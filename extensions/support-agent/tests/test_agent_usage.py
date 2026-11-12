"""Usage accounting never turns missing telemetry into zero billed tokens."""
import math

import pytest

from evaluation.usage import aggregate_usage, estimate_cost, normalize_usage


MODEL = "deepseek-test-model"


def event(inputs=100, outputs=20, *, hit=40, seq=1, **extra):
    result = {"input_tokens": inputs, "output_tokens": outputs, "total_tokens": inputs + outputs,
              "model": MODEL, "provider": "openAI", "runId": "run-test", "seq": seq}
    if hit is not None:
        result["input_token_details"] = {"cache_read": hit}
    return {**result, **extra}


def prices(currency="CNY"):
    # Synthetic prices verify arithmetic; they are not current provider prices.
    return {"currency": currency, "model": MODEL, "rates": {
        "off_peak": {"input_per_million": 2, "cache_hit_per_million": 0.5, "output_per_million": 4},
        "peak": {"input_per_million": 4, "cache_hit_per_million": 1, "output_per_million": 8},
    }}


def test_sse_all_model_calls_and_reconnect_deduplication():
    first, second = event(), event(seq=2)
    wrapped = {"event": "on_token_usage", "data": first}
    usage = normalize_usage(usage_events=[wrapped, first, second, {"event": "other", "data": second}])
    assert usage["source"] == "provider-reported"
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (200, 40, 240)
    assert (usage["cache_hit_tokens"], usage["cache_miss_tokens"]) == (80, 120)
    assert usage["model_calls"] == 2
    assert usage["coverage"] == {"records": 2, "input_known": 2, "output_known": 2,
                                  "cache_known": 2, "provider_reported": 2, "estimated": 0, "missing": 0}


def test_identical_unkeyed_events_are_distinct_calls():
    value = event()
    del value["runId"], value["seq"]
    assert normalize_usage(usage_events=[value, value])["input_tokens"] == 200


def test_auxiliary_call_sequence_does_not_collide_with_main_call():
    usage = normalize_usage(usage_events=[event(), event(usage_type="subagent")])
    assert usage["model_calls"] == 2
    assert usage["input_tokens"] == 200


def test_conflicting_replayed_event_is_not_summed_twice():
    usage = normalize_usage(usage_events=[event(), event(outputs=30)])
    assert usage["output_tokens"] == 20
    assert "conflicting_replayed_usage_event" in usage["notes"]


def test_sse_precedes_message_rollup_without_double_counting():
    message = {"metadata": {"usage": {"input": 600, "output": 90, "cacheRead": 400, "cacheWrite": 0}}}
    usage = normalize_usage(message, [event()])
    assert usage["input_tokens"] == 100
    assert usage["measurement"] == "sse-provider-usage"


def test_cache_missing_is_unknown_and_explicit_zero_is_known():
    unknown = normalize_usage(usage_events=[event(hit=None)])
    known_zero = normalize_usage(usage_events=[event(hit=0)])
    assert unknown["cache_hit_tokens"] is None and unknown["cache_miss_tokens"] is None
    assert (unknown["cache_hit_lower_bound"], unknown["cache_hit_upper_bound"]) == (0, 100)
    assert unknown["cache_coverage"] == "unknown"
    assert known_zero["cache_hit_tokens"] == 0 and known_zero["cache_miss_tokens"] == 100
    assert known_zero["cache_coverage"] == "known"


def test_partial_cache_coverage_retains_bounds():
    usage = normalize_usage(usage_events=[event(hit=40), event(hit=None, seq=2)])
    assert usage["cache_hit_tokens"] is None and usage["cache_miss_tokens"] is None
    assert (usage["cache_hit_lower_bound"], usage["cache_hit_upper_bound"]) == (40, 140)
    assert usage["input_unknown_cache_tokens"] == 100
    assert usage["cache_coverage"] == "partial"


def test_raw_deepseek_cache_hit_miss_contract():
    usage = normalize_usage(usage_events=[{
        "prompt_tokens": 100, "completion_tokens": 15,
        "prompt_cache_hit_tokens": 60, "prompt_cache_miss_tokens": 40, "model": MODEL,
    }])
    assert (usage["cache_hit_tokens"], usage["cache_miss_tokens"]) == (60, 40)
    assert usage["input_tokens"] == 100


def test_persisted_real_shape_input_excludes_cache_and_cannot_prove_coverage():
    message = {"model": "agent_demo", "tokenCount": 712, "metadata": {"usage": {
        "input": 3114, "output": 712, "cacheWrite": 0, "cacheRead": 5248,
    }}}
    usage = normalize_usage(message, model=MODEL)
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (8362, 712, 9074)
    assert usage["model_calls"] is None
    assert usage["cache_hit_tokens"] is None and usage["cache_miss_tokens"] is None
    assert (usage["cache_hit_lower_bound"], usage["cache_hit_upper_bound"]) == (5248, 8362)
    assert usage["source"] == "provider-reported" and usage["cache_coverage"] == "partial"


def test_token_count_alone_is_not_full_usage_or_automatically_estimated():
    usage = normalize_usage({"tokenCount": 800})
    assert usage["source"] == "missing"
    assert usage["input_tokens"] is None and usage["output_tokens"] is None
    assert usage["cache_hit_tokens"] is None


def test_explicit_estimate_has_distinct_provenance():
    usage = normalize_usage(estimated={"input_tokens": 100, "output_tokens": 20}, model=MODEL)
    assert usage["source"] == "estimated"
    assert usage["cache_coverage"] == "unknown"
    assert estimate_cost(usage, prices())["status"] == "estimate"


def test_aggregate_missing_response_preserves_coverage_and_known_subtotals():
    usage = aggregate_usage([normalize_usage(usage_events=[event()]), normalize_usage(model=MODEL)])
    assert usage["input_tokens"] is None and usage["output_tokens"] is None
    assert usage["known_input_tokens"] == 100 and usage["known_output_tokens"] == 20
    assert usage["coverage"]["records"] == 2 and usage["coverage"]["missing"] == 1
    cost = estimate_cost(usage, prices())
    assert cost["status"] == "incomplete_token_coverage"
    assert cost["upper_bound"] is None


def test_empty_aggregate_is_missing_not_zero_usage():
    usage = aggregate_usage([])
    assert usage["source"] == "missing" and usage["input_tokens"] is None
    assert usage["coverage"]["records"] == 0


@pytest.mark.parametrize("invalid", [-1, True, math.nan, math.inf, "100", 2.5])
def test_invalid_token_values_are_missing(invalid):
    usage = normalize_usage(usage_events=[event(input_tokens=invalid)])
    assert usage["input_tokens"] is None


def test_invalid_or_conflicting_cache_counts_are_not_clamped_to_fake_certainty():
    for value in [event(hit=101), event(prompt_cache_miss_tokens=80)]:
        usage = normalize_usage(usage_events=[value])
        assert usage["cache_hit_tokens"] is None
        assert usage["cache_hit_upper_bound"] == 100
        assert usage["notes"]


def test_unsupported_provider_does_not_guess_additive_cache_convention():
    usage = normalize_usage(usage_events=[event(provider="bedrock")])
    assert usage["input_tokens"] is None
    assert "unsupported_provider_input_convention" in usage["notes"]


@pytest.mark.parametrize("currency", ["CNY", "USD"])
def test_cost_known_cache_uses_supplied_currency_and_scenarios(currency):
    usage = normalize_usage(usage_events=[event()])
    cost = estimate_cost(usage, prices(currency))
    assert cost["currency"] == currency and cost["actual_charge"] is False
    assert cost["lower_bound"] == pytest.approx((60 * 2 + 40 * 0.5 + 20 * 4) / 1e6)
    assert cost["upper_bound"] == pytest.approx((60 * 4 + 40 * 1 + 20 * 8) / 1e6)
    assert cost["conservative_budget"] == pytest.approx((100 * 4 + 20 * 8) / 1e6)


def test_cost_unknown_cache_includes_all_hit_to_all_miss_interval():
    cost = estimate_cost(normalize_usage(usage_events=[event(hit=None)]), prices())
    assert cost["lower_bound"] == pytest.approx((100 * 0.5 + 20 * 4) / 1e6)
    assert cost["upper_bound"] == pytest.approx((100 * 4 + 20 * 8) / 1e6)


def test_single_flat_price_configuration():
    config = {"model": MODEL, "currency": "CNY", **prices()["rates"]["off_peak"]}
    cost = estimate_cost(normalize_usage(usage_events=[event()]), config)
    assert cost["lower_bound"] == cost["upper_bound"]


def test_mixed_or_unverified_model_cannot_be_priced_as_one_model():
    for usage in [normalize_usage({"metadata": {"usage": {"input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0}}}),
                  normalize_usage(usage_events=[event(), event(seq=2, model="another-model")])]:
        cost = estimate_cost(usage, prices())
        assert cost["status"] == "unpriced_model_missing_or_mismatch"
        assert cost["upper_bound"] is None


@pytest.mark.parametrize("change", [{"currency": "EUR"}, {"model": None},
    {"rates": {}}, {"rates": {"bad": {"input_per_million": -1, "cache_hit_per_million": 1, "output_per_million": 2}}}])
def test_invalid_prices_rejected(change):
    with pytest.raises(ValueError):
        estimate_cost(normalize_usage(usage_events=[event()]), {**prices(), **change})
