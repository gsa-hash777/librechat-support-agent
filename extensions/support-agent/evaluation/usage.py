"""Conservative usage accounting for the OpenAI-compatible DeepSeek evaluation.

Prefer each ``on_token_usage`` SSE data object. A LibreChat message rollup is a
fallback: its ``input`` excludes cache, and upstream fills absent cache fields
with zero. Neither tokenCount nor the local transaction ledger proves provider
billing. This module performs no network access and contains no model prices.
"""
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
import math


def _count(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0 or int(value) != value:
        return None
    return int(value)


def _first(*values):
    return next((number for value in values if (number := _count(value)) is not None), None)


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _model(value):
    # LibreChat persists an agent ID in message.model, not the model it called.
    return value if isinstance(value, str) and value and not value.startswith("agent_") else None


def _record(source, measurement, input_tokens=None, output_tokens=None, *, hit=None,
            miss=None, write=None, hit_lower=None, hit_upper=None, model=None,
            notes=(), model_calls=None):
    known = input_tokens is not None and output_tokens is not None
    if input_tokens is not None and input_tokens == 0:
        hit = miss = hit_lower = hit_upper = 0
    if hit_lower is None and input_tokens is not None:
        hit_lower = hit if hit is not None else 0
    if hit_upper is None and input_tokens is not None:
        hit_upper = hit if hit is not None else input_tokens
    cache_coverage = "known" if hit is not None and miss is not None else "partial" if (hit_lower or 0) > 0 else "unknown"
    return {
        "source": source, "measurement": measurement, "model": _model(model),
        "models": [_model(model)] if _model(model) else [],
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens if known else None,
        "known_input_tokens": input_tokens, "known_output_tokens": output_tokens,
        "cache_hit_tokens": hit, "cache_miss_tokens": miss, "cache_write_tokens": write,
        "cache_hit_lower_bound": hit_lower, "cache_hit_upper_bound": hit_upper,
        "input_unknown_cache_tokens": hit_upper - hit_lower if hit_upper is not None and hit_lower is not None else None,
        "cache_coverage": cache_coverage, "model_calls": model_calls,
        "coverage": {"records": 1, "input_known": int(input_tokens is not None),
                     "output_known": int(output_tokens is not None), "cache_known": int(cache_coverage == "known"),
                     "provider_reported": int(source == "provider-reported"),
                     "estimated": int(source == "estimated"), "missing": int(source == "missing")},
        "notes": list(notes),
    }


def _event_record(event, model):
    details = _mapping(event.get("input_token_details"))
    prompt_details = _mapping(event.get("prompt_tokens_details"))
    inputs = _first(event.get("input_tokens"), event.get("prompt_tokens"))
    outputs = _first(event.get("output_tokens"), event.get("completion_tokens"))
    hit = _first(details.get("cache_read"), prompt_details.get("cached_tokens"), event.get("prompt_cache_hit_tokens"))
    miss = _first(event.get("prompt_cache_miss_tokens"))
    write = _first(details.get("cache_creation"))
    notes = []
    # The current evaluation uses DeepSeek's OpenAI-compatible convention:
    # input_tokens already includes cached reads. Do not guess for other APIs.
    provider = str(event.get("provider", "")).casefold().replace("_", "").replace("-", "")
    if provider and provider not in {"openai", "deepseek", "azure", "azureopenai", "custom"}:
        notes.append("unsupported_provider_input_convention")
        inputs = hit = miss = write = None
    if inputs is not None:
        if hit is not None and hit > inputs or miss is not None and miss > inputs:
            notes.append("invalid_cache_breakdown")
            hit = miss = None
        elif hit is not None and miss is not None and hit + miss != inputs:
            notes.append("conflicting_cache_breakdown")
            hit = miss = None
        elif hit is not None:
            miss = inputs - hit
        elif miss is not None:
            hit = inputs - miss
        if write is not None and (write > inputs or hit is not None and hit + write > inputs):
            notes.append("invalid_cache_write_count")
            write = None
    else:
        hit = miss = None
    total = _first(event.get("total_tokens"))
    if inputs is not None and outputs is not None and total is not None and total != inputs + outputs:
        notes.append("provider_total_does_not_match_input_plus_output")
    source = "provider-reported" if inputs is not None or outputs is not None else "missing"
    return _record(source, "sse-provider-usage", inputs, outputs, hit=hit, miss=miss,
                   write=write, model=_model(event.get("model")) or model, notes=notes, model_calls=1)


def aggregate_usage(records: Iterable[Mapping]):
    """Sum independent normalized responses; missing values remain missing.

    Known subtotals and coverage are retained, so a failed/partially observed
    request cannot silently become zero tokens. Callers must not add SSE and
    message-rollup records for the same response; normalize_usage chooses one.
    """
    rows = list(records)
    if not rows:
        result = _record("missing", "unavailable", notes=["no_usage_records"])
        result["coverage"] = {key: 0 for key in result["coverage"]}
        return result

    def complete_sum(key):
        values = [_count(row.get(key)) for row in rows]
        return sum(values) if all(value is not None for value in values) else None

    def known_sum(key):
        values = [_count(row.get(key)) for row in rows]
        return sum(value for value in values if value is not None) if any(value is not None for value in values) else None

    sources = {row.get("source", "missing") for row in rows}
    source = next(iter(sources)) if len(sources) == 1 else "mixed"
    models = sorted({model for row in rows for model in row.get("models", []) if _model(model)})
    result = _record(source, "aggregate", complete_sum("input_tokens"), complete_sum("output_tokens"),
                     hit=complete_sum("cache_hit_tokens"), miss=complete_sum("cache_miss_tokens"),
                     write=complete_sum("cache_write_tokens"), hit_lower=complete_sum("cache_hit_lower_bound"),
                     hit_upper=complete_sum("cache_hit_upper_bound"),
                     model=models[0] if len(models) == 1 else None,
                     model_calls=complete_sum("model_calls"))
    result["models"] = models
    result["known_input_tokens"] = known_sum("known_input_tokens")
    result["known_output_tokens"] = known_sum("known_output_tokens")
    # Explicitly retain unknown bounds rather than _record's default bounds.
    result["cache_hit_lower_bound"] = complete_sum("cache_hit_lower_bound")
    result["cache_hit_upper_bound"] = complete_sum("cache_hit_upper_bound")
    result["input_unknown_cache_tokens"] = complete_sum("input_unknown_cache_tokens")
    result["coverage"] = {key: sum(_count(_mapping(row.get("coverage")).get(key)) or 0 for row in rows)
                          for key in result["coverage"]}
    result["notes"] = sorted({note for row in rows for note in row.get("notes", [])})
    if result["coverage"]["cache_known"] != result["coverage"]["records"]:
        result["cache_coverage"] = "partial" if result["coverage"]["cache_known"] or any((row.get("cache_hit_lower_bound") or 0) > 0 for row in rows) else "unknown"
    return result


def normalize_usage(message=None, usage_events=None, estimated=None, *, model=None):
    """Prefer SSE usage, then the saved message rollup, then explicit estimates.

    ``usage_events`` accepts raw data dictionaries or {event, data} wrappers.
    Replayed SSE events are deduplicated only using runId+seq+usage_type+model;
    distinct calls with identical token counts must both be counted.
    ``estimated`` accepts input_tokens/output_tokens supplied by the caller.
    """
    records, seen, notes = [], {}, set()
    for supplied in usage_events or []:
        supplied = _mapping(supplied)
        if "event" in supplied:
            if supplied.get("event") != "on_token_usage":
                continue
            event = _mapping(supplied.get("data"))
        else:
            event = supplied
        if not any(key in event for key in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens")):
            continue
        seq = _count(event.get("seq"))
        run_id = event.get("runId")
        if isinstance(run_id, str) and run_id and seq is not None:
            identity = (run_id, seq, event.get("usage_type"), event.get("model"), event.get("provider"))
            if identity in seen:
                if seen[identity] != event:
                    notes.add("conflicting_replayed_usage_event")
                continue
            seen[identity] = event
        records.append(_event_record(event, model))
    if records:
        result = aggregate_usage(records)
        result["measurement"] = "sse-provider-usage"
        result["notes"] = sorted(set(result["notes"]) | notes)
        result["scope"] = "observed_completed_model_calls_in_response"
        return result

    message = _mapping(message)
    rollup = _mapping(_mapping(message.get("metadata")).get("usage"))
    if rollup:
        plain, read, write, output = (_count(rollup.get(key)) for key in ("input", "cacheRead", "cacheWrite", "output"))
        inputs = plain + read + write if all(value is not None for value in (plain, read, write)) else None
        result = _record("provider-reported", "librechat-message-rollup", inputs, output,
                         write=write, hit_lower=read, hit_upper=inputs - write if inputs is not None else None,
                         model=model or _model(message.get("model")), notes=[
                             "librechat_normalized_provider_usage",
                             "rollup_zero_fills_missing_cache_fields_cache_totals_are_bounds",
                             "rollup_may_include_auxiliary_models_price_only_when_single_model_is_verified",
                         ])
        result["scope"] = "persisted_observed_model_calls_in_response_excludes_separate_title_jobs"
        return result

    estimated = _mapping(estimated)
    inputs, outputs = _count(estimated.get("input_tokens")), _count(estimated.get("output_tokens"))
    if inputs is not None or outputs is not None:
        result = _record("estimated", "caller-estimate", inputs, outputs,
                         model=model or _model(estimated.get("model")), notes=["caller_estimate_not_provider_measurement"])
        result["scope"] = "caller_defined_estimate"
        return result
    result = _record("missing", "unavailable", model=model, notes=["tokenCount_is_not_full_request_usage"])
    result["scope"] = "unknown"
    return result


def _rate(value):
    if isinstance(value, bool):
        raise ValueError("Prices must be finite nonnegative numbers.")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("Prices must be finite nonnegative numbers.") from None
    if not result.is_finite() or result < 0:
        raise ValueError("Prices must be finite nonnegative numbers.")
    return result


def estimate_cost(usage, prices):
    """Return price-based cost bounds, never an actual provider charge.

    Prices: {currency: 'CNY'|'USD', model: str, input_per_million: number,
    cache_hit_per_million: number, output_per_million: number}. Alternatively put
    those three rates under rates: {off_peak: {...}, peak: {...}}. The interval
    then includes both tariff scenarios and unknown cache coverage. Non-hit
    input is priced at input_per_million (DeepSeek has no separate write rate).
    """
    prices = _mapping(prices)
    currency = prices.get("currency")
    if currency not in {"CNY", "USD"}:
        raise ValueError("Price currency must be CNY or USD; currencies are never converted.")
    price_model = _model(prices.get("model"))
    if price_model is None:
        raise ValueError("A model name is required in the price configuration.")
    sets = prices.get("rates", {"configured": prices})
    if not isinstance(sets, Mapping) or not sets:
        raise ValueError("Prices must contain at least one named rate set.")
    parsed = {}
    for name, rates in sets.items():
        rates = _mapping(rates)
        parsed[str(name)] = tuple(_rate(rates.get(key)) for key in (
            "input_per_million", "cache_hit_per_million", "output_per_million"))

    models = usage.get("models", [])
    result = {"currency": currency, "model": price_model, "actual_charge": False,
              "basis": "supplied_prices_and_observed_or_estimated_tokens",
              "lower_bound": None, "upper_bound": None, "conservative_budget": None,
              "scenarios": {}, "status": "unavailable"}
    if models != [price_model]:
        result["status"] = "unpriced_model_missing_or_mismatch"
        return result
    inputs = _count(usage.get("input_tokens"))
    outputs = _count(usage.get("output_tokens"))
    if inputs is None or outputs is None:
        result["status"] = "incomplete_token_coverage"
        return result
    lower_hit = _count(usage.get("cache_hit_lower_bound"))
    upper_hit = _count(usage.get("cache_hit_upper_bound"))
    lower_hit = 0 if lower_hit is None else lower_hit
    upper_hit = inputs if upper_hit is None else upper_hit
    if not 0 <= lower_hit <= upper_hit <= inputs:
        raise ValueError("Cache bounds must be within total input tokens.")

    scale = Decimal(1000000)
    def amount(value):
        return float(value / scale)
    for name, (miss_rate, hit_rate, output_rate) in parsed.items():
        possibilities = [(inputs - hit) * miss_rate + hit * hit_rate + outputs * output_rate
                         for hit in (lower_hit, upper_hit)]
        result["scenarios"][name] = {
            "lower_bound": amount(min(possibilities)), "upper_bound": amount(max(possibilities)),
            "all_input_cache_miss": amount(inputs * miss_rate + outputs * output_rate),
        }
    result["lower_bound"] = min(value["lower_bound"] for value in result["scenarios"].values())
    result["upper_bound"] = max(value["upper_bound"] for value in result["scenarios"].values())
    result["conservative_budget"] = max(max(value["upper_bound"], value["all_input_cache_miss"])
                                        for value in result["scenarios"].values())
    result["status"] = "estimate" if usage.get("source") in {"estimated", "mixed"} else "price_based_estimate"
    return result
