"""Budget boundaries, no-retry behavior and fixed-denominator reporting."""
import json
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import task_success_run as runner


class BudgetViewTests(unittest.TestCase):
    @staticmethod
    def approved_amendment():
        return {"from_limit_cny": 5, "to_limit_cny": 5.5, "additional_cny": .5, "authorized_by": "user",
                "authorized_at": "2026-11-12T10:00:00Z", "authorization_text": "Only finish the five unattempted tasks; do not retry P05.",
                "only_unattempted_case_ids": ["P06", "P07", "P08", "P09", "P10"], "do_not_retry_case_ids": ["P05"],
                "original_ledger_sha256": "a" * 64}

    def test_amendment_requires_explicit_scope_and_preserves_spending(self):
        from evaluation.task_success_proxy import Ledger, GuardError
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            state = {"schema_version": "task-success-request-guard-v1", "limit_cny": 5.5,
                     "attempts": [{"attempt_id": "old", "reserved_cny": 5.49, "cost_cny": 5.49, "status": "settled"}]}
            runner.save_state(path, state)
            with self.assertRaises(runner.BootstrapError):
                runner.Budget(path)
            state["amendments"] = [self.approved_amendment()]
            runner.save_state(path, state)
            self.assertEqual(str(runner.Budget(path).occupied), "5.49")
            ledger = Ledger(path)
            with self.assertRaisesRegex(GuardError, "budget_exhausted"):
                ledger.reserve(b"x", 1500)
            self.assertEqual(runner.read_json(path)["attempts"], state["attempts"])
            state["amendments"][0]["only_unattempted_case_ids"] = ["P05"]
            runner.save_state(path, state)
            with self.assertRaises(GuardError):
                Ledger(path)

    def test_unresolved_reservation_is_not_zero_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            path.write_text(json.dumps({"limit_cny": 5, "attempts": [
                {"reserved_cny": .3, "status": "unknown_usage"},
                {"reserved_cny": .5, "cost_cny": .12, "status": "settled"}]}))
            budget = runner.Budget(path)
            self.assertFalse(budget.settled())
            self.assertEqual(str(budget.occupied), "0.42")

    def test_budget_limit_cannot_be_silently_raised(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            path.write_text(json.dumps({"limit_cny": 10, "attempts": []}))
            with self.assertRaises(runner.BootstrapError):
                runner.Budget(path)


class JudgeTests(unittest.TestCase):
    def test_oversize_input_never_calls_provider(self):
        with patch.object(runner, "build_opener") as network:
            verdict, record = runner.judge_once([{"role": "user", "content": "x" * 100001}], "dummy", None, "r", "K01", runner.price_snapshot())
            self.assertEqual(verdict["status"], "unscored")
            self.assertIsNone(record)
            network.assert_not_called()

    def test_failed_submission_is_never_retried_or_treated_as_free(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            path.write_text(json.dumps({"limit_cny": 5, "attempts": []}))
            budget = runner.Budget(path)
            with patch.object(runner, "build_opener") as network:
                network.return_value.open.side_effect = TimeoutError("not for output")
                verdict, record = runner.judge_once([{"role": "user", "content": "test"}], "dummy", budget, "r", "K01", runner.price_snapshot())
                self.assertEqual(network.return_value.open.call_count, 1)
                self.assertEqual(verdict["status"], "unscored")
                self.assertFalse(record["settled"])
                self.assertNotIn("not for output", json.dumps(verdict))
                request = network.return_value.open.call_args.args[0]
                self.assertEqual(request.full_url, "http://127.0.0.1:8093/v1/chat/completions")


class ReportingTests(unittest.TestCase):
    def test_unscored_and_unattempted_stay_in_primary_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            path.write_text(json.dumps({"limit_cny": 5, "attempts": []}))
            results = []
            for cid, status, duration, turns in (("K01", "pass", 5, 1), ("K02", "fail", 20, 1), ("I01", "unscored", 30, 2)):
                results.append({"case_id": cid, "category": "knowledge", "outcome": {"status": status},
                                "hard": {"passed": True}, "semantic": {"status": status}, "error": None,
                                "planned_messages": turns, "completed_messages": turns, "elapsed_seconds": duration,
                                "usage": runner.normalize_usage(model=runner.MODEL), "judge_record": {}})
            result = runner.report({"categories": {"knowledge": 100}}, results, runner.Budget(path))
            self.assertEqual(result["task_success_rate"], .01)
            self.assertEqual(result["not_attempted"], 97)
            self.assertEqual(result["unscored"], 1)
            self.assertEqual(result["failure"], 1)
            self.assertEqual(result["latency"]["single_message"]["p95_seconds"], 20)
            self.assertEqual(result["latency"]["two_message"]["n"], 1)

    def test_nearest_rank_p95_is_explicit(self):
        self.assertEqual(runner.percentile(list(range(1, 101)), .95), 95)
        self.assertIsNone(runner.percentile([], .95))


class ContinuationTests(unittest.TestCase):
    def test_budget_continuation_never_retries_interrupted_case(self):
        cases = runner.read_json(runner.SERVICE / "evaluation/agent_cases.json")["cases"]
        with tempfile.TemporaryDirectory() as directory:
            root, local, old, current, _, frozen = self.fixture(directory)
            template = runner.read_json(old / "K01.json")
            rows = [{**template, "case_id": c["id"], "category": c["category"], "outcome": {"status": "pass"}}
                    for c in cases[:95]]
            rows[-1].update(outcome={"status": "fail", "success": False}, stop_reason="agent_runtime_or_usage_requires_review", judge_record={})
            self.assertEqual(rows[-1]["case_id"], "P05")
            manifest = runner.read_json(old / "manifest.json")
            manifest["case_ids"] = [c["id"] for c in cases]
            runner.save_state(old / "manifest.json", manifest)
            runner.save_state(old / "results.json", rows)
            for row in rows:
                runner.save_state(old / (row["case_id"] + ".json"), row)
            ledger = runner.read_json(local / "budget.json")
            amendment = BudgetViewTests.approved_amendment()
            ledger.update(limit_cny=5.5, amendments=[amendment])
            runner.save_state(local / "budget.json", ledger)
            with patch.object(runner, "ROOT", root), patch.object(runner, "LOCAL", local), patch.object(runner, "source_paths", return_value=[current]):
                folder, latest, combined = runner.make_continuation(old, cases, frozen, runner.price_snapshot(), runner.Budget(local / "budget.json"), 150,
                    {"kind": "budget-amendment-v1", "amendment": amendment})
            self.assertEqual(latest["max_budget_cny"], 5.5)
            self.assertEqual(latest["continuation"]["inherited_execution_count"], 95)
            self.assertEqual(combined[-1]["administrative_status"], "admin_budget_interruption")
            self.assertEqual(combined[-1]["outcome"]["status"], "unscored")
            self.assertEqual(combined[-1]["outcome_before_administrative_classification"]["status"], "fail")
            self.assertEqual(runner.read_json(old / "P05.json")["outcome"]["status"], "fail")
            self.assertEqual([c["id"] for c in cases[len(combined):]], ["P06", "P07", "P08", "P09", "P10"])

    def fixture(self, directory):
        local = Path(directory) / ".local/task-success-v1"
        old = local / "results/old-run"
        frozen = old / "frozen" / runner.RUNNER_RELATIVE
        frozen.parent.mkdir(parents=True)
        frozen.write_text("old runner")
        root = Path(directory)
        current = root / runner.RUNNER_RELATIVE
        current.parent.mkdir(parents=True, exist_ok=True)
        current.write_text("new continuation runner")
        cases = [{"id": "K01"}, {"id": "K02"}]
        row = {"case_id": "K01", "category": "knowledge", "outcome": {"status": "unscored"},
               "hard": {"passed": True}, "semantic": {"status": "unscored"}, "error": None,
               "planned_messages": 1, "completed_messages": 1, "elapsed_seconds": 5,
               "usage": runner.normalize_usage(model=runner.MODEL), "judge_record": {}}
        manifest = {"run_id": "old-run", "protocol": "support-task-success-v1", "status": "stopped", "stop_reason": "frozen_sources_changed",
                    "case_ids": ["K01", "K02"], "prices": runner.price_snapshot(), "max_budget_cny": 5,
                    "judge_model": runner.MODEL, "judge_temperature": 0, "judge_max_tokens": runner.JUDGE_OUTPUT,
                    "judge_max_input_bytes": runner.JUDGE_INPUT_BYTES, "source_sha256": {runner.RUNNER_RELATIVE: runner.digest(frozen)},
                    "categories": {"knowledge": 100}}
        ledger = {"limit_cny": 5, "attempts": [{"attempt_id": "paid-1", "reserved_cny": .3, "cost_cny": .1, "status": "settled"}]}
        for path, value in ((old / "manifest.json", manifest), (old / "results.json", [row]),
                            (old / "K01.json", row), (old / "report.json", {"budget": ledger}), (local / "budget.json", ledger)):
            path.write_text(json.dumps(value), encoding="utf-8")
        return root, local, old, current, cases, {runner.RUNNER_RELATIVE: runner.digest(current)}

    def test_continuation_preserves_old_outcomes_and_files_without_reexecuting(self):
        with tempfile.TemporaryDirectory() as directory:
            root, local, old, current, cases, frozen = self.fixture(directory)
            original = {str(path): runner.digest(path) for path in old.rglob("*") if path.is_file()}
            with patch.object(runner, "ROOT", root), patch.object(runner, "LOCAL", local), patch.object(runner, "source_paths", return_value=[current]):
                folder, manifest, rows = runner.make_continuation(old, cases, frozen, runner.price_snapshot(), runner.Budget(local / "budget.json"), 150)
                self.assertNotEqual(folder, old)
                self.assertEqual(rows, runner.read_json(old / "results.json"))
                self.assertEqual(rows[0]["outcome"]["status"], "unscored")
                self.assertEqual(manifest["continuation"]["inherited_execution_count"], 1)
                self.assertEqual(manifest["continuation"]["agent_executions_repeated"], 0)
                self.assertEqual(manifest["continuation"]["source_changes"][0]["path"], runner.RUNNER_RELATIVE)
                with self.assertRaises(FileExistsError):
                    runner.make_continuation(old, cases, frozen, runner.price_snapshot(), runner.Budget(local / "budget.json"), 150)
            self.assertEqual(original, {str(path): runner.digest(path) for path in old.rglob("*") if path.is_file()})

    def test_unresolved_case_and_extra_paid_attempt_prevent_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            root, local, old, current, cases, frozen = self.fixture(directory)
            manifest = runner.read_json(old / "manifest.json")
            manifest["pending_case"] = "K02"
            (old / "manifest.json").write_text(json.dumps(manifest))
            with patch.object(runner, "LOCAL", local):
                with self.assertRaisesRegex(runner.BootstrapError, "unresolved"):
                    runner.inspect_continuation(old, cases, frozen, runner.price_snapshot(), runner.Budget(local / "budget.json"))
                manifest.pop("pending_case")
                (old / "manifest.json").write_text(json.dumps(manifest))
                ledger = runner.read_json(local / "budget.json")
                ledger["attempts"].append({"attempt_id": "unmapped-paid-2", "reserved_cny": .2, "cost_cny": .1, "status": "settled"})
                (local / "budget.json").write_text(json.dumps(ledger))
                with self.assertRaisesRegex(runner.BootstrapError, "New paid calls"):
                    runner.inspect_continuation(old, cases, frozen, runner.price_snapshot(), runner.Budget(local / "budget.json"))

    def test_changed_outcomes_or_nonprefix_cannot_be_imported(self):
        with self.assertRaises(runner.BootstrapError):
            runner.validate_result_prefix([{"case_id": "K02", "outcome": {"status": "pass"}}], [{"id": "K01"}, {"id": "K02"}])
        with tempfile.TemporaryDirectory() as directory:
            root, local, old, current, cases, frozen = self.fixture(directory)
            rows = runner.read_json(old / "results.json")
            rows[0]["outcome"]["status"] = "pass"
            (old / "results.json").write_text(json.dumps(rows))
            with patch.object(runner, "LOCAL", local), self.assertRaisesRegex(runner.BootstrapError, "per-case"):
                runner.inspect_continuation(old, cases, frozen, runner.price_snapshot(), runner.Budget(local / "budget.json"))

    def test_agent_policy_or_reference_change_is_never_implicit(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)
            path = archive / "extensions/support-agent/integration/bootstrap.py"
            path.parent.mkdir(parents=True)
            path.write_text("policy")
            name = "extensions/support-agent/integration/bootstrap.py"
            with self.assertRaisesRegex(runner.BootstrapError, "Agent policy"):
                runner.source_change_review({name: runner.digest(path)}, {name: "0" * 64}, archive)

    def test_format_review_cannot_change_judge_prompt_or_other_oracle_logic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "archive"
            before = archive / runner.ORACLE_RELATIVE
            after = root / runner.ORACLE_RELATIVE
            before.parent.mkdir(parents=True); after.parent.mkdir(parents=True)
            before.write_text('JUDGE_SYSTEM = "original"\ndef parse_judge_response(x):\n return x\n')
            after.write_text('JUDGE_SYSTEM = "changed"\ndef parse_judge_response(x):\n return x\n')
            previous, current = {runner.ORACLE_RELATIVE: runner.digest(before)}, {runner.ORACLE_RELATIVE: runner.digest(after)}
            review = {"path": runner.ORACLE_RELATIVE, "before_sha256": runner.digest(before), "after_sha256": runner.digest(after),
                      "semantic_rules_unchanged": True, "reason": "format only"}
            with patch.object(runner, "ROOT", root), self.assertRaisesRegex(runner.BootstrapError, "outside"):
                runner.source_change_review(previous, current, archive, review)

    def test_structure_diagnostic_contains_no_unexpected_generated_content(self):
        content = json.dumps({"dimensions": {"goal_completion": {"status": "pass", "reason": "secret-prose", "evidence": ["unexpected-array"]}, "malicious-field": "secret-prose"}, "reasons": [], "secret-prose": "secret-prose"})
        shape = runner.judge_response_shape(content)
        self.assertEqual(shape["dimensions"]["goal_completion"]["evidence_type"], "list")
        self.assertEqual(shape["unexpected_top_level_key_count"], 1)
        self.assertNotIn("secret-prose", json.dumps(shape))
        self.assertNotIn("malicious-field", json.dumps(shape))
        self.assertNotIn("unexpected-array", json.dumps(shape))

    def test_checkpoint_executes_only_next_unattempted_task(self):
        from evaluation import task_success as oracle, task_success_setup as setup
        cases = runner.read_json(runner.SERVICE / "evaluation/agent_cases.json")["cases"]
        def result_for(case):
            return {"case_id": case["id"], "category": case["category"], "outcome": {"status": "unscored"},
                    "hard": {"passed": True}, "semantic": {"status": "unscored"}, "error": None,
                    "planned_messages": 1, "completed_messages": 1, "elapsed_seconds": 5,
                    "usage": runner.normalize_usage(model=runner.MODEL), "judge_record": {}, "stop_reason": None}
        inherited = [result_for(case) for case in cases[:37]]
        self.assertEqual(cases[37]["id"], "N08")
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            local = root / "tasks"
            folder = local / "results/new"
            folder.mkdir(parents=True)
            (root / ".local/evaluation").mkdir(parents=True)
            (root / ".local/evaluation/demo.json").write_text("{}")
            (local / "agents.json").write_text("{}")
            (local / "budget.json").write_text(json.dumps({"limit_cny": 5, "attempts": []}))
            manifest = {"run_id": "continuation", "source_sha256": {}, "categories": dict(runner.Counter(c["category"] for c in cases)),
                        "continuation": {"inherited_execution_count": 37}, "status": "running", "stop_reason": None}
            for name, value in (("ROOT", root), ("LOCAL", local)):
                stack.enter_context(patch.object(runner, name, value))
            stack.enter_context(patch.object(sys, "argv", ["runner", "--run", "--continue-from", str(local / "results/old"), "--stop-after-new", "1"]))
            stack.enter_context(patch.object(runner, "exclusive_run", return_value=nullcontext()))
            stack.enter_context(patch.object(runner, "authenticate"))
            stack.enter_context(patch.object(runner, "renew_session"))
            api = stack.enter_context(patch.object(runner, "LocalApi"))
            api.return_value.token = "safe-test-token"
            stack.enter_context(patch.object(setup, "validate_agents"))
            stack.enter_context(patch.object(runner, "secrets_and_key", return_value=([], "safe-test-key")))
            stack.enter_context(patch.object(runner, "fingerprints", return_value={}))
            stack.enter_context(patch.object(runner, "make_continuation", return_value=(folder, manifest, inherited)))
            execute = stack.enter_context(patch.object(runner, "execute_task", return_value=result_for(cases[37])))
            stack.enter_context(patch.object(oracle, "deterministic_checks", return_value={"passed": True, "input_valid": True, "checks": {}}))
            stack.enter_context(patch.object(oracle, "build_judge_messages", return_value=[]))
            judge = stack.enter_context(patch.object(runner, "judge_once", return_value=({"status": "pass"}, {"settled": True})))
            self.assertEqual(runner.main(), 0)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(execute.call_args.args[2]["id"], "N08")
            self.assertEqual(judge.call_count, 1)
            self.assertEqual(manifest["status"], "checkpoint_stopped")
            self.assertEqual(manifest["checkpoint"]["next_case_id"], "N09")
            saved = runner.read_json(folder / "results.json")
            self.assertEqual(len(saved), 38)
            self.assertTrue(all(row["outcome"]["status"] == "unscored" for row in saved[:37]))


class UniformFormatUpgradeTests(unittest.TestCase):
    def fixture(self, directory, stack):
        from evaluation import task_success as oracle
        root, local, old, current, _, frozen = ContinuationTests().fixture(directory)
        old_prompt = oracle.JUDGE_SYSTEM.split("\n\nFORMAT CONTRACT v1.1", 1)[0]
        self.assertNotEqual(old_prompt, oracle.JUDGE_SYSTEM)
        before = old / "frozen" / runner.ORACLE_RELATIVE
        after = root / runner.ORACLE_RELATIVE
        before.write_text("JUDGE_SYSTEM = " + repr(old_prompt) + "\ndef parse_judge_response(x):\n return x\n", encoding="utf-8")
        after.write_text("JUDGE_SYSTEM = " + repr(oracle.JUDGE_SYSTEM) + "\ndef parse_judge_response(x):\n return x\n", encoding="utf-8")
        cases = [{"id": "K01"}, {"id": "K02"}, {"id": "K03"}]
        template = runner.read_json(old / "K01.json")
        rows = []
        for case, status in zip(cases, ["pass", "fail", "unscored"]):
            row = {**template, "case_id": case["id"], "hard": {"input_valid": True, "passed": status != "fail", "checks": {}},
                   "outcome": {"status": status}, "semantic": {"status": status}, "evidence": {"kept": case["id"]}}
            rows.append(row)
            runner.save_state(old / (case["id"] + ".json"), row)
            runner.save_state(old / (case["id"] + ".judge-input.json"), [{"role": "system", "content": old_prompt},
                                                                        {"role": "user", "content": "original-" + case["id"]}])
        manifest = runner.read_json(old / "manifest.json")
        manifest["case_ids"] = [case["id"] for case in cases]
        manifest["source_sha256"][runner.ORACLE_RELATIVE] = runner.digest(before)
        runner.save_state(old / "manifest.json", manifest)
        runner.save_state(old / "results.json", rows)
        frozen[runner.ORACLE_RELATIVE] = runner.digest(after)
        for name, value in (("ROOT", root), ("LOCAL", local)):
            stack.enter_context(patch.object(runner, name, value))
        stack.enter_context(patch.object(runner, "source_paths", return_value=[current, after]))
        stack.enter_context(patch.object(runner, "fingerprints", return_value=frozen))
        hard = stack.enter_context(patch.object(oracle, "deterministic_checks", return_value={"input_valid": True, "passed": True, "checks": {}}))
        packets = stack.enter_context(patch.object(oracle, "build_judge_messages", side_effect=lambda row: [
            {"role": "system", "content": oracle.JUDGE_SYSTEM}, {"role": "user", "content": "recomputed-" + row["case_id"]}]))
        return root, local, old, cases, frozen, hard, packets

    def test_uniform_upgrade_preserves_every_old_verdict_and_execution(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root, local, old, cases, frozen, hard, packets = self.fixture(directory, stack)
            old_hashes = {str(p): runner.digest(p) for p in old.rglob("*") if p.is_file()}
            folder, manifest, rows = runner.make_format_upgrade(old, cases, frozen, runner.price_snapshot(), runner.Budget(local / "budget.json"), 150)
            self.assertEqual(hard.call_count, 3)
            self.assertEqual(packets.call_count, 3)
            self.assertEqual([r["semantic_history"][0]["outcome"]["status"] for r in rows], ["pass", "fail", "unscored"])
            self.assertTrue(all(r["outcome"]["status"] == "unscored" for r in rows))
            self.assertTrue(all(r["hard"]["passed"] for r in rows))
            self.assertFalse(rows[1]["hard_history"][0]["hard"]["passed"])
            self.assertEqual([r["evidence"] for r in rows], [{"kept": c["id"]} for c in cases])
            self.assertTrue(all(r["agent_execution_provenance"]["reexecuted"] is False for r in rows))
            self.assertEqual(manifest["semantic_regrade"]["planned_case_ids"], ["K01", "K02", "K03"])
            self.assertEqual(old_hashes, {str(p): runner.digest(p) for p in old.rglob("*") if p.is_file()})

    def test_format_suffix_cannot_change_rubric_or_parser(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root, local, old, cases, frozen, _, _ = self.fixture(directory, stack)
            review = runner.format_upgrade_review(old, frozen)
            prior = runner.read_json(old / "manifest.json")["source_sha256"]
            self.assertEqual(len(runner.source_change_review(prior, frozen, old / "frozen", review)), 2)
            path = root / runner.ORACLE_RELATIVE
            path.write_text(path.read_text(encoding="utf-8").replace("return x", "return None"), encoding="utf-8")
            frozen[runner.ORACLE_RELATIVE] = runner.digest(path)
            review = runner.format_upgrade_review(old, frozen)
            with self.assertRaisesRegex(runner.BootstrapError, "rubric, hard checks and parser"):
                runner.source_change_review(prior, frozen, old / "frozen", review)

    def test_regrade_all_labels_once_and_resume_without_duplicates(self):
        from evaluation import task_success as oracle
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            _, local, old, cases, frozen, _, _ = self.fixture(directory, stack)
            budget = runner.Budget(local / "budget.json")
            folder, manifest, rows = runner.make_format_upgrade(old, cases, frozen, runner.price_snapshot(), budget, 150)
            verdict = {"status": "pass", "dimensions": {name: {"status": "pass", "reason": "ok", "evidence": "task.prompt"} for name in oracle.DIMENSIONS}}
            judge = stack.enter_context(patch.object(runner, "judge_once", return_value=(verdict, {"settled": True, "finish_reason": "stop"})))
            execute = stack.enter_context(patch.object(runner, "execute_task"))
            self.assertTrue(runner.rejudge_inherited(folder, manifest, rows, "dummy", [], budget, runner.price_snapshot()))
            self.assertEqual([call.args[4] for call in judge.call_args_list], ["K01", "K02", "K03"])
            self.assertEqual([call.args[0][1]["content"] for call in judge.call_args_list], ["recomputed-K01", "recomputed-K02", "recomputed-K03"])
            self.assertTrue(manifest["semantic_regrade"]["format_checkpoint_passed"])
            self.assertTrue(runner.rejudge_inherited(folder, manifest, rows, "dummy", [], budget, runner.price_snapshot()))
            self.assertEqual(judge.call_count, 3)
            execute.assert_not_called()

    def test_first_two_malformed_responses_stop_without_extra_call(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            _, local, old, cases, frozen, _, _ = self.fixture(directory, stack)
            budget = runner.Budget(local / "budget.json")
            folder, manifest, rows = runner.make_format_upgrade(old, cases, frozen, runner.price_snapshot(), budget, 150)
            judge = stack.enter_context(patch.object(runner, "judge_once", return_value=({"status": "unscored"}, {"settled": True, "finish_reason": "stop"})))
            self.assertFalse(runner.rejudge_inherited(folder, manifest, rows, "dummy", [], budget, runner.price_snapshot()))
            self.assertEqual(judge.call_count, 2)
            self.assertEqual(manifest["stop_reason"], "v1_1_initial_format_check_failed")
            self.assertEqual(rows[2]["semantic"]["reasons"], ["awaiting_v1_1_uniform_regrade"])

    def test_unknown_judge_usage_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            _, local, old, cases, frozen, _, _ = self.fixture(directory, stack)
            budget = runner.Budget(local / "budget.json")
            folder, manifest, rows = runner.make_format_upgrade(old, cases, frozen, runner.price_snapshot(), budget, 150)
            judge = stack.enter_context(patch.object(runner, "judge_once", return_value=({"status": "unscored"}, {"settled": False})))
            self.assertFalse(runner.rejudge_inherited(folder, manifest, rows, "dummy", [], budget, runner.price_snapshot()))
            self.assertEqual(judge.call_count, 1)
            self.assertEqual(manifest["stop_reason"], "judge_usage_or_admission_unresolved")
            with self.assertRaisesRegex(runner.BootstrapError, "no implicit paid retry"):
                runner.rejudge_inherited(folder, manifest, rows, "dummy", [], budget, runner.price_snapshot())
            self.assertEqual(judge.call_count, 1)


if __name__ == "__main__":
    unittest.main()
