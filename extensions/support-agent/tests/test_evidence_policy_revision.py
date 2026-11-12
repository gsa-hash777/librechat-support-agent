"""Business-language regressions independent of held-out evaluation fixtures."""
import re

import pytest

from app.evidence import needs_for
from app.retrieval import retrieve


def examine(query, content, title="独立模拟业务规则"):
    row = {
        "id": "policy-revision-source", "doc_id": "policy-revision-source",
        "chunk_id": "policy-revision-chunk", "revision": 1, "org_id": "demo",
        "title": title, "source": "独立规则边界样例", "content": content,
        "start_offset": 0, "end_offset": len(content), "start_line": 1, "end_line": 1,
    }
    # Fix retrieval to a real source row; only the evidence policy is under test.
    return retrieve([row], query, ranker=lambda safe, _query, limit: safe[:limit])


def supported(result):
    return result["evidence_assessment"]["supported_claims"]


@pytest.mark.parametrize("query,content", [
    (
        "交接处理保证多久完成？",
        "交接清单须提前3个工作日发起，这只是准备要求，不是处理时限承诺。",
    ),
    (
        "现场审阅是否保证2小时完成？",
        "现场审阅材料须提前2小时登记，登记是准备步骤，不代表审阅处理承诺。",
    ),
])
def test_preparation_lead_time_is_not_a_promised_processing_duration(query, content):
    result = examine(query, content)
    assert not any(claim["requirement"].startswith("duration") for claim in supported(result))
    assert result["evidence_assessment"]["coverage"] != "full"
    assert result["evidence_assessment"]["missing_information"]


@pytest.mark.parametrize("query,content", [
    (
        "审批多久一定完成？",
        "审批申请须提前3个工作日登记。不保证通知一定送达。",
    ),
    (
        "是否保证备份可以恢复？",
        "备份结果需要核验，不保证通知准时送达。",
    ),
    (
        "归档后保证多久可以导出资料？",
        "归档项目保留原有权限，但不保证外部附件可下载。",
    ),
])
def test_negative_guarantee_cannot_move_to_another_event(query, content):
    result = examine(query, content)
    assert not any(claim["requirement"] == "guarantee" for claim in supported(result))
    assert result["evidence_assessment"]["coverage"] != "full"


def test_same_event_explicit_commitment_remains_available():
    content = "审阅材料提交后承诺2个工作日内完成审阅。"
    result = examine("审阅材料提交后保证多久完成？", content)
    assert result["evidence_status"] == "candidates"
    assert result["evidence_assessment"]["coverage"] == "full"
    assert any(claim["requirement"] == "guarantee" for claim in supported(result))
    assert any(claim["requirement"].startswith("duration") for claim in supported(result))


def test_same_event_explicit_no_commitment_is_a_known_negative_fact():
    content = "资料审阅不承诺固定完成时限。"
    result = examine("资料审阅是否保证1小时内完成？", content)
    assert result["evidence_status"] == "candidates"
    assert any(claim["requirement"] == "guarantee" for claim in supported(result))
    assert all(item["quote"] == content for claim in supported(result) for item in claim["evidence"])


@pytest.mark.parametrize("query,content", [
    (
        "采购验收发现型号异常时应该由谁确认？",
        "采购验收发现型号不符时，由资产保管人与采购经办人共同核对。",
    ),
    (
        "业务报表发现数量异常，核对哪些筛选条件？",
        "业务报表数量不一致时，先核对项目、统计日期和状态筛选条件。",
    ),
    (
        "交接清单有问题时，规范要求先核对什么？",
        "交接清单有问题时，先核对项目编号、原负责人和接收人。",
    ),
    (
        "出现审批异常时，需要保留哪些业务材料？",
        "出现审批异常时，应保留申请编号、修订材料与原审批意见。",
    ),
])
def test_specific_business_rule_question_is_not_an_unspecified_fault(query, content):
    result = examine(query, content)
    assert result["evidence_status"] == "candidates"
    assert result["results"]
    assert not result["evidence_assessment"]["clarification_questions"]


@pytest.mark.parametrize("query,content,quote", [
    (
        "首次登记试验最多用多少条记录核对映射？",
        "首次登记试验最多使用12条记录核对字段映射，确认后再登记其余数据。",
        "12条",
    ),
    (
        "试点用于核对映射的样本记录是多少条？",
        "试点应先用12条记录核对映射。",
        "12条",
    ),
    (
        "手册示例中的名单有多少人？",
        "手册示例名单有8人，这只是展示样例，不是组织人数上限。",
        "8人",
    ),
])
def test_explicit_trial_or_example_question_can_use_that_scoped_number(query, content, quote):
    result = examine(query, content)
    assert result["evidence_status"] == "candidates"
    assert result["evidence_assessment"]["coverage"] == "full"
    assert any(quote in item["quote"] for claim in supported(result) for item in claim["evidence"])


@pytest.mark.parametrize("query,content", [
    (
        "单批登记上限是多少条记录？",
        "首次登记试验最多使用12条记录核对字段映射，确认后再登记其余数据。",
    ),
    (
        "组织成员最多可以有多少人？",
        "手册示例名单有8人，这只是展示样例，不是组织人数上限。",
    ),
])
def test_trial_or_example_number_does_not_establish_a_global_limit(query, content):
    result = examine(query, content)
    assert result["evidence_assessment"]["coverage"] != "full"
    assert not any(claim["requirement"] == "count_limit" for claim in supported(result))


@pytest.mark.parametrize("query,content", [
    ("归档资料保留多少个月？", "归档资料保留6个月。"),
    ("审计日志保存多少天？", "审计日志保存45天。"),
    ("聊天记录留存几个月？", "聊天记录留存4个月。"),
    ("业务记录最多保留多少个月？", "业务记录最多保留8个月。"),
])
def test_retention_time_units_are_not_entity_counts(query, content):
    codes = [need.code for need in needs_for(query)]
    assert "retention" in codes
    assert "count_limit" not in codes
    result = examine(query, content)
    assert result["evidence_status"] == "candidates"
    assert result["evidence_assessment"]["coverage"] == "full"
    assert any(claim["requirement"] == "retention" for claim in supported(result))


def test_people_count_is_still_a_count_after_retention_fix():
    query = "组织成员最多多少个人？"
    codes = [need.code for need in needs_for(query)]
    assert "count_limit" in codes and "retention" not in codes
    result = examine(query, "组织成员最多8个人。")
    assert result["evidence_status"] == "candidates"
    assert result["evidence_assessment"]["coverage"] == "full"


@pytest.mark.parametrize("query", [
    "还是这样，帮我处理一下。",
    "这个能继续吗？",
    "刚才那一步有问题，帮我弄好。",
    "帮忙看下为什么这样。",
    "能不能重新来一次？",
])
def test_contextless_reference_requests_minimal_context_without_secrets(query):
    result = examine(query, "采购材料由采购经办人核对。")
    assert result["evidence_status"] == "clarify"
    assert result["results"] == []
    assessment = result["evidence_assessment"]
    assert "object" in assessment["missing_fields"]
    assert set(assessment["missing_fields"]) <= {"object", "operation", "symptom"}
    assert len(assessment["missing_fields"]) <= 2
    questions = assessment["clarification_questions"]
    assert 1 <= len(questions) <= 2
    # Permit reminders not to share secrets, never a positive request for them.
    for question in questions:
        assert not re.search(r"请(?:提供|发送|输入|贴出|上传|告诉我).{0,6}(?:密码|验证码|完整密钥|API\s*Key)", question, re.I)
        assert not re.search(r"(?:不要|不得|不能|别).{0,2}(?:省略|遗漏|隐藏|隐瞒).{0,4}(?:密码|验证码|密钥)", question)


def test_concrete_error_code_does_not_need_generic_context_again():
    result = examine("上传附件提示 HTTP 415 应该怎样排查？", "HTTP 415 时核对文件实际格式与扩展名。")
    assert result["evidence_status"] == "candidates"
    assert not result["evidence_assessment"]["clarification_questions"]
