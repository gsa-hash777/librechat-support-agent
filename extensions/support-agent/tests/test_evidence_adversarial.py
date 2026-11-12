"""Independent business boundaries; no benchmark questions or model inference."""
import pytest

from app.retrieval import retrieve


def check(query, content, title):
    row = {
        "id": "independent-business-boundary", "doc_id": "independent-business-boundary",
        "chunk_id": "boundary-chunk", "revision": 1, "org_id": "demo",
        "title": title, "content": content, "source": "独立模拟业务边界",
        "start_offset": 0, "end_offset": len(content), "start_line": 1, "end_line": 1,
    }
    # Hold recall fixed so these tests constrain the policy, not BM25 ranking.
    return retrieve([row], query, ranker=lambda safe, _query, limit: safe[:limit])


def claims(result):
    return {item["requirement"]: item for item in result["evidence_assessment"]["supported_claims"]}


@pytest.mark.parametrize("query,title,content,requirement", [
    (
        "审计日志保留多久？", "审计日志与聊天记录",
        "审计日志需要业务负责人申请查看。聊天记录保留30天。", "retention",
    ),
    (
        "账号锁定后要等多久？", "账号锁定与密码重置",
        "账号锁定后，请在30分钟内使用重置链接。", "duration_lock",
    ),
    (
        "企业版单个文件上传上限多少MB？", "企业版与标准版上传",
        "标准版单个文件上传上限100MB。", "count_limit",
    ),
    (
        "企业版价格多少钱？", "企业版定价与报销",
        "企业版套餐用户申请报销时，垫付金额上限300元。", "pricing",
    ),
    (
        "是否支持本地部署？", "本地部署与热线",
        "本地部署需要管理员安排，不提供全天热线。", "private_deployment",
    ),
])
def test_precise_fact_cannot_borrow_other_subject_or_relationship(query, title, content, requirement):
    result = check(query, content, title)
    assert requirement not in claims(result)
    assert result["evidence_assessment"]["coverage"] != "full"


def test_two_link_types_have_distinct_supported_durations():
    result = check(
        "邀请链接和密码重置链接分别多久过期？",
        "邀请链接有效期72小时。密码重置链接有效期30分钟。",
        "邀请链接与密码重置链接",
    )
    supported = claims(result)
    assert "72小时" in supported["duration_invitation"]["evidence"][0]["quote"]
    assert "30分钟" in supported["duration_reset"]["evidence"][0]["quote"]


def test_partial_answer_preserves_known_procedure_when_price_is_missing():
    content = "忘记密码后通过邮箱验证重置密码。"
    result = check("忘记密码如何重置，年度价格多少？", content, "密码重置")
    assert result["evidence_status"] == "candidates"
    assert result["evidence_assessment"]["coverage"] == "partial"
    assert "pricing" in result["evidence_assessment"]["missing_requirements"]
    assert "pricing" not in claims(result)
    assert any(content in row["content"] for row in result["results"])


def test_partial_answer_retains_a_known_duration_and_declares_missing_price():
    result = check(
        "邀请链接有效期多久，年度报价多少钱？",
        "邀请链接有效期72小时。", "成员邀请",
    )
    assert result["evidence_status"] == "candidates"
    assert result["evidence_assessment"]["coverage"] == "partial"
    assert "duration_invitation" in claims(result)
    assert "pricing" in result["evidence_assessment"]["missing_requirements"]


def test_explicit_negative_deployment_fact_remains_answerable():
    result = check("是否支持本地部署？", "当前版本不支持本地部署。", "部署限制")
    assert result["evidence_status"] == "candidates"
    assert claims(result)["private_deployment"]["evidence"][0]["quote"] == "当前版本不支持本地部署。"


def test_explicit_no_guarantee_is_not_replaced_with_a_promised_time():
    result = check(
        "是否保证工单在2小时内响应？",
        "不保证所有工单在2小时内响应。", "工单响应范围",
    )
    assert result["evidence_status"] == "candidates"
    assert claims(result)["guarantee"]["evidence"][0]["quote"].startswith("不保证")


def test_known_business_object_needs_symptom_not_a_second_object_question():
    result = check("成员邀请有问题", "邀请链接有效期72小时。", "成员邀请")
    assert result["evidence_status"] == "clarify"
    assessed = result["evidence_assessment"]
    assert "symptom" in assessed["missing_fields"]
    assert "object" not in assessed["missing_fields"]
    assert 1 <= len(assessed["clarification_questions"]) <= 2


def test_observed_upload_status_is_sufficient_to_start_troubleshooting():
    result = check(
        "上传的时候提示415怎么办？",
        "HTTP 415 时核对实际格式和扩展名。", "上传排查",
    )
    assert result["evidence_status"] == "candidates"
    assert not result["evidence_assessment"]["clarification_questions"]
