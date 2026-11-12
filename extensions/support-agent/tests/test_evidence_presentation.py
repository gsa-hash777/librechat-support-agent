"""Evidence coverage is actionable without exposing internal retrieval material."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.knowledge_ui import assessment_html
from app.retrieval import retrieve
from test_service import login


@pytest.fixture
def settings(settings):
    return replace(settings, retrieval_enabled=True)


def test_partial_evidence_shows_missing_fact_and_source_without_raw_internals(client, app, users):
    def engine(chunks, query, mode, limit):
        result = retrieve(chunks, "忘记密码如何重置", limit)
        result.update(evidence_assessment={
            "coverage": "partial",
            "supported_claims": [{"requirement": "reset", "label": "重置方式", "evidence": []}],
            "missing_information": ["企业版套餐的具体价格尚无资料。"],
            "clarification_questions": ["你使用的是标准版还是企业版？"],
            "missing_fields": ["internal_dimension_marker"],
            "raw": "PRIVATE_ASSESSMENT_DEBUG",
        }, ranked_results=[{"content": "PRIVATE_UNGATED_CANDIDATE"}])
        return result

    app.state.retrieval_engine = SimpleNamespace(retrieve=engine)
    login(client, users)
    response = client.get("/console/knowledge", params={"q": "忘记密码如何重置及套餐价格", "mode": "hybrid"})
    assert response.status_code == 200
    for text in ("仅部分问题找到证据", "企业版套餐的具体价格尚无资料。", "你使用的是标准版还是企业版？", "通过邮箱验证重置密码", "不是答案正确概率"):
        assert text in response.text
    assert '/console/knowledge/reset-password?revision=1#' in response.text
    for internal in ("PRIVATE_ASSESSMENT_DEBUG", "PRIVATE_UNGATED_CANDIDATE", "internal_dimension_marker", "supported_claims", "ranked_results"):
        assert internal not in response.text


def test_clarification_ui_shows_specific_minimal_question(client, app, users):
    app.state.retrieval_engine = SimpleNamespace(retrieve=lambda *args, **kwargs: {
        "results": [], "evidence_status": "clarify", "clarification": "请补充遇到问题的功能和现象。",
        "evidence_assessment": {"coverage": "none", "missing_information": [],
                                "clarification_questions": ["你在哪个功能遇到了什么报错？"],
                                "missing_fields": ["object", "symptom"]},
    })
    login(client, users)
    response = client.get("/console/knowledge", params={"q": "这个怎么办", "mode": "hybrid"})
    assert response.status_code == 200
    assert "你在哪个功能遇到了什么报错？" in response.text
    assert "object" not in response.text and "symptom" not in response.text


def test_presentation_escapes_untrusted_text_and_ignores_non_public_fields():
    rendered = assessment_html({"evidence_assessment": {
        "coverage": "full", "missing_information": ['<img src=x onerror="alert(1)">'],
        "clarification_questions": ["普通追问", "普通追问", {"raw": "HIDDEN_OBJECT"}],
        "missing_fields": ["HIDDEN_INTERNAL_FIELD"], "raw": "HIDDEN_DEBUG", "probability": 1,
    }})
    assert '<img' not in rendered and '&lt;img' in rendered
    assert rendered.count("普通追问") == 1
    assert "HIDDEN_" not in rendered and "100%" not in rendered
    assert "不是答案正确概率" in rendered
    assert assessment_html({}) == ""
    assert assessment_html({"evidence_assessment": []}) == ""


@pytest.mark.parametrize("scope,message,excluded", [
    ("no_precise_fact_need_detected_candidates_require_verification", "找到相关候选，尚未逐项验证", "已识别的信息项找到对应原文"),
    ("heuristic_information_needs_not_semantic_entailment", "已识别的信息项找到对应原文", "找到相关候选，尚未逐项验证"),
])
def test_full_coverage_console_explains_actual_assessment_scope(client, app, users, scope, message, excluded):
    app.state.retrieval_engine = SimpleNamespace(retrieve=lambda *args, **kwargs: {
        "results": [], "evidence_status": "candidates",
        "evidence_assessment": {"coverage": "full", "assessment_scope": scope},
    })
    login(client, users)
    response = client.get("/console/knowledge", params={"q": "如何检查资料", "mode": "hybrid"})
    assert response.status_code == 200
    assert message in response.text and excluded not in response.text
    assert "不是答案正确概率" in response.text
    assert "本次问题各项均找到对应证据" not in response.text
    assert scope not in response.text
