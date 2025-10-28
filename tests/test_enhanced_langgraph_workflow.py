# -*- coding: utf-8 -*-
"""
Enhanced LangGraph Workflow 통합 테스트
긴급도 평가, 문서 분석, 법령 검증, 전문가 라우팅 기능 테스트
"""

import pytest

from source.services.langgraph.state_definitions import create_initial_legal_state
from source.services.langgraph.workflow_service import LangGraphWorkflowService
from source.utils.langgraph_config import LangGraphConfig


@pytest.fixture
def workflow_service():
    """워크플로우 서비스 생성"""
    config = LangGraphConfig.from_env()
    return LangGraphWorkflowService(config)


@pytest.mark.asyncio
async def test_urgency_assessment(workflow_service):
    """긴급도 평가 노드 테스트"""
    query = "급하게 답변 필요합니다. 내일까지 소송 서류 제출해야 합니다."
    session_id = "test_session_001"

    initial_state = create_initial_legal_state(query, session_id)
    result = await workflow_service.app.ainvoke(initial_state)

    assert "urgency_level" in result
    assert result["urgency_level"] in ["high", "critical"]
    assert result.get("emergency_type") in ["legal_deadline", "case_progress", None]

    print(f"✅ 긴급도 평가 테스트 통과: {result['urgency_level']}")


@pytest.mark.asyncio
async def test_legal_field_classification(workflow_service):
    """법률분야 분류 테스트"""
    test_cases = [
        ("이혼 소송에서 재산분할을 어떻게 하나요?", "family"),
        ("회사 주주총회 절차가 궁금합니다.", "corporate"),
        ("특허 출원 절차를 알려주세요.", "intellectual_property"),
        ("계약서를 검토해주세요.", "civil"),
    ]

    for query, expected_field in test_cases:
        session_id = f"test_session_{expected_field}"
        initial_state = create_initial_legal_state(query, session_id)
        result = await workflow_service.app.ainvoke(initial_state)

        assert "legal_field" in result
        assert result["legal_field"] == expected_field or result["legal_field"] in ["general", "civil"]

    print("✅ 법률분야 분류 테스트 통과")


@pytest.mark.asyncio
async def test_document_analysis(workflow_service):
    """문서 분석 노드 테스트"""
    contract = """
    제1조 (목적) 본 계약은...
    제2조 (대금) 계약 금액은...
    """

    query = "이 계약서를 검토해주세요"
    session_id = "test_session_doc_001"

    initial_state = create_initial_legal_state(query, session_id)
    initial_state["uploaded_document"] = contract

    result = await workflow_service.app.ainvoke(initial_state)

    assert "document_type" in result
    assert result.get("document_type") == "contract"
    assert "document_analysis" in result

    print(f"✅ 문서 분석 테스트 통과: {result.get('document_type')}")


@pytest.mark.asyncio
async def test_complexity_assessment(workflow_service):
    """복잡도 평가 및 전문가 라우팅 테스트"""
    complex_query = """
    복잡한 이혼 소송에서 양육권과 재산분할을 어떻게 진행해야 하나요?
    전 부인이 아이 양육권을 요구하고 있고, 공동부채도 많습니다.
    재산은 아파트 한 채와 예금이 있는데 어떻게 나눠야 할까요?
    """

    session_id = "test_session_complex_001"
    initial_state = create_initial_legal_state(complex_query, session_id)

    result = await workflow_service.app.ainvoke(initial_state)

    assert "complexity_level" in result
    assert "requires_expert" in result
    assert result["legal_field"] == "family"
    assert result["complexity_level"] in ["medium", "complex"]

    print(f"✅ 복잡도 평가 테스트 통과: {result['complexity_level']}")


@pytest.mark.asyncio
async def test_workflow_complete_flow(workflow_service):
    """전체 워크플로우 통합 테스트"""
    query = "민법 제750조에 따른 손해배상 청구가 가능한가요?"
    session_id = "test_session_complete_001"

    initial_state = create_initial_legal_state(query, session_id)
    result = await workflow_service.app.ainvoke(initial_state)

    # 기본 필드 확인
    assert "answer" in result
    assert "legal_field" in result
    assert "urgency_level" in result
    assert "complexity_level" in result
    assert "legal_validity_check" in result

    # 답변 품질 확인
    assert len(result["answer"]) > 50

    print("✅ 전체 워크플로우 통합 테스트 통과")
    print(f"   - 법률분야: {result['legal_field']}")
    print(f"   - 긴급도: {result['urgency_level']}")
    print(f"   - 복잡도: {result['complexity_level']}")
    print(f"   - 법령검증: {result['legal_validity_check']}")


@pytest.mark.asyncio
async def test_multi_turn_conversation(workflow_service):
    """멀티턴 대화 테스트"""
    # 첫 번째 턴
    query1 = "손해배상 청구 가능하나요?"
    session_id = "test_session_multi_001"

    initial_state1 = create_initial_legal_state(query1, session_id)
    result1 = await workflow_service.app.ainvoke(initial_state1)

    assert "resolved_query" in result1

    # 두 번째 턴 (대명사 사용)
    query2 = "그 절차는 어떻게 되나요?"
    initial_state2 = create_initial_legal_state(query2, session_id)
    result2 = await workflow_service.app.ainvoke(initial_state2)

    assert "is_multi_turn" in result2
    assert result2.get("conversation_history") is not None

    print("✅ 멀티턴 대화 테스트 통과")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
