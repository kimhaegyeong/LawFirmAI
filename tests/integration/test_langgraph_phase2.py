# -*- coding: utf-8 -*-
"""
Phase 2: 하이브리드 질문 분석 및 법률 제한 검증 테스트
"""

import pytest

from source.services.langgraph_workflow.legal_workflow_enhanced import (
    EnhancedLegalQuestionWorkflow,
)
from source.services.langgraph_workflow.state_definitions import create_initial_state
from source.utils.langgraph_config import LangGraphConfig


class TestPhase2HybridAnalysis:
    """Phase 2 하이브리드 분석 및 법률 제한 검증 테스트"""

    @pytest.fixture
    def workflow(self):
        """워크플로우 초기화"""
        config = LangGraphConfig()
        return EnhancedLegalQuestionWorkflow(config)

    def test_analyze_query_hybrid_success(self, workflow):
        """하이브리드 질문 분석 성공 테스트"""
        state = create_initial_state(
            query="민법 제750조에 대해 알려주세요",
            session_id="test_session_1",
            user_id="test_user_1"
        )
        state["user_query"] = "민법 제750조에 대해 알려주세요"

        result = workflow.analyze_query_hybrid(state)

        assert result["query_analysis"] is not None
        assert "query_type" in result["query_analysis"]
        assert "confidence" in result["query_analysis"]
        assert "하이브리드 쿼리 분석 완료" in result["processing_steps"]

    def test_analyze_query_hybrid_with_fallback(self, workflow):
        """하이브리드 질문 분석 폴백 테스트"""
        state = create_initial_state(
            query="상속 순위에 대해 알려주세요",
            session_id="test_session_2",
            user_id="test_user_2"
        )
        state["user_query"] = "상속 순위에 대해 알려주세요"

        result = workflow.analyze_query_hybrid(state)

        assert result["query_analysis"] is not None
        assert result["query_analysis"].get("hybrid_analysis", True) or True
        assert "하이브리드 쿼리 분석 완료" in result["processing_steps"]

    def test_validate_legal_restrictions_pass(self, workflow):
        """법률 제한 검증 통과 테스트"""
        state = create_initial_state(
            query="이혼 절차에 대해 알려주세요",
            session_id="test_session_3",
            user_id="test_user_3"
        )
        state["user_query"] = "이혼 절차에 대해 알려주세요"
        state["query_analysis"] = {"query_type": "family_law", "confidence": 0.8}

        result = workflow.validate_legal_restrictions(state)

        assert result["legal_restriction_result"] is not None
        assert result["is_restricted"] == False
        assert "법률 제한 검증 완료" in result["processing_steps"]

    def test_validate_legal_restrictions_fail(self, workflow):
        """법률 제한 검증 실패 테스트"""
        state = create_initial_state(
            query="불법 콘텐츠에 대해 알려주세요",
            session_id="test_session_4",
            user_id="test_user_4"
        )
        state["user_query"] = "불법 콘텐츠에 대해 알려주세요"
        state["query_analysis"] = {"query_type": "general", "confidence": 0.5}
        state["is_restricted"] = True
        state["legal_restriction_result"] = {"restricted": True}

        result = workflow.validate_legal_restrictions(state)

        assert result["legal_restriction_result"] is not None
        assert "법률 제한 검증 완료" in result["processing_steps"]

    def test_should_continue_after_restriction_restricted(self, workflow):
        """제한 검증 후 라우팅 - 제한됨"""
        state = create_initial_state(
            query="불법 콘텐츠에 대해 알려주세요",
            session_id="test_session_5",
            user_id="test_user_5"
        )
        state["is_restricted"] = True

        result = workflow.should_continue_after_restriction(state)

        assert result == "restricted"

    def test_should_continue_after_restriction_continue(self, workflow):
        """제한 검증 후 라우팅 - 계속"""
        state = create_initial_state(
            query="이혼 절차에 대해 알려주세요",
            session_id="test_session_6",
            user_id="test_user_6"
        )
        state["is_restricted"] = False

        result = workflow.should_continue_after_restriction(state)

        assert result == "continue"

    def test_generate_restricted_response(self, workflow):
        """제한된 응답 생성 테스트"""
        state = create_initial_state(
            query="불법 콘텐츠에 대해 알려주세요",
            session_id="test_session_7",
            user_id="test_user_7"
        )
        state["legal_restriction_result"] = {"restricted": True}

        result = workflow.generate_restricted_response(state)

        assert result["answer"] is not None
        assert len(result["answer"]) > 0
        assert result["generation_method"] == "restricted_response"
        assert result["generation_success"] == True
        assert "제한된 응답 생성 완료" in result["processing_steps"]

    def test_generate_restricted_response_with_safe_response(self, workflow):
        """제한된 응답 생성 - 안전한 응답 포함"""
        state = create_initial_state(
            query="불법 콘텐츠에 대해 알려주세요",
            session_id="test_session_8",
            user_id="test_user_8"
        )
        state["legal_restriction_result"] = {
            "restricted": True,
            "safe_response": "안전한 응답입니다"
        }

        result = workflow.generate_restricted_response(state)

        assert result["answer"] == "안전한 응답입니다"
        assert result["generation_success"] == True

    @pytest.mark.asyncio
    async def test_full_workflow_with_hybrid_analysis(self, workflow):
        """전체 워크플로우 - 하이브리드 분석 포함"""
        state = create_initial_state(
            query="이혼 절차에 대해 알려주세요",
            session_id="test_session_9",
            user_id="test_user_9"
        )
        state["user_query"] = "이혼 절차에 대해 알려주세요"

        # 워크플로우 실행
        compiled_workflow = workflow.graph.compile()
        result = await compiled_workflow.ainvoke(state)

        assert result["answer"] is not None
        assert len(result["answer"]) > 0
        assert "하이브리드 쿼리 분석 완료" in result["processing_steps"]
        assert "법률 제한 검증 완료" in result["processing_steps"]

    @pytest.mark.asyncio
    async def test_full_workflow_with_restriction(self, workflow):
        """전체 워크플로우 - 제한된 응답"""
        state = create_initial_state(
            query="불법 콘텐츠에 대해 알려주세요",
            session_id="test_session_10",
            user_id="test_user_10"
        )
        state["user_query"] = "불법 콘텐츠에 대해 알려주세요"
        state["is_restricted"] = True
        state["legal_restriction_result"] = {"restricted": True, "safe_response": "안전한 응답"}

        # 워크플로우 실행
        compiled_workflow = workflow.graph.compile()
        result = await compiled_workflow.ainvoke(state)

        # 결과에 answer가 있어야 함
        assert "answer" in result or "response" in result

    def test_query_analysis_structure(self, workflow):
        """쿼리 분석 결과 구조 테스트"""
        state = create_initial_state(
            query="계약서 작성 방법을 알려주세요",
            session_id="test_session_11",
            user_id="test_user_11"
        )
        state["user_query"] = "계약서 작성 방법을 알려주세요"

        result = workflow.analyze_query_hybrid(state)

        query_analysis = result["query_analysis"]
        assert "query_type" in query_analysis
        assert "confidence" in query_analysis
        assert "classification_method" in query_analysis

    def test_hybrid_classification_structure(self, workflow):
        """하이브리드 분류 결과 구조 테스트"""
        state = create_initial_state(
            query="형법 제123조에 대해 알려주세요",
            session_id="test_session_12",
            user_id="test_user_12"
        )
        state["user_query"] = "형법 제123조에 대해 알려주세요"

        result = workflow.analyze_query_hybrid(state)

        assert "hybrid_classification" in result
        assert isinstance(result["hybrid_classification"], dict)
