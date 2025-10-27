# -*- coding: utf-8 -*-
"""
Phase 1: 입력 검증 및 특수 쿼리 감지 테스트
"""

import pytest

from source.services.langgraph_workflow.legal_workflow_enhanced import (
    EnhancedLegalQuestionWorkflow,
)
from source.services.langgraph_workflow.state_definitions import create_initial_state
from source.utils.langgraph_config import LangGraphConfig


class TestPhase1InputValidation:
    """Phase 1 입력 검증 테스트"""

    @pytest.fixture
    def workflow(self):
        """워크플로우 초기화"""
        config = LangGraphConfig()
        return EnhancedLegalQuestionWorkflow(config)

    @pytest.fixture
    def sample_state(self):
        """샘플 상태 생성"""
        return create_initial_state(
            query="민법 제750조에 대해 알려주세요",
            session_id="test_session_1",
            user_id="test_user_1"
        )

    def test_validate_input_success(self, workflow, sample_state):
        """정상 입력 검증 테스트"""
        sample_state["user_query"] = "민법 제750조에 대해 알려주세요"

        result = workflow.validate_input(sample_state)

        assert result["validation_results"]["valid"] == True
        assert len(result["errors"]) == 0
        assert "입력 검증 완료" in result["processing_steps"]

    def test_validate_input_empty(self, workflow):
        """빈 입력 검증 테스트"""
        state = create_initial_state("", "test_session_2", "test_user_2")
        state["user_query"] = ""

        result = workflow.validate_input(state)

        assert result["validation_results"]["valid"] == False
        assert len(result["errors"]) > 0
        assert "메시지가 비어있습니다" in str(result["errors"])

    def test_validate_input_too_long(self, workflow):
        """너무 긴 입력 검증 테스트"""
        state = create_initial_state(
            query="A" * 10001,
            session_id="test_session_3",
            user_id="test_user_3"
        )
        state["user_query"] = "A" * 10001

        result = workflow.validate_input(state)

        assert result["validation_results"]["valid"] == False
        assert len(result["errors"]) > 0

    def test_detect_special_queries_law_article(self, workflow):
        """법률 조문 쿼리 감지 테스트"""
        state = create_initial_state(
            query="민법 제750조에 대해 알려주세요",
            session_id="test_session_4",
            user_id="test_user_4"
        )
        state["user_query"] = "민법 제750조에 대해 알려주세요"

        result = workflow.detect_special_queries(state)

        assert result["is_law_article_query"] == True
        assert "특수 쿼리 감지 완료" in result["processing_steps"]

    def test_detect_special_queries_contract(self, workflow):
        """계약서 쿼리 감지 테스트"""
        state = create_initial_state(
            query="계약서 작성 방법을 알려주세요",
            session_id="test_session_5",
            user_id="test_user_5"
        )
        state["user_query"] = "계약서 작성 방법을 알려주세요"

        result = workflow.detect_special_queries(state)

        assert result["is_contract_query"] == True
        assert "특수 쿼리 감지 완료" in result["processing_steps"]

    def test_detect_special_queries_regular(self, workflow):
        """일반 쿼리 감지 테스트"""
        state = create_initial_state(
            query="상속 순위에 대해 알려주세요",
            session_id="test_session_6",
            user_id="test_user_6"
        )
        state["user_query"] = "상속 순위에 대해 알려주세요"

        result = workflow.detect_special_queries(state)

        assert result["is_law_article_query"] == False
        assert result["is_contract_query"] == False
        assert "특수 쿼리 감지 완료" in result["processing_steps"]

    def test_should_route_special_law_article(self, workflow):
        """법률 조문 라우팅 테스트"""
        state = create_initial_state(
            query="민법 제750조에 대해 알려주세요",
            session_id="test_session_7",
            user_id="test_user_7"
        )
        state["is_law_article_query"] = True
        state["is_contract_query"] = False

        result = workflow.should_route_special(state)

        assert result == "law_article"

    def test_should_route_special_contract(self, workflow):
        """계약서 라우팅 테스트"""
        state = create_initial_state(
            query="계약서 작성 방법을 알려주세요",
            session_id="test_session_8",
            user_id="test_user_8"
        )
        state["is_law_article_query"] = False
        state["is_contract_query"] = True

        result = workflow.should_route_special(state)

        assert result == "contract"

    def test_should_route_special_regular(self, workflow):
        """일반 라우팅 테스트"""
        state = create_initial_state(
            query="상속 순위에 대해 알려주세요",
            session_id="test_session_9",
            user_id="test_user_9"
        )
        state["is_law_article_query"] = False
        state["is_contract_query"] = False

        result = workflow.should_route_special(state)

        assert result == "regular"

    def test_handle_law_article_query(self, workflow):
        """법률 조문 쿼리 처리 테스트"""
        state = create_initial_state(
            query="민법 제750조에 대해 알려주세요",
            session_id="test_session_10",
            user_id="test_user_10"
        )
        state["user_query"] = "민법 제750조에 대해 알려주세요"

        result = workflow.handle_law_article_query(state)

        assert result["answer"] is not None
        assert len(result["answer"]) > 0
        assert result["generation_method"] == "law_article_query"
        assert result["generation_success"] == True

    def test_handle_contract_query(self, workflow):
        """계약서 쿼리 처리 테스트"""
        state = create_initial_state(
            query="계약서 작성 방법을 알려주세요",
            session_id="test_session_11",
            user_id="test_user_11"
        )
        state["user_query"] = "계약서 작성 방법을 알려주세요"

        result = workflow.handle_contract_query(state)

        assert result["answer"] is not None
        assert len(result["answer"]) > 0
        assert result["generation_method"] == "contract_query"
        assert result["generation_success"] == True

    @pytest.mark.asyncio
    async def test_full_workflow_law_article(self, workflow):
        """전체 워크플로우 - 법률 조문 쿼리"""
        state = create_initial_state(
            query="민법 제750조에 대해 알려주세요",
            session_id="test_session_12",
            user_id="test_user_12"
        )
        state["user_query"] = "민법 제750조에 대해 알려주세요"

        # 워크플로우 실행
        compiled_workflow = workflow.graph.compile()
        result = await compiled_workflow.ainvoke(state)

        assert result["answer"] is not None
        assert len(result["answer"]) > 0
        assert "입력 검증 완료" in result["processing_steps"]

    @pytest.mark.asyncio
    async def test_full_workflow_contract(self, workflow):
        """전체 워크플로우 - 계약서 쿼리"""
        state = create_initial_state(
            query="계약서 작성 방법을 알려주세요",
            session_id="test_session_13",
            user_id="test_user_13"
        )
        state["user_query"] = "계약서 작성 방법을 알려주세요"

        # 워크플로우 실행
        compiled_workflow = workflow.graph.compile()
        result = await compiled_workflow.ainvoke(state)

        assert result["answer"] is not None
        assert len(result["answer"]) > 0
        assert "계약서 쿼리 처리 완료" in result["processing_steps"]

    @pytest.mark.asyncio
    async def test_full_workflow_regular_query(self, workflow):
        """전체 워크플로우 - 일반 쿼리"""
        state = create_initial_state(
            query="상속 순위에 대해 알려주세요",
            session_id="test_session_14",
            user_id="test_user_14"
        )
        state["user_query"] = "상속 순위에 대해 알려주세요"

        # 워크플로우 실행
        compiled_workflow = workflow.graph.compile()
        result = await compiled_workflow.ainvoke(state)

        assert result["answer"] is not None
        assert len(result["answer"]) > 0
        assert "질문 분류 완료" in result["processing_steps"]
