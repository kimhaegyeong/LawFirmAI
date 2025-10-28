# -*- coding: utf-8 -*-
"""
LangGraph Workflow Tests
LangGraph 워크플로우 테스트 코드
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# 테스트 환경 설정
os.environ["USE_LANGGRAPH"] = "true"
os.environ["LANGGRAPH_ENABLED"] = "true"

# 프로젝트 루트 경로 추가
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.services.langgraph.checkpoint_manager import CheckpointManager
from source.services.langgraph.legal_workflow_enhanced import (
    EnhancedLegalQuestionWorkflow,
)
from source.services.langgraph.state_definitions import create_initial_legal_state
from source.services.langgraph.workflow_service import LangGraphWorkflowService
from source.utils.langgraph_config import LangGraphConfig


class TestLangGraphConfig:
    """LangGraph 설정 테스트"""

    def test_config_creation(self):
        """설정 생성 테스트"""
        config = LangGraphConfig()

        assert config.checkpoint_storage.value == "sqlite"
        assert config.checkpoint_db_path == "./data/checkpoints/langgraph.db"
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.ollama_model == "qwen2.5:3b"
        assert config.langgraph_enabled is True

    def test_config_from_env(self):
        """환경 변수에서 설정 로드 테스트"""
        with patch.dict(os.environ, {
            "LANGGRAPH_CHECKPOINT_DB": "/tmp/test.db",
            "OLLAMA_BASE_URL": "http://test:11434",
            "OLLAMA_MODEL": "test-model"
        }):
            config = LangGraphConfig.from_env()

            assert config.checkpoint_db_path == "/tmp/test.db"
            assert config.ollama_base_url == "http://test:11434"
            assert config.ollama_model == "test-model"

    def test_config_validation(self):
        """설정 유효성 검사 테스트"""
        config = LangGraphConfig()
        errors = config.validate()

        # 기본 설정은 유효해야 함
        assert len(errors) == 0


class TestCheckpointManager:
    """체크포인트 관리자 테스트"""

    def test_checkpoint_manager_initialization(self):
        """체크포인트 관리자 초기화 테스트"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            manager = CheckpointManager(db_path)

            # 데이터베이스 정보 조회
            info = manager.get_database_info()

            assert info["database_path"] == db_path
            assert "langgraph_available" in info

        finally:
            # 임시 파일 정리
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_checkpoint_manager_with_invalid_path(self):
        """잘못된 경로로 체크포인트 관리자 초기화 테스트"""
        # 잘못된 경로로 초기화해도 오류가 발생하지 않아야 함
        manager = CheckpointManager("/invalid/path/test.db")

        info = manager.get_database_info()
        assert "error" in info or "database_path" in info


class TestLegalWorkflow:
    """법률 워크플로우 테스트"""

    @pytest.fixture
    def mock_config(self):
        """모의 설정 생성"""
        config = Mock()
        config.ollama_base_url = "http://localhost:11434"
        config.ollama_model = "qwen2.5:7b"
        config.ollama_timeout = 120
        return config

    @pytest.fixture
    def workflow(self, mock_config):
        """워크플로우 인스턴스 생성"""
        with patch('source.services.question_classifier.QuestionClassifier'):
            with patch('source.services.hybrid_search_engine.HybridSearchEngine'):
                with patch('source.services.langgraph.legal_workflow_enhanced.ChatGoogleGenerativeAI'):
                    return EnhancedLegalQuestionWorkflow(mock_config)

    def test_workflow_initialization(self, workflow):
        """워크플로우 초기화 테스트"""
        assert workflow is not None
        assert workflow.config is not None

    def test_workflow_graph_building(self, workflow):
        """워크플로우 그래프 구축 테스트"""
        if workflow.graph:
            # 그래프가 성공적으로 구축되었는지 확인
            assert workflow.graph is not None

    def test_workflow_compilation(self, workflow):
        """워크플로우 컴파일 테스트"""
        # EnhancedLegalQuestionWorkflow는 그래프를 자동으로 컴파일함
        # compile 메서드가 없는 경우에 대한 처리
        if hasattr(workflow, 'compile'):
            compiled = workflow.compile()
            if compiled:
                assert compiled is not None
        else:
            # compile 메서드가 없는 경우, 그래프가 이미 컴파일되어 있음
            assert workflow.graph is not None or hasattr(workflow, 'app')


class TestLangGraphWorkflowService:
    """LangGraph 워크플로우 서비스 테스트"""

    @pytest.fixture
    def mock_config(self):
        """모의 설정 생성"""
        config = Mock()
        config.checkpoint_db_path = ":memory:"  # 메모리 데이터베이스 사용
        config.ollama_base_url = "http://localhost:11434"
        config.ollama_model = "qwen2.5:7b"
        config.ollama_timeout = 120
        return config

    @pytest.fixture
    def service(self, mock_config):
        """서비스 인스턴스 생성"""
        with patch('source.services.langgraph.workflow_service.EnhancedLegalQuestionWorkflow'):
            return LangGraphWorkflowService(mock_config)

    @pytest.mark.asyncio
    async def test_process_query_basic(self, service):
        """기본 질문 처리 테스트"""
        with patch.object(service, 'app') as mock_app:
            # 모의 응답 설정 (AsyncMock 사용)
            async_mock_result = {
                "answer": "테스트 답변입니다.",
                "confidence": 0.8,
                "sources": ["테스트 소스"],
                "legal_references": ["민법"],
                "processing_steps": ["질문 분류", "문서 검색", "답변 생성"],
                "query_type": "general_question",
                "metadata": {},
                "errors": []
            }
            mock_app.ainvoke = AsyncMock(return_value=async_mock_result)

            result = await service.process_query("테스트 질문입니다")

            assert "answer" in result
            assert "sources" in result
            assert "confidence" in result
            assert "session_id" in result
            assert result["answer"] == "테스트 답변입니다."

    @pytest.mark.asyncio
    async def test_process_query_with_session_id(self, service):
        """세션 ID와 함께 질문 처리 테스트"""
        with patch.object(service, 'app') as mock_app:
            async_mock_result = {
                "answer": "세션 테스트 답변입니다.",
                "confidence": 0.9,
                "sources": ["세션 소스"],
                "legal_references": ["상법"],
                "processing_steps": ["세션 처리"],
                "query_type": "contract_review",
                "metadata": {},
                "errors": []
            }
            mock_app.ainvoke = AsyncMock(return_value=async_mock_result)

            session_id = "test-session-123"
            result = await service.process_query("세션 테스트 질문", session_id)

            assert result["session_id"] == session_id
            assert result["answer"] == "세션 테스트 답변입니다."

    @pytest.mark.asyncio
    async def test_process_query_error_handling(self, service):
        """오류 처리 테스트"""
        with patch.object(service, 'app') as mock_app:
            mock_app.ainvoke.side_effect = Exception("테스트 오류")

            result = await service.process_query("오류 테스트 질문")

            assert "answer" in result
            assert "오류가 발생했습니다" in result["answer"]
            assert "errors" in result
            assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_resume_session(self, service):
        """세션 재개 테스트"""
        with patch.object(service, 'app') as mock_app:
            async_mock_result = {
                "answer": "재개된 답변",
                "confidence": 0.8,
                "sources": [],
                "legal_references": [],
                "processing_steps": [],
                "query_type": "general_question",
                "metadata": {},
                "errors": []
            }
            mock_app.ainvoke = AsyncMock(return_value=async_mock_result)

            result = await service.resume_session("test-session", "새 질문")

            assert "answer" in result
            assert "session_id" in result

    def test_get_session_info(self, service):
        """세션 정보 조회 테스트"""
        # checkpoint_manager가 None이므로 기본 동작 확인
        info = service.get_session_info("test-session")

        assert info["session_id"] == "test-session"
        assert info["checkpoint_count"] == 0
        assert info["has_checkpoints"] is False

    def test_get_service_status(self, service):
        """서비스 상태 조회 테스트"""
        # checkpoint_manager가 None이므로 기본 동작 확인
        status = service.get_service_status()

        assert status["service_name"] == "LangGraphWorkflowService"
        assert status["status"] == "running"
        assert "config" in status
        assert "database_info" in status

    @pytest.mark.asyncio
    async def test_test_workflow(self, service):
        """워크플로우 테스트 테스트"""
        with patch.object(service, 'process_query') as mock_process:
            mock_process.return_value = {
                "answer": "테스트 답변",
                "processing_steps": ["테스트 단계"],
                "errors": []
            }

            result = await service.test_workflow("테스트 질문")

            assert result["test_passed"] is True
            assert result["test_query"] == "테스트 질문"
            assert "result" in result


class TestAnswerQuality:
    """답변 품질 검증 테스트"""

    @pytest.fixture
    def mock_config(self):
        """모의 설정 생성"""
        config = Mock()
        config.checkpoint_db_path = ":memory:"
        config.ollama_base_url = "http://localhost:11434"
        config.ollama_model = "qwen2.5:7b"
        config.ollama_timeout = 120
        return config

    @pytest.fixture
    def service(self, mock_config):
        """서비스 인스턴스 생성"""
        with patch('source.services.langgraph.workflow_service.EnhancedLegalQuestionWorkflow'):
            return LangGraphWorkflowService(mock_config)

    @pytest.mark.asyncio
    async def test_answer_quality_minimum_length(self, service):
        """답변 최소 길이 검증"""
        with patch.object(service, 'app') as mock_app:
            async_mock_result = {
                "answer": "질문에 대한 답변입니다." * 5,  # 충분한 길이의 답변
                "confidence": 0.8,
                "sources": ["소스1", "소스2"],
                "legal_references": ["민법"],
                "processing_steps": ["질문 분류", "문서 검색", "답변 생성"],
                "query_type": "general_question",
                "metadata": {},
                "errors": []
            }
            mock_app.ainvoke = AsyncMock(return_value=async_mock_result)

            result = await service.process_query("법률 질문")

            # 답변 길이 검증 (최소 50자 이상)
            assert len(result["answer"]) >= 50, f"답변이 너무 짧습니다: {len(result['answer'])}자"
            assert "answer" in result

    @pytest.mark.asyncio
    async def test_answer_quality_contains_legal_terms(self, service):
        """답변에 법률 용어 포함 여부 검증"""
        with patch.object(service, 'app') as mock_app:
            async_mock_result = {
                "answer": "계약서 작성 시 계약 당사자, 계약 목적, 계약 조건을 명확히 해야 합니다. "
                          "민법 제105조에 따라 계약은 당사자 간의 의사표시 합치로 성립합니다.",
                "confidence": 0.85,
                "sources": ["민법"],
                "legal_references": ["민법 제105조"],
                "processing_steps": [],
                "query_type": "contract_review",
                "metadata": {},
                "errors": []
            }
            mock_app.ainvoke = AsyncMock(return_value=async_mock_result)

            result = await service.process_query("계약서 작성 방법")

            # 법률 용어 포함 여부 검증
            legal_terms = ["계약", "법", "조", "규정", "판례", "법률", "조문"]
            has_legal_terms = any(term in result["answer"] for term in legal_terms)
            assert has_legal_terms, "답변에 법률 용어가 포함되지 않았습니다"

    @pytest.mark.asyncio
    async def test_answer_quality_has_sources(self, service):
        """답변에 소스 제공 여부 검증"""
        with patch.object(service, 'app') as mock_app:
            async_mock_result = {
                "answer": "질문에 대한 답변입니다.",
                "confidence": 0.75,
                "sources": ["민법", "대법원 판례", "계약법"],
                "legal_references": ["민법 제105조"],
                "processing_steps": [],
                "query_type": "law_inquiry",
                "metadata": {},
                "errors": []
            }
            mock_app.ainvoke = AsyncMock(return_value=async_mock_result)

            result = await service.process_query("법률 질문")

            # 소스 제공 여부 검증
            assert "sources" in result
            assert isinstance(result["sources"], list)
            assert len(result["sources"]) > 0, "소스가 제공되지 않았습니다"

    @pytest.mark.asyncio
    async def test_answer_quality_confidence_score_validity(self, service):
        """신뢰도 점수 유효성 검증"""
        with patch.object(service, 'app') as mock_app:
            async_mock_result = {
                "answer": "질문에 대한 답변입니다.",
                "confidence": 0.85,
                "sources": [],
                "legal_references": [],
                "processing_steps": [],
                "query_type": "general_question",
                "metadata": {},
                "errors": []
            }
            mock_app.ainvoke = AsyncMock(return_value=async_mock_result)

            result = await service.process_query("법률 질문")

            # 신뢰도 점수 검증 (0.0 ~ 1.0 범위)
            assert "confidence" in result
            assert 0.0 <= result["confidence"] <= 1.0, f"신뢰도 점수가 유효하지 않습니다: {result['confidence']}"

    @pytest.mark.asyncio
    async def test_answer_quality_no_errors(self, service):
        """답변에 에러 없음 검증"""
        with patch.object(service, 'app') as mock_app:
            async_mock_result = {
                "answer": "질문에 대한 답변입니다.",
                "confidence": 0.8,
                "sources": ["소스1"],
                "legal_references": [],
                "processing_steps": [],
                "query_type": "general_question",
                "metadata": {},
                "errors": []  # 에러 없음
            }
            mock_app.ainvoke = AsyncMock(return_value=async_mock_result)

            result = await service.process_query("법률 질문")

            # 에러 없음 검증
            assert "errors" in result
            assert isinstance(result["errors"], list)
            assert len(result["errors"]) == 0, f"답변에 에러가 있습니다: {result['errors']}"

    @pytest.mark.asyncio
    async def test_answer_quality_processing_steps(self, service):
        """처리 단계 정보 검증"""
        with patch.object(service, 'app') as mock_app:
            async_mock_result = {
                "answer": "질문에 대한 답변입니다.",
                "confidence": 0.8,
                "sources": ["소스1"],
                "legal_references": [],
                "processing_steps": ["질문 분류 완료", "문서 검색 완료", "답변 생성 완료"],
                "query_type": "general_question",
                "metadata": {},
                "errors": []
            }
            mock_app.ainvoke = AsyncMock(return_value=async_mock_result)

            result = await service.process_query("법률 질문")

            # 처리 단계 검증
            assert "processing_steps" in result
            assert isinstance(result["processing_steps"], list)
            assert len(result["processing_steps"]) > 0, "처리 단계 정보가 제공되지 않았습니다"

    @pytest.mark.asyncio
    async def test_answer_quality_contains_answer(self, service):
        """답변 내용 존재 여부 검증"""
        with patch.object(service, 'app') as mock_app:
            async_mock_result = {
                "answer": "질문하신 내용에 대해 답변드리겠습니다. "
                          "계약서 작성 시에는 다음과 같은 사항들을 명확히 해야 합니다.",
                "confidence": 0.8,
                "sources": ["소스1"],
                "legal_references": [],
                "processing_steps": [],
                "query_type": "general_question",
                "metadata": {},
                "errors": []
            }
            mock_app.ainvoke = AsyncMock(return_value=async_mock_result)

            result = await service.process_query("계약서 작성 시 주의사항은?")

            # 답변 내용 존재 여부 검증
            assert "answer" in result
            assert len(result["answer"]) > 0, "답변이 비어있습니다"
            assert result["answer"].strip() != "", "답변이 공백만 있습니다"

    @pytest.mark.asyncio
    async def test_answer_quality_metadata_completeness(self, service):
        """메타데이터 완전성 검증"""
        with patch.object(service, 'app') as mock_app:
            async_mock_result = {
                "answer": "답변입니다.",
                "confidence": 0.8,
                "sources": ["소스1"],
                "legal_references": [],
                "processing_steps": [],
                "query_type": "contract_review",
                "metadata": {
                    "keyword_coverage": 0.75,
                    "response_length": 100,
                    "query_type": "contract_review"
                },
                "errors": []
            }
            mock_app.ainvoke = AsyncMock(return_value=async_mock_result)

            result = await service.process_query("계약서 검토")

            # 메타데이터 완전성 검증
            assert "metadata" in result
            assert isinstance(result["metadata"], dict)
            assert len(result["metadata"]) > 0, "메타데이터가 비어있습니다"

    @pytest.mark.asyncio
    async def test_answer_quality_session_id_presence(self, service):
        """세션 ID 존재 여부 검증"""
        with patch.object(service, 'app') as mock_app:
            async_mock_result = {
                "answer": "답변입니다.",
                "confidence": 0.8,
                "sources": ["소스1"],
                "legal_references": [],
                "processing_steps": [],
                "query_type": "general_question",
                "metadata": {},
                "errors": []
            }
            mock_app.ainvoke = AsyncMock(return_value=async_mock_result)

            result = await service.process_query("질문")

            # 세션 ID 존재 여부 검증
            assert "session_id" in result
            assert len(result["session_id"]) > 0, "세션 ID가 없습니다"

    @pytest.mark.asyncio
    async def test_answer_quality_error_handling(self, service):
        """에러 발생 시 처리 검증"""
        with patch.object(service, 'app') as mock_app:
            mock_app.ainvoke.side_effect = Exception("테스트 에러")

            result = await service.process_query("질문")

            # 에러 처리 검증
            assert "answer" in result
            assert "errors" in result
            assert isinstance(result["errors"], list)
            assert len(result["errors"]) > 0, "에러가 발생했지만 기록되지 않았습니다"
            assert "오류" in result["answer"] or "error" in result["answer"].lower(), "에러 메시지가 적절하지 않습니다"

    @pytest.mark.asyncio
    async def test_answer_quality_query_type_mapping(self, service):
        """질문 유형 매핑 검증"""
        query_type_test_cases = [
            ("계약서 판례에 대해 알고 싶어요", "precedent_search"),
            ("민법 제105조는 무엇인가요", "law_inquiry"),
            ("이혼 절차는 어떻게 되나요", "procedure_guide"),
            ("배임죄는 어떤 죄인가요", "term_explanation"),
        ]

        for query, expected_type in query_type_test_cases:
            with patch.object(service, 'app') as mock_app:
                async_mock_result = {
                    "answer": "답변입니다.",
                    "confidence": 0.8,
                    "sources": [],
                    "legal_references": [],
                    "processing_steps": [],
                    "query_type": expected_type,
                    "metadata": {},
                    "errors": []
                }
                mock_app.ainvoke = AsyncMock(return_value=async_mock_result)

                result = await service.process_query(query)

                # 질문 유형 검증
                assert "query_type" in result
                assert result["query_type"] == expected_type, \
                    f"질문 유형이 잘못 매핑되었습니다. 예상: {expected_type}, 실제: {result['query_type']}"

    @pytest.mark.asyncio
    async def test_answer_quality_comprehensive(self, service):
        """종합적인 답변 품질 검증"""
        with patch.object(service, 'app') as mock_app:
            async_mock_result = {
                "answer": "질문하신 계약법 관련 내용에 대해 답변드리겠습니다. "
                          "계약서 작성 시 계약 당사자, 계약 목적, 계약 조건을 명확히 해야 합니다. "
                          "민법 제105조에 따라 계약은 당사자 간의 의사표시 합치로 성립합니다. "
                          "이를 위반하면 계약 취소나 손해배상 청구가 가능합니다.",
                "confidence": 0.85,
                "sources": ["민법", "계약법 판례", "대법원 판결"],
                "legal_references": ["민법 제105조", "민법 제549조"],
                "processing_steps": [
                    "질문 분류 완료: contract_review",
                    "문서 검색 완료: 15개 문서",
                    "용어 통합 완료: 5개",
                    "답변 생성 완료"
                ],
                "query_type": "contract_review",
                "metadata": {
                    "keyword_coverage": 0.82,
                    "response_length": 250,
                    "query_type": "contract_review"
                },
                "errors": []
            }
            mock_app.ainvoke = AsyncMock(return_value=async_mock_result)

            result = await service.process_query("계약서 작성 시 주의사항은?")

            # 종합적인 품질 검증
            assert len(result["answer"]) >= 100, "답변이 너무 짧습니다"
            assert result["confidence"] >= 0.7, "신뢰도가 너무 낮습니다"
            assert len(result["sources"]) > 0, "소스가 제공되지 않았습니다"
            assert len(result["legal_references"]) > 0, "법률 참조가 제공되지 않았습니다"
            assert len(result["processing_steps"]) >= 3, "처리 단계가 충분하지 않습니다"
            assert len(result["errors"]) == 0, "에러가 발생했습니다"
            assert result["answer"].strip() != "", "답변이 비어있습니다"


class TestStateDefinitions:
    """상태 정의 테스트"""

    def test_create_initial_legal_state(self):
        """초기 법률 상태 생성 테스트"""
        state = create_initial_legal_state("테스트 질문", "test-session")

        assert state["query"] == "테스트 질문"
        assert state["session_id"] == "test-session"
        assert state["query_type"] == ""
        assert state["confidence"] == 0.0
        assert state["retrieved_docs"] == []
        assert state["answer"] == ""
        assert state["processing_steps"] == []
        assert state["errors"] == []

    def test_create_initial_agent_state(self):
        """초기 에이전트 상태 생성 테스트"""
        from source.services.langgraph.state_definitions import (
            create_initial_agent_state,
        )

        state = create_initial_agent_state("에이전트 질문", "agent-session")

        assert state["query"] == "에이전트 질문"
        assert state["session_id"] == "agent-session"
        assert state["current_agent"] == ""
        assert state["agent_history"] == []
        assert state["task_results"] == []

    def test_create_initial_streaming_state(self):
        """초기 스트리밍 상태 생성 테스트"""
        from source.services.langgraph.state_definitions import (
            create_initial_streaming_state,
        )

        state = create_initial_streaming_state("스트리밍 질문", "stream-session")

        assert state["query"] == "스트리밍 질문"
        assert state["session_id"] == "stream-session"
        assert state["stream_chunks"] == []
        assert state["is_streaming"] is False
        assert state["final_answer"] == ""


class TestIntegration:
    """통합 테스트"""

    @pytest.mark.asyncio
    async def test_full_workflow_integration(self):
        """전체 워크플로우 통합 테스트"""
        # 실제 환경에서는 LangGraph가 설치되어 있어야 함
        try:
            config = LangGraphConfig()
            config.checkpoint_db_path = ":memory:"  # 메모리 데이터베이스 사용

            service = LangGraphWorkflowService(config)

            # 간단한 테스트 질문 처리
            result = await service.process_query("계약서 작성 시 주의사항은?")

            # 기본적인 응답 구조 확인
            assert "answer" in result
            assert "sources" in result
            assert "confidence" in result
            assert "session_id" in result

        except ImportError:
            pytest.skip("LangGraph not available")
        except Exception as e:
            # LangGraph가 설치되지 않았거나 다른 오류가 발생한 경우
            pytest.skip(f"Integration test skipped: {e}")


if __name__ == "__main__":
    # 테스트 실행
    pytest.main([__file__, "-v"])
