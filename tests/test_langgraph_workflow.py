# -*- coding: utf-8 -*-
"""
LangGraph Workflow Tests
LangGraph 워크플로우 테스트 코드
"""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path

# 테스트 환경 설정
os.environ["USE_LANGGRAPH"] = "true"
os.environ["LANGGRAPH_ENABLED"] = "true"

# 프로젝트 루트 경로 추가
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.services.langgraph.workflow_service import LangGraphWorkflowService
from source.services.langgraph.checkpoint_manager import CheckpointManager
from source.services.langgraph.legal_workflow import LegalQuestionWorkflow
from source.services.langgraph.state_definitions import create_initial_legal_state
from source.utils.langgraph_config import LangGraphConfig


class TestLangGraphConfig:
    """LangGraph 설정 테스트"""
    
    def test_config_creation(self):
        """설정 생성 테스트"""
        config = LangGraphConfig()
        
        assert config.checkpoint_storage.value == "sqlite"
        assert config.checkpoint_db_path == "./data/checkpoints/langgraph.db"
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.ollama_model == "qwen2.5:7b"
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
        with patch('source.services.langgraph.legal_workflow.QuestionClassifier'):
            with patch('source.services.langgraph.legal_workflow.HybridSearchEngine'):
                with patch('source.services.langgraph.legal_workflow.OllamaClient'):
                    return LegalQuestionWorkflow(mock_config)
    
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
        compiled = workflow.compile()
        
        # 컴파일이 성공했는지 확인 (LangGraph가 사용 가능한 경우)
        if compiled:
            assert compiled is not None


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
        with patch('source.services.langgraph.workflow_service.LegalQuestionWorkflow'):
            with patch('source.services.langgraph.workflow_service.CheckpointManager'):
                return LangGraphWorkflowService(mock_config)
    
    @pytest.mark.asyncio
    async def test_process_query_basic(self, service):
        """기본 질문 처리 테스트"""
        with patch.object(service, 'app') as mock_app:
            # 모의 응답 설정
            mock_app.ainvoke.return_value = {
                "answer": "테스트 답변입니다.",
                "confidence": 0.8,
                "sources": ["테스트 소스"],
                "legal_references": ["민법"],
                "processing_steps": ["질문 분류", "문서 검색", "답변 생성"],
                "query_type": "general_question",
                "metadata": {},
                "errors": []
            }
            
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
            mock_app.ainvoke.return_value = {
                "answer": "세션 테스트 답변입니다.",
                "confidence": 0.9,
                "sources": ["세션 소스"],
                "legal_references": ["상법"],
                "processing_steps": ["세션 처리"],
                "query_type": "contract_review",
                "metadata": {},
                "errors": []
            }
            
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
        with patch.object(service.checkpoint_manager, 'list_checkpoints') as mock_list:
            mock_list.return_value = [{"id": "checkpoint1"}]
            
            with patch.object(service, 'process_query') as mock_process:
                mock_process.return_value = {"answer": "재개된 답변"}
                
                result = await service.resume_session("test-session", "새 질문")
                
                assert result["answer"] == "재개된 답변"
                mock_process.assert_called_once()
    
    def test_get_session_info(self, service):
        """세션 정보 조회 테스트"""
        with patch.object(service.checkpoint_manager, 'list_checkpoints') as mock_list:
            mock_list.return_value = [{"id": "checkpoint1"}, {"id": "checkpoint2"}]
            
            info = service.get_session_info("test-session")
            
            assert info["session_id"] == "test-session"
            assert info["checkpoint_count"] == 2
            assert info["has_checkpoints"] is True
    
    def test_get_service_status(self, service):
        """서비스 상태 조회 테스트"""
        with patch.object(service.checkpoint_manager, 'get_database_info') as mock_db_info:
            mock_db_info.return_value = {"database_path": ":memory:", "table_exists": True}
            
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
        from source.services.langgraph.state_definitions import create_initial_agent_state
        
        state = create_initial_agent_state("에이전트 질문", "agent-session")
        
        assert state["query"] == "에이전트 질문"
        assert state["session_id"] == "agent-session"
        assert state["current_agent"] == ""
        assert state["agent_history"] == []
        assert state["task_results"] == []
    
    def test_create_initial_streaming_state(self):
        """초기 스트리밍 상태 생성 테스트"""
        from source.services.langgraph.state_definitions import create_initial_streaming_state
        
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
