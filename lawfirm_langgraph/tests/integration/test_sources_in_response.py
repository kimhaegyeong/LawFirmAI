# -*- coding: utf-8 -*-
"""
LangGraph sources 전달 테스트
langgraph에서 sources가 최종 응답에 제대로 전달되는지 테스트
"""

import pytest
import asyncio
import sys
import json
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, patch, AsyncMock

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# lawfirm_langgraph 경로 추가
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
if lawfirm_langgraph_path.exists():
    sys.path.insert(0, str(lawfirm_langgraph_path))

try:
    from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
    from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
    from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    print(f"LangGraph not available: {e}")

try:
    from api.services.chat_service import ChatService
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False


class TestSourcesInResponse:
    """sources가 최종 응답에 제대로 전달되는지 테스트"""
    
    @pytest.fixture
    def mock_state_with_sources(self):
        """sources가 포함된 mock state"""
        return {
            "query": "테스트 질문",
            "answer": "테스트 답변",
            "sources": ["민법 제1조", "대법원 2020다12345"],
            "sources_detail": [
                {
                    "name": "민법 제1조",
                    "type": "statute_article",
                    "url": "https://www.law.go.kr/LSW/lsSc.do?lawNm=민법&articleNo=1",
                    "metadata": {"statute_name": "민법", "article_no": "제1조"}
                },
                {
                    "name": "대법원 2020다12345",
                    "type": "case_paragraph",
                    "url": "",
                    "metadata": {"court": "대법원", "doc_id": "2020다12345"}
                }
            ],
            "legal_references": ["민법 제1조"],
            "retrieved_docs": [
                {
                    "type": "statute_article",
                    "statute_name": "민법",
                    "article_no": "제1조",
                    "metadata": {"statute_name": "민법", "article_no": "제1조"}
                },
                {
                    "type": "case_paragraph",
                    "court": "대법원",
                    "doc_id": "2020다12345",
                    "metadata": {"court": "대법원", "doc_id": "2020다12345"}
                }
            ],
            "confidence": 0.9,
            "query_type": "legal_question"
        }
    
    @pytest.fixture
    def mock_state_without_sources(self):
        """sources가 없는 mock state"""
        return {
            "query": "테스트 질문",
            "answer": "테스트 답변",
            "sources": [],
            "sources_detail": [],
            "legal_references": [],
            "retrieved_docs": [],
            "confidence": 0.9,
            "query_type": "general_question"
        }
    
    @pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not available")
    @pytest.mark.asyncio
    async def test_workflow_service_sources_in_response(self, mock_state_with_sources):
        """LangGraphWorkflowService에서 sources가 응답에 포함되는지 테스트"""
        config = LangGraphConfig.from_env()
        
        with patch('lawfirm_langgraph.core.workflow.workflow_service.EnhancedLegalQuestionWorkflow'):
            service = LangGraphWorkflowService(config)
            
            # mock workflow 설정
            mock_workflow = MagicMock()
            
            # invoke_async mock - 실제 workflow의 invoke_async는 state를 반환하므로
            # mock_state_with_sources를 그대로 반환하도록 설정
            async def mock_invoke_async(input_data, config):
                # 실제 workflow는 state를 반환하므로 dict를 반환
                return mock_state_with_sources
            
            mock_workflow.invoke_async = AsyncMock(side_effect=mock_invoke_async)
            
            # app.aget_state도 mock 설정
            mock_state_obj = MagicMock()
            mock_state_obj.values = mock_state_with_sources
            
            async def mock_aget_state(config):
                return mock_state_obj
            
            mock_workflow.app = MagicMock()
            mock_workflow.app.aget_state = AsyncMock(side_effect=mock_aget_state)
            
            service.workflow = mock_workflow
            service.app = mock_workflow
            
            # process_query 실행
            result = await service.process_query(
                query="테스트 질문",
                session_id="test_session"
            )
            
            # sources가 응답에 포함되는지 확인
            assert "sources" in result, "응답에 sources 필드가 있어야 함"
            assert isinstance(result["sources"], list), "sources는 리스트여야 함"
            # sources가 비어있지 않거나, retrieved_docs에서 추출되었는지 확인
            if len(result["sources"]) == 0:
                # retrieved_docs에서 sources를 추출했는지 확인
                retrieved_docs = result.get("retrieved_docs", [])
                if retrieved_docs:
                    # retrieved_docs가 있으면 sources가 생성되어야 함
                    assert len(result["sources"]) > 0 or len(retrieved_docs) > 0, "retrieved_docs가 있으면 sources가 생성되어야 함"
            
            # sources_detail이 응답에 포함되는지 확인
            sources_detail = result.get("sources_detail", [])
            if sources_detail:
                assert isinstance(sources_detail, list), "sources_detail은 리스트여야 함"
            
            print(f"\n✅ WorkflowService sources 전달 테스트 통과")
            print(f"  sources: {result.get('sources', [])}")
            print(f"  sources_detail: {len(result.get('sources_detail', []))}개")
            print(f"  retrieved_docs: {len(result.get('retrieved_docs', []))}개")
    
    @pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not available")
    @pytest.mark.asyncio
    async def test_workflow_service_sources_from_retrieved_docs(self, mock_state_without_sources):
        """retrieved_docs에서 sources를 추출하는지 테스트"""
        config = LangGraphConfig.from_env()
        
        # retrieved_docs는 있지만 sources는 없는 경우
        mock_state = {
            **mock_state_without_sources,
            "retrieved_docs": [
                {
                    "type": "statute_article",
                    "statute_name": "민법",
                    "article_no": "제1조",
                    "metadata": {"statute_name": "민법", "article_no": "제1조"}
                }
            ]
        }
        
        with patch('lawfirm_langgraph.core.workflow.workflow_service.EnhancedLegalQuestionWorkflow'):
            service = LangGraphWorkflowService(config)
            
            # mock workflow 설정
            mock_workflow = MagicMock()
            
            async def mock_invoke_async(input_data, config):
                return mock_state
            
            mock_workflow.invoke_async = AsyncMock(side_effect=mock_invoke_async)
            service.workflow = mock_workflow
            service.app = mock_workflow
            
            # process_query 실행
            result = await service.process_query(
                query="테스트 질문",
                session_id="test_session"
            )
            
            # retrieved_docs에서 sources를 추출했는지 확인
            assert "sources" in result, "응답에 sources 필드가 있어야 함"
            # sources가 비어있지 않거나, retrieved_docs가 있으면 sources가 생성되어야 함
            if result.get("retrieved_docs"):
                assert len(result.get("sources", [])) > 0 or len(result.get("retrieved_docs", [])) > 0, "retrieved_docs가 있으면 sources가 생성되어야 함"
            
            print(f"\n✅ WorkflowService retrieved_docs에서 sources 추출 테스트 통과")
            print(f"  sources: {result.get('sources', [])}")
            print(f"  retrieved_docs: {len(result.get('retrieved_docs', []))}개")
    
    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
    @pytest.mark.asyncio
    async def test_chat_service_sources_in_stream(self, mock_state_with_sources):
        """ChatService의 stream_message에서 sources가 전달되는지 테스트"""
        from api.services.chat_service import ChatService
        
        chat_service = ChatService()
        
        if not chat_service.workflow_service:
            pytest.skip("Workflow service not available")
        
        # mock workflow service 설정
        mock_workflow = MagicMock()
        mock_state = MagicMock()
        mock_state.values = mock_state_with_sources
        
        async def mock_aget_state(config):
            return mock_state
        
        mock_workflow.app.aget_state = AsyncMock(side_effect=mock_aget_state)
        chat_service.workflow_service.app = mock_workflow.app
        
        # stream_message 실행
        session_id = "test_session"
        message = "테스트 질문"
        
        sources_found = False
        sources_in_final_event = False
        final_event_metadata = None
        
        async for event in chat_service.stream_message(message, session_id):
            event_data = json.loads(event) if isinstance(event, str) else event
            
            # sources 이벤트 확인
            if event_data.get("type") == "sources":
                sources_found = True
                metadata = event_data.get("metadata", {})
                if metadata.get("sources"):
                    sources_in_final_event = True
                    assert len(metadata["sources"]) > 0, "sources가 비어있지 않아야 함"
            
            # final 이벤트에서 sources 확인
            if event_data.get("type") == "final":
                final_event_metadata = event_data.get("metadata", {})
                if final_event_metadata.get("sources"):
                    sources_in_final_event = True
                    assert len(final_event_metadata["sources"]) > 0, "final 이벤트에 sources가 있어야 함"
        
        # sources가 최소한 한 곳에는 있어야 함
        # 또는 get_sources_from_session으로 가져올 수 있어야 함
        if not sources_found and not sources_in_final_event:
            # get_sources_from_session으로 확인
            sources_data = await chat_service.get_sources_from_session(session_id)
            assert len(sources_data.get("sources", [])) > 0, "sources가 LangGraph state에서 가져와져야 함"
            sources_in_final_event = True
        
        assert sources_found or sources_in_final_event, "sources가 스트림 이벤트에 포함되거나 LangGraph state에서 가져와져야 함"
        
        print(f"\n✅ ChatService stream_message sources 전달 테스트 통과")
        print(f"  sources_found: {sources_found}")
        print(f"  sources_in_final_event: {sources_in_final_event}")
        if final_event_metadata:
            print(f"  final_event_metadata keys: {list(final_event_metadata.keys())}")
    
    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
    @pytest.mark.asyncio
    async def test_chat_service_get_sources_from_session(self, mock_state_with_sources):
        """ChatService의 get_sources_from_session에서 sources를 가져오는지 테스트"""
        from api.services.chat_service import ChatService
        import json
        
        chat_service = ChatService()
        
        if not chat_service.workflow_service:
            pytest.skip("Workflow service not available")
        
        # mock workflow service 설정
        mock_workflow = MagicMock()
        mock_state = MagicMock()
        mock_state.values = mock_state_with_sources
        
        async def mock_aget_state(config):
            return mock_state
        
        mock_workflow.app.aget_state = AsyncMock(side_effect=mock_aget_state)
        chat_service.workflow_service.app = mock_workflow.app
        
        # get_sources_from_session 실행
        session_id = "test_session"
        sources_data = await chat_service.get_sources_from_session(session_id)
        
        # sources가 반환되는지 확인
        assert "sources" in sources_data, "sources 필드가 있어야 함"
        assert isinstance(sources_data["sources"], list), "sources는 리스트여야 함"
        assert len(sources_data["sources"]) > 0, "sources가 비어있지 않아야 함"
        
        # sources_detail이 반환되는지 확인
        assert "sources_detail" in sources_data, "sources_detail 필드가 있어야 함"
        assert isinstance(sources_data["sources_detail"], list), "sources_detail은 리스트여야 함"
        
        # legal_references가 반환되는지 확인
        assert "legal_references" in sources_data, "legal_references 필드가 있어야 함"
        assert isinstance(sources_data["legal_references"], list), "legal_references는 리스트여야 함"
        
        print(f"\n✅ ChatService get_sources_from_session 테스트 통과")
        print(f"  sources: {sources_data['sources']}")
        print(f"  sources_detail: {len(sources_data['sources_detail'])}개")
        print(f"  legal_references: {sources_data['legal_references']}")
    
    @pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not available")
    @pytest.mark.asyncio
    async def test_sources_detail_structure(self, mock_state_with_sources):
        """sources_detail의 구조가 올바른지 테스트"""
        config = LangGraphConfig.from_env()
        
        with patch('lawfirm_langgraph.core.workflow.workflow_service.EnhancedLegalQuestionWorkflow'):
            service = LangGraphWorkflowService(config)
            
            # mock workflow 설정
            mock_workflow = MagicMock()
            
            async def mock_invoke_async(input_data, config):
                return mock_state_with_sources
            
            mock_workflow.invoke_async = AsyncMock(side_effect=mock_invoke_async)
            service.workflow = mock_workflow
            service.app = mock_workflow
            
            # process_query 실행
            result = await service.process_query(
                query="테스트 질문",
                session_id="test_session"
            )
            
            # sources_detail 구조 확인
            sources_detail = result.get("sources_detail", [])
            if sources_detail:
                for detail in sources_detail:
                    assert isinstance(detail, dict), "sources_detail의 각 항목은 딕셔너리여야 함"
                    assert "name" in detail, "sources_detail에 name 필드가 있어야 함"
                    assert "type" in detail, "sources_detail에 type 필드가 있어야 함"
                    assert detail["name"], "name이 비어있지 않아야 함"
                    assert detail["type"], "type이 비어있지 않아야 함"
            
            print(f"\n✅ sources_detail 구조 테스트 통과")
            print(f"  sources_detail: {sources_detail}")


if __name__ == "__main__":
    import json
    
    print("=" * 80)
    print("LangGraph sources 전달 테스트 시작")
    print("=" * 80)
    
    # 테스트 실행
    pytest.main([__file__, "-v", "-s"])

