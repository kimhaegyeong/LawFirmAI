#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StreamHandler 테스트 코드
"""
import pytest
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, Any, List, AsyncGenerator

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.services.streaming.stream_handler import StreamHandler
from api.services.streaming.event_builder import StreamEventBuilder
from api.services.streaming.token_extractor import TokenExtractor
from api.services.streaming.node_filter import NodeFilter


class TestStreamHandler:
    """StreamHandler 클래스 테스트"""
    
    @pytest.fixture
    def mock_workflow_service(self):
        """WorkflowService 모킹"""
        mock_service = MagicMock()
        mock_service.app = MagicMock()
        return mock_service
    
    @pytest.fixture
    def mock_sources_extractor(self):
        """SourcesExtractor 모킹"""
        mock_extractor = MagicMock()
        
        # _get_sources_by_type 모킹
        mock_extractor._get_sources_by_type = Mock(return_value={
            "statute_article": [],
            "case_paragraph": [{"type": "case_paragraph", "doc_id": "case_2024다209769"}],
            "decision_paragraph": [],
            "interpretation_paragraph": []
        })
        
        # _get_sources_by_type_with_reference_statutes 모킹
        mock_extractor._get_sources_by_type_with_reference_statutes = Mock(return_value={
            "statute_article": [
                {
                    "type": "statute_article",
                    "statute_name": "민법",
                    "article_no": "105",
                    "source_from": "case_paragraph",
                    "source_doc_id": "case_2024다209769"
                }
            ],
            "case_paragraph": [{"type": "case_paragraph", "doc_id": "case_2024다209769"}],
            "decision_paragraph": [],
            "interpretation_paragraph": []
        })
        
        # _extract_statutes_from_reference_clauses 모킹
        mock_extractor._extract_statutes_from_reference_clauses = Mock(return_value=[
            {
                "type": "statute_article",
                "statute_name": "민법",
                "article_no": "105",
                "source_from": "case_paragraph",
                "source_doc_id": "case_2024다209769"
            }
        ])
        
        return mock_extractor
    
    @pytest.fixture
    def stream_handler(self, mock_workflow_service, mock_sources_extractor):
        """StreamHandler 인스턴스 생성"""
        return StreamHandler(
            workflow_service=mock_workflow_service,
            sources_extractor=mock_sources_extractor,
            extract_related_questions_fn=None
        )
    
    def test_init(self, mock_workflow_service, mock_sources_extractor):
        """StreamHandler 초기화 테스트"""
        handler = StreamHandler(
            workflow_service=mock_workflow_service,
            sources_extractor=mock_sources_extractor
        )
        
        assert handler.workflow_service == mock_workflow_service
        assert handler.sources_extractor == mock_sources_extractor
        assert handler.token_extractor is not None
        assert handler.node_filter is not None
        assert handler.event_builder is not None
    
    @pytest.mark.asyncio
    async def test_get_final_metadata_with_reference_statutes(self, stream_handler, mock_workflow_service):
        """_get_final_metadata에서 참조 법령이 포함된 sources_by_type 생성 테스트"""
        # Mock state
        mock_state = MagicMock()
        mock_state.values = {
            "sources": [],
            "legal_references": [],
            "sources_detail": [
                {
                    "type": "case_paragraph",
                    "doc_id": "case_2024다209769",
                    "metadata": {}
                }
            ],
            "metadata": {}
        }
        
        # Mock config
        config = {"configurable": {"thread_id": "test_session"}}
        
        # Mock aget_state
        async def mock_aget_state(cfg):
            return mock_state
        
        mock_workflow_service.app.aget_state = AsyncMock(side_effect=mock_aget_state)
        
        # 테스트 실행
        result = await stream_handler._get_final_metadata(
            config=config,
            initial_state={},
            message="테스트 질문",
            full_answer="테스트 답변",
            session_id="test_session"
        )
        
        # 검증
        assert result is not None
        assert "sources_by_type" in result
        
        sources_by_type = result.get("sources_by_type")
        assert sources_by_type is not None
        
        # 참조 법령이 포함되었는지 확인
        statute_articles = sources_by_type.get("statute_article", [])
        assert len(statute_articles) > 0
        
        # 첫 번째 법령이 참조 법령인지 확인
        first_statute = statute_articles[0]
        assert first_statute.get("source_from") == "case_paragraph"
        assert first_statute.get("source_doc_id") == "case_2024다209769"
    
    @pytest.mark.asyncio
    async def test_get_final_metadata_exception_handling(self, stream_handler, mock_workflow_service):
        """_get_final_metadata에서 예외 발생 시 기본값 반환 테스트"""
        # Mock state
        mock_state = MagicMock()
        mock_state.values = {
            "sources": [],
            "legal_references": [],
            "sources_detail": [
                {
                    "type": "case_paragraph",
                    "doc_id": "case_2024다209769"
                }
            ],
            "metadata": {}
        }
        
        config = {"configurable": {"thread_id": "test_session"}}
        
        # _get_sources_by_type_with_reference_statutes에서 예외 발생하도록 모킹
        stream_handler.sources_extractor._get_sources_by_type_with_reference_statutes = Mock(
            side_effect=Exception("Database connection error")
        )
        
        # Mock aget_state
        async def mock_aget_state(cfg):
            return mock_state
        
        mock_workflow_service.app.aget_state = AsyncMock(side_effect=mock_aget_state)
        
        # 테스트 실행
        result = await stream_handler._get_final_metadata(
            config=config,
            initial_state={},
            message="테스트 질문",
            full_answer="테스트 답변",
            session_id="test_session"
        )
        
        # 검증: 예외 발생해도 기본 sources_by_type이 반환되어야 함
        assert result is not None
        assert "sources_by_type" in result
        
        sources_by_type = result.get("sources_by_type")
        assert sources_by_type is not None
        assert isinstance(sources_by_type, dict)
        assert "statute_article" in sources_by_type
        assert "case_paragraph" in sources_by_type
    
    @pytest.mark.asyncio
    async def test_get_final_metadata_fallback_on_error(self, stream_handler, mock_workflow_service):
        """_get_final_metadata에서 fallback 동작 테스트"""
        mock_state = MagicMock()
        mock_state.values = {
            "sources": [],
            "legal_references": [],
            "sources_detail": [
                {
                    "type": "case_paragraph",
                    "doc_id": "case_2024다209769"
                }
            ],
            "metadata": {}
        }
        
        config = {"configurable": {"thread_id": "test_session"}}
        
        # 첫 번째 호출에서 예외, 두 번째 호출에서 성공하도록 모킹
        call_count = 0
        def mock_with_reference_statutes(sources_detail):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First error")
            return {
                "statute_article": [],
                "case_paragraph": sources_detail,
                "decision_paragraph": [],
                "interpretation_paragraph": []
            }
        
        stream_handler.sources_extractor._get_sources_by_type_with_reference_statutes = Mock(
            side_effect=mock_with_reference_statutes
        )
        
        # Mock aget_state
        async def mock_aget_state(cfg):
            return mock_state
        
        mock_workflow_service.app.aget_state = AsyncMock(side_effect=mock_aget_state)
        
        # 테스트 실행
        result = await stream_handler._get_final_metadata(
            config=config,
            initial_state={},
            message="테스트 질문",
            full_answer="테스트 답변",
            session_id="test_session"
        )
        
        # 검증: fallback이 호출되어야 함
        assert result is not None
        assert "sources_by_type" in result
        # _get_sources_by_type이 fallback으로 호출되었는지 확인
        assert stream_handler.sources_extractor._get_sources_by_type.called
    
    @pytest.mark.asyncio
    async def test_get_final_metadata_timeout(self, stream_handler, mock_workflow_service):
        """_get_final_metadata에서 타임아웃 처리 테스트"""
        config = {"configurable": {"thread_id": "test_session"}}
        
        # 타임아웃 발생하도록 모킹
        async def mock_aget_state_timeout(cfg):
            await asyncio.sleep(10)  # 타임아웃보다 긴 대기
            return None
        
        mock_workflow_service.app.aget_state = AsyncMock(side_effect=mock_aget_state_timeout)
        
        # 테스트 실행
        result = await stream_handler._get_final_metadata(
            config=config,
            initial_state={},
            message="테스트 질문",
            full_answer="테스트 답변",
            session_id="test_session"
        )
        
        # 검증: 타임아웃 시 빈 딕셔너리 반환
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_get_final_metadata_with_empty_sources_detail(self, stream_handler, mock_workflow_service):
        """sources_detail이 비어있을 때 테스트"""
        mock_state = MagicMock()
        mock_state.values = {
            "sources": [],
            "legal_references": [],
            "sources_detail": [],
            "metadata": {}
        }
        
        config = {"configurable": {"thread_id": "test_session"}}
        
        async def mock_aget_state(cfg):
            return mock_state
        
        mock_workflow_service.app.aget_state = AsyncMock(side_effect=mock_aget_state)
        
        # 테스트 실행
        result = await stream_handler._get_final_metadata(
            config=config,
            initial_state={},
            message="테스트 질문",
            full_answer="테스트 답변",
            session_id="test_session"
        )
        
        # 검증: sources_detail이 비어있으면 sources_by_type도 기본 구조만 반환
        assert result is not None
        sources_by_type = result.get("sources_by_type")
        # sources_detail이 비어있으면 None이거나 기본 구조
        assert sources_by_type is None or isinstance(sources_by_type, dict)
    
    def test_validate_and_augment_state(self, stream_handler):
        """_validate_and_augment_state 테스트"""
        # 정상 케이스
        initial_state = {
            "input": {},
            "query": ""
        }
        message = "테스트 질문"
        session_id = "test_session"
        
        result = stream_handler._validate_and_augment_state(initial_state, message, session_id)
        
        assert result == message
        assert initial_state["input"]["query"] == message
        assert initial_state["input"]["session_id"] == session_id
        assert initial_state["query"] == message
        assert initial_state["session_id"] == session_id
    
    def test_validate_and_augment_state_empty_message(self, stream_handler):
        """빈 메시지 처리 테스트"""
        initial_state = {
            "input": {},
            "query": ""
        }
        message = ""
        session_id = "test_session"
        
        result = stream_handler._validate_and_augment_state(initial_state, message, session_id)
        
        assert result is None


@pytest.mark.asyncio
async def test_stream_handler_integration():
    """StreamHandler 통합 테스트"""
    print("\n" + "=" * 80)
    print("StreamHandler 통합 테스트")
    print("=" * 80)
    
    try:
        from api.services.chat_service import get_chat_service
        
        chat_service = get_chat_service()
        stream_handler = chat_service.stream_handler
        
        if not stream_handler:
            print("❌ StreamHandler가 초기화되지 않았습니다.")
            return False
        
        print("✅ StreamHandler 초기화 확인")
        
        # sources_extractor 확인
        if stream_handler.sources_extractor:
            print("✅ SourcesExtractor 확인")
            
            # 테스트용 sources_detail
            test_sources_detail = [
                {
                    "type": "case_paragraph",
                    "doc_id": "case_2024다209769",
                    "case_number": "2024다209769",
                    "metadata": {}
                }
            ]
            
            # _get_sources_by_type_with_reference_statutes 테스트
            try:
                result = stream_handler.sources_extractor._get_sources_by_type_with_reference_statutes(
                    test_sources_detail
                )
                
                print(f"✅ _get_sources_by_type_with_reference_statutes 실행 성공")
                print(f"   - statute_article 개수: {len(result.get('statute_article', []))}")
                print(f"   - case_paragraph 개수: {len(result.get('case_paragraph', []))}")
                
                # 참조 법령 확인
                statutes = result.get('statute_article', [])
                if statutes:
                    print(f"\n📋 추출된 참조 법령:")
                    for i, statute in enumerate(statutes[:3], 1):
                        print(f"   {i}. {statute.get('statute_name', 'N/A')} 제{statute.get('article_no', 'N/A')}조")
                        print(f"      - source_from: {statute.get('source_from', 'N/A')}")
                else:
                    print("\n⚠️  참조 법령이 추출되지 않았습니다.")
                
                return True
            except Exception as e:
                print(f"❌ _get_sources_by_type_with_reference_statutes 실행 실패: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("⚠️  SourcesExtractor가 없습니다.")
            return False
            
    except Exception as e:
        print(f"❌ 통합 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 단위 테스트 실행
    print("\n" + "=" * 80)
    print("StreamHandler 단위 테스트")
    print("=" * 80)
    
    pytest.main([__file__, "-v", "-s"])

