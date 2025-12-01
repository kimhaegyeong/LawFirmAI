"""
StreamHandler 통합 테스트
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, Any, List, AsyncGenerator
from api.services.streaming.stream_handler import StreamHandler
from api.services.streaming.event_builder import StreamEventBuilder
from api.services.streaming.token_extractor import TokenExtractor
from api.services.streaming.node_filter import NodeFilter


@pytest.mark.integration
class TestStreamHandler:
    """StreamHandler 클래스 통합 테스트"""
    
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
            sources_extractor=mock_sources_extractor
        )
    
    @pytest.mark.asyncio
    async def test_stream_handler_initialization(self, stream_handler):
        """StreamHandler 초기화 테스트"""
        assert stream_handler is not None
        assert stream_handler.workflow_service is not None
        assert stream_handler.sources_extractor is not None

