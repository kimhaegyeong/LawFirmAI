# -*- coding: utf-8 -*-
"""
Workflow Routes 테스트
langgraph_core/utils/workflow_routes.py 단위 테스트
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from lawfirm_langgraph.langgraph_core.utils.workflow_routes import (
    WorkflowRoutes,
    QueryComplexity
)


class TestQueryComplexity:
    """QueryComplexity 테스트"""
    
    def test_complexity_values(self):
        """복잡도 값 확인"""
        assert QueryComplexity.SIMPLE == "simple"
        assert QueryComplexity.MODERATE == "moderate"
        assert QueryComplexity.COMPLEX == "complex"
        assert QueryComplexity.MULTI_HOP == "multi_hop"


class TestWorkflowRoutes:
    """WorkflowRoutes 테스트"""
    
    @pytest.fixture
    def mock_retry_manager(self):
        """Mock RetryCounterManager"""
        manager = Mock()
        manager.get_retry_counts = Mock(return_value={
            "generation": 0,
            "validation": 0,
            "total": 0
        })
        return manager
    
    @pytest.fixture
    def workflow_routes(self, mock_retry_manager):
        """WorkflowRoutes 인스턴스 생성"""
        return WorkflowRoutes(
            retry_manager=mock_retry_manager,
            answer_generator=None,
            ai_keyword_generator=None
        )
    
    def test_init(self, mock_retry_manager):
        """초기화 테스트"""
        routes = WorkflowRoutes(
            retry_manager=mock_retry_manager,
            answer_generator=None,
            ai_keyword_generator=None
        )
        
        assert routes.retry_manager == mock_retry_manager
        assert routes.answer_generator is None
        assert routes.ai_keyword_generator is None
        assert routes.logger is not None
    
    def test_route_by_complexity(self, workflow_routes):
        """복잡도별 라우팅 테스트"""
        assert workflow_routes.route_by_complexity({"query_complexity": "simple"}) == "simple"
        assert workflow_routes.route_by_complexity({"query_complexity": "moderate"}) == "moderate"
        assert workflow_routes.route_by_complexity({"query_complexity": "complex"}) == "complex"
        assert workflow_routes.route_by_complexity({}) in ["simple", "moderate", "complex"]
        assert workflow_routes.route_by_complexity({"common": {"query_complexity": "moderate"}}) == "moderate"
    
    def test_route_expert(self, workflow_routes):
        """전문가 라우팅 테스트"""
        state = {
            "query": "복잡한 법률 질문",
            "legal_field": "family",
            "extracted_keywords": ["키워드1", "키워드2"]
        }
        
        with patch.object(workflow_routes, 'assess_complexity', return_value="complex"):
            result = workflow_routes.route_expert(state)
            
            assert "complexity_level" in result
            assert "requires_expert" in result
            assert "expert_subgraph" in result
    
    def test_assess_complexity(self, workflow_routes):
        """복잡도 평가 테스트"""
        state = {
            "query": "A" * 250,
            "extracted_keywords": ["kw1"] * 15,
            "urgency_level": "high"
        }
        
        result = workflow_routes.assess_complexity(state)
        
        assert result in ["simple", "medium", "complex", "moderate"]
    
    def test_assess_complexity_simple(self, workflow_routes):
        """간단한 복잡도 평가 테스트"""
        state = {
            "query": "짧은 질문",
            "extracted_keywords": ["kw1"],
            "urgency_level": "low"
        }
        
        result = workflow_routes.assess_complexity(state)
        
        assert result == "simple"
    
    def test_should_analyze_document(self, workflow_routes):
        """문서 분석 필요 여부 테스트"""
        assert workflow_routes.should_analyze_document({"uploaded_document": {"content": "test"}}) == "analyze"
        assert workflow_routes.should_analyze_document({}) == "skip"
    
    def test_should_skip_search(self, workflow_routes):
        """검색 스킵 여부 테스트"""
        assert workflow_routes.should_skip_search({"search_cache_hit": True}) == "skip"
        assert workflow_routes.should_skip_search({"search_cache_hit": False}) == "continue"
    
    def test_should_skip_search_adaptive(self, workflow_routes):
        """Adaptive 검색 스킵 여부 테스트"""
        assert workflow_routes.should_skip_search_adaptive({"needs_search": False, "query_complexity": "simple"}) == "skip"
        assert workflow_routes.should_skip_search_adaptive({"needs_search": True, "query_complexity": "complex"}) == "continue"
    
    def test_should_expand_keywords_ai(self, workflow_routes):
        """AI 키워드 확장 여부 테스트"""
        workflow_routes.ai_keyword_generator = None
        assert workflow_routes.should_expand_keywords_ai({"extracted_keywords": ["kw1", "kw2", "kw3"]}) == "skip"
        
        workflow_routes.ai_keyword_generator = Mock()
        assert workflow_routes.should_expand_keywords_ai({"extracted_keywords": ["kw1", "kw2"]}) == "skip"
        assert workflow_routes.should_expand_keywords_ai({"extracted_keywords": ["kw1", "kw2", "kw3", "kw4"], "query_type": "legal_advice"}) == "expand"
    
    def test_should_retry_generation(self, workflow_routes, mock_retry_manager):
        """재시도 여부 테스트"""
        mock_retry_manager.get_retry_counts.return_value = {"generation": 2, "validation": 0, "total": 2}
        assert workflow_routes.should_retry_generation({"answer": "Test answer"}) == "format"
        
        mock_retry_manager.get_retry_counts.return_value = {"generation": 0, "validation": 0, "total": 0}
        assert workflow_routes.should_retry_generation({"answer": "Short"}) in ["validate", "retry_search", "format"]

