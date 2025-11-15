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
    
    def test_route_by_complexity_simple(self, workflow_routes):
        """간단한 복잡도 라우팅 테스트"""
        state = {"query_complexity": "simple"}
        result = workflow_routes.route_by_complexity(state)
        
        assert result == "simple"
    
    def test_route_by_complexity_moderate(self, workflow_routes):
        """중간 복잡도 라우팅 테스트"""
        state = {"query_complexity": "moderate"}
        result = workflow_routes.route_by_complexity(state)
        
        assert result == "moderate"
    
    def test_route_by_complexity_complex(self, workflow_routes):
        """복잡한 복잡도 라우팅 테스트"""
        state = {"query_complexity": "complex"}
        result = workflow_routes.route_by_complexity(state)
        
        assert result == "complex"
    
    def test_route_by_complexity_default(self, workflow_routes):
        """기본 복잡도 라우팅 테스트"""
        state = {}
        result = workflow_routes.route_by_complexity(state)
        
        assert result in ["simple", "moderate", "complex"]
    
    def test_route_by_complexity_common_group(self, workflow_routes):
        """common 그룹에서 복잡도 조회 테스트"""
        state = {"common": {"query_complexity": "moderate"}}
        result = workflow_routes.route_by_complexity(state)
        
        assert result == "moderate"
    
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
        
        assert result in ["simple", "medium", "complex"]
    
    def test_assess_complexity_simple(self, workflow_routes):
        """간단한 복잡도 평가 테스트"""
        state = {
            "query": "짧은 질문",
            "extracted_keywords": ["kw1"],
            "urgency_level": "low"
        }
        
        result = workflow_routes.assess_complexity(state)
        
        assert result == "simple"
    
    def test_should_analyze_document_with_document(self, workflow_routes):
        """문서가 있는 경우 문서 분석 필요 여부 테스트"""
        state = {"uploaded_document": {"content": "test"}}
        result = workflow_routes.should_analyze_document(state)
        
        assert result == "analyze"
    
    def test_should_analyze_document_without_document(self, workflow_routes):
        """문서가 없는 경우 문서 분석 필요 여부 테스트"""
        state = {}
        result = workflow_routes.should_analyze_document(state)
        
        assert result == "skip"
    
    def test_should_skip_search_cache_hit(self, workflow_routes):
        """캐시 히트 시 검색 스킵 테스트"""
        state = {"search_cache_hit": True}
        result = workflow_routes.should_skip_search(state)
        
        assert result == "skip"
    
    def test_should_skip_search_no_cache(self, workflow_routes):
        """캐시 미스 시 검색 계속 테스트"""
        state = {"search_cache_hit": False}
        result = workflow_routes.should_skip_search(state)
        
        assert result == "continue"
    
    def test_should_skip_search_adaptive_simple(self, workflow_routes):
        """간단한 질문 검색 스킵 테스트"""
        state = {
            "needs_search": False,
            "query_complexity": "simple"
        }
        result = workflow_routes.should_skip_search_adaptive(state)
        
        assert result == "skip"
    
    def test_should_skip_search_adaptive_complex(self, workflow_routes):
        """복잡한 질문 검색 계속 테스트"""
        state = {
            "needs_search": True,
            "query_complexity": "complex"
        }
        result = workflow_routes.should_skip_search_adaptive(state)
        
        assert result == "continue"
    
    def test_should_expand_keywords_ai_no_generator(self, workflow_routes):
        """AI 생성기가 없는 경우 키워드 확장 스킵 테스트"""
        workflow_routes.ai_keyword_generator = None
        state = {"extracted_keywords": ["kw1", "kw2", "kw3"]}
        result = workflow_routes.should_expand_keywords_ai(state)
        
        assert result == "skip"
    
    def test_should_expand_keywords_ai_few_keywords(self, workflow_routes):
        """키워드가 적은 경우 확장 스킵 테스트"""
        workflow_routes.ai_keyword_generator = Mock()
        state = {"extracted_keywords": ["kw1", "kw2"]}
        result = workflow_routes.should_expand_keywords_ai(state)
        
        assert result == "skip"
    
    def test_should_expand_keywords_ai_complex_type(self, workflow_routes):
        """복잡한 질문 유형 키워드 확장 테스트"""
        workflow_routes.ai_keyword_generator = Mock()
        state = {
            "extracted_keywords": ["kw1", "kw2", "kw3", "kw4"],
            "query_type": "legal_advice"
        }
        result = workflow_routes.should_expand_keywords_ai(state)
        
        assert result == "expand"
    
    def test_should_retry_generation_max_retries(self, workflow_routes, mock_retry_manager):
        """최대 재시도 횟수 도달 시 포맷팅 진행 테스트"""
        mock_retry_manager.get_retry_counts.return_value = {
            "generation": 2,
            "validation": 0,
            "total": 2
        }
        
        state = {"answer": "Test answer"}
        result = workflow_routes.should_retry_generation(state)
        
        assert result == "format"
    
    def test_should_retry_generation_short_answer(self, workflow_routes, mock_retry_manager):
        """짧은 답변 재시도 테스트"""
        mock_retry_manager.get_retry_counts.return_value = {
            "generation": 0,
            "validation": 0,
            "total": 0
        }
        
        state = {"answer": "Short"}
        result = workflow_routes.should_retry_generation(state)
        
        assert result in ["validate", "retry_search", "format"]

