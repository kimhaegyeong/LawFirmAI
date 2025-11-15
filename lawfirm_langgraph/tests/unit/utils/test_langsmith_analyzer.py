# -*- coding: utf-8 -*-
"""
LangSmith Analyzer 테스트
core/utils/langsmith_analyzer.py 단위 테스트
"""

import os
import pytest
import uuid
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from collections import OrderedDict

try:
    from langsmith.schemas import Run
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    Run = None


@pytest.fixture
def mock_langsmith_available():
    """LangSmith 사용 가능 모킹"""
    with patch('lawfirm_langgraph.core.utils.langsmith_analyzer.LANGSMITH_AVAILABLE', True):
        with patch('lawfirm_langgraph.core.utils.langsmith_analyzer.Client') as mock_client:
            with patch('lawfirm_langgraph.core.utils.langsmith_analyzer.Run') as mock_run:
                yield mock_client, mock_run


@pytest.fixture
def mock_analyzer(mock_langsmith_available):
    """LangGraphQueryAnalyzer 인스턴스 생성"""
    mock_client, mock_run = mock_langsmith_available
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    
    with patch.dict(os.environ, {'LANGSMITH_API_KEY': 'test_key'}):
        from lawfirm_langgraph.core.utils.langsmith_analyzer import LangGraphQueryAnalyzer
        analyzer = LangGraphQueryAnalyzer(api_key='test_key', project_name='TestProject')
        analyzer.client = mock_client_instance
        return analyzer


class TestLangGraphQueryAnalyzer:
    """LangGraphQueryAnalyzer 테스트"""
    
    def test_init_success(self, mock_langsmith_available, monkeypatch):
        """초기화 성공 테스트"""
        mock_client, _ = mock_langsmith_available
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        monkeypatch.setenv("LANGSMITH_API_KEY", "test_key")
        
        from lawfirm_langgraph.core.utils.langsmith_analyzer import LangGraphQueryAnalyzer
        analyzer = LangGraphQueryAnalyzer(api_key="test_key", project_name="TestProject")
        
        assert analyzer.api_key == "test_key"
        assert analyzer.project_name == "TestProject"
        assert analyzer.client is not None
    
    def test_init_no_api_key(self, mock_langsmith_available, monkeypatch):
        """API 키 없이 초기화 테스트"""
        mock_client, _ = mock_langsmith_available
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        
        from lawfirm_langgraph.core.utils.langsmith_analyzer import LangGraphQueryAnalyzer
        
        with pytest.raises(ValueError):
            LangGraphQueryAnalyzer()
    
    @patch('lawfirm_langgraph.core.utils.langsmith_analyzer.LANGSMITH_AVAILABLE', False)
    def test_init_langsmith_not_available(self):
        """LangSmith 미설치 시 초기화 테스트"""
        from lawfirm_langgraph.core.utils.langsmith_analyzer import LangGraphQueryAnalyzer
        
        with pytest.raises(ImportError):
            LangGraphQueryAnalyzer(api_key="test_key")
    
    def test_clear_cache(self, mock_analyzer):
        """캐시 클리어 테스트"""
        mock_analyzer._run_cache['test_id'] = Mock()
        mock_analyzer._tree_cache['test_id'] = {}
        mock_analyzer._cache_stats = {"hits": 5, "misses": 3}
        
        mock_analyzer.clear_cache()
        
        assert len(mock_analyzer._run_cache) == 0
        assert len(mock_analyzer._tree_cache) == 0
        assert mock_analyzer._cache_stats == {"hits": 0, "misses": 0}
    
    def test_get_cache_stats(self, mock_analyzer):
        """캐시 통계 반환 테스트"""
        mock_analyzer._cache_stats = {"hits": 5, "misses": 3}
        mock_analyzer._run_cache['id1'] = Mock()
        mock_analyzer._tree_cache['id2'] = {}
        
        stats = mock_analyzer.get_cache_stats()
        
        assert stats["hits"] == 5
        assert stats["misses"] == 3
        assert stats["hit_rate"] == 62.5
        assert stats["run_cache_size"] == 1
        assert stats["tree_cache_size"] == 1
    
    def test_get_cache_stats_empty(self, mock_analyzer):
        """빈 캐시 통계 테스트"""
        mock_analyzer._cache_stats = {"hits": 0, "misses": 0}
        
        stats = mock_analyzer.get_cache_stats()
        
        assert stats["hit_rate"] == 0
    
    def test_manage_cache_size(self, mock_analyzer):
        """캐시 크기 관리 테스트"""
        cache = OrderedDict()
        for i in range(5):
            cache[f'key_{i}'] = f'value_{i}'
        
        mock_analyzer._max_cache_size = 3
        mock_analyzer._manage_cache_size(cache, 3)
        
        assert len(cache) == 3
    
    def test_calculate_cost(self, mock_analyzer):
        """비용 계산 테스트"""
        cost = mock_analyzer._calculate_cost(1000, "gpt-3.5-turbo")
        
        assert cost > 0
        assert isinstance(cost, float)
    
    def test_calculate_cost_zero_tokens(self, mock_analyzer):
        """0 토큰 비용 계산 테스트"""
        cost = mock_analyzer._calculate_cost(0, "gpt-3.5-turbo")
        
        assert cost == 0.0
    
    def test_calculate_cost_with_token_usage(self, mock_analyzer):
        """토큰 사용량 포함 비용 계산 테스트"""
        token_usage = {"input_tokens": 500, "output_tokens": 500}
        cost = mock_analyzer._calculate_cost(1000, "gpt-3.5-turbo", token_usage=token_usage)
        
        assert cost > 0
    
    def test_validate_run_id(self, mock_analyzer):
        """Run ID 검증 테스트"""
        valid_id = str(uuid.uuid4())
        assert mock_analyzer._validate_run_id(valid_id) is True
        assert mock_analyzer._validate_run_id("invalid") is False
        assert mock_analyzer._validate_run_id("") is False
    
    def test_extract_query(self, mock_analyzer):
        """질의 추출 테스트"""
        mock_run = Mock()
        mock_run.inputs = {"query": "Test query"}
        
        query = mock_analyzer._extract_query(mock_run)
        
        assert query == "Test query"
    
    def test_extract_query_nested(self, mock_analyzer):
        """중첩된 질의 추출 테스트"""
        mock_run = Mock()
        mock_run.inputs = {"input": {"query": "Nested query"}}
        
        query = mock_analyzer._extract_query(mock_run)
        
        assert query == "Nested query"
    
    def test_extract_query_no_inputs(self, mock_analyzer):
        """입력 없는 질의 추출 테스트"""
        mock_run = Mock()
        mock_run.inputs = None
        
        query = mock_analyzer._extract_query(mock_run)
        
        assert query == "N/A"
    
    @patch('lawfirm_langgraph.core.utils.langsmith_analyzer.time.sleep')
    def test_read_run_with_retry_success(self, mock_sleep, mock_analyzer):
        """재시도 포함 run 읽기 성공 테스트"""
        mock_run = Mock()
        mock_analyzer.client.read_run.return_value = mock_run
        
        result = mock_analyzer._read_run_with_retry("test_id")
        
        assert result == mock_run
        mock_sleep.assert_not_called()
    
    @patch('lawfirm_langgraph.core.utils.langsmith_analyzer.time.sleep')
    def test_read_run_with_retry_rate_limit(self, mock_sleep, mock_analyzer):
        """Rate limit 재시도 테스트"""
        mock_analyzer.client.read_run.side_effect = [
            Exception("Rate limit exceeded"),
            Mock()
        ]
        
        result = mock_analyzer._read_run_with_retry("test_id")
        
        assert result is not None
        assert mock_sleep.called
    
    @patch('lawfirm_langgraph.core.utils.langsmith_analyzer.time.sleep')
    def test_read_run_with_retry_failure(self, mock_sleep, mock_analyzer):
        """재시도 실패 테스트"""
        mock_analyzer.client.read_run.side_effect = Exception("Test error")
        
        result = mock_analyzer._read_run_with_retry("test_id")
        
        assert result is None
    
    def test_get_recent_runs(self, mock_analyzer):
        """최근 runs 조회 테스트"""
        mock_run1 = Mock()
        mock_run2 = Mock()
        mock_analyzer.client.list_runs.return_value = [mock_run1, mock_run2]
        
        runs = mock_analyzer.get_recent_runs(hours=24, limit=100)
        
        assert len(runs) == 2
        mock_analyzer.client.list_runs.assert_called_once()
    
    def test_get_recent_runs_exception(self, mock_analyzer):
        """최근 runs 조회 예외 테스트"""
        mock_analyzer.client.list_runs.side_effect = Exception("Test error")
        
        runs = mock_analyzer.get_recent_runs(hours=24, limit=100)
        
        assert len(runs) == 0
    
    def test_analyze_run_performance(self, mock_analyzer):
        """Run 성능 분석 테스트"""
        mock_run = Mock()
        mock_run.id = uuid.uuid4()
        mock_run.start_time = datetime.now()
        mock_run.end_time = datetime.now() + timedelta(seconds=10)
        mock_run.status = "success"
        mock_run.error = None
        mock_run.inputs = {"query": "Test query"}
        mock_run.outputs = {}
        
        mock_analyzer._get_child_runs = Mock(return_value=[])
        mock_analyzer._analyze_nodes = Mock(return_value={"nodes": [], "total_tokens": 0, "total_cost": 0.0})
        mock_analyzer._identify_bottlenecks = Mock(return_value=[])
        mock_analyzer._generate_recommendations = Mock(return_value=[])
        mock_analyzer._extract_state_info = Mock(return_value={})
        
        analysis = mock_analyzer.analyze_run_performance(mock_run)
        
        assert "run_id" in analysis
        assert "query" in analysis
        assert "duration" in analysis
        assert "status" in analysis
        assert "nodes" in analysis
        assert "total_tokens" in analysis
        assert "total_cost" in analysis
    
    def test_get_improvement_suggestions(self, mock_analyzer):
        """개선 제안 생성 테스트"""
        analysis = {
            "error": None,
            "duration": 5.0,
            "bottlenecks": [],
            "total_tokens": 100
        }
        
        suggestions = mock_analyzer.get_improvement_suggestions(analysis)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
    
    def test_get_improvement_suggestions_with_error(self, mock_analyzer):
        """오류 포함 개선 제안 테스트"""
        analysis = {
            "error": "Test error",
            "duration": 5.0,
            "bottlenecks": [],
            "total_tokens": 100
        }
        
        suggestions = mock_analyzer.get_improvement_suggestions(analysis)
        
        assert any("오류" in s for s in suggestions)
    
    def test_get_improvement_suggestions_slow(self, mock_analyzer):
        """느린 실행 개선 제안 테스트"""
        analysis = {
            "error": None,
            "duration": 35.0,
            "bottlenecks": [],
            "total_tokens": 100
        }
        
        suggestions = mock_analyzer.get_improvement_suggestions(analysis)
        
        assert any("느립니다" in s for s in suggestions)
    
    def test_get_improvement_suggestions_high_tokens(self, mock_analyzer):
        """높은 토큰 사용량 개선 제안 테스트"""
        analysis = {
            "error": None,
            "duration": 5.0,
            "bottlenecks": [],
            "total_tokens": 15000
        }
        
        suggestions = mock_analyzer.get_improvement_suggestions(analysis)
        
        assert any("토큰" in s for s in suggestions)
    
    def test_analyze_query_patterns(self, mock_analyzer):
        """질의 패턴 분석 테스트"""
        mock_run1 = Mock()
        mock_run1.start_time = datetime.now()
        mock_run1.end_time = datetime.now() + timedelta(seconds=5)
        mock_run1.status = "success"
        mock_run1.error = None
        mock_run1.inputs = {"query": "Test query 1"}
        
        mock_run2 = Mock()
        mock_run2.start_time = datetime.now()
        mock_run2.end_time = datetime.now() + timedelta(seconds=40)
        mock_run2.status = "success"
        mock_run2.error = None
        mock_run2.inputs = {"query": "Test query 2"}
        
        mock_analyzer._extract_query = Mock(side_effect=["Query 1", "Query 2"])
        mock_analyzer._get_child_runs = Mock(return_value=[])
        
        patterns = mock_analyzer.analyze_query_patterns([mock_run1, mock_run2])
        
        assert "slow_queries" in patterns
        assert "error_queries" in patterns
        assert "common_nodes" in patterns
        assert "average_durations" in patterns
        assert "token_usage" in patterns
    
    def test_analyze_query_patterns_with_errors(self, mock_analyzer):
        """오류 포함 질의 패턴 분석 테스트"""
        mock_run = Mock()
        mock_run.start_time = datetime.now()
        mock_run.end_time = datetime.now() + timedelta(seconds=5)
        mock_run.status = "error"
        mock_run.error = "Test error"
        mock_run.inputs = {"query": "Test query"}
        
        mock_analyzer._extract_query = Mock(return_value="Test query")
        mock_analyzer._get_child_runs = Mock(return_value=[])
        
        patterns = mock_analyzer.analyze_query_patterns([mock_run])
        
        assert len(patterns["error_queries"]) == 1
        assert patterns["error_queries"][0]["error"] == "Test error"

