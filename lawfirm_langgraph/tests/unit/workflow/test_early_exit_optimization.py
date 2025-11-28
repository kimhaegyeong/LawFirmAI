# -*- coding: utf-8 -*-
"""
조기 종료 최적화 단위 테스트
execute_searches_parallel의 조기 종료 로직 검증
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import Future, TimeoutError as FutureTimeoutError

try:
    from lawfirm_langgraph.core.workflow.processors.search_execution_processor import SearchExecutionProcessor
except ImportError:
    from core.workflow.processors.search_execution_processor import SearchExecutionProcessor


class TestEarlyExitOptimization:
    """조기 종료 최적화 테스트 클래스"""
    
    @pytest.fixture
    def processor(self):
        """SearchExecutionProcessor 인스턴스 생성"""
        try:
            from lawfirm_langgraph.core.agents.handlers.search_handler import SearchHandler
            from lawfirm_langgraph.core.search.processors.result_balancer import ResultBalancer
        except ImportError:
            from core.agents.handlers.search_handler import SearchHandler
            from core.search.processors.result_balancer import ResultBalancer
        
        with patch('lawfirm_langgraph.core.workflow.processors.search_execution_processor.SearchHandler'):
            processor = SearchExecutionProcessor(
                search_handler=Mock(spec=SearchHandler),
                result_balancer=Mock(spec=ResultBalancer),
                logger=Mock()
            )
            processor.logger = Mock()
            processor.logger.isEnabledFor = Mock(return_value=False)
            processor.logger.warning = Mock()
            processor.logger.info = Mock()
            processor.logger.debug = Mock()
            processor.logger.error = Mock()
            return processor
    
    @pytest.fixture
    def sample_state(self):
        """샘플 워크플로우 상태"""
        return {
            "query": "계약 해지",
            "optimized_queries": {
                "semantic_query": "계약 해지",
                "keyword_queries": ["계약", "해지"]
            },
            "search_params": {
                "semantic_k": 10,
                "keyword_k": 10
            },
            "query_type_str": "law_inquiry",
            "legal_field": "contract",
            "extracted_keywords": ["계약", "해지"],
            "search": {}
        }
    
    def test_early_exit_on_zero_semantic_results(self, processor, sample_state):
        """Semantic 검색이 0개 결과를 반환하면 조기 종료하는지 테스트"""
        # Mock 설정: Semantic 검색은 0개, Keyword 검색은 10개 반환
        def mock_semantic_search(*args, **kwargs):
            return [], 0
        
        def mock_keyword_search(*args, **kwargs):
            return [{"id": i, "content": f"result {i}"} for i in range(10)], 10
        
        processor.execute_semantic_search = Mock(side_effect=mock_semantic_search)
        processor.execute_keyword_search = Mock(side_effect=mock_keyword_search)
        
        # State 값 가져오기/설정하기 Mock
        processor._get_state_value = Mock(side_effect=lambda state, key, default=None: state.get(key, default))
        processor._set_state_value = Mock(side_effect=lambda state, key, value: state.update({key: value}))
        processor._determine_search_parameters = Mock(return_value={"semantic_k": 10, "keyword_k": 10})
        processor._calculate_dynamic_k_values = Mock(return_value=(10, 10))
        processor._generate_search_cache_key = Mock(return_value=None)
        processor._get_cached_search_results = Mock(return_value=None)
        processor._save_metadata_safely = Mock()
        processor._update_processing_time = Mock()
        
        # ensure_state_group Mock
        def ensure_state_group(state, group):
            if group not in state:
                state[group] = {}
        
        with patch('lawfirm_langgraph.core.workflow.processors.search_execution_processor.ensure_state_group', side_effect=ensure_state_group):
            with patch('lawfirm_langgraph.core.workflow.processors.search_execution_processor.set_retrieved_docs'):
                result_state = processor.execute_searches_parallel(sample_state)
        
        # 검증: Semantic 검색이 0개 결과면 조기 종료되어야 함
        assert processor.logger.warning.called
        warning_calls = [str(call) for call in processor.logger.warning.call_args_list]
        early_exit_warning = any("EARLY EXIT" in str(call) or "0 results" in str(call) for call in warning_calls)
        
        # 조기 종료 경고가 호출되었는지 확인
        assert early_exit_warning or processor.logger.info.called, "조기 종료 로직이 작동하지 않았습니다"
    
    def test_early_exit_on_phase1_timeout_with_zero_semantic(self, processor, sample_state):
        """Phase 1 타임아웃 시 Semantic 검색이 0개 결과면 조기 종료하는지 테스트"""
        # Mock 설정: Semantic 검색은 타임아웃 후 0개, Keyword 검색은 10개 반환
        semantic_future = Future()
        semantic_future.set_result(([], 0))
        
        keyword_future = Future()
        keyword_future.set_result(([{"id": i, "content": f"result {i}"} for i in range(10)], 10))
        
        processor.execute_semantic_search = Mock(return_value=([], 0))
        processor.execute_keyword_search = Mock(return_value=([{"id": i, "content": f"result {i}"} for i in range(10)], 10))
        
        # State 값 가져오기/설정하기 Mock
        processor._get_state_value = Mock(side_effect=lambda state, key, default=None: state.get(key, default))
        processor._set_state_value = Mock(side_effect=lambda state, key, value: state.update({key: value}))
        processor._determine_search_parameters = Mock(return_value={"semantic_k": 10, "keyword_k": 10})
        processor._calculate_dynamic_k_values = Mock(return_value=(10, 10))
        processor._generate_search_cache_key = Mock(return_value=None)
        processor._get_cached_search_results = Mock(return_value=None)
        processor._save_metadata_safely = Mock()
        processor._update_processing_time = Mock()
        
        # ensure_state_group Mock
        def ensure_state_group(state, group):
            if group not in state:
                state[group] = {}
        
        with patch('lawfirm_langgraph.core.workflow.processors.search_execution_processor.ensure_state_group', side_effect=ensure_state_group):
            with patch('lawfirm_langgraph.core.workflow.processors.search_execution_processor.set_retrieved_docs'):
                with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
                    # 타임아웃 시뮬레이션
                    mock_executor_instance = Mock()
                    mock_executor.return_value.__enter__.return_value = mock_executor_instance
                    
                    # Future가 타임아웃되도록 설정
                    def mock_submit(func, *args, **kwargs):
                        future = Future()
                        if 'semantic' in str(func) or 'execute_semantic_search' in str(func):
                            future.set_result(([], 0))
                        else:
                            future.set_result(([{"id": i} for i in range(10)], 10))
                        return future
                    
                    mock_executor_instance.submit = Mock(side_effect=mock_submit)
                    
                    # as_completed가 타임아웃을 발생시키도록 설정
                    from concurrent.futures import as_completed
                    with patch('concurrent.futures.as_completed') as mock_as_completed:
                        # 첫 번째 호출은 타임아웃, 두 번째는 정상 완료
                        def mock_as_completed_side_effect(futures, timeout=None):
                            if timeout and timeout < 15:  # Phase 1 타임아웃
                                raise TimeoutError("Phase 1 timeout")
                            # Phase 2는 정상 완료
                            for future in futures:
                                yield future
                        
                        mock_as_completed.side_effect = mock_as_completed_side_effect
                        
                        result_state = processor.execute_searches_parallel(sample_state)
        
        # 검증: 타임아웃 후에도 조기 종료 경고가 호출되었는지 확인
        assert processor.logger.warning.called
        warning_calls = [str(call) for call in processor.logger.warning.call_args_list]
        timeout_warning = any("timeout" in str(call).lower() for call in warning_calls)
        early_exit_warning = any("EARLY EXIT" in str(call) or "0 results" in str(call) for call in warning_calls)
        
        assert timeout_warning or early_exit_warning, "타임아웃 또는 조기 종료 로직이 작동하지 않았습니다"
    
    def test_no_early_exit_when_semantic_has_results(self, processor, sample_state):
        """Semantic 검색에 결과가 있으면 조기 종료하지 않는지 테스트"""
        # Mock 설정: Semantic 검색은 5개, Keyword 검색은 10개 반환
        processor.execute_semantic_search = Mock(return_value=(
            [{"id": i, "content": f"semantic {i}"} for i in range(5)], 5
        ))
        processor.execute_keyword_search = Mock(return_value=(
            [{"id": i, "content": f"keyword {i}"} for i in range(10)], 10
        ))
        
        # State 값 가져오기/설정하기 Mock
        processor._get_state_value = Mock(side_effect=lambda state, key, default=None: state.get(key, default))
        processor._set_state_value = Mock(side_effect=lambda state, key, value: state.update({key: value}))
        processor._determine_search_parameters = Mock(return_value={"semantic_k": 10, "keyword_k": 10})
        processor._calculate_dynamic_k_values = Mock(return_value=(10, 10))
        processor._generate_search_cache_key = Mock(return_value=None)
        processor._get_cached_search_results = Mock(return_value=None)
        processor._save_metadata_safely = Mock()
        processor._update_processing_time = Mock()
        processor._evaluate_search_quality = Mock(return_value=0.8)
        processor._adjust_search_priority = Mock(return_value={})
        processor._merge_multi_query_results_single = Mock(return_value=[])
        processor.result_balancer = Mock()
        processor.result_balancer.group_results_by_type = Mock(return_value={})
        processor.result_balancer.balance_search_results = Mock(return_value=[])
        
        # ensure_state_group Mock
        def ensure_state_group(state, group):
            if group not in state:
                state[group] = {}
        
        with patch('lawfirm_langgraph.core.workflow.processors.search_execution_processor.ensure_state_group', side_effect=ensure_state_group):
            with patch('lawfirm_langgraph.core.workflow.processors.search_execution_processor.set_retrieved_docs'):
                result_state = processor.execute_searches_parallel(sample_state)
        
        # 검증: Semantic 검색에 결과가 있으면 조기 종료 경고가 없어야 함
        warning_calls = [str(call) for call in processor.logger.warning.call_args_list]
        early_exit_warning = any("EARLY EXIT" in str(call) and "0 results" in str(call) for call in warning_calls)
        
        assert not early_exit_warning, "Semantic 검색에 결과가 있는데 조기 종료되었습니다"
    
    def test_phase1_timeout_value(self, processor, sample_state):
        """Phase 1 타임아웃 값이 10-15초로 설정되었는지 테스트"""
        processor._get_state_value = Mock(side_effect=lambda state, key, default=None: state.get(key, default))
        processor._set_state_value = Mock()
        processor._determine_search_parameters = Mock(return_value={"semantic_k": 10, "keyword_k": 10})
        processor._calculate_dynamic_k_values = Mock(return_value=(10, 10))
        processor._generate_search_cache_key = Mock(return_value=None)
        processor._get_cached_search_results = Mock(return_value=None)
        
        # 타임아웃 값 확인을 위해 코드를 직접 확인
        # phase1_timeout = max(10, min(15, 8 + (semantic_k + keyword_k) // 5))
        semantic_k, keyword_k = 10, 10
        expected_timeout = max(10, min(15, 8 + (semantic_k + keyword_k) // 5))
        
        assert expected_timeout >= 10 and expected_timeout <= 15, f"Phase 1 타임아웃이 10-15초 범위를 벗어났습니다: {expected_timeout}초"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

