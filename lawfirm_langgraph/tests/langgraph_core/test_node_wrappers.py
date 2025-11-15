# -*- coding: utf-8 -*-
"""
Node Wrappers 테스트
langgraph_core/nodes/node_wrappers.py 단위 테스트
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from lawfirm_langgraph.langgraph_core.nodes.node_wrappers import (
    with_state_optimization,
    _global_search_results_cache
)


class TestWithStateOptimization:
    """with_state_optimization 데코레이터 테스트"""
    
    def test_decorator_basic(self):
        """기본 데코레이터 동작 테스트"""
        @with_state_optimization("test_node", enable_reduction=False)
        def test_node(state: Dict[str, Any]) -> Dict[str, Any]:
            return {"result": "success"}
        
        state = {"query": "Test query"}
        result = test_node(state)
        
        assert "result" in result
        assert result["result"] == "success"
    
    def test_decorator_with_input_group(self):
        """input 그룹이 있는 state 테스트"""
        @with_state_optimization("test_node", enable_reduction=False)
        def test_node(state: Dict[str, Any]) -> Dict[str, Any]:
            return {"result": "success"}
        
        state = {"input": {"query": "Test query"}}
        result = test_node(state)
        
        assert "result" in result
    
    def test_decorator_with_invalid_state(self):
        """유효하지 않은 state 테스트"""
        @with_state_optimization("test_node", enable_reduction=False)
        def test_node(state: Dict[str, Any]) -> Dict[str, Any]:
            return {"result": "success"}
        
        state = "invalid_state"
        result = test_node(state)
        
        assert isinstance(result, dict)
    
    def test_decorator_with_no_args(self):
        """인자가 없는 경우 테스트"""
        @with_state_optimization("test_node", enable_reduction=False)
        def test_node(state: Dict[str, Any]) -> Dict[str, Any]:
            return {"result": "success"}
        
        result = test_node()
        
        assert isinstance(result, dict)
    
    @patch('lawfirm_langgraph.langgraph_core.nodes.node_wrappers.validate_state_for_node')
    def test_decorator_with_validation(self, mock_validate):
        """검증 포함 데코레이터 테스트"""
        mock_validate.return_value = (True, None, {"query": "Test query"})
        
        @with_state_optimization("test_node", enable_reduction=False)
        def test_node(state: Dict[str, Any]) -> Dict[str, Any]:
            return {"result": "success"}
        
        state = {"query": "Test query"}
        result = test_node(state)
        
        assert "result" in result
        mock_validate.assert_called_once()
    
    @patch('lawfirm_langgraph.langgraph_core.nodes.node_wrappers.StateReducer')
    def test_decorator_with_reduction(self, mock_reducer_class):
        """State Reduction 포함 데코레이터 테스트"""
        mock_reducer = MagicMock()
        mock_reducer.reduce_state_for_node.return_value = {"query": "Test query"}
        mock_reducer_class.return_value = mock_reducer
        
        @with_state_optimization("test_node", enable_reduction=True)
        def test_node(state: Dict[str, Any]) -> Dict[str, Any]:
            return {"result": "success"}
        
        state = {"query": "Test query"}
        result = test_node(state)
        
        assert "result" in result
        mock_reducer.reduce_state_for_node.assert_called_once()
    
    def test_decorator_preserves_function_metadata(self):
        """함수 메타데이터 보존 테스트"""
        @with_state_optimization("test_node", enable_reduction=False)
        def test_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Test function docstring"""
            return {"result": "success"}
        
        assert test_node.__name__ == "test_node"
        assert "Test function docstring" in test_node.__doc__

