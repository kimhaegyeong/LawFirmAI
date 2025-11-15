# -*- coding: utf-8 -*-
"""
Workflow Utils 테스트
langgraph_core/utils/workflow_utils.py 단위 테스트
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from lawfirm_langgraph.langgraph_core.utils.workflow_utils import WorkflowUtils


class TestWorkflowUtils:
    """WorkflowUtils 테스트"""
    
    def test_get_state_value_exists(self):
        """State 값 가져오기 - 값이 있는 경우"""
        state = {"query": "Test query", "answer": "Test answer"}
        value = WorkflowUtils.get_state_value(state, "query")
        
        assert value == "Test query"
    
    def test_get_state_value_not_exists(self):
        """State 값 가져오기 - 값이 없는 경우"""
        state = {"query": "Test query"}
        value = WorkflowUtils.get_state_value(state, "non_existent", "default")
        
        assert value == "default"
    
    def test_get_state_value_none(self):
        """State 값 가져오기 - None인 경우"""
        state = {"query": None}
        value = WorkflowUtils.get_state_value(state, "query", "default")
        
        assert value == "default"
    
    def test_set_state_value(self):
        """State 값 설정 테스트"""
        state = {"query": "Test query"}
        
        with patch('lawfirm_langgraph.langgraph_core.utils.workflow_utils.set_field') as mock_set_field:
            WorkflowUtils.set_state_value(state, "answer", "Test answer")
            mock_set_field.assert_called_once_with(state, "answer", "Test answer", None)
    
    def test_set_state_value_with_logger(self):
        """Logger 포함 State 값 설정 테스트"""
        state = {"query": "Test query"}
        logger = Mock()
        
        with patch('lawfirm_langgraph.langgraph_core.utils.workflow_utils.set_field') as mock_set_field:
            WorkflowUtils.set_state_value(state, "answer", "Test answer", logger)
            mock_set_field.assert_called_once_with(state, "answer", "Test answer", logger)

