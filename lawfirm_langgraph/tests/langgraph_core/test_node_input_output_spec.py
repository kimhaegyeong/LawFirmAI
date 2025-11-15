# -*- coding: utf-8 -*-
"""
Node Input/Output Spec 테스트
langgraph_core/nodes/node_input_output_spec.py 단위 테스트
"""

import pytest
from typing import Dict, Any

from lawfirm_langgraph.langgraph_core.nodes.node_input_output_spec import (
    NodeIOSpec,
    NodeCategory,
    NODE_SPECS,
    get_node_spec,
    validate_node_input,
    get_required_state_groups,
    get_output_state_groups,
    get_all_node_names,
    get_nodes_by_category,
    validate_workflow_flow
)


class TestNodeCategory:
    """NodeCategory Enum 테스트"""
    
    def test_enum_values(self):
        """Enum 값 확인"""
        assert NodeCategory.INPUT.value == "input"
        assert NodeCategory.CLASSIFICATION.value == "classification"
        assert NodeCategory.SEARCH.value == "search"
        assert NodeCategory.GENERATION.value == "generation"
        assert NodeCategory.VALIDATION.value == "validation"
        assert NodeCategory.ENHANCEMENT.value == "enhancement"
        assert NodeCategory.CONTROL.value == "control"


class TestNodeIOSpec:
    """NodeIOSpec 테스트"""
    
    def test_init(self):
        """초기화 테스트"""
        spec = NodeIOSpec(
            node_name="test_node",
            category=NodeCategory.INPUT,
            description="Test node",
            required_input={"query": "질문"},
            optional_input={},
            output={"result": "결과"},
            required_state_groups={"input"},
            output_state_groups={"output"}
        )
        
        assert spec.node_name == "test_node"
        assert spec.category == NodeCategory.INPUT
        assert spec.description == "Test node"
        assert "query" in spec.required_input
        assert "result" in spec.output
    
    def test_validate_input_success(self):
        """Input 유효성 검증 성공 테스트"""
        spec = NodeIOSpec(
            node_name="test_node",
            category=NodeCategory.INPUT,
            description="Test node",
            required_input={"query": "질문"},
            optional_input={},
            output={},
            required_state_groups={"input"},
            output_state_groups={}
        )
        
        state = {"query": "Test query"}
        is_valid, error = spec.validate_input(state)
        
        assert is_valid is True
        assert error is None
    
    def test_validate_input_missing_field(self):
        """Input 유효성 검증 실패 테스트 (필수 필드 누락)"""
        spec = NodeIOSpec(
            node_name="test_node",
            category=NodeCategory.INPUT,
            description="Test node",
            required_input={"query": "질문", "answer": "답변"},
            optional_input={},
            output={},
            required_state_groups={"input"},
            output_state_groups={}
        )
        
        state = {"query": "Test query"}
        is_valid, error = spec.validate_input(state)
        
        assert is_valid is False
        assert "answer" in error
    
    def test_check_field_in_state_nested(self):
        """State에서 필드 확인 - 중첩 구조 테스트"""
        spec = NodeIOSpec(
            node_name="test_node",
            category=NodeCategory.INPUT,
            description="Test node",
            required_input={"query": "질문"},
            optional_input={},
            output={},
            required_state_groups={"input"},
            output_state_groups={}
        )
        
        state = {"input": {"query": "Test query"}}
        result = spec._check_field_in_state("query", state)
        
        assert result is True
    
    def test_check_field_in_state_flat(self):
        """State에서 필드 확인 - 평면 구조 테스트"""
        spec = NodeIOSpec(
            node_name="test_node",
            category=NodeCategory.INPUT,
            description="Test node",
            required_input={"query": "질문"},
            optional_input={},
            output={},
            required_state_groups={"input"},
            output_state_groups={}
        )
        
        state = {"query": "Test query"}
        result = spec._check_field_in_state("query", state)
        
        assert result is True
    
    def test_check_field_in_state_group(self):
        """State에서 필드 확인 - 그룹 내부 테스트"""
        spec = NodeIOSpec(
            node_name="test_node",
            category=NodeCategory.INPUT,
            description="Test node",
            required_input={"query": "질문"},
            optional_input={},
            output={},
            required_state_groups={"input"},
            output_state_groups={}
        )
        
        state = {"search": {"query": "Test query"}}
        result = spec._check_field_in_state("query", state)
        
        assert result is True


class TestNodeSpecs:
    """NODE_SPECS 테스트"""
    
    def test_node_specs_not_empty(self):
        """노드 사양이 비어있지 않은지 확인"""
        assert len(NODE_SPECS) > 0
    
    def test_classify_query_spec(self):
        """classify_query 노드 사양 테스트"""
        spec = NODE_SPECS.get("classify_query")
        assert spec is not None
        assert spec.category == NodeCategory.CLASSIFICATION
        assert "query" in spec.required_input
        assert "query_type" in spec.output
    
    def test_generate_answer_enhanced_spec(self):
        """generate_answer_enhanced 노드 사양 테스트"""
        spec = NODE_SPECS.get("generate_answer_enhanced")
        assert spec is not None
        assert spec.category == NodeCategory.GENERATION
        assert "query" in spec.required_input
        assert "answer" in spec.output


class TestHelperFunctions:
    """헬퍼 함수 테스트"""
    
    def test_get_node_spec_exists(self):
        """노드 사양 조회 - 존재하는 경우"""
        spec = get_node_spec("classify_query")
        assert spec is not None
        assert spec.node_name == "classify_query"
    
    def test_get_node_spec_not_exists(self):
        """노드 사양 조회 - 존재하지 않는 경우"""
        spec = get_node_spec("non_existent_node")
        assert spec is None
    
    def test_validate_node_input_valid(self):
        """노드 Input 유효성 검증 - 유효한 경우"""
        state = {"query": "Test query"}
        is_valid, error = validate_node_input("classify_query", state)
        
        assert is_valid is True
        assert error is None
    
    def test_validate_node_input_invalid(self):
        """노드 Input 유효성 검증 - 유효하지 않은 경우"""
        state = {}
        is_valid, error = validate_node_input("classify_query", state)
        
        assert is_valid is False
        assert error is not None
    
    def test_validate_node_input_no_spec(self):
        """노드 Input 유효성 검증 - 사양이 없는 경우"""
        state = {}
        is_valid, error = validate_node_input("non_existent_node", state)
        
        assert is_valid is True
        assert error is None
    
    def test_get_required_state_groups(self):
        """필수 State 그룹 조회 테스트"""
        groups = get_required_state_groups("classify_query")
        assert isinstance(groups, set)
        assert "input" in groups
    
    def test_get_required_state_groups_no_spec(self):
        """필수 State 그룹 조회 - 사양이 없는 경우"""
        groups = get_required_state_groups("non_existent_node")
        assert groups == set()
    
    def test_get_output_state_groups(self):
        """출력 State 그룹 조회 테스트"""
        groups = get_output_state_groups("classify_query")
        assert isinstance(groups, set)
        assert "classification" in groups
    
    def test_get_output_state_groups_no_spec(self):
        """출력 State 그룹 조회 - 사양이 없는 경우"""
        groups = get_output_state_groups("non_existent_node")
        assert groups == set()
    
    def test_get_all_node_names(self):
        """모든 노드 이름 조회 테스트"""
        node_names = get_all_node_names()
        assert isinstance(node_names, list)
        assert len(node_names) > 0
        assert "classify_query" in node_names
    
    def test_get_nodes_by_category(self):
        """카테고리별 노드 조회 테스트"""
        classification_nodes = get_nodes_by_category(NodeCategory.CLASSIFICATION)
        assert isinstance(classification_nodes, list)
        assert len(classification_nodes) > 0
        assert all(node.category == NodeCategory.CLASSIFICATION for node in classification_nodes)
    
    def test_get_nodes_by_category_empty(self):
        """카테고리별 노드 조회 - 해당 카테고리 노드가 없는 경우"""
        nodes = get_nodes_by_category(NodeCategory.INPUT)
        assert isinstance(nodes, list)


class TestWorkflowFlowValidation:
    """워크플로우 흐름 검증 테스트"""
    
    def test_validate_workflow_flow(self):
        """워크플로우 흐름 검증 테스트"""
        result = validate_workflow_flow()
        
        assert "valid" in result
        assert "issues" in result
        assert "total_nodes" in result
        assert isinstance(result["valid"], bool)
        assert isinstance(result["issues"], list)
        assert isinstance(result["total_nodes"], int)
        assert result["total_nodes"] > 0

