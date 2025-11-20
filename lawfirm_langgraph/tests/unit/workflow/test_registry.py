# -*- coding: utf-8 -*-
"""
레지스트리 패턴 테스트
"""

import pytest
from unittest.mock import Mock, MagicMock
from core.workflow.registry.node_registry import NodeRegistry
from core.workflow.registry.subgraph_registry import SubgraphRegistry
from core.workflow.nodes.classification_nodes import ClassificationNodes


class TestNodeRegistry:
    """NodeRegistry 테스트"""
    
    @pytest.fixture
    def registry(self):
        """NodeRegistry 인스턴스"""
        return NodeRegistry()
    
    def test_register_node(self, registry):
        """노드 등록 테스트"""
        def test_node(state):
            return state
        
        registry.register("test_node", test_node)
        
        assert registry.has_node("test_node")
        assert registry.get_node("test_node") == test_node
    
    def test_register_class(self, registry):
        """노드 클래스 등록 테스트"""
        mock_workflow = Mock()
        classification_nodes = ClassificationNodes(
            workflow_instance=mock_workflow,
            logger_instance=Mock()
        )
        
        registry.register_class(classification_nodes, prefix="classification_")
        
        assert registry.has_node("classification_classify_query_and_complexity")
        assert registry.has_node("classification_direct_answer")
        assert registry.has_node("classification_classification_parallel")
        assert registry.has_node("classification_assess_urgency")
        assert registry.has_node("classification_resolve_multi_turn")
        assert registry.has_node("classification_route_expert")
    
    def test_get_all_nodes(self, registry):
        """모든 노드 조회 테스트"""
        registry.register("node1", Mock())
        registry.register("node2", Mock())
        
        all_nodes = registry.get_all_nodes()
        assert "node1" in all_nodes
        assert "node2" in all_nodes
        assert len(all_nodes) == 2
    
    def test_remove_node(self, registry):
        """노드 제거 테스트"""
        registry.register("test_node", Mock())
        registry.remove_node("test_node")
        
        assert not registry.has_node("test_node")
    
    def test_register_invalid_node(self, registry):
        """잘못된 노드 등록 테스트"""
        with pytest.raises(ValueError, match="callable이어야 합니다"):
            registry.register("invalid_node", "not_callable")
    
    def test_get_nonexistent_node(self, registry):
        """존재하지 않는 노드 조회 테스트"""
        result = registry.get_node("nonexistent")
        assert result is None
    
    def test_has_nonexistent_node(self, registry):
        """존재하지 않는 노드 확인 테스트"""
        assert not registry.has_node("nonexistent")
    
    def test_register_duplicate_node(self, registry):
        """중복 노드 등록 테스트 (덮어쓰기)"""
        def node1(state):
            return state
        
        def node2(state):
            return state
        
        registry.register("duplicate_node", node1)
        registry.register("duplicate_node", node2)
        
        assert registry.get_node("duplicate_node") == node2
    
    def test_initialization_with_logger(self):
        """로거 인스턴스로 초기화 테스트"""
        mock_logger = Mock()
        registry = NodeRegistry(logger_instance=mock_logger)
        
        assert registry.logger == mock_logger


class TestSubgraphRegistry:
    """SubgraphRegistry 테스트"""
    
    @pytest.fixture
    def registry(self):
        """SubgraphRegistry 인스턴스"""
        return SubgraphRegistry()
    
    def test_register_subgraph(self, registry):
        """서브그래프 등록 테스트"""
        mock_subgraph = Mock()
        registry.register("test_subgraph", mock_subgraph)
        
        assert registry.has_subgraph("test_subgraph")
        assert registry.get_subgraph("test_subgraph") == mock_subgraph
    
    def test_get_all_subgraphs(self, registry):
        """모든 서브그래프 조회 테스트"""
        mock_subgraph1 = Mock()
        mock_subgraph2 = Mock()
        
        registry.register("subgraph1", mock_subgraph1)
        registry.register("subgraph2", mock_subgraph2)
        
        all_subgraphs = registry.get_all_subgraphs()
        assert "subgraph1" in all_subgraphs
        assert "subgraph2" in all_subgraphs
        assert len(all_subgraphs) == 2
    
    def test_get_nonexistent_subgraph(self, registry):
        """존재하지 않는 서브그래프 조회 테스트"""
        result = registry.get_subgraph("nonexistent")
        assert result is None
    
    def test_has_nonexistent_subgraph(self, registry):
        """존재하지 않는 서브그래프 확인 테스트"""
        assert not registry.has_subgraph("nonexistent")
    
    def test_remove_subgraph(self, registry):
        """서브그래프 제거 테스트"""
        mock_subgraph = Mock()
        registry.register("test_subgraph", mock_subgraph)
        registry.remove_subgraph("test_subgraph")
        
        assert not registry.has_subgraph("test_subgraph")
    
    def test_register_duplicate_subgraph(self, registry):
        """중복 서브그래프 등록 테스트 (덮어쓰기)"""
        mock_subgraph1 = Mock()
        mock_subgraph2 = Mock()
        
        registry.register("duplicate_subgraph", mock_subgraph1)
        registry.register("duplicate_subgraph", mock_subgraph2)
        
        assert registry.get_subgraph("duplicate_subgraph") == mock_subgraph2
    
    def test_initialization_with_logger(self):
        """로거 인스턴스로 초기화 테스트"""
        mock_logger = Mock()
        registry = SubgraphRegistry(logger_instance=mock_logger)
        
        assert registry.logger == mock_logger

