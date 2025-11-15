"""
노드 필터링 로직
"""
from typing import Any, Optional, List
from .constants import StreamingConstants


class NodeFilter:
    """노드 필터링 로직"""
    
    # 분류/평가 노드 패턴
    CLASSIFICATION_NODE_PATTERNS = [
        "assessment", "classification", "complexity", "necessity", 
        "query_type", "urgency", "domain", "classify", "evaluate"
    ]
    
    def __init__(self, target_nodes: Optional[List[str]] = None):
        self.target_nodes = target_nodes or StreamingConstants.TARGET_NODES
    
    def is_target_node(
        self, 
        event_name: str, 
        event_parent: Any, 
        last_node_name: Optional[str]
    ) -> bool:
        """타겟 노드인지 확인"""
        if "generate_answer" in event_name.lower() or \
           "generate_and_validate" in event_name.lower() or \
           event_name in self.target_nodes:
            return True
        
        if isinstance(event_parent, dict):
            parent_node_name = event_parent.get("name", "")
            if parent_node_name and (
                "generate_answer" in parent_node_name.lower() or 
                "generate_and_validate" in parent_node_name.lower() or
                parent_node_name in self.target_nodes
            ):
                return True
        
        if last_node_name in self.target_nodes:
            if "Chat" in event_name or "LLM" in event_name or "Model" in event_name:
                return True
        
        return False
    
    def is_answer_generation_node(self, node_name: str) -> bool:
        """답변 생성 노드인지 확인"""
        return node_name in StreamingConstants.ANSWER_GENERATION_NODES
    
    def is_answer_completion_node(self, node_name: str) -> bool:
        """답변 생성 완료 노드인지 확인"""
        return node_name in StreamingConstants.ANSWER_COMPLETION_NODES
    
    def is_classification_node(self, node_name: str) -> bool:
        """분류/평가 노드인지 확인 (정상 동작이므로 스트림에서 제외)"""
        node_name_lower = node_name.lower()
        return any(pattern in node_name_lower for pattern in self.CLASSIFICATION_NODE_PATTERNS)

