# -*- coding: utf-8 -*-
"""
Node Registry
노드 레지스트리 패턴 구현
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Callable, Dict, Optional, Type

logger = get_logger(__name__)


class NodeRegistry:
    """노드 레지스트리"""
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """
        NodeRegistry 초기화
        
        Args:
            logger_instance: 로거 인스턴스
        """
        self._nodes: Dict[str, Callable] = {}
        self.logger = logger_instance or logger
    
    def register(self, name: str, node_func: Callable) -> None:
        """
        노드 등록
        
        Args:
            name: 노드 이름
            node_func: 노드 함수
        """
        if not callable(node_func):
            raise ValueError(f"node_func는 callable이어야 합니다: {name}")
        
        self._nodes[name] = node_func
        self.logger.debug(f"노드 등록: {name}")
    
    def register_class(self, node_class: Type, prefix: str = "") -> None:
        """
        노드 클래스의 모든 정적 메서드를 등록
        
        Args:
            node_class: 노드 클래스
            prefix: 노드 이름 접두사
        """
        for attr_name in dir(node_class):
            if attr_name.startswith('_'):
                continue
            
            attr = getattr(node_class, attr_name)
            
            # 정적 메서드 또는 일반 메서드 확인
            if callable(attr) and not isinstance(attr, type):
                node_name = f"{prefix}_{attr_name}" if prefix else attr_name
                self._nodes[node_name] = attr
                self.logger.debug(f"노드 클래스에서 등록: {node_name}")
    
    def register_instance_methods(
        self,
        instance: object,
        method_names: Optional[list] = None,
        prefix: str = ""
    ) -> None:
        """
        인스턴스의 메서드들을 등록
        
        Args:
            instance: 인스턴스 객체
            method_names: 등록할 메서드 이름 리스트 (None이면 모든 public 메서드)
            prefix: 노드 이름 접두사
        """
        if method_names is None:
            # 모든 public 메서드 찾기
            method_names = [
                name for name in dir(instance)
                if not name.startswith('_') and callable(getattr(instance, name))
            ]
        
        for method_name in method_names:
            if hasattr(instance, method_name):
                method = getattr(instance, method_name)
                if callable(method):
                    node_name = f"{prefix}_{method_name}" if prefix else method_name
                    # 인스턴스 메서드를 바인딩하여 등록
                    self._nodes[node_name] = method
                    self.logger.debug(f"인스턴스 메서드 등록: {node_name}")
    
    def get_all_nodes(self) -> Dict[str, Callable]:
        """
        모든 노드 반환
        
        Returns:
            노드 딕셔너리
        """
        return self._nodes.copy()
    
    def get_node(self, name: str) -> Optional[Callable]:
        """
        특정 노드 반환
        
        Args:
            name: 노드 이름
        
        Returns:
            노드 함수 또는 None
        """
        return self._nodes.get(name)
    
    def has_node(self, name: str) -> bool:
        """
        노드 존재 여부 확인
        
        Args:
            name: 노드 이름
        
        Returns:
            존재 여부
        """
        return name in self._nodes
    
    def remove_node(self, name: str) -> bool:
        """
        노드 제거
        
        Args:
            name: 노드 이름
        
        Returns:
            제거 성공 여부
        """
        if name in self._nodes:
            del self._nodes[name]
            self.logger.debug(f"노드 제거: {name}")
            return True
        return False
    
    def clear(self) -> None:
        """모든 노드 제거"""
        self._nodes.clear()
        self.logger.debug("모든 노드 제거")

