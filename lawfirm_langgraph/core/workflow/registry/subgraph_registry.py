# -*- coding: utf-8 -*-
"""
Subgraph Registry
서브그래프 레지스트리 패턴 구현
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SubgraphRegistry:
    """서브그래프 레지스트리"""
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """
        SubgraphRegistry 초기화
        
        Args:
            logger_instance: 로거 인스턴스
        """
        self._subgraphs: Dict[str, Any] = {}
        self.logger = logger_instance or logger
    
    def register(self, name: str, subgraph: Any) -> None:
        """
        서브그래프 등록
        
        Args:
            name: 서브그래프 이름
            subgraph: 컴파일된 서브그래프
        """
        self._subgraphs[name] = subgraph
        self.logger.debug(f"서브그래프 등록: {name}")
    
    def get_all_subgraphs(self) -> Dict[str, Any]:
        """
        모든 서브그래프 반환
        
        Returns:
            서브그래프 딕셔너리
        """
        return self._subgraphs.copy()
    
    def get_subgraph(self, name: str) -> Optional[Any]:
        """
        특정 서브그래프 반환
        
        Args:
            name: 서브그래프 이름
        
        Returns:
            서브그래프 또는 None
        """
        return self._subgraphs.get(name)
    
    def has_subgraph(self, name: str) -> bool:
        """
        서브그래프 존재 여부 확인
        
        Args:
            name: 서브그래프 이름
        
        Returns:
            존재 여부
        """
        return name in self._subgraphs
    
    def remove_subgraph(self, name: str) -> bool:
        """
        서브그래프 제거
        
        Args:
            name: 서브그래프 이름
        
        Returns:
            제거 성공 여부
        """
        if name in self._subgraphs:
            del self._subgraphs[name]
            self.logger.debug(f"서브그래프 제거: {name}")
            return True
        return False
    
    def clear(self) -> None:
        """모든 서브그래프 제거"""
        self._subgraphs.clear()
        self.logger.debug("모든 서브그래프 제거")

