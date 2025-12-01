"""
청킹 전략 팩토리

전략 선택 및 인스턴스 생성
"""
from typing import Dict, Any, Optional, Type, List
from .base_chunker import ChunkingStrategy
from .strategies.standard_chunker import StandardChunkingStrategy
from .strategies.dynamic_chunker import DynamicChunkingStrategy
from .strategies.hybrid_chunker import HybridChunkingStrategy
from .config import get_chunking_config


class ChunkingFactory:
    """청킹 전략 팩토리"""
    
    # 전략 이름과 클래스 매핑
    STRATEGY_CLASSES: Dict[str, Type[ChunkingStrategy]] = {
        "standard": StandardChunkingStrategy,
        "dynamic": DynamicChunkingStrategy,
        "hybrid": HybridChunkingStrategy
    }
    
    @classmethod
    def create_strategy(
        cls,
        strategy_name: str = "standard",
        config: Optional[Dict[str, Any]] = None,
        query_type: Optional[str] = None,
        **kwargs
    ) -> ChunkingStrategy:
        """
        청킹 전략 인스턴스 생성
        
        Args:
            strategy_name: 전략 이름 ('standard', 'dynamic', 'hybrid')
            config: 청킹 설정 딕셔너리
            query_type: 질문 유형 (동적 청킹 시 사용)
            **kwargs: 추가 파라미터
        
        Returns:
            ChunkingStrategy 인스턴스
        """
        strategy_class = cls.STRATEGY_CLASSES.get(strategy_name)
        
        if not strategy_class:
            raise ValueError(f"Unknown chunking strategy: {strategy_name}")
        
        # 설정이 없으면 기본 설정 로드
        if config is None:
            config = get_chunking_config(query_type=query_type)
        
        # 전략별 특수 처리
        if strategy_name == "dynamic":
            return strategy_class(config=config, query_type=query_type)
        else:
            return strategy_class(config=config)
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """
        사용 가능한 전략 목록 반환
        
        Returns:
            전략 이름 리스트
        """
        return list(cls.STRATEGY_CLASSES.keys())
    
    @classmethod
    def register_strategy(
        cls,
        name: str,
        strategy_class: Type[ChunkingStrategy]
    ):
        """
        새로운 전략 등록 (확장성)
        
        Args:
            name: 전략 이름
            strategy_class: 전략 클래스
        """
        if not issubclass(strategy_class, ChunkingStrategy):
            raise TypeError(f"Strategy class must inherit from ChunkingStrategy")
        
        cls.STRATEGY_CLASSES[name] = strategy_class

