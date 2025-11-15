"""
청킹 전략 추상 클래스

모든 청킹 전략은 이 클래스를 상속받아 구현해야 합니다.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ChunkResult:
    """청킹 결과 데이터 클래스"""
    text: str
    chunk_index: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "text": self.text,
            "chunk_index": self.chunk_index,
            **self.metadata
        }


class ChunkingStrategy(ABC):
    """청킹 전략 추상 클래스"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        청킹 전략 초기화
        
        Args:
            config: 청킹 설정 딕셔너리
        """
        self.config = config or {}
        self.strategy_name = self.__class__.__name__
    
    @abstractmethod
    def chunk(
        self,
        content: List[str],
        source_type: str,
        source_id: int,
        **kwargs
    ) -> List[ChunkResult]:
        """
        내용을 청크로 분할
        
        Args:
            content: 청킹할 내용 (문장 리스트 또는 단락 리스트)
            source_type: 원본 문서 타입 (statute_article, case_paragraph, etc.)
            source_id: 원본 문서 ID
            **kwargs: 추가 파라미터
        
        Returns:
            ChunkResult 리스트
        """
        pass
    
    def _add_standard_metadata(
        self,
        chunk: Dict[str, Any],
        source_type: str,
        source_id: int,
        chunking_strategy: str,
        **additional_metadata
    ) -> Dict[str, Any]:
        """
        표준 메타데이터 추가
        
        Args:
            chunk: 청크 딕셔너리
            source_type: 원본 문서 타입
            source_id: 원본 문서 ID
            chunking_strategy: 청킹 전략 이름
            **additional_metadata: 추가 메타데이터
        
        Returns:
            메타데이터가 추가된 딕셔너리
        """
        metadata = {
            "chunking_strategy": chunking_strategy,
            "source_type": source_type,
            "source_id": source_id,
            **additional_metadata
        }
        
        # 기존 chunk의 메타데이터와 병합
        if isinstance(chunk, dict):
            chunk_metadata = chunk.get("metadata", {})
            if chunk_metadata:
                metadata.update(chunk_metadata)
            chunk["metadata"] = metadata
        
        return metadata
    
    def _categorize_chunk_size(self, text: str) -> str:
        """
        청크 크기 카테고리 분류
        
        Args:
            text: 청크 텍스트
        
        Returns:
            'small', 'medium', 'large'
        """
        length = len(text)
        if length < 800:
            return "small"
        elif length < 1500:
            return "medium"
        else:
            return "large"

