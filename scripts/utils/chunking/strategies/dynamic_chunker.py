"""
동적 청킹 전략

질문 유형에 따라 청킹 사이즈를 자동으로 조정
"""
from typing import List, Dict, Any, Optional
from ..base_chunker import ChunkingStrategy, ChunkResult
from ..config import ChunkingConfig, get_chunking_config
from ...text_chunker import chunk_paragraphs, chunk_statute


class DynamicChunkingStrategy(ChunkingStrategy):
    """질문 유형별 동적 청킹 전략"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, query_type: Optional[str] = None):
        """
        동적 청킹 전략 초기화
        
        Args:
            config: 청킹 설정 딕셔너리
            query_type: 질문 유형 (law_inquiry, precedent_search, etc.)
        """
        super().__init__(config)
        self.query_type = query_type
    
    def chunk(
        self,
        content: List[str],
        source_type: str,
        source_id: int,
        query_type: Optional[str] = None,
        **kwargs
    ) -> List[ChunkResult]:
        """
        질문 유형별 동적 청킹 수행
        
        Args:
            content: 청킹할 내용
            source_type: 원본 문서 타입
            source_id: 원본 문서 ID
            query_type: 질문 유형 (우선순위: 파라미터 > 인스턴스 변수)
            **kwargs: 추가 파라미터
        
        Returns:
            ChunkResult 리스트
        """
        # 질문 유형 결정 (파라미터 우선)
        effective_query_type = query_type or self.query_type
        
        # 질문 유형별 설정 가져오기
        chunk_config = get_chunking_config(query_type=effective_query_type, document_type=source_type)
        
        min_chars = chunk_config.get("min_chars", 400)
        max_chars = chunk_config.get("max_chars", 1800)
        overlap_ratio = chunk_config.get("overlap_ratio", 0.25)
        
        # 문서 타입에 따라 다른 청킹 함수 사용
        if source_type == "statute_article":
            chunks = chunk_statute(content, min_chars=min_chars, max_chars=max_chars, overlap_ratio=overlap_ratio)
        else:
            chunks = chunk_paragraphs(content, min_chars=min_chars, max_chars=max_chars, overlap_ratio=overlap_ratio)
        
        # ChunkResult로 변환
        results = []
        for i, chunk in enumerate(chunks):
            # 표준 메타데이터 추가
            metadata = self._add_standard_metadata(
                chunk,
                source_type,
                source_id,
                "dynamic",
                query_type=effective_query_type,
                chunk_size_category=self._categorize_chunk_size(chunk.get("text", ""))
            )
            
            # 기존 chunk의 메타데이터도 포함
            if isinstance(chunk, dict):
                for key, value in chunk.items():
                    if key not in ["text", "chunk_index"] and key not in metadata:
                        metadata[key] = value
            
            results.append(ChunkResult(
                text=chunk.get("text", ""),
                chunk_index=chunk.get("chunk_index", i),
                metadata=metadata
            ))
        
        return results

