"""
기본 청킹 전략

기존 chunk_paragraphs와 chunk_statute 로직을 래핑
"""
from typing import List, Dict, Any
from ..base_chunker import ChunkingStrategy, ChunkResult
from ...text_chunker import chunk_paragraphs, chunk_statute


class StandardChunkingStrategy(ChunkingStrategy):
    """기본 청킹 전략 (기존 로직 유지)"""
    
    def chunk(
        self,
        content: List[str],
        source_type: str,
        source_id: int,
        **kwargs
    ) -> List[ChunkResult]:
        """
        기본 청킹 수행
        
        Args:
            content: 청킹할 내용
            source_type: 원본 문서 타입
            source_id: 원본 문서 ID
            **kwargs: 추가 파라미터
                - min_chars: 최소 문자 수
                - max_chars: 최대 문자 수
                - overlap_ratio: 오버랩 비율
        
        Returns:
            ChunkResult 리스트
        """
        # 설정에서 파라미터 가져오기
        min_chars = kwargs.get("min_chars") or self.config.get("min_chars", 400)
        max_chars = kwargs.get("max_chars") or self.config.get("max_chars", 1800)
        overlap_ratio = kwargs.get("overlap_ratio") or self.config.get("overlap_ratio", 0.25)
        
        # 문서 타입에 따라 다른 청킹 함수 사용
        if source_type == "statute_article":
            # 법령 조문은 chunk_statute 사용
            chunks = chunk_statute(content, min_chars=min_chars, max_chars=max_chars, overlap_ratio=overlap_ratio)
        else:
            # 판례/결정례/해석례는 chunk_paragraphs 사용
            chunks = chunk_paragraphs(content, min_chars=min_chars, max_chars=max_chars, overlap_ratio=overlap_ratio)
        
        # ChunkResult로 변환
        results = []
        for i, chunk in enumerate(chunks):
            # 표준 메타데이터 추가
            metadata = self._add_standard_metadata(
                chunk,
                source_type,
                source_id,
                "standard",
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

