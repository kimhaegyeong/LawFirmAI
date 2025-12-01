"""
하이브리드 청킹 전략

여러 크기의 청크를 생성하여 검색 품질 향상
"""
import uuid
from typing import List, Dict, Any, Optional
from ..base_chunker import ChunkingStrategy, ChunkResult
from ..config import ChunkingConfig
from ...text_chunker import chunk_paragraphs, chunk_statute


class HybridChunkingStrategy(ChunkingStrategy):
    """하이브리드 청킹 전략 (여러 크기의 청크 생성)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        하이브리드 청킹 전략 초기화
        
        Args:
            config: 청킹 설정 딕셔너리
        """
        super().__init__(config)
        self.config_manager = ChunkingConfig()
    
    def chunk(
        self,
        content: List[str],
        source_type: str,
        source_id: int,
        **kwargs
    ) -> List[ChunkResult]:
        """
        하이브리드 청킹 수행 (여러 크기의 청크 생성)
        
        Args:
            content: 청킹할 내용
            source_type: 원본 문서 타입
            source_id: 원본 문서 ID
            **kwargs: 추가 파라미터
        
        Returns:
            ChunkResult 리스트 (여러 크기 카테고리 포함)
        """
        # 청크 그룹 ID 생성 (같은 원본에서 나온 청크들을 그룹화)
        chunk_group_id = f"{source_type}_{source_id}_{uuid.uuid4().hex[:8]}"
        
        all_results = []
        
        # 각 크기 카테고리별로 청킹
        for size_category, size_config in self.config_manager.HYBRID_SIZE_CATEGORIES.items():
            min_chars = size_config["min_chars"]
            max_chars = size_config["max_chars"]
            overlap_ratio = size_config["overlap_ratio"]
            
            # 문서 타입에 따라 다른 청킹 함수 사용
            if source_type == "statute_article":
                chunks = chunk_statute(content, min_chars=min_chars, max_chars=max_chars, overlap_ratio=overlap_ratio)
            else:
                chunks = chunk_paragraphs(content, min_chars=min_chars, max_chars=max_chars, overlap_ratio=overlap_ratio)
            
            # ChunkResult로 변환
            for i, chunk in enumerate(chunks):
                # 표준 메타데이터 추가
                metadata = self._add_standard_metadata(
                    chunk,
                    source_type,
                    source_id,
                    "hybrid",
                    chunk_size_category=size_category,
                    chunk_group_id=chunk_group_id,
                    original_document_id=source_id
                )
                
                # 기존 chunk의 메타데이터도 포함
                if isinstance(chunk, dict):
                    for key, value in chunk.items():
                        if key not in ["text", "chunk_index"] and key not in metadata:
                            metadata[key] = value
                
                all_results.append(ChunkResult(
                    text=chunk.get("text", ""),
                    chunk_index=chunk.get("chunk_index", i),
                    metadata=metadata
                ))
        
        return all_results

