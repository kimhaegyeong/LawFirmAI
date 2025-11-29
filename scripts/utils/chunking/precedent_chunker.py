# -*- coding: utf-8 -*-
"""
판례 전용 청킹 클래스
PostgreSQL precedent_contents 테이블의 섹션 타입별 차등 청킹
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    get_logger = lambda name: logging.getLogger(name)


@dataclass
class PrecedentChunk:
    """판례 청크 데이터 클래스"""
    chunk_content: str
    chunk_index: int
    chunk_length: int
    metadata: Dict[str, Any]
    metadata: Dict[str, Any]


class PrecedentChunker:
    """판례 전용 청킹 클래스"""
    
    # 섹션 타입별 청킹 설정
    SECTION_TYPE_CONFIG = {
        "판시사항": {
            "max_length": 500,  # 500자 이하면 청킹 안 함
            "chunk_size": None,  # 청킹 불필요
            "chunk_overlap": 0
        },
        "판결요지": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "min_length": 100  # 최소 길이
        },
        "판례내용": {
            "chunk_size": 800,
            "chunk_overlap": 150,
            "min_length": 200  # 최소 길이
        }
    }
    
    # 기본 설정 (섹션 타입이 명시되지 않은 경우)
    DEFAULT_CONFIG = {
        "chunk_size": 800,
        "chunk_overlap": 150,
        "min_length": 200
    }
    
    def __init__(self):
        """청킹 클래스 초기화"""
        self.logger = get_logger(__name__)
        
        # 문장 구분 패턴
        self.sentence_pattern = re.compile(r'[다이다]\.|\.\s+')
        
        # 문단 구분 패턴
        self.paragraph_pattern = re.compile(r'\n\n+')
    
    def chunk_precedent_content(
        self,
        precedent_content_id: int,
        section_type: str,
        section_content: str,
        precedent_metadata: Dict[str, Any],
        referenced_articles: Optional[str] = None,
        referenced_precedents: Optional[str] = None
    ) -> List[PrecedentChunk]:
        """
        판례 내용 청킹
        
        Args:
            precedent_content_id: precedent_contents 테이블의 id
            section_type: 섹션 유형 (판시사항, 판결요지, 판례내용)
            section_content: 섹션 내용
            precedent_metadata: precedents 테이블의 메타데이터
            referenced_articles: 참조조문
            referenced_precedents: 참조판례
        
        Returns:
            PrecedentChunk 리스트
        """
        try:
            # 섹션 타입별 설정 가져오기
            config = self.SECTION_TYPE_CONFIG.get(
                section_type,
                self.DEFAULT_CONFIG
            )
            
            # 판시사항은 짧으면 청킹 안 함
            if section_type == "판시사항":
                if len(section_content) <= config["max_length"]:
                    return [self._create_chunk(
                        content=section_content,
                        chunk_index=0,
                        precedent_content_id=precedent_content_id,
                        section_type=section_type,
                        precedent_metadata=precedent_metadata,
                        referenced_articles=referenced_articles,
                        referenced_precedents=referenced_precedents
                    )]
                # 길면 판례내용과 동일하게 처리
                config = self.SECTION_TYPE_CONFIG["판례내용"]
            
            # 의미 단위 청킹
            chunks = self._chunk_by_meaning(
                content=section_content,
                section_type=section_type,
                config=config
            )
            
            # PrecedentChunk 객체로 변환
            result_chunks = []
            for i, chunk_content in enumerate(chunks):
                chunk = self._create_chunk(
                    content=chunk_content,
                    chunk_index=i,
                    precedent_content_id=precedent_content_id,
                    section_type=section_type,
                    precedent_metadata=precedent_metadata,
                    referenced_articles=referenced_articles,
                    referenced_precedents=referenced_precedents
                )
                result_chunks.append(chunk)
            
            self.logger.debug(
                f"청킹 완료: precedent_content_id={precedent_content_id}, "
                f"section_type={section_type}, chunks={len(result_chunks)}"
            )
            
            return result_chunks
            
        except Exception as e:
            self.logger.error(f"청킹 실패: {e}")
            return []
    
    def _chunk_by_meaning(
        self,
        content: str,
        section_type: str,
        config: Dict[str, Any]
    ) -> List[str]:
        """
        의미 단위 청킹
        
        Args:
            content: 청킹할 내용
            section_type: 섹션 유형
            config: 청킹 설정
        
        Returns:
            청크 리스트
        """
        chunk_size = config.get("chunk_size", 800)
        chunk_overlap = config.get("chunk_overlap", 150)
        min_length = config.get("min_length", 200)
        
        # 1단계: 문단 단위 분할
        paragraphs = self.paragraph_pattern.split(content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return [content] if content.strip() else []
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # 문단이 너무 길면 문장 단위로 분할
            if len(para) > chunk_size:
                # 현재 청크 저장
                if current_chunk and len(current_chunk) >= min_length:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # 긴 문단을 문장 단위로 분할
                sentences = self._split_into_sentences(para)
                for sent in sentences:
                    if len(current_chunk) + len(sent) > chunk_size:
                        if current_chunk and len(current_chunk) >= min_length:
                            chunks.append(current_chunk)
                            # Overlap 적용
                            current_chunk = self._apply_overlap(
                                current_chunk,
                                chunk_overlap
                            ) + sent
                        else:
                            current_chunk = sent
                    else:
                        current_chunk += sent if not current_chunk else "\n\n" + sent
            else:
                # 문단을 현재 청크에 추가
                if len(current_chunk) + len(para) > chunk_size:
                    if current_chunk and len(current_chunk) >= min_length:
                        chunks.append(current_chunk)
                        # Overlap 적용
                        current_chunk = self._apply_overlap(
                            current_chunk,
                            chunk_overlap
                        ) + "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
        
        # 마지막 청크 추가
        if current_chunk and len(current_chunk) >= min_length:
            chunks.append(current_chunk)
        elif current_chunk:
            # 최소 길이 미만이면 이전 청크에 병합
            if chunks:
                chunks[-1] += "\n\n" + current_chunk
            else:
                chunks.append(current_chunk)
        
        return chunks if chunks else [content]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """문장 단위로 분할"""
        # 문장 끝 패턴으로 분할
        sentences = self.sentence_pattern.split(text)
        result = []
        for i, sent in enumerate(sentences):
            if sent.strip():
                # 문장 끝 기호 복원
                if i < len(sentences) - 1:
                    sent += "."
                result.append(sent.strip())
        return result
    
    def _apply_overlap(self, text: str, overlap_size: int) -> str:
        """Overlap 적용 (뒤에서 overlap_size만큼 추출)"""
        if len(text) <= overlap_size:
            return text
        return text[-overlap_size:]
    
    def _create_chunk(
        self,
        content: str,
        chunk_index: int,
        precedent_content_id: int,
        section_type: str,
        precedent_metadata: Dict[str, Any],
        referenced_articles: Optional[str] = None,
        referenced_precedents: Optional[str] = None
    ) -> PrecedentChunk:
        """PrecedentChunk 객체 생성"""
        metadata = {
            "precedent_content_id": precedent_content_id,
            "section_type": section_type,
            "precedent_id": precedent_metadata.get("precedent_id"),
            "case_name": precedent_metadata.get("case_name"),
            "case_number": precedent_metadata.get("case_number"),
            "decision_date": str(precedent_metadata.get("decision_date")) if precedent_metadata.get("decision_date") else None,
            "court_name": precedent_metadata.get("court_name"),
            "domain": precedent_metadata.get("domain"),
            "referenced_articles": referenced_articles,
            "referenced_precedents": referenced_precedents
        }
        
        return PrecedentChunk(
            chunk_content=content,
            chunk_index=chunk_index,
            chunk_length=len(content),
            metadata=metadata
        )

