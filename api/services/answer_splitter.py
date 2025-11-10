"""
답변 분할 유틸리티 모듈
긴 답변을 논리적 섹션으로 분할하여 청크 단위로 제공
"""
import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class AnswerChunk:
    """답변 청크 정보"""
    content: str
    chunk_index: int
    total_chunks: int
    is_complete: bool
    has_more: bool


class AnswerSplitter:
    """답변을 논리적 섹션으로 분할하는 클래스"""
    
    SECTION_PATTERNS = [
        r'^##\s+',  # 마크다운 헤더 (##)
        r'^###\s+',  # 마크다운 서브헤더 (###)
        r'^[0-9]+[\.\)]\s+',  # 번호 목록
        r'^[가-힣]+[\.\)]\s+',  # 한글 번호 목록
        r'^\*\s+',  # 불릿 포인트
        r'^-\s+',  # 대시 목록
    ]
    
    PARAGRAPH_BREAK = r'\n\n+'
    
    def __init__(self, chunk_size: int = 2000):
        """
        Args:
            chunk_size: 각 청크의 최대 길이 (문자 수)
        """
        self.chunk_size = chunk_size
    
    def split_answer(self, answer: str) -> List[AnswerChunk]:
        """
        답변을 논리적 섹션으로 분할
        
        Args:
            answer: 분할할 답변
            
        Returns:
            AnswerChunk 리스트
        """
        if not answer:
            return []
        
        if len(answer) <= self.chunk_size:
            return [AnswerChunk(
                content=answer,
                chunk_index=0,
                total_chunks=1,
                is_complete=True,
                has_more=False
            )]
        
        sections = self._split_into_sections(answer)
        chunks = self._group_sections_into_chunks(sections)
        
        return chunks
    
    def _split_into_sections(self, answer: str) -> List[Dict[str, Any]]:
        """답변을 논리적 섹션으로 분리"""
        sections = []
        current_section = []
        current_section_type = "paragraph"
        
        lines = answer.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            is_header = False
            for pattern in self.SECTION_PATTERNS:
                if re.match(pattern, line_stripped):
                    if current_section:
                        sections.append({
                            "type": current_section_type,
                            "content": '\n'.join(current_section),
                            "start_index": i - len(current_section),
                            "end_index": i - 1
                        })
                    current_section = [line]
                    current_section_type = "header"
                    is_header = True
                    break
            
            if not is_header:
                if not line_stripped:
                    if current_section and current_section[-1].strip():
                        current_section.append(line)
                else:
                    current_section.append(line)
        
        if current_section:
            sections.append({
                "type": current_section_type,
                "content": '\n'.join(current_section),
                "start_index": len(lines) - len(current_section),
                "end_index": len(lines) - 1
            })
        
        return sections
    
    def _group_sections_into_chunks(
        self, 
        sections: List[Dict[str, Any]]
    ) -> List[AnswerChunk]:
        """섹션을 청크로 그룹화"""
        chunks = []
        current_chunk_content = []
        current_chunk_length = 0
        chunk_index = 0
        
        for section in sections:
            section_content = section["content"]
            section_length = len(section_content)
            
            if current_chunk_length + section_length <= self.chunk_size:
                current_chunk_content.append(section_content)
                current_chunk_length += section_length + 2
            else:
                if current_chunk_content:
                    chunks.append(AnswerChunk(
                        content='\n\n'.join(current_chunk_content),
                        chunk_index=chunk_index,
                        total_chunks=0,
                        is_complete=False,
                        has_more=True
                    ))
                    chunk_index += 1
                
                if section_length > self.chunk_size:
                    sub_chunks = self._split_large_section(section_content)
                    for sub_chunk in sub_chunks:
                        chunks.append(AnswerChunk(
                            content=sub_chunk,
                            chunk_index=chunk_index,
                            total_chunks=0,
                            is_complete=False,
                            has_more=True
                        ))
                        chunk_index += 1
                    current_chunk_content = []
                    current_chunk_length = 0
                else:
                    current_chunk_content = [section_content]
                    current_chunk_length = section_length
        
        if current_chunk_content:
            chunks.append(AnswerChunk(
                content='\n\n'.join(current_chunk_content),
                chunk_index=chunk_index,
                total_chunks=0,
                is_complete=True,
                has_more=False
            ))
        
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
            if chunk.chunk_index == total - 1:
                chunk.is_complete = True
                chunk.has_more = False
        
        return chunks
    
    def _split_large_section(self, section_content: str) -> List[str]:
        """너무 큰 섹션을 강제로 분할"""
        chunks = []
        paragraphs = re.split(self.PARAGRAPH_BREAK, section_content)
        
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            if current_length + para_length <= self.chunk_size:
                current_chunk.append(para)
                current_length += para_length + 2
            else:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = para_length
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks

