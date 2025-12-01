# -*- coding: utf-8 -*-
"""
통일된 출처 포맷터
모든 출처 타입에 대해 일관된 포맷팅 규칙 적용
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import re

logger = get_logger(__name__)


@dataclass
class SourceInfo:
    """출처 정보 데이터 클래스"""
    name: str
    type: str
    url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None


class UnifiedSourceFormatter:
    """통일된 출처 포맷터"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def format_source(self, source_type: str, metadata: Dict[str, Any]) -> SourceInfo:
        """
        통일된 형식으로 출처 포맷팅
        
        Args:
            source_type: 출처 타입 (statute_article, case_paragraph 등)
            metadata: 출처 메타데이터
            
        Returns:
            SourceInfo 객체
        """
        if source_type == "statute_article":
            return self._format_statute_article(metadata)
        elif source_type == "case_paragraph":
            return self._format_case_paragraph(metadata)
        elif source_type == "decision_paragraph":
            return self._format_decision_paragraph(metadata)
        elif source_type == "interpretation_paragraph":
            return self._format_interpretation_paragraph(metadata)
        elif source_type == "regulation_paragraph":
            return self._format_regulation_paragraph(metadata)
        else:
            return SourceInfo(name="Unknown", type=source_type)
    
    def _format_statute_article(self, metadata: Dict[str, Any]) -> SourceInfo:
        """법령 조문 포맷팅"""
        statute_name = metadata.get("statute_name") or "법령"
        article_no = metadata.get("article_no") or ""
        clause_no = metadata.get("clause_no") or ""
        item_no = metadata.get("item_no") or ""
        
        parts = [statute_name]
        if article_no:
            parts.append(article_no)
        if clause_no:
            parts.append(f"제{clause_no}항")
        if item_no:
            parts.append(f"제{item_no}호")
        
        name = " ".join(parts)
        url = self._generate_statute_url(statute_name, article_no, metadata)
        
        return SourceInfo(
            name=name,
            type="statute_article",
            url=url,
            metadata={
                "statute_name": statute_name,
                "article_no": article_no,
                "clause_number": article_no,
                "clause_no": clause_no,
                "item_no": item_no,
                "abbrv": metadata.get("abbrv"),
                "category": metadata.get("category"),
                "statute_type": metadata.get("statute_type")
            }
        )
    
    def _format_case_paragraph(self, metadata: Dict[str, Any]) -> SourceInfo:
        """판례 포맷팅 (casenames 우선, doc_id fallback)"""
        court = metadata.get("court", "")
        doc_id = metadata.get("doc_id", "")
        casenames = metadata.get("casenames", "")
        announce_date = metadata.get("announce_date", "")
        
        # name 우선순위: casenames > doc_id > "판례"
        if casenames and casenames.strip():
            name = casenames.strip()
        elif doc_id and doc_id.strip():
            name = doc_id.strip()
        else:
            name = "판례"
        
        url = self._generate_case_url(doc_id, metadata)
        
        return SourceInfo(
            name=name,
            type="case_paragraph",
            url=url,
            metadata={
                "court": court,
                "doc_id": doc_id,
                "casenames": casenames,
                "announce_date": announce_date,
                "case_type": metadata.get("case_type")
            }
        )
    
    def _format_decision_paragraph(self, metadata: Dict[str, Any]) -> SourceInfo:
        """결정례 포맷팅 (decision_number만 표시)"""
        org = metadata.get("org", "")
        doc_id = metadata.get("doc_id", "")
        decision_date = metadata.get("decision_date", "")
        
        name = doc_id if doc_id else "결정례"
        url = self._generate_decision_url(doc_id, metadata)
        
        return SourceInfo(
            name=name,
            type="decision_paragraph",
            url=url,
            metadata={
                "org": org,
                "doc_id": doc_id,
                "decision_date": decision_date,
                "result": metadata.get("result")
            }
        )
    
    def _format_interpretation_paragraph(self, metadata: Dict[str, Any]) -> SourceInfo:
        """해석례 포맷팅 (interpretation_number만 표시)"""
        org = metadata.get("org", "")
        title = metadata.get("title", "")
        doc_id = metadata.get("doc_id", "")
        response_date = metadata.get("response_date", "")
        
        name = doc_id if doc_id else "해석례"
        url = self._generate_interpretation_url(doc_id, metadata)
        
        return SourceInfo(
            name=name,
            type="interpretation_paragraph",
            url=url,
            metadata={
                "org": org,
                "title": title,
                "doc_id": doc_id,
                "response_date": response_date
            }
        )
    
    def _format_regulation_paragraph(self, metadata: Dict[str, Any]) -> SourceInfo:
        """기타 참고자료 포맷팅"""
        title = metadata.get("title", "") or metadata.get("name", "")
        doc_id = metadata.get("doc_id", "") or metadata.get("id", "")
        content = metadata.get("content", "") or metadata.get("text", "")
        
        if title and title.strip():
            name = title.strip()
        elif doc_id and doc_id.strip():
            name = doc_id.strip()
        elif content and len(content) > 0:
            name = content[:50] + "..." if len(content) > 50 else content
        else:
            name = "기타 참고자료"
        
        url = metadata.get("url", "") or metadata.get("detail_url", "")
        
        return SourceInfo(
            name=name,
            type="regulation_paragraph",
            url=url,
            metadata={
                "title": title,
                "doc_id": doc_id,
                "content": content,
                **{k: v for k, v in metadata.items() if k not in ["title", "doc_id", "id", "content", "text", "url", "detail_url"]}
            }
        )
    
    def _format_article_no(self, article_no: str) -> str:
        """조문번호를 6자리 형식으로 변환 (예: 제2조 -> 000200, 제10조의2 -> 001002)"""
        if not article_no:
            return ""
        
        numbers = re.findall(r'\d+', article_no)
        if not numbers:
            return ""
        
        main_no = int(numbers[0])
        sub_no = int(numbers[1]) if len(numbers) > 1 else 0
        
        return f"{main_no:04d}{sub_no:02d}"
    
    def _generate_statute_url(self, statute_name: str, article_no: str, metadata: Dict[str, Any]) -> str:
        """법령 조문 URL 생성 (Open Law API 형식)"""
        base_url = "http://www.law.go.kr/DRF/lawService.do"
        
        law_id = metadata.get("law_id") or metadata.get("법령ID") or metadata.get("ID")
        if law_id:
            url = f"{base_url}?target=eflaw&ID={law_id}&type=HTML"
            if article_no:
                jo_no = self._format_article_no(article_no)
                if jo_no:
                    url += f"&JO={jo_no}"
            return url
        
        mst = metadata.get("mst") or metadata.get("MST") or metadata.get("lsi_seq")
        effective_date = metadata.get("effective_date") or metadata.get("efYd") or metadata.get("시행일자")
        
        if mst and effective_date:
            ef_yd = str(effective_date).replace("-", "")
            url = f"{base_url}?target=eflaw&MST={mst}&efYd={ef_yd}&type=HTML"
            if article_no:
                jo_no = self._format_article_no(article_no)
                if jo_no:
                    url += f"&JO={jo_no}"
            return url
        
        if not statute_name or not article_no:
            return ""
        
        proclamation_number = metadata.get("proclamation_number", "")
        effective_date = metadata.get("effective_date", "")
        
        if effective_date:
            effective_date = effective_date.replace("-", "")
            return f"https://www.law.go.kr/LSW/lsInfoP.do?efYd={effective_date}&lsiSeq={proclamation_number}"
        elif proclamation_number:
            return f"https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq={proclamation_number}"
        else:
            return f"https://www.law.go.kr/LSW/lsSc.do?lawNm={statute_name}&articleNo={article_no.replace('제', '').replace('조', '')}"
    
    def _generate_case_url(self, doc_id: str, metadata: Dict[str, Any]) -> str:
        """판례 URL 생성 (Open Law API 형식)"""
        detail_url = metadata.get("detail_url", "")
        if detail_url:
            return detail_url
        
        precedent_serial_number = (
            metadata.get("precedent_serial_number") or 
            metadata.get("판례일련번호") or 
            metadata.get("판례정보일련번호") or
            doc_id
        )
        
        if not precedent_serial_number:
            return ""
        
        base_url = "http://www.law.go.kr/DRF/lawService.do"
        return f"{base_url}?target=prec&ID={precedent_serial_number}&type=HTML"
    
    def _generate_decision_url(self, doc_id: str, metadata: Dict[str, Any]) -> str:
        """헌재결정례 URL 생성 (Open Law API 형식)"""
        decision_serial_number = (
            metadata.get("decision_serial_number") or 
            metadata.get("헌재결정례일련번호") or 
            metadata.get("결정ID") or
            doc_id
        )
        
        if not decision_serial_number:
            return ""
        
        base_url = "http://www.law.go.kr/DRF/lawService.do"
        return f"{base_url}?target=detc&ID={decision_serial_number}&type=HTML"
    
    def _generate_interpretation_url(self, doc_id: str, metadata: Dict[str, Any]) -> str:
        """법령해석례 URL 생성 (Open Law API 형식)"""
        interpretation_serial_number = (
            metadata.get("interpretation_serial_number") or 
            metadata.get("법령해석례일련번호") or 
            metadata.get("해석ID") or 
            metadata.get("expcId") or
            doc_id
        )
        
        if not interpretation_serial_number:
            return ""
        
        base_url = "http://www.law.go.kr/DRF/lawService.do"
        return f"{base_url}?target=expc&ID={interpretation_serial_number}&type=HTML"
    
    def format_sources_list(self, sources: List[Dict[str, Any]]) -> List[SourceInfo]:
        """출처 리스트를 통일된 형식으로 포맷팅"""
        formatted_sources = []
        for source in sources:
            if not isinstance(source, dict):
                continue
            
            source_type = source.get("type") or source.get("source_type") or ""
            metadata = source.get("metadata", {})
            
            if not source_type:
                continue
            
            formatted = self.format_source(source_type, metadata)
            formatted_sources.append(formatted)
        
        return formatted_sources

