# -*- coding: utf-8 -*-
"""
통일된 출처 포맷터
모든 출처 타입에 대해 일관된 포맷팅 규칙 적용
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


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
        self.logger = logging.getLogger(__name__)
    
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
        """판례 포맷팅"""
        court = metadata.get("court", "")
        doc_id = metadata.get("doc_id", "")
        casenames = metadata.get("casenames", "")
        announce_date = metadata.get("announce_date", "")
        
        parts = []
        if court:
            parts.append(court)
        if casenames:
            parts.append(casenames)
        if doc_id:
            parts.append(f"({doc_id})")
        if announce_date:
            parts.append(f"[{announce_date}]")
        
        name = " ".join(parts) if parts else "판례"
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
        """결정례 포맷팅"""
        org = metadata.get("org", "")
        doc_id = metadata.get("doc_id", "")
        decision_date = metadata.get("decision_date", "")
        
        parts = []
        if org:
            parts.append(org)
        if doc_id:
            parts.append(f"({doc_id})")
        if decision_date:
            parts.append(f"[{decision_date}]")
        
        name = " ".join(parts) if parts else "결정례"
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
        """해석례 포맷팅"""
        org = metadata.get("org", "")
        title = metadata.get("title", "")
        doc_id = metadata.get("doc_id", "")
        response_date = metadata.get("response_date", "")
        
        parts = []
        if org:
            parts.append(org)
        if title:
            parts.append(title)
        if doc_id:
            parts.append(f"({doc_id})")
        if response_date:
            parts.append(f"[{response_date}]")
        
        name = " ".join(parts) if parts else "해석례"
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
    
    def _generate_statute_url(self, statute_name: str, article_no: str, metadata: Dict[str, Any]) -> str:
        """법령 조문 URL 생성"""
        if not statute_name or not article_no:
            return ""
        
        base_url = "https://www.law.go.kr"
        effective_date = metadata.get("effective_date", "")
        proclamation_number = metadata.get("proclamation_number", "")
        
        if effective_date:
            effective_date = effective_date.replace("-", "")
            return f"{base_url}/LSW/lsInfoP.do?efYd={effective_date}&lsiSeq={proclamation_number}"
        elif proclamation_number:
            return f"{base_url}/LSW/lsInfoP.do?lsiSeq={proclamation_number}"
        else:
            return f"{base_url}/LSW/lsSc.do?lawNm={statute_name}&articleNo={article_no.replace('제', '').replace('조', '')}"
    
    def _generate_case_url(self, doc_id: str, metadata: Dict[str, Any]) -> str:
        """판례 URL 생성"""
        if not doc_id:
            return ""
        
        detail_url = metadata.get("detail_url", "")
        if detail_url:
            return detail_url
        
        base_url = "https://glaw.scourt.go.kr"
        return f"{base_url}/wsjo/panre/sjo100.do?contId={doc_id}"
    
    def _generate_decision_url(self, doc_id: str, metadata: Dict[str, Any]) -> str:
        """결정례 URL 생성"""
        if not doc_id:
            return ""
        
        base_url = "https://www.law.go.kr"
        return f"{base_url}/LSW/lsInfoP.do?lsiSeq={doc_id}"
    
    def _generate_interpretation_url(self, doc_id: str, metadata: Dict[str, Any]) -> str:
        """해석례 URL 생성"""
        if not doc_id:
            return ""
        
        base_url = "https://www.law.go.kr"
        return f"{base_url}/LSW/lsInfoP.do?lsiSeq={doc_id}"
    
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

