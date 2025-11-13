# -*- coding: utf-8 -*-
"""소스 추출 클래스"""

import logging
from typing import Any, Dict, List, Optional, Tuple


class SourceExtractor:
    """소스 및 legal_references 추출 담당"""
    
    STATUTE_FIELDS = ["statute_name", "law_name", "abbrv", "statute_abbrv", "law_abbrv", "name"]
    ARTICLE_FIELDS = ["article_no", "article_number", "article", "article_num"]
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def extract_statute_info(self, detail: Dict) -> Optional[Tuple[str, str]]:
        """sources_detail에서 statute_name과 article_no 추출"""
        if not isinstance(detail, dict):
            return None
        
        detail_metadata = detail.get("metadata", {}) if isinstance(detail.get("metadata"), dict) else {}
        
        # detail의 최상위 레벨 필드도 확인 (강화)
        merged_metadata = {**detail_metadata}
        for key in ["statute_name", "law_name", "article_no", "article_number", "clause_no", "item_no",
                   "law_id", "statute_id", "abbrv", "statute_abbrv", "law_abbrv", "name", "article", "article_num"]:
            if key in detail and detail[key] is not None and detail[key] != "":
                if not merged_metadata.get(key):
                    merged_metadata[key] = detail[key]
        
        statute_name = self._extract_field(detail, merged_metadata, self.STATUTE_FIELDS)
        
        # statute_name이 없으면 abbrv를 fallback으로 사용
        if not statute_name:
            statute_name = self._extract_field(detail, merged_metadata, ["abbrv", "statute_abbrv", "law_abbrv"])
        
        article_no = self._extract_field(detail, merged_metadata, self.ARTICLE_FIELDS)
        
        return (statute_name, article_no) if statute_name else None
    
    def _extract_field(
        self, 
        detail: Dict, 
        metadata: Dict, 
        field_names: List[str]
    ) -> Optional[str]:
        """여러 필드명을 시도하여 값 추출"""
        if isinstance(detail, dict):
            for field in field_names:
                value = detail.get(field)
                if value is not None and value != "":
                    value_str = str(value).strip()
                    if value_str:
                        return value_str
        
        if isinstance(metadata, dict):
            for field in field_names:
                value = metadata.get(field)
                if value is not None and value != "":
                    value_str = str(value).strip()
                    if value_str:
                        return value_str
        
        return None
    
    def format_legal_reference(
        self, 
        statute_name: str, 
        article_no: Optional[str] = None,
        clause_no: Optional[str] = None,
        item_no: Optional[str] = None
    ) -> str:
        """법령 참조 형식으로 변환"""
        parts = [statute_name]
        
        if article_no:
            article_no_str = str(article_no) if article_no else ""
            if article_no_str.startswith("제") and article_no_str.endswith("조"):
                parts.append(article_no_str)
            else:
                article_no_clean = article_no_str.strip()
                if article_no_clean:
                    parts.append(f"제{article_no_clean}조")
        
        if clause_no:
            parts.append(f"제{clause_no}항")
        
        if item_no:
            parts.append(f"제{item_no}호")
        
        return " ".join(parts)
    
    def extract_legal_references_from_sources_detail(
        self, 
        sources_detail: List[Dict[str, Any]]
    ) -> List[str]:
        """sources_detail에서 legal_references 추출"""
        legal_refs = []
        seen_legal_refs = set()
        
        self.logger.debug(f"[LEGAL_REFERENCES] Checking {len(sources_detail)} sources_detail for statute_article documents")
        
        for detail in sources_detail:
            if not isinstance(detail, dict):
                continue
            
            detail_type = detail.get("type") or detail.get("source_type")
            if detail_type != "statute_article":
                continue
            
            statute_info = self.extract_statute_info(detail)
            if not statute_info:
                continue
            
            statute_name, article_no = statute_info
            
            detail_metadata = detail.get("metadata", {}) if isinstance(detail.get("metadata"), dict) else {}
            
            # detail의 최상위 레벨 필드도 확인 (강화)
            merged_metadata = {**detail_metadata}
            for key in ["statute_name", "law_name", "article_no", "article_number", "clause_no", "item_no",
                       "law_id", "statute_id", "abbrv", "statute_abbrv", "law_abbrv", "name", "article", "article_num"]:
                if key in detail and detail[key] is not None and detail[key] != "":
                    if not merged_metadata.get(key):
                        merged_metadata[key] = detail[key]
            
            clause_no = self._extract_field(detail, merged_metadata, ["clause_no", "clause", "clause_number"])
            item_no = self._extract_field(detail, merged_metadata, ["item_no", "item", "item_number"])
            
            self.logger.debug(
                f"[LEGAL_REFERENCES] Found statute_article in sources_detail: "
                f"statute_name={statute_name}, article_no={article_no}, "
                f"detail keys={list(detail.keys())}, "
                f"metadata keys={list(detail_metadata.keys()) if isinstance(detail_metadata, dict) else 'N/A'}"
            )
            
            legal_ref = self.format_legal_reference(statute_name, article_no, clause_no, item_no)
            
            if legal_ref and legal_ref not in seen_legal_refs:
                legal_refs.append(legal_ref)
                seen_legal_refs.add(legal_ref)
        
        return legal_refs
    
    def extract_legal_references_from_docs(
        self, 
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[str]:
        """retrieved_docs에서 legal_references 추출"""
        legal_refs = []
        seen_legal_refs = set()
        
        self.logger.debug(f"[LEGAL_REFERENCES] No legal_references found in sources_detail, trying retrieved_docs directly")
        
        for doc in retrieved_docs:
            if not isinstance(doc, dict):
                continue
            
            source_type = (
                doc.get("type") or 
                doc.get("source_type") or 
                doc.get("metadata", {}).get("source_type", "") if isinstance(doc.get("metadata"), dict) else ""
            )
            
            if source_type != "statute_article":
                continue
            
            metadata_raw = doc.get("metadata", {})
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
            
            # doc의 최상위 레벨 필드도 확인 (강화)
            merged_metadata = {**metadata}
            for key in ["statute_name", "law_name", "article_no", "article_number", "clause_no", "item_no",
                       "law_id", "statute_id", "abbrv", "statute_abbrv", "law_abbrv", "name", "article", "article_num"]:
                if key in doc and doc[key] is not None and doc[key] != "":
                    if not merged_metadata.get(key):
                        merged_metadata[key] = doc[key]
            
            statute_name = self._extract_field(doc, merged_metadata, self.STATUTE_FIELDS)
            
            # statute_name이 없으면 abbrv를 fallback으로 사용
            if not statute_name:
                statute_name = self._extract_field(doc, merged_metadata, ["abbrv", "statute_abbrv", "law_abbrv"])
            
            article_no = self._extract_field(doc, merged_metadata, self.ARTICLE_FIELDS)
            clause_no = self._extract_field(doc, merged_metadata, ["clause_no", "clause", "clause_number"])
            item_no = self._extract_field(doc, merged_metadata, ["item_no", "item", "item_number"])
            
            if statute_name:
                legal_ref = self.format_legal_reference(statute_name, article_no, clause_no, item_no)
                
                if legal_ref and legal_ref not in seen_legal_refs:
                    legal_refs.append(legal_ref)
                    seen_legal_refs.add(legal_ref)
                    self.logger.debug(
                        f"[LEGAL_REFERENCES] Added legal reference: {legal_ref} "
                        f"(statute_name={statute_name}, article_no={article_no})"
                    )
            else:
                self.logger.debug(
                    f"[LEGAL_REFERENCES] Skipping statute_article doc: statute_name is missing. "
                    f"doc keys: {list(doc.keys())[:15]}, "
                    f"doc has statute_name={doc.get('statute_name')}, law_name={doc.get('law_name')}, abbrv={doc.get('abbrv')}, "
                    f"metadata has statute_name={merged_metadata.get('statute_name') if isinstance(merged_metadata, dict) else 'N/A'}, "
                    f"metadata keys: {list(merged_metadata.keys())[:15] if isinstance(merged_metadata, dict) else 'N/A'}"
                )
        
        return legal_refs

