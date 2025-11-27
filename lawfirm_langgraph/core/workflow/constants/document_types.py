# -*- coding: utf-8 -*-
"""
문서 타입 정의 및 분류 시스템

메타데이터 필드 기준으로만 문서 타입을 분류합니다.
데이터베이스 테이블 기준은 사용하지 않습니다.
"""

from enum import Enum
from typing import Any, Dict, List, Optional


class DocumentType(str, Enum):
    """
    검색 문서 타입 정의
    
    분류 기준: 메타데이터 필드 기준만 사용
    - 문서의 metadata 필드에 있는 키를 기반으로 타입을 결정
    - type, source_type 필드가 있으면 우선 사용
    - 없으면 metadata 필드의 키를 기반으로 추론
    """
    
    # 법률 조문
    STATUTE_ARTICLE = "statute_article"
    """법률 조문 - statute_name, law_name, article_no 메타데이터 필드 존재"""
    
    # 판례 본문
    PRECEDENT_CONTENT = "precedent_content"
    """판례 본문 - case_id, court, casenames, doc_id, precedent_id 메타데이터 필드 존재"""
    
    # 알 수 없음
    UNKNOWN = "unknown"
    """알 수 없는 타입"""
    
    @classmethod
    def from_string(cls, value: str) -> "DocumentType":
        """문자열로부터 DocumentType 반환 (대소문자 무시)"""
        if not value:
            return cls.UNKNOWN
        
        value_lower = value.lower().strip()
        
        # 정확한 매칭
        for doc_type in cls:
            if doc_type.value == value_lower:
                return doc_type
        
        # 레거시 호환: case, case_paragraph는 precedent_content로 매핑
        if value_lower in ["case", "case_paragraph", "precedent", "precedent_content"]:
            return cls.PRECEDENT_CONTENT
        
        return cls.UNKNOWN
    
    @classmethod
    def all_types(cls) -> List[str]:
        """모든 타입 리스트 반환 (UNKNOWN 제외)"""
        return [dt.value for dt in cls if dt != cls.UNKNOWN]
    
    @classmethod
    def from_metadata(cls, doc: Dict[str, Any]) -> "DocumentType":
        """
        문서 딕셔너리로부터 DocumentType 추론 (메타데이터 필드 기준)
        
        우선순위:
        1. doc.get("type") 또는 doc.get("source_type")
        2. doc.get("metadata", {}).get("source_type")
        3. metadata 필드의 키를 기반으로 추론
        
        Args:
            doc: 문서 딕셔너리 (type, source_type, metadata 필드 포함 가능)
        
        Returns:
            DocumentType
        """
        if not isinstance(doc, dict):
            return cls.UNKNOWN
        
        # 1단계: type 또는 source_type 필드 직접 확인
        doc_type = (
            doc.get("type") or
            doc.get("source_type") or
            (doc.get("metadata", {}).get("source_type") if isinstance(doc.get("metadata"), dict) else None)
        )
        
        if doc_type:
            return cls.from_string(doc_type)
        
        # 2단계: metadata 필드의 키를 기반으로 추론
        metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
        
        # statute_article 판단: statute_name, law_name, article_no 중 하나라도 있으면
        if any(key in metadata for key in ["statute_name", "law_name", "article_no"]):
            return cls.STATUTE_ARTICLE
        
        # precedent_content 판단: case_id, court, casenames, doc_id, precedent_id 중 하나라도 있으면
        if any(key in metadata for key in ["case_id", "court", "casenames", "doc_id", "precedent_id"]):
            return cls.PRECEDENT_CONTENT
        
        return cls.UNKNOWN


class DocumentTypeConfig:
    """문서 타입별 설정"""
    
    # 타입별 최소 개수 보장 (검색 결과 다양성 확보)
    MIN_COUNTS: Dict[DocumentType, int] = {
        DocumentType.STATUTE_ARTICLE: 1,
        DocumentType.PRECEDENT_CONTENT: 2,
    }
    
    # 타입별 우선순위 부스팅 (점수에 추가)
    PRIORITY_BOOST: Dict[DocumentType, float] = {
        DocumentType.STATUTE_ARTICLE: 1.0,  # 가장 높은 우선순위
        DocumentType.PRECEDENT_CONTENT: 0.9,
    }
    
    # 타입별 메타데이터 키 매핑 (타입 추론용)
    METADATA_KEYS: Dict[DocumentType, Dict[str, List[str]]] = {
        DocumentType.STATUTE_ARTICLE: {
            "required": ["statute_name", "law_name", "article_no"],
            "optional": ["clause_no", "item_no", "statute_id"]
        },
        DocumentType.PRECEDENT_CONTENT: {
            "required": ["doc_id"],
            "optional": ["case_id", "court", "casenames", "precedent_id"]
        },
    }
    
    # 레거시 타입 매핑 (하위 호환성)
    LEGACY_TYPE_MAPPING: Dict[str, DocumentType] = {
        "case": DocumentType.PRECEDENT_CONTENT,
        "case_paragraph": DocumentType.PRECEDENT_CONTENT,
        "precedent": DocumentType.PRECEDENT_CONTENT,
        "statute": DocumentType.STATUTE_ARTICLE,
    }
    
    @classmethod
    def get_min_count(cls, doc_type: DocumentType) -> int:
        """타입별 최소 개수 반환"""
        return cls.MIN_COUNTS.get(doc_type, 0)
    
    @classmethod
    def get_priority_boost(cls, doc_type: DocumentType) -> float:
        """타입별 우선순위 부스팅 값 반환"""
        return cls.PRIORITY_BOOST.get(doc_type, 0.5)
    
    @classmethod
    def get_required_metadata_keys(cls, doc_type: DocumentType) -> List[str]:
        """타입별 필수 메타데이터 키 반환"""
        return cls.METADATA_KEYS.get(doc_type, {}).get("required", [])
    
    @classmethod
    def get_optional_metadata_keys(cls, doc_type: DocumentType) -> List[str]:
        """타입별 선택적 메타데이터 키 반환"""
        return cls.METADATA_KEYS.get(doc_type, {}).get("optional", [])
    
    @classmethod
    def is_valid_metadata(cls, doc_type: DocumentType, metadata: Dict[str, Any]) -> bool:
        """메타데이터가 해당 타입의 필수 필드를 포함하는지 확인"""
        required_keys = cls.get_required_metadata_keys(doc_type)
        if not required_keys:
            return True  # 필수 필드가 없으면 항상 유효
        
        return any(key in metadata for key in required_keys)

