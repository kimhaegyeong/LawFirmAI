# -*- coding: utf-8 -*-
"""
문서 타입 정규화 유틸리티

문서의 type 정보를 doc.type으로 통합하는 공통 함수 제공.
단일 소스 원칙을 적용하여 모든 위치의 type 정보를 doc.type으로 통합합니다.
"""

from typing import Dict, Any, Optional
from lawfirm_langgraph.core.workflow.constants.document_types import DocumentType


def normalize_document_type(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    문서의 type을 doc.type으로 통합 (단일 소스 원칙)
    
    모든 위치의 type 정보를 수집하여 doc.type으로 통합하고,
    다른 위치의 type 필드는 doc.type과 동기화합니다.
    
    우선순위:
    1. doc.type (unknown이 아닌 경우만)
    2. doc.source_type
    3. metadata.source_type
    4. metadata.type
    5. content.metadata.source_type (content가 dict인 경우)
    6. content.metadata.type (content가 dict인 경우)
    7. DocumentType.from_metadata 추론
    
    Args:
        doc: 문서 딕셔너리
    
    Returns:
        type이 통합된 문서 딕셔너리 (in-place 수정)
    """
    if not isinstance(doc, dict):
        return doc
    
    # 1단계: 모든 위치에서 type 수집 (우선순위 순)
    collected_type = None
    
    # 우선순위 1: doc.type (unknown이 아닌 경우만)
    doc_type = doc.get("type")
    if doc_type and doc_type.lower() != "unknown":
        collected_type = doc_type
    
    # 우선순위 2: doc.source_type
    if not collected_type:
        source_type = doc.get("source_type")
        if source_type and source_type.lower() != "unknown":
            collected_type = source_type
    
    # 우선순위 3: metadata.source_type
    if not collected_type:
        metadata = doc.get("metadata", {})
        if isinstance(metadata, dict):
            metadata_source_type = metadata.get("source_type")
            if metadata_source_type and metadata_source_type.lower() != "unknown":
                collected_type = metadata_source_type
    
    # 우선순위 4: metadata.type
    if not collected_type:
        metadata = doc.get("metadata", {})
        if isinstance(metadata, dict):
            metadata_type = metadata.get("type")
            if metadata_type and metadata_type.lower() != "unknown":
                collected_type = metadata_type
    
    # 우선순위 5: content.metadata.source_type (content가 dict인 경우)
    if not collected_type:
        content = doc.get("content")
        if isinstance(content, dict):
            content_metadata = content.get("metadata", {})
            if isinstance(content_metadata, dict):
                content_source_type = content_metadata.get("source_type")
                if content_source_type and content_source_type.lower() != "unknown":
                    collected_type = content_source_type
    
    # 우선순위 6: content.metadata.type (content가 dict인 경우)
    if not collected_type:
        content = doc.get("content")
        if isinstance(content, dict):
            content_metadata = content.get("metadata", {})
            if isinstance(content_metadata, dict):
                content_type = content_metadata.get("type")
                if content_type and content_type.lower() != "unknown":
                    collected_type = content_type
    
    # 우선순위 7: DocumentType.from_metadata 추론
    if not collected_type:
        try:
            doc_type_enum = DocumentType.from_metadata(doc)
            if doc_type_enum != DocumentType.UNKNOWN:
                collected_type = doc_type_enum.value
                # 추론 성공 로깅
                from lawfirm_langgraph.core.utils.logger import get_logger
                logger = get_logger(__name__)
                logger.debug(
                    f"[normalize_document_type] ✅ DocumentType.from_metadata 추론 성공: "
                    f"doc_type={collected_type}, "
                    f"doc_keys={list(doc.keys())[:15]}, "
                    f"metadata_keys={list(doc.get('metadata', {}).keys())[:15] if isinstance(doc.get('metadata'), dict) else []}"
                )
        except (ImportError, AttributeError) as e:
            from lawfirm_langgraph.core.utils.logger import get_logger
            logger = get_logger(__name__)
            logger.warning(f"[normalize_document_type] DocumentType.from_metadata 호출 실패: {e}")
    
    # 우선순위 8: content 필드 텍스트 분석 제거
    # 원래의 type 필드를 신뢰하도록 변경 (텍스트 분석은 오분류를 유발할 수 있음)
    # 판례 데이터에 법령 조문이 포함되어 있어서 잘못 분류되는 문제 방지
    
    # 2단계: doc.type에 통합 (단일 소스)
    if collected_type:
        doc["type"] = collected_type
        # source_type도 동기화 (레거시 호환)
        doc["source_type"] = collected_type
        
        # metadata에도 동기화
        if "metadata" not in doc:
            doc["metadata"] = {}
        if not isinstance(doc["metadata"], dict):
            doc["metadata"] = {}
        doc["metadata"]["type"] = collected_type
        doc["metadata"]["source_type"] = collected_type
        
        # content.metadata에도 동기화 (content가 dict인 경우)
        content = doc.get("content")
        if isinstance(content, dict):
            if "metadata" not in content:
                content["metadata"] = {}
            if not isinstance(content["metadata"], dict):
                content["metadata"] = {}
            content["metadata"]["type"] = collected_type
            content["metadata"]["source_type"] = collected_type
    else:
        # collected_type이 없으면 로깅 (디버깅용)
        from lawfirm_langgraph.core.utils.logger import get_logger
        logger = get_logger(__name__)
        metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
        # DocumentType.from_metadata가 확인하는 필드들 (현재 테이블 컬럼명 기준)
        type_hint_fields = {
            "statute_name": doc.get("statute_name") or metadata.get("statute_name"),
            "law_name": doc.get("law_name") or metadata.get("law_name"),
            "law_name_kr": doc.get("law_name_kr") or metadata.get("law_name_kr"),
            "article_no": doc.get("article_no") or metadata.get("article_no"),
            "statute_id": doc.get("statute_id") or metadata.get("statute_id"),
            "precedent_id": doc.get("precedent_id") or metadata.get("precedent_id"),
            "case_id": doc.get("case_id") or metadata.get("case_id"),
            "court": doc.get("court") or metadata.get("court"),
            "court_name": doc.get("court_name") or metadata.get("court_name"),
            "doc_id": doc.get("doc_id") or metadata.get("doc_id"),
            "case_number": doc.get("case_number") or metadata.get("case_number"),
            "casenames": doc.get("casenames") or metadata.get("casenames"),
            "case_name": doc.get("case_name") or metadata.get("case_name"),
        }
        logger.debug(
            f"[normalize_document_type] ⚠️ type을 찾을 수 없습니다. "
            f"doc.type={doc.get('type')}, "
            f"doc.source_type={doc.get('source_type')}, "
            f"metadata.type={metadata.get('type')}, "
            f"metadata.source_type={metadata.get('source_type')}, "
            f"type_hint_fields={type_hint_fields}, "
            f"doc_keys={list(doc.keys())[:20]}, "
            f"metadata_keys={list(metadata.keys())[:20] if isinstance(metadata, dict) else []}"
        )
    
    return doc


def normalize_documents_type(docs: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """
    문서 리스트의 type을 일괄 정규화
    
    Args:
        docs: 문서 딕셔너리 리스트
    
    Returns:
        type이 통합된 문서 딕셔너리 리스트
    """
    if not docs:
        return docs
    
    normalized_docs = []
    for doc in docs:
        normalized_doc = normalize_document_type(doc)
        normalized_docs.append(normalized_doc)
    
    return normalized_docs

