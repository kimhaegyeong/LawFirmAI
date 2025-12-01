# -*- coding: utf-8 -*-
"""
문서 타입 정규화 유틸리티

문서의 type 정보를 doc.type으로 통합하는 공통 함수 제공.
단일 소스 원칙을 적용하여 모든 위치의 type 정보를 doc.type으로 통합합니다.
"""

from typing import Dict, Any


def normalize_document_type(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    문서의 type을 doc.type으로 통합 (단일 소스 원칙)
    
    우선순위:
    1. doc.type (unknown이 아닌 경우만)
    2. metadata.type
    3. DocumentType.from_metadata() 추론
    
    Args:
        doc: 문서 딕셔너리
    
    Returns:
        type이 통합된 문서 딕셔너리 (in-place 수정)
    """
    if not isinstance(doc, dict):
        return doc
    
    # 1단계: doc.type 확인 (unknown이 아닌 경우만)
    doc_type = doc.get("type")
    if doc_type and doc_type.lower() != "unknown":
        # 이미 유효한 type이 있으면 그대로 사용
        return doc
    
    # 2단계: metadata.type 확인
    metadata = doc.get("metadata", {})
    metadata_type = None
    if isinstance(metadata, dict):
        metadata_type = metadata.get("type")
        if metadata_type and metadata_type.lower() != "unknown":
            doc["type"] = metadata_type
            return doc
    
    # 3단계: DocumentType.from_metadata() 추론
    inferred_type = None
    inference_error = None
    try:
        from lawfirm_langgraph.core.workflow.constants.document_types import DocumentType
        inferred_type_enum = DocumentType.from_metadata(doc)
        if inferred_type_enum != DocumentType.UNKNOWN:
            inferred_type = inferred_type_enum.value
            doc["type"] = inferred_type
            # metadata에도 동기화 (일관성 유지)
            if "metadata" not in doc:
                doc["metadata"] = {}
            if not isinstance(doc["metadata"], dict):
                doc["metadata"] = {}
            doc["metadata"]["type"] = inferred_type
            return doc
    except (ImportError, AttributeError) as e:
        from lawfirm_langgraph.core.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.warning(f"[normalize_document_type] DocumentType.from_metadata 호출 실패: {e}")
        inference_error = str(e)
    
    # 4단계: 추론 실패 시 unknown으로 설정
    doc["type"] = "unknown"
    if "metadata" not in doc:
        doc["metadata"] = {}
    if not isinstance(doc["metadata"], dict):
        doc["metadata"] = {}
    doc["metadata"]["type"] = "unknown"
    
    # 디버깅 로깅 (실제 값 포함)
    from lawfirm_langgraph.core.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.debug(
        f"[normalize_document_type] ⚠️ type을 찾을 수 없습니다. "
        f"doc_type={repr(doc_type)}, "
        f"metadata_type={repr(metadata_type)}, "
        f"inferred_type={repr(inferred_type)}, "
        f"inference_error={repr(inference_error)}, "
        f"doc_keys={list(doc.keys())[:20]}, "
        f"metadata_keys={list(doc.get('metadata', {}).keys())[:20] if isinstance(doc.get('metadata'), dict) else []}"
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
