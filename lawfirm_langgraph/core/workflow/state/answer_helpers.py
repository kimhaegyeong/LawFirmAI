"""
답변 상태 관리 헬퍼 함수

StructuredAnswer와 AnswerState 간의 변환 및 관리
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from .answer_models import (
    StructuredAnswer,
    DocumentUsageInfo,
    CoverageMetrics,
)

logger = logging.getLogger(__name__)


def create_structured_answer_from_state(
    answer_text: str,
    retrieved_docs: Optional[List[Dict[str, Any]]] = None,
    validation_result: Optional[Dict[str, Any]] = None,
    sources: Optional[List[str]] = None,
    structure_confidence: float = 0.0,
    query_type: Optional[str] = None
) -> StructuredAnswer:
    """
    답변 생성 결과로부터 StructuredAnswer 생성
    
    Args:
        answer_text: 답변 본문
        retrieved_docs: 검색된 문서 목록
        validation_result: 검증 결과 (coverage 정보 포함)
        sources: 참고 출처 목록
        structure_confidence: 구조 신뢰도
        query_type: 질문 유형
    
    Returns:
        StructuredAnswer 객체
    """
    # 문서 사용 정보 생성
    document_usage: List[DocumentUsageInfo] = []
    
    if retrieved_docs:
        # 답변에서 인용 추출
        import re
        citation_pattern = r'\[문서\s*(\d+)\]'
        citations_in_answer = re.findall(citation_pattern, answer_text)
        citation_counts = {}
        for cit in citations_in_answer:
            doc_num = int(cit)
            citation_counts[doc_num] = citation_counts.get(doc_num, 0) + 1
        
        # 인용 위치 찾기
        citation_positions: Dict[int, List[int]] = {}
        for match in re.finditer(citation_pattern, answer_text):
            doc_num = int(match.group(1))
            if doc_num not in citation_positions:
                citation_positions[doc_num] = []
            citation_positions[doc_num].append(match.start())
        
        # 각 문서에 대한 사용 정보 생성
        for idx, doc in enumerate(retrieved_docs, start=1):
            doc_num = idx
            used_in_answer = doc_num in citation_counts
            
            # 문서 정보 추출
            source = doc.get("source", doc.get("name", f"문서 {doc_num}"))
            source_type = doc.get("type", doc.get("source_type", "unknown"))
            doc_id = doc.get("doc_id") or doc.get("id") or doc.get("chunk_id")
            
            # 관련성 점수
            relevance_score = doc.get("relevance_score") or doc.get("similarity") or doc.get("score")
            semantic_similarity = doc.get("semantic_similarity") or doc.get("cross_encoder_score")
            
            document_usage.append(
                DocumentUsageInfo(
                    document_number=doc_num,
                    document_id=str(doc_id) if doc_id else None,
                    source=str(source),
                    source_type=str(source_type),
                    used_in_answer=used_in_answer,
                    citation_count=citation_counts.get(doc_num, 0),
                    citation_positions=citation_positions.get(doc_num, []),
                    relevance_score=float(relevance_score) if relevance_score is not None else None,
                    semantic_similarity=float(semantic_similarity) if semantic_similarity is not None else None,
                    key_content=doc.get("content", doc.get("text", ""))[:200] if doc.get("content") or doc.get("text") else None
                )
            )
    
    # 커버리지 메트릭 생성
    coverage = CoverageMetrics()
    
    if validation_result:
        # 키워드 커버리지
        coverage.keyword_coverage = validation_result.get("keyword_coverage", 0.0)
        coverage.keyword_total = validation_result.get("keyword_total", 0)
        coverage.keyword_matched = validation_result.get("keyword_matched", 0)
        
        # 인용 커버리지
        coverage.citation_coverage = validation_result.get("citation_coverage", 0.0)
        coverage.citation_count = validation_result.get("citation_count", 0)
        coverage.citation_expected = validation_result.get("citations_expected", 0)
        
        # 문서 활용도
        coverage.document_usage_rate = validation_result.get("document_usage_rate", 0.0)
        coverage.documents_used = validation_result.get("used_doc_count", 0)
        coverage.documents_total = validation_result.get("total_doc_count", 0)
        coverage.documents_min_required = validation_result.get("min_required_citations", 0)
        
        # 법률 참조 커버리지
        coverage.legal_reference_coverage = validation_result.get("legal_reference_coverage")
        coverage.legal_references_found = validation_result.get("legal_references_found", 0)
        
        # 전체 커버리지
        coverage.overall_coverage = validation_result.get("coverage_score", 0.0)
        
        # 커버리지 세부 정보
        coverage.coverage_breakdown = {
            "keyword": coverage.keyword_coverage,
            "citation": coverage.citation_coverage,
            "document_usage": coverage.document_usage_rate,
        }
        if coverage.legal_reference_coverage is not None:
            coverage.coverage_breakdown["legal_reference"] = coverage.legal_reference_coverage
    
    return StructuredAnswer(
        answer_text=answer_text,
        document_usage=document_usage,
        coverage=coverage,
        sources=sources or [],
        structure_confidence=structure_confidence,
        query_type=query_type
    )


def parse_answer_with_metadata(answer_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    답변 텍스트에서 메타데이터를 파싱하여 분리
    
    Args:
        answer_text: LLM이 생성한 답변 (텍스트 + 메타데이터 포함 가능)
    
    Returns:
        (답변 본문, 메타데이터 dict 또는 None)
    """
    if not answer_text:
        return answer_text, None
    
    # <metadata> 태그로 감싸진 JSON 추출
    metadata_pattern = r'<metadata>\s*(\{.*?\})\s*</metadata>'
    match = re.search(metadata_pattern, answer_text, re.DOTALL | re.IGNORECASE)
    
    if match:
        try:
            metadata_json = match.group(1)
            # JSON 파싱
            metadata = json.loads(metadata_json)
            
            # 메타데이터 부분 제거하여 순수 답변 본문만 추출
            answer_body = answer_text[:match.start()].rstrip()
            # "---" 구분선 제거
            answer_body = re.sub(r'\n*---\s*\n*$', '', answer_body, flags=re.MULTILINE)
            
            logger.debug(f"✅ [METADATA PARSE] Successfully parsed metadata from answer")
            return answer_body.strip(), metadata
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ [METADATA PARSE] Failed to parse JSON metadata: {e}")
            # JSON 파싱 실패 시 메타데이터 부분만 제거
            answer_body = answer_text[:match.start()].rstrip()
            answer_body = re.sub(r'\n*---\s*\n*$', '', answer_body, flags=re.MULTILINE)
            return answer_body.strip(), None
        except Exception as e:
            logger.warning(f"⚠️ [METADATA PARSE] Unexpected error parsing metadata: {e}")
            return answer_text, None
    
    # 메타데이터가 없는 경우 원본 반환
    return answer_text, None


def create_structured_answer_from_llm_response(
    answer_text: str,
    retrieved_docs: Optional[List[Dict[str, Any]]] = None,
    validation_result: Optional[Dict[str, Any]] = None,
    sources: Optional[List[str]] = None,
    structure_confidence: float = 0.0,
    query_type: Optional[str] = None
) -> StructuredAnswer:
    """
    LLM 응답에서 StructuredAnswer 생성 (메타데이터 파싱 포함)
    
    Args:
        answer_text: LLM이 생성한 답변 (메타데이터 포함 가능)
        retrieved_docs: 검색된 문서 목록
        validation_result: 검증 결과 (coverage 정보 포함)
        sources: 참고 출처 목록
        structure_confidence: 구조 신뢰도
        query_type: 질문 유형
    
    Returns:
        StructuredAnswer 객체
    """
    # 메타데이터 파싱
    answer_body, metadata = parse_answer_with_metadata(answer_text)
    
    # 메타데이터가 있으면 우선 사용, 없으면 기존 방식 사용
    if metadata:
        try:
            # 메타데이터에서 document_usage 추출
            document_usage: List[DocumentUsageInfo] = []
            if metadata.get("document_usage"):
                for doc_dict in metadata["document_usage"]:
                    try:
                        document_usage.append(DocumentUsageInfo(**doc_dict))
                    except Exception as e:
                        logger.warning(f"⚠️ [METADATA PARSE] Failed to parse document_usage item: {e}")
                        continue
            
            # 메타데이터에서 coverage 추출
            coverage = CoverageMetrics()
            if metadata.get("coverage"):
                try:
                    coverage = CoverageMetrics(**metadata["coverage"])
                except Exception as e:
                    logger.warning(f"⚠️ [METADATA PARSE] Failed to parse coverage: {e}")
                    # 검증 결과로 대체
                    coverage = CoverageMetrics()
                    if validation_result:
                        coverage.overall_coverage = validation_result.get("coverage_score", 0.0)
            
            # 메타데이터가 불완전한 경우 검증 결과로 보완
            if not document_usage and retrieved_docs:
                logger.debug("⚠️ [METADATA PARSE] document_usage not found in metadata, using fallback")
                return create_structured_answer_from_state(
                    answer_text=answer_body,
                    retrieved_docs=retrieved_docs,
                    validation_result=validation_result,
                    sources=sources,
                    structure_confidence=structure_confidence,
                    query_type=query_type
                )
            
            logger.info(f"✅ [METADATA PARSE] Created StructuredAnswer from LLM metadata: {len(document_usage)} documents")
            return StructuredAnswer(
                answer_text=answer_body,
                document_usage=document_usage,
                coverage=coverage,
                sources=sources or [],
                structure_confidence=structure_confidence,
                query_type=query_type
            )
        except Exception as e:
            logger.warning(f"⚠️ [METADATA PARSE] Failed to create StructuredAnswer from metadata: {e}, using fallback")
            # 메타데이터 파싱 실패 시 기존 방식 사용
            return create_structured_answer_from_state(
                answer_text=answer_body,
                retrieved_docs=retrieved_docs,
                validation_result=validation_result,
                sources=sources,
                structure_confidence=structure_confidence,
                query_type=query_type
            )
    else:
        # 메타데이터가 없는 경우 기존 방식 사용
        return create_structured_answer_from_state(
            answer_text=answer_body,
            retrieved_docs=retrieved_docs,
            validation_result=validation_result,
            sources=sources,
            structure_confidence=structure_confidence,
            query_type=query_type
        )


def update_answer_state_with_structured(
    answer_state: Dict[str, Any],
    structured_answer: StructuredAnswer
) -> Dict[str, Any]:
    """
    AnswerState에 StructuredAnswer 정보 업데이트
    
    Args:
        answer_state: 기존 AnswerState dict
        structured_answer: StructuredAnswer 객체
    
    Returns:
        업데이트된 AnswerState dict
    """
    answer_state["answer"] = structured_answer.answer_text
    answer_state["sources"] = structured_answer.sources
    answer_state["structure_confidence"] = structured_answer.structure_confidence
    
    # 구조화된 정보 추가
    if structured_answer.document_usage:
        answer_state["document_usage"] = [doc.model_dump() for doc in structured_answer.document_usage]
    if structured_answer.coverage:
        answer_state["coverage"] = structured_answer.coverage.model_dump()
    
    return answer_state

