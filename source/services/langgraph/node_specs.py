# -*- coding: utf-8 -*-
"""
LangGraph 노드별 Input/Output 사양 정의
source/services/langgraph용 - Flat State 구조에 맞춤

각 노드가 사용하는 입력 데이터와 출력 데이터를 명확히 정의하여
State Reduction을 통해 메모리 최적화
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set


@dataclass
class FlatNodeIOSpec:
    """노드별 Input/Output 사양 (Flat State용)"""
    node_name: str
    description: str
    required_fields: Set[str]  # 필수 입력 필드
    optional_fields: Set[str]  # 선택적 입력 필드
    output_fields: Set[str]  # 출력 필드
    always_include: Set[str]  # 항상 포함할 필드 (예: session_id, processing_steps)

    def get_required_fields(self) -> Set[str]:
        """필요한 모든 필드 (필수 + 항상 포함)"""
        return self.required_fields | self.always_include


# ============================================
# 노드별 Input/Output 사양 정의 (Flat State)
# ============================================

FLAT_NODE_SPECS: Dict[str, FlatNodeIOSpec] = {
    "classify_query": FlatNodeIOSpec(
        node_name="classify_query",
        description="질문 유형 분류 및 법률 분야 판단",
        required_fields={"query"},
        optional_fields={"conversation_history", "legal_field"},
        output_fields={"query_type", "confidence", "legal_field", "legal_domain"},
        always_include={"session_id", "processing_steps", "errors", "metadata"}
    ),

    "assess_urgency": FlatNodeIOSpec(
        node_name="assess_urgency",
        description="질문의 긴급도 평가",
        required_fields={"query"},
        optional_fields={"query_type", "legal_field", "confidence"},
        output_fields={"urgency_level", "urgency_reasoning", "emergency_type"},
        always_include={"session_id", "processing_steps", "errors", "metadata"}
    ),

    "resolve_multi_turn": FlatNodeIOSpec(
        node_name="resolve_multi_turn",
        description="멀티턴 대화 처리",
        required_fields={"query"},
        optional_fields={"conversation_history", "conversation_context", "query_type"},
        output_fields={"is_multi_turn", "multi_turn_confidence", "resolved_query", "original_query",
                       "conversation_history", "conversation_context"},
        always_include={"session_id", "processing_steps", "errors", "metadata"}
    ),

    "route_expert": FlatNodeIOSpec(
        node_name="route_expert",
        description="전문가 라우팅 결정",
        required_fields={"query", "query_type"},
        optional_fields={"legal_field", "legal_domain", "urgency_level", "complexity_level"},
        output_fields={"complexity_level", "requires_expert", "expert_subgraph"},
        always_include={"session_id", "processing_steps", "errors", "metadata"}
    ),

    "analyze_document": FlatNodeIOSpec(
        node_name="analyze_document",
        description="업로드된 문서 분석",
        required_fields={"query"},
        optional_fields={"document_file", "query_type"},
        output_fields={"document_type", "document_analysis", "key_clauses", "potential_issues"},
        always_include={"session_id", "processing_steps", "errors", "metadata"}
    ),

    "expand_keywords_ai": FlatNodeIOSpec(
        node_name="expand_keywords_ai",
        description="AI 기반 키워드 확장",
        required_fields={"query", "query_type"},
        optional_fields={"legal_field", "extracted_keywords", "legal_domain"},
        output_fields={"search_query", "extracted_keywords", "ai_keyword_expansion"},
        always_include={"session_id", "processing_steps", "errors", "metadata"}
    ),

    "retrieve_documents": FlatNodeIOSpec(
        node_name="retrieve_documents",
        description="문서 검색 (하이브리드: 벡터 + 키워드)",
        required_fields={"query", "search_query"},
        optional_fields={"query_type", "extracted_keywords", "legal_field", "legal_domain", "ai_keyword_expansion"},
        output_fields={"retrieved_docs"},
        always_include={"session_id", "processing_steps", "errors", "metadata"}
    ),

    "process_legal_terms": FlatNodeIOSpec(
        node_name="process_legal_terms",
        description="법률 용어 처리 및 통합",
        required_fields={"query", "retrieved_docs"},
        optional_fields={"legal_field", "query_type", "search_query"},
        output_fields={"legal_references", "legal_citations", "analysis"},
        always_include={"session_id", "processing_steps", "errors", "metadata"}
    ),

    "generate_answer_enhanced": FlatNodeIOSpec(
        node_name="generate_answer_enhanced",
        description="향상된 답변 생성 (LLM 활용)",
        required_fields={"query", "retrieved_docs"},
        optional_fields={"query_type", "legal_field", "legal_domain", "analysis", "legal_references",
                        "legal_citations", "document_analysis"},
        output_fields={"answer", "enhanced_answer", "confidence", "sources", "legal_references", "legal_citations"},
        always_include={"session_id", "processing_steps", "errors", "metadata"}
    ),

    "validate_answer_quality": FlatNodeIOSpec(
        node_name="validate_answer_quality",
        description="답변 품질 및 법령 검증",
        required_fields={"answer", "query"},
        optional_fields={"retrieved_docs", "sources", "legal_references", "query_type"},
        output_fields={"quality_check_passed", "quality_score", "legal_validity_check",
                      "legal_basis_validation", "retry_count", "needs_enhancement"},
        always_include={"session_id", "processing_steps", "errors", "metadata"}
    ),

    "enhance_answer_structure": FlatNodeIOSpec(
        node_name="enhance_answer_structure",
        description="답변 구조화 및 법적 근거 강화",
        required_fields={"answer", "query_type"},
        optional_fields={"legal_references", "legal_citations", "retrieved_docs", "legal_field"},
        output_fields={"answer", "enhanced_answer", "structure_confidence"},
        always_include={"session_id", "processing_steps", "errors", "metadata"}
    ),

    "apply_visual_formatting": FlatNodeIOSpec(
        node_name="apply_visual_formatting",
        description="시각적 포맷팅 적용",
        required_fields={"answer"},
        optional_fields={"query_type", "legal_references", "structure_confidence"},
        output_fields={"answer"},
        always_include={"session_id", "processing_steps", "errors", "metadata"}
    ),

    "prepare_final_response": FlatNodeIOSpec(
        node_name="prepare_final_response",
        description="최종 응답 준비",
        required_fields={"answer"},
        optional_fields={"sources", "legal_references", "confidence", "legal_validity_check",
                        "query_type", "legal_field"},
        output_fields={"answer", "sources", "confidence", "processing_time", "tokens_used"},
        always_include={"session_id", "processing_steps", "errors", "metadata"}
    )
}


# ============================================
# 헬퍼 함수
# ============================================

def get_node_spec(node_name: str) -> Optional[FlatNodeIOSpec]:
    """노드별 사양 조회"""
    return FLAT_NODE_SPECS.get(node_name)


def get_required_fields(node_name: str) -> Set[str]:
    """노드에 필요한 필드 반환"""
    spec = get_node_spec(node_name)
    if spec:
        return spec.get_required_fields()
    return set()


def get_output_fields(node_name: str) -> Set[str]:
    """노드가 출력하는 필드 반환"""
    spec = get_node_spec(node_name)
    if spec:
        return spec.output_fields
    return set()


def validate_node_input(node_name: str, state: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    노드 Input 유효성 검증

    Args:
        node_name: 노드 이름
        state: State 객체

    Returns:
        (is_valid, error_message) 튜플
    """
    spec = get_node_spec(node_name)
    if not spec:
        return True, None  # 사양이 없으면 검증 통과

    missing_fields = []
    for field in spec.required_fields:
        if field not in state:
            missing_fields.append(field)

    if missing_fields:
        return False, f"Missing required fields in {node_name}: {missing_fields}"

    return True, None


def get_all_node_names() -> List[str]:
    """모든 노드 이름 반환"""
    return list(FLAT_NODE_SPECS.keys())
