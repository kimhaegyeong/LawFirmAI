# -*- coding: utf-8 -*-
"""
LangGraph 노드별 Input/Output 사양 정의
각 노드가 사용하는 입력 데이터와 출력 데이터를 명확히 정의

효과:
- 메모리 사용량 최적화: 필요한 데이터만 전달
- 타입 안전성 향상: 런타임 검증
- 디버깅 용이: 명확한 Input/Output
- 문서화: 각 노드의 역할 명확화
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class NodeCategory(str, Enum):
    """노드 카테고리"""
    INPUT = "input"
    CLASSIFICATION = "classification"
    SEARCH = "search"
    GENERATION = "generation"
    VALIDATION = "validation"
    ENHANCEMENT = "enhancement"
    CONTROL = "control"


@dataclass
class NodeIOSpec:
    """노드별 Input/Output 사양"""
    node_name: str
    category: NodeCategory
    description: str
    required_input: Dict[str, str]  # {필드명: 설명}
    optional_input: Dict[str, str]
    output: Dict[str, str]
    required_state_groups: Set[str]  # 필요한 State 그룹
    output_state_groups: Set[str]  # 출력되는 State 그룹

    def validate_input(self, state: Dict) -> tuple[bool, Optional[str]]:
        """Input 유효성 검증"""
        missing_fields = []
        for field in self.required_input:
            if self._check_field_in_state(field, state):
                continue
            missing_fields.append(field)

        if missing_fields:
            return False, f"Missing required fields in {self.node_name}: {missing_fields}"
        return True, None

    def _check_field_in_state(self, field: str, state: Dict) -> bool:
        """State에서 필드 존재 확인 (nested/flat 모두 지원)"""
        # Nested 구조 확인
        if "input" in state and isinstance(state["input"], dict) and field in state.get("input", {}):
            return True

        # Flat 구조 확인
        if field in state:
            return True

        # Search, Answer 등 그룹 내 확인
        for group in ["search", "answer", "classification", "validation", "control", "common"]:
            if group in state and isinstance(state[group], dict) and field in state[group]:
                return True

        return False


# ============================================
# 노드별 Input/Output 사양 정의
# ============================================

NODE_SPECS: Dict[str, NodeIOSpec] = {
    "classify_query": NodeIOSpec(
        node_name="classify_query",
        category=NodeCategory.CLASSIFICATION,
        description="질문 유형 분류 및 법률 분야 판단",
        required_input={
            "query": "사용자 질문",
        },
        optional_input={
            "legal_field": "법률 분야 힌트"
        },
        output={
            "query_type": "질문 유형",
            "confidence": "신뢰도 점수",
            "legal_field": "법률 분야",
            "legal_domain": "법률 도메인"
        },
        required_state_groups={"input"},
        output_state_groups={"classification"}
    ),

    "assess_urgency": NodeIOSpec(
        node_name="assess_urgency",
        category=NodeCategory.CLASSIFICATION,
        description="질문의 긴급도 평가",
        required_input={
            "query": "사용자 질문",
        },
        optional_input={
            "query_type": "질문 유형",
            "legal_field": "법률 분야"
        },
        output={
            "urgency_level": "긴급도 레벨 (low/medium/high/critical)",
            "urgency_reasoning": "긴급도 평가 근거",
            "emergency_type": "긴급 상황 유형"
        },
        required_state_groups={"input"},
        output_state_groups={"classification"}
    ),

    "resolve_multi_turn": NodeIOSpec(
        node_name="resolve_multi_turn",
        category=NodeCategory.CLASSIFICATION,
        description="멀티턴 대화 처리",
        required_input={
            "query": "사용자 질문"
        },
        optional_input={
            # 대화 이력은 내부 어댑터가 보존하므로 노드의 선택 입력에서 제외
        },
        output={
            "is_multi_turn": "멀티턴 여부",
            "multi_turn_confidence": "멀티턴 확신도",
            "conversation_history": "대화 이력",
            "conversation_context": "대화 컨텍스트"
        },
        required_state_groups={"input"},
        output_state_groups={"multi_turn"}
    ),

    "route_expert": NodeIOSpec(
        node_name="route_expert",
        category=NodeCategory.CLASSIFICATION,
        description="전문가 라우팅 결정",
        required_input={
            "query": "사용자 질문",
            "query_type": "질문 유형"
        },
        optional_input={
            "legal_field": "법률 분야",
            "urgency_level": "긴급도"
        },
        output={
            "complexity_level": "복잡도 레벨 (simple/medium/complex)",
            "requires_expert": "전문가 필요 여부",
            "expert_subgraph": "전문가 서브그래프"
        },
        required_state_groups={"input", "classification"},
        output_state_groups={"classification"}
    ),

    "analyze_document": NodeIOSpec(
        node_name="analyze_document",
        category=NodeCategory.CLASSIFICATION,
        description="업로드된 문서 분석",
        required_input={
            "query": "사용자 질문"
        },
        optional_input={
            "document_file": "업로드된 문서"
        },
        output={
            "document_type": "문서 유형",
            "document_analysis": "문서 분석 결과",
            "key_clauses": "핵심 조항",
            "potential_issues": "잠재적 문제점"
        },
        required_state_groups={"input"},
        output_state_groups={"document"}
    ),

    "expand_keywords_ai": NodeIOSpec(
        node_name="expand_keywords_ai",
        category=NodeCategory.SEARCH,
        description="AI 기반 키워드 확장",
        required_input={
            "query": "사용자 질문",
            "query_type": "질문 유형"
        },
        optional_input={
            "legal_field": "법률 분야",
            "extracted_keywords": "기존 키워드"
        },
        output={
            "search_query": "개선된 검색 쿼리",
            "extracted_keywords": "추출된 키워드",
            "ai_keyword_expansion": "AI 키워드 확장 결과"
        },
        required_state_groups={"input", "classification"},
        output_state_groups={"search"}
    ),

    "prepare_search_query": NodeIOSpec(
        node_name="prepare_search_query",
        category=NodeCategory.SEARCH,
        description="검색 쿼리 준비 및 최적화",
        required_input={
            "query": "사용자 질문",
            "query_type": "질문 유형"
        },
        optional_input={
            "legal_field": "법률 분야",
            "extracted_keywords": "추출된 키워드",
            "search_query": "기존 검색 쿼리"
        },
        output={
            "optimized_queries": "최적화된 검색 쿼리",
            "search_params": "검색 파라미터",
            "search_cache_hit": "캐시 히트 여부"
        },
        required_state_groups={"input", "classification"},  # query가 필요하므로 input 그룹 필수
        output_state_groups={"search"}
    ),

    "process_legal_terms": NodeIOSpec(
        node_name="process_legal_terms",
        category=NodeCategory.ENHANCEMENT,
        description="법률 용어 처리 및 통합",
        required_input={
            "query": "사용자 질문",
            "retrieved_docs": "검색된 문서"
        },
        optional_input={
            "legal_field": "법률 분야"
        },
        output={
            "legal_references": "법령 참조 리스트",
            "legal_citations": "법령 인용 정보",
            "analysis": "법률 분석 결과"
        },
        required_state_groups={"input", "search"},
        output_state_groups={"analysis"}
    ),

    "prepare_document_context_for_prompt": NodeIOSpec(
        node_name="prepare_document_context_for_prompt",
        category=NodeCategory.ENHANCEMENT,
        description="프롬프트용 문서 컨텍스트 준비",
        required_input={
            "query": "사용자 질문",
            "retrieved_docs": "검색된 문서"
        },
        optional_input={
            "query_type": "질문 유형",
            "extracted_keywords": "추출된 키워드",
            "legal_field": "법률 분야"
        },
        output={
            "prompt_optimized_context": "프롬프트 최적화된 문서 컨텍스트"
        },
        required_state_groups={"input", "search"},
        output_state_groups={"search", "common"}  # common에도 포함하여 보존
    ),

    "generate_answer_enhanced": NodeIOSpec(
        node_name="generate_answer_enhanced",
        category=NodeCategory.GENERATION,
        description="향상된 답변 생성 (LLM 활용)",
        required_input={
            "query": "사용자 질문",
            "retrieved_docs": "검색된 문서"
        },
        optional_input={
            "query_type": "질문 유형",
            "legal_field": "법률 분야",
            "analysis": "법률 분석",
            "legal_references": "법령 참조",
            "prompt_optimized_context": "프롬프트 최적화된 문서 컨텍스트"
        },
        output={
            "answer": "생성된 답변",
            "confidence": "신뢰도 점수",
            "legal_references": "법령 참조",
            "legal_citations": "법령 인용"
        },
        required_state_groups={"input", "search"},  # 최소 의존성만 필수
        output_state_groups={"answer", "analysis", "common"}  # common 출력 그룹에 포함
    ),

    "validate_answer_quality": NodeIOSpec(
        node_name="validate_answer_quality",
        category=NodeCategory.VALIDATION,
        description="답변 품질 및 법령 검증",
        required_input={
            "answer": "생성된 답변",
            "query": "원본 질문"
        },
        optional_input={
            "retrieved_docs": "검색 문서",
            "sources": "소스",
            "legal_references": "법령 참조"
        },
        output={
            "quality_check_passed": "품질 검증 통과 여부",
            "quality_score": "품질 점수",
            "legal_validity_check": "법령 검증",
            "legal_basis_validation": "법적 근거 검증"
        },
        required_state_groups={"input", "answer"},  # 최소 의존성만 필수
        output_state_groups={"validation", "control", "common"}  # common 출력 그룹에 포함
    ),

    "enhance_answer_structure": NodeIOSpec(
        node_name="enhance_answer_structure",
        category=NodeCategory.ENHANCEMENT,
        description="답변 구조화 및 법적 근거 강화",
        required_input={
            "answer": "생성된 답변",
            "query_type": "질문 유형"
        },
        optional_input={
            "legal_references": "법령 참조",
            "legal_citations": "법령 인용",
            "retrieved_docs": "검색 문서"
        },
        output={
            "answer": "구조화된 답변",
            "structure_confidence": "구조화 신뢰도"
        },
        required_state_groups={"answer", "classification"},
        output_state_groups={"answer"}
    ),

    "apply_visual_formatting": NodeIOSpec(
        node_name="apply_visual_formatting",
        category=NodeCategory.ENHANCEMENT,
        description="시각적 포맷팅 적용",
        required_input={
            "answer": "답변",
        },
        optional_input={
            "query_type": "질문 유형",
            "legal_references": "법령 참조"
        },
        output={
            "answer": "포맷팅된 답변"
        },
        required_state_groups={"answer"},
        output_state_groups={"answer"}
    ),

    "prepare_final_response": NodeIOSpec(
        node_name="prepare_final_response",
        category=NodeCategory.GENERATION,
        description="최종 응답 준비",
        required_input={
            "answer": "답변"
        },
        optional_input={
            "sources": "소스",
            "legal_references": "법령 참조",
            "confidence": "신뢰도",
            "legal_validity_check": "법령 검증 결과"
        },
        output={
            "answer": "최종 답변",
            "sources": "최종 소스",
            "confidence": "최종 신뢰도"
        },
        required_state_groups={"answer"},
        output_state_groups={"answer", "common"}
    ),

    "generate_and_validate_answer": NodeIOSpec(
        node_name="generate_and_validate_answer",
        category=NodeCategory.GENERATION,
        description="통합된 답변 생성, 검증, 포맷팅 및 최종 준비",
        required_input={
            "query": "사용자 질문",
            "retrieved_docs": "검색된 문서"
        },
        optional_input={
            "query_type": "질문 유형",
            "legal_field": "법률 분야",
            "legal_references": "법령 참조"
        },
        output={
            "answer": "생성 및 검증된 답변",
            "confidence": "신뢰도 점수",
            "quality_check_passed": "품질 검증 통과 여부",
            "legal_validity_check": "법령 검증"
        },
        required_state_groups={"input", "search"},  # Phase 6: answer 보존을 위해 입력에서 answer 그룹은 선택적
        output_state_groups={"answer", "validation", "control", "common"}  # Phase 6: answer 그룹 필수 출력
    ),

    "direct_answer": NodeIOSpec(
        node_name="direct_answer",
        category=NodeCategory.GENERATION,
        description="간단한 질문 - 검색 없이 LLM만 사용하여 답변 생성",
        required_input={
            "query": "사용자 질문",
            "query_type": "질문 유형"
        },
        optional_input={
            "legal_field": "법률 분야"
        },
        output={
            "answer": "직접 생성된 답변",
            "confidence": "신뢰도 점수",
            "sources": "소스 목록 (빈 목록)"
        },
        required_state_groups={"input", "classification"},  # Phase 6: answer 보존을 위해 입력에서 answer 그룹은 선택적
        output_state_groups={"answer", "common"}  # Phase 6: answer 그룹 필수 출력
    ),

    "execute_searches_parallel": NodeIOSpec(
        node_name="execute_searches_parallel",
        category=NodeCategory.SEARCH,
        description="의미적 검색과 키워드 검색을 병렬로 실행",
        required_input={
            "query": "사용자 질문",
            "optimized_queries": "최적화된 검색 쿼리",
            "search_params": "검색 파라미터"
        },
        optional_input={
            "query_type": "질문 유형",
            "legal_field": "법률 분야",
            "extracted_keywords": "추출된 키워드"
        },
        output={
            "semantic_results": "의미적 검색 결과",
            "keyword_results": "키워드 검색 결과",
            "semantic_count": "의미적 검색 결과 수",
            "keyword_count": "키워드 검색 결과 수"
        },
        required_state_groups={"input", "search"},  # search 그룹 필요
        output_state_groups={"search"}  # search 그룹에 저장
    ),

    "evaluate_search_quality": NodeIOSpec(
        node_name="evaluate_search_quality",
        category=NodeCategory.SEARCH,
        description="검색 결과 품질 평가",
        required_input={
            "semantic_results": "의미적 검색 결과",
            "keyword_results": "키워드 검색 결과"
        },
        optional_input={
            "query": "사용자 질문",
            "query_type": "질문 유형",
            "search_params": "검색 파라미터"
        },
        output={
            "search_quality_evaluation": "검색 품질 평가 결과"
        },
        required_state_groups={"input", "search"},
        output_state_groups={"search", "common"}
    ),

    "conditional_retry_search": NodeIOSpec(
        node_name="conditional_retry_search",
        category=NodeCategory.SEARCH,
        description="검색 품질에 따른 조건부 재검색",
        required_input={
            "search_quality_evaluation": "검색 품질 평가 결과",
            "semantic_results": "의미적 검색 결과",
            "keyword_results": "키워드 검색 결과"
        },
        optional_input={
            "query": "사용자 질문",
            "optimized_queries": "최적화된 검색 쿼리"
        },
        output={
            "semantic_results": "재검색된 의미적 결과",
            "keyword_results": "재검색된 키워드 결과"
        },
        required_state_groups={"input", "search"},
        output_state_groups={"search"}
    ),

    "merge_and_rerank_with_keyword_weights": NodeIOSpec(
        node_name="merge_and_rerank_with_keyword_weights",
        category=NodeCategory.SEARCH,
        description="키워드별 가중치를 적용한 결과 병합 및 Reranking",
        required_input={
            "semantic_results": "의미적 검색 결과",
            "keyword_results": "키워드 검색 결과"
        },
        optional_input={
            "query": "사용자 질문",
            "optimized_queries": "최적화된 검색 쿼리",
            "search_params": "검색 파라미터",
            "extracted_keywords": "추출된 키워드",
            "legal_field": "법률 분야"
        },
        output={
            "merged_documents": "병합 및 Reranking된 문서",
            "keyword_weights": "키워드별 가중치",
            "retrieved_docs": "검색된 문서 (최종 결과)"
        },
        required_state_groups={"input", "search"},  # search 그룹 필요 (semantic_results, keyword_results 포함)
        output_state_groups={"search"}  # search 그룹에 저장
    ),

    "filter_and_validate_results": NodeIOSpec(
        node_name="filter_and_validate_results",
        category=NodeCategory.SEARCH,
        description="검색 결과 필터링 및 품질 검증",
        required_input={
            "merged_documents": "병합된 문서"
        },
        optional_input={
            "query": "사용자 질문",
            "query_type": "질문 유형",
            "legal_field": "법률 분야",
            "search_params": "검색 파라미터",
            "retrieved_docs": "기존 검색된 문서"
        },
        output={
            "retrieved_docs": "필터링된 검색 문서"
        },
        required_state_groups={"input", "search"},  # search 그룹 필요
        output_state_groups={"search"}
    ),

    "update_search_metadata": NodeIOSpec(
        node_name="update_search_metadata",
        category=NodeCategory.SEARCH,
        description="검색 메타데이터 업데이트",
        required_input={
            "retrieved_docs": "검색된 문서"
        },
        optional_input={
            "semantic_count": "의미적 검색 결과 수",
            "keyword_count": "키워드 검색 결과 수",
            "optimized_queries": "최적화된 검색 쿼리"
        },
        output={
            "search_metadata": "업데이트된 검색 메타데이터"
        },
        required_state_groups={"input", "search"},
        output_state_groups={"search", "common"}
    ),

    "process_search_results_combined": NodeIOSpec(
        node_name="process_search_results_combined",
        category=NodeCategory.SEARCH,
        description="검색 결과 처리 통합 노드 (6개 노드를 1개로 병합)",
        required_input={
            "semantic_results": "의미적 검색 결과",
            "keyword_results": "키워드 검색 결과"
        },
        optional_input={
            "query": "사용자 질문",
            "query_type": "질문 유형",
            "optimized_queries": "최적화된 검색 쿼리",
            "search_params": "검색 파라미터",
            "extracted_keywords": "추출된 키워드",
            "legal_field": "법률 분야"
        },
        output={
            "retrieved_docs": "검색된 문서 (최종 결과)",
            "merged_documents": "병합된 문서",
            "search_metadata": "검색 메타데이터",
            "search_quality_evaluation": "검색 품질 평가 결과"
        },
        required_state_groups={"input", "search"},
        output_state_groups={"search", "common"}  # search와 common 그룹에 저장하여 보존
    )
}


# ============================================
# 헬퍼 함수
# ============================================

def get_node_spec(node_name: str) -> Optional[NodeIOSpec]:
    """노드별 사양 조회"""
    return NODE_SPECS.get(node_name)


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

    return spec.validate_input(state)


def get_required_state_groups(node_name: str) -> Set[str]:
    """노드에 필요한 State 그룹 반환"""
    spec = get_node_spec(node_name)
    if spec:
        return spec.required_state_groups
    return set()


def get_output_state_groups(node_name: str) -> Set[str]:
    """노드가 출력하는 State 그룹 반환"""
    spec = get_node_spec(node_name)
    if spec:
        return spec.output_state_groups
    return set()


def get_all_node_names() -> List[str]:
    """모든 노드 이름 반환"""
    return list(NODE_SPECS.keys())


def get_nodes_by_category(category: NodeCategory) -> List[NodeIOSpec]:
    """카테고리별 노드 반환"""
    return [spec for spec in NODE_SPECS.values() if spec.category == category]


# ============================================
# 검증 및 디버깅
# ============================================

def validate_workflow_flow() -> Dict[str, Any]:
    """전체 워크플로우 흐름 검증"""
    issues = []

    # 각 노드의 Input이 이전 노드의 Output과 일치하는지 확인
    node_names = get_all_node_names()

    for node_name in node_names:
        spec = get_node_spec(node_name)
        if not spec:
            continue

        # Required input 체크
        for required_field in spec.required_input:
            # 이전 노드에서 제공되는지 확인
            found = False
            for other_node in node_names:
                if other_node == node_name:
                    continue
                other_spec = get_node_spec(other_node)
                if other_spec and required_field in other_spec.output:
                    found = True
                    break

            if not found and not required_field.startswith("query"):  # query는 초기 입력
                issues.append(f"{node_name}: 필수 입력 '{required_field}'이 이전 노드에서 제공되지 않음")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "total_nodes": len(node_names)
    }


if __name__ == "__main__":
    # 검증 실행
    result = validate_workflow_flow()
    print(f"워크플로우 검증 결과: {'✅ Valid' if result['valid'] else '❌ Invalid'}")
    print(f"총 노드 수: {result['total_nodes']}")

    if result['issues']:
        print("\n문제점:")
        for issue in result['issues']:
            print(f"  - {issue}")
