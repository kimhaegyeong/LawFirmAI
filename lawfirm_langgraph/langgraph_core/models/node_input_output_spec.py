# -*- coding: utf-8 -*-
"""
LangGraph ?�드�?Input/Output ?�양 ?�의
�??�드가 ?�용?�는 ?�력 ?�이?��? 출력 ?�이?��? 명확???�의

?�과:
- 메모�??�용??최적?? ?�요???�이?�만 ?�달
- ?�???�전???�상: ?��???검�?
- ?�버�??�이: 명확??Input/Output
- 문서?? �??�드????�� 명확??
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class NodeCategory(str, Enum):
    """?�드 카테고리"""
    INPUT = "input"
    CLASSIFICATION = "classification"
    SEARCH = "search"
    GENERATION = "generation"
    VALIDATION = "validation"
    ENHANCEMENT = "enhancement"
    CONTROL = "control"


@dataclass
class NodeIOSpec:
    """?�드�?Input/Output ?�양"""
    node_name: str
    category: NodeCategory
    description: str
    required_input: Dict[str, str]  # {?�드�? ?�명}
    optional_input: Dict[str, str]
    output: Dict[str, str]
    required_state_groups: Set[str]  # ?�요??State 그룹
    output_state_groups: Set[str]  # 출력?�는 State 그룹

    def validate_input(self, state: Dict) -> tuple[bool, Optional[str]]:
        """Input ?�효??검�?""
        missing_fields = []
        for field in self.required_input:
            if self._check_field_in_state(field, state):
                continue
            missing_fields.append(field)

        if missing_fields:
            return False, f"Missing required fields in {self.node_name}: {missing_fields}"
        return True, None

    def _check_field_in_state(self, field: str, state: Dict) -> bool:
        """State?�서 ?�드 존재 ?�인 (nested/flat 모두 지??"""
        # Nested 구조 ?�인
        if "input" in state and isinstance(state["input"], dict) and field in state.get("input", {}):
            return True

        # Flat 구조 ?�인
        if field in state:
            return True

        # Search, Answer ??그룹 ???�인
        for group in ["search", "answer", "classification", "validation", "control", "common"]:
            if group in state and isinstance(state[group], dict) and field in state[group]:
                return True

        return False


# ============================================
# ?�드�?Input/Output ?�양 ?�의
# ============================================

NODE_SPECS: Dict[str, NodeIOSpec] = {
    "classify_query": NodeIOSpec(
        node_name="classify_query",
        category=NodeCategory.CLASSIFICATION,
        description="질문 ?�형 분류 �?법률 분야 ?�단",
        required_input={
            "query": "?�용??질문",
        },
        optional_input={
            "legal_field": "법률 분야 ?�트"
        },
        output={
            "query_type": "질문 ?�형",
            "confidence": "?�뢰???�수",
            "legal_field": "법률 분야",
            "legal_domain": "법률 ?�메??
        },
        required_state_groups={"input"},
        output_state_groups={"classification"}
    ),

    "assess_urgency": NodeIOSpec(
        node_name="assess_urgency",
        category=NodeCategory.CLASSIFICATION,
        description="질문??긴급???��?",
        required_input={
            "query": "?�용??질문",
        },
        optional_input={
            "query_type": "질문 ?�형",
            "legal_field": "법률 분야"
        },
        output={
            "urgency_level": "긴급???�벨 (low/medium/high/critical)",
            "urgency_reasoning": "긴급???��? 근거",
            "emergency_type": "긴급 ?�황 ?�형"
        },
        required_state_groups={"input"},
        output_state_groups={"classification"}
    ),

    "resolve_multi_turn": NodeIOSpec(
        node_name="resolve_multi_turn",
        category=NodeCategory.CLASSIFICATION,
        description="멀?�턴 ?�??처리",
        required_input={
            "query": "?�용??질문"
        },
        optional_input={
            # ?�???�력?� ?��? ?�댑?��? 보존?��?�??�드???�택 ?�력?�서 ?�외
        },
        output={
            "is_multi_turn": "멀?�턴 ?��?",
            "multi_turn_confidence": "멀?�턴 ?�신??,
            "conversation_history": "?�???�력",
            "conversation_context": "?�??컨텍?�트"
        },
        required_state_groups={"input"},
        output_state_groups={"multi_turn"}
    ),

    "route_expert": NodeIOSpec(
        node_name="route_expert",
        category=NodeCategory.CLASSIFICATION,
        description="?�문가 ?�우??결정",
        required_input={
            "query": "?�용??질문",
            "query_type": "질문 ?�형"
        },
        optional_input={
            "legal_field": "법률 분야",
            "urgency_level": "긴급??
        },
        output={
            "complexity_level": "복잡???�벨 (simple/medium/complex)",
            "requires_expert": "?�문가 ?�요 ?��?",
            "expert_subgraph": "?�문가 ?�브그래??
        },
        required_state_groups={"input", "classification"},
        output_state_groups={"classification"}
    ),

    "analyze_document": NodeIOSpec(
        node_name="analyze_document",
        category=NodeCategory.CLASSIFICATION,
        description="?�로?�된 문서 분석",
        required_input={
            "query": "?�용??질문"
        },
        optional_input={
            "document_file": "?�로?�된 문서"
        },
        output={
            "document_type": "문서 ?�형",
            "document_analysis": "문서 분석 결과",
            "key_clauses": "?�심 조항",
            "potential_issues": "?�재??문제??
        },
        required_state_groups={"input"},
        output_state_groups={"document"}
    ),

    "expand_keywords_ai": NodeIOSpec(
        node_name="expand_keywords_ai",
        category=NodeCategory.SEARCH,
        description="AI 기반 ?�워???�장",
        required_input={
            "query": "?�용??질문",
            "query_type": "질문 ?�형"
        },
        optional_input={
            "legal_field": "법률 분야",
            "extracted_keywords": "기존 ?�워??
        },
        output={
            "search_query": "개선??검??쿼리",
            "extracted_keywords": "추출???�워??,
            "ai_keyword_expansion": "AI ?�워???�장 결과"
        },
        required_state_groups={"input", "classification"},
        output_state_groups={"search"}
    ),

    "prepare_search_query": NodeIOSpec(
        node_name="prepare_search_query",
        category=NodeCategory.SEARCH,
        description="검??쿼리 준�?�?최적??,
        required_input={
            "query": "?�용??질문",
            "query_type": "질문 ?�형"
        },
        optional_input={
            "legal_field": "법률 분야",
            "extracted_keywords": "추출???�워??,
            "search_query": "기존 검??쿼리"
        },
        output={
            "optimized_queries": "최적?�된 검??쿼리",
            "search_params": "검???�라미터",
            "search_cache_hit": "캐시 ?�트 ?��?"
        },
        required_state_groups={"input", "classification"},  # query가 ?�요?��?�?input 그룹 ?�수
        output_state_groups={"search"}
    ),

    "process_legal_terms": NodeIOSpec(
        node_name="process_legal_terms",
        category=NodeCategory.ENHANCEMENT,
        description="법률 ?�어 처리 �??�합",
        required_input={
            "query": "?�용??질문",
            "retrieved_docs": "검?�된 문서"
        },
        optional_input={
            "legal_field": "법률 분야"
        },
        output={
            "legal_references": "법령 참조 리스??,
            "legal_citations": "법령 ?�용 ?�보",
            "analysis": "법률 분석 결과"
        },
        required_state_groups={"input", "search"},
        output_state_groups={"analysis"}
    ),

    "prepare_document_context_for_prompt": NodeIOSpec(
        node_name="prepare_document_context_for_prompt",
        category=NodeCategory.ENHANCEMENT,
        description="?�롬?�트??문서 컨텍?�트 준�?,
        required_input={
            "query": "?�용??질문",
            "retrieved_docs": "검?�된 문서"
        },
        optional_input={
            "query_type": "질문 ?�형",
            "extracted_keywords": "추출???�워??,
            "legal_field": "법률 분야"
        },
        output={
            "prompt_optimized_context": "?�롬?�트 최적?�된 문서 컨텍?�트"
        },
        required_state_groups={"input", "search"},
        output_state_groups={"search", "common"}  # common?�도 ?�함?�여 보존
    ),

    "generate_answer_enhanced": NodeIOSpec(
        node_name="generate_answer_enhanced",
        category=NodeCategory.GENERATION,
        description="?�상???��? ?�성 (LLM ?�용)",
        required_input={
            "query": "?�용??질문",
            "retrieved_docs": "검?�된 문서"
        },
        optional_input={
            "query_type": "질문 ?�형",
            "legal_field": "법률 분야",
            "analysis": "법률 분석",
            "legal_references": "법령 참조",
            "prompt_optimized_context": "?�롬?�트 최적?�된 문서 컨텍?�트"
        },
        output={
            "answer": "?�성???��?",
            "confidence": "?�뢰???�수",
            "legal_references": "법령 참조",
            "legal_citations": "법령 ?�용"
        },
        required_state_groups={"input", "search"},  # 최소 ?�존?�만 ?�수
        output_state_groups={"answer", "analysis", "common"}  # common 출력 그룹???�함
    ),

    "validate_answer_quality": NodeIOSpec(
        node_name="validate_answer_quality",
        category=NodeCategory.VALIDATION,
        description="?��? ?�질 �?법령 검�?,
        required_input={
            "answer": "?�성???��?",
            "query": "?�본 질문"
        },
        optional_input={
            "retrieved_docs": "검??문서",
            "sources": "?�스",
            "legal_references": "법령 참조"
        },
        output={
            "quality_check_passed": "?�질 검�??�과 ?��?",
            "quality_score": "?�질 ?�수",
            "legal_validity_check": "법령 검�?,
            "legal_basis_validation": "법적 근거 검�?
        },
        required_state_groups={"input", "answer"},  # 최소 ?�존?�만 ?�수
        output_state_groups={"validation", "control", "common"}  # common 출력 그룹???�함
    ),

    "enhance_answer_structure": NodeIOSpec(
        node_name="enhance_answer_structure",
        category=NodeCategory.ENHANCEMENT,
        description="?��? 구조??�?법적 근거 강화",
        required_input={
            "answer": "?�성???��?",
            "query_type": "질문 ?�형"
        },
        optional_input={
            "legal_references": "법령 참조",
            "legal_citations": "법령 ?�용",
            "retrieved_docs": "검??문서"
        },
        output={
            "answer": "구조?�된 ?��?",
            "structure_confidence": "구조???�뢰??
        },
        required_state_groups={"answer", "classification"},
        output_state_groups={"answer"}
    ),

    "apply_visual_formatting": NodeIOSpec(
        node_name="apply_visual_formatting",
        category=NodeCategory.ENHANCEMENT,
        description="?�각???�맷???�용",
        required_input={
            "answer": "?��?",
        },
        optional_input={
            "query_type": "질문 ?�형",
            "legal_references": "법령 참조"
        },
        output={
            "answer": "?�맷?�된 ?��?"
        },
        required_state_groups={"answer"},
        output_state_groups={"answer"}
    ),

    "prepare_final_response": NodeIOSpec(
        node_name="prepare_final_response",
        category=NodeCategory.GENERATION,
        description="최종 ?�답 준�?,
        required_input={
            "answer": "?��?"
        },
        optional_input={
            "sources": "?�스",
            "legal_references": "법령 참조",
            "confidence": "?�뢰??,
            "legal_validity_check": "법령 검�?결과"
        },
        output={
            "answer": "최종 ?��?",
            "sources": "최종 ?�스",
            "confidence": "최종 ?�뢰??
        },
        required_state_groups={"answer"},
        output_state_groups={"answer", "common"}
    ),

    "generate_and_validate_answer": NodeIOSpec(
        node_name="generate_and_validate_answer",
        category=NodeCategory.GENERATION,
        description="?�합???��? ?�성, 검�? ?�맷??�?최종 준�?,
        required_input={
            "query": "?�용??질문",
            "retrieved_docs": "검?�된 문서"
        },
        optional_input={
            "query_type": "질문 ?�형",
            "legal_field": "법률 분야",
            "legal_references": "법령 참조"
        },
        output={
            "answer": "?�성 �?검증된 ?��?",
            "confidence": "?�뢰???�수",
            "quality_check_passed": "?�질 검�??�과 ?��?",
            "legal_validity_check": "법령 검�?
        },
        required_state_groups={"input", "search"},  # Phase 6: answer 보존???�해 ?�력?�서 answer 그룹?� ?�택??
        output_state_groups={"answer", "validation", "control", "common"}  # Phase 6: answer 그룹 ?�수 출력
    ),

    "direct_answer": NodeIOSpec(
        node_name="direct_answer",
        category=NodeCategory.GENERATION,
        description="간단??질문 - 검???�이 LLM�??�용?�여 ?��? ?�성",
        required_input={
            "query": "?�용??질문",
            "query_type": "질문 ?�형"
        },
        optional_input={
            "legal_field": "법률 분야"
        },
        output={
            "answer": "직접 ?�성???��?",
            "confidence": "?�뢰???�수",
            "sources": "?�스 목록 (�?목록)"
        },
        required_state_groups={"input", "classification"},  # Phase 6: answer 보존???�해 ?�력?�서 answer 그룹?� ?�택??
        output_state_groups={"answer", "common"}  # Phase 6: answer 그룹 ?�수 출력
    ),

    "execute_searches_parallel": NodeIOSpec(
        node_name="execute_searches_parallel",
        category=NodeCategory.SEARCH,
        description="?��???검?�과 ?�워??검?�을 병렬�??�행",
        required_input={
            "query": "?�용??질문",
            "optimized_queries": "최적?�된 검??쿼리",
            "search_params": "검???�라미터"
        },
        optional_input={
            "query_type": "질문 ?�형",
            "legal_field": "법률 분야",
            "extracted_keywords": "추출???�워??
        },
        output={
            "semantic_results": "?��???검??결과",
            "keyword_results": "?�워??검??결과",
            "semantic_count": "?��???검??결과 ??,
            "keyword_count": "?�워??검??결과 ??
        },
        required_state_groups={"input", "search"},  # search 그룹 ?�요
        output_state_groups={"search"}  # search 그룹???�??
    ),

    "evaluate_search_quality": NodeIOSpec(
        node_name="evaluate_search_quality",
        category=NodeCategory.SEARCH,
        description="검??결과 ?�질 ?��?",
        required_input={
            "semantic_results": "?��???검??결과",
            "keyword_results": "?�워??검??결과"
        },
        optional_input={
            "query": "?�용??질문",
            "query_type": "질문 ?�형",
            "search_params": "검???�라미터"
        },
        output={
            "search_quality_evaluation": "검???�질 ?��? 결과"
        },
        required_state_groups={"input", "search"},
        output_state_groups={"search", "common"}
    ),

    "conditional_retry_search": NodeIOSpec(
        node_name="conditional_retry_search",
        category=NodeCategory.SEARCH,
        description="검???�질???�른 조건부 ?��???,
        required_input={
            "search_quality_evaluation": "검???�질 ?��? 결과",
            "semantic_results": "?��???검??결과",
            "keyword_results": "?�워??검??결과"
        },
        optional_input={
            "query": "?�용??질문",
            "optimized_queries": "최적?�된 검??쿼리"
        },
        output={
            "semantic_results": "?��??�된 ?��???결과",
            "keyword_results": "?��??�된 ?�워??결과"
        },
        required_state_groups={"input", "search"},
        output_state_groups={"search"}
    ),

    "merge_and_rerank_with_keyword_weights": NodeIOSpec(
        node_name="merge_and_rerank_with_keyword_weights",
        category=NodeCategory.SEARCH,
        description="?�워?�별 가중치�??�용??결과 병합 �?Reranking",
        required_input={
            "semantic_results": "?��???검??결과",
            "keyword_results": "?�워??검??결과"
        },
        optional_input={
            "query": "?�용??질문",
            "optimized_queries": "최적?�된 검??쿼리",
            "search_params": "검???�라미터",
            "extracted_keywords": "추출???�워??,
            "legal_field": "법률 분야"
        },
        output={
            "merged_documents": "병합 �?Reranking??문서",
            "keyword_weights": "?�워?�별 가중치",
            "retrieved_docs": "검?�된 문서 (최종 결과)"
        },
        required_state_groups={"input", "search"},  # search 그룹 ?�요 (semantic_results, keyword_results ?�함)
        output_state_groups={"search"}  # search 그룹???�??
    ),

    "filter_and_validate_results": NodeIOSpec(
        node_name="filter_and_validate_results",
        category=NodeCategory.SEARCH,
        description="검??결과 ?�터�?�??�질 검�?,
        required_input={
            "merged_documents": "병합??문서"
        },
        optional_input={
            "query": "?�용??질문",
            "query_type": "질문 ?�형",
            "legal_field": "법률 분야",
            "search_params": "검???�라미터",
            "retrieved_docs": "기존 검?�된 문서"
        },
        output={
            "retrieved_docs": "?�터링된 검??문서"
        },
        required_state_groups={"input", "search"},  # search 그룹 ?�요
        output_state_groups={"search"}
    ),

    "update_search_metadata": NodeIOSpec(
        node_name="update_search_metadata",
        category=NodeCategory.SEARCH,
        description="검??메�??�이???�데?�트",
        required_input={
            "retrieved_docs": "검?�된 문서"
        },
        optional_input={
            "semantic_count": "?��???검??결과 ??,
            "keyword_count": "?�워??검??결과 ??,
            "optimized_queries": "최적?�된 검??쿼리"
        },
        output={
            "search_metadata": "?�데?�트??검??메�??�이??
        },
        required_state_groups={"input", "search"},
        output_state_groups={"search", "common"}
    ),

    "process_search_results_combined": NodeIOSpec(
        node_name="process_search_results_combined",
        category=NodeCategory.SEARCH,
        description="검??결과 처리 ?�합 ?�드 (6�??�드�?1개로 병합)",
        required_input={
            "semantic_results": "?��???검??결과",
            "keyword_results": "?�워??검??결과"
        },
        optional_input={
            "query": "?�용??질문",
            "query_type": "질문 ?�형",
            "optimized_queries": "최적?�된 검??쿼리",
            "search_params": "검???�라미터",
            "extracted_keywords": "추출???�워??,
            "legal_field": "법률 분야"
        },
        output={
            "retrieved_docs": "검?�된 문서 (최종 결과)",
            "merged_documents": "병합??문서",
            "search_metadata": "검??메�??�이??,
            "search_quality_evaluation": "검???�질 ?��? 결과"
        },
        required_state_groups={"input", "search"},
        output_state_groups={"search", "common"}  # search?� common 그룹???�?�하??보존
    )
}


# ============================================
# ?�퍼 ?�수
# ============================================

def get_node_spec(node_name: str) -> Optional[NodeIOSpec]:
    """?�드�??�양 조회"""
    return NODE_SPECS.get(node_name)


def validate_node_input(node_name: str, state: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    ?�드 Input ?�효??검�?

    Args:
        node_name: ?�드 ?�름
        state: State 객체

    Returns:
        (is_valid, error_message) ?�플
    """
    spec = get_node_spec(node_name)
    if not spec:
        return True, None  # ?�양???�으�?검�??�과

    return spec.validate_input(state)


def get_required_state_groups(node_name: str) -> Set[str]:
    """?�드???�요??State 그룹 반환"""
    spec = get_node_spec(node_name)
    if spec:
        return spec.required_state_groups
    return set()


def get_output_state_groups(node_name: str) -> Set[str]:
    """?�드가 출력?�는 State 그룹 반환"""
    spec = get_node_spec(node_name)
    if spec:
        return spec.output_state_groups
    return set()


def get_all_node_names() -> List[str]:
    """모든 ?�드 ?�름 반환"""
    return list(NODE_SPECS.keys())


def get_nodes_by_category(category: NodeCategory) -> List[NodeIOSpec]:
    """카테고리�??�드 반환"""
    return [spec for spec in NODE_SPECS.values() if spec.category == category]


# ============================================
# 검�?�??�버�?
# ============================================

def validate_workflow_flow() -> Dict[str, Any]:
    """?�체 ?�크?�로???�름 검�?""
    issues = []

    # �??�드??Input???�전 ?�드??Output�??�치?�는지 ?�인
    node_names = get_all_node_names()

    for node_name in node_names:
        spec = get_node_spec(node_name)
        if not spec:
            continue

        # Required input 체크
        for required_field in spec.required_input:
            # ?�전 ?�드?�서 ?�공?�는지 ?�인
            found = False
            for other_node in node_names:
                if other_node == node_name:
                    continue
                other_spec = get_node_spec(other_node)
                if other_spec and required_field in other_spec.output:
                    found = True
                    break

            if not found and not required_field.startswith("query"):  # query??초기 ?�력
                issues.append(f"{node_name}: ?�수 ?�력 '{required_field}'???�전 ?�드?�서 ?�공?��? ?�음")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "total_nodes": len(node_names)
    }


if __name__ == "__main__":
    # 검�??�행
    result = validate_workflow_flow()
    print(f"?�크?�로??검�?결과: {'??Valid' if result['valid'] else '??Invalid'}")
    print(f"�??�드 ?? {result['total_nodes']}")

    if result['issues']:
        print("\n문제??")
        for issue in result['issues']:
            print(f"  - {issue}")
