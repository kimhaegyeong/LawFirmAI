# -*- coding: utf-8 -*-
"""
노드 함수 래퍼
State Reduction과 Adapter를 자동으로 적용하는 데코레이터 및 헬퍼 함수
"""

import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional

from core.workflow.state.state_adapter import (
    StateAdapter,
    validate_state_for_node,
)
from core.workflow.state.state_reduction import StateReducer
from core.agents.node_input_output_spec import validate_node_input

logger = logging.getLogger(__name__)

# 전역 검색 결과 캐시 (LangGraph reducer 손실 대비)
# node_wrappers에서 저장하고, 이후 노드에서 복원
_global_search_results_cache: Optional[Dict[str, Any]] = None


def with_state_optimization(node_name: str, enable_reduction: bool = True):
    """
    State 최적화를 적용하는 데코레이터

    적용 기능:
    1. Input 검증
    2. State 자동 변환 (flat ↔ nested)
    3. State Reduction (선택적)

    Args:
        node_name: 노드 이름
        enable_reduction: State Reduction 활성화 여부

    Returns:
        데코레이터 함수
    """
    def decorator(func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Dict[str, Any]:
            # 전역 변수 선언 (try 블록 전에 먼저 선언 - Python 문법 요구사항)
            global _global_search_results_cache

            try:
                # 0. 인자 처리: 첫 번째 인자가 self인지 확인
                # 바운드 메서드의 경우 self가 이미 바인딩되어 있으므로,
                # LangGraph는 state만 전달해야 하지만, 혹시 모를 상황 대비
                if len(args) == 0:
                    logger.error(f"No arguments provided to {node_name}")
                    return {}

                # 첫 번째 인자가 dict가 아닌 경우 (self가 전달된 것으로 간주)
                if len(args) > 1:
                    # args[0]은 self, args[1]은 state
                    self_arg = args[0]
                    state = args[1]
                    rest_args = args[2:]
                else:
                    # args[0]은 state
                    state = args[0]
                    rest_args = args[1:]

                # 0-1. State가 딕셔너리인지 확인
                if not isinstance(state, dict):
                    error_msg = (
                        f"State parameter must be a dict for node {node_name}, "
                        f"got {type(state).__name__}. "
                        f"Attempting to call original function without optimization."
                    )
                    logger.error(error_msg)
                    # 원본 함수 직접 호출 (최소한의 처리)
                    if len(args) > 1:
                        return func(*args, **kwargs)
                    else:
                        return func(state, *rest_args, **kwargs)

                # 중요: 노드 실행 전에 state에 input 그룹이 있는지 확인하고 복원
                # LangGraph가 이전 노드의 결과만 전달하는 경우, input이 사라질 수 있음
                state_has_input = "input" in state and isinstance(state.get("input"), dict)
                state_has_query = state_has_input and bool(state["input"].get("query"))

                # 중요: 검색 결과 노드들에 대해 search 그룹 복원
                # execute_searches_parallel의 결과가 다음 노드에 전달되지 않을 수 있음
                search_dependent_nodes = [
                    "merge_and_rerank_with_keyword_weights",
                    "filter_and_validate_results",
                    "update_search_metadata",
                    "process_search_results_combined",
                    "prepare_document_context_for_prompt",
                    "generate_answer_enhanced"
                ]

                # 디버깅: search_dependent_nodes 체크
                is_search_dependent = node_name in search_dependent_nodes
                if is_search_dependent:
                    print(f"[DEBUG] node_wrappers ({node_name}): IS a search_dependent node")

                if is_search_dependent:
                    # 전역 캐시에서 검색 결과 복원 (노드 실행 전에 state에 추가)
                    print(f"[DEBUG] node_wrappers ({node_name}): Checking global cache - cache exists={_global_search_results_cache is not None}")
                    if _global_search_results_cache:
                        state_search = state.get("search", {}) if isinstance(state.get("search"), dict) else {}
                        has_results = len(state_search.get("semantic_results", [])) > 0 or len(state_search.get("keyword_results", [])) > 0
                        print(f"[DEBUG] node_wrappers ({node_name}): State has results={has_results}, state_search semantic={len(state_search.get('semantic_results', []))}, keyword={len(state_search.get('keyword_results', []))}")

                        if not has_results:
                            print(f"[DEBUG] node_wrappers ({node_name}): Restoring search results from global cache BEFORE function execution")
                            if "search" not in state:
                                state["search"] = {}
                            state["search"].update(_global_search_results_cache)
                            # 최상위 레벨에도 추가 (flat 구조 호환)
                            if "semantic_results" not in state:
                                state["semantic_results"] = _global_search_results_cache.get("semantic_results", [])
                            if "keyword_results" not in state:
                                state["keyword_results"] = _global_search_results_cache.get("keyword_results", [])
                            # retrieved_docs와 merged_documents도 복원 (답변 생성에 필요)
                            if "retrieved_docs" not in state or not state.get("retrieved_docs"):
                                state["retrieved_docs"] = _global_search_results_cache.get("retrieved_docs", [])
                            if "merged_documents" not in state or not state.get("merged_documents"):
                                state["merged_documents"] = _global_search_results_cache.get("merged_documents", [])
                            restored_semantic = len(state["search"].get("semantic_results", []))
                            restored_keyword = len(state["search"].get("keyword_results", []))
                            restored_docs = len(state.get("retrieved_docs", []))
                            print(f"[DEBUG] node_wrappers ({node_name}): Restored to state BEFORE execution - semantic={restored_semantic}, keyword={restored_keyword}, retrieved_docs={restored_docs}")

                    # search 그룹이 없으면 state에서 직접 찾기 (flat 구조에서)
                    if "search" not in state or not isinstance(state.get("search"), dict):
                        # flat 구조에서 semantic_results, keyword_results 찾기
                        has_search_data = any(
                            key in state for key in [
                                "semantic_results", "keyword_results", "semantic_count", "keyword_count",
                                "optimized_queries", "search_params", "merged_documents", "keyword_weights"
                            ]
                        )
                        print(f"[DEBUG] node_wrappers ({node_name}): Checking flat state for search data - has_search_data={has_search_data}, state keys={list(state.keys())[:10] if isinstance(state, dict) else 'N/A'}")
                        if has_search_data:
                            # search 그룹 생성
                            if "search" not in state:
                                state["search"] = {}
                            search_group = state["search"]
                            # flat 구조의 데이터를 search 그룹으로 복사
                            if "semantic_results" in state and not search_group.get("semantic_results"):
                                search_group["semantic_results"] = state.get("semantic_results", [])
                                print(f"[DEBUG] node_wrappers ({node_name}): Copied semantic_results from flat state: {len(search_group['semantic_results'])}")
                            if "keyword_results" in state and not search_group.get("keyword_results"):
                                search_group["keyword_results"] = state.get("keyword_results", [])
                                print(f"[DEBUG] node_wrappers ({node_name}): Copied keyword_results from flat state: {len(search_group['keyword_results'])}")
                            if "semantic_count" in state and not search_group.get("semantic_count"):
                                search_group["semantic_count"] = state.get("semantic_count", 0)
                            if "keyword_count" in state and not search_group.get("keyword_count"):
                                search_group["keyword_count"] = state.get("keyword_count", 0)
                            if "optimized_queries" in state and not search_group.get("optimized_queries"):
                                search_group["optimized_queries"] = state.get("optimized_queries", {})
                            if "search_params" in state and not search_group.get("search_params"):
                                search_group["search_params"] = state.get("search_params", {})
                            # retrieved_docs와 merged_documents도 flat state에서 복원
                            if "retrieved_docs" in state and not state.get("retrieved_docs"):
                                # 이미 최상위 레벨에 있으면 사용
                                retrieved_docs = state.get("retrieved_docs", [])
                                if retrieved_docs:
                                    search_group["retrieved_docs"] = retrieved_docs
                            if "merged_documents" in state and not state.get("merged_documents"):
                                # 이미 최상위 레벨에 있으면 사용
                                merged_docs = state.get("merged_documents", [])
                                if merged_docs:
                                    search_group["merged_documents"] = merged_docs
                            logger.info(f"Restored search group from flat state for node {node_name}")
                            print(f"[DEBUG] node_wrappers ({node_name}): Restored search group - semantic_results={len(search_group.get('semantic_results', []))}, keyword_results={len(search_group.get('keyword_results', []))}, retrieved_docs={len(search_group.get('retrieved_docs', []))}")

                # 디버깅: state 구조 확인
                if node_name in ["classify_query", "prepare_search_query", "merge_and_rerank_with_keyword_weights"]:
                    print(f"[DEBUG] node_wrappers ({node_name}): State keys={list(state.keys()) if isinstance(state, dict) else 'N/A'}")
                    print(f"[DEBUG] node_wrappers ({node_name}): state_has_input={state_has_input}, state_has_query={state_has_query}")
                    if node_name == "merge_and_rerank_with_keyword_weights":
                        search_group = state.get("search", {}) if isinstance(state.get("search"), dict) else {}
                        print(f"[DEBUG] node_wrappers ({node_name}): search group exists={bool(search_group)}, semantic_results={len(search_group.get('semantic_results', []))}, keyword_results={len(search_group.get('keyword_results', []))}")

                if not state_has_input or not state_has_query:
                    # state에 input이 없거나 query가 없으면 최상위 레벨에서 찾기
                    query_from_top = state.get("query", "")
                    session_id_from_top = state.get("session_id", "")

                    if query_from_top:
                        if "input" not in state:
                            state["input"] = {}
                        state["input"]["query"] = query_from_top
                        if session_id_from_top:
                            state["input"]["session_id"] = session_id_from_top
                        logger.info(f"Restored input group from top-level for node {node_name}: query length={len(query_from_top)}")
                        if node_name in ["classify_query", "prepare_search_query"]:
                            print(f"[DEBUG] node_wrappers ({node_name}): Restored query from top-level: '{query_from_top[:50]}...'")
                    else:
                        # 다른 그룹에서 찾기
                        found_query = None
                        if "search" in state and isinstance(state.get("search"), dict):
                            found_query = state["search"].get("search_query", "")
                        elif "classification" in state and isinstance(state.get("classification"), dict):
                            # classification에는 query가 없지만 혹시 모르니 확인
                            pass

                        if found_query:
                            if "input" not in state:
                                state["input"] = {}
                            state["input"]["query"] = found_query
                            logger.info(f"Restored input group from search.search_query for node {node_name}: query length={len(found_query)}")
                            if node_name in ["classify_query", "prepare_search_query"]:
                                print(f"[DEBUG] node_wrappers ({node_name}): Restored query from search.search_query: '{found_query[:50]}...'")
                        elif node_name in ["classify_query", "prepare_search_query"]:
                            print(f"[DEBUG] node_wrappers ({node_name}): WARNING - No query found anywhere in state!")

                # 1. Input 검증 및 자동 변환
                # 디버깅: 원본 state의 query 확인
                original_query_before = state.get("query") or (state.get("input") and isinstance(state.get("input"), dict) and state["input"].get("query"))
                if node_name == "classify_query":
                    print(f"[DEBUG] node_wrappers.classify_query: original state query='{original_query_before[:50] if original_query_before else 'EMPTY'}...'")

                is_valid, error, converted_state = validate_state_for_node(
                    state,
                    node_name,
                    auto_convert=True
                )

                # 변환 후에도 input 확인 및 복원
                if isinstance(converted_state, dict):
                    converted_has_input = "input" in converted_state and isinstance(converted_state.get("input"), dict)
                    converted_has_query = converted_has_input and bool(converted_state["input"].get("query"))

                    if not converted_has_input or not converted_has_query:
                        # state에서 다시 찾기
                        query_from_state = state.get("query") or (state.get("input") and isinstance(state.get("input"), dict) and state["input"].get("query"))
                        session_id_from_state = state.get("session_id") or (state.get("input") and isinstance(state.get("input"), dict) and state["input"].get("session_id"))

                        if query_from_state:
                            if "input" not in converted_state:
                                converted_state["input"] = {}
                            converted_state["input"]["query"] = query_from_state
                            if session_id_from_state:
                                converted_state["input"]["session_id"] = session_id_from_state
                            logger.debug(f"Restored input group after conversion for node {node_name}")

                # 디버깅: converted_state의 query 확인
                converted_query = converted_state.get("query") or (converted_state.get("input") and isinstance(converted_state.get("input"), dict) and converted_state["input"].get("query"))
                if node_name == "classify_query":
                    print(f"[DEBUG] node_wrappers.classify_query: converted_state query='{converted_query[:50] if converted_query else 'EMPTY'}...'")

                # 개선 사항 1: Input Validation - 실패 시 자동 복구 로직 추가
                if not is_valid:
                    # 경고를 debug 레벨로 낮춤 (자동 복구가 있으므로)
                    logger.debug(f"Input validation failed for {node_name}: {error} (attempting auto-recovery)")
                    
                    # 개선 사항 1: retrieved_docs 필드 누락 시 자동 복구
                    if error and "retrieved_docs" in error:
                        try:
                            # Global cache에서 복구 시도 (전역 변수는 이미 wrapper 함수 시작 부분에서 선언됨)
                            if _global_search_results_cache:
                                retrieved_docs = (
                                    _global_search_results_cache.get("retrieved_docs", []) or
                                    (_global_search_results_cache.get("search") and isinstance(_global_search_results_cache["search"], dict) and 
                                     _global_search_results_cache["search"].get("retrieved_docs", [])) or
                                    []
                                )
                                if retrieved_docs:
                                    # state에 복구된 retrieved_docs 저장
                                    converted_state["retrieved_docs"] = retrieved_docs
                                    if "search" not in converted_state:
                                        converted_state["search"] = {}
                                    converted_state["search"]["retrieved_docs"] = retrieved_docs
                                    if "common" not in converted_state:
                                        converted_state["common"] = {}
                                    if "search" not in converted_state["common"]:
                                        converted_state["common"]["search"] = {}
                                    converted_state["common"]["search"]["retrieved_docs"] = retrieved_docs
                                    logger.info(f"✅ [AUTO-RECOVER] Restored {len(retrieved_docs)} retrieved_docs from global cache for {node_name}")
                                    # 재검증
                                    is_valid, error = validate_node_input(node_name, converted_state)
                                    if is_valid:
                                        logger.info(f"✅ [AUTO-RECOVER] Input validation passed after auto-recovery for {node_name}")
                                else:
                                    # 빈 리스트로라도 저장하여 validation 통과
                                    converted_state["retrieved_docs"] = []
                                    if "search" not in converted_state:
                                        converted_state["search"] = {}
                                    converted_state["search"]["retrieved_docs"] = []
                                    logger.warning(f"⚠️ [AUTO-RECOVER] No retrieved_docs found, using empty list for {node_name}")
                                    # 재검증
                                    is_valid, error = validate_node_input(node_name, converted_state)
                        except Exception as e:
                            logger.debug(f"Auto-recovery failed for {node_name}: {e}")
                    
                    # 개선 사항 2: query_type 필드 누락 시 자동 복구
                    if error and ("query_type" in error or "query_type" in str(error)):
                        try:
                            query_type = None
                            # 1. converted_state에서 직접 확인 (stream_mode="updates" 사용 시)
                            if isinstance(converted_state, dict):
                                # classification 그룹에서 확인 (우선순위 1)
                                if "classification" in converted_state and isinstance(converted_state["classification"], dict):
                                    query_type = converted_state["classification"].get("query_type")
                                # 최상위 레벨에서 확인 (우선순위 2)
                                if not query_type:
                                    query_type = converted_state.get("query_type")
                                # metadata에서 확인 (우선순위 3)
                                if not query_type and "metadata" in converted_state and isinstance(converted_state["metadata"], dict):
                                    query_type = converted_state["metadata"].get("query_type")
                                # analysis에서 확인 (우선순위 4)
                                if not query_type and "analysis" in converted_state and isinstance(converted_state["analysis"], dict):
                                    query_type = converted_state["analysis"].get("query_type")
                                # common 그룹에서 확인 (우선순위 5)
                                if not query_type and "common" in converted_state and isinstance(converted_state["common"], dict):
                                    common_classification = converted_state["common"].get("classification", {})
                                    if isinstance(common_classification, dict):
                                        query_type = common_classification.get("query_type")
                            
                            # 2. Global cache에서 복구 시도 (없는 경우)
                            if not query_type and _global_search_results_cache:
                                query_type = (
                                    _global_search_results_cache.get("common", {}).get("classification", {}).get("query_type", "") or
                                    _global_search_results_cache.get("metadata", {}).get("query_type", "") or
                                    _global_search_results_cache.get("analysis", {}).get("query_type", "") or
                                    _global_search_results_cache.get("classification", {}).get("query_type", "") or
                                    _global_search_results_cache.get("query_type", "") or
                                    ""
                                ) or None
                            
                            # 3. 기본값 설정 (마지막 수단)
                            if not query_type:
                                query_type = "simple_question"  # direct_answer 노드의 기본값과 일치
                                logger.warning(f"⚠️ [AUTO-RECOVER] query_type not found, using default: {query_type} for {node_name}")
                            
                            if query_type:
                                # state에 복구된 query_type 저장 (모든 위치에)
                                converted_state["query_type"] = query_type
                                # classification 그룹에 저장 (필수)
                                if "classification" not in converted_state:
                                    converted_state["classification"] = {}
                                converted_state["classification"]["query_type"] = query_type
                                # metadata에도 저장
                                if "metadata" not in converted_state:
                                    converted_state["metadata"] = {}
                                converted_state["metadata"]["query_type"] = query_type
                                logger.info(f"✅ [AUTO-RECOVER] Restored query_type={query_type} for {node_name}")
                                # 재검증
                                is_valid, error = validate_node_input(node_name, converted_state)
                                if is_valid:
                                    logger.info(f"✅ [AUTO-RECOVER] Input validation passed after auto-recovery for {node_name}")
                        except Exception as e:
                            logger.debug(f"Auto-recovery failed for {node_name}: {e}")

                # 2. State Reduction (활성화된 경우)
                if enable_reduction:
                    reducer = StateReducer(aggressive_reduction=True)
                    working_state = reducer.reduce_state_for_node(converted_state, node_name)

                    # Reduction 결과가 비어있으면 원본 사용
                    if not working_state:
                        logger.warning(f"State reduction returned empty dict for {node_name}, using converted_state")
                        working_state = converted_state

                    # 상태 크기 로깅
                    if logger.isEnabledFor(logging.DEBUG):
                        original_size = _estimate_state_size(state)
                        reduced_size = _estimate_state_size(working_state)
                        reduction_pct = (1 - reduced_size / original_size) * 100 if original_size > 0 else 0
                        logger.debug(
                            f"State reduction for {node_name}: "
                            f"{reduction_pct:.1f}% reduction "
                            f"({original_size:.0f} → {reduced_size:.0f} bytes)"
                        )
                else:
                    working_state = converted_state

                # 중요: state_reduction 후에도 query_type과 retrieved_docs 보존
                # State reduction으로 인한 손실 방지
                preserved_query_type = (
                    converted_state.get("query_type") or
                    (converted_state.get("metadata", {}).get("query_type") if isinstance(converted_state.get("metadata"), dict) else None) or
                    (converted_state.get("common", {}).get("classification", {}).get("query_type") if isinstance(converted_state.get("common"), dict) and isinstance(converted_state["common"].get("classification"), dict) else None) or
                    (converted_state.get("classification", {}).get("query_type") if isinstance(converted_state.get("classification"), dict) else None)
                )
                preserved_retrieved_docs = (
                    converted_state.get("retrieved_docs") or
                    (converted_state.get("search", {}).get("retrieved_docs") if isinstance(converted_state.get("search"), dict) else None) or
                    (converted_state.get("common", {}).get("search", {}).get("retrieved_docs") if isinstance(converted_state.get("common"), dict) and isinstance(converted_state["common"].get("search"), dict) else None) or
                    (converted_state.get("metadata", {}).get("retrieved_docs") if isinstance(converted_state.get("metadata"), dict) else None)
                )
                
                # working_state에 보존된 필드 복원
                if preserved_query_type and not working_state.get("query_type"):
                    working_state["query_type"] = preserved_query_type
                    if "metadata" not in working_state:
                        working_state["metadata"] = {}
                    if not isinstance(working_state["metadata"], dict):
                        working_state["metadata"] = {}
                    working_state["metadata"]["query_type"] = preserved_query_type
                    if "common" not in working_state:
                        working_state["common"] = {}
                    if not isinstance(working_state["common"], dict):
                        working_state["common"] = {}
                    if "classification" not in working_state["common"]:
                        working_state["common"]["classification"] = {}
                    working_state["common"]["classification"]["query_type"] = preserved_query_type
                    logger.debug(f"Preserved query_type={preserved_query_type} for node {node_name} after reduction")
                
                if preserved_retrieved_docs and not working_state.get("retrieved_docs"):
                    working_state["retrieved_docs"] = preserved_retrieved_docs
                    if "search" not in working_state:
                        working_state["search"] = {}
                    if not isinstance(working_state["search"], dict):
                        working_state["search"] = {}
                    working_state["search"]["retrieved_docs"] = preserved_retrieved_docs
                    if "common" not in working_state:
                        working_state["common"] = {}
                    if not isinstance(working_state["common"], dict):
                        working_state["common"] = {}
                    if "search" not in working_state["common"]:
                        working_state["common"]["search"] = {}
                    working_state["common"]["search"]["retrieved_docs"] = preserved_retrieved_docs
                    logger.debug(f"Preserved {len(preserved_retrieved_docs)} retrieved_docs for node {node_name} after reduction")
                
                # 중요: state_reduction 후에도 전역 캐시에서 검색 결과 복원
                # search_dependent 노드들에 대해 working_state에 검색 결과 추가
                # is_search_dependent 변수는 위에서 정의됨
                if is_search_dependent:
                    print(f"[DEBUG] node_wrappers ({node_name}): After reduction - Checking global cache - cache exists={_global_search_results_cache is not None}")
                    if _global_search_results_cache:
                        working_search = working_state.get("search", {}) if isinstance(working_state.get("search"), dict) else {}
                        has_results = len(working_search.get("semantic_results", [])) > 0 or len(working_search.get("keyword_results", [])) > 0
                        print(f"[DEBUG] node_wrappers ({node_name}): After reduction - working_state has results={has_results}, semantic={len(working_search.get('semantic_results', []))}, keyword={len(working_search.get('keyword_results', []))}")

                        if not has_results:
                            print(f"[DEBUG] node_wrappers ({node_name}): Restoring search results to working_state AFTER reduction")
                            if "search" not in working_state:
                                working_state["search"] = {}
                            working_state["search"].update(_global_search_results_cache)
                            # 최상위 레벨에도 추가 (flat 구조 호환)
                            if "semantic_results" not in working_state:
                                working_state["semantic_results"] = _global_search_results_cache.get("semantic_results", [])
                            if "keyword_results" not in working_state:
                                working_state["keyword_results"] = _global_search_results_cache.get("keyword_results", [])
                            restored_semantic = len(working_state["search"].get("semantic_results", []))
                            restored_keyword = len(working_state["search"].get("keyword_results", []))
                            print(f"[DEBUG] node_wrappers ({node_name}): Restored to working_state AFTER reduction - semantic={restored_semantic}, keyword={restored_keyword}")

                # 3. 원본 함수 호출
                # 중요: working_state에 query가 있는지 확인하고 없으면 원본 state에서 복원
                # converted_state와 원본 state 모두 확인
                original_query = None
                original_session_id = None

                # 원본 state에서 query 찾기 (여러 위치 확인)
                if isinstance(state, dict):
                    original_query = state.get("query") or (state.get("input") and isinstance(state.get("input"), dict) and state["input"].get("query"))
                    original_session_id = state.get("session_id") or (state.get("input") and isinstance(state.get("input"), dict) and state["input"].get("session_id"))

                # converted_state에서도 확인
                if not original_query and isinstance(converted_state, dict):
                    original_query = converted_state.get("query") or (converted_state.get("input") and isinstance(converted_state.get("input"), dict) and converted_state["input"].get("query"))
                    original_session_id = converted_state.get("session_id") or (converted_state.get("input") and isinstance(converted_state.get("input"), dict) and converted_state["input"].get("session_id"))

                # working_state에 query가 없으면 복원 (state reduction 후에도 보장)
                if "input" not in working_state or not working_state.get("input") or not working_state["input"].get("query"):
                    # 원본 state에서 query 찾기
                    if original_query:
                        if "input" not in working_state:
                            working_state["input"] = {}
                        working_state["input"]["query"] = original_query
                        if original_session_id:
                            working_state["input"]["session_id"] = original_session_id
                        logger.info(f"Restored query in working_state for node {node_name}: '{original_query[:50]}...'")
                    # converted_state에서도 확인
                    elif isinstance(converted_state, dict):
                        converted_query = converted_state.get("query") or (converted_state.get("input") and isinstance(converted_state.get("input"), dict) and converted_state["input"].get("query"))
                        converted_session_id = converted_state.get("session_id") or (converted_state.get("input") and isinstance(converted_state.get("input"), dict) and converted_state["input"].get("session_id"))

                        if converted_query:
                            if "input" not in working_state:
                                working_state["input"] = {}
                            working_state["input"]["query"] = converted_query
                            if converted_session_id:
                                working_state["input"]["session_id"] = converted_session_id
                            logger.info(f"Restored query from converted_state in working_state for node {node_name}: '{converted_query[:50]}...'")
                        else:
                            logger.warning(f"No query found in state for node {node_name}, state keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")
                    else:
                        logger.warning(f"No query found in state for node {node_name}, state keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")

                # 디버깅: working_state의 query 확인
                if node_name in ["prepare_search_query"]:
                    working_query = working_state.get("query") or (working_state.get("input") and isinstance(working_state.get("input"), dict) and working_state["input"].get("query", ""))
                    print(f"[DEBUG] node_wrappers ({node_name}): working_state before function call - query='{working_query[:50] if working_query else 'EMPTY'}...'")
                    print(f"[DEBUG] node_wrappers ({node_name}): working_state keys={list(working_state.keys()) if isinstance(working_state, dict) else 'N/A'}")

                if len(args) > 1:
                    # self가 있는 경우
                    result = func(args[0], working_state, *rest_args, **kwargs)
                else:
                    # self가 없는 경우
                    result = func(working_state, *rest_args, **kwargs)

                # 디버깅: result의 query 확인
                if node_name in ["prepare_search_query"]:
                    result_query = result.get("query") if isinstance(result, dict) else None
                    result_input_query = result.get("input", {}).get("query", "") if isinstance(result, dict) and result.get("input") else None
                    print(f"[DEBUG] node_wrappers ({node_name}): result after function call - query='{result_query[:50] if result_query else 'N/A'}...', input.query='{result_input_query[:50] if result_input_query else 'N/A'}...'")

                # 4. 결과를 원본 State에 병합
                # 중요: result에 query가 없으면 원본 state의 query를 보존
                if isinstance(result, dict) and isinstance(state, dict):
                    # result의 input에 query가 없으면 원본 state의 query 복원
                    result_has_query = False
                    if "input" in result and isinstance(result.get("input"), dict):
                        result_has_query = bool(result["input"].get("query"))
                    elif "query" in result:
                        result_has_query = bool(result.get("query"))

                    # working_state에서도 query 확인 (reduction 후에도 query가 있을 수 있음)
                    if not result_has_query and isinstance(working_state, dict):
                        working_query = working_state.get("query") or (working_state.get("input") and isinstance(working_state.get("input"), dict) and working_state["input"].get("query"))
                        if working_query:
                            result_has_query = True
                            if "input" not in result:
                                result["input"] = {}
                            if not isinstance(result["input"], dict):
                                result["input"] = {}
                            result["input"]["query"] = working_query
                            working_session_id = working_state.get("session_id") or (working_state.get("input") and isinstance(working_state.get("input"), dict) and working_state["input"].get("session_id"))
                            if working_session_id:
                                result["input"]["session_id"] = working_session_id
                            logger.debug(f"Preserved query from working_state for node {node_name}")

                    if not result_has_query:
                        # 원본 state에서 query 복원
                        original_query = state.get("query") or (state.get("input") and isinstance(state.get("input"), dict) and state["input"].get("query"))
                        original_session_id = state.get("session_id") or (state.get("input") and isinstance(state.get("input"), dict) and state["input"].get("session_id"))

                        if original_query:
                            if "input" not in result:
                                result["input"] = {}
                            if not isinstance(result["input"], dict):
                                result["input"] = {}
                            result["input"]["query"] = original_query
                            if original_session_id:
                                result["input"]["session_id"] = original_session_id
                            logger.info(f"Preserved query in result for node {node_name}: '{original_query[:50]}...'")

                # 중요: result에 input 그룹이 있으면 모든 필수 필드를 포함해야 함
                # LangGraph는 TypedDict의 각 필드를 병합하는데, input이 없으면 이전 값이 사라질 수 있음
                # 따라서 항상 result에 input 그룹을 포함시켜야 함
                if isinstance(result, dict) and isinstance(state, dict):
                    # 항상 input 그룹 보존 보장
                    input_to_preserve = None

                    # 1. state에서 input 찾기
                    if "input" in state and isinstance(state.get("input"), dict):
                        input_to_preserve = state["input"].copy()
                    elif "query" in state or "session_id" in state:
                        input_to_preserve = {
                            "query": state.get("query", ""),
                            "session_id": state.get("session_id", "")
                        }

                    # 2. result에서 input 찾기
                    result_has_input = "input" in result and isinstance(result.get("input"), dict)
                    result_has_query = result_has_input and bool(result["input"].get("query"))

                    # 3. result에 input이 없거나 query가 없으면 state의 input 보존
                    if not result_has_input or not result_has_query:
                        if input_to_preserve:
                            if "input" not in result:
                                result["input"] = {}
                            result["input"]["query"] = input_to_preserve.get("query", result.get("input", {}).get("query", ""))
                            result["input"]["session_id"] = input_to_preserve.get("session_id", result.get("input", {}).get("session_id", ""))
                            logger.debug(f"Preserved input group from state for node {node_name}")

                    # 4. result에 input이 있어도 query가 없으면 state의 input 보존
                    elif result_has_input and not result_has_query:
                        if input_to_preserve and input_to_preserve.get("query"):
                            result["input"]["query"] = input_to_preserve["query"]
                            if input_to_preserve.get("session_id"):
                                result["input"]["session_id"] = input_to_preserve["session_id"]
                            logger.debug(f"Restored query from state input for node {node_name}")

                    # 중요: execute_searches_parallel 이후 노드들에 대해 전역 캐시에서 복원 (result에만 적용)
                    # 참고: 노드 실행 전 state 복원은 위쪽에서 이미 처리됨
                    if node_name in ["merge_and_rerank_with_keyword_weights", "filter_and_validate_results", "update_search_metadata", "prepare_document_context_for_prompt"]:
                        if _global_search_results_cache and isinstance(result, dict):
                            # result에 search 그룹이 없거나 비어있으면 캐시에서 복원
                            result_search = result.get("search", {}) if isinstance(result.get("search"), dict) else {}
                            has_results = len(result_search.get("semantic_results", [])) > 0 or len(result_search.get("keyword_results", [])) > 0

                            if not has_results:
                                print(f"[DEBUG] node_wrappers ({node_name}): Restoring search results from global cache")
                                if "search" not in result:
                                    result["search"] = {}
                                result["search"].update(_global_search_results_cache)
                                # 최상위 레벨에도 추가
                                if "semantic_results" not in result:
                                    result["semantic_results"] = _global_search_results_cache.get("semantic_results", [])
                                if "keyword_results" not in result:
                                    result["keyword_results"] = _global_search_results_cache.get("keyword_results", [])
                                restored_semantic = len(result["search"].get("semantic_results", []))
                                restored_keyword = len(result["search"].get("keyword_results", []))
                                print(f"[DEBUG] node_wrappers ({node_name}): Restored from cache - semantic={restored_semantic}, keyword={restored_keyword}")

                    # 중요: execute_searches_parallel의 경우 search 그룹 보존
                    # LangGraph는 TypedDict를 병합할 때 SearchState에 없는 필드가 손실될 수 있음
                    # 따라서 result에 search 그룹이 있으면 항상 보존
                    if node_name == "execute_searches_parallel":
                        result_search = result.get("search") if isinstance(result.get("search"), dict) else {}
                        state_search = state.get("search") if isinstance(state.get("search"), dict) else {}

                        # result에 search 그룹이 있으면 확인 및 로깅
                        if result_search:
                            semantic_count = len(result_search.get("semantic_results", []))
                            keyword_count = len(result_search.get("keyword_results", []))
                            print(f"[DEBUG] node_wrappers ({node_name}): result has search group - semantic_results={semantic_count}, keyword_results={keyword_count}")
                            # result에 명시적으로 보존 (LangGraph 병합 보장)
                            if "search" not in result or not isinstance(result.get("search"), dict):
                                result["search"] = {}
                            result["search"]["semantic_results"] = result_search.get("semantic_results", [])
                            result["search"]["keyword_results"] = result_search.get("keyword_results", [])
                            result["search"]["semantic_count"] = result_search.get("semantic_count", semantic_count)
                            result["search"]["keyword_count"] = result_search.get("keyword_count", keyword_count)
                        elif state_search:
                            # state에 search 그룹이 있으면 result에도 복사
                            print(f"[DEBUG] node_wrappers ({node_name}): Copying search group from state to result")
                            result["search"] = state_search.copy()

                    # processing_steps 전역 캐시에 저장 (state reduction 손실 방지)
                    if isinstance(result, dict):
                        # common 그룹에서 processing_steps 확인
                        result_common = result.get("common", {})
                        if isinstance(result_common, dict):
                            result_steps = result_common.get("processing_steps", [])
                            if isinstance(result_steps, list) and len(result_steps) > 0:
                                # 전역 캐시에 저장
                                if not _global_search_results_cache:
                                    _global_search_results_cache = {}
                                if "processing_steps" not in _global_search_results_cache:
                                    _global_search_results_cache["processing_steps"] = []
                                # 기존 steps와 병합 (중복 제거)
                                for step in result_steps:
                                    if isinstance(step, str) and step not in _global_search_results_cache["processing_steps"]:
                                        _global_search_results_cache["processing_steps"].append(step)

                        # 최상위 레벨에서도 확인
                        result_top_steps = result.get("processing_steps", [])
                        if isinstance(result_top_steps, list) and len(result_top_steps) > 0:
                            # 전역 캐시에 저장
                            if not _global_search_results_cache:
                                _global_search_results_cache = {}
                            if "processing_steps" not in _global_search_results_cache:
                                _global_search_results_cache["processing_steps"] = []
                            # 기존 steps와 병합 (중복 제거)
                            for step in result_top_steps:
                                if isinstance(step, str) and step not in _global_search_results_cache["processing_steps"]:
                                    _global_search_results_cache["processing_steps"].append(step)

                    # 5. Nested 구조면 그대로 반환, Flat 구조면 병합
                    # 중요: LangGraph reducer가 TypedDict 필드만 보존하므로,
                    # result에 모든 필수 그룹을 명시적으로 포함시켜야 함
                    # 특히 execute_searches_parallel의 경우 search 그룹이 반드시 포함되어야 함
                    if node_name == "execute_searches_parallel":
                        # result에 search 그룹이 없으면 state에서 복사
                        if "search" not in result or not isinstance(result.get("search"), dict):
                            if "search" in state and isinstance(state.get("search"), dict):
                                result["search"] = state["search"].copy()
                                print(f"[DEBUG] node_wrappers ({node_name}): Copied search group from state to result before return")

                        # result의 search 그룹에 모든 필수 필드 포함 확인
                        if "search" in result and isinstance(result.get("search"), dict):
                            result_search = result["search"]
                            # semantic_results와 keyword_results가 있으면 반드시 보존
                            semantic_results = result_search.get("semantic_results", [])
                            keyword_results = result_search.get("keyword_results", [])
                            if semantic_results or keyword_results:
                                semantic_count = len(semantic_results)
                                keyword_count = len(keyword_results)

                                # 전역 캐시에 저장 (LangGraph reducer 손실 대비)
                                # global 선언은 wrapper 함수 시작 부분에 이미 있음
                                # 중요: query_complexity와 needs_search 보존 (classify_complexity에서 저장한 값)
                                preserved_complexity = _global_search_results_cache.get("query_complexity") if _global_search_results_cache else None
                                preserved_needs_search = _global_search_results_cache.get("needs_search") if _global_search_results_cache else None

                                if not _global_search_results_cache:
                                    _global_search_results_cache = {}
                                _global_search_results_cache.update({
                                    "semantic_results": semantic_results,
                                    "keyword_results": keyword_results,
                                    "semantic_count": semantic_count,
                                    "keyword_count": keyword_count
                                })

                                # 보존된 query_complexity 복원
                                if preserved_complexity:
                                    _global_search_results_cache["query_complexity"] = preserved_complexity
                                    if preserved_needs_search is not None:
                                        _global_search_results_cache["needs_search"] = preserved_needs_search

                                print(f"[DEBUG] node_wrappers ({node_name}): Saved to global cache - semantic={semantic_count}, keyword={keyword_count}, complexity={preserved_complexity}")

                    # 중요: merge_and_rerank_with_keyword_weights의 경우 retrieved_docs 캐시 보존
                    if node_name == "merge_and_rerank_with_keyword_weights":
                        result_search = result.get("search") if isinstance(result.get("search"), dict) else {}
                        retrieved_docs = result_search.get("retrieved_docs", [])
                        merged_documents = result_search.get("merged_documents", [])

                        # 최상위 레벨에서도 확인
                        top_retrieved_docs = result.get("retrieved_docs", [])
                        top_merged_docs = result.get("merged_documents", [])

                        print(f"[DEBUG] node_wrappers ({node_name}): result - search_group retrieved_docs={len(retrieved_docs) if isinstance(retrieved_docs, list) else 0}, merged_documents={len(merged_documents) if isinstance(merged_documents, list) else 0}, top_retrieved_docs={len(top_retrieved_docs) if isinstance(top_retrieved_docs, list) else 0}, top_merged_docs={len(top_merged_docs) if isinstance(top_merged_docs, list) else 0}")
                        print(f"[DEBUG] node_wrappers ({node_name}): result - retrieved_docs type={type(retrieved_docs).__name__}, is_list={isinstance(retrieved_docs, list)}, has_length={len(retrieved_docs) if isinstance(retrieved_docs, list) else 'N/A'}")

                        # retrieved_docs 또는 merged_documents가 있으면 전역 캐시에 저장
                        final_retrieved_docs = (retrieved_docs if isinstance(retrieved_docs, list) and len(retrieved_docs) > 0 else
                                               top_retrieved_docs if isinstance(top_retrieved_docs, list) and len(top_retrieved_docs) > 0 else
                                               merged_documents if isinstance(merged_documents, list) and len(merged_documents) > 0 else
                                               top_merged_docs if isinstance(top_merged_docs, list) and len(top_merged_docs) > 0 else [])

                        print(f"[DEBUG] node_wrappers ({node_name}): final_retrieved_docs={len(final_retrieved_docs) if isinstance(final_retrieved_docs, list) else 0}, type={type(final_retrieved_docs).__name__}, is_list={isinstance(final_retrieved_docs, list)}, has_length={len(final_retrieved_docs) if isinstance(final_retrieved_docs, list) else 'N/A'}")

                        # 개선 2.1: process_search_results_combined 실행 후 retrieved_docs 전역 캐시 저장 확인
                        if node_name == "process_search_results_combined":
                            print(f"[DEBUG] node_wrappers ({node_name}): process_search_results_combined 실행 완료 - result 구조 분석 중...", flush=True)

                            # result 전체 구조 출력
                            if isinstance(result, dict):
                                result_keys = list(result.keys())
                                print(f"[DEBUG] node_wrappers ({node_name}): result keys: {result_keys}", flush=True)

                                # retrieved_docs 찾기 시도 (모든 가능한 경로 확인)
                                possible_paths = {
                                    "top_level": result.get("retrieved_docs"),
                                    "search_group": result.get("search", {}).get("retrieved_docs") if isinstance(result.get("search"), dict) else None,
                                    "common_group": result.get("common", {}).get("search", {}).get("retrieved_docs") if isinstance(result.get("common"), dict) and isinstance(result.get("common").get("search"), dict) else None,
                                    "input_group": result.get("input", {}).get("retrieved_docs") if isinstance(result.get("input"), dict) else None,
                                    "merged_documents_top": result.get("merged_documents"),
                                    "merged_documents_search": result.get("search", {}).get("merged_documents") if isinstance(result.get("search"), dict) else None,
                                }

                                found_path = None
                                found_docs = None
                                for path_name, docs in possible_paths.items():
                                    if docs and isinstance(docs, list) and len(docs) > 0:
                                        found_path = path_name
                                        found_docs = docs
                                        print(f"[DEBUG] node_wrappers ({node_name}): ✅ retrieved_docs를 {path_name}에서 찾음 - 개수: {len(docs)}", flush=True)
                                        break

                                if found_docs:
                                    final_retrieved_docs = found_docs
                                else:
                                    print(f"[DEBUG] node_wrappers ({node_name}): ❌ retrieved_docs를 찾을 수 없음 - 모든 경로 확인 완료", flush=True)
                                    # 각 경로의 상세 정보 출력
                                    for path_name, docs in possible_paths.items():
                                        docs_type = type(docs).__name__
                                        docs_len = len(docs) if isinstance(docs, list) else 'N/A'
                                        docs_sample = docs[:1] if isinstance(docs, list) and len(docs) > 0 else None
                                        print(f"[DEBUG] node_wrappers ({node_name}):   - {path_name}: type={docs_type}, len={docs_len}, sample={docs_sample}", flush=True)
                            else:
                                print(f"[DEBUG] node_wrappers ({node_name}): ⚠️ result가 dict가 아님 - type: {type(result).__name__}", flush=True)

                            # final_retrieved_docs 현재 상태 확인
                            print(f"[DEBUG] node_wrappers ({node_name}): final_retrieved_docs={len(final_retrieved_docs) if isinstance(final_retrieved_docs, list) else 0}, type={type(final_retrieved_docs).__name__}", flush=True)

                        if isinstance(final_retrieved_docs, list) and len(final_retrieved_docs) > 0:
                            # global 선언은 wrapper 함수 시작 부분에 이미 있음
                            # 전역 캐시 초기화 (없으면 생성)
                            if not _global_search_results_cache:
                                _global_search_results_cache = {}
                            # 중요: query_complexity와 needs_search 보존 (초기화되지 않은 경우에만)
                            # 이미 존재하면 덮어쓰지 않음

                            # retrieved_docs와 merged_documents를 전역 캐시에 저장
                            _global_search_results_cache["retrieved_docs"] = final_retrieved_docs
                            _global_search_results_cache["merged_documents"] = final_retrieved_docs

                            # search 그룹 전체를 전역 캐시에 저장
                            if result_search:
                                if "search" not in _global_search_results_cache:
                                    _global_search_results_cache["search"] = {}
                                _global_search_results_cache["search"].update(result_search)
                                # retrieved_docs와 merged_documents도 search 그룹에 포함
                                _global_search_results_cache["search"]["retrieved_docs"] = final_retrieved_docs
                                _global_search_results_cache["search"]["merged_documents"] = final_retrieved_docs
                            else:
                                # search 그룹이 없으면 생성하여 저장
                                if "search" not in _global_search_results_cache:
                                    _global_search_results_cache["search"] = {}
                                _global_search_results_cache["search"]["retrieved_docs"] = final_retrieved_docs
                                _global_search_results_cache["search"]["merged_documents"] = final_retrieved_docs

                            print(f"[DEBUG] node_wrappers ({node_name}): ✅ Saved retrieved_docs to global cache - count={len(final_retrieved_docs)}, cache has search group={bool(_global_search_results_cache.get('search'))}")
                            # 개선 3: 저장 후 검증
                            cached_count = len(_global_search_results_cache.get("retrieved_docs", []))
                            cached_search_count = len(_global_search_results_cache.get("search", {}).get("retrieved_docs", []))
                            print(f"[DEBUG] node_wrappers ({node_name}): 전역 캐시 검증 - 최상위: {cached_count}, search 그룹: {cached_search_count}")
                        else:
                            print(f"[DEBUG] node_wrappers ({node_name}): ⚠️ WARNING - result has no retrieved_docs or merged_documents in search group or top level")
                            if node_name == "process_search_results_combined":
                                print(f"[DEBUG] node_wrappers ({node_name}): ❌ process_search_results_combined에서 retrieved_docs가 저장되지 않았습니다!")

                    # execute_searches_parallel의 result 처리 (이전 코드 유지)
                    if node_name == "execute_searches_parallel":
                        if "search" in result and isinstance(result.get("search"), dict):
                            result_search = result["search"]
                            semantic_results = result_search.get("semantic_results", [])
                            keyword_results = result_search.get("keyword_results", [])
                            if semantic_results or keyword_results:
                                semantic_count = len(semantic_results)
                                keyword_count = len(keyword_results)
                                print(f"[DEBUG] node_wrappers ({node_name}): Result has search group with data before return - semantic={semantic_count}, keyword={keyword_count}")
                                print(f"[DEBUG] node_wrappers ({node_name}): Result keys before return: {list(result.keys())}")

                                # LangGraph reducer 손실 대비: result의 모든 필드를 명시적으로 포함
                                # 특히 search 그룹의 모든 필드를 최상위 레벨에도 포함 (Flat 구조 호환)
                                if not isinstance(result.get("semantic_results"), list):
                                    result["semantic_results"] = semantic_results
                                if not isinstance(result.get("keyword_results"), list):
                                    result["keyword_results"] = keyword_results
                                if "semantic_count" not in result:
                                    result["semantic_count"] = semantic_count
                                if "keyword_count" not in result:
                                    result["keyword_count"] = keyword_count
                                print(f"[DEBUG] node_wrappers ({node_name}): Added search fields to top level - semantic_results={len(result.get('semantic_results', []))}, keyword_results={len(result.get('keyword_results', []))}")

                                # 전역 캐시에 저장 (LangGraph reducer 손실 대비)
                                # 중요: query_complexity와 needs_search 보존 (classify_complexity에서 저장한 값)
                                preserved_complexity = _global_search_results_cache.get("query_complexity") if _global_search_results_cache else None
                                preserved_needs_search = _global_search_results_cache.get("needs_search") if _global_search_results_cache else None

                                _global_search_results_cache = result_search.copy()

                                # 보존된 query_complexity 복원
                                if preserved_complexity:
                                    _global_search_results_cache["query_complexity"] = preserved_complexity
                                    if preserved_needs_search is not None:
                                        _global_search_results_cache["needs_search"] = preserved_needs_search

                                print(f"[DEBUG] node_wrappers ({node_name}): Saved to global cache - semantic={semantic_count}, keyword={keyword_count}, complexity={preserved_complexity}")

                    if "input" in state and isinstance(state.get("input"), dict):
                        # Nested 구조면 그대로 반환하되, 모든 필수 그룹 포함 확인
                        # LangGraph reducer가 전체 result를 병합하므로, 모든 그룹을 포함해야 함
                        if "input" not in result:
                            result["input"] = state["input"].copy()
                        return result
                    else:
                        # Flat 구조면 병합
                        state.update(result)
                        return state

                return result

            except Exception as e:
                logger.error(f"Error in state optimization wrapper for {node_name}: {e}", exc_info=True)
                # 에러 발생 시 원본 함수 실행 (최소한의 처리)
                try:
                    return func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback function call also failed for {node_name}: {fallback_error}",
                        exc_info=True
                    )
                    # 마지막 수단: 빈 딕셔너리 반환
                    return {}

        return wrapper
    return decorator


def _estimate_state_size(state: Dict[str, Any]) -> int:
    """State 크기 추정"""
    import sys
    try:
        return sys.getsizeof(str(state))
    except:
        return len(str(state))


def with_input_validation(node_name: str):
    """
    Input 검증만 적용하는 데코레이터

    State Reduction 없이 Input 검증만 수행
    """
    def decorator(func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Dict[str, Any]:
            try:
                # 0. 인자 처리: 첫 번째 인자가 self인지 확인
                if len(args) == 0:
                    logger.error(f"No arguments provided to {node_name}")
                    return {}

                # 첫 번째 인자가 dict가 아닌 경우 (self가 전달된 것으로 간주)
                if len(args) > 1:
                    # args[0]은 self, args[1]은 state
                    self_arg = args[0]
                    state = args[1]
                    rest_args = args[2:]
                else:
                    # args[0]은 state
                    state = args[0]
                    rest_args = args[1:]

                # 0-1. State가 딕셔너리인지 확인
                if not isinstance(state, dict):
                    error_msg = (
                        f"State parameter must be a dict for node {node_name}, "
                        f"got {type(state).__name__}. "
                        f"Attempting to call original function without validation."
                    )
                    logger.error(error_msg)
                    return func(*args, **kwargs)

                # Input 검증 및 자동 변환
                is_valid, error, converted_state = validate_state_for_node(
                    state,
                    node_name,
                    auto_convert=True
                )

                if not is_valid:
                    # 경고를 debug 레벨로 낮춤 (자동 복구 로직이 있으므로)
                    logger.debug(f"Input validation failed for {node_name}: {error} (continuing with converted state)")

                # 원본 함수 호출
                if len(args) > 1:
                    # self가 있는 경우
                    result = func(args[0], converted_state, *rest_args, **kwargs)
                else:
                    # self가 없는 경우
                    result = func(converted_state, *rest_args, **kwargs)

                # 결과 반환
                return result

            except Exception as e:
                logger.error(f"Error in input validation wrapper for {node_name}: {e}", exc_info=True)
                try:
                    return func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback function call also failed for {node_name}: {fallback_error}",
                        exc_info=True
                    )
                    return {}

        return wrapper
    return decorator


def adapt_state_if_needed(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    State를 필요시 자동 변환 (편의 함수)

    Args:
        state: State 객체 (flat 또는 nested)

    Returns:
        변환된 State 객체
    """
    return StateAdapter.to_nested(state)


def flatten_state_if_needed(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    State를 Flat 구조로 변환 (편의 함수)

    Args:
        state: State 객체 (nested)

    Returns:
        Flat 구조의 State
    """
    return StateAdapter.to_flat(state)
