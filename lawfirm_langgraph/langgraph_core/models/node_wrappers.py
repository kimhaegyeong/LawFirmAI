# -*- coding: utf-8 -*-
"""
?¸ë“œ ?¨ìˆ˜ ?˜í¼
State Reductionê³?Adapterë¥??ë™?¼ë¡œ ?ìš©?˜ëŠ” ?°ì½”?ˆì´??ë°??¬í¼ ?¨ìˆ˜
"""

import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional

from langgraph_core.utils.state_adapter import (
    StateAdapter,
    validate_state_for_node,
)
from langgraph_core.utils.state_reduction import StateReducer

logger = logging.getLogger(__name__)

# ?„ì—­ ê²€??ê²°ê³¼ ìºì‹œ (LangGraph reducer ?ì‹¤ ?€ë¹?
# node_wrappers?ì„œ ?€?¥í•˜ê³? ?´í›„ ?¸ë“œ?ì„œ ë³µì›
_global_search_results_cache: Optional[Dict[str, Any]] = None


def with_state_optimization(node_name: str, enable_reduction: bool = True):
    """
    State ìµœì ?”ë? ?ìš©?˜ëŠ” ?°ì½”?ˆì´??

    ?ìš© ê¸°ëŠ¥:
    1. Input ê²€ì¦?
    2. State ?ë™ ë³€??(flat ??nested)
    3. State Reduction (? íƒ??

    Args:
        node_name: ?¸ë“œ ?´ë¦„
        enable_reduction: State Reduction ?œì„±???¬ë?

    Returns:
        ?°ì½”?ˆì´???¨ìˆ˜
    """
    def decorator(func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Dict[str, Any]:
            # ?„ì—­ ë³€??? ì–¸ (try ë¸”ë¡ ?„ì— ë¨¼ì? ? ì–¸ - Python ë¬¸ë²• ?”êµ¬?¬í•­)
            global _global_search_results_cache

            try:
                # 0. ?¸ì ì²˜ë¦¬: ì²?ë²ˆì§¸ ?¸ìê°€ self?¸ì? ?•ì¸
                # ë°”ìš´??ë©”ì„œ?œì˜ ê²½ìš° selfê°€ ?´ë? ë°”ì¸?©ë˜???ˆìœ¼ë¯€ë¡?
                # LangGraph??stateë§??„ë‹¬?´ì•¼ ?˜ì?ë§? ?¹ì‹œ ëª¨ë? ?í™© ?€ë¹?
                if len(args) == 0:
                    logger.error(f"No arguments provided to {node_name}")
                    return {}

                # ì²?ë²ˆì§¸ ?¸ìê°€ dictê°€ ?„ë‹Œ ê²½ìš° (selfê°€ ?„ë‹¬??ê²ƒìœ¼ë¡?ê°„ì£¼)
                if len(args) > 1:
                    # args[0]?€ self, args[1]?€ state
                    self_arg = args[0]
                    state = args[1]
                    rest_args = args[2:]
                else:
                    # args[0]?€ state
                    state = args[0]
                    rest_args = args[1:]

                # 0-1. Stateê°€ ?•ì…”?ˆë¦¬?¸ì? ?•ì¸
                if not isinstance(state, dict):
                    error_msg = (
                        f"State parameter must be a dict for node {node_name}, "
                        f"got {type(state).__name__}. "
                        f"Attempting to call original function without optimization."
                    )
                    logger.error(error_msg)
                    # ?ë³¸ ?¨ìˆ˜ ì§ì ‘ ?¸ì¶œ (ìµœì†Œ?œì˜ ì²˜ë¦¬)
                    if len(args) > 1:
                        return func(*args, **kwargs)
                    else:
                        return func(state, *rest_args, **kwargs)

                # ì¤‘ìš”: ?¸ë“œ ?¤í–‰ ?„ì— state??input ê·¸ë£¹???ˆëŠ”ì§€ ?•ì¸?˜ê³  ë³µì›
                # LangGraphê°€ ?´ì „ ?¸ë“œ??ê²°ê³¼ë§??„ë‹¬?˜ëŠ” ê²½ìš°, input???¬ë¼ì§????ˆìŒ
                state_has_input = "input" in state and isinstance(state.get("input"), dict)
                state_has_query = state_has_input and bool(state["input"].get("query"))

                # ì¤‘ìš”: ê²€??ê²°ê³¼ ?¸ë“œ?¤ì— ?€??search ê·¸ë£¹ ë³µì›
                # execute_searches_parallel??ê²°ê³¼ê°€ ?¤ìŒ ?¸ë“œ???„ë‹¬?˜ì? ?Šì„ ???ˆìŒ
                search_dependent_nodes = [
                    "merge_and_rerank_with_keyword_weights",
                    "filter_and_validate_results",
                    "update_search_metadata",
                    "process_search_results_combined",
                    "prepare_document_context_for_prompt",
                    "generate_answer_enhanced"
                ]

                # ?”ë²„ê¹? search_dependent_nodes ì²´í¬
                is_search_dependent = node_name in search_dependent_nodes
                if is_search_dependent:
                    print(f"[DEBUG] node_wrappers ({node_name}): IS a search_dependent node")

                if is_search_dependent:
                    # ?„ì—­ ìºì‹œ?ì„œ ê²€??ê²°ê³¼ ë³µì› (?¸ë“œ ?¤í–‰ ?„ì— state??ì¶”ê?)
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
                            # ìµœìƒ???ˆë²¨?ë„ ì¶”ê? (flat êµ¬ì¡° ?¸í™˜)
                            if "semantic_results" not in state:
                                state["semantic_results"] = _global_search_results_cache.get("semantic_results", [])
                            if "keyword_results" not in state:
                                state["keyword_results"] = _global_search_results_cache.get("keyword_results", [])
                            # retrieved_docs?€ merged_documents??ë³µì› (?µë? ?ì„±???„ìš”)
                            if "retrieved_docs" not in state or not state.get("retrieved_docs"):
                                state["retrieved_docs"] = _global_search_results_cache.get("retrieved_docs", [])
                            if "merged_documents" not in state or not state.get("merged_documents"):
                                state["merged_documents"] = _global_search_results_cache.get("merged_documents", [])
                            restored_semantic = len(state["search"].get("semantic_results", []))
                            restored_keyword = len(state["search"].get("keyword_results", []))
                            restored_docs = len(state.get("retrieved_docs", []))
                            print(f"[DEBUG] node_wrappers ({node_name}): Restored to state BEFORE execution - semantic={restored_semantic}, keyword={restored_keyword}, retrieved_docs={restored_docs}")

                    # search ê·¸ë£¹???†ìœ¼ë©?state?ì„œ ì§ì ‘ ì°¾ê¸° (flat êµ¬ì¡°?ì„œ)
                    if "search" not in state or not isinstance(state.get("search"), dict):
                        # flat êµ¬ì¡°?ì„œ semantic_results, keyword_results ì°¾ê¸°
                        has_search_data = any(
                            key in state for key in [
                                "semantic_results", "keyword_results", "semantic_count", "keyword_count",
                                "optimized_queries", "search_params", "merged_documents", "keyword_weights"
                            ]
                        )
                        print(f"[DEBUG] node_wrappers ({node_name}): Checking flat state for search data - has_search_data={has_search_data}, state keys={list(state.keys())[:10] if isinstance(state, dict) else 'N/A'}")
                        if has_search_data:
                            # search ê·¸ë£¹ ?ì„±
                            if "search" not in state:
                                state["search"] = {}
                            search_group = state["search"]
                            # flat êµ¬ì¡°???°ì´?°ë? search ê·¸ë£¹?¼ë¡œ ë³µì‚¬
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
                            # retrieved_docs?€ merged_documents??flat state?ì„œ ë³µì›
                            if "retrieved_docs" in state and not state.get("retrieved_docs"):
                                # ?´ë? ìµœìƒ???ˆë²¨???ˆìœ¼ë©??¬ìš©
                                retrieved_docs = state.get("retrieved_docs", [])
                                if retrieved_docs:
                                    search_group["retrieved_docs"] = retrieved_docs
                            if "merged_documents" in state and not state.get("merged_documents"):
                                # ?´ë? ìµœìƒ???ˆë²¨???ˆìœ¼ë©??¬ìš©
                                merged_docs = state.get("merged_documents", [])
                                if merged_docs:
                                    search_group["merged_documents"] = merged_docs
                            logger.info(f"Restored search group from flat state for node {node_name}")
                            print(f"[DEBUG] node_wrappers ({node_name}): Restored search group - semantic_results={len(search_group.get('semantic_results', []))}, keyword_results={len(search_group.get('keyword_results', []))}, retrieved_docs={len(search_group.get('retrieved_docs', []))}")

                # ?”ë²„ê¹? state êµ¬ì¡° ?•ì¸
                if node_name in ["classify_query", "prepare_search_query", "merge_and_rerank_with_keyword_weights"]:
                    print(f"[DEBUG] node_wrappers ({node_name}): State keys={list(state.keys()) if isinstance(state, dict) else 'N/A'}")
                    print(f"[DEBUG] node_wrappers ({node_name}): state_has_input={state_has_input}, state_has_query={state_has_query}")
                    if node_name == "merge_and_rerank_with_keyword_weights":
                        search_group = state.get("search", {}) if isinstance(state.get("search"), dict) else {}
                        print(f"[DEBUG] node_wrappers ({node_name}): search group exists={bool(search_group)}, semantic_results={len(search_group.get('semantic_results', []))}, keyword_results={len(search_group.get('keyword_results', []))}")

                if not state_has_input or not state_has_query:
                    # state??input???†ê±°??queryê°€ ?†ìœ¼ë©?ìµœìƒ???ˆë²¨?ì„œ ì°¾ê¸°
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
                        # ?¤ë¥¸ ê·¸ë£¹?ì„œ ì°¾ê¸°
                        found_query = None
                        if "search" in state and isinstance(state.get("search"), dict):
                            found_query = state["search"].get("search_query", "")
                        elif "classification" in state and isinstance(state.get("classification"), dict):
                            # classification?ëŠ” queryê°€ ?†ì?ë§??¹ì‹œ ëª¨ë¥´???•ì¸
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

                # 1. Input ê²€ì¦?ë°??ë™ ë³€??
                # ?”ë²„ê¹? ?ë³¸ state??query ?•ì¸
                original_query_before = state.get("query") or (state.get("input") and isinstance(state.get("input"), dict) and state["input"].get("query"))
                if node_name == "classify_query":
                    print(f"[DEBUG] node_wrappers.classify_query: original state query='{original_query_before[:50] if original_query_before else 'EMPTY'}...'")

                is_valid, error, converted_state = validate_state_for_node(
                    state,
                    node_name,
                    auto_convert=True
                )

                # ë³€???„ì—??input ?•ì¸ ë°?ë³µì›
                if isinstance(converted_state, dict):
                    converted_has_input = "input" in converted_state and isinstance(converted_state.get("input"), dict)
                    converted_has_query = converted_has_input and bool(converted_state["input"].get("query"))

                    if not converted_has_input or not converted_has_query:
                        # state?ì„œ ?¤ì‹œ ì°¾ê¸°
                        query_from_state = state.get("query") or (state.get("input") and isinstance(state.get("input"), dict) and state["input"].get("query"))
                        session_id_from_state = state.get("session_id") or (state.get("input") and isinstance(state.get("input"), dict) and state["input"].get("session_id"))

                        if query_from_state:
                            if "input" not in converted_state:
                                converted_state["input"] = {}
                            converted_state["input"]["query"] = query_from_state
                            if session_id_from_state:
                                converted_state["input"]["session_id"] = session_id_from_state
                            logger.debug(f"Restored input group after conversion for node {node_name}")

                # ?”ë²„ê¹? converted_state??query ?•ì¸
                converted_query = converted_state.get("query") or (converted_state.get("input") and isinstance(converted_state.get("input"), dict) and converted_state["input"].get("query"))
                if node_name == "classify_query":
                    print(f"[DEBUG] node_wrappers.classify_query: converted_state query='{converted_query[:50] if converted_query else 'EMPTY'}...'")

                if not is_valid:
                    logger.warning(f"Input validation failed for {node_name}: {error}")

                # 2. State Reduction (?œì„±?”ëœ ê²½ìš°)
                if enable_reduction:
                    reducer = StateReducer(aggressive_reduction=True)
                    working_state = reducer.reduce_state_for_node(converted_state, node_name)

                    # Reduction ê²°ê³¼ê°€ ë¹„ì–´?ˆìœ¼ë©??ë³¸ ?¬ìš©
                    if not working_state:
                        logger.warning(f"State reduction returned empty dict for {node_name}, using converted_state")
                        working_state = converted_state

                    # ?íƒœ ?¬ê¸° ë¡œê¹…
                    if logger.isEnabledFor(logging.DEBUG):
                        original_size = _estimate_state_size(state)
                        reduced_size = _estimate_state_size(working_state)
                        reduction_pct = (1 - reduced_size / original_size) * 100 if original_size > 0 else 0
                        logger.debug(
                            f"State reduction for {node_name}: "
                            f"{reduction_pct:.1f}% reduction "
                            f"({original_size:.0f} ??{reduced_size:.0f} bytes)"
                        )
                else:
                    working_state = converted_state

                # ì¤‘ìš”: state_reduction ?„ì—???„ì—­ ìºì‹œ?ì„œ ê²€??ê²°ê³¼ ë³µì›
                # search_dependent ?¸ë“œ?¤ì— ?€??working_state??ê²€??ê²°ê³¼ ì¶”ê?
                # is_search_dependent ë³€?˜ëŠ” ?„ì—???•ì˜??
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
                            # ìµœìƒ???ˆë²¨?ë„ ì¶”ê? (flat êµ¬ì¡° ?¸í™˜)
                            if "semantic_results" not in working_state:
                                working_state["semantic_results"] = _global_search_results_cache.get("semantic_results", [])
                            if "keyword_results" not in working_state:
                                working_state["keyword_results"] = _global_search_results_cache.get("keyword_results", [])
                            restored_semantic = len(working_state["search"].get("semantic_results", []))
                            restored_keyword = len(working_state["search"].get("keyword_results", []))
                            print(f"[DEBUG] node_wrappers ({node_name}): Restored to working_state AFTER reduction - semantic={restored_semantic}, keyword={restored_keyword}")

                # 3. ?ë³¸ ?¨ìˆ˜ ?¸ì¶œ
                # ì¤‘ìš”: working_state??queryê°€ ?ˆëŠ”ì§€ ?•ì¸?˜ê³  ?†ìœ¼ë©??ë³¸ state?ì„œ ë³µì›
                # converted_state?€ ?ë³¸ state ëª¨ë‘ ?•ì¸
                original_query = None
                original_session_id = None

                # ?ë³¸ state?ì„œ query ì°¾ê¸° (?¬ëŸ¬ ?„ì¹˜ ?•ì¸)
                if isinstance(state, dict):
                    original_query = state.get("query") or (state.get("input") and isinstance(state.get("input"), dict) and state["input"].get("query"))
                    original_session_id = state.get("session_id") or (state.get("input") and isinstance(state.get("input"), dict) and state["input"].get("session_id"))

                # converted_state?ì„œ???•ì¸
                if not original_query and isinstance(converted_state, dict):
                    original_query = converted_state.get("query") or (converted_state.get("input") and isinstance(converted_state.get("input"), dict) and converted_state["input"].get("query"))
                    original_session_id = converted_state.get("session_id") or (converted_state.get("input") and isinstance(converted_state.get("input"), dict) and converted_state["input"].get("session_id"))

                # working_state??queryê°€ ?†ìœ¼ë©?ë³µì› (state reduction ?„ì—??ë³´ì¥)
                if "input" not in working_state or not working_state.get("input") or not working_state["input"].get("query"):
                    # ?ë³¸ state?ì„œ query ì°¾ê¸°
                    if original_query:
                        if "input" not in working_state:
                            working_state["input"] = {}
                        working_state["input"]["query"] = original_query
                        if original_session_id:
                            working_state["input"]["session_id"] = original_session_id
                        logger.info(f"Restored query in working_state for node {node_name}: '{original_query[:50]}...'")
                    # converted_state?ì„œ???•ì¸
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

                # ?”ë²„ê¹? working_state??query ?•ì¸
                if node_name in ["prepare_search_query"]:
                    working_query = working_state.get("query") or (working_state.get("input") and isinstance(working_state.get("input"), dict) and working_state["input"].get("query", ""))
                    print(f"[DEBUG] node_wrappers ({node_name}): working_state before function call - query='{working_query[:50] if working_query else 'EMPTY'}...'")
                    print(f"[DEBUG] node_wrappers ({node_name}): working_state keys={list(working_state.keys()) if isinstance(working_state, dict) else 'N/A'}")

                if len(args) > 1:
                    # selfê°€ ?ˆëŠ” ê²½ìš°
                    result = func(args[0], working_state, *rest_args, **kwargs)
                else:
                    # selfê°€ ?†ëŠ” ê²½ìš°
                    result = func(working_state, *rest_args, **kwargs)

                # ?”ë²„ê¹? result??query ?•ì¸
                if node_name in ["prepare_search_query"]:
                    result_query = result.get("query") if isinstance(result, dict) else None
                    result_input_query = result.get("input", {}).get("query", "") if isinstance(result, dict) and result.get("input") else None
                    print(f"[DEBUG] node_wrappers ({node_name}): result after function call - query='{result_query[:50] if result_query else 'N/A'}...', input.query='{result_input_query[:50] if result_input_query else 'N/A'}...'")

                # 4. ê²°ê³¼ë¥??ë³¸ State??ë³‘í•©
                # ì¤‘ìš”: result??queryê°€ ?†ìœ¼ë©??ë³¸ state??queryë¥?ë³´ì¡´
                if isinstance(result, dict) and isinstance(state, dict):
                    # result??input??queryê°€ ?†ìœ¼ë©??ë³¸ state??query ë³µì›
                    result_has_query = False
                    if "input" in result and isinstance(result.get("input"), dict):
                        result_has_query = bool(result["input"].get("query"))
                    elif "query" in result:
                        result_has_query = bool(result.get("query"))

                    # working_state?ì„œ??query ?•ì¸ (reduction ?„ì—??queryê°€ ?ˆì„ ???ˆìŒ)
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
                        # ?ë³¸ state?ì„œ query ë³µì›
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

                # ì¤‘ìš”: result??input ê·¸ë£¹???ˆìœ¼ë©?ëª¨ë“  ?„ìˆ˜ ?„ë“œë¥??¬í•¨?´ì•¼ ??
                # LangGraph??TypedDict??ê°??„ë“œë¥?ë³‘í•©?˜ëŠ”?? input???†ìœ¼ë©??´ì „ ê°’ì´ ?¬ë¼ì§????ˆìŒ
                # ?°ë¼????ƒ result??input ê·¸ë£¹???¬í•¨?œì¼œ????
                if isinstance(result, dict) and isinstance(state, dict):
                    # ??ƒ input ê·¸ë£¹ ë³´ì¡´ ë³´ì¥
                    input_to_preserve = None

                    # 1. state?ì„œ input ì°¾ê¸°
                    if "input" in state and isinstance(state.get("input"), dict):
                        input_to_preserve = state["input"].copy()
                    elif "query" in state or "session_id" in state:
                        input_to_preserve = {
                            "query": state.get("query", ""),
                            "session_id": state.get("session_id", "")
                        }

                    # 2. result?ì„œ input ì°¾ê¸°
                    result_has_input = "input" in result and isinstance(result.get("input"), dict)
                    result_has_query = result_has_input and bool(result["input"].get("query"))

                    # 3. result??input???†ê±°??queryê°€ ?†ìœ¼ë©?state??input ë³´ì¡´
                    if not result_has_input or not result_has_query:
                        if input_to_preserve:
                            if "input" not in result:
                                result["input"] = {}
                            result["input"]["query"] = input_to_preserve.get("query", result.get("input", {}).get("query", ""))
                            result["input"]["session_id"] = input_to_preserve.get("session_id", result.get("input", {}).get("session_id", ""))
                            logger.debug(f"Preserved input group from state for node {node_name}")

                    # 4. result??input???ˆì–´??queryê°€ ?†ìœ¼ë©?state??input ë³´ì¡´
                    elif result_has_input and not result_has_query:
                        if input_to_preserve and input_to_preserve.get("query"):
                            result["input"]["query"] = input_to_preserve["query"]
                            if input_to_preserve.get("session_id"):
                                result["input"]["session_id"] = input_to_preserve["session_id"]
                            logger.debug(f"Restored query from state input for node {node_name}")

                    # ì¤‘ìš”: execute_searches_parallel ?´í›„ ?¸ë“œ?¤ì— ?€???„ì—­ ìºì‹œ?ì„œ ë³µì› (result?ë§Œ ?ìš©)
                    # ì°¸ê³ : ?¸ë“œ ?¤í–‰ ??state ë³µì›?€ ?„ìª½?ì„œ ?´ë? ì²˜ë¦¬??
                    if node_name in ["merge_and_rerank_with_keyword_weights", "filter_and_validate_results", "update_search_metadata", "prepare_document_context_for_prompt"]:
                        if _global_search_results_cache and isinstance(result, dict):
                            # result??search ê·¸ë£¹???†ê±°??ë¹„ì–´?ˆìœ¼ë©?ìºì‹œ?ì„œ ë³µì›
                            result_search = result.get("search", {}) if isinstance(result.get("search"), dict) else {}
                            has_results = len(result_search.get("semantic_results", [])) > 0 or len(result_search.get("keyword_results", [])) > 0

                            if not has_results:
                                print(f"[DEBUG] node_wrappers ({node_name}): Restoring search results from global cache")
                                if "search" not in result:
                                    result["search"] = {}
                                result["search"].update(_global_search_results_cache)
                                # ìµœìƒ???ˆë²¨?ë„ ì¶”ê?
                                if "semantic_results" not in result:
                                    result["semantic_results"] = _global_search_results_cache.get("semantic_results", [])
                                if "keyword_results" not in result:
                                    result["keyword_results"] = _global_search_results_cache.get("keyword_results", [])
                                restored_semantic = len(result["search"].get("semantic_results", []))
                                restored_keyword = len(result["search"].get("keyword_results", []))
                                print(f"[DEBUG] node_wrappers ({node_name}): Restored from cache - semantic={restored_semantic}, keyword={restored_keyword}")

                    # ì¤‘ìš”: execute_searches_parallel??ê²½ìš° search ê·¸ë£¹ ë³´ì¡´
                    # LangGraph??TypedDictë¥?ë³‘í•©????SearchState???†ëŠ” ?„ë“œê°€ ?ì‹¤?????ˆìŒ
                    # ?°ë¼??result??search ê·¸ë£¹???ˆìœ¼ë©???ƒ ë³´ì¡´
                    if node_name == "execute_searches_parallel":
                        result_search = result.get("search") if isinstance(result.get("search"), dict) else {}
                        state_search = state.get("search") if isinstance(state.get("search"), dict) else {}

                        # result??search ê·¸ë£¹???ˆìœ¼ë©??•ì¸ ë°?ë¡œê¹…
                        if result_search:
                            semantic_count = len(result_search.get("semantic_results", []))
                            keyword_count = len(result_search.get("keyword_results", []))
                            print(f"[DEBUG] node_wrappers ({node_name}): result has search group - semantic_results={semantic_count}, keyword_results={keyword_count}")
                            # result??ëª…ì‹œ?ìœ¼ë¡?ë³´ì¡´ (LangGraph ë³‘í•© ë³´ì¥)
                            if "search" not in result or not isinstance(result.get("search"), dict):
                                result["search"] = {}
                            result["search"]["semantic_results"] = result_search.get("semantic_results", [])
                            result["search"]["keyword_results"] = result_search.get("keyword_results", [])
                            result["search"]["semantic_count"] = result_search.get("semantic_count", semantic_count)
                            result["search"]["keyword_count"] = result_search.get("keyword_count", keyword_count)
                        elif state_search:
                            # state??search ê·¸ë£¹???ˆìœ¼ë©?result?ë„ ë³µì‚¬
                            print(f"[DEBUG] node_wrappers ({node_name}): Copying search group from state to result")
                            result["search"] = state_search.copy()

                    # processing_steps ?„ì—­ ìºì‹œ???€??(state reduction ?ì‹¤ ë°©ì?)
                    if isinstance(result, dict):
                        # common ê·¸ë£¹?ì„œ processing_steps ?•ì¸
                        result_common = result.get("common", {})
                        if isinstance(result_common, dict):
                            result_steps = result_common.get("processing_steps", [])
                            if isinstance(result_steps, list) and len(result_steps) > 0:
                                # ?„ì—­ ìºì‹œ???€??
                                if not _global_search_results_cache:
                                    _global_search_results_cache = {}
                                if "processing_steps" not in _global_search_results_cache:
                                    _global_search_results_cache["processing_steps"] = []
                                # ê¸°ì¡´ steps?€ ë³‘í•© (ì¤‘ë³µ ?œê±°)
                                for step in result_steps:
                                    if isinstance(step, str) and step not in _global_search_results_cache["processing_steps"]:
                                        _global_search_results_cache["processing_steps"].append(step)

                        # ìµœìƒ???ˆë²¨?ì„œ???•ì¸
                        result_top_steps = result.get("processing_steps", [])
                        if isinstance(result_top_steps, list) and len(result_top_steps) > 0:
                            # ?„ì—­ ìºì‹œ???€??
                            if not _global_search_results_cache:
                                _global_search_results_cache = {}
                            if "processing_steps" not in _global_search_results_cache:
                                _global_search_results_cache["processing_steps"] = []
                            # ê¸°ì¡´ steps?€ ë³‘í•© (ì¤‘ë³µ ?œê±°)
                            for step in result_top_steps:
                                if isinstance(step, str) and step not in _global_search_results_cache["processing_steps"]:
                                    _global_search_results_cache["processing_steps"].append(step)

                    # 5. Nested êµ¬ì¡°ë©?ê·¸ë?ë¡?ë°˜í™˜, Flat êµ¬ì¡°ë©?ë³‘í•©
                    # ì¤‘ìš”: LangGraph reducerê°€ TypedDict ?„ë“œë§?ë³´ì¡´?˜ë?ë¡?
                    # result??ëª¨ë“  ?„ìˆ˜ ê·¸ë£¹??ëª…ì‹œ?ìœ¼ë¡??¬í•¨?œì¼œ????
                    # ?¹íˆ execute_searches_parallel??ê²½ìš° search ê·¸ë£¹??ë°˜ë“œ???¬í•¨?˜ì–´????
                    if node_name == "execute_searches_parallel":
                        # result??search ê·¸ë£¹???†ìœ¼ë©?state?ì„œ ë³µì‚¬
                        if "search" not in result or not isinstance(result.get("search"), dict):
                            if "search" in state and isinstance(state.get("search"), dict):
                                result["search"] = state["search"].copy()
                                print(f"[DEBUG] node_wrappers ({node_name}): Copied search group from state to result before return")

                        # result??search ê·¸ë£¹??ëª¨ë“  ?„ìˆ˜ ?„ë“œ ?¬í•¨ ?•ì¸
                        if "search" in result and isinstance(result.get("search"), dict):
                            result_search = result["search"]
                            # semantic_results?€ keyword_resultsê°€ ?ˆìœ¼ë©?ë°˜ë“œ??ë³´ì¡´
                            semantic_results = result_search.get("semantic_results", [])
                            keyword_results = result_search.get("keyword_results", [])
                            if semantic_results or keyword_results:
                                semantic_count = len(semantic_results)
                                keyword_count = len(keyword_results)

                                # ?„ì—­ ìºì‹œ???€??(LangGraph reducer ?ì‹¤ ?€ë¹?
                                # global ? ì–¸?€ wrapper ?¨ìˆ˜ ?œì‘ ë¶€ë¶„ì— ?´ë? ?ˆìŒ
                                # ì¤‘ìš”: query_complexity?€ needs_search ë³´ì¡´ (classify_complexity?ì„œ ?€?¥í•œ ê°?
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

                                # ë³´ì¡´??query_complexity ë³µì›
                                if preserved_complexity:
                                    _global_search_results_cache["query_complexity"] = preserved_complexity
                                    if preserved_needs_search is not None:
                                        _global_search_results_cache["needs_search"] = preserved_needs_search

                                print(f"[DEBUG] node_wrappers ({node_name}): Saved to global cache - semantic={semantic_count}, keyword={keyword_count}, complexity={preserved_complexity}")

                    # ì¤‘ìš”: merge_and_rerank_with_keyword_weights??ê²½ìš° retrieved_docs ìºì‹œ ë³´ì¡´
                    if node_name == "merge_and_rerank_with_keyword_weights":
                        result_search = result.get("search") if isinstance(result.get("search"), dict) else {}
                        retrieved_docs = result_search.get("retrieved_docs", [])
                        merged_documents = result_search.get("merged_documents", [])

                        # ìµœìƒ???ˆë²¨?ì„œ???•ì¸
                        top_retrieved_docs = result.get("retrieved_docs", [])
                        top_merged_docs = result.get("merged_documents", [])

                        print(f"[DEBUG] node_wrappers ({node_name}): result - search_group retrieved_docs={len(retrieved_docs) if isinstance(retrieved_docs, list) else 0}, merged_documents={len(merged_documents) if isinstance(merged_documents, list) else 0}, top_retrieved_docs={len(top_retrieved_docs) if isinstance(top_retrieved_docs, list) else 0}, top_merged_docs={len(top_merged_docs) if isinstance(top_merged_docs, list) else 0}")
                        print(f"[DEBUG] node_wrappers ({node_name}): result - retrieved_docs type={type(retrieved_docs).__name__}, is_list={isinstance(retrieved_docs, list)}, has_length={len(retrieved_docs) if isinstance(retrieved_docs, list) else 'N/A'}")

                        # retrieved_docs ?ëŠ” merged_documentsê°€ ?ˆìœ¼ë©??„ì—­ ìºì‹œ???€??
                        final_retrieved_docs = (retrieved_docs if isinstance(retrieved_docs, list) and len(retrieved_docs) > 0 else
                                               top_retrieved_docs if isinstance(top_retrieved_docs, list) and len(top_retrieved_docs) > 0 else
                                               merged_documents if isinstance(merged_documents, list) and len(merged_documents) > 0 else
                                               top_merged_docs if isinstance(top_merged_docs, list) and len(top_merged_docs) > 0 else [])

                        print(f"[DEBUG] node_wrappers ({node_name}): final_retrieved_docs={len(final_retrieved_docs) if isinstance(final_retrieved_docs, list) else 0}, type={type(final_retrieved_docs).__name__}, is_list={isinstance(final_retrieved_docs, list)}, has_length={len(final_retrieved_docs) if isinstance(final_retrieved_docs, list) else 'N/A'}")

                        # ê°œì„  2.1: process_search_results_combined ?¤í–‰ ??retrieved_docs ?„ì—­ ìºì‹œ ?€???•ì¸
                        if node_name == "process_search_results_combined":
                            print(f"[DEBUG] node_wrappers ({node_name}): process_search_results_combined ?¤í–‰ ?„ë£Œ - result êµ¬ì¡° ë¶„ì„ ì¤?..", flush=True)

                            # result ?„ì²´ êµ¬ì¡° ì¶œë ¥
                            if isinstance(result, dict):
                                result_keys = list(result.keys())
                                print(f"[DEBUG] node_wrappers ({node_name}): result keys: {result_keys}", flush=True)

                                # retrieved_docs ì°¾ê¸° ?œë„ (ëª¨ë“  ê°€?¥í•œ ê²½ë¡œ ?•ì¸)
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
                                        print(f"[DEBUG] node_wrappers ({node_name}): ??retrieved_docsë¥?{path_name}?ì„œ ì°¾ìŒ - ê°œìˆ˜: {len(docs)}", flush=True)
                                        break

                                if found_docs:
                                    final_retrieved_docs = found_docs
                                else:
                                    print(f"[DEBUG] node_wrappers ({node_name}): ??retrieved_docsë¥?ì°¾ì„ ???†ìŒ - ëª¨ë“  ê²½ë¡œ ?•ì¸ ?„ë£Œ", flush=True)
                                    # ê°?ê²½ë¡œ???ì„¸ ?•ë³´ ì¶œë ¥
                                    for path_name, docs in possible_paths.items():
                                        docs_type = type(docs).__name__
                                        docs_len = len(docs) if isinstance(docs, list) else 'N/A'
                                        docs_sample = docs[:1] if isinstance(docs, list) and len(docs) > 0 else None
                                        print(f"[DEBUG] node_wrappers ({node_name}):   - {path_name}: type={docs_type}, len={docs_len}, sample={docs_sample}", flush=True)
                            else:
                                print(f"[DEBUG] node_wrappers ({node_name}): ? ï¸ resultê°€ dictê°€ ?„ë‹˜ - type: {type(result).__name__}", flush=True)

                            # final_retrieved_docs ?„ì¬ ?íƒœ ?•ì¸
                            print(f"[DEBUG] node_wrappers ({node_name}): final_retrieved_docs={len(final_retrieved_docs) if isinstance(final_retrieved_docs, list) else 0}, type={type(final_retrieved_docs).__name__}", flush=True)

                        if isinstance(final_retrieved_docs, list) and len(final_retrieved_docs) > 0:
                            # global ? ì–¸?€ wrapper ?¨ìˆ˜ ?œì‘ ë¶€ë¶„ì— ?´ë? ?ˆìŒ
                            # ?„ì—­ ìºì‹œ ì´ˆê¸°??(?†ìœ¼ë©??ì„±)
                            if not _global_search_results_cache:
                                _global_search_results_cache = {}
                            # ì¤‘ìš”: query_complexity?€ needs_search ë³´ì¡´ (ì´ˆê¸°?”ë˜ì§€ ?Šì? ê²½ìš°?ë§Œ)
                            # ?´ë? ì¡´ì¬?˜ë©´ ??–´?°ì? ?ŠìŒ

                            # retrieved_docs?€ merged_documentsë¥??„ì—­ ìºì‹œ???€??
                            _global_search_results_cache["retrieved_docs"] = final_retrieved_docs
                            _global_search_results_cache["merged_documents"] = final_retrieved_docs

                            # search ê·¸ë£¹ ?„ì²´ë¥??„ì—­ ìºì‹œ???€??
                            if result_search:
                                if "search" not in _global_search_results_cache:
                                    _global_search_results_cache["search"] = {}
                                _global_search_results_cache["search"].update(result_search)
                                # retrieved_docs?€ merged_documents??search ê·¸ë£¹???¬í•¨
                                _global_search_results_cache["search"]["retrieved_docs"] = final_retrieved_docs
                                _global_search_results_cache["search"]["merged_documents"] = final_retrieved_docs
                            else:
                                # search ê·¸ë£¹???†ìœ¼ë©??ì„±?˜ì—¬ ?€??
                                if "search" not in _global_search_results_cache:
                                    _global_search_results_cache["search"] = {}
                                _global_search_results_cache["search"]["retrieved_docs"] = final_retrieved_docs
                                _global_search_results_cache["search"]["merged_documents"] = final_retrieved_docs

                            print(f"[DEBUG] node_wrappers ({node_name}): ??Saved retrieved_docs to global cache - count={len(final_retrieved_docs)}, cache has search group={bool(_global_search_results_cache.get('search'))}")
                            # ê°œì„  3: ?€????ê²€ì¦?
                            cached_count = len(_global_search_results_cache.get("retrieved_docs", []))
                            cached_search_count = len(_global_search_results_cache.get("search", {}).get("retrieved_docs", []))
                            print(f"[DEBUG] node_wrappers ({node_name}): ?„ì—­ ìºì‹œ ê²€ì¦?- ìµœìƒ?? {cached_count}, search ê·¸ë£¹: {cached_search_count}")
                        else:
                            print(f"[DEBUG] node_wrappers ({node_name}): ? ï¸ WARNING - result has no retrieved_docs or merged_documents in search group or top level")
                            if node_name == "process_search_results_combined":
                                print(f"[DEBUG] node_wrappers ({node_name}): ??process_search_results_combined?ì„œ retrieved_docsê°€ ?€?¥ë˜ì§€ ?Šì•˜?µë‹ˆ??")

                    # execute_searches_parallel??result ì²˜ë¦¬ (?´ì „ ì½”ë“œ ? ì?)
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

                                # LangGraph reducer ?ì‹¤ ?€ë¹? result??ëª¨ë“  ?„ë“œë¥?ëª…ì‹œ?ìœ¼ë¡??¬í•¨
                                # ?¹íˆ search ê·¸ë£¹??ëª¨ë“  ?„ë“œë¥?ìµœìƒ???ˆë²¨?ë„ ?¬í•¨ (Flat êµ¬ì¡° ?¸í™˜)
                                if not isinstance(result.get("semantic_results"), list):
                                    result["semantic_results"] = semantic_results
                                if not isinstance(result.get("keyword_results"), list):
                                    result["keyword_results"] = keyword_results
                                if "semantic_count" not in result:
                                    result["semantic_count"] = semantic_count
                                if "keyword_count" not in result:
                                    result["keyword_count"] = keyword_count
                                print(f"[DEBUG] node_wrappers ({node_name}): Added search fields to top level - semantic_results={len(result.get('semantic_results', []))}, keyword_results={len(result.get('keyword_results', []))}")

                                # ?„ì—­ ìºì‹œ???€??(LangGraph reducer ?ì‹¤ ?€ë¹?
                                # ì¤‘ìš”: query_complexity?€ needs_search ë³´ì¡´ (classify_complexity?ì„œ ?€?¥í•œ ê°?
                                preserved_complexity = _global_search_results_cache.get("query_complexity") if _global_search_results_cache else None
                                preserved_needs_search = _global_search_results_cache.get("needs_search") if _global_search_results_cache else None

                                _global_search_results_cache = result_search.copy()

                                # ë³´ì¡´??query_complexity ë³µì›
                                if preserved_complexity:
                                    _global_search_results_cache["query_complexity"] = preserved_complexity
                                    if preserved_needs_search is not None:
                                        _global_search_results_cache["needs_search"] = preserved_needs_search

                                print(f"[DEBUG] node_wrappers ({node_name}): Saved to global cache - semantic={semantic_count}, keyword={keyword_count}, complexity={preserved_complexity}")

                    if "input" in state and isinstance(state.get("input"), dict):
                        # Nested êµ¬ì¡°ë©?ê·¸ë?ë¡?ë°˜í™˜?˜ë˜, ëª¨ë“  ?„ìˆ˜ ê·¸ë£¹ ?¬í•¨ ?•ì¸
                        # LangGraph reducerê°€ ?„ì²´ resultë¥?ë³‘í•©?˜ë?ë¡? ëª¨ë“  ê·¸ë£¹???¬í•¨?´ì•¼ ??
                        if "input" not in result:
                            result["input"] = state["input"].copy()
                        return result
                    else:
                        # Flat êµ¬ì¡°ë©?ë³‘í•©
                        state.update(result)
                        return state

                return result

            except Exception as e:
                logger.error(f"Error in state optimization wrapper for {node_name}: {e}", exc_info=True)
                # ?ëŸ¬ ë°œìƒ ???ë³¸ ?¨ìˆ˜ ?¤í–‰ (ìµœì†Œ?œì˜ ì²˜ë¦¬)
                try:
                    return func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback function call also failed for {node_name}: {fallback_error}",
                        exc_info=True
                    )
                    # ë§ˆì?ë§??˜ë‹¨: ë¹??•ì…”?ˆë¦¬ ë°˜í™˜
                    return {}

        return wrapper
    return decorator


def _estimate_state_size(state: Dict[str, Any]) -> int:
    """State ?¬ê¸° ì¶”ì •"""
    import sys
    try:
        return sys.getsizeof(str(state))
    except:
        return len(str(state))


def with_input_validation(node_name: str):
    """
    Input ê²€ì¦ë§Œ ?ìš©?˜ëŠ” ?°ì½”?ˆì´??

    State Reduction ?†ì´ Input ê²€ì¦ë§Œ ?˜í–‰
    """
    def decorator(func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Dict[str, Any]:
            try:
                # 0. ?¸ì ì²˜ë¦¬: ì²?ë²ˆì§¸ ?¸ìê°€ self?¸ì? ?•ì¸
                if len(args) == 0:
                    logger.error(f"No arguments provided to {node_name}")
                    return {}

                # ì²?ë²ˆì§¸ ?¸ìê°€ dictê°€ ?„ë‹Œ ê²½ìš° (selfê°€ ?„ë‹¬??ê²ƒìœ¼ë¡?ê°„ì£¼)
                if len(args) > 1:
                    # args[0]?€ self, args[1]?€ state
                    self_arg = args[0]
                    state = args[1]
                    rest_args = args[2:]
                else:
                    # args[0]?€ state
                    state = args[0]
                    rest_args = args[1:]

                # 0-1. Stateê°€ ?•ì…”?ˆë¦¬?¸ì? ?•ì¸
                if not isinstance(state, dict):
                    error_msg = (
                        f"State parameter must be a dict for node {node_name}, "
                        f"got {type(state).__name__}. "
                        f"Attempting to call original function without validation."
                    )
                    logger.error(error_msg)
                    return func(*args, **kwargs)

                # Input ê²€ì¦?ë°??ë™ ë³€??
                is_valid, error, converted_state = validate_state_for_node(
                    state,
                    node_name,
                    auto_convert=True
                )

                if not is_valid:
                    logger.warning(f"Input validation failed for {node_name}: {error}")

                # ?ë³¸ ?¨ìˆ˜ ?¸ì¶œ
                if len(args) > 1:
                    # selfê°€ ?ˆëŠ” ê²½ìš°
                    result = func(args[0], converted_state, *rest_args, **kwargs)
                else:
                    # selfê°€ ?†ëŠ” ê²½ìš°
                    result = func(converted_state, *rest_args, **kwargs)

                # ê²°ê³¼ ë°˜í™˜
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
    Stateë¥??„ìš”???ë™ ë³€??(?¸ì˜ ?¨ìˆ˜)

    Args:
        state: State ê°ì²´ (flat ?ëŠ” nested)

    Returns:
        ë³€?˜ëœ State ê°ì²´
    """
    return StateAdapter.to_nested(state)


def flatten_state_if_needed(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stateë¥?Flat êµ¬ì¡°ë¡?ë³€??(?¸ì˜ ?¨ìˆ˜)

    Args:
        state: State ê°ì²´ (nested)

    Returns:
        Flat êµ¬ì¡°??State
    """
    return StateAdapter.to_flat(state)
