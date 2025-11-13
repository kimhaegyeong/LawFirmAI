# -*- coding: utf-8 -*-
"""
Search Execution Processor
검색 실행 로직을 처리하는 프로세서
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from core.workflow.state.state_definitions import LegalWorkflowState
from core.workflow.state.state_helpers import ensure_state_group
from core.workflow.utils.workflow_constants import WorkflowConstants


class SearchExecutionProcessor:
    """검색 실행 프로세서"""

    def __init__(
        self,
        search_handler,
        logger,
        config,
        keyword_search_func=None,
        get_state_value_func=None,
        set_state_value_func=None,
        get_query_type_str_func=None,
        determine_search_parameters_func=None,
        save_metadata_safely_func=None,
        update_processing_time_func=None,
        handle_error_func=None
    ):
        self.search_handler = search_handler
        self.logger = logger
        self.config = config
        self.keyword_search_func = keyword_search_func
        self._get_state_value_func = get_state_value_func
        self._set_state_value_func = set_state_value_func
        self._get_query_type_str_func = get_query_type_str_func
        self._determine_search_parameters_func = determine_search_parameters_func
        self._save_metadata_safely_func = save_metadata_safely_func
        self._update_processing_time_func = update_processing_time_func
        self._handle_error_func = handle_error_func

    def get_search_params(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """검색에 필요한 모든 파라미터를 한 번에 가져오기 (State 접근 최적화)"""
        from core.workflow.state.state_helpers import get_field

        optimized_queries = self._get_state_value(state, "optimized_queries", {})
        search_params = self._get_state_value(state, "search_params", {})
        query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
        legal_field = self._get_state_value(state, "legal_field", "")
        extracted_keywords = self._get_state_value(state, "extracted_keywords", [])
        original_query = self._get_state_value(state, "query", "")

        if "search" in state and isinstance(state["search"], dict):
            search_group = state["search"]
            if "extracted_keywords" in search_group and search_group["extracted_keywords"]:
                extracted_keywords = search_group["extracted_keywords"]

            if search_group.get("optimized_queries") and isinstance(search_group["optimized_queries"], dict) and len(search_group["optimized_queries"]) > 0:
                optimized_queries = search_group["optimized_queries"]
                if not extracted_keywords and "expanded_keywords" in optimized_queries:
                    extracted_keywords = optimized_queries.get("expanded_keywords", [])

            if search_group.get("search_params") and isinstance(search_group["search_params"], dict) and len(search_group["search_params"]) > 0:
                search_params = search_group["search_params"]

        if not extracted_keywords:
            extracted_keywords_raw = get_field(state, "extracted_keywords")
            if extracted_keywords_raw and len(extracted_keywords_raw) > 0:
                extracted_keywords = extracted_keywords_raw

        if not optimized_queries or len(optimized_queries) == 0:
            optimized_queries_raw = get_field(state, "optimized_queries")
            if optimized_queries_raw and len(optimized_queries_raw) > 0:
                optimized_queries = optimized_queries_raw
                if not extracted_keywords and "expanded_keywords" in optimized_queries:
                    extracted_keywords = optimized_queries.get("expanded_keywords", [])

        if not search_params or len(search_params) == 0:
            search_params_raw = get_field(state, "search_params")
            if search_params_raw and len(search_params_raw) > 0:
                search_params = search_params_raw

        if not original_query and "input" in state and isinstance(state.get("input"), dict):
            original_query = state["input"].get("query", "")

        return {
            "optimized_queries": optimized_queries,
            "search_params": search_params,
            "query_type_str": query_type_str,
            "legal_field": legal_field,
            "extracted_keywords": extracted_keywords,
            "original_query": original_query
        }

    def execute_searches_parallel(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """의미적 검색과 키워드 검색을 병렬로 실행"""
        try:
            start_time = time.time()

            debug_mode = os.getenv("DEBUG_SEARCH", "false").lower() == "true"

            params = self.get_search_params(state)
            optimized_queries = params["optimized_queries"]
            search_params = params["search_params"]
            query_type_str = params["query_type_str"]
            legal_field = params["legal_field"]
            extracted_keywords = params["extracted_keywords"]
            original_query = params["original_query"]

            if not extracted_keywords or len(extracted_keywords) == 0:
                extracted_keywords = self._get_state_value(state, "extracted_keywords", [])
                if not extracted_keywords and "search" in state and isinstance(state.get("search"), dict):
                    extracted_keywords = state["search"].get("extracted_keywords", [])
                if not extracted_keywords:
                    extracted_keywords = state.get("extracted_keywords", [])
                self.logger.info(f"🔍 [SEARCH] extracted_keywords from batch was empty, got {len(extracted_keywords)} from state directly")
            else:
                self.logger.info(f"🔍 [SEARCH] extracted_keywords from batch: {len(extracted_keywords)} keywords")

            if debug_mode:
                self.logger.debug(f"execute_searches_parallel: START")
                self.logger.debug(f"  - optimized_queries: {type(optimized_queries).__name__}, exists={bool(optimized_queries)}")
                self.logger.debug(f"  - search_params: {type(search_params).__name__}, exists={bool(search_params)}")

            semantic_query_value = optimized_queries.get("semantic_query", "") if optimized_queries else ""

            if not semantic_query_value or not str(semantic_query_value).strip():
                if original_query:
                    if debug_mode:
                        self.logger.warning(f"semantic_query is empty in execute_searches_parallel, using base query: '{original_query[:50]}...'")
                    optimized_queries["semantic_query"] = original_query
                    semantic_query_value = original_query

            has_semantic_query = optimized_queries and semantic_query_value and len(str(semantic_query_value).strip()) > 0
            keyword_queries_value = optimized_queries.get("keyword_queries", []) if optimized_queries else []

            if not keyword_queries_value or len(keyword_queries_value) == 0:
                if original_query:
                    if debug_mode:
                        self.logger.warning(f"keyword_queries is empty in execute_searches_parallel, using base query")
                    optimized_queries["keyword_queries"] = [original_query]
                    keyword_queries_value = [original_query]

            has_keyword_queries = optimized_queries and keyword_queries_value and len(keyword_queries_value) > 0

            if debug_mode:
                self.logger.debug(f"  - Validation: semantic_query='{semantic_query_value[:50] if semantic_query_value else 'EMPTY'}...', has_semantic_query={has_semantic_query}")
                self.logger.debug(f"  - Validation: keyword_queries={len(keyword_queries_value) if keyword_queries_value else 0}, has_keyword_queries={has_keyword_queries}")
                self.logger.debug(f"  - Validation: search_params is None={search_params is None}, is empty={search_params == {}}, keys={list(search_params.keys()) if search_params else []}")

            if not search_params or not isinstance(search_params, dict) or len(search_params) == 0:
                self.logger.warning(f"🔍 [SEARCH] search_params is empty, setting default values")
                search_params = self._determine_search_parameters(
                    query_type=query_type_str,
                    query_complexity=len(original_query) if original_query else 0,
                    keyword_count=len(extracted_keywords) if extracted_keywords else 0,
                    is_retry=False
                )
                self.logger.info(f"🔍 [SEARCH] Default search_params set: {search_params}")

            optimized_queries_valid = optimized_queries and isinstance(optimized_queries, dict) and len(optimized_queries) > 0
            search_params_valid = search_params and isinstance(search_params, dict) and len(search_params) > 0
            self.logger.info(f"🔍 [SEARCH] Validation check: optimized_queries_valid={optimized_queries_valid} (type: {type(optimized_queries).__name__}, len: {len(optimized_queries) if isinstance(optimized_queries, dict) else 'N/A'}), search_params_valid={search_params_valid} (type: {type(search_params).__name__}, len: {len(search_params) if isinstance(search_params, dict) else 'N/A'}), has_semantic_query={has_semantic_query}")

            if not optimized_queries_valid or not search_params_valid or not has_semantic_query:
                self.logger.warning(f"🔍 [SEARCH] PARALLEL SEARCH SKIP: optimized_queries_valid={optimized_queries_valid}, search_params_valid={search_params_valid}, has_semantic_query={has_semantic_query}")
                if debug_mode:
                    self.logger.warning("Optimized queries or search params not found")
                    self.logger.debug(f"PARALLEL SEARCH SKIP: optimized_queries={optimized_queries is not None}, search_params={search_params is not None}")
                self._set_state_value(state, "semantic_results", [])
                self._set_state_value(state, "keyword_results", [])
                self._set_state_value(state, "semantic_count", 0)
                self._set_state_value(state, "keyword_count", 0)
                return state

            semantic_results = []
            semantic_count = 0
            keyword_results = []
            keyword_count = 0

            if debug_mode:
                self.logger.debug(f"PARALLEL SEARCH START: semantic_query={optimized_queries.get('semantic_query', 'N/A')[:50]}, keyword_queries={len(optimized_queries.get('keyword_queries', []))}, original_query={original_query[:50] if original_query else 'N/A'}...")

            self.logger.info(f"🔍 [SEARCH] Before check: extracted_keywords={len(extracted_keywords) if extracted_keywords else 0} (type: {type(extracted_keywords).__name__})")
            if not extracted_keywords or len(extracted_keywords) == 0:
                extracted_keywords = self._get_state_value(state, "extracted_keywords", [])
                if not extracted_keywords and "search" in state and isinstance(state.get("search"), dict):
                    extracted_keywords = state["search"].get("extracted_keywords", [])
                if not extracted_keywords:
                    extracted_keywords = state.get("extracted_keywords", [])
                self.logger.info(f"🔍 [SEARCH] Re-fetched extracted_keywords for semantic search: {len(extracted_keywords)} keywords")
            else:
                self.logger.info(f"🔍 [SEARCH] extracted_keywords already has {len(extracted_keywords)} keywords, skipping re-fetch")

            final_keywords = extracted_keywords if extracted_keywords else []
            self.logger.info(f"🔍 [SEARCH] Final extracted_keywords before ThreadPoolExecutor: {len(final_keywords)} keywords (type: {type(final_keywords).__name__}, is_empty: {not final_keywords})")

            keywords_copy = list(final_keywords) if final_keywords else []
            self.logger.info(f"🔍 [SEARCH] keywords_copy created: {len(keywords_copy)} keywords (type: {type(keywords_copy).__name__})")

            with ThreadPoolExecutor(max_workers=2) as executor:
                semantic_future = executor.submit(
                    self.execute_semantic_search,
                    optimized_queries,
                    search_params,
                    original_query,
                    keywords_copy
                )

                keyword_future = executor.submit(
                    self.execute_keyword_search,
                    optimized_queries,
                    search_params,
                    query_type_str,
                    legal_field,
                    extracted_keywords,
                    original_query
                )

                try:
                    semantic_results, semantic_count = semantic_future.result(timeout=20)
                    if debug_mode:
                        self.logger.debug(f"Semantic future completed: {semantic_count} results")
                except Exception as e:
                    self.logger.error(f"Semantic search failed: {e}")
                    if debug_mode:
                        self.logger.debug(f"Semantic search exception: {e}")
                    semantic_results, semantic_count = [], 0

                try:
                    keyword_results, keyword_count = keyword_future.result(timeout=20)
                    if debug_mode:
                        self.logger.debug(f"Keyword future completed: {keyword_count} results")
                except Exception as e:
                    self.logger.error(f"Keyword search failed: {e}")
                    if debug_mode:
                        self.logger.debug(f"Keyword search exception: {e}")
                    keyword_results, keyword_count = [], 0

            # 법령 조문 직접 검색 추가 (개선 #10) - ThreadPoolExecutor 완료 후 병합
            direct_statute_results = []
            try:
                if original_query and query_type_str == "law_inquiry":
                    from core.agents.legal_data_connector_v2 import LegalDataConnectorV2
                    data_connector = LegalDataConnectorV2()
                    direct_statute_results = data_connector.search_statute_article_direct(original_query, limit=5)
                    if direct_statute_results:
                        self.logger.info(f"⚖️ [DIRECT STATUTE] {len(direct_statute_results)}개 조문 직접 검색 성공")
                        # 직접 검색된 조문을 keyword_results 최상위에 추가 (relevance_score=1.0이므로 최상위로)
                        keyword_results = direct_statute_results + keyword_results
                        keyword_count += len(direct_statute_results)
                        self.logger.info(f"⚖️ [DIRECT STATUTE] keyword_results에 {len(direct_statute_results)}개 조문 추가 완료 (총 {keyword_count}개)")
            except Exception as e:
                self.logger.warning(f"법령 조문 직접 검색 실패: {e}")

            ensure_state_group(state, "search")

            if debug_mode:
                self.logger.debug(f"PARALLEL SEARCH: Before save - semantic_results={len(semantic_results)}, keyword_results={len(keyword_results)}")

            self._set_state_value(state, "semantic_results", semantic_results)
            self._set_state_value(state, "keyword_results", keyword_results)
            self._set_state_value(state, "semantic_count", semantic_count)
            self._set_state_value(state, "keyword_count", keyword_count)

            if debug_mode:
                stored_semantic = self._get_state_value(state, "semantic_results", [])
                stored_keyword = self._get_state_value(state, "keyword_results", [])
                self.logger.debug(f"PARALLEL SEARCH: After save - semantic_results={len(stored_semantic)}, keyword_results={len(stored_keyword)}")

                if "search" in state and isinstance(state.get("search"), dict):
                    direct_semantic = state["search"].get("semantic_results", [])
                    direct_keyword = state["search"].get("keyword_results", [])
                    self.logger.debug(f"PARALLEL SEARCH: Direct state['search'] check - semantic={len(direct_semantic)}, keyword={len(direct_keyword)}")
                else:
                    self.logger.debug(f"PARALLEL SEARCH: state['search'] not found or not dict, state keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")

            self._save_metadata_safely(state, "_last_executed_node", "execute_searches_parallel")
            self._update_processing_time(state, start_time)

            elapsed_time = time.time() - start_time

            self.logger.info(
                f"✅ [PARALLEL SEARCH] Completed in {elapsed_time:.3f}s - "
                f"Semantic: {semantic_count} results, Keyword: {keyword_count} results"
            )

            if debug_mode:
                self.logger.debug(f"PARALLEL SEARCH: Semantic={semantic_count}, Keyword={keyword_count}")

                if semantic_results:
                    semantic_scores = [doc.get("relevance_score", 0.0) for doc in semantic_results[:5]]
                    self.logger.info(
                        f"🔍 [DEBUG] Semantic search details: "
                        f"Top scores: {semantic_scores}, "
                        f"Sample sources: {[doc.get('source', 'Unknown')[:30] for doc in semantic_results[:3]]}"
                    )
                else:
                    self.logger.warning("⚠️ [DEBUG] Semantic search returned 0 results")

                if keyword_results:
                    keyword_scores = [doc.get("relevance_score", doc.get("score", 0.0)) for doc in keyword_results[:5]]
                    self.logger.info(
                        f"🔍 [DEBUG] Keyword search details: "
                        f"Top scores: {keyword_scores}, "
                        f"Sample sources: {[doc.get('source', 'Unknown')[:30] for doc in keyword_results[:3]]}"
                    )
                else:
                    self.logger.warning("⚠️ [DEBUG] Keyword search returned 0 results")

        except Exception as e:
            self.logger.error(f"Error in parallel search: {e}", exc_info=True)
            return self.fallback_sequential_search(state)

        debug_mode = os.getenv("DEBUG_SEARCH", "false").lower() == "true"

        if debug_mode:
            if "search" in state and isinstance(state.get("search"), dict):
                final_search = state["search"]
                final_semantic = len(final_search.get("semantic_results", []))
                final_keyword = len(final_search.get("keyword_results", []))
                self.logger.debug(f"[DEBUG] execute_searches_parallel: Returning state with search group - semantic_results={final_semantic}, keyword_results={final_keyword}")
                self.logger.debug(f"[DEBUG] execute_searches_parallel: Returning state keys={list(state.keys()) if isinstance(state, dict) else 'N/A'}")
            else:
                self.logger.debug(f"[DEBUG] execute_searches_parallel: WARNING - Returning state WITHOUT search group!")
                self.logger.debug(f"[DEBUG] execute_searches_parallel: Returning state keys={list(state.keys()) if isinstance(state, dict) else 'N/A'}")

        return state

    def execute_semantic_search(
        self,
        optimized_queries: Dict[str, Any],
        search_params: Dict[str, Any],
        original_query: str = "",
        extracted_keywords: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """의미적 검색 실행"""
        semantic_results = []
        semantic_count = 0

        semantic_query = optimized_queries.get("semantic_query", "")
        semantic_k = search_params.get("semantic_k", WorkflowConstants.SEMANTIC_SEARCH_K)

        if extracted_keywords is None:
            extracted_keywords = []

        self.logger.info(
            f"🔍 [DEBUG] _execute_semantic_search_internal received: extracted_keywords={len(extracted_keywords)} (type: {type(extracted_keywords).__name__}), query='{semantic_query[:50]}...', k={semantic_k}"
        )

        self.logger.info(
            f"🔍 [DEBUG] Executing semantic search: query='{semantic_query[:50]}...', k={semantic_k}, original_query='{original_query[:50] if original_query else 'N/A'}...', extracted_keywords={len(extracted_keywords)}"
        )

        enhanced_semantic_query = semantic_query
        if extracted_keywords and len(extracted_keywords) > 0:
            core_keywords = []
            for kw in extracted_keywords[:5]:
                if isinstance(kw, str):
                    if any(term in kw for term in ["법", "조", "제", "민법", "형법", "상법", "임대차", "계약"]):
                        core_keywords.insert(0, kw)
                    else:
                        core_keywords.append(kw)

            if core_keywords:
                query_keywords = set(semantic_query.split())
                new_keywords = [kw for kw in core_keywords if kw not in query_keywords]
                if new_keywords:
                    enhanced_semantic_query = f"{semantic_query} {' '.join(new_keywords[:3])}"
                    self.logger.info(f"🔍 [SEMANTIC SEARCH] Enhanced semantic query with keywords: '{enhanced_semantic_query[:100]}...'")

        main_semantic, main_count = self.search_handler.semantic_search(
            enhanced_semantic_query,
            k=semantic_k,
            extracted_keywords=extracted_keywords
        )
        semantic_results.extend(main_semantic)
        semantic_count += main_count

        self.logger.info(
            f"🔍 [DEBUG] Main semantic search: {main_count} results (query: '{enhanced_semantic_query[:50]}...')"
        )

        if original_query and original_query.strip():
            enhanced_original_query = original_query
            if extracted_keywords and len(extracted_keywords) > 0:
                core_keywords = [str(kw) for kw in extracted_keywords[:3] if isinstance(kw, str)]
                if core_keywords:
                    query_keywords = set(original_query.split())
                    new_keywords = [kw for kw in core_keywords if kw not in query_keywords]
                    if new_keywords:
                        enhanced_original_query = f"{original_query} {' '.join(new_keywords[:2])}"

            original_semantic, original_count = self.search_handler.semantic_search(
                enhanced_original_query,
                k=semantic_k // 2,
                extracted_keywords=extracted_keywords
            )
            semantic_results.extend(original_semantic)
            semantic_count += original_count
            self.logger.info(
                f"🔍 [DEBUG] Original query semantic search: {original_count} results (query: '{enhanced_original_query[:50]}...')"
            )
            print(f"[DEBUG] _execute_semantic_search_internal: Added {original_count} results from original query search")

        keyword_queries = optimized_queries.get("keyword_queries", [])[:2]
        for i, kw_query in enumerate(keyword_queries, 1):
            if kw_query and kw_query.strip() and kw_query != semantic_query:
                kw_semantic, kw_count = self.search_handler.semantic_search(
                    kw_query,
                    k=semantic_k // 2,
                    extracted_keywords=extracted_keywords
                )
                semantic_results.extend(kw_semantic)
                semantic_count += kw_count
                self.logger.info(
                    f"🔍 [DEBUG] Keyword-based semantic search #{i}: {kw_count} results (query: '{kw_query[:50]}...')"
                )
                print(f"[DEBUG] _execute_semantic_search_internal: Added {kw_count} results from keyword query #{i}")

        self.logger.info(
            f"🔍 [DEBUG] Total semantic search results: {semantic_count} (unique: {len(semantic_results)})"
        )
        print(f"[DEBUG] SEMANTIC SEARCH INTERNAL: Total={semantic_count}, Unique={len(semantic_results)}")

        search_queries_used = []
        if semantic_query:
            search_queries_used.append(f"semantic_query({len(semantic_query)} chars)")
        if original_query:
            search_queries_used.append(f"original_query({len(original_query)} chars)")
        keyword_queries_used = optimized_queries.get("keyword_queries", [])[:2]
        if keyword_queries_used:
            search_queries_used.append(f"keyword_queries({len(keyword_queries_used)} queries)")
        print(f"[DEBUG] SEMANTIC SEARCH INTERNAL: Queries used: {', '.join(search_queries_used)}")

        return semantic_results, semantic_count

    def execute_keyword_search(
        self,
        optimized_queries: Dict[str, Any],
        search_params: Dict[str, Any],
        query_type_str: str,
        legal_field: str,
        extracted_keywords: List[str],
        original_query: str = ""
    ) -> Tuple[List[Dict[str, Any]], int]:
        """키워드 검색 실행"""
        keyword_results = []
        keyword_count = 0

        keyword_queries = optimized_queries.get("keyword_queries", [])
        keyword_limit = search_params.get("keyword_limit", WorkflowConstants.CATEGORY_SEARCH_LIMIT)

        self.logger.info(
            f"🔍 [DEBUG] Executing keyword search: {len(keyword_queries)} queries, "
            f"limit={keyword_limit}, field={legal_field}, "
            f"keywords={extracted_keywords[:5] if extracted_keywords else []}, "
            f"original_query='{original_query[:50] if original_query else 'N/A'}...'"
        )

        if original_query and original_query.strip():
            if self.keyword_search_func:
                original_kw_results, original_kw_count = self.keyword_search_func(
                    query=original_query,
                    query_type_str=query_type_str,
                    limit=keyword_limit,
                    legal_field=legal_field,
                    extracted_keywords=extracted_keywords
                )
            else:
                original_kw_results, original_kw_count = [], 0
            keyword_results.extend(original_kw_results)
            keyword_count += original_kw_count
            self.logger.info(
                f"🔍 [DEBUG] Original query keyword search: {original_kw_count} results (query: '{original_query[:50]}...')"
            )

        for i, kw_query in enumerate(keyword_queries, 1):
            if kw_query and kw_query.strip() and kw_query != original_query:
                if self.keyword_search_func:
                    kw_results, kw_count = self.keyword_search_func(
                        query=kw_query,
                        query_type_str=query_type_str,
                        limit=keyword_limit,
                        legal_field=legal_field,
                        extracted_keywords=extracted_keywords
                    )
                else:
                    kw_results, kw_count = [], 0
                keyword_results.extend(kw_results)
                keyword_count += kw_count
                self.logger.info(
                    f"🔍 [DEBUG] Keyword search #{i}: {kw_count} results (query: '{kw_query[:50]}...')"
                )

        self.logger.info(
            f"🔍 [DEBUG] Total keyword search results: {keyword_count} (unique: {len(keyword_results)})"
        )
        print(f"[DEBUG] KEYWORD SEARCH INTERNAL: Total={keyword_count}, Unique={len(keyword_results)}")

        return keyword_results, keyword_count

    def fallback_sequential_search(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """순차 검색 실행 (폴백)"""
        try:
            self.logger.warning("Falling back to sequential search")

            optimized_queries = self._get_state_value(state, "optimized_queries", {})
            search_params = self._get_state_value(state, "search_params", {})
            query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
            legal_field = self._get_state_value(state, "legal_field", "")
            extracted_keywords = optimized_queries.get("expanded_keywords", [])

            original_query = self._get_state_value(state, "query", "")
            if not original_query and "input" in state and isinstance(state.get("input"), dict):
                original_query = state["input"].get("query", "")

            extracted_keywords_for_search = self._get_state_value(state, "extracted_keywords", [])
            semantic_results, semantic_count = self.execute_semantic_search(
                optimized_queries, search_params, original_query, extracted_keywords_for_search
            )

            keyword_results, keyword_count = self.execute_keyword_search(
                optimized_queries, search_params, query_type_str, legal_field, extracted_keywords, original_query
            )

            self._set_state_value(state, "semantic_results", semantic_results)
            self._set_state_value(state, "keyword_results", keyword_results)
            self._set_state_value(state, "semantic_count", semantic_count)
            self._set_state_value(state, "keyword_count", keyword_count)

            self.logger.info(f"Sequential search completed: {semantic_count} semantic, {keyword_count} keyword")

        except Exception as e:
            self.logger.error(f"Error in sequential search: {e}", exc_info=True)
            self._handle_error(state, str(e), "순차 검색 중 오류 발생")

        return state

    def _get_state_value(self, state: LegalWorkflowState, key: str, default: Any = None) -> Any:
        """State에서 값 가져오기"""
        if self._get_state_value_func:
            return self._get_state_value_func(state, key, default)
        if isinstance(state, dict):
            if key in state:
                return state[key]
            if "search" in state and isinstance(state.get("search"), dict) and key in state["search"]:
                return state["search"][key]
        return default

    def _set_state_value(self, state: LegalWorkflowState, key: str, value: Any) -> None:
        """State에 값 설정"""
        if self._set_state_value_func:
            self._set_state_value_func(state, key, value)
        elif isinstance(state, dict):
            if "search" not in state or not isinstance(state.get("search"), dict):
                state["search"] = {}
            state["search"][key] = value

    def _get_query_type_str(self, query_type) -> str:
        """QueryType을 문자열로 변환"""
        if self._get_query_type_str_func:
            return self._get_query_type_str_func(query_type)
        if isinstance(query_type, str):
            return query_type
        if hasattr(query_type, 'value'):
            return query_type.value
        return str(query_type) if query_type else ""

    def _determine_search_parameters(
        self,
        query_type: str,
        query_complexity: int,
        keyword_count: int,
        is_retry: bool
    ) -> Dict[str, Any]:
        """검색 파라미터 결정"""
        if self._determine_search_parameters_func:
            return self._determine_search_parameters_func(query_type, query_complexity, keyword_count, is_retry)
        return {
            "semantic_k": WorkflowConstants.SEMANTIC_SEARCH_K,
            "keyword_limit": WorkflowConstants.CATEGORY_SEARCH_LIMIT,
            "min_relevance": self.config.similarity_threshold if hasattr(self.config, 'similarity_threshold') else 0.5,
            "max_results": WorkflowConstants.MAX_DOCUMENTS
        }

    def _save_metadata_safely(self, state: LegalWorkflowState, key: str, value: Any) -> None:
        """메타데이터 안전하게 저장"""
        if self._save_metadata_safely_func:
            self._save_metadata_safely_func(state, key, value)
        elif isinstance(state, dict):
            if "metadata" not in state or not isinstance(state.get("metadata"), dict):
                state["metadata"] = {}
            state["metadata"][key] = value

    def _update_processing_time(self, state: LegalWorkflowState, start_time: float) -> None:
        """처리 시간 업데이트"""
        if self._update_processing_time_func:
            self._update_processing_time_func(state, start_time)
        elif isinstance(state, dict):
            elapsed = time.time() - start_time
            if "metadata" not in state or not isinstance(state.get("metadata"), dict):
                state["metadata"] = {}
            if "processing_time" not in state["metadata"]:
                state["metadata"]["processing_time"] = 0.0
            state["metadata"]["processing_time"] += elapsed

    def _handle_error(self, state: LegalWorkflowState, error_msg: str, context: str) -> None:
        """에러 처리"""
        if self._handle_error_func:
            self._handle_error_func(state, error_msg, context)
        else:
            self.logger.error(f"{context}: {error_msg}")
            if isinstance(state, dict):
                if "errors" not in state:
                    state["errors"] = []
                state["errors"].append(f"{context}: {error_msg}")

