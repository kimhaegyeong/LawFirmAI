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
from core.workflow.utils.query_diversifier import QueryDiversifier
from core.workflow.utils.search_result_balancer import SearchResultBalancer


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
        handle_error_func=None,
        semantic_search_engine=None
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
        
        # semantic_search_engine 저장 (타입별 검색용)
        self.semantic_search_engine = semantic_search_engine
        
        # 검색 쿼리 다변화 및 결과 균형 조정 유틸리티
        self.query_diversifier = QueryDiversifier()
        self.result_balancer = SearchResultBalancer(min_per_type=1, max_per_type=5)

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

            # 검색 결과 타입 균형 조정 (개선)
            try:
                # numpy 타입 변환 함수 (msgpack 직렬화 오류 방지)
                def convert_numpy_types(obj):
                    import numpy as np
                    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_numpy_types(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert_numpy_types(item) for item in obj]
                    return obj
                
                # 검색 결과에 numpy 타입 변환 적용
                semantic_results = [convert_numpy_types(doc) for doc in semantic_results]
                keyword_results = [convert_numpy_types(doc) for doc in keyword_results]
                
                # semantic_results와 keyword_results를 타입별로 그룹화
                all_results = semantic_results + keyword_results
                grouped_results = self.result_balancer.group_results_by_type(all_results)
                
                # Phase 3: 타입별 분포 확인 및 경고
                type_distribution = {}
                for doc_type, docs in grouped_results.items():
                    count = len(docs)
                    type_distribution[doc_type] = count
                    self.logger.info(f"📊 [SEARCH BALANCE] {doc_type}: {count}개")
                
                # 단일 타입만 검색된 경우 경고
                non_zero_types = [t for t, c in type_distribution.items() if c > 0]
                if len(non_zero_types) == 1:
                    single_type = non_zero_types[0]
                    self.logger.warning(
                        f"⚠️ [TYPE DIVERSITY] 단일 타입만 검색됨: {single_type} ({type_distribution[single_type]}개). "
                        f"다른 타입의 문서가 검색되지 않았습니다. 데이터 불균형 또는 검색 쿼리 최적화가 필요할 수 있습니다."
                    )
                elif len(non_zero_types) == 0:
                    self.logger.warning(
                        f"⚠️ [TYPE DIVERSITY] 검색 결과가 없습니다. 검색 쿼리나 데이터를 확인하세요."
                    )
                else:
                    # 타입 다양성 점수 계산 (0.0 ~ 1.0)
                    total_docs = sum(type_distribution.values())
                    if total_docs > 0:
                        # 엔트로피 기반 다양성 점수
                        import math
                        entropy = 0.0
                        for count in type_distribution.values():
                            if count > 0:
                                p = count / total_docs
                                entropy -= p * math.log2(p)
                        max_entropy = math.log2(len(non_zero_types)) if len(non_zero_types) > 1 else 1.0
                        diversity_score = entropy / max_entropy if max_entropy > 0 else 0.0
                        
                        self.logger.info(
                            f"✅ [TYPE DIVERSITY] 타입 다양성 점수: {diversity_score:.2f} "
                            f"(검색된 타입: {len(non_zero_types)}개, 총 문서: {total_docs}개)"
                        )
                        
                        if diversity_score < 0.5:
                            self.logger.warning(
                                f"⚠️ [TYPE DIVERSITY] 타입 다양성이 낮습니다 (점수: {diversity_score:.2f}). "
                                f"검색 쿼리 다변화 또는 데이터 균형 조정을 고려하세요."
                            )
                
                # 균형 조정된 결과 생성
                semantic_k = search_params.get("semantic_k", WorkflowConstants.SEMANTIC_SEARCH_K)
                keyword_k = search_params.get("keyword_k", WorkflowConstants.KEYWORD_SEARCH_K)
                balanced_results = self.result_balancer.balance_search_results(
                    grouped_results,
                    total_limit=semantic_k + keyword_k
                )
                
                # 균형 조정된 결과를 semantic_results와 keyword_results로 재분배
                # (기존 로직과의 호환성을 위해 유지하되, 균형 조정된 결과를 우선 사용)
                if balanced_results:
                    # semantic_results와 keyword_results를 균형 조정된 결과로 업데이트
                    # 관련도가 높은 결과를 semantic_results에, 나머지를 keyword_results에 배치
                    semantic_results_balanced = [
                        doc for doc in balanced_results 
                        if doc.get("relevance_score", 0.0) >= 0.5
                    ]
                    keyword_results_balanced = [
                        doc for doc in balanced_results 
                        if doc.get("relevance_score", 0.0) < 0.5 or doc not in semantic_results_balanced
                    ]
                    
                    # 기존 결과와 병합 (중복 제거)
                    existing_semantic_ids = {id(doc) for doc in semantic_results}
                    existing_keyword_ids = {id(doc) for doc in keyword_results}
                    
                    semantic_results = semantic_results + [
                        doc for doc in semantic_results_balanced 
                        if id(doc) not in existing_semantic_ids
                    ]
                    keyword_results = keyword_results + [
                        doc for doc in keyword_results_balanced 
                        if id(doc) not in existing_keyword_ids and id(doc) not in existing_semantic_ids
                    ]
                    
                    semantic_count = len(semantic_results)
                    keyword_count = len(keyword_results)
                    
                    self.logger.info(
                        f"✅ [SEARCH BALANCE] 균형 조정 완료: "
                        f"semantic={semantic_count}, keyword={keyword_count}, "
                        f"타입별 분포={dict((k, len(v)) for k, v in grouped_results.items())}"
                    )
            except Exception as e:
                self.logger.warning(f"검색 결과 균형 조정 실패 (기존 결과 사용): {e}")

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

        # Phase 1 + Phase 2: 타입별 별도 검색 수행 및 쿼리 다변화 적용 (타입 다양성 개선)
        document_types = {
            "statute_article": "statute",
            "case_paragraph": "case",
            "decision_paragraph": "decision",
            "interpretation_paragraph": "interpretation"
        }
        type_specific_results = {}
        type_specific_count = 0
        
        # semantic_search_engine 확인 (여러 방법 시도)
        semantic_engine = None
        if self.semantic_search_engine:
            semantic_engine = self.semantic_search_engine
            self.logger.info(f"🔍 [TYPE DIVERSITY] semantic_search_engine from self: {type(semantic_engine).__name__}")
        elif hasattr(self.search_handler, 'semantic_search_engine') and self.search_handler.semantic_search_engine:
            semantic_engine = self.search_handler.semantic_search_engine
            self.logger.info(f"🔍 [TYPE DIVERSITY] semantic_search_engine from search_handler: {type(semantic_engine).__name__}")
        elif hasattr(self.search_handler, 'semantic_search') and self.search_handler.semantic_search:
            semantic_engine = self.search_handler.semantic_search
            self.logger.info(f"🔍 [TYPE DIVERSITY] semantic_search_engine from search_handler.semantic_search: {type(semantic_engine).__name__}")
        else:
            self.logger.warning(f"⚠️ [TYPE DIVERSITY] semantic_search_engine not found: self.semantic_search_engine={self.semantic_search_engine}, search_handler.semantic_search_engine={getattr(self.search_handler, 'semantic_search_engine', 'N/A')}, search_handler.semantic_search={getattr(self.search_handler, 'semantic_search', 'N/A')}")
        
        if semantic_engine:
            # Phase 2: QueryDiversifier로 타입별 쿼리 생성
            try:
                diversified_queries = self.query_diversifier.diversify_search_queries(original_query or enhanced_semantic_query)
                self.logger.info(
                    f"🔍 [TYPE DIVERSITY] 다변화된 쿼리 생성: "
                    f"statute={len(diversified_queries.get('statute', []))}, "
                    f"case={len(diversified_queries.get('case', []))}, "
                    f"decision={len(diversified_queries.get('decision', []))}, "
                    f"interpretation={len(diversified_queries.get('interpretation', []))}"
                )
            except Exception as e:
                self.logger.warning(f"⚠️ [TYPE DIVERSITY] 쿼리 다변화 실패: {e}")
                diversified_queries = {}
            
            self.logger.info("🔍 [TYPE DIVERSITY] 타입별 검색 시작")
            for doc_type, query_type in document_types.items():
                try:
                    # Phase 2: 타입별 최적화된 쿼리 사용
                    type_queries = diversified_queries.get(query_type, [])
                    search_query = enhanced_semantic_query  # 기본 쿼리
                    
                    # 타입별 최적화된 쿼리가 있으면 사용
                    if type_queries:
                        search_query = type_queries[0]  # 첫 번째 최적화된 쿼리 사용
                        self.logger.info(f"🔍 [TYPE DIVERSITY] {doc_type}: 최적화된 쿼리 사용: '{search_query[:50]}...'")
                    else:
                        self.logger.info(f"🔍 [TYPE DIVERSITY] {doc_type}: 기본 쿼리 사용: '{search_query[:50]}...'")
                    
                    # 각 타입별로 별도 의미적 검색 수행 (재시도 로직 활용)
                    self.logger.info(f"🔍 [TYPE DIVERSITY] {doc_type} 검색 시작 (k=20, threshold=0.05, source_types=[{doc_type}])")
                    type_results = semantic_engine.search(
                        search_query,
                        k=20,  # 더 많은 결과 검색
                        source_types=[doc_type],  # 타입별 필터 적용
                        similarity_threshold=0.05,  # 낮은 임계값으로 시작 (재시도 로직이 더 낮춤)
                        min_results=1,  # 최소 1개는 보장
                        disable_retry=False  # 재시도 로직 활성화 (임계값 자동 조정)
                    )
                    
                    # 결과가 없으면 더 일반적인 쿼리로 재시도
                    if not type_results:
                        # 원본 쿼리에서 핵심 키워드만 추출하여 재시도
                        core_keywords = original_query.split()[:3] if original_query else search_query.split()[:3]
                        fallback_query = " ".join(core_keywords)
                        self.logger.info(f"🔍 [TYPE DIVERSITY] {doc_type}: 폴백 쿼리로 재시도: '{fallback_query}'")
                        try:
                            type_results = semantic_engine.search(
                                fallback_query,
                                k=20,
                                source_types=[doc_type],
                                similarity_threshold=0.0,  # 최소 임계값
                                min_results=1,
                                disable_retry=False  # 재시도 로직 활성화
                            )
                            if type_results:
                                self.logger.info(f"✅ [TYPE DIVERSITY] {doc_type}: 폴백 쿼리로 {len(type_results)}개 검색 성공")
                        except Exception as e:
                            self.logger.debug(f"⚠️ [TYPE DIVERSITY] {doc_type} 폴백 검색 실패: {e}")
                    
                    # 결과가 여전히 없으면 매우 일반적인 키워드로 재시도
                    if not type_results:
                        # 타입별 일반 키워드 사용
                        type_keywords = {
                            "statute_article": "법령 조문",
                            "case_paragraph": "판례",
                            "decision_paragraph": "결정례",
                            "interpretation_paragraph": "해석례"
                        }
                        general_query = type_keywords.get(doc_type, "법률")
                        self.logger.info(f"🔍 [TYPE DIVERSITY] {doc_type}: 일반 키워드로 재시도: '{general_query}'")
                        try:
                            type_results = semantic_engine.search(
                                general_query,
                                k=5,  # 최소한 5개만
                                source_types=[doc_type],
                                similarity_threshold=0.0,
                                min_results=1,
                                disable_retry=False
                            )
                            if type_results:
                                self.logger.info(f"✅ [TYPE DIVERSITY] {doc_type}: 일반 키워드로 {len(type_results)}개 검색 성공")
                        except Exception as e:
                            self.logger.debug(f"⚠️ [TYPE DIVERSITY] {doc_type} 일반 키워드 검색 실패: {e}")
                    
                    self.logger.info(f"🔍 [TYPE DIVERSITY] {doc_type}: 최종 {len(type_results)}개 검색됨")
                    
                    # 최종 방안: 검색 결과가 없으면 타입별 샘플링으로 대체
                    if not type_results:
                        self.logger.info(f"🔍 [TYPE DIVERSITY] {doc_type}: 샘플링으로 대체 시도")
                        try:
                            type_results = self._get_type_sample(semantic_engine, doc_type, k=2)
                            if type_results:
                                self.logger.info(f"✅ [TYPE DIVERSITY] {doc_type}: 샘플링으로 {len(type_results)}개 가져옴")
                        except Exception as e:
                            self.logger.debug(f"⚠️ [TYPE DIVERSITY] {doc_type} 샘플링 실패: {e}")
                    
                    if type_results:
                        type_specific_results[doc_type] = type_results
                        semantic_results.extend(type_results)
                        type_specific_count += len(type_results)
                        self.logger.info(
                            f"✅ [TYPE DIVERSITY] {doc_type}: {len(type_results)}개 검색 성공 (쿼리: '{search_query[:30]}...')"
                        )
                    else:
                        self.logger.warning(
                            f"⚠️ [TYPE DIVERSITY] {doc_type}: 검색 결과 없음 (데이터 없음 또는 쿼리 관련성 낮음, 쿼리: '{search_query[:30]}...')"
                        )
                except Exception as e:
                    self.logger.error(f"❌ [TYPE DIVERSITY] 타입별 검색 실패 ({doc_type}): {e}")
                    import traceback
                    self.logger.debug(f"타입별 검색 예외 상세: {traceback.format_exc()}")
        else:
            self.logger.warning("⚠️ [TYPE DIVERSITY] semantic_search_engine을 찾을 수 없어 타입별 검색을 수행할 수 없습니다")
        
        semantic_count += type_specific_count
        
        if type_specific_count > 0:
            self.logger.info(
                f"✅ [TYPE DIVERSITY] 타입별 검색 완료: 총 {type_specific_count}개 추가 "
                f"(타입별 분포: {dict((k, len(v)) for k, v in type_specific_results.items())})"
            )
        else:
            self.logger.info("⚠️ [TYPE DIVERSITY] 타입별 검색 결과 없음 (데이터 불균형 또는 검색 실패)")

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

    def _get_type_sample(self, semantic_engine, doc_type: str, k: int = 2) -> List[Dict[str, Any]]:
        """
        특정 타입의 랜덤 샘플 가져오기 (검색 실패 시 사용)
        
        Args:
            semantic_engine: SemanticSearchEngineV2 인스턴스
            doc_type: 문서 타입
            k: 가져올 샘플 수
            
        Returns:
            List[Dict[str, Any]]: 샘플 문서 리스트
        """
        try:
            # DB에서 해당 타입의 랜덤 문서 가져오기 (성능 최적화: 인덱스 활용)
            conn = semantic_engine._get_connection()
            
            # 먼저 해당 타입의 문서 수 확인
            count_cursor = conn.execute(
                "SELECT COUNT(*) as count FROM text_chunks WHERE source_type = ? AND text IS NOT NULL AND LENGTH(text) > 50",
                (doc_type,)
            )
            count_row = count_cursor.fetchone()
            total_count = count_row['count'] if count_row else 0
            
            if total_count == 0:
                self.logger.debug(f"⚠️ [TYPE DIVERSITY] {doc_type}: 샘플링할 문서 없음 (총 0개)")
                return []
            
            # 랜덤 샘플링 (성능 최적화: LIMIT 사용)
            cursor = conn.execute(
                """
                SELECT id, text, source_id, source_type
                FROM text_chunks
                WHERE source_type = ? AND text IS NOT NULL AND LENGTH(text) > 50
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (doc_type, k)
            )
            rows = cursor.fetchall()
            
            if not rows:
                self.logger.debug(f"⚠️ [TYPE DIVERSITY] {doc_type}: 샘플링 결과 없음 (총 {total_count}개 중)")
                return []
            
            self.logger.info(f"✅ [TYPE DIVERSITY] {doc_type}: {len(rows)}개 샘플링 성공 (총 {total_count}개 중)")
            
            # 검색 결과 형식으로 변환
            samples = []
            for row in rows:
                chunk_id = row['id']
                text = row['text'] or ""
                source_id = row['source_id']
                
                # 메타데이터 조회 (오류 처리 강화)
                source_meta = {}
                try:
                    if hasattr(semantic_engine, '_get_source_metadata'):
                        source_meta = semantic_engine._get_source_metadata(conn, doc_type, source_id)
                        if not source_meta:
                            # 메타데이터 조회 실패 시 text_chunks에서 기본 정보 가져오기
                            cursor_meta = conn.execute(
                                "SELECT source_type, source_id, text FROM text_chunks WHERE id = ?",
                                (chunk_id,)
                            )
                            row_meta = cursor_meta.fetchone()
                            if row_meta:
                                source_meta = {
                                    "source_type": row_meta['source_type'],
                                    "source_id": row_meta['source_id'],
                                    "text": row_meta['text']
                                }
                except Exception as e:
                    self.logger.debug(f"⚠️ [TYPE DIVERSITY] 메타데이터 조회 실패 ({doc_type}, source_id={source_id}): {e}")
                    # 기본 메타데이터 설정
                    source_meta = {
                        "source_type": doc_type,
                        "source_id": source_id
                    }
                
                # UnifiedSourceFormatter로 출처 정보 생성 (메타데이터 기반 개선)
                try:
                    from core.generation.formatters.unified_source_formatter import UnifiedSourceFormatter
                    formatter = UnifiedSourceFormatter()
                    source_info = formatter.format_source(doc_type, source_meta)
                    source_name = source_info.name
                    source_url = source_info.url
                    
                    # source_name이 비어있거나 기본값이면 메타데이터에서 추출 시도
                    if not source_name or source_name == doc_type:
                        if doc_type == "statute_article":
                            source_name = source_meta.get("statute_name") or source_meta.get("name") or "법령 조문"
                        elif doc_type == "case_paragraph":
                            source_name = source_meta.get("casenames") or source_meta.get("doc_id") or "판례"
                        elif doc_type == "decision_paragraph":
                            source_name = f"{source_meta.get('org', '')} {source_meta.get('doc_id', '')}".strip() or "결정례"
                        elif doc_type == "interpretation_paragraph":
                            source_name = f"{source_meta.get('org', '')} {source_meta.get('title', '')}".strip() or "해석례"
                except Exception as e:
                    self.logger.debug(f"⚠️ [TYPE DIVERSITY] 출처 정보 생성 실패 ({doc_type}): {e}")
                    source_name = doc_type
                    source_url = ""
                
                samples.append({
                    "id": f"chunk_{chunk_id}",
                    "text": text,
                    "content": text,
                    "score": 0.3,  # 낮은 점수 (강제 샘플링)
                    "similarity": 0.3,
                    "type": doc_type,
                    "source_type": doc_type,
                    "source": source_name,
                    "source_url": source_url,
                    "source_id": source_id,
                    "metadata": {
                        "chunk_id": chunk_id,
                        "source_type": doc_type,
                        "source_id": source_id,
                        "text": text,
                        "is_sample": True,  # 샘플링된 문서 표시
                        **source_meta
                    },
                    "relevance_score": 0.3,
                    "search_type": "type_sample"
                })
            
            return samples
        except Exception as e:
            self.logger.warning(f"⚠️ [TYPE DIVERSITY] 타입 샘플링 실패 ({doc_type}): {e}")
            import traceback
            self.logger.debug(f"타입 샘플링 예외 상세: {traceback.format_exc()}")
            return []

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

