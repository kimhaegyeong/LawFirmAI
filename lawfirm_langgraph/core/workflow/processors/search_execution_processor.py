# -*- coding: utf-8 -*-
"""
Search Execution Processor
검색 실행 로직을 처리하는 프로세서
"""

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, CancelledError as FutureCancelledError
from typing import Any, Dict, List, Optional, Tuple

try:
    from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
except ImportError:
    from core.workflow.state.state_definitions import LegalWorkflowState
try:
    from lawfirm_langgraph.core.workflow.state.state_helpers import ensure_state_group, set_retrieved_docs
except ImportError:
    from core.workflow.state.state_helpers import ensure_state_group, set_retrieved_docs
try:
    from lawfirm_langgraph.core.workflow.utils.workflow_constants import WorkflowConstants
except ImportError:
    from core.workflow.utils.workflow_constants import WorkflowConstants
try:
    from lawfirm_langgraph.core.workflow.utils.query_diversifier import QueryDiversifier
except ImportError:
    from core.workflow.utils.query_diversifier import QueryDiversifier
try:
    from lawfirm_langgraph.core.workflow.utils.search_result_balancer import SearchResultBalancer
except ImportError:
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
        
        # State 접근 캐싱 (성능 최적화)
        self._state_cache = {}
        self._state_cache_key = None
        
        # 검색 쿼리 중복 방지 캐시
        self._executed_queries = set()  # 실행된 쿼리 추적

    def get_search_params(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """검색에 필요한 모든 파라미터를 한 번에 가져오기 (State 접근 최적화)"""
        from core.workflow.state.state_helpers import get_field
        import hashlib

        # State 캐싱: state 해시로 캐시 키 생성
        state_str = str(sorted(state.items())) if isinstance(state, dict) else str(state)
        state_hash = hashlib.md5(state_str.encode()).hexdigest()
        
        # 캐시 히트 확인
        if self._state_cache_key == state_hash and self._state_cache:
            self.logger.debug("✅ [PERFORMANCE] State cache hit in get_search_params")
            return self._state_cache.copy()
        
        # 캐시 미스: State에서 값 가져오기
        # Multi-Query 강화: state의 여러 위치에서 optimized_queries 찾기 (순서 중요)
        # _get_state_value가 None을 반환할 수 있으므로, 먼저 직접 확인
        optimized_queries = None
        
        # 디버깅: state 구조 확인 (디버그 모드에서만)
        if os.getenv("DEBUG_STATE_ACCESS", "false").lower() == "true":
            state_keys = list(state.keys()) if isinstance(state, dict) else []
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"🔍 [MULTI-QUERY] get_search_params: state keys={state_keys}")
        
        # search와 common 그룹의 구조도 확인
        if "search" in state and isinstance(state["search"], dict):
            search_keys = list(state["search"].keys())
            self.logger.debug(f"[MULTI-QUERY] search group keys={search_keys}")
        if "common" in state and isinstance(state.get("common"), dict):
            common_keys = list(state["common"].keys())
            self.logger.debug(f"[MULTI-QUERY] common group keys={common_keys}")
            if "search" in state["common"] and isinstance(state["common"]["search"], dict):
                common_search_keys = list(state["common"]["search"].keys())
                self.logger.debug(f"[MULTI-QUERY] common.search keys={common_search_keys}")
        
        # 1. top-level state에서 직접 확인 (가장 우선)
        if "optimized_queries" in state and isinstance(state["optimized_queries"], dict) and len(state["optimized_queries"]) > 0:
            optimized_queries = state["optimized_queries"]
            self.logger.debug(f"[MULTI-QUERY] Found optimized_queries in top-level state (keys: {list(optimized_queries.keys())})")
            self.logger.info(f"🔍 [MULTI-QUERY] Found optimized_queries in top-level state (keys: {list(optimized_queries.keys())})")
        
        # 2. search group에서 확인 (top-level에 없으면)
        if (not optimized_queries or (isinstance(optimized_queries, dict) and len(optimized_queries) == 0)) and "search" in state and isinstance(state["search"], dict):
            search_group = state["search"]
            search_optimized = search_group.get("optimized_queries")
            self.logger.debug(f"[MULTI-QUERY] Checking search group: optimized_queries type={type(search_optimized)}, value={search_optimized}")
            if search_optimized and isinstance(search_optimized, dict):
                if len(search_optimized) > 0:
                    optimized_queries = search_optimized
                    self.logger.debug(f"[MULTI-QUERY] Found optimized_queries in search group (keys: {list(optimized_queries.keys())})")
                    self.logger.info(f"🔍 [MULTI-QUERY] Found optimized_queries in search group (keys: {list(optimized_queries.keys())})")
                else:
                    self.logger.debug("[MULTI-QUERY] search group optimized_queries is empty dict")
            else:
                self.logger.debug(f"[MULTI-QUERY] search group optimized_queries is not a dict or None: {search_optimized}")
        
        # 3. common.search에서 확인 (위에서 찾지 못했으면)
        if (not optimized_queries or len(optimized_queries) == 0) and "common" in state and isinstance(state.get("common"), dict):
            common_search = state["common"].get("search", {})
            if isinstance(common_search, dict) and common_search.get("optimized_queries"):
                optimized_queries = common_search["optimized_queries"]
                self.logger.debug(f"[MULTI-QUERY] Found optimized_queries in common.search (keys: {list(optimized_queries.keys())})")
                self.logger.info(f"🔍 [MULTI-QUERY] Found optimized_queries in common.search (keys: {list(optimized_queries.keys())})")
        # 4. _get_state_value로 확인 (fallback)
        if not optimized_queries or len(optimized_queries) == 0:
            optimized_queries = self._get_state_value(state, "optimized_queries", {})
            if optimized_queries and len(optimized_queries) > 0:
                self.logger.debug(f"[MULTI-QUERY] Found optimized_queries via _get_state_value (keys: {list(optimized_queries.keys())})")
                self.logger.info(f"🔍 [MULTI-QUERY] Found optimized_queries via _get_state_value (keys: {list(optimized_queries.keys())})")
            else:
                self.logger.debug(f"[MULTI-QUERY] _get_state_value returned: {optimized_queries}")
        
        # optimized_queries가 None이면 빈 딕셔너리로 초기화
        if optimized_queries is None:
            optimized_queries = {}
            self.logger.debug("[MULTI-QUERY] optimized_queries was None, initialized to empty dict")
        
        # 5. Global cache에서 확인 (state reduction 대응)
        if (not optimized_queries or len(optimized_queries) == 0):
            try:
                from core.shared.wrappers.node_wrappers import _global_search_results_cache
                if _global_search_results_cache and isinstance(_global_search_results_cache, dict):
                    if "search" in _global_search_results_cache and isinstance(_global_search_results_cache["search"], dict):
                        cached_optimized = _global_search_results_cache["search"].get("optimized_queries")
                        if cached_optimized and isinstance(cached_optimized, dict) and len(cached_optimized) > 0:
                            optimized_queries = cached_optimized.copy()
                            self.logger.debug(f"[MULTI-QUERY] Found optimized_queries in global cache (keys: {list(optimized_queries.keys())})")
                            self.logger.info(f"🔍 [MULTI-QUERY] Found optimized_queries in global cache (keys: {list(optimized_queries.keys())})")
            except Exception as e:
                self.logger.debug(f"Failed to get optimized_queries from global cache: {e}")
        
        # 6. get_field로 확인 (최후의 수단)
        if not optimized_queries or len(optimized_queries) == 0:
            optimized_queries_raw = get_field(state, "optimized_queries")
            if optimized_queries_raw and isinstance(optimized_queries_raw, dict) and len(optimized_queries_raw) > 0:
                optimized_queries = optimized_queries_raw
                self.logger.info("🔍 [MULTI-QUERY] Found optimized_queries via get_field")
        
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
                # search group의 optimized_queries가 더 완전하면 사용
                if "multi_queries" in search_group["optimized_queries"] or len(search_group["optimized_queries"]) > len(optimized_queries):
                    optimized_queries = search_group["optimized_queries"]
                    self.logger.debug("🔍 [MULTI-QUERY] Using optimized_queries from search group (more complete)")
                if not extracted_keywords and "expanded_keywords" in optimized_queries:
                    extracted_keywords = optimized_queries.get("expanded_keywords", [])

            if search_group.get("search_params") and isinstance(search_group["search_params"], dict) and len(search_group["search_params"]) > 0:
                search_params = search_group["search_params"]

        if not extracted_keywords:
            extracted_keywords_raw = get_field(state, "extracted_keywords")
            if extracted_keywords_raw and len(extracted_keywords_raw) > 0:
                extracted_keywords = extracted_keywords_raw
        
        # Multi-Query 복원: optimized_queries가 있지만 multi_queries가 없는 경우 state에서 직접 확인
        if optimized_queries and "multi_queries" not in optimized_queries:
            # state의 여러 위치에서 multi_queries 확인 (순서 중요)
            state_multi_queries = None
            # 1. top-level state에서 직접 확인 (가장 우선)
            if "optimized_queries" in state and isinstance(state["optimized_queries"], dict):
                state_multi_queries = state["optimized_queries"].get("multi_queries")
                if state_multi_queries:
                    self.logger.info(f"🔍 [MULTI-QUERY] Found multi_queries in top-level state (count: {len(state_multi_queries)})")
            # 2. search group에서 확인
            if not state_multi_queries and "search" in state and isinstance(state.get("search"), dict):
                search_optimized = state["search"].get("optimized_queries", {})
                if isinstance(search_optimized, dict):
                    state_multi_queries = search_optimized.get("multi_queries")
                    if state_multi_queries:
                        self.logger.info(f"🔍 [MULTI-QUERY] Found multi_queries in search group (count: {len(state_multi_queries)})")
            # 3. common.search에서 확인
            if not state_multi_queries and "common" in state and isinstance(state.get("common"), dict):
                common_search = state["common"].get("search", {})
                if isinstance(common_search, dict) and common_search.get("optimized_queries"):
                    common_optimized = common_search["optimized_queries"]
                    if isinstance(common_optimized, dict):
                        state_multi_queries = common_optimized.get("multi_queries")
                        if state_multi_queries:
                            self.logger.info(f"🔍 [MULTI-QUERY] Found multi_queries in common.search (count: {len(state_multi_queries)})")
            # 4. common group에서 직접 확인
            if not state_multi_queries and "common" in state and isinstance(state.get("common"), dict):
                common_optimized = state["common"].get("optimized_queries", {})
                if isinstance(common_optimized, dict):
                    state_multi_queries = common_optimized.get("multi_queries")
                    if state_multi_queries:
                        self.logger.info(f"🔍 [MULTI-QUERY] Found multi_queries in common group (count: {len(state_multi_queries)})")
            
            if state_multi_queries:
                if not optimized_queries:
                    optimized_queries = {}
                optimized_queries["multi_queries"] = state_multi_queries
                self.logger.info(f"✅ [MULTI-QUERY] Restored multi_queries from state (count: {len(state_multi_queries)})")
            else:
                # 디버깅: state 구조 확인
                self.logger.warning(f"⚠️ [MULTI-QUERY] Could not find multi_queries in state. State keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")
                if isinstance(state, dict) and "search" in state:
                    self.logger.warning(f"⚠️ [MULTI-QUERY] search group keys: {list(state['search'].keys()) if isinstance(state['search'], dict) else 'N/A'}")
        
        # Multi-Query 확인 로그 (항상 출력)
        has_multi = optimized_queries and "multi_queries" in optimized_queries
        keys_str = list(optimized_queries.keys()) if optimized_queries else "None"
        self.logger.debug(f"[MULTI-QUERY] get_search_params: optimized_queries keys={keys_str}, has_multi_queries={has_multi}")
        if has_multi:
            self.logger.info(f"🔍 [MULTI-QUERY] get_search_params: Found multi_queries with {len(optimized_queries.get('multi_queries', []))} queries")
        elif optimized_queries:
            self.logger.warning(f"⚠️ [MULTI-QUERY] get_search_params: optimized_queries exists but no multi_queries (keys: {keys_str})")

        if not search_params or len(search_params) == 0:
            search_params_raw = get_field(state, "search_params")
            if search_params_raw and len(search_params_raw) > 0:
                search_params = search_params_raw

        if not original_query and "input" in state and isinstance(state.get("input"), dict):
            original_query = state["input"].get("query", "")

        result = {
            "optimized_queries": optimized_queries,
            "search_params": search_params,
            "query_type_str": query_type_str,
            "legal_field": legal_field,
            "extracted_keywords": extracted_keywords,
            "original_query": original_query
        }
        
        # 캐시 저장
        self._state_cache = result.copy()
        self._state_cache_key = state_hash
        
        return result

    def _generate_search_cache_key(
        self,
        optimized_queries: Dict[str, Any],
        search_params: Dict[str, Any],
        original_query: str
    ) -> Optional[str]:
        """검색 결과 캐싱을 위한 키 생성"""
        import hashlib
        
        try:
            # 캐시 키 생성 요소
            cache_key_parts = [
                str(optimized_queries.get("semantic_query", "")),
                str(original_query),
                str(search_params.get("semantic_k", "")),
                str(search_params.get("keyword_k", "")),
                ",".join(sorted(optimized_queries.get("keyword_queries", []))) if optimized_queries.get("keyword_queries") else ""
            ]
            cache_key_str = ":".join(cache_key_parts)
            cache_key = hashlib.md5(cache_key_str.encode('utf-8')).hexdigest()
            return f"search_results:{cache_key}"
        except Exception as e:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Failed to generate cache key: {e}")
            return None
    
    def _get_cached_search_results(
        self,
        cache_key: str
    ) -> Optional[Dict[str, Any]]:
        """캐시된 검색 결과 가져오기"""
        try:
            # PerformanceCache가 있으면 사용
            if hasattr(self, 'performance_optimizer') and hasattr(self.performance_optimizer, 'cache'):
                cached_result = self.performance_optimizer.cache.get_cached_answer(cache_key, "search")
                if cached_result and isinstance(cached_result, dict):
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f"✅ [SEARCH CACHE HIT] Found cached search results: {cache_key[:16]}...")
                    return cached_result
        except Exception as e:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Failed to get cached search results: {e}")
        return None
    
    def _cache_search_results(
        self,
        cache_key: str,
        search_results: Dict[str, Any]
    ) -> None:
        """검색 결과 캐싱"""
        try:
            # PerformanceCache가 있으면 사용
            if hasattr(self, 'performance_optimizer') and hasattr(self.performance_optimizer, 'cache'):
                self.performance_optimizer.cache.cache_answer(
                    cache_key,
                    search_results,
                    query_type="search",
                    ttl=3600  # 1시간 TTL
                )
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"✅ [SEARCH CACHE] Cached search results: {cache_key[:16]}...")
        except Exception as e:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Failed to cache search results: {e}")
    
    def _apply_cached_results(
        self,
        state: LegalWorkflowState,
        cached_results: Dict[str, Any]
    ) -> LegalWorkflowState:
        """캐시된 검색 결과를 state에 적용"""
        semantic_results = cached_results.get("semantic_results", [])
        keyword_results = cached_results.get("keyword_results", [])
        semantic_count = len(semantic_results)
        keyword_count = len(keyword_results)
        
        self._set_state_value(state, "semantic_results", semantic_results)
        self._set_state_value(state, "keyword_results", keyword_results)
        self._set_state_value(state, "semantic_count", semantic_count)
        self._set_state_value(state, "keyword_count", keyword_count)
        
        merged_docs = semantic_results + keyword_results
        set_retrieved_docs(state, merged_docs)
        
        return state

    def _calculate_result_quality(
        self,
        results: List[Dict[str, Any]]
    ) -> float:
        """검색 결과 품질 점수 계산"""
        if not results:
            return 0.0
        
        # 평균 관련도 점수 계산
        scores = []
        for doc in results:
            score = doc.get("relevance_score") or doc.get("score") or doc.get("final_weighted_score", 0.0)
            if isinstance(score, (int, float)):
                scores.append(float(score))
        
        if not scores:
            return 0.0
        
        avg_score = sum(scores) / len(scores)
        return min(1.0, max(0.0, avg_score))
    
    def _adjust_search_priority(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        query_type: str
    ) -> Dict[str, int]:
        """검색 품질에 따른 우선순위 조정"""
        
        # 품질 점수 계산
        semantic_quality = self._calculate_result_quality(semantic_results)
        keyword_quality = self._calculate_result_quality(keyword_results)
        
        # 기본 우선순위 (낮은 숫자가 높은 우선순위)
        priorities = {
            "semantic": 1,  # 기본 우선순위
            "keyword": 1,
            "multi_query": 2,  # 낮은 우선순위
            "direct_statute": 2
        }
        
        # 품질이 낮으면 보조 검색 우선순위 상향
        if semantic_quality < 0.5 and keyword_quality < 0.5:
            priorities["multi_query"] = 1  # 우선순위 상향
            priorities["direct_statute"] = 1
        
        # 쿼리 타입에 따른 우선순위 조정
        if query_type == "law_inquiry":
            priorities["direct_statute"] = 0  # 최우선
        elif query_type == "precedent_search":
            priorities["semantic"] = 0  # semantic 최우선
        
        return priorities

    def _calculate_dynamic_k_values(
        self,
        query_type_str: str,
        query_complexity: int,
        keyword_count: int,
        is_retry: bool,
        original_query: str,
        search_params: Dict[str, Any]
    ) -> Tuple[int, int]:
        """다차원 동적 k 값 계산 (개선: 쿼리 타입, 복잡도, 키워드 수 등 종합 고려)"""
        
        # 기본값
        base_semantic_k = search_params.get("semantic_k", WorkflowConstants.SEMANTIC_SEARCH_K)
        base_keyword_k = search_params.get("keyword_k", WorkflowConstants.KEYWORD_SEARCH_K)
        
        # 1. 쿼리 타입에 따른 조정
        type_multiplier = {
            "precedent_search": 1.3,  # 판례 검색: 더 많은 결과 (1.5 → 1.3으로 완화)
            "law_inquiry": 1.2,        # 법령 조회: 더 많은 결과 (1.3 → 1.2로 완화)
            "legal_advice": 1.1,       # 법률 상담: 약간 증가 (1.2 → 1.1로 완화)
            "general_question": 1.0,   # 일반 질문: 기본값
            "term_explanation": 0.9,   # 용어 설명: 감소 (빠른 응답)
            "procedure_guide": 1.0     # 절차 안내: 기본값
        }
        multiplier = type_multiplier.get(query_type_str, 1.0)
        
        # 2. 쿼리 복잡도에 따른 조정
        query_length = len(original_query) if original_query else 0
        if query_complexity > 100 or query_length > 100:
            multiplier += 0.2  # 복잡한 쿼리: 더 많은 결과
        elif query_complexity > 50 or query_length > 50:
            multiplier += 0.1  # 중간 복잡도: 약간 증가
        elif query_length < 15:  # 매우 짧은 쿼리
            multiplier = max(0.7, multiplier - 0.2)  # 감소 (빠른 응답)
        
        # 3. 키워드 수에 따른 조정
        if keyword_count > 10:
            multiplier += 0.15  # 많은 키워드: 더 많은 결과
        elif keyword_count > 5:
            multiplier += 0.1
        elif keyword_count == 0:
            multiplier = max(0.8, multiplier - 0.1)  # 키워드 없음: 감소
        
        # 4. 재시도 여부에 따른 조정
        if is_retry:
            multiplier += 0.3  # 재시도: 더 많은 결과 (0.5 → 0.3으로 완화)
        
        # 5. 성능 최적화 모드 적용
        performance_mode = os.getenv("SEARCH_PERFORMANCE_MODE", "balanced").lower()
        if performance_mode == "fast":
            # 빠른 모드: k 값을 20-30% 감소
            multiplier *= 0.75
        elif performance_mode == "balanced":
            # 균형 모드: 기본값 유지
            pass
        elif performance_mode == "quality":
            # 품질 우선 모드: k 값을 10-20% 증가
            multiplier *= 1.15
        
        # 6. 최종 k 값 계산 (상한/하한 적용)
        semantic_k = int(base_semantic_k * multiplier)
        keyword_k = int(base_keyword_k * multiplier)
        
        # 상한/하한 적용
        semantic_k = max(5, min(semantic_k, 20))   # 최소 5개, 최대 20개
        keyword_k = max(3, min(keyword_k, 15))     # 최소 3개, 최대 15개
        
        return semantic_k, keyword_k

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

            # 성능 최적화: extracted_keywords를 한 번만 확인 (중복 접근 제거)
            if not extracted_keywords or len(extracted_keywords) == 0:
                # 한 번에 모든 가능한 위치 확인
                extracted_keywords = (
                    self._get_state_value(state, "extracted_keywords", []) or
                    (state.get("search", {}).get("extracted_keywords", []) if isinstance(state.get("search"), dict) else []) or
                    state.get("extracted_keywords", []) or
                    []
                )
                if debug_mode:
                    self.logger.debug(f"extracted_keywords from batch was empty, got {len(extracted_keywords)} from state directly")
            elif debug_mode:
                self.logger.debug(f"extracted_keywords from batch: {len(extracted_keywords)} keywords")

            # 로깅 최적화: 로깅 레벨 체크 및 배치 로깅
            if self.logger.isEnabledFor(logging.DEBUG):
                debug_info = {
                    "optimized_queries": {
                        "type": type(optimized_queries).__name__,
                        "exists": bool(optimized_queries),
                        "keys": list(optimized_queries.keys()) if isinstance(optimized_queries, dict) else None
                    },
                    "search_params": {
                        "type": type(search_params).__name__,
                        "exists": bool(search_params),
                        "keys": list(search_params.keys()) if isinstance(search_params, dict) else None
                    }
                }
                self.logger.debug(f"execute_searches_parallel: START - {debug_info}")

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
                        self.logger.warning("keyword_queries is empty in execute_searches_parallel, using base query")
                    optimized_queries["keyword_queries"] = [original_query]
                    keyword_queries_value = [original_query]

            has_keyword_queries = optimized_queries and keyword_queries_value and len(keyword_queries_value) > 0

            # 로깅 최적화: 검증 정보 배치 로깅
            if self.logger.isEnabledFor(logging.DEBUG):
                validation_info = {
                    "semantic_query": semantic_query_value[:50] if semantic_query_value else 'EMPTY',
                    "has_semantic_query": has_semantic_query,
                    "keyword_queries_count": len(keyword_queries_value) if keyword_queries_value else 0,
                    "has_keyword_queries": has_keyword_queries,
                    "search_params": {
                        "is_none": search_params is None,
                        "is_empty": search_params == {},
                        "keys": list(search_params.keys()) if search_params else []
                    }
                }
                self.logger.debug(f"Validation: {validation_info}")

            if not search_params or not isinstance(search_params, dict) or len(search_params) == 0:
                self.logger.warning("🔍 [SEARCH] search_params is empty, setting default values")
                search_params = self._determine_search_parameters(
                    query_type=query_type_str,
                    query_complexity=len(original_query) if original_query else 0,
                    keyword_count=len(extracted_keywords) if extracted_keywords else 0,
                    is_retry=False
                )
                self.logger.info(f"🔍 [SEARCH] Default search_params set: {search_params}")

            optimized_queries_valid = optimized_queries and isinstance(optimized_queries, dict) and len(optimized_queries) > 0
            search_params_valid = search_params and isinstance(search_params, dict) and len(search_params) > 0
            # 로깅 최적화: 검증 체크 배치 로깅
            if self.logger.isEnabledFor(logging.DEBUG):
                validation_check_info = {
                    "optimized_queries_valid": optimized_queries_valid,
                    "optimized_queries": {
                        "type": type(optimized_queries).__name__,
                        "len": len(optimized_queries) if isinstance(optimized_queries, dict) else 'N/A'
                    },
                    "search_params_valid": search_params_valid,
                    "search_params": {
                        "type": type(search_params).__name__,
                        "len": len(search_params) if isinstance(search_params, dict) else 'N/A'
                    },
                    "has_semantic_query": has_semantic_query
                }
                self.logger.debug(f"🔍 [SEARCH] Validation check: {validation_check_info}")

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

            # Multi-Query 확인 로그 (로깅 최적화)
            multi_queries = optimized_queries.get("multi_queries", [])
            if multi_queries and debug_mode:
                self.logger.debug(f"[MULTI-QUERY] execute_searches_parallel: Found {len(multi_queries)} multi-queries in optimized_queries")
                self.logger.debug(f"🔍 [MULTI-QUERY] execute_searches_parallel: Found {len(multi_queries)} multi-queries")
            elif not multi_queries and debug_mode:
                self.logger.debug("⚠️ [MULTI-QUERY] execute_searches_parallel: No multi_queries in optimized_queries")
            
            if debug_mode:
                self.logger.debug(f"PARALLEL SEARCH START: semantic_query={optimized_queries.get('semantic_query', 'N/A')[:50]}, keyword_queries={len(optimized_queries.get('keyword_queries', []))}, multi_queries={len(multi_queries) if multi_queries else 0}, original_query={original_query[:50] if original_query else 'N/A'}...")

            # 성능 최적화: extracted_keywords 재확인 제거 (이미 위에서 확인함)
            final_keywords = extracted_keywords if extracted_keywords else []
            keywords_copy = list(final_keywords) if final_keywords else []
            
            if debug_mode:
                self.logger.debug(f"Final extracted_keywords: {len(final_keywords)} keywords, keywords_copy: {len(keywords_copy)} keywords")

            # 성능 최적화: ThreadPoolExecutor를 사용하되 더 효율적으로 실행
            # 법령 조문 직접 검색도 병렬화하여 성능 향상
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            semantic_results, semantic_count = [], 0
            keyword_results, keyword_count = [], 0
            
            # 조기 종료 최적화: 동적 임계값 계산 (개선: 다차원 동적 k 값 조정)
            # query_complexity와 is_retry 값 가져오기
            query_complexity = self._get_state_value(state, "query_complexity", len(original_query) if original_query else 0)
            if isinstance(query_complexity, str):
                # complexity_level이 문자열인 경우 숫자로 변환
                complexity_map = {"simple": 20, "moderate": 50, "complex": 100}
                query_complexity = complexity_map.get(query_complexity.lower(), 50)
            elif not isinstance(query_complexity, int):
                query_complexity = len(original_query) if original_query else 0
            
            is_retry = self._get_state_value(state, "needs_retry", False)
            if not isinstance(is_retry, bool):
                is_retry = False
            
            # 다차원 동적 k 값 계산
            semantic_k, keyword_k = self._calculate_dynamic_k_values(
                query_type_str=query_type_str,
                query_complexity=query_complexity,
                keyword_count=len(extracted_keywords) if extracted_keywords else 0,
                is_retry=is_retry,
                original_query=original_query,
                search_params=search_params
            )
            
            # search_params에 동적으로 계산된 k 값 업데이트
            search_params["semantic_k"] = semantic_k
            search_params["keyword_k"] = keyword_k
            
            # 검색 결과 캐싱 확인 (재시도가 아닌 경우만)
            if not is_retry:
                cache_key = self._generate_search_cache_key(
                    optimized_queries, search_params, original_query
                )
                if cache_key:
                    cached_results = self._get_cached_search_results(cache_key)
                    if cached_results:
                        self.logger.info("✅ [SEARCH CACHE HIT] Using cached search results")
                        return self._apply_cached_results(state, cached_results)
            
            min_required_results = semantic_k + keyword_k
            early_exit_threshold = int(min_required_results * 0.9)  # 10% 여유 (1.0 → 0.9)
            max_results_threshold = min_required_results * 2  # 최대 2배까지만
            
            # 조기 종료 조건 강화: 최소 결과 수 설정 (동적 k 값에 맞춰 조정)
            min_semantic_for_early_exit = max(3, int(semantic_k * 0.4))  # 40% (기존 //2 대신)
            min_keyword_for_early_exit = max(2, int(keyword_k * 0.4))
            
            # 법령 조문 직접 검색도 병렬 실행 (max_workers=3)
            needs_direct_statute = original_query and query_type_str == "law_inquiry"
            
            # Multi-Query 병렬 처리 최적화: Multi-Query 준비 (적응형 활용)
            multi_queries = optimized_queries.get("multi_queries", [])
            multi_queries_to_process = []
            if multi_queries and len(multi_queries) > 1:
                multi_queries_to_process = multi_queries[1:]  # 첫 번째는 이미 처리됨
                # 최대 개수는 Phase 2에서 결과 부족 정도에 따라 동적으로 결정
            
            # 동적 worker 수 계산 (Multi-Query 포함)
            base_workers = 2  # semantic + keyword
            if needs_direct_statute:
                base_workers += 1
            if multi_queries_to_process:
                base_workers += len(multi_queries_to_process)
            max_workers = min(base_workers, 6)  # 최대 6개로 제한
            
            # 동적 타임아웃 계산: 작업 수와 예상 결과 수에 따라 조정 (최적화: 타임아웃 감소)
            base_timeout = 6  # 기본 타임아웃 6초 (8 → 6초로 감소)
            worker_count = base_workers
            timeout_per_worker = 1.5  # 작업당 1.5초 추가 (2 → 1.5초로 감소)
            dynamic_timeout = base_timeout + (worker_count * timeout_per_worker)
            dynamic_timeout = min(dynamic_timeout, 15)  # 최대 15초로 제한 (20 → 15초로 감소)
            
            # 조기 종료 플래그
            early_exit_triggered = False
            early_exit_reason = None
            
            # 2단계 우선순위 검색: Phase 1 (핵심 검색) 먼저 실행
            phase1_sufficient = False
            
            # 검색 쿼리 중복 방지: 동일한 쿼리로 이미 검색했는지 확인
            import hashlib
            query_hash = hashlib.md5(
                f"{original_query}:{str(optimized_queries.get('semantic_query', ''))}".encode('utf-8')
            ).hexdigest()
            
            if query_hash in self._executed_queries:
                self.logger.info(f"⚠️ [DUPLICATE SEARCH] 동일한 쿼리로 이미 검색됨: {query_hash[:16]}... (스킵)")
            else:
                self._executed_queries.add(query_hash)
            
            # Phase 1: 핵심 검색 작업만 먼저 실행 (semantic + keyword)
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
                
                # Phase 1 완료 대기 (동적 타임아웃 조정 - 최적화: 10-15초로 단축)
                # 동적 k 값에 따라 타임아웃 조정: 최소 10초, 최대 15초, k 값에 따라 조정
                # 처리 시간 최적화를 위해 타임아웃 단축 (20-35초 -> 10-15초)
                phase1_timeout = max(10, min(15, 8 + (semantic_k + keyword_k) // 5))
                
                try:
                    for future in as_completed([semantic_future, keyword_future], timeout=phase1_timeout):
                        try:
                            if future == semantic_future:
                                semantic_results, semantic_count = future.result()
                                # 🔥 최적화: Semantic 검색이 0개 결과면 즉시 조기 종료
                                if len(semantic_results) == 0 and semantic_count == 0:
                                    self.logger.warning(
                                        "⚠️ [EARLY EXIT] Semantic search returned 0 results, "
                                        "skipping Phase 2 and retries"
                                    )
                                    # Keyword 검색 결과만 기다리고 종료
                                    if not keyword_future.done():
                                        try:
                                            keyword_results, keyword_count = keyword_future.result(timeout=5.0)
                                        except (TimeoutError, FutureCancelledError, Exception):
                                            keyword_results, keyword_count = [], 0
                                    early_exit_triggered = True
                                    early_exit_reason = "Semantic search returned 0 results"
                                    break
                            elif future == keyword_future:
                                keyword_results, keyword_count = future.result()
                            
                            # Phase 1 조기 종료 강화: 각 future 완료 시마다 즉시 체크
                            phase1_total = len(semantic_results) + len(keyword_results)
                            phase1_sufficient = (
                                phase1_total >= early_exit_threshold and
                                len(semantic_results) >= min_semantic_for_early_exit and
                                len(keyword_results) >= min_keyword_for_early_exit
                            )
                            
                            if phase1_sufficient:
                                # 나머지 future 취소 시도
                                remaining_futures = [f for f in [semantic_future, keyword_future] if not f.done()]
                                for remaining_future in remaining_futures:
                                    if not remaining_future.running():
                                        try:
                                            remaining_future.cancel()
                                            if self.logger.isEnabledFor(logging.DEBUG):
                                                self.logger.debug("Cancelled remaining search (early exit)")
                                        except (FutureCancelledError, Exception) as cancel_error:
                                            # 취소 중 발생하는 예외는 무시
                                            if self.logger.isEnabledFor(logging.DEBUG):
                                                self.logger.debug(f"Future cancellation error (ignored): {cancel_error}")
                                
                                self.logger.info(
                                    f"⚡ [PRIORITY SEARCH] Phase 1 sufficient "
                                    f"(total: {phase1_total}, semantic: {len(semantic_results)}, keyword: {len(keyword_results)}), "
                                    f"skipping Phase 2"
                                )
                                early_exit_triggered = True
                                early_exit_reason = f"Phase 1 sufficient: {phase1_total} results"
                                break  # 조기 종료
                            
                        except (FutureCancelledError, Exception) as e:
                            # CancelledError는 정상적인 취소이므로 경고만 로깅
                            if isinstance(e, FutureCancelledError):
                                if self.logger.isEnabledFor(logging.DEBUG):
                                    self.logger.debug(f"Search future cancelled: {future}")
                                if future == semantic_future:
                                    semantic_results, semantic_count = [], 0
                                elif future == keyword_future:
                                    keyword_results, keyword_count = [], 0
                            else:
                                if future == semantic_future:
                                    self.logger.error(f"Semantic search failed: {e}")
                                    semantic_results, semantic_count = [], 0
                                elif future == keyword_future:
                                    self.logger.error(f"Keyword search failed: {e}")
                                    keyword_results, keyword_count = [], 0
                    
                    # Phase 1 결과 평가 (조기 종료되지 않은 경우)
                    if not early_exit_triggered:
                        # 🔥 최적화: Semantic 검색이 0개 결과면 즉시 조기 종료
                        if len(semantic_results) == 0 and semantic_count == 0:
                            self.logger.warning(
                                "⚠️ [EARLY EXIT] Semantic search returned 0 results after Phase 1, "
                                "skipping Phase 2 and retries"
                            )
                            early_exit_triggered = True
                            early_exit_reason = "Semantic search returned 0 results"
                        else:
                            phase1_total = len(semantic_results) + len(keyword_results)
                            phase1_sufficient = (
                                phase1_total >= early_exit_threshold and
                                len(semantic_results) >= min_semantic_for_early_exit and
                                len(keyword_results) >= min_keyword_for_early_exit
                            )
                            
                            if phase1_sufficient:
                                self.logger.info(
                                    f"⚡ [PRIORITY SEARCH] Phase 1 sufficient "
                                    f"(total: {phase1_total}, semantic: {len(semantic_results)}, keyword: {len(keyword_results)}), "
                                    f"skipping Phase 2"
                                )
                                early_exit_triggered = True
                                early_exit_reason = f"Phase 1 sufficient: {phase1_total} results"
                
                except (TimeoutError, FutureCancelledError) as e:
                    # Phase 1 타임아웃 또는 취소: 부분 결과라도 사용
                    if isinstance(e, TimeoutError):
                        self.logger.warning("⚠️ Phase 1 timeout, using partial results")
                    else:
                        self.logger.warning("⚠️ Phase 1 cancelled, using partial results")
                    try:
                        if not semantic_results and semantic_future.done():
                            try:
                                semantic_results, semantic_count = semantic_future.result()
                            except (FutureCancelledError, Exception):
                                semantic_results, semantic_count = [], 0
                    except Exception:
                        semantic_results, semantic_count = [], 0
                    try:
                        if not keyword_results and keyword_future.done():
                            try:
                                keyword_results, keyword_count = keyword_future.result()
                            except (FutureCancelledError, Exception):
                                keyword_results, keyword_count = [], 0
                    except Exception:
                        keyword_results, keyword_count = [], 0
                    
                    # 🔥 최적화: 타임아웃 발생 후에도 Semantic 검색이 0개 결과면 조기 종료
                    if len(semantic_results) == 0 and semantic_count == 0:
                        self.logger.warning(
                            "⚠️ [EARLY EXIT] Phase 1 timeout with 0 semantic results, "
                            "skipping Phase 2 and retries"
                        )
                        early_exit_triggered = True
                        early_exit_reason = "Phase 1 timeout with 0 semantic results"
                        phase1_sufficient = False  # Phase 2 스킵을 위해 명시적으로 설정
            
            # Phase 2: 보조 검색 작업 실행 (결과가 부족한 경우만)
            # 🔥 최적화: Semantic 검색이 0개 결과면 Phase 2 스킵
            if not phase1_sufficient and not (len(semantic_results) == 0 and semantic_count == 0):
                # 🔥 개선 1: Phase 1 품질 평가 (방안 5)
                phase1_quality = self._evaluate_search_quality(
                    semantic_results,
                    keyword_results,
                    original_query
                )
                
                if phase1_quality >= 0.7:
                    # 품질이 충분하면 multi-query 스킵
                    self.logger.info(
                        f"✅ [MULTI-QUERY] Phase 1 quality sufficient "
                        f"(score: {phase1_quality:.2f}), skipping multi-query"
                    )
                    multi_queries_to_process = []
                else:
                    self.logger.info(
                        f"🔄 [MULTI-QUERY] Phase 1 quality insufficient "
                        f"(score: {phase1_quality:.2f}), will execute multi-query"
                    )
                
                self.logger.info(
                    f"🔄 [PRIORITY SEARCH] Phase 1 insufficient "
                    f"(total: {len(semantic_results) + len(keyword_results)}, "
                    f"semantic: {len(semantic_results)}, keyword: {len(keyword_results)}, "
                    f"quality: {phase1_quality:.2f}), starting Phase 2"
                )
                
                # 적응형 우선순위 조정
                priorities = self._adjust_search_priority(
                    semantic_results=semantic_results,
                    keyword_results=keyword_results,
                    query_type=query_type_str
                )
                
                # 🔥 개선 2: 적응형 Multi-Query 개수 결정 (방안 1)
                current_total = len(semantic_results) + len(keyword_results)
                missing_results = max(0, early_exit_threshold - current_total)
                
                if multi_queries_to_process:
                    if missing_results > 10:
                        # 많이 부족하면 최대 3개 사용
                        multi_queries_to_use = multi_queries_to_process[:3]
                        self.logger.info(
                            f"🔄 [MULTI-QUERY] High missing results ({missing_results}), "
                            f"using {len(multi_queries_to_use)} multi-queries"
                        )
                    elif missing_results > 5:
                        # 중간 정도 부족하면 2개 사용
                        multi_queries_to_use = multi_queries_to_process[:2]
                        self.logger.info(
                            f"🔄 [MULTI-QUERY] Moderate missing results ({missing_results}), "
                            f"using {len(multi_queries_to_use)} multi-queries"
                        )
                    else:
                        # 조금만 부족하면 1개만 사용
                        multi_queries_to_use = multi_queries_to_process[:1]
                        self.logger.info(
                            f"🔄 [MULTI-QUERY] Low missing results ({missing_results}), "
                            f"using {len(multi_queries_to_use)} multi-query"
                        )
                else:
                    multi_queries_to_use = []
                
                # Phase 2를 위한 ThreadPoolExecutor (우선순위에 따라)
                phase2_workers = len(multi_queries_to_use)  # Multi-Query 개수에 따라 조정
                if needs_direct_statute and priorities.get("direct_statute", 2) <= 1:
                    phase2_workers += 1  # direct_statute 추가
                phase2_workers = min(phase2_workers, 6)  # 최대 6개로 제한
                
                with ThreadPoolExecutor(max_workers=phase2_workers) as executor:
                    # 법령 조문 직접 검색 (법령 조회인 경우만)
                    direct_statute_future = None
                    direct_statute_results = []
                    if needs_direct_statute:
                        def _search_direct_statute():
                            try:
                                from core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2
                                data_connector = LegalDataConnectorV2()
                                return data_connector.search_statute_article_direct(original_query, limit=5)
                            except Exception as e:
                                if debug_mode:
                                    self.logger.debug(f"Direct statute search error: {e}")
                                return []
                        
                        direct_statute_future = executor.submit(_search_direct_statute)

                    # 🔥 개선 3: Multi-Query 병렬 실행 (방안 1)
                    multi_query_futures = {}
                    if multi_queries_to_use:
                        for idx, mq in enumerate(multi_queries_to_use):
                            # 부족한 결과 수를 multi-query 개수로 나눠서 각각 할당
                            k_per_query = max(3, missing_results // len(multi_queries_to_use)) if missing_results > 0 else max(3, semantic_k // 4)
                            mq_future = executor.submit(
                                self._execute_semantic_search_single,
                                mq,
                                k_per_query,
                                keywords_copy,
                                original_query
                            )
                            multi_query_futures[mq_future] = ('multi_query', mq[:50], idx)
                    
                    # Phase 2 futures map
                    futures_map = {}
                    if direct_statute_future:
                        futures_map[direct_statute_future] = ('direct_statute', None)
                    # Multi-query futures 추가 (튜플 형식으로 저장)
                    for mq_future, mq_info in multi_query_futures.items():
                        futures_map[mq_future] = mq_info
                
                completed_count = 0
                direct_statute_results = []
                unfinished_futures = []
                
                # 동적 타임아웃 사용 (작업 수에 따라 조정)
                # 로깅 최적화: 완료된 작업을 모아서 한 번에 로깅
                completed_tasks = []
                # 성능 최적화: 중복 제거를 위한 seen_ids와 seen_hashes를 한 번만 생성하고 재사용
                seen_ids = set()
                seen_hashes = set()
                
                try:
                    for future in as_completed(futures_map.keys(), timeout=dynamic_timeout):
                        # futures_map 값이 튜플일 수 있으므로 안전하게 처리
                        future_info = futures_map[future]
                        if isinstance(future_info, tuple):
                            if len(future_info) >= 2:
                                search_type, query_type = future_info[0], future_info[1:]
                            else:
                                search_type, query_type = future_info[0], None
                        else:
                            search_type, query_type = future_info, None
                        try:
                            if search_type == 'semantic':
                                if query_type == 'main':
                                    semantic_results, semantic_count = future.result()
                                    # seen_ids와 seen_hashes 업데이트
                                    for doc in semantic_results:
                                        doc_id = doc.get("id") or doc.get("doc_id")
                                        if doc_id:
                                            seen_ids.add(doc_id)
                                        # content hash 기반 중복 제거 (처음 100자 해시)
                                        content = doc.get("content", "") or doc.get("text", "")
                                        if content:
                                            content_hash = hash(content[:100])
                                            seen_hashes.add(content_hash)
                                    completed_tasks.append(('semantic', semantic_count))
                            elif search_type == 'multi_query':
                                # 🔥 개선 4: Multi-Query 결과 처리 (다양성 보장 병합)
                                mq_results, mq_count = future.result()
                                # query_type은 튜플 또는 리스트: (mq_query, idx) 또는 [mq_query, idx]
                                if isinstance(query_type, (tuple, list)) and len(query_type) >= 2:
                                    mq_query = query_type[0] if query_type[0] else "unknown"
                                    mq_idx = query_type[1] if len(query_type) > 1 else 0
                                else:
                                    mq_query = "unknown"
                                    mq_idx = 0
                                    
                                    if mq_results:
                                        # 다양성 보장 병합
                                        new_results = self._merge_multi_query_results_single(
                                            main_results=semantic_results,
                                            mq_results=mq_results,
                                            mq_query=mq_query or "unknown",
                                            original_query=original_query,
                                            seen_ids=seen_ids,
                                            seen_hashes=seen_hashes
                                        )
                                        
                                        # seen_ids와 seen_hashes 업데이트
                                        for doc in new_results:
                                            doc_id = doc.get("id") or doc.get("doc_id")
                                            if doc_id:
                                                seen_ids.add(doc_id)
                                            content = doc.get("content", "") or doc.get("text", "")
                                            if content:
                                                content_hash = hash(content[:100])
                                                seen_hashes.add(content_hash)
                                        
                                        semantic_results.extend(new_results)
                                        completed_tasks.append(('multi_query', len(new_results), mq_idx))
                            elif search_type == 'keyword':
                                keyword_results, keyword_count = future.result()
                                # seen_ids와 seen_hashes 업데이트 (semantic_results와의 중복 제거를 위해)
                                for doc in keyword_results:
                                    doc_id = doc.get("id") or doc.get("doc_id")
                                    if doc_id:
                                        seen_ids.add(doc_id)
                                    # content hash 기반 중복 제거 (처음 100자 해시)
                                    content = doc.get("content", "") or doc.get("text", "")
                                    if content:
                                        content_hash = hash(content[:100])
                                        seen_hashes.add(content_hash)
                                completed_tasks.append(('keyword', keyword_count))
                            elif search_type == 'direct_statute':
                                direct_statute_results = future.result()
                                completed_tasks.append(('direct_statute', len(direct_statute_results) if direct_statute_results else 0))
                            
                            completed_count += 1
                            
                            # 조기 종료 체크 강화: 각 작업 완료 시마다 체크 (개선: 조건 강화)
                            current_total = len(semantic_results) + len(keyword_results)
                            semantic_count_current = len(semantic_results)
                            keyword_count_current = len(keyword_results)
                            
                            # 조기 종료 조건 강화: 여러 조건 중 하나만 만족해도 종료
                            has_both_types_sufficient = (
                                semantic_count_current >= min_semantic_for_early_exit and
                                keyword_count_current >= min_keyword_for_early_exit
                            )
                            has_semantic_sufficient = semantic_count_current >= semantic_k and semantic_count_current >= 5
                            has_keyword_sufficient = keyword_count_current >= keyword_k and keyword_count_current >= 5
                            
                            if (current_total >= early_exit_threshold and has_both_types_sufficient) or \
                               has_semantic_sufficient or \
                               has_keyword_sufficient:
                                early_exit_triggered = True
                                if has_both_types_sufficient:
                                    early_exit_reason = f"Sufficient results (both types): {current_total} >= {early_exit_threshold} (semantic: {semantic_count_current}, keyword: {keyword_count_current})"
                                elif has_semantic_sufficient:
                                    early_exit_reason = f"Sufficient semantic results: {semantic_count_current} >= {semantic_k}"
                                else:
                                    early_exit_reason = f"Sufficient keyword results: {keyword_count_current} >= {keyword_k}"
                                
                                # 나머지 미완료 future 취소 (개선: Multi-Query 우선 취소)
                                remaining_futures = [f for f in futures_map.keys() if not f.done()]
                                
                                # Multi-Query futures 우선 취소 (덜 중요한 작업)
                                for remaining_future in remaining_futures:
                                    remaining_type, _ = futures_map[remaining_future]
                                    if remaining_type == 'multi_query' and not remaining_future.running():
                                        try:
                                            remaining_future.cancel()
                                            if self.logger.isEnabledFor(logging.DEBUG):
                                                self.logger.debug(f"Cancelled {remaining_type} search (early exit)")
                                        except (FutureCancelledError, Exception) as cancel_error:
                                            # 취소 중 발생하는 예외는 무시
                                            if self.logger.isEnabledFor(logging.DEBUG):
                                                self.logger.debug(f"Future cancellation error (ignored): {cancel_error}")
                                
                                # 다른 futures 취소
                                for remaining_future in remaining_futures:
                                    remaining_type, _ = futures_map[remaining_future]
                                    if remaining_type != 'multi_query' and not remaining_future.running():
                                        try:
                                            remaining_future.cancel()
                                            if self.logger.isEnabledFor(logging.DEBUG):
                                                self.logger.debug(f"Cancelled {remaining_type} search (early exit)")
                                        except (FutureCancelledError, Exception) as cancel_error:
                                            # 취소 중 발생하는 예외는 무시
                                            if self.logger.isEnabledFor(logging.DEBUG):
                                                self.logger.debug(f"Future cancellation error (ignored): {cancel_error}")
                                
                                if self.logger.isEnabledFor(logging.DEBUG):
                                    self.logger.debug(f"⚡ [EARLY EXIT] {early_exit_reason}")
                                break
                                
                        except (FutureCancelledError, Exception) as e:
                            # CancelledError는 정상적인 취소이므로 경고만 로깅
                            if isinstance(e, FutureCancelledError):
                                if self.logger.isEnabledFor(logging.DEBUG):
                                    self.logger.debug(f"{search_type} search cancelled: {e}")
                                if search_type == 'direct_statute':
                                    direct_statute_results = []
                                elif search_type == 'semantic':
                                    semantic_results, semantic_count = [], 0
                                else:
                                    keyword_results, keyword_count = [], 0
                            else:
                                if search_type == 'direct_statute':
                                    if self.logger.isEnabledFor(logging.DEBUG):
                                        self.logger.debug(f"Direct statute search failed: {e}")
                                    direct_statute_results = []
                                    completed_tasks.append(('direct_statute', 'error', str(e)))
                                else:
                                    self.logger.error(f"{search_type} search failed: {e}")
                                    if self.logger.isEnabledFor(logging.DEBUG):
                                        self.logger.debug(f"{search_type} search exception: {e}")
                                    completed_tasks.append((search_type, 'error', str(e)))
                                    if search_type == 'semantic':
                                        semantic_results, semantic_count = [], 0
                                    else:
                                        keyword_results, keyword_count = [], 0
                            completed_count += 1
                    
                    # 로깅 최적화: 완료된 작업 한 번에 로깅
                    if self.logger.isEnabledFor(logging.DEBUG) and completed_tasks:
                        self.logger.debug(f"Completed tasks: {completed_tasks}")
                except (TimeoutError, FutureCancelledError) as e:
                    # 타임아웃 또는 취소 발생 시 완료되지 않은 future 수집
                    unfinished_futures = [f for f in futures_map.keys() if not f.done()]
                    if isinstance(e, TimeoutError):
                        self.logger.warning(
                            f"⚠️ 병렬 검색 타임아웃 발생: {len(unfinished_futures)} (of {len(futures_map)}) futures unfinished"
                        )
                    else:
                        self.logger.warning(
                            f"⚠️ 병렬 검색 취소 발생: {len(unfinished_futures)} (of {len(futures_map)}) futures unfinished"
                        )
                
                    # Phase 2 타임아웃 처리 (최적화: 타임아웃 단축)
                    phase2_timeout = 5  # 5초 (8 → 5초로 감소)
                    
                    try:
                        for future in as_completed(futures_map.keys(), timeout=phase2_timeout):
                            search_type, query_type = futures_map[future]
                            try:
                                if search_type == 'multi_query':
                                    # Multi-Query 결과 처리 (중복 제거)
                                    mq_results, mq_count = future.result()
                                    if mq_results:
                                        new_results = []
                                        for doc in mq_results:
                                            doc_id = doc.get("id") or doc.get("doc_id")
                                            content = doc.get("content", "") or doc.get("text", "")
                                            content_hash = hash(content[:100]) if content else None
                                            
                                            # ID와 content hash 모두 확인하여 중복 제거
                                            if (not doc_id or doc_id not in seen_ids) and \
                                               (not content_hash or content_hash not in seen_hashes):
                                                if doc_id:
                                                    seen_ids.add(doc_id)
                                                if content_hash:
                                                    seen_hashes.add(content_hash)
                                                new_results.append(doc)
                                        
                                        semantic_results.extend(new_results)
                                        semantic_count += len(new_results)
                                        self.logger.info(f"✅ [PHASE 2] Multi-Query added {len(new_results)} new results")
                                
                                elif search_type == 'direct_statute':
                                    direct_statute_results = future.result()
                                    if direct_statute_results:
                                        # 중복 제거 후 keyword_results에 추가
                                        new_statute_results = []
                                        for doc in direct_statute_results:
                                            doc_id = doc.get("id") or doc.get("doc_id")
                                            if doc_id and doc_id not in seen_ids:
                                                seen_ids.add(doc_id)
                                                new_statute_results.append(doc)
                                        keyword_results.extend(new_statute_results)
                                        keyword_count += len(new_statute_results)
                                        self.logger.info(f"✅ [PHASE 2] Direct statute added {len(new_statute_results)} new results")
                            
                            except (FutureCancelledError, Exception) as e:
                                if isinstance(e, FutureCancelledError):
                                    if self.logger.isEnabledFor(logging.DEBUG):
                                        self.logger.debug(f"{search_type} search cancelled in Phase 2")
                                else:
                                    if search_type == 'direct_statute':
                                        if self.logger.isEnabledFor(logging.DEBUG):
                                            self.logger.debug(f"Direct statute search failed: {e}")
                                        direct_statute_results = []
                                    else:
                                        self.logger.warning(f"Multi-query search failed: {e}")
                    
                    except (TimeoutError, FutureCancelledError) as e:
                        if isinstance(e, TimeoutError):
                            self.logger.warning("⚠️ Phase 2 timeout, using partial results")
                        else:
                            self.logger.warning("⚠️ Phase 2 cancelled, using partial results")
                
                # 조기 종료 로깅
                if early_exit_triggered:
                    self.logger.info(
                        f"⚡ [EARLY EXIT] {early_exit_reason} - "
                        f"Semantic: {len(semantic_results)}, Keyword: {len(keyword_results)}"
                    )

            # 검색 결과 타입 균형 조정 (성능 최적화: 결과가 많을 때만 수행)
            total_results = len(semantic_results) + len(keyword_results)
            should_balance = total_results > 20  # 결과가 20개 이상일 때만 균형 조정
            
            if should_balance:
                try:
                    # numpy 타입 변환 함수 (최적화: 필요한 경우에만 변환)
                    def convert_numpy_types(obj, _depth=0):
                        # 재귀 깊이 제한으로 성능 향상
                        if _depth > 5:
                            return obj
                        import numpy as np
                        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                            return int(obj)
                        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, dict):
                            return {k: convert_numpy_types(v, _depth + 1) for k, v in obj.items()}
                        elif isinstance(obj, (list, tuple)):
                            return [convert_numpy_types(item, _depth + 1) for item in obj]
                        return obj
                    
                    # 검색 결과에 numpy 타입 변환 적용 (필요한 경우에만)
                    has_numpy = False
                    for doc in semantic_results[:5] + keyword_results[:5]:
                        import numpy as np
                        if any(isinstance(v, (np.integer, np.floating, np.ndarray)) for v in (doc.values() if isinstance(doc, dict) else [])):
                            has_numpy = True
                            break
                    
                    if has_numpy:
                        semantic_results = [convert_numpy_types(doc) for doc in semantic_results]
                        keyword_results = [convert_numpy_types(doc) for doc in keyword_results]
                    
                    # semantic_results와 keyword_results를 타입별로 그룹화
                    all_results = semantic_results + keyword_results
                    grouped_results = self.result_balancer.group_results_by_type(all_results)
                    
                    # 타입별 분포 확인 (로깅 최적화)
                    type_distribution = {doc_type: len(docs) for doc_type, docs in grouped_results.items()}
                    # 로깅 최적화: 타입별 분포 배치 로깅
                    if self.logger.isEnabledFor(logging.DEBUG):
                        non_zero_types = {k: v for k, v in type_distribution.items() if v > 0}
                        if non_zero_types:
                            self.logger.debug(f"📊 [SEARCH BALANCE] Type distribution: {non_zero_types}")
                    
                    # 단일 타입만 검색된 경우 경고 (로깅 최적화)
                    non_zero_types = [t for t, c in type_distribution.items() if c > 0]
                    if len(non_zero_types) == 1:
                        single_type = non_zero_types[0]
                        self.logger.warning(
                            f"⚠️ [TYPE DIVERSITY] 단일 타입만 검색됨: {single_type} ({type_distribution[single_type]}개)"
                        )
                    elif len(non_zero_types) == 0:
                        self.logger.warning("⚠️ [TYPE DIVERSITY] 검색 결과가 없습니다")
                    elif debug_mode:
                        # 타입 다양성 점수 계산 (디버그 모드에서만)
                        total_docs = sum(type_distribution.values())
                        if total_docs > 0:
                            import math
                            entropy = 0.0
                            for count in type_distribution.values():
                                if count > 0:
                                    p = count / total_docs
                                    entropy -= p * math.log2(p)
                            max_entropy = math.log2(len(non_zero_types)) if len(non_zero_types) > 1 else 1.0
                            diversity_score = entropy / max_entropy if max_entropy > 0 else 0.0
                            self.logger.debug(
                                f"✅ [TYPE DIVERSITY] 타입 다양성 점수: {diversity_score:.2f} "
                                f"(검색된 타입: {len(non_zero_types)}개, 총 문서: {total_docs}개)"
                            )
                    
                    # 균형 조정된 결과 생성
                    semantic_k = search_params.get("semantic_k", WorkflowConstants.SEMANTIC_SEARCH_K)
                    keyword_k = search_params.get("keyword_k", WorkflowConstants.KEYWORD_SEARCH_K)
                    balanced_results = self.result_balancer.balance_search_results(
                        grouped_results,
                        total_limit=semantic_k + keyword_k
                    )
                    
                    # 균형 조정된 결과를 semantic_results와 keyword_results로 재분배
                    if balanced_results:
                        semantic_results_balanced = [
                            doc for doc in balanced_results 
                            if doc.get("relevance_score", 0.0) >= 0.5
                        ]
                        keyword_results_balanced = [
                            doc for doc in balanced_results 
                            if doc.get("relevance_score", 0.0) < 0.5 or doc not in semantic_results_balanced
                        ]
                        
                        # 기존 결과와 병합 (중복 제거 - 성능 최적화)
                        existing_ids = {id(doc) for doc in semantic_results + keyword_results}
                        
                        semantic_results = semantic_results + [
                            doc for doc in semantic_results_balanced 
                            if id(doc) not in existing_ids
                        ]
                        keyword_results = keyword_results + [
                            doc for doc in keyword_results_balanced 
                            if id(doc) not in existing_ids
                        ]
                        
                        semantic_count = len(semantic_results)
                        keyword_count = len(keyword_results)
                        
                        if debug_mode:
                            self.logger.debug(
                                f"✅ [SEARCH BALANCE] 균형 조정 완료: "
                                f"semantic={semantic_count}, keyword={keyword_count}"
                            )
                except Exception as e:
                    if debug_mode:
                        self.logger.warning(f"검색 결과 균형 조정 실패 (기존 결과 사용): {e}")

            ensure_state_group(state, "search")

            if debug_mode:
                self.logger.debug(f"PARALLEL SEARCH: Before save - semantic_results={len(semantic_results)}, keyword_results={len(keyword_results)}")

            self._set_state_value(state, "semantic_results", semantic_results)
            self._set_state_value(state, "keyword_results", keyword_results)
            self._set_state_value(state, "semantic_count", semantic_count)
            self._set_state_value(state, "keyword_count", keyword_count)
            
            # State 구조 일관성 확보: retrieved_docs를 헬퍼 함수로 저장
            merged_docs = semantic_results + keyword_results
            set_retrieved_docs(state, merged_docs)
            
            # 검색 결과 캐싱 저장 (재시도가 아닌 경우만)
            if not is_retry:
                cache_key = self._generate_search_cache_key(
                    optimized_queries, search_params, original_query
                )
                if cache_key and (semantic_results or keyword_results):
                    self._cache_search_results(cache_key, {
                        'semantic_results': semantic_results,
                        'keyword_results': keyword_results,
                        'semantic_count': semantic_count,
                        'keyword_count': keyword_count
                    })

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

        except (TimeoutError, FutureCancelledError) as timeout_err:
            # CancelledError는 정상적인 취소이므로 경고만 로깅
            if isinstance(timeout_err, FutureCancelledError):
                self.logger.warning(f"⚠️ 병렬 검색 취소 발생: {timeout_err}")
            else:
                self.logger.warning(f"⚠️ 병렬 검색 타임아웃 발생: {timeout_err}")
            
            # 🔥 개선 3: 검색 결과가 0개일 때 즉시 반환 (timeout 방지)
            if semantic_count == 0 and keyword_count == 0:
                self.logger.warning(
                    "⚠️ [SEARCH TIMEOUT PREVENTION] 검색 결과가 0개입니다. "
                    "타임아웃 방지를 위해 즉시 반환합니다."
                )
                ensure_state_group(state, "search")
                state["search"]["semantic_results"] = []
                state["search"]["keyword_results"] = []
                state["search"]["semantic_count"] = 0
                state["search"]["keyword_count"] = 0
                return state
            
            self.logger.info("🔄 순차 검색으로 폴백 시도 중...")
            try:
                return self.fallback_sequential_search(state)
            except Exception as fallback_err:
                self.logger.error(f"❌ 순차 검색 폴백도 실패: {fallback_err}", exc_info=True)
                # 최후의 수단: 빈 결과로라도 계속 진행
                self.logger.warning("⚠️ 최소한의 결과로 계속 진행합니다.")
                semantic_results, semantic_count = [], 0
                keyword_results, keyword_count = [], 0
                # 미완료된 future 취소 시도
                try:
                    if 'semantic_future' in locals() and not semantic_future.done():
                        semantic_future.cancel()
                    if 'keyword_future' in locals() and not keyword_future.done():
                        keyword_future.cancel()
                except Exception:
                    pass
                # 빈 결과라도 state에 저장하여 워크플로우가 계속 진행되도록 함
                ensure_state_group(state, "search")
                state["search"]["semantic_results"] = semantic_results
                state["search"]["keyword_results"] = keyword_results
                state["search"]["semantic_count"] = semantic_count
                state["search"]["keyword_count"] = keyword_count
                return state
        except (FutureCancelledError, Exception) as e:
            # CancelledError는 정상적인 취소이므로 경고만 로깅하고 빈 결과 반환
            if isinstance(e, FutureCancelledError):
                self.logger.warning(f"⚠️ 병렬 검색 취소 발생: {e}")
                # 취소된 경우 빈 결과로라도 계속 진행
                semantic_results, semantic_count = [], 0
                keyword_results, keyword_count = [], 0
                ensure_state_group(state, "search")
                state["search"]["semantic_results"] = semantic_results
                state["search"]["keyword_results"] = keyword_results
                state["search"]["semantic_count"] = semantic_count
                state["search"]["keyword_count"] = keyword_count
                return state
            
            self.logger.error(f"❌ 병렬 검색 중 예상치 못한 오류: {e}", exc_info=True)
            self.logger.info("🔄 순차 검색으로 폴백 시도 중...")
            try:
                return self.fallback_sequential_search(state)
            except Exception as fallback_err:
                self.logger.error(f"❌ 순차 검색 폴백도 실패: {fallback_err}", exc_info=True)
                # 최후의 수단: 빈 결과로라도 계속 진행
                self.logger.warning("⚠️ 최소한의 결과로 계속 진행합니다.")
                semantic_results, semantic_count = [], 0
                keyword_results, keyword_count = [], 0
                ensure_state_group(state, "search")
                state["search"]["semantic_results"] = semantic_results
                state["search"]["keyword_results"] = keyword_results
                state["search"]["semantic_count"] = semantic_count
                state["search"]["keyword_count"] = keyword_count
                return state

        debug_mode = os.getenv("DEBUG_SEARCH", "false").lower() == "true"

        if debug_mode:
            if "search" in state and isinstance(state.get("search"), dict):
                final_search = state["search"]
                final_semantic = len(final_search.get("semantic_results", []))
                final_keyword = len(final_search.get("keyword_results", []))
                self.logger.debug(f"[DEBUG] execute_searches_parallel: Returning state with search group - semantic_results={final_semantic}, keyword_results={final_keyword}")
                self.logger.debug(f"[DEBUG] execute_searches_parallel: Returning state keys={list(state.keys()) if isinstance(state, dict) else 'N/A'}")
            else:
                self.logger.debug("[DEBUG] execute_searches_parallel: WARNING - Returning state WITHOUT search group!")
                self.logger.debug(f"[DEBUG] execute_searches_parallel: Returning state keys={list(state.keys()) if isinstance(state, dict) else 'N/A'}")

        return state

    def _execute_semantic_search_single(
        self,
        query: str,
        k: int,
        extracted_keywords: Optional[List[str]] = None,
        original_query: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """단일 semantic 검색 실행 (Multi-Query용)"""
        if not query or not query.strip():
            return [], 0
        
        try:
            results, count = self.search_handler.semantic_search(
                query,
                k=k,
                extracted_keywords=extracted_keywords
            )
            return results, count
        except Exception as e:
            self.logger.warning(f"Single semantic search failed for '{query[:30]}...': {e}")
            return [], 0

    def execute_semantic_search(
        self,
        optimized_queries: Dict[str, Any],
        search_params: Dict[str, Any],
        original_query: str = "",
        extracted_keywords: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """의미적 검색 실행"""
        self.logger.info("🔍 [EXECUTE_SEMANTIC_SEARCH] 메서드 호출됨")
        self.logger.info(f"🔍 [EXECUTE_SEMANTIC_SEARCH] original_query: {original_query[:50] if original_query else 'N/A'}...")
        semantic_results = []
        semantic_count = 0

        semantic_query = optimized_queries.get("semantic_query", "") if optimized_queries else ""
        semantic_k = search_params.get("semantic_k", WorkflowConstants.SEMANTIC_SEARCH_K) if search_params else WorkflowConstants.SEMANTIC_SEARCH_K
        expanded_keywords = optimized_queries.get("expanded_keywords", []) if optimized_queries else []
        
        # 빈 쿼리 검증: semantic_query가 비어있으면 original_query 사용, 둘 다 비어있으면 빈 결과 반환
        if not semantic_query or not str(semantic_query).strip():
            if original_query and original_query.strip():
                semantic_query = original_query
                if optimized_queries:
                    optimized_queries["semantic_query"] = original_query
                self.logger.info(f"🔍 [EXECUTE_SEMANTIC_SEARCH] semantic_query가 비어있어 original_query 사용: '{original_query[:50]}...'")
            else:
                self.logger.warning("⚠️ [EXECUTE_SEMANTIC_SEARCH] semantic_query와 original_query가 모두 비어있어 검색을 수행할 수 없습니다.")
                return [], 0
        
        # 개선: textToSQL 라우팅 확인 및 적용 (우선순위 1)
        if original_query and original_query.strip():
            from core.search.connectors.legal_data_connector_v2 import route_query, LegalDataConnectorV2
            route = route_query(original_query)
            self.logger.info(f"🔍 [TEXT2SQL SEMANTIC] route_query result: '{route}' for query: '{original_query[:50]}...'")
            if route == "text2sql":
                self.logger.info(f"🔍 [TEXT2SQL SEMANTIC] Detected text2sql route for semantic search: '{original_query[:50]}...'")
                try:
                    data_connector = LegalDataConnectorV2()
                    text2sql_results = data_connector.search_documents(original_query, limit=semantic_k)
                    if text2sql_results:
                        semantic_results.extend(text2sql_results)
                        semantic_count += len(text2sql_results)
                        self.logger.info(f"✅ [TEXT2SQL SEMANTIC] {len(text2sql_results)}개 결과 검색 성공 (semantic_results에 추가)")
                    else:
                        self.logger.warning("⚠️ [TEXT2SQL SEMANTIC] 검색 결과 없음")
                except Exception as e:
                    self.logger.warning(f"⚠️ [TEXT2SQL SEMANTIC] 검색 실패: {e}")

        if extracted_keywords is None:
            extracted_keywords = []

        self.logger.info(
            f"🔍 [QUERY USAGE] semantic_query from optimized_queries: '{semantic_query[:100]}...' "
            f"(length={len(semantic_query)}, expanded_keywords_count={len(expanded_keywords) if expanded_keywords else 0})"
        )
        if expanded_keywords:
            self.logger.info(
                f"🔍 [QUERY USAGE] expanded_keywords: {expanded_keywords[:10]} "
                f"(total={len(expanded_keywords)}, included_in_query={len([t for t in expanded_keywords if t in semantic_query])})"
            )

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
                    self.logger.info(
                        f"🔍 [QUERY ENHANCEMENT] Enhanced semantic query: "
                        f"original='{semantic_query[:80]}...', "
                        f"enhanced='{enhanced_semantic_query[:100]}...', "
                        f"added_keywords={new_keywords[:3]}"
                    )
                else:
                    self.logger.info(
                        f"🔍 [QUERY ENHANCEMENT] No new keywords to add (all keywords already in query): "
                        f"query='{semantic_query[:80]}...'"
                    )
            else:
                self.logger.debug("🔍 [QUERY ENHANCEMENT] No core keywords extracted from extracted_keywords")
        else:
            self.logger.info(
                f"🔍 [QUERY ENHANCEMENT] Using original semantic_query (no extracted_keywords): "
                f"'{semantic_query[:100]}...'"
            )

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

        # 조기 종료 체크: 메인 검색만으로 충분한 경우
        max_results_threshold = semantic_k * 3
        if len(semantic_results) >= max_results_threshold:
            self.logger.info(
                f"⏭️ [EARLY EXIT] Main semantic search sufficient: "
                f"{len(semantic_results)} >= {max_results_threshold}, skipping additional searches"
            )
            return semantic_results, semantic_count

        if original_query and original_query.strip():
            enhanced_original_query = original_query
            if extracted_keywords and len(extracted_keywords) > 0:
                core_keywords = [str(kw) for kw in extracted_keywords[:3] if isinstance(kw, str)]
                if core_keywords:
                    query_keywords = set(original_query.split())
                    new_keywords = [kw for kw in core_keywords if kw not in query_keywords]
                    if new_keywords:
                        enhanced_original_query = f"{original_query} {' '.join(new_keywords[:2])}"

            # 조기 종료 체크: original_query 검색 전 확인
            if len(semantic_results) >= max_results_threshold:
                self.logger.info(
                    f"⏭️ [EARLY EXIT] Skipping original query search: "
                    f"{len(semantic_results)} >= {max_results_threshold}"
                )
            else:
                # 중복 검색 제거: semantic_query와 original_query가 같으면 스킵
                semantic_query_normalized = str(semantic_query).strip().lower()
                original_query_normalized = str(original_query).strip().lower()
                
                if semantic_query_normalized == original_query_normalized:
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            "⏭️ [SKIP DUPLICATE] semantic_query와 original_query가 동일하여 "
                            "original_query 검색 스킵"
                        )
                else:
                    original_semantic, original_count = self.search_handler.semantic_search(
                        enhanced_original_query,
                        k=semantic_k // 2,
                        extracted_keywords=extracted_keywords
                    )
                    semantic_results.extend(original_semantic)
                    semantic_count += original_count
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            f"🔍 [DEBUG] Original query semantic search: {original_count} results "
                            f"(query: '{enhanced_original_query[:50]}...')"
                        )
                
                # 다시 조기 종료 체크
                if len(semantic_results) >= max_results_threshold:
                    self.logger.info(
                        f"⏭️ [EARLY EXIT] After original query search: "
                        f"{len(semantic_results)} >= {max_results_threshold}, skipping multi-query"
                    )
                    return semantic_results, semantic_count

        # Multi-Query Retrieval 적용 (LLM 기반 질문 재작성)
        # 개선: 조기 종료 조건 및 병렬 실행
        multi_queries = optimized_queries.get("multi_queries", [])
        min_results_threshold = semantic_k  # 최소 결과 수
        max_results_threshold = semantic_k * 3  # 최대 결과 수 제한
        
        # 조기 종료: 이미 충분한 결과가 있으면 멀티 쿼리 스킵
        if len(semantic_results) >= max_results_threshold:
            self.logger.info(f"⏭️ [MULTI-QUERY] Skipping multi-query: already have {len(semantic_results)} results (threshold: {max_results_threshold})")
        elif multi_queries and len(multi_queries) > 1:
            # 개선: Multi-Query 병렬 실행
            # 개선: Multi-Query 병렬 실행 (성능 최적화: 결과가 충분하면 스킵)
            max_semantic_results_before_multi = semantic_k * 2  # Multi-Query 전 최대 결과 수
            if len(semantic_results) >= max_semantic_results_before_multi:
                self.logger.info(
                    f"⚡ [PERFORMANCE] Skipping multi-query search "
                    f"(already have {len(semantic_results)} results, threshold={max_semantic_results_before_multi})"
                )
                multi_queries_to_process = []
            else:
                multi_queries_to_process = multi_queries[1:]  # 첫 번째는 이미 처리됨
                # 최대 처리 개수 제한 (성능 최적화)
                max_multi_queries = min(len(multi_queries_to_process), 2)  # 최대 2개로 감소 (3 → 2)
                multi_queries_to_process = multi_queries_to_process[:max_multi_queries]
            
            if multi_queries_to_process:
                self.logger.debug(f"[MULTI-QUERY] Found {len(multi_queries)} queries, processing {len(multi_queries_to_process)} in parallel...")
                self.logger.info(f"🔍 [MULTI-QUERY] Found {len(multi_queries)} queries, processing {len(multi_queries_to_process)} in parallel...")
                
                # 중복 제거를 위한 seen_ids 및 내용 유사도 추적
                seen_ids = set()
                seen_contents = {}  # content_hash -> doc
                
                # 원본 쿼리 결과의 ID와 내용 해시 수집
                for doc in semantic_results:
                    doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id")
                    if doc_id:
                        seen_ids.add(doc_id)
                    # 내용 해시로도 중복 확인
                    content = doc.get("content") or doc.get("text", "")
                    if content:
                        import hashlib
                        content_hash = hashlib.md5(content[:200].encode('utf-8')).hexdigest()
                        if content_hash not in seen_contents:
                            seen_contents[content_hash] = doc
                from concurrent.futures import as_completed
                import threading
                
                # 스레드 안전을 위한 락
                results_lock = threading.Lock()
                
                def process_multi_query(mq: str) -> List[Dict[str, Any]]:
                    """단일 Multi-Query 처리 함수"""
                    if not mq or not mq.strip() or mq == semantic_query:
                        return []
                    
                    try:
                        # 성능 최적화: 결과 수 제한
                        mq_semantic, mq_count = self.search_handler.semantic_search(
                            mq,
                            k=max(5, semantic_k // 3),  # 최소 5개, 최대 semantic_k // 3
                            extracted_keywords=extracted_keywords
                        )
                        return mq_semantic
                    except Exception as e:
                        self.logger.warning(f"⚠️ [MULTI-QUERY] Query '{mq[:30]}...' failed: {e}")
                        return []
                
                # 배치 검색 최적화: 여러 쿼리를 한 번에 배치로 검색
                if hasattr(self.search_handler, 'semantic_search_batch') and len(multi_queries_to_process) > 1:
                    try:
                        self.logger.info(f"✅ [BATCH SEARCH] Processing {len(multi_queries_to_process)} queries in batch")
                        batch_k = max(5, semantic_k // 3)
                        batch_results = self.search_handler.semantic_search_batch(
                            queries=multi_queries_to_process,
                            k=batch_k,
                            extracted_keywords=extracted_keywords
                        )
                        
                        # 배치 결과 처리
                        for query, (mq_semantic, mq_count) in zip(multi_queries_to_process, batch_results):
                            if mq_semantic:
                                # 조기 종료 확인
                                if len(semantic_results) >= max_results_threshold:
                                    self.logger.info(
                                        f"⏭️ [MULTI-QUERY] Early exit: {len(semantic_results)} results "
                                        f"(threshold: {max_results_threshold})"
                                    )
                                    break
                                
                                # 중복 제거 및 결과 추가
                                added_count = 0
                                for doc in mq_semantic:
                                    doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id")
                                    content = doc.get("content") or doc.get("text", "")
                                    
                                    # ID 기반 중복 확인
                                    if doc_id and doc_id in seen_ids:
                                        continue
                                    
                                    # 내용 기반 중복 확인
                                    is_duplicate = False
                                    if content:
                                        import hashlib
                                        content_hash = hashlib.md5(content[:200].encode('utf-8')).hexdigest()
                                        if content_hash in seen_contents:
                                            existing_doc = seen_contents[content_hash]
                                            existing_content = existing_doc.get("content") or existing_doc.get("text", "")
                                            if len(content) > 0 and len(existing_content) > 0:
                                                common_chars = len(set(content[:100]) & set(existing_content[:100]))
                                                similarity = common_chars / max(len(set(content[:100])), len(set(existing_content[:100])), 1)
                                                if similarity > 0.8:
                                                    is_duplicate = True
                                        else:
                                            seen_contents[content_hash] = doc
                                    
                                    if not is_duplicate:
                                        semantic_results.append(doc)
                                        if doc_id:
                                            seen_ids.add(doc_id)
                                        added_count += 1
                                
                                if added_count > 0:
                                    self.logger.info(
                                        f"🔍 [MULTI-QUERY] Query '{query[:30]}...' added {added_count} unique results"
                                    )
                    except Exception as e:
                        self.logger.warning(f"⚠️ [BATCH SEARCH] Batch search failed: {e}, falling back to parallel search")
                        # 폴백: 기존 병렬 처리 방식 사용
                        with ThreadPoolExecutor(max_workers=min(len(multi_queries_to_process), 4)) as executor:
                            future_to_query = {
                                executor.submit(process_multi_query, mq): mq 
                                for mq in multi_queries_to_process
                            }
                            
                            for future in as_completed(future_to_query, timeout=20):
                                query = future_to_query[future]
                                try:
                                    mq_semantic = future.result()
                                    if mq_semantic:
                                        with results_lock:
                                            if len(semantic_results) >= max_results_threshold:
                                                break
                                            
                                            added_count = 0
                                            for doc in mq_semantic:
                                                doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id")
                                                content = doc.get("content") or doc.get("text", "")
                                                
                                                if doc_id and doc_id in seen_ids:
                                                    continue
                                                
                                                is_duplicate = False
                                                if content:
                                                    import hashlib
                                                    content_hash = hashlib.md5(content[:200].encode('utf-8')).hexdigest()
                                                    if content_hash in seen_contents:
                                                        existing_doc = seen_contents[content_hash]
                                                        existing_content = existing_doc.get("content") or existing_doc.get("text", "")
                                                        if len(content) > 0 and len(existing_content) > 0:
                                                            common_chars = len(set(content[:100]) & set(existing_content[:100]))
                                                            similarity = common_chars / max(len(set(content[:100])), len(set(existing_content[:100])), 1)
                                                            if similarity > 0.8:
                                                                is_duplicate = True
                                                    else:
                                                        seen_contents[content_hash] = doc
                                                
                                                if not is_duplicate:
                                                    semantic_results.append(doc)
                                                    if doc_id:
                                                        seen_ids.add(doc_id)
                                                    added_count += 1
                                            
                                            if added_count > 0:
                                                self.logger.info(
                                                    f"🔍 [MULTI-QUERY] Query '{query[:30]}...' added {added_count} unique results"
                                                )
                                except Exception as e:
                                    self.logger.warning(f"⚠️ [MULTI-QUERY] Query '{query[:30]}...' failed: {e}")
                else:
                    # 단일 쿼리이거나 배치 검색 미지원: 기존 병렬 처리 방식 사용
                    with ThreadPoolExecutor(max_workers=min(len(multi_queries_to_process), 4)) as executor:
                        future_to_query = {
                            executor.submit(process_multi_query, mq): mq 
                            for mq in multi_queries_to_process
                        }
                        
                        for future in as_completed(future_to_query, timeout=20):
                            query = future_to_query[future]
                            try:
                                mq_semantic = future.result()
                                if mq_semantic:
                                    with results_lock:
                                        if len(semantic_results) >= max_results_threshold:
                                            break
                                        
                                        added_count = 0
                                        for doc in mq_semantic:
                                            doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id")
                                            content = doc.get("content") or doc.get("text", "")
                                            
                                            if doc_id and doc_id in seen_ids:
                                                continue
                                            
                                            is_duplicate = False
                                            if content:
                                                import hashlib
                                                content_hash = hashlib.md5(content[:200].encode('utf-8')).hexdigest()
                                                if content_hash in seen_contents:
                                                    existing_doc = seen_contents[content_hash]
                                                    existing_content = existing_doc.get("content") or existing_doc.get("text", "")
                                                    if len(content) > 0 and len(existing_content) > 0:
                                                        common_chars = len(set(content[:100]) & set(existing_content[:100]))
                                                        similarity = common_chars / max(len(set(content[:100])), len(set(existing_content[:100])), 1)
                                                        if similarity > 0.8:
                                                            is_duplicate = True
                                                else:
                                                    seen_contents[content_hash] = doc
                                            
                                            if not is_duplicate:
                                                semantic_results.append(doc)
                                                if doc_id:
                                                    seen_ids.add(doc_id)
                                                added_count += 1
                                        
                                        if added_count > 0:
                                            self.logger.info(
                                                f"🔍 [MULTI-QUERY] Query '{query[:30]}...' added {added_count} unique results"
                                            )
                            except Exception as e:
                                self.logger.warning(f"⚠️ [MULTI-QUERY] Query '{query[:30]}...' failed: {e}")
            
            # 최종 결과 수 업데이트
            semantic_count = len(semantic_results)
            self.logger.info(
                f"✅ [MULTI-QUERY] Processing completed: {semantic_count} total results "
                f"(from {len(multi_queries_to_process)} queries)"
            )
        
        # 키워드 쿼리로 추가 의미적 검색 (Multi-Query가 없거나 부족한 경우)
        # 성능 최적화: 결과 수가 이미 충분하면 스킵
        max_semantic_results = semantic_k * 3  # 최대 결과 수 제한 (예: 12 * 3 = 36)
        if len(semantic_results) < max_semantic_results and (not multi_queries or len(multi_queries) <= 1):
            keyword_queries = optimized_queries.get("keyword_queries", [])[:1]  # 2개 → 1개로 감소
            for i, kw_query in enumerate(keyword_queries, 1):
                if kw_query and kw_query.strip() and kw_query != semantic_query:
                    # 성능 최적화: 결과 수 제한
                    kw_semantic, kw_count = self.search_handler.semantic_search(
                        kw_query,
                        k=max(5, semantic_k // 3),  # 최소 5개, 최대 semantic_k // 3
                        extracted_keywords=extracted_keywords
                    )
                    semantic_results.extend(kw_semantic)
                    semantic_count += kw_count
                    self.logger.info(
                        f"🔍 [DEBUG] Keyword-based semantic search #{i}: {kw_count} results (query: '{kw_query[:50]}...')"
                    )
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f"[DEBUG] _execute_semantic_search_internal: Added {kw_count} results from keyword query #{i}")
                    
                    # 성능 최적화: 결과 수가 이미 충분하면 중단
                    if len(semantic_results) >= max_semantic_results:
                        self.logger.info(f"⚡ [PERFORMANCE] Stopping keyword-based search (already have {len(semantic_results)} results)")
                        break

        # Phase 1 + Phase 2: 타입별 별도 검색 수행 및 쿼리 다변화 적용 (타입 다양성 개선)
        # 개선: 타입별 검색 전 조건 체크 (조기 스킵)
        max_semantic_results = semantic_k * 3  # 최대 결과 수 제한
        
        # 조건 1: 결과가 충분하면 스킵 (60% 이상으로 완화하여 더 빨리 스킵)
        should_skip_type_diversity = False
        if len(semantic_results) >= max_semantic_results * 0.6:
            self.logger.info(
                f"⚡ [PERFORMANCE] Skipping type diversity search "
                f"(already have {len(semantic_results)} results, threshold={max_semantic_results * 0.6:.0f})"
            )
            should_skip_type_diversity = True
        else:
            # 조건 2: 타입 분포 확인
            def _calculate_type_distribution(docs):
                """타입 분포 계산"""
                type_counts = {}
                for doc in docs:
                    doc_type = (
                        doc.get("type") or
                        doc.get("source_type") or
                        (doc.get("metadata", {}).get("source_type") if isinstance(doc.get("metadata"), dict) else "") or
                        "unknown"
                    )
                    # 타입 매핑
                    type_mapping = {
                        "statute_article": "statute",
                        "case_paragraph": "case",
                        "decision_paragraph": "decision",
                        "interpretation_paragraph": "interpretation"
                    }
                    mapped_type = type_mapping.get(doc_type, doc_type)
                    type_counts[mapped_type] = type_counts.get(mapped_type, 0) + 1
                return type_counts
            
            type_distribution = _calculate_type_distribution(semantic_results)
            
            # 이미 2개 이상 타입이면 스킵 (3개 → 2개로 완화)
            if len(type_distribution) >= 2:
                self.logger.info(
                    f"⚡ [PERFORMANCE] Skipping type diversity search "
                    f"(sufficient type diversity: {len(type_distribution)} types)"
                )
                should_skip_type_diversity = True
        
        type_specific_results = {}
        type_specific_count = 0
        
        if not should_skip_type_diversity:
            document_types = {
                "statute_article": "statute",
                "case_paragraph": "case",
                "decision_paragraph": "decision",
                "interpretation_paragraph": "interpretation"
            }
            
            # semantic_search_engine 확인 (여러 방법 시도)
            semantic_engine = None
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("🔍 [TYPE DIVERSITY] semantic_search_engine 확인 시작")
        self.logger.info(f"🔍 [TYPE DIVERSITY] self.semantic_search_engine: {self.semantic_search_engine is not None}")
        
        # SemanticSearchEngineV2 인스턴스인지 확인하는 헬퍼 함수
        def is_semantic_search_engine(obj):
            """SemanticSearchEngineV2 인스턴스인지 확인"""
            if obj is None:
                return False
            # 타입 이름으로 확인 (import 없이)
            type_name = type(obj).__name__
            if type_name == 'SemanticSearchEngineV2':
                return True
            # hasattr로 search 메서드 확인
            if hasattr(obj, 'search') and callable(getattr(obj, 'search', None)):
                # 함수가 아닌 인스턴스인지 확인
                if not callable(obj) or hasattr(obj, '__class__'):
                    return True
            return False
        
        if self.semantic_search_engine and is_semantic_search_engine(self.semantic_search_engine):
            semantic_engine = self.semantic_search_engine
            self.logger.info(f"✅ [TYPE DIVERSITY] semantic_search_engine from self: {type(semantic_engine).__name__}")
        elif hasattr(self.search_handler, 'semantic_search_engine') and self.search_handler.semantic_search_engine:
            candidate = self.search_handler.semantic_search_engine
            if is_semantic_search_engine(candidate):
                semantic_engine = candidate
                self.logger.info(f"✅ [TYPE DIVERSITY] semantic_search_engine from search_handler: {type(semantic_engine).__name__}")
            else:
                self.logger.warning(f"⚠️ [TYPE DIVERSITY] search_handler.semantic_search_engine is not a valid engine: {type(candidate).__name__}")
        elif hasattr(self.search_handler, 'semantic_search') and self.search_handler.semantic_search:
            candidate = self.search_handler.semantic_search
            # semantic_search가 함수인지 확인
            if callable(candidate) and not is_semantic_search_engine(candidate):
                self.logger.warning("⚠️ [TYPE DIVERSITY] search_handler.semantic_search is a function, not an engine instance")
            elif is_semantic_search_engine(candidate):
                semantic_engine = candidate
                self.logger.info(f"✅ [TYPE DIVERSITY] semantic_search_engine from search_handler.semantic_search: {type(semantic_engine).__name__}")
            else:
                self.logger.warning(f"⚠️ [TYPE DIVERSITY] search_handler.semantic_search is not a valid engine: {type(candidate).__name__}")
        else:
            self.logger.warning("⚠️ [TYPE DIVERSITY] semantic_search_engine not found")
            self.logger.warning(f"   - self.semantic_search_engine: {self.semantic_search_engine} ({type(self.semantic_search_engine).__name__ if self.semantic_search_engine else 'None'})")
            self.logger.warning(f"   - search_handler.semantic_search_engine: {getattr(self.search_handler, 'semantic_search_engine', 'N/A')}")
            self.logger.warning(f"   - search_handler.semantic_search: {getattr(self.search_handler, 'semantic_search', 'N/A')}")
        
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"🔍 [TYPE DIVERSITY] semantic_engine 확인 결과: {semantic_engine is not None}")
        
        if semantic_engine and not should_skip_type_diversity:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("✅ [TYPE DIVERSITY] semantic_engine 발견, 타입별 검색 진행")
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
            
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("🔍 [TYPE DIVERSITY] 타입별 검색 시작 (병렬 실행)")
                self.logger.debug(f"🔍 [TYPE DIVERSITY] 검색할 타입: {list(document_types.keys())}")
            
            # 타입별 검색 병렬화
            def search_by_type(doc_type, query_type):
                """타입별 검색 함수 (병렬 실행용)"""
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"🔍 [TYPE DIVERSITY] {doc_type} 검색 시작 (query_type={query_type})")
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
                    
                    # 우선순위 5: 타입별 검색 강화 (성능 최적화: k 값 감소)
                    k_per_type = 15  # 20 → 15로 감소 (성능 개선)
                    min_score_by_type = {
                        "statute_article": 0.4,  # 법령 조문: 낮은 임계값
                        "case_paragraph": 0.5,
                        "decision_paragraph": 0.5,
                        "interpretation_paragraph": 0.5
                    }
                    min_score = min_score_by_type.get(doc_type, 0.5)
                    
                    self.logger.info(f"🔍 [TYPE DIVERSITY] {doc_type} 검색 시작 (k={k_per_type}, threshold={min_score}, source_types=[{doc_type}])")
                    type_results = semantic_engine.search(
                        search_query,
                        k=k_per_type,  # k * 2 → k로 감소 (성능 개선)
                        source_types=[doc_type],  # 타입별 필터 적용
                        similarity_threshold=min_score,  # 타입별 최소 점수
                        min_results=1,  # 최소 1개는 보장
                        disable_retry=False  # 재시도 로직 활성화 (임계값 자동 조정)
                    )
                    
                    # 품질 필터링 (타입별 최소 점수)
                    if type_results:
                        filtered_results = [
                            doc for doc in type_results
                            if doc.get("similarity", doc.get("relevance_score", 0.0)) >= min_score
                        ]
                        # 상위 k_per_type개 선택
                        type_results = filtered_results[:k_per_type]
                        self.logger.info(f"🔍 [TYPE DIVERSITY] {doc_type}: {len(type_results)}개 결과 (필터링 후)")
                    
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
                    
                    # 개선: 일반 키워드 재시도 제거 (3단계 → 2단계로 단순화)
                    # 결과가 없으면 샘플링으로 대체
                    self.logger.info(f"🔍 [TYPE DIVERSITY] {doc_type}: 최종 {len(type_results)}개 검색됨")
                    
                    # 최종 방안: 검색 결과가 없으면 타입별 샘플링으로 대체
                    if not type_results:
                        self.logger.info(f"🔍 [TYPE DIVERSITY] {doc_type}: 샘플링으로 대체 시도")
                        try:
                            type_results = self._get_type_sample(semantic_engine, doc_type, k=2)
                            if type_results:
                                if self.logger.isEnabledFor(logging.DEBUG):
                                    self.logger.debug(f"✅ [TYPE DIVERSITY] {doc_type}: 샘플링으로 {len(type_results)}개 가져옴")
                                # 샘플링된 문서 상세 로그
                                for idx, sample_doc in enumerate(type_results, 1):
                                    self.logger.debug(
                                        f"   샘플 {idx}: id={sample_doc.get('id')}, "
                                        f"source_type={sample_doc.get('source_type')}, "
                                        f"type={sample_doc.get('type')}, "
                                        f"relevance_score={sample_doc.get('relevance_score')}"
                                    )
                            else:
                                self.logger.warning(f"⚠️ [TYPE DIVERSITY] {doc_type}: 샘플링 결과도 없음")
                        except Exception as e:
                            self.logger.error(f"❌ [TYPE DIVERSITY] {doc_type} 샘플링 실패: {e}")
                            import traceback
                            self.logger.debug(f"샘플링 예외 상세: {traceback.format_exc()}")
                    
                    return doc_type, type_results
                except Exception as e:
                    self.logger.error(f"❌ [TYPE DIVERSITY] 타입별 검색 실패 ({doc_type}): {e}")
                    import traceback
                    self.logger.debug(f"타입별 검색 예외 상세: {traceback.format_exc()}")
                    return doc_type, []
            
            # 병렬 실행
            with ThreadPoolExecutor(max_workers=len(document_types)) as executor:
                futures = {
                    executor.submit(search_by_type, doc_type, query_type): doc_type
                    for doc_type, query_type in document_types.items()
                }
                
                for future in futures:
                    doc_type = futures[future]
                    try:
                        result_doc_type, type_results = future.result(timeout=15)  # 30초 → 15초로 최적화
                        if type_results:
                            type_specific_results[result_doc_type] = type_results
                            semantic_results.extend(type_results)
                            type_specific_count += len(type_results)
                            if self.logger.isEnabledFor(logging.DEBUG):
                                self.logger.debug(
                                    f"✅ [TYPE DIVERSITY] {result_doc_type}: {len(type_results)}개 검색 성공 "
                                    f"(검색 결과에 추가됨, 총 semantic_results: {len(semantic_results)}개)"
                                )
                        else:
                            if self.logger.isEnabledFor(logging.DEBUG):
                                self.logger.debug(
                                    f"⚠️ [TYPE DIVERSITY] {result_doc_type}: 검색 결과 없음 (데이터 없음 또는 쿼리 관련성 낮음)"
                                )
                    except Exception as e:
                        self.logger.error(f"❌ [TYPE DIVERSITY] {doc_type} 병렬 검색 실패: {e}")
                
                # 타입별 검색 결과 로깅
                if 'type_specific_count' in locals() and type_specific_count > 0:
                    self.logger.info(
                        f"✅ [TYPE DIVERSITY] 타입별 검색 완료: 총 {type_specific_count}개 추가 "
                        f"(총 semantic_results: {len(semantic_results)}개)"
                    )
        elif not semantic_engine:
            self.logger.warning("⚠️ [TYPE DIVERSITY] semantic_search_engine을 찾을 수 없어 타입별 검색을 수행할 수 없습니다")
            self.logger.warning(f"⚠️ [TYPE DIVERSITY] semantic_search_engine 확인: self.semantic_search_engine={self.semantic_search_engine is not None}")
            if hasattr(self, 'search_handler'):
                self.logger.warning(f"⚠️ [TYPE DIVERSITY] search_handler 확인: {self.search_handler is not None}")
                if self.search_handler:
                    self.logger.warning(f"⚠️ [TYPE DIVERSITY] search_handler.semantic_search 확인: {hasattr(self.search_handler, 'semantic_search')}")
                    if hasattr(self.search_handler, 'semantic_search_engine'):
                        self.logger.warning(f"⚠️ [TYPE DIVERSITY] search_handler.semantic_search_engine 확인: {self.search_handler.semantic_search_engine is not None}")
        
        semantic_count += type_specific_count
        
        if type_specific_count > 0:
            type_distribution = dict((k, len(v)) for k, v in type_specific_results.items())
            self.logger.info(
                f"✅ [TYPE DIVERSITY] 타입별 검색 완료: 총 {type_specific_count}개 추가 "
                f"(타입별 분포: {type_distribution})"
            )
            # interpretation_paragraph 확인
            if "interpretation_paragraph" in type_specific_results:
                self.logger.info(
                    f"✅ [TYPE DIVERSITY] interpretation_paragraph: {len(type_specific_results['interpretation_paragraph'])}개 "
                    f"검색 결과에 포함됨"
                )
            else:
                self.logger.warning(
                    f"⚠️ [TYPE DIVERSITY] interpretation_paragraph: 검색 결과에 없음 "
                    f"(type_specific_results keys: {list(type_specific_results.keys())})"
                )
        else:
            self.logger.info("⚠️ [TYPE DIVERSITY] 타입별 검색 결과 없음 (데이터 불균형 또는 검색 실패)")

        self.logger.info(
            f"🔍 [DEBUG] Total semantic search results: {semantic_count} (unique: {len(semantic_results)})"
        )
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"[DEBUG] SEMANTIC SEARCH INTERNAL: Total={semantic_count}, Unique={len(semantic_results)}")

        search_queries_used = []
        if semantic_query:
            search_queries_used.append(f"semantic_query({len(semantic_query)} chars)")
        if original_query:
            search_queries_used.append(f"original_query({len(original_query)} chars)")
        keyword_queries_used = optimized_queries.get("keyword_queries", [])[:2]
        if keyword_queries_used:
            search_queries_used.append(f"keyword_queries({len(keyword_queries_used)} queries)")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"[DEBUG] SEMANTIC SEARCH INTERNAL: Queries used: {', '.join(search_queries_used)}")

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

        # 개선: textToSQL 라우팅 확인 및 적용
        from core.search.connectors.legal_data_connector_v2 import route_query, LegalDataConnectorV2
        
        # original_query에 대해 라우팅 확인
        self.logger.debug(f"[TEXT2SQL DEBUG] original_query='{original_query[:50] if original_query else 'EMPTY'}...', has_query={bool(original_query and original_query.strip())}")
        self.logger.info(f"🔍 [TEXT2SQL DEBUG] original_query='{original_query[:50] if original_query else 'EMPTY'}...', has_query={bool(original_query and original_query.strip())}")
        if original_query and original_query.strip():
            route = route_query(original_query)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"🔍 [TEXT2SQL DEBUG] route_query result: '{route}' for query: '{original_query[:50]}...'")
            if route == "text2sql":
                # textToSQL 방식으로 검색
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"🔍 [TEXT2SQL] Detected text2sql route for query: '{original_query[:50]}...'")
                try:
                    data_connector = LegalDataConnectorV2()
                    text2sql_results = data_connector.search_documents(original_query, limit=keyword_limit)
                    if text2sql_results:
                        keyword_results.extend(text2sql_results)
                        keyword_count += len(text2sql_results)
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.debug(f"✅ [TEXT2SQL] {len(text2sql_results)}개 결과 검색 성공")
                    else:
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.debug("⚠️ [TEXT2SQL] 검색 결과 없음")
                except Exception as e:
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f"⚠️ [TEXT2SQL] 검색 실패: {e}")
            
            # 기존 keyword_search_func 로직도 유지 (하이브리드)
            if self.keyword_search_func:
                original_kw_results, original_kw_count = self.keyword_search_func(
                    query=original_query,
                    query_type_str=query_type_str,
                    limit=keyword_limit,
                    legal_field=legal_field,
                    extracted_keywords=extracted_keywords
                )
                keyword_results.extend(original_kw_results)
                keyword_count += original_kw_count
                self.logger.info(
                    f"🔍 [DEBUG] Original query keyword search: {original_kw_count} results (query: '{original_query[:50]}...')"
                )

        # keyword_queries에 대해서도 라우팅 확인
        for i, kw_query in enumerate(keyword_queries, 1):
            if kw_query and kw_query.strip() and kw_query != original_query:
                route = route_query(kw_query)
                if route == "text2sql":
                    self.logger.info(f"🔍 [TEXT2SQL] Detected text2sql route for keyword query #{i}: '{kw_query[:50]}...'")
                    try:
                        data_connector = LegalDataConnectorV2()
                        text2sql_results = data_connector.search_documents(kw_query, limit=keyword_limit)
                        if text2sql_results:
                            keyword_results.extend(text2sql_results)
                            keyword_count += len(text2sql_results)
                            self.logger.info(f"✅ [TEXT2SQL] Query #{i}: {len(text2sql_results)}개 결과 검색 성공")
                    except Exception as e:
                        self.logger.warning(f"⚠️ [TEXT2SQL] Query #{i} 검색 실패: {e}")
                
                # 기존 keyword_search_func 로직도 유지 (하이브리드)
                if self.keyword_search_func:
                    kw_results, kw_count = self.keyword_search_func(
                        query=kw_query,
                        query_type_str=query_type_str,
                        limit=keyword_limit,
                        legal_field=legal_field,
                        extracted_keywords=extracted_keywords
                    )
                    keyword_results.extend(kw_results)
                    keyword_count += kw_count
                    self.logger.info(
                        f"🔍 [DEBUG] Keyword search #{i}: {kw_count} results (query: '{kw_query[:50]}...')"
                    )

        self.logger.info(
            f"🔍 [DEBUG] Total keyword search results: {keyword_count} (unique: {len(keyword_results)})"
        )
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"[DEBUG] KEYWORD SEARCH INTERNAL: Total={keyword_count}, Unique={len(keyword_results)}")

        return keyword_results, keyword_count

    def fallback_sequential_search(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """순차 검색 실행 (폴백) - 안전한 오류 처리"""
        semantic_results, semantic_count = [], 0
        keyword_results, keyword_count = [], 0
        
        try:
            self.logger.info("🔄 순차 검색 실행 중...")

            optimized_queries = self._get_state_value(state, "optimized_queries", {})
            search_params = self._get_state_value(state, "search_params", {})
            query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
            legal_field = self._get_state_value(state, "legal_field", "")
            extracted_keywords = optimized_queries.get("expanded_keywords", [])

            original_query = self._get_state_value(state, "query", "")
            if not original_query and "input" in state and isinstance(state.get("input"), dict):
                original_query = state["input"].get("query", "")
            
            # 빈 쿼리 검증 추가
            if not original_query or not original_query.strip():
                self.logger.error("❌ 순차 검색: query가 비어있어 검색을 수행할 수 없습니다.")
                self.logger.warning("⚠️ 순차 검색: query를 찾을 수 없습니다. state 구조 확인:")
                self.logger.debug(f"   state keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")
                if "input" in state:
                    self.logger.debug(f"   input type: {type(state.get('input'))}")
                    if isinstance(state.get("input"), dict):
                        self.logger.debug(f"   input keys: {list(state.get('input', {}).keys())}")
                # 빈 결과 반환
                self._set_state_value(state, "semantic_results", [])
                self._set_state_value(state, "keyword_results", [])
                self._set_state_value(state, "semantic_count", 0)
                self._set_state_value(state, "keyword_count", 0)
                return state

            extracted_keywords_for_search = self._get_state_value(state, "extracted_keywords", [])
            
            # 의미적 검색 시도 (오류 발생해도 계속 진행)
            try:
                semantic_results, semantic_count = self.execute_semantic_search(
                    optimized_queries, search_params, original_query, extracted_keywords_for_search
                )
                self.logger.info(f"✅ 순차 검색: 의미적 검색 완료 ({semantic_count}개 결과)")
            except Exception as semantic_err:
                self.logger.warning(f"⚠️ 순차 검색: 의미적 검색 실패: {semantic_err}")
                self.logger.debug(f"   의미적 검색 오류 상세: {semantic_err}", exc_info=True)
                semantic_results, semantic_count = [], 0

            # 키워드 검색 시도 (오류 발생해도 계속 진행)
            try:
                keyword_results, keyword_count = self.execute_keyword_search(
                    optimized_queries, search_params, query_type_str, legal_field, extracted_keywords, original_query
                )
                self.logger.info(f"✅ 순차 검색: 키워드 검색 완료 ({keyword_count}개 결과)")
            except Exception as keyword_err:
                self.logger.warning(f"⚠️ 순차 검색: 키워드 검색 실패: {keyword_err}")
                self.logger.debug(f"   키워드 검색 오류 상세: {keyword_err}", exc_info=True)
                keyword_results, keyword_count = [], 0

            # 결과 저장 (일부라도 성공하면 저장)
            self._set_state_value(state, "semantic_results", semantic_results)
            self._set_state_value(state, "keyword_results", keyword_results)
            self._set_state_value(state, "semantic_count", semantic_count)
            self._set_state_value(state, "keyword_count", keyword_count)
            
            # State 구조 일관성 확보: retrieved_docs를 헬퍼 함수로 저장
            merged_docs = semantic_results + keyword_results
            set_retrieved_docs(state, merged_docs)

            total_results = semantic_count + keyword_count
            if total_results > 0:
                self.logger.info(f"✅ 순차 검색 완료: 의미적 {semantic_count}개, 키워드 {keyword_count}개 (총 {total_results}개)")
            else:
                self.logger.warning(f"⚠️ 순차 검색 완료: 결과 없음 (의미적 {semantic_count}개, 키워드 {keyword_count}개)")

        except Exception as e:
            self.logger.error(f"❌ 순차 검색 중 예상치 못한 오류: {e}", exc_info=True)
            self._handle_error(state, str(e), "순차 검색 중 오류 발생")
            # 최소한의 결과라도 저장
            self._set_state_value(state, "semantic_results", semantic_results)
            self._set_state_value(state, "keyword_results", keyword_results)
            self._set_state_value(state, "semantic_count", semantic_count)
            self._set_state_value(state, "keyword_count", keyword_count)
            
            # State 구조 일관성 확보: retrieved_docs를 헬퍼 함수로 저장
            merged_docs = semantic_results + keyword_results
            set_retrieved_docs(state, merged_docs)

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
                
                # 고유한 ID 생성 (중복 방지)
                unique_id = f"sample_{doc_type}_{chunk_id}_{source_id}"
                samples.append({
                    "id": unique_id,  # 고유 ID로 변경
                    "content_id": unique_id,  # 중복 제거를 위한 대체 ID
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
                        "search_type": "type_sample",  # 메타데이터에도 추가
                        **source_meta
                    },
                    "relevance_score": 0.3,
                    "search_type": "type_sample"
                })
                self.logger.debug(f"🔍 [TYPE DIVERSITY] 샘플 문서 생성: id={unique_id}, doc_type={doc_type}, chunk_id={chunk_id}")
            
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

    def _evaluate_search_quality(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        original_query: str
    ) -> float:
        """검색 결과 품질 평가 (0.0 ~ 1.0)
        
        Args:
            semantic_results: 의미적 검색 결과
            keyword_results: 키워드 검색 결과
            original_query: 원본 쿼리
            
        Returns:
            품질 점수 (0.0 ~ 1.0)
        """
        try:
            # 1. 결과 수 기반 점수 (0.0 ~ 0.4)
            total_results = len(semantic_results) + len(keyword_results)
            semantic_count = len(semantic_results)
            keyword_count = len(keyword_results)
            
            # 결과 수 점수 계산
            count_score = min(0.4, (total_results / 20.0) * 0.4)  # 20개 이상이면 0.4점
            
            # 2. 결과 다양성 점수 (0.0 ~ 0.3)
            diversity_score = 0.0
            if semantic_count > 0 and keyword_count > 0:
                # 두 타입 모두 있으면 다양성 점수
                diversity_score = 0.3
            elif total_results > 0:
                # 하나라도 있으면 부분 점수
                diversity_score = 0.15
            
            # 3. 관련성 점수 (0.0 ~ 0.3) - 평균 relevance_score 기반
            relevance_score = 0.0
            all_scores = []
            
            for doc in semantic_results + keyword_results:
                score = doc.get("relevance_score") or doc.get("score") or 0.0
                if score > 0:
                    all_scores.append(score)
            
            if all_scores:
                avg_score = sum(all_scores) / len(all_scores)
                # 평균 점수가 0.7 이상이면 높은 관련성
                relevance_score = min(0.3, (avg_score / 0.7) * 0.3)
            
            # 최종 품질 점수
            quality_score = count_score + diversity_score + relevance_score
            
            return min(1.0, quality_score)
            
        except Exception as e:
            self.logger.debug(f"Error evaluating search quality: {e}")
            # 오류 발생 시 기본 점수 반환
            total_results = len(semantic_results) + len(keyword_results)
            return min(1.0, total_results / 20.0)

    def _merge_multi_query_results_single(
        self,
        main_results: List[Dict[str, Any]],
        mq_results: List[Dict[str, Any]],
        mq_query: str,
        original_query: str,
        seen_ids: set,
        seen_hashes: set
    ) -> List[Dict[str, Any]]:
        """Multi-Query 결과를 다양성과 관련성을 고려하여 병합 (단일)
        
        Args:
            main_results: 기존 검색 결과
            mq_results: Multi-Query 검색 결과
            mq_query: Multi-Query 텍스트
            original_query: 원본 쿼리
            seen_ids: 이미 본 문서 ID 집합
            seen_hashes: 이미 본 콘텐츠 해시 집합
            
        Returns:
            병합된 새로운 결과 리스트
        """
        new_results = []
        
        for doc in mq_results:
            doc_id = doc.get("id") or doc.get("doc_id")
            content = doc.get("content", "") or doc.get("text", "")
            content_hash = hash(content[:100]) if content else None
            
            # 중복 체크
            if (not doc_id or doc_id not in seen_ids) and \
               (not content_hash or content_hash not in seen_hashes):
                # Multi-query 출처 정보 추가
                doc["multi_query_source"] = mq_query
                doc["multi_query_boost"] = 0.9  # Multi-query 결과는 약간 낮은 가중치
                
                # 원본 쿼리와의 관련성 점수 계산 (간단한 키워드 기반)
                if content:
                    similarity = self._calculate_query_similarity(original_query, content)
                    doc["original_query_similarity"] = similarity
                    
                    # 관련성 점수 반영
                    base_score = doc.get("relevance_score", 0.0) or doc.get("score", 0.0)
                    if base_score > 0:
                        # 원본 쿼리와의 관련성을 고려하여 점수 조정
                        adjusted_score = base_score * (0.7 + 0.3 * similarity)
                        doc["relevance_score"] = adjusted_score
                        doc["score"] = adjusted_score
                
                new_results.append(doc)
        
        return new_results

    def _calculate_query_similarity(self, query1: str, text: str) -> float:
        """쿼리와 텍스트 간의 유사도 계산 (간단한 키워드 기반)
        
        Args:
            query1: 쿼리 텍스트
            text: 비교할 텍스트
            
        Returns:
            유사도 점수 (0.0 ~ 1.0)
        """
        try:
            if not query1 or not text:
                return 0.0
            
            # 간단한 키워드 기반 유사도 계산
            query_words = set(query1.lower().split())
            text_words = set(text.lower().split())
            
            if not query_words or not text_words:
                return 0.0
            
            # Jaccard 유사도
            intersection = len(query_words & text_words)
            union = len(query_words | text_words)
            
            if union == 0:
                return 0.0
            
            similarity = intersection / union
            
            # 정규화 (0.0 ~ 1.0)
            return min(1.0, similarity)
            
        except Exception as e:
            self.logger.debug(f"Error calculating query similarity: {e}")
            return 0.5  # 기본값

