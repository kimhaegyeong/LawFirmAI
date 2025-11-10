# -*- coding: utf-8 -*-
"""
Search Mixin
검색 관련 노드 및 메서드들을 제공하는 Mixin 클래스
"""

import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from core.workflow.state.state_definitions import LegalWorkflowState
from core.workflow.utils.workflow_constants import WorkflowConstants, RetryConfig
from core.workflow.state.workflow_types import QueryComplexity
from core.shared.wrappers.node_wrappers import with_state_optimization
from core.generation.validators.quality_validators import SearchValidator
from core.workflow.state.state_utils import prune_retrieved_docs, MAX_RETRIEVED_DOCS, MAX_DOCUMENT_CONTENT_LENGTH

try:
    from langfuse import observe
except ImportError:
    def observe(**kwargs):
        def decorator(func):
            return func
        return decorator

# 성능 최적화: 정규식 패턴 컴파일 (모듈 레벨)
LAW_PATTERN = re.compile(r'[가-힣]+법\s*제?\s*\d+\s*조')
PRECEDENT_PATTERN = re.compile(r'대법원|법원.*\d{4}[다나마]\d+')


class SearchMixin:
    """검색 관련 노드 및 메서드들을 제공하는 Mixin 클래스"""
    
    # ============================================================================
    # prepare_search_query 헬퍼 메서드들
    # ============================================================================
    
    def _get_query_info_for_optimization(
        self,
        state: LegalWorkflowState
    ) -> Dict[str, Any]:
        """쿼리 정보 가져오기 및 검증 (중복 코드 제거)"""
        query = None
        
        if "input" in state and isinstance(state["input"], dict):
            query = state["input"].get("query", "")
        
        if not query or not str(query).strip():
            query = self._get_state_value(state, "query", "")
        
        if not query or not str(query).strip():
            if isinstance(state, dict) and "query" in state:
                query = state["query"]
        
        search_query = self._get_state_value(state, "search_query") or query
        
        if not query or not str(query).strip():
            self.logger.error(f"prepare_search_query: query is empty! State keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")
            if "input" in state:
                self.logger.error(f"prepare_search_query: state['input'] = {state['input']}")
            return {
                "query": None,
                "search_query": None,
                "query_type_str": "",
                "extracted_keywords": [],
                "legal_field": ""
            }
        
        query_type_raw = self._get_state_value(state, "query_type", "")
        query_type_str = self._get_query_type_str(query_type_raw)
        query_type_str = self._normalize_query_type_for_prompt(query_type_str)
        
        extracted_keywords_raw = self._get_state_value(state, "extracted_keywords", [])
        if not isinstance(extracted_keywords_raw, list):
            self.logger.warning(f"extracted_keywords is not a list: {type(extracted_keywords_raw)}, converting to empty list")
            extracted_keywords = []
        else:
            extracted_keywords = [kw for kw in extracted_keywords_raw if kw and isinstance(kw, str) and len(str(kw).strip()) > 0]
        
        legal_field_raw = self._get_state_value(state, "legal_field", "")
        legal_field = str(legal_field_raw).strip() if legal_field_raw else ""
        
        self.logger.debug(
            f"📋 [PREPARE SEARCH QUERY] Data for query optimization:\n"
            f"   query: '{query[:50]}{'...' if len(query) > 50 else ''}'\n"
            f"   search_query: '{search_query[:50]}{'...' if len(search_query) > 50 else ''}'\n"
            f"   query_type (raw): '{query_type_raw}' → (normalized): '{query_type_str}'\n"
            f"   extracted_keywords: {len(extracted_keywords)} items {extracted_keywords[:5] if extracted_keywords else '[]'}\n"
            f"   legal_field: '{legal_field}'"
        )
        
        return {
            "query": query,
            "search_query": search_query,
            "query_type_str": query_type_str,
            "extracted_keywords": extracted_keywords,
            "legal_field": legal_field
        }
    
    def _optimize_query_with_cache(
        self,
        search_query: str,
        query_type_str: str,
        extracted_keywords: List[str],
        legal_field: str,
        is_retry: bool
    ) -> Tuple[Dict[str, Any], bool]:
        """쿼리 최적화 (캐싱 포함) (중복 코드 제거)"""
        import hashlib
        
        optimized_queries = None
        cache_hit = False
        
        if not is_retry:
            cache_key_parts = [
                search_query,
                query_type_str,
                ",".join(sorted(extracted_keywords)) if extracted_keywords else "",
                legal_field
            ]
            cache_key = hashlib.md5(":".join(cache_key_parts).encode('utf-8')).hexdigest()
            
            try:
                cached_result = self.performance_optimizer.cache.get_cached_answer(
                    f"query_opt:{cache_key}", query_type_str
                )
                if cached_result and isinstance(cached_result, dict) and "optimized_queries" in cached_result:
                    optimized_queries = cached_result.get("optimized_queries")
                    cache_hit = True
                    self.logger.info(f"✅ [CACHE HIT] 쿼리 최적화 결과 캐시 히트: {cache_key[:16]}...")
            except Exception as e:
                self.logger.debug(f"캐시 확인 중 오류 (무시): {e}")
        
        if not optimized_queries:
            optimized_queries = self._optimize_search_query(
                query=search_query,
                query_type=query_type_str,
                extracted_keywords=extracted_keywords,
                legal_field=legal_field
            )
            
            if not is_retry:
                try:
                    cache_key_parts = [
                        search_query,
                        query_type_str,
                        ",".join(sorted(extracted_keywords)) if extracted_keywords else "",
                        legal_field
                    ]
                    cache_key = hashlib.md5(":".join(cache_key_parts).encode('utf-8')).hexdigest()
                    self.performance_optimizer.cache.cache_answer(
                        f"query_opt:{cache_key}",
                        query_type_str,
                        {"optimized_queries": optimized_queries},
                        confidence=1.0,
                        sources=[]
                    )
                    self.logger.debug(f"✅ [CACHE STORE] 쿼리 최적화 결과 캐시 저장: {cache_key[:16]}...")
                except Exception as e:
                    self.logger.debug(f"캐시 저장 중 오류 (무시): {e}")
        
        return optimized_queries, cache_hit
    
    def _validate_and_fix_optimized_queries(
        self,
        state: LegalWorkflowState,
        optimized_queries: Dict[str, Any],
        query: str
    ) -> Dict[str, Any]:
        """최적화된 쿼리 검증 및 수정 (중복 코드 제거)"""
        semantic_query_created = optimized_queries.get("semantic_query", "")
        if not semantic_query_created or not str(semantic_query_created).strip():
            self.logger.warning(f"semantic_query is empty, using base query: '{query[:50]}...'")
            optimized_queries["semantic_query"] = query
            semantic_query_created = query
            self._set_state_value(state, "optimized_queries", optimized_queries)
        
        keyword_queries_created = optimized_queries.get("keyword_queries", [])
        if not keyword_queries_created or len(keyword_queries_created) == 0:
            self.logger.warning("keyword_queries is empty, using base query")
            optimized_queries["keyword_queries"] = [query]
            keyword_queries_created = [query]
            self._set_state_value(state, "optimized_queries", optimized_queries)
        
        return {
            "optimized_queries": optimized_queries,
            "semantic_query_created": semantic_query_created,
            "keyword_queries_created": keyword_queries_created
        }
    
    @observe(name="prepare_search_query")
    @with_state_optimization("prepare_search_query", enable_reduction=False)
    def prepare_search_query(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """검색 쿼리 준비 및 최적화 전용 노드 (Part 2)"""
        try:
            start_time = time.time()

            preserved = self._preserve_metadata(state, ["query_complexity", "needs_search"])
            if "metadata" not in state or not isinstance(state.get("metadata"), dict):
                state["metadata"] = {}
            state["metadata"] = dict(state["metadata"])
            state["metadata"].update(preserved)
            state["metadata"]["_last_executed_node"] = "prepare_search_query"
            
            if "common" not in state or not isinstance(state.get("common"), dict):
                state["common"] = {}
            if "metadata" not in state["common"]:
                state["common"]["metadata"] = {}
            state["common"]["metadata"]["_last_executed_node"] = "prepare_search_query"

            self._ensure_input_group(state)
            query_value, session_id_value = self._restore_query_from_state(state)
            if query_value:
                query_value = self._normalize_query_encoding(query_value)
                state["input"]["query"] = query_value
                if session_id_value:
                    state["input"]["session_id"] = session_id_value

            # 재시도 카운터 관리
            metadata = state.get("metadata", {}) if isinstance(state.get("metadata"), dict) else {}
            if not isinstance(metadata, dict):
                metadata = {}

            # 중요: state.get("common")이 None일 수 있으므로 안전하게 처리
            common_state = state.get("common")
            if common_state and isinstance(common_state, dict):
                common_metadata = common_state.get("metadata", {})
                if isinstance(common_metadata, dict):
                    metadata = {**metadata, **common_metadata}

            last_executed_node = metadata.get("_last_executed_node", "")
            is_retry_from_generation = (last_executed_node == "generate_answer_enhanced")
            is_retry_from_validation = (last_executed_node == "validate_answer_quality")

            # 재시도 카운터 증가
            if is_retry_from_generation:
                if self.retry_manager.should_allow_retry(state, "generation"):
                    self.retry_manager.increment_retry_count(state, "generation")

            if is_retry_from_validation:
                if self.retry_manager.should_allow_retry(state, "validation"):
                    self.retry_manager.increment_retry_count(state, "validation")

            # 재시도 횟수 체크
            retry_counts = self.retry_manager.get_retry_counts(state)
            if retry_counts["total"] >= RetryConfig.MAX_TOTAL_RETRIES:
                self.logger.error("Maximum total retry count reached")
                if not self._get_state_value(state, "answer", ""):
                    query = self._get_state_value(state, "query", "")
                    self._set_answer_safely(state,
                        f"죄송합니다. 질문 '{query}'에 대한 답변을 생성하는데 어려움이 있습니다.")
                return state

            query_info = self._get_query_info_for_optimization(state)
            if not query_info["query"]:
                self._set_answer_safely(state, "죄송합니다. 질문을 이해하지 못했습니다. 다시 질문해주세요.")
                return state
            
            query = query_info["query"]
            search_query = query_info["search_query"]
            query_type_str = query_info["query_type_str"]
            extracted_keywords = query_info["extracted_keywords"]
            legal_field = query_info["legal_field"]

            is_retry = (last_executed_node == "validate_answer_quality")

            optimized_queries, cache_hit_optimization = self._optimize_query_with_cache(
                search_query=search_query,
                query_type_str=query_type_str,
                extracted_keywords=extracted_keywords,
                legal_field=legal_field,
                is_retry=is_retry
            )

            if is_retry:
                quality_feedback = self.answer_generator.get_quality_feedback_for_retry(state)
                improved_query = self._improve_search_query_for_retry(
                    optimized_queries["semantic_query"],
                    quality_feedback,
                    state
                )
                if improved_query != optimized_queries["semantic_query"]:
                    self.logger.info(
                        f"🔍 [SEARCH RETRY] Improved query: '{optimized_queries['semantic_query']}' → '{improved_query}'"
                    )
                    optimized_queries["semantic_query"] = improved_query
                    optimized_queries["keyword_queries"][0] = improved_query

            search_params = self._determine_search_parameters(
                query_type=query_type_str,
                query_complexity=len(query),
                keyword_count=len(extracted_keywords),
                is_retry=is_retry
            )

            self._set_state_value(state, "optimized_queries", optimized_queries)
            self._set_state_value(state, "search_params", search_params)
            self._set_state_value(state, "is_retry_search", is_retry)
            self._set_state_value(state, "search_start_time", start_time)

            validated_queries = self._validate_and_fix_optimized_queries(state, optimized_queries, query)
            optimized_queries = validated_queries["optimized_queries"]
            semantic_query_created = validated_queries["semantic_query_created"]
            keyword_queries_created = validated_queries["keyword_queries_created"]

            self._set_state_value(state, "search_query", semantic_query_created)

            # 캐시 확인 (재시도 시에는 캐시 우회)
            cache_hit = False
            if not is_retry:
                cached_documents = self.performance_optimizer.cache.get_cached_documents(
                    optimized_queries["semantic_query"],
                    query_type_str
                )
                if cached_documents:
                    self._set_state_value(state, "retrieved_docs", cached_documents)
                    self._set_state_value(state, "search_cache_hit", True)
                    cache_hit = True
                    self._add_step(state, "캐시 히트", f"캐시 히트: {len(cached_documents)}개 문서")

            self._set_state_value(state, "search_cache_hit", cache_hit)
            self._save_metadata_safely(state, "_last_executed_node", "prepare_search_query")
            self._update_processing_time(state, start_time)
            self._add_step(state, "검색 쿼리 준비", f"검색 쿼리 준비 완료: {semantic_query_created[:50]}...")

            if cache_hit:
                self.logger.info(f"✅ [CACHE HIT] 캐시 히트: {len(cached_documents)}개 문서, 검색 스킵")
            else:
                self.logger.info(
                    f"✅ [PREPARE SEARCH QUERY] "
                    f"semantic_query: '{semantic_query_created[:50]}...', "
                    f"keyword_queries: {len(keyword_queries_created)}개, "
                    f"search_params: k={search_params.get('semantic_k', 'N/A')}"
                )

        except Exception as e:
            self._handle_error(state, str(e), "검색 쿼리 준비 중 오류 발생")

        self._ensure_input_group(state)
        query_value, session_id_value = self._restore_query_from_state(state)
        if query_value:
            state["input"]["query"] = query_value
            if session_id_value:
                state["input"]["session_id"] = session_id_value

        return state

