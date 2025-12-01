# -*- coding: utf-8 -*-
"""
Classification Mixin
분류 관련 노드 및 메서드들을 제공하는 Mixin 클래스
"""

import time
from typing import Any, Dict, Tuple

try:
    from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
except ImportError:
    from core.workflow.state.state_definitions import LegalWorkflowState
try:
    from lawfirm_langgraph.core.workflow.state.workflow_types import QueryComplexity
except ImportError:
    from core.workflow.state.workflow_types import QueryComplexity
try:
    from lawfirm_langgraph.core.shared.wrappers.node_wrappers import with_state_optimization
except ImportError:
    from core.shared.wrappers.node_wrappers import with_state_optimization

# Mock observe decorator (Langfuse 제거됨)
def observe(**kwargs):
    def decorator(func):
        return func
    return decorator


class ClassificationMixin:
    """분류 관련 노드 및 메서드들을 제공하는 Mixin 클래스"""
    
    # ============================================================================
    # Classification 헬퍼 메서드들
    # ============================================================================
    
    def _restore_and_validate_query(self, state: LegalWorkflowState) -> str:
        """Query 복원 및 검증 (중복 코드 제거)"""
        self._ensure_input_group(state)
        
        current_query = state["input"].get("query", "")
        if not current_query:
            query_from_top = state.get("query", "")
            session_id_from_top = state.get("session_id", "")
            if query_from_top:
                state["input"]["query"] = query_from_top
                if session_id_from_top:
                    state["input"]["session_id"] = session_id_from_top
        
        query_value = self._get_state_value(state, "query", "")
        if not query_value or not str(query_value).strip():
            if "input" in state and isinstance(state.get("input"), dict):
                query_value = state["input"].get("query", "")
            elif isinstance(state, dict) and "query" in state:
                query_value = state["query"]
            else:
                if "input" not in state:
                    state["input"] = {}
                state["input"]["query"] = ""
        else:
            if "input" not in state:
                state["input"] = {}
            state["input"]["query"] = query_value
        
        return self._get_state_value(state, "query", "")
    
    def _execute_classification(
        self,
        query: str
    ) -> Tuple[Any, float, QueryComplexity, bool]:
        """분류 실행 (LLM 기반 또는 폴백) (중복 코드 제거)"""
        if not query:
            classified_type, confidence = self._fallback_classification("")
            complexity = QueryComplexity.MODERATE
            needs_search = True
        else:
            if self.config.use_llm_for_complexity:
                try:
                    classified_type, confidence, complexity, needs_search = self._classify_query_with_chain(query)
                    self.logger.info(
                        f"✅ [CHAIN CLASSIFICATION] "
                        f"QuestionType={classified_type.value}, complexity={complexity.value}, "
                        f"needs_search={needs_search}, confidence={confidence:.2f}"
                    )
                except Exception as e:
                    self.logger.warning(f"체인 LLM 분류 실패, 폴백 사용: {e}")
                    classified_type, confidence = self._fallback_classification(query)
                    complexity, needs_search = self._fallback_complexity_classification(query)
            else:
                classified_type, confidence = self._fallback_classification(query)
                complexity, needs_search = self._fallback_complexity_classification(query)
        
        return (classified_type, confidence, complexity, needs_search)
    
    def _save_classification_results(
        self,
        state: LegalWorkflowState,
        query_type_str: str,
        confidence: float,
        legal_field: str
    ) -> None:
        """분류 결과를 State에 저장 (중복 코드 제거)"""
        self._set_state_value(state, "query_type", query_type_str)
        self._set_state_value(state, "confidence", confidence)
        self._set_state_value(state, "legal_field", legal_field)
        self._set_state_value(state, "legal_domain", self._map_to_legal_domain(legal_field))
        
        if "classification" not in state:
            state["classification"] = {}
        state["classification"]["query_type"] = query_type_str
        state["classification"]["confidence"] = confidence
        state["classification"]["legal_field"] = legal_field
        
        if "common" not in state:
            state["common"] = {}
        if "classification" not in state["common"]:
            state["common"]["classification"] = {}
        state["common"]["classification"]["query_type"] = query_type_str
        state["common"]["classification"]["confidence"] = confidence
        
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["query_type"] = query_type_str
        state["metadata"]["confidence"] = confidence
        state["metadata"]["legal_field"] = legal_field
        
        if "metadata" not in state["common"]:
            state["common"]["metadata"] = {}
        state["common"]["metadata"]["query_type"] = query_type_str
        state["common"]["metadata"]["confidence"] = confidence
        
        try:
            from core.shared.wrappers.node_wrappers import _global_search_results_cache
            import core.agents.node_wrappers as node_wrappers_module
            if node_wrappers_module._global_search_results_cache is None:
                node_wrappers_module._global_search_results_cache = {}
            
            if "common" not in _global_search_results_cache:
                _global_search_results_cache["common"] = {}
            if "classification" not in _global_search_results_cache["common"]:
                _global_search_results_cache["common"]["classification"] = {}
            _global_search_results_cache["common"]["classification"]["query_type"] = query_type_str
            _global_search_results_cache["common"]["classification"]["confidence"] = confidence
            
            if "metadata" not in _global_search_results_cache:
                _global_search_results_cache["metadata"] = {}
            _global_search_results_cache["metadata"]["query_type"] = query_type_str
            _global_search_results_cache["metadata"]["confidence"] = confidence
            
            _global_search_results_cache["query_type"] = query_type_str
            _global_search_results_cache["confidence"] = confidence
            
            saved_query_type = (
                _global_search_results_cache.get("common", {}).get("classification", {}).get("query_type") or
                _global_search_results_cache.get("metadata", {}).get("query_type") or
                _global_search_results_cache.get("query_type")
            )
            if saved_query_type == query_type_str:
                self.logger.info(f"✅ [QUERY_TYPE] Saved to global cache and verified: {query_type_str}")
            else:
                self.logger.warning(f"⚠️ [QUERY_TYPE] Global cache save verification failed: expected {query_type_str}, got {saved_query_type}")
        except (ImportError, AttributeError, TypeError) as e:
            self.logger.debug(f"Could not save to global cache: {e}")
    
    def _save_complexity_results(
        self,
        state: LegalWorkflowState,
        complexity: QueryComplexity,
        needs_search: bool
    ) -> None:
        """복잡도 결과를 State에 저장 (중복 코드 제거)"""
        self._set_state_value(state, "query_complexity", complexity.value)
        self._set_state_value(state, "needs_search", needs_search)
        
        if "classification" not in state:
            state["classification"] = {}
        state["classification"]["query_complexity"] = complexity.value
        state["classification"]["needs_search"] = needs_search
        state["query_complexity"] = complexity.value
        state["needs_search"] = needs_search
        
        if "common" not in state:
            state["common"] = {}
        state["common"]["query_complexity"] = complexity.value
        state["common"]["needs_search"] = needs_search
        
        if "metadata" not in state:
            state["metadata"] = {}
        elif not isinstance(state.get("metadata"), dict):
            state["metadata"] = {}
        state["metadata"]["query_complexity"] = complexity.value
        state["metadata"]["needs_search"] = needs_search
        
        try:
            # 여러 import 방법 시도 (모듈 경로 오류 방지)
            try:
                from lawfirm_langgraph.core.agents import node_wrappers
            except ImportError:
                try:
                    from core.agents import node_wrappers
                except ImportError:
                    # 현재 파일이 core.agents 내부에 있으므로 상대 import 시도
                    from .. import node_wrappers
            if not hasattr(node_wrappers, '_global_search_results_cache') or node_wrappers._global_search_results_cache is None:
                node_wrappers._global_search_results_cache = {}
            node_wrappers._global_search_results_cache["query_complexity"] = complexity.value
            node_wrappers._global_search_results_cache["needs_search"] = needs_search
        except Exception as e:
            self.logger.warning(f"Global cache 저장 실패: {e}")
    
    # ============================================================================
    # Classification 노드들
    # ============================================================================
    
    @observe(name="classify_query")
    @with_state_optimization("classify_query", enable_reduction=True)
    def classify_query(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """질문 분류 (LLM 기반)"""
        self._ensure_input_group(state)
        
        query_value, session_id_value = self._restore_query_from_state(state)
        if query_value:
            state["input"]["query"] = query_value
            if session_id_value:
                state["input"]["session_id"] = session_id_value
        else:
            query_value = self._get_state_value(state, "query", "")
            if query_value:
                state["input"]["query"] = query_value

        try:
            start_time = time.time()

            query = self._get_state_value(state, "query", "")
            classified_type, confidence = self._classify_with_llm(query)

            query_type_str = classified_type.value if hasattr(classified_type, 'value') else str(classified_type)
            self._set_state_value(state, "query_type", query_type_str)
            self._set_state_value(state, "confidence", confidence)

            self.logger.info(
                f"✅ [QUESTION CLASSIFICATION] "
                f"QuestionType={classified_type.name if hasattr(classified_type, 'name') else classified_type} "
                f"(confidence: {confidence:.2f})"
            )

            legal_field = self._extract_legal_field(query_type_str, query)
            self._set_state_value(state, "legal_field", legal_field)
            self._set_state_value(state, "legal_domain", self._map_to_legal_domain(legal_field))

            processing_time = self._update_processing_time(state, start_time)
            self._add_step(state, "질문 분류 완료",
                         f"질문 분류 완료: {query_type_str}, 법률분야: {legal_field} (시간: {processing_time:.3f}s)")

            self.logger.info(f"LLM classified query as {query_type_str} with confidence {confidence}, field: {legal_field}")

            self._ensure_input_group(state)
            query_value, session_id_value = self._restore_query_from_state(state)
            state["input"]["query"] = query_value or state.get("input", {}).get("query", "")
            state["input"]["session_id"] = session_id_value or state.get("input", {}).get("session_id", "")

            if not state["input"]["query"]:
                self.logger.warning(f"classify_query: query is empty after ensuring input group!")
            else:
                self.logger.debug(f"Ensured input group in state after classify_query: query length={len(state['input']['query'])}")

        except Exception as e:
            self._handle_error(state, str(e), "LLM 질문 분류 중 오류 발생")
            query = self._get_state_value(state, "query", "")
            classified_type, confidence = self._fallback_classification(query)
            query_type_str = classified_type.value if hasattr(classified_type, 'value') else str(classified_type)
            self._set_state_value(state, "query_type", query_type_str)
            self._set_state_value(state, "confidence", confidence)

            legal_field = self._extract_legal_field(query_type_str, query)
            self._set_state_value(state, "legal_field", legal_field)
            self._set_state_value(state, "legal_domain", self._map_to_legal_domain(legal_field))

            self._add_step(state, "폴백 키워드 기반 분류 사용", "폴백 키워드 기반 분류 사용")

            self._ensure_input_group(state)
            query_value, session_id_value = self._restore_query_from_state(state)
            state["input"]["query"] = query_value or state.get("input", {}).get("query", "")
            state["input"]["session_id"] = session_id_value or state.get("input", {}).get("session_id", "")

            if not state["input"]["query"]:
                self.logger.warning(f"classify_query (fallback): query is empty after ensuring input group!")
            else:
                self.logger.debug(f"Ensured input group in state after classify_query (fallback): query length={len(state['input']['query'])}")

        self._ensure_input_group(state)
        query_value, session_id_value = self._restore_query_from_state(state)
        if query_value:
            state["input"]["query"] = query_value
            if session_id_value:
                state["input"]["session_id"] = session_id_value
        else:
            self.logger.error(f"classify_query: query not found, state keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")

        return state

    @observe(name="classify_query_and_complexity")
    @with_state_optimization("classify_query_and_complexity", enable_reduction=False)
    def classify_query_and_complexity(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """통합된 질문 분류 및 복잡도 판단 (classify_query + classify_complexity)"""
        try:
            overall_start_time = time.time()
            query_start_time = time.time()

            query = self._restore_and_validate_query(state)
            classified_type, confidence, complexity, needs_search = self._execute_classification(query)

            query_type_str = classified_type.value if hasattr(classified_type, 'value') else str(classified_type)
            legal_field = self._extract_legal_field(query_type_str, query)

            self._save_classification_results(state, query_type_str, confidence, legal_field)

            self._update_processing_time(state, query_start_time)
            self._add_step(state, "질문 분류 완료",
                         f"질문 분류 완료: {query_type_str}, 법률분야: {legal_field}")

            self._preserve_metadata_and_common_state(state)

            self._save_complexity_results(state, complexity, needs_search)

            self._add_step(
                state,
                "복잡도 분류",
                f"질문 복잡도: {complexity.value}, 검색 필요: {needs_search}"
            )

            self._update_processing_time(state, overall_start_time)

        except Exception as e:
            self._handle_error(state, str(e), "질문 분류 및 복잡도 판단 중 오류 발생")
            try:
                query = self._get_state_value(state, "query", "")
                classified_type, confidence = self._fallback_classification(query)
                query_type_str = classified_type.value if hasattr(classified_type, 'value') else str(classified_type)
                legal_field = self._extract_legal_field(query_type_str, query)
                self._set_state_value(state, "query_type", query_type_str)
                self._set_state_value(state, "confidence", confidence)
                self._set_state_value(state, "legal_field", legal_field)
                self._set_state_value(state, "legal_domain", self._map_to_legal_domain(legal_field))
            except Exception:
                self._set_state_value(state, "query_type", "general_question")
                self._set_state_value(state, "confidence", 0.5)
                self._set_state_value(state, "legal_field", "general")
                self._set_state_value(state, "legal_domain", "general")

            self._set_state_value(state, "query_complexity", QueryComplexity.MODERATE.value)
            self._set_state_value(state, "needs_search", True)

        self._preserve_metadata_and_common_state(state)

        return state

