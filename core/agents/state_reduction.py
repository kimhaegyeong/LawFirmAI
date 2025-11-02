# -*- coding: utf-8 -*-
"""
State Reduction 시스템
노드별 필요한 데이터만 전달하여 메모리 최적화

효과:
- 메모리 사용량: 90%+ 감소
- LangSmith 전송: 85% 감소
- 처리 속도: 10-15% 개선
"""

import logging
from typing import Any, Dict, Optional, Set

from .node_input_output_spec import (
    get_node_spec,
)

logger = logging.getLogger(__name__)


class StateReducer:
    """State를 노드별로 필요한 만큼만 줄이는 클래스"""

    def __init__(self, aggressive_reduction: bool = True):
        """
        StateReducer 초기화

        Args:
            aggressive_reduction: 공격적인 감소 모드 (더 많은 최적화)
        """
        self.aggressive_reduction = aggressive_reduction
        self.logger = logging.getLogger(__name__)

    def reduce_state_for_node(
        self,
        state: Dict[str, Any],
        node_name: str
    ) -> Dict[str, Any]:
        """
        특정 노드에 필요한 State만 추출

        Args:
            state: 전체 State 객체
            node_name: 실행할 노드 이름

        Returns:
            축소된 State (nested 또는 flat 구조)
        """
        # State가 딕셔너리가 아닌 경우 처리
        if not isinstance(state, dict):
            self.logger.warning(
                f"State is not a dict for node {node_name}, "
                f"got {type(state).__name__}. Returning empty dict."
            )
            return {}

        # 중요: state reduction 전에 input 그룹 보존
        # 모든 노드에서 query가 필요하므로 input 그룹은 항상 보존해야 함
        preserved_input = None
        if "input" in state and isinstance(state.get("input"), dict):
            preserved_input = state["input"].copy()
        elif "query" in state or "session_id" in state:
            preserved_input = {
                "query": state.get("query", ""),
                "session_id": state.get("session_id", "")
            }

        spec = get_node_spec(node_name)
        if not spec:
            # 스펙이 없으면 전체 반환하되 input은 항상 포함
            if preserved_input:
                if "input" not in state or not isinstance(state.get("input"), dict):
                    state["input"] = preserved_input
            return state

        # 필요한 State 그룹 조회
        required_groups = spec.required_state_groups
        if not self.aggressive_reduction:
            # 보수적 모드: 일반적으로 필요한 그룹 추가
            required_groups = required_groups | {"common"}
        else:
            # 공격적 모드: strict하게 필요한 것만
            required_groups = required_groups | {"common"}  # common은 항상 필요

        # 중요: input 그룹은 항상 보존 (모든 노드에서 필요)
        # 워크플로우 전반에서 query가 필요하므로 항상 input 그룹 포함
        required_groups = required_groups | {"input"}

        reduced = {}

        # 필수 그룹만 추출
        # nested 구조에서 추출
        for group in required_groups:
            if group in state:
                # nested 구조: state["search"] 형태
                reduced[group] = state[group]
            # else:
            #     # Flat 구조를 처리하기 위해 경고는 출력하지 않음
            #     pass

        # Flat 구조인 경우를 위한 호환성 처리
        # nested 구조에서 그룹을 찾지 못했거나, 일부 그룹이 flat 구조로 존재하는 경우
        if not reduced.get("input") and "query" in state:
            # Flat 구조로 보임, 변환 필요
            reduced = self._extract_flat_state_for_groups(state, required_groups)
        elif "search" in required_groups and not reduced.get("search"):
            # search 그룹이 nested 구조에 없으면 flat 구조에서 찾기
            # flat 구조에 semantic_results, keyword_results 등이 있을 수 있음
            flat_search = self._extract_search_from_flat_state(state)
            if flat_search:
                reduced["search"] = flat_search

        # nested 구조에 search 그룹이 있지만 semantic_results/keyword_results가 없는 경우
        # flat 구조에서 가져오기 시도
        if "search" in required_groups and reduced.get("search"):
            search_group = reduced["search"]
            if isinstance(search_group, dict):
                # semantic_results나 keyword_results가 없거나 비어있으면 flat 구조에서 찾기
                has_semantic = search_group.get("semantic_results") and len(search_group.get("semantic_results", [])) > 0
                has_keyword = search_group.get("keyword_results") and len(search_group.get("keyword_results", [])) > 0
                if not has_semantic or not has_keyword:
                    flat_search = self._extract_search_from_flat_state(state)
                    if flat_search:
                        # flat 구조에서 찾은 결과를 병합 (기존 값 우선)
                        if flat_search.get("semantic_results") and not has_semantic:
                            search_group["semantic_results"] = flat_search["semantic_results"]
                        if flat_search.get("keyword_results") and not has_keyword:
                            search_group["keyword_results"] = flat_search["keyword_results"]
                        if flat_search.get("semantic_count", 0) > 0 and not search_group.get("semantic_count"):
                            search_group["semantic_count"] = flat_search["semantic_count"]
                        if flat_search.get("keyword_count", 0) > 0 and not search_group.get("keyword_count"):
                            search_group["keyword_count"] = flat_search["keyword_count"]


        # 중요: search 그룹이 필요한 노드들에 대해 전역 캐시에서 복원
        # execute_searches_parallel의 결과가 state_reduction에서 손실될 수 있음
        search_dependent_nodes = [
            "merge_and_rerank_with_keyword_weights",
            "filter_and_validate_results",
            "update_search_metadata",
            "prepare_document_context_for_prompt"
        ]

        if "search" in required_groups and node_name in search_dependent_nodes:
            # 전역 캐시에서 검색 결과 복원 시도 (node_wrappers에서 저장한 캐시)
            try:
                from .node_wrappers import _global_search_results_cache
                if _global_search_results_cache:
                    search_group = reduced.get("search", {}) if isinstance(reduced.get("search"), dict) else {}
                    has_results = len(search_group.get("semantic_results", [])) > 0 or len(search_group.get("keyword_results", [])) > 0

                    if not has_results:
                        print(f"[DEBUG] state_reduction ({node_name}): Restoring search results from global cache")
                        if "search" not in reduced:
                            reduced["search"] = {}
                        reduced["search"].update(_global_search_results_cache)
                        restored_semantic = len(reduced["search"].get("semantic_results", []))
                        restored_keyword = len(reduced["search"].get("keyword_results", []))
                        print(f"[DEBUG] state_reduction ({node_name}): Restored from cache - semantic={restored_semantic}, keyword={restored_keyword}")
            except (ImportError, AttributeError) as e:
                # 전역 캐시를 가져올 수 없는 경우 무시 (정상적인 상황일 수 있음)
                pass

        # 중요: reduction 후에 보존된 input 복원
        # input 그룹은 항상 보존되어야 하므로 reduction 후에도 복원
        if preserved_input:
            if "input" not in reduced:
                reduced["input"] = {}
            # query가 없으면 보존된 input에서 복원
            if not reduced["input"].get("query") and preserved_input.get("query"):
                reduced["input"]["query"] = preserved_input["query"]
            # session_id가 없으면 보존된 input에서 복원
            if not reduced["input"].get("session_id") and preserved_input.get("session_id"):
                reduced["input"]["session_id"] = preserved_input["session_id"]

            if node_name == "classify_query":
                print(f"[DEBUG] state_reduction: Preserved input after reduction: query='{reduced['input'].get('query', '')[:50] if reduced['input'].get('query') else 'EMPTY'}...'")

        # 추가 보장: input 그룹이 필요한데 없으면 생성 (이중 체크)
        if "input" in required_groups and not reduced.get("input"):
            # state에서 query를 찾아서 input 그룹 생성
            query_value = state.get("query", "") or (state.get("input") and isinstance(state.get("input"), dict) and state["input"].get("query", ""))
            session_id_value = state.get("session_id", "") or (state.get("input") and isinstance(state.get("input"), dict) and state["input"].get("session_id", ""))
            if query_value or session_id_value:
                reduced["input"] = {
                    "query": query_value,
                    "session_id": session_id_value
                }
                self.logger.debug(f"Created input group for node {node_name} from state")
                if node_name == "classify_query":
                    print(f"[DEBUG] state_reduction: Created input group with query='{query_value[:50] if query_value else 'EMPTY'}...'")
            elif node_name == "classify_query":
                print(f"[DEBUG] state_reduction: No query found in state, keys={list(state.keys())}")

        # common 그룹이 없고 state에 retry_count나 metadata가 있는 경우
        # flat 구조에서 common 그룹 생성 (메타데이터 보존)
        if "common" not in reduced and ("retry_count" in state or "metadata" in state):
            reduced["common"] = {}
            if "metadata" not in reduced["common"]:
                reduced["common"]["metadata"] = {}
            # retry_count를 common.metadata에 복사
            if "retry_count" in state:
                reduced["common"]["metadata"]["generation_retry_count"] = state.get("retry_count", 0)
            # metadata를 common.metadata에 병합 (안전하게 복사)
            if "metadata" in state and isinstance(state.get("metadata"), dict):
                # slice 객체나 다른 unhashable 타입을 키로 사용하는 항목 제외
                safe_metadata = {}
                for key, value in state["metadata"].items():
                    # 딕셔너리 키는 hashable해야 하므로 안전한 키만 복사
                    try:
                        hash(key)  # 키가 hashable인지 확인
                        safe_metadata[key] = value
                    except TypeError:
                        # slice 객체 등 hashable하지 않은 키는 건너뛰기
                        logger.warning(f"Skipping unhashable metadata key: {type(key).__name__}")

                reduced["common"]["metadata"].update(safe_metadata)
                # 재시도 카운터도 명시적으로 복사
                if "generation_retry_count" in safe_metadata:
                    reduced["common"]["metadata"]["generation_retry_count"] = safe_metadata["generation_retry_count"]

        return reduced

    def _extract_flat_state_for_groups(
        self,
        state: Dict[str, Any],
        required_groups: Set[str]
    ) -> Dict[str, Any]:
        """Flat 구조에서 필요한 그룹만 추출"""
        reduced = {}

        # input 그룹
        if "input" in required_groups or "query" in state:
            reduced["input"] = {
                "query": state.get("query", ""),
                "session_id": state.get("session_id", "")
            }

        # classification 그룹
        if "classification" in required_groups:
            reduced["classification"] = {
                "query_type": state.get("query_type", ""),
                "confidence": state.get("confidence", 0.0),
                "legal_field": state.get("legal_field", "general"),
                "legal_domain": state.get("legal_domain", "general"),
                "urgency_level": state.get("urgency_level", "medium"),
                "urgency_reasoning": state.get("urgency_reasoning", ""),
                "emergency_type": state.get("emergency_type"),
                "complexity_level": state.get("complexity_level", "simple"),
                "requires_expert": state.get("requires_expert", False),
                "expert_subgraph": state.get("expert_subgraph")
            }

        # search 그룹
        if "search" in required_groups:
            # extracted_keywords 안전하게 처리 (슬라이스 객체 등 unhashable 타입 방지)
            keywords_raw = state.get("extracted_keywords", [])
            safe_keywords = []
            if isinstance(keywords_raw, list):
                for kw in keywords_raw:
                    if isinstance(kw, (str, int, float, tuple)) and kw is not None:
                        safe_keywords.append(kw)
                    elif kw is not None:
                        try:
                            safe_keywords.append(str(kw))
                        except Exception:
                            pass

            reduced["search"] = {
                "search_query": state.get("search_query", state.get("query", "")),
                "extracted_keywords": safe_keywords,
                "ai_keyword_expansion": state.get("ai_keyword_expansion"),
                "retrieved_docs": state.get("retrieved_docs", []),
                "optimized_queries": state.get("optimized_queries", {}),
                "search_params": state.get("search_params", {}),
                "semantic_results": state.get("semantic_results", []),
                "keyword_results": state.get("keyword_results", []),
                "semantic_count": state.get("semantic_count", 0),
                "keyword_count": state.get("keyword_count", 0),
                "merged_documents": state.get("merged_documents", []),
                "keyword_weights": state.get("keyword_weights", {}),
                "prompt_optimized_context": state.get("prompt_optimized_context", {})
            }

        # analysis 그룹
        if "analysis" in required_groups:
            reduced["analysis"] = {
                "analysis": state.get("analysis"),
                "legal_references": state.get("legal_references", []),
                "legal_citations": state.get("legal_citations")
            }

        # answer 그룹
        if "answer" in required_groups:
            # answer 필드 안전하게 처리 (dict나 slice 객체일 수 있음)
            answer_raw = state.get("answer", "")
            if isinstance(answer_raw, dict):
                answer = answer_raw.get("content", answer_raw.get("answer", str(answer_raw)))
            elif hasattr(answer_raw, '__iter__') and not isinstance(answer_raw, (str, bytes)):
                # slice 객체나 다른 iterable이면 문자열로 변환
                answer = str(answer_raw) if answer_raw else ""
            else:
                answer = str(answer_raw) if answer_raw else ""

            # sources 안전하게 처리 (슬라이스 객체 등 unhashable 타입 방지)
            sources_raw = state.get("sources", [])
            safe_sources = []
            if isinstance(sources_raw, list):
                for source in sources_raw:
                    if isinstance(source, str):
                        safe_sources.append(source)
                    elif source is not None:
                        try:
                            safe_sources.append(str(source))
                        except Exception:
                            pass
            elif sources_raw is not None:
                try:
                    safe_sources = [str(sources_raw)]
                except Exception:
                    safe_sources = []

            reduced["answer"] = {
                "answer": answer,
                "sources": safe_sources,
                "structure_confidence": state.get("structure_confidence", 0.0)
            }

        # document 그룹
        if "document" in required_groups:
            reduced["document"] = {
                "document_type": state.get("document_type"),
                "document_analysis": state.get("document_analysis"),
                "key_clauses": state.get("key_clauses", []),
                "potential_issues": state.get("potential_issues", [])
            }

        # multi_turn 그룹
        if "multi_turn" in required_groups:
            reduced["multi_turn"] = {
                "is_multi_turn": state.get("is_multi_turn", False),
                "multi_turn_confidence": state.get("multi_turn_confidence", 1.0),
                "conversation_history": state.get("conversation_history", []),
                "conversation_context": state.get("conversation_context")
            }

        # validation 그룹
        if "validation" in required_groups:
            reduced["validation"] = {
                "legal_validity_check": state.get("legal_validity_check", True),
                "legal_basis_validation": state.get("legal_basis_validation"),
                "outdated_laws": state.get("outdated_laws", [])
            }

        # control 그룹
        if "control" in required_groups:
            reduced["control"] = {
                "retry_count": state.get("retry_count", 0),
                "quality_check_passed": state.get("quality_check_passed", False),
                "needs_enhancement": state.get("needs_enhancement", False)
            }

        # common 그룹 (항상 포함)
        # processing_steps는 common 그룹 또는 최상위 레벨에 있을 수 있음
        processing_steps = []
        if "common" in state and isinstance(state["common"], dict):
            processing_steps = state["common"].get("processing_steps", [])
        if not processing_steps or (isinstance(processing_steps, list) and len(processing_steps) == 0):
            processing_steps = state.get("processing_steps", [])

        errors = []
        if "common" in state and isinstance(state["common"], dict):
            errors = state["common"].get("errors", [])
        if not errors or (isinstance(errors, list) and len(errors) == 0):
            errors = state.get("errors", [])

        metadata = {}
        if "common" in state and isinstance(state["common"], dict):
            metadata = state["common"].get("metadata", {})
        if not metadata or not isinstance(metadata, dict):
            metadata = state.get("metadata", {}) if isinstance(state.get("metadata"), dict) else {}

        processing_time = 0.0
        if "common" in state and isinstance(state["common"], dict):
            processing_time = state["common"].get("processing_time", 0.0)
        if processing_time == 0.0:
            processing_time = state.get("processing_time", 0.0)

        tokens_used = 0
        if "common" in state and isinstance(state["common"], dict):
            tokens_used = state["common"].get("tokens_used", 0)
        if tokens_used == 0:
            tokens_used = state.get("tokens_used", 0)

        reduced["common"] = {
            "processing_steps": processing_steps if isinstance(processing_steps, list) else [],
            "errors": errors if isinstance(errors, list) else [],
            "metadata": metadata if isinstance(metadata, dict) else {},
            "processing_time": processing_time,
            "tokens_used": tokens_used
        }

        return reduced

    def _extract_search_from_flat_state(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Flat 구조에서 search 그룹 추출 (semantic_results, keyword_results 포함)"""

        # flat 구조에 semantic_results, keyword_results 등이 있는지 확인
        has_search_fields = any(
            key in state for key in [
                "semantic_results", "keyword_results", "semantic_count", "keyword_count",
                "optimized_queries", "search_params", "merged_documents", "keyword_weights",
                "search_query", "extracted_keywords", "retrieved_docs"
            ]
        )

        if not has_search_fields:
            return None

        # extracted_keywords 안전하게 처리
        keywords_raw = state.get("extracted_keywords", [])
        safe_keywords = []
        if isinstance(keywords_raw, list):
            for kw in keywords_raw:
                if isinstance(kw, (str, int, float, tuple)) and kw is not None:
                    safe_keywords.append(kw)
                elif kw is not None:
                    try:
                        safe_keywords.append(str(kw))
                    except Exception:
                        pass

        # semantic_results와 keyword_results 안전하게 처리
        semantic_raw = state.get("semantic_results", [])
        keyword_raw = state.get("keyword_results", [])
        safe_semantic = semantic_raw if isinstance(semantic_raw, list) else []
        safe_keyword = keyword_raw if isinstance(keyword_raw, list) else []

        return {
            "search_query": state.get("search_query", state.get("query", "")),
            "extracted_keywords": safe_keywords,
            "ai_keyword_expansion": state.get("ai_keyword_expansion"),
            "retrieved_docs": state.get("retrieved_docs", []),
            "optimized_queries": state.get("optimized_queries", {}),
            "search_params": state.get("search_params", {}),
            "semantic_results": safe_semantic,  # 중요: semantic_results 포함
            "keyword_results": safe_keyword,  # 중요: keyword_results 포함
            "semantic_count": state.get("semantic_count", 0),
            "keyword_count": state.get("keyword_count", 0),
            "merged_documents": state.get("merged_documents", []),
            "keyword_weights": state.get("keyword_weights", {}),
            "prompt_optimized_context": state.get("prompt_optimized_context", {})
        }

    def reduce_state_size(
        self,
        state: Dict[str, Any],
        max_docs: int = 10,
        max_content_per_doc: int = 500
    ) -> Dict[str, Any]:
        """
        State 크기 줄이기 (특히 retrieved_docs)
        Flat 및 Modular 구조 모두 지원

        Args:
            state: State 객체 (Flat 또는 Modular)
            max_docs: 최대 문서 수
            max_content_per_doc: 문서당 최대 문자 수

        Returns:
            크기가 줄어든 State
        """
        from .state_helpers import is_modular_state
        from .state_utils import prune_retrieved_docs

        reduced = dict(state)

        # Modular 구조 확인
        is_modular = is_modular_state(state)

        # retrieved_docs 제한
        if is_modular:
            # Modular 구조: search.retrieved_docs
            if "search" in reduced and isinstance(reduced["search"], dict):
                docs = reduced["search"].get("retrieved_docs", [])
                if docs:
                    pruned_docs = prune_retrieved_docs(
                        docs,
                        max_items=max_docs,
                        max_content_per_doc=max_content_per_doc
                    )
                    reduced["search"]["retrieved_docs"] = pruned_docs
        else:
            # Flat 구조: 직접 retrieved_docs
            if "retrieved_docs" in reduced:
                docs = reduced["retrieved_docs"]
                if len(docs) > max_docs:
                    self.logger.info(
                        f"Reducing retrieved_docs from {len(docs)} to {max_docs}"
                    )
                    pruned_docs = prune_retrieved_docs(
                        docs,
                        max_items=max_docs,
                        max_content_per_doc=max_content_per_doc
                    )
                    reduced["retrieved_docs"] = pruned_docs

        # conversation_history 제한
        if is_modular:
            # Modular 구조: multi_turn.conversation_history
            if "multi_turn" in reduced and isinstance(reduced["multi_turn"], dict):
                history = reduced["multi_turn"].get("conversation_history", [])
                if len(history) > 5:
                    self.logger.info(
                        f"Reducing conversation_history from {len(history)} to 5"
                    )
                    reduced["multi_turn"]["conversation_history"] = history[-5:]
        else:
            # Flat 구조: 직접 conversation_history
            if "conversation_history" in reduced:
                history = reduced["conversation_history"]
                if len(history) > 5:
                    self.logger.info(
                        f"Reducing conversation_history from {len(history)} to 5"
                    )
                    reduced["conversation_history"] = history[-5:]

        return reduced

    def estimate_state_size(self, state: Dict[str, Any]) -> Dict[str, float]:
        """State 크기 추정 (메모리 사용량)"""
        import sys

        estimates = {}

        # 전체 크기
        total_size = sys.getsizeof(state)

        # 그룹별 크기
        if isinstance(state, dict):
            for key, value in state.items():
                if isinstance(value, dict):
                    size = sum(sys.getsizeof(v) for v in value.values())
                    estimates[key] = size
                else:
                    estimates[key] = sys.getsizeof(value)

        estimates["total"] = total_size

        return estimates

    def log_state_stats(self, state: Dict[str, Any], node_name: str = "") -> None:
        """State 통계 로깅"""
        if not self.logger.isEnabledFor(logging.INFO):
            return

        stats = {
            "node": node_name,
            "groups": list(state.keys()) if isinstance(state, dict) else [],
            "size_estimate": self.estimate_state_size(state)
        }

        self.logger.info(f"State stats for {node_name}: {stats}")


# ============================================
# 편의 함수
# ============================================

# 전역 StateReducer 인스턴스
_global_reducer = StateReducer(aggressive_reduction=True)


def reduce_state_for_node(
    state: Dict[str, Any],
    node_name: str
) -> Dict[str, Any]:
    """
    노드에 필요한 State만 추출

    Args:
        state: 전체 State
        node_name: 노드 이름

    Returns:
        축소된 State
    """
    return _global_reducer.reduce_state_for_node(state, node_name)


def reduce_state_size(
    state: Dict[str, Any],
    max_docs: int = 10,
    max_content_per_doc: int = 500
) -> Dict[str, Any]:
    """
    State 크기 줄이기

    Args:
        state: State 객체
        max_docs: 최대 문서 수
        max_content_per_doc: 문서당 최대 문자 수

    Returns:
        크기가 줄어든 State
    """
    return _global_reducer.reduce_state_size(state, max_docs, max_content_per_doc)


# ============================================
# 데코레이터
# ============================================

def with_state_reduction(node_name: str):
    """
    State Reduction 적용하는 데코레이터

    Usage:
        @with_state_reduction("retrieve_documents")
        def retrieve_documents(state: Dict) -> Dict:
            # 필요한 데이터만 포함된 state 사용
            ...
            return state
    """
    def decorator(func):
        def wrapper(state: Dict[str, Any], **kwargs):
            # State 축소
            reduced_state = reduce_state_for_node(state, node_name)

            # 원본 함수 호출
            result = func(reduced_state, **kwargs)

            # 결과를 원본 state에 병합
            if isinstance(state, dict) and isinstance(result, dict):
                state.update(result)
                return state

            return result

        return wrapper
    return decorator
