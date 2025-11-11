# -*- coding: utf-8 -*-
"""
Context Expansion Processor
컨텍스트 확장 로직을 처리하는 프로세서
"""

import logging
import time
from typing import Any, Dict, List, Optional

from core.workflow.state.state_definitions import LegalWorkflowState


class ContextExpansionProcessor:
    """컨텍스트 확장 프로세서"""

    def __init__(
        self,
        search_handler,
        logger,
        keyword_search_func=None,
        get_state_value_func=None,
        set_state_value_func=None
    ):
        self.search_handler = search_handler
        self.logger = logger
        self.keyword_search_func = keyword_search_func
        self._get_state_value_func = get_state_value_func
        self._set_state_value_func = set_state_value_func

    def expand_context(
        self,
        state: LegalWorkflowState,
        validation_results: Dict[str, Any]
    ) -> LegalWorkflowState:
        """적응형 컨텍스트 확장"""
        try:
            if not validation_results.get("needs_expansion", False):
                return state

            existing_docs = self._get_state_value(state, "retrieved_docs", [])

            if not self.should_expand(validation_results, existing_docs):
                return state

            metadata = self._get_state_value(state, "metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            expansion_count = metadata.get("context_expansion_count", 0)
            if expansion_count >= 1:
                self.logger.info("Context expansion skipped: maximum expansion count reached")
                return state

            missing_info = validation_results.get("missing_information", [])
            query = self._get_state_value(state, "query", "")
            query_type = self._get_state_value(state, "query_type", "")

            self.logger.info(f"🔧 [CONTEXT EXPANSION] Expanding context for missing: {missing_info[:3]}")

            expansion_start_time = time.time()
            initial_doc_count = len(existing_docs)
            initial_overall_score = validation_results.get("overall_score", 0.0)

            expanded_query = self.build_expanded_query(query, missing_info, query_type)

            try:
                semantic_results, semantic_count = self.search_handler.semantic_search(expanded_query, k=5)
                keyword_results, keyword_count = [], 0
                if self.keyword_search_func:
                    keyword_results, keyword_count = self.keyword_search_func(
                        expanded_query,
                        query_type,
                        limit=3
                    )

                existing_docs = self._get_state_value(state, "retrieved_docs", [])
                all_docs = existing_docs + semantic_results + keyword_results

                seen_ids = set()
                unique_docs = []
                for doc in all_docs:
                    doc_id = doc.get("id") or hash(doc.get("content", "")[:100])
                    if isinstance(doc_id, (str, int)) and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        unique_docs.append(doc)

                expansion_end_time = time.time()
                expansion_duration = expansion_end_time - expansion_start_time
                added_doc_count = len(unique_docs) - initial_doc_count
                final_doc_count = len(unique_docs)

                expansion_stats = {
                    "expansion_count": expansion_count + 1,
                    "expansion_duration": expansion_duration,
                    "initial_doc_count": initial_doc_count,
                    "final_doc_count": final_doc_count,
                    "added_doc_count": added_doc_count,
                    "initial_overall_score": initial_overall_score,
                    "expanded_query": expanded_query,
                    "missing_info": missing_info[:3]
                }

                self._set_state_value(state, "retrieved_docs", unique_docs[:10])
                metadata["context_expansion_count"] = expansion_count + 1
                metadata["context_expansion_stats"] = expansion_stats
                self._set_state_value(state, "metadata", metadata)

                self.logger.info(
                    f"✅ [CONTEXT EXPANSION] Added {added_doc_count} documents, "
                    f"total: {final_doc_count} (duration: {expansion_duration:.2f}s, "
                    f"initial_score: {initial_overall_score:.2f})"
                )

            except Exception as e:
                self.logger.warning(f"Context expansion search failed: {e}")

            return state

        except Exception as e:
            self.logger.error(f"Adaptive context expansion failed: {e}")
            return state

    def should_expand(
        self,
        validation_results: Dict[str, Any],
        existing_docs: List[Dict[str, Any]]
    ) -> bool:
        """컨텍스트 확장 필요 여부 판단"""
        overall_score = validation_results.get("overall_score", 1.0)
        missing_info = validation_results.get("missing_information", [])
        missing_count = len(missing_info) if missing_info else 0
        avg_relevance = validation_results.get("avg_relevance", 0.0)

        if not existing_docs or len(existing_docs) == 0:
            self.logger.info(
                f"✅ [CONTEXT EXPANSION] Will expand: no existing docs "
                f"(overall_score={overall_score:.2f}, missing_info={missing_count})"
            )
            return True

        if existing_docs:
            relevance_scores = [
                doc.get("relevance_score", doc.get("score", 0.0))
                for doc in existing_docs
            ]
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

            if avg_relevance >= 0.3:
                self.logger.info(
                    f"🔍 [CONTEXT EXPANSION] Skipped: existing docs avg relevance ({avg_relevance:.2f}) >= 0.3 "
                    f"(overall_score={overall_score:.2f}, missing_info={missing_count}, docs={len(existing_docs)})"
                )
                return False

        if missing_count < 3:
            self.logger.info(
                f"🔍 [CONTEXT EXPANSION] Skipped: missing_info count ({missing_count}) < 3 "
                f"(overall_score={overall_score:.2f}, avg_relevance={avg_relevance:.2f}, docs={len(existing_docs) if existing_docs else 0})"
            )
            return False

        if overall_score >= 0.5:
            self.logger.info(
                f"🔍 [CONTEXT EXPANSION] Skipped: overall_score ({overall_score:.2f}) >= 0.5 "
                f"(missing_info={missing_count}, avg_relevance={avg_relevance:.2f}, docs={len(existing_docs) if existing_docs else 0})"
            )
            return False

        self.logger.info(
            f"✅ [CONTEXT EXPANSION] Will expand: overall_score={overall_score:.2f}, "
            f"missing_info={missing_count}, avg_relevance={avg_relevance:.2f}, docs={len(existing_docs) if existing_docs else 0}"
        )
        return True

    def build_expanded_query(
        self,
        query: str,
        missing_info: List[str],
        query_type: str
    ) -> str:
        """확장된 검색 쿼리 생성"""
        keywords = missing_info[:3]
        type_lower = query_type.lower() if query_type else ""
        if "precedent" in type_lower or "판례" in type_lower:
            expanded_query = f"{query} {' '.join(keywords)}"
        elif "law" in type_lower or "법령" in type_lower:
            expanded_query = query
        else:
            expanded_query = f"{query} {' '.join(keywords)}"

        return expanded_query

    def validate_context_quality(
        self,
        context: Dict[str, Any],
        query: str,
        query_type: str,
        extracted_keywords: List[str]
    ) -> Dict[str, Any]:
        """컨텍스트 품질 검증"""
        def calculate_relevance(context_text: str, query: str) -> float:
            if not context_text or not query:
                return 0.0
            query_words = set(query.lower().split())
            context_words = set(context_text.lower().split())
            if not query_words:
                return 0.0
            common_words = query_words.intersection(context_words)
            return len(common_words) / len(query_words)

        context_text = context.get("text", "") or context.get("content", "")
        if not context_text and isinstance(context, dict):
            context_text = str(context).get("text", "") or str(context).get("content", "")

        relevance = calculate_relevance(context_text, query)
        coverage = len(extracted_keywords) / max(len(query.split()), 1) if extracted_keywords else 0.0

        overall_score = (relevance * 0.7 + coverage * 0.3)

        missing_info = []
        if relevance < 0.3:
            missing_info.append("낮은 관련성")
        if coverage < 0.5:
            missing_info.append("키워드 커버리지 부족")

        return {
            "overall_score": overall_score,
            "relevance": relevance,
            "coverage": coverage,
            "missing_information": missing_info,
            "needs_expansion": overall_score < 0.5,
            "avg_relevance": relevance
        }

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

