# -*- coding: utf-8 -*-
"""
Query Utils Mixin
Query 관련 유틸리티 메서드들을 제공하는 Mixin 클래스
"""

from typing import Any, Dict, List, Optional

from core.agents.query_enhancer import QueryEnhancer
from core.agents.workflow_utils import WorkflowUtils


class QueryUtilsMixin:
    """Query 관련 유틸리티 메서드들을 제공하는 Mixin 클래스"""
    
    def _get_query_type_str(self, query_type) -> str:
        """WorkflowUtils.get_query_type_str 래퍼"""
        return WorkflowUtils.get_query_type_str(query_type)

    def _normalize_query_type_for_prompt(self, query_type) -> str:
        """WorkflowUtils.normalize_query_type_for_prompt 래퍼"""
        return WorkflowUtils.normalize_query_type_for_prompt(query_type, self.logger)

    def _optimize_search_query(
        self,
        query: str,
        query_type: str,
        extracted_keywords: List[str],
        legal_field: str
    ) -> Dict[str, Any]:
        """QueryEnhancer.optimize_search_query 래퍼"""
        return self.query_enhancer.optimize_search_query(
            query, query_type, extracted_keywords, legal_field
        )

    def _normalize_legal_terms(self, query: str, keywords: List[str]) -> List[str]:
        """QueryEnhancer.normalize_legal_terms 래퍼"""
        return self.query_enhancer.normalize_legal_terms(query, keywords)

    def _expand_legal_terms(
        self,
        terms: List[str],
        legal_field: str
    ) -> List[str]:
        """QueryEnhancer.expand_legal_terms 래퍼"""
        return self.query_enhancer.expand_legal_terms(terms, legal_field)

    def _clean_query_for_fallback(self, query: str) -> str:
        """QueryEnhancer.clean_query_for_fallback 래퍼"""
        return self.query_enhancer.clean_query_for_fallback(query)

    def _build_semantic_query(self, query: str, expanded_terms: List[str]) -> str:
        """QueryEnhancer.build_semantic_query 래퍼"""
        return self.query_enhancer.build_semantic_query(query, expanded_terms)

    def _build_keyword_queries(
        self,
        query: str,
        expanded_terms: List[str],
        query_type: str
    ) -> List[str]:
        """QueryEnhancer.build_keyword_queries 래퍼"""
        return self.query_enhancer.build_keyword_queries(query, expanded_terms, query_type)

    def _enhance_query_with_llm(
        self,
        query: str,
        query_type: str,
        extracted_keywords: List[str],
        legal_field: str
    ) -> Optional[Dict[str, Any]]:
        """QueryEnhancer.enhance_query_with_llm 래퍼"""
        return self.query_enhancer.enhance_query_with_llm(
            query, query_type, extracted_keywords, legal_field
        )

    def _build_query_enhancement_prompt(
        self,
        query: str,
        query_type: str,
        extracted_keywords: List[str],
        legal_field: str
    ) -> str:
        """QueryEnhancer.build_query_enhancement_prompt 래퍼"""
        return self.query_enhancer.build_query_enhancement_prompt(query, query_type, extracted_keywords, legal_field)

    def _format_field_info(self, legal_field: str, field_info: Dict[str, Any]) -> str:
        """QueryEnhancer.format_field_info 래퍼"""
        return self.query_enhancer.format_field_info(legal_field, field_info)

    def _enhance_query_with_chain(
        self,
        query: str,
        query_type: str,
        extracted_keywords: List[str],
        legal_field: str
    ) -> Optional[Dict[str, Any]]:
        """QueryEnhancer.enhance_query_with_chain 래퍼"""
        return self.query_enhancer.enhance_query_with_chain(
            query, query_type, extracted_keywords, legal_field
        )

    def _parse_llm_query_enhancement(self, llm_output: str) -> Optional[Dict[str, Any]]:
        """QueryEnhancer.parse_llm_query_enhancement 래퍼"""
        return self.query_enhancer.parse_llm_query_enhancement(llm_output)

    def _determine_search_parameters(
        self,
        query_type: str,
        query_complexity: int,
        keyword_count: int,
        is_retry: bool
    ) -> Dict[str, Any]:
        """QueryEnhancer.determine_search_parameters 래퍼"""
        return self.query_enhancer.determine_search_parameters(
            query_type, query_complexity, keyword_count, is_retry
        )

