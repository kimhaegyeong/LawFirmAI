# -*- coding: utf-8 -*-
"""
State Cache Helper Functions
전역 캐시에서 상태 값을 복구하는 유틸리티 함수들

리팩토링: legal_workflow_enhanced.py에서 중복된 캐시 복구 로직을 분리
"""

from typing import Any, Dict, Optional


def get_query_type_from_cache() -> Optional[str]:
    """
    전역 캐시에서 query_type을 복구
    
    Returns:
        query_type 문자열 또는 None
    """
    try:
        from core.agents.node_wrappers import _global_search_results_cache
        if _global_search_results_cache and isinstance(_global_search_results_cache, dict):
            # common.classification 그룹에서 찾기 (우선순위 1)
            if "common" in _global_search_results_cache and isinstance(_global_search_results_cache["common"], dict):
                if "classification" in _global_search_results_cache["common"] and isinstance(_global_search_results_cache["common"]["classification"], dict):
                    query_type = _global_search_results_cache["common"]["classification"].get("query_type", "")
                    if query_type:
                        return query_type
            # analysis에서 찾기 (우선순위 2)
            if "analysis" in _global_search_results_cache and isinstance(_global_search_results_cache["analysis"], dict):
                query_type = _global_search_results_cache["analysis"].get("query_type", "")
                if query_type:
                    return query_type
            # metadata에서 찾기 (우선순위 3)
            if "metadata" in _global_search_results_cache and isinstance(_global_search_results_cache["metadata"], dict):
                query_type = _global_search_results_cache["metadata"].get("query_type", "")
                if query_type:
                    return query_type
            # 최상위 레벨에서 찾기 (우선순위 4)
            query_type = _global_search_results_cache.get("query_type", "")
            if query_type:
                return query_type
    except Exception:
        pass
    return None


def get_prompt_optimized_context_from_cache() -> Optional[Dict[str, Any]]:
    """
    전역 캐시에서 prompt_optimized_context를 복구
    
    Returns:
        prompt_optimized_context 딕셔너리 또는 None
    """
    try:
        from core.agents.node_wrappers import _global_search_results_cache
        if _global_search_results_cache:
            context = (
                _global_search_results_cache.get("prompt_optimized_context", {}) or
                (_global_search_results_cache.get("search") and isinstance(_global_search_results_cache["search"], dict) and 
                 _global_search_results_cache["search"].get("prompt_optimized_context", {})) or
                (_global_search_results_cache.get("common") and isinstance(_global_search_results_cache["common"], dict) and 
                 _global_search_results_cache["common"].get("search") and isinstance(_global_search_results_cache["common"]["search"], dict) and
                 _global_search_results_cache["common"]["search"].get("prompt_optimized_context", {})) or
                {}
            )
            if context and isinstance(context, dict) and len(context) > 0:
                return context
    except Exception:
        pass
    return None


def get_retrieved_docs_from_cache() -> list:
    """
    전역 캐시에서 retrieved_docs를 복구
    
    Returns:
        retrieved_docs 리스트 (없으면 빈 리스트)
    """
    try:
        from core.agents.node_wrappers import _global_search_results_cache
        if _global_search_results_cache:
            retrieved_docs = (
                _global_search_results_cache.get("retrieved_docs", []) or
                (_global_search_results_cache.get("search") and isinstance(_global_search_results_cache["search"], dict) and 
                 _global_search_results_cache["search"].get("retrieved_docs", [])) or
                []
            )
            if retrieved_docs and isinstance(retrieved_docs, list):
                return retrieved_docs
    except Exception:
        pass
    return []

