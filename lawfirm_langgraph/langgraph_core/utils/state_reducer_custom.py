# -*- coding: utf-8 -*-
"""
Custom State Reducer for LangGraph
TypedDict reducer의 기본 동작을 확장하여 search 그룹 보존

LangGraph의 기본 TypedDict reducer는 TypedDict에 정의된 필드만 보존하는데,
이 커스텀 reducer는 search 그룹 내부의 모든 필드도 보존합니다.

NOTE: 이 reducer는 현재 사용되지 않습니다.
      Phase 1에서 Annotated reducer를 적용했으므로 커스텀 reducer가 더 이상 필요하지 않습니다.
      하지만 하위 호환성을 위해 보관합니다.
"""

from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


def custom_state_reducer(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """
    커스텀 State Reducer - search 그룹 보존
    
    DEPRECATED: Annotated reducer (merge_search_lists, merge_dict_updates) 사용 권장
                이 reducer는 하위 호환성을 위해 보관됩니다.

    LangGraph의 기본 reducer는 TypedDict의 필드만 병합하는데,
    이 reducer는 search 그룹 내부의 모든 필드도 병합합니다.

    Args:
        left: 기존 state (누적된 상태)
        right: 노드 반환값 (새로운 업데이트)

    Returns:
        병합된 state
    """
    result = {}

    # 1. left (기존 state)의 모든 필드 복사
    if isinstance(left, dict):
        result.update(left)

    # 2. right (노드 반환값)의 필드로 업데이트
    if isinstance(right, dict):
        # 일반 필드는 기본 업데이트
        for key, value in right.items():
            if key == "search" and isinstance(value, dict):
                # search 그룹의 경우 특별 처리
                if "search" not in result or not isinstance(result.get("search"), dict):
                    result["search"] = {}

                # search 그룹 내부의 모든 필드 병합
                result_search = result["search"]
                for search_key, search_value in value.items():
                    # 리스트나 딕셔너리는 병합, 아닌 것은 덮어쓰기
                    if search_key in ["semantic_results", "keyword_results", "merged_documents"]:
                        # 리스트 필드: right의 값이 있으면 그것을 사용
                        if search_value:
                            result_search[search_key] = search_value
                        elif search_key not in result_search:
                            result_search[search_key] = []
                    elif search_key in ["optimized_queries", "search_params", "keyword_weights", "prompt_optimized_context"]:
                        # 딕셔너리 필드: 병합
                        if isinstance(search_value, dict) and search_key in result_search and isinstance(result_search[search_key], dict):
                            result_search[search_key] = {**result_search[search_key], **search_value}
                        elif search_value:
                            result_search[search_key] = search_value
                    else:
                        # 일반 필드: right의 값이 있으면 사용
                        if search_value or search_key not in result_search:
                            result_search[search_key] = search_value

                # 디버깅: search 그룹 병합 확인
                semantic_count = len(result_search.get("semantic_results", []))
                keyword_count = len(result_search.get("keyword_results", []))
                if semantic_count > 0 or keyword_count > 0:
                    print(f"[DEBUG] custom_state_reducer: Merged search group - semantic_results={semantic_count}, keyword_results={keyword_count}")
            else:
                # 일반 필드는 기본 업데이트
                result[key] = value

    return result
