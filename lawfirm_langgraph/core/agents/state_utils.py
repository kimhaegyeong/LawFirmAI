# -*- coding: utf-8 -*-
"""
State Utility Helper Module
LegalWorkflowState 최적화를 위한 유틸리티 함수들
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Any, Dict, List

logger = get_logger(__name__)

# Configuration constants
MAX_RETRIEVED_DOCS = 10
MAX_DOCUMENT_CONTENT_LENGTH = 500
MAX_CONVERSATION_HISTORY = 5
MAX_PROCESSING_STEPS = 20


def summarize_document(doc: Dict[str, Any], max_content_length: int = 500) -> Dict[str, Any]:
    """
    문서의 content 필드를 요약하여 크기를 줄임

    Args:
        doc: 문서 딕셔너리
        max_content_length: 최대 content 길이

    Returns:
        요약된 문서 딕셔너리
    """
    if not isinstance(doc, dict):
        return doc

    summarized_doc = doc.copy()

    # content 필드가 있는 경우 요약
    if "content" in summarized_doc and isinstance(summarized_doc["content"], str):
        original_content = summarized_doc["content"]

        if len(original_content) > max_content_length:
            # 앞부분과 뒷부분을 보존하되, 최종 길이가 max_content_length를 초과하지 않도록
            if max_content_length > 100:
                # truncate 메시지를 고려하여 각 부분의 길이 조정
                truncate_msg = f"\n... (truncated from {len(original_content)} chars) ...\n"
                available_space = max_content_length - len(truncate_msg)

                front_len = available_space // 2
                back_len = available_space - front_len

                front = original_content[:front_len]
                back = original_content[-back_len:]
                summarized_doc["content"] = f"{front}{truncate_msg}{back}"
            else:
                summarized_doc["content"] = original_content[:max_content_length] + "..."

            # 최종 길이 검증
            if len(summarized_doc["content"]) > max_content_length:
                summarized_doc["content"] = summarized_doc["content"][:max_content_length] + "..."

            # 요약 플래그 추가
            summarized_doc["is_summarized"] = True
            summarized_doc["original_content_length"] = len(original_content)
            logger.debug(f"Summarized document: {len(original_content)} → {len(summarized_doc['content'])} chars")
        else:
            summarized_doc["is_summarized"] = False

    return summarized_doc


def prune_retrieved_docs(
    docs: List[Dict],
    max_items: int = 10,
    max_content_per_doc: int = 500
) -> List[Dict]:
    """
    검색된 문서 목록을 정제하고 요약

    Args:
        docs: 문서 목록
        max_items: 최대 문서 수
        max_content_per_doc: 문서당 최대 content 길이

    Returns:
        정제 및 요약된 문서 목록
    """
    if not docs:
        return []

    # relevance_score 기준으로 정렬 (내림차순)
    sorted_docs = sorted(
        docs,
        key=lambda x: x.get("relevance_score", x.get("score", 0.0)),
        reverse=True
    )

    # 상위 N개만 선택
    top_docs = sorted_docs[:max_items]

    # 각 문서의 content 요약
    pruned_docs = [
        summarize_document(doc, max_content_per_doc)
        for doc in top_docs
    ]

    logger.info(f"Pruned retrieved_docs: {len(docs)} → {len(pruned_docs)}")

    return pruned_docs


def prune_conversation_history(history: List[Dict], max_items: int = 5) -> List[Dict]:
    """
    대화 이력을 정제

    Args:
        history: 대화 이력
        max_items: 최대 항목 수

    Returns:
        정제된 대화 이력 (최근 N개만 유지)
    """
    if not history:
        return []

    if len(history) <= max_items:
        return history

    # 최근 N개만 유지
    pruned_history = history[-max_items:]

    logger.debug(f"Pruned conversation_history: {len(history)} → {len(pruned_history)}")

    return pruned_history


def prune_processing_steps(steps: List[str], max_items: int = 20) -> List[str]:
    """
    처리 단계 목록을 정제

    Args:
        steps: 처리 단계 목록
        max_items: 최대 항목 수

    Returns:
        정제된 처리 단계 목록 (최근 N개만 유지)
    """
    if not steps:
        return []

    if len(steps) <= max_items:
        return steps

    # 최근 N개만 유지
    pruned_steps = steps[-max_items:]

    logger.debug(f"Pruned processing_steps: {len(steps)} → {len(pruned_steps)}")

    return pruned_steps


def consolidate_metadata(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    여러 개의 메타데이터 딕셔너리를 단일 metadata 딕셔너리로 통합

    Args:
        state: 워크플로우 상태

    Returns:
        통합된 metadata 딕셔너리
    """
    consolidated = {}

    # 기존 metadata가 있으면 복사
    if "metadata" in state and isinstance(state["metadata"], dict):
        consolidated = state["metadata"].copy()

    # search_metadata 통합
    if "search_metadata" in state and isinstance(state["search_metadata"], dict):
        consolidated["search"] = state["search_metadata"]

    # format_metadata 통합
    if "format_metadata" in state and isinstance(state["format_metadata"], dict):
        consolidated["format"] = state["format_metadata"]

    # quality_metrics 통합
    if "quality_metrics" in state and isinstance(state["quality_metrics"], dict):
        consolidated["quality"] = state["quality_metrics"]

    return consolidated
