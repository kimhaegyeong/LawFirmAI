# -*- coding: utf-8 -*-
"""
검색 결과 처리 병렬 Task 모듈
LangGraph의 병렬 처리 패턴을 활용한 비동기 작업들
"""

import asyncio
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Any, Dict, List, Optional, Tuple

logger = get_logger(__name__)


class SearchResultTasks:
    """검색 결과 처리 병렬 Task 클래스"""

    @staticmethod
    async def evaluate_quality_parallel(
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        query: str,
        query_type: str,
        search_params: Dict[str, Any],
        evaluate_semantic_func: Optional[callable] = None,
        evaluate_keyword_func: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        품질 평가 병렬 실행
        
        Args:
            semantic_results: 의미 검색 결과
            keyword_results: 키워드 검색 결과
            query: 질의
            query_type: 질의 유형
            search_params: 검색 파라미터
            evaluate_semantic_func: 의미 검색 품질 평가 함수
            evaluate_keyword_func: 키워드 검색 품질 평가 함수
        
        Returns:
            품질 평가 결과 딕셔너리
        """
        async def eval_semantic():
            if evaluate_semantic_func:
                return evaluate_semantic_func(
                    semantic_results=semantic_results,
                    query=query,
                    query_type=query_type,
                    min_results=search_params.get("semantic_k", 5) // 2
                )
            return {
                "score": 0.8 if len(semantic_results) >= 5 else 0.5,
                "needs_retry": len(semantic_results) < 5
            }

        async def eval_keyword():
            if evaluate_keyword_func:
                return evaluate_keyword_func(
                    keyword_results=keyword_results,
                    query=query,
                    query_type=query_type,
                    min_results=search_params.get("keyword_limit", 3) // 2
                )
            return {
                "score": 0.8 if len(keyword_results) >= 3 else 0.5,
                "needs_retry": len(keyword_results) < 3
            }

        semantic_quality, keyword_quality = await asyncio.gather(
            eval_semantic(),
            eval_keyword()
        )

        overall_quality = (semantic_quality["score"] + keyword_quality["score"]) / 2.0
        needs_retry = semantic_quality.get("needs_retry", False) or keyword_quality.get("needs_retry", False)

        return {
            "semantic_quality": semantic_quality,
            "keyword_quality": keyword_quality,
            "overall_quality": overall_quality,
            "needs_retry": needs_retry
        }

    @staticmethod
    async def apply_keyword_weights_parallel(
        documents: List[Dict[str, Any]],
        extracted_keywords: List[str],
        query: str,
        query_type: str,
        legal_field: str,
        calculate_keyword_weights_func: Optional[callable] = None,
        calculate_keyword_match_score_func: Optional[callable] = None,
        calculate_weighted_final_score_func: Optional[callable] = None,
        search_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        키워드 가중치 적용 병렬 실행
        
        Args:
            documents: 문서 리스트
            extracted_keywords: 추출된 키워드
            query: 질의
            query_type: 질의 유형
            legal_field: 법률 분야
            calculate_keyword_weights_func: 키워드 가중치 계산 함수
            calculate_keyword_match_score_func: 키워드 매칭 점수 계산 함수
            calculate_weighted_final_score_func: 가중치 최종 점수 계산 함수
            search_params: 검색 파라미터
        
        Returns:
            가중치가 적용된 문서 리스트 (점수순 정렬)
        """
        if not documents:
            return []

        search_params = search_params or {}

        if calculate_keyword_weights_func:
            keyword_weights = calculate_keyword_weights_func(
                extracted_keywords=extracted_keywords,
                query=query,
                query_type=query_type,
                legal_field=legal_field
            )
        else:
            keyword_weights = {kw: 1.0 for kw in extracted_keywords}

        async def apply_weight(doc: Dict[str, Any]) -> Dict[str, Any]:
            if calculate_keyword_match_score_func:
                keyword_scores = calculate_keyword_match_score_func(
                    document=doc,
                    keyword_weights=keyword_weights,
                    query=query
                )
            else:
                keyword_scores = {
                    "keyword_match_score": 0.5,
                    "keyword_coverage": 0.5,
                    "matched_keywords": [],
                    "weighted_keyword_score": 0.5
                }

            if calculate_weighted_final_score_func:
                final_score = calculate_weighted_final_score_func(
                    document=doc,
                    keyword_scores=keyword_scores,
                    search_params=search_params,
                    query_type=query_type
                )
            else:
                final_score = doc.get("relevance_score", 0.0) * 0.7 + keyword_scores.get("weighted_keyword_score", 0.0) * 0.3

            doc["keyword_match_score"] = keyword_scores.get("keyword_match_score", 0.0)
            doc["keyword_coverage"] = keyword_scores.get("keyword_coverage", 0.0)
            doc["matched_keywords"] = keyword_scores.get("matched_keywords", [])
            doc["weighted_keyword_score"] = keyword_scores.get("weighted_keyword_score", 0.0)
            doc["final_weighted_score"] = final_score

            return doc

        tasks = [apply_weight(doc) for doc in documents]
        weighted_docs = await asyncio.gather(*tasks)

        return sorted(weighted_docs, key=lambda x: x.get("final_weighted_score", x.get("relevance_score", 0.0)), reverse=True)

    @staticmethod
    async def filter_documents_parallel(
        documents: List[Dict[str, Any]],
        min_relevance: float = 0.80,
        min_content_length: int = 5,
        min_final_score: float = 0.55
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        문서 필터링 병렬 실행
        
        Args:
            documents: 문서 리스트
            min_relevance: 최소 관련도 점수
            min_content_length: 최소 콘텐츠 길이
            min_final_score: 최소 최종 점수
        
        Returns:
            (필터링된 문서 리스트, 통계 딕셔너리)
        """
        if not documents:
            return [], {"total": 0, "filtered": 0, "skipped": 0}

        async def filter_doc(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            relevance_score = (
                doc.get("relevance_score") or
                doc.get("score") or
                doc.get("similarity") or
                0.0
            )
            
            final_weighted_score = doc.get("final_weighted_score", relevance_score)
            
            content = (
                doc.get("content", "") or
                doc.get("text", "") or
                doc.get("content_text", "") or
                ""
            )
            
            if not isinstance(content, str):
                content = str(content) if content else ""

            if relevance_score < min_relevance:
                return None
            
            if final_weighted_score < min_final_score:
                return None
            
            if not content or len(content.strip()) < min_content_length:
                return None

            return doc

        tasks = [filter_doc(doc) for doc in documents]
        filtered = await asyncio.gather(*tasks)

        valid_docs = [doc for doc in filtered if doc is not None]
        stats = {
            "total": len(documents),
            "filtered": len(valid_docs),
            "skipped": len(documents) - len(valid_docs)
        }

        return valid_docs, stats

