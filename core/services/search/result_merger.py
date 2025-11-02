# -*- coding: utf-8 -*-
"""
Result Merger and Ranker
검색 결과 병합 및 순위 결정
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MergedResult:
    """병합된 검색 결과"""
    text: str
    score: float
    source: str
    metadata: Dict[str, Any]


class ResultMerger:
    """검색 결과 병합기"""

    def __init__(self):
        """병합기 초기화"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("ResultMerger initialized")

    def merge_results(self,
                     exact_results: Dict[str, List[Dict[str, Any]]],
                     semantic_results: List[Dict[str, Any]],
                     weights: Dict[str, float] = None) -> List[MergedResult]:
        """
        검색 결과 병합

        Args:
            exact_results: 정확한 검색 결과 (딕셔너리)
            semantic_results: 의미적 검색 결과 (리스트)
            weights: 가중치 딕셔너리

        Returns:
            List[MergedResult]: 병합된 결과
        """
        if weights is None:
            weights = {"exact": 0.7, "semantic": 0.3}

        merged_results = []

        # 정확한 검색 결과 처리 (딕셔너리 형태)
        for search_type, results in exact_results.items():
            for result in results:
                if isinstance(result, dict):
                    merged_result = MergedResult(
                        text=result.get('text', ''),
                        score=result.get('similarity', result.get('score', 0.0)) * weights["exact"],
                        source=f"exact_{search_type}",
                        metadata=result.get('metadata', result)
                    )
                    merged_results.append(merged_result)

        # 의미적 검색 결과 처리 (리스트 형태)
        for result in semantic_results:
            if isinstance(result, dict):
                # 개선 1.1: content 필드 우선, 없으면 text, 없으면 빈 문자열
                text_value = (
                    result.get('content') or
                    result.get('text') or
                    result.get('document') or
                    ''
                )

                # score 필드도 다양한 이름으로 시도
                score_value = (
                    result.get('similarity') or
                    result.get('score') or
                    result.get('relevance_score') or
                    0.0
                ) * weights["semantic"]

                merged_result = MergedResult(
                    text=str(text_value) if text_value else '',
                    score=score_value,
                    source="semantic",
                    metadata=result.get('metadata', result)
                )
                merged_results.append(merged_result)

        return merged_results


class ResultRanker:
    """검색 결과 순위 결정기"""

    def __init__(self):
        """순위 결정기 초기화"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("ResultRanker initialized")

    def rank_results(self, results: List[MergedResult], top_k: int = 10) -> List[MergedResult]:
        """
        검색 결과 순위 결정

        Args:
            results: 병합된 검색 결과
            top_k: 반환할 결과 수

        Returns:
            List[MergedResult]: 순위가 매겨진 결과
        """
        if not results:
            return []

        # 중복 제거 (텍스트 기준)
        unique_results = {}
        for result in results:
            if result.text not in unique_results:
                unique_results[result.text] = result
            else:
                # 더 높은 점수로 업데이트
                if result.score > unique_results[result.text].score:
                    unique_results[result.text] = result

        # 점수순 정렬
        ranked_results = list(unique_results.values())
        ranked_results.sort(key=lambda x: x.score, reverse=True)

        return ranked_results[:top_k]

    def apply_diversity_filter(self, results: List[MergedResult], max_per_type: int = 5) -> List[MergedResult]:
        """
        다양성 필터 적용

        Args:
            results: 순위가 매겨진 결과
            max_per_type: 타입별 최대 결과 수

        Returns:
            List[MergedResult]: 다양성이 적용된 결과
        """
        # 타입별 카운터
        type_counts = {}
        filtered_results = []

        for result in results:
            result_type = result.source
            if result_type not in type_counts:
                type_counts[result_type] = 0

            if type_counts[result_type] < max_per_type:
                filtered_results.append(result)
                type_counts[result_type] += 1

        return filtered_results


# 기본 인스턴스 생성
def create_result_merger() -> ResultMerger:
    """기본 결과 병합기 생성"""
    return ResultMerger()


def create_result_ranker() -> ResultRanker:
    """기본 결과 순위 결정기 생성"""
    return ResultRanker()


if __name__ == "__main__":
    # 테스트 코드
    merger = create_result_merger()
    ranker = create_result_ranker()

    # 샘플 결과
    exact_results = [
        type('obj', (object,), {'text': '민법 제543조', 'score': 0.9, 'metadata': {}})()
    ]
    semantic_results = [
        type('obj', (object,), {'text': '계약 해지 규정', 'score': 0.7, 'metadata': {}})()
    ]

    # 병합 및 순위 결정
    merged = merger.merge_results(exact_results, semantic_results)
    ranked = ranker.rank_results(merged)

    print(f"Ranked results: {len(ranked)}")
    for result in ranked:
        print(f"  Score: {result.score:.3f}, Source: {result.source}")
        print(f"  Text: {result.text}")
