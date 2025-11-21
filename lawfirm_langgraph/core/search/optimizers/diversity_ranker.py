# -*- coding: utf-8 -*-
"""
Diversity Ranker
MMR (Maximal Marginal Relevance) 기반 다양성 보장 랭커
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import List, Dict, Any, Optional
import numpy as np

logger = get_logger(__name__)


class DiversityRanker:
    """MMR 기반 다양성 보장 랭커"""
    
    def __init__(self):
        """초기화"""
        self.logger = get_logger(__name__)
        self.logger.info("DiversityRanker initialized")
    
    def rank_with_diversity(
        self,
        results: List[Dict[str, Any]],
        query: str,
        lambda_param: float = 0.5,
        top_k: int = 10,
        similarity_key: str = "relevance_score"
    ) -> List[Dict[str, Any]]:
        """
        MMR 기반 다양성 보장 랭킹
        
        Args:
            results: 검색 결과 리스트
            query: 검색 쿼리
            lambda_param: MMR 람다 파라미터 (0.0=다양성만, 1.0=관련성만)
            top_k: 반환할 결과 수
            similarity_key: 유사도 점수 키
        
        Returns:
            List[Dict[str, Any]]: 다양성을 고려한 랭킹된 결과
        """
        try:
            if not results:
                return []
            
            if len(results) <= top_k:
                # 결과가 적으면 그대로 반환
                return results
            
            # MMR 알고리즘 적용
            selected = []
            remaining = results.copy()
            
            # 첫 번째 결과는 가장 관련성 높은 것으로 선택
            first_result = max(remaining, key=lambda x: x.get(similarity_key, 0.0))
            selected.append(first_result)
            remaining.remove(first_result)
            
            # 나머지 결과 선택
            while len(selected) < top_k and remaining:
                best_score = -float('inf')
                best_result = None
                
                for candidate in remaining:
                    # 관련성 점수
                    relevance = candidate.get(similarity_key, 0.0)
                    
                    # 다양성 점수 (이미 선택된 결과와의 최대 유사도)
                    max_similarity = 0.0
                    for selected_result in selected:
                        similarity = self._calculate_similarity(
                            candidate, selected_result
                        )
                        max_similarity = max(max_similarity, similarity)
                    
                    diversity = 1.0 - max_similarity
                    
                    # MMR 점수
                    mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_result = candidate
                
                if best_result:
                    selected.append(best_result)
                    remaining.remove(best_result)
                else:
                    break
            
            self.logger.debug(
                f"MMR ranking completed: {len(selected)} results "
                f"(lambda={lambda_param})"
            )
            
            return selected
        
        except Exception as e:
            self.logger.error(f"MMR ranking failed: {e}")
            # 실패 시 관련성만으로 정렬
            return sorted(
                results,
                key=lambda x: x.get(similarity_key, 0.0),
                reverse=True
            )[:top_k]
    
    def _calculate_similarity(
        self,
        doc1: Dict[str, Any],
        doc2: Dict[str, Any]
    ) -> float:
        """
        두 문서 간 유사도 계산
        
        Args:
            doc1: 첫 번째 문서
            doc2: 두 번째 문서
        
        Returns:
            float: 유사도 (0.0-1.0)
        """
        try:
            # 1. 텍스트 유사도 (간단한 Jaccard 유사도)
            text1 = doc1.get("text", doc1.get("content", ""))
            text2 = doc2.get("text", doc2.get("content", ""))
            
            if not text1 or not text2:
                return 0.0
            
            # 단어 집합
            words1 = set(text1.split())
            words2 = set(text2.split())
            
            if not words1 or not words2:
                return 0.0
            
            # Jaccard 유사도
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            jaccard = intersection / union if union > 0 else 0.0
            
            # 2. 메타데이터 유사도
            metadata_similarity = 0.0
            
            # 출처가 같으면 높은 유사도
            source1 = doc1.get("source", doc1.get("title", ""))
            source2 = doc2.get("source", doc2.get("title", ""))
            
            if source1 and source2 and source1 == source2:
                metadata_similarity += 0.3
            
            # 카테고리가 같으면 유사도 증가
            category1 = doc1.get("metadata", {}).get("category", "")
            category2 = doc2.get("metadata", {}).get("category", "")
            
            if category1 and category2 and category1 == category2:
                metadata_similarity += 0.2
            
            # 최종 유사도 (텍스트 70%, 메타데이터 30%)
            final_similarity = 0.7 * jaccard + 0.3 * metadata_similarity
            
            return min(1.0, final_similarity)
        
        except Exception as e:
            self.logger.warning(f"Similarity calculation failed: {e}")
            return 0.0
    
    def ensure_category_diversity(
        self,
        results: List[Dict[str, Any]],
        top_k: int = 10,
        min_per_category: int = 1
    ) -> List[Dict[str, Any]]:
        """
        카테고리 다양성 보장
        
        Args:
            results: 검색 결과
            top_k: 반환할 결과 수
            min_per_category: 카테고리당 최소 결과 수
        
        Returns:
            List[Dict[str, Any]]: 카테고리 다양성을 고려한 결과
        """
        try:
            if not results:
                return []
            
            # 카테고리별로 그룹화
            categories = {}
            no_category = []
            
            for result in results:
                category = result.get("metadata", {}).get("category", "")
                if category:
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(result)
                else:
                    no_category.append(result)
            
            # 각 카테고리에서 최소 개수만큼 선택
            selected = []
            for category, category_results in categories.items():
                selected.extend(category_results[:min_per_category])
            
            # 나머지는 관련성 순으로 추가
            remaining = []
            for category, category_results in categories.items():
                remaining.extend(category_results[min_per_category:])
            remaining.extend(no_category)
            
            # 관련성 순 정렬
            remaining.sort(
                key=lambda x: x.get("relevance_score", 0.0),
                reverse=True
            )
            
            # top_k까지 채우기
            selected.extend(remaining[:top_k - len(selected)])
            
            return selected[:top_k]
        
        except Exception as e:
            self.logger.error(f"Category diversity failed: {e}")
            return results[:top_k]

