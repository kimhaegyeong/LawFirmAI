"""
결과 통합 및 랭킹 시스템
정확한 매칭과 의미적 검색 결과를 통합하고 랭킹하는 시스템
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

class ResultMerger:
    """검색 결과 통합 클래스"""
    
    def __init__(self):
        self.weights = {
            "exact_match": 1.0,
            "semantic": 0.8,
            "semantic_similarity": 0.6
        }
    
    def merge_results(self, 
                     exact_results: Dict[str, List[Dict[str, Any]]],
                     semantic_results: List[Dict[str, Any]],
                     query: str) -> List[Dict[str, Any]]:
        """검색 결과 통합"""
        try:
            logger.info("Starting result merging process")
            
            # 모든 결과를 하나의 리스트로 수집
            all_results = []
            
            # 정확한 매칭 결과 추가
            for doc_type, results in exact_results.items():
                for result in results:
                    result["doc_type"] = doc_type
                    result["merge_score"] = self._calculate_merge_score(result, query, "exact")
                    all_results.append(result)
            
            # 의미적 검색 결과 추가
            for result in semantic_results:
                result["doc_type"] = result.get("type", "unknown")
                result["merge_score"] = self._calculate_merge_score(result, query, "semantic")
                all_results.append(result)
            
            # 중복 제거 및 점수 통합
            merged_results = self._deduplicate_and_merge(all_results)
            
            # 최종 랭킹
            ranked_results = self._rank_results(merged_results, query)
            
            logger.info(f"Result merging completed: {len(ranked_results)} final results")
            return ranked_results
            
        except Exception as e:
            logger.error(f"Result merging failed: {e}")
            return []
    
    def _calculate_merge_score(self, result: Dict[str, Any], query: str, search_type: str) -> float:
        """통합 점수 계산"""
        try:
            base_score = result.get("relevance_score", 0.0)
            search_weight = self.weights.get(search_type, 0.5)
            
            # 쿼리 길이에 따른 가중치
            query_length_factor = min(len(query) / 10, 1.0)  # 최대 1.0
            
            # 문서 타입별 가중치
            doc_type_weights = {
                "law": 1.0,
                "precedent": 0.9,
                "constitutional": 0.8,
                "administrative_rule": 0.7,
                "legal_interpretation": 0.6
            }
            
            doc_type = result.get("doc_type", "unknown")
            doc_type_weight = doc_type_weights.get(doc_type, 0.5)
            
            # 최종 점수 계산
            merge_score = base_score * search_weight * query_length_factor * doc_type_weight
            
            return merge_score
            
        except Exception as e:
            logger.error(f"Failed to calculate merge score: {e}")
            return 0.0
    
    def _deduplicate_and_merge(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """중복 제거 및 결과 통합"""
        try:
            # ID별로 그룹화
            grouped_results = defaultdict(list)
            
            for result in results:
                doc_id = result.get("id")
                if doc_id:
                    grouped_results[doc_id].append(result)
            
            merged_results = []
            
            for doc_id, group in grouped_results.items():
                if len(group) == 1:
                    # 중복이 없는 경우
                    merged_results.append(group[0])
                else:
                    # 중복이 있는 경우 통합
                    merged_result = self._merge_duplicate_results(group)
                    merged_results.append(merged_result)
            
            return merged_results
            
        except Exception as e:
            logger.error(f"Failed to deduplicate results: {e}")
            return results
    
    def _merge_duplicate_results(self, duplicate_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """중복 결과 통합"""
        try:
            # 첫 번째 결과를 기본으로 사용
            merged = duplicate_results[0].copy()
            
            # 최고 점수 선택
            max_score = max(r.get("merge_score", 0.0) for r in duplicate_results)
            merged["merge_score"] = max_score
            
            # 검색 타입 통합
            search_types = [r.get("search_type", "") for r in duplicate_results]
            merged["search_types"] = list(set(search_types))
            
            # 관련성 점수 통합 (가중 평균)
            relevance_scores = [r.get("relevance_score", 0.0) for r in duplicate_results]
            weights = [self.weights.get(r.get("search_type", ""), 0.5) for r in duplicate_results]
            
            if sum(weights) > 0:
                weighted_avg = sum(s * w for s, w in zip(relevance_scores, weights)) / sum(weights)
                merged["relevance_score"] = weighted_avg
            
            return merged
            
        except Exception as e:
            logger.error(f"Failed to merge duplicate results: {e}")
            return duplicate_results[0]
    
    def _rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """결과 랭킹"""
        try:
            # 점수 기반 정렬
            ranked_results = sorted(results, key=lambda x: x.get("merge_score", 0.0), reverse=True)
            
            # 상위 결과에 랭킹 점수 추가
            for i, result in enumerate(ranked_results):
                result["rank"] = i + 1
                result["final_score"] = result.get("merge_score", 0.0)
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Failed to rank results: {e}")
            return results


class ResultRanker:
    """검색 결과 랭킹 클래스"""
    
    def __init__(self):
        self.ranking_factors = {
            "relevance_score": 0.4,
            "recency": 0.2,
            "authority": 0.2,
            "popularity": 0.1,
            "completeness": 0.1
        }
    
    def rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """검색 결과 랭킹"""
        try:
            logger.info(f"Ranking {len(results)} results")
            
            # 각 결과에 대해 랭킹 점수 계산
            for result in results:
                ranking_score = self._calculate_ranking_score(result, query)
                result["ranking_score"] = ranking_score
            
            # 랭킹 점수로 정렬
            ranked_results = sorted(results, key=lambda x: x.get("ranking_score", 0.0), reverse=True)
            
            # 최종 랭킹 번호 부여
            for i, result in enumerate(ranked_results):
                result["final_rank"] = i + 1
            
            logger.info("Ranking completed")
            return ranked_results
            
        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            return results
    
    def _calculate_ranking_score(self, result: Dict[str, Any], query: str) -> float:
        """랭킹 점수 계산"""
        try:
            score = 0.0
            
            # 관련성 점수
            relevance_score = result.get("relevance_score", 0.0)
            score += relevance_score * self.ranking_factors["relevance_score"]
            
            # 최신성 점수
            recency_score = self._calculate_recency_score(result)
            score += recency_score * self.ranking_factors["recency"]
            
            # 권위성 점수
            authority_score = self._calculate_authority_score(result)
            score += authority_score * self.ranking_factors["authority"]
            
            # 완성도 점수
            completeness_score = self._calculate_completeness_score(result)
            score += completeness_score * self.ranking_factors["completeness"]
            
            return score
            
        except Exception as e:
            logger.error(f"Failed to calculate ranking score: {e}")
            return 0.0
    
    def _calculate_recency_score(self, result: Dict[str, Any]) -> float:
        """최신성 점수 계산"""
        try:
            # 날짜 정보 추출
            date_fields = ["decision_date", "effective_date", "created_at"]
            date_value = None
            
            for field in date_fields:
                if result.get(field):
                    date_value = result[field]
                    break
            
            if not date_value:
                return 0.5  # 날짜 정보가 없으면 중간 점수
            
            # 간단한 최신성 점수 (실제로는 날짜 파싱 필요)
            # 여기서는 예시로 문자열 길이 기반 점수 사용
            return min(len(str(date_value)) / 10, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate recency score: {e}")
            return 0.5
    
    def _calculate_authority_score(self, result: Dict[str, Any]) -> float:
        """권위성 점수 계산"""
        try:
            doc_type = result.get("doc_type", "")
            court_name = result.get("court_name", "")
            
            # 문서 타입별 권위성
            type_authority = {
                "constitutional": 1.0,  # 헌재결정례
                "law": 0.9,           # 법령
                "precedent": 0.8,      # 판례
                "administrative_rule": 0.7,
                "legal_interpretation": 0.6
            }
            
            authority = type_authority.get(doc_type, 0.5)
            
            # 법원별 권위성 (판례의 경우)
            if doc_type == "precedent":
                court_authority = {
                    "대법원": 1.0,
                    "고등법원": 0.8,
                    "지방법원": 0.6,
                    "가정법원": 0.6,
                    "행정법원": 0.7
                }
                court_score = court_authority.get(court_name, 0.5)
                authority = (authority + court_score) / 2
            
            return authority
            
        except Exception as e:
            logger.error(f"Failed to calculate authority score: {e}")
            return 0.5
    
    def _calculate_completeness_score(self, result: Dict[str, Any]) -> float:
        """완성도 점수 계산"""
        try:
            content = result.get("content", "")
            title = result.get("title", "")
            
            # 내용 길이 기반 점수
            content_length = len(content)
            if content_length > 1000:
                length_score = 1.0
            elif content_length > 500:
                length_score = 0.8
            elif content_length > 200:
                length_score = 0.6
            else:
                length_score = 0.4
            
            # 제목 존재 여부
            title_score = 1.0 if title else 0.5
            
            # 필수 필드 존재 여부
            required_fields = ["id", "content"]
            field_score = sum(1 for field in required_fields if result.get(field)) / len(required_fields)
            
            # 종합 점수
            completeness = (length_score * 0.5 + title_score * 0.3 + field_score * 0.2)
            
            return completeness
            
        except Exception as e:
            logger.error(f"Failed to calculate completeness score: {e}")
            return 0.5
    
    def apply_diversity_filter(self, results: List[Dict[str, Any]], max_per_type: int = 5) -> List[Dict[str, Any]]:
        """다양성 필터 적용 (타입별 최대 개수 제한)"""
        try:
            type_counts = defaultdict(int)
            filtered_results = []
            
            for result in results:
                doc_type = result.get("doc_type", "unknown")
                
                if type_counts[doc_type] < max_per_type:
                    filtered_results.append(result)
                    type_counts[doc_type] += 1
            
            logger.info(f"Diversity filter applied: {len(filtered_results)} results")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Diversity filter failed: {e}")
            return results
