# -*- coding: utf-8 -*-
"""
Search Quality Improvement Integration Example
검색 품질 개선 기능 통합 예제
"""

import logging
from typing import List, Dict, Any, Optional

from core.search.optimizers.enhanced_query_expander import EnhancedQueryExpander
from core.search.optimizers.adaptive_hybrid_weights import AdaptiveHybridWeights
from core.search.optimizers.adaptive_threshold import AdaptiveThreshold
from core.search.optimizers.diversity_ranker import DiversityRanker
from core.search.optimizers.advanced_reranker import AdvancedReranker
from core.search.optimizers.multi_dimensional_quality import MultiDimensionalQualityScorer
from core.search.optimizers.metadata_enhancer import MetadataEnhancer
from core.search.mlflow_tracker import SearchQualityTracker

logger = logging.getLogger(__name__)


class ImprovedSearchPipeline:
    """개선된 검색 파이프라인"""
    
    def __init__(
        self,
        semantic_search_engine: Any,
        keyword_search_engine: Any,
        result_merger: Any,
        result_ranker: Any,
        enable_mlflow: bool = True
    ):
        """
        초기화
        
        Args:
            semantic_search_engine: 의미적 검색 엔진
            keyword_search_engine: 키워드 검색 엔진
            result_merger: 결과 병합기
            result_ranker: 결과 재정렬기
            enable_mlflow: MLflow 추적 활성화 여부
        """
        self.semantic_search_engine = semantic_search_engine
        self.keyword_search_engine = keyword_search_engine
        self.result_merger = result_merger
        self.result_ranker = result_ranker
        
        # 개선 기능 초기화
        self.query_expander = EnhancedQueryExpander()
        self.weight_calculator = AdaptiveHybridWeights()
        self.threshold_calculator = AdaptiveThreshold()
        self.diversity_ranker = DiversityRanker()
        self.advanced_reranker = AdvancedReranker(
            primary_model="ko-reranker",
            use_ensemble=False
        )
        self.quality_scorer = MultiDimensionalQualityScorer()
        self.metadata_enhancer = MetadataEnhancer()
        
        # MLflow 추적기
        self.mlflow_tracker = SearchQualityTracker() if enable_mlflow else None
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("ImprovedSearchPipeline initialized")
    
    def search(
        self,
        query: str,
        query_type: str = "general_question",
        extracted_keywords: Optional[List[str]] = None,
        top_k: int = 10,
        use_query_expansion: bool = True,
        use_adaptive_weights: bool = True,
        use_adaptive_threshold: bool = True,
        use_diversity: bool = True,
        track_mlflow: bool = True
    ) -> List[Dict[str, Any]]:
        """
        개선된 검색 실행
        
        Args:
            query: 검색 쿼리
            query_type: 질문 유형
            extracted_keywords: 추출된 키워드
            top_k: 반환할 결과 수
            use_query_expansion: Query Expansion 사용 여부
            use_adaptive_weights: 동적 가중치 사용 여부
            use_adaptive_threshold: 적응형 임계값 사용 여부
            use_diversity: 다양성 보장 사용 여부
            track_mlflow: MLflow 추적 여부
        
        Returns:
            List[Dict[str, Any]]: 검색 결과
        """
        try:
            # 1. Query Expansion
            expanded_query = None
            if use_query_expansion:
                expanded_query = self.query_expander.expand_query(
                    query=query,
                    query_type=query_type,
                    extracted_keywords=extracted_keywords
                )
                self.logger.info(f"Query expanded: {len(expanded_query.expanded_keywords)} keywords")
            
            # 2. 적응형 임계값 계산
            threshold = 0.5
            if use_adaptive_threshold:
                threshold = self.threshold_calculator.calculate_threshold(
                    query=query,
                    query_type=query_type
                )
                self.logger.info(f"Adaptive threshold: {threshold:.3f}")
            
            # 3. 의미적 검색
            semantic_results, semantic_count = self.semantic_search_engine.search(
                query=query,
                k=top_k * 2,  # 다양성을 위해 더 많이 검색
                similarity_threshold=threshold
            )
            
            # 4. 키워드 검색
            keyword_results, keyword_count = self.keyword_search_engine.search(
                query=query,
                k=top_k * 2
            )
            
            # 5. Keyword Coverage 계산 (간단한 버전)
            keyword_coverage = self._calculate_keyword_coverage(
                results=semantic_results + keyword_results,
                keywords=extracted_keywords or []
            )
            
            # 6. 동적 가중치 계산
            weights = {"semantic": 0.6, "keyword": 0.4}
            if use_adaptive_weights:
                weights = self.weight_calculator.calculate_weights(
                    query=query,
                    query_type=query_type,
                    keyword_coverage=keyword_coverage
                )
                self.logger.info(f"Adaptive weights: semantic={weights['semantic']:.2f}, keyword={weights['keyword']:.2f}")
            
            # 7. 결과 병합
            merged_results = self.result_merger.merge_results(
                exact_results={"semantic": semantic_results},
                semantic_results=keyword_results,
                weights=weights,
                query=query
            )
            
            # 8. 메타데이터 강화 및 부스팅
            for doc in merged_results:
                metadata_boost = self.metadata_enhancer.boost_by_metadata(
                    doc, query, query_type
                )
                original_score = doc.get("relevance_score", doc.get("similarity", 0.0))
                doc["metadata_boost"] = metadata_boost
                doc["boosted_score"] = original_score * (1.0 + metadata_boost * 0.2)
            
            # 9. 재정렬
            ranked_results = self.result_ranker.rank_results(
                results=merged_results,
                top_k=top_k * 2,
                query=query
            )
            
            # 10. 고급 Reranking (선택적)
            if self.advanced_reranker.primary_reranker:
                reranked_results = self.advanced_reranker.rerank(
                    query=query,
                    documents=ranked_results[:top_k * 2],
                    top_k=top_k * 2
                )
                ranked_results = reranked_results
            
            # 11. 다차원 품질 점수 계산
            for doc in ranked_results:
                quality_scores = self.quality_scorer.calculate_quality(
                    doc, query, query_type, extracted_keywords
                )
                doc["quality_scores"] = {
                    "relevance": quality_scores.relevance,
                    "accuracy": quality_scores.accuracy,
                    "completeness": quality_scores.completeness,
                    "recency": quality_scores.recency,
                    "source_credibility": quality_scores.source_credibility,
                    "overall": quality_scores.overall
                }
                # 품질 점수를 최종 점수에 반영
                final_score = doc.get("final_score", doc.get("relevance_score", 0.0))
                doc["final_score"] = 0.7 * final_score + 0.3 * quality_scores.overall
            
            # 12. 최종 점수로 재정렬
            ranked_results.sort(
                key=lambda x: x.get("final_score", x.get("relevance_score", 0.0)),
                reverse=True
            )
            
            # 13. 다양성 보장
            if use_diversity:
                diverse_results = self.diversity_ranker.rank_with_diversity(
                    results=ranked_results,
                    query=query,
                    lambda_param=0.5,
                    top_k=top_k
                )
                final_results = diverse_results
            else:
                final_results = ranked_results[:top_k]
            
            # 10. MLflow 추적
            if track_mlflow and self.mlflow_tracker:
                self.mlflow_tracker.track_search_experiment(
                    query=query,
                    results=final_results,
                    feature_name="improved_search_pipeline",
                    params={
                        "similarity_threshold": threshold,
                        "semantic_weight": weights["semantic"],
                        "keyword_weight": weights["keyword"],
                        "use_query_expansion": use_query_expansion,
                        "use_adaptive_weights": use_adaptive_weights,
                        "use_adaptive_threshold": use_adaptive_threshold,
                        "use_diversity": use_diversity
                    },
                    query_type=query_type
                )
            
            self.logger.info(f"Search completed: {len(final_results)} results")
            return final_results
        
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def _calculate_keyword_coverage(
        self,
        results: List[Dict[str, Any]],
        keywords: List[str]
    ) -> float:
        """Keyword Coverage 계산"""
        if not keywords or not results:
            return 0.5
        
        matched_count = 0
        for result in results[:10]:  # 상위 10개만 확인
            text = result.get("text", result.get("content", ""))
            if text:
                text_lower = text.lower()
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        matched_count += 1
                        break
        
        coverage = matched_count / min(len(keywords), 10)
        return min(1.0, coverage)

