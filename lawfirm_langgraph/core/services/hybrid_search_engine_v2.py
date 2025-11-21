# -*- coding: utf-8 -*-
"""
Hybrid Search Engine V2
lawfirm_v2.db 기반 하이브리드 검색 엔진 (FTS5 + 벡터 검색 통합)
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Any, Dict, List, Optional

from ..utils.config import Config
from .exact_search_engine_v2 import ExactSearchEngineV2
from .question_classifier import QuestionClassifier, QuestionType
from .result_merger import ResultMerger, ResultRanker
from .semantic_search_engine_v2 import SemanticSearchEngineV2

logger = get_logger(__name__)


class HybridSearchEngineV2:
    """lawfirm_v2.db 기반 하이브리드 검색 엔진"""

    def __init__(self,
                 db_path: Optional[str] = None,
                 model_name: Optional[str] = None):
        """
        검색 엔진 초기화

        Args:
            db_path: lawfirm_v2.db 경로 (None이면 환경변수 DATABASE_PATH 사용)
            model_name: 임베딩 모델명 (None이면 환경변수 EMBEDDING_MODEL 또는 Config 사용)
        """
        if db_path is None:
            config = Config()
            db_path = config.database_path
        
        if model_name is None:
            import os
            model_name = os.getenv("EMBEDDING_MODEL")
            if model_name is None:
                config = Config()
                model_name = config.embedding_model
        
        self.db_path = db_path
        self.model_name = model_name
        self.logger = get_logger(__name__)

        # 검색 엔진 초기화
        self.exact_search = ExactSearchEngineV2(db_path)
        self.semantic_search = SemanticSearchEngineV2(db_path, model_name)
        self.question_classifier = QuestionClassifier()
        self.result_merger = ResultMerger()
        self.result_ranker = ResultRanker()

        # 검색 설정
        self.search_config = {
            "exact_search_weight": 0.6,
            "semantic_search_weight": 0.4,
            "max_results": 50,
            "semantic_threshold": 0.3,
            "diversity_max_per_type": 5
        }

        self.logger.info("HybridSearchEngineV2 initialized")

    def search(self,
               query: str,
               search_types: Optional[List[str]] = None,
               max_results: Optional[int] = None,
               include_exact: bool = True,
               include_semantic: bool = True,
               keyword_coverage: float = 0.0,
               extracted_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        하이브리드 검색 실행 (개선: Keyword Coverage 기반 동적 가중치 조정)

        Args:
            query: 검색 쿼리
            search_types: 검색할 문서 타입 목록
            max_results: 최대 결과 수
            include_exact: FTS5 검색 포함 여부
            include_semantic: 벡터 검색 포함 여부
            keyword_coverage: Keyword Coverage 점수 (0.0 ~ 1.0)
            extracted_keywords: 추출된 키워드 리스트

        Returns:
            검색 결과 딕셔너리
        """
        try:
            self.logger.info(f"Starting hybrid search for query: '{query}'")

            if max_results is None:
                max_results = self.search_config["max_results"]

            if search_types is None:
                search_types = ["law", "precedent", "decision", "interpretation"]

            # 질문 유형 분석 (개선: 질문 유형별 가중치 동적 조정)
            query_type = self.question_classifier.classify(query)
            
            # Keyword Coverage 계산 (제공되지 않은 경우)
            if keyword_coverage == 0.0 and extracted_keywords:
                # 간단한 키워드 매칭으로 coverage 추정
                query_lower = query.lower()
                matched_keywords = sum(1 for kw in extracted_keywords if isinstance(kw, str) and kw.lower() in query_lower)
                if extracted_keywords:
                    keyword_coverage = matched_keywords / len(extracted_keywords)
            
            # 질문 유형별 가중치 설정 (Keyword Coverage 기반 동적 조정)
            type_weights = self._get_query_type_weights(
                query_type,
                keyword_coverage=keyword_coverage,
                document_count=max_results
            )
            
            # 검색 결과 수집
            exact_results = {}
            semantic_results = []

            # 정확한 매칭 검색 (FTS5)
            if include_exact:
                exact_results = self._execute_exact_search(query, search_types)

            # 의미적 검색 (벡터) - 가중치에 따라 검색 결과 수 조정
            if include_semantic:
                semantic_k = int(max_results * type_weights["semantic"] * 2)
                semantic_results = self._execute_semantic_search(query, search_types, k=semantic_k)

            # 결과 통합 (가중치 적용)
            merged_results = self.result_merger.merge_results(
                exact_results, semantic_results, weights=type_weights, query=query
            )

            # 결과 랭킹
            ranked_results = self.result_ranker.rank_results(merged_results, top_k=max_results, query=query)

            # 다양성 필터 적용
            filtered_results = self.result_ranker.apply_diversity_filter(
                ranked_results, self.search_config["diversity_max_per_type"]
            )

            # 최종 결과 제한
            final_results = filtered_results[:max_results]

            # MergedResult를 Dict로 변환
            final_results_dict = []
            for result in final_results:
                if hasattr(result, 'text'):
                    final_results_dict.append({
                        "text": result.text,
                        "score": result.score,
                        "metadata": result.metadata,
                        "type": getattr(result, 'type', 'unknown'),
                        "source": getattr(result, 'source', 'unknown'),
                        "relevance_score": result.score,
                        "search_type": getattr(result, 'search_type', 'hybrid')
                    })
                elif isinstance(result, dict):
                    final_results_dict.append(result)

            return {
                "results": final_results_dict,
                "total": len(final_results_dict),
                "exact_count": sum(len(v) if isinstance(v, list) else 1 for v in exact_results.values()),
                "semantic_count": len(semantic_results),
                "query": query
            }

        except Exception as e:
            self.logger.error(f"Error in hybrid search: {e}", exc_info=True)
            return {
                "results": [],
                "total": 0,
                "exact_count": 0,
                "semantic_count": 0,
                "query": query,
                "error": str(e)
            }

    def _execute_exact_search(self, query: str, search_types: List[str]) -> Dict[str, Any]:
        """FTS5 기반 정확 검색 실행"""
        try:
            results = self.exact_search.search(query, search_types=search_types)
            return results
        except Exception as e:
            self.logger.error(f"Error in exact search: {e}")
            return {}

    def _execute_semantic_search(self, query: str, search_types: List[str], k: Optional[int] = None) -> List[Dict[str, Any]]:
        """벡터 기반 의미 검색 실행"""
        try:
            # search_types를 source_types로 매핑
            source_type_mapping = {
                "law": ["statute_article"],
                "precedent": ["case_paragraph"],
                "decision": ["decision_paragraph"],
                "interpretation": ["interpretation_paragraph"]
            }

            source_types = []
            for st in search_types:
                if st in source_type_mapping:
                    source_types.extend(source_type_mapping[st])

            if k is None:
                k = self.search_config["max_results"]

            results = self.semantic_search.search(
                query,
                k=k,
                source_types=source_types if source_types else None,
                similarity_threshold=self.search_config["semantic_threshold"]
            )
            return results
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return []
    
    def _get_query_type_weights(
        self, 
        query_type: QuestionType,
        keyword_coverage: float = 0.0,
        document_count: int = 10
    ) -> Dict[str, float]:
        """질문 유형별 가중치 반환 (개선: Keyword Coverage 기반 동적 조정)"""
        # 기본 가중치
        base_weights = {
            QuestionType.LAW_INQUIRY: {
                "exact": 0.6,  # 법령 조문은 키워드 검색이 중요
                "semantic": 0.4
            },
            QuestionType.PRECEDENT_SEARCH: {
                "exact": 0.4,  # 판례는 의미적 검색이 중요
                "semantic": 0.6
            },
            QuestionType.COMPLEX_QUESTION: {
                "exact": 0.5,  # 균형
                "semantic": 0.5
            },
            QuestionType.GENERAL: {
                "exact": 0.5,  # 기본값
                "semantic": 0.5
            }
        }
        
        weights = base_weights.get(query_type, {"exact": 0.5, "semantic": 0.5}).copy()
        
        # Keyword Coverage 기반 동적 조정 (개선: Phase 1)
        if keyword_coverage > 0.0:
            # Keyword Coverage가 낮으면 키워드 검색 가중치 증가
            if keyword_coverage < 0.5:
                keyword_boost = (0.5 - keyword_coverage) * 0.3  # 최대 0.15 증가
                weights["exact"] = min(0.7, weights["exact"] + keyword_boost)
                weights["semantic"] = 1.0 - weights["exact"]
                self.logger.debug(
                    f"Keyword Coverage 기반 가중치 조정: "
                    f"coverage={keyword_coverage:.3f}, "
                    f"exact={weights['exact']:.3f}, semantic={weights['semantic']:.3f}"
                )
        
        return weights
