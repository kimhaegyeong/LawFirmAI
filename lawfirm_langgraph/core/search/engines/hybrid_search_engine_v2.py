# -*- coding: utf-8 -*-
"""
Hybrid Search Engine V2
lawfirm_v2.db 기반 하이브리드 검색 엔진 (FTS5 + 벡터 검색 통합)
"""

import os
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Any, Dict, List, Optional

try:
    from lawfirm_langgraph.core.utils.config import Config
except ImportError:
    from core.utils.config import Config
from .exact_search_engine_v2 import ExactSearchEngineV2
try:
    from lawfirm_langgraph.core.classification.classifiers.question_classifier import QuestionClassifier, QuestionType
except ImportError:
    from core.classification.classifiers.question_classifier import QuestionClassifier, QuestionType
try:
    from lawfirm_langgraph.core.search.processors.result_merger import ResultMerger, ResultRanker
except ImportError:
    from core.search.processors.result_merger import ResultMerger, ResultRanker
from .semantic_search_engine_v2 import SemanticSearchEngineV2

logger = get_logger(__name__)


class HybridSearchEngineV2:
    """lawfirm_v2.db 기반 하이브리드 검색 엔진"""

    # 검색 타입 매핑
    SEARCH_TYPE_TO_SOURCE_TYPE = {
        "law": ["statute_article"],
        "precedent": ["case_paragraph"],
        "decision": ["decision_paragraph"],
        "interpretation": ["interpretation_paragraph"]
    }

    # 질문 유형별 가중치 설정
    QUESTION_TYPE_WEIGHTS = {
        "law_inquiry": {
            "exact": 0.6,
            "semantic": 0.4
        },
        "precedent_search": {
            "exact": 0.4,
            "semantic": 0.6
        },
        "complex_question": {
            "exact": 0.5,
            "semantic": 0.5
        }
    }

    # 기본 가중치
    DEFAULT_WEIGHTS = {"exact": 0.6, "semantic": 0.4}

    # 기본 검색 타입
    DEFAULT_SEARCH_TYPES = ["law", "precedent", "decision", "interpretation"]

    def __init__(self,
                 db_path: Optional[str] = None,
                 model_name: Optional[str] = None):
        """
        검색 엔진 초기화

        Args:
            db_path: lawfirm_v2.db 경로 (None이면 환경변수 DATABASE_PATH 사용)
            model_name: 임베딩 모델명 (None이면 환경변수 EMBEDDING_MODEL 또는 Config 사용)
        """
        config = Config()
        
        if db_path is None:
            db_path = config.database_path
        
        if model_name is None:
            model_name = os.getenv("EMBEDDING_MODEL") or config.embedding_model
        
        self.db_path = db_path
        self.model_name = model_name
        self.logger = get_logger(__name__)

        # MLflow 인덱스 사용 설정 확인
        use_mlflow_index = getattr(config, 'use_mlflow_index', False)
        mlflow_run_id = getattr(config, 'mlflow_run_id', None)

        # 검색 엔진 초기화
        self.exact_search = ExactSearchEngineV2(db_path)
        self.semantic_search = SemanticSearchEngineV2(
            db_path=db_path,
            model_name=model_name,
            use_mlflow_index=use_mlflow_index,
            mlflow_run_id=mlflow_run_id
        )
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
               include_semantic: bool = True) -> Dict[str, Any]:
        """
        하이브리드 검색 실행

        Args:
            query: 검색 쿼리
            search_types: 검색할 문서 타입 목록
            max_results: 최대 결과 수
            include_exact: FTS5 검색 포함 여부
            include_semantic: 벡터 검색 포함 여부

        Returns:
            검색 결과 딕셔너리
        """
        try:
            self.logger.info(f"Starting hybrid search for query: '{query}'")

            if max_results is None:
                max_results = self.search_config["max_results"]

            if search_types is None:
                search_types = self.DEFAULT_SEARCH_TYPES

            # 검색 결과 수집
            exact_results = {}
            semantic_results = []

            # 정확한 매칭 검색 (FTS5)
            if include_exact:
                exact_results = self._execute_exact_search(query, search_types)

            # 의미적 검색 (벡터)
            if include_semantic:
                semantic_results = self._execute_semantic_search(query, search_types)

            # 질문 유형 분석 및 가중치 동적 조정
            weights = self._get_weights_for_query_type(query)
            
            # 결과 통합 (동적 가중치 적용)
            merged_results = self.result_merger.merge_results(
                exact_results, semantic_results, weights=weights, query=query
            )

            # 결과 랭킹 및 필터링
            final_results = self._rank_and_filter_results(
                merged_results, max_results, query
            )

            # MergedResult를 Dict로 변환
            final_results_dict = self._convert_results_to_dict(final_results)

            return {
                "results": final_results_dict,
                "total": len(final_results_dict),
                "exact_count": self._count_exact_results(exact_results),
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

    def _execute_semantic_search(self, query: str, search_types: List[str]) -> List[Dict[str, Any]]:
        """벡터 기반 의미 검색 실행"""
        try:
            source_types = self._map_search_types_to_source_types(search_types)

            results = self.semantic_search.search(
                query,
                k=self.search_config["max_results"],
                source_types=source_types if source_types else None,
                similarity_threshold=self.search_config["semantic_threshold"]
            )
            return results
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return []

    def _get_weights_for_query_type(self, query: str) -> Dict[str, float]:
        """
        질문 유형에 따른 가중치 조회

        Args:
            query: 검색 쿼리

        Returns:
            가중치 딕셔너리
        """
        query_type = self.question_classifier.classify(query)
        weights = self.QUESTION_TYPE_WEIGHTS.get(query_type, self.DEFAULT_WEIGHTS)
        
        self.logger.info(
            f"Query type: {query_type}, weights: exact={weights['exact']:.2f}, "
            f"semantic={weights['semantic']:.2f}"
        )
        
        return weights

    def _map_search_types_to_source_types(self, search_types: List[str]) -> List[str]:
        """
        검색 타입을 source_type으로 매핑

        Args:
            search_types: 검색 타입 목록

        Returns:
            source_type 목록
        """
        source_types = []
        for search_type in search_types:
            if search_type in self.SEARCH_TYPE_TO_SOURCE_TYPE:
                source_types.extend(self.SEARCH_TYPE_TO_SOURCE_TYPE[search_type])
        return source_types

    def _rank_and_filter_results(self, merged_results: List[Any], max_results: int, query: str) -> List[Any]:
        """
        결과 랭킹 및 필터링

        Args:
            merged_results: 병합된 검색 결과
            max_results: 최대 결과 수
            query: 검색 쿼리

        Returns:
            랭킹 및 필터링된 결과
        """
        ranked_results = self.result_ranker.rank_results(
            merged_results, top_k=max_results, query=query
        )

        filtered_results = self.result_ranker.apply_diversity_filter(
            ranked_results, self.search_config["diversity_max_per_type"]
        )

        return filtered_results[:max_results]

    def _convert_results_to_dict(self, results: List[Any]) -> List[Dict[str, Any]]:
        """
        MergedResult를 Dict로 변환

        Args:
            results: 검색 결과 리스트

        Returns:
            Dict 형태의 결과 리스트
        """
        converted_results = []
        for result in results:
            if hasattr(result, 'text'):
                converted_results.append({
                    "text": result.text,
                    "score": result.score,
                    "metadata": result.metadata,
                    "type": getattr(result, 'type', 'unknown'),
                    "source": getattr(result, 'source', 'unknown'),
                    "relevance_score": result.score,
                    "search_type": getattr(result, 'search_type', 'hybrid')
                })
            elif isinstance(result, dict):
                converted_results.append(result)
        return converted_results

    def _count_exact_results(self, exact_results: Dict[str, Any]) -> int:
        """
        정확 검색 결과 수 계산

        Args:
            exact_results: 정확 검색 결과 딕셔너리

        Returns:
            결과 수
        """
        return sum(len(v) if isinstance(v, list) else 1 for v in exact_results.values())
