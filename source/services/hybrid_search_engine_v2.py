# -*- coding: utf-8 -*-
"""
Hybrid Search Engine V2
lawfirm_v2.db 기반 하이브리드 검색 엔진 (FTS5 + 벡터 검색 통합)
"""

import logging
from typing import Any, Dict, List, Optional

from ..utils.config import Config
from .exact_search_engine_v2 import ExactSearchEngineV2
from .question_classifier import QuestionClassifier, QuestionType
from .result_merger import ResultMerger, ResultRanker
from .semantic_search_engine_v2 import SemanticSearchEngineV2

logger = logging.getLogger(__name__)


class HybridSearchEngineV2:
    """lawfirm_v2.db 기반 하이브리드 검색 엔진"""

    def __init__(self,
                 db_path: Optional[str] = None,
                 model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        """
        검색 엔진 초기화

        Args:
            db_path: lawfirm_v2.db 경로 (None이면 환경변수 DATABASE_PATH 사용)
            model_name: 임베딩 모델명
        """
        if db_path is None:
            config = Config()
            db_path = config.database_path
        self.db_path = db_path
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

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
                search_types = ["law", "precedent", "decision", "interpretation"]

            # 검색 결과 수집
            exact_results = {}
            semantic_results = []

            # 정확한 매칭 검색 (FTS5)
            if include_exact:
                exact_results = self._execute_exact_search(query, search_types)

            # 의미적 검색 (벡터)
            if include_semantic:
                semantic_results = self._execute_semantic_search(query, search_types)

            # 결과 통합
            merged_results = self.result_merger.merge_results(
                exact_results, semantic_results
            )

            # 결과 랭킹
            ranked_results = self.result_ranker.rank_results(merged_results)

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

    def _execute_semantic_search(self, query: str, search_types: List[str]) -> List[Dict[str, Any]]:
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
