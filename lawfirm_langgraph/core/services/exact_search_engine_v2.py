# -*- coding: utf-8 -*-
"""
Exact Search Engine V2
lawfirm_v2.db의 FTS5 기반 키워드 검색 엔진
"""

import logging
from typing import Any, Dict, List, Optional

from ..utils.config import Config
from ..search.connectors.legal_data_connector import LegalDataConnectorV2

logger = logging.getLogger(__name__)


class ExactSearchEngineV2:
    """lawfirm_v2.db 기반 정확한 매칭 검색 엔진 (FTS5)"""

    def __init__(self, db_path: Optional[str] = None):
        """
        검색 엔진 초기화

        Args:
            db_path: lawfirm_v2.db 경로 (None이면 환경변수 DATABASE_PATH 사용)
        """
        if db_path is None:
            config = Config()
            db_path = config.database_path
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.connector = LegalDataConnectorV2(db_path)
        self.logger.info("ExactSearchEngineV2 initialized")

    def search(self,
               query: str,
               search_types: Optional[List[str]] = None,
               max_results: int = 50) -> Dict[str, Any]:
        """
        FTS5 기반 키워드 검색

        Args:
            query: 검색 쿼리
            search_types: 검색할 문서 타입 목록 (['law', 'precedent', 'decision', 'interpretation'])
            max_results: 최대 결과 수

        Returns:
            {문서타입: [결과리스트]} 형태의 딕셔너리
        """
        try:
            results_by_type = {
                "law": [],
                "precedent": [],
                "decision": [],
                "interpretation": []
            }

            # search_types가 지정되지 않으면 전체 검색
            if search_types is None:
                search_types = ["law", "precedent", "decision", "interpretation"]

            # 각 타입별로 FTS 검색 수행
            if "law" in search_types:
                statute_results = self.connector.search_statutes_fts(query, limit=max_results)
                results_by_type["law"] = statute_results

            if "precedent" in search_types:
                case_results = self.connector.search_cases_fts(query, limit=max_results)
                results_by_type["precedent"] = case_results

            if "decision" in search_types:
                decision_results = self.connector.search_decisions_fts(query, limit=max_results)
                results_by_type["decision"] = decision_results

            if "interpretation" in search_types:
                interp_results = self.connector.search_interpretations_fts(query, limit=max_results)
                results_by_type["interpretation"] = interp_results

            # 전체 결과 수 계산
            total_results = sum(len(v) for v in results_by_type.values())

            self.logger.info(f"Exact search found {total_results} results across {len(search_types)} types")

            return results_by_type

        except Exception as e:
            self.logger.error(f"Error in exact search: {e}")
            return {
                "law": [],
                "precedent": [],
                "decision": [],
                "interpretation": []
            }
