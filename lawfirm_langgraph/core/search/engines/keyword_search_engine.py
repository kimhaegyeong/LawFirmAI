# -*- coding: utf-8 -*-
"""
Keyword Search Engine
키워드 기반 검색 엔진
"""

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExactSearchResult:
    """정확한 검색 결과"""
    text: str
    score: float
    match_type: str
    metadata: Dict[str, Any]


class KeywordSearchEngine:
    """
    키워드 기반 검색 엔진

    DEPRECATED: 이 클래스는 lawfirm.db를 사용합니다.
    새로운 프로젝트는 ExactSearchEngineV2 (lawfirm_v2.db의 FTS5 사용)를 사용하세요.
    """

    def __init__(self, db_path: str = "data/lawfirm.db"):
        """검색 엔진 초기화"""
        import warnings
        warnings.warn(
            "KeywordSearchEngine is deprecated. Use ExactSearchEngineV2 instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(db_path)
        self.logger.warning("⚠️ KeywordSearchEngine is deprecated. Migrate to ExactSearchEngineV2.")
        self.logger.info("KeywordSearchEngine initialized")

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        쿼리를 파싱하여 검색에 필요한 정보를 추출합니다.
        Args:
            query: 검색 쿼리
        Returns:
            Dict[str, Any]: 파싱된 쿼리 정보
        """
        return {
            "original_query": query,
            "raw_query": query,  # hybrid_search_engine에서 사용하는 키 추가
            "keywords": query.split(),
            "search_type": "exact",
            "law_name": None,
            "article_number": None,
            "case_number": None,
            "court_name": None
        }

    def search(self, query: str, documents: List[Dict[str, Any]], top_k: int = 10) -> List[ExactSearchResult]:
        """
        정확한 검색 수행

        Args:
            query: 검색 쿼리
            documents: 검색할 문서 목록
            top_k: 반환할 결과 수

        Returns:
            List[ExactSearchResult]: 검색 결과
        """
        results = []

        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})

            # 정확한 매칭 점수 계산
            score = self._calculate_exact_score(query, text)

            if score > 0:
                match_type = self._determine_match_type(query, text)
                result = ExactSearchResult(
                    text=text,
                    score=score,
                    match_type=match_type,
                    metadata=metadata
                )
                results.append(result)

        # 점수순 정렬 및 상위 k개 반환
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def search_laws(self, query: str, law_name: str = None, article_number: str = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """법령 검색"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # 실제 테이블 구조에 맞는 쿼리
                sql_query = """
                SELECT
                    id,
                    law_name,
                    article_number,
                    content,
                    law_type,
                    effective_date
                FROM assembly_laws
                WHERE 1=1
                """
                params = []

                # 검색 조건 추가
                if query:
                    sql_query += " AND (law_name LIKE ? OR content LIKE ?)"
                    search_term = f"%{query}%"
                    params.extend([search_term, search_term])

                if law_name:
                    sql_query += " AND law_name LIKE ?"
                    params.append(f"%{law_name}%")

                if article_number:
                    sql_query += " AND article_number LIKE ?"
                    params.append(f"%{article_number}%")

                sql_query += " ORDER BY law_name, article_number LIMIT ?"
                params.append(top_k)

                cursor.execute(sql_query, params)
                rows = cursor.fetchall()

                results = []
                for row in rows:
                    result = {
                        'law_id': row['id'],
                        'law_name': row['law_name'],
                        'article_number': row['article_number'],
                        'content': row['content'],
                        'law_type': row['law_type'],
                        'effective_date': row['effective_date'],
                        'search_type': 'exact_law'
                    }
                    results.append(result)

                self.logger.info(f"Found {len(results)} law results for query: {query}")
                return results

        except Exception as e:
            self.logger.error(f"Error searching laws: {e}")
            return []

    def search_precedents(self, query: str, case_number: str = None, court_name: str = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """판례 검색"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # 실제 테이블 구조에 맞는 쿼리
                sql_query = """
                SELECT
                    case_id,
                    case_name,
                    case_number,
                    court,
                    decision_date,
                    category,
                    field,
                    full_text
                FROM precedent_cases
                WHERE 1=1
                """
                params = []

                # 검색 조건 추가
                if query:
                    sql_query += " AND (case_name LIKE ? OR full_text LIKE ?)"
                    search_term = f"%{query}%"
                    params.extend([search_term, search_term])

                if case_number:
                    sql_query += " AND case_number LIKE ?"
                    params.append(f"%{case_number}%")

                if court_name:
                    sql_query += " AND court LIKE ?"
                    params.append(f"%{court_name}%")

                sql_query += " ORDER BY decision_date DESC LIMIT ?"
                params.append(top_k)

                cursor.execute(sql_query, params)
                rows = cursor.fetchall()

                results = []
                for row in rows:
                    result = {
                        'case_id': row['case_id'],
                        'case_name': row['case_name'],
                        'case_number': row['case_number'],
                        'court': row['court'],
                        'decision_date': row['decision_date'],
                        'category': row['category'],
                        'field': row['field'],
                        'full_text': row['full_text'],
                        'search_type': 'exact_precedent'
                    }
                    results.append(result)

                self.logger.info(f"Found {len(results)} precedent results for query: {query}")
                return results

        except Exception as e:
            self.logger.error(f"Error searching precedents: {e}")
            return []

    def search_constitutional_decisions(self, query: str, case_number: str = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """헌법재판소 결정 검색"""
        # 임시 구현 - 실제로는 데이터베이스에서 검색
        self.logger.info(f"Searching constitutional decisions for query: {query}, case_number: {case_number}")
        return []

    def search_assembly_laws(self, query: str, law_name: str = None, article_number: str = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """국회 법률 검색"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # 실제 테이블 구조에 맞는 쿼리
                sql_query = """
                SELECT
                    id,
                    law_id,
                    law_name,
                    law_type,
                    category,
                    promulgation_date,
                    enforcement_date,
                    full_text,
                    summary
                FROM assembly_laws
                WHERE 1=1
                """
                params = []

                # 검색 조건 추가
                if query:
                    sql_query += " AND (law_name LIKE ? OR full_text LIKE ? OR summary LIKE ?)"
                    search_term = f"%{query}%"
                    params.extend([search_term, search_term, search_term])

                if law_name:
                    sql_query += " AND law_name LIKE ?"
                    params.append(f"%{law_name}%")

                if article_number:
                    sql_query += " AND law_id LIKE ?"
                    params.append(f"%{article_number}%")

                sql_query += " ORDER BY promulgation_date DESC LIMIT ?"
                params.append(top_k)

                cursor.execute(sql_query, params)
                rows = cursor.fetchall()

                results = []
                for row in rows:
                    result = {
                        'law_id': row['id'],
                        'law_name': row['law_name'],
                        'law_type': row['law_type'],
                        'category': row['category'],
                        'promulgation_date': row['promulgation_date'],
                        'enforcement_date': row['enforcement_date'],
                        'full_text': row['full_text'],
                        'summary': row['summary'],
                        'search_type': 'exact_assembly_law'
                    }
                    results.append(result)

                self.logger.info(f"Found {len(results)} assembly law results for query: {query}")
                return results

        except Exception as e:
            self.logger.error(f"Error searching assembly laws: {e}")
            return []

    def _calculate_exact_score(self, query: str, text: str) -> float:
        """
        정확한 매칭 점수 계산

        Args:
            query: 검색 쿼리
            text: 검색할 텍스트

        Returns:
            float: 매칭 점수 (0.0-1.0)
        """
        query_lower = query.lower()
        text_lower = text.lower()

        # 완전 일치
        if query_lower == text_lower:
            return 1.0

        # 부분 일치
        if query_lower in text_lower:
            # 일치 비율 계산
            match_ratio = len(query_lower) / len(text_lower)
            return min(0.9, match_ratio)

        # 단어별 일치
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())

        if query_words.issubset(text_words):
            word_match_ratio = len(query_words) / len(text_words)
            return min(0.8, word_match_ratio)

        return 0.0

    def _determine_match_type(self, query: str, text: str) -> str:
        """
        매칭 타입 결정

        Args:
            query: 검색 쿼리
            text: 검색할 텍스트

        Returns:
            str: 매칭 타입
        """
        query_lower = query.lower()
        text_lower = text.lower()

        if query_lower == text_lower:
            return "exact"
        elif query_lower in text_lower:
            return "partial"
        else:
            return "word_match"


# 기본 인스턴스 생성
def create_keyword_search_engine(db_path: str = "data/lawfirm.db") -> KeywordSearchEngine:
    """기본 키워드 검색 엔진 생성"""
    return KeywordSearchEngine(db_path)


if __name__ == "__main__":
    # 테스트 코드
    engine = create_keyword_search_engine()

    # 샘플 문서
    documents = [
        {"text": "민법 제543조 계약의 해지", "metadata": {"category": "civil"}},
        {"text": "형법 제250조 살인", "metadata": {"category": "criminal"}},
        {"text": "가족법 이혼 절차", "metadata": {"category": "family"}}
    ]

    # 검색 테스트
    results = engine.search("계약 해지", documents)
    print(f"Search results: {len(results)}")
    for result in results:
        print(f"  Score: {result.score:.3f}, Type: {result.match_type}")
        print(f"  Text: {result.text}")
