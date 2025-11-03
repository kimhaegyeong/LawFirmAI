# -*- coding: utf-8 -*-
"""
lawfirm_v2.db 전용 법률 데이터 연동 서비스
FTS5 키워드 검색 + 벡터 의미 검색 지원
"""

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Query routing patterns
ARTICLE_PATTERN = re.compile(r"제\s*\d+\s*조")
DATE_PATTERN = re.compile(r"\d{4}[.\-]\s*\d{1,2}[.\-]\s*\d{1,2}")
COURT_KEYWORDS = ["대법원", "고등법원", "지방법원", "가정법원", "행정법원"]


def route_query(query: str) -> str:
    """Query routing: 'text2sql' or 'vector'"""
    q = (query or "").strip()
    if not q:
        return "vector"

    # Strong textual cues → Text2SQL
    if ARTICLE_PATTERN.search(q):
        return "text2sql"
    if DATE_PATTERN.search(q):
        return "text2sql"
    if any(k in q for k in COURT_KEYWORDS):
        return "text2sql"
    if re.search(r"(사건|사건번호|doc_id|문서번호)", q):
        return "text2sql"

    return "vector"


class LegalDataConnectorV2:
    """lawfirm_v2.db 전용 법률 데이터베이스 연결 및 검색 서비스"""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            import os
            import sys
            # source 모듈 경로 추가
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from langgraph_core.utils.config import Config
            config = Config()
            db_path = config.database_path
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        if not Path(db_path).exists():
            self.logger.warning(f"Database {db_path} not found. Please initialize it first.")

    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _table_exists(self, table_name: str) -> bool:
        """
        테이블 존재 여부 확인
        
        Args:
            table_name: 확인할 테이블명
            
        Returns:
            테이블 존재 여부
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (table_name,))
            exists = cursor.fetchone() is not None
            conn.close()
            return exists
        except Exception as e:
            self.logger.debug(f"Error checking table existence for {table_name}: {e}")
            return False

    def _is_table_error(self, error: Exception) -> bool:
        """
        테이블 관련 오류인지 확인 (no such table 등)
        
        Args:
            error: 확인할 예외 객체
            
        Returns:
            테이블 관련 오류 여부
        """
        error_str = str(error).lower()
        return "no such table" in error_str or "table" in error_str

    def _analyze_query_plan(self, query: str, table_name: str) -> Optional[Dict[str, Any]]:
        """
        FTS5 쿼리 실행 계획 분석

        Args:
            query: 검색 쿼리
            table_name: FTS5 테이블명

        Returns:
            실행 계획 정보 딕셔너리 또는 None
        """
        try:
            safe_query = self._sanitize_fts5_query(query)
            if not safe_query:
                return None

            conn = self._get_connection()
            cursor = conn.cursor()

            # EXPLAIN QUERY PLAN 실행
            explain_query = f"""
                EXPLAIN QUERY PLAN
                SELECT rowid, bm25({table_name}) as rank_score
                FROM {table_name}
                WHERE {table_name} MATCH ?
                ORDER BY rank_score
                LIMIT 10
            """
            cursor.execute(explain_query, (safe_query,))
            plan_rows = cursor.fetchall()

            conn.close()

            # 실행 계획 분석
            plan_info = {
                "uses_index": any("FTS" in str(row) or "MATCH" in str(row) for row in plan_rows),
                "scan_type": "FTS" if any("FTS" in str(row) for row in plan_rows) else "UNKNOWN",
                "plan_detail": [str(row) for row in plan_rows]
            }

            self.logger.debug(f"Query plan for '{query[:30]}': {plan_info}")
            return plan_info

        except Exception as e:
            self.logger.debug(f"Error analyzing query plan: {e}")
            return None

    def _optimize_fts5_query(self, query: str) -> str:
        """
        FTS5 쿼리 최적화 (토크나이저 고려, 실행 계획 분석)

        Args:
            query: 원본 쿼리

        Returns:
            최적화된 쿼리
        """
        # 현재는 unicode61 토크나이저 사용 (기본값)
        # 포터 토크나이저는 영어 어간 추출에 유용하지만 한글에는 적합하지 않음
        # 따라서 unicode61 유지

        # 최적화: 불필요한 공백 제거, 중복 단어 제거
        words = query.split()
        unique_words = list(dict.fromkeys(words))  # 순서 유지하며 중복 제거

        # FTS5는 최대 3개 키워드 권장 (이미 _sanitize_fts5_query에서 처리)
        optimized = " ".join(unique_words[:3])

        return optimized

    def _sanitize_fts5_query(self, query: str) -> str:
        """
        FTS5 쿼리를 안전하게 변환
        - 특수 문자가 있으면 이스케이프 처리
        - 빈 쿼리는 빈 문자열 반환
        - 단순 키워드 검색에 최적화
        - FTS5는 기본적으로 공백으로 구분된 단어를 AND 조건으로 처리
        """
        if not query or not query.strip():
            return ""

        query = query.strip()

        # FTS5 특수 문자 제거 및 이스케이프
        # FTS5에서 문제가 되는 문자: ", :, (, ), ?, *, - 등
        # 하지만 단순 키워드 검색을 위해서는 특수 문자를 제거하거나 이스케이프

        # 특수 문자 목록 (FTS5 문법에서 사용되는 문자)
        special_chars = ['"', ':', '(', ')', '?', '*', '-', 'OR', 'AND', 'NOT']

        # 단어 개수 확인
        words = query.split()

        # 특수 문자가 포함되어 있는지 확인
        has_special = any(char in query for char in special_chars)

        if has_special:
            # 특수 문자가 있으면 제거하고 단어만 추출
            import re
            # 한글, 영문, 숫자만 추출
            clean_words = re.findall(r'[가-힣a-zA-Z0-9]+', query)
            if not clean_words:
                # 단어가 없으면 빈 문자열 반환
                return ""
            # 최대 5개 단어만 사용
            clean_words = clean_words[:5]
            # OR 조건으로 연결 (검색 범위 확장)
            sanitized = " OR ".join(clean_words)
            # SQL injection 방지: 작은따옴표 이스케이프
            sanitized = sanitized.replace("'", "''")
            return sanitized
        elif len(words) > 3:
            # 키워드가 3개 이상이면 상위 3개만 사용 (AND 조건, 기본 FTS5 동작)
            # 너무 많은 키워드는 검색 범위를 좁힘
            result = " ".join(words[:3])
            # SQL injection 방지: 작은따옴표 이스케이프
            result = result.replace("'", "''")
            return result
        else:
            # 단순 키워드 검색 (AND 조건, 기본 FTS5 동작)
            # SQL injection 방지: 작은따옴표 이스케이프
            return query.replace("'", "''")

    def search_statutes_fts(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """FTS5 키워드 검색: 법령 조문 (최적화됨)"""
        try:
            # FTS5 쿼리 최적화 및 안전화
            optimized_query = self._optimize_fts5_query(query)
            safe_query = self._sanitize_fts5_query(optimized_query)
            if not safe_query:
                self.logger.warning(f"Empty or invalid FTS5 query: '{query}'")
                return []

            # 실행 계획 분석 (디버그 모드)
            if self.logger.isEnabledFor(logging.DEBUG):
                plan_info = self._analyze_query_plan(query, "statute_articles_fts")
                if plan_info:
                    self.logger.debug(f"Query plan analysis: {plan_info}")

            conn = self._get_connection()
            cursor = conn.cursor()

            # FTS5 검색 (BM25 랭킹)
            cursor.execute("""
                SELECT
                    sa.id,
                    sa.statute_id,
                    sa.article_no,
                    sa.clause_no,
                    sa.item_no,
                    sa.heading,
                    sa.text,
                    s.name as statute_name,
                    s.abbrv as statute_abbrv,
                    s.statute_type,
                    s.category,
                    bm25(statute_articles_fts) as rank_score
                FROM statute_articles_fts
                JOIN statute_articles sa ON statute_articles_fts.rowid = sa.id
                JOIN statutes s ON sa.statute_id = s.id
                WHERE statute_articles_fts MATCH ?
                ORDER BY rank_score
                LIMIT ?
            """, (safe_query, limit))

            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": f"statute_article_{row['id']}",
                    "type": "statute",
                    "content": row['text'],
                    "source": row['statute_name'],
                    "metadata": {
                        "statute_id": row['statute_id'],
                        "article_no": row['article_no'],
                        "clause_no": row['clause_no'],
                        "item_no": row['item_no'],
                        "heading": row['heading'],
                        "statute_abbrv": row['statute_abbrv'],
                        "statute_type": row['statute_type'],
                        "category": row['category'],
                    },
                    "relevance_score": max(0.0, -row['rank_score'] / 100.0) if row['rank_score'] else 0.5,
                    "search_type": "keyword"
                })

            conn.close()
            self.logger.info(f"FTS search found {len(results)} statute articles for query: {query}")
            return results

        except Exception as e:
            # 테이블이 없는 경우는 warning으로 처리 (정상적인 초기 상태일 수 있음)
            if self._is_table_error(e):
                self.logger.warning(f"FTS table 'statute_articles_fts' not found: {e}. "
                                  f"Database may need migration. Returning empty results.")
            else:
                self.logger.error(f"Error in FTS statute search: {e}")
            return []

    def search_cases_fts(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """FTS5 키워드 검색: 판례 (최적화됨)"""
        try:
            # FTS5 쿼리 최적화 및 안전화
            optimized_query = self._optimize_fts5_query(query)
            safe_query = self._sanitize_fts5_query(optimized_query)
            if not safe_query:
                self.logger.warning(f"Empty or invalid FTS5 query: '{query}'")
                return []

            # 실행 계획 분석 (디버그 모드)
            if self.logger.isEnabledFor(logging.DEBUG):
                plan_info = self._analyze_query_plan(query, "case_paragraphs_fts")
                if plan_info:
                    self.logger.debug(f"Query plan analysis: {plan_info}")

            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(f"""
                SELECT
                    cp.id,
                    cp.case_id,
                    cp.para_index,
                    cp.text,
                    c.doc_id,
                    c.court,
                    c.case_type,
                    c.casenames,
                    c.announce_date,
                    bm25(case_paragraphs_fts) as rank_score
                FROM case_paragraphs_fts
                JOIN case_paragraphs cp ON case_paragraphs_fts.rowid = cp.id
                JOIN cases c ON cp.case_id = c.id
                WHERE case_paragraphs_fts MATCH '{safe_query}'
                ORDER BY rank_score
                LIMIT {limit}
            """)

            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": f"case_para_{row['id']}",
                    "type": "case",
                    "content": row['text'],
                    "source": f"{row['court']} {row['doc_id']}",
                    "metadata": {
                        "case_id": row['case_id'],
                        "doc_id": row['doc_id'],
                        "court": row['court'],
                        "case_type": row['case_type'],
                        "casenames": row['casenames'],
                        "announce_date": row['announce_date'],
                        "para_index": row['para_index'],
                    },
                    "relevance_score": max(0.0, -row['rank_score'] / 100.0) if row['rank_score'] else 0.5,
                    "search_type": "keyword"
                })

            conn.close()
            self.logger.info(f"FTS search found {len(results)} case paragraphs for query: {query}")
            return results

        except Exception as e:
            # 테이블이 없는 경우는 warning으로 처리 (정상적인 초기 상태일 수 있음)
            if self._is_table_error(e):
                self.logger.warning(f"FTS table 'case_paragraphs_fts' not found: {e}. "
                                  f"Database may need migration. Returning empty results.")
            else:
                self.logger.error(f"Error in FTS case search: {e}")
            return []

    def search_decisions_fts(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """FTS5 키워드 검색: 심결례 (최적화됨)"""
        try:
            # FTS5 쿼리 최적화 및 안전화
            optimized_query = self._optimize_fts5_query(query)
            safe_query = self._sanitize_fts5_query(optimized_query)
            if not safe_query:
                self.logger.warning(f"Empty or invalid FTS5 query: '{query}'")
                return []

            # 실행 계획 분석 (디버그 모드)
            if self.logger.isEnabledFor(logging.DEBUG):
                plan_info = self._analyze_query_plan(query, "decision_paragraphs_fts")
                if plan_info:
                    self.logger.debug(f"Query plan analysis: {plan_info}")

            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(f"""
                SELECT
                    dp.id,
                    dp.decision_id,
                    dp.para_index,
                    dp.text,
                    d.org,
                    d.doc_id,
                    d.decision_date,
                    d.result,
                    bm25(decision_paragraphs_fts) as rank_score
                FROM decision_paragraphs_fts
                JOIN decision_paragraphs dp ON decision_paragraphs_fts.rowid = dp.id
                JOIN decisions d ON dp.decision_id = d.id
                WHERE decision_paragraphs_fts MATCH '{safe_query}'
                ORDER BY rank_score
                LIMIT {limit}
            """)

            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": f"decision_para_{row['id']}",
                    "type": "decision",
                    "content": row['text'],
                    "source": f"{row['org']} {row['doc_id']}",
                    "metadata": {
                        "decision_id": row['decision_id'],
                        "org": row['org'],
                        "doc_id": row['doc_id'],
                        "decision_date": row['decision_date'],
                        "result": row['result'],
                        "para_index": row['para_index'],
                    },
                    "relevance_score": max(0.0, -row['rank_score'] / 100.0) if row['rank_score'] else 0.5,
                    "search_type": "keyword"
                })

            conn.close()
            self.logger.info(f"FTS search found {len(results)} decision paragraphs for query: {query}")
            return results

        except Exception as e:
            # 테이블이 없는 경우는 warning으로 처리 (정상적인 초기 상태일 수 있음)
            if self._is_table_error(e):
                self.logger.warning(f"FTS table 'decision_paragraphs_fts' not found: {e}. "
                                  f"Database may need migration. Returning empty results.")
            else:
                self.logger.error(f"Error in FTS decision search: {e}")
            return []

    def search_interpretations_fts(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """FTS5 키워드 검색: 유권해석 (최적화됨)"""
        try:
            # FTS5 쿼리 최적화 및 안전화
            optimized_query = self._optimize_fts5_query(query)
            safe_query = self._sanitize_fts5_query(optimized_query)
            if not safe_query:
                self.logger.warning(f"Empty or invalid FTS5 query: '{query}'")
                return []

            # 실행 계획 분석 (디버그 모드)
            if self.logger.isEnabledFor(logging.DEBUG):
                plan_info = self._analyze_query_plan(query, "interpretation_paragraphs_fts")
                if plan_info:
                    self.logger.debug(f"Query plan analysis: {plan_info}")

            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(f"""
                SELECT
                    ip.id,
                    ip.interpretation_id,
                    ip.para_index,
                    ip.text,
                    i.org,
                    i.doc_id,
                    i.title,
                    i.response_date,
                    bm25(interpretation_paragraphs_fts) as rank_score
                FROM interpretation_paragraphs_fts
                JOIN interpretation_paragraphs ip ON interpretation_paragraphs_fts.rowid = ip.id
                JOIN interpretations i ON ip.interpretation_id = i.id
                WHERE interpretation_paragraphs_fts MATCH '{safe_query}'
                ORDER BY rank_score
                LIMIT {limit}
            """)

            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": f"interpretation_para_{row['id']}",
                    "type": "interpretation",
                    "content": row['text'],
                    "source": f"{row['org']} {row['title']}",
                    "metadata": {
                        "interpretation_id": row['interpretation_id'],
                        "org": row['org'],
                        "doc_id": row['doc_id'],
                        "title": row['title'],
                        "response_date": row['response_date'],
                        "para_index": row['para_index'],
                    },
                    "relevance_score": max(0.0, -row['rank_score'] / 100.0) if row['rank_score'] else 0.5,
                    "search_type": "keyword"
                })

            conn.close()
            self.logger.info(f"FTS search found {len(results)} interpretation paragraphs for query: {query}")
            return results

        except Exception as e:
            # 테이블이 없는 경우는 warning으로 처리 (정상적인 초기 상태일 수 있음)
            if self._is_table_error(e):
                self.logger.warning(f"FTS table 'interpretation_paragraphs_fts' not found: {e}. "
                                  f"Database may need migration. Returning empty results.")
            else:
                self.logger.error(f"Error in FTS interpretation search: {e}")
            return []

    def search_documents(self, query: str, category: Optional[str] = None, limit: int = 10, force_fts: bool = False) -> List[Dict[str, Any]]:
        """
        통합 검색: 라우팅에 따라 FTS5 또는 벡터 검색

        Args:
            query: 검색 쿼리
            category: 카테고리 (하위 호환성을 위해 유지, 현재 사용하지 않음)
            limit: 최대 결과 수
            force_fts: True이면 라우팅과 관계없이 강제로 FTS5 검색 수행

        Returns:
            검색 결과 리스트
        """
        # force_fts가 True이면 라우팅 무시하고 강제로 FTS5 검색
        if force_fts:
            results = []
            results.extend(self.search_statutes_fts(query, limit=limit))
            results.extend(self.search_cases_fts(query, limit=limit))
            results.extend(self.search_decisions_fts(query, limit=limit))
            results.extend(self.search_interpretations_fts(query, limit=limit))

            # relevance_score 기준 정렬
            results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
            return results[:limit]

        # 기존 라우팅 로직
        route = route_query(query)

        if route == "text2sql":
            # FTS5 키워드 검색
            results = []
            results.extend(self.search_statutes_fts(query, limit=limit))
            results.extend(self.search_cases_fts(query, limit=limit))
            results.extend(self.search_decisions_fts(query, limit=limit))
            results.extend(self.search_interpretations_fts(query, limit=limit))

            # relevance_score 기준 정렬
            results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
            return results[:limit]
        else:
            # 벡터 검색은 별도 SemanticSearchEngineV2에서 처리
            # 여기서는 빈 리스트 반환하거나 기본 FTS 결과 반환
            self.logger.info(f"Vector search requested for: {query}, delegating to SemanticSearchEngineV2")
            return []

    def get_all_categories(self) -> List[str]:
        """도메인 목록 반환 (하위 호환성)"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT name FROM domains ORDER BY name")
            categories = [row[0] for row in cursor.fetchall()]
            conn.close()
            return categories
        except Exception as e:
            self.logger.error(f"Error getting categories: {e}")
            return []
