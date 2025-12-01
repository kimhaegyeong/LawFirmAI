# -*- coding: utf-8 -*-
"""
SQL Router for hybrid Text-to-SQL + RAG workflow

- Decides if a query is suitable for Text-to-SQL
- Generates safe SQL (best-effort, simple patterns)
- Validates SQL via allow-list (SELECT-only, tables/columns)
- Executes SQL using DatabaseManager
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from core.data.database import DatabaseManager
except Exception:  # pragma: no cover
    # Fallback import path if running from different root
    from data.database import DatabaseManager  # type: ignore


logger = get_logger(__name__)


class SQLRouter:
    """Hybrid SQL Router with conservative safety checks."""

    def __init__(self, db_path: str = "data/lawfirm_v2.db") -> None:
        """DEPRECATED: sql_router는 lawfirm.db의 DatabaseManager를 사용합니다.
        lawfirm_v2.db를 사용하려면 LegalDataConnectorV2를 사용하세요."""
        import warnings
        warnings.warn(
            "SQLRouter using DatabaseManager (lawfirm.db) is deprecated. "
            "Use LegalDataConnectorV2 (lawfirm_v2.db) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # DatabaseManager는 deprecated이므로 사용 불가
        raise RuntimeError(
            "SQLRouter is deprecated because it uses DatabaseManager (lawfirm.db). "
            "Use LegalDataConnectorV2 (lawfirm_v2.db) instead."
        )
        self.db = DatabaseManager(db_path=db_path)
        # Allow-list tables/columns (extend as schema evolves)
        self.allowed_tables = {
            "laws": {"id", "law_name"},
            "articles": {"id", "law_name", "article_number", "content"},
            "cases": {"id", "case_number", "court", "decision_date", "summary"},
            "case_citations": {"id", "from_case_id", "to_case_id"},
            "amendments": {"id", "law_name", "effective_date", "description"},
        }

    def get_schema_overview(self) -> str:
        """Return a compact schema description for Text-to-SQL prompting (PostgreSQL)."""
        return (
            "PostgreSQL 테이블/뷰 설명:\n"
            "- domains(id, name) — 법률 도메인\n"
            "- sources(id, source_type, path, hash, created_at) — 소스 추적\n"
            "- statutes(id, domain_id, name, abbrv, statute_type, proclamation_date, effective_date, category) — 법률 정보\n"
            "- statute_articles(id, statute_id, article_no, clause_no, item_no, heading, text, version_effective_date) — 조문 본문\n"
            "- text_chunks(id, source_type, source_id, chunk_index, text, meta) — 텍스트 청크\n"
            "- embeddings(id, chunk_id, model, dim, vector) — 벡터 임베딩\n"
            "- embedding_versions(id, version_name, chunking_strategy, model_name, is_active) — 임베딩 버전\n"
            "- retrieval_cache(query_hash, topk_ids, created_at) — 검색 캐시\n"
            "제약: SELECT만 허용, LIMIT 필수. 조문번호는 정확히 매칭하세요. JOIN을 사용하여 관련 테이블을 연결하세요."
        )

    def is_sql_suitable(self, query: str) -> bool:
        if not query:
            return False
        q = query.strip()
        # Heuristic: presence of law article markers / case metadata terms / numeric filters
        patterns = [
            r"민법\s*제?\d+\s*조", r"형법\s*제?\d+\s*조", r"제\s*\d+\s*조",
            r"사건번호|선고일|대법원|고등법원|지방법원",
            r"최근\s*\d+\s*년|\d{4}년|\d{4}-\d{2}-\d{2}",
            r"건수|갯수|개수|통계|집계|정렬|필터",
        ]
        return any(re.search(p, q) for p in patterns)

    def route_and_execute(self, query: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Generate, validate and execute SQL. Returns (sql, rows)."""
        try:
            sql, params = self._generate_sql(query)
            if not sql:
                return None, []
            # Enforce LIMIT for safety if missing
            sql = self._ensure_limit(sql)
            if not self._is_sql_safe(sql):
                logger.warning(f"Rejected unsafe SQL: {sql}")
                return None, []
            rows = self._execute(sql, params)
            if rows:
                return sql, rows
            # Fallback: content LIKE search over articles
            try:
                fallback_sql = (
                    "SELECT law_name, article_number, content FROM articles WHERE content LIKE ?"
                )
                fallback_sql = self._ensure_limit(fallback_sql)
                fallback_rows = self._execute(fallback_sql, (f"%{query}%",))
                if fallback_rows:
                    return fallback_sql, fallback_rows
            except Exception:
                pass
            # Fallback: summary LIKE search over cases
            try:
                fallback_sql2 = (
                    "SELECT case_number, court, decision_date, summary FROM cases WHERE summary LIKE ?"
                )
                fallback_sql2 = self._ensure_limit(fallback_sql2)
                fallback_rows2 = self._execute(fallback_sql2, (f"%{query}%",))
                if fallback_rows2:
                    return fallback_sql2, fallback_rows2
            except Exception:
                pass
            return sql, []
        except Exception as e:
            logger.warning(f"SQL routing failed: {e}")
            return None, []

    # --- internals ---

    def _generate_sql(self, query: str) -> Tuple[Optional[str], Tuple[Any, ...]]:
        q = query.strip()
        # Simple article lookup: "민법 제750조" → articles
        m = re.search(r"([가-힣A-Za-z]+)\s*제\s*(\d+)\s*조", q)
        if m:
            law_name = m.group(1)
            article = m.group(2)
            sql = (
                "SELECT law_name, article_number, content "
                "FROM articles WHERE law_name LIKE ? AND article_number = ? LIMIT 5"
            )
            return sql, (f"%{law_name}%", article)

        # Case number lookup: "대법원 2021다12345" → cases
        m = re.search(r"(\d{4}[가-힣]{1,2}\d{1,6})", q)
        if m:
            case_no = m.group(1)
            sql = (
                "SELECT case_number, court, decision_date, summary "
                "FROM cases WHERE case_number = ? LIMIT 5"
            )
            return sql, (case_no,)

        # Recent N years count of a term (very naive): "최근 3년 불법행위 판결 수"
        m = re.search(r"최근\s*(\d+)\s*년.*(판결|사건).*(수|건수|개수)", q)
        if m:
            years = int(m.group(1))
            sql = (
                "SELECT COUNT(*) AS cnt FROM cases "
                "WHERE decision_date >= date('now', ?)"
            )
            return sql, (f"-{years} years",)

        return None, tuple()

    def _is_sql_safe(self, sql: str) -> bool:
        s = sql.strip()
        sl = s.lower()
        if not sl.startswith("select"):
            return False
        # Disallow dangerous tokens
        forbidden = [";", "--", "/*", "*/", " drop ", " delete ", " update ", " insert ", " alter ", " create ", " pragma ", " attach ", " detach ", " vacuum "]
        if any(tok in sl for tok in forbidden):
            return False

        # Extract table name from FROM clause
        m = re.search(r"from\s+([a-zA-Z_][a-zA-Z0-9_]*)", sl)
        if not m:
            return False
        table = m.group(1)
        if table not in self.allowed_tables:
            return False

        # Validate selected columns unless it's only '*'
        m_sel = re.search(r"select\s+(.*?)\s+from", sl)
        if not m_sel:
            return False
        select_part = m_sel.group(1).strip()
        if select_part != "*":
            cols = [c.strip() for c in select_part.split(",")]
            for col in cols:
                # Allow aggregates/functions e.g., count(*) as cnt
                if re.match(r"[a-zA-Z_]+\s*\(.*\)", col):
                    continue
                # strip alias
                col_name = re.split(r"\s+as\s+", col)[0].strip()
                # strip table prefix
                if "." in col_name:
                    col_name = col_name.split(".")[-1]
                if col_name and col_name != "*" and col_name not in self.allowed_tables[table]:
                    return False

        # Validate WHERE identifiers
        m_where = re.search(r"where\s+(.*?)(group\s+by|order\s+by|limit|$)", sl)
        if m_where:
            where_expr = m_where.group(1)
            idents = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(=|<|>|like|in|>=|<=)", where_expr)
            for ident, _ in idents:
                if ident not in self.allowed_tables[table]:
                    return False

        # LIMIT must exist and be <= 100
        m_lim = re.search(r"limit\s+(\d+)", sl)
        if not m_lim:
            return False
        try:
            lim_val = int(m_lim.group(1))
            if lim_val > 100:
                return False
        except Exception:
            return False
        return True

    def _ensure_limit(self, sql: str, default_limit: int = 20) -> str:
        s = sql.strip()
        lowered = s.lower()
        m_lim = re.search(r"limit\s+(\d+)", lowered)
        if m_lim:
            try:
                val = int(m_lim.group(1))
                if val > 100:
                    return re.sub(r"limit\s+\d+", "LIMIT 100", s, flags=re.IGNORECASE)
                return s
            except Exception:
                return re.sub(r"limit\s+[^\s]+", f"LIMIT {default_limit}", s, flags=re.IGNORECASE)
        return f"{s} LIMIT {default_limit}"

    def _execute(self, sql: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
        try:
            return self.db.execute_query(sql, params)
        except Exception as e:
            logger.warning(f"SQL execution failed: {e}")
            return []
