# -*- coding: utf-8 -*-
"""
SQLRouter 강제 경로(직접 ?�출) ?�위 ?�스??

- ?�턴�?SQL ?�성 �??�전??검�?
- LIMIT ?�동 부???�한(<=100) 보정 ?�인
"""

import sys
from pathlib import Path

import pytest

# ?�로?�트 루트 경로 추�?
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _mk_router():
    from source.services.sql_router import SQLRouter
    return SQLRouter(db_path="data/lawfirm.db")


def test_article_lookup_sql_generation_and_safety():
    router = _mk_router()
    q = "민법 ??50�?조문 보여�?
    sql, rows = router.route_and_execute(q)
    # SQL???�성?�고 SELECT�??�작?�며 LIMIT ?�함
    assert sql is not None and isinstance(sql, str)
    assert sql.strip().lower().startswith("select")
    assert " from articles " in sql.lower()
    assert " limit " in sql.lower()
    # ?�전??검�??�과
    assert router._is_sql_safe(sql) is True
    # rows??리스??0건이?�도 리스??
    assert isinstance(rows, list)


def test_case_lookup_sql_generation_and_safety():
    router = _mk_router()
    q = "?�법원 2021??2345 ?�건 ?��?"
    sql, rows = router.route_and_execute(q)
    assert sql is not None and sql.lower().startswith("select")
    assert " from cases " in sql.lower()
    assert " limit " in sql.lower()
    assert router._is_sql_safe(sql) is True
    assert isinstance(rows, list)


def test_recent_years_count_query_adds_limit():
    router = _mk_router()
    q = "최근 3??민사 ?�해배상 ?�결 ??
    sql, _ = router.route_and_execute(q)
    assert sql is not None
    assert sql.lower().startswith("select")
    # count(*) 쿼리?�도 LIMIT가 ?�동 부?�되?�야 ??
    assert " limit " in sql.lower()
    assert router._is_sql_safe(sql) is True


def test_limit_cap_enforced():
    router = _mk_router()
    # ?��? ?�퍼 직접 ?�인 (100 초과 ??100?�로 보정)
    s = "SELECT law_name FROM articles WHERE law_name LIKE '%민법%' LIMIT 1000"
    capped = router._ensure_limit(s)
    assert " limit 100" in capped.lower()
