# -*- coding: utf-8 -*-
"""
SQLRouter 강제 경로(직접 호출) 단위 테스트

- 패턴별 SQL 생성 및 안전성 검증
- LIMIT 자동 부여/상한(<=100) 보정 확인
"""

import sys
from pathlib import Path

import pytest

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _mk_router():
    from core.services.search.sql_router import SQLRouter
    return SQLRouter(db_path="data/lawfirm.db")


def test_article_lookup_sql_generation_and_safety():
    router = _mk_router()
    q = "민법 제750조 조문 보여줘"
    sql, rows = router.route_and_execute(q)
    # SQL이 생성되고 SELECT로 시작하며 LIMIT 포함
    assert sql is not None and isinstance(sql, str)
    assert sql.strip().lower().startswith("select")
    assert " from articles " in sql.lower()
    assert " limit " in sql.lower()
    # 안전성 검증 통과
    assert router._is_sql_safe(sql) is True
    # rows는 리스트(0건이어도 리스트)
    assert isinstance(rows, list)


def test_case_lookup_sql_generation_and_safety():
    router = _mk_router()
    q = "대법원 2021다12345 사건 요지"
    sql, rows = router.route_and_execute(q)
    assert sql is not None and sql.lower().startswith("select")
    assert " from cases " in sql.lower()
    assert " limit " in sql.lower()
    assert router._is_sql_safe(sql) is True
    assert isinstance(rows, list)


def test_recent_years_count_query_adds_limit():
    router = _mk_router()
    q = "최근 3년 민사 손해배상 판결 수"
    sql, _ = router.route_and_execute(q)
    assert sql is not None
    assert sql.lower().startswith("select")
    # count(*) 쿼리에도 LIMIT가 자동 부여되어야 함
    assert " limit " in sql.lower()
    assert router._is_sql_safe(sql) is True


def test_limit_cap_enforced():
    router = _mk_router()
    # 내부 헬퍼 직접 확인 (100 초과 시 100으로 보정)
    s = "SELECT law_name FROM articles WHERE law_name LIKE '%민법%' LIMIT 1000"
    capped = router._ensure_limit(s)
    assert " limit 100" in capped.lower()
