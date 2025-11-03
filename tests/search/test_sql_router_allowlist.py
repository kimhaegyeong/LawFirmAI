# -*- coding: utf-8 -*-
"""
SQLRouter allow-list Î∞??àÏ†Ñ??Í≤ÄÏ¶??®ÏúÑ ?åÏä§??
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _router():
    from source.services.sql_router import SQLRouter
    return SQLRouter(db_path="data/lawfirm.db")


def test_disallow_write_and_ddl_keywords():
    r = _router()
    # ?¥Î? ?àÏ†Ñ??Í≤Ä??ÏßÅÏ†ë ?∏Ï∂ú
    assert r._is_sql_safe("SELECT * FROM articles LIMIT 10") is True
    assert r._is_sql_safe("DELETE FROM articles WHERE id=1 LIMIT 1") is False
    assert r._is_sql_safe("DROP TABLE articles") is False
    assert r._is_sql_safe("UPDATE cases SET court='X' LIMIT 1") is False


def test_disallow_unknown_table():
    r = _router()
    assert r._is_sql_safe("SELECT id FROM unknown WHERE id=1 LIMIT 1") is False


def test_disallow_unknown_columns_in_select():
    r = _router()
    # articles?êÎäî content, law_name, article_number, idÎß??àÏö©
    assert r._is_sql_safe("SELECT content, id FROM articles LIMIT 5") is True
    assert r._is_sql_safe("SELECT content, hacker FROM articles LIMIT 5") is False


def test_disallow_unknown_columns_in_where():
    r = _router()
    assert r._is_sql_safe("SELECT content FROM articles WHERE article_number = 1 LIMIT 5") is True
    assert r._is_sql_safe("SELECT content FROM articles WHERE secret = 1 LIMIT 5") is False


def test_limit_required_and_capped():
    r = _router()
    assert r._is_sql_safe("SELECT content FROM articles WHERE article_number=1 LIMIT 5") is True
    # LIMIT ÎØ∏Ï°¥?????àÏ†Ñ???§Ìå®
    assert r._is_sql_safe("SELECT content FROM articles WHERE article_number=1") is False
    # 100 Ï¥àÍ≥º ???àÏ†Ñ???§Ìå®
    assert r._is_sql_safe("SELECT content FROM articles WHERE article_number=1 LIMIT 1000") is False
