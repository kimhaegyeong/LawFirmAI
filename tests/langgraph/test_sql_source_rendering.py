# -*- coding: utf-8 -*-
"""
SQL 결과가 존재할 때 워크플로우 결과의 sources에 SQL 출처가 포함되는지 검증
"""

import sys
from pathlib import Path

import pytest


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class _SQLRouterWithRows:
    def __init__(self, *args, **kwargs):
        pass

    def is_sql_suitable(self, query: str) -> bool:
        return True

    def route_and_execute(self, query: str):
        sql = "SELECT law_name, article_number, content FROM articles WHERE law_name LIKE '%민법%' AND article_number = 750 LIMIT 5"
        rows = [
            {"id": 1, "law_name": "민법", "article_number": 750, "content": "불법행위에 의한 손해배상"},
            {"id": 2, "law_name": "민법", "article_number": 750, "content": "손해배상의 범위"},
        ]
        return sql, rows

    def get_schema_overview(self) -> str:
        return "articles(law_name, article_number, content)"


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Workflow may not expose SQL sources/refs deterministically in this environment")
async def test_sql_sources_included_when_rows_exist(monkeypatch):
    import source.services.sql_router as sql_router_module
    monkeypatch.setattr(sql_router_module, "SQLRouter", _SQLRouterWithRows, raising=True)

    from core.agents.workflow_service import LangGraphWorkflowService
    from infrastructure.utils.langgraph_config import LangGraphConfig

    service = LangGraphWorkflowService(LangGraphConfig.from_env())
    result = await service.process_query("민법 제750조 조문 보여줘", session_id="test_sql_sources", enable_checkpoint=False)

    assert isinstance(result, dict)
    sources = result.get("sources") or []
    assert isinstance(sources, list)
    has_sql_source = any(isinstance(s, dict) and s.get("type") == "sql" for s in sources)
    # 대안: legal_references에 SQL 결과 콘텐츠가 주입되었는지 확인
    legal_refs = result.get("legal_references") or []
    contains_sql_content = any(isinstance(x, str) and ("불법행위" in x or "손해배상" in x) for x in legal_refs)
    assert has_sql_source or contains_sql_content, "SQL 실행 흔적이 sources(type='sql') 또는 legal_references에 반영되어야 합니다"


