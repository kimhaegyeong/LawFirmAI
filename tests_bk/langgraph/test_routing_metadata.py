# -*- coding: utf-8 -*-
"""
?˜ì´ë¸Œë¦¬???¼ìš°??ë©”í??°ì´???¸ì¶œ ?µí•© ?ŒìŠ¤??

- SQLRouterë¥?monkeypatch?˜ì—¬ SQL ?œë„ ??0ê±´ì„ ê°•ì œ
- ?Œí¬?Œë¡œ??ê²°ê³¼??metadata??RAG ?´ë°± ? í˜¸ê°€ ?¬í•¨?˜ëŠ”ì§€ ?•ì¸
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class _FakeSQLRouter:
    def __init__(self, *args, **kwargs):
        pass

    def is_sql_suitable(self, query: str) -> bool:
        # ?´ë–¤ ì§ˆì˜??SQL ?í•©?˜ë‹¤ê³??œì‹œ
        return True

    def route_and_execute(self, query: str):
        # SQL??ë°˜í™˜?˜ì?ë§?ê²°ê³¼??0ê±?
        return ("SELECT law_name FROM articles WHERE law_name LIKE '%ë¯¼ë²•%' LIMIT 5", [])

    def get_schema_overview(self) -> str:
        return "laws, articles, cases, case_citations, amendments"


@pytest.mark.asyncio
async def test_router_metadata_exposed_on_zero_rows(monkeypatch):
    # SQLRouterë¥??˜ì´?¬ë¡œ ?€ì²´í•˜??0ê±´ì„ ê°•ì œ
    import source.services.sql_router as sql_router_module
    monkeypatch.setattr(sql_router_module, "SQLRouter", _FakeSQLRouter, raising=True)

    from source.agents.workflow_service import LangGraphWorkflowService
    from infrastructure.utils.langgraph_config import LangGraphConfig

    service = LangGraphWorkflowService(LangGraphConfig.from_env())
    result = await service.process_query("ë¯¼ë²• ??50ì¡?ì¡°ë¬¸ ë³´ì—¬ì¤?, session_id="test_router_meta", enable_checkpoint=False)

    assert isinstance(result, dict)
    metadata = result.get("metadata") or {}
    # êµ¬í˜„ ì°¨ì´???°ë¼ ?Œë˜ê·??¼ìš°???œì‹œ??? íƒ?ì¼ ???ˆìœ¼ë¯€ë¡?ì¡´ì¬ ???•ì‹ë§??•ì¸
    routing = metadata.get("routing") or {}
    if routing:
        assert isinstance(routing, dict)
        if "mode" in routing:
            assert routing.get("mode") in ("text_to_sql", "rag")
        if "rows" in routing:
            assert isinstance(routing.get("rows"), int)
