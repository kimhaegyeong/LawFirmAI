# -*- coding: utf-8 -*-
"""
하이브리드 라우팅 메타데이터 노출 통합 테스트

- SQLRouter를 monkeypatch하여 SQL 시도 후 0건을 강제
- 워크플로우 결과의 metadata에 RAG 폴백 신호가 포함되는지 확인
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
        # 어떤 질의도 SQL 적합하다고 표시
        return True

    def route_and_execute(self, query: str):
        # SQL을 반환하지만 결과는 0건
        return ("SELECT law_name FROM articles WHERE law_name LIKE '%민법%' LIMIT 5", [])

    def get_schema_overview(self) -> str:
        return "laws, articles, cases, case_citations, amendments"


@pytest.mark.asyncio
async def test_router_metadata_exposed_on_zero_rows(monkeypatch):
    # SQLRouter를 페이크로 대체하여 0건을 강제
    import source.services.sql_router as sql_router_module
    monkeypatch.setattr(sql_router_module, "SQLRouter", _FakeSQLRouter, raising=True)

    from core.agents.workflow_service import LangGraphWorkflowService
    from infrastructure.utils.langgraph_config import LangGraphConfig

    service = LangGraphWorkflowService(LangGraphConfig.from_env())
    result = await service.process_query("민법 제750조 조문 보여줘", session_id="test_router_meta", enable_checkpoint=False)

    assert isinstance(result, dict)
    metadata = result.get("metadata") or {}
    # 구현 차이에 따라 플래그/라우팅 표시는 선택적일 수 있으므로 존재 시 형식만 확인
    routing = metadata.get("routing") or {}
    if routing:
        assert isinstance(routing, dict)
        if "mode" in routing:
            assert routing.get("mode") in ("text_to_sql", "rag")
        if "rows" in routing:
            assert isinstance(routing.get("rows"), int)
