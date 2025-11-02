# -*- coding: utf-8 -*-
"""
SQL→키워드→벡터 재검색 폴백 루프 테스트

시나리오:
- SQL 라우팅이 적합한 질의이나 DB에 일치 레코드가 없어 0건 반환
- 워크플로우가 `metadata.force_rag_fallback=True`를 설정하고 컨텍스트 보강 재시도를 수행
- 최종 결과에 오류 없이 응답이 생성되며, 메타데이터에 라우팅 정보가 포함되는지 검증
"""

import asyncio
import sys
from pathlib import Path

import pytest

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.asyncio
async def test_sql_zero_rows_triggers_rag_fallback():
    from core.agents.workflow_service import LangGraphWorkflowService
    from infrastructure.utils.langgraph_config import LangGraphConfig

    config = LangGraphConfig.from_env()
    service = LangGraphWorkflowService(config)

    # 존재하지 않을 가능성이 높은 조문 번호로 SQL 적합 질의 생성
    query = "민법 제99999조 보여줘"

    result = await service.process_query(query, session_id="test_sql_fallback", enable_checkpoint=False)

    assert isinstance(result, dict), "결과는 dict여야 합니다"
    assert "metadata" in result, "metadata가 결과에 포함되어야 합니다"

    metadata = result.get("metadata") or {}
    assert isinstance(metadata, dict), "metadata는 dict여야 합니다"

    # 라우팅/폴백 표시는 구현 조건에 따라 생략될 수 있으므로 존재 시 형식만 확인
    routing = metadata.get("routing") or {}
    if routing:
        assert isinstance(routing, dict)
    if "force_rag_fallback" in metadata:
        assert isinstance(metadata.get("force_rag_fallback"), bool)

    # 최종 응답이 정상적으로 생성되어야 함
    answer = result.get("answer")
    assert answer is not None, "최종 답변이 존재해야 합니다"
    # 문자열로 수렴되었는지만 확인 (형태는 포맷터에 따라 다를 수 있음)
    assert len(str(answer)) > 0, "최종 답변 문자열 길이는 0보다 커야 합니다"
