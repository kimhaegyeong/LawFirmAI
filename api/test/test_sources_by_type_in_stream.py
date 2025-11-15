"""
스트리밍 API에서 sources_by_type 포함 여부 테스트
"""
import pytest
import json
import sys
import os
from typing import Dict, Any
from fastapi.testclient import TestClient

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.main import app


@pytest.fixture
async def client():
    """테스트용 AsyncClient fixture"""
    from fastapi.testclient import TestClient
    return TestClient(app)


def test_stream_api_includes_sources_by_type(client):
    """스트리밍 API 응답에 sources_by_type이 포함되는지 테스트"""
    # 스트리밍 테스트는 실제 서버가 필요하므로 skip
    pytest.skip("스트리밍 테스트는 실제 서버 실행이 필요합니다")
    
    response = client.post(
        "/api/v1/chat/stream",
        json={
            "message": "계약 해제 조건은?",
            "session_id": "test_session_sources_by_type"
        },
        headers={"Accept": "text/event-stream"},
        stream=True
    )
    
    assert response.status_code == 200
    
    sources_event = None
    final_event = None
    
    for line in response.iter_lines():
        if not line:
            continue
        
        if line.startswith(b"data: "):
            try:
                data = json.loads(line[6:].decode('utf-8'))
                event_type = data.get("type")
                
                if event_type == "sources":
                    sources_event = data
                elif event_type == "final":
                    final_event = data
                    
                # sources 이벤트를 찾으면 종료
                if sources_event:
                    break
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    
    # sources 이벤트 검증
    if sources_event:
        assert "metadata" in sources_event
        metadata = sources_event["metadata"]
        
        # sources_by_type 필드 확인
        assert "sources_by_type" in metadata or "sources_detail" in metadata
        
        # sources_detail이 있으면 sources_by_type도 있어야 함 (또는 생성 가능해야 함)
        if "sources_detail" in metadata and metadata["sources_detail"]:
            sources_detail = metadata["sources_detail"]
            assert isinstance(sources_detail, list)
            
            # sources_by_type이 있으면 구조 검증
            if "sources_by_type" in metadata:
                sources_by_type = metadata["sources_by_type"]
                assert isinstance(sources_by_type, dict)
                assert "statute_article" in sources_by_type
                assert "case_paragraph" in sources_by_type
                assert "decision_paragraph" in sources_by_type
                assert "interpretation_paragraph" in sources_by_type
    
    # final 이벤트 검증
    if final_event:
        assert "metadata" in final_event
        final_metadata = final_event["metadata"]
        
        # final 이벤트에도 sources_by_type이 있을 수 있음
        if "sources_by_type" in final_metadata:
            sources_by_type = final_metadata["sources_by_type"]
            assert isinstance(sources_by_type, dict)


@pytest.mark.skip(reason="통합 테스트는 서버 실행이 필요합니다")
def test_sources_by_type_structure(client):
    """sources_by_type 구조 검증 테스트"""
    response = await client.post(
        "/api/v1/chat/stream",
        json={
            "message": "민법 계약 해제에 대해 알려줘",
            "session_id": "test_session_structure"
        },
        headers={"Accept": "text/event-stream"}
    )
    
    assert response.status_code == 200
    
    sources_event = None
    
    async for line in response.iter_lines():
        if not line:
            continue
        
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                if data.get("type") == "sources":
                    sources_event = data
                    break
            except json.JSONDecodeError:
                continue
    
    if sources_event and "metadata" in sources_event:
        metadata = sources_event["metadata"]
        
        if "sources_by_type" in metadata:
            sources_by_type = metadata["sources_by_type"]
            
            # 구조 검증
            assert isinstance(sources_by_type, dict)
            assert "statute_article" in sources_by_type
            assert "case_paragraph" in sources_by_type
            assert "decision_paragraph" in sources_by_type
            assert "interpretation_paragraph" in sources_by_type
            
            # 각 타입이 리스트인지 확인
            for key, value in sources_by_type.items():
                assert isinstance(value, list), f"{key} should be a list"
                
                # 리스트의 각 항목이 딕셔너리인지 확인
                for item in value:
                    assert isinstance(item, dict), f"Items in {key} should be dictionaries"


@pytest.mark.skip(reason="통합 테스트는 서버 실행이 필요합니다")
def test_legal_references_backward_compatibility(client):
    """legal_references 하위 호환성 테스트"""
    response = await client.post(
        "/api/v1/chat/stream",
        json={
            "message": "민법 계약 해제에 대해 알려줘",
            "session_id": "test_session_compatibility"
        },
        headers={"Accept": "text/event-stream"}
    )
    
    assert response.status_code == 200
    
    sources_event = None
    
    async for line in response.iter_lines():
        if not line:
            continue
        
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                if data.get("type") == "sources":
                    sources_event = data
                    break
            except json.JSONDecodeError:
                continue
    
    if sources_event and "metadata" in sources_event:
        metadata = sources_event["metadata"]
        
        # legal_references 필드가 있어야 함 (하위 호환성)
        assert "legal_references" in metadata
        assert isinstance(metadata["legal_references"], list)
        
        # sources_detail이 있으면 legal_references도 추출되어야 함
        if "sources_detail" in metadata and metadata["sources_detail"]:
            sources_detail = metadata["sources_detail"]
            statute_count = sum(1 for item in sources_detail if item.get("type") == "statute_article")
            
            # statute_article이 있으면 legal_references도 있어야 함 (또는 빈 배열)
            if statute_count > 0:
                # legal_references는 sources_detail에서 추출되므로 있을 수 있음
                assert isinstance(metadata["legal_references"], list)


@pytest.mark.skip(reason="통합 테스트는 서버 실행이 필요합니다")
def test_sources_by_type_matches_sources_detail(client):
    """sources_by_type과 sources_detail의 일관성 테스트"""
    response = await client.post(
        "/api/v1/chat/stream",
        json={
            "message": "계약 해제 조건은?",
            "session_id": "test_session_consistency"
        },
        headers={"Accept": "text/event-stream"}
    )
    
    assert response.status_code == 200
    
    sources_event = None
    
    async for line in response.iter_lines():
        if not line:
            continue
        
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                if data.get("type") == "sources":
                    sources_event = data
                    break
            except json.JSONDecodeError:
                continue
    
    if sources_event and "metadata" in sources_event:
        metadata = sources_event["metadata"]
        
        if "sources_by_type" in metadata and "sources_detail" in metadata:
            sources_by_type = metadata["sources_by_type"]
            sources_detail = metadata["sources_detail"]
            
            # sources_by_type의 총 개수가 sources_detail과 일치해야 함
            total_in_by_type = (
                len(sources_by_type.get("statute_article", [])) +
                len(sources_by_type.get("case_paragraph", [])) +
                len(sources_by_type.get("decision_paragraph", [])) +
                len(sources_by_type.get("interpretation_paragraph", []))
            )
            
            assert total_in_by_type == len(sources_detail), \
                f"sources_by_type total ({total_in_by_type}) should match sources_detail length ({len(sources_detail)})"

