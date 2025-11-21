"""
테스트 클라이언트 헬퍼 함수
"""
from fastapi.testclient import TestClient
from typing import Dict, Any, Optional


def create_test_client():
    """테스트 클라이언트 생성"""
    from api.main import app
    return TestClient(app)


def make_chat_request(
    client: TestClient,
    message: str,
    session_id: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    endpoint: str = "/api/v1/chat"
) -> Any:
    """채팅 요청 헬퍼"""
    data = {"message": message}
    if session_id:
        data["session_id"] = session_id
    
    response = client.post(
        endpoint,
        json=data,
        headers=headers or {}
    )
    return response


def make_stream_request(
    client: TestClient,
    message: str,
    session_id: Optional[str] = None,
    endpoint: str = "/api/v1/chat/stream"
):
    """스트리밍 요청 헬퍼"""
    data = {"message": message}
    if session_id:
        data["session_id"] = session_id
    
    return client.post(
        endpoint,
        json=data,
        headers={"Accept": "text/event-stream"}
    )

