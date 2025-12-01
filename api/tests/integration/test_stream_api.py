"""
스트리밍 API 통합 테스트
"""
import pytest
import requests
from api.tests.helpers.server_helpers import check_server_health

BASE_URL = "http://localhost:8000/api/v1"
STREAM_ENDPOINT = f"{BASE_URL}/chat/stream"


@pytest.mark.integration
@pytest.mark.slow
class TestStreamAPI:
    """스트리밍 API 통합 테스트"""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_server(self):
        """서버가 실행 중인지 확인"""
        if not check_server_health():
            pytest.skip("서버가 실행 중이지 않습니다. 서버를 시작한 후 테스트를 실행하세요.")
    
    def test_stream_api_basic(self):
        """기본 스트리밍 API 테스트"""
        request_data = {
            "message": "민법 제750조 손해배상에 대해 설명해주세요",
            "session_id": "test-stream-session-001"
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        response = requests.post(
            STREAM_ENDPOINT,
            json=request_data,
            headers=headers,
            stream=True,
            timeout=60
        )
        
        assert response.status_code == 200, f"응답 상태 코드: {response.status_code}"
        assert response.headers.get("Content-Type", "").startswith("text/event-stream")
        
        # SSE 데이터 파싱
        events = []
        buffer = ""
        
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            
            if line.startswith("data: "):
                data_str = line[6:]  # "data: " 제거
                try:
                    event_data = requests.compat.json.loads(data_str)
                    events.append(event_data)
                except:
                    pass
        
        # 최소한의 이벤트가 있어야 함
        assert len(events) > 0, "이벤트가 수신되지 않았습니다."
        
        # done 이벤트가 있어야 함
        event_types = [event.get("type") for event in events]
        assert "done" in event_types or "final" in event_types, "완료 이벤트가 없습니다."

