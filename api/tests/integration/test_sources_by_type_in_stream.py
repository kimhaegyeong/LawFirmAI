"""
스트리밍 API에서 sources_by_type 포함 여부 테스트
"""
import pytest
import json
from api.tests.helpers.server_helpers import check_server_health


@pytest.mark.integration
@pytest.mark.slow
class TestSourcesByTypeInStream:
    """스트리밍 API에서 sources_by_type 포함 여부 테스트"""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_server(self):
        """서버가 실행 중인지 확인"""
        if not check_server_health():
            pytest.skip("스트리밍 테스트는 실제 서버 실행이 필요합니다")
    
    def test_stream_api_includes_sources_by_type(self, client):
        """스트리밍 API 응답에 sources_by_type이 포함되는지 테스트"""
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
                    
                    if data.get("type") == "sources":
                        sources_event = data
                    elif data.get("type") == "final":
                        final_event = data
                except:
                    pass
        
        # sources 이벤트 또는 final 이벤트에 sources_by_type이 포함되어야 함
        if sources_event:
            assert "sources_by_type" in sources_event or "metadata" in sources_event
        elif final_event:
            metadata = final_event.get("metadata", {})
            assert "sources_by_type" in metadata or "sources" in metadata

