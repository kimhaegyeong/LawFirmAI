"""
Chat API E2E 플로우 테스트
"""
import pytest
import requests
from api.tests.helpers.server_helpers import check_server_health, wait_for_server

API_BASE_URL = "http://localhost:8000"


@pytest.mark.e2e
@pytest.mark.slow
class TestChatAPIFlow:
    """Chat API 전체 플로우 테스트"""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_server(self):
        """서버가 실행 중인지 확인"""
        if not check_server_health():
            pytest.skip("서버가 실행 중이지 않습니다. 서버를 시작한 후 테스트를 실행하세요.")
    
    def test_chat_api_basic_flow(self):
        """기본 Chat API 플로우 테스트"""
        test_data = {
            "message": "전세금 반환 보증에 대해 설명해주세요",
            "session_id": None,
            "enable_checkpoint": False
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/chat",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        assert response.status_code == 200, f"응답 상태 코드: {response.status_code}, 응답: {response.text}"
        
        result = response.json()
        
        # 필수 필드 확인
        assert "answer" in result, "응답에 'answer' 필드가 없습니다."
        assert "sources" in result, "응답에 'sources' 필드가 없습니다."
        assert "sources_detail" in result, "응답에 'sources_detail' 필드가 없습니다."
        assert "confidence" in result, "응답에 'confidence' 필드가 없습니다."
        assert "related_questions" in result, "응답에 'related_questions' 필드가 없습니다."
        
        # 답변 내용 확인
        answer = result.get("answer", "")
        assert len(answer) > 0, "답변이 비어있습니다."
        
        # Sources 확인
        sources = result.get("sources", [])
        assert isinstance(sources, list), "sources는 리스트여야 합니다."
        
        # Sources Detail 확인
        sources_detail = result.get("sources_detail", [])
        assert isinstance(sources_detail, list), "sources_detail은 리스트여야 합니다."
        
        # Related Questions 확인
        related_questions = result.get("related_questions", [])
        assert isinstance(related_questions, list), "related_questions는 리스트여야 합니다."
    
    def test_chat_api_with_session(self):
        """세션을 사용한 Chat API 테스트"""
        # 세션 생성
        session_data = {"title": "테스트 세션"}
        session_response = requests.post(
            f"{API_BASE_URL}/api/v1/sessions",
            json=session_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if session_response.status_code == 200:
            session_id = session_response.json().get("session_id")
            
            # 세션 ID를 사용한 채팅
            test_data = {
                "message": "계약 해지 사유에 대해 알려주세요",
                "session_id": session_id,
                "enable_checkpoint": False
            }
            
            response = requests.post(
                f"{API_BASE_URL}/api/v1/chat",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            
            assert response.status_code == 200
            result = response.json()
            assert "answer" in result

