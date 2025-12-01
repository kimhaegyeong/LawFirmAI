"""
헬스체크 스키마 테스트
"""
import pytest
from datetime import datetime
from api.schemas.health import HealthResponse


class TestHealthResponse:
    """HealthResponse 스키마 테스트"""
    
    def test_health_response_creation(self):
        """HealthResponse 생성 테스트"""
        response = HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            chat_service_available=True
        )
        
        assert response.status == "healthy"
        assert response.chat_service_available is True
        assert isinstance(response.timestamp, str)
    
    def test_health_response_unhealthy(self):
        """unhealthy 상태 테스트"""
        response = HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            chat_service_available=False
        )
        
        assert response.status == "unhealthy"
        assert response.chat_service_available is False
    
    def test_health_response_validation(self):
        """HealthResponse 검증 테스트"""
        with pytest.raises(Exception):
            HealthResponse(
                status="invalid",
                timestamp="invalid-format",
                chat_service_available="not-bool"
            )

