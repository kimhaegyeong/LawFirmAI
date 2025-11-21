"""
익명 사용자 질의 제한 테스트
"""
import pytest
from api.services.anonymous_quota_service import anonymous_quota_service
from api.config import api_config


@pytest.mark.integration
class TestAnonymousQuota:
    """익명 사용자 질의 제한 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        # 테스트 전에 익명 사용자 제한 활성화
        api_config.anonymous_quota_enabled = True
        api_config.anonymous_quota_limit = 3
        anonymous_quota_service.enabled = True
        anonymous_quota_service.quota_limit = 3
        
        # 테스트를 위해 카운터 리셋
        anonymous_quota_service._quota_store.clear()
    
    def test_anonymous_quota_service_initialization(self):
        """익명 사용자 제한 서비스 초기화 테스트"""
        assert anonymous_quota_service is not None
        assert anonymous_quota_service.is_enabled() is True
        assert anonymous_quota_service.quota_limit == 3
    
    def test_anonymous_quota_check_quota(self):
        """익명 사용자 질의 가능 여부 확인 테스트"""
        test_ip = "127.0.0.1"
        
        # 처음에는 질의 가능
        assert anonymous_quota_service.check_quota(test_ip) is True
        
        # 3회까지 질의 가능
        for i in range(3):
            anonymous_quota_service.increment_quota(test_ip)
            if i < 2:
                assert anonymous_quota_service.check_quota(test_ip) is True
            else:
                # 3회째는 제한에 도달
                assert anonymous_quota_service.check_quota(test_ip) is False
    
    def test_anonymous_quota_reset(self):
        """익명 사용자 제한 리셋 테스트"""
        test_ip = "127.0.0.1"
        
        # 제한까지 사용
        for _ in range(3):
            anonymous_quota_service.increment_quota(test_ip)
        
        assert anonymous_quota_service.check_quota(test_ip) is False
        
        # 리셋
        anonymous_quota_service._quota_store.clear()
        assert anonymous_quota_service.check_quota(test_ip) is True
    
    def test_anonymous_quota_different_ips(self):
        """다른 IP 주소별 독립적인 제한 테스트"""
        ip1 = "127.0.0.1"
        ip2 = "192.168.1.1"
        
        # IP1이 제한에 도달해도 IP2는 영향받지 않음
        for _ in range(3):
            anonymous_quota_service.increment_quota(ip1)
        
        assert anonymous_quota_service.check_quota(ip1) is False
        assert anonymous_quota_service.check_quota(ip2) is True

