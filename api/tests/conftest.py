"""
공통 pytest fixture 및 설정
"""
import pytest
import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# 프로젝트 루트를 sys.path에 추가 (한 번만)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

api_path = project_root / "api"
if str(api_path) not in sys.path:
    sys.path.insert(0, str(api_path))

from api.main import app
from api.config import api_config
from api.services.auth_service import auth_service


@pytest.fixture(scope="session")
def client():
    """테스트 클라이언트 fixture"""
    return TestClient(app)


@pytest.fixture
def mock_auth_disabled():
    """인증 비활성화 모킹"""
    with patch.object(api_config, 'auth_enabled', False):
        with patch.object(auth_service, 'is_auth_enabled', return_value=False):
            yield


@pytest.fixture
def mock_auth_enabled():
    """인증 활성화 모킹"""
    with patch.object(api_config, 'auth_enabled', True):
        with patch.object(api_config, 'jwt_secret_key', 'test_secret_key'):
            with patch.object(auth_service, 'is_auth_enabled', return_value=True):
                with patch.object(auth_service, 'verify_token', return_value={'sub': 'test_user'}):
                    yield


@pytest.fixture
def mock_rate_limit_disabled():
    """Rate limit 비활성화 모킹"""
    with patch.object(api_config, 'rate_limit_enabled', False):
        yield


@pytest.fixture
def mock_rate_limit_enabled():
    """Rate limit 활성화 모킹"""
    with patch.object(api_config, 'rate_limit_enabled', True):
        with patch.object(api_config, 'rate_limit_per_minute', 5):
            yield

