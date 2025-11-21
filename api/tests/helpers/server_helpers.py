"""
서버 관련 헬퍼 함수
"""
import requests
import time
from typing import Optional


def wait_for_server(
    url: str = "http://localhost:8000/health",
    max_attempts: int = 30,
    delay: float = 1.0
) -> bool:
    """서버가 시작될 때까지 대기"""
    for i in range(max_attempts):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code in [200, 404]:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(delay)
    return False


def check_server_health(url: str = "http://localhost:8000/health") -> bool:
    """서버 상태 확인"""
    try:
        response = requests.get(url, timeout=2)
        return response.status_code == 200
    except:
        return False

