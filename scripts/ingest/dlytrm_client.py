# -*- coding: utf-8 -*-
"""
dlytrm (일상용어) API 클라이언트
국가법령정보 공동활용 LAW OPEN DATA - 일상용어 검색 API 클라이언트
"""

import logging
import time
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


class DlytrmClient:
    """dlytrm (일상용어) API 클라이언트"""
    
    def __init__(self, oc: str, base_url: str = "https://www.law.go.kr/DRF"):
        """
        Args:
            oc: 사용자 이메일 ID (g4c@korea.kr일 경우 oc=g4c)
            base_url: API 기본 URL
        """
        self.oc = oc
        self.base_url = base_url
        self.rate_limit_delay = 0.5  # 요청 간 지연 (초)
    
    def search_terms(
        self,
        query: str = "",
        page: int = 1,
        display: int = 100
    ) -> Dict[str, Any]:
        """일상용어 검색"""
        params = {
            'OC': self.oc,
            'target': 'dlytrm',
            'type': 'JSON',
            'query': query,
            'page': page,
            'display': display
        }
        
        return self._make_request(params)
    
    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """API 요청 실행 (재시도 로직 포함)"""
        url = f"{self.base_url}/lawSearch.do"
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                time.sleep(self.rate_limit_delay)
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                # 응답 내용 확인
                content_type = response.headers.get('Content-Type', '')
                if 'json' not in content_type.lower():
                    logger.warning(f"응답이 JSON이 아닙니다. Content-Type: {content_type}")
                    logger.debug(f"응답 내용 (처음 500자): {response.text[:500]}")
                
                # JSON 파싱 시도
                try:
                    return response.json()
                except ValueError as json_error:
                    logger.error(f"JSON 파싱 실패: {json_error}")
                    logger.error(f"응답 상태 코드: {response.status_code}")
                    logger.error(f"응답 헤더: {dict(response.headers)}")
                    logger.error(f"응답 내용 (처음 1000자): {response.text[:1000]}")
                    logger.error(f"요청 URL: {response.url}")
                    raise
                    
            except requests.exceptions.Timeout as e:
                logger.warning(f"요청 타임아웃 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # 지수 백오프
                    continue
                raise
            except requests.exceptions.RequestException as e:
                logger.warning(f"요청 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # 지수 백오프
                    continue
                raise
            except Exception as e:
                logger.error(f"예상치 못한 오류: {e}")
                raise
        
        raise Exception("최대 재시도 횟수 초과")

