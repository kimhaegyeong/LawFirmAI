# -*- coding: utf-8 -*-
"""
Open Law API 클라이언트
국가법령정보 공동활용 LAW OPEN DATA API 클라이언트
"""

import logging
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class OpenLawClient:
    """Open Law API 클라이언트"""
    
    def __init__(self, oc: str, base_url: str = "http://www.law.go.kr/DRF"):
        """
        Args:
            oc: 사용자 이메일 ID (g4c@korea.kr일 경우 oc=g4c)
            base_url: API 기본 URL
        """
        self.oc = oc
        self.base_url = base_url
        self.rate_limit_delay = 0.5  # 요청 간 지연 (초)
    
    def search_statutes(
        self,
        query: str = "",
        page: int = 1,
        display: int = 100,
        nw: int = 3,
        sort: str = "lasc"
    ) -> Dict[str, Any]:
        """현행법령(시행일) 목록 조회 (lsEfYdListGuide)"""
        params = {
            'OC': self.oc,
            'target': 'eflaw',
            'type': 'JSON',
            'query': query,
            'nw': nw,  # 3: 현행법령만
            'page': page,
            'display': display,
            'sort': sort
        }
        return self._make_request("lawSearch.do", params)
    
    def get_statute_info(
        self,
        law_id: Optional[int] = None,
        mst: Optional[int] = None,
        ef_yd: Optional[int] = None,
        jo: Optional[str] = None
    ) -> Dict[str, Any]:
        """현행법령(시행일) 본문 조회 (lsEfYdInfoGuide)"""
        params = {
            'OC': self.oc,
            'target': 'eflaw',
            'type': 'JSON'
        }
        
        if law_id:
            params['ID'] = law_id
        elif mst and ef_yd:
            params['MST'] = mst
            params['efYd'] = ef_yd
        else:
            raise ValueError("ID 또는 (MST + efYd) 중 하나는 필수입니다")
        
        if jo:
            params['JO'] = jo
        
        return self._make_request("lawService.do", params)
    
    def get_statute_article(
        self,
        law_id: Optional[int] = None,
        mst: Optional[int] = None,
        ef_yd: int = None,
        jo: str = None,
        hang: Optional[str] = None,
        ho: Optional[str] = None,
        mok: Optional[str] = None
    ) -> Dict[str, Any]:
        """현행법령(시행일) 본문 조항호목 조회 (lsEfYdJoListGuide)"""
        params = {
            'OC': self.oc,
            'target': 'eflawjosub',
            'type': 'JSON',
            'efYd': ef_yd
        }
        
        if law_id:
            params['ID'] = law_id
        elif mst:
            params['MST'] = mst
        else:
            raise ValueError("ID 또는 MST 중 하나는 필수입니다")
        
        if jo:
            params['JO'] = jo
        if hang:
            params['HANG'] = hang
        if ho:
            params['HO'] = ho
        if mok:
            params['MOK'] = mok
        
        return self._make_request("lawService.do", params)
    
    def search_precedents(
        self,
        query: str = "",
        page: int = 1,
        display: int = 100,
        jo: Optional[str] = None,
        org: Optional[str] = None,
        sort: str = "ddes"
    ) -> Dict[str, Any]:
        """판례 목록 조회 (precListGuide)"""
        params = {
            'OC': self.oc,
            'target': 'prec',
            'type': 'JSON',
            'query': query,
            'page': page,
            'display': display,
            'sort': sort
        }
        
        if jo:
            params['JO'] = jo
        if org:
            params['org'] = org
        
        return self._make_request("lawSearch.do", params)
    
    def get_precedent_info(
        self,
        precedent_id: int,
        lm: Optional[str] = None
    ) -> Dict[str, Any]:
        """판례 본문 조회 (precInfoGuide)"""
        params = {
            'OC': self.oc,
            'target': 'prec',
            'type': 'JSON',
            'ID': precedent_id
        }
        
        if lm:
            params['LM'] = lm
        
        return self._make_request("lawService.do", params)
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """API 요청 실행 (재시도 로직 포함)"""
        url = f"{self.base_url}/{endpoint}"
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

