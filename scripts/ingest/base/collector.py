# -*- coding: utf-8 -*-
"""
공통 수집기 베이스 클래스
모든 수집기의 공통 로직을 제공
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from lawfirm_langgraph.core.data.connection_pool import get_connection_pool

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """공통 수집기 베이스 클래스"""
    
    def __init__(self, client, db_path: str):
        """
        Args:
            client: API 클라이언트 인스턴스
            db_path: 데이터베이스 파일 경로
        """
        self.client = client
        self.db_path = db_path
        self.connection_pool = get_connection_pool(db_path)
    
    @property
    @abstractmethod
    def table_name(self) -> str:
        """저장할 테이블명"""
        pass
    
    @property
    @abstractmethod
    def search_wrapper_key(self) -> str:
        """응답 래퍼 키 (예: 'dlytrmSearch', 'lstrmAISearch')"""
        pass
    
    @property
    @abstractmethod
    def items_key(self) -> str:
        """항목 배열 키 (예: 'items', '일상용어', '법령용어')"""
        pass
    
    def collect_by_keywords(
        self,
        keywords: List[str],
        max_pages_per_keyword: Optional[int] = None,
        start_page: int = 1
    ) -> int:
        """키워드 기반 수집"""
        total_saved = 0
        for keyword in keywords:
            logger.info(f"키워드 '{keyword}' 수집 시작 (시작 페이지: {start_page})")
            saved = self.collect_all_pages(
                query=keyword,
                max_pages=max_pages_per_keyword,
                start_page=start_page
            )
            total_saved += saved
            logger.info(f"키워드 '{keyword}' 수집 완료: {saved}건")
        return total_saved
    
    def collect_all_pages(
        self,
        query: str = "",
        max_pages: Optional[int] = None,
        start_page: int = 1
    ) -> int:
        """전체 페이지 수집"""
        page = start_page
        total_saved = 0
        
        while True:
            if max_pages and page > max_pages:
                break
            
            try:
                response = self.client.search_terms(
                    query=query,
                    page=page,
                    display=100
                )
                
                # 응답 검증
                if not response:
                    logger.warning(f"페이지 {page}: 빈 응답")
                    break
                
                # 응답 구조 확인: 래퍼 키로 래핑되어 있을 수 있음
                search_data = response.get(self.search_wrapper_key) or response
                
                # 검색결과개수 확인
                total_count = self._extract_total_count(search_data)
                
                if total_count == 0:
                    logger.info(f"페이지 {page}: 검색 결과 없음")
                    break
                
                # response를 search_data로 교체
                response = search_data
                
                # 데이터 저장
                saved = self._save_response(
                    response=response,
                    search_keyword=query,
                    page=page,
                    display=100
                )
                total_saved += saved
                
                logger.info(f"페이지 {page} 수집 완료: {saved}건 저장 (전체: {total_count}건)")
                
                # 다음 페이지 확인
                if not self._has_more_pages(response, page, saved, total_count):
                    break
                
                page += 1
                
            except Exception as e:
                logger.error(f"페이지 {page} 수집 실패: {e}", exc_info=True)
                break
        
        return total_saved
    
    def _extract_total_count(self, response: Dict[str, Any]) -> int:
        """총 개수 추출"""
        total_count_str = (
            response.get('검색결과개수') or 
            response.get('totalCnt') or 
            response.get('totalCount') or
            '0'
        )
        try:
            return int(total_count_str)
        except (ValueError, TypeError):
            return 0
    
    def _has_more_pages(
        self,
        response: Dict[str, Any],
        page: int,
        saved: int,
        total_count: int
    ) -> bool:
        """더 많은 페이지가 있는지 확인"""
        num_of_rows_str = response.get('numOfRows', '0')
        try:
            num_of_rows = int(num_of_rows_str)
        except (ValueError, TypeError):
            num_of_rows = saved  # 저장된 개수로 추정
        
        current_page_str = response.get('page', str(page))
        try:
            current_page = int(current_page_str)
        except (ValueError, TypeError):
            current_page = page
        
        if num_of_rows == 0 or saved == 0:
            logger.info(f"페이지 {page}: 더 이상 데이터 없음")
            return False
        
        # 마지막 페이지 확인
        if current_page * num_of_rows >= total_count:
            logger.info(f"페이지 {page}: 마지막 페이지 도달")
            return False
        
        return True
    
    @abstractmethod
    def _save_response(
        self,
        response: Dict[str, Any],
        search_keyword: str,
        page: int,
        display: int
    ) -> int:
        """응답 데이터를 DB에 저장"""
        pass
    
    @abstractmethod
    def _extract_item_fields(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """항목 필드 추출"""
        pass
    
    def _build_request_url(
        self,
        query: str,
        page: int,
        display: int,
        **kwargs
    ) -> str:
        """요청 URL 생성"""
        params = {
            'OC': self.client.oc,
            'target': self.client.target,
            'type': 'JSON',
            'query': query,
            'page': page,
            'display': display
        }
        params.update(kwargs)
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items() if v])
        return f"{self.client.base_url}/lawSearch.do?{query_string}"

