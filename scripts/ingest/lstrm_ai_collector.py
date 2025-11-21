# -*- coding: utf-8 -*-
"""
lstrmAI 데이터 수집기
API 응답을 데이터베이스에 저장하는 수집기
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from lawfirm_langgraph.core.data.connection_pool import get_connection_pool

from scripts.ingest.lstrm_ai_client import LstrmAIClient

logger = logging.getLogger(__name__)


class LstrmAICollector:
    """lstrmAI 데이터 수집기"""
    
    def __init__(self, client: LstrmAIClient, db_path: str):
        """
        Args:
            client: LstrmAIClient 인스턴스
            db_path: 데이터베이스 파일 경로
        """
        self.client = client
        self.db_path = db_path
        self.connection_pool = get_connection_pool(db_path)
    
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
                
                # 응답 구조 확인: lstrmAISearch로 래핑되어 있을 수 있음
                search_data = response.get('lstrmAISearch') or response
                
                # 검색결과개수 확인 (문자열일 수 있으므로 변환)
                total_count_str = (
                    search_data.get('검색결과개수') or 
                    search_data.get('totalCnt') or 
                    search_data.get('totalCount') or
                    '0'
                )
                try:
                    total_count = int(total_count_str)
                except (ValueError, TypeError):
                    total_count = 0
                
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
                    break
                
                # 마지막 페이지 확인
                if current_page * num_of_rows >= total_count:
                    logger.info(f"페이지 {page}: 마지막 페이지 도달")
                    break
                
                page += 1
                
            except Exception as e:
                logger.error(f"페이지 {page} 수집 실패: {e}", exc_info=True)
                break
        
        return total_saved
    
    def _save_response(
        self,
        response: Dict[str, Any],
        search_keyword: str,
        page: int,
        display: int,
        homonym_yn: Optional[str] = None
    ) -> int:
        """응답 데이터를 DB에 저장 (원본 JSON 포함)"""
        conn = self.connection_pool.get_connection()
        try:
            cursor = conn.cursor()
            
            # 전체 응답을 JSON 문자열로 저장
            raw_json = json.dumps(response, ensure_ascii=False, indent=None)
            
            # 각 결과 항목을 개별 레코드로 저장
            items = response.get('items', []) or []
            if not items:
                # items가 없을 경우 다른 필드명 확인
                items = response.get('법령용어', []) or []
            
            if not items:
                logger.warning(f"응답에 항목이 없습니다. 응답 키: {list(response.keys())}")
                logger.debug(f"응답 내용: {response}")
                return 0
            
            # items가 딕셔너리 배열인지 확인
            if items and isinstance(items[0], str):
                logger.warning(f"items가 문자열 배열입니다. 응답 구조를 확인해야 합니다.")
                logger.debug(f"items 샘플: {items[:3]}")
                # 다른 구조일 수 있으므로 빈 배열로 처리
                items = []
            
            saved_count = 0
            
            for item in items:
                # item이 딕셔너리가 아닌 경우 스킵
                if not isinstance(item, dict):
                    logger.warning(f"항목이 딕셔너리가 아닙니다: {type(item)}, 값: {item}")
                    continue
                
                # 요청 URL 생성
                request_url = self._build_request_url(
                    search_keyword, page, display, homonym_yn
                )
                
                # 필드명 정규화 (공백 제거 및 다양한 필드명 시도)
                term_id = (
                    item.get('법령용어 id') or 
                    item.get('법령용어id') or 
                    item.get('법령용어_id') or
                    item.get('id') or
                    None
                )
                term_name = item.get('법령용어명') or item.get('법령용어 명')
                homonym_exists = (
                    item.get('동음이의어존재여부') or 
                    item.get('동음이의어 존재여부') or
                    item.get('동음이의어존재 여부')
                )
                homonym_note = item.get('비고')
                term_relation_link = (
                    item.get('용어간관계링크') or 
                    item.get('용어 간관계링크') or
                    item.get('용어간관계 링크')
                )
                article_relation_link = (
                    item.get('조문간관계링크') or
                    item.get('조문 간관계링크') or
                    item.get('조문간관계 링크')
                )
                
                cursor.execute("""
                    INSERT OR IGNORE INTO open_law_lstrm_ai_data (
                        search_keyword, search_page, search_display, homonym_yn,
                        raw_response_json,
                        term_id, term_name, homonym_exists, homonym_note,
                        term_relation_link, article_relation_link,
                        collection_method, api_request_url,
                        total_count, page_number, num_of_rows,
                        collected_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    search_keyword, page, display, homonym_yn,
                    raw_json,  # 원본 JSON 저장
                    term_id,
                    term_name,
                    homonym_exists,
                    homonym_note,
                    term_relation_link,
                    article_relation_link,
                    'keyword' if search_keyword else 'all',
                    request_url,
                    response.get('검색결과개수'),
                    response.get('page'),
                    response.get('numOfRows'),
                    datetime.now().isoformat()
                ))
                if cursor.rowcount > 0:
                    saved_count += 1
            
            conn.commit()
            return saved_count
        except Exception as e:
            conn.rollback()
            logger.error(f"데이터 저장 실패: {e}", exc_info=True)
            raise
        finally:
            # 연결 풀 사용 시 close() 불필요
            pass
    
    def _build_request_url(
        self,
        query: str,
        page: int,
        display: int,
        homonym_yn: Optional[str] = None
    ) -> str:
        """요청 URL 생성"""
        params = {
            'OC': self.client.oc,
            'target': 'lstrmAI',
            'type': 'JSON',
            'query': query,
            'page': page,
            'display': display
        }
        if homonym_yn:
            params['homonymYn'] = homonym_yn
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items() if v])
        return f"{self.client.base_url}/lawSearch.do?{query_string}"

