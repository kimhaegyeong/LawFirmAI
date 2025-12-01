# -*- coding: utf-8 -*-
"""
lstrmAI 데이터 수집기
API 응답을 데이터베이스에 저장하는 수집기
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from scripts.ingest.base.collector import BaseCollector
from scripts.ingest.config.api_configs import LSTRM_AI_CONFIG
from scripts.ingest.lstrm_ai_client import LstrmAIClient

logger = logging.getLogger(__name__)


class LstrmAICollector(BaseCollector):
    """lstrmAI 데이터 수집기"""
    
    def __init__(self, client: LstrmAIClient, db_path: str):
        """
        Args:
            client: LstrmAIClient 인스턴스
            db_path: 데이터베이스 파일 경로
        """
        super().__init__(client, db_path)
        self.config = LSTRM_AI_CONFIG
    
    @property
    def table_name(self) -> str:
        """저장할 테이블명"""
        return self.config.table_name
    
    @property
    def search_wrapper_key(self) -> str:
        """응답 래퍼 키"""
        return self.config.search_wrapper_key
    
    @property
    def items_key(self) -> str:
        """항목 배열 키"""
        return self.config.items_key
    
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
            items = response.get(self.items_key, []) or []
            if not items:
                # items가 없을 경우 다른 필드명 확인
                items = response.get('법령용어', []) or []
            
            if not items:
                logger.warning(f"응답에 항목이 없습니다. 응답 키: {list(response.keys())}")
                logger.debug(f"응답 내용: {response}")
                return 0
            
            # items가 딕셔너리 배열인지 확인
            if items and isinstance(items[0], str):
                logger.warning("items가 문자열 배열입니다. 응답 구조를 확인해야 합니다.")
                logger.debug(f"items 샘플: {items[:3]}")
                items = []
            
            saved_count = 0
            
            for item in items:
                # item이 딕셔너리가 아닌 경우 스킵
                if not isinstance(item, dict):
                    logger.warning(f"항목이 딕셔너리가 아닙니다: {type(item)}, 값: {item}")
                    continue
                
                # 요청 URL 생성
                request_url = self._build_request_url(
                    search_keyword, page, display, homonym_yn=homonym_yn
                )
                
                # 필드 추출
                fields = self._extract_item_fields(item)
                
                cursor.execute(f"""
                    INSERT OR IGNORE INTO {self.table_name} (
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
                    raw_json,
                    fields.get('term_id'),
                    fields.get('term_name'),
                    fields.get('homonym_exists'),
                    fields.get('homonym_note'),
                    fields.get('term_relation_link'),
                    fields.get('article_relation_link'),
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
            pass
    
    def _extract_item_fields(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """항목 필드 추출"""
        def get_field(field_name: str) -> Any:
            """설정된 필드명 변형들을 시도하여 값 추출"""
            mappings = self.config.field_mappings.get(field_name, [])
            for mapping in mappings:
                value = item.get(mapping)
                if value is not None:
                    return value
            return None
        
        return {
            'term_id': get_field('term_id'),
            'term_name': get_field('term_name'),
            'homonym_exists': get_field('homonym_exists'),
            'homonym_note': get_field('homonym_note'),
            'term_relation_link': get_field('term_relation_link'),
            'article_relation_link': get_field('article_relation_link')
        }

