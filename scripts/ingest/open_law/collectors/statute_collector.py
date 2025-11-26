# -*- coding: utf-8 -*-
"""
법령 수집기
Open Law API를 통해 법령 데이터를 수집하고 PostgreSQL에 저장
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from scripts.ingest.open_law.client import OpenLawClient

logger = logging.getLogger(__name__)


class StatuteCollector:
    """법령 수집기"""
    
    def __init__(self, client: OpenLawClient, database_url: str):
        """
        Args:
            client: OpenLawClient 인스턴스
            database_url: PostgreSQL 데이터베이스 URL
        """
        self.client = client
        self.engine = create_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False
        )
        self.Session = sessionmaker(bind=self.engine)
    
    def collect_statute_list(
        self,
        query: str = "",
        domain: str = "civil_law",
        max_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """법령 목록 수집"""
        statutes = []
        page = 1
        
        while True:
            if max_pages and page > max_pages:
                break
            
            try:
                response = self.client.search_statutes(
                    query=query,
                    page=page,
                    display=100
                )
                
                # 응답 구조 확인
                search_data = response.get('LawSearch') or response
                
                # 검색결과개수 확인
                total_count_str = (
                    search_data.get('totalCnt') or 
                    search_data.get('검색결과개수') or 
                    '0'
                )
                try:
                    total_count = int(total_count_str)
                except (ValueError, TypeError):
                    total_count = 0
                
                if total_count == 0:
                    logger.info(f"페이지 {page}: 검색 결과 없음")
                    break
                
                # 법령 목록 추출
                items = search_data.get('law', [])
                if not isinstance(items, list):
                    items = [items] if items else []
                
                for item in items:
                    item['domain'] = domain
                    statutes.append(item)
                
                logger.info(f"페이지 {page}: {len(items)}개 법령 수집 (전체: {len(statutes)}개)")
                
                # 다음 페이지가 없으면 종료
                if len(items) < 100:
                    break
                
                page += 1
                
            except Exception as e:
                logger.error(f"페이지 {page} 수집 실패: {e}")
                break
        
        return statutes
    
    def save_statute_list(self, statutes: List[Dict[str, Any]], output_path: str):
        """법령 목록을 JSON 파일로 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(statutes, f, ensure_ascii=False, indent=2)
        
        logger.info(f"법령 목록 저장 완료: {output_path} ({len(statutes)}개)")
    
    def load_statute_list(self, input_path: str) -> List[Dict[str, Any]]:
        """법령 목록을 JSON 파일에서 로드"""
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def collect_and_save_statute_content(
        self,
        statute: Dict[str, Any],
        domain: str
    ) -> int:
        """법령 본문 및 조문 수집 및 저장"""
        session = self.Session()
        article_count = 0
        
        try:
            law_id = statute.get('법령ID') or statute.get('법령일련번호')
            if not law_id:
                logger.warning(f"법령ID가 없습니다: {statute}")
                return 0
            
            # 법령 메타데이터 저장
            statute_id = self._save_statute_metadata(session, statute, domain)
            
            # 법령 본문 조회
            try:
                law_info = self.client.get_statute_info(law_id=law_id)
                
                # 응답 구조 확인 (디버깅 - 첫 번째 법령만 상세 로깅)
                if article_count == 0:  # 첫 번째 법령만
                    import json
                    logger.info(f"법령 응답 구조 (첫 번째 법령, ID: {law_id}):")
                    logger.info(f"  최상위 키: {list(law_info.keys())}")
                    if 'LawSearch' in law_info:
                        logger.info(f"  LawSearch 키: {list(law_info['LawSearch'].keys())}")
                        if 'law' in law_info['LawSearch']:
                            law_item = law_info['LawSearch']['law']
                            if isinstance(law_item, dict):
                                logger.info(f"  law 키: {list(law_item.keys())}")
                                # JSON 일부 출력 (너무 길지 않게)
                                law_json = json.dumps(law_item, ensure_ascii=False, indent=2)
                                if len(law_json) > 2000:
                                    logger.info(f"  law 내용 (처음 2000자):\n{law_json[:2000]}...")
                                else:
                                    logger.info(f"  law 내용:\n{law_json}")
                
                # 다양한 응답 구조 시도
                law_data = (
                    law_info.get('법령', {}) or
                    law_info.get('LawSearch', {}).get('law', {}) or
                    law_info.get('law', {}) or
                    law_info
                )
                
                # 조문 수집
                if law_data:
                    articles = self._extract_articles(law_data, law_id)
                    logger.debug(f"추출된 조문 수: {len(articles)}")
                    
                    for article in articles:
                        try:
                            if self._save_article(session, statute_id, article):
                                article_count += 1
                            # 중복인 경우는 False를 반환하므로 카운트하지 않음
                        except Exception as e:
                            logger.warning(f"조문 저장 실패: {e}")
                            session.rollback()
                else:
                    logger.warning(f"법령 데이터가 비어있습니다 (ID: {law_id})")
                
                session.commit()
                logger.info(f"법령 '{statute.get('법령명한글')}' 수집 완료: {article_count}개 조문")
                
            except Exception as e:
                logger.error(f"법령 본문 수집 실패 (ID: {law_id}): {e}")
                session.rollback()
        
        finally:
            session.close()
        
        return article_count
    
    def _save_statute_metadata(
        self,
        session,
        statute: Dict[str, Any],
        domain: str
    ) -> int:
        """법령 메타데이터 저장"""
        law_id = statute.get('법령ID') or statute.get('법령일련번호')
        
        # 날짜 변환
        proclamation_date = self._parse_date(statute.get('공포일자'))
        effective_date = self._parse_date(statute.get('시행일자'))
        
        # 기존 레코드 확인
        result = session.execute(
            text("SELECT id FROM statutes WHERE law_id = :law_id"),
            {"law_id": law_id}
        )
        existing = result.fetchone()
        
        if existing:
            # 업데이트
            session.execute(
                text("""
                    UPDATE statutes SET
                        law_name_kr = :law_name_kr,
                        law_abbrv = :law_abbrv,
                        law_type = :law_type,
                        proclamation_date = :proclamation_date,
                        proclamation_number = :proclamation_number,
                        effective_date = :effective_date,
                        ministry_code = :ministry_code,
                        ministry_name = :ministry_name,
                        amendment_type = :amendment_type,
                        domain = :domain,
                        raw_response_json = :raw_response_json,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE law_id = :law_id
                """),
                {
                    "law_id": law_id,
                    "law_name_kr": statute.get('법령명한글', ''),
                    "law_abbrv": statute.get('법령약칭명'),
                    "law_type": statute.get('법령구분명'),
                    "proclamation_date": proclamation_date,
                    "proclamation_number": statute.get('공포번호'),
                    "effective_date": effective_date,
                    "ministry_code": statute.get('소관부처코드'),
                    "ministry_name": statute.get('소관부처명'),
                    "amendment_type": statute.get('제개정구분명'),
                    "domain": domain,
                    "raw_response_json": json.dumps(statute, ensure_ascii=False)
                }
            )
            return existing[0]
        else:
            # 삽입
            result = session.execute(
                text("""
                    INSERT INTO statutes (
                        law_id, law_name_kr, law_abbrv, law_type,
                        proclamation_date, proclamation_number, effective_date,
                        ministry_code, ministry_name, amendment_type, domain,
                        raw_response_json
                    ) VALUES (
                        :law_id, :law_name_kr, :law_abbrv, :law_type,
                        :proclamation_date, :proclamation_number, :effective_date,
                        :ministry_code, :ministry_name, :amendment_type, :domain,
                        :raw_response_json
                    ) RETURNING id
                """),
                {
                    "law_id": law_id,
                    "law_name_kr": statute.get('법령명한글', ''),
                    "law_abbrv": statute.get('법령약칭명'),
                    "law_type": statute.get('법령구분명'),
                    "proclamation_date": proclamation_date,
                    "proclamation_number": statute.get('공포번호'),
                    "effective_date": effective_date,
                    "ministry_code": statute.get('소관부처코드'),
                    "ministry_name": statute.get('소관부처명'),
                    "amendment_type": statute.get('제개정구분명'),
                    "domain": domain,
                    "raw_response_json": json.dumps(statute, ensure_ascii=False)
                }
            )
            return result.fetchone()[0]
    
    def _extract_articles(self, law_data: Dict[str, Any], law_id: int) -> List[Dict[str, Any]]:
        """법령 데이터에서 조문 추출"""
        articles = []
        
        # 다양한 응답 구조 시도
        jo_list = None
        
        # 1. 조문.조문단위 구조 (실제 API 응답 구조)
        if '조문' in law_data:
            jo_data = law_data.get('조문', {})
            if isinstance(jo_data, dict):
                # 조문단위가 있는 경우
                if '조문단위' in jo_data:
                    jo_list = jo_data.get('조문단위', [])
                # 조문이 직접 배열인 경우
                elif isinstance(jo_data, list):
                    jo_list = jo_data
            elif isinstance(jo_data, list):
                jo_list = jo_data
        
        # 2. 법령본문.조문 구조
        if not jo_list and '법령본문' in law_data:
            jo_list = law_data.get('법령본문', {}).get('조문', [])
        
        # 3. 본문.조문 구조
        if not jo_list and '본문' in law_data:
            jo_list = law_data.get('본문', {}).get('조문', [])
        
        # 4. 조문목록 구조
        if not jo_list and '조문목록' in law_data:
            jo_list = law_data.get('조문목록', {}).get('조문', [])
            if not jo_list:
                jo_list = law_data.get('조문목록', [])
        
        if not jo_list:
            # 조문이 없으면 조문 목록 API로 시도
            logger.debug(f"조문이 없어 조문 목록 API로 시도 (ID: {law_id})")
            try:
                # 시행일자 가져오기
                ef_yd = law_data.get('시행일자') or law_data.get('시행일')
                if ef_yd:
                    # 조문 목록 조회 (JO 없이 전체 조문 목록)
                    article_list_response = self.client.get_statute_article(
                        law_id=law_id,
                        ef_yd=int(str(ef_yd).replace('-', '').replace('/', '')[:8])
                    )
                    logger.debug(f"조문 목록 API 응답 키: {list(article_list_response.keys())}")
                    
                    # 조문 목록에서 조문 번호 추출
                    article_list_data = (
                        article_list_response.get('LawSearch', {}) or
                        article_list_response.get('law', {}) or
                        article_list_response
                    )
                    
                    # 조문 번호 목록 추출
                    jo_numbers = []
                    if isinstance(article_list_data, dict):
                        # 다양한 구조 시도
                        if '조문' in article_list_data:
                            jo_list = article_list_data.get('조문', [])
                        elif 'jo' in article_list_data:
                            jo_list = article_list_data.get('jo', [])
                        elif 'article' in article_list_data:
                            jo_list = article_list_data.get('article', [])
                        else:
                            # 조문 번호만 있는 경우
                            for key, value in article_list_data.items():
                                if '조문' in key.lower() or 'jo' in key.lower():
                                    if isinstance(value, list):
                                        jo_list = value
                                        break
                                    elif isinstance(value, dict):
                                        jo_list = value.get('list', []) or [value]
                                        break
                    
                    if jo_list:
                        logger.debug(f"조문 목록 API에서 {len(jo_list)}개 조문 발견")
            except Exception as e:
                logger.debug(f"조문 목록 API 호출 실패: {e}")
        
        if not jo_list:
            logger.warning(f"조문을 찾을 수 없습니다 (ID: {law_id})")
            return []
        
        if not isinstance(jo_list, list):
            jo_list = [jo_list] if jo_list else []
        
        for jo in jo_list:
            article_no = str(jo.get('조문번호', '')).zfill(4) + '00'
            article_title = jo.get('조문제목', '')
            article_content = jo.get('조문내용', '')
            
            # 항 추출
            hang_list = jo.get('항', [])
            if not isinstance(hang_list, list):
                hang_list = [hang_list] if hang_list else []
            
            if hang_list:
                for hang in hang_list:
                    hang_no = str(hang.get('항번호', '')).zfill(4) + '00'
                    hang_content = hang.get('항내용', '')
                    
                    # 호 추출
                    ho_list = hang.get('호', [])
                    if not isinstance(ho_list, list):
                        ho_list = [ho_list] if ho_list else []
                    
                    if ho_list:
                        for ho in ho_list:
                            ho_no = str(ho.get('호번호', '')).zfill(4) + '00'
                            ho_content = ho.get('호내용', '')
                            
                            # 목 추출
                            mok_list = ho.get('목', [])
                            if not isinstance(mok_list, list):
                                mok_list = [mok_list] if mok_list else []
                            
                            if mok_list:
                                for mok in mok_list:
                                    articles.append({
                                        'article_no': article_no,
                                        'article_title': article_title,
                                        'article_content': article_content,
                                        'clause_no': hang_no,
                                        'clause_content': hang_content,
                                        'item_no': ho_no,
                                        'item_content': ho_content,
                                        'sub_item_no': mok.get('목번호', ''),
                                        'sub_item_content': mok.get('목내용', ''),
                                        'raw_response_json': json.dumps(jo, ensure_ascii=False)
                                    })
                            else:
                                articles.append({
                                    'article_no': article_no,
                                    'article_title': article_title,
                                    'article_content': article_content,
                                    'clause_no': hang_no,
                                    'clause_content': hang_content,
                                    'item_no': ho_no,
                                    'item_content': ho_content,
                                    'sub_item_no': None,
                                    'sub_item_content': None,
                                    'raw_response_json': json.dumps(jo, ensure_ascii=False)
                                })
                    else:
                        articles.append({
                            'article_no': article_no,
                            'article_title': article_title,
                            'article_content': article_content,
                            'clause_no': hang_no,
                            'clause_content': hang_content,
                            'item_no': None,
                            'item_content': None,
                            'sub_item_no': None,
                            'sub_item_content': None,
                            'raw_response_json': json.dumps(jo, ensure_ascii=False)
                        })
            else:
                articles.append({
                    'article_no': article_no,
                    'article_title': article_title,
                    'article_content': article_content,
                    'clause_no': None,
                    'clause_content': None,
                    'item_no': None,
                    'item_content': None,
                    'sub_item_no': None,
                    'sub_item_content': None,
                    'raw_response_json': json.dumps(jo, ensure_ascii=False)
                })
        
        return articles
    
    def _save_article(
        self,
        session,
        statute_id: int,
        article: Dict[str, Any]
    ) -> bool:
        """
        조문 저장 (중복 방지)
        
        중복 판단 기준:
        - statute_id
        - article_no
        - article_title
        - article_content
        - clause_no
        
        Returns:
            bool: 저장 성공 여부 (중복인 경우 False)
        """
        article_no = article.get('article_no', '')
        article_title = article.get('article_title') or ''
        article_content = article.get('article_content', '')
        clause_no = article.get('clause_no') or ''
        
        # 중복 체크: 동일한 statute_id, article_no, article_title, article_content, clause_no 조합이 이미 있는지 확인
        check_result = session.execute(
            text("""
                SELECT id FROM statutes_articles
                WHERE statute_id = :statute_id
                  AND article_no = :article_no
                  AND COALESCE(article_title, '') = COALESCE(:article_title, '')
                  AND article_content = :article_content
                  AND COALESCE(clause_no, '') = COALESCE(:clause_no, '')
                LIMIT 1
            """),
            {
                "statute_id": statute_id,
                "article_no": article_no,
                "article_title": article_title,
                "article_content": article_content,
                "clause_no": clause_no
            }
        )
        existing_article = check_result.fetchone()
        
        if existing_article:
            # 이미 존재하는 경우 건너뛰기 (중복 방지)
            logger.debug(
                f"조문 중복 건너뛰기: statute_id={statute_id}, "
                f"article_no={article_no}, clause_no={clause_no}"
            )
            return False
        
        # 새로 삽입
        session.execute(
            text("""
                INSERT INTO statutes_articles (
                    statute_id, article_no, article_title, article_content,
                    clause_no, clause_content, item_no, item_content,
                    sub_item_no, sub_item_content, raw_response_json
                ) VALUES (
                    :statute_id, :article_no, :article_title, :article_content,
                    :clause_no, :clause_content, :item_no, :item_content,
                    :sub_item_no, :sub_item_content, :raw_response_json
                )
            """),
            {
                "statute_id": statute_id,
                "article_no": article_no,
                "article_title": article.get('article_title'),
                "article_content": article_content,
                "clause_no": article.get('clause_no'),
                "clause_content": article.get('clause_content'),
                "item_no": article.get('item_no'),
                "item_content": article.get('item_content'),
                "sub_item_no": article.get('sub_item_no'),
                "sub_item_content": article.get('sub_item_content'),
                "raw_response_json": article.get('raw_response_json', '{}')
            }
        )
        return True
    
    def _parse_date(self, date_str: Optional[Any]) -> Optional[str]:
        """날짜 문자열 파싱 (YYYYMMDD -> YYYY-MM-DD)"""
        if not date_str:
            return None
        
        date_str = str(date_str)
        if len(date_str) == 8 and date_str.isdigit():
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        return None

