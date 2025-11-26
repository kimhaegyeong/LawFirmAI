# -*- coding: utf-8 -*-
"""
판례 수집기
Open Law API를 통해 판례 데이터를 수집하고 PostgreSQL에 저장
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


class PrecedentCollector:
    """판례 수집기"""
    
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
    
    def collect_precedent_list(
        self,
        query: str = "",
        jo: Optional[str] = None,
        domain: str = "civil_law",
        max_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """판례 목록 수집"""
        precedents = []
        page = 1
        
        while True:
            if max_pages and page > max_pages:
                break
            
            try:
                response = self.client.search_precedents(
                    query=query,
                    page=page,
                    display=100,
                    jo=jo
                )
                
                # 응답 구조 확인
                search_data = response.get('PrecSearch') or response
                
                # 검색결과개수 확인
                total_count_str = (
                    search_data.get('totalCnt') or 
                    search_data.get('검색결과갯수') or 
                    '0'
                )
                try:
                    total_count = int(total_count_str)
                except (ValueError, TypeError):
                    total_count = 0
                
                if total_count == 0:
                    logger.info(f"페이지 {page}: 검색 결과 없음")
                    break
                
                # 판례 목록 추출
                items = search_data.get('prec', [])
                if not isinstance(items, list):
                    items = [items] if items else []
                
                for item in items:
                    item['domain'] = domain
                    precedents.append(item)
                
                logger.info(f"페이지 {page}: {len(items)}개 판례 수집 (전체: {len(precedents)}개)")
                
                # 다음 페이지가 없으면 종료
                if len(items) < 100:
                    break
                
                page += 1
                
            except Exception as e:
                logger.error(f"페이지 {page} 수집 실패: {e}")
                break
        
        return precedents
    
    def save_precedent_list(self, precedents: List[Dict[str, Any]], output_path: str):
        """판례 목록을 JSON 파일로 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(precedents, f, ensure_ascii=False, indent=2)
        
        logger.info(f"판례 목록 저장 완료: {output_path} ({len(precedents)}개)")
    
    def load_precedent_list(self, input_path: str) -> List[Dict[str, Any]]:
        """판례 목록을 JSON 파일에서 로드"""
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_processed_precedent_ids(self, processed_path: str) -> set:
        """처리 완료된 판례 ID 집합 로드"""
        processed_path = Path(processed_path)
        if not processed_path.exists():
            return set()
        
        try:
            with open(processed_path, 'r', encoding='utf-8') as f:
                processed_list = json.load(f)
                # 판례일련번호 추출
                processed_ids = set()
                for item in processed_list:
                    precedent_id = item.get('판례일련번호') or item.get('prec id')
                    if precedent_id:
                        try:
                            processed_ids.add(int(precedent_id))
                        except (ValueError, TypeError):
                            pass
                return processed_ids
        except Exception as e:
            logger.warning(f"처리 완료 파일 로드 실패: {e}")
            return set()
    
    def save_processed_precedent(self, processed_path: str, precedent: Dict[str, Any]):
        """처리 완료된 판례를 파일에 추가"""
        processed_path = Path(processed_path)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 기존 처리 완료 목록 로드
        processed_list = []
        if processed_path.exists():
            try:
                with open(processed_path, 'r', encoding='utf-8') as f:
                    processed_list = json.load(f)
            except Exception:
                processed_list = []
        
        # 중복 체크 (판례일련번호 기준)
        precedent_id = precedent.get('판례일련번호') or precedent.get('prec id')
        existing_ids = {
            item.get('판례일련번호') or item.get('prec id')
            for item in processed_list
        }
        
        if precedent_id not in existing_ids:
            processed_list.append(precedent)
            
            # 파일 저장
            with open(processed_path, 'w', encoding='utf-8') as f:
                json.dump(processed_list, f, ensure_ascii=False, indent=2)
    
    def filter_unprocessed_precedents(
        self,
        precedents: List[Dict[str, Any]],
        processed_path: str
    ) -> List[Dict[str, Any]]:
        """처리 완료된 판례를 제외한 목록 반환"""
        processed_ids = self.load_processed_precedent_ids(processed_path)
        
        unprocessed = []
        for precedent in precedents:
            precedent_id = precedent.get('판례일련번호') or precedent.get('prec id')
            if precedent_id:
                try:
                    precedent_id_int = int(precedent_id)
                    if precedent_id_int not in processed_ids:
                        unprocessed.append(precedent)
                except (ValueError, TypeError):
                    # ID 변환 실패 시 포함
                    unprocessed.append(precedent)
            else:
                # ID가 없으면 포함
                unprocessed.append(precedent)
        
        return unprocessed
    
    def collect_and_save_precedent_content(
        self,
        precedent: Dict[str, Any],
        domain: str,
        verbose: bool = True,
        existing_sections: Optional[set] = None
    ) -> int:
        """
        판례 본문 수집 및 저장
        
        Args:
            precedent: 판례 정보 딕셔너리
            domain: 법률 분야 ('civil_law', 'criminal_law')
            verbose: 상세 로깅 여부
            existing_sections: 이미 수집된 섹션 타입 집합 (최적화용)
        
        Returns:
            int: 새로 저장된 섹션 개수
        """
        if existing_sections is None:
            existing_sections = set()
        
        session = self.Session()
        content_count = 0
        
        try:
            precedent_id = precedent.get('판례일련번호') or precedent.get('prec id')
            if not precedent_id:
                logger.warning(f"판례일련번호가 없습니다: {precedent}")
                return 0
            
            # 판례 ID를 정수로 변환
            try:
                precedent_id = int(precedent_id)
            except (ValueError, TypeError):
                logger.warning(f"판례일련번호를 정수로 변환할 수 없습니다: {precedent_id}")
                return 0
            
            # 판례 메타데이터 저장
            db_precedent_id = self._save_precedent_metadata(session, precedent, domain)
            
            # 판례 본문 조회
            try:
                prec_info = self.client.get_precedent_info(precedent_id=precedent_id)
                
                # '일치하는 판례가 없습니다' 에러 체크 - 있으면 건너뛰기
                if isinstance(prec_info.get('Law'), str):
                    law_str = prec_info.get('Law', '')
                    if '일치하는 판례가 없습니다' in law_str or '판례명을 확인하여 주십시오' in law_str:
                        session.commit()
                        if verbose:
                            logger.debug(f"판례를 찾을 수 없음 (ID: {precedent_id}): {law_str}")
                        return 0
                
                # 다양한 응답 구조 시도
                prec_data = None
                
                # PrecService가 dict인 경우 (가장 일반적)
                if isinstance(prec_info.get('PrecService'), dict):
                    prec_service = prec_info.get('PrecService', {})
                    # PrecService.prec 경로 확인
                    if 'prec' in prec_service:
                        prec_item = prec_service['prec']
                        if isinstance(prec_item, dict):
                            prec_data = prec_item
                        elif isinstance(prec_item, list) and len(prec_item) > 0:
                            prec_data = prec_item[0]
                    # PrecService 자체가 판례 데이터인 경우
                    elif any(key in prec_service for key in ['판시사항', '판결요지', '판례내용']):
                        prec_data = prec_service
                
                # Law가 dict인 경우
                if not prec_data and isinstance(prec_info.get('Law'), dict):
                    law_dict = prec_info.get('Law', {})
                    # Law.판례 경로 확인
                    if '판례' in law_dict:
                        prec_data = law_dict['판례']
                    # Law 자체가 판례 데이터인 경우
                    elif any(key in law_dict for key in ['판시사항', '판결요지', '판례내용']):
                        prec_data = law_dict
                
                # 다른 경로 시도
                if not prec_data:
                    prec_data = (
                        prec_info.get('판례', {}) or
                        prec_info.get('prec', {}) or
                        (prec_info if isinstance(prec_info, dict) and any(key in prec_info for key in ['판시사항', '판결요지', '판례내용']) else None)
                    )
                
                # 판례 본문 섹션 저장
                if prec_data and isinstance(prec_data, dict):
                    sections = self._extract_sections(prec_data)
                    for section in sections:
                        section_type = section.get('section_type', '')
                        
                        # 이미 수집된 섹션은 건너뛰기 (메모리 체크로 빠르게)
                        if section_type in existing_sections:
                            continue
                        
                        try:
                            # 중복 체크는 _save_precedent_content 내부에서 처리
                            saved = self._save_precedent_content(session, db_precedent_id, section)
                            if saved:
                                content_count += 1
                                # 새로 저장된 섹션을 existing_sections에 추가 (다음 섹션 체크 최적화)
                                existing_sections.add(section_type)
                        except IntegrityError:
                            # UNIQUE 제약 위반 (이미 존재)
                            session.rollback()
                            # DB에 존재하므로 existing_sections에 추가
                            existing_sections.add(section_type)
                            if verbose:
                                logger.debug(f"판례 본문 중복 (이미 존재): {section_type}")
                        except Exception as e:
                            logger.warning(f"판례 본문 저장 실패: {e}")
                            session.rollback()
                elif prec_data:
                    logger.debug(f"판례 데이터가 dict가 아님: {type(prec_data)}")
                
                session.commit()
                if verbose:
                    logger.info(f"판례 '{precedent.get('사건명')}' 수집 완료: {content_count}개 섹션")
                
            except Exception as e:
                logger.error(f"판례 본문 수집 실패 (ID: {precedent_id}): {e}")
                session.rollback()
        
        finally:
            session.close()
        
        return content_count
    
    def _save_precedent_metadata(
        self,
        session,
        precedent: Dict[str, Any],
        domain: str
    ) -> int:
        """판례 메타데이터 저장"""
        precedent_id = precedent.get('판례일련번호') or precedent.get('prec id')
        
        # 날짜 변환
        decision_date = self._parse_date(precedent.get('선고일자'))
        
        # 기존 레코드 확인
        result = session.execute(
            text("SELECT id FROM precedents WHERE precedent_id = :precedent_id"),
            {"precedent_id": precedent_id}
        )
        existing = result.fetchone()
        
        if existing:
            # 업데이트
            session.execute(
                text("""
                    UPDATE precedents SET
                        case_name = :case_name,
                        case_number = :case_number,
                        decision_date = :decision_date,
                        court_name = :court_name,
                        court_type_code = :court_type_code,
                        case_type_name = :case_type_name,
                        case_type_code = :case_type_code,
                        decision_type = :decision_type,
                        decision_result = :decision_result,
                        domain = :domain,
                        raw_response_json = :raw_response_json,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE precedent_id = :precedent_id
                """),
                {
                    "precedent_id": precedent_id,
                    "case_name": precedent.get('사건명', ''),
                    "case_number": precedent.get('사건번호'),
                    "decision_date": decision_date,
                    "court_name": precedent.get('법원명'),
                    "court_type_code": precedent.get('법원종류코드'),
                    "case_type_name": precedent.get('사건종류명'),
                    "case_type_code": precedent.get('사건종류코드'),
                    "decision_type": precedent.get('판결유형'),
                    "decision_result": precedent.get('선고'),
                    "domain": domain,
                    "raw_response_json": json.dumps(precedent, ensure_ascii=False)
                }
            )
            return existing[0]
        else:
            # 삽입
            result = session.execute(
                text("""
                    INSERT INTO precedents (
                        precedent_id, case_name, case_number, decision_date,
                        court_name, court_type_code, case_type_name, case_type_code,
                        decision_type, decision_result, domain, raw_response_json
                    ) VALUES (
                        :precedent_id, :case_name, :case_number, :decision_date,
                        :court_name, :court_type_code, :case_type_name, :case_type_code,
                        :decision_type, :decision_result, :domain, :raw_response_json
                    ) RETURNING id
                """),
                {
                    "precedent_id": precedent_id,
                    "case_name": precedent.get('사건명', ''),
                    "case_number": precedent.get('사건번호'),
                    "decision_date": decision_date,
                    "court_name": precedent.get('법원명'),
                    "court_type_code": precedent.get('법원종류코드'),
                    "case_type_name": precedent.get('사건종류명'),
                    "case_type_code": precedent.get('사건종류코드'),
                    "decision_type": precedent.get('판결유형'),
                    "decision_result": precedent.get('선고'),
                    "domain": domain,
                    "raw_response_json": json.dumps(precedent, ensure_ascii=False)
                }
            )
            return result.fetchone()[0]
    
    def _extract_sections(self, prec_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """판례 데이터에서 섹션 추출"""
        sections = []
        
        # 판시사항
        if prec_data.get('판시사항'):
            sections.append({
                'section_type': '판시사항',
                'section_content': prec_data.get('판시사항', ''),
                'referenced_articles': prec_data.get('참조조문'),
                'referenced_precedents': prec_data.get('참조판례'),
                'raw_response_json': json.dumps(prec_data, ensure_ascii=False)
            })
        
        # 판결요지
        if prec_data.get('판결요지'):
            sections.append({
                'section_type': '판결요지',
                'section_content': prec_data.get('판결요지', ''),
                'referenced_articles': prec_data.get('참조조문'),
                'referenced_precedents': prec_data.get('참조판례'),
                'raw_response_json': json.dumps(prec_data, ensure_ascii=False)
            })
        
        # 판례내용
        if prec_data.get('판례내용'):
            sections.append({
                'section_type': '판례내용',
                'section_content': prec_data.get('판례내용', ''),
                'referenced_articles': prec_data.get('참조조문'),
                'referenced_precedents': prec_data.get('참조판례'),
                'raw_response_json': json.dumps(prec_data, ensure_ascii=False)
            })
        
        return sections
    
    def _save_precedent_content(
        self,
        session,
        precedent_id: int,
        section: Dict[str, Any]
    ) -> bool:
        """
        판례 본문 저장 (중복 방지)
        
        Returns:
            bool: 저장 성공 여부 (중복인 경우 False)
        """
        section_type = section.get('section_type', '')
        
        # 중복 확인: 이미 존재하는지 체크
        try:
            result = session.execute(
                text("""
                    SELECT 1 FROM precedent_contents
                    WHERE precedent_id = :precedent_id AND section_type = :section_type
                    LIMIT 1
                """),
                {
                    "precedent_id": precedent_id,
                    "section_type": section_type
                }
            )
            if result.fetchone():
                # 이미 존재함
                return False
        except Exception as e:
            logger.warning(f"중복 확인 실패: {e}")
            # 확인 실패 시 삽입 시도 (에러 발생 가능)
        
        # 중복이 없으면 삽입
        try:
            result = session.execute(
                text("""
                    INSERT INTO precedent_contents (
                        precedent_id, section_type, section_content,
                        referenced_articles, referenced_precedents, raw_response_json
                    ) VALUES (
                        :precedent_id, :section_type, :section_content,
                        :referenced_articles, :referenced_precedents, :raw_response_json
                    )
                """),
                {
                    "precedent_id": precedent_id,
                    "section_type": section_type,
                    "section_content": section.get('section_content', ''),
                    "referenced_articles": section.get('referenced_articles'),
                    "referenced_precedents": section.get('referenced_precedents'),
                    "raw_response_json": section.get('raw_response_json', '{}')
                }
            )
            # 삽입 성공
            return result.rowcount > 0
        except IntegrityError:
            # UNIQUE 제약 위반 (다른 프로세스에서 동시에 삽입한 경우)
            return False
        except Exception as e:
            logger.warning(f"판례 본문 저장 실패: {e}")
            return False
    
    def _parse_date(self, date_str: Optional[Any]) -> Optional[str]:
        """날짜 문자열 파싱 (YYYYMMDD -> YYYY-MM-DD)"""
        if not date_str:
            return None
        
        date_str = str(date_str)
        if len(date_str) == 8 and date_str.isdigit():
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        return None

