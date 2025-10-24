"""
법률 용어 수집 서비스

이 모듈은 법령정보개방포털 API를 통해 법률 용어를 수집하고
데이터베이스에 저장하는 기능을 제공합니다.
"""

import asyncio
import aiohttp
import json
import logging
import sqlite3
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from scripts.data_collection.law_open_api.legal_terms.legal_term_collection_config import LegalTermCollectionConfig as Config
from source.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

@dataclass
class LegalTermListItem:
    """법률 용어 목록 아이템"""
    법령용어ID: str
    법령용어명: str
    법령용어상세검색: str
    사전구분코드: str
    법령용어상세링크: str
    법령종류코드: int
    lstrm_id: int

@dataclass
class LegalTermDetail:
    """법률 용어 상세 정보"""
    법령용어일련번호: int
    법령용어명_한글: str
    법령용어명_한자: str
    법령용어코드: int
    법령용어코드명: str
    출처: str
    법령용어정의: str

@dataclass
class CollectionProgress:
    """수집 진행 상황"""
    current_page: int = 1
    total_pages: int = 0
    collected_count: int = 0
    failed_count: int = 0
    last_collected_time: Optional[datetime] = None
    resume_page: Optional[int] = None

class LegalTermCollector:
    """법률 용어 수집기"""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = "http://www.law.go.kr/DRF"
        self.list_url = f"{self.base_url}/lawSearch.do"
        self.detail_url = f"{self.base_url}/lawService.do"
        
        # API 설정
        self.oc_id = config.get("LEGAL_API_OC_ID", "test")
        self.display_count = config.get("LEGAL_API_DISPLAY_COUNT", 100)
        self.max_retries = config.get("LEGAL_API_MAX_RETRIES", 3)
        self.retry_delays = [180, 300, 600]  # 3분, 5분, 10분
        
        # 데이터 저장 경로 (설정에서 가져오기)
        file_storage = config.get("file_storage", {})
        self.raw_data_dir = Path(file_storage.get("raw_data_dir", "data/raw/law_open_api/legal_terms"))
        self.processed_data_dir = Path(file_storage.get("processed_data_dir", "data/processed/legal_terms"))
        self.db_path = Path(config.get("database", {}).get("db_path", "data/lawfirm.db"))
        
        # 디렉토리 생성
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 진행 상황 파일
        monitoring = config.get("monitoring", {})
        self.progress_file = Path(monitoring.get("progress_file", "data/legal_term_collection_progress.json"))
        
        # 세션 설정
        self.session: Optional[aiohttp.ClientSession] = None
        self.progress = CollectionProgress()
        
        # 데이터베이스 초기화
        self._init_database()
        
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 법률 용어 목록 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS legal_term_list (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        법령용어ID TEXT UNIQUE NOT NULL,
                        법령용어명 TEXT NOT NULL,
                        법령용어상세검색 TEXT,
                        사전구분코드 TEXT,
                        법령용어상세링크 TEXT,
                        법령종류코드 INTEGER,
                        lstrm_id INTEGER,
                        collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        processed BOOLEAN DEFAULT FALSE,
                        vectorized BOOLEAN DEFAULT FALSE
                    )
                """)
                
                # 법률 용어 상세 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS legal_term_details (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        법령용어일련번호 INTEGER UNIQUE NOT NULL,
                        법령용어명_한글 TEXT NOT NULL,
                        법령용어명_한자 TEXT,
                        법령용어코드 INTEGER,
                        법령용어코드명 TEXT,
                        출처 TEXT,
                        법령용어정의 TEXT,
                        collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        processed BOOLEAN DEFAULT FALSE
                    )
                """)
                
                # 수집 로그 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS collection_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        operation_type TEXT NOT NULL,
                        page_number INTEGER,
                        success BOOLEAN NOT NULL,
                        error_message TEXT,
                        collected_count INTEGER DEFAULT 0,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info("데이터베이스 초기화 완료")
                
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
            raise
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    def _load_progress(self) -> CollectionProgress:
        """진행 상황 로드"""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return CollectionProgress(**data)
        except Exception as e:
            logger.warning(f"진행 상황 로드 실패: {e}")
        
        return CollectionProgress()
    
    def _save_progress(self):
        """진행 상황 저장"""
        try:
            # datetime 객체를 문자열로 변환
            progress_dict = asdict(self.progress)
            if 'last_collected_time' in progress_dict and progress_dict['last_collected_time']:
                progress_dict['last_collected_time'] = progress_dict['last_collected_time'].isoformat()
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_dict, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"진행 상황 저장 실패: {e}")
    
    async def _make_request_with_retry(self, url: str, params: Dict[str, Any]) -> Optional[Any]:
        """재시도 로직이 포함된 API 요청 - GET 방식"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
            'Referer': 'https://www.law.go.kr/',
            'Origin': 'https://www.law.go.kr'
        }
        
        for attempt in range(self.max_retries):
            try:
                # URL 직접 구성 (이중 인코딩 방지)
                import urllib.parse
                
                # 수동으로 URL 구성하여 이중 인코딩 방지
                query_parts = []
                for key, value in params.items():
                    if key == 'query':
                        # query는 이미 인코딩되어 있으므로 그대로 사용
                        query_parts.append(f"{key}={value}")
                    else:
                        query_parts.append(f"{key}={urllib.parse.quote(str(value), safe='')}")
                
                query_string = "&".join(query_parts)
                full_url = f"{url}?{query_string}"
                
                # GET 요청으로 명시적 호출
                async with self.session.get(full_url, headers=headers) as response:
                    logger.debug(f"GET 요청: {url} with params: {params}")
                    logger.debug(f"응답 상태: {response.status}")
                    
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '')
                        logger.debug(f"응답 Content-Type: {content_type}")
                        
                        if 'json' in content_type.lower():
                            try:
                                data = await response.json()
                                return data
                            except Exception as e:
                                logger.warning(f"JSON 파싱 실패: {e}")
                                return None
                        elif 'xml' in content_type.lower():
                            # XML 응답 처리
                            text = await response.text()
                            if "error500" in text or "페이지 접속에 실패" in text:
                                logger.error("API 오류 페이지 응답")
                                return None
                            return text  # XML 문자열 반환
                        else:
                            text = await response.text()
                            if "error500" in text or "페이지 접속에 실패" in text:
                                logger.error("API 오류 페이지 응답")
                                return None
                            return None
                    else:
                        logger.warning(f"API 응답 오류: {response.status}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"API 요청 타임아웃 (시도 {attempt + 1}/{self.max_retries})")
            except Exception as e:
                logger.warning(f"API 요청 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
            
            # 마지막 시도가 아니면 대기
            if attempt < self.max_retries - 1:
                delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                await asyncio.sleep(delay)
        
        logger.error(f"API 요청 최종 실패: {url}")
        return None
    
    async def get_term_list(self, page: int = 1, query: str = "", gana: str = "") -> Optional[Dict]:
        """법률 용어 목록 조회"""
        params = {
            "OC": self.oc_id,
            "target": "lstrm",
            "type": "JSON",
            "display": self.display_count,
            "page": page,
            "sort": "lasc"
        }
        
        if query:
            params["query"] = query
        if gana:
            params["gana"] = gana
        
        logger.info(f"API 요청 파라미터: {params}")
        return await self._make_request_with_retry(self.list_url, params)
    
    async def get_term_detail(self, term_name: str, detail_link: str = None) -> Optional[Any]:
        """법률 용어 상세 조회 - API 가이드 준수"""
        import urllib.parse
        
        # API 가이드에 따른 파라미터 설정 - 실제 용어명 사용
        params = {
            "OC": "schema9",  # 가이드에 명시된 OC 값 사용
            "target": "lstrm",
            "type": "JSON",  # JSON 형태로 요청
            "query": urllib.parse.quote(term_name, encoding='utf-8')  # 실제 용어명 사용
        }
        
        logger.debug(f"상세 조회 요청 (실제 용어명): {term_name}")
        logger.debug(f"URL 인코딩된 query: {params['query']}")
        
        response = await self._make_request_with_retry(self.detail_url, params)
        
        # 응답 파싱하여 LegalTermDetail 객체 반환
        if response is not None:
            return self._parse_term_detail_response(response, term_name)
        return None
    
    def _parse_term_detail_response(self, response: Any, term_name: str) -> Optional[LegalTermDetail]:
        """법률 용어 상세 응답 파싱 - API 가이드 준수"""
        try:
            # JSON 응답 처리 (우선)
            if isinstance(response, dict):
                # "일치하는 법령용어가 없습니다" 메시지 처리
                if "Law" in response and "일치하는 법령용어가 없습니다" in str(response["Law"]):
                    logger.warning(f"일치하는 법령용어가 없음: {term_name}")
                    return None
                
                # API 가이드에 따른 직접 필드 접근
                if any(key in response for key in ["법령용어일련번호", "법령용어명_한글"]):
                    detail_item = LegalTermDetail(
                        법령용어일련번호=int(response.get("법령용어일련번호", 0)),
                        법령용어명_한글=response.get("법령용어명_한글", ""),
                        법령용어명_한자=response.get("법령용어명_한자", ""),
                        법령용어코드=int(response.get("법령용어코드", 0)),
                        법령용어코드명=response.get("법령용어코드명", ""),
                        출처=response.get("출처", ""),
                        법령용어정의=response.get("법령용어정의", "")
                    )
                    logger.debug(f"가이드 방식 파싱 성공: {term_name}")
                    return detail_item
                
                # 중첩된 구조에서 데이터 찾기 (fallback)
                detail_data = None
                for key in ["LsTrm", "lstrm", "result", "LsTrmService"]:
                    if key in response:
                        detail_data = response[key]
                        break
                
                if detail_data:
                    # 배열인 경우 첫 번째 요소 사용
                    if isinstance(detail_data, list) and len(detail_data) > 0:
                        item = detail_data[0]
                    elif isinstance(detail_data, dict):
                        item = detail_data
                    else:
                        return None
                    
                    # 안전한 변환 함수들
                    def safe_int(value, default=0):
                        if isinstance(value, list) and len(value) > 0:
                            value = value[0]
                        try:
                            return int(value) if value else default
                        except (ValueError, TypeError):
                            return default
                    
                    def safe_str(value, default=""):
                        if isinstance(value, list) and len(value) > 0:
                            value = value[0]
                        return str(value) if value else default
                    
                    detail_item = LegalTermDetail(
                        법령용어일련번호=safe_int(item.get("법령용어일련번호", item.get("법령용어 일련번호", 0))),
                        법령용어명_한글=safe_str(item.get("법령용어명_한글", "")),
                        법령용어명_한자=safe_str(item.get("법령용어명_한자", "")),
                        법령용어코드=safe_int(item.get("법령용어코드", 0)),
                        법령용어코드명=safe_str(item.get("법령용어코드명", "")),
                        출처=safe_str(item.get("출처", "")),
                        법령용어정의=safe_str(item.get("법령용어정의", ""))
                    )
                    
                    logger.debug(f"중첩 구조 파싱 성공: {term_name}")
                    return detail_item
                
                logger.warning(f"상세 정보 파싱 실패 - 데이터 없음: {term_name}")
                logger.debug(f"응답 구조: {list(response.keys()) if isinstance(response, dict) else type(response)}")
                return None
            
            # XML 응답 처리 (fallback)
            elif isinstance(response, str) and response.startswith('<?xml'):
                return self._parse_xml_response(response, term_name)
            
            else:
                logger.warning(f"예상치 못한 응답 타입: {type(response)}")
                return None
            
        except Exception as e:
            logger.error(f"상세 응답 파싱 오류: {e}")
            return None
    
    def _parse_xml_response(self, xml_content: str, term_name: str) -> Optional[LegalTermDetail]:
        """XML 응답 파싱 - 실제 구조에 맞게 수정"""
        try:
            import xml.etree.ElementTree as ET
            
            # XML 파싱
            root = ET.fromstring(xml_content)
            
            # 실제 XML 구조: 루트에서 직접 필드들이 나옴
            # LsTrmService 래퍼가 없는 경우도 처리
            detail_element = None
            
            # LsTrmService 요소가 있는지 확인
            service_element = root.find('LsTrmService')
            if service_element is not None:
                detail_element = service_element
                logger.debug(f"LsTrmService 요소 발견: {term_name}")
            else:
                # LsTrmService가 없으면 루트 자체가 데이터
                detail_element = root
                logger.debug(f"루트 요소를 데이터로 사용: {term_name}")
            
            # CDATA를 포함한 텍스트 추출 함수
            def get_text_with_cdata(element, field_name):
                elem = element.find(field_name)
                if elem is not None:
                    # CDATA 섹션이 있는 경우와 없는 경우 모두 처리
                    if elem.text:
                        return elem.text.strip()
                    
                    # CDATA 섹션 처리 - 자식 요소가 CDATA인 경우
                    if len(elem) > 0:
                        for child in elem:
                            if child.text:
                                return child.text.strip()
                    
                    # 전체 텍스트 내용 확인 (CDATA 포함)
                    full_text = ET.tostring(elem, encoding='unicode', method='text')
                    if full_text:
                        return full_text.strip()
                return ""
            
            def get_int_with_cdata(element, field_name, default=0):
                text = get_text_with_cdata(element, field_name)
                try:
                    return int(text) if text else default
                except ValueError:
                    return default
            
            # 필드 추출
            법령용어일련번호 = get_int_with_cdata(detail_element, '법령용어일련번호')
            법령용어명_한글 = get_text_with_cdata(detail_element, '법령용어명_한글')
            법령용어명_한자 = get_text_with_cdata(detail_element, '법령용어명_한자')
            법령용어코드 = get_int_with_cdata(detail_element, '법령용어코드')
            법령용어코드명 = get_text_with_cdata(detail_element, '법령용어코드명')
            출처 = get_text_with_cdata(detail_element, '출처')
            법령용어정의 = get_text_with_cdata(detail_element, '법령용어정의')
            
            # 필수 필드 확인 - 법령용어일련번호 또는 법령용어명_한글 중 하나라도 있으면 성공
            if not 법령용어일련번호 and not 법령용어명_한글:
                logger.warning(f"XML에서 필수 필드를 찾을 수 없음: {term_name}")
                logger.debug(f"XML 구조: {[child.tag for child in detail_element]}")
                logger.debug(f"XML 내용: {xml_content[:200]}...")
                return None
            
            # LegalTermDetail 객체 생성
            detail = LegalTermDetail(
                법령용어일련번호=법령용어일련번호,
                법령용어명_한글=법령용어명_한글,
                법령용어명_한자=법령용어명_한자,
                법령용어코드=법령용어코드,
                법령용어코드명=법령용어코드명,
                출처=출처,
                법령용어정의=법령용어정의
            )
            
            logger.debug(f"XML 파싱 성공: {term_name}")
            logger.debug(f"추출된 데이터: 일련번호={법령용어일련번호}, 한글명={법령용어명_한글}")
            return detail
            
        except ET.ParseError as e:
            logger.error(f"XML 파싱 오류: {e}")
            logger.debug(f"XML 내용: {xml_content[:200]}...")
            return None
        except Exception as e:
            logger.error(f"XML 응답 처리 오류: {e}")
            logger.debug(f"XML 내용: {xml_content[:200]}...")
            return None
    
    def _parse_term_list_response(self, response: Dict) -> Tuple[List[LegalTermListItem], int]:
        """법률 용어 목록 응답 파싱"""
        try:
            items = []
            total_pages = 0
            
            logger.info(f"API 응답 구조: {list(response.keys())}")
            logger.info(f"전체 응답: {response}")
            
            # 다양한 응답 구조 처리
            term_data = None
            if "LsTrmSearch" in response and "lstrm" in response["LsTrmSearch"]:
                term_data = response["LsTrmSearch"]["lstrm"]
                total_count = int(response["LsTrmSearch"].get("totalCnt", 0))
            elif "lstrm" in response:
                term_data = response["lstrm"]
                total_count = int(response.get("totalCnt", 0))
            elif "result" in response:
                term_data = response["result"]
                total_count = int(response.get("totalCnt", 0))
            elif isinstance(response, list):
                term_data = response
                total_count = len(response)
            
            if term_data is not None:
                # 총 페이지 수 계산
                total_pages = (total_count + self.display_count - 1) // self.display_count
                
                # 용어 목록 파싱
                if isinstance(term_data, list):
                    for item in term_data:
                        if isinstance(item, dict):
                            # 법령종류코드 처리 (쉼표로 구분된 경우 첫 번째 값 사용)
                            law_type_code = item.get("법령종류코드", item.get("type", "0"))
                            if isinstance(law_type_code, str) and "," in law_type_code:
                                law_type_code = law_type_code.split(",")[0]
                            
                            term_item = LegalTermListItem(
                                법령용어ID=item.get("법령용어ID", item.get("id", "")),
                                법령용어명=item.get("법령용어명", item.get("name", "")),
                                법령용어상세검색=item.get("법령용어상세검색", item.get("detail", "")),
                                사전구분코드=item.get("사전구분코드", item.get("code", "")),
                                법령용어상세링크=item.get("법령용어상세링크", item.get("link", "")),
                                법령종류코드=int(law_type_code),
                                lstrm_id=int(item.get("id", item.get("lstrm_id", 0)))
                            )
                            items.append(term_item)
                elif isinstance(term_data, dict):
                    # 단일 아이템인 경우
                    term_item = LegalTermListItem(
                        법령용어ID=term_data.get("법령용어ID", term_data.get("id", "")),
                        법령용어명=term_data.get("법령용어명", term_data.get("name", "")),
                        법령용어상세검색=term_data.get("법령용어상세검색", term_data.get("detail", "")),
                        사전구분코드=term_data.get("사전구분코드", term_data.get("code", "")),
                        법령용어상세링크=term_data.get("법령용어상세링크", term_data.get("link", "")),
                        법령종류코드=int(term_data.get("법령종류코드", term_data.get("type", 0))),
                        lstrm_id=int(term_data.get("lstrm id", term_data.get("lstrm_id", 0)))
                    )
                    items.append(term_item)
            
            logger.info(f"파싱된 항목 수: {len(items)}")
            return items, total_pages
            
        except Exception as e:
            logger.error(f"법률 용어 목록 파싱 실패: {e}")
            logger.error(f"응답 데이터: {response}")
            return [], 0
    
    def _save_to_file(self, data: Any, filename: str, batch_number: int, page_info: str = ""):
        """데이터를 파일로 저장"""
        try:
            # 빈 배열이나 유효하지 않은 데이터 체크
            if isinstance(data, list) and len(data) == 0:
                logger.warning("빈 배열은 저장하지 않습니다.")
                return
            
            # 데이터 유효성 검사
            if isinstance(data, list):
                # 리스트 내 모든 항목이 유효한지 확인
                valid_items = []
                for item in data:
                    if isinstance(item, dict):
                        # "일치하는 법령용어가 없습니다" 메시지가 포함된 경우 제외
                        if "Law" in item and "일치하는 법령용어가 없습니다" in str(item["Law"]):
                            logger.warning("유효하지 않은 응답 데이터는 저장하지 않습니다.")
                            continue
                        valid_items.append(item)
                    elif hasattr(item, '법령용어일련번호') and hasattr(item, '법령용어명_한글'):
                        # LegalTermDetail 객체인 경우 - 딕셔너리로 변환
                        from dataclasses import asdict
                        item_dict = asdict(item)
                        valid_items.append(item_dict)
                    else:
                        logger.warning(f"유효하지 않은 데이터 타입: {type(item)}")
                        continue
                
                if len(valid_items) == 0:
                    logger.warning("유효한 데이터가 없어 저장하지 않습니다.")
                    return
                
                data = valid_items
            
            # 디렉토리 생성
            self.raw_data_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if page_info:
                file_path = self.raw_data_dir / f"{filename}_batch_{batch_number:04d}_{page_info}_{timestamp}.json"
            else:
                file_path = self.raw_data_dir / f"{filename}_batch_{batch_number:04d}_{timestamp}.json"
            
            logger.info(f"파일 저장 시작: {file_path}")
            logger.info(f"저장할 데이터 개수: {len(data) if isinstance(data, list) else 1}")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"데이터 파일 저장 완료: {file_path}")
            logger.info(f"파일 크기: {file_path.stat().st_size} bytes")
            
        except Exception as e:
            logger.error(f"데이터 파일 저장 실패: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
    
    async def save_term_detail_response(self, term_name: str, response: Dict, term_id: str = None) -> str:
        """법률용어 상세 응답을 파일로 저장"""
        try:
            # 날짜별 폴더 생성
            date_folder = datetime.now().strftime("%Y-%m-%d")
            detail_dir = Path(self.config.get("file_storage", {}).get("detail_responses_dir", "data/raw/law_open_api/legal_terms/responses/term_details"))
            date_path = detail_dir / date_folder
            date_path.mkdir(parents=True, exist_ok=True)
            
            # 파일명 생성
            if term_id:
                filename = f"term_{term_id}_detail.json"
            else:
                # 용어명에서 특수문자 제거
                safe_name = "".join(c for c in term_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_name = safe_name.replace(' ', '_')
                filename = f"term_{safe_name}_detail.json"
            
            file_path = date_path / filename
            
            # 저장할 데이터 구조
            save_data = {
                "request_info": {
                    "term_name": term_name,
                    "term_id": term_id,
                    "timestamp": datetime.now().isoformat(),
                    "api_url": self.detail_url
                },
                "response_data": response,
                "collection_metadata": {
                    "collected_at": datetime.now().isoformat(),
                    "processing_status": "raw"
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"상세 응답 파일 저장 완료: {file_path}")
            return str(file_path)
                
        except Exception as e:
            logger.error(f"상세 응답 파일 저장 실패: {e}")
            return ""
    
    async def save_term_details_batch(self, details: List[LegalTermDetail], batch_number: int) -> str:
        """법률용어 상세 정보 배치를 파일로 저장"""
        try:
            batch_dir = Path(self.config.get("file_storage", {}).get("detail_batches_dir", "data/raw/law_open_api/legal_terms/batches"))
            batch_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"legal_term_details_batch_{batch_number:04d}_{timestamp}.json"
            file_path = batch_dir / filename
            
            # 배치 데이터 구조
            batch_data = {
                "batch_info": {
                    "batch_id": f"legal_term_details_batch_{batch_number:04d}",
                    "collection_date": datetime.now().strftime("%Y-%m-%d"),
                    "batch_size": len(details),
                    "start_time": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat()
                },
                "terms": [asdict(detail) for detail in details],
                "statistics": {
                    "total_terms": len(details),
                    "successful_collections": len(details),
                    "failed_collections": 0
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"상세 배치 파일 저장 완료: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"상세 배치 파일 저장 실패: {e}")
            return ""
    
    
    async def collect_term_list(self, start_page: int = 1, end_page: Optional[int] = None, 
                              batch_size: int = 10, query: str = "", gana: str = ""):
        """법률 용어 목록 수집"""
        logger.info(f"법률 용어 목록 수집 시작: 페이지 {start_page}부터")
        
        # 진행 상황 로드
        self.progress = self._load_progress()
        if self.progress.resume_page:
            start_page = self.progress.resume_page
            logger.info(f"이전 진행 상황에서 재개: 페이지 {start_page}")
        
        current_page = start_page
        batch_number = 1
        batch_data = []
        
        try:
            while True:
                if end_page and current_page > end_page:
                    break
                
                logger.info(f"페이지 {current_page} 수집 중...")
                
                # API 요청
                response = await self.get_term_list(current_page, query, gana)
                
                if response is None:
                    logger.error(f"페이지 {current_page} 수집 실패")
                    self._log_collection_attempt("term_list", current_page, False, "API 요청 실패")
                    self.progress.failed_count += 1
                    current_page += 1
                    continue
                
                # 응답 파싱
                items, total_pages = self._parse_term_list_response(response)
                
                if not items:
                    logger.warning(f"페이지 {current_page}에서 데이터 없음")
                    current_page += 1
                    continue
                
                # 배치 데이터에 추가
                batch_data.extend([asdict(item) for item in items])
                
                # 진행 상황 업데이트
                self.progress.current_page = current_page
                self.progress.total_pages = total_pages
                self.progress.collected_count += len(items)
                self.progress.last_collected_time = datetime.now()
                
                logger.info(f"페이지 {current_page} 수집 완료: {len(items)}개")
                
                # 진행 상황을 더 자주 저장 (데이터 유실 방지)
                save_interval = self.config.get("collection", {}).get("progress_save_interval", 5)
                if current_page % save_interval == 0:  # 설정된 간격마다 진행 상황 저장
                    self._save_progress()
                    logger.info(f"진행 상황 저장: 페이지 {current_page}")
                
                # 배치 크기에 도달하면 파일 저장 (페이지 단위로)
                if len(batch_data) >= batch_size * self.display_count:
                    page_info = f"page{current_page}"
                    self._save_to_file(batch_data, "legal_term_list", batch_number, page_info)
                    batch_data = []
                    batch_number += 1
                    logger.info(f"배치 파일 저장 완료: 배치 {batch_number-1}")
                
                # 또는 매 페이지마다 저장 (누락 방지)
                if self.config.get("collection", {}).get("save_every_page", False):
                    page_info = f"page{current_page}"
                    self._save_to_file([asdict(item) for item in items], "legal_term_list", batch_number, page_info)
                    batch_number += 1
                
                current_page += 1
                
                # API 부하 방지를 위한 대기
                await asyncio.sleep(self.config.get("api", {}).get("rate_limit_delay", 1.0))
            
            # 남은 배치 데이터 저장
            if batch_data:
                page_info = f"page{current_page-1}"
                self._save_to_file(batch_data, "legal_term_list", batch_number, page_info)
            
            # 진행 상황 저장
            self._save_progress()
            
            logger.info(f"법률 용어 목록 수집 완료: 총 {self.progress.collected_count}개")
            
        except Exception as e:
            logger.error(f"법률 용어 목록 수집 중 오류: {e}")
            self.progress.resume_page = current_page
            self._save_progress()  # 오류 발생 시에도 진행 상황 저장
            logger.info(f"오류 발생으로 인한 진행 상황 저장: 페이지 {current_page}")
            raise
    
    async def collect_term_details(self, batch_size: int = 50):
        """법률 용어 상세 수집 (마지막 파일만)"""
        logger.info("법률 용어 상세 수집 시작")
        
        try:
            # 목록 파일에서 용어들 읽기
            list_files = list(self.raw_data_dir.glob("legal_term_list_batch_*.json"))
            if not list_files:
                logger.warning("목록 파일이 없습니다. 먼저 목록을 수집해주세요.")
                return
            
            # 가장 최근 파일 사용
            latest_file = max(list_files, key=lambda f: f.stat().st_mtime)
            logger.info(f"목록 파일 사용: {latest_file.name}")
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                list_data = json.load(f)
            
            # 용어명과 상세링크 추출
            terms = []
            for item in list_data:
                if isinstance(item, dict) and '법령용어명' in item and '법령용어상세링크' in item:
                    terms.append((item['법령용어명'], item['법령용어상세링크']))
            
                total_terms = len(terms)
                logger.info(f"수집할 용어 수: {total_terms}개")
                
                batch_data = []
                batch_number = 1
                collected_count = 0
                
            for i, (term_name, detail_link) in enumerate(terms):
                if (i + 1) % 50 == 0:  # 50개마다 진행 상황 로그
                    logger.info(f"진행 상황: {i+1}/{total_terms} ({((i+1)/total_terms)*100:.1f}%)")
                    
                # API 요청 (상세링크 사용)
                response = await self.get_term_detail(term_name, detail_link)
                
                if response is None:
                    continue
                
                # 응답 파싱
                detail = self._parse_term_detail_response(response, term_name)
                
                if detail is None:
                    continue
                
                collected_count += 1
                batch_data.append(asdict(detail))
                
                # 배치 크기에 도달하면 파일 저장
                batch_size = self.config.get("collection", {}).get("detail_batch_size", 20)
                if len(batch_data) >= batch_size:
                    page_info = f"details_batch{batch_number}"
                    self._save_to_file(batch_data, "legal_term_details", batch_number, page_info)
                    batch_data = []
                    batch_number += 1
                    logger.info(f"배치 저장 완료: {i+1}/{total_terms}")
                
                # API 부하 방지를 위한 대기
                await asyncio.sleep(self.config.get("collection", {}).get("detail_collection_delay", 1.0))
                
            # 남은 배치 데이터 저장
            if batch_data:
                page_info = f"details_final"
                self._save_to_file(batch_data, "legal_term_details", batch_number, page_info)
                
                logger.info(f"법률 용어 상세 수집 완료: 총 {collected_count}개")
                
        except Exception as e:
            logger.error(f"법률 용어 상세 수집 중 오류: {e}")
            raise
    
    async def collect_all_term_details(self, batch_size: int = 10, max_files: int = None):
        """모든 목록 파일에서 법률 용어 상세 수집"""
        logger.info("전체 목록 파일 대상 법률 용어 상세 수집 시작")
        
        try:
            # 목록 파일들 읽기
            list_files = list(self.raw_data_dir.glob("legal_term_list_batch_*.json"))
            if not list_files:
                logger.warning("목록 파일이 없습니다. 먼저 목록을 수집해주세요.")
                return
            
            # 파일 개수 제한 (테스트용)
            if max_files:
                list_files = list_files[:max_files]
            
            logger.info(f"처리할 목록 파일 수: {len(list_files)}개")
            
            # 모든 용어 수집
            all_terms = []
            processed_files = 0
            
            for file_path in list_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        list_data = json.load(f)
                    
                    # 용어명과 상세링크 추출
                    file_terms = []
                    for item in list_data:
                        if isinstance(item, dict) and '법령용어명' in item and '법령용어상세링크' in item:
                            file_terms.append((item['법령용어명'], item['법령용어상세링크']))
                    
                    all_terms.extend(file_terms)
                    processed_files += 1
                    
                    if processed_files % 50 == 0:
                        logger.info(f"파일 처리 진행: {processed_files}/{len(list_files)} (용어 수: {len(all_terms)})")
                        
                except Exception as e:
                    logger.error(f"파일 처리 오류 {file_path}: {e}")
                    continue
            
            total_terms = len(all_terms)
            logger.info(f"총 수집할 용어 수: {total_terms}개")
            
            # 중복 제거 (같은 용어명이 여러 파일에 있을 수 있음)
            unique_terms = list(set(all_terms))
            unique_count = len(unique_terms)
            logger.info(f"중복 제거 후 용어 수: {unique_count}개")
            
            batch_data = []
            batch_number = 1
            collected_count = 0
            failed_count = 0
            
            for i, (term_name, detail_link) in enumerate(unique_terms):
                if (i + 1) % 100 == 0:  # 100개마다 진행 상황 로그
                    logger.info(f"진행 상황: {i+1}/{unique_count} ({((i+1)/unique_count)*100:.1f}%) - 수집: {collected_count}, 실패: {failed_count}")
                
                try:
                    # API 요청 (상세링크 사용)
                    response = await self.get_term_detail(term_name, detail_link)
                    
                    if response is None:
                        failed_count += 1
                        continue
                    
                    # 응답 파싱
                    detail = self._parse_term_detail_response(response, term_name)
                    
                    if detail is None:
                        failed_count += 1
                        continue
                    
                    collected_count += 1
                    batch_data.append(asdict(detail))
                    
                    # 배치 크기에 도달하면 파일 저장
                    if len(batch_data) >= batch_size:
                        page_info = f"all_details_batch{batch_number}"
                        self._save_to_file(batch_data, "legal_term_details", batch_number, page_info)
                        batch_data = []
                        batch_number += 1
                        logger.info(f"배치 저장 완료: {i+1}/{unique_count}")
                    
                    # API 부하 방지를 위한 대기
                    await asyncio.sleep(self.config.get("collection", {}).get("detail_collection_delay", 1.0))
                    
                except Exception as e:
                    logger.error(f"용어 상세 수집 오류 '{term_name}': {e}")
                    failed_count += 1
                    continue
            
            # 남은 배치 데이터 저장
            if batch_data:
                page_info = f"all_details_final"
                self._save_to_file(batch_data, "legal_term_details", batch_number, page_info)
            
            logger.info(f"전체 법률 용어 상세 수집 완료: 총 {collected_count}개 (실패: {failed_count}개)")
                
        except Exception as e:
            logger.error(f"전체 법률 용어 상세 수집 중 오류: {e}")
            raise
    
    async def collect_alternating(self, start_page: int = 1, end_page: Optional[int] = None,
                                list_batch_size: int = 50, detail_batch_size: int = 10,
                                query: str = "", gana: str = "") -> None:
        """목록 수집과 상세 수집을 번갈아가면서 진행"""
        logger.info("번갈아가면서 수집 시작")
        
        current_page = start_page
        total_list_collected = 0
        total_detail_collected = 0
        
        while True:
            if end_page and current_page > end_page:
                break
            
            # 1단계: 목록 수집 (1페이지)
            logger.info(f"=== 목록 수집: 페이지 {current_page} ===")
            
            try:
                response = await self.get_term_list(current_page, query=query, gana=gana)
                
                if not response:
                    logger.warning(f"페이지 {current_page} 응답 없음")
                    current_page += 1
                    continue
                
                items, total_count = self._parse_term_list_response(response)
                
                if not items:
                    logger.warning(f"페이지 {current_page}에서 항목 없음")
                    current_page += 1
                    continue
                
                # 목록 저장
                from dataclasses import asdict
                self._save_to_file([asdict(item) for item in items], "legal_term_list", current_page, f"page{current_page}")
                
                total_list_collected += len(items)
                logger.info(f"목록 수집 완료: {len(items)}개 항목 저장")
                
                # 2단계: 해당 페이지의 상세 정보 수집
                logger.info(f"=== 상세 수집: 페이지 {current_page} 용어들 ===")
                
                page_detail_count = 0
                page_detail_failed = 0
                
                for i, item in enumerate(items, 1):
                    if isinstance(item, dict) and '법령용어명' in item:
                        term_name = item['법령용어명']
                    elif hasattr(item, '법령용어명'):
                        term_name = item.법령용어명
                    else:
                        continue
                    
                    if term_name and len(term_name.strip()) > 0:
                        try:
                            logger.info(f"상세 수집: {i}/{len(items)} - {term_name}")
                            
                            detail = await self.get_term_detail(term_name)
                            
                            if detail:
                                # LegalTermDetail 객체인지 확인하고 유효한 데이터인지 검증
                                if hasattr(detail, '법령용어일련번호') and hasattr(detail, '법령용어명_한글'):
                                    # 유효한 데이터인 경우에만 저장 - 딕셔너리로 변환
                                    from dataclasses import asdict
                                    detail_dict = asdict(detail)
                                    safe_term_name = "".join(c for c in term_name if c.isalnum() or c in (' ', '-', '_')).strip()
                                    safe_term_name = safe_term_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                                    self._save_to_file([detail_dict], "legal_term_detail", i, f"page{current_page}_{safe_term_name}")
                                    
                                    page_detail_count += 1
                                    total_detail_collected += 1
                                    logger.info(f"상세 정보 수집 성공: {term_name}")
                                else:
                                    page_detail_failed += 1
                                    logger.warning(f"상세 정보 수집 실패 (유효하지 않은 데이터): {term_name}")
                            else:
                                page_detail_failed += 1
                                logger.warning(f"상세 정보 수집 실패 (용어 없음): {term_name}")
                            
                            # API 부하 방지
                            await asyncio.sleep(self.config.get("collection", {}).get("detail_collection_delay", 1.0))
                            
                        except Exception as e:
                            logger.error(f"상세 수집 오류 {term_name}: {e}")
                            page_detail_failed += 1
                
                logger.info(f"페이지 {current_page} 상세 수집 완료: 성공 {page_detail_count}개, 실패 {page_detail_failed}개")
                
                # 진행 상황 저장
                progress_data = {
                    "last_page": current_page,
                    "total_list_collected": total_list_collected,
                    "total_detail_collected": total_detail_collected,
                    "status": "alternating_collection"
                }
                self.progress.current_page = current_page
                self.progress.collected_count = total_list_collected
                self._save_progress()
                
                current_page += 1
                
                # 페이지 간 대기
                await asyncio.sleep(self.config.get("collection", {}).get("detail_collection_delay", 1.0) * 2)
                
            except Exception as e:
                logger.error(f"페이지 {current_page} 처리 중 오류: {e}")
                current_page += 1
                continue
        
        logger.info(f"번갈아가면서 수집 완료: 목록 {total_list_collected}개, 상세 {total_detail_collected}개")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """수집 통계 조회"""
        try:
            # 목록 파일 통계
            list_files = list(self.raw_data_dir.glob("legal_term_list_batch_*.json"))
            list_count = 0
            for file_path in list_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    list_count += len(data) if isinstance(data, list) else 1
            
            # 상세 파일 통계
            detail_files = list(self.raw_data_dir.glob("legal_term_details_batch_*.json"))
            detail_count = 0
            for file_path in detail_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    detail_count += len(data) if isinstance(data, list) else 1
            
            return {
                "total_terms": list_count,
                "collected_details": detail_count,
                "list_files": len(list_files),
                "detail_files": len(detail_files),
                "progress": asdict(self.progress)
            }
                
        except Exception as e:
            logger.error(f"수집 통계 조회 실패: {e}")
            return {}
    
    def reset_collection_progress(self):
        """수집 진행 상황 초기화"""
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
            
            self.progress = CollectionProgress()
            logger.info("수집 진행 상황 초기화 완료")
            
        except Exception as e:
            logger.error(f"수집 진행 상황 초기화 실패: {e}")


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='법률용어 수집기')
    
    parser.add_argument('--start-page', type=int, default=1, 
                       help='시작 페이지 번호 (기본값: 1)')
    parser.add_argument('--end-page', type=int, default=None,
                       help='종료 페이지 번호 (기본값: None, 무제한)')
    parser.add_argument('--collect-details', action='store_true',
                       help='상세 정보 수집 여부')
    parser.add_argument('--collect-all-details', action='store_true',
                       help='모든 목록 파일에서 상세 정보 수집')
    parser.add_argument('--collect-alternating', action='store_true',
                       help='목록 수집과 상세 수집을 번갈아가면서 진행')
    parser.add_argument('--list-only', action='store_true',
                       help='목록만 수집 (상세 수집 제외)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='목록 수집 배치 크기 (기본값: 10)')
    parser.add_argument('--detail-batch-size', type=int, default=50,
                       help='상세 수집 배치 크기 (기본값: 50)')
    parser.add_argument('--query', type=str, default='',
                       help='검색 쿼리 (기본값: 빈 문자열)')
    parser.add_argument('--gana', type=str, default='',
                       help='가나다 검색 (기본값: 빈 문자열)')
    parser.add_argument('--display-count', type=int, default=100,
                       help='페이지당 표시 개수 (기본값: 100)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='최대 재시도 횟수 (기본값: 3)')
    parser.add_argument('--rate-limit-delay', type=float, default=1.0,
                       help='요청 간 대기 시간(초) (기본값: 1.0)')
    parser.add_argument('--detail-delay', type=float, default=1.0,
                       help='상세 조회 간 대기 시간(초) (기본값: 1.0)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='처리할 최대 파일 수 (테스트용)')
    parser.add_argument('--config-file', type=str, default=None,
                       help='설정 파일 경로 (기본값: 기본 설정 사용)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='상세 로그 출력')
    
    return parser.parse_args()

async def main():
    """메인 실행 함수"""
    args = parse_arguments()
    
    # 설정 로드
    if args.config_file:
        config = Config(args.config_file)
    else:
        config = Config()
    
    # 명령행 인수로 설정 오버라이드
    config_dict = config.get_all_config()
    config_dict['api']['rate_limit_delay'] = args.rate_limit_delay
    config_dict['collection']['detail_collection_delay'] = args.detail_delay
    config_dict['collection']['start_page'] = args.start_page
    config_dict['collection']['end_page'] = args.end_page
    config_dict['collection']['query'] = args.query
    config_dict['collection']['gana'] = args.gana
    config_dict['collection']['list_batch_size'] = args.batch_size
    config_dict['collection']['detail_batch_size'] = args.detail_batch_size
    config_dict['api']['display_count'] = args.display_count
    config_dict['api']['max_retries'] = args.max_retries
    
    # 로그 레벨 설정
    if args.verbose:
        config_dict['logging']['level'] = 'DEBUG'
    
    # 설정 업데이트
    config.update_config(config_dict)
    
    logger.info(f"=== 법률용어 수집 시작 ===")
    logger.info(f"시작 페이지: {args.start_page}")
    logger.info(f"종료 페이지: {args.end_page or '무제한'}")
    logger.info(f"상세 수집: {'예' if args.collect_details else '아니오'}")
    logger.info(f"전체 상세 수집: {'예' if args.collect_all_details else '아니오'}")
    logger.info(f"목록만 수집: {'예' if args.list_only else '아니오'}")
    logger.info(f"배치 크기: 목록 {args.batch_size}, 상세 {args.detail_batch_size}")
    logger.info(f"검색 쿼리: '{args.query}'")
    logger.info(f"가나다 검색: '{args.gana}'")
    
    async with LegalTermCollector(config) as collector:
        try:
            # 번갈아가면서 수집 우선 처리
            if args.collect_alternating and not args.list_only:
                logger.info("번갈아가면서 수집 시작")
                await collector.collect_alternating(
                    start_page=args.start_page,
                    end_page=args.end_page,
                    list_batch_size=args.batch_size,
                    detail_batch_size=args.detail_batch_size,
                    query=args.query,
                    gana=args.gana
                )
            # 상세 수집 우선 처리
            elif args.collect_all_details and not args.list_only:
                logger.info("전체 목록 파일 대상 상세 정보 수집 시작")
                await collector.collect_all_term_details(
                    batch_size=args.detail_batch_size, 
                    max_files=args.max_files
                )
            elif args.collect_details and not args.list_only:
                logger.info("상세 정보 수집 시작")
                await collector.collect_term_details(batch_size=args.detail_batch_size)
            elif args.list_only:
                logger.info("목록만 수집 모드 - 상세 수집 건너뜀")
            else:
                # 목록 수집
                logger.info(f"목록 수집 시작: 페이지 {args.start_page}~{args.end_page or '무제한'}")
            await collector.collect_term_list(
                    start_page=args.start_page,
                    end_page=args.end_page,
                    batch_size=args.batch_size,
                    query=args.query,
                    gana=args.gana
                )
            
            # 수집 통계 출력
            stats = collector.get_collection_stats()
            if stats:
                logger.info(f"=== 수집 완료 통계 ===")
                logger.info(f"총 수집된 용어: {stats.get('total_terms', 0)}개")
                logger.info(f"상세 수집된 용어: {stats.get('collected_details', 0)}개")
                logger.info(f"목록 파일 수: {stats.get('list_files', 0)}개")
                logger.info(f"상세 파일 수: {stats.get('detail_files', 0)}개")
                
                # 진행 상황 출력
                progress = stats.get('progress', {})
                logger.info(f"현재 페이지: {progress.get('current_page', 0)}")
                logger.info(f"수집된 개수: {progress.get('collected_count', 0)}")
                logger.info(f"실패한 개수: {progress.get('failed_count', 0)}")
            else:
                logger.warning("수집 통계를 가져올 수 없습니다.")
            
        except KeyboardInterrupt:
            logger.info("사용자에 의해 수집 중단됨")
            # 중단 시 현재까지의 통계 출력
            stats = collector.get_collection_stats()
            logger.info(f"중단 시점 통계: {stats}")
        except Exception as e:
            logger.error(f"수집 중 오류 발생: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
