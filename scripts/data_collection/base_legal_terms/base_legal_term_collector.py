"""
법령정보지식베이스 법령용어 수집기

이 모듈은 법령정보지식베이스 API를 통해 법령용어를 수집하고
base_legal_terms 폴더 구조에 저장하는 기능을 제공합니다.
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

# 설정 파일 import
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'base_legal_terms', 'config'))
# from base_legal_term_collection_config import BaseLegalTermCollectionConfig as Config

# 인라인 설정 클래스
class BaseLegalTermCollectionConfig:
    """Base Legal Term Collection 설정 클래스"""
    
    def __init__(self):
        self._config = self._get_default_config()
    
    def _get_default_config(self):
        """기본 설정 반환"""
        return {
            "api": {
                "base_url": "https://www.law.go.kr/DRF",
                "list_endpoint": "/lawSearch.do",
                "detail_endpoint": "/lawService.do",
                "oc_id": os.getenv("LAW_OPEN_API_OC", "test"),
                "target": "lstrmAI",
                "type": "JSON",
                "display_count": 100,
                "max_retries": 3,
                "retry_delays": [180, 300, 600],
                "request_timeout": 30,
                "rate_limit_delay": 1.0
            },
            "collection": {
                "start_page": 1,
                "end_page": None,
                "query": "",
                "homonym_yn": "Y",
                "list_batch_size": 20,
                "detail_batch_size": 50,
                "relation_batch_size": 30,
                "auto_resume": True,
                "save_raw_data": True,
                "save_batch_files": True,
                "progress_save_interval": 5,
                "detail_progress_save_interval": 10,
                "collect_details": True,
                "collect_relations": True,
                "detail_collection_delay": 1.0,
                "max_detail_retries": 3,
                "detail_timeout": 30,
                "skip_existing_details": True,
                "save_every_page": True
            },
            "file_storage": {
                "base_dir": "data",
                "raw_data_dir": "data/raw",
                "processed_data_dir": "data/processed",
                "embeddings_dir": "data/embeddings",
                "database_dir": "data/database",
                "logs_dir": "data/logs",
                "progress_dir": "data/progress",
                "reports_dir": "data/reports",
                "term_lists_dir": "data/raw/law_open_api/base_legal_terms",
                "term_details_dir": "data/raw/law_open_api/base_legal_terms",
                "term_relations_dir": "data/raw/law_open_api/base_legal_terms",
                "api_responses_dir": "data/raw/law_open_api/base_legal_terms",
                "cleaned_terms_dir": "data/processed/law_open_api/base_legal_terms",
                "normalized_terms_dir": "data/processed/law_open_api/base_legal_terms",
                "validated_terms_dir": "data/processed/law_open_api/base_legal_terms",
                "integrated_terms_dir": "data/processed/law_open_api/base_legal_terms",
                "list_batch_prefix": "term_list_batch_",
                "detail_batch_prefix": "term_detail_batch_",
                "relation_batch_prefix": "term_relation_batch_",
                "file_suffix": ".json",
                "save_raw_responses": True,
                "save_batch_files": True,
                "save_daily_folders": True,
                "file_format": "json",
                "max_file_size_mb": 50
            },
            "database": {
                "db_path": "data/lawfirm.db",  # 메인 데이터베이스로 변경
                "backup_enabled": True,
                "backup_interval": 1000,
                "connection_timeout": 30,
                "enable_wal_mode": True,
                "cache_size": 10000
            },
            "vector_store": {
                "model_name": os.getenv("SENTENCE_BERT_MODEL", "jhgan/ko-sroberta-multitask"),
                "vector_dimension": 768,
                "index_type": "flat",
                "similarity_metric": "l2",
                "embeddings_dir": "data/embeddings",
                "index_file": "base_legal_terms_index.faiss",
                "metadata_file": "base_legal_terms_metadata.json",
                "cache_dir": "data/embeddings/cache"
            },
            "processing": {
                "enable_cleaning": True,
                "enable_normalization": True,
                "enable_validation": True,
                "enable_relation_extraction": True,
                "quality_threshold": 0.8,
                "duplicate_threshold": 0.9,
                "min_term_length": 2,
                "max_term_length": 100,
                "enable_homonym_processing": True,
                "enable_relation_mapping": True
            },
            "logging": {
                "level": "INFO",
                "log_file": "data/logs/base_legal_terms_collection.log",
                "error_log_file": "data/logs/base_legal_terms_error.log",
                "max_file_size": "10MB",
                "backup_count": 5,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "enable_console_logging": True
            },
            "monitoring": {
                "enabled": True,
                "progress_update_interval": 10,
                "status_file": "data/progress/base_legal_terms_status.json",
                "progress_file": "data/progress/base_legal_terms_progress.json",
                "metrics_file": "data/progress/base_legal_terms_metrics.json",
                "quality_file": "data/progress/base_legal_terms_quality.json"
            },
            "performance": {
                "max_concurrent_requests": 5,
                "memory_limit_mb": 2048,
                "cache_size": 1000,
                "enable_compression": True,
                "chunk_size": 1024,
                "enable_async_processing": True,
                "batch_processing_size": 100
            }
        }
    
    def get_api_config(self):
        return self._config["api"]
    
    def get_collection_config(self):
        return self._config["collection"]
    
    def get_file_storage_config(self):
        return self._config["file_storage"]
    
    def get_database_config(self):
        return self._config["database"]
    
    def get_vector_store_config(self):
        return self._config["vector_store"]
    
    def get_processing_config(self):
        return self._config["processing"]
    
    def get_logging_config(self):
        return self._config["logging"]
    
    def get_monitoring_config(self):
        return self._config["monitoring"]
    
    def get_performance_config(self):
        return self._config["performance"]
    
    def get_all_config(self):
        return self._config
    
    def update_config(self, new_config):
        """설정 업데이트"""
        self._config.update(new_config)

Config = BaseLegalTermCollectionConfig

# 로거 설정
import logging

# 로거 초기화
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class BaseLegalTermListItem:
    """법령정보지식베이스 법령용어 목록 아이템"""
    법령용어ID: str
    법령용어명: str
    동음이의어존재여부: str
    비고: str
    용어간관계링크: str
    조문간관계링크: str
    수집일시: str = ""

@dataclass
class BaseLegalTermDetail:
    """법령정보지식베이스 법령용어 상세 정보"""
    법령용어일련번호: int
    법령용어명_한글: str
    법령용어명_한자: str
    법령용어코드: int
    법령용어코드명: str
    출처: str
    법령용어정의: str
    동음이의어내용: str = ""
    용어관계정보: List[Dict] = None
    조문관계정보: List[Dict] = None
    수집일시: str = ""

@dataclass
class CollectionProgress:
    """수집 진행 상황"""
    current_page: int = 1
    total_pages: int = 0
    collected_count: int = 0
    failed_count: int = 0
    last_collected_time: Optional[datetime] = None
    resume_page: Optional[int] = None
    collection_type: str = "lists"  # lists, details, relations

class BaseLegalTermCollector:
    """법령정보지식베이스 법령용어 수집기"""
    
    def __init__(self, config: Config):
        self.config = config
        self.api_config = config.get_api_config()
        self.collection_config = config.get_collection_config()
        self.file_storage_config = config.get_file_storage_config()
        self.database_config = config.get_database_config()
        
        # API 설정
        self.base_url = self.api_config.get("base_url", "https://www.law.go.kr/DRF")
        self.list_url = f"{self.base_url}{self.api_config.get('list_endpoint', '/lawSearch.do')}"
        self.detail_url = f"{self.base_url}{self.api_config.get('detail_endpoint', '/lawService.do')}"
        
        self.oc_id = self.api_config.get("oc_id", "test")
        self.target = self.api_config.get("target", "lstrmAI")
        self.type = self.api_config.get("type", "JSON")
        self.display_count = self.api_config.get("display_count", 100)
        self.max_retries = self.api_config.get("max_retries", 3)
        self.retry_delays = self.api_config.get("retry_delays", [180, 300, 600])
        
        # 파일 저장 경로
        self.base_dir = Path(self.file_storage_config.get("base_dir", "data/base_legal_terms"))
        self.raw_dir = Path(self.file_storage_config.get("raw_data_dir", "data/base_legal_terms/raw"))
        self.processed_dir = Path(self.file_storage_config.get("processed_data_dir", "data/base_legal_terms/processed"))
        self.db_path = Path(self.database_config.get("db_path", "data/lawfirm.db"))
        
        # 세부 디렉토리
        self.term_lists_dir = Path(self.file_storage_config.get("term_lists_dir", "data/base_legal_terms/raw/term_lists"))
        self.term_details_dir = Path(self.file_storage_config.get("term_details_dir", "data/base_legal_terms/raw/term_details"))
        self.term_relations_dir = Path(self.file_storage_config.get("term_relations_dir", "data/base_legal_terms/raw/term_relations"))
        self.api_responses_dir = Path(self.file_storage_config.get("api_responses_dir", "data/base_legal_terms/raw/api_responses"))
        
        # 디렉토리 생성
        self._create_directories()
        
        # 진행 상황 파일
        monitoring_config = config.get_monitoring_config()
        self.progress_file = Path(monitoring_config.get("progress_file", "data/base_legal_terms/progress/collection_progress.json"))
        
        # 세션 설정
        self.session: Optional[aiohttp.ClientSession] = None
        self.progress = CollectionProgress()
        
        # 데이터베이스 초기화
        self._init_database()
        
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            self.base_dir,
            self.raw_dir,
            self.processed_dir,
            self.term_lists_dir,
            self.term_details_dir,
            self.term_relations_dir,
            self.api_responses_dir,
            self.base_dir / "logs",
            self.base_dir / "progress",
            self.base_dir / "reports",
            self.base_dir / "database"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            # 데이터베이스 디렉토리 생성
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 법령용어 목록 테이블
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS base_legal_term_lists (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        법령용어ID TEXT UNIQUE,
                        법령용어명 TEXT,
                        동음이의어존재여부 TEXT,
                        비고 TEXT,
                        용어간관계링크 TEXT,
                        조문간관계링크 TEXT,
                        수집일시 TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 법령용어 상세 테이블
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS base_legal_term_details (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        법령용어일련번호 INTEGER UNIQUE,
                        법령용어명_한글 TEXT,
                        법령용어명_한자 TEXT,
                        법령용어코드 INTEGER,
                        법령용어코드명 TEXT,
                        출처 TEXT,
                        법령용어정의 TEXT,
                        동음이의어내용 TEXT,
                        용어관계정보 TEXT,
                        조문관계정보 TEXT,
                        수집일시 TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 수집 진행 상황 테이블
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS collection_progress (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        collection_type TEXT,
                        current_page INTEGER,
                        total_pages INTEGER,
                        collected_count INTEGER,
                        failed_count INTEGER,
                        last_collected_time TEXT,
                        resume_page INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info("데이터베이스 초기화 완료")
                
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(self):
        """HTTP 세션 컨텍스트 매니저"""
        timeout = aiohttp.ClientTimeout(total=self.api_config.get("request_timeout", 30))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.session = session
            try:
                yield session
            finally:
                self.session = None
    
    async def _make_request(self, url: str, params: Dict[str, Any], retry_count: int = 0) -> Optional[Dict]:
        """API 요청 실행 (재시도 로직 포함)"""
        try:
            async with self.session.get(url, params=params) as response:
                logger.info(f"응답 상태: {response.status}")
                logger.info(f"응답 헤더: {dict(response.headers)}")
                
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    logger.info(f"Content-Type: {content_type}")
                    
                    # JSON 응답 처리
                    if 'application/json' in content_type or 'text/json' in content_type:
                        try:
                            json_data = await response.json()
                            logger.info(f"JSON 응답 수신: {type(json_data)}")
                            return json_data
                        except Exception as e:
                            logger.error(f"JSON 파싱 실패: {e}")
                            text = await response.text()
                            logger.error(f"응답 내용: {text[:500]}...")
                            return None
                    else:
                        text = await response.text()
                        logger.warning(f"JSON이 아닌 응답 (Content-Type: {content_type})")
                        logger.warning(f"응답 내용: {text[:200]}...")
                        return None
                else:
                    logger.error(f"HTTP 오류: {response.status}")
                    text = await response.text()
                    logger.error(f"오류 응답: {text[:200]}...")
                    return None
                    
        except Exception as e:
            logger.error(f"요청 실패: {e}")
            
            if retry_count < self.max_retries:
                delay = self.retry_delays[min(retry_count, len(self.retry_delays) - 1)]
                logger.info(f"{delay}초 후 재시도 ({retry_count + 1}/{self.max_retries})")
                await asyncio.sleep(delay)
                return await self._make_request(url, params, retry_count + 1)
            else:
                logger.error(f"최대 재시도 횟수 초과")
                return None
    
    async def get_term_list(self, page: int, query: str = "") -> Optional[Dict]:
        """법령용어 목록 조회"""
        params = {
            'OC': self.oc_id,
            'target': self.target,
            'type': self.type
        }
        
        # 선택적 파라미터들
        if query:
            params['query'] = query
        if self.display_count:
            params['display'] = self.display_count
        if page > 1:
            params['page'] = page
        
        logger.info(f"법령용어 목록 조회: 페이지 {page}, 쿼리: '{query}'")
        logger.info(f"요청 URL: {self.list_url}")
        logger.info(f"요청 파라미터: {params}")
        
        response = await self._make_request(self.list_url, params)
        
        if response:
            # lstrmAISearch 구조에서 데이터 추출
            search_data = response.get('lstrmAISearch', {})
            items = search_data.get('items', [])
            logger.info(f"페이지 {page} 응답 수신: {len(items)}개 항목")
            logger.info(f"응답 구조: {list(response.keys())}")
            logger.info(f"검색 결과 개수: {search_data.get('검색결과개수', '0')}")
            logger.info(f"search_data 키들: {list(search_data.keys())}")
            logger.info(f"search_data 전체: {search_data}")
            
            # 실제 데이터가 있는 경우만 로그 출력
            if items:
                logger.info(f"첫 번째 항목: {items[0]}")
        
        return response
    
    async def get_term_detail(self, term_id: str) -> Optional[Dict]:
        """법령용어 상세 정보 조회"""
        params = {
            'OC': self.oc_id,
            'target': 'lstrm',  # 상세 조회는 다른 타겟 사용
            'trmSeqs': term_id,
            'type': self.type
        }
        
        logger.debug(f"법령용어 상세 조회: ID {term_id}")
        
        response = await self._make_request(self.detail_url, params)
        
        if response:
            logger.debug(f"용어 {term_id} 상세 정보 수신")
        
        return response
    
    def _parse_term_list_response(self, response: Dict) -> Tuple[List[BaseLegalTermListItem], int]:
        """법령용어 목록 응답 파싱"""
        try:
            if not response:
                logger.warning("응답이 비어있습니다.")
                return [], 0
            
            # lstrmAISearch 구조에서 데이터 추출
            search_data = response.get('lstrmAISearch', {})
            if not search_data:
                logger.warning("응답에 lstrmAISearch 필드가 없습니다.")
                return [], 0
            
            # 실제 데이터는 '법령용어' 필드에 있음
            items = search_data.get('법령용어', search_data.get('items', search_data.get('item', search_data.get('data', []))))
            
            if not isinstance(items, list):
                logger.warning("items가 리스트가 아닙니다.")
                logger.info(f"search_data 구조: {list(search_data.keys())}")
                logger.info(f"search_data 전체 내용: {search_data}")
                
                # 다른 가능한 필드들 확인
                for key in search_data.keys():
                    if isinstance(search_data[key], list) and len(search_data[key]) > 0:
                        logger.info(f"리스트 형태의 데이터 발견: {key} = {len(search_data[key])}개 항목")
                        items = search_data[key]
                        break
                
                if not isinstance(items, list):
                    return [], 0
            
            parsed_items = []
            for item in items:
                try:
                    parsed_item = BaseLegalTermListItem(
                        법령용어ID=str(item.get('id', item.get('법령용어ID', item.get('법령용어일련번호', '')))),
                        법령용어명=str(item.get('법령용어명', '')),
                        동음이의어존재여부=str(item.get('동음이의어존재여부', '')),
                        비고=str(item.get('비고', '')),
                        용어간관계링크=str(item.get('용어간관계링크', '')),
                        조문간관계링크=str(item.get('조문간관계링크', '')),
                        수집일시=datetime.now().isoformat()
                    )
                    parsed_items.append(parsed_item)
                except Exception as e:
                    logger.warning(f"아이템 파싱 실패: {e}, 아이템: {item}")
                    continue
            
            # 총 페이지 수 계산
            total_count = int(search_data.get('검색결과개수', '0'))
            num_of_rows = int(search_data.get('numOfRows', '100'))
            total_pages = (total_count + num_of_rows - 1) // num_of_rows if total_count > 0 else 0
            
            logger.info(f"파싱 완료: {len(parsed_items)}개 항목, 총 페이지: {total_pages}")
            
            return parsed_items, total_pages
            
        except Exception as e:
            logger.error(f"응답 파싱 실패: {e}")
            logger.error(f"응답 데이터: {response}")
            return [], 0
    
    def _parse_term_detail_response(self, response: Dict) -> Optional[BaseLegalTermDetail]:
        """법령용어 상세 정보 응답 파싱"""
        try:
            if not response or 'items' not in response:
                logger.warning("상세 응답에 items 필드가 없습니다.")
                return None
            
            items = response['items']
            if not isinstance(items, list) or len(items) == 0:
                logger.warning("상세 응답 items가 비어있습니다.")
                return None
            
            item = items[0]  # 첫 번째 항목 사용
            
            parsed_item = BaseLegalTermDetail(
                법령용어일련번호=int(item.get('법령용어일련번호', 0)),
                법령용어명_한글=str(item.get('법령용어명_한글', '')),
                법령용어명_한자=str(item.get('법령용어명_한자', '')),
                법령용어코드=int(item.get('법령용어코드', 0)),
                법령용어코드명=str(item.get('법령용어코드명', '')),
                출처=str(item.get('출처', '')),
                법령용어정의=str(item.get('법령용어정의', '')),
                동음이의어내용=str(item.get('동음이의어내용', '')),
                수집일시=datetime.now().isoformat()
            )
            
            logger.debug(f"상세 정보 파싱 완료: {parsed_item.법령용어명_한글}")
            return parsed_item
            
        except Exception as e:
            logger.error(f"상세 응답 파싱 실패: {e}")
            logger.error(f"응답 데이터: {response}")
            return None
    
    def _save_to_file(self, data: Any, filename: str, batch_number: int, page_info: str = ""):
        """데이터를 파일로 저장"""
        try:
            # 빈 배열이나 유효하지 않은 데이터 체크
            if isinstance(data, list) and len(data) == 0:
                logger.warning("빈 배열은 저장하지 않습니다.")
                return
            
            # 데이터 유효성 검사
            if isinstance(data, list):
                valid_items = []
                for item in data:
                    if isinstance(item, dict):
                        valid_items.append(item)
                    elif hasattr(item, '__dict__'):
                        # dataclass 객체인 경우 딕셔너리로 변환
                        item_dict = asdict(item)
                        valid_items.append(item_dict)
                    else:
                        logger.warning(f"유효하지 않은 데이터 타입: {type(item)}")
                        continue
                
                if len(valid_items) == 0:
                    logger.warning("유효한 데이터가 없어 저장하지 않습니다.")
                    return
                
                data = valid_items
            
            # 파일 경로 결정
            if "list" in filename.lower():
                save_dir = self.term_lists_dir
            elif "detail" in filename.lower():
                save_dir = self.term_details_dir
            elif "relation" in filename.lower():
                save_dir = self.term_relations_dir
            else:
                save_dir = self.api_responses_dir
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if page_info:
                file_path = save_dir / f"{filename}_{batch_number:03d}_{page_info}_{timestamp}.json"
            else:
                file_path = save_dir / f"{filename}_{batch_number:03d}_{timestamp}.json"
            
            # 파일 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"데이터 저장 완료: {file_path}")
            
        except Exception as e:
            logger.error(f"파일 저장 실패: {e}")
    
    def _save_progress(self, collection_type: str, current_page: int, collected_count: int):
        """진행 상황 저장"""
        try:
            self.progress.collection_type = collection_type
            self.progress.current_page = current_page
            self.progress.collected_count += collected_count
            self.progress.last_collected_time = datetime.now()
            
            progress_data = {
                "collection_type": self.progress.collection_type,
                "current_page": self.progress.current_page,
                "total_pages": self.progress.total_pages,
                "collected_count": self.progress.collected_count,
                "failed_count": self.progress.failed_count,
                "last_collected_time": self.progress.last_collected_time.isoformat() if self.progress.last_collected_time else None,
                "resume_page": self.progress.resume_page,
                "updated_at": datetime.now().isoformat()
            }
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"진행 상황 저장: {collection_type} 페이지 {current_page}, 수집 {self.progress.collected_count}개")
            
        except Exception as e:
            logger.error(f"진행 상황 저장 실패: {e}")
    
    def _load_progress(self) -> CollectionProgress:
        """진행 상황 로드"""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                progress = CollectionProgress(
                    current_page=data.get("current_page", 1),
                    total_pages=data.get("total_pages", 0),
                    collected_count=data.get("collected_count", 0),
                    failed_count=data.get("failed_count", 0),
                    resume_page=data.get("resume_page"),
                    collection_type=data.get("collection_type", "lists")
                )
                
                if data.get("last_collected_time"):
                    try:
                        progress.last_collected_time = datetime.fromisoformat(data["last_collected_time"])
                    except:
                        progress.last_collected_time = None
                
                logger.info(f"진행 상황 로드: {progress.collection_type} 페이지 {progress.current_page}")
                return progress
            else:
                logger.info("진행 상황 파일이 없습니다. 새로 시작합니다.")
                return CollectionProgress()
                
        except Exception as e:
            logger.error(f"진행 상황 로드 실패: {e}")
            return CollectionProgress()
    
    async def collect_term_lists(self, start_page: int = 1, end_page: Optional[int] = None, 
                                batch_size: int = 20, query: str = ""):
        """법령용어 목록 수집"""
        logger.info(f"법령용어 목록 수집 시작: 페이지 {start_page}부터")
        
        # 진행 상황 로드
        self.progress = self._load_progress()
        if self.progress.resume_page and self.progress.collection_type == "lists":
            start_page = self.progress.resume_page
            logger.info(f"이전 진행 상황에서 재개: 페이지 {start_page}")
        
        current_page = start_page
        batch_number = 1
        batch_data = []
        
        try:
            async with self.get_session():
                while True:
                    if end_page and current_page > end_page:
                        break
                    
                    logger.info(f"페이지 {current_page} 수집 중...")
                    
                    # API 요청
                    response = await self.get_term_list(current_page, query)
                    
                    if response is None:
                        logger.error(f"페이지 {current_page} 수집 실패")
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
                    
                    # 배치 크기 도달 시 저장
                    if len(batch_data) >= batch_size:
                        self._save_to_file(batch_data, "term_list_batch", batch_number, f"page{current_page}")
                        batch_data = []
                        batch_number += 1
                    
                    # 진행 상황 저장
                    self._save_progress("lists", current_page, len(items))
                    
                    # 요청 간 대기
                    delay = self.api_config.get("rate_limit_delay", 1.0)
                    await asyncio.sleep(delay)
                    
                    current_page += 1
                    
                    # 마지막 페이지 확인
                    if total_pages > 0 and current_page > total_pages:
                        logger.info(f"모든 페이지 수집 완료: {total_pages}페이지")
                        break
                
                # 남은 배치 데이터 저장
                if batch_data:
                    self._save_to_file(batch_data, "term_list_batch", batch_number, f"page{current_page-1}")
                
                logger.info(f"법령용어 목록 수집 완료: 총 {self.progress.collected_count}개 수집")
                
        except Exception as e:
            logger.error(f"법령용어 목록 수집 중 오류: {e}")
            raise
    
    async def collect_term_details(self, batch_size: int = 50):
        """법령용어 상세 정보 수집"""
        logger.info("법령용어 상세 정보 수집 시작")
        
        # 목록 파일들 로드
        list_files = list(self.term_lists_dir.glob("*.json"))
        if not list_files:
            logger.warning("수집할 목록 파일이 없습니다.")
            return
        
        logger.info(f"{len(list_files)}개 목록 파일에서 상세 정보 수집")
        
        batch_number = 1
        batch_data = []
        total_collected = 0
        
        try:
            async with self.get_session():
                for list_file in list_files:
                    logger.info(f"목록 파일 처리: {list_file.name}")
                    
                    with open(list_file, 'r', encoding='utf-8') as f:
                        terms = json.load(f)
                    
                    for term in terms:
                        term_id = term.get('법령용어ID', '')
                        if not term_id:
                            continue
                        
                        # 상세 정보 요청
                        detail_response = await self.get_term_detail(term_id)
                        
                        if detail_response:
                            detail = self._parse_term_detail_response(detail_response)
                            if detail:
                                batch_data.append(asdict(detail))
                                total_collected += 1
                        
                        # 배치 크기 도달 시 저장
                        if len(batch_data) >= batch_size:
                            self._save_to_file(batch_data, "term_detail_batch", batch_number)
                            batch_data = []
                            batch_number += 1
                        
                        # 요청 간 대기
                        delay = self.collection_config.get("detail_collection_delay", 1.0)
                        await asyncio.sleep(delay)
                
                # 남은 배치 데이터 저장
                if batch_data:
                    self._save_to_file(batch_data, "term_detail_batch", batch_number)
                
                logger.info(f"법령용어 상세 정보 수집 완료: 총 {total_collected}개 수집")
                
        except Exception as e:
            logger.error(f"법령용어 상세 정보 수집 중 오류: {e}")
            raise
    
    async def collect_alternating(self, start_page: int = 1, end_page: Optional[int] = None,
                                list_batch_size: int = 20, detail_batch_size: int = 50,
                                query: str = ""):
        """목록 수집과 상세 수집을 번갈아가면서 진행"""
        logger.info("번갈아가면서 수집 시작")
        
        # 먼저 목록 수집
        await self.collect_term_lists(
            start_page=start_page,
            end_page=end_page,
            batch_size=list_batch_size,
            query=query
        )
        
        # 그 다음 상세 수집
        await self.collect_term_details(batch_size=detail_batch_size)
        
        logger.info("번갈아가면서 수집 완료")

    def load_collected_data_to_database(self):
        """수집된 데이터를 데이터베이스에 적재"""
        logger.info("수집된 데이터를 데이터베이스에 적재 시작")
        
        # 수집된 파일들 찾기
        batch_files = list(self.term_lists_dir.glob("term_list_batch_*.json"))
        logger.info(f"발견된 배치 파일: {len(batch_files)}개")
        
        total_loaded = 0
        total_skipped = 0
        
        try:
            with sqlite3.connect(self.db_path, timeout=self.database_config.get("connection_timeout", 30)) as conn:
                cursor = conn.cursor()
                
                for batch_file in batch_files:
                    logger.info(f"처리 중: {batch_file.name}")
                    
                    try:
                        with open(batch_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        batch_loaded = 0
                        batch_skipped = 0
                        
                        for item in data:
                            try:
                                # 데이터베이스에 삽입 (중복 체크)
                                cursor.execute('''
                                    INSERT OR IGNORE INTO base_legal_term_lists 
                                    (법령용어ID, 법령용어명, 동음이의어존재여부, 비고, 용어간관계링크, 조문간관계링크, 수집일시)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    item.get('법령용어ID', ''),
                                    item.get('법령용어명', ''),
                                    item.get('동음이의어존재여부', ''),
                                    item.get('비고', ''),
                                    item.get('용어간관계링크', ''),
                                    item.get('조문간관계링크', ''),
                                    item.get('수집일시', '')
                                ))
                                
                                if cursor.rowcount > 0:
                                    batch_loaded += 1
                                else:
                                    batch_skipped += 1
                                    
                            except Exception as e:
                                logger.warning(f"아이템 삽입 실패: {e}, 아이템: {item}")
                                batch_skipped += 1
                                continue
                        
                        total_loaded += batch_loaded
                        total_skipped += batch_skipped
                        
                        logger.info(f"{batch_file.name}: {batch_loaded}개 삽입, {batch_skipped}개 스킵")
                        
                    except Exception as e:
                        logger.error(f"파일 처리 실패 {batch_file.name}: {e}")
                        continue
                
                conn.commit()
                logger.info(f"데이터베이스 적재 완료: 총 {total_loaded}개 삽입, {total_skipped}개 스킵")
                
                # 최종 통계 확인
                cursor.execute("SELECT COUNT(*) FROM base_legal_term_lists")
                total_count = cursor.fetchone()[0]
                logger.info(f"데이터베이스 총 용어 수: {total_count}개")
                
        except Exception as e:
            logger.error(f"데이터베이스 적재 실패: {e}")
            raise


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='법령정보지식베이스 법령용어 수집기')
    
    # 수집 옵션
    parser.add_argument('--collect-lists', action='store_true',
                       help='법령용어 목록만 수집')
    parser.add_argument('--collect-details', action='store_true',
                       help='법령용어 상세 정보만 수집')
    parser.add_argument('--collect-alternating', action='store_true',
                       help='목록 수집과 상세 수집을 번갈아가면서 진행')
    parser.add_argument('--load-to-database', action='store_true',
                       help='수집된 데이터를 데이터베이스에 적재')
    parser.add_argument('--init-database', action='store_true',
                       help='데이터베이스 테이블 초기화')
    
    # 페이지 설정
    parser.add_argument('--start-page', type=int, default=1,
                       help='시작 페이지 (기본값: 1)')
    parser.add_argument('--end-page', type=int, default=None,
                       help='종료 페이지 (기본값: 무제한)')
    
    # 배치 설정
    parser.add_argument('--batch-size', type=int, default=20,
                       help='목록 수집 배치 크기 (기본값: 20)')
    parser.add_argument('--detail-batch-size', type=int, default=50,
                       help='상세 수집 배치 크기 (기본값: 50)')
    
    # 검색 설정
    parser.add_argument('--query', type=str, default='',
                       help='검색 쿼리')
    
    # API 설정
    parser.add_argument('--display-count', type=int, default=100,
                       help='페이지당 결과 수 (기본값: 100)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='최대 재시도 횟수 (기본값: 3)')
    
    # 대기 시간 설정
    parser.add_argument('--rate-limit-delay', type=float, default=1.0,
                       help='요청 간 대기 시간(초) (기본값: 1.0)')
    parser.add_argument('--detail-delay', type=float, default=1.0,
                       help='상세 조회 간 대기 시간(초) (기본값: 1.0)')
    
    # 기타 옵션
    parser.add_argument('--config-file', type=str, default=None,
                       help='설정 파일 경로')
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
    config_dict['api']['display_count'] = args.display_count
    config_dict['api']['max_retries'] = args.max_retries
    config_dict['api']['rate_limit_delay'] = args.rate_limit_delay
    config_dict['collection']['start_page'] = args.start_page
    config_dict['collection']['end_page'] = args.end_page
    config_dict['collection']['query'] = args.query
    config_dict['collection']['list_batch_size'] = args.batch_size
    config_dict['collection']['detail_batch_size'] = args.detail_batch_size
    config_dict['collection']['detail_collection_delay'] = args.detail_delay
    
    # 로그 레벨 설정
    if args.verbose:
        config_dict['logging']['level'] = 'DEBUG'
    
    # 설정 업데이트
    config.update_config(config_dict)
    
    logger.info(f"=== 법령정보지식베이스 법령용어 수집 시작 ===")
    logger.info(f"시작 페이지: {args.start_page}")
    logger.info(f"종료 페이지: {args.end_page or '무제한'}")
    logger.info(f"목록 수집: {'예' if args.collect_lists else '아니오'}")
    logger.info(f"상세 수집: {'예' if args.collect_details else '아니오'}")
    logger.info(f"번갈아가면서 수집: {'예' if args.collect_alternating else '아니오'}")
    logger.info(f"배치 크기: 목록 {args.batch_size}, 상세 {args.detail_batch_size}")
    logger.info(f"검색 쿼리: '{args.query}'")
    
    # 수집기 생성 및 실행
    collector = BaseLegalTermCollector(config)
    
    try:
        if args.init_database:
            logger.info("데이터베이스 초기화 시작")
            collector._init_database()
            logger.info("=== 데이터베이스 초기화 완료 ===")
        elif args.load_to_database:
            logger.info("데이터베이스 적재 시작")
            collector.load_collected_data_to_database()
            logger.info("=== 데이터베이스 적재 완료 ===")
        elif args.collect_alternating:
            logger.info("번갈아가면서 수집 시작")
            await collector.collect_alternating(
                start_page=args.start_page,
                end_page=args.end_page,
                list_batch_size=args.batch_size,
                detail_batch_size=args.detail_batch_size,
                query=args.query
            )
        elif args.collect_details:
            logger.info("상세 정보 수집 시작")
            await collector.collect_term_details(batch_size=args.detail_batch_size)
        elif args.collect_lists:
            logger.info("목록 수집 시작")
            await collector.collect_term_lists(
                start_page=args.start_page,
                end_page=args.end_page,
                batch_size=args.batch_size,
                query=args.query
            )
        else:
            logger.info("기본 모드: 목록 수집 시작")
            await collector.collect_term_lists(
                start_page=args.start_page,
                end_page=args.end_page,
                batch_size=args.batch_size,
                query=args.query
            )

        logger.info("=== 작업 완료 ===")

    except Exception as e:
        logger.error(f"작업 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
