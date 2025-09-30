#!/usr/bin/env python3
"""
날짜 기반 판례 수집기

이 모듈은 날짜별로 체계적인 판례 수집을 수행합니다.
- 연도별, 분기별, 월별, 주별 수집 전략
- 선고일자 내림차순 최적화
- 폴더별 raw 데이터 저장 구조
- 중복 방지 및 체크포인트 지원
"""

import json
import time
import random
import hashlib
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient, LawOpenAPIConfig
from scripts.precedent.precedent_models import (
    CollectionStats, PrecedentData, CollectionStatus, PrecedentCategory,
    COURT_CODES, CASE_TYPE_CODES
)
import logging

logger = logging.getLogger(__name__)


class DateCollectionStrategy(Enum):
    """날짜 수집 전략 열거형"""
    YEARLY = "yearly"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    WEEKLY = "weekly"
    DAILY = "daily"


@dataclass
class DateCollectionConfig:
    """날짜 수집 설정 클래스"""
    strategy: DateCollectionStrategy
    start_date: str
    end_date: str
    target_count: int
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: int = 5
    api_delay_range: Tuple[float, float] = (1.0, 3.0)
    output_subdir: Optional[str] = None


class DateBasedPrecedentCollector:
    """날짜 기반 판례 수집 클래스"""
    
    def __init__(self, config: LawOpenAPIConfig, base_output_dir: Optional[Path] = None, 
                 include_details: bool = True):
        """
        날짜 기반 판례 수집기 초기화
        
        Args:
            config: API 설정 객체
            base_output_dir: 기본 출력 디렉토리 (기본값: data/raw/precedents)
            include_details: 판례본문 포함 여부 (기본값: True)
        """
        self.client = LawOpenAPIClient(config)
        self.base_output_dir = base_output_dir or Path("data/raw/precedents")
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.include_details = include_details  # 판례본문 수집 여부
        
        # 데이터 관리 (메모리 최적화)
        self.collected_precedents: Set[str] = set()
        self.processed_date_ranges: Set[str] = set()
        self.pending_precedents: List[PrecedentData] = []
        self.max_memory_precedents = 50000  # 최대 메모리 보관 건수
        
        # 통계 및 상태
        self.stats = CollectionStats()
        self.stats.status = CollectionStatus.PENDING
        
        # 에러 처리
        self.error_count = 0
        self.max_errors = 50
        
        # 시간 인터벌 설정 (기본값)
        self.request_interval_base = 2.0  # 기본 간격
        self.request_interval_range = 2.0  # 간격 범위
        
        # 기존 수집된 데이터 로드
        self._load_existing_data()
        
        logger.info(f"날짜 기반 판례 수집기 초기화 완료 (판례본문 포함: {include_details})")
    
    def set_request_interval(self, base_interval: float, interval_range: float):
        """API 요청 간격 설정"""
        self.request_interval_base = base_interval
        self.request_interval_range = interval_range
        logger.info(f"⏱️ 요청 간격 설정: {base_interval:.1f} ± {interval_range:.1f}초")
    
    def _load_existing_data(self, target_year: Optional[int] = None):
        """기존 수집된 데이터 로드하여 중복 방지 (메모리 최적화)"""
        logger.info("기존 수집된 데이터 확인 중...")
        
        loaded_count = 0
        error_count = 0
        
        # 모든 하위 디렉토리에서 데이터 로드 (최신 폴더 우선)
        subdirs = sorted([d for d in self.base_output_dir.iterdir() if d.is_dir()], 
                        key=lambda x: x.name, reverse=True)
        
        for subdir in subdirs:
            if len(self.collected_precedents) >= self.max_memory_precedents:
                logger.info(f"⚠️ 메모리 한계 도달: {len(self.collected_precedents):,}건, 추가 로드 중단")
                break
                
                for file_path in subdir.glob("*.json"):
                    try:
                        loaded_count += self._load_precedents_from_file(file_path, target_year)
                        
                        # 메모리 사용량 체크
                        if len(self.collected_precedents) >= self.max_memory_precedents:
                            logger.info(f"⚠️ 메모리 한계 도달: {len(self.collected_precedents):,}건, 추가 로드 중단")
                            break
                            
                    except Exception as e:
                        error_count += 1
                        logger.debug(f"파일 로드 실패 {file_path}: {e}")
            
            if len(self.collected_precedents) >= self.max_memory_precedents:
                break
        
        logger.info(f"기존 데이터 로드 완료: {loaded_count:,}건, 오류: {error_count:,}건")
        self.stats.collected_count = len(self.collected_precedents)
        logger.info(f"중복 방지를 위한 판례 ID {len(self.collected_precedents):,}개 로드됨 (메모리 최적화)")
    
    def _load_precedents_from_file(self, file_path: Path, target_year: Optional[int] = None) -> int:
        """파일에서 판례 데이터 로드 (특정 연도 필터링)"""
        loaded_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 다양한 데이터 구조 처리
            precedents = []
            if isinstance(data, dict):
                if 'precedents' in data:
                    precedents = data['precedents']
                elif 'basic_info' in data:
                    precedents = [data]
                elif 'by_category' in data:
                    for category_data in data['by_category'].values():
                        precedents.extend(category_data)
            elif isinstance(data, list):
                precedents = data
            
            # 판례 ID 추출 (특정 연도 필터링)
            for precedent in precedents:
                if isinstance(precedent, dict):
                    # 특정 연도가 지정된 경우 해당 연도의 판례만 로드
                    if target_year:
                        decision_date = precedent.get('선고일자', '') or precedent.get('판결일자', '')
                        if decision_date:
                            try:
                                # 날짜 파싱 (YYYY.MM.DD 형식)
                                if '.' in decision_date:
                                    date_parts = decision_date.split('.')
                                    if len(date_parts) >= 1:
                                        precedent_year = int(date_parts[0])
                                        if precedent_year != target_year:
                                            continue  # 다른 연도는 건너뛰기
                                else:
                                    continue  # 날짜 형식이 잘못된 경우 건너뛰기
                            except (ValueError, IndexError):
                                continue  # 날짜 파싱 오류 시 건너뛰기
                        else:
                            continue  # 날짜가 없는 경우 건너뛰기
                    
                    precedent_id = precedent.get('판례일련번호') or precedent.get('precedent_id')
                    if precedent_id:
                        self.collected_precedents.add(str(precedent_id))
                        loaded_count += 1
        
        except Exception as e:
            logger.debug(f"파일 로드 실패 {file_path}: {e}")
        
        return loaded_count
    
    def _create_output_subdir(self, strategy: DateCollectionStrategy, date_range: str) -> Path:
        """출력 하위 디렉토리 생성"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 전략별 디렉토리 구조
        if strategy == DateCollectionStrategy.YEARLY:
            subdir_name = f"yearly_{date_range}_{timestamp}"
        elif strategy == DateCollectionStrategy.QUARTERLY:
            subdir_name = f"quarterly_{date_range}_{timestamp}"
        elif strategy == DateCollectionStrategy.MONTHLY:
            subdir_name = f"monthly_{date_range}_{timestamp}"
        elif strategy == DateCollectionStrategy.WEEKLY:
            subdir_name = f"weekly_{date_range}_{timestamp}"
        elif strategy == DateCollectionStrategy.DAILY:
            subdir_name = f"daily_{date_range}_{timestamp}"
        else:
            subdir_name = f"date_based_{date_range}_{timestamp}"
        
        output_dir = self.base_output_dir / subdir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def _is_duplicate_precedent(self, precedent: Dict[str, Any]) -> bool:
        """판례 중복 여부 확인 (개선된 로직)"""
        precedent_id = (
            precedent.get('판례일련번호') or 
            precedent.get('precedent_id') or 
            precedent.get('prec id') or
            precedent.get('id')
        )
        
        # 판례일련번호로 중복 확인
        if precedent_id and str(precedent_id) in self.collected_precedents:
            return True
        
        # 대체 식별자로 확인 (더 엄격한 조건)
        case_number = precedent.get('사건번호', '')
        case_name = precedent.get('사건명', '')
        decision_date = precedent.get('선고일자', '')
        
        # 사건번호, 사건명, 선고일자가 모두 일치하는 경우만 중복으로 처리
        if case_number and case_name and decision_date:
            alternative_id = f"{case_number}_{case_name}_{decision_date}"
            if alternative_id in self.collected_precedents:
                return True
        
        return False
    
    def _mark_precedent_collected(self, precedent: Dict[str, Any]):
        """판례를 수집됨으로 표시 (메모리 최적화)"""
        precedent_id = (
            precedent.get('판례일련번호') or 
            precedent.get('precedent_id') or 
            precedent.get('prec id') or
            precedent.get('id')
        )
        
        if precedent_id:
            self.collected_precedents.add(str(precedent_id))
        
        # 대체 식별자로도 저장
        case_number = precedent.get('사건번호', '')
        case_name = precedent.get('사건명', '')
        decision_date = precedent.get('선고일자', '')
        
        if case_number and case_name and decision_date:
            alternative_id = f"{case_number}_{case_name}_{decision_date}"
            self.collected_precedents.add(alternative_id)
        
        # 메모리 사용량 체크 및 최적화
        self._check_memory_usage()
    
    def _check_memory_usage(self):
        """메모리 사용량 체크 및 최적화"""
        if len(self.collected_precedents) > self.max_memory_precedents:
            logger.warning(f"⚠️ 메모리 사용량 초과: {len(self.collected_precedents):,}건 > {self.max_memory_precedents:,}건")
            logger.info("🔄 메모리 최적화를 위해 중복 데이터 일부 정리 중...")
            
            # 오래된 데이터 일부 제거 (최신 데이터 우선 보존)
            items_to_remove = len(self.collected_precedents) - self.max_memory_precedents
            items_list = list(self.collected_precedents)
            
            # 판례일련번호는 보존하고 대체 식별자부터 제거
            precedent_ids = [item for item in items_list if item.isdigit()]
            alternative_ids = [item for item in items_list if not item.isdigit()]
            
            # 대체 식별자부터 제거
            removed_count = 0
            for alt_id in alternative_ids[:items_to_remove]:
                self.collected_precedents.discard(alt_id)
                removed_count += 1
            
            # 여전히 초과하면 판례일련번호도 제거
            if len(self.collected_precedents) > self.max_memory_precedents:
                remaining_to_remove = len(self.collected_precedents) - self.max_memory_precedents
                for prec_id in precedent_ids[:remaining_to_remove]:
                    self.collected_precedents.discard(prec_id)
                    removed_count += 1
            
            logger.info(f"✅ 메모리 최적화 완료: {removed_count:,}건 제거, 현재 {len(self.collected_precedents):,}건 보관")
    
    def _create_precedent_data(self, raw_data: Dict[str, Any]) -> Optional[PrecedentData]:
        """원시 데이터에서 PrecedentData 객체 생성"""
        try:
            # 판례 ID 추출
            precedent_id = (
                raw_data.get('판례일련번호') or 
                raw_data.get('precedent_id') or 
                raw_data.get('prec id') or
                raw_data.get('id')
            )
            
            if not precedent_id:
                case_number = raw_data.get('사건번호', '')
                case_name = raw_data.get('사건명', '')
                if case_name:
                    precedent_id = f"{case_number}_{case_name}"
                else:
                    precedent_id = f"case_{case_number}"
            
            # 카테고리 분류
            category = self._categorize_precedent(raw_data)
            
            # PrecedentData 객체 생성
            precedent_data = PrecedentData(
                precedent_id=str(precedent_id),
                case_name=raw_data.get('사건명', ''),
                case_number=raw_data.get('사건번호', ''),
                court=COURT_CODES.get(raw_data.get('법원코드', ''), ''),
                case_type=CASE_TYPE_CODES.get(raw_data.get('사건유형코드', ''), ''),
                decision_date=raw_data.get('판결일자', '') or raw_data.get('선고일자', ''),
                category=category,
                raw_data=raw_data
            )
            
            return precedent_data
            
        except Exception as e:
            logger.error(f"PrecedentData 생성 실패: {e}")
            return None
    
    def _create_precedent_data_with_detail(self, raw_data: Dict[str, Any]) -> Optional[PrecedentData]:
        """원시 데이터에서 PrecedentData 객체 생성 (판례본문 포함)"""
        try:
            # 기본 PrecedentData 생성
            precedent_data = self._create_precedent_data(raw_data)
            if not precedent_data:
                return None
            
            # 판례본문 수집 (재시도 메커니즘 포함)
            precedent_id = raw_data.get('판례일련번호')
            if precedent_id:
                logger.info(f"🔍 판례본문 수집 시작: {raw_data.get('사건명', 'Unknown')} (ID: {precedent_id})")
                detail_info = self._collect_precedent_detail_with_retry(precedent_id)
                precedent_data.detail_info = detail_info
                logger.info(f"✅ 판례본문 수집 완료: {raw_data.get('사건명', 'Unknown')} (ID: {precedent_id})")
            else:
                logger.warning(f"⚠️ 판례일련번호가 없어 판례본문 수집 불가: {raw_data.get('사건명', 'Unknown')}")
                precedent_data.detail_info = {}
            
            return precedent_data
            
        except Exception as e:
            logger.error(f"PrecedentData 생성 실패 (판례본문 포함): {e}")
            return None
    
    def _collect_precedent_detail_with_retry(self, precedent_id: str, max_retries: int = 3) -> Dict[str, Any]:
        """판례본문 수집 (재시도 메커니즘 포함)"""
        import time
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"판례본문 수집 시도 {attempt + 1}/{max_retries}: {precedent_id}")
                
                # API 요청 간 지연 (API 부하 방지)
                if attempt > 0:
                    delay = min(2 ** attempt, 10)  # 지수 백오프, 최대 10초
                    logger.info(f"API 재시도 전 {delay}초 대기...")
                    time.sleep(delay)
                else:
                    # 첫 번째 요청도 0.5초 지연 (API 부하 방지, 속도 개선)
                    time.sleep(0.5)
                
                detail_response = self.client.get_precedent_detail(precedent_id=precedent_id)
                
                if detail_response:
                    # 판례본문 정보 추출
                    detail_info = self._extract_precedent_detail(detail_response)
                    logger.info(f"✅ 판례본문 수집 완료: {precedent_id}")
                    return detail_info
                else:
                    logger.warning(f"판례본문 API 응답 없음 (시도 {attempt + 1}/{max_retries}): {precedent_id}")
                    
            except Exception as e:
                logger.warning(f"판례본문 수집 오류 (시도 {attempt + 1}/{max_retries}): {precedent_id} - {e}")
                
                # 마지막 시도가 아니면 계속
                if attempt < max_retries - 1:
                    continue
                else:
                    logger.error(f"판례본문 수집 최종 실패: {precedent_id} - {e}")
        
        # 모든 시도 실패 시 빈 딕셔너리 반환
        logger.error(f"판례본문 수집 완전 실패: {precedent_id} (최대 재시도 횟수 초과)")
        return {'오류': f'판례본문 수집 실패 (재시도 {max_retries}회 초과)', '판례일련번호': precedent_id}
    
    def _extract_precedent_detail(self, detail_response: Dict[str, Any]) -> Dict[str, Any]:
        """판례 상세 응답에서 정보 추출"""
        try:
            extracted_info = {}
            
            # 다양한 API 응답 구조 처리
            if 'PrecService' in detail_response:
                prec_service = detail_response['PrecService']
                logger.debug(f"PrecService 구조 발견: {type(prec_service)}")
                
                # 실제 API 응답 구조에 맞게 수정
                # PrecService에 직접 판례 정보가 있음 (배열이 아님)
                if isinstance(prec_service, dict):
                    # 주요 정보 추출
                    extracted_info = {
                        '판례일련번호': prec_service.get('판례정보일련번호', ''),
                        '사건명': prec_service.get('사건명', ''),
                        '사건번호': prec_service.get('사건번호', ''),
                        '법원명': prec_service.get('법원명', ''),
                        '선고일자': prec_service.get('선고일자', ''),
                        '판결유형': prec_service.get('판결유형', ''),
                        '사건유형': prec_service.get('사건종류명', ''),
                        '판결요지': prec_service.get('판결요지', ''),
                        '판시사항': prec_service.get('판시사항', ''),
                        '참조조문': prec_service.get('참조조문', ''),
                        '참조판례': prec_service.get('참조판례', ''),
                        '판례내용': prec_service.get('판례내용', ''),
                        '사건종류코드': prec_service.get('사건종류코드', ''),
                        '법원종류코드': prec_service.get('법원종류코드', ''),
                        '선고': prec_service.get('선고', '')
                    }
                    
                    # 추가 메타데이터 (중복 제거)
                    extracted_info['수집일시'] = datetime.now().isoformat()
                    # API_응답_원본은 저장하지 않음 (중복 데이터 방지)
                    
                    logger.info(f"판례본문 정보 추출 성공: {extracted_info.get('사건명', 'Unknown')}")
                else:
                    logger.warning(f"PrecService가 딕셔너리가 아님: {type(prec_service)}")
                    
            elif 'Law' in detail_response:
                # Law 구조 처리 (국세청 판례 등)
                law_data = detail_response['Law']
                logger.debug(f"Law 구조 발견 (국세청 판례): {type(law_data)}")
                
                if isinstance(law_data, dict):
                    # Law 구조에서 정보 추출 시도
                    extracted_info = {
                        '판례일련번호': law_data.get('판례정보일련번호', ''),
                        '사건명': law_data.get('사건명', ''),
                        '사건번호': law_data.get('사건번호', ''),
                        '법원명': law_data.get('법원명', ''),
                        '선고일자': law_data.get('선고일자', ''),
                        '판결유형': law_data.get('판결유형', ''),
                        '사건유형': law_data.get('사건종류명', ''),
                        '판결요지': law_data.get('판결요지', ''),
                        '판시사항': law_data.get('판시사항', ''),
                        '참조조문': law_data.get('참조조문', ''),
                        '참조판례': law_data.get('참조판례', ''),
                        '판례내용': law_data.get('판례내용', ''),
                        '사건종류코드': law_data.get('사건종류코드', ''),
                        '법원종류코드': law_data.get('법원종류코드', ''),
                        '선고': law_data.get('선고', ''),
                        '데이터출처': '국세청'
                    }
                    
                    # 추가 메타데이터 (중복 제거)
                    extracted_info['수집일시'] = datetime.now().isoformat()
                    # API_응답_원본은 저장하지 않음 (중복 데이터 방지)
                    
                    logger.info(f"판례본문 정보 추출 성공 (국세청 판례): {extracted_info.get('사건명', 'Unknown')}")
                    
                elif isinstance(law_data, str):
                    extracted_info = {
                        '판례일련번호': '',
                        '사건명': '',
                        '사건번호': '',
                        '법원명': '',
                        '선고일자': '',
                        '판결유형': '',
                        '사건유형': '',
                        '판결요지': '',
                        '판시사항': '',
                        '참조조문': '',
                        '참조판례': '',
                        '판례내용': law_data[:1000] + '...' if len(law_data) > 1000 else law_data,  # HTML 내용 일부 저장
                        '사건종류코드': '',
                        '법원종류코드': '',
                        '선고': '',
                        '데이터출처': '국세청 (HTML 형태)',
                        '수집일시': datetime.now().isoformat(),
                        '오류': 'HTML 형태의 응답으로 JSON 파싱 불가'
                    }
                    
                    logger.info(f"판례본문 정보 추출 성공 (국세청 HTML): 길이 {len(law_data)}")
                    
                else:
                    logger.warning(f"Law가 예상치 못한 타입: {type(law_data)}")
                    extracted_info = {
                        '수집일시': datetime.now().isoformat(),
                        '오류': f"Law 타입 오류: {type(law_data)}"
                    }
                    
            else:
                logger.warning(f"알 수 없는 응답 구조: {list(detail_response.keys())}")
                # 전체 응답을 그대로 저장 (중복 제거)
                extracted_info = {
                    '수집일시': datetime.now().isoformat(),
                    '오류': f"알 수 없는 응답 구조: {list(detail_response.keys())}"
                }
                        
            return extracted_info
            
        except Exception as e:
            logger.error(f"판례 상세 정보 추출 실패: {e}")
            return {'오류': str(e)}
    
    def _categorize_precedent(self, precedent: Dict[str, Any]) -> PrecedentCategory:
        """판례 카테고리 분류 (사건유형코드 기반)"""
        case_type_code = precedent.get('사건유형코드', '')
        
        # 사건유형코드 기반 분류
        case_type_mapping = {
            '01': PrecedentCategory.CIVIL_CONTRACT,
            '02': PrecedentCategory.CRIMINAL,
            '03': PrecedentCategory.ADMINISTRATIVE,
            '04': PrecedentCategory.CIVIL_FAMILY,
            '05': PrecedentCategory.OTHER
        }
        
        return case_type_mapping.get(case_type_code, PrecedentCategory.OTHER)
    
    def _random_delay(self, min_seconds: float = None, max_seconds: float = None):
        """API 요청 간 랜덤 지연 - 사용자 설정 간격 사용"""
        if min_seconds is None:
            min_seconds = max(0.1, self.request_interval_base - self.request_interval_range)
        if max_seconds is None:
            max_seconds = self.request_interval_base + self.request_interval_range
        
        delay = random.uniform(min_seconds, max_seconds)
        logger.debug(f"API 요청 간 {delay:.2f}초 대기...")
        time.sleep(delay)
    
    def _collect_by_date_range(self, start_date: str, end_date: str, max_count: int, 
                              output_dir: Path, base_params: Dict[str, Any]) -> List[PrecedentData]:
        """날짜 범위로 판례 수집"""
        precedents = []
        page = 1
        consecutive_empty_pages = 0
        max_empty_pages = 3
        total_collected = 0
        total_duplicates = 0
        total_errors = 0
        
        # 무제한 모드 확인
        unlimited_mode = max_count >= 999999999
        
        if unlimited_mode:
            logger.info(f"📅 날짜 범위 {start_date} ~ {end_date} 수집 시작 (무제한 모드)")
        else:
            logger.info(f"📅 날짜 범위 {start_date} ~ {end_date} 수집 시작 (목표: {max_count:,}건)")
        
        logger.info("=" * 80)
        
        while (unlimited_mode or len(precedents) < max_count) and consecutive_empty_pages < max_empty_pages:
            try:
                # API 파라미터 구성 (올바른 prncYd 파라미터 사용)
                params = base_params.copy()
                params.update({
                    "from_date": start_date,
                    "to_date": end_date,
                    "display": 20,  # 배치 크기를 100에서 20으로 줄임 (판례본문 수집 시 속도 개선)
                    "page": page
                })
                
                # API 요청 간 지연
                if page > 1:
                    self._random_delay()
                
                # API 호출
                logger.info(f"🔍 페이지 {page} 요청 중... (날짜 필터링: {start_date}~{end_date})")
                results = self.client.get_precedent_list(**params)
                
                if not results:
                    consecutive_empty_pages += 1
                    logger.warning(f"⚠️  페이지 {page}: 결과 없음 (연속 빈 페이지: {consecutive_empty_pages}/{max_empty_pages})")
                    page += 1
                    continue
                else:
                    consecutive_empty_pages = 0
                
                # 결과 처리
                new_count = 0
                duplicate_count = 0
                page_precedents = []
                
                for result in results:
                    # 무제한 모드가 아닌 경우에만 건수 제한 확인
                    if not unlimited_mode and len(precedents) >= max_count:
                        break
                    
                    # 중복 확인 (개선된 로직)
                    if self._is_duplicate_precedent(result):
                        duplicate_count += 1
                        self.stats.duplicate_count += 1
                        continue
                    
                    # PrecedentData 객체 생성 (판례본문 포함 여부에 따라)
                    if self.include_details:
                        logger.info(f"📄 판례본문 수집 중: {result.get('사건명', 'Unknown')} (ID: {result.get('판례일련번호', 'N/A')})")
                        precedent_data = self._create_precedent_data_with_detail(result)
                    else:
                        precedent_data = self._create_precedent_data(result)
                    
                    if not precedent_data:
                        self.stats.failed_count += 1
                        total_errors += 1
                        continue
                    
                    # 신규 판례 추가
                    precedents.append(precedent_data)
                    page_precedents.append(precedent_data)
                    self._mark_precedent_collected(result)
                    new_count += 1
                
                # 페이지별 즉시 저장
                if page_precedents:
                    self._save_page_precedents(page_precedents, output_dir, page, start_date, end_date)
                
                # 통계 업데이트
                total_collected += new_count
                total_duplicates += duplicate_count
                
                # 진행상황 로그
                progress_percent = (len(precedents) / max_count * 100) if not unlimited_mode else 0
                logger.info(f"✅ 페이지 {page} 완료: 신규 {new_count:,}건, 중복 {duplicate_count:,}건")
                logger.info(f"📊 누적 현황: 총 {len(precedents):,}건 수집 (신규: {total_collected:,}, 중복: {total_duplicates:,}, 오류: {total_errors:,})")
                if not unlimited_mode:
                    logger.info(f"📈 진행률: {progress_percent:.1f}% ({len(precedents):,}/{max_count:,}건)")
                logger.info("-" * 60)
                
                page += 1
                
                # API 요청 제한 확인
                if self._check_api_limits():
                    break
                
            except Exception as e:
                total_errors += 1
                self.error_count += 1
                logger.error(f"❌ 페이지 {page} 오류 발생: {e}")
                logger.error(f"🔄 오류 카운트: {self.error_count}/{self.max_errors}")
                
                if self.error_count >= self.max_errors:
                    logger.error("🛑 최대 오류 수에 도달하여 수집 중단")
                    break
                
                # 오류 발생 시에도 현재까지 수집된 데이터 저장
                if precedents:
                    logger.info(f"💾 오류 발생 전까지 수집된 {len(precedents):,}건 저장 중...")
                    self._save_page_precedents(precedents, output_dir, page, start_date, end_date)
                
                continue
        
        # 최종 통계 로그
        logger.info("=" * 80)
        logger.info(f"🎉 날짜 범위 {start_date} ~ {end_date} 수집 완료!")
        logger.info(f"📊 최종 통계: 총 {len(precedents):,}건 수집 (신규: {total_collected:,}, 중복: {total_duplicates:,}, 오류: {total_errors:,})")
        logger.info("=" * 80)
        
        return precedents
    
    def _save_page_precedents(self, precedents: List[PrecedentData], output_dir: Path, 
                             page: int, start_date: str, end_date: str):
        """페이지별 판례 즉시 저장 (판례일련번호 기준)"""
        if not precedents:
            return
        
        try:
            # 카테고리별로 그룹화
            by_category = {}
            for precedent in precedents:
                category = precedent.category.value
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(precedent)
            
            # 카테고리별 파일 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            saved_files = []
            
            for category, category_precedents in by_category.items():
                # 판례일련번호 범위로 파일명 생성
                serial_numbers = [p.raw_data.get('판례일련번호', '') for p in category_precedents]
                serial_numbers = [s for s in serial_numbers if s]  # 빈 값 제거
                
                if serial_numbers:
                    # 판례일련번호 정렬
                    serial_numbers.sort()
                    start_serial = serial_numbers[0]
                    end_serial = serial_numbers[-1]
                    
                    # 파일명 생성 (카테고리 제거)
                    filename = f"page_{page:03d}_{start_serial}-{end_serial}_{len(category_precedents)}건_{timestamp}.json"
                else:
                    # 판례일련번호가 없는 경우 대체 파일명
                    filename = f"page_{page:03d}_{len(category_precedents)}건_{timestamp}.json"
                
                filepath = output_dir / filename
                
                # 판례 데이터 구성 (기본 정보 + 상세 정보)
                precedents_data = []
                for p in category_precedents:
                    precedent_data = p.raw_data.copy()
                    if p.detail_info:
                        precedent_data['detail_info'] = p.detail_info
                    precedents_data.append(precedent_data)
                
                batch_data = {
                    'metadata': {
                        'page': page,
                        'category': category,
                        'count': len(category_precedents),
                        'date_range': f"{start_date}~{end_date}",
                        'saved_at': datetime.now().isoformat(),
                        'batch_id': f"page_{page}_{timestamp}",
                        'serial_number_range': f"{start_serial}-{end_serial}" if serial_numbers else None,
                        'include_details': self.include_details
                    },
                    'precedents': precedents_data
                }
                
                with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
                    json.dump(batch_data, f, ensure_ascii=False, indent=2)
                
                saved_files.append(filepath)
                logger.info(f"💾 페이지 {page} 저장 완료: {category} 카테고리 {len(category_precedents):,}건 -> {filename}")
            
            # 통계 업데이트
            self.stats.saved_count += len(precedents)
            
            logger.info(f"📁 페이지 {page} 저장 완료: 총 {len(saved_files):,}개 파일")
            
        except Exception as e:
            logger.error(f"❌ 페이지 {page} 저장 실패: {e}")
            logger.error(traceback.format_exc())
    
    def _save_batch_precedents(self, output_dir: Path):
        """배치 단위로 판례 저장 (판례일련번호 기준)"""
        if not self.pending_precedents:
            return
        
        try:
            # 카테고리별로 그룹화
            by_category = {}
            for precedent in self.pending_precedents:
                category = precedent.category.value
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(precedent)
            
            # 카테고리별 파일 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            saved_files = []
            
            for category, precedents in by_category.items():
                # 판례일련번호 범위로 파일명 생성
                serial_numbers = [p.raw_data.get('판례일련번호', '') for p in precedents]
                serial_numbers = [s for s in serial_numbers if s]  # 빈 값 제거
                
                if serial_numbers:
                    # 판례일련번호 정렬
                    serial_numbers.sort()
                    start_serial = serial_numbers[0]
                    end_serial = serial_numbers[-1]
                    
                    # 안전한 파일명 생성
                    safe_category = category.replace('_', '-')
                    filename = f"batch_{safe_category}_{start_serial}-{end_serial}_{len(precedents)}건_{timestamp}.json"
                else:
                    # 판례일련번호가 없는 경우 대체 파일명
                    safe_category = category.replace('_', '-')
                    filename = f"batch_{safe_category}_{len(precedents)}건_{timestamp}.json"
                
                filepath = output_dir / filename
                
                batch_data = {
                    'metadata': {
                        'category': category,
                        'count': len(precedents),
                        'saved_at': datetime.now().isoformat(),
                        'batch_id': timestamp,
                        'serial_number_range': f"{start_serial}-{end_serial}" if serial_numbers else None
                    },
                    'precedents': [p.raw_data for p in precedents]
                }
                
                with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
                    json.dump(batch_data, f, ensure_ascii=False, indent=2)
                
                saved_files.append(filepath)
                logger.info(f"💾 배치 저장 완료: {category} 카테고리 {len(precedents):,}건 -> {filename}")
            
            # 통계 업데이트
            self.stats.saved_count += len(self.pending_precedents)
            
            # 임시 저장소 초기화
            self.pending_precedents = []
            
            logger.info(f"📁 배치 저장 완료: 총 {len(saved_files):,}개 파일")
            
        except Exception as e:
            logger.error(f"❌ 배치 저장 실패: {e}")
            logger.error(traceback.format_exc())
    
    def _check_api_limits(self) -> bool:
        """API 요청 제한 확인"""
        try:
            stats = self.client.get_request_stats()
            remaining = stats.get('remaining_requests', 0)
            if remaining < 100:
                logger.warning(f"API 요청 한도가 부족합니다. 남은 요청: {remaining}회")
                return True
        except Exception as e:
            logger.warning(f"API 요청 제한 확인 실패: {e}")
        return False
    
    def collect_by_yearly_strategy(self, years: List[int], target_per_year: int) -> Dict[str, Any]:
        """연도별 수집 전략"""
        logger.info(f"연도별 수집 시작: {years}년, 연간 목표 {target_per_year}건")
        
        all_precedents = []
        collection_summary = {
            'strategy': 'yearly',
            'years': years,
            'target_per_year': target_per_year,
            'collected_by_year': {},
            'total_collected': 0,
            'start_time': datetime.now().isoformat()
        }
        
        for year in years:
            if self.stats.collected_count >= self.stats.target_count:
                break
            
            year_str = str(year)
            start_date = f"{year_str}0101"
            end_date = f"{year_str}1231"
            
            # 해당 연도의 기존 데이터만 중복 확인하도록 수집기 재초기화
            logger.info(f"🔄 {year}년 중복 확인을 위한 데이터 로드 중...")
            self.collected_precedents.clear()  # 기존 중복 데이터 초기화
            self._load_existing_data(target_year=year)  # 해당 연도만 로드
            
            # 출력 디렉토리 생성
            output_dir = self._create_output_subdir(DateCollectionStrategy.YEARLY, year_str)
            
            logger.info(f"📊 {year}년 판례 수집 중... (목표: {target_per_year}건)")
            
            year_precedents = self._collect_by_date_range(
                start_date, end_date, target_per_year, output_dir,
                {"search": 1, "sort": "ddes"}
            )
            
            all_precedents.extend(year_precedents)
            collection_summary['collected_by_year'][year_str] = len(year_precedents)
            
            logger.info(f"✅ {year}년 완료: {len(year_precedents)}건 수집 (누적: {len(all_precedents)}건)")
        
        # 최종 배치 저장
        if self.pending_precedents:
            final_output_dir = self._create_output_subdir(DateCollectionStrategy.YEARLY, "final")
            self._save_batch_precedents(final_output_dir)
        
        collection_summary['total_collected'] = len(all_precedents)
        collection_summary['end_time'] = datetime.now().isoformat()
        
        # 수집 요약 저장
        summary_file = self.base_output_dir / f"yearly_collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(collection_summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"연도별 수집 완료: 총 {len(all_precedents)}건")
        return collection_summary
    
    def collect_by_quarterly_strategy(self, quarters: List[Tuple[str, str, str]], target_per_quarter: int) -> Dict[str, Any]:
        """분기별 수집 전략"""
        logger.info(f"분기별 수집 시작: {len(quarters)}개 분기, 분기당 목표 {target_per_quarter}건")
        
        all_precedents = []
        collection_summary = {
            'strategy': 'quarterly',
            'quarters': [q[0] for q in quarters],
            'target_per_quarter': target_per_quarter,
            'collected_by_quarter': {},
            'total_collected': 0,
            'start_time': datetime.now().isoformat()
        }
        
        for quarter_name, start_date, end_date in quarters:
            if self.stats.collected_count >= self.stats.target_count:
                break
            
            # 출력 디렉토리 생성
            output_dir = self._create_output_subdir(DateCollectionStrategy.QUARTERLY, quarter_name)
            
            logger.info(f"📊 {quarter_name} 판례 수집 중... (목표: {target_per_quarter}건)")
            
            quarter_precedents = self._collect_by_date_range(
                start_date, end_date, target_per_quarter, output_dir,
                {"search": 1, "sort": "ddes"}
            )
            
            all_precedents.extend(quarter_precedents)
            collection_summary['collected_by_quarter'][quarter_name] = len(quarter_precedents)
            
            logger.info(f"✅ {quarter_name} 완료: {len(quarter_precedents)}건 수집 (누적: {len(all_precedents)}건)")
        
        # 최종 배치 저장
        if self.pending_precedents:
            final_output_dir = self._create_output_subdir(DateCollectionStrategy.QUARTERLY, "final")
            self._save_batch_precedents(final_output_dir)
        
        collection_summary['total_collected'] = len(all_precedents)
        collection_summary['end_time'] = datetime.now().isoformat()
        
        # 수집 요약 저장
        summary_file = self.base_output_dir / f"quarterly_collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(collection_summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"분기별 수집 완료: 총 {len(all_precedents)}건")
        return collection_summary
    
    def collect_by_monthly_strategy(self, months: List[Tuple[str, str, str]], target_per_month: int) -> Dict[str, Any]:
        """월별 수집 전략"""
        logger.info(f"월별 수집 시작: {len(months)}개 월, 월간 목표 {target_per_month}건")
        
        all_precedents = []
        collection_summary = {
            'strategy': 'monthly',
            'months': [m[0] for m in months],
            'target_per_month': target_per_month,
            'collected_by_month': {},
            'total_collected': 0,
            'start_time': datetime.now().isoformat()
        }
        
        for month_name, start_date, end_date in months:
            if self.stats.collected_count >= self.stats.target_count:
                break
            
            # 출력 디렉토리 생성
            output_dir = self._create_output_subdir(DateCollectionStrategy.MONTHLY, month_name)
            
            logger.info(f"📊 {month_name} 판례 수집 중... (목표: {target_per_month}건)")
            
            month_precedents = self._collect_by_date_range(
                start_date, end_date, target_per_month, output_dir,
                {"search": 1, "sort": "ddes"}
            )
            
            all_precedents.extend(month_precedents)
            collection_summary['collected_by_month'][month_name] = len(month_precedents)
            
            logger.info(f"✅ {month_name} 완료: {len(month_precedents)}건 수집 (누적: {len(all_precedents)}건)")
        
        # 최종 배치 저장
        if self.pending_precedents:
            final_output_dir = self._create_output_subdir(DateCollectionStrategy.MONTHLY, "final")
            self._save_batch_precedents(final_output_dir)
        
        collection_summary['total_collected'] = len(all_precedents)
        collection_summary['end_time'] = datetime.now().isoformat()
        
        # 수집 요약 저장
        summary_file = self.base_output_dir / f"monthly_collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(collection_summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"월별 수집 완료: 총 {len(all_precedents)}건")
        return collection_summary
    
    def collect_by_weekly_strategy(self, weeks: List[Tuple[str, str, str]], target_per_week: int) -> Dict[str, Any]:
        """주별 수집 전략"""
        logger.info(f"주별 수집 시작: {len(weeks)}개 주, 주간 목표 {target_per_week}건")
        
        all_precedents = []
        collection_summary = {
            'strategy': 'weekly',
            'weeks': [w[0] for w in weeks],
            'target_per_week': target_per_week,
            'collected_by_week': {},
            'total_collected': 0,
            'start_time': datetime.now().isoformat()
        }
        
        for week_name, start_date, end_date in weeks:
            if self.stats.collected_count >= self.stats.target_count:
                break
            
            # 출력 디렉토리 생성
            output_dir = self._create_output_subdir(DateCollectionStrategy.WEEKLY, week_name)
            
            logger.info(f"📊 {week_name} 판례 수집 중... (목표: {target_per_week}건)")
            
            week_precedents = self._collect_by_date_range(
                start_date, end_date, target_per_week, output_dir,
                {"search": 1, "sort": "ddes"}
            )
            
            all_precedents.extend(week_precedents)
            collection_summary['collected_by_week'][week_name] = len(week_precedents)
            
            logger.info(f"✅ {week_name} 완료: {len(week_precedents)}건 수집 (누적: {len(all_precedents)}건)")
        
        # 최종 배치 저장
        if self.pending_precedents:
            final_output_dir = self._create_output_subdir(DateCollectionStrategy.WEEKLY, "final")
            self._save_batch_precedents(final_output_dir)
        
        collection_summary['total_collected'] = len(all_precedents)
        collection_summary['end_time'] = datetime.now().isoformat()
        
        # 수집 요약 저장
        summary_file = self.base_output_dir / f"weekly_collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(collection_summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"주별 수집 완료: 총 {len(all_precedents)}건")
        return collection_summary
    
    def generate_date_ranges(self, strategy: DateCollectionStrategy, count: int) -> List[Tuple[str, str, str]]:
        """날짜 범위 생성"""
        ranges = []
        current = datetime.now()
        
        if strategy == DateCollectionStrategy.YEARLY:
            for i in range(count):
                year = current.year - i
                ranges.append((f"{year}년", f"{year}0101", f"{year}1231"))
        
        elif strategy == DateCollectionStrategy.QUARTERLY:
            for i in range(count):
                target_date = current - timedelta(days=90*i)
                year = target_date.year
                quarter = (target_date.month - 1) // 3 + 1
                
                if quarter == 1:
                    start_date = f"{year}0101"
                    end_date = f"{year}0331"
                elif quarter == 2:
                    start_date = f"{year}0401"
                    end_date = f"{year}0630"
                elif quarter == 3:
                    start_date = f"{year}0701"
                    end_date = f"{year}0930"
                else:
                    start_date = f"{year}1001"
                    end_date = f"{year}1231"
                
                ranges.append((f"{year}Q{quarter}", start_date, end_date))
        
        elif strategy == DateCollectionStrategy.MONTHLY:
            for i in range(count):
                target_date = current - timedelta(days=30*i)
                year = target_date.year
                month = target_date.month
                
                start_date = datetime(year, month, 1)
                if month == 12:
                    end_date = datetime(year+1, 1, 1) - timedelta(days=1)
                else:
                    end_date = datetime(year, month+1, 1) - timedelta(days=1)
                
                ranges.append((
                    f"{year}년{month:02d}월",
                    start_date.strftime('%Y%m%d'),
                    end_date.strftime('%Y%m%d')
                ))
        
        elif strategy == DateCollectionStrategy.WEEKLY:
            for i in range(count):
                target_date = current - timedelta(weeks=i)
                start_of_week = target_date - timedelta(days=target_date.weekday())
                end_of_week = start_of_week + timedelta(days=6)
                
                ranges.append((
                    f"{start_of_week.strftime('%Y%m%d')}주",
                    start_of_week.strftime('%Y%m%d'),
                    end_of_week.strftime('%Y%m%d')
                ))
        
        return ranges
