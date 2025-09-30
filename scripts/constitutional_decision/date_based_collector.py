#!/usr/bin/env python3
"""
헌재결정례 날짜 기반 수집기

이 모듈은 날짜별로 체계적인 헌재결정례 수집을 수행합니다.
- 연도별, 분기별, 월별, 주별 수집 전략
- 결정일자 내림차순 최적화
- 폴더별 raw 데이터 저장 구조
- 중복 방지 및 체크포인트 지원
"""

import json
import time
import random
import hashlib
import traceback
import gc
import psutil
import os
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


@dataclass
class ConstitutionalDecisionData:
    """헌재결정례 데이터 클래스 - 목록 데이터 내부에 본문 데이터 포함"""
    # 목록 조회 API 응답 (기본 정보)
    id: str  # 검색결과번호
    사건번호: str
    종국일자: str
    헌재결정례일련번호: str
    사건명: str
    헌재결정례상세링크: str
    
    # 상세 조회 API 응답 (본문 데이터) - 목록 데이터 내부에 포함
    사건종류명: Optional[str] = None
    판시사항: Optional[str] = None
    결정요지: Optional[str] = None
    전문: Optional[str] = None
    참조조문: Optional[str] = None
    참조판례: Optional[str] = None
    심판대상조문: Optional[str] = None
    
    # 메타데이터
    document_type: str = "constitutional_decision"
    collected_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CollectionStats:
    """수집 통계 클래스"""
    total_collected: int = 0
    total_duplicates: int = 0
    total_errors: int = 0
    api_requests_made: int = 0
    api_errors: int = 0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    status: str = "PENDING"
    target_count: int = 0
    retry_delay: int = 5
    collected_decisions: Set[str] = field(default_factory=set)


class DateBasedConstitutionalCollector:
    """날짜 기반 헌재결정례 수집 클래스"""
    
    def __init__(self, config: LawOpenAPIConfig, base_output_dir: Optional[Path] = None):
        """
        날짜 기반 헌재결정례 수집기 초기화
        
        Args:
            config: API 설정 객체
            base_output_dir: 기본 출력 디렉토리 (기본값: data/raw/constitutional_decisions)
        """
        self.client = LawOpenAPIClient(config)
        self.base_output_dir = base_output_dir or Path("data/raw/constitutional_decisions")
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 관리 (메모리 최적화)
        self.collected_decisions: Set[str] = set()
        self.processed_date_ranges: Set[str] = set()
        self.pending_decisions: List[ConstitutionalDecisionData] = []
        self.max_memory_decisions = 10000  # 최대 메모리 보관 건수
        
        # 통계 및 상태
        self.stats = CollectionStats()
        
        # 에러 처리
        self.error_count = 0
        self.max_errors = 50
        
        # 기존 수집된 데이터 로드
        self._load_existing_data()
        
        # 시간 인터벌 설정 (기본값)
        self.request_interval_base = 2.0  # 기본 간격
        self.request_interval_range = 2.0  # 간격 범위
        
        # 체크포인트 재개 모드 (기본값: False)
        self.resume_mode = False
        
        logger.info("날짜 기반 헌재결정례 수집기 초기화 완료")
    
    def _monitor_memory_usage(self):
        """메모리 사용량 모니터링"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            if memory_mb > 1000:  # 1GB 이상 사용 시 경고
                logger.warning(f"메모리 사용량이 높습니다: {memory_mb:.1f}MB")
                self._cleanup_memory()
                
            return memory_mb
        except Exception as e:
            logger.debug(f"메모리 모니터링 오류: {e}")
            return 0
    
    def _cleanup_memory(self):
        """메모리 정리"""
        try:
            # 가비지 컬렉션 강제 실행
            collected = gc.collect()
            logger.debug(f"가비지 컬렉션 완료: {collected}개 객체 정리")
            
            # 대용량 데이터 구조 정리
            if len(self.collected_decisions) > self.max_memory_decisions:
                # 오래된 데이터 일부 제거 (최근 50%만 유지)
                sorted_decisions = sorted(self.collected_decisions)
                keep_count = len(sorted_decisions) // 2
                self.collected_decisions = set(sorted_decisions[-keep_count:])
                logger.info(f"메모리 정리: 수집된 결정례 {len(sorted_decisions) - keep_count}개 제거")
            
            # 대기 중인 데이터 정리
            if len(self.pending_decisions) > 1000:
                self.pending_decisions = self.pending_decisions[-500:]  # 최근 500개만 유지
                logger.info("메모리 정리: 대기 중인 결정례 데이터 정리")
                
        except Exception as e:
            logger.error(f"메모리 정리 중 오류: {e}")
    
    def _check_memory_and_cleanup(self):
        """메모리 체크 및 정리"""
        memory_mb = self._monitor_memory_usage()
        
        # 메모리 사용량이 높으면 정리
        if memory_mb > 800:  # 800MB 이상
            self._cleanup_memory()
            
        return memory_mb
    
    def set_request_interval(self, base_interval: float, interval_range: float):
        """API 요청 간격 설정"""
        self.request_interval_base = base_interval
        self.request_interval_range = interval_range
        logger.info(f"⏱️ 요청 간격 설정: {base_interval:.1f} ± {interval_range:.1f}초")
    
    def enable_resume_mode(self):
        """체크포인트 재개 모드 활성화"""
        self.resume_mode = True
        logger.info("🔄 체크포인트 재개 모드가 활성화되었습니다")
    
    def _load_existing_data(self, target_year: Optional[int] = None):
        """기존 수집된 데이터 로드"""
        try:
            if target_year:
                # 특정 연도 데이터만 로드
                pattern = f"yearly_{target_year}_*"
                existing_dirs = list(self.base_output_dir.glob(pattern))
            else:
                # 모든 기존 데이터 로드
                existing_dirs = list(self.base_output_dir.glob("*"))
            
            for dir_path in existing_dirs:
                if dir_path.is_dir():
                    json_files = list(dir_path.glob("page_*.json"))
                    for json_file in json_files:
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                decisions = data.get('decisions', [])
                                for decision in decisions:
                                    decision_id = decision.get('헌재결정례일련번호')
                                    if decision_id:
                                        self.collected_decisions.add(decision_id)
                        except Exception as e:
                            logger.warning(f"기존 데이터 로드 실패: {json_file} - {e}")
            
            logger.info(f"기존 수집된 헌재결정례: {len(self.collected_decisions)}건")
            
        except Exception as e:
            logger.error(f"기존 데이터 로드 중 오류: {e}")
    
    def _create_output_directory(self, strategy: DateCollectionStrategy, 
                               year: Optional[int] = None, 
                               quarter: Optional[int] = None,
                               month: Optional[int] = None,
                               week_start: Optional[str] = None) -> Path:
        """출력 디렉토리 생성 (체크포인트가 있으면 기존 디렉토리 사용)"""
        
        # 체크포인트가 있는 기존 디렉토리 찾기
        if self.resume_mode:
            existing_dir = self._find_existing_directory(strategy, year, quarter, month, week_start)
            if existing_dir:
                logger.info(f"🔄 기존 디렉토리 사용: {existing_dir}")
                return existing_dir
        
        # 새로운 디렉토리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if strategy == DateCollectionStrategy.YEARLY:
            if year:
                dir_name = f"yearly_{year}_{timestamp}"
            else:
                dir_name = f"yearly_collection_{timestamp}"
        elif strategy == DateCollectionStrategy.QUARTERLY:
            if year and quarter:
                dir_name = f"quarterly_{year}Q{quarter}_{timestamp}"
            else:
                dir_name = f"quarterly_collection_{timestamp}"
        elif strategy == DateCollectionStrategy.MONTHLY:
            if year and month:
                dir_name = f"monthly_{year}년{month}월_{timestamp}"
            else:
                dir_name = f"monthly_collection_{timestamp}"
        elif strategy == DateCollectionStrategy.WEEKLY:
            if week_start:
                dir_name = f"weekly_{week_start}주_{timestamp}"
            else:
                dir_name = f"weekly_collection_{timestamp}"
        else:
            dir_name = f"daily_collection_{timestamp}"
        
        output_dir = self.base_output_dir / dir_name
        
        # 디렉토리 생성 강화
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"출력 디렉토리 생성: {output_dir}")
        except Exception as e:
            logger.error(f"디렉토리 생성 실패: {output_dir} - {e}")
            raise
        
        return output_dir
    
    def _find_existing_directory(self, strategy: DateCollectionStrategy, 
                               year: Optional[int] = None, 
                               quarter: Optional[int] = None,
                               month: Optional[int] = None,
                               week_start: Optional[str] = None) -> Optional[Path]:
        """체크포인트가 있는 기존 디렉토리 찾기"""
        try:
            # 패턴 생성
            if strategy == DateCollectionStrategy.YEARLY and year:
                pattern = f"yearly_{year}_*"
            elif strategy == DateCollectionStrategy.QUARTERLY and year and quarter:
                pattern = f"quarterly_{year}Q{quarter}_*"
            elif strategy == DateCollectionStrategy.MONTHLY and year and month:
                pattern = f"monthly_{year}년{month}월_*"
            elif strategy == DateCollectionStrategy.WEEKLY and week_start:
                pattern = f"weekly_{week_start}주_*"
            else:
                return None
            
            # 해당 패턴의 디렉토리들 찾기
            matching_dirs = list(self.base_output_dir.glob(pattern))
            
            # 체크포인트 파일이 있는 디렉토리 찾기
            for dir_path in sorted(matching_dirs, reverse=True):  # 최신순으로 정렬
                checkpoint_file = dir_path / "checkpoint.json"
                if checkpoint_file.exists():
                    logger.info(f"📋 체크포인트 발견: {checkpoint_file}")
                    return dir_path
            
            logger.info(f"📋 체크포인트가 있는 디렉토리를 찾지 못했습니다. 패턴: {pattern}")
            return None
            
        except Exception as e:
            logger.warning(f"기존 디렉토리 찾기 실패: {e}")
            return None
    
    def _save_batch(self, decisions: List[ConstitutionalDecisionData], 
                   output_dir: Path, page_num: int, 
                   category: str = "constitutional") -> bool:
        """배치 데이터 저장"""
        try:
            if not decisions:
                return True
            
            # 출력 디렉토리 확인 및 생성
            if not output_dir.exists():
                logger.warning(f"출력 디렉토리가 존재하지 않습니다: {output_dir}")
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"출력 디렉토리 재생성: {output_dir}")
            
            # 파일명 생성
            start_id = decisions[0].헌재결정례일련번호
            end_id = decisions[-1].헌재결정례일련번호
            count = len(decisions)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"page_{page_num:03d}_{category}_{start_id}-{end_id}_{count}건_{timestamp}.json"
            file_path = output_dir / filename
            
            # 데이터 구조화
            batch_data = {
                "metadata": {
                    "category": category,
                    "page": page_num,
                    "count": count,
                    "start_id": start_id,
                    "end_id": end_id,
                    "collected_at": timestamp,
                    "strategy": "date_based"
                },
                "decisions": [decision.__dict__ for decision in decisions]
            }
            
            # JSON 파일 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 배치 저장 완료: {filename} ({count}건)")
            
            # 체크포인트 저장 (진행 상황 기록)
            self._save_checkpoint(output_dir, page_num, collected_count=len(decisions))
            
            return True
            
        except Exception as e:
            logger.error(f"배치 저장 실패: {e}")
            return False
    
    def _save_checkpoint(self, output_dir: Path, page_num: int, collected_count: int):
        """체크포인트 저장 (진행 상황 기록)"""
        try:
            checkpoint_data = {
                "checkpoint_info": {
                    "last_page": page_num,
                    "collected_count": collected_count,
                    "timestamp": datetime.now().isoformat(),
                    "status": "in_progress"
                }
            }
            
            checkpoint_file = output_dir / "checkpoint.json"
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.warning(f"체크포인트 저장 실패: {e}")
    
    def _load_checkpoint(self, output_dir: Path) -> dict:
        """체크포인트 로드 (중단된 수집 재개)"""
        try:
            checkpoint_file = output_dir / "checkpoint.json"
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                logger.info(f"📋 체크포인트 발견: 페이지 {checkpoint_data['checkpoint_info']['last_page']}, 수집된 건수 {checkpoint_data['checkpoint_info']['collected_count']}")
                return checkpoint_data
        except Exception as e:
            logger.warning(f"체크포인트 로드 실패: {e}")
        
        return None
    
    def _save_summary(self, output_dir: Path, strategy: DateCollectionStrategy, 
                     total_collected: int, total_duplicates: int, 
                     total_errors: int, duration: timedelta):
        """수집 요약 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_data = {
                "collection_info": {
                    "strategy": strategy.value,
                    "start_time": self.stats.start_time,
                    "end_time": self.stats.end_time,
                    "duration_seconds": duration.total_seconds(),
                    "duration_str": str(duration)
                },
                "statistics": {
                    "total_collected": total_collected,
                    "total_duplicates": total_duplicates,
                    "total_errors": total_errors,
                    "api_requests_made": self.stats.api_requests_made,
                    "api_errors": self.stats.api_errors,
                    "success_rate": (total_collected / (total_collected + total_errors)) * 100 if (total_collected + total_errors) > 0 else 0
                },
                "collected_decisions": list(self.collected_decisions),
                "metadata": {
                    "collected_at": timestamp,
                    "output_directory": str(output_dir),
                    "total_files": len(list(output_dir.glob("page_*.json")))
                }
            }
            
            summary_file = output_dir / f"{strategy.value}_collection_summary_{timestamp}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📊 수집 요약 저장: {summary_file.name}")
            
        except Exception as e:
            logger.error(f"수집 요약 저장 실패: {e}")
    
    def collect_by_year(self, year: int, target_count: Optional[int] = None, 
                       unlimited: bool = False, use_final_date: bool = False) -> bool:
        """특정 연도 헌재결정례 수집"""
        try:
            date_type = "종국일자" if use_final_date else "선고일자"
            logger.info(f"🗓️ {year}년 헌재결정례 수집 시작 ({date_type} 기준)")
            
            # 출력 디렉토리 생성
            output_dir = self._create_output_directory(DateCollectionStrategy.YEARLY, year=year)
            
            # 체크포인트 확인 (중단된 수집 재개)
            checkpoint = None
            start_page = 1
            if self.resume_mode:
                checkpoint = self._load_checkpoint(output_dir)
                if checkpoint:
                    start_page = checkpoint['checkpoint_info']['last_page'] + 1
                    logger.info(f"🔄 중단된 수집 재개: 페이지 {start_page}부터 시작")
                else:
                    logger.info("📋 체크포인트가 없습니다. 처음부터 시작합니다.")
            else:
                logger.info("🆕 새로운 수집을 시작합니다.")
            
            # 기존 데이터 로드 (특정 연도만)
            self._load_existing_data(target_year=year)
            
            # 날짜 범위 설정
            start_date = f"{year}0101"
            end_date = f"{year}1231"
            
            # 목표 건수 설정
            if unlimited:
                target_count = 999999  # 무제한
            elif target_count is None:
                target_count = 2000  # 기본값
            
            self.stats.target_count = target_count
            self.stats.start_time = datetime.now().isoformat()
            
            logger.info(f"📅 수집 기간: {start_date} ~ {end_date} ({date_type} 기준)")
            logger.info(f"🎯 목표 건수: {target_count:,}건")
            logger.info(f"📁 출력 디렉토리: {output_dir}")
            
            # 수집 실행
            success = self._collect_decisions_by_date_range(
                start_date=start_date,
                end_date=end_date,
                target_count=target_count,
                output_dir=output_dir,
                category=f"{year}년",
                use_final_date=use_final_date,
                start_page=start_page
            )
            
            # 수집 완료 처리
            self.stats.end_time = datetime.now().isoformat()
            duration = datetime.fromisoformat(self.stats.end_time) - datetime.fromisoformat(self.stats.start_time)
            
            # 요약 저장
            self._save_summary(
                output_dir=output_dir,
                strategy=DateCollectionStrategy.YEARLY,
                total_collected=self.stats.total_collected,
                total_duplicates=self.stats.total_duplicates,
                total_errors=self.stats.total_errors,
                duration=duration
            )
            
            logger.info(f"✅ {year}년 헌재결정례 수집 완료")
            logger.info(f"📊 수집 결과: {self.stats.total_collected:,}건 수집, {self.stats.total_duplicates:,}건 중복, {self.stats.total_errors:,}건 오류")
            logger.info(f"⏱️ 소요 시간: {duration}")
            
            return success
            
        except Exception as e:
            logger.error(f"{year}년 헌재결정례 수집 실패: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def collect_by_quarter(self, year: int, quarter: int, target_count: int = 500) -> bool:
        """특정 분기 헌재결정례 수집"""
        try:
            logger.info(f"🗓️ {year}년 {quarter}분기 헌재결정례 수집 시작")
            
            # 출력 디렉토리 생성
            output_dir = self._create_output_directory(DateCollectionStrategy.QUARTERLY, year=year, quarter=quarter)
            
            # 체크포인트 확인 (중단된 수집 재개)
            checkpoint = None
            start_page = 1
            if self.resume_mode:
                checkpoint = self._load_checkpoint(output_dir)
                if checkpoint:
                    start_page = checkpoint['checkpoint_info']['last_page'] + 1
                    logger.info(f"🔄 중단된 수집 재개: 페이지 {start_page}부터 시작")
                else:
                    logger.info("📋 체크포인트가 없습니다. 처음부터 시작합니다.")
            else:
                logger.info("🆕 새로운 수집을 시작합니다.")
            
            # 분기별 날짜 범위 설정
            quarter_months = {
                1: (1, 3),    # 1분기: 1-3월
                2: (4, 6),    # 2분기: 4-6월
                3: (7, 9),    # 3분기: 7-9월
                4: (10, 12)   # 4분기: 10-12월
            }
            
            start_month, end_month = quarter_months[quarter]
            start_date = f"{year}{start_month:02d}01"
            end_date = f"{year}{end_month:02d}31"
            
            self.stats.target_count = target_count
            self.stats.start_time = datetime.now().isoformat()
            
            logger.info(f"📅 수집 기간: {start_date} ~ {end_date}")
            logger.info(f"🎯 목표 건수: {target_count:,}건")
            logger.info(f"📁 출력 디렉토리: {output_dir}")
            
            # 수집 실행
            success = self._collect_decisions_by_date_range(
                start_date=start_date,
                end_date=end_date,
                target_count=target_count,
                output_dir=output_dir,
                category=f"{year}년{quarter}분기",
                start_page=start_page
            )
            
            # 수집 완료 처리
            self.stats.end_time = datetime.now().isoformat()
            duration = datetime.fromisoformat(self.stats.end_time) - datetime.fromisoformat(self.stats.start_time)
            
            # 요약 저장
            self._save_summary(
                output_dir=output_dir,
                strategy=DateCollectionStrategy.QUARTERLY,
                total_collected=self.stats.total_collected,
                total_duplicates=self.stats.total_duplicates,
                total_errors=self.stats.total_errors,
                duration=duration
            )
            
            logger.info(f"✅ {year}년 {quarter}분기 헌재결정례 수집 완료")
            logger.info(f"📊 수집 결과: {self.stats.total_collected:,}건 수집, {self.stats.total_duplicates:,}건 중복, {self.stats.total_errors:,}건 오류")
            logger.info(f"⏱️ 소요 시간: {duration}")
            
            return success
            
        except Exception as e:
            logger.error(f"{year}년 {quarter}분기 헌재결정례 수집 실패: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def collect_by_month(self, year: int, month: int, target_count: int = 200) -> bool:
        """특정 월 헌재결정례 수집"""
        try:
            logger.info(f"🗓️ {year}년 {month}월 헌재결정례 수집 시작")
            
            # 출력 디렉토리 생성
            output_dir = self._create_output_directory(DateCollectionStrategy.MONTHLY, year=year, month=month)
            
            # 체크포인트 확인 (중단된 수집 재개)
            checkpoint = None
            start_page = 1
            if self.resume_mode:
                checkpoint = self._load_checkpoint(output_dir)
                if checkpoint:
                    start_page = checkpoint['checkpoint_info']['last_page'] + 1
                    logger.info(f"🔄 중단된 수집 재개: 페이지 {start_page}부터 시작")
                else:
                    logger.info("📋 체크포인트가 없습니다. 처음부터 시작합니다.")
            else:
                logger.info("🆕 새로운 수집을 시작합니다.")
            
            # 월별 날짜 범위 설정
            start_date = f"{year}{month:02d}01"
            # 월말 날짜 계산
            if month == 12:
                end_date = f"{year}1231"
            else:
                next_month = month + 1
                next_year = year if next_month <= 12 else year + 1
                if next_month > 12:
                    next_month = 1
                end_date = f"{next_year}{next_month:02d}01"
                # 하루 전으로 설정
                end_date_obj = datetime.strptime(end_date, "%Y%m%d") - timedelta(days=1)
                end_date = end_date_obj.strftime("%Y%m%d")
            
            self.stats.target_count = target_count
            self.stats.start_time = datetime.now().isoformat()
            
            logger.info(f"📅 수집 기간: {start_date} ~ {end_date}")
            logger.info(f"🎯 목표 건수: {target_count:,}건")
            logger.info(f"📁 출력 디렉토리: {output_dir}")
            
            # 수집 실행
            success = self._collect_decisions_by_date_range(
                start_date=start_date,
                end_date=end_date,
                target_count=target_count,
                output_dir=output_dir,
                category=f"{year}년{month}월",
                start_page=start_page
            )
            
            # 수집 완료 처리
            self.stats.end_time = datetime.now().isoformat()
            duration = datetime.fromisoformat(self.stats.end_time) - datetime.fromisoformat(self.stats.start_time)
            
            # 요약 저장
            self._save_summary(
                output_dir=output_dir,
                strategy=DateCollectionStrategy.MONTHLY,
                total_collected=self.stats.total_collected,
                total_duplicates=self.stats.total_duplicates,
                total_errors=self.stats.total_errors,
                duration=duration
            )
            
            logger.info(f"✅ {year}년 {month}월 헌재결정례 수집 완료")
            logger.info(f"📊 수집 결과: {self.stats.total_collected:,}건 수집, {self.stats.total_duplicates:,}건 중복, {self.stats.total_errors:,}건 오류")
            logger.info(f"⏱️ 소요 시간: {duration}")
            
            return success
            
        except Exception as e:
            logger.error(f"{year}년 {month}월 헌재결정례 수집 실패: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _collect_decisions_by_date_range(self, start_date: str, end_date: str, 
                                       target_count: int, output_dir: Path, 
                                       category: str, use_final_date: bool = False, 
                                       start_page: int = 1) -> bool:
        """날짜 범위별 헌재결정례 수집 (선고일자 내림차순)"""
        try:
            page = start_page
            collected_count = 0
            batch_decisions = []
            date_type = "종국일자" if use_final_date else "선고일자"
            
            while collected_count < target_count:
                try:
                    # 메모리 체크 및 정리 (매 10페이지마다)
                    if page % 10 == 0:
                        memory_mb = self._check_memory_and_cleanup()
                        logger.info(f"🧠 메모리 사용량: {memory_mb:.1f}MB")
                    
                    # API 요청 지연 - 사용자 설정 간격 사용
                    min_interval = max(0.1, self.request_interval_base - self.request_interval_range)
                    max_interval = self.request_interval_base + self.request_interval_range
                    delay = random.uniform(min_interval, max_interval)
                    time.sleep(delay)
                    
                    # 헌재결정례 목록 조회 (선고일자 내림차순)
                    logger.info(f"📄 페이지 {page} 조회 중... (수집된 건수: {collected_count:,}/{target_count:,}) [{date_type} 기준]")
                    
                    # 종국일자 기준 수집인 경우 edYd 파라미터 사용
                    if use_final_date:
                        # 종국일자 기간 검색 (edYd: YYYYMMDD-YYYYMMDD 형식)
                        edYd_range = f"{start_date}-{end_date}"
                        results = self.client.get_constitutional_list(
                            query=None,  # 키워드 없이 날짜 범위로만 검색
                            display=100,  # 페이지당 최대 건수
                            page=page,
                            search=1,  # 검색 범위
                            sort="efdes",  # 종국일자 내림차순 정렬
                            edYd=edYd_range  # 종국일자 기간 검색
                        )
                    else:
                        # 선고일자 기준 수집 (기존 방식)
                        results = self.client.get_constitutional_list(
                            query=None,  # 키워드 없이 날짜 범위로만 검색
                            display=100,  # 페이지당 최대 건수
                            page=page,
                            from_date=start_date,
                            to_date=end_date,
                            search=1,  # 검색 범위
                            sort="ddes"  # 선고일자 내림차순 정렬
                        )
                    
                    if not results:
                        logger.info(f"📄 페이지 {page}: 더 이상 데이터가 없습니다.")
                        break
                    
                    new_decisions = 0
                    for result in results:
                        if collected_count >= target_count:
                            break
                        
                        # 헌재결정례 ID 확인
                        decision_id = result.get('헌재결정례일련번호')
                        if not decision_id:
                            continue
                        
                        # 중복 확인
                        if decision_id in self.collected_decisions:
                            self.stats.total_duplicates += 1
                            continue
                        
                        # 상세 정보 조회
                        try:
                            detail = self.client.get_constitutional_detail(decision_id)
                            
                            # ConstitutionalDecisionData 객체 생성 (목록 데이터 내부에 본문 데이터 포함)
                            decision_data = ConstitutionalDecisionData(
                                # 목록 조회 API 응답 (기본 정보)
                                id=result.get('id', ''),
                                사건번호=result.get('사건번호', ''),
                                종국일자=result.get('종국일자', ''),
                                헌재결정례일련번호=decision_id,
                                사건명=result.get('사건명', ''),
                                헌재결정례상세링크=result.get('헌재결정례상세링크', ''),
                                
                                # 상세 조회 API 응답 (본문 데이터)
                                사건종류명=detail.get('사건종류명') if detail else None,
                                판시사항=detail.get('판시사항') if detail else None,
                                결정요지=detail.get('결정요지') if detail else None,
                                전문=detail.get('전문') if detail else None,
                                참조조문=detail.get('참조조문') if detail else None,
                                참조판례=detail.get('참조판례') if detail else None,
                                심판대상조문=detail.get('심판대상조문') if detail else None,
                                
                                # 메타데이터
                                document_type="constitutional_decision",
                                collected_at=datetime.now().isoformat()
                            )
                            
                            batch_decisions.append(decision_data)
                            self.collected_decisions.add(decision_id)
                            collected_count += 1
                            new_decisions += 1
                            
                            logger.info(f"✅ 새로운 헌재결정례 수집: {decision_data.사건명} (ID: {decision_id})")
                            
                            # 배치 단위로 중간 저장 (10건마다)
                            if len(batch_decisions) >= 10:
                                self._save_batch(batch_decisions, output_dir, page, category)
                                batch_decisions = []  # 배치 초기화
                            
                        except Exception as e:
                            logger.error(f"상세 정보 조회 실패: {decision_id} - {e}")
                            # 상세 정보 조회 실패해도 기본 정보는 저장
                            decision_data = ConstitutionalDecisionData(
                                # 목록 조회 API 응답 (기본 정보)
                                id=result.get('id', ''),
                                사건번호=result.get('사건번호', ''),
                                종국일자=result.get('종국일자', ''),
                                헌재결정례일련번호=decision_id,
                                사건명=result.get('사건명', ''),
                                헌재결정례상세링크=result.get('헌재결정례상세링크', ''),
                                
                                # 상세 조회 실패로 본문 데이터는 None
                                사건종류명=None,
                                판시사항=None,
                                결정요지=None,
                                전문=None,
                                참조조문=None,
                                참조판례=None,
                                심판대상조문=None,
                                
                                # 메타데이터
                                document_type="constitutional_decision",
                                collected_at=datetime.now().isoformat()
                            )
                            
                            batch_decisions.append(decision_data)
                            self.collected_decisions.add(decision_id)
                            collected_count += 1
                            new_decisions += 1
                            
                            logger.warning(f"⚠️ 기본 정보만 수집: {decision_data.사건명} (ID: {decision_id})")
                            
                            # 배치 단위로 중간 저장 (10건마다)
                            if len(batch_decisions) >= 10:
                                self._save_batch(batch_decisions, output_dir, page, category)
                                batch_decisions = []  # 배치 초기화
                            self.stats.total_errors += 1
                    
                    # 배치 저장 (100건마다) - 추가 안전장치
                    if len(batch_decisions) >= 100:
                        self._save_batch(batch_decisions, output_dir, page, category)
                        batch_decisions = []
                    
                    logger.info(f"📄 페이지 {page} 완료: {new_decisions}건의 새로운 결정례 수집")
                    logger.info(f"   📊 누적 수집: {collected_count:,}/{target_count:,}건 ({collected_count/target_count*100:.1f}%)")
                    
                    page += 1
                    self.stats.api_requests_made += 1
                    
                    # API 요청 제한 확인
                    stats = self.client.get_request_stats()
                    if stats['remaining_requests'] < 10:
                        logger.warning("API 요청 한도가 거의 소진되었습니다.")
                        break
                    
                except Exception as e:
                    logger.error(f"페이지 {page} 처리 중 오류: {e}")
                    self.stats.api_errors += 1
                    self.error_count += 1
                    
                    if self.error_count >= self.max_errors:
                        logger.error(f"최대 오류 횟수({self.max_errors})에 도달했습니다.")
                        break
                    
                    # 재시도 지연
                    time.sleep(self.stats.retry_delay)
                    continue
            
            # 남은 배치 저장
            if batch_decisions:
                self._save_batch(batch_decisions, output_dir, page, category)
            
            self.stats.total_collected = collected_count
            logger.info(f"🎯 수집 완료: {collected_count:,}건 수집")
            
            return True
            
        except Exception as e:
            logger.error(f"날짜 범위별 수집 실패: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def collect_multiple_years(self, start_year: int, end_year: int, 
                              target_per_year: int = 2000) -> bool:
        """여러 연도 헌재결정례 수집"""
        try:
            logger.info(f"🗓️ {start_year}년 ~ {end_year}년 헌재결정례 수집 시작")
            
            total_success = True
            for year in range(start_year, end_year + 1):
                logger.info(f"📅 {year}년 수집 시작...")
                success = self.collect_by_year(year, target_per_year)
                if not success:
                    logger.warning(f"⚠️ {year}년 수집 실패")
                    total_success = False
                else:
                    logger.info(f"✅ {year}년 수집 완료")
                
                # 연도 간 지연
                time.sleep(5)
            
            logger.info(f"🎯 다중 연도 수집 완료: {start_year}년 ~ {end_year}년")
            return total_success
            
        except Exception as e:
            logger.error(f"다중 연도 수집 실패: {e}")
            logger.error(traceback.format_exc())
            return False
