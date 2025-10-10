#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
법령해석례 날짜 기반 수집기

이 모듈은 날짜별로 체계적인 법령해석례 수집을 수행합니다.
- 연도별, 분기별, 월별 수집 전략
- 해석일자 내림차순 최적화
- 폴더별 raw 데이터 저장 구조
- 중복 방지 및 체크포인트 지원
"""

import sys
import os
import json
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

# source 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'source'))

from data.law_open_api_client import LawOpenAPIClient
import logging

logger = logging.getLogger(__name__)


class DateCollectionStrategy(Enum):
    """날짜 수집 전략"""
    YEARLY = "yearly"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"


@dataclass
class CollectionConfig:
    """수집 설정 클래스"""
    base_output_dir: Path = Path("data/raw/legal_interpretations")
    max_retries: int = 3
    retry_delay: int = 5
    api_delay_range: Tuple[float, float] = (1.0, 3.0)
    output_subdir: Optional[str] = None


@dataclass
class LegalInterpretationData:
    """법령해석례 데이터 클래스 - 목록 데이터 내부에 본문 데이터 포함"""
    # 목록 조회 API 응답 (기본 정보)
    id: str  # 검색결과번호
    법령해석례일련번호: str
    안건명: str
    안건번호: str
    질의기관코드: str
    질의기관명: str
    회신기관코드: str
    회신기관명: str
    회신일자: str
    법령해석례상세링크: str
    
    # 상세 조회 API 응답 (본문 데이터)
    해석일자: Optional[str] = None
    해석기관코드: Optional[str] = None
    해석기관명: Optional[str] = None
    관리기관코드: Optional[str] = None
    등록일시: Optional[str] = None
    질의요지: Optional[str] = None
    회답: Optional[str] = None
    이유: Optional[str] = None
    
    # 메타데이터
    document_type: str = "legal_interpretation"
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


class DateBasedLegalInterpretationCollector:
    """날짜 기반 법령해석례 수집 클래스"""
    
    def __init__(self, config: CollectionConfig = None):
        self.config = config or CollectionConfig()
        # 간단한 설정으로 API 클라이언트 생성
        class SimpleConfig:
            def __init__(self):
                self.law_open_api_oc = os.getenv('LAW_OPEN_API_OC', '{OC}')
                self.oc = os.getenv('LAW_OPEN_API_OC', '{OC}')  # API 클라이언트가 oc 속성을 찾음
                self.base_url = 'https://www.law.go.kr/DRF/'
                self.rate_limit = 1000
                self.request_timeout = 30
                self.connect_timeout = 30
                self.timeout = 30
                self.max_retries = 3
                self.retry_delay = 5
                self.retry_delay_base = 1
                self.retry_delay_max = 60
                self.law_firm_ai_api_key = os.getenv('LAW_FIRM_AI_API_KEY', 'your-api-key-here')
                self.api_host = '0.0.0.0'
                self.api_port = 8000
                self.debug = False
                self.database_url = 'sqlite:///./data/lawfirm.db'
                self.database_path = './data/lawfirm.db'
                self.model_path = './models'
                self.device = 'cpu'
                self.model_cache_dir = './model_cache'
                self.chroma_db_path = './data/chroma_db'
                self.embedding_model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        
        api_config = SimpleConfig()
        self.client = LawOpenAPIClient(api_config)
        self.stats = CollectionStats()
        self.collected_decisions: Set[str] = set()
        
        # 출력 디렉토리 생성
        self.config.base_output_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_output_directory(self, strategy: DateCollectionStrategy, 
                               year: int = None, quarter: int = None, 
                               month: int = None) -> Path:
        """출력 디렉토리 생성"""
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
                dir_name = f"monthly_{year}{month:02d}_{timestamp}"
            else:
                dir_name = f"monthly_collection_{timestamp}"
        else:
            dir_name = f"daily_collection_{timestamp}"
        
        output_dir = self.config.base_output_dir / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _save_batch(self, interpretations: List[LegalInterpretationData], 
                   output_dir: Path, page_num: int, 
                   category: str = "legal_interpretation") -> bool:
        """배치 데이터 저장"""
        try:
            if not interpretations:
                return True
            
            # 파일명 생성
            start_id = interpretations[0].법령해석례일련번호
            end_id = interpretations[-1].법령해석례일련번호
            count = len(interpretations)
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
                "interpretations": [interpretation.__dict__ for interpretation in interpretations]
            }
            
            # JSON 파일 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 배치 저장 완료: {filename} ({count}건)")
            
            # 체크포인트 저장 (진행 상황 기록)
            self._save_checkpoint(output_dir, page_num, collected_count=len(interpretations))
            
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
                    "success_rate": (total_collected / (total_collected + total_errors) * 100) if (total_collected + total_errors) > 0 else 0
                },
                "collected_interpretations": list(self.collected_decisions),
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
                       unlimited: bool = False, use_interpretation_date: bool = True) -> bool:
        """특정 연도 법령해석례 수집"""
        try:
            date_type = "해석일자" if use_interpretation_date else "회신일자"
            logger.info(f"🗓️ {year}년 법령해석례 수집 시작 ({date_type} 기준)")
            
            # 출력 디렉토리 생성
            output_dir = self._create_output_directory(DateCollectionStrategy.YEARLY, year=year)
            
            # 체크포인트 확인 (중단된 수집 재개)
            checkpoint = self._load_checkpoint(output_dir)
            start_page = 1
            if checkpoint:
                start_page = checkpoint['checkpoint_info']['last_page'] + 1
                logger.info(f"🔄 중단된 수집 재개: 페이지 {start_page}부터 시작")
            
            # 기존 데이터 로드 (특정 연도만)
            self._load_existing_data(target_year=year)
            
            # 날짜 범위 설정
            start_date = f"{year}0101"
            end_date = f"{year}1231"
            
            # 목표 건수 설정
            if unlimited:
                target_count = 2000  # 무제한이지만 안전을 위해 제한
            elif target_count is None:
                target_count = 1000  # 기본값
            
            self.stats.start_time = datetime.now().isoformat()
            self.stats.target_count = target_count
            self.stats.status = "RUNNING"
            
            logger.info(f"📅 수집 기간: {start_date} ~ {end_date} ({date_type} 기준)")
            logger.info(f"🎯 목표 건수: {target_count:,}건")
            logger.info(f"📁 출력 디렉토리: {output_dir}")
            
            # 수집 실행
            success = self._collect_interpretations_by_date_range(
                start_date=start_date,
                end_date=end_date,
                target_count=target_count,
                output_dir=output_dir,
                category=f"{year}년",
                use_interpretation_date=use_interpretation_date,
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
            
            logger.info(f"✅ {year}년 법령해석례 수집 완료")
            logger.info(f"📊 수집 결과: {self.stats.total_collected:,}건 수집, {self.stats.total_duplicates:,}건 중복, {self.stats.total_errors:,}건 오류")
            logger.info(f"⏱️ 소요 시간: {duration}")
            
            return success
            
        except Exception as e:
            logger.error(f"{year}년 법령해석례 수집 실패: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def collect_by_quarter(self, year: int, quarter: int, target_count: int = 500) -> bool:
        """특정 분기 법령해석례 수집"""
        try:
            logger.info(f"🗓️ {year}년 {quarter}분기 법령해석례 수집 시작")
            
            # 출력 디렉토리 생성
            output_dir = self._create_output_directory(DateCollectionStrategy.QUARTERLY, year=year, quarter=quarter)
            
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
            
            self.stats.start_time = datetime.now().isoformat()
            self.stats.target_count = target_count
            self.stats.status = "RUNNING"
            
            logger.info(f"📅 수집 기간: {start_date} ~ {end_date}")
            logger.info(f"🎯 목표 건수: {target_count:,}건")
            logger.info(f"📁 출력 디렉토리: {output_dir}")
            
            # 수집 실행
            success = self._collect_interpretations_by_date_range(
                start_date=start_date,
                end_date=end_date,
                target_count=target_count,
                output_dir=output_dir,
                category=f"{year}년{quarter}분기",
                start_page=1
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
            
            logger.info(f"✅ {year}년 {quarter}분기 법령해석례 수집 완료")
            logger.info(f"📊 수집 결과: {self.stats.total_collected:,}건 수집, {self.stats.total_duplicates:,}건 중복, {self.stats.total_errors:,}건 오류")
            logger.info(f"⏱️ 소요 시간: {duration}")
            
            return success
            
        except Exception as e:
            logger.error(f"{year}년 {quarter}분기 법령해석례 수집 실패: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def collect_by_month(self, year: int, month: int, target_count: int = 200) -> bool:
        """특정 월 법령해석례 수집"""
        try:
            logger.info(f"🗓️ {year}년 {month}월 법령해석례 수집 시작")
            
            # 출력 디렉토리 생성
            output_dir = self._create_output_directory(DateCollectionStrategy.MONTHLY, year=year, month=month)
            
            # 월별 날짜 범위 설정
            start_date = f"{year}{month:02d}01"
            # 월말 날짜 계산
            if month == 12:
                end_date = f"{year}1231"
            else:
                next_month = month + 1
                end_date = f"{year}{next_month:02d}01"
            
            self.stats.start_time = datetime.now().isoformat()
            self.stats.target_count = target_count
            self.stats.status = "RUNNING"
            
            logger.info(f"📅 수집 기간: {start_date} ~ {end_date}")
            logger.info(f"🎯 목표 건수: {target_count:,}건")
            logger.info(f"📁 출력 디렉토리: {output_dir}")
            
            # 수집 실행
            success = self._collect_interpretations_by_date_range(
                start_date=start_date,
                end_date=end_date,
                target_count=target_count,
                output_dir=output_dir,
                category=f"{year}년{month}월",
                start_page=1
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
            
            logger.info(f"✅ {year}년 {month}월 법령해석례 수집 완료")
            logger.info(f"📊 수집 결과: {self.stats.total_collected:,}건 수집, {self.stats.total_duplicates:,}건 중복, {self.stats.total_errors:,}건 오류")
            logger.info(f"⏱️ 소요 시간: {duration}")
            
            return success
            
        except Exception as e:
            logger.error(f"{year}년 {month}월 법령해석례 수집 실패: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _collect_interpretations_by_date_range(self, start_date: str, end_date: str, 
                                             target_count: int, output_dir: Path, 
                                             category: str, use_interpretation_date: bool = True, 
                                             start_page: int = 1) -> bool:
        """날짜 범위별 법령해석례 수집 (해석일자 내림차순)"""
        try:
            page = start_page
            collected_count = 0
            batch_interpretations = []
            date_type = "해석일자" if use_interpretation_date else "회신일자"
            
            while collected_count < target_count:
                try:
                    logger.info(f"📄 페이지 {page} 처리 중...")
                    
                    # API 호출
                    if use_interpretation_date:
                        # 해석일자 기준 검색
                        interpretations = self.client.get_legal_interpretation_list(
                            page=page,
                            display=100,
                            explYd=f"{start_date}~{end_date}",
                            sort="ddes"  # 해석일자 내림차순
                        )
                    else:
                        # 회신일자 기준 검색
                        interpretations = self.client.get_legal_interpretation_list(
                            page=page,
                            display=100,
                            regYd=f"{start_date}~{end_date}",
                            sort="ddes"  # 회신일자 내림차순
                        )
                    
                    if not interpretations:
                        logger.info(f"📄 페이지 {page}: 더 이상 데이터가 없습니다.")
                        break
                    
                    self.stats.api_requests_made += 1
                    new_interpretations = 0
                    
                    for result in interpretations:
                        interpretation_id = result.get('법령해석례일련번호', '')
                        
                        if not interpretation_id:
                            continue
                        
                        # 중복 확인
                        if interpretation_id in self.collected_decisions:
                            self.stats.total_duplicates += 1
                            continue
                        
                        # 상세 정보 조회
                        try:
                            detail = self.client.get_legal_interpretation_detail(interpretation_id)
                            
                            # LegalInterpretationData 객체 생성 (목록 데이터 내부에 본문 데이터 포함)
                            interpretation_data = LegalInterpretationData(
                                # 목록 조회 API 응답 (기본 정보)
                                id=result.get('id', ''),
                                법령해석례일련번호=interpretation_id,
                                안건명=result.get('안건명', ''),
                                안건번호=result.get('안건번호', ''),
                                질의기관코드=result.get('질의기관코드', ''),
                                질의기관명=result.get('질의기관명', ''),
                                회신기관코드=result.get('회신기관코드', ''),
                                회신기관명=result.get('회신기관명', ''),
                                회신일자=result.get('회신일자', ''),
                                법령해석례상세링크=result.get('법령해석례상세링크', ''),
                                
                                # 상세 조회 API 응답 (본문 데이터)
                                해석일자=detail.get('해석일자') if detail else None,
                                해석기관코드=detail.get('해석기관코드') if detail else None,
                                해석기관명=detail.get('해석기관명') if detail else None,
                                관리기관코드=detail.get('관리기관코드') if detail else None,
                                등록일시=detail.get('등록일시') if detail else None,
                                질의요지=detail.get('질의요지') if detail else None,
                                회답=detail.get('회답') if detail else None,
                                이유=detail.get('이유') if detail else None,
                                
                                # 메타데이터
                                document_type="legal_interpretation",
                                collected_at=datetime.now().isoformat()
                            )
                            
                            batch_interpretations.append(interpretation_data)
                            self.collected_decisions.add(interpretation_id)
                            collected_count += 1
                            new_interpretations += 1
                            
                            logger.info(f"✅ 새로운 법령해석례 수집: {interpretation_data.안건명} (ID: {interpretation_id})")
                            
                            # 배치 단위로 중간 저장 (10건마다)
                            if len(batch_interpretations) >= 10:
                                self._save_batch(batch_interpretations, output_dir, page, category)
                                batch_interpretations = []  # 배치 초기화
                            
                        except Exception as e:
                            logger.error(f"상세 정보 조회 실패: {interpretation_id} - {e}")
                            # 상세 정보 조회 실패해도 기본 정보는 저장
                            interpretation_data = LegalInterpretationData(
                                # 목록 조회 API 응답 (기본 정보)
                                id=result.get('id', ''),
                                법령해석례일련번호=interpretation_id,
                                안건명=result.get('안건명', ''),
                                안건번호=result.get('안건번호', ''),
                                질의기관코드=result.get('질의기관코드', ''),
                                질의기관명=result.get('질의기관명', ''),
                                회신기관코드=result.get('회신기관코드', ''),
                                회신기관명=result.get('회신기관명', ''),
                                회신일자=result.get('회신일자', ''),
                                법령해석례상세링크=result.get('법령해석례상세링크', ''),
                                
                                # 상세 조회 실패로 본문 데이터는 None
                                해석일자=None,
                                해석기관코드=None,
                                해석기관명=None,
                                관리기관코드=None,
                                등록일시=None,
                                질의요지=None,
                                회답=None,
                                이유=None,
                                
                                # 메타데이터
                                document_type="legal_interpretation",
                                collected_at=datetime.now().isoformat()
                            )
                            
                            batch_interpretations.append(interpretation_data)
                            self.collected_decisions.add(interpretation_id)
                            collected_count += 1
                            new_interpretations += 1
                            
                            logger.warning(f"⚠️ 기본 정보만 수집: {interpretation_data.안건명} (ID: {interpretation_id})")
                            
                            # 배치 단위로 중간 저장 (10건마다)
                            if len(batch_interpretations) >= 10:
                                self._save_batch(batch_interpretations, output_dir, page, category)
                                batch_interpretations = []  # 배치 초기화
                            self.stats.total_errors += 1
                    
                    # 배치 저장 (100건마다) - 추가 안전장치
                    if len(batch_interpretations) >= 100:
                        self._save_batch(batch_interpretations, output_dir, page, category)
                        batch_interpretations = []
                    
                    logger.info(f"📄 페이지 {page} 완료: {new_interpretations}건의 새로운 해석례 수집")
                    logger.info(f"   📊 누적 수집: {collected_count:,}/{target_count:,}건 ({collected_count/target_count*100:.1f}%)")
                    
                    page += 1
                    
                    # API 호출 간격 조절
                    time.sleep(1.0)
                    
                except Exception as e:
                    logger.error(f"페이지 {page} 처리 중 오류: {e}")
                    self.stats.total_errors += 1
                    
                    # 재시도 로직
                    if self.stats.total_errors < self.config.max_retries:
                        logger.info(f"재시도 중... ({self.stats.total_errors}/{self.config.max_retries})")
                        time.sleep(self.stats.retry_delay)
                        continue
                    else:
                        logger.error(f"최대 재시도 횟수 초과. 수집을 중단합니다.")
                        break
            
            # 남은 배치 저장
            if batch_interpretations:
                self._save_batch(batch_interpretations, output_dir, page, category)
            
            logger.info(f"✅ 날짜 범위별 수집 완료: {collected_count:,}건 수집")
            return True
            
        except Exception as e:
            logger.error(f"날짜 범위별 수집 실패: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _load_existing_data(self, target_year: int = None):
        """기존 수집된 데이터 로드 (중복 방지)"""
        try:
            # 기존 데이터 디렉토리 스캔
            for year_dir in self.config.base_output_dir.glob("yearly_*"):
                if target_year and str(target_year) not in str(year_dir):
                    continue
                
                # JSON 파일들에서 이미 수집된 ID 추출
                for json_file in year_dir.glob("page_*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        if 'interpretations' in data:
                            for interpretation in data['interpretations']:
                                interpretation_id = interpretation.get('법령해석례일련번호', '')
                                if interpretation_id:
                                    self.collected_decisions.add(interpretation_id)
                    except Exception as e:
                        logger.warning(f"기존 데이터 로드 실패: {json_file} - {e}")
            
            logger.info(f"📋 기존 수집된 법령해석례: {len(self.collected_decisions):,}건")
            
        except Exception as e:
            logger.warning(f"기존 데이터 로드 실패: {e}")


if __name__ == "__main__":
    # 테스트 실행
    collector = DateBasedLegalInterpretationCollector()
    
    # 2025년 법령해석례 수집 테스트 (10건)
    success = collector.collect_by_year(2025, target_count=10, use_interpretation_date=True)
    
    if success:
        print("✅ 법령해석례 수집 테스트 성공")
    else:
        print("❌ 법령해석례 수집 테스트 실패")
