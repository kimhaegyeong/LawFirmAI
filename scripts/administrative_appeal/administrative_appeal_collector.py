#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
행정심판례 수집기 클래스 (collect_precedents.py 구조 참고)
"""

import json
import time
import random
import signal
import atexit
import hashlib
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from contextlib import contextmanager

import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient, LawOpenAPIConfig
from scripts.administrative_appeal.administrative_appeal_models import (
    CollectionStats, AdministrativeAppealData, CollectionStatus, AppealType,
    ADMINISTRATIVE_APPEAL_KEYWORDS, PRIORITY_KEYWORDS, KEYWORD_TARGET_COUNTS, DEFAULT_KEYWORD_COUNT,
    APPEAL_TYPE_KEYWORDS, FALLBACK_KEYWORDS
)
from scripts.administrative_appeal.administrative_appeal_logger import setup_logging

logger = setup_logging()


class AdministrativeAppealCollector:
    """행정심판례 수집 클래스 (개선된 버전)"""
    
    def __init__(self, config: LawOpenAPIConfig, output_dir: Optional[Path] = None):
        """
        행정심판례 수집기 초기화
        
        Args:
            config: API 설정 객체
            output_dir: 출력 디렉토리 (기본값: data/raw/administrative_appeals)
        """
        self.client = LawOpenAPIClient(config)
        self.output_dir = output_dir or Path("data/raw/administrative_appeals")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 관리
        self.collected_appeals: Set[str] = set()  # 중복 방지
        self.processed_keywords: Set[str] = set()  # 처리된 키워드 추적
        self.pending_appeals: List[AdministrativeAppealData] = []  # 임시 저장소
        
        # 설정
        self.batch_size = 30  # 배치 저장 크기
        self.max_retries = 3  # 최대 재시도 횟수
        self.retry_delay = 5  # 재시도 간격 (초)
        self.api_delay_range = (1.0, 3.0)  # API 요청 간 지연 범위
        
        # 통계 및 상태
        self.stats = CollectionStats()
        self.stats.total_keywords = len(ADMINISTRATIVE_APPEAL_KEYWORDS)
        self.checkpoint_file: Optional[Path] = None
        self.resume_mode = False
        
        # 에러 처리
        self.error_count = 0
        self.max_errors = 50  # 최대 허용 에러 수
        
        # Graceful shutdown 관련
        self.shutdown_requested = False
        self.shutdown_reason = None
        self._setup_signal_handlers()
        
        # 기존 수집된 데이터 로드
        self._load_existing_data()
        
        # 체크포인트 파일 확인 및 복구
        self._check_and_resume_from_checkpoint()
        
        logger.info(f"행정심판례 수집기 초기화 완료 - 목표: {self.stats.target_count}건")
    
    def _setup_signal_handlers(self):
        """시그널 핸들러 설정 (Graceful shutdown)"""
        def signal_handler(signum, frame):
            """시그널 핸들러"""
            signal_name = signal.Signals(signum).name
            logger.warning(f"시그널 {signal_name} ({signum}) 수신됨. Graceful shutdown 시작...")
            self.shutdown_requested = True
            self.shutdown_reason = f"Signal {signal_name} ({signum})"
        
        # SIGINT (Ctrl+C), SIGTERM 처리
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Windows에서 SIGBREAK 처리
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
        
        # 프로그램 종료 시 정리 함수 등록
        atexit.register(self._cleanup_on_exit)
    
    def _cleanup_on_exit(self):
        """프로그램 종료 시 정리 작업"""
        if self.pending_appeals:
            logger.info("프로그램 종료 시 임시 데이터 저장 중...")
            self._save_batch_appeals()
        
        if self.checkpoint_file:
            logger.info("최종 체크포인트 저장 중...")
            self._save_checkpoint(self.checkpoint_file)
    
    def _check_shutdown_requested(self) -> bool:
        """종료 요청 확인"""
        return self.shutdown_requested
    
    def _request_shutdown(self, reason: str):
        """종료 요청"""
        self.shutdown_requested = True
        self.shutdown_reason = reason
        logger.warning(f"종료 요청됨: {reason}")
    
    def _load_existing_data(self):
        """기존 수집된 데이터 로드하여 중복 방지"""
        logger.info("기존 수집된 데이터 확인 중...")
        
        loaded_count = 0
        error_count = 0
        
        # 다양한 파일 패턴에서 데이터 로드
        file_patterns = [
            "administrative_appeal_*.json",
            "batch_*.json", 
            "checkpoints/**/*.json",
            "*.json"
        ]
        
        for pattern in file_patterns:
            files = list(self.output_dir.glob(pattern))
            for file_path in files:
                try:
                    loaded_count += self._load_appeals_from_file(file_path)
                except Exception as e:
                    error_count += 1
                    logger.debug(f"파일 로드 실패 {file_path}: {e}")
        
        logger.info(f"기존 데이터 로드 완료: {loaded_count:,}건, 오류: {error_count:,}건")
        self.stats.collected_count = len(self.collected_appeals)
        logger.info(f"중복 방지를 위한 심판례 ID {len(self.collected_appeals):,}개 로드됨")
    
    def _load_appeals_from_file(self, file_path: Path) -> int:
        """파일에서 행정심판례 데이터 로드"""
        loaded_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 다양한 데이터 구조 처리
        appeals = []
        
        if isinstance(data, dict):
            if 'appeals' in data:
                appeals = data['appeals']
            elif 'basic_info' in data:
                appeals = [data]
            elif 'by_category' in data:
                for category_data in data['by_category'].values():
                    appeals.extend(category_data)
        elif isinstance(data, list):
            appeals = data
        
        # 심판례 ID 추출
        for appeal in appeals:
            if isinstance(appeal, dict):
                appeal_id = appeal.get('판례일련번호') or appeal.get('appeal_id')
                if appeal_id:
                    self.collected_appeals.add(str(appeal_id))
                    loaded_count += 1
        
        return loaded_count
    
    def _is_duplicate_appeal(self, appeal: Dict[str, Any]) -> bool:
        """행정심판례 중복 여부 확인"""
        appeal_id = appeal.get('판례일련번호') or appeal.get('appeal_id')
        
        if appeal_id and str(appeal_id).strip() != '':
            if str(appeal_id) in self.collected_appeals:
                logger.debug(f"심판례일련번호로 중복 확인: {appeal_id}")
                return True
        
        return False
    
    def _mark_appeal_collected(self, appeal: Dict[str, Any]):
        """행정심판례를 수집됨으로 표시"""
        appeal_id = appeal.get('판례일련번호') or appeal.get('appeal_id')
        
        if appeal_id and str(appeal_id).strip() != '':
            self.collected_appeals.add(str(appeal_id))
            logger.debug(f"심판례일련번호로 저장: {appeal_id}")
    
    def _validate_appeal_data(self, appeal: Dict[str, Any]) -> bool:
        """행정심판례 데이터 검증"""
        appeal_id = appeal.get('판례일련번호') or appeal.get('appeal_id')
        case_name = appeal.get('사건명')
        
        if not appeal_id or str(appeal_id).strip() == '':
            if not case_name:
                logger.warning(f"행정심판례 식별 정보 부족 - 심판례ID: {appeal_id}, 사건명: {case_name}")
                return False
            logger.debug(f"심판례일련번호 없음, 사건명으로 대체: {case_name}")
        elif not case_name:
            logger.warning(f"사건명이 없습니다: {appeal}")
            return False
        
        return True
    
    def _create_appeal_data(self, raw_data: Dict[str, Any]) -> Optional[AdministrativeAppealData]:
        """원시 데이터에서 AdministrativeAppealData 객체 생성"""
        try:
            # 데이터 검증
            if not self._validate_appeal_data(raw_data):
                return None
            
            # 심판례 ID 추출
            appeal_id = raw_data.get('판례일련번호') or raw_data.get('appeal_id')
            
            # 심판례 ID가 없는 경우 대체 ID 생성
            if not appeal_id or str(appeal_id).strip() == '':
                case_name = raw_data.get('사건명', '')
                if case_name:
                    appeal_id = f"appeal_{case_name}"
                else:
                    appeal_id = f"appeal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                logger.debug(f"대체 ID 생성: {appeal_id}")
            
            # 심판 유형 분류
            appeal_type = self.classify_appeal_type(raw_data)
            
            # AdministrativeAppealData 객체 생성
            appeal_data = AdministrativeAppealData(
                appeal_id=str(appeal_id),
                case_name=raw_data.get('사건명', ''),
                case_number=raw_data.get('사건번호', ''),
                appeal_type=appeal_type,
                decision_date=raw_data.get('판결일자', '') or raw_data.get('선고일자', ''),
                raw_data=raw_data
            )
            
            return appeal_data
            
        except Exception as e:
            logger.error(f"AdministrativeAppealData 생성 실패: {e}")
            logger.error(f"원시 데이터: {raw_data}")
            return None
    
    def classify_appeal_type(self, appeal: Dict[str, Any]) -> AppealType:
        """행정심판례 유형 분류"""
        case_name = appeal.get('사건명', '').lower()
        case_content = appeal.get('판시사항', '') + ' ' + appeal.get('판결요지', '')
        case_content = case_content.lower()
        
        # 키워드 기반 분류
        for appeal_type, keywords in APPEAL_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in case_name or keyword in case_content:
                    return AppealType(appeal_type)
        
        return AppealType.OTHER
    
    def _random_delay(self, min_seconds: Optional[float] = None, max_seconds: Optional[float] = None):
        """API 요청 간 랜덤 지연"""
        min_delay = min_seconds or self.api_delay_range[0]
        max_delay = max_seconds or self.api_delay_range[1]
        delay = random.uniform(min_delay, max_delay)
        logger.debug(f"API 요청 간 {delay:.2f}초 대기...")
        time.sleep(delay)
    
    @contextmanager
    def _api_request_with_retry(self, operation_name: str):
        """API 요청 재시도 컨텍스트 매니저"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                self.stats.api_requests_made += 1
                yield
                return
            except Exception as e:
                last_exception = e
                self.stats.api_errors += 1
                self.error_count += 1
                
                if attempt < self.max_retries - 1:
                    logger.warning(f"{operation_name} 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))  # 지수 백오프
                else:
                    logger.error(f"{operation_name} 최종 실패: {e}")
                    break
        
        # 모든 재시도가 실패한 경우 예외 발생
        if last_exception:
            raise last_exception
    
    def collect_all_appeals(self, target_count: int = 1000):
        """모든 행정심판례 수집 (개선된 버전)"""
        self.stats.target_count = target_count
        self.stats.status = CollectionStatus.IN_PROGRESS
        
        # 체크포인트 파일 설정
        checkpoint_file = self._setup_checkpoint_file()
        
        try:
            logger.info(f"행정심판례 수집 시작 - 목표: {target_count}건")
            logger.info("Graceful shutdown 지원: Ctrl+C 또는 SIGTERM으로 안전하게 중단 가능")
            logger.info("중단 시 현재까지 수집된 데이터가 자동으로 저장됩니다")
            
            # 키워드별 검색 실행
            self._collect_by_keywords(target_count, checkpoint_file)
            
            # 목표 달성하지 못한 경우 백업 키워드 사용
            if self.stats.collected_count < target_count:
                remaining_count = target_count - self.stats.collected_count
                logger.info(f"목표 달성 실패. 백업 키워드로 {remaining_count}건 추가 수집 시도")
                self._collect_with_fallback_keywords(remaining_count, checkpoint_file)
            
            # 종료 요청 확인
            if self._check_shutdown_requested():
                logger.warning(f"수집이 중단되었습니다: {self.shutdown_reason}")
                self.stats.status = CollectionStatus.CANCELLED
                self.stats.end_time = datetime.now()
                self._save_final_checkpoint(checkpoint_file)
                return
            
            # 최종 통계 출력
            self._print_final_stats()
            
            # 체크포인트 파일 정리
            self._cleanup_checkpoint_file(checkpoint_file)
            
            self.stats.status = CollectionStatus.COMPLETED
            self.stats.end_time = datetime.now()
            
        except KeyboardInterrupt:
            logger.warning("사용자에 의해 수집이 중단되었습니다.")
            self.stats.status = CollectionStatus.CANCELLED
            self.stats.end_time = datetime.now()
            self._save_final_checkpoint(checkpoint_file)
            return
        except Exception as e:
            logger.error(f"행정심판례 수집 중 오류 발생: {e}")
            self.stats.status = CollectionStatus.FAILED
            self.stats.end_time = datetime.now()
            self._save_final_checkpoint(checkpoint_file)
            raise
    
    def _setup_checkpoint_file(self) -> Path:
        """체크포인트 파일 설정"""
        if self.resume_mode and self.checkpoint_file:
            return self.checkpoint_file
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return self.output_dir / f"collection_checkpoint_{timestamp}.json"
    
    def _collect_by_keywords(self, target_count: int, checkpoint_file: Path):
        """우선순위 기반 키워드별 행정심판례 수집"""
        # 우선순위 키워드와 일반 키워드 분리
        priority_keywords = [kw for kw in PRIORITY_KEYWORDS if kw in ADMINISTRATIVE_APPEAL_KEYWORDS]
        remaining_keywords = [kw for kw in ADMINISTRATIVE_APPEAL_KEYWORDS if kw not in PRIORITY_KEYWORDS]
        
        # 전체 키워드 목록 (우선순위 먼저, 나머지 나중에)
        ordered_keywords = priority_keywords + remaining_keywords
        
        # 이미 처리된 키워드 제외
        unprocessed_keywords = [kw for kw in ordered_keywords if kw not in self.processed_keywords]
        
        logger.info(f"우선순위 기반 키워드 수집 시작")
        logger.info(f"1순위 키워드: {len(priority_keywords)}개 (우선 수집)")
        logger.info(f"2순위 키워드: {len(remaining_keywords)}개 (추가 수집)")
        logger.info(f"총 키워드: {len(ordered_keywords)}개")
        logger.info(f"이미 처리된 키워드: {len(self.processed_keywords)}개")
        logger.info(f"처리 대기 키워드: {len(unprocessed_keywords)}개")
        
        if not unprocessed_keywords:
            logger.info("모든 키워드가 이미 처리되었습니다.")
            return
        
        # 진행 상황 추적
        total_keywords = len(unprocessed_keywords)
        logger.info(f"총 {total_keywords}개 미처리 키워드 처리 시작")
        
        for i, keyword in enumerate(unprocessed_keywords):
            # 종료 요청 확인
            if self._check_shutdown_requested():
                logger.warning(f"종료 요청으로 키워드 검색 중단: {self.shutdown_reason}")
                break
            
            if self.stats.collected_count >= target_count:
                logger.info(f"목표 수량 {target_count:,}건 달성으로 키워드 검색 중단")
                break
            
            try:
                # 키워드별 목표 건수 결정
                if keyword in KEYWORD_TARGET_COUNTS:
                    keyword_target = KEYWORD_TARGET_COUNTS[keyword]
                    priority_level = "우선순위"
                else:
                    keyword_target = DEFAULT_KEYWORD_COUNT
                    priority_level = "일반"
                
                # 진행 상황 로깅
                progress_percent = ((i + 1) / total_keywords) * 100
                logger.info(f"[{i+1}/{total_keywords}] ({progress_percent:.1f}%) 키워드 '{keyword}' 처리 시작 ({priority_level}, 목표: {keyword_target}건)")
                
                if i > 0:
                    self._random_delay()
                
                # 키워드별 수집
                appeals = self.collect_appeals_by_keyword(keyword, keyword_target)
                
                # 통계 업데이트
                self.stats.keywords_processed = len(self.processed_keywords)
                
                # 체크포인트 저장 (매 키워드마다)
                self._save_checkpoint(checkpoint_file)
                
                # 키워드 완료 로깅
                logger.info(f"키워드 '{keyword}' 완료 ({priority_level}, 목표: {keyword_target}건). 누적: {self.stats.collected_count:,}/{target_count:,}건")
                
                # API 요청 제한 확인
                if self._check_api_limits():
                    break
                    
            except KeyboardInterrupt:
                logger.warning(f"키워드 '{keyword}' 검색이 중단되었습니다.")
                break
            except Exception as e:
                logger.error(f"키워드 '{keyword}' 검색 실패: {e}")
                continue
    
    def collect_appeals_by_keyword(self, keyword: str, max_count: int = 30) -> List[AdministrativeAppealData]:
        """키워드로 행정심판례 검색 및 수집"""
        # 이미 처리된 키워드인지 확인
        if keyword in self.processed_keywords:
            logger.info(f"키워드 '{keyword}'는 이미 처리되었습니다. 건너뜁니다.")
            return []
        
        logger.info(f"키워드 '{keyword}'로 행정심판례 검색 시작 (목표: {max_count}건)")
        
        appeals = []
        page = 1
        consecutive_empty_pages = 0
        max_empty_pages = 3
        
        while len(appeals) < max_count and consecutive_empty_pages < max_empty_pages:
            # 종료 요청 확인
            if self._check_shutdown_requested():
                logger.warning(f"종료 요청으로 '{keyword}' 검색 중단: {self.shutdown_reason}")
                break
            
            try:
                # API 요청 간 랜덤 지연
                if page > 1:
                    self._random_delay()
                
                # 진행 상황 로깅
                logger.debug(f"키워드 '{keyword}' 페이지 {page} 검색 중...")
                
                # API 요청 실행
                try:
                    with self._api_request_with_retry(f"키워드 '{keyword}' 검색"):
                        results = self.client.get_administrative_appeal_list(
                            query=keyword,
                            display=100,
                            page=page
                        )
                except Exception as api_error:
                    logger.error(f"API 요청 실패: {api_error}")
                    consecutive_empty_pages += 1
                    page += 1
                    continue
                
                if not results:
                    consecutive_empty_pages += 1
                    logger.debug(f"키워드 '{keyword}' 페이지 {page}에서 결과 없음 (연속 빈 페이지: {consecutive_empty_pages})")
                    page += 1
                    continue
                else:
                    consecutive_empty_pages = 0
                
                # 결과 처리
                new_count, duplicate_count = self._process_search_results(results, appeals, max_count)
                
                # 페이지별 결과 로깅
                logger.debug(f"페이지 {page}: {new_count}건 신규, {duplicate_count}건 중복 (누적: {len(appeals)}/{max_count}건)")
                
                page += 1
                
                # API 요청 제한 확인
                if self._check_api_limits():
                    break
                    
            except KeyboardInterrupt:
                logger.warning(f"키워드 '{keyword}' 검색이 중단되었습니다.")
                break
            except Exception as e:
                logger.error(f"키워드 '{keyword}' 검색 중 오류: {e}")
                self.stats.failed_count += 1
                break
        
        # 수집된 심판례를 임시 저장소에 추가
        for appeal in appeals:
            self.pending_appeals.append(appeal)
            self.stats.collected_count += 1
        
        # 키워드 처리 완료 표시
        self.processed_keywords.add(keyword)
        
        logger.info(f"키워드 '{keyword}' 수집 완료: {len(appeals)}건")
        return appeals
    
    def _process_search_results(self, results: List[Dict[str, Any]], appeals: List[AdministrativeAppealData], 
                              max_count: int) -> Tuple[int, int]:
        """검색 결과 처리"""
        new_count = 0
        duplicate_count = 0
        
        for result in results:
            # 종료 요청 확인
            if self._check_shutdown_requested():
                logger.warning(f"종료 요청으로 결과 처리 중단: {self.shutdown_reason}")
                break
            
            # 중복 확인
            if self._is_duplicate_appeal(result):
                duplicate_count += 1
                self.stats.duplicate_count += 1
                continue
            
            # AdministrativeAppealData 객체 생성
            appeal_data = self._create_appeal_data(result)
            if not appeal_data:
                self.stats.failed_count += 1
                continue
            
            # 신규 심판례 추가
            appeals.append(appeal_data)
            self._mark_appeal_collected(result)
            new_count += 1
            
            if len(appeals) >= max_count:
                break
        
        return new_count, duplicate_count
    
    def _save_batch_appeals(self):
        """배치 단위로 행정심판례 저장"""
        if not self.pending_appeals:
            return
        
        try:
            # 유형별로 그룹화
            by_type = {}
            for appeal in self.pending_appeals:
                appeal_type = appeal.appeal_type.value
                if appeal_type not in by_type:
                    by_type[appeal_type] = []
                by_type[appeal_type].append(appeal)
            
            # 유형별 파일 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            saved_files = []
            
            for appeal_type, appeals in by_type.items():
                # 안전한 파일명 생성
                safe_type = appeal_type.replace('_', '-')
                filename = f"batch_{safe_type}_{len(appeals)}건_{timestamp}.json"
                filepath = self.output_dir / filename
                
                batch_data = {
                    'metadata': {
                        'appeal_type': appeal_type,
                        'count': len(appeals),
                        'saved_at': datetime.now().isoformat(),
                        'batch_id': timestamp
                    },
                    'appeals': [a.raw_data for a in appeals]
                }
                
                with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
                    json.dump(batch_data, f, ensure_ascii=False, indent=2)
                
                saved_files.append(filepath)
                logger.info(f"배치 저장 완료: {appeal_type} 유형 {len(appeals):,}건 -> {filename}")
            
            # 통계 업데이트
            self.stats.saved_count += len(self.pending_appeals)
            
            # 임시 저장소 초기화
            self.pending_appeals = []
            
            logger.info(f"배치 저장 완료: 총 {len(saved_files):,}개 파일")
            
        except Exception as e:
            logger.error(f"배치 저장 실패: {e}")
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
    
    def _print_final_stats(self):
        """최종 통계 출력"""
        logger.info("=" * 60)
        logger.info("수집 완료 통계")
        logger.info("=" * 60)
        logger.info(f"목표 수집 건수: {self.stats.target_count:,}건")
        logger.info(f"실제 수집 건수: {self.stats.collected_count:,}건")
        logger.info(f"중복 제외 건수: {self.stats.duplicate_count:,}건")
        logger.info(f"실패 건수: {self.stats.failed_count:,}건")
        logger.info(f"처리된 키워드: {len(self.processed_keywords):,}개")
        logger.info(f"저장된 배치 수: {self.stats.saved_count:,}건")
        logger.info(f"API 요청 수: {self.stats.api_requests_made:,}회")
        logger.info(f"API 오류 수: {self.stats.api_errors:,}회")
        logger.info(f"성공률: {self.stats.success_rate:.1f}%")
        if self.stats.duration:
            logger.info(f"소요 시간: {self.stats.duration}")
        logger.info("=" * 60)
    
    def _cleanup_checkpoint_file(self, checkpoint_file: Path):
        """체크포인트 파일 정리"""
        if checkpoint_file and checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                logger.info("체크포인트 파일 정리 완료")
            except Exception as e:
                logger.warning(f"체크포인트 파일 정리 실패: {e}")
    
    def _save_final_checkpoint(self, checkpoint_file: Path):
        """최종 체크포인트 저장"""
        try:
            self._save_checkpoint(checkpoint_file)
            logger.info(f"현재까지 수집된 데이터는 {checkpoint_file}에 저장되었습니다.")
            logger.info("나중에 다시 실행하면 이어서 계속할 수 있습니다.")
        except Exception as e:
            logger.error(f"최종 체크포인트 저장 실패: {e}")
    
    def _save_checkpoint(self, checkpoint_file: Path):
        """진행 상황 체크포인트 저장"""
        try:
            checkpoint_data = {
                'stats': {
                    'start_time': self.stats.start_time.isoformat(),
                    'end_time': self.stats.end_time.isoformat() if self.stats.end_time else None,
                    'target_count': self.stats.target_count,
                    'collected_count': self.stats.collected_count,
                    'saved_count': self.stats.saved_count,
                    'duplicate_count': self.stats.duplicate_count,
                    'failed_count': self.stats.failed_count,
                    'keywords_processed': self.stats.keywords_processed,
                    'total_keywords': self.stats.total_keywords,
                    'api_requests_made': self.stats.api_requests_made,
                    'api_errors': self.stats.api_errors,
                    'status': self.stats.status.value,
                    'processed_keywords': list(self.processed_keywords),
                    'collected_appeals_count': len(self.collected_appeals)
                },
                'appeals': [a.raw_data for a in self.pending_appeals],
                'saved_at': datetime.now().isoformat(),
                'resume_info': {
                    'can_resume': True,
                    'last_keyword_processed': list(self.processed_keywords)[-1] if self.processed_keywords else None,
                    'progress_percentage': (self.stats.collected_count / self.stats.target_count) * 100 if self.stats.target_count > 0 else 0
                },
                'shutdown_info': {
                    'shutdown_requested': self.shutdown_requested,
                    'shutdown_reason': self.shutdown_reason,
                    'graceful_shutdown_supported': True
                }
            }
            
            with open(checkpoint_file, 'w', encoding='utf-8', newline='\n') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"체크포인트 저장 완료: {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"체크포인트 저장 실패: {e}")
            logger.error(traceback.format_exc())
    
    def _check_and_resume_from_checkpoint(self):
        """체크포인트 파일 확인 및 복구"""
        logger.info("체크포인트 파일 확인 중...")
        
        # 체크포인트 파일 찾기
        checkpoint_files = list(self.output_dir.glob("collection_checkpoint_*.json"))
        
        if not checkpoint_files:
            logger.info("체크포인트 파일이 없습니다. 새로 시작합니다.")
            return
        
        # 가장 최근 체크포인트 파일 선택
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        self.checkpoint_file = latest_checkpoint
        
        logger.info(f"체크포인트 파일 발견: {latest_checkpoint.name}")
        
        try:
            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # 체크포인트 데이터 복구
            self._restore_from_checkpoint(checkpoint_data)
            
            self.resume_mode = True
            self.stats.status = CollectionStatus.IN_PROGRESS
            
            logger.info("=" * 60)
            logger.info("이전 작업 복구 완료")
            logger.info("=" * 60)
            logger.info(f"복구된 수집 건수: {self.stats.collected_count:,}건")
            logger.info(f"처리된 키워드: {len(self.processed_keywords):,}개")
            logger.info(f"저장된 배치 수: {self.stats.saved_count:,}건")
            logger.info(f"중복 제외 건수: {self.stats.duplicate_count:,}건")
            logger.info(f"API 요청 수: {self.stats.api_requests_made:,}회")
            logger.info("=" * 60)
            
            # 자동으로 계속 진행
            logger.info("이전 작업을 이어서 진행합니다.")
            
        except Exception as e:
            logger.error(f"체크포인트 파일 복구 실패: {e}")
            logger.info("새로 시작합니다.")
            self.resume_mode = False
            self.checkpoint_file = None
    
    def _restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """체크포인트 데이터에서 상태 복구"""
        stats_data = checkpoint_data.get('stats', {})
        appeals = checkpoint_data.get('appeals', [])
        
        # 통계 복구
        self.stats.collected_count = stats_data.get('collected_count', 0)
        self.stats.saved_count = stats_data.get('saved_count', 0)
        self.stats.duplicate_count = stats_data.get('duplicate_count', 0)
        self.stats.failed_count = stats_data.get('failed_count', 0)
        self.stats.keywords_processed = stats_data.get('keywords_processed', 0)
        self.stats.api_requests_made = stats_data.get('api_requests_made', 0)
        self.stats.api_errors = stats_data.get('api_errors', 0)
        
        # 처리된 키워드 복구
        processed_keywords = stats_data.get('processed_keywords', [])
        self.processed_keywords = set(processed_keywords)
        
        # 수집된 심판례 ID 복구
        for appeal in appeals:
            if isinstance(appeal, dict):
                appeal_id = appeal.get('판례일련번호') or appeal.get('appeal_id')
                if appeal_id:
                    self.collected_appeals.add(str(appeal_id))
    
    def _collect_with_fallback_keywords(self, remaining_count: int, checkpoint_file: Path):
        """백업 키워드로 추가 수집"""
        logger.info(f"백업 키워드 수집 시작 - 목표: {remaining_count}건")
        
        # 백업 키워드 중 아직 처리되지 않은 것들만 선택
        unprocessed_fallback = [kw for kw in FALLBACK_KEYWORDS if kw not in self.processed_keywords]
        
        if not unprocessed_fallback:
            logger.info("모든 백업 키워드가 이미 처리되었습니다.")
            return
        
        logger.info(f"백업 키워드 {len(unprocessed_fallback)}개로 추가 수집 시도")
        
        for i, keyword in enumerate(unprocessed_fallback):
            if self.stats.collected_count >= self.stats.target_count:
                logger.info(f"목표 수량 {self.stats.target_count:,}건 달성으로 백업 키워드 수집 중단")
                break
            
            if self._check_shutdown_requested():
                logger.warning(f"종료 요청으로 백업 키워드 수집 중단: {self.shutdown_reason}")
                break
            
            try:
                # 백업 키워드는 각각 5건씩 수집
                keyword_target = min(5, remaining_count)
                remaining_count -= keyword_target
                
                logger.info(f"백업 키워드 '{keyword}' 처리 시작 (목표: {keyword_target}건)")
                
                # 심판례 수집
                appeals = self.collect_appeals_by_keyword(keyword, keyword_target)
                
                # 배치 저장
                if len(self.pending_appeals) >= self.batch_size:
                    self._save_batch_appeals()
                
                # 체크포인트 저장
                self._save_checkpoint(checkpoint_file)
                
                # 진행 상황 로깅
                progress_percent = (self.stats.collected_count / self.stats.target_count) * 100
                logger.info(f"백업 키워드 '{keyword}' 완료: {len(appeals)}건 수집 (총 {self.stats.collected_count:,}건, {progress_percent:.1f}%)")
                
                # API 요청 제한 확인
                if self._check_api_limits():
                    logger.warning("API 요청 제한에 도달하여 백업 키워드 수집 중단")
                    break
                    
            except Exception as e:
                logger.error(f"백업 키워드 '{keyword}' 수집 중 오류: {e}")
                self.stats.failed_count += 1
                continue
        
        logger.info(f"백업 키워드 수집 완료 - 총 {self.stats.collected_count:,}건 수집")
            
        # 배치 저장 (임시 저장소가 가득 찬 경우)
        if len(self.pending_appeals) >= self.batch_size:
            self._save_batch_appeals()
