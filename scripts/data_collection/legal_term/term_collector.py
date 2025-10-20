# -*- coding: utf-8 -*-
"""
법률 용어 수집기 (메모리 최적화 및 체크포인트 지원)

국가법령정보센터 OpenAPI를 활용하여 법률 용어를 수집하고 관리합니다.
- 메모리 효율적인 배치 처리
- 체크포인트 시스템으로 중단 시 재개 가능
- 실시간 진행률 모니터링
- 메모리 사용량 추적 및 최적화
"""

import os
import sys
import json
import gc
import psutil
import logging
import signal
import atexit
import glob
import time
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.legal_term_collection_api import LegalTermCollectionAPI, TermCollectionConfig
from source.data.legal_term_dictionary import LegalTermDictionary

logger = logging.getLogger(__name__)


@dataclass
class CheckpointData:
    """체크포인트 데이터 구조"""
    session_id: str
    start_time: str
    last_update: str
    total_target: int
    collected_count: int
    processed_categories: List[str] = field(default_factory=list)
    processed_keywords: List[str] = field(default_factory=list)
    current_category: Optional[str] = None
    current_keyword: Optional[str] = None
    memory_usage_mb: float = 0.0
    api_requests_made: int = 0
    errors_count: int = 0
    can_resume: bool = True
    # 페이지네이션 정보 추가
    current_page: int = 1
    page_size: int = 100
    consecutive_empty_pages: int = 0
    last_page_terms_count: int = 0


@dataclass
class MemoryConfig:
    """메모리 관리 설정"""
    max_memory_mb: int = 2048  # 최대 메모리 사용량 (MB)
    batch_size: int = 10  # 배치 크기 (10개씩 처리)
    checkpoint_interval: int = 2  # 체크포인트 저장 간격 (2개 배치마다 저장 - 더 자주 저장)
    gc_threshold: int = 500  # 가비지 컬렉션 임계값
    memory_check_interval: int = 10  # 메모리 체크 간격
    batch_delay_min: float = 1.0  # 배치 간 최소 지연 시간 (초)
    batch_delay_max: float = 3.0  # 배치 간 최대 지연 시간 (초)


class LegalTermCollector:
    """법률 용어 수집기 클래스 (메모리 최적화 및 체크포인트 지원)"""
    
    def __init__(self, config: TermCollectionConfig = None, memory_config: MemoryConfig = None):
        """수집기 초기화"""
        self.config = config or self._create_default_config()
        self.memory_config = memory_config or MemoryConfig()
        self.api_client = LegalTermCollectionAPI(self.config)
        self.dictionary = LegalTermDictionary()
        
        # 체크포인트 관련
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_file = Path("data/raw/legal_terms/checkpoint.json")
        self.checkpoint_data = None
        
        # 동적 파일명 생성 (세션별 구분)
        self.dynamic_file_prefix = None
        self.dictionary_file = None
        self.shutdown_requested = False
        
        # 메모리 관리
        self.process = psutil.Process()
        self.memory_check_counter = 0
        self.gc_counter = 0
        
        # 기본 통계
        self.stats = {
            'total_collected': 0,
            'start_time': None,
            'end_time': None
        }
        
        # 시그널 핸들러 등록
        self._setup_signal_handlers()
        
        # 종료 시 정리 작업 등록
        atexit.register(self._cleanup_on_exit)
        
        logger.info("LegalTermCollector initialized with memory optimization")
    
    def _create_default_config(self) -> TermCollectionConfig:
        """기본 설정 생성"""
        config = TermCollectionConfig()
        config.batch_size = 10  # 기본 배치 크기
        config.delay_between_requests = 0.05
        config.max_retries = 3
        return config
    
    def _setup_signal_handlers(self):
        """시그널 핸들러 설정 (graceful shutdown 지원)"""
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            logger.info(f"시그널 {signal_name}({signum}) 수신 - graceful shutdown 시작")
            
            # 중복 종료 요청 방지
            if self.shutdown_requested:
                logger.warning("이미 종료 요청이 진행 중입니다. 강제 종료합니다.")
                sys.exit(1)
            
            self.shutdown_requested = True
            
            # 진행 상황 로깅
            if self.checkpoint_data:
                progress = (self.checkpoint_data.collected_count / self.checkpoint_data.total_target * 100) if self.checkpoint_data.total_target > 0 else 0
                logger.info(f"현재 진행률: {progress:.1f}% ({self.checkpoint_data.collected_count}/{self.checkpoint_data.total_target})")
            
            # 체크포인트 저장
            try:
                self._save_checkpoint()
                logger.info("체크포인트 저장 완료")
            except Exception as e:
                logger.error(f"체크포인트 저장 실패: {e}")
            
            # 정리 작업 수행
            self._perform_cleanup()
            
            logger.info("graceful shutdown 완료")
            sys.exit(0)
        
        # 다양한 시그널에 대한 핸들러 등록
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)    # 종료 요청
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal_handler)  # 터미널 연결 끊김
        if hasattr(signal, 'SIGQUIT'):
            signal.signal(signal.SIGQUIT, signal_handler)  # 종료 + 코어 덤프
    
    def _perform_cleanup(self):
        """정리 작업 수행 (graceful shutdown)"""
        logger.info("정리 작업 시작...")
        
        try:
            # 1. 진행 중인 작업 완료 대기
            logger.info("진행 중인 작업 완료 대기 중...")
            time.sleep(1)  # 현재 진행 중인 작업이 완료될 시간 제공
            
            # 2. 메모리 정리
            logger.info("메모리 정리 중...")
            self._force_garbage_collection()
            
            # 3. 통계 정보 로깅
            if self.checkpoint_data:
                logger.info("=" * 50)
                logger.info("수집 세션 요약:")
                logger.info(f"  세션 ID: {self.checkpoint_data.session_id}")
                logger.info(f"  수집된 용어: {self.checkpoint_data.collected_count}개")
                logger.info(f"  목표 용어: {self.checkpoint_data.total_target}개")
                logger.info(f"  진행률: {self.checkpoint_data.collected_count/self.checkpoint_data.total_target*100:.1f}%")
                logger.info(f"  API 요청 수: {self.checkpoint_data.api_requests_made}")
                logger.info(f"  에러 수: {self.checkpoint_data.errors_count}")
                logger.info(f"  메모리 사용량: {self.checkpoint_data.memory_usage_mb:.1f}MB")
                logger.info("=" * 50)
            
            # 4. 리소스 정리
            logger.info("리소스 정리 중...")
            if hasattr(self, 'api_client') and hasattr(self.api_client, 'session'):
                self.api_client.session.close()
            
            # 5. 임시 파일 정리 (필요한 경우)
            self._cleanup_temp_files()
            
            logger.info("정리 작업 완료")
            
        except Exception as e:
            logger.error(f"정리 작업 중 오류 발생: {e}")
    
    def _cleanup_on_exit(self):
        """종료 시 정리 작업 (atexit 핸들러)"""
        try:
            self._perform_cleanup()
        except Exception as e:
            logger.error(f"종료 시 정리 작업 실패: {e}")
    
    def _cleanup_temp_files(self):
        """임시 파일 정리"""
        try:
            # 임시 파일이 있다면 정리
            temp_patterns = [
                "data/raw/legal_terms/temp_*.json",
                "data/raw/legal_terms/*.tmp",
                "data/raw/legal_terms/session_*/temp_*.json",
                "data/raw/legal_terms/session_*/*.tmp",
                "logs/temp_*.log"
            ]
            
            for pattern in temp_patterns:
                temp_files = glob.glob(pattern)
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                        logger.debug(f"임시 파일 삭제: {temp_file}")
                    except OSError:
                        pass  # 파일이 이미 삭제되었거나 접근할 수 없음
                        
        except Exception as e:
            logger.debug(f"임시 파일 정리 중 오류 (무시됨): {e}")
    
    def _get_memory_usage_mb(self) -> float:
        """현재 메모리 사용량 조회 (MB)"""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # MB 단위
        except Exception as e:
            logger.warning(f"메모리 사용량 조회 실패: {e}")
            return 0.0
    
    def _check_memory_limit(self) -> bool:
        """메모리 사용량 제한 확인"""
        current_memory = self._get_memory_usage_mb()
        
        
        # 메모리 체크 카운터 증가
        self.memory_check_counter += 1
        
        # 메모리 제한 초과 시 가비지 컬렉션 실행
        if current_memory > self.memory_config.max_memory_mb:
            logger.warning(f"메모리 사용량 초과: {current_memory:.1f}MB > {self.memory_config.max_memory_mb}MB")
            self._force_garbage_collection()
            return False
        
        return True
    
    def _force_garbage_collection(self):
        """강제 가비지 컬렉션 실행"""
        gc.collect()
    
    def _save_checkpoint(self) -> bool:
        """체크포인트 저장 (비정상 종료 방지)"""
        try:
            if not self.checkpoint_data:
                return False
            
            # 현재 상태 업데이트
            self.checkpoint_data.last_update = datetime.now().isoformat()
            self.checkpoint_data.collected_count = len(self.dictionary.terms)
            self.checkpoint_data.memory_usage_mb = self._get_memory_usage_mb()
            self.checkpoint_data.api_requests_made = self.api_client.request_count
            
            # 체크포인트 파일 저장 (원자적 쓰기)
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 임시 파일에 먼저 쓰고 나중에 이름 변경 (원자적 쓰기)
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.checkpoint_data.__dict__, f, ensure_ascii=False, indent=2)
                
                # 임시 파일을 실제 파일로 이름 변경 (원자적 연산)
                temp_file.replace(self.checkpoint_file)
                
                logger.debug(f"체크포인트 저장 완료: {self.checkpoint_data.collected_count}개 용어")
                return True
                
            except Exception as e:
                # 임시 파일 정리
                if temp_file.exists():
                    temp_file.unlink()
                raise e
            
        except Exception as e:
            logger.error(f"체크포인트 저장 실패: {e}")
            return False
    
    def _load_checkpoint(self) -> Optional[CheckpointData]:
        """체크포인트 로드 (기존 세션 우선 검색, 비정상 종료 복구 지원)"""
        try:
            # 1. 현재 설정된 체크포인트 파일 확인
            if self.checkpoint_file.exists():
                checkpoint_data = self._load_checkpoint_file(self.checkpoint_file)
                if checkpoint_data:
                    logger.info(f"현재 세션 체크포인트 로드 완료: {checkpoint_data.collected_count}개 용어 수집됨")
                    return checkpoint_data
            
            # 2. 모든 세션 폴더에서 체크포인트 파일 검색 (비정상 종료 복구)
            session_folders = list(Path("data/raw/legal_terms").glob("session_*"))
            if not session_folders:
                logger.info("기존 세션 폴더가 없습니다.")
                return None
            
            # 체크포인트 파일들을 수집하고 정렬
            all_checkpoints = []
            for session_folder in session_folders:
                checkpoint_files = list(session_folder.glob("checkpoint_*.json"))
                for checkpoint_file in checkpoint_files:
                    try:
                        checkpoint_data = self._load_checkpoint_file(checkpoint_file)
                        if checkpoint_data:
                            # 체크포인트 파일의 수정 시간도 고려
                            file_mtime = checkpoint_file.stat().st_mtime
                            all_checkpoints.append((checkpoint_data, checkpoint_file, file_mtime))
                    except Exception as e:
                        logger.debug(f"체크포인트 파일 읽기 실패 ({checkpoint_file}): {e}")
                        continue
            
            if not all_checkpoints:
                logger.info("사용 가능한 체크포인트 파일이 없습니다.")
                return None
            
            # 가장 최근 체크포인트 선택 (시간순 정렬)
            all_checkpoints.sort(key=lambda x: x[2], reverse=True)  # 파일 수정 시간 기준
            
            # 현재 연도와 매칭되는 체크포인트 우선 선택
            current_year = self._extract_current_year()
            matching_checkpoints = []
            other_checkpoints = []
            
            for checkpoint_data, checkpoint_file, file_mtime in all_checkpoints:
                folder_year = self._extract_folder_year(checkpoint_file.parent.name)
                if current_year and folder_year and current_year == folder_year:
                    matching_checkpoints.append((checkpoint_data, checkpoint_file, file_mtime))
                else:
                    other_checkpoints.append((checkpoint_data, checkpoint_file, file_mtime))
            
            # 우선순위: 같은 연도 > 다른 연도 > 가장 최근
            selected_checkpoint = None
            if matching_checkpoints:
                selected_checkpoint = matching_checkpoints[0]  # 같은 연도 중 가장 최근
                logger.info(f"같은 연도({current_year}) 체크포인트 발견")
            elif other_checkpoints:
                selected_checkpoint = other_checkpoints[0]  # 다른 연도 중 가장 최근
                logger.info("다른 연도 체크포인트 발견")
            
            if selected_checkpoint:
                checkpoint_data, checkpoint_file, file_mtime = selected_checkpoint
                
                # 현재 세션으로 설정
                self.checkpoint_file = checkpoint_file
                self.session_id = checkpoint_data.session_id
                
                # 기존 사전 파일도 로드
                self._load_existing_dictionary_files(checkpoint_file.parent)
                
                logger.info(f"🔄 기존 세션 체크포인트 로드 완료: {checkpoint_data.collected_count}개 용어 수집됨")
                logger.info(f"📁 사용된 체크포인트: {checkpoint_file}")
                logger.info(f"🆔 원본 세션 ID: {checkpoint_data.session_id}")
                logger.info(f"🆔 현재 세션 ID: {self.session_id}")
                logger.info(f"⏰ 체크포인트 수정 시간: {datetime.fromtimestamp(file_mtime)}")
                logger.info(f"📊 목표 용어 수: {checkpoint_data.total_target}개")
                logger.info(f"📊 진행률: {checkpoint_data.collected_count/checkpoint_data.total_target*100:.1f}%")
                
                return checkpoint_data
            
            logger.info("사용 가능한 체크포인트 파일이 없습니다.")
            return None
            
        except Exception as e:
            logger.error(f"체크포인트 로드 실패: {e}")
            return None
    
    def _load_checkpoint_file(self, checkpoint_file: Path) -> Optional[CheckpointData]:
        """체크포인트 파일 로드 및 검증"""
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_dict = json.load(f)
            
            # 필수 필드 검증
            required_fields = ['session_id', 'collected_count', 'total_target']
            for field in required_fields:
                if field not in checkpoint_dict:
                    logger.warning(f"체크포인트 파일에 필수 필드 '{field}'가 없습니다: {checkpoint_file}")
                    return None
            
            checkpoint_data = CheckpointData(**checkpoint_dict)
            
            # 데이터 무결성 검증
            if checkpoint_data.collected_count < 0:
                logger.warning(f"체크포인트 데이터 무결성 오류 (음수 수집 수): {checkpoint_file}")
                return None
            
            if checkpoint_data.total_target <= 0:
                logger.warning(f"체크포인트 데이터 무결성 오류 (잘못된 목표 수): {checkpoint_file}")
                return None
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"체크포인트 파일 로드 실패 ({checkpoint_file}): {e}")
            return None
    
    def _extract_current_year(self) -> Optional[str]:
        """현재 설정에서 연도 추출"""
        try:
            if hasattr(self, 'dynamic_file_prefix') and self.dynamic_file_prefix:
                parts = self.dynamic_file_prefix.split('_')
                if 'year' in parts:
                    year_index = parts.index('year')
                    if year_index + 1 < len(parts):
                        return parts[year_index + 1]
        except Exception:
            pass
        return None
    
    def _extract_folder_year(self, folder_name: str) -> Optional[str]:
        """폴더명에서 연도 추출"""
        try:
            if 'year_' in folder_name:
                year_part = folder_name.split('year_')[1]
                return year_part
        except Exception:
            pass
        return None
    
    def _load_existing_dictionary_files(self, session_folder: Path):
        """기존 사전 파일들 로드"""
        try:
            # 메인 사전 파일 찾기
            dictionary_files = list(session_folder.glob("legal_terms_*.json"))
            if dictionary_files:
                # 가장 최근 사전 파일 사용
                latest_dictionary = max(dictionary_files, key=lambda f: f.stat().st_mtime)
                self.dictionary_file = latest_dictionary
                logger.info(f"기존 사전 파일 로드: {latest_dictionary}")
                
                # 사전 데이터 로드
                if self.dictionary_file.exists():
                    self.dictionary.load_terms_from_file(str(self.dictionary_file))
                    logger.info(f"기존 사전 데이터 로드 완료: {len(self.dictionary.terms)}개 용어")
            else:
                logger.info("기존 사전 파일이 없습니다.")
                
        except Exception as e:
            logger.error(f"기존 사전 파일 로드 실패: {e}")
    
    def _merge_existing_data(self, target_year: int):
        """기존 데이터 병합 (같은 연도의 다른 세션에서)"""
        try:
            # 같은 연도의 다른 세션 폴더들 찾기
            session_folders = list(Path("data/raw/legal_terms").glob(f"session_*_year_{target_year}"))
            
            merged_count = 0
            for session_folder in session_folders:
                # 현재 세션 폴더는 제외
                if session_folder == self.dictionary_file.parent:
                    continue
                
                # 해당 세션의 사전 파일들 로드
                dictionary_files = list(session_folder.glob("legal_terms_*.json"))
                for dict_file in dictionary_files:
                    try:
                        # 임시 사전 객체로 로드
                        temp_dict = LegalTermDictionary()
                        temp_dict.load_terms_from_file(str(dict_file))
                        
                        # 현재 사전에 병합
                        for term_id, term_data in temp_dict.terms.items():
                            if term_id not in self.dictionary.terms:
                                self.dictionary.add_term(term_data)
                                merged_count += 1
                        
                        logger.info(f"세션 병합 완료: {dict_file.name} ({len(temp_dict.terms)}개 용어)")
                        
                    except Exception as e:
                        logger.debug(f"세션 병합 실패 ({dict_file}): {e}")
                        continue
            
            if merged_count > 0:
                logger.info(f"기존 데이터 병합 완료: {merged_count}개 용어 추가")
                self.checkpoint_data.collected_count = len(self.dictionary.terms)
            else:
                logger.info("병합할 기존 데이터가 없습니다.")
                
        except Exception as e:
            logger.error(f"기존 데이터 병합 실패: {e}")
    
    def _create_checkpoint_data(self, total_target: int) -> CheckpointData:
        """체크포인트 데이터 생성"""
        return CheckpointData(
            session_id=self.session_id,
            start_time=datetime.now().isoformat(),
            last_update=datetime.now().isoformat(),
            total_target=total_target,
            collected_count=0,
            memory_usage_mb=self._get_memory_usage_mb(),
            api_requests_made=0,
            errors_count=0,
            can_resume=True
        )
    
    def collect_all_terms(self, max_terms: int = 5000, use_mock_data: bool = False, resume: bool = True) -> bool:
        """모든 법률 용어 수집 (메모리 최적화 및 체크포인트 지원)"""
        logger.info(f"전체 법률 용어 수집 시작 - 최대 {max_terms}개 (메모리 최적화)")
        
        # 동적 파일명 설정
        self._setup_dynamic_filenames("all")
        
        # 체크포인트 로드 또는 생성
        if resume:
            self.checkpoint_data = self._load_checkpoint()
        
        if not self.checkpoint_data:
            self.checkpoint_data = self._create_checkpoint_data(max_terms)
            logger.info("새로운 수집 세션 시작")
        else:
            # 기존 사전 로드 후 실제 용어 수로 동기화
            self.load_dictionary()
            actual_count = len(self.dictionary.terms)
            if actual_count != self.checkpoint_data.collected_count:
                logger.info(f"체크포인트와 실제 사전 용어 수 불일치 - 체크포인트: {self.checkpoint_data.collected_count}개, 실제: {actual_count}개")
                logger.info(f"실제 사전 용어 수로 동기화: {actual_count}개")
                self.checkpoint_data.collected_count = actual_count
            else:
                logger.info(f"기존 수집 세션 재개: {self.checkpoint_data.collected_count}개 용어 수집됨")
        
        self.stats['start_time'] = datetime.now()
        
        try:
            # 배치 단위로 용어 수집
            success = self._collect_terms_in_batches(max_terms, use_mock_data)
            
            self.stats['end_time'] = datetime.now()
            self._log_collection_summary()
            
            return success
            
        except Exception as e:
            logger.error(f"용어 수집 실패: {e}")
            self.checkpoint_data.errors_count += 1
            self._save_checkpoint()
            return False
    
    def _collect_terms_in_batches(self, max_terms: int, use_mock_data: bool) -> bool:
        """배치 단위로 용어 수집"""
        batch_size = self.memory_config.batch_size
        collected_count = self.checkpoint_data.collected_count
        remaining_terms = max_terms - collected_count
        
        logger.info(f"배치 수집 시작 - 배치 크기: {batch_size}, 남은 용어: {remaining_terms}개")
        
        batch_count = 0
        while collected_count < max_terms and not self.shutdown_requested:
            # 메모리 체크
            if not self._check_memory_limit():
                logger.warning("메모리 제한으로 인한 일시 중단")
                break
            
            # 현재 배치 크기 계산
            current_batch_size = min(batch_size, remaining_terms)
            
            try:
                # 배치 단위로 용어 수집
                batch_terms = self.api_client.collect_legal_terms(
                    max_terms=current_batch_size, 
                    use_mock_data=use_mock_data
                )
                
                if not batch_terms:
                    logger.info("더 이상 수집할 용어가 없습니다.")
                    break
                
                # 사전에 배치 용어 추가
                batch_success = self._add_terms_to_dictionary_batch(batch_terms)
                
                if batch_success:
                    # 실제로 사전에 저장된 용어 수로 업데이트
                    collected_count = len(self.dictionary.terms)
                    self.checkpoint_data.collected_count = collected_count
                    remaining_terms = max_terms - collected_count
                    
                    logger.info(f"배치 {batch_count + 1} 완료 - 수집된 용어: {collected_count}/{max_terms}개")
                    
                    # 체크포인트 저장
                    if (batch_count + 1) % self.memory_config.checkpoint_interval == 0:
                        self._save_checkpoint()
                
                batch_count += 1
                
                # 배치 간 랜덤 지연 (2~5초)
                if not self.shutdown_requested:
                    delay = random.uniform(self.memory_config.batch_delay_min, self.memory_config.batch_delay_max)
                    logger.info(f"배치 간 지연: {delay:.1f}초")
                    time.sleep(delay)
                
                # 가비지 컬렉션 임계값 체크
                if batch_count % self.memory_config.gc_threshold == 0:
                    self._force_garbage_collection()
                
            except Exception as e:
                logger.error(f"배치 {batch_count + 1} 수집 실패: {e}")
                self.checkpoint_data.errors_count += 1
                
                # 연속 에러 시 중단
                if self.checkpoint_data.errors_count > 10:
                    logger.error("연속 에러로 인한 수집 중단")
                    break
        
        # 최종 체크포인트 저장 (실제 사전 용어 수로 업데이트)
        self.checkpoint_data.collected_count = len(self.dictionary.terms)
        self._save_checkpoint()
        
        logger.info(f"배치 수집 완료 - 총 {collected_count}개 용어 수집")
        return collected_count > 0
    
    def collect_terms_by_categories(self, categories: List[str], max_terms_per_category: int = 500, resume: bool = True) -> bool:
        """카테고리별 용어 수집 (메모리 최적화 및 체크포인트 지원)"""
        logger.info(f"카테고리별 용어 수집 시작 - {len(categories)}개 카테고리 (메모리 최적화)")
        
        # 동적 파일명 설정
        self._setup_dynamic_filenames("categories")
        
        # 체크포인트 로드 또는 생성
        if resume:
            self.checkpoint_data = self._load_checkpoint()
        
        if not self.checkpoint_data:
            total_target = len(categories) * max_terms_per_category
            self.checkpoint_data = self._create_checkpoint_data(total_target)
            logger.info("새로운 카테고리별 수집 세션 시작")
        else:
            logger.info(f"기존 카테고리별 수집 세션 재개: {self.checkpoint_data.collected_count}개 용어 수집됨")
        
        self.stats['start_time'] = datetime.now()
        
        try:
            # 카테고리별 배치 수집
            success = self._collect_categories_in_batches(categories, max_terms_per_category)
            
            self.stats['end_time'] = datetime.now()
            self._log_collection_summary()
            
            return success
            
        except Exception as e:
            logger.error(f"카테고리별 용어 수집 실패: {e}")
            self.checkpoint_data.errors_count += 1
            self._save_checkpoint()
            return False
    
    def _collect_categories_in_batches(self, categories: List[str], max_terms_per_category: int) -> bool:
        """카테고리별 배치 수집"""
        batch_size = self.memory_config.batch_size
        processed_categories = set(self.checkpoint_data.processed_categories)
        
        logger.info(f"카테고리별 배치 수집 시작 - 배치 크기: {batch_size}")
        
        for category in categories:
            if self.shutdown_requested:
                logger.info("중단 요청으로 인한 수집 중단")
                break
            
            if category in processed_categories:
                logger.info(f"카테고리 '{category}' 이미 처리됨 - 건너뛰기")
                continue
            
            logger.info(f"카테고리 '{category}' 수집 시작...")
            
            try:
                # 카테고리별 배치 수집
                category_success = self._collect_single_category_batches(category, max_terms_per_category, batch_size)
                
                if category_success:
                    self.checkpoint_data.processed_categories.append(category)
                    logger.info(f"카테고리 '{category}' 완료")
                else:
                    logger.warning(f"카테고리 '{category}' 수집 실패")
                
                # 체크포인트 저장
                self._save_checkpoint()
                
                # 메모리 체크 및 가비지 컬렉션
                if not self._check_memory_limit():
                    logger.warning("메모리 제한으로 인한 일시 중단")
                    break
                
            except Exception as e:
                logger.error(f"카테고리 '{category}' 수집 중 오류: {e}")
                self.checkpoint_data.errors_count += 1
        
        logger.info(f"카테고리별 배치 수집 완료 - 총 {self.checkpoint_data.collected_count}개 용어 수집")
        return self.checkpoint_data.collected_count > 0
    
    def _collect_single_category_batches(self, category: str, max_terms: int, batch_size: int) -> bool:
        """단일 카테고리 배치 수집"""
        collected_count = 0
        batch_count = 0
        
        while collected_count < max_terms and not self.shutdown_requested:
            # 메모리 체크
            if not self._check_memory_limit():
                break
            
            # 현재 배치 크기 계산
            current_batch_size = min(batch_size, max_terms - collected_count)
            
            try:
                # 배치 단위로 용어 수집
                batch_terms = self.api_client.collect_legal_terms(category, current_batch_size)
                
                if not batch_terms:
                    logger.info(f"카테고리 '{category}'에서 더 이상 수집할 용어가 없습니다.")
                    break
                
                # 사전에 배치 용어 추가
                batch_success = self._add_terms_to_dictionary_batch(batch_terms)
                
                if batch_success:
                    # 실제로 사전에 저장된 용어 수로 업데이트
                    collected_count = len(self.dictionary.terms)
                    self.checkpoint_data.collected_count = collected_count
                    
                    logger.info(f"카테고리 '{category}' 배치 {batch_count + 1} 완료 - 수집된 용어: {collected_count}/{max_terms}개")
                
                batch_count += 1
                
                # 배치 간 랜덤 지연 (2~5초)
                if not self.shutdown_requested:
                    delay = random.uniform(self.memory_config.batch_delay_min, self.memory_config.batch_delay_max)
                    logger.info(f"카테고리 '{category}' 배치 간 지연: {delay:.1f}초")
                    time.sleep(delay)
                
                # 가비지 컬렉션 임계값 체크
                if batch_count % self.memory_config.gc_threshold == 0:
                    self._force_garbage_collection()
                
            except Exception as e:
                logger.error(f"카테고리 '{category}' 배치 {batch_count + 1} 수집 실패: {e}")
                self.checkpoint_data.errors_count += 1
                break
        
        return collected_count > 0
    
    def collect_terms_by_keywords(self, keywords: List[str], max_terms_per_keyword: int = 100, resume: bool = True) -> bool:
        """키워드별 용어 수집 (메모리 최적화 및 체크포인트 지원)"""
        logger.info(f"키워드별 용어 수집 시작 - {len(keywords)}개 키워드 (메모리 최적화)")
        
        # 동적 파일명 설정
        self._setup_dynamic_filenames("keywords")
        
        # 체크포인트 로드 또는 생성
        if resume:
            self.checkpoint_data = self._load_checkpoint()
        
        if not self.checkpoint_data:
            total_target = len(keywords) * max_terms_per_keyword
            self.checkpoint_data = self._create_checkpoint_data(total_target)
            logger.info("새로운 키워드별 수집 세션 시작")
        else:
            logger.info(f"기존 키워드별 수집 세션 재개: {self.checkpoint_data.collected_count}개 용어 수집됨")
        
        self.stats['start_time'] = datetime.now()
        
        try:
            # 키워드별 배치 수집
            success = self._collect_keywords_in_batches(keywords, max_terms_per_keyword)
            
            self.stats['end_time'] = datetime.now()
            self._log_collection_summary()
            
            return success
            
        except Exception as e:
            logger.error(f"키워드별 용어 수집 실패: {e}")
            self.checkpoint_data.errors_count += 1
            self._save_checkpoint()
            return False
    
    def _collect_keywords_in_batches(self, keywords: List[str], max_terms_per_keyword: int) -> bool:
        """키워드별 배치 수집"""
        batch_size = self.memory_config.batch_size
        processed_keywords = set(self.checkpoint_data.processed_keywords)
        
        logger.info(f"키워드별 배치 수집 시작 - 배치 크기: {batch_size}")
        
        for keyword in keywords:
            if self.shutdown_requested:
                logger.info("중단 요청으로 인한 수집 중단")
                break
            
            if keyword in processed_keywords:
                logger.info(f"키워드 '{keyword}' 이미 처리됨 - 건너뛰기")
                continue
            
            logger.info(f"키워드 '{keyword}' 수집 시작...")
            
            try:
                # 키워드별 배치 수집
                keyword_success = self._collect_single_keyword_batches(keyword, max_terms_per_keyword, batch_size)
                
                if keyword_success:
                    self.checkpoint_data.processed_keywords.append(keyword)
                    logger.info(f"키워드 '{keyword}' 완료")
                else:
                    logger.warning(f"키워드 '{keyword}' 수집 실패")
                
                # 체크포인트 저장
                self._save_checkpoint()
                
                # 메모리 체크 및 가비지 컬렉션
                if not self._check_memory_limit():
                    logger.warning("메모리 제한으로 인한 일시 중단")
                    break
                
            except Exception as e:
                logger.error(f"키워드 '{keyword}' 수집 중 오류: {e}")
                self.checkpoint_data.errors_count += 1
        
        logger.info(f"키워드별 배치 수집 완료 - 총 {self.checkpoint_data.collected_count}개 용어 수집")
        return self.checkpoint_data.collected_count > 0
    
    def _collect_single_keyword_batches(self, keyword: str, max_terms: int, batch_size: int) -> bool:
        """단일 키워드 배치 수집"""
        collected_count = 0
        batch_count = 0
        
        while collected_count < max_terms and not self.shutdown_requested:
            # 메모리 체크
            if not self._check_memory_limit():
                break
            
            # 현재 배치 크기 계산
            current_batch_size = min(batch_size, max_terms - collected_count)
            
            try:
                # 배치 단위로 용어 수집
                batch_terms = self.api_client.collect_legal_terms(keyword, current_batch_size)
                
                if not batch_terms:
                    logger.info(f"키워드 '{keyword}'에서 더 이상 수집할 용어가 없습니다.")
                    break
                
                # 사전에 배치 용어 추가
                batch_success = self._add_terms_to_dictionary_batch(batch_terms)
                
                if batch_success:
                    # 실제로 사전에 저장된 용어 수로 업데이트
                    collected_count = len(self.dictionary.terms)
                    self.checkpoint_data.collected_count = collected_count
                    
                    logger.info(f"키워드 '{keyword}' 배치 {batch_count + 1} 완료 - 수집된 용어: {collected_count}/{max_terms}개")
                
                batch_count += 1
                
                # 배치 간 랜덤 지연 (2~5초)
                if not self.shutdown_requested:
                    delay = random.uniform(self.memory_config.batch_delay_min, self.memory_config.batch_delay_max)
                    logger.info(f"키워드 '{keyword}' 배치 간 지연: {delay:.1f}초")
                    time.sleep(delay)
                
                # 가비지 컬렉션 임계값 체크
                if batch_count % self.memory_config.gc_threshold == 0:
                    self._force_garbage_collection()
                
            except Exception as e:
                logger.error(f"키워드 '{keyword}' 배치 {batch_count + 1} 수집 실패: {e}")
                self.checkpoint_data.errors_count += 1
                break
        
        return collected_count > 0
    
    def _setup_dynamic_filenames(self, collection_type: str, target_year: int = None):
        """동적 파일명 설정 (세션별 구분)"""
        try:
            # 파일명 접두사 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if target_year:
                self.dynamic_file_prefix = f"{timestamp}_{collection_type}_{target_year}"
            else:
                self.dynamic_file_prefix = f"{timestamp}_{collection_type}"
            
            # 세션별 폴더 생성
            session_folder = Path(f"data/raw/legal_terms/session_{self.dynamic_file_prefix}")
            session_folder.mkdir(parents=True, exist_ok=True)
            
            # 파일 경로 설정 (세션 폴더 내부)
            self.dictionary_file = session_folder / f"legal_terms_{self.dynamic_file_prefix}.json"
            self.checkpoint_file = session_folder / f"checkpoint_{self.dynamic_file_prefix}.json"
            
            logger.info(f"동적 파일명 설정 완료:")
            logger.info(f"  세션 폴더: {session_folder}")
            logger.info(f"  사전 파일: {self.dictionary_file}")
            logger.info(f"  체크포인트 파일: {self.checkpoint_file}")
            
        except Exception as e:
            logger.error(f"동적 파일명 설정 실패: {e}")
            # 기본 파일명으로 폴백
            self.dictionary_file = Path("data/raw/legal_terms/legal_term_dictionary.json")
            self.checkpoint_file = Path("data/raw/legal_terms/checkpoint.json")
    
    def collect_terms_by_year(self, year: int, max_terms: int = None, resume: bool = True) -> bool:
        """지정 연도 용어 수집 (메모리 최적화 및 체크포인트 지원)"""
        # 무제한 수집 모드 (max_terms가 None이거나 매우 큰 값인 경우)
        if max_terms is None:
            max_terms = 999999999  # 거의 무제한
            logger.info(f"{year}년 용어 수집 시작 - 무제한 모드 (메모리 최적화)")
        else:
            logger.info(f"{year}년 용어 수집 시작 - 최대 {max_terms}개 (메모리 최적화)")
        
        # 동적 파일명 설정
        self._setup_dynamic_filenames("year", year)
        
        # 체크포인트 로드 또는 생성
        if resume:
            self.checkpoint_data = self._load_checkpoint()
        
        if not self.checkpoint_data:
            self.checkpoint_data = self._create_checkpoint_data(max_terms)
            logger.info(f"🆕 새로운 {year}년 수집 세션 시작")
            logger.info(f"📁 세션 폴더: {self.dictionary_file.parent}")
            logger.info(f"📁 사전 파일: {self.dictionary_file}")
            logger.info(f"📁 체크포인트 파일: {self.checkpoint_file}")
            logger.info(f"📊 목표 용어 수: {max_terms}개")
        else:
            # 기존 체크포인트의 목표 용어 수 업데이트 (무제한 모드인 경우)
            if max_terms == 999999999 or self.checkpoint_data.total_target < max_terms:
                logger.info(f"목표 용어 수 업데이트: {self.checkpoint_data.total_target} → {max_terms}")
                self.checkpoint_data.total_target = max_terms
            
            # 기존 데이터 병합
            self._merge_existing_data(year)
            
            # 기존 사전 로드 후 실제 용어 수로 동기화
            self.load_dictionary()
            actual_count = len(self.dictionary.terms)
            if actual_count != self.checkpoint_data.collected_count:
                logger.info(f"체크포인트와 실제 사전 용어 수 불일치 - 체크포인트: {self.checkpoint_data.collected_count}개, 실제: {actual_count}개")
                logger.info(f"실제 사전 용어 수로 동기화: {actual_count}개")
                self.checkpoint_data.collected_count = actual_count
            else:
                logger.info(f"기존 {year}년 수집 세션 재개: {self.checkpoint_data.collected_count}개 용어 수집됨")
        
        self.stats['start_time'] = datetime.now()
        
        try:
            # 단일 연도 배치 수집
            success = self._collect_single_year_batches(year, max_terms)
            
            self.stats['end_time'] = datetime.now()
            self._log_collection_summary()
            
            return success
            
        except Exception as e:
            logger.error(f"{year}년 용어 수집 실패: {e}")
            self.checkpoint_data.errors_count += 1
            self._save_checkpoint()
            return False
    
    
    def _collect_single_year_batches(self, year: int, max_terms: int) -> bool:
        """단일 연도 배치 수집"""
        batch_size = self.memory_config.batch_size
        collected_count = self.checkpoint_data.collected_count
        
        # 무제한 모드인 경우 남은 용어 계산 생략
        if max_terms == 999999999:
            remaining_terms = "무제한"
            logger.info(f"{year}년 배치 수집 시작 - 배치 크기: {batch_size}, 모드: 무제한")
        else:
            remaining_terms = max_terms - collected_count
            logger.info(f"{year}년 배치 수집 시작 - 배치 크기: {batch_size}, 남은 용어: {remaining_terms}개")
        
        # 연도별 날짜 범위 설정
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        batch_count = 0
        consecutive_empty_batches = 0  # 연속으로 빈 배치가 나온 횟수
        max_empty_batches = 5  # 최대 연속 빈 배치 허용 횟수
        
        while (max_terms == 999999999 or collected_count < max_terms) and not self.shutdown_requested:
            # shutdown 요청 확인 (더 자주 체크)
            if self.shutdown_requested:
                logger.info("종료 요청으로 인한 수집 중단")
                break
            
            # 메모리 체크
            if not self._check_memory_limit():
                logger.warning("메모리 제한으로 인한 일시 중단")
                break
            
            # 현재 배치 크기 계산 (무제한 모드인 경우 배치 크기 그대로 사용)
            if max_terms == 999999999:
                current_batch_size = batch_size
            else:
                current_batch_size = min(batch_size, remaining_terms)
            
            try:
                # 배치 단위로 용어 수집 (날짜 범위 포함)
                batch_terms = self.api_client.collect_legal_terms(
                    max_terms=current_batch_size,
                    start_date=start_date,
                    end_date=end_date,
                    checkpoint_data=self.checkpoint_data.__dict__ if self.checkpoint_data else None,
                    session_folder=str(self.dictionary_file.parent) if self.dictionary_file else None
                )
                
                # shutdown 요청 재확인 (API 호출 후)
                if self.shutdown_requested:
                    logger.info("API 호출 후 종료 요청 확인 - 수집 중단")
                    break
                
                if not batch_terms:
                    consecutive_empty_batches += 1
                    logger.info(f"{year}년 배치 {batch_count + 1} - 수집된 용어 없음 (연속 빈 배치: {consecutive_empty_batches}/{max_empty_batches})")
                    
                    # 연속으로 빈 배치가 많이 나오면 중단
                    if consecutive_empty_batches >= max_empty_batches:
                        logger.info(f"{year}년에서 연속 {max_empty_batches}회 빈 배치로 인한 수집 중단")
                        break
                    
                    # 빈 배치 후에도 지연 적용
                    if not self.shutdown_requested:
                        delay = random.uniform(self.memory_config.batch_delay_min, self.memory_config.batch_delay_max)
                        logger.info(f"빈 배치 후 지연: {delay:.1f}초")
                        time.sleep(delay)
                    
                    batch_count += 1
                    continue
                else:
                    # 용어가 수집되면 연속 빈 배치 카운터 리셋
                    consecutive_empty_batches = 0
                
                # 사전에 배치 용어 추가
                batch_success = self._add_terms_to_dictionary_batch(batch_terms)
                
                if batch_success:
                    # 실제로 사전에 저장된 용어 수로 업데이트 (가장 정확한 방법)
                    collected_count = len(self.dictionary.terms)
                    self.checkpoint_data.collected_count = collected_count
                    
                    # 무제한 모드가 아닌 경우에만 남은 용어 계산
                    if max_terms != 999999999:
                        remaining_terms = max_terms - collected_count
                        logger.info(f"✅ {year}년 배치 {batch_count + 1} 완료 - 수집된 용어: {collected_count}/{max_terms}개")
                    else:
                        logger.info(f"✅ {year}년 배치 {batch_count + 1} 완료 - 수집된 용어: {collected_count}개 (무제한 모드)")
                    
                    # 파일 생성 상황 로그
                    logger.info(f"📁 현재 사전 파일: {self.dictionary_file}")
                    logger.info(f"📁 현재 체크포인트: {self.checkpoint_file}")
                    logger.info(f"📊 사전에 저장된 용어 수: {len(self.dictionary.terms)}개")
                    
                    
                    # 체크포인트 저장 및 페이지 파일 즉시 저장
                    if (batch_count + 1) % self.memory_config.checkpoint_interval == 0:
                        logger.info(f"💾 체크포인트 저장 중... (배치 {batch_count + 1}마다 저장)")
                        self._save_checkpoint()
                        logger.info(f"💾 페이지 파일 저장 중...")
                        self._save_page_files_immediately()
                        logger.info(f"✅ 파일 저장 완료!")
                
                batch_count += 1
                
                # 배치 간 랜덤 지연 (2~5초)
                if not self.shutdown_requested:
                    delay = random.uniform(self.memory_config.batch_delay_min, self.memory_config.batch_delay_max)
                    logger.info(f"⏳ {year}년 배치 간 지연: {delay:.1f}초")
                    time.sleep(delay)
                
                # 가비지 컬렉션 임계값 체크
                if batch_count % self.memory_config.gc_threshold == 0:
                    self._force_garbage_collection()
                
            except Exception as e:
                logger.error(f"{year}년 배치 {batch_count + 1} 수집 실패: {e}")
                self.checkpoint_data.errors_count += 1
                
                # 연속 에러 시 중단
                if self.checkpoint_data.errors_count > 10:
                    logger.error("연속 에러로 인한 수집 중단")
                    break
        
        # 최종 체크포인트 저장 (실제 사전 용어 수로 업데이트)
        logger.info(f"💾 최종 체크포인트 저장 중...")
        self.checkpoint_data.collected_count = len(self.dictionary.terms)
        self._save_checkpoint()
        logger.info(f"💾 최종 페이지 파일 저장 중...")
        self._save_page_files_immediately()
        
        logger.info(f"🎉 {year}년 배치 수집 완료 - 총 {collected_count}개 용어 수집")
        logger.info(f"📁 최종 사전 파일: {self.dictionary_file}")
        logger.info(f"📁 최종 체크포인트: {self.checkpoint_file}")
        logger.info(f"📊 최종 사전 용어 수: {len(self.dictionary.terms)}개")
        return collected_count > 0
    
    def _add_terms_to_dictionary_batch(self, terms: List[Dict[str, Any]]) -> bool:
        """배치 단위로 사전에 용어 추가 (메모리 최적화)"""
        
        success_count = 0
        fail_count = 0
        category_stats = {}
        
        for term_data in terms:
            try:
                if self.dictionary.add_term(term_data):
                    success_count += 1
                    
                    # 카테고리별 통계 수집
                    category = term_data.get('category', '기타')
                    category_stats[category] = category_stats.get(category, 0) + 1
                else:
                    fail_count += 1
                    
            except Exception as e:
                logger.error(f"용어 추가 실패: {e}")
                fail_count += 1
        
        # 기본 통계 업데이트
        self.stats['total_collected'] = len(self.dictionary.terms)
        
        return success_count > 0
    
    def _add_terms_to_dictionary(self, terms: List[Dict[str, Any]]) -> bool:
        """수집된 용어를 사전에 추가"""
        logger.info(f"사전에 {len(terms)}개 용어 추가 중...")
        
        success_count = 0
        fail_count = 0
        category_stats = {}
        
        for i, term_data in enumerate(terms):
            try:
                if self.dictionary.add_term(term_data):
                    success_count += 1
                    
                    # 카테고리별 통계 수집
                    category = term_data.get('category', '기타')
                    category_stats[category] = category_stats.get(category, 0) + 1
                else:
                    fail_count += 1
                
                # 진행률 로그 (100개마다)
                if (i + 1) % 100 == 0:
                    logger.info(f"용어 추가 진행: {i + 1}/{len(terms)} ({success_count}개 성공, {fail_count}개 실패)")
                    
            except Exception as e:
                logger.error(f"용어 추가 실패: {e}")
                fail_count += 1
        
        # 기본 통계 업데이트
        self.stats['total_collected'] = len(self.dictionary.terms)
        
        logger.info(f"사전 추가 완료 - {success_count}개 성공, {fail_count}개 실패")
        
        # 카테고리별 통계 출력
        logger.info("카테고리별 용어 수집 통계:")
        for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {category}: {count}개")
        
        return success_count > 0
    
    def _log_collection_summary(self):
        """수집 요약 로그 출력"""
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            logger.info(f"수집 완료 - 총 용어: {len(self.dictionary.terms)}개, 소요 시간: {duration.total_seconds():.2f}초")
        else:
            logger.info(f"수집 완료 - 총 용어: {len(self.dictionary.terms)}개")
    
    
    def get_progress_status(self) -> Dict[str, Any]:
        """진행 상태 조회"""
        if not self.checkpoint_data:
            return {'status': 'not_started'}
        
        progress_percent = (self.checkpoint_data.collected_count / self.checkpoint_data.total_target) * 100
        
        return {
            'status': 'in_progress' if not self.shutdown_requested else 'paused',
            'collected_count': len(self.dictionary.terms),
            'total_target': self.checkpoint_data.total_target
        }
    
    def clear_checkpoint(self) -> bool:
        """체크포인트 삭제"""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                logger.info("체크포인트 삭제 완료")
                return True
            else:
                logger.info("체크포인트 파일이 존재하지 않습니다.")
                return True
        except Exception as e:
            logger.error(f"체크포인트 삭제 실패: {e}")
            return False
    
    def optimize_memory_settings(self, available_memory_mb: int = None) -> bool:
        """메모리 설정 최적화"""
        try:
            if available_memory_mb is None:
                # 시스템 메모리 정보 조회
                memory_info = psutil.virtual_memory()
                available_memory_mb = memory_info.available / 1024 / 1024
            
            # 안전한 메모리 사용량 설정 (전체의 70%)
            safe_memory_mb = int(available_memory_mb * 0.7)
            
            # 배치 크기 동적 조정
            if safe_memory_mb > 4096:  # 4GB 이상
                batch_size = 100
            elif safe_memory_mb > 2048:  # 2GB 이상
                batch_size = 50
            else:  # 2GB 미만
                batch_size = 25
            
            # 메모리 설정 업데이트
            self.memory_config.max_memory_mb = safe_memory_mb
            self.memory_config.batch_size = batch_size
            
            logger.info(f"메모리 설정 최적화 완료:")
            logger.info(f"  사용 가능 메모리: {available_memory_mb:.1f}MB")
            logger.info(f"  안전 메모리 한계: {safe_memory_mb:.1f}MB")
            logger.info(f"  배치 크기: {batch_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"메모리 설정 최적화 실패: {e}")
            return False
    
    def save_dictionary(self, file_path: str = None) -> bool:
        """사전 저장 (배치별 분리 파일 지원)"""
        try:
            if file_path is None:
                # 동적 파일명이 설정되어 있으면 사용, 아니면 기본 파일명 사용
                if self.dictionary_file:
                    file_path = str(self.dictionary_file)
                else:
                    file_path = "data/raw/legal_terms/legal_term_dictionary.json"
            
            # 배치별 분리 파일로 저장
            return self._save_dictionary_batch_separated(file_path)
            
        except Exception as e:
            logger.error(f"사전 저장 중 오류: {e}")
            return False
    
    def _save_dictionary_batch_separated(self, base_file_path: str) -> bool:
        """배치별로 분리된 파일로 사전 저장 (중복 방지)"""
        try:
            base_path = Path(base_file_path)
            session_folder = base_path.parent
            
            # 기존 메인 사전 파일 확인
            if base_path.exists():
                logger.info(f"메인 사전 파일 {base_path.name}이 이미 존재함 - 건너뛰기")
            else:
                # 1. 전체 사전 파일 저장 (기존 방식)
                self.dictionary.dictionary_path = base_path
                success = self.dictionary.save_dictionary()
                
                if not success:
                    return False
            
            # 2. 페이지별로 분리 저장 (기존 페이지 파일들이 있으면 그대로 유지)
            existing_page_files = list(session_folder.glob("legal_terms_page_*.json"))
            if existing_page_files:
                logger.info(f"기존 페이지 파일 {len(existing_page_files)}개 발견 - 추가 저장 생략")
            else:
                # 페이지별로 분리 저장
                self._save_batch_files(session_folder)
            
            logger.info(f"배치별 분리 사전 저장 완료: {base_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"배치별 분리 사전 저장 실패: {e}")
            return False
    
    def _save_batch_files(self, session_folder: Path):
        """API 페이지별로 파일 분리 저장 (중복 방지)"""
        try:
            # API 페이지 크기 설정 (기본 100개 - API 기본값)
            page_size = 100
            
            # 기존 페이지 파일들 확인
            existing_page_files = list(session_folder.glob("legal_terms_page_*.json"))
            existing_page_numbers = set()
            
            for file_path in existing_page_files:
                try:
                    # 파일명에서 페이지 번호 추출 (예: legal_terms_page_001.json -> 1)
                    page_num = int(file_path.stem.split('_')[-1])
                    existing_page_numbers.add(page_num)
                except (ValueError, IndexError):
                    continue
            
            if existing_page_numbers:
                logger.info(f"기존 페이지 파일 {len(existing_page_numbers)}개 발견: {sorted(existing_page_numbers)}")
            
            # 용어들을 페이지별로 그룹화
            terms_list = list(self.dictionary.terms.items())
            total_terms = len(terms_list)
            
            if total_terms == 0:
                logger.info("저장할 용어가 없습니다.")
                return
            
            # 페이지 파일 저장
            page_count = 0
            for i in range(0, total_terms, page_size):
                page_terms = dict(terms_list[i:i + page_size])
                page_count += 1
                
                # 페이지 파일명 생성
                page_file = session_folder / f"legal_terms_page_{page_count:03d}.json"
                
                # 기존 파일이 있는지 확인
                if page_file.exists():
                    logger.info(f"페이지 파일 {page_file.name}이 이미 존재함 - 건너뛰기")
                    continue
                
                # 페이지 데이터 구성
                page_data = {
                    "metadata": {
                        "page_number": page_count,
                        "page_size": len(page_terms),
                        "total_pages": (total_terms + page_size - 1) // page_size,
                        "saved_at": datetime.now().isoformat(),
                        "session_id": self.session_id,
                        "api_page_size": page_size
                    },
                    "terms": page_terms
                }
                
                # 페이지 파일 저장
                with open(page_file, 'w', encoding='utf-8') as f:
                    json.dump(page_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"📄 페이지 파일 저장 완료: {page_file.name} ({len(page_terms)}개 용어)")
            
            logger.info(f"📊 총 {page_count}개 페이지 파일 생성 완료")
            
        except Exception as e:
            logger.error(f"페이지 파일 저장 실패: {e}")
    
    def _save_page_files_immediately(self):
        """체크포인트마다 즉시 페이지 파일 저장"""
        try:
            if not self.dictionary_file:
                logger.warning("사전 파일 경로가 설정되지 않았습니다.")
                return
            
            session_folder = self.dictionary_file.parent
            logger.info(f"📁 세션 폴더: {session_folder}")
            self._save_batch_files(session_folder)
            logger.info("✅ 체크포인트마다 페이지 파일 저장 완료")
            
        except Exception as e:
            logger.error(f"❌ 즉시 페이지 파일 저장 실패: {e}")
    
    def load_dictionary(self, file_path: str = None) -> bool:
        """사전 로드 (페이지 파일 지원)"""
        try:
            if file_path is None:
                file_path = "data/raw/legal_terms/legal_term_dictionary.json"
            
            # 페이지 파일들도 함께 로드
            return self._load_dictionary_with_pages(file_path)
            
        except Exception as e:
            logger.error(f"사전 로드 중 오류: {e}")
            return False
    
    def _load_dictionary_with_pages(self, base_file_path: str) -> bool:
        """페이지 파일들을 포함하여 사전 로드"""
        try:
            base_path = Path(base_file_path)
            session_folder = base_path.parent
            
            # 1. 기본 사전 파일 로드
            success = self.dictionary.load_terms_from_file(base_file_path)
            if success:
                logger.info(f"기본 사전 로드 완료: {base_file_path}")
            
            # 2. 페이지 파일들 로드
            self._load_page_files(session_folder)
            
            return success
            
        except Exception as e:
            logger.error(f"페이지 파일 포함 사전 로드 실패: {e}")
            return False
    
    def _load_page_files(self, session_folder: Path):
        """페이지 파일들 로드"""
        try:
            page_files = list(session_folder.glob("legal_terms_page_*.json"))
            if not page_files:
                logger.info("페이지 파일이 없습니다.")
                return
            
            # 페이지 번호 순으로 정렬
            page_files.sort(key=lambda f: int(f.stem.split('_')[-1]))
            
            loaded_count = 0
            for page_file in page_files:
                try:
                    with open(page_file, 'r', encoding='utf-8') as f:
                        page_data = json.load(f)
                    
                    # 페이지 데이터에서 용어들 추출
                    terms = page_data.get('terms', {})
                    page_number = page_data.get('metadata', {}).get('page_number', 0)
                    
                    for term_id, term_data in terms.items():
                        if self.dictionary.add_term(term_data):
                            loaded_count += 1
                    
                    logger.info(f"페이지 파일 로드: {page_file.name} (페이지 #{page_number}, {len(terms)}개 용어)")
                    
                except Exception as e:
                    logger.debug(f"페이지 파일 로드 실패 ({page_file}): {e}")
                    continue
            
            if loaded_count > 0:
                logger.info(f"페이지 파일 로드 완료: {loaded_count}개 용어 추가")
                
        except Exception as e:
            logger.error(f"페이지 파일 로드 실패: {e}")
    
    
    
