#!/usr/bin/env python3
"""
헌재결정례 수집 전용 로거

헌재결정례 수집 과정에서 발생하는 로그를 체계적으로 관리합니다.
- 로그 레벨별 분리 (INFO, WARNING, ERROR)
- 파일 및 콘솔 출력 지원
- 로그 로테이션 기능
- 성능 모니터링 로그
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    헌재결정례 수집용 로거 설정
    
    Args:
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 로그 파일 경로 (None이면 자동 생성)
        console_output: 콘솔 출력 여부
    
    Returns:
        설정된 로거 객체
    """
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 로그 파일명 설정
    if not log_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"constitutional_decision_collection_{timestamp}.log"
    else:
        log_file = Path(log_file)
    
    # 로거 생성
    logger = logging.getLogger('constitutional_decision_collector')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 기존 핸들러 제거 (중복 방지)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 로그 포맷 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 파일 핸들러 추가
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # 파일에는 모든 레벨 로그
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러 추가
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Windows에서 UTF-8 환경 설정
    if sys.platform.startswith('win'):
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        try:
            import subprocess
            subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
        except:
            pass
    
    return logger


def log_collection_start(logger: logging.Logger, target_count: int, resume: bool = False):
    """수집 시작 로그"""
    logger.info("=" * 80)
    logger.info("헌재결정례 수집 시작")
    logger.info("=" * 80)
    logger.info(f"목표 수집 건수: {target_count:,}건")
    logger.info(f"재시작 모드: {'예' if resume else '아니오'}")
    logger.info(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


def log_collection_progress(
    logger: logging.Logger,
    current_count: int,
    target_count: int,
    keyword: str,
    batch_count: int = 0
):
    """수집 진행 상황 로그"""
    progress_percentage = (current_count / target_count * 100) if target_count > 0 else 0
    
    if batch_count > 0:
        logger.info(
            f"진행률: {progress_percentage:.1f}% ({current_count:,}/{target_count:,}) "
            f"- 키워드: '{keyword}' - 배치: {batch_count}건"
        )
    else:
        logger.info(
            f"진행률: {progress_percentage:.1f}% ({current_count:,}/{target_count:,}) "
            f"- 키워드: '{keyword}'"
        )


def log_keyword_completion(
    logger: logging.Logger,
    keyword: str,
    collected_count: int,
    total_collected: int,
    api_requests: int,
    api_errors: int
):
    """키워드 완료 로그"""
    logger.info(f"키워드 '{keyword}' 완료 - 수집: {collected_count}건, 누적: {total_collected:,}건")
    logger.info(f"API 요청: {api_requests:,}회, 오류: {api_errors:,}회")


def log_batch_save(
    logger: logging.Logger,
    batch_count: int,
    category: str,
    file_path: str
):
    """배치 저장 로그"""
    logger.debug(f"배치 저장 완료 - {category}: {batch_count}건 -> {file_path}")


def log_api_error(
    logger: logging.Logger,
    operation: str,
    error: Exception,
    retry_count: int = 0
):
    """API 오류 로그"""
    if retry_count > 0:
        logger.warning(f"API 오류 ({operation}) - 재시도 {retry_count}회: {error}")
    else:
        logger.error(f"API 오류 ({operation}): {error}")


def log_collection_completion(
    logger: logging.Logger,
    total_collected: int,
    target_count: int,
    start_time: datetime,
    end_time: datetime,
    api_stats: dict
):
    """수집 완료 로그"""
    duration = end_time - start_time
    
    logger.info("=" * 80)
    logger.info("헌재결정례 수집 완료")
    logger.info("=" * 80)
    logger.info(f"수집 건수: {total_collected:,}건")
    logger.info(f"목표 건수: {target_count:,}건")
    logger.info(f"달성률: {(total_collected/target_count*100):.1f}%")
    logger.info(f"소요 시간: {duration}")
    logger.info(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"완료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"API 요청 수: {api_stats.get('total_requests', 0):,}회")
    logger.info(f"API 오류 수: {api_stats.get('error_count', 0):,}회")
    logger.info(f"남은 요청 수: {api_stats.get('remaining_requests', 0):,}회")
    logger.info("=" * 80)


def log_collection_interruption(
    logger: logging.Logger,
    total_collected: int,
    reason: str = "사용자 중단"
):
    """수집 중단 로그"""
    logger.warning("=" * 80)
    logger.warning("헌재결정례 수집 중단")
    logger.warning("=" * 80)
    logger.warning(f"중단 사유: {reason}")
    logger.warning(f"수집된 건수: {total_collected:,}건")
    logger.warning(f"중단 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.warning("재시작하려면 동일한 명령을 다시 실행하세요.")
    logger.warning("=" * 80)


def log_error_with_traceback(
    logger: logging.Logger,
    error: Exception,
    context: str = ""
):
    """상세 오류 로그 (스택 트레이스 포함)"""
    if context:
        logger.error(f"오류 발생 ({context}): {error}")
    else:
        logger.error(f"오류 발생: {error}")
    
    logger.error("스택 트레이스:")
    logger.error(traceback.format_exc())


def log_performance_metrics(
    logger: logging.Logger,
    operation: str,
    duration: float,
    count: int = 0
):
    """성능 메트릭 로그"""
    if count > 0:
        rate = count / duration if duration > 0 else 0
        logger.info(f"성능 메트릭 - {operation}: {duration:.2f}초, {count}건, {rate:.2f}건/초")
    else:
        logger.info(f"성능 메트릭 - {operation}: {duration:.2f}초")


def log_memory_usage(logger: logging.Logger, context: str = ""):
    """메모리 사용량 로그"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        if context:
            logger.debug(f"메모리 사용량 ({context}): {memory_mb:.2f}MB")
        else:
            logger.debug(f"메모리 사용량: {memory_mb:.2f}MB")
    except ImportError:
        logger.debug("psutil이 설치되지 않아 메모리 사용량을 확인할 수 없습니다.")
    except Exception as e:
        logger.debug(f"메모리 사용량 확인 실패: {e}")


def log_checkpoint_save(logger: logging.Logger, checkpoint_file: str, data_count: int):
    """체크포인트 저장 로그"""
    logger.debug(f"체크포인트 저장: {checkpoint_file} ({data_count}건)")


def log_checkpoint_load(logger: logging.Logger, checkpoint_file: str, data_count: int):
    """체크포인트 로드 로그"""
    logger.info(f"체크포인트 로드: {checkpoint_file} ({data_count}건)")


def log_api_rate_limit(
    logger: logging.Logger,
    remaining_requests: int,
    reset_time: Optional[str] = None
):
    """API 요청 제한 로그"""
    if remaining_requests < 100:
        logger.warning(f"API 요청 한도 부족: {remaining_requests}회 남음")
        if reset_time:
            logger.warning(f"요청 한도 리셋 시간: {reset_time}")
    elif remaining_requests < 500:
        logger.info(f"API 요청 한도: {remaining_requests}회 남음")


def log_duplicate_detection(
    logger: logging.Logger,
    decision_id: str,
    keyword: str
):
    """중복 감지 로그"""
    logger.debug(f"중복 감지 - 결정례 ID: {decision_id}, 키워드: '{keyword}'")


def log_decision_classification(
    logger: logging.Logger,
    decision_id: str,
    decision_type: str,
    confidence: float = 0.0
):
    """결정유형 분류 로그"""
    if confidence > 0:
        logger.debug(f"결정유형 분류 - ID: {decision_id}, 유형: {decision_type}, 신뢰도: {confidence:.2f}")
    else:
        logger.debug(f"결정유형 분류 - ID: {decision_id}, 유형: {decision_type}")


def log_summary_generation(
    logger: logging.Logger,
    summary_file: str,
    total_decisions: int,
    decision_types: dict
):
    """요약 생성 로그"""
    logger.info(f"수집 결과 요약 생성: {summary_file}")
    logger.info(f"총 결정례 수: {total_decisions:,}건")
    
    if decision_types:
        logger.info("결정유형별 분포:")
        for decision_type, count in sorted(decision_types.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {decision_type}: {count:,}건")


def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """진행률 바 생성"""
    if total <= 0:
        return "[" + " " * width + "] 0.0%"
    
    percentage = current / total
    filled_width = int(width * percentage)
    bar = "█" * filled_width + "░" * (width - filled_width)
    return f"[{bar}] {percentage:.1f}%"


def log_progress_bar(
    logger: logging.Logger,
    current: int,
    total: int,
    prefix: str = "진행률",
    width: int = 50
):
    """진행률 바 로그"""
    progress_bar = create_progress_bar(current, total, width)
    logger.info(f"{prefix}: {progress_bar} ({current:,}/{total:,})")
