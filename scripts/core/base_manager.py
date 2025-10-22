#!/usr/bin/env python3
"""
공통 기본 매니저 클래스
모든 스크립트 매니저의 기본 클래스
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json


@dataclass
class BaseConfig:
    """기본 설정 클래스"""
    log_level: str = "INFO"
    log_dir: str = "logs"
    results_dir: str = "results"
    backup_enabled: bool = True
    timeout_seconds: int = 300


class BaseManager(ABC):
    """모든 매니저 클래스의 기본 클래스"""
    
    def __init__(self, config: BaseConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.start_time = None
        self.metrics = {}
        
    def _setup_logger(self) -> logging.Logger:
        """표준화된 로거 설정"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # 핸들러가 이미 있으면 제거
        if logger.handlers:
            logger.handlers.clear()
        
        # 로그 디렉토리 생성
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일 핸들러
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"{self.__class__.__name__.lower()}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def start_operation(self, operation_name: str) -> None:
        """작업 시작"""
        self.start_time = datetime.now()
        self.logger.info(f"Starting {operation_name}")
        
    def end_operation(self, operation_name: str) -> Dict[str, Any]:
        """작업 종료"""
        if self.start_time is None:
            return {}
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.logger.info(f"Completed {operation_name} in {duration:.2f} seconds")
        
        return {
            'operation_name': operation_name,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration
        }
    
    def _handle_error(self, error: Exception, context: str) -> None:
        """표준화된 에러 처리"""
        error_msg = f"Error in {context}: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        
        # 메트릭 업데이트
        self.metrics.setdefault('errors', []).append({
            'context': context,
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        })
    
    def _collect_metrics(self, operation: str, duration: float, **kwargs) -> None:
        """메트릭 수집"""
        if 'operations' not in self.metrics:
            self.metrics['operations'] = []
        
        self.metrics['operations'].append({
            'operation': operation,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        })
    
    def save_metrics(self, filepath: Optional[str] = None) -> None:
        """메트릭 저장"""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = f"{self.config.results_dir}/metrics_{self.__class__.__name__.lower()}_{timestamp}.json"
            
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Metrics saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """실행 메서드 (하위 클래스에서 구현)"""
        pass


class ScriptConfigManager:
    """스크립트 전용 설정 관리자"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 로드"""
        default_config = {
            'database': {
                'path': 'data/lawfirm.db',
                'backup_enabled': True,
                'backup_dir': 'data/backups'
            },
            'vector': {
                'embeddings_dir': 'data/embeddings',
                'model': 'jhgan/ko-sroberta-multitask',
                'batch_size': 32,
                'chunk_size': 1000
            },
            'testing': {
                'results_dir': 'results',
                'max_workers': 4,
                'batch_size': 100,
                'timeout_seconds': 300
            },
            'logging': {
                'level': 'INFO',
                'dir': 'logs',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
        
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # 기본 설정과 사용자 설정 병합
                    self._merge_config(default_config, user_config)
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")
        
        return default_config
    
    def _merge_config(self, default: Dict[str, Any], user: Dict[str, Any]) -> None:
        """설정 병합"""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def get_database_config(self) -> Dict[str, Any]:
        """데이터베이스 설정"""
        return self.config.get('database', {})
    
    def get_vector_config(self) -> Dict[str, Any]:
        """벡터 임베딩 설정"""
        return self.config.get('vector', {})
    
    def get_test_config(self) -> Dict[str, Any]:
        """테스트 설정"""
        return self.config.get('testing', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """로깅 설정"""
        return self.config.get('logging', {})
    
    def save_config(self, filepath: str) -> None:
        """설정 저장"""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"Failed to save config: {e}")


class ProgressTracker:
    """진행률 추적기"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1) -> None:
        """진행률 업데이트"""
        self.current += increment
        self._print_progress()
    
    def _print_progress(self) -> None:
        """진행률 출력"""
        if self.total == 0:
            return
        
        percentage = (self.current / self.total) * 100
        elapsed = datetime.now() - self.start_time
        
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = f"ETA: {eta.total_seconds():.0f}s"
        else:
            eta_str = "ETA: --"
        
        print(f"\r{self.description}: {self.current}/{self.total} ({percentage:.1f}%) {eta_str}", end="", flush=True)
        
        if self.current >= self.total:
            print()  # 줄바꿈
    
    def finish(self) -> None:
        """완료"""
        self.current = self.total
        self._print_progress()


class ErrorHandler:
    """에러 핸들러"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.error_count = 0
        self.errors = []
    
    def handle_error(self, error: Exception, context: str, critical: bool = False) -> None:
        """에러 처리"""
        self.error_count += 1
        
        error_info = {
            'context': context,
            'error': str(error),
            'type': type(error).__name__,
            'timestamp': datetime.now().isoformat(),
            'critical': critical
        }
        
        self.errors.append(error_info)
        
        if critical:
            self.logger.critical(f"Critical error in {context}: {error}", exc_info=True)
        else:
            self.logger.error(f"Error in {context}: {error}", exc_info=True)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """에러 요약"""
        if not self.errors:
            return {'total_errors': 0, 'critical_errors': 0}
        
        critical_errors = len([e for e in self.errors if e['critical']])
        
        return {
            'total_errors': self.error_count,
            'critical_errors': critical_errors,
            'errors': self.errors
        }
    
    def clear_errors(self) -> None:
        """에러 목록 초기화"""
        self.error_count = 0
        self.errors.clear()


class PerformanceMonitor:
    """성능 모니터"""
    
    def __init__(self):
        self.measurements = {}
        self.start_times = {}
    
    def start_timer(self, operation: str) -> None:
        """타이머 시작"""
        self.start_times[operation] = datetime.now()
    
    def end_timer(self, operation: str) -> float:
        """타이머 종료"""
        if operation not in self.start_times:
            return 0.0
        
        duration = (datetime.now() - self.start_times[operation]).total_seconds()
        
        if operation not in self.measurements:
            self.measurements[operation] = []
        
        self.measurements[operation].append(duration)
        
        del self.start_times[operation]
        return duration
    
    def get_average_time(self, operation: str) -> float:
        """평균 실행 시간"""
        if operation not in self.measurements or not self.measurements[operation]:
            return 0.0
        
        return sum(self.measurements[operation]) / len(self.measurements[operation])
    
    def get_summary(self) -> Dict[str, Any]:
        """성능 요약"""
        summary = {}
        
        for operation, times in self.measurements.items():
            if times:
                summary[operation] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'average_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        
        return summary


def setup_project_path() -> None:
    """프로젝트 경로 설정"""
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def create_backup(filepath: str, backup_dir: str = "data/backups") -> Optional[str]:
    """파일 백업 생성"""
    try:
        source_path = Path(filepath)
        if not source_path.exists():
            return None
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = backup_path / f"{source_path.stem}_backup_{timestamp}{source_path.suffix}"
        
        import shutil
        shutil.copy2(source_path, backup_file)
        
        return str(backup_file)
        
    except Exception as e:
        print(f"Failed to create backup: {e}")
        return None


def validate_file_exists(filepath: str, description: str = "File") -> bool:
    """파일 존재 여부 검증"""
    path = Path(filepath)
    if not path.exists():
        print(f"Error: {description} not found: {filepath}")
        return False
    return True


def validate_directory_exists(dirpath: str, description: str = "Directory") -> bool:
    """디렉토리 존재 여부 검증"""
    path = Path(dirpath)
    if not path.exists():
        print(f"Error: {description} not found: {dirpath}")
        return False
    return True


def ensure_directory_exists(dirpath: str) -> bool:
    """디렉토리 존재 확인 및 생성"""
    try:
        path = Path(dirpath)
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Failed to create directory {dirpath}: {e}")
        return False


if __name__ == "__main__":
    # 테스트 코드
    config = BaseConfig()
    manager = BaseManager(config)
    
    print("BaseManager initialized successfully")
    print(f"Logger name: {manager.logger.name}")
    print(f"Config: {manager.config}")
