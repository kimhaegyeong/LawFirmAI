#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
일일 법령용어 수집 스케줄러

Python schedule 라이브러리를 사용하여 법령용어를 주기적으로 수집하는 스케줄러입니다.
- 매일 자동 수집
- 수동 실행 지원
- 에러 처리 및 복구
- 로깅 및 모니터링
"""

import sys
import time
import signal
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient
from scripts.data_collection.law_open_api.collectors import IncrementalLegalTermCollector
from scripts.data_collection.law_open_api.utils import (
    setup_scheduler_logger,
    CollectionLogger,
    TimestampManager
)

logger = logging.getLogger(__name__)


class DailyLegalTermScheduler:
    """일일 법령용어 수집 스케줄러"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        스케줄러 초기화
        
        Args:
            config: 설정 정보
        """
        self.config = config
        self.running = False
        self.shutdown_requested = False
        
        # 컴포넌트 초기화
        self.client = LawOpenAPIClient()
        self.collector = IncrementalLegalTermCollector(self.client)
        self.timestamp_manager = TimestampManager()
        
        # 로거 설정
        self.logger = setup_scheduler_logger("DailyLegalTermScheduler")
        self.collection_logger = CollectionLogger("ScheduledCollection")
        
        # 시그널 핸들러 설정
        self._setup_signal_handlers()
        
        self.logger.info("DailyLegalTermScheduler 초기화 완료")
    
    def _setup_signal_handlers(self):
        """시그널 핸들러 설정"""
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            self.logger.info(f"시그널 {signal_name}({signum}) 수신 - graceful shutdown 시작")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def setup_schedule(self):
        """스케줄 설정"""
        try:
            import schedule
        except ImportError:
            self.logger.error("schedule 모듈이 설치되지 않았습니다. pip install schedule로 설치하세요.")
            raise ImportError("schedule 모듈이 필요합니다")
        
        # 기존 스케줄 클리어
        schedule.clear()
        
        # 설정에서 스케줄 정보 가져오기
        scheduling_config = self.config.get("collection", {}).get("scheduling", {})
        
        # 일일 수집 스케줄
        daily_config = scheduling_config.get("daily_collection", {})
        if daily_config.get("enabled", True):
            collection_time = daily_config.get("time", "02:00")
            schedule.every().day.at(collection_time).do(self._collect_legal_terms)
            self.logger.info(f"일일 수집 스케줄 설정 완료: 매일 {collection_time}")
        
        # 주간 전체 동기화 스케줄
        weekly_config = scheduling_config.get("weekly_full_sync", {})
        if weekly_config.get("enabled", False):
            sync_day = weekly_config.get("day", "sunday")
            sync_time = weekly_config.get("time", "01:00")
            
            if sync_day.lower() == "sunday":
                schedule.every().sunday.at(sync_time).do(self._collect_full_data)
            elif sync_day.lower() == "monday":
                schedule.every().monday.at(sync_time).do(self._collect_full_data)
            # 다른 요일들도 추가 가능
            
            self.logger.info(f"주간 전체 동기화 스케줄 설정 완료: 매주 {sync_day} {sync_time}")
        
        self.logger.info("모든 스케줄 설정 완료")
    
    def _collect_legal_terms(self):
        """법령용어 수집 실행"""
        self.logger.info("스케줄된 법령용어 수집 시작")
        
        try:
            # 연결 테스트
            if not self.client.test_connection():
                self.logger.error("API 연결 실패 - 수집 중단")
                return
            
            # 증분 수집 실행
            result = self.collector.collect_incremental_updates()
            
            if result["status"] == "success":
                self.logger.info(f"스케줄된 수집 완료: {result}")
                self._log_collection_result(result)
            else:
                self.logger.error(f"스케줄된 수집 실패: {result}")
                self._log_error(result.get("error", "Unknown error"))
            
        except Exception as e:
            self.logger.error(f"스케줄된 수집 중 예외 발생: {e}")
            self._log_error(str(e))
    
    def _collect_full_data(self):
        """전체 데이터 수집 실행"""
        self.logger.info("스케줄된 전체 데이터 수집 시작")
        
        try:
            # 연결 테스트
            if not self.client.test_connection():
                self.logger.error("API 연결 실패 - 수집 중단")
                return
            
            # 전체 수집 실행
            result = self.collector.collect_full_data()
            
            if result["status"] == "success":
                self.logger.info(f"스케줄된 전체 수집 완료: {result}")
                self._log_collection_result(result)
            else:
                self.logger.error(f"스케줄된 전체 수집 실패: {result}")
                self._log_error(result.get("error", "Unknown error"))
            
        except Exception as e:
            self.logger.error(f"스케줄된 전체 수집 중 예외 발생: {e}")
            self._log_error(str(e))
    
    def _log_collection_result(self, result: Dict[str, Any]):
        """수집 결과 로깅"""
        try:
            # 수집 결과를 별도 파일에 저장
            log_dir = Path("logs/legal_term_collection")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            result_file = log_dir / f"collection_results_{datetime.now().strftime('%Y%m%d')}.json"
            
            import json
            results = []
            
            # 기존 결과 로드
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            
            # 새 결과 추가
            results.append({
                "timestamp": datetime.now().isoformat(),
                "result": result
            })
            
            # 최근 100개만 유지
            if len(results) > 100:
                results = results[-100:]
            
            # 저장
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"수집 결과 로그 저장 완료: {result_file}")
            
        except Exception as e:
            self.logger.error(f"수집 결과 로깅 실패: {e}")
    
    def _log_error(self, error_message: str):
        """에러 로깅"""
        try:
            # 에러 로그를 별도 파일에 저장
            log_dir = Path("logs/legal_term_collection")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            error_file = log_dir / f"collection_errors_{datetime.now().strftime('%Y%m%d')}.json"
            
            import json
            errors = []
            
            # 기존 에러 로드
            if error_file.exists():
                with open(error_file, 'r', encoding='utf-8') as f:
                    errors = json.load(f)
            
            # 새 에러 추가
            errors.append({
                "timestamp": datetime.now().isoformat(),
                "error": error_message
            })
            
            # 최근 100개만 유지
            if len(errors) > 100:
                errors = errors[-100:]
            
            # 저장
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(errors, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"에러 로그 저장 완료: {error_file}")
            
        except Exception as e:
            self.logger.error(f"에러 로깅 실패: {e}")
    
    def run(self):
        """스케줄러 실행"""
        try:
            import schedule
        except ImportError:
            self.logger.error("schedule 모듈이 설치되지 않았습니다.")
            return
        
        self.running = True
        self.logger.info("법령용어 스케줄러 시작")
        
        try:
            while self.running and not self.shutdown_requested:
                # 스케줄된 작업 실행
                schedule.run_pending()
                
                # 1분마다 체크
                time.sleep(60)
                
                # 주기적으로 상태 로깅
                if datetime.now().minute == 0:  # 매시 정각
                    self._log_scheduler_status()
        
        except KeyboardInterrupt:
            self.logger.info("사용자에 의한 중단 요청")
        except Exception as e:
            self.logger.error(f"스케줄러 실행 중 예외 발생: {e}")
        finally:
            self.running = False
            self.logger.info("법령용어 스케줄러 종료")
    
    def _log_scheduler_status(self):
        """스케줄러 상태 로깅"""
        try:
            status = self.collector.get_collection_status()
            self.logger.info(f"스케줄러 상태 - 마지막 수집: {status['last_collection']}, "
                           f"수집 횟수: {status['stats']['collection_count']}, "
                           f"성공률: {status['stats']['success_rate']}%")
        except Exception as e:
            self.logger.error(f"상태 로깅 실패: {e}")
    
    def stop(self):
        """스케줄러 중지"""
        self.logger.info("스케줄러 중지 요청")
        self.running = False
    
    def get_status(self) -> Dict[str, Any]:
        """스케줄러 상태 조회"""
        return {
            "running": self.running,
            "shutdown_requested": self.shutdown_requested,
            "collection_status": self.collector.get_collection_status(),
            "config": self.config
        }
    
    def run_manual_collection(self, mode: str = "incremental") -> Dict[str, Any]:
        """
        수동 수집 실행
        
        Args:
            mode: 수집 모드 ("incremental" 또는 "full")
            
        Returns:
            수집 결과
        """
        self.logger.info(f"수동 수집 실행 - 모드: {mode}")
        
        try:
            if mode == "incremental":
                result = self.collector.collect_incremental_updates()
            elif mode == "full":
                result = self.collector.collect_full_data()
            else:
                raise ValueError(f"지원하지 않는 모드: {mode}")
            
            self.logger.info(f"수동 수집 완료: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"수동 수집 실패: {e}")
            return {
                "status": "error",
                "error": str(e),
                "collection_time": datetime.now().isoformat()
            }


# 편의 함수들
def create_scheduler(config: Dict[str, Any]) -> DailyLegalTermScheduler:
    """
    스케줄러 생성 (편의 함수)
    
    Args:
        config: 설정 정보
        
    Returns:
        DailyLegalTermScheduler 인스턴스
    """
    return DailyLegalTermScheduler(config)


def run_scheduler(config: Dict[str, Any]):
    """
    스케줄러 실행 (편의 함수)
    
    Args:
        config: 설정 정보
    """
    scheduler = create_scheduler(config)
    scheduler.setup_schedule()
    scheduler.run()


if __name__ == "__main__":
    # 테스트 실행
    logging.basicConfig(level=logging.INFO)
    
    print("DailyLegalTermScheduler 테스트")
    print("=" * 50)
    
    # 환경변수 확인
    import os
    if not os.getenv("LAW_OPEN_API_OC"):
        print("❌ LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다.")
        print("다음과 같이 설정해주세요:")
        print("export LAW_OPEN_API_OC='your_email@example.com'")
        exit(1)
    
    # 기본 설정
    test_config = {
        "collection": {
            "enabled": True,
            "data_type": "legal_terms",
            "scheduling": {
                "daily_collection": {
                    "enabled": True,
                    "time": "02:00",
                    "timeout_minutes": 30
                },
                "weekly_full_sync": {
                    "enabled": False,
                    "day": "sunday",
                    "time": "01:00",
                    "timeout_minutes": 60
                }
            }
        },
        "api": {
            "base_url": "https://open.law.go.kr/LSO/openApi",
            "timeout": 30,
            "max_retries": 3,
            "retry_delay": 5
        },
        "logging": {
            "level": "INFO",
            "log_dir": "logs/legal_term_collection",
            "max_file_size_mb": 10,
            "backup_count": 5
        }
    }
    
    try:
        # 스케줄러 생성
        scheduler = create_scheduler(test_config)
        
        # 연결 테스트
        if scheduler.client.test_connection():
            print("✅ API 연결 테스트 성공")
            
            # 수동 수집 테스트
            print("\n수동 수집 테스트 시작...")
            result = scheduler.run_manual_collection("incremental")
            
            print(f"\n수동 수집 결과:")
            print(f"  - 상태: {result['status']}")
            if result['status'] == 'success':
                print(f"  - 새로운 레코드: {result['new_records']}개")
                print(f"  - 업데이트된 레코드: {result['updated_records']}개")
                print(f"  - 삭제된 레코드: {result['deleted_records']}개")
            else:
                print(f"  - 에러: {result.get('error', 'Unknown error')}")
            
            # 상태 조회
            status = scheduler.get_status()
            print(f"\n스케줄러 상태:")
            print(f"  - 실행 중: {status['running']}")
            print(f"  - 마지막 수집: {status['collection_status']['last_collection']}")
            
            # 스케줄 설정 테스트
            print("\n스케줄 설정 테스트...")
            scheduler.setup_schedule()
            print("✅ 스케줄 설정 완료")
            
        else:
            print("❌ API 연결 테스트 실패")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
    
    print("\n✅ DailyLegalTermScheduler 테스트 완료")




