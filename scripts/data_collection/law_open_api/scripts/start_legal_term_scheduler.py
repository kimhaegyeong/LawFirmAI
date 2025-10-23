#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
법령용어 스케줄된 수집 시작 스크립트

Python schedule 라이브러리를 사용하여 법령용어를 주기적으로 수집하는 스케줄러를 시작합니다.
"""

import sys
import os
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

# 현재 작업 디렉토리를 프로젝트 루트로 변경
os.chdir(project_root)

from scripts.data_collection.law_open_api.schedulers import DailyLegalTermScheduler
from scripts.data_collection.law_open_api.utils import setup_scheduler_logger

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """
    설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        설정 정보
    """
    if not config_path:
        config_path = project_root / "config" / "legal_term_collection_config.yaml"
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.error(f"설정 파일을 찾을 수 없습니다: {config_file}")
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_file}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"설정 파일 로드 완료: {config_file}")
        return config
        
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {e}")
        raise


def validate_config(config: dict) -> bool:
    """
    설정 검증
    
    Args:
        config: 설정 정보
        
    Returns:
        검증 성공 여부
    """
    try:
        # 필수 설정 확인
        if not config.get("collection", {}).get("enabled"):
            logger.error("수집이 비활성화되어 있습니다.")
            return False
        
        # API 키 확인
        if not os.getenv("LAW_OPEN_API_OC"):
            logger.error("LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다.")
            return False
        
        # 스케줄링 설정 확인
        scheduling = config.get("collection", {}).get("scheduling", {})
        if not scheduling.get("daily_collection", {}).get("enabled"):
            logger.warning("일일 수집이 비활성화되어 있습니다.")
        
        logger.info("설정 검증 완료")
        return True
        
    except Exception as e:
        logger.error(f"설정 검증 실패: {e}")
        return False


def setup_logging(config: dict):
    """로깅 설정"""
    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO")
    log_dir = log_config.get("log_dir", "logs/legal_term_collection")
    
    # 스케줄러 로거 설정
    scheduler_logger = setup_scheduler_logger("LegalTermScheduler", log_dir, log_level)
    
    return scheduler_logger


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='법령용어 스케줄된 수집 시작')
    parser.add_argument('--config', type=str, 
                       help='설정 파일 경로 (기본값: config/legal_term_collection_config.yaml)')
    parser.add_argument('--test', action='store_true',
                       help='테스트 모드 (스케줄러 설정만 확인)')
    parser.add_argument('--manual', action='store_true',
                       help='수동 수집 실행 (스케줄링 없이)')
    parser.add_argument('--mode', choices=['incremental', 'full'], default='incremental',
                       help='수동 수집 모드 (기본값: incremental)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='상세 로깅')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("법령용어 스케줄된 수집 시작")
    print("=" * 60)
    print(f"시작 시간: {datetime.now()}")
    print(f"프로젝트 루트: {project_root}")
    
    try:
        # 설정 로드
        print("\n1. 설정 파일 로드 중...")
        config = load_config(args.config)
        
        # 로깅 설정
        print("2. 로깅 설정 중...")
        scheduler_logger = setup_logging(config)
        
        # 설정 검증
        print("3. 설정 검증 중...")
        if not validate_config(config):
            print("❌ 설정 검증 실패")
            return 1
        
        print("✅ 설정 검증 완료")
        
        # 스케줄러 생성
        print("4. 스케줄러 생성 중...")
        scheduler = DailyLegalTermScheduler(config)
        
        # 테스트 모드
        if args.test:
            print("\n🧪 테스트 모드 - 스케줄러 설정 확인")
            scheduler.setup_schedule()
            status = scheduler.get_status()
            
            print(f"스케줄러 상태:")
            print(f"  - 실행 중: {status['running']}")
            print(f"  - 설정: {status['config']['collection']['scheduling']}")
            
            print("✅ 테스트 모드 완료")
            return 0
        
        # 수동 수집 모드
        if args.manual:
            print(f"\n🔧 수동 수집 모드 - {args.mode}")
            
            # 연결 테스트
            if not scheduler.client.test_connection():
                print("❌ API 연결 실패")
                return 1
            
            print("✅ API 연결 성공")
            
            # 수집 실행
            result = scheduler.run_manual_collection(args.mode)
            
            print(f"\n수집 결과:")
            print(f"  - 상태: {result['status']}")
            if result['status'] == 'success':
                print(f"  - 새로운 레코드: {result['new_records']}개")
                print(f"  - 업데이트된 레코드: {result['updated_records']}개")
                print(f"  - 삭제된 레코드: {result['deleted_records']}개")
                print(f"  - 수집 시간: {result['collection_time']}")
            else:
                print(f"  - 에러: {result.get('error', 'Unknown error')}")
            
            print("✅ 수동 수집 완료")
            return 0
        
        # 스케줄러 실행
        print("\n5. 스케줄러 설정 중...")
        scheduler.setup_schedule()
        
        print("6. 스케줄러 시작 중...")
        print("   - Ctrl+C로 중지할 수 있습니다")
        print("   - 로그는 logs/legal_term_collection/ 디렉토리에 저장됩니다")
        
        # 스케줄러 실행
        scheduler.run()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의한 중단")
        return 0
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        logger.error(f"스크립트 실행 실패: {e}")
        return 1
    
    print("\n✅ 스케줄러 종료")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
