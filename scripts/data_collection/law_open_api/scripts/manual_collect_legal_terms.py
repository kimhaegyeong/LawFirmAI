#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
법령용어 수동 수집 스크립트

국가법령정보센터 OPEN API를 통해 법령용어 데이터를 수동으로 수집합니다.
- 증분 수집 모드
- 전체 수집 모드
- 수집 결과 상세 출력
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient
from scripts.data_collection.law_open_api.collectors import IncrementalLegalTermCollector
from scripts.data_collection.law_open_api.utils import setup_collection_logger

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    로깅 설정
    
    Args:
        verbose: 상세 로깅 여부
        
    Returns:
        설정된 로거
    """
    log_level = "DEBUG" if verbose else "INFO"
    return setup_collection_logger("ManualCollection", level=log_level)


def validate_environment() -> bool:
    """
    환경 검증
    
    Returns:
        검증 성공 여부
    """
    # API 키 확인
    if not os.getenv("LAW_OPEN_API_OC"):
        print("❌ LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다.")
        print("다음과 같이 설정해주세요:")
        print("export LAW_OPEN_API_OC='your_email@example.com'")
        return False
    
    print(f"✅ API 키 확인: {os.getenv('LAW_OPEN_API_OC')}")
    return True


def test_api_connection(client: LawOpenAPIClient) -> bool:
    """
    API 연결 테스트
    
    Args:
        client: API 클라이언트
        
    Returns:
        연결 성공 여부
    """
    print("\n🔗 API 연결 테스트 중...")
    
    try:
        if client.test_connection():
            print("✅ API 연결 성공")
            return True
        else:
            print("❌ API 연결 실패")
            return False
    except Exception as e:
        print(f"❌ API 연결 테스트 중 오류: {e}")
        return False


def run_incremental_collection(collector: IncrementalLegalTermCollector, 
                              include_details: bool = True,
                              resume_from_checkpoint: bool = True,
                              batch_size: int = 1000) -> dict:
    """
    증분 수집 실행
    
    Args:
        collector: 수집기
        include_details: 상세 정보 포함 여부
        resume_from_checkpoint: 체크포인트에서 재시작 여부
        batch_size: 배치 크기
        
    Returns:
        수집 결과
    """
    print(f"\n📥 증분 수집 시작... (상세정보: {include_details}, 체크포인트 재시작: {resume_from_checkpoint}, 배치크기: {batch_size})")
    
    try:
        result = collector.collect_incremental_updates(include_details, resume_from_checkpoint, batch_size)
        return result
    except Exception as e:
        logger.error(f"증분 수집 실패: {e}")
        return {
            "status": "error",
            "error": str(e),
            "collection_time": datetime.now().isoformat()
        }


def run_full_collection(collector: IncrementalLegalTermCollector) -> dict:
    """
    전체 수집 실행
    
    Args:
        collector: 수집기
        
    Returns:
        수집 결과
    """
    print("\n📥 전체 수집 시작...")
    
    try:
        result = collector.collect_full_data()
        return result
    except Exception as e:
        logger.error(f"전체 수집 실패: {e}")
        return {
            "status": "error",
            "error": str(e),
            "collection_time": datetime.now().isoformat()
        }


def print_collection_result(result: dict, mode: str):
    """
    수집 결과 출력
    
    Args:
        result: 수집 결과
        mode: 수집 모드
    """
    print(f"\n📊 {mode.upper()} 수집 결과")
    print("=" * 50)
    
    print(f"상태: {result['status']}")
    print(f"수집 시간: {result['collection_time']}")
    
    if result['status'] == 'success':
        if mode == 'incremental':
            print(f"새로운 레코드: {result['new_records']}개")
            print(f"업데이트된 레코드: {result['updated_records']}개")
            print(f"삭제된 레코드: {result['deleted_records']}개")
            print(f"변경 없는 레코드: {result['unchanged_records']}개")
            
            # 요약 정보
            summary = result.get('summary', {})
            if summary:
                print(f"\n📈 요약 정보:")
                print(f"  - 기존 데이터: {summary.get('total_existing', 0)}개")
                print(f"  - 새 데이터: {summary.get('total_new', 0)}개")
                print(f"  - 변경사항: {summary.get('new_count', 0) + summary.get('updated_count', 0) + summary.get('deleted_count', 0)}개")
        
        elif mode == 'full':
            print(f"총 레코드: {result['total_records']}개")
    
    else:
        print(f"에러: {result.get('error', 'Unknown error')}")


def print_collection_status(collector: IncrementalLegalTermCollector):
    """
    수집 상태 출력
    
    Args:
        collector: 수집기
    """
    print(f"\n📋 수집 상태")
    print("=" * 30)
    
    try:
        status = collector.get_collection_status()
        
        print(f"데이터 타입: {status['data_type']}")
        print(f"마지막 수집: {status['last_collection'] or '없음'}")
        
        stats = status['stats']
        print(f"수집 횟수: {stats['collection_count']}회")
        print(f"성공 횟수: {stats['success_count']}회")
        print(f"실패 횟수: {stats['error_count']}회")
        print(f"성공률: {stats['success_rate']}%")
        
        if stats['first_collection']:
            print(f"첫 수집: {stats['first_collection']}")
        if stats['last_successful_collection']:
            print(f"마지막 성공: {stats['last_successful_collection']}")
        
        print(f"데이터 디렉토리: {status['data_directory']}")
        print(f"메타데이터 디렉토리: {status['metadata_directory']}")
        
    except Exception as e:
        print(f"상태 조회 실패: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='법령용어 수동 수집')
    parser.add_argument('--mode', choices=['incremental', 'full'], default='incremental',
                       help='수집 모드 (기본값: incremental)')
    parser.add_argument('--include-details', action='store_true', default=True,
                       help='법령용어 상세 정보 포함 여부 (기본값: True)')
    parser.add_argument('--no-resume', action='store_true',
                       help='체크포인트에서 재시작하지 않고 처음부터 시작합니다.')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='배치 크기 (기본값: 1000개)')
    parser.add_argument('--output', type=str, 
                       help='출력 디렉토리 (기본값: data/raw/law_open_api/legal_terms)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='상세 로깅')
    parser.add_argument('--status', action='store_true',
                       help='수집 상태만 조회')
    parser.add_argument('--test', action='store_true',
                       help='API 연결 테스트만 실행')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("법령용어 수동 수집")
    print("=" * 60)
    print(f"시작 시간: {datetime.now()}")
    print(f"수집 모드: {args.mode}")
    
    # 로깅 설정
    logger = setup_logging(args.verbose)
    
    try:
        # 환경 검증
        print("\n1. 환경 검증 중...")
        if not validate_environment():
            return 1
        
        # 클라이언트 생성
        print("2. API 클라이언트 생성 중...")
        client = LawOpenAPIClient()
        
        # 수집기 생성
        data_dir = args.output or "data/raw/law_open_api/legal_terms"
        collector = IncrementalLegalTermCollector(client, data_dir)
        
        # API 연결 테스트
        print("3. API 연결 테스트 중...")
        if not test_api_connection(client):
            return 1
        
        # 테스트 모드
        if args.test:
            print("\n✅ API 연결 테스트 완료")
            return 0
        
        # 상태 조회 모드
        if args.status:
            print_collection_status(collector)
            return 0
        
        # 수집 실행
        print(f"4. {args.mode} 수집 실행 중...")
        
        if args.mode == 'incremental':
            result = run_incremental_collection(collector, args.include_details, not args.no_resume, args.batch_size)
        else:
            result = run_full_collection(collector)
        
        # 결과 출력
        print_collection_result(result, args.mode)
        
        # 상태 조회
        print_collection_status(collector)
        
        # 성공/실패 판정
        if result['status'] == 'success':
            print(f"\n✅ {args.mode.upper()} 수집 완료")
            return 0
        else:
            print(f"\n❌ {args.mode.upper()} 수집 실패")
            return 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의한 중단")
        return 0
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        logger.error(f"스크립트 실행 실패: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)




