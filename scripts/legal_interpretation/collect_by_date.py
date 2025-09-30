#!/usr/bin/env python3
"""
법령해석례 날짜 기반 수집 메인 스크립트

사용법:
    python scripts/legal_interpretation/collect_by_date.py --strategy yearly --year 2025 --interpretation-date
    python scripts/legal_interpretation/collect_by_date.py --strategy quarterly --year 2025 --quarter 1
    python scripts/legal_interpretation/collect_by_date.py --strategy monthly --year 2025 --month 8
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# source 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'source'))

# 현재 디렉토리의 date_based_collector 모듈 import
from date_based_collector import (
    DateBasedLegalInterpretationCollector, 
    CollectionConfig, 
    DateCollectionStrategy
)
from data.law_open_api_client import load_env_file

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="법령해석례 날짜 기반 수집",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 2025년 법령해석례 수집 (해석일자 기준)
  python scripts/legal_interpretation/collect_by_date.py --strategy yearly --year 2025 --interpretation-date
  
  # 2024년 법령해석례 수집 (회신일자 기준)
  python scripts/legal_interpretation/collect_by_date.py --strategy yearly --year 2024
  
  # 특정 건수만 수집
  python scripts/legal_interpretation/collect_by_date.py --strategy yearly --year 2025 --target 100 --interpretation-date
  
  # 분기별 수집
  python scripts/legal_interpretation/collect_by_date.py --strategy quarterly --year 2025 --quarter 1
  
  # 월별 수집
  python scripts/legal_interpretation/collect_by_date.py --strategy monthly --year 2025 --month 8
        """
    )
    
    # 수집 전략
    parser.add_argument(
        '--strategy', 
        choices=['yearly', 'quarterly', 'monthly'], 
        required=True,
        help='수집 전략 (yearly: 연도별, quarterly: 분기별, monthly: 월별)'
    )
    
    # 연도 관련
    parser.add_argument(
        '--year', 
        type=int, 
        required=True,
        help='수집할 연도 (예: 2025)'
    )
    
    # 분기 관련
    parser.add_argument(
        '--quarter', 
        type=int, 
        choices=[1, 2, 3, 4],
        help='수집할 분기 (1-4분기, quarterly 전략에서만 사용)'
    )
    
    # 월 관련
    parser.add_argument(
        '--month', 
        type=int, 
        choices=list(range(1, 13)),
        help='수집할 월 (1-12월, monthly 전략에서만 사용)'
    )
    
    # 목표 건수
    parser.add_argument(
        '--target', 
        type=int,
        help='수집할 목표 건수 (기본값: 전략별 기본값)'
    )
    
    # 무제한 수집
    parser.add_argument(
        '--unlimited', 
        action='store_true',
        help='무제한 수집 (모든 데이터 수집)'
    )
    
    # 날짜 기준
    parser.add_argument(
        '--interpretation-date', 
        action='store_true',
        help='해석일자 기준으로 수집 (기본값: 회신일자 기준)'
    )
    
    # 출력 디렉토리
    parser.add_argument(
        '--output-dir', 
        type=str,
        help='출력 디렉토리 (기본값: data/raw/legal_interpretations)'
    )
    
    # 체크포인트 확인
    parser.add_argument(
        '--check', 
        action='store_true',
        help='기존 수집 데이터 확인만 (수집하지 않음)'
    )
    
    # 상세 로그
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='상세 로그 출력'
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """로깅 설정"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # 로그 디렉토리 생성
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/legal_interpretation_collection.log', encoding='utf-8')
        ]
    )
    
    return logging.getLogger(__name__)


def validate_arguments(args):
    """인수 검증"""
    errors = []
    
    # 전략별 필수 인수 검증
    if args.strategy == 'quarterly' and not args.quarter:
        errors.append("quarterly 전략 사용 시 --quarter 인수가 필요합니다")
    
    if args.strategy == 'monthly' and not args.month:
        errors.append("monthly 전략 사용 시 --month 인수가 필요합니다")
    
    # 연도 검증
    if args.year < 2000 or args.year > 2030:
        errors.append("연도는 2000-2030 사이여야 합니다")
    
    # 목표 건수 검증
    if args.target and args.target <= 0:
        errors.append("목표 건수는 0보다 커야 합니다")
    
    if errors:
        for error in errors:
            print(f"❌ 오류: {error}")
        return False
    
    return True


def get_default_target_count(strategy: str) -> int:
    """전략별 기본 목표 건수 반환"""
    defaults = {
        'yearly': 1000,
        'quarterly': 500,
        'monthly': 200
    }
    return defaults.get(strategy, 100)


def main():
    """메인 함수"""
    try:
        # 환경 변수 로드
        load_env_file()
        
        # 인수 파싱
        args = parse_arguments()
        
        # 인수 검증
        if not validate_arguments(args):
            return 1
        
        # 로깅 설정
        logger = setup_logging(args.verbose)
        
        # 출력 디렉토리 설정
        output_dir = Path(args.output_dir) if args.output_dir else None
        
        # 수집 설정 생성
        config = CollectionConfig()
        if output_dir:
            config.base_output_dir = output_dir
        
        # 수집기 생성
        collector = DateBasedLegalInterpretationCollector(config)
        
        # 목표 건수 설정
        target_count = args.target
        if not target_count and not args.unlimited:
            target_count = get_default_target_count(args.strategy)
        
        # 체크포인트 확인만 하는 경우
        if args.check:
            logger.info("📋 기존 수집 데이터 확인 중...")
            collector._load_existing_data(target_year=args.year)
            logger.info(f"📊 기존 수집된 법령해석례: {len(collector.collected_decisions):,}건")
            return 0
        
        # 수집 실행
        success = False
        date_type = "해석일자" if args.interpretation_date else "회신일자"
        
        if args.strategy == 'yearly':
            logger.info(f"🗓️ {args.year}년 법령해석례 수집 시작 ({date_type} 기준)")
            success = collector.collect_by_year(
                year=args.year,
                target_count=target_count,
                unlimited=args.unlimited,
                use_interpretation_date=args.interpretation_date
            )
            
        elif args.strategy == 'quarterly':
            logger.info(f"🗓️ {args.year}년 {args.quarter}분기 법령해석례 수집 시작")
            success = collector.collect_by_quarter(
                year=args.year,
                quarter=args.quarter,
                target_count=target_count
            )
            
        elif args.strategy == 'monthly':
            logger.info(f"🗓️ {args.year}년 {args.month}월 법령해석례 수집 시작")
            success = collector.collect_by_month(
                year=args.year,
                month=args.month,
                target_count=target_count
            )
        
        if success:
            logger.info("✅ 법령해석례 날짜 기반 수집이 성공적으로 완료되었습니다.")
            return 0
        else:
            logger.error("❌ 법령해석례 날짜 기반 수집이 실패했습니다.")
            return 1
            
    except KeyboardInterrupt:
        logger.info("⚠️ 사용자에 의해 프로그램이 중단되었습니다.")
        return 1
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류 발생: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
