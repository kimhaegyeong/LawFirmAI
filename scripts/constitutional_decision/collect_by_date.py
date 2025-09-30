#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
헌재결정례 날짜 기반 수집 스크립트

이 스크립트는 날짜별로 체계적인 헌재결정례 수집을 수행합니다.
- 연도별, 분기별, 월별 수집 전략
- 선고일자 내림차순 최적화 (최신 결정례 우선)
- 폴더별 raw 데이터 저장 구조
- 중복 방지 및 체크포인트 지원
"""

import os
import sys
import json
import argparse
import traceback
import logging
from datetime import datetime, timedelta
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIConfig, load_env_file
from scripts.constitutional_decision.date_based_collector import (
    DateBasedConstitutionalCollector, DateCollectionStrategy
)
from scripts.constitutional_decision.constitutional_logger import setup_logging

logger = setup_logging()


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="헌재결정례 날짜 기반 수집 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 특정 연도 수집 (무제한, 선고일자 기준)
  python collect_by_date.py --strategy yearly --year 2024 --unlimited
  
  # 특정 연도 수집 (목표 건수 지정, 선고일자 기준)
  python collect_by_date.py --strategy yearly --year 2023 --target 1000
  
  # 특정 연도 수집 (종국일자 기준)
  python collect_by_date.py --strategy yearly --year 2025 --unlimited --final-date
  
  # 특정 분기 수집
  python collect_by_date.py --strategy quarterly --year 2024 --quarter 4 --target 500
  
  # 특정 월 수집
  python collect_by_date.py --strategy monthly --year 2024 --month 12 --target 200
  
  # 여러 연도 수집
  python collect_by_date.py --strategy yearly --start-year 2020 --end-year 2024 --target 2000
        """
    )
    
    # 필수 인수
    parser.add_argument(
        "--strategy", 
        choices=["yearly", "quarterly", "monthly"],
        required=False,
        help="수집 전략 선택"
    )
    
    # 연도 관련 인수
    parser.add_argument(
        "--year", 
        type=int,
        help="수집할 연도 (단일 연도 수집 시)"
    )
    parser.add_argument(
        "--start-year", 
        type=int,
        help="수집 시작 연도 (다중 연도 수집 시)"
    )
    parser.add_argument(
        "--end-year", 
        type=int,
        help="수집 종료 연도 (다중 연도 수집 시)"
    )
    
    # 분기 관련 인수
    parser.add_argument(
        "--quarter", 
        type=int,
        choices=[1, 2, 3, 4],
        help="수집할 분기 (분기별 수집 시)"
    )
    
    # 월 관련 인수
    parser.add_argument(
        "--month", 
        type=int,
        choices=list(range(1, 13)),
        help="수집할 월 (월별 수집 시)"
    )
    
    # 목표 건수 관련 인수
    parser.add_argument(
        "--target", 
        type=int,
        default=2000,
        help="목표 수집 건수 (기본값: 2000)"
    )
    parser.add_argument(
        "--unlimited", 
        action="store_true",
        help="무제한 수집 (해당 기간의 모든 데이터)"
    )
    
    # 출력 관련 인수
    parser.add_argument(
        "--output-dir", 
        type=str,
        help="출력 디렉토리 (기본값: data/raw/constitutional_decisions)"
    )
    
    # 기타 인수
    parser.add_argument(
        "--check", 
        action="store_true",
        help="기존 수집 데이터 확인"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="상세 로그 출력"
    )
    parser.add_argument(
        "--final-date", 
        action="store_true",
        help="종국일자 기준으로 수집 (기본값: 선고일자 기준)"
    )
    parser.add_argument(
        "--interval", 
        type=float,
        default=2.0,
        help="API 요청 간격 (초) - 기본값: 2.0초, 권장: 2.0-5.0초"
    )
    parser.add_argument(
        "--interval-range", 
        type=float,
        default=2.0,
        help="API 요청 간격 범위 (초) - 기본값: 2.0초, 실제 간격은 interval ± interval_range"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="체크포인트부터 수집 재개 (중단된 수집 이어서 진행)"
    )
    
    return parser.parse_args()


def check_existing_data():
    """기존 수집 데이터 확인"""
    try:
        output_dir = Path("data/raw/constitutional_decisions")
        
        if not output_dir.exists():
            print("❌ 수집된 데이터가 없습니다.")
            return
        
        print("=" * 80)
        print("📊 헌재결정례 날짜 기반 수집 데이터 확인")
        print("=" * 80)
        
        # 수집 전략별 디렉토리 확인
        strategies = ["yearly", "quarterly", "monthly"]
        total_collections = 0
        
        for strategy in strategies:
            pattern = f"{strategy}_*"
            dirs = list(output_dir.glob(pattern))
            
            if dirs:
                print(f"\n📁 {strategy.upper()} 수집 데이터:")
                for dir_path in sorted(dirs):
                    try:
                        # 요약 파일 확인
                        summary_files = list(dir_path.glob(f"{strategy}_collection_summary_*.json"))
                        if summary_files:
                            latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)
                            with open(latest_summary, 'r', encoding='utf-8') as f:
                                summary_data = json.load(f)
                            
                            stats = summary_data.get('statistics', {})
                            collection_info = summary_data.get('collection_info', {})
                            
                            print(f"  📂 {dir_path.name}")
                            print(f"     수집 건수: {stats.get('total_collected', 0):,}건")
                            print(f"     중복 건수: {stats.get('total_duplicates', 0):,}건")
                            print(f"     오류 건수: {stats.get('total_errors', 0):,}건")
                            print(f"     성공률: {stats.get('success_rate', 0):.1f}%")
                            print(f"     수집 시간: {collection_info.get('duration_str', 'N/A')}")
                            print(f"     파일 수: {summary_data.get('metadata', {}).get('total_files', 0)}개")
                            
                            total_collections += 1
                        else:
                            # 요약 파일이 없는 경우 페이지 파일로 확인
                            page_files = list(dir_path.glob("page_*.json"))
                            if page_files:
                                total_count = 0
                                for page_file in page_files:
                                    try:
                                        with open(page_file, 'r', encoding='utf-8') as f:
                                            data = json.load(f)
                                            count = data.get('metadata', {}).get('count', 0)
                                            total_count += count
                                    except:
                                        pass
                                
                                print(f"  📂 {dir_path.name}")
                                print(f"     추정 수집 건수: {total_count:,}건")
                                print(f"     파일 수: {len(page_files)}개")
                                total_collections += 1
                    
                    except Exception as e:
                        print(f"  📂 {dir_path.name} (오류: {e})")
        
        print(f"\n📊 총 수집 실행 횟수: {total_collections}회")
        print("=" * 80)
        
    except Exception as e:
        print(f"데이터 확인 중 오류 발생: {e}")
        print(traceback.format_exc())


def validate_arguments(args):
    """인수 유효성 검증"""
    errors = []
    
    # 환경변수 확인
    if not os.getenv("LAW_OPEN_API_OC"):
        errors.append("LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다.")
    
    # 전략별 필수 인수 확인
    if args.strategy == "yearly":
        if not args.year and not (args.start_year and args.end_year):
            errors.append("연도별 수집 시 --year 또는 --start-year/--end-year이 필요합니다.")
        if args.year and (args.start_year or args.end_year):
            errors.append("--year과 --start-year/--end-year을 동시에 사용할 수 없습니다.")
    
    elif args.strategy == "quarterly":
        if not args.year or not args.quarter:
            errors.append("분기별 수집 시 --year과 --quarter가 필요합니다.")
    
    elif args.strategy == "monthly":
        if not args.year or not args.month:
            errors.append("월별 수집 시 --year과 --month가 필요합니다.")
    
    # 목표 건수 확인
    if not args.unlimited and args.target <= 0:
        errors.append("목표 건수는 0보다 커야 합니다.")
    
    # 연도 범위 확인
    current_year = datetime.now().year
    if args.year and (args.year < 2000 or args.year > current_year):
        errors.append(f"연도는 2000년부터 {current_year}년까지 가능합니다.")
    
    if args.start_year and (args.start_year < 2000 or args.start_year > current_year):
        errors.append(f"시작 연도는 2000년부터 {current_year}년까지 가능합니다.")
    
    if args.end_year and (args.end_year < 2000 or args.end_year > current_year):
        errors.append(f"종료 연도는 2000년부터 {current_year}년까지 가능합니다.")
    
    if args.start_year and args.end_year and args.start_year > args.end_year:
        errors.append("시작 연도가 종료 연도보다 클 수 없습니다.")
    
    return errors


def main():
    """메인 함수"""
    try:
        # 환경변수 파일 로딩
        load_env_file()
        
        # Windows에서 UTF-8 환경 설정
        if sys.platform.startswith('win'):
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            try:
                import subprocess
                subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
            except:
                pass
        
        # 인수 파싱
        args = parse_arguments()
        
        # 데이터 확인 모드
        if args.check:
            check_existing_data()
            return 0
        
        # strategy가 없는 경우 오류
        if not args.strategy:
            logger.error("--strategy 인수가 필요합니다.")
            return 1
        
        # 인수 유효성 검증
        errors = validate_arguments(args)
        if errors:
            for error in errors:
                logger.error(f"❌ {error}")
            return 1
        
        # 로그 레벨 설정
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # API 설정
        config = LawOpenAPIConfig(oc=os.getenv("LAW_OPEN_API_OC"))
        
        # 출력 디렉토리 설정
        output_dir = Path(args.output_dir) if args.output_dir else None
        
        # 수집기 생성 (시간 인터벌 설정 포함)
        collector = DateBasedConstitutionalCollector(config, output_dir)
        
        # 시간 인터벌 설정
        collector.set_request_interval(args.interval, args.interval_range)
        logger.info(f"⏱️ API 요청 간격 설정: {args.interval:.1f} ± {args.interval_range:.1f}초")
        
        # 체크포인트 재개 모드 설정
        if args.resume:
            logger.info("🔄 체크포인트 재개 모드 활성화")
            collector.enable_resume_mode()
        
        # 수집 실행
        success = False
        
        if args.strategy == "yearly":
            if args.year:
                # 단일 연도 수집
                target_count = None if args.unlimited else args.target
                date_type = "종국일자" if args.final_date else "선고일자"
                logger.info(f"🗓️ {args.year}년 헌재결정례 수집 시작 (목표: {target_count or '무제한'}건, {date_type} 기준)")
                success = collector.collect_by_year(args.year, target_count, args.unlimited, args.final_date)
            else:
                # 다중 연도 수집
                date_type = "종국일자" if args.final_date else "선고일자"
                logger.info(f"🗓️ {args.start_year}년 ~ {args.end_year}년 헌재결정례 수집 시작 ({date_type} 기준)")
                success = collector.collect_multiple_years(args.start_year, args.end_year, args.target)
        
        elif args.strategy == "quarterly":
            logger.info(f"🗓️ {args.year}년 {args.quarter}분기 헌재결정례 수집 시작 (목표: {args.target}건)")
            success = collector.collect_by_quarter(args.year, args.quarter, args.target)
        
        elif args.strategy == "monthly":
            logger.info(f"🗓️ {args.year}년 {args.month}월 헌재결정례 수집 시작 (목표: {args.target}건)")
            success = collector.collect_by_month(args.year, args.month, args.target)
        
        if success:
            logger.info("✅ 헌재결정례 날짜 기반 수집이 성공적으로 완료되었습니다.")
            return 0
        else:
            logger.error("❌ 헌재결정례 날짜 기반 수집이 실패했습니다.")
            return 1
        
    except KeyboardInterrupt:
        logger.warning("⚠️ 사용자에 의해 프로그램이 중단되었습니다.")
        logger.info("💾 현재까지 수집된 데이터는 저장되었습니다.")
        return 130
    except Exception as e:
        logger.error(f"❌ 프로그램 실행 중 오류 발생: {e}")
        logger.error(f"🔍 오류 상세: {traceback.format_exc()}")
        
        # 네트워크 관련 오류인지 확인
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['dns', 'connection', 'timeout', 'network', 'resolve']):
            logger.error("🌐 네트워크 연결 문제가 발생했습니다.")
            logger.error("💡 해결 방법:")
            logger.error("   1. 인터넷 연결 상태를 확인하세요")
            logger.error("   2. 방화벽이나 프록시 설정을 확인하세요")
            logger.error("   3. DNS 서버 설정을 확인하세요")
            logger.error("   4. 잠시 후 다시 시도해보세요")
        elif 'memory' in error_msg or 'torch' in error_msg:
            logger.error("🧠 메모리 관련 문제가 발생했습니다.")
            logger.error("💡 해결 방법:")
            logger.error("   1. 다른 프로그램을 종료하여 메모리를 확보하세요")
            logger.error("   2. 목표 건수를 줄여서 다시 시도하세요")
            logger.error("   3. 시스템을 재시작한 후 다시 시도하세요")
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
