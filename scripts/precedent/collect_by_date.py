#!/usr/bin/env python3
"""
날짜 기반 판례 수집 실행 스크립트

이 스크립트는 날짜별로 체계적인 판례 수집을 수행합니다.
- 연도별, 분기별, 월별, 주별 수집 전략 지원
- 폴더별 raw 데이터 저장 구조
- 선고일자 내림차순 최적화
- 중복 방지 및 체크포인트 지원

사용법:
    python scripts/precedent/collect_by_date.py --strategy yearly --target 10000
    python scripts/precedent/collect_by_date.py --strategy quarterly --target 4000
    python scripts/precedent/collect_by_date.py --strategy monthly --target 2400
    python scripts/precedent/collect_by_date.py --strategy weekly --target 1200
    python scripts/precedent/collect_by_date.py --strategy all --target 20000
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIConfig, LawOpenAPIClient
from scripts.precedent.date_based_collector import (
    DateBasedPrecedentCollector, DateCollectionStrategy
)
from scripts.precedent.precedent_logger import setup_logging

logger = setup_logging()


def generate_date_ranges(strategy: DateCollectionStrategy, count: int) -> List[Tuple[str, str, str]]:
    """날짜 범위 생성"""
    ranges = []
    current = datetime.now()
    
    if strategy == DateCollectionStrategy.YEARLY:
        for i in range(count):
            year = current.year - i
            ranges.append((f"{year}년", f"{year}0101", f"{year}1231"))
    
    elif strategy == DateCollectionStrategy.QUARTERLY:
        for i in range(count):
            target_date = current - timedelta(days=90*i)
            year = target_date.year
            quarter = (target_date.month - 1) // 3 + 1
            
            if quarter == 1:
                start_date = f"{year}0101"
                end_date = f"{year}0331"
            elif quarter == 2:
                start_date = f"{year}0401"
                end_date = f"{year}0630"
            elif quarter == 3:
                start_date = f"{year}0701"
                end_date = f"{year}0930"
            else:
                start_date = f"{year}1001"
                end_date = f"{year}1231"
            
            ranges.append((f"{year}Q{quarter}", start_date, end_date))
    
    elif strategy == DateCollectionStrategy.MONTHLY:
        for i in range(count):
            target_date = current - timedelta(days=30*i)
            year = target_date.year
            month = target_date.month
            
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year+1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(year, month+1, 1) - timedelta(days=1)
            
            ranges.append((
                f"{year}년{month:02d}월",
                start_date.strftime('%Y%m%d'),
                end_date.strftime('%Y%m%d')
            ))
    
    elif strategy == DateCollectionStrategy.WEEKLY:
        for i in range(count):
            target_date = current - timedelta(weeks=i)
            start_of_week = target_date - timedelta(days=target_date.weekday())
            end_of_week = start_of_week + timedelta(days=6)
            
            ranges.append((
                f"{start_of_week.strftime('%Y%m%d')}주",
                start_of_week.strftime('%Y%m%d'),
                end_of_week.strftime('%Y%m%d')
            ))
    
    return ranges


def main():
    parser = argparse.ArgumentParser(
        description="날짜 기반 판례 수집",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 특정 연도 수집 (2024년만)
  python scripts/precedent/collect_by_date.py --strategy yearly --year 2024 --unlimited

  # 특정 연도 수집 (2023년만)
  python scripts/precedent/collect_by_date.py --strategy yearly --year 2023 --unlimited

  # 연도별 수집 (최근 5년, 연간 2000건)
  python scripts/precedent/collect_by_date.py --strategy yearly --target 10000

  # 분기별 수집 (최근 2년, 분기당 500건)
  python scripts/precedent/collect_by_date.py --strategy quarterly --target 4000

  # 월별 수집 (최근 1년, 월간 200건)
  python scripts/precedent/collect_by_date.py --strategy monthly --target 2400

  # 주별 수집 (최근 3개월, 주간 100건)
  python scripts/precedent/collect_by_date.py --strategy weekly --target 1200

  # 모든 전략 순차 실행
  python scripts/precedent/collect_by_date.py --strategy all --target 20000
        """
    )
    
    parser.add_argument(
        "--strategy", 
        choices=["yearly", "quarterly", "monthly", "weekly", "all"], 
        default="all", 
        help="수집 전략 선택 (기본값: all)"
    )
    parser.add_argument(
        "--target", 
        type=int, 
        default=None, 
        help="목표 수집 건수 (기본값: None, 제한 없음)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/raw/precedents", 
        help="출력 디렉토리 (기본값: data/raw/precedents)"
    )
    parser.add_argument(
        "--count", 
        type=int, 
        default=None, 
        help="수집할 기간 수 (예: 연도별 5년, 분기별 8분기)"
    )
    parser.add_argument(
        "--resume", 
        action="store_true", 
        help="중단된 지점부터 재시작"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="실제 수집 없이 계획만 출력"
    )
    parser.add_argument(
        "--unlimited", 
        action="store_true", 
        help="건수 제한 없이 최대한 수집 (기본값: True)"
    )
    parser.add_argument(
        "--year", 
        type=int, 
        default=None, 
        help="특정 연도 지정 (예: 2024, 2023, 2022)"
    )
    parser.add_argument(
        "--no-details", 
        action="store_true", 
        help="판례본문 제외하고 수집 (빠르지만 기본 정보만)"
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
    
    args = parser.parse_args()
    
    # 건수 제한 설정
    if args.unlimited or args.target is None:
        args.target = 999999999  # 매우 큰 수로 설정
        unlimited_mode = True
    else:
        unlimited_mode = False
    
    # 특정 연도 설정
    if args.year:
        if args.year < 2000 or args.year > 2030:
            logger.error(f"❌ 잘못된 연도입니다: {args.year}. 2000-2030 사이의 연도를 입력하세요.")
            return
        logger.info(f"📅 특정 연도 지정: {args.year}년")
    
    # 로깅 설정
    logger.info("=" * 80)
    logger.info("📅 날짜 기반 판례 수집 시작")
    logger.info("=" * 80)
    logger.info(f"🎯 수집 전략: {args.strategy}")
    if unlimited_mode:
        logger.info(f"📊 목표 건수: 제한 없음 (최대한 수집)")
    else:
        logger.info(f"📊 목표 건수: {args.target:,}건")
    logger.info(f"📁 출력 디렉토리: {args.output}")
    logger.info(f"🔄 재시작 모드: {args.resume}")
    logger.info(f"🔍 드라이런 모드: {args.dry_run}")
    logger.info(f"🚀 무제한 모드: {unlimited_mode}")
    if args.year:
        logger.info(f"📅 특정 연도: {args.year}년")
    logger.info("=" * 80)
    
    if args.dry_run:
        logger.info("🔍 드라이런 모드: 실제 수집 없이 계획만 출력합니다.")
        _print_collection_plan(args.strategy, args.target, args.count)
        return
    
    try:
        # API 클라이언트 설정
        config = LawOpenAPIConfig()
        client = LawOpenAPIClient(config)
        
        # 출력 디렉토리 설정
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 날짜 기반 수집기 초기화
        collector = DateBasedPrecedentCollector(config, output_dir, not args.no_details)
        
        # 시간 인터벌 설정
        collector.set_request_interval(args.interval, args.interval_range)
        logger.info(f"⏱️ API 요청 간격 설정: {args.interval:.1f} ± {args.interval_range:.1f}초")
        
        # 수집 전략별 실행
        if args.strategy == "all":
            _run_all_strategies(collector, args.target, args.count, args.year)
        else:
            _run_single_strategy(collector, args.strategy, args.target, args.count, args.year)
        
        logger.info("=" * 80)
        logger.info("🎉 날짜 기반 판례 수집 완료!")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("⚠️ 사용자에 의해 수집이 중단되었습니다.")
        logger.info("💾 현재까지 수집된 데이터는 저장되었습니다.")
    except Exception as e:
        logger.error(f"❌ 수집 중 오류 발생: {e}")
        
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
        
        raise


def _print_collection_plan(strategy: str, target: int, count: Optional[int]):
    """수집 계획 출력"""
    logger.info("📋 수집 계획:")
    
    if strategy == "yearly":
        years = count or 5
        target_per_year = target // years
        logger.info(f"  📅 연도별 수집: 최근 {years}년, 연간 {target_per_year:,}건")
        logger.info(f"  📊 총 목표: {target:,}건")
        
    elif strategy == "quarterly":
        quarters = count or 8
        target_per_quarter = target // quarters
        logger.info(f"  📅 분기별 수집: 최근 {quarters}분기, 분기당 {target_per_quarter:,}건")
        logger.info(f"  📊 총 목표: {target:,}건")
        
    elif strategy == "monthly":
        months = count or 12
        target_per_month = target // months
        logger.info(f"  📅 월별 수집: 최근 {months}개월, 월간 {target_per_month:,}건")
        logger.info(f"  📊 총 목표: {target:,}건")
        
    elif strategy == "weekly":
        weeks = count or 12
        target_per_week = target // weeks
        logger.info(f"  📅 주별 수집: 최근 {weeks}주, 주간 {target_per_week:,}건")
        logger.info(f"  📊 총 목표: {target:,}건")
        
    elif strategy == "all":
        logger.info(f"  📅 모든 전략 순차 실행: 총 {target:,}건")
        logger.info(f"    - 연도별: 5년 × 2,000건 = 10,000건")
        logger.info(f"    - 분기별: 8분기 × 500건 = 4,000건")
        logger.info(f"    - 월별: 12개월 × 200건 = 2,400건")
        logger.info(f"    - 주별: 12주 × 100건 = 1,200건")
        logger.info(f"    - 총 예상: 17,600건")


def _run_all_strategies(collector: DateBasedPrecedentCollector, target: int, count: Optional[int], year: Optional[int]):
    """모든 전략 순차 실행"""
    logger.info("🚀 모든 수집 전략 순차 실행 시작")
    
    total_collected = 0
    strategies = [
        ("yearly", 5, 2000),
        ("quarterly", 8, 500),
        ("monthly", 12, 200),
        ("weekly", 12, 100)
    ]
    
    for strategy_name, default_count, default_target in strategies:
        if total_collected >= target:
            logger.info(f"🎯 목표 {target:,}건 달성으로 {strategy_name} 전략 건너뛰기")
            break
        
        remaining = target - total_collected
        strategy_count = count or default_count
        strategy_target = min(remaining, default_target * strategy_count)
        
        logger.info(f"📅 {strategy_name} 전략 시작 (목표: {strategy_target:,}건)")
        
        try:
            if strategy_name == "yearly":
                years = [datetime.now().year - i for i in range(strategy_count)]
                result = collector.collect_by_yearly_strategy(years, strategy_target // strategy_count)
            elif strategy_name == "quarterly":
                quarters = collector.generate_date_ranges(DateCollectionStrategy.QUARTERLY, strategy_count)
                result = collector.collect_by_quarterly_strategy(quarters, strategy_target // strategy_count)
            elif strategy_name == "monthly":
                months = collector.generate_date_ranges(DateCollectionStrategy.MONTHLY, strategy_count)
                result = collector.collect_by_monthly_strategy(months, strategy_target // strategy_count)
            elif strategy_name == "weekly":
                weeks = collector.generate_date_ranges(DateCollectionStrategy.WEEKLY, strategy_count)
                result = collector.collect_by_weekly_strategy(weeks, strategy_target // strategy_count)
            
            collected = result.get('total_collected', 0)
            total_collected += collected
            
            logger.info(f"✅ {strategy_name} 전략 완료: {collected:,}건 (총 {total_collected:,}건)")
            
        except Exception as e:
            logger.error(f"❌ {strategy_name} 전략 실패: {e}")
            continue
    
    logger.info(f"🎉 모든 전략 완료: 총 {total_collected:,}건 수집")


def _run_single_strategy(collector: DateBasedPrecedentCollector, strategy: str, target: int, count: Optional[int], year: Optional[int]):
    """단일 전략 실행"""
    logger.info(f"🚀 {strategy} 전략 실행 시작")
    
    try:
        if strategy == "yearly":
            if year:
                # 특정 연도 지정된 경우
                years = [year]
                target_per_year = target
                logger.info(f"📅 특정 연도 {year}년 수집")
            else:
                # 기본 연도 범위
                years_count = count or 5
                years = [datetime.now().year - i for i in range(years_count)]
                target_per_year = target // years_count
            result = collector.collect_by_yearly_strategy(years, target_per_year)
            
        elif strategy == "quarterly":
            quarters_count = count or 8
            quarters = collector.generate_date_ranges(DateCollectionStrategy.QUARTERLY, quarters_count)
            target_per_quarter = target // quarters_count
            result = collector.collect_by_quarterly_strategy(quarters, target_per_quarter)
            
        elif strategy == "monthly":
            months_count = count or 12
            months = collector.generate_date_ranges(DateCollectionStrategy.MONTHLY, months_count)
            target_per_month = target // months_count
            result = collector.collect_by_monthly_strategy(months, target_per_month)
            
        elif strategy == "weekly":
            weeks_count = count or 12
            weeks = collector.generate_date_ranges(DateCollectionStrategy.WEEKLY, weeks_count)
            target_per_week = target // weeks_count
            result = collector.collect_by_weekly_strategy(weeks, target_per_week)
        
        collected = result.get('total_collected', 0)
        logger.info(f"✅ {strategy} 전략 완료: {collected:,}건 수집")
        
    except Exception as e:
        logger.error(f"❌ {strategy} 전략 실패: {e}")
        raise


if __name__ == "__main__":
    main()
