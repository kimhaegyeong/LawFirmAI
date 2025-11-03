#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
? ì§œ ê¸°ë°˜ ?ë? ?˜ì§‘ ?¤í–‰ ?¤í¬ë¦½íŠ¸

???¤í¬ë¦½íŠ¸??? ì§œë³„ë¡œ ì²´ê³„?ì¸ ?ë? ?˜ì§‘???˜í–‰?©ë‹ˆ??
- ?°ë„ë³? ë¶„ê¸°ë³? ?”ë³„, ì£¼ë³„ ?˜ì§‘ ?„ëµ ì§€??
- ?´ë”ë³?raw ?°ì´???€??êµ¬ì¡°
- ? ê³ ?¼ì ?´ë¦¼ì°¨ìˆœ ìµœì ??
- ì¤‘ë³µ ë°©ì? ë°?ì²´í¬?¬ì¸??ì§€??

?¬ìš©ë²?
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

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIConfig, LawOpenAPIClient
from scripts.precedent.date_based_collector import (
    DateBasedPrecedentCollector, DateCollectionStrategy
)
from scripts.precedent.precedent_logger import setup_logging

logger = setup_logging()


def generate_date_ranges(strategy: DateCollectionStrategy, count: int) -> List[Tuple[str, str, str]]:
    """? ì§œ ë²”ìœ„ ?ì„±"""
    ranges = []
    current = datetime.now()
    
    if strategy == DateCollectionStrategy.YEARLY:
        for i in range(count):
            year = current.year - i
            ranges.append((f"{year}??, f"{year}0101", f"{year}1231"))
    
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
                f"{year}??month:02d}??,
                start_date.strftime('%Y%m%d'),
                end_date.strftime('%Y%m%d')
            ))
    
    elif strategy == DateCollectionStrategy.WEEKLY:
        for i in range(count):
            target_date = current - timedelta(weeks=i)
            start_of_week = target_date - timedelta(days=target_date.weekday())
            end_of_week = start_of_week + timedelta(days=6)
            
            ranges.append((
                f"{start_of_week.strftime('%Y%m%d')}ì£?,
                start_of_week.strftime('%Y%m%d'),
                end_of_week.strftime('%Y%m%d')
            ))
    
    return ranges


def main():
    parser = argparse.ArgumentParser(
        description="? ì§œ ê¸°ë°˜ ?ë? ?˜ì§‘",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
?¬ìš© ?ˆì‹œ:
  # ?¹ì • ?°ë„ ?˜ì§‘ (2024?„ë§Œ)
  python scripts/precedent/collect_by_date.py --strategy yearly --year 2024 --unlimited

  # ?¹ì • ?°ë„ ?˜ì§‘ (2023?„ë§Œ)
  python scripts/precedent/collect_by_date.py --strategy yearly --year 2023 --unlimited

  # ?°ë„ë³??˜ì§‘ (ìµœê·¼ 5?? ?°ê°„ 2000ê±?
  python scripts/precedent/collect_by_date.py --strategy yearly --target 10000

  # ë¶„ê¸°ë³??˜ì§‘ (ìµœê·¼ 2?? ë¶„ê¸°??500ê±?
  python scripts/precedent/collect_by_date.py --strategy quarterly --target 4000

  # ?”ë³„ ?˜ì§‘ (ìµœê·¼ 1?? ?”ê°„ 200ê±?
  python scripts/precedent/collect_by_date.py --strategy monthly --target 2400

  # ì£¼ë³„ ?˜ì§‘ (ìµœê·¼ 3ê°œì›”, ì£¼ê°„ 100ê±?
  python scripts/precedent/collect_by_date.py --strategy weekly --target 1200

  # ëª¨ë“  ?„ëµ ?œì°¨ ?¤í–‰
  python scripts/precedent/collect_by_date.py --strategy all --target 20000
        """
    )
    
    parser.add_argument(
        "--strategy", 
        choices=["yearly", "quarterly", "monthly", "weekly", "all"], 
        default="all", 
        help="?˜ì§‘ ?„ëµ ? íƒ (ê¸°ë³¸ê°? all)"
    )
    parser.add_argument(
        "--target", 
        type=int, 
        default=None, 
        help="ëª©í‘œ ?˜ì§‘ ê±´ìˆ˜ (ê¸°ë³¸ê°? None, ?œí•œ ?†ìŒ)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/raw/precedents", 
        help="ì¶œë ¥ ?”ë ‰? ë¦¬ (ê¸°ë³¸ê°? data/raw/precedents)"
    )
    parser.add_argument(
        "--count", 
        type=int, 
        default=None, 
        help="?˜ì§‘??ê¸°ê°„ ??(?? ?°ë„ë³?5?? ë¶„ê¸°ë³?8ë¶„ê¸°)"
    )
    parser.add_argument(
        "--resume", 
        action="store_true", 
        help="ì¤‘ë‹¨??ì§€?ë????¬ì‹œ??
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="?¤ì œ ?˜ì§‘ ?†ì´ ê³„íšë§?ì¶œë ¥"
    )
    parser.add_argument(
        "--unlimited", 
        action="store_true", 
        help="ê±´ìˆ˜ ?œí•œ ?†ì´ ìµœë????˜ì§‘ (ê¸°ë³¸ê°? True)"
    )
    parser.add_argument(
        "--year", 
        type=int, 
        default=None, 
        help="?¹ì • ?°ë„ ì§€??(?? 2024, 2023, 2022)"
    )
    parser.add_argument(
        "--no-details", 
        action="store_true", 
        help="?ë?ë³¸ë¬¸ ?œì™¸?˜ê³  ?˜ì§‘ (ë¹ ë¥´ì§€ë§?ê¸°ë³¸ ?•ë³´ë§?"
    )
    parser.add_argument(
        "--interval", 
        type=float,
        default=2.0,
        help="API ?”ì²­ ê°„ê²© (ì´? - ê¸°ë³¸ê°? 2.0ì´? ê¶Œì¥: 2.0-5.0ì´?
    )
    parser.add_argument(
        "--interval-range", 
        type=float,
        default=2.0,
        help="API ?”ì²­ ê°„ê²© ë²”ìœ„ (ì´? - ê¸°ë³¸ê°? 2.0ì´? ?¤ì œ ê°„ê²©?€ interval Â± interval_range"
    )
    
    args = parser.parse_args()
    
    # ê±´ìˆ˜ ?œí•œ ?¤ì •
    if args.unlimited or args.target is None:
        args.target = 999999999  # ë§¤ìš° ???˜ë¡œ ?¤ì •
        unlimited_mode = True
    else:
        unlimited_mode = False
    
    # ?¹ì • ?°ë„ ?¤ì •
    if args.year:
        if args.year < 2000 or args.year > 2030:
            logger.error(f"???˜ëª»???°ë„?…ë‹ˆ?? {args.year}. 2000-2030 ?¬ì´???°ë„ë¥??…ë ¥?˜ì„¸??")
            return
        logger.info(f"?“… ?¹ì • ?°ë„ ì§€?? {args.year}??)
    
    # ë¡œê¹… ?¤ì •
    logger.info("=" * 80)
    logger.info("?“… ? ì§œ ê¸°ë°˜ ?ë? ?˜ì§‘ ?œì‘")
    logger.info("=" * 80)
    logger.info(f"?¯ ?˜ì§‘ ?„ëµ: {args.strategy}")
    if unlimited_mode:
        logger.info(f"?“Š ëª©í‘œ ê±´ìˆ˜: ?œí•œ ?†ìŒ (ìµœë????˜ì§‘)")
    else:
        logger.info(f"?“Š ëª©í‘œ ê±´ìˆ˜: {args.target:,}ê±?)
    logger.info(f"?“ ì¶œë ¥ ?”ë ‰? ë¦¬: {args.output}")
    logger.info(f"?”„ ?¬ì‹œ??ëª¨ë“œ: {args.resume}")
    logger.info(f"?” ?œë¼?´ëŸ° ëª¨ë“œ: {args.dry_run}")
    logger.info(f"?? ë¬´ì œ??ëª¨ë“œ: {unlimited_mode}")
    if args.year:
        logger.info(f"?“… ?¹ì • ?°ë„: {args.year}??)
    logger.info("=" * 80)
    
    if args.dry_run:
        logger.info("?” ?œë¼?´ëŸ° ëª¨ë“œ: ?¤ì œ ?˜ì§‘ ?†ì´ ê³„íšë§?ì¶œë ¥?©ë‹ˆ??")
        _print_collection_plan(args.strategy, args.target, args.count)
        return
    
    try:
        # API ?´ë¼?´ì–¸???¤ì •
        config = LawOpenAPIConfig()
        client = LawOpenAPIClient(config)
        
        # ì¶œë ¥ ?”ë ‰? ë¦¬ ?¤ì •
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ? ì§œ ê¸°ë°˜ ?˜ì§‘ê¸?ì´ˆê¸°??
        collector = DateBasedPrecedentCollector(config, output_dir, not args.no_details)
        
        # ?œê°„ ?¸í„°ë²??¤ì •
        collector.set_request_interval(args.interval, args.interval_range)
        logger.info(f"?±ï¸ API ?”ì²­ ê°„ê²© ?¤ì •: {args.interval:.1f} Â± {args.interval_range:.1f}ì´?)
        
        # ?˜ì§‘ ?„ëµë³??¤í–‰
        if args.strategy == "all":
            _run_all_strategies(collector, args.target, args.count, args.year)
        else:
            _run_single_strategy(collector, args.strategy, args.target, args.count, args.year)
        
        logger.info("=" * 80)
        logger.info("?‰ ? ì§œ ê¸°ë°˜ ?ë? ?˜ì§‘ ?„ë£Œ!")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("? ï¸ ?¬ìš©?ì— ?˜í•´ ?˜ì§‘??ì¤‘ë‹¨?˜ì—ˆ?µë‹ˆ??")
        logger.info("?’¾ ?„ì¬ê¹Œì? ?˜ì§‘???°ì´?°ëŠ” ?€?¥ë˜?ˆìŠµ?ˆë‹¤.")
    except Exception as e:
        logger.error(f"???˜ì§‘ ì¤??¤ë¥˜ ë°œìƒ: {e}")
        
        # ?¤íŠ¸?Œí¬ ê´€???¤ë¥˜?¸ì? ?•ì¸
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['dns', 'connection', 'timeout', 'network', 'resolve']):
            logger.error("?Œ ?¤íŠ¸?Œí¬ ?°ê²° ë¬¸ì œê°€ ë°œìƒ?ˆìŠµ?ˆë‹¤.")
            logger.error("?’¡ ?´ê²° ë°©ë²•:")
            logger.error("   1. ?¸í„°???°ê²° ?íƒœë¥??•ì¸?˜ì„¸??)
            logger.error("   2. ë°©í™”ë²½ì´???„ë¡???¤ì •???•ì¸?˜ì„¸??)
            logger.error("   3. DNS ?œë²„ ?¤ì •???•ì¸?˜ì„¸??)
            logger.error("   4. ? ì‹œ ???¤ì‹œ ?œë„?´ë³´?¸ìš”")
        elif 'memory' in error_msg or 'torch' in error_msg:
            logger.error("?§  ë©”ëª¨ë¦?ê´€??ë¬¸ì œê°€ ë°œìƒ?ˆìŠµ?ˆë‹¤.")
            logger.error("?’¡ ?´ê²° ë°©ë²•:")
            logger.error("   1. ?¤ë¥¸ ?„ë¡œê·¸ë¨??ì¢…ë£Œ?˜ì—¬ ë©”ëª¨ë¦¬ë? ?•ë³´?˜ì„¸??)
            logger.error("   2. ëª©í‘œ ê±´ìˆ˜ë¥?ì¤„ì—¬???¤ì‹œ ?œë„?˜ì„¸??)
            logger.error("   3. ?œìŠ¤?œì„ ?¬ì‹œ?‘í•œ ???¤ì‹œ ?œë„?˜ì„¸??)
        
        raise


def _print_collection_plan(strategy: str, target: int, count: Optional[int]):
    """?˜ì§‘ ê³„íš ì¶œë ¥"""
    logger.info("?“‹ ?˜ì§‘ ê³„íš:")
    
    if strategy == "yearly":
        years = count or 5
        target_per_year = target // years
        logger.info(f"  ?“… ?°ë„ë³??˜ì§‘: ìµœê·¼ {years}?? ?°ê°„ {target_per_year:,}ê±?)
        logger.info(f"  ?“Š ì´?ëª©í‘œ: {target:,}ê±?)
        
    elif strategy == "quarterly":
        quarters = count or 8
        target_per_quarter = target // quarters
        logger.info(f"  ?“… ë¶„ê¸°ë³??˜ì§‘: ìµœê·¼ {quarters}ë¶„ê¸°, ë¶„ê¸°??{target_per_quarter:,}ê±?)
        logger.info(f"  ?“Š ì´?ëª©í‘œ: {target:,}ê±?)
        
    elif strategy == "monthly":
        months = count or 12
        target_per_month = target // months
        logger.info(f"  ?“… ?”ë³„ ?˜ì§‘: ìµœê·¼ {months}ê°œì›”, ?”ê°„ {target_per_month:,}ê±?)
        logger.info(f"  ?“Š ì´?ëª©í‘œ: {target:,}ê±?)
        
    elif strategy == "weekly":
        weeks = count or 12
        target_per_week = target // weeks
        logger.info(f"  ?“… ì£¼ë³„ ?˜ì§‘: ìµœê·¼ {weeks}ì£? ì£¼ê°„ {target_per_week:,}ê±?)
        logger.info(f"  ?“Š ì´?ëª©í‘œ: {target:,}ê±?)
        
    elif strategy == "all":
        logger.info(f"  ?“… ëª¨ë“  ?„ëµ ?œì°¨ ?¤í–‰: ì´?{target:,}ê±?)
        logger.info(f"    - ?°ë„ë³? 5??Ã— 2,000ê±?= 10,000ê±?)
        logger.info(f"    - ë¶„ê¸°ë³? 8ë¶„ê¸° Ã— 500ê±?= 4,000ê±?)
        logger.info(f"    - ?”ë³„: 12ê°œì›” Ã— 200ê±?= 2,400ê±?)
        logger.info(f"    - ì£¼ë³„: 12ì£?Ã— 100ê±?= 1,200ê±?)
        logger.info(f"    - ì´??ˆìƒ: 17,600ê±?)


def _run_all_strategies(collector: DateBasedPrecedentCollector, target: int, count: Optional[int], year: Optional[int]):
    """ëª¨ë“  ?„ëµ ?œì°¨ ?¤í–‰"""
    logger.info("?? ëª¨ë“  ?˜ì§‘ ?„ëµ ?œì°¨ ?¤í–‰ ?œì‘")
    
    total_collected = 0
    strategies = [
        ("yearly", 5, 2000),
        ("quarterly", 8, 500),
        ("monthly", 12, 200),
        ("weekly", 12, 100)
    ]
    
    for strategy_name, default_count, default_target in strategies:
        if total_collected >= target:
            logger.info(f"?¯ ëª©í‘œ {target:,}ê±??¬ì„±?¼ë¡œ {strategy_name} ?„ëµ ê±´ë„ˆ?°ê¸°")
            break
        
        remaining = target - total_collected
        strategy_count = count or default_count
        strategy_target = min(remaining, default_target * strategy_count)
        
        logger.info(f"?“… {strategy_name} ?„ëµ ?œì‘ (ëª©í‘œ: {strategy_target:,}ê±?")
        
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
            
            logger.info(f"??{strategy_name} ?„ëµ ?„ë£Œ: {collected:,}ê±?(ì´?{total_collected:,}ê±?")
            
        except Exception as e:
            logger.error(f"??{strategy_name} ?„ëµ ?¤íŒ¨: {e}")
            continue
    
    logger.info(f"?‰ ëª¨ë“  ?„ëµ ?„ë£Œ: ì´?{total_collected:,}ê±??˜ì§‘")


def _run_single_strategy(collector: DateBasedPrecedentCollector, strategy: str, target: int, count: Optional[int], year: Optional[int]):
    """?¨ì¼ ?„ëµ ?¤í–‰"""
    logger.info(f"?? {strategy} ?„ëµ ?¤í–‰ ?œì‘")
    
    try:
        if strategy == "yearly":
            if year:
                # ?¹ì • ?°ë„ ì§€?•ëœ ê²½ìš°
                years = [year]
                target_per_year = target
                logger.info(f"?“… ?¹ì • ?°ë„ {year}???˜ì§‘")
            else:
                # ê¸°ë³¸ ?°ë„ ë²”ìœ„
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
        logger.info(f"??{strategy} ?„ëµ ?„ë£Œ: {collected:,}ê±??˜ì§‘")
        
    except Exception as e:
        logger.error(f"??{strategy} ?„ëµ ?¤íŒ¨: {e}")
        raise


if __name__ == "__main__":
    main()
