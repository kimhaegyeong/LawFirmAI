#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?Œì¬ê²°ì •ë¡€ ? ì§œ ê¸°ë°˜ ?˜ì§‘ ?¤í¬ë¦½íŠ¸

???¤í¬ë¦½íŠ¸??? ì§œë³„ë¡œ ì²´ê³„?ì¸ ?Œì¬ê²°ì •ë¡€ ?˜ì§‘???˜í–‰?©ë‹ˆ??
- ?°ë„ë³? ë¶„ê¸°ë³? ?”ë³„ ?˜ì§‘ ?„ëµ
- ? ê³ ?¼ì ?´ë¦¼ì°¨ìˆœ ìµœì ??(ìµœì‹  ê²°ì •ë¡€ ?°ì„ )
- ?´ë”ë³?raw ?°ì´???€??êµ¬ì¡°
- ì¤‘ë³µ ë°©ì? ë°?ì²´í¬?¬ì¸??ì§€??
"""

import os
import sys
import json
import argparse
import traceback
import logging
from datetime import datetime, timedelta
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIConfig, load_env_file
from scripts.constitutional_decision.date_based_collector import (
    DateBasedConstitutionalCollector, DateCollectionStrategy
)
from scripts.constitutional_decision.constitutional_logger import setup_logging

logger = setup_logging()


def parse_arguments():
    """ëª…ë ¹???¸ìˆ˜ ?Œì‹±"""
    parser = argparse.ArgumentParser(
        description="?Œì¬ê²°ì •ë¡€ ? ì§œ ê¸°ë°˜ ?˜ì§‘ ?¤í¬ë¦½íŠ¸",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
?¬ìš© ?ˆì‹œ:
  # ?¹ì • ?°ë„ ?˜ì§‘ (ë¬´ì œ?? ? ê³ ?¼ì ê¸°ì?)
  python collect_by_date.py --strategy yearly --year 2024 --unlimited
  
  # ?¹ì • ?°ë„ ?˜ì§‘ (ëª©í‘œ ê±´ìˆ˜ ì§€?? ? ê³ ?¼ì ê¸°ì?)
  python collect_by_date.py --strategy yearly --year 2023 --target 1000
  
  # ?¹ì • ?°ë„ ?˜ì§‘ (ì¢…êµ­?¼ì ê¸°ì?)
  python collect_by_date.py --strategy yearly --year 2025 --unlimited --final-date
  
  # ?¹ì • ë¶„ê¸° ?˜ì§‘
  python collect_by_date.py --strategy quarterly --year 2024 --quarter 4 --target 500
  
  # ?¹ì • ???˜ì§‘
  python collect_by_date.py --strategy monthly --year 2024 --month 12 --target 200
  
  # ?¬ëŸ¬ ?°ë„ ?˜ì§‘
  python collect_by_date.py --strategy yearly --start-year 2020 --end-year 2024 --target 2000
        """
    )
    
    # ?„ìˆ˜ ?¸ìˆ˜
    parser.add_argument(
        "--strategy", 
        choices=["yearly", "quarterly", "monthly"],
        required=False,
        help="?˜ì§‘ ?„ëµ ? íƒ"
    )
    
    # ?°ë„ ê´€???¸ìˆ˜
    parser.add_argument(
        "--year", 
        type=int,
        help="?˜ì§‘???°ë„ (?¨ì¼ ?°ë„ ?˜ì§‘ ??"
    )
    parser.add_argument(
        "--start-year", 
        type=int,
        help="?˜ì§‘ ?œì‘ ?°ë„ (?¤ì¤‘ ?°ë„ ?˜ì§‘ ??"
    )
    parser.add_argument(
        "--end-year", 
        type=int,
        help="?˜ì§‘ ì¢…ë£Œ ?°ë„ (?¤ì¤‘ ?°ë„ ?˜ì§‘ ??"
    )
    
    # ë¶„ê¸° ê´€???¸ìˆ˜
    parser.add_argument(
        "--quarter", 
        type=int,
        choices=[1, 2, 3, 4],
        help="?˜ì§‘??ë¶„ê¸° (ë¶„ê¸°ë³??˜ì§‘ ??"
    )
    
    # ??ê´€???¸ìˆ˜
    parser.add_argument(
        "--month", 
        type=int,
        choices=list(range(1, 13)),
        help="?˜ì§‘????(?”ë³„ ?˜ì§‘ ??"
    )
    
    # ëª©í‘œ ê±´ìˆ˜ ê´€???¸ìˆ˜
    parser.add_argument(
        "--target", 
        type=int,
        default=2000,
        help="ëª©í‘œ ?˜ì§‘ ê±´ìˆ˜ (ê¸°ë³¸ê°? 2000)"
    )
    parser.add_argument(
        "--unlimited", 
        action="store_true",
        help="ë¬´ì œ???˜ì§‘ (?´ë‹¹ ê¸°ê°„??ëª¨ë“  ?°ì´??"
    )
    
    # ì¶œë ¥ ê´€???¸ìˆ˜
    parser.add_argument(
        "--output-dir", 
        type=str,
        help="ì¶œë ¥ ?”ë ‰? ë¦¬ (ê¸°ë³¸ê°? data/raw/constitutional_decisions)"
    )
    
    # ê¸°í? ?¸ìˆ˜
    parser.add_argument(
        "--check", 
        action="store_true",
        help="ê¸°ì¡´ ?˜ì§‘ ?°ì´???•ì¸"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="?ì„¸ ë¡œê·¸ ì¶œë ¥"
    )
    parser.add_argument(
        "--final-date", 
        action="store_true",
        help="ì¢…êµ­?¼ì ê¸°ì??¼ë¡œ ?˜ì§‘ (ê¸°ë³¸ê°? ? ê³ ?¼ì ê¸°ì?)"
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="ì²´í¬?¬ì¸?¸ë????˜ì§‘ ?¬ê°œ (ì¤‘ë‹¨???˜ì§‘ ?´ì–´??ì§„í–‰)"
    )
    
    return parser.parse_args()


def check_existing_data():
    """ê¸°ì¡´ ?˜ì§‘ ?°ì´???•ì¸"""
    try:
        output_dir = Path("data/raw/constitutional_decisions")
        
        if not output_dir.exists():
            print("???˜ì§‘???°ì´?°ê? ?†ìŠµ?ˆë‹¤.")
            return
        
        print("=" * 80)
        print("?“Š ?Œì¬ê²°ì •ë¡€ ? ì§œ ê¸°ë°˜ ?˜ì§‘ ?°ì´???•ì¸")
        print("=" * 80)
        
        # ?˜ì§‘ ?„ëµë³??”ë ‰? ë¦¬ ?•ì¸
        strategies = ["yearly", "quarterly", "monthly"]
        total_collections = 0
        
        for strategy in strategies:
            pattern = f"{strategy}_*"
            dirs = list(output_dir.glob(pattern))
            
            if dirs:
                print(f"\n?“ {strategy.upper()} ?˜ì§‘ ?°ì´??")
                for dir_path in sorted(dirs):
                    try:
                        # ?”ì•½ ?Œì¼ ?•ì¸
                        summary_files = list(dir_path.glob(f"{strategy}_collection_summary_*.json"))
                        if summary_files:
                            latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)
                            with open(latest_summary, 'r', encoding='utf-8') as f:
                                summary_data = json.load(f)
                            
                            stats = summary_data.get('statistics', {})
                            collection_info = summary_data.get('collection_info', {})
                            
                            print(f"  ?“‚ {dir_path.name}")
                            print(f"     ?˜ì§‘ ê±´ìˆ˜: {stats.get('total_collected', 0):,}ê±?)
                            print(f"     ì¤‘ë³µ ê±´ìˆ˜: {stats.get('total_duplicates', 0):,}ê±?)
                            print(f"     ?¤ë¥˜ ê±´ìˆ˜: {stats.get('total_errors', 0):,}ê±?)
                            print(f"     ?±ê³µë¥? {stats.get('success_rate', 0):.1f}%")
                            print(f"     ?˜ì§‘ ?œê°„: {collection_info.get('duration_str', 'N/A')}")
                            print(f"     ?Œì¼ ?? {summary_data.get('metadata', {}).get('total_files', 0)}ê°?)
                            
                            total_collections += 1
                        else:
                            # ?”ì•½ ?Œì¼???†ëŠ” ê²½ìš° ?˜ì´ì§€ ?Œì¼ë¡??•ì¸
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
                                
                                print(f"  ?“‚ {dir_path.name}")
                                print(f"     ì¶”ì • ?˜ì§‘ ê±´ìˆ˜: {total_count:,}ê±?)
                                print(f"     ?Œì¼ ?? {len(page_files)}ê°?)
                                total_collections += 1
                    
                    except Exception as e:
                        print(f"  ?“‚ {dir_path.name} (?¤ë¥˜: {e})")
        
        print(f"\n?“Š ì´??˜ì§‘ ?¤í–‰ ?Ÿìˆ˜: {total_collections}??)
        print("=" * 80)
        
    except Exception as e:
        print(f"?°ì´???•ì¸ ì¤??¤ë¥˜ ë°œìƒ: {e}")
        print(traceback.format_exc())


def validate_arguments(args):
    """?¸ìˆ˜ ? íš¨??ê²€ì¦?""
    errors = []
    
    # ?˜ê²½ë³€???•ì¸
    if not os.getenv("LAW_OPEN_API_OC"):
        errors.append("LAW_OPEN_API_OC ?˜ê²½ë³€?˜ê? ?¤ì •?˜ì? ?Šì•˜?µë‹ˆ??")
    
    # ?„ëµë³??„ìˆ˜ ?¸ìˆ˜ ?•ì¸
    if args.strategy == "yearly":
        if not args.year and not (args.start_year and args.end_year):
            errors.append("?°ë„ë³??˜ì§‘ ??--year ?ëŠ” --start-year/--end-year???„ìš”?©ë‹ˆ??")
        if args.year and (args.start_year or args.end_year):
            errors.append("--yearê³?--start-year/--end-year???™ì‹œ???¬ìš©?????†ìŠµ?ˆë‹¤.")
    
    elif args.strategy == "quarterly":
        if not args.year or not args.quarter:
            errors.append("ë¶„ê¸°ë³??˜ì§‘ ??--yearê³?--quarterê°€ ?„ìš”?©ë‹ˆ??")
    
    elif args.strategy == "monthly":
        if not args.year or not args.month:
            errors.append("?”ë³„ ?˜ì§‘ ??--yearê³?--monthê°€ ?„ìš”?©ë‹ˆ??")
    
    # ëª©í‘œ ê±´ìˆ˜ ?•ì¸
    if not args.unlimited and args.target <= 0:
        errors.append("ëª©í‘œ ê±´ìˆ˜??0ë³´ë‹¤ ì»¤ì•¼ ?©ë‹ˆ??")
    
    # ?°ë„ ë²”ìœ„ ?•ì¸
    current_year = datetime.now().year
    if args.year and (args.year < 2000 or args.year > current_year):
        errors.append(f"?°ë„??2000?„ë???{current_year}?„ê¹Œì§€ ê°€?¥í•©?ˆë‹¤.")
    
    if args.start_year and (args.start_year < 2000 or args.start_year > current_year):
        errors.append(f"?œì‘ ?°ë„??2000?„ë???{current_year}?„ê¹Œì§€ ê°€?¥í•©?ˆë‹¤.")
    
    if args.end_year and (args.end_year < 2000 or args.end_year > current_year):
        errors.append(f"ì¢…ë£Œ ?°ë„??2000?„ë???{current_year}?„ê¹Œì§€ ê°€?¥í•©?ˆë‹¤.")
    
    if args.start_year and args.end_year and args.start_year > args.end_year:
        errors.append("?œì‘ ?°ë„ê°€ ì¢…ë£Œ ?°ë„ë³´ë‹¤ ?????†ìŠµ?ˆë‹¤.")
    
    return errors


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    try:
        # ?˜ê²½ë³€???Œì¼ ë¡œë”©
        load_env_file()
        
        # Windows?ì„œ UTF-8 ?˜ê²½ ?¤ì •
        if sys.platform.startswith('win'):
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            try:
                import subprocess
                subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
            except:
                pass
        
        # ?¸ìˆ˜ ?Œì‹±
        args = parse_arguments()
        
        # ?°ì´???•ì¸ ëª¨ë“œ
        if args.check:
            check_existing_data()
            return 0
        
        # strategyê°€ ?†ëŠ” ê²½ìš° ?¤ë¥˜
        if not args.strategy:
            logger.error("--strategy ?¸ìˆ˜ê°€ ?„ìš”?©ë‹ˆ??")
            return 1
        
        # ?¸ìˆ˜ ? íš¨??ê²€ì¦?
        errors = validate_arguments(args)
        if errors:
            for error in errors:
                logger.error(f"??{error}")
            return 1
        
        # ë¡œê·¸ ?ˆë²¨ ?¤ì •
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # API ?¤ì •
        config = LawOpenAPIConfig(oc=os.getenv("LAW_OPEN_API_OC"))
        
        # ì¶œë ¥ ?”ë ‰? ë¦¬ ?¤ì •
        output_dir = Path(args.output_dir) if args.output_dir else None
        
        # ?˜ì§‘ê¸??ì„± (?œê°„ ?¸í„°ë²??¤ì • ?¬í•¨)
        collector = DateBasedConstitutionalCollector(config, output_dir)
        
        # ?œê°„ ?¸í„°ë²??¤ì •
        collector.set_request_interval(args.interval, args.interval_range)
        logger.info(f"?±ï¸ API ?”ì²­ ê°„ê²© ?¤ì •: {args.interval:.1f} Â± {args.interval_range:.1f}ì´?)
        
        # ì²´í¬?¬ì¸???¬ê°œ ëª¨ë“œ ?¤ì •
        if args.resume:
            logger.info("?”„ ì²´í¬?¬ì¸???¬ê°œ ëª¨ë“œ ?œì„±??)
            collector.enable_resume_mode()
        
        # ?˜ì§‘ ?¤í–‰
        success = False
        
        if args.strategy == "yearly":
            if args.year:
                # ?¨ì¼ ?°ë„ ?˜ì§‘
                target_count = None if args.unlimited else args.target
                date_type = "ì¢…êµ­?¼ì" if args.final_date else "? ê³ ?¼ì"
                logger.info(f"?—“ï¸?{args.year}???Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?œì‘ (ëª©í‘œ: {target_count or 'ë¬´ì œ??}ê±? {date_type} ê¸°ì?)")
                success = collector.collect_by_year(args.year, target_count, args.unlimited, args.final_date)
            else:
                # ?¤ì¤‘ ?°ë„ ?˜ì§‘
                date_type = "ì¢…êµ­?¼ì" if args.final_date else "? ê³ ?¼ì"
                logger.info(f"?—“ï¸?{args.start_year}??~ {args.end_year}???Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?œì‘ ({date_type} ê¸°ì?)")
                success = collector.collect_multiple_years(args.start_year, args.end_year, args.target)
        
        elif args.strategy == "quarterly":
            logger.info(f"?—“ï¸?{args.year}??{args.quarter}ë¶„ê¸° ?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?œì‘ (ëª©í‘œ: {args.target}ê±?")
            success = collector.collect_by_quarter(args.year, args.quarter, args.target)
        
        elif args.strategy == "monthly":
            logger.info(f"?—“ï¸?{args.year}??{args.month}???Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?œì‘ (ëª©í‘œ: {args.target}ê±?")
            success = collector.collect_by_month(args.year, args.month, args.target)
        
        if success:
            logger.info("???Œì¬ê²°ì •ë¡€ ? ì§œ ê¸°ë°˜ ?˜ì§‘???±ê³µ?ìœ¼ë¡??„ë£Œ?˜ì—ˆ?µë‹ˆ??")
            return 0
        else:
            logger.error("???Œì¬ê²°ì •ë¡€ ? ì§œ ê¸°ë°˜ ?˜ì§‘???¤íŒ¨?ˆìŠµ?ˆë‹¤.")
            return 1
        
    except KeyboardInterrupt:
        logger.warning("? ï¸ ?¬ìš©?ì— ?˜í•´ ?„ë¡œê·¸ë¨??ì¤‘ë‹¨?˜ì—ˆ?µë‹ˆ??")
        logger.info("?’¾ ?„ì¬ê¹Œì? ?˜ì§‘???°ì´?°ëŠ” ?€?¥ë˜?ˆìŠµ?ˆë‹¤.")
        return 130
    except Exception as e:
        logger.error(f"???„ë¡œê·¸ë¨ ?¤í–‰ ì¤??¤ë¥˜ ë°œìƒ: {e}")
        logger.error(f"?” ?¤ë¥˜ ?ì„¸: {traceback.format_exc()}")
        
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
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
