#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ë²•ë ¹?´ì„ë¡€ ? ì§œ ê¸°ë°˜ ?˜ì§‘ ë©”ì¸ ?¤í¬ë¦½íŠ¸

?¬ìš©ë²?
    python scripts/legal_interpretation/collect_by_date.py --strategy yearly --year 2025 --interpretation-date
    python scripts/legal_interpretation/collect_by_date.py --strategy quarterly --year 2025 --quarter 1
    python scripts/legal_interpretation/collect_by_date.py --strategy monthly --year 2025 --month 8
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# source ëª¨ë“ˆ ê²½ë¡œ ì¶”ê?
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'source'))

# ?„ì¬ ?”ë ‰? ë¦¬??date_based_collector ëª¨ë“ˆ import
from date_based_collector import (
    DateBasedLegalInterpretationCollector, 
    CollectionConfig, 
    DateCollectionStrategy
)
from data.law_open_api_client import load_env_file

def parse_arguments():
    """ëª…ë ¹???¸ìˆ˜ ?Œì‹±"""
    parser = argparse.ArgumentParser(
        description="ë²•ë ¹?´ì„ë¡€ ? ì§œ ê¸°ë°˜ ?˜ì§‘",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
?¬ìš© ?ˆì‹œ:
  # 2025??ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ (?´ì„?¼ì ê¸°ì?)
  python scripts/legal_interpretation/collect_by_date.py --strategy yearly --year 2025 --interpretation-date
  
  # 2024??ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ (?Œì‹ ?¼ì ê¸°ì?)
  python scripts/legal_interpretation/collect_by_date.py --strategy yearly --year 2024
  
  # ?¹ì • ê±´ìˆ˜ë§??˜ì§‘
  python scripts/legal_interpretation/collect_by_date.py --strategy yearly --year 2025 --target 100 --interpretation-date
  
  # ë¶„ê¸°ë³??˜ì§‘
  python scripts/legal_interpretation/collect_by_date.py --strategy quarterly --year 2025 --quarter 1
  
  # ?”ë³„ ?˜ì§‘
  python scripts/legal_interpretation/collect_by_date.py --strategy monthly --year 2025 --month 8
        """
    )
    
    # ?˜ì§‘ ?„ëµ
    parser.add_argument(
        '--strategy', 
        choices=['yearly', 'quarterly', 'monthly'], 
        required=True,
        help='?˜ì§‘ ?„ëµ (yearly: ?°ë„ë³? quarterly: ë¶„ê¸°ë³? monthly: ?”ë³„)'
    )
    
    # ?°ë„ ê´€??
    parser.add_argument(
        '--year', 
        type=int, 
        required=True,
        help='?˜ì§‘???°ë„ (?? 2025)'
    )
    
    # ë¶„ê¸° ê´€??
    parser.add_argument(
        '--quarter', 
        type=int, 
        choices=[1, 2, 3, 4],
        help='?˜ì§‘??ë¶„ê¸° (1-4ë¶„ê¸°, quarterly ?„ëµ?ì„œë§??¬ìš©)'
    )
    
    # ??ê´€??
    parser.add_argument(
        '--month', 
        type=int, 
        choices=list(range(1, 13)),
        help='?˜ì§‘????(1-12?? monthly ?„ëµ?ì„œë§??¬ìš©)'
    )
    
    # ëª©í‘œ ê±´ìˆ˜
    parser.add_argument(
        '--target', 
        type=int,
        help='?˜ì§‘??ëª©í‘œ ê±´ìˆ˜ (ê¸°ë³¸ê°? ?„ëµë³?ê¸°ë³¸ê°?'
    )
    
    # ë¬´ì œ???˜ì§‘
    parser.add_argument(
        '--unlimited', 
        action='store_true',
        help='ë¬´ì œ???˜ì§‘ (ëª¨ë“  ?°ì´???˜ì§‘)'
    )
    
    # ? ì§œ ê¸°ì?
    parser.add_argument(
        '--interpretation-date', 
        action='store_true',
        help='?´ì„?¼ì ê¸°ì??¼ë¡œ ?˜ì§‘ (ê¸°ë³¸ê°? ?Œì‹ ?¼ì ê¸°ì?)'
    )
    
    # ì¶œë ¥ ?”ë ‰? ë¦¬
    parser.add_argument(
        '--output-dir', 
        type=str,
        help='ì¶œë ¥ ?”ë ‰? ë¦¬ (ê¸°ë³¸ê°? data/raw/legal_interpretations)'
    )
    
    # ì²´í¬?¬ì¸???•ì¸
    parser.add_argument(
        '--check', 
        action='store_true',
        help='ê¸°ì¡´ ?˜ì§‘ ?°ì´???•ì¸ë§?(?˜ì§‘?˜ì? ?ŠìŒ)'
    )
    
    # ?ì„¸ ë¡œê·¸
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='?ì„¸ ë¡œê·¸ ì¶œë ¥'
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """ë¡œê¹… ?¤ì •"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # ë¡œê·¸ ?”ë ‰? ë¦¬ ?ì„±
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
    """?¸ìˆ˜ ê²€ì¦?""
    errors = []
    
    # ?„ëµë³??„ìˆ˜ ?¸ìˆ˜ ê²€ì¦?
    if args.strategy == 'quarterly' and not args.quarter:
        errors.append("quarterly ?„ëµ ?¬ìš© ??--quarter ?¸ìˆ˜ê°€ ?„ìš”?©ë‹ˆ??)
    
    if args.strategy == 'monthly' and not args.month:
        errors.append("monthly ?„ëµ ?¬ìš© ??--month ?¸ìˆ˜ê°€ ?„ìš”?©ë‹ˆ??)
    
    # ?°ë„ ê²€ì¦?
    if args.year < 2000 or args.year > 2030:
        errors.append("?°ë„??2000-2030 ?¬ì´?¬ì•¼ ?©ë‹ˆ??)
    
    # ëª©í‘œ ê±´ìˆ˜ ê²€ì¦?
    if args.target and args.target <= 0:
        errors.append("ëª©í‘œ ê±´ìˆ˜??0ë³´ë‹¤ ì»¤ì•¼ ?©ë‹ˆ??)
    
    if errors:
        for error in errors:
            print(f"???¤ë¥˜: {error}")
        return False
    
    return True


def get_default_target_count(strategy: str) -> int:
    """?„ëµë³?ê¸°ë³¸ ëª©í‘œ ê±´ìˆ˜ ë°˜í™˜"""
    defaults = {
        'yearly': 1000,
        'quarterly': 500,
        'monthly': 200
    }
    return defaults.get(strategy, 100)


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    try:
        # ?˜ê²½ ë³€??ë¡œë“œ
        load_env_file()
        
        # ?¸ìˆ˜ ?Œì‹±
        args = parse_arguments()
        
        # ?¸ìˆ˜ ê²€ì¦?
        if not validate_arguments(args):
            return 1
        
        # ë¡œê¹… ?¤ì •
        logger = setup_logging(args.verbose)
        
        # ì¶œë ¥ ?”ë ‰? ë¦¬ ?¤ì •
        output_dir = Path(args.output_dir) if args.output_dir else None
        
        # ?˜ì§‘ ?¤ì • ?ì„±
        config = CollectionConfig()
        if output_dir:
            config.base_output_dir = output_dir
        
        # ?˜ì§‘ê¸??ì„±
        collector = DateBasedLegalInterpretationCollector(config)
        
        # ëª©í‘œ ê±´ìˆ˜ ?¤ì •
        target_count = args.target
        if not target_count and not args.unlimited:
            target_count = get_default_target_count(args.strategy)
        
        # ì²´í¬?¬ì¸???•ì¸ë§??˜ëŠ” ê²½ìš°
        if args.check:
            logger.info("?“‹ ê¸°ì¡´ ?˜ì§‘ ?°ì´???•ì¸ ì¤?..")
            collector._load_existing_data(target_year=args.year)
            logger.info(f"?“Š ê¸°ì¡´ ?˜ì§‘??ë²•ë ¹?´ì„ë¡€: {len(collector.collected_decisions):,}ê±?)
            return 0
        
        # ?˜ì§‘ ?¤í–‰
        success = False
        date_type = "?´ì„?¼ì" if args.interpretation_date else "?Œì‹ ?¼ì"
        
        if args.strategy == 'yearly':
            logger.info(f"?—“ï¸?{args.year}??ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ ?œì‘ ({date_type} ê¸°ì?)")
            success = collector.collect_by_year(
                year=args.year,
                target_count=target_count,
                unlimited=args.unlimited,
                use_interpretation_date=args.interpretation_date
            )
            
        elif args.strategy == 'quarterly':
            logger.info(f"?—“ï¸?{args.year}??{args.quarter}ë¶„ê¸° ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ ?œì‘")
            success = collector.collect_by_quarter(
                year=args.year,
                quarter=args.quarter,
                target_count=target_count
            )
            
        elif args.strategy == 'monthly':
            logger.info(f"?—“ï¸?{args.year}??{args.month}??ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ ?œì‘")
            success = collector.collect_by_month(
                year=args.year,
                month=args.month,
                target_count=target_count
            )
        
        if success:
            logger.info("??ë²•ë ¹?´ì„ë¡€ ? ì§œ ê¸°ë°˜ ?˜ì§‘???±ê³µ?ìœ¼ë¡??„ë£Œ?˜ì—ˆ?µë‹ˆ??")
            return 0
        else:
            logger.error("??ë²•ë ¹?´ì„ë¡€ ? ì§œ ê¸°ë°˜ ?˜ì§‘???¤íŒ¨?ˆìŠµ?ˆë‹¤.")
            return 1
            
    except KeyboardInterrupt:
        logger.info("? ï¸ ?¬ìš©?ì— ?˜í•´ ?„ë¡œê·¸ë¨??ì¤‘ë‹¨?˜ì—ˆ?µë‹ˆ??")
        return 1
    except Exception as e:
        logger.error(f"???ˆìƒì¹?ëª»í•œ ?¤ë¥˜ ë°œìƒ: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
