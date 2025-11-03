#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
êµ?šŒ ë²•ë¥ ?•ë³´?œìŠ¤??ë¶„ì•¼ë³??ë? ?˜ì§‘ (Playwright + ?ì§„??

?¬ìš©ë²?
  python collect_precedents_by_category.py --category civil --sample 50
  python collect_precedents_by_category.py --category criminal --sample 100
  python collect_precedents_by_category.py --category family --sample 30
  python collect_precedents_by_category.py --all-categories --sample 20
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.assembly_playwright_client import AssemblyPlaywrightClient
from scripts.data_collection.common.assembly_collector import AssemblyCollector
from scripts.data_collection.common.checkpoint_manager import CheckpointManager
from scripts.data_collection.common.common_utils import (
    CollectionConfig, CollectionLogger, SignalHandler, 
    MemoryManager, RetryManager, DataOptimizer, 
    check_system_requirements, memory_monitor, retry_on_failure
)

# ë¡œê¹… ?¤ì •
logger = CollectionLogger.setup_logging("precedent_category_collection")

# ?œê·¸???¸ë“¤???±ë¡
signal_handler = SignalHandler()

# ë¶„ì•¼ë³?ì½”ë“œ ë§¤í•‘ (?¤ì œ êµ?šŒ ?œìŠ¤??ê¸°ì?)
CATEGORY_CODES = {
    'civil': 'PREC00_001',      # ë¯¼ì‚¬
    'criminal': 'PREC00_002',   # ?•ì‚¬
    'tax': 'PREC00_003',        # ì¡°ì„¸
    'administrative': 'PREC00_004',  # ?‰ì •
    'family': 'PREC00_005',     # ê°€??
    'patent': 'PREC00_006',     # ?¹í—ˆ
    'maritime': 'PREC00_009',   # ?´ì‚¬
    'military': 'PREC00_010'    # êµ°ì‚¬
}

CATEGORY_NAMES = {
    'civil': 'ë¯¼ì‚¬',
    'criminal': '?•ì‚¬', 
    'tax': 'ì¡°ì„¸',
    'administrative': '?‰ì •',
    'family': 'ê°€??,
    'patent': '?¹í—ˆ',
    'maritime': '?´ì‚¬',
    'military': 'êµ°ì‚¬'
}

@memory_monitor(threshold_mb=500.0)  # 300MB ??500MBë¡?ì¡°ì •
def collect_precedents_by_category(
    category: str,
    target_count: int = None,
    page_size: int = 10,
    resume: bool = True,
    start_page: int = 1,
    config: CollectionConfig = None
):
    """
    ë¶„ì•¼ë³??ë? ?˜ì§‘ (ê°œì„ ??ë²„ì „)
    
    Args:
        category: ë¶„ì•¼ ì½”ë“œ (civil, criminal, family ??
        target_count: ëª©í‘œ ?˜ì§‘ ê±´ìˆ˜ (None=?„ì²´)
        page_size: ?˜ì´ì§€????ª© ??(?¤ì œë¡œëŠ” 10ê°?ê³ ì •)
        resume: ì²´í¬?¬ì¸?¸ì—???¬ê°œ ?¬ë?
        start_page: ?œì‘ ?˜ì´ì§€ ë²ˆí˜¸
        config: ?˜ì§‘ ?¤ì • ê°ì²´
    """
    if config is None:
        config = CollectionConfig()
    
    # ?œìŠ¤???”êµ¬?¬í•­ ?•ì¸
    if not check_system_requirements(min_memory_gb=2.0):
        logger.warning("Proceeding with caution due to system constraints")
    category_code = CATEGORY_CODES.get(category)
    category_name = CATEGORY_NAMES.get(category, category)
    
    if not category_code:
        print(f"??Unknown category: {category}")
        print(f"Available categories: {', '.join(CATEGORY_CODES.keys())}")
        return
    
    print(f"\n{'='*60}")
    print(f"?? PRECEDENT COLLECTION BY CATEGORY STARTED")
    print(f"?“‹ Category: {category_name} ({category_code})")
    print(f"{'='*60}")
    
    # ì²´í¬?¬ì¸??ë§¤ë‹ˆ?€
    checkpoint_mgr = CheckpointManager(f"data/checkpoints/precedents_{category}")
    print(f"?“ Checkpoint directory: data/checkpoints/precedents_{category}")
    
    # ì²´í¬?¬ì¸??ë¡œë“œ
    actual_start_page = start_page
    checkpoint = None
    
    if resume:
        print(f"?” Checking for existing checkpoint...")
        checkpoint = checkpoint_mgr.load_checkpoint()
        if checkpoint:
            print(f"?“‚ Resuming from checkpoint")
            print(f"   Data type: {checkpoint.get('data_type', 'unknown')}")
            print(f"   Category: {checkpoint.get('category', 'None')}")
            print(f"   Page: {checkpoint.get('current_page', 0)}/{checkpoint.get('total_pages', 0)}")
            print(f"   Collected: {checkpoint.get('collected_count', 0)} items")
            print(f"   Memory: {checkpoint.get('memory_usage_mb', 0):.1f}MB")
            actual_start_page = checkpoint['current_page'] + 1
        else:
            print(f"?“‚ No checkpoint found, starting from page {start_page}")
    else:
        print(f"?“‚ Resume disabled, starting from page {start_page}")
    
    # ë©”ëª¨ë¦?ë§¤ë‹ˆ?€ ë°??¬ì‹œ??ë§¤ë‹ˆ?€ ì´ˆê¸°??
    from scripts.data_collection.common.common_utils import MemoryManager
    memory_manager = MemoryManager(
        memory_limit_mb=config.get('memory_limit_mb', 1000),
        cleanup_threshold=config.get('cleanup_threshold', 0.3)
    )
    retry_manager = RetryManager(
        max_retries=config.get('max_retries'),
        base_delay=1.0
    )
    
    # ?˜ì§‘ê¸?ì´ˆê¸°??(?¤ì • ê¸°ë°˜)
    logger.info("Initializing collector...")
    collector = AssemblyCollector(
        base_dir="data/raw/assembly",
        data_type="precedent",
        category=category,
        batch_size=config.get('batch_size'),
        memory_limit_mb=config.get('memory_limit_mb')
    )
    print(f"??Collector initialized")
    
    # ?œì‘ ?œê°„ ?¤ì •
    start_time = datetime.now().isoformat()
    collector.set_start_time(start_time)
    
    # ?„ì²´ ?˜ì´ì§€ ê³„ì‚° (?¤ì œë¡œëŠ” ?˜ì´ì§€??10ê°œì”© ?œì‹œ??
    if target_count:
        total_pages = actual_start_page + (target_count + 10 - 1) // 10 - 1  # ?˜ì´ì§€??10ê°?
    else:
        total_pages = 100  # ?€?µì ???˜ì´ì§€ ??
    
    print(f"\n?“Š Collection Parameters:")
    print(f"   Category: {category_name} ({category_code})")
    print(f"   Target: {target_count or 'ALL'}")
    print(f"   Pages: {actual_start_page} to {total_pages}")
    print(f"   Page size: 10 (fixed)")
    print(f"   Batch size: {collector.batch_size}")
    print(f"   Memory limit: {collector.memory_limit_mb}MB")
    print(f"   Start time: {start_time}")
    
    collected_this_run = 0
    
    try:
        print(f"\n?Œ Starting Playwright browser...")
        with AssemblyPlaywrightClient(
            rate_limit=config.get('rate_limit'),
            headless=True,
            memory_limit_mb=config.get('memory_limit_mb')
        ) as client:
            print(f"??Playwright browser started")
            
            for page in range(actual_start_page, total_pages + 1):
                if signal_handler.is_interrupted():
                    logger.warning("Collection interrupted by user signal")
                    break
                
                print(f"\n{'?€'*50}")
                print(f"?“„ Processing Page {page}/{total_pages}")
                print(f"?“‹ Category: {category_name}")
                print(f"{'?€'*50}")
                
                # ?˜ì´ì§€ ?•ë³´ ?¤ì •
                collector.set_page_info(page)
                
                # ë©”ëª¨ë¦??íƒœ ?•ì¸
                memory_mb = memory_manager.get_memory_usage()
                CollectionLogger.log_memory_usage(logger, memory_mb, config.get('memory_limit_mb'))
                
                # ?¬ì‹œ??ë¡œì§?¼ë¡œ ?ë? ëª©ë¡ ê°€?¸ì˜¤ê¸?
                logger.info(f"Fetching {category_name} precedent list from page {page}...")
                precedents = retry_manager.retry(
                    client.get_precedent_list_page_by_category,
                    category_code=category_code,
                    page_num=page, 
                    page_size=10
                )
                print(f"??Found {len(precedents)} precedents on page")
                
                if not precedents:
                    print(f"? ï¸ No precedents found on page {page}, skipping...")
                    continue
                
                # ê°??ë? ?ì„¸ ?˜ì§‘ (?¤íŠ¸ë¦¬ë° ì²˜ë¦¬)
                print(f"?“‹ Processing {len(precedents)} precedents...")
                
                for idx, precedent_item in enumerate(precedents, 1):
                    if signal_handler.is_interrupted():
                        print(f"\n? ï¸ INTERRUPTED during precedent processing")
                        break
                    
                    try:
                        print(f"   [{idx:2d}/{len(precedents)}] Processing: {precedent_item['case_name'][:50]}...")
                        
                        detail = client.get_precedent_detail(precedent_item)
                        
                        # ë¶„ì•¼ ?•ë³´ ì¶”ê?
                        detail.update({
                            'category': category_name,
                            'category_code': category_code
                        })
                        
                        # ?°ì´??ìµœì ???ìš©
                        from scripts.data_collection.common.common_utils import DataOptimizer
                        detail = DataOptimizer.optimize_item(detail, 'precedent')
                        print(f"      ?”§ Data optimized for memory efficiency")
                        
                        # ?¤íŠ¸ë¦¬ë° ì²˜ë¦¬: ì¦‰ì‹œ ?€?¥í•˜ê³?ë©”ëª¨ë¦¬ì—???´ì œ
                        collector.save_item(detail)
                        collected_this_run += 1
                        
                        print(f"      ??Collected (Total: {collector.collected_count})")
                        
                        # ì¦‰ì‹œ ë©”ëª¨ë¦¬ì—???´ì œ
                        del detail
                        
                        # ë©”ëª¨ë¦??•ë¦¬ (ë§?2ê°œë§ˆ??- ???ì£¼)
                        if idx % 2 == 0:
                            import gc
                            gc.collect()
                            print(f"      ?§¹ Memory cleanup at item {idx}")
                            
                            # ë©”ëª¨ë¦??¬ìš©??ì²´í¬
                            from scripts.data_collection.common.common_utils import MemoryManager
                            memory_mgr = MemoryManager()
                            memory_mb = memory_mgr.get_memory_usage()
                            if memory_mb > 500:  # 500MB ?´ìƒ?´ë©´ ì¶”ê? ?•ë¦¬
                                print(f"      ? ï¸ High memory usage: {memory_mb:.1f}MB, performing additional cleanup")
                                memory_mgr.aggressive_cleanup()
                        
                        # ëª©í‘œ ?¬ì„± ì²´í¬
                        if target_count and collected_this_run >= target_count:
                            print(f"\n?¯ TARGET REACHED: {collected_this_run}/{target_count}")
                            break
                        
                    except Exception as e:
                        print(f"      ??Failed: {str(e)[:100]}...")
                        collector.add_failed_item(precedent_item, str(e))
                        continue
                
                # ?¤íŠ¸ë¦¬ë° ì²˜ë¦¬ë¡??¸í•´ ?˜ì´ì§€ë³??€?¥ì? ?ëµ (?´ë? ê°œë³„ ?€?¥ë¨)
                # ë©”ëª¨ë¦??•ë¦¬ë§??˜í–‰
                import gc
                gc.collect()
                
                # ì¶”ê? ë©”ëª¨ë¦??•ë¦¬
                from scripts.data_collection.common.common_utils import MemoryManager
                memory_mgr = MemoryManager()
                memory_mb = memory_mgr.get_memory_usage()
                if memory_mb > 400:  # 400MB ?´ìƒ?´ë©´ ê°•ì œ ?•ë¦¬
                    print(f"?§¹ Page {page} - High memory usage: {memory_mb:.1f}MB, performing cleanup")
                    memory_mgr.aggressive_cleanup()
                    memory_after = memory_mgr.get_memory_usage()
                    print(f"?§¹ After cleanup: {memory_after:.1f}MB")
                else:
                    print(f"?§¹ Page {page} processing completed and memory cleaned up")
                
                # ì§„í–‰ë¥?ë¡œê·¸
                print(f"\n?“ˆ Progress Summary:")
                print(f"   Category: {category_name}")
                print(f"   Page: {page}/{total_pages} ({page/total_pages*100:.1f}%)")
                print(f"   Collected this run: {collected_this_run}")
                print(f"   Total collected: {collector.collected_count}")
                print(f"   Failed: {len(collector.failed_items)}")
                print(f"   Success rate: {collector.collected_count/(collector.collected_count + len(collector.failed_items))*100:.1f}%" if (collector.collected_count + len(collector.failed_items)) > 0 else "   Success rate: N/A")
                
                checkpoint_data = {
                    'data_type': 'precedent',
                    'category': category,
                    'category_code': category_code,
                    'current_page': page,
                    'total_pages': total_pages,
                    'collected_count': collector.collected_count,
                    'collected_this_run': collected_this_run,
                    'start_time': checkpoint['start_time'] if checkpoint else start_time,
                    'memory_usage_mb': memory_mb,
                    'target_count': target_count,
                    'page_size': page_size
                }
                
                checkpoint_mgr.save_checkpoint(checkpoint_data)
                print(f"?’¾ Checkpoint saved at page {page}")
                
                if target_count and collected_this_run >= target_count:
                    print(f"\n?¯ Target achieved, stopping collection")
                    break
            
            print(f"\n? Finalizing collection...")
            collector.finalize()
            
            if not signal_handler.is_interrupted():
                checkpoint_mgr.clear_checkpoint()
                print(f"\n??COLLECTION COMPLETED SUCCESSFULLY!")
            else:
                print(f"\n? ï¸ COLLECTION INTERRUPTED (progress saved)")
            
            # ìµœì¢… ?µê³„
            print(f"\n?“Š Final Statistics:")
            print(f"   Category: {category_name}")
            print(f"   Total collected: {collector.collected_count} items")
            print(f"   Failed: {len(collector.failed_items)} items")
            print(f"   Requests made: {client.request_count}")
            print(f"   Rate limit: {client.get_stats()['rate_limit']}s")
            print(f"   Timeout: {client.get_stats()['timeout']}ms")
            
    except Exception as e:
        print(f"\n??CRITICAL ERROR: {e}")
        print(f"?”§ Finalizing collector...")
        collector.finalize()
        raise

def collect_all_categories(target_count_per_category: int = 50):
    """ëª¨ë“  ë¶„ì•¼ë³„ë¡œ ?ë? ?˜ì§‘"""
    categories = ['civil', 'criminal', 'tax', 'administrative', 'family', 'patent']
    
    print(f"\n{'='*60}")
    print(f"?? COLLECTING PRECEDENTS FOR ALL CATEGORIES")
    print(f"?“‹ Categories: {', '.join([CATEGORY_NAMES[cat] for cat in categories])}")
    print(f"?“Š Target per category: {target_count_per_category}")
    print(f"{'='*60}")
    
    total_collected = 0
    
    for category in categories:
        try:
            print(f"\n?”„ Starting collection for {CATEGORY_NAMES[category]}...")
            collect_precedents_by_category(
                category=category,
                target_count=target_count_per_category,
                resume=False,  # ê°?ë¶„ì•¼ë³„ë¡œ ?ˆë¡œ ?œì‘
                start_page=1
            )
            print(f"??Completed {CATEGORY_NAMES[category]} collection")
            
        except Exception as e:
            print(f"??Failed to collect {CATEGORY_NAMES[category]}: {e}")
            continue
    
    print(f"\n?‰ ALL CATEGORIES COLLECTION COMPLETED!")
    print(f"?“Š Total collected: {total_collected} precedents")

def main():
    """ë©”ì¸ ?¨ìˆ˜ (ê°œì„ ??ë²„ì „)"""
    parser = argparse.ArgumentParser(
        description='êµ?šŒ ë²•ë¥ ?•ë³´?œìŠ¤??ë¶„ì•¼ë³??ë? ?˜ì§‘ (Playwright + ê°œì„ ??ë©”ëª¨ë¦?ê´€ë¦?',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available categories:
  civil          - ë¯¼ì‚¬ (PREC00_001)
  criminal        - ?•ì‚¬ (PREC00_002)  
  tax             - ì¡°ì„¸ (PREC00_003)
  administrative  - ?‰ì • (PREC00_004)
  family          - ê°€??(PREC00_005)
  patent          - ?¹í—ˆ (PREC00_006)
  maritime        - ?´ì‚¬ (PREC00_009)
  military        - êµ°ì‚¬ (PREC00_010)

Examples:
  python collect_precedents_by_category.py --category civil --sample 50
  python collect_precedents_by_category.py --category criminal --sample 100 --memory-limit 500
  python collect_precedents_by_category.py --all-categories --sample 20 --batch-size 15
        """
    )
    
    parser.add_argument('--category', type=str, 
                        choices=list(CATEGORY_CODES.keys()),
                        help='?˜ì§‘??ë¶„ì•¼ ? íƒ')
    parser.add_argument('--all-categories', action='store_true',
                        help='ëª¨ë“  ë¶„ì•¼ ?˜ì§‘ (ë¯¼ì‚¬, ?•ì‚¬, ê°€??')
    parser.add_argument('--sample', type=int, metavar='N',
                        help='?˜í”Œ ?˜ì§‘ ê°œìˆ˜ (10, 50, 100 ??')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='ì²´í¬?¬ì¸?¸ì—???¬ê°œ (ê¸°ë³¸ê°?')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='ì²˜ìŒë¶€???œì‘')
    parser.add_argument('--page-size', type=int, default=50,
                        help='?˜ì´ì§€????ª© ??(ê¸°ë³¸: 50)')
    parser.add_argument('--start-page', type=int, default=1,
                        help='?œì‘ ?˜ì´ì§€ ë²ˆí˜¸ (ê¸°ë³¸: 1)')
    parser.add_argument('--memory-limit', type=int, default=1000,
                        help='ë©”ëª¨ë¦??œí•œ (MB, ê¸°ë³¸: 1000)')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='ë°°ì¹˜ ?¬ê¸° (ê¸°ë³¸: 5)')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='ìµœë? ?¬ì‹œ???Ÿìˆ˜ (ê¸°ë³¸: 3)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='ë¡œê·¸ ?ˆë²¨ (ê¸°ë³¸: INFO)')
    
    args = parser.parse_args()
    
    # ?¤ì • ê°ì²´ ?ì„±
    config = CollectionConfig(
        memory_limit_mb=args.memory_limit,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        log_level=args.log_level
    )
    
    # ë¡œê·¸ ?ˆë²¨ ?¬ì„¤??
    if args.log_level != 'INFO':
        import logging
        logger.setLevel(getattr(logging, args.log_level))
    
    if args.all_categories:
        if not args.sample:
            args.sample = 50  # ê¸°ë³¸ê°?
        print(f"?“¦ All categories mode: {args.sample} items per category")
        collect_all_categories(target_count_per_category=args.sample)
    elif args.category:
        if not args.sample:
            print("??Please specify --sample N")
            sys.exit(1)
        print(f"?“¦ Category mode: {CATEGORY_NAMES[args.category]} - {args.sample} items")
        collect_precedents_by_category(
            category=args.category,
            target_count=args.sample,
            page_size=args.page_size,
            resume=args.resume,
            start_page=args.start_page,
            config=config
        )
    else:
        parser.print_help()
        logger.error("\n??Please specify --category CATEGORY or --all-categories")
        sys.exit(1)

if __name__ == "__main__":
    main()
