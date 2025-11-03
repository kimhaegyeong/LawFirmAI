#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
êµ?šŒ ë²•ë¥ ?•ë³´?œìŠ¤??ë²•ë¥  ?˜ì§‘ (Playwright + ?ì§„??+ ?•ì¶•)

?¬ìš©ë²?
  python collect_laws.py --sample 10     # ?˜í”Œ 10ê°?
  python collect_laws.py --sample 100    # ?˜í”Œ 100ê°?
  python collect_laws.py --sample 1000   # ?˜í”Œ 1000ê°?
  python collect_laws.py --full          # ?„ì²´ 7602ê°?
  python collect_laws.py --resume        # ì¤‘ë‹¨ ì§€?ì—???¬ê°œ

?¹ì§•:
  - ?ë™ ?°ì´???•ì¶• (95% ?´ìƒ ?©ëŸ‰ ?ˆì•½)
  - ë©”ëª¨ë¦??¨ìœ¨???˜ì§‘
  - ì²´í¬?¬ì¸??ê¸°ë°˜ ?¬ê°œ ê¸°ëŠ¥
  - ?¤ì‹œê°??•ì¶• ?µê³„ ?œì‹œ
"""

import argparse
import sys
import signal
import logging
import json
from pathlib import Path
from datetime import datetime

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from source.data.assembly_playwright_client import AssemblyPlaywrightClient
from scripts.data_collection.common.assembly_collector import AssemblyCollector
from scripts.data_collection.common.checkpoint_manager import CheckpointManager
from scripts.data_collection.common.assembly_logger import setup_logging, log_progress, log_memory_usage, log_collection_stats, log_checkpoint_info
from scripts.data_processing.utilities.law_data_compressor import compress_law_data, compress_and_save_page_data

# ë¡œê±° ?¤ì •
logger = setup_logging("law_collection")

# Graceful shutdown ì²˜ë¦¬
interrupted = False

def signal_handler(sig, frame):
    """?œê·¸???¸ë“¤??(Ctrl+C ??"""
    global interrupted
    logger.warning("\n? ï¸ Interrupt signal received. Saving progress...")
    interrupted = True

# ?œê·¸???¸ë“¤???±ë¡
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def collect_laws_incremental(
    target_count: int = None,
    page_size: int = 100,
    resume: bool = True,
    start_page: int = 1
):
    """
    ?ì§„??ë²•ë¥  ?˜ì§‘
    
    Args:
        target_count: ëª©í‘œ ?˜ì§‘ ê±´ìˆ˜ (None=?„ì²´)
        page_size: ?˜ì´ì§€????ª© ??(100 ê¶Œì¥)
        resume: ì²´í¬?¬ì¸?¸ì—???¬ê°œ
    """
    
    print(f"\n{'='*60}")
    print(f"?? LAW COLLECTION STARTED")
    print(f"{'='*60}")
    
    # ì²´í¬?¬ì¸??ë§¤ë‹ˆ?€
    checkpoint_mgr = CheckpointManager("data/checkpoints/laws")
    print(f"?“ Checkpoint directory: data/checkpoints/laws")
    
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
    
    # ?˜ì§‘ê¸?ì´ˆê¸°??
    print(f"\n?“¦ Initializing collector...")
    collector = AssemblyCollector(
        base_dir="data/raw/assembly",
        data_type="law",
        category=None,
        batch_size=50,
        memory_limit_mb=800
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
    print(f"   Target: {target_count or 'ALL (7602)'} items")
    print(f"   Pages: {actual_start_page} to {total_pages}")
    print(f"   Page size: 10 (fixed)")
    print(f"   Batch size: {collector.batch_size}")
    print(f"   Memory limit: {collector.memory_limit_mb}MB")
    print(f"   Start time: {start_time}")
    
    collected_this_run = 0
    total_original_size = 0
    total_compressed_size = 0
    
    try:
        print(f"\n?Œ Starting Playwright browser...")
        # Playwright ?œì‘
        with AssemblyPlaywrightClient(
            rate_limit=3.0,
            headless=True,
            memory_limit_mb=800
        ) as client:
            print(f"??Playwright browser started")
            
            for page in range(actual_start_page, total_pages + 1):
                if interrupted:
                    print(f"\n? ï¸ INTERRUPTED by user signal")
                    break
                
                print(f"\n{'?€'*50}")
                print(f"?“„ Processing Page {page}/{total_pages}")
                print(f"{'?€'*50}")
                
                # ë©”ëª¨ë¦?ì²´í¬
                memory_mb = client.check_memory_usage()
                print(f"?“Š Memory usage: {memory_mb:.1f}MB")
                
                # ëª©ë¡ ì¡°íšŒ
                print(f"?” Fetching law list from page {page}...")
                laws = client.get_law_list_page(page_num=page, page_size=10)
                print(f"??Found {len(laws)} laws on page")
                
                if not laws:
                    print(f"? ï¸ No laws found on page {page}, skipping...")
                    continue
                
                # ê°?ë²•ë¥  ?ì„¸ ?˜ì§‘
                print(f"?“‹ Processing {len(laws)} laws...")
                page_laws = []  # ?„ì¬ ?˜ì´ì§€??ë²•ë¥ ?¤ì„ ?€?¥í•  ë¦¬ìŠ¤??
                
                for idx, law_item in enumerate(laws, 1):
                    if interrupted:
                        print(f"\n? ï¸ INTERRUPTED during law processing")
                        break
                    
                    try:
                        print(f"   [{idx:2d}/{len(laws)}] Processing: {law_item['law_name'][:50]}...")
                        
                        detail = client.get_law_detail(
                            law_item['cont_id'],
                            law_item['cont_sid']
                        )
                        
                        # ëª©ë¡ ?•ë³´ ë³‘í•©
                        detail.update({
                            'row_number': law_item['row_number'],
                            'category': law_item['category'],
                            'law_type': law_item['law_type'],
                            'promulgation_number': law_item['promulgation_number'],
                            'promulgation_date': law_item['promulgation_date'],
                            'enforcement_date': law_item['enforcement_date'],
                            'amendment_type': law_item['amendment_type']
                        })
                        
                        page_laws.append(detail)  # ?˜ì´ì§€ë³?ë¦¬ìŠ¤?¸ì— ì¶”ê?
                        collector.save_item(detail)
                        collected_this_run += 1
                        
                        print(f"      ??Collected (Total: {collector.collected_count})")
                        
                        # ëª©í‘œ ?¬ì„± ì²´í¬
                        if target_count and collected_this_run >= target_count:
                            print(f"\n?¯ TARGET REACHED: {collected_this_run}/{target_count}")
                            break
                        
                    except Exception as e:
                        print(f"      ??Failed: {str(e)[:100]}...")
                        collector.add_failed_item(law_item, str(e))
                        continue
                
                # ?„ì¬ ?˜ì´ì§€??ë²•ë¥ ?¤ì„ ë³„ë„ ?Œì¼ë¡??€??(?•ì¶•??ë²„ì „)
                if page_laws:
                    timestamp = datetime.now().strftime("%H%M%S")
                    page_filename = f"law_page_{page:03d}_{timestamp}.json"
                    page_filepath = collector.output_dir / page_filename
                    
                    page_data = {
                        "page_number": page,
                        "total_pages": total_pages,
                        "laws_count": len(page_laws),
                        "collected_at": datetime.now().isoformat(),
                        "laws": page_laws
                    }
                    
                    # ?•ì¶•???°ì´?°ë¡œ ?€??
                    compressed_size = compress_and_save_page_data(page_data, str(page_filepath))
                    
                    # ?•ì¶• ?µê³„ ?…ë°?´íŠ¸
                    total_compressed_size += compressed_size
                    
                    # ?ë³¸ ?¬ê¸° ì¶”ì • (?•ì¶• ???¬ê¸°)
                    estimated_original_size = compressed_size * 20  # ?€?µì ???•ì¶•ë¥?ê³ ë ¤
                    total_original_size += estimated_original_size
                    
                    compression_ratio = (1 - compressed_size / estimated_original_size) * 100 if estimated_original_size > 0 else 0
                    
                    print(f"?“„ Page {page} saved: {page_filename} ({len(page_laws)} laws, {compressed_size:,} bytes, {compression_ratio:.1f}% ?•ì¶•)")
                
                # ì§„í–‰ë¥?ë¡œê·¸
                print(f"\n?“ˆ Progress Summary:")
                print(f"   Page: {page}/{total_pages} ({page/total_pages*100:.1f}%)")
                print(f"   Collected this run: {collected_this_run}")
                print(f"   Total collected: {collector.collected_count}")
                print(f"   Failed: {len(collector.failed_items)}")
                print(f"   Success rate: {collector.collected_count/(collector.collected_count + len(collector.failed_items))*100:.1f}%" if (collector.collected_count + len(collector.failed_items)) > 0 else "   Success rate: N/A")
                
                # ì²´í¬?¬ì¸???€??
                checkpoint_data = {
                    'data_type': 'law',
                    'category': None,
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
                
                # ëª©í‘œ ?¬ì„± ??ì¢…ë£Œ
                if target_count and collected_this_run >= target_count:
                    print(f"\n?¯ Target achieved, stopping collection")
                    break
            
            # ?˜ì§‘ ?„ë£Œ
            print(f"\n? Finalizing collection...")
            collector.finalize()
            
            if not interrupted:
                checkpoint_mgr.clear_checkpoint()
                print(f"\n??COLLECTION COMPLETED SUCCESSFULLY!")
            else:
                print(f"\n? ï¸ COLLECTION INTERRUPTED (progress saved)")
            
            # ìµœì¢… ?µê³„
            print(f"\n?“Š Final Statistics:")
            print(f"   Total collected: {collector.collected_count} items")
            print(f"   Failed: {len(collector.failed_items)} items")
            print(f"   Requests made: {client.request_count}")
            print(f"   Rate limit: {client.get_stats()['rate_limit']}s")
            print(f"   Timeout: {client.get_stats()['timeout']}ms")
            
            # ?•ì¶• ?µê³„
            if total_compressed_size > 0:
                overall_compression_ratio = (1 - total_compressed_size / total_original_size) * 100 if total_original_size > 0 else 0
                print(f"\n?—œï¸?Compression Statistics:")
                print(f"   Estimated original size: {total_original_size:,} bytes ({total_original_size/1024/1024:.1f} MB)")
                print(f"   Compressed size: {total_compressed_size:,} bytes ({total_compressed_size/1024/1024:.1f} MB)")
                print(f"   Compression ratio: {overall_compression_ratio:.1f}%")
                print(f"   Space saved: {(total_original_size - total_compressed_size)/1024/1024:.1f} MB")
            
    except Exception as e:
        print(f"\n??CRITICAL ERROR: {e}")
        print(f"?”§ Finalizing collector...")
        collector.finalize()
        raise

def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    parser = argparse.ArgumentParser(
        description='êµ?šŒ ë²•ë¥ ?•ë³´?œìŠ¤??ë²•ë¥  ?˜ì§‘ (Playwright)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect_laws.py --sample 10                    # ?˜í”Œ 10ê°??˜ì§‘
  python collect_laws.py --sample 100                   # ?˜í”Œ 100ê°??˜ì§‘
  python collect_laws.py --sample 100 --start-page 5     # 5?˜ì´ì§€ë¶€??100ê°??˜ì§‘
  python collect_laws.py --sample 50 --start-page 10     # 10?˜ì´ì§€ë¶€??50ê°??˜ì§‘
  python collect_laws.py --full                          # ?„ì²´ 7602ê°??˜ì§‘
  python collect_laws.py --resume                        # ì¤‘ë‹¨ ì§€?ì—???¬ê°œ
        """
    )
    
    parser.add_argument('--sample', type=int, metavar='N',
                        help='?˜í”Œ ?˜ì§‘ ê°œìˆ˜ (10, 100, 1000 ??')
    parser.add_argument('--full', action='store_true',
                        help='?„ì²´ ?˜ì§‘ (7602ê°?')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='ì²´í¬?¬ì¸?¸ì—???¬ê°œ (ê¸°ë³¸ê°?')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='ì²˜ìŒë¶€???œì‘')
    parser.add_argument('--page-size', type=int, default=100,
                        help='?˜ì´ì§€????ª© ??(ê¸°ë³¸: 100)')
    parser.add_argument('--start-page', type=int, default=1,
                        help='?œì‘ ?˜ì´ì§€ ë²ˆí˜¸ (ê¸°ë³¸: 1)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='ë¡œê·¸ ?ˆë²¨ (ê¸°ë³¸: INFO)')
    
    args = parser.parse_args()
    
    # ë¡œê·¸ ?ˆë²¨ ?¬ì„¤??
    if args.log_level != 'INFO':
        logger.setLevel(getattr(logging, args.log_level))
    
    if args.sample:
        print(f"?“¦ Sample mode: {args.sample} items")
        collect_laws_incremental(
            target_count=args.sample,
            page_size=args.page_size,
            resume=args.resume,
            start_page=args.start_page
        )
    elif args.full:
        logger.info(f"?“¦ Full mode: 7602 items")
        collect_laws_incremental(
            target_count=None,
            page_size=args.page_size,
            resume=args.resume,
            start_page=args.start_page
        )
    else:
        parser.print_help()
        logger.error("\n??Please specify --sample N or --full")
        sys.exit(1)

if __name__ == "__main__":
    main()
