#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
êµ?šŒ ë²•ë¥ ?•ë³´?œìŠ¤??ë²•ë¥  ?˜ì§‘ (ìµœì ??ë²„ì „)

?±ëŠ¥ ê°œì„  ?¬í•­:
1. ë°°ì¹˜ ì²˜ë¦¬ë¡?ë©”ëª¨ë¦??¨ìœ¨???¥ìƒ
2. ë¶ˆí•„?”í•œ ë©”ëª¨ë¦?ì²´í¬ ìµœì†Œ??
3. ?Œì¼ I/O ìµœì ??
4. ?˜ì´ì§€ ?¤ë¹„ê²Œì´??ìµœì ??
5. ?ëŸ¬ ì²˜ë¦¬ ê°œì„ 

?¬ìš©ë²?
  python collect_laws_optimized.py --sample 10     # ?˜í”Œ 10ê°?
  python collect_laws_optimized.py --sample 100    # ?˜í”Œ 100ê°?
  python collect_laws_optimized.py --sample 1000   # ?˜í”Œ 1000ê°?
  python collect_laws_optimized.py --full          # ?„ì²´ 7602ê°?
  python collect_laws_optimized.py --resume        # ì¤‘ë‹¨ ì§€?ì—???¬ê°œ
"""

import argparse
import sys
import signal
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import time

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.assembly_playwright_client import AssemblyPlaywrightClient
from scripts.assembly.assembly_collector import AssemblyCollector
from scripts.assembly.checkpoint_manager import CheckpointManager
from scripts.assembly.assembly_logger import setup_logging, log_progress, log_memory_usage, log_collection_stats, log_checkpoint_info
from scripts.monitoring.metrics_collector import LawCollectionMetrics

# ë¡œê±° ?¤ì •
logger = setup_logging("law_collection_optimized")

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

class OptimizedLawCollector:
    """ìµœì ?”ëœ ë²•ë¥  ?˜ì§‘ê¸?(?˜ì´ì§€ë³??€??"""
    
    def __init__(self, base_dir: str = "data/raw/assembly", page_size: int = 10, enable_metrics: bool = True):
        self.base_dir = Path(base_dir)
        self.page_size = page_size
        self.collected_items = []
        self.failed_items = []
        self.start_time = None
        self.enable_metrics = enable_metrics
        
        # ì¶œë ¥ ?”ë ‰? ë¦¬ ?ì„±
        self.output_dir = self.base_dir / "law" / datetime.now().strftime("%Y%m%d")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"?“ Output directory: {self.output_dir}")
        
        # ë©”íŠ¸ë¦??˜ì§‘ê¸?ì´ˆê¸°??
        if self.enable_metrics:
            try:
                # ê¸°ì¡´ ë©”íŠ¸ë¦??œë²„ê°€ ?¤í–‰ ì¤‘ì¸ì§€ ?•ì¸
                import requests
                try:
                    response = requests.get("http://localhost:8000/metrics", timeout=1)
                    if response.status_code == 200:
                        print(f"?“Š Using existing metrics server")
                        # ê¸°ì¡´ ?œë²„ë¥??¬ìš©?˜ë˜, ë©”íŠ¸ë¦??¸ìŠ¤?´ìŠ¤???ˆë¡œ ?ì„±
                        self.metrics = LawCollectionMetrics()
                        self.metrics.start_collection()
                        print(f"?“Š Connected to existing metrics server")
                    else:
                        raise Exception("Metrics server not responding")
                except:
                    # ë©”íŠ¸ë¦??œë²„ê°€ ?†ìœ¼ë©??ˆë¡œ ?œì‘
                    print(f"?“Š Starting new metrics server")
                    self.metrics = LawCollectionMetrics()
                    self.metrics.start_server()
                    self.metrics.start_collection()
                    print(f"?“Š Metrics server started")
                
                print(f"?“Š Metrics collection enabled")
            except Exception as e:
                print(f"? ï¸ Failed to start metrics: {e}")
                self.enable_metrics = False
                self.metrics = None
        else:
            self.metrics = None
            print(f"?“Š Metrics collection disabled")
    
    def add_item(self, item: Dict):
        """?„ì´??ì¶”ê? (?˜ì´ì§€ë³?ì²˜ë¦¬??"""
        self.collected_items.append(item)
    
    def save_page(self, page_number: int):
        """?˜ì´ì§€ë³??€??(10ê°œì”©)"""
        if not self.collected_items:
            return
        
        # ?˜ì´ì§€ ì²˜ë¦¬ ?œê°„ ê¸°ë¡
        page_start_time = time.time()
        
        timestamp = datetime.now().strftime("%H%M%S")
        page_filename = f"law_page_{page_number:03d}_{timestamp}.json"
        page_filepath = self.output_dir / page_filename
        
        page_data = {
            "page_info": {
                "page_number": page_number,
                "laws_count": len(self.collected_items),
                "saved_at": datetime.now().isoformat(),
                "page_size": self.page_size
            },
            "laws": self.collected_items
        }
        
        try:
            with open(page_filepath, 'w', encoding='utf-8') as f:
                json.dump(page_data, f, ensure_ascii=False, indent=2)
            
            print(f"?“„ Page {page_number} saved: {page_filename} ({len(self.collected_items)} laws)")
            
            # ë©”íŠ¸ë¦?ê¸°ë¡
            if self.metrics:
                page_duration = time.time() - page_start_time
                self.metrics.record_page_processed(page_number)
                self.metrics.record_laws_collected(len(self.collected_items))
                self.metrics.record_page_processing_time(page_duration)
                
                # ?˜ì´ì§€ ?•ë³´??ì²˜ë¦¬ ?œê°„ ì¶”ê?
                page_data["page_info"]["processing_time"] = page_duration
            
            self.collected_items.clear()
            
        except Exception as e:
            print(f"??Failed to save page {page_number}: {e}")
            if self.metrics:
                self.metrics.record_error("file_save_error")
    
    def add_failed_item(self, item: Dict, error: str):
        """?¤íŒ¨???„ì´??ì¶”ê?"""
        self.failed_items.append({
            'item': item,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        
        # ?ëŸ¬ ë©”íŠ¸ë¦?ê¸°ë¡
        if self.metrics:
            error_type = "network_error" if "network" in error.lower() else "parsing_error"
            self.metrics.record_error(error_type)
    
    def finalize(self):
        """ìµœì¢… ?€??""
        if self.collected_items:
            # ?¨ì? ?„ì´?œë“¤??ë§ˆì?ë§??˜ì´ì§€ë¡??€??
            timestamp = datetime.now().strftime("%H%M%S")
            page_filename = f"law_page_final_{timestamp}.json"
            page_filepath = self.output_dir / page_filename
            
            page_data = {
                "page_info": {
                    "page_number": "final",
                    "laws_count": len(self.collected_items),
                    "saved_at": datetime.now().isoformat(),
                    "page_size": self.page_size
                },
                "laws": self.collected_items
            }
            
            try:
                with open(page_filepath, 'w', encoding='utf-8') as f:
                    json.dump(page_data, f, ensure_ascii=False, indent=2)
                
                print(f"?“„ Final page saved: {page_filename} ({len(self.collected_items)} laws)")
                
                # ë©”íŠ¸ë¦?ê¸°ë¡
                if self.metrics:
                    self.metrics.record_laws_collected(len(self.collected_items))
                
                self.collected_items.clear()
                
            except Exception as e:
                print(f"??Failed to save final page: {e}")
                if self.metrics:
                    self.metrics.record_error("file_save_error")
        
        # ?¤íŒ¨???„ì´?œë“¤ ?€??
        if self.failed_items:
            failed_filename = f"failed_items_{datetime.now().strftime('%H%M%S')}.json"
            failed_filepath = self.output_dir / failed_filename
            
            with open(failed_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.failed_items, f, ensure_ascii=False, indent=2)
            
            print(f"??Failed items saved: {failed_filename} ({len(self.failed_items)} items)")
        
        # ë©”íŠ¸ë¦??˜ì§‘ ì¤‘ì?
        if self.metrics:
            self.metrics.stop_collection()
    
    @property
    def collected_count(self) -> int:
        """?˜ì§‘??ì´??„ì´????""
        # ?˜ì´ì§€ ?Œì¼?¤ì—??ì´?ê°œìˆ˜ ê³„ì‚°
        total_count = len(self.collected_items)  # ?„ì¬ ë©”ëª¨ë¦¬ì— ?ˆëŠ” ?„ì´?œë“¤
        
        # ?€?¥ëœ ?˜ì´ì§€ ?Œì¼?¤ì—??ê°œìˆ˜ ?©ì‚°
        for page_file in self.output_dir.glob("law_page_*.json"):
            try:
                with open(page_file, 'r', encoding='utf-8') as f:
                    page_data = json.load(f)
                    if 'page_info' in page_data and 'laws_count' in page_data['page_info']:
                        total_count += page_data['page_info']['laws_count']
            except Exception:
                continue
        
        return total_count

def collect_laws_optimized(
    target_count: int = None,
    page_size: int = 100,
    resume: bool = True,
    start_page: int = 1,
    laws_per_page: int = 10,
    enable_metrics: bool = True
):
    """
    ìµœì ?”ëœ ?ì§„??ë²•ë¥  ?˜ì§‘ (?˜ì´ì§€ë³??€??
    
    Args:
        target_count: ëª©í‘œ ?˜ì§‘ ê±´ìˆ˜ (None=?„ì²´)
        page_size: ?˜ì´ì§€????ª© ??(100 ê¶Œì¥)
        resume: ì²´í¬?¬ì¸?¸ì—???¬ê°œ
        start_page: ?œì‘ ?˜ì´ì§€ ë²ˆí˜¸
        laws_per_page: ?˜ì´ì§€??ë²•ë¥  ??(10ê°?ê³ ì •)
        enable_metrics: ë©”íŠ¸ë¦??˜ì§‘ ?œì„±???¬ë?
    """
    
    print(f"\n{'='*60}")
    print(f"?? OPTIMIZED LAW COLLECTION STARTED")
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
    
    # ìµœì ?”ëœ ?˜ì§‘ê¸?ì´ˆê¸°??
    print(f"\n?“¦ Initializing optimized collector...")
    collector = OptimizedLawCollector(
        base_dir="data/raw/assembly",
        page_size=laws_per_page,
        enable_metrics=enable_metrics
    )
    print(f"??Optimized collector initialized (page size: {laws_per_page})")
    
    # ?œì‘ ?œê°„ ?¤ì •
    start_time = datetime.now().isoformat()
    collector.start_time = start_time
    
    # ?„ì²´ ?˜ì´ì§€ ê³„ì‚°
    if target_count:
        total_pages = actual_start_page + (target_count + 9) // 10 - 1  # ?˜ì´ì§€??10ê°?
    else:
        total_pages = 100  # ?€?µì ???˜ì´ì§€ ??
    
    print(f"\n?“Š Collection Parameters:")
    print(f"   Target: {target_count or 'ALL (7602)'} items")
    print(f"   Pages: {actual_start_page} to {total_pages}")
    print(f"   Laws per page: {laws_per_page} (fixed)")
    print(f"   Save mode: Page-by-page")
    print(f"   Start time: {start_time}")
    
    collected_this_run = 0
    last_memory_check = 0
    
    try:
        print(f"\n?Œ Starting Playwright browser...")
        # Playwright ?œì‘ (ìµœì ?”ëœ ?¤ì •)
        with AssemblyPlaywrightClient(
            rate_limit=3.0,  # Rate limiting ? ì?
            headless=True,
            memory_limit_mb=1000  # ë©”ëª¨ë¦??œí•œ ì¦ê?
        ) as client:
            print(f"??Playwright browser started")
            
            for page in range(actual_start_page, total_pages + 1):
                if interrupted:
                    print(f"\n? ï¸ INTERRUPTED by user signal")
                    break
                
                print(f"\n{'?€'*50}")
                print(f"?“„ Processing Page {page}/{total_pages}")
                print(f"{'?€'*50}")
                
                # ë©”ëª¨ë¦?ì²´í¬ ìµœì ??(10?˜ì´ì§€ë§ˆë‹¤ë§?ì²´í¬)
                if page - last_memory_check >= 10:
                    memory_mb = client.check_memory_usage()
                    print(f"?“Š Memory usage: {memory_mb:.1f}MB")
                    last_memory_check = page
                
                # ëª©ë¡ ì¡°íšŒ
                print(f"?” Fetching law list from page {page}...")
                laws = client.get_law_list_page(page_num=page, page_size=10)
                print(f"??Found {len(laws)} laws on page")
                
                if not laws:
                    print(f"? ï¸ No laws found on page {page}, skipping...")
                    continue
                
                # ê°?ë²•ë¥  ?ì„¸ ?˜ì§‘ (ìµœì ?”ëœ ì²˜ë¦¬)
                print(f"?“‹ Processing {len(laws)} laws...")
                page_start_time = time.time()
                
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
                        
                        collector.add_item(detail)
                        collected_this_run += 1
                        
                        print(f"      ??Collected (Page: {len(collector.collected_items)}/{laws_per_page})")
                        
                        # ëª©í‘œ ?¬ì„± ì²´í¬
                        if target_count and collected_this_run >= target_count:
                            print(f"\n?¯ TARGET REACHED: {collected_this_run}/{target_count}")
                            break
                        
                    except Exception as e:
                        print(f"      ??Failed: {str(e)[:100]}...")
                        collector.add_failed_item(law_item, str(e))
                        continue
                
                # ?˜ì´ì§€ë³??€??
                collector.save_page(page)
                
                # ?˜ì´ì§€ ì²˜ë¦¬ ?œê°„ ê³„ì‚°
                page_time = time.time() - page_start_time
                print(f"?±ï¸ Page {page} processed in {page_time:.1f}s")
                
                # ì§„í–‰ë¥?ë¡œê·¸ (ìµœì ?”ëœ ì¶œë ¥)
                print(f"\n?“ˆ Progress Summary:")
                print(f"   Page: {page}/{total_pages} ({page/total_pages*100:.1f}%)")
                print(f"   Collected this run: {collected_this_run}")
                print(f"   Total collected: {collector.collected_count}")
                print(f"   Failed: {len(collector.failed_items)}")
                print(f"   Success rate: {collected_this_run/(collected_this_run + len(collector.failed_items))*100:.1f}%" if (collected_this_run + len(collector.failed_items)) > 0 else "   Success rate: N/A")
                
                # ì²´í¬?¬ì¸???€??(ìµœì ?”ëœ ë¹ˆë„)
                if page % 5 == 0:  # 5?˜ì´ì§€ë§ˆë‹¤ë§?ì²´í¬?¬ì¸???€??
                    checkpoint_data = {
                        'data_type': 'law',
                        'category': None,
                        'current_page': page,
                        'total_pages': total_pages,
                        'collected_count': collector.collected_count,
                        'collected_this_run': collected_this_run,
                        'start_time': checkpoint['start_time'] if checkpoint else start_time,
                        'memory_usage_mb': client.check_memory_usage(),
                        'target_count': target_count,
                        'page_size': page_size,
                        'laws_per_page': laws_per_page
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
            
    except Exception as e:
        print(f"\n??CRITICAL ERROR: {e}")
        print(f"?”§ Finalizing collector...")
        collector.finalize()
        raise

def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    parser = argparse.ArgumentParser(
        description='êµ?šŒ ë²•ë¥ ?•ë³´?œìŠ¤??ë²•ë¥  ?˜ì§‘ (ìµœì ??ë²„ì „)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect_laws_optimized.py --sample 10                    # ?˜í”Œ 10ê°??˜ì§‘
  python collect_laws_optimized.py --sample 100                   # ?˜í”Œ 100ê°??˜ì§‘
  python collect_laws_optimized.py --sample 100 --start-page 5     # 5?˜ì´ì§€ë¶€??100ê°??˜ì§‘
  python collect_laws_optimized.py --sample 50 --start-page 10     # 10?˜ì´ì§€ë¶€??50ê°??˜ì§‘
  python collect_laws_optimized.py --full                          # ?„ì²´ 7602ê°??˜ì§‘
  python collect_laws_optimized.py --resume                        # ì¤‘ë‹¨ ì§€?ì—???¬ê°œ
  python collect_laws_optimized.py --sample 100 --laws-per-page 10   # ?˜ì´ì§€??10ê°œë¡œ ?˜ì§‘
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
    parser.add_argument('--laws-per-page', type=int, default=10,
                        help='?˜ì´ì§€??ë²•ë¥  ??(ê¸°ë³¸: 10)')
    parser.add_argument('--enable-metrics', action='store_true', default=True,
                        help='ë©”íŠ¸ë¦??˜ì§‘ ?œì„±??(ê¸°ë³¸ê°?')
    parser.add_argument('--disable-metrics', dest='enable_metrics', action='store_false',
                        help='ë©”íŠ¸ë¦??˜ì§‘ ë¹„í™œ?±í™”')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='ë¡œê·¸ ?ˆë²¨ (ê¸°ë³¸: INFO)')
    
    args = parser.parse_args()
    
    # ë¡œê·¸ ?ˆë²¨ ?¬ì„¤??
    if args.log_level != 'INFO':
        logger.setLevel(getattr(logging, args.log_level))
    
    if args.sample:
        print(f"?“¦ Sample mode: {args.sample} items")
        collect_laws_optimized(
            target_count=args.sample,
            page_size=args.page_size,
            resume=args.resume,
            start_page=args.start_page,
            laws_per_page=args.laws_per_page,
            enable_metrics=args.enable_metrics
        )
    elif args.full:
        logger.info(f"?“¦ Full mode: 7602 items")
        collect_laws_optimized(
            target_count=None,
            page_size=args.page_size,
            resume=args.resume,
            start_page=args.start_page,
            laws_per_page=args.laws_per_page,
            enable_metrics=args.enable_metrics
        )
    else:
        parser.print_help()
        logger.error("\n??Please specify --sample N or --full")
        sys.exit(1)

if __name__ == "__main__":
    main()
