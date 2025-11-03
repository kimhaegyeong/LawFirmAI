#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
êµ?šŒ ë²•ë¥ ?•ë³´?œìŠ¤??ë²•ë¥ ë§??˜ì§‘ (?¹ì • URL??

?œê³µ??URL??ë²•ë¥ ë§??˜ì§‘?˜ëŠ” ?¤í¬ë¦½íŠ¸:
https://likms.assembly.go.kr/law/lawsLawtInqyList2020.do?genActiontypeCd=2ACT1010&genMenuId=menu_serv_nlaw_lawt_1020&uid=R310CV1620579091049F522&genDoctreattypeCd=DOCT2041&topicCd=PJJG00_000&pageSize=100&srchNm=&srchType=contNm&orderType=&orderObj=&srchDtType=promDt&srchStaDt=&srchEndDt=

ì´?1,895ê±´ì˜ ë²•ë¥  ?˜ì§‘

?¬ìš©ë²?
  python collect_laws_only.py --sample 10     # ?˜í”Œ 10ê°?
  python collect_laws_only.py --sample 100    # ?˜í”Œ 100ê°?
  python collect_laws_only.py --full          # ?„ì²´ 1895ê°?
  python collect_laws_only.py --resume        # ì¤‘ë‹¨ ì§€?ì—???¬ê°œ
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

# ë¡œê±° ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/law_collection_only.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Graceful shutdown ì²˜ë¦¬
interrupted = False

def signal_handler(sig, frame):
    """?œê·¸???¸ë“¤??(Ctrl+C ??"""
    global interrupted
    logger.warning("\nInterrupt signal received. Saving progress...")
    interrupted = True

# ?œê·¸???¸ë“¤???±ë¡
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class LawOnlyCollector:
    """ë²•ë¥ ë§??˜ì§‘?˜ëŠ” ?´ë˜??""
    
    def __init__(self, base_dir: str = "data/raw/assembly"):
        self.base_dir = Path(base_dir)
        self.collected_items = []
        self.failed_items = []
        self.start_time = None
        
        # ì¶œë ¥ ?”ë ‰? ë¦¬ ?ì„±
        self.output_dir = self.base_dir / "law_only" / datetime.now().strftime("%Y%m%d")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
        
        # ?¹ì • URL ?¤ì • (ë²•ë¥ ë§??„í„°ë§ëœ URL)
        self.base_url = "https://likms.assembly.go.kr/law/lawsLawtInqyList2020.do"
        self.url_params = {
            'genActiontypeCd': '2ACT1010',
            'genMenuId': 'menu_serv_nlaw_lawt_1020',
            'uid': 'R310CV1620579091049F522',
            'genDoctreattypeCd': 'DOCT2041',  # ë²•ë¥ ë§??„í„°ë§?
            'topicCd': 'PJJG00_000',
            'pageSize': '100',
            'srchNm': '',
            'srchType': 'contNm',
            'orderType': '',
            'orderObj': '',
            'srchDtType': 'promDt',
            'srchStaDt': '',
            'srchEndDt': ''
        }
        
        logger.info(f"Target URL: {self.base_url}")
        logger.info(f"Expected total: 1,895 laws")
    
    def build_url(self, page_num: int = 1) -> str:
        """?˜ì´ì§€ ë²ˆí˜¸???°ë¥¸ URL êµ¬ì„±"""
        params = self.url_params.copy()
        params['pageNum'] = str(page_num)
        
        param_str = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{self.base_url}?{param_str}"
    
    def collect_laws(self, target_count: int = None, page_size: int = 100, resume: bool = True, start_page: int = 1):
        """ë²•ë¥  ?˜ì§‘ ë©”ì¸ ?¨ìˆ˜"""
        
        print(f"\n{'='*60}")
        print(f"LAW ONLY COLLECTION STARTED")
        print(f"Target: Laws only (1,895 total)")
        print(f"{'='*60}")
        
        # ì²´í¬?¬ì¸??ë§¤ë‹ˆ?€
        checkpoint_mgr = CheckpointManager("data/checkpoints/laws_only")
        print(f"Checkpoint directory: data/checkpoints/laws_only")
        
        # ì²´í¬?¬ì¸??ë¡œë“œ
        actual_start_page = start_page
        checkpoint = None
        
        if resume:
            print(f"Checking for existing checkpoint...")
            checkpoint = checkpoint_mgr.load_checkpoint()
            if checkpoint:
                print(f"Resuming from checkpoint")
                print(f"   Page: {checkpoint.get('current_page', 0)}")
                print(f"   Collected: {checkpoint.get('collected_count', 0)} items")
                actual_start_page = checkpoint['current_page'] + 1
            else:
                print(f"No checkpoint found, starting from page {start_page}")
        else:
            print(f"Resume disabled, starting from page {start_page}")
        
        # ?˜ì§‘ê¸?ì´ˆê¸°??
        print(f"\nInitializing collector...")
        collector = AssemblyCollector(
            base_dir=str(self.base_dir),
            data_type="law_only",
            batch_size=20,
            memory_limit_mb=600
        )
        
        self.start_time = datetime.now()
        
        try:
            with AssemblyPlaywrightClient(rate_limit=2.0, timeout=30000, headless=True) as client:
                print(f"Browser started successfully")
                
                page_num = actual_start_page
                collected_count = checkpoint.get('collected_count', 0) if checkpoint else 0
                
                while True:
                    if interrupted:
                        print(f"\nCollection interrupted by user")
                        break
                    
                    if target_count and collected_count >= target_count:
                        print(f"\nTarget count reached: {collected_count}/{target_count}")
                        break
                    
                    print(f"\nProcessing page {page_num}...")
                    
                    try:
                        # ?˜ì´ì§€ ?•ë³´ ?¤ì • (page_numberë¥??„í•´ ?„ìš”)
                        collector.set_page_info(page_num)
                        
                        # ?¹ì • URLë¡??˜ì´ì§€ ?´ë™
                        url = self.build_url(page_num)
                        print(f"URL: {url}")
                        
                        # ?˜ì´ì§€ ë¡œë“œ
                        client.page.goto(url, wait_until='domcontentloaded')
                        client.page.wait_for_timeout(3000)  # 3ì´??€ê¸?
                        
                        # ë²•ë¥  ëª©ë¡ ?Œì‹±
                        laws = client._parse_law_table()
                        
                        if not laws:
                            print(f"No laws found on page {page_num}, stopping...")
                            break
                        
                        print(f"Found {len(laws)} laws on page {page_num}")
                        
                        # ë²•ë¥  ?ì„¸ ?•ë³´ ?˜ì§‘
                        for i, law in enumerate(laws):
                            if interrupted:
                                break
                            
                            if target_count and collected_count >= target_count:
                                break
                            
                            try:
                                print(f"   Collecting law {collected_count + 1}: {law['law_name'][:50]}...")
                                
                                # ë²•ë¥  ?ì„¸ ?•ë³´ ?˜ì§‘
                                detail = client.get_law_detail(law['cont_id'], law['cont_sid'])
                                
                                if detail:
                                    # ?˜ì§‘ê¸°ì— ì¶”ê?
                                    collector.save_item(detail)
                                    collected_count += 1
                                    
                                    print(f"   Collected: {detail['law_name'][:50]}...")
                                else:
                                    print(f"   Failed to collect: {law['law_name'][:50]}...")
                                    self.failed_items.append(law)
                                
                                # ë°°ì¹˜ ?€??
                                if collected_count % collector.batch_size == 0:
                                    collector._save_batch()
                                    print(f"   Batch saved: {collected_count} items")
                                
                            except Exception as e:
                                print(f"   Error collecting law: {e}")
                                self.failed_items.append(law)
                                continue
                        
                        # ?˜ì´ì§€ ?„ë£Œ ??ì²´í¬?¬ì¸???€??
                        checkpoint_data = {
                            'data_type': 'law_only',
                            'current_page': page_num,
                            'collected_count': collected_count,
                            'failed_count': len(self.failed_items),
                            'timestamp': datetime.now().isoformat(),
                            'target_count': target_count
                        }
                        checkpoint_mgr.save_checkpoint(checkpoint_data)
                        
                        print(f"Page {page_num} completed: {collected_count} total items")
                        
                        page_num += 1
                        
                    except Exception as e:
                        print(f"Error processing page {page_num}: {e}")
                        logger.error(f"Page {page_num} error: {e}")
                        break
                
                # ìµœì¢… ë°°ì¹˜ ?€??
                collector._save_batch()
                
        except Exception as e:
            print(f"Collection failed: {e}")
            logger.error(f"Collection failed: {e}")
            return False
        
        # ìµœì¢… ?µê³„
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print(f"\n{'='*60}")
        print(f"LAW COLLECTION COMPLETED!")
        print(f"{'='*60}")
        print(f"Final Statistics:")
        print(f"   Total collected: {collected_count} laws")
        print(f"   Failed items: {len(self.failed_items)}")
        print(f"   Duration: {duration}")
        print(f"   Output directory: {self.output_dir}")
        print(f"{'='*60}")
        
        return True

def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    parser = argparse.ArgumentParser(
        description='êµ?šŒ ë²•ë¥ ?•ë³´?œìŠ¤??ë²•ë¥ ë§??˜ì§‘ (?¹ì • URL??',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect_laws_only.py --sample 10     # ?˜í”Œ 10ê°?
  python collect_laws_only.py --sample 100    # ?˜í”Œ 100ê°?
  python collect_laws_only.py --full          # ?„ì²´ 1895ê°?
  python collect_laws_only.py --resume        # ì¤‘ë‹¨ ì§€?ì—???¬ê°œ
        """
    )
    
    parser.add_argument('--sample', type=int, metavar='N',
                       help='?˜í”Œ ?˜ì§‘ ê°œìˆ˜ (10, 100, 1000 ??')
    parser.add_argument('--full', action='store_true',
                       help='?„ì²´ ?˜ì§‘ (1895ê°?')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='ì²´í¬?¬ì¸?¸ì—???¬ê°œ (ê¸°ë³¸ê°?')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                       help='ì²˜ìŒë¶€???œì‘')
    parser.add_argument('--start-page', type=int, default=1,
                       help='?œì‘ ?˜ì´ì§€ ë²ˆí˜¸ (ê¸°ë³¸: 1)')
    parser.add_argument('--page-size', type=int, default=100,
                       help='?˜ì´ì§€????ª© ??(ê¸°ë³¸: 100)')
    
    args = parser.parse_args()
    
    # ëª©í‘œ ?˜ì§‘ ê°œìˆ˜ ê²°ì •
    if args.full:
        target_count = 1895  # ?„ì²´ ë²•ë¥  ??
        print(f"Full collection mode: {target_count} laws")
    elif args.sample:
        target_count = args.sample
        print(f"Sample mode: {target_count} items")
    else:
        target_count = 100  # ê¸°ë³¸ê°?
        print(f"Default mode: {target_count} items")
    
    # ?˜ì§‘ê¸??ì„± ë°??¤í–‰
    collector = LawOnlyCollector()
    
    try:
        success = collector.collect_laws(
            target_count=target_count,
            page_size=args.page_size,
            resume=args.resume,
            start_page=args.start_page
        )
        
        if success:
            print(f"\nCollection completed successfully!")
            return 0
        else:
            print(f"\nCollection failed!")
            return 1
            
    except KeyboardInterrupt:
        print(f"\nCollection interrupted by user")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
