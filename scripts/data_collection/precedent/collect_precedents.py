#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?ë? ?˜ì§‘ ?¤í¬ë¦½íŠ¸ (ë¦¬íŒ©? ë§??ë²„ì „)

êµ??ë²•ë ¹?•ë³´?¼í„° LAW OPEN APIë¥??¬ìš©?˜ì—¬ ?ë?ë¥??˜ì§‘?©ë‹ˆ??
- ìµœê·¼ 5?„ê°„ ?ë? 5,000ê±??˜ì§‘
- ?ë?ëª?ê²€?? ë³¸ë¬¸ ê²€?? ? ì§œ ë²”ìœ„ ê²€???œìš©
- ?¬ê±´ë²ˆí˜¸, ë²•ì›, ?ê²°????ë©”í??°ì´???•ì œ
- ?ë? ì¹´í…Œê³ ë¦¬ ë¶„ë¥˜ (ë¯¼ì‚¬, ?•ì‚¬, ?‰ì •, ?ì‚¬ ??
- ?¥ìƒ???ëŸ¬ ì²˜ë¦¬, ?±ëŠ¥ ìµœì ?? ëª¨ë‹ˆ?°ë§ ê¸°ëŠ¥
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIConfig
from scripts.precedent.precedent_collector import PrecedentCollector
from scripts.precedent.precedent_logger import setup_logging

logger = setup_logging()


def check_progress():
    """ì¤‘ë‹¨??ì§€???•ì¸ ? í‹¸ë¦¬í‹° ?¨ìˆ˜"""
    try:
        output_dir = Path("data/raw/precedents")
        
        if not output_dir.exists():
            print("?˜ì§‘???°ì´?°ê? ?†ìŠµ?ˆë‹¤.")
            return
        
        # ì²´í¬?¬ì¸???Œì¼ ?•ì¸
        checkpoint_files = list(output_dir.glob("collection_checkpoint_*.json"))
        
        if not checkpoint_files:
            print("ì²´í¬?¬ì¸???Œì¼???†ìŠµ?ˆë‹¤.")
            # ê¸°ì¡´ ?˜ì§‘???Œì¼???•ì¸
            batch_files = list(output_dir.glob("batch_*.json"))
            if batch_files:
                print(f"?˜ì§‘??ë°°ì¹˜ ?Œì¼: {len(batch_files)}ê°?)
                total_count = 0
                for file_path in batch_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            count = data.get('metadata', {}).get('count', 0)
                            total_count += count
                    except:
                        pass
                print(f"ì¶”ì • ?˜ì§‘ ê±´ìˆ˜: {total_count}ê±?)
            return
        
        # ê°€??ìµœê·¼ ì²´í¬?¬ì¸???Œì¼ ë¡œë“œ
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_checkpoint, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        stats = data['stats']
        resume_info = data['resume_info']
        shutdown_info = data.get('shutdown_info', {})
        
        print("=" * 60)
        print("?˜ì§‘ ì§„í–‰ ?í™© ?•ì¸")
        print("=" * 60)
        print(f"ì²´í¬?¬ì¸???Œì¼: {latest_checkpoint.name}")
        print(f"?˜ì§‘ ì§„í–‰ë¥? {resume_info['progress_percentage']:.1f}%")
        print(f"?˜ì§‘??ê±´ìˆ˜: {stats['collected_count']:,}ê±?)
        print(f"ëª©í‘œ ê±´ìˆ˜: {stats['target_count']:,}ê±?)
        print(f"ì¤‘ë³µ ?œì™¸ ê±´ìˆ˜: {stats['duplicate_count']:,}ê±?)
        print(f"?¤íŒ¨ ê±´ìˆ˜: {stats['failed_count']:,}ê±?)
        print(f"ì²˜ë¦¬???¤ì›Œ?? {stats['keywords_processed']:,}ê°?)
        print(f"ì´??¤ì›Œ?? {stats['total_keywords']:,}ê°?)
        print(f"ë§ˆì?ë§?ì²˜ë¦¬???¤ì›Œ?? {resume_info.get('last_keyword_processed', '?†ìŒ')}")
        print(f"API ?”ì²­ ?? {stats['api_requests_made']:,}??)
        print(f"API ?¤ë¥˜ ?? {stats['api_errors']:,}??)
        print(f"?íƒœ: {stats['status']}")
        
        # Graceful shutdown ?•ë³´
        if shutdown_info.get('graceful_shutdown_supported'):
            print(f"Graceful shutdown ì§€?? ??)
            if shutdown_info.get('shutdown_requested'):
                print(f"ì¢…ë£Œ ?”ì²­?? {shutdown_info.get('shutdown_reason', '?????†ìŒ')}")
        else:
            print(f"Graceful shutdown ì§€?? ?„ë‹ˆ??)
        
        if stats.get('start_time'):
            start_time = datetime.fromisoformat(stats['start_time'])
            print(f"?œì‘ ?œê°„: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if stats.get('end_time'):
            end_time = datetime.fromisoformat(stats['end_time'])
            print(f"ì¢…ë£Œ ?œê°„: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            duration = end_time - start_time
            print(f"?Œìš” ?œê°„: {duration}")
        
        print("=" * 60)
        
        # ?˜ì§‘??ë°°ì¹˜ ?Œì¼???•ì¸
        batch_files = list(output_dir.glob("batch_*.json"))
        if batch_files:
            print(f"?˜ì§‘??ë°°ì¹˜ ?Œì¼: {len(batch_files)}ê°?)
            
            # ì¹´í…Œê³ ë¦¬ë³??µê³„
            category_stats = {}
            for file_path in batch_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    category = data.get('metadata', {}).get('category', 'unknown')
                    count = data.get('metadata', {}).get('count', 0)
                    category_stats[category] = category_stats.get(category, 0) + count
                except:
                    pass
            
            if category_stats:
                print("\nì¹´í…Œê³ ë¦¬ë³??˜ì§‘ ?„í™©:")
                for category, count in sorted(category_stats.items()):
                    print(f"  {category}: {count:,}ê±?)
        
        print("\n?¬ì‹œ?‘í•˜?¤ë©´ ?¤ìŒ ëª…ë ¹???¤í–‰?˜ì„¸??")
        print("LAW_OPEN_API_OC=your_email_id python collect_precedents.py")
        
    except Exception as e:
        print(f"ì§„í–‰ ?í™© ?•ì¸ ì¤??¤ë¥˜ ë°œìƒ: {e}")
        print(traceback.format_exc())


def main():
    """ë©”ì¸ ?¨ìˆ˜ (ë¦¬íŒ©? ë§??ë²„ì „)"""
    try:
        # Windows?ì„œ UTF-8 ?˜ê²½ ?¤ì •
        if sys.platform.startswith('win'):
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            # ì½˜ì†” ì½”ë“œ?˜ì´ì§€ë¥?UTF-8ë¡??¤ì •
            try:
                import subprocess
                subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
            except:
                pass
        
        # ?˜ê²½ë³€???•ì¸
        oc = os.getenv("LAW_OPEN_API_OC")
        if not oc:
            logger.error("LAW_OPEN_API_OC ?˜ê²½ë³€?˜ê? ?¤ì •?˜ì? ?Šì•˜?µë‹ˆ??")
            logger.info("?¬ìš©ë²? LAW_OPEN_API_OC=your_email_id python collect_precedents.py")
            return 1
        
        # ë¡œê·¸ ?”ë ‰? ë¦¬ ?ì„±
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # API ?¤ì •
        config = LawOpenAPIConfig(oc=oc)
        
        # ?ë? ?˜ì§‘ ?¤í–‰
        collector = PrecedentCollector(config)
        
        # ëª…ë ¹???¸ìˆ˜ ì²˜ë¦¬
        if len(sys.argv) > 1:
            if sys.argv[1] == "--check" or sys.argv[1] == "-c":
                # ì§„í–‰ ?í™© ?•ì¸ ëª¨ë“œ
                check_progress()
                return 0
            elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
                # ?„ì?ë§?ì¶œë ¥
                print("?ë? ?˜ì§‘ ?¤í¬ë¦½íŠ¸ ?¬ìš©ë²?(?°ì„ ?œìœ„ ê¸°ë°˜):")
                print("  python collect_precedents.py [?µì…˜] [ëª©í‘œ?˜ëŸ‰]")
                print("")
                print("?µì…˜:")
                print("  --check, -c     ì§„í–‰ ?í™© ?•ì¸")
                print("  --help, -h      ?„ì?ë§?ì¶œë ¥")
                print("")
                print("?˜ì§‘ ë°©ì‹:")
                print("  - ?°ì„ ?œìœ„ ?¤ì›Œ???°ì„  ?˜ì§‘ (ê³„ì•½, ?í•´ë°°ìƒ, ?¬ê¸° ??")
                print("  - ?¤ì›Œ?œë³„ ì°¨ë“± ëª©í‘œ ê±´ìˆ˜ (?°ì„ ?œìœ„: 15-100ê±? ?¼ë°˜: 15ê±?")
                print("  - ì´?200ê°??´ìƒ ?¤ì›Œ?œë¡œ ì²´ê³„???˜ì§‘")
                print("  - ?´ë? ì²˜ë¦¬???¤ì›Œ?œëŠ” ?ë™?¼ë¡œ ê±´ë„ˆ?°ê¸°")
                print("")
                print("?ˆì‹œ:")
                print("  python collect_precedents.py              # ê¸°ë³¸ 5,000ê±??˜ì§‘")
                print("  python collect_precedents.py 10000        # 10,000ê±??˜ì§‘")
                print("  python collect_precedents.py --check      # ì§„í–‰ ?í™© ?•ì¸")
                return 0
            else:
                try:
                    target_count = int(sys.argv[1])
                    logger.info(f"ëª…ë ¹???¸ìˆ˜ë¡?ëª©í‘œ ?˜ëŸ‰ ?¤ì •: {target_count}ê±?)
                except ValueError:
                    logger.warning(f"?˜ëª»??ëª©í‘œ ?˜ëŸ‰: {sys.argv[1]}, ê¸°ë³¸ê°??¬ìš©: 5000ê±?)
                    target_count = 5000
        else:
            target_count = 5000
        
        # ?˜ì§‘ ?¤í–‰
        collector.collect_all_precedents(target_count=target_count)
        
        logger.info("?ë? ?˜ì§‘???±ê³µ?ìœ¼ë¡??„ë£Œ?˜ì—ˆ?µë‹ˆ??")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("?¬ìš©?ì— ?˜í•´ ?„ë¡œê·¸ë¨??ì¤‘ë‹¨?˜ì—ˆ?µë‹ˆ??")
        return 130
    except Exception as e:
        logger.error(f"?„ë¡œê·¸ë¨ ?¤í–‰ ì¤??¤ë¥˜ ë°œìƒ: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)