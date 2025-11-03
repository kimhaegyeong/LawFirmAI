#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?¤í¬ë¦½íŠ¸ (ë¦¬íŒ©? ë§??ë²„ì „)

êµ??ë²•ë ¹?•ë³´?¼í„° LAW OPEN APIë¥??¬ìš©?˜ì—¬ ?Œì¬ê²°ì •ë¡€ë¥??˜ì§‘?©ë‹ˆ??
- ìµœê·¼ 5?„ê°„ ?Œì¬ê²°ì •ë¡€ 1,000ê±??˜ì§‘
- ?Œë²•?¬íŒ??ê²°ì •ë¡€???ì„¸ ?´ìš© ?˜ì§‘
- ê²°ì •? í˜•ë³?ë¶„ë¥˜ (?„í—Œ, ?©í—Œ, ê°í•˜, ê¸°ê° ??
- ?¥ìƒ???ëŸ¬ ì²˜ë¦¬, ?±ëŠ¥ ìµœì ?? ëª¨ë‹ˆ?°ë§ ê¸°ëŠ¥
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIConfig, load_env_file
from scripts.constitutional_decision.constitutional_collector import ConstitutionalDecisionCollector
from scripts.constitutional_decision.constitutional_logger import setup_logging

logger = setup_logging()


def check_progress():
    """ì¤‘ë‹¨??ì§€???•ì¸ ? í‹¸ë¦¬í‹° ?¨ìˆ˜"""
    try:
        output_dir = Path("data/raw/constitutional_decisions")
        
        if not output_dir.exists():
            print("???˜ì§‘???°ì´?°ê? ?†ìŠµ?ˆë‹¤.")
            return
        
        print("=" * 60)
        print("?“Š ?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ì§„í–‰ ?í™© ?•ì¸")
        print("=" * 60)
        
        # ì²´í¬?¬ì¸???Œì¼ ?•ì¸
        checkpoint_files = list(output_dir.glob("collection_checkpoint_*.json"))
        
        if not checkpoint_files:
            print("? ï¸ ì²´í¬?¬ì¸???Œì¼???†ìŠµ?ˆë‹¤.")
            # ê¸°ì¡´ ?˜ì§‘???Œì¼???•ì¸
            batch_files = list(output_dir.glob("batch_*.json"))
            if batch_files:
                print(f"?“ ?˜ì§‘??ë°°ì¹˜ ?Œì¼: {len(batch_files)}ê°?)
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
        print("?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ì§„í–‰ ?í™© ?•ì¸")
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
            
            # ê²°ì •? í˜•ë³??µê³„
            decision_type_stats = {}
            for file_path in batch_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    category = data.get('metadata', {}).get('category', 'unknown')
                    count = data.get('metadata', {}).get('count', 0)
                    decision_type_stats[category] = decision_type_stats.get(category, 0) + count
                except:
                    pass
            
            if decision_type_stats:
                print("\nê²°ì •? í˜•ë³??˜ì§‘ ?„í™©:")
                for decision_type, count in sorted(decision_type_stats.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {decision_type}: {count:,}ê±?)
        
        print("\n?¬ì‹œ?‘í•˜?¤ë©´ ?¤ìŒ ëª…ë ¹???¤í–‰?˜ì„¸??")
        print("LAW_OPEN_API_OC=your_email_id python collect_constitutional_decisions.py")
        
    except Exception as e:
        print(f"ì§„í–‰ ?í™© ?•ì¸ ì¤??¤ë¥˜ ë°œìƒ: {e}")
        print(traceback.format_exc())


def main():
    """ë©”ì¸ ?¨ìˆ˜ (ë¦¬íŒ©? ë§??ë²„ì „)"""
    try:
        # ?˜ê²½ë³€???Œì¼ ë¡œë”©
        load_env_file()
        
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
            logger.info("?¬ìš©ë²? LAW_OPEN_API_OC=your_email_id python collect_constitutional_decisions.py")
            return 1
        
        # ë¡œê·¸ ?”ë ‰? ë¦¬ ?ì„±
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # API ?¤ì •
        config = LawOpenAPIConfig(oc=oc)
        
        # ?Œì¬ê²°ì •ë¡€ ?˜ì§‘ê¸??ì„±
        collector = ConstitutionalDecisionCollector(config)
        
        # ëª…ë ¹???¸ìˆ˜ ì²˜ë¦¬
        if len(sys.argv) > 1:
            if sys.argv[1] == "--check" or sys.argv[1] == "-c":
                # ì§„í–‰ ?í™© ?•ì¸ ëª¨ë“œ
                check_progress()
                return 0
            elif sys.argv[1] == "--monitor" or sys.argv[1] == "-m":
                # ?¤ì‹œê°?ëª¨ë‹ˆ?°ë§ ëª¨ë“œ
                from scripts.constitutional_decision.monitor_progress import monitor_progress
                monitor_progress()
                return 0
            elif sys.argv[1] == "--no-keyword" or sys.argv[1] == "-n":
                # ?¤ì›Œ???†ì´ ?„ì²´ ?°ì´???˜ì§‘ ëª¨ë“œ
                logger.info("?” ?¤ì›Œ???†ì´ ?„ì²´ ?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ëª¨ë“œ")
                logger.info("?“Š ?¤ì‹œê°?ëª¨ë‹ˆ?°ë§???„í•´ ë³„ë„ ?°ë??ì—???¤ìŒ ëª…ë ¹???¤í–‰?˜ì„¸??")
                logger.info("   python scripts/constitutional_decision/collect_constitutional_decisions.py --monitor")
                collector.collect_all_decisions(target_count=1000, keyword_mode=False)
                logger.info("?Œì¬ê²°ì •ë¡€ ?˜ì§‘???±ê³µ?ìœ¼ë¡??„ë£Œ?˜ì—ˆ?µë‹ˆ??")
                return 0
            elif sys.argv[1] == "--date-based" or sys.argv[1] == "-d":
                # ? ì§œ ê¸°ë°˜ ?˜ì§‘ ëª¨ë“œ
                logger.info("?—“ï¸?? ì§œ ê¸°ë°˜ ?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ëª¨ë“œ")
                logger.info("?“‹ ?¬ìš©ë²?")
                logger.info("   python scripts/constitutional_decision/collect_by_date.py --strategy yearly --year 2024 --unlimited")
                logger.info("   python scripts/constitutional_decision/collect_by_date.py --strategy quarterly --year 2024 --quarter 4 --target 500")
                logger.info("   python scripts/constitutional_decision/collect_by_date.py --strategy monthly --year 2024 --month 12 --target 200")
                logger.info("   python scripts/constitutional_decision/collect_by_date.py --check  # ê¸°ì¡´ ?°ì´???•ì¸")
                return 0
            elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
                # ?„ì?ë§?ì¶œë ¥
                print("?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?¤í¬ë¦½íŠ¸ ?¬ìš©ë²?(?°ì„ ?œìœ„ ê¸°ë°˜):")
                print("  python collect_constitutional_decisions.py [?µì…˜] [ëª©í‘œ?˜ëŸ‰]")
                print("")
                print("?µì…˜:")
                print("  --check, -c     ì§„í–‰ ?í™© ?•ì¸")
                print("  --monitor, -m   ?¤ì‹œê°?ëª¨ë‹ˆ?°ë§")
                print("  --no-keyword, -n  ?¤ì›Œ???†ì´ ?„ì²´ ?°ì´???˜ì§‘")
                print("  --date-based, -d  ? ì§œ ê¸°ë°˜ ?˜ì§‘ ëª¨ë“œ ?ˆë‚´")
                print("  --help, -h      ?„ì?ë§?ì¶œë ¥")
                print("")
                print("?˜ì§‘ ë°©ì‹:")
                print("  - ?°ì„ ?œìœ„ ?¤ì›Œ???°ì„  ?˜ì§‘ (?Œë²•?Œì›, ?„í—Œë²•ë¥ ?¬íŒ ??")
                print("  - ?¤ì›Œ?œë³„ ì°¨ë“± ëª©í‘œ ê±´ìˆ˜ (ìµœê³ ?°ì„ : 100ê±? ê³ ìš°?? 50ê±? ì¤‘ìš°?? 30ê±?")
                print("  - ì´?50ê°??´ìƒ ?¤ì›Œ?œë¡œ ì²´ê³„???˜ì§‘")
                print("  - ?´ë? ì²˜ë¦¬???¤ì›Œ?œëŠ” ?ë™?¼ë¡œ ê±´ë„ˆ?°ê¸°")
                print("")
                print("?ˆì‹œ:")
                print("  python collect_constitutional_decisions.py              # ê¸°ë³¸ 1,000ê±??˜ì§‘ (?¤ì›Œ??ê¸°ë°˜)")
                print("  python collect_constitutional_decisions.py 2000        # 2,000ê±??˜ì§‘ (?¤ì›Œ??ê¸°ë°˜)")
                print("  python collect_constitutional_decisions.py --no-keyword # ?¤ì›Œ???†ì´ ?„ì²´ ?°ì´???˜ì§‘")
                print("  python collect_constitutional_decisions.py --date-based # ? ì§œ ê¸°ë°˜ ?˜ì§‘ ëª¨ë“œ ?ˆë‚´")
                print("  python collect_constitutional_decisions.py --check     # ì§„í–‰ ?í™© ?•ì¸")
                print("  python collect_constitutional_decisions.py --monitor   # ?¤ì‹œê°?ëª¨ë‹ˆ?°ë§")
                print("")
                print("? ì§œ ê¸°ë°˜ ?˜ì§‘ ?ˆì‹œ:")
                print("  python collect_by_date.py --strategy yearly --year 2024 --unlimited")
                print("  python collect_by_date.py --strategy quarterly --year 2024 --quarter 4 --target 500")
                print("  python collect_by_date.py --strategy monthly --year 2024 --month 12 --target 200")
                return 0
            else:
                try:
                    target_count = int(sys.argv[1])
                    logger.info(f"ëª…ë ¹???¸ìˆ˜ë¡?ëª©í‘œ ?˜ëŸ‰ ?¤ì •: {target_count}ê±?)
                except ValueError:
                    logger.warning(f"?˜ëª»??ëª©í‘œ ?˜ëŸ‰: {sys.argv[1]}, ê¸°ë³¸ê°??¬ìš©: 1000ê±?)
                    target_count = 1000
        else:
            target_count = 1000
        
        # ?˜ì§‘ ?¤í–‰ (?¤ì›Œ??ëª¨ë“œ ê¸°ë³¸ê°?
        collector.collect_all_decisions(target_count=target_count, keyword_mode=True)
        
        logger.info("?Œì¬ê²°ì •ë¡€ ?˜ì§‘???±ê³µ?ìœ¼ë¡??„ë£Œ?˜ì—ˆ?µë‹ˆ??")
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
