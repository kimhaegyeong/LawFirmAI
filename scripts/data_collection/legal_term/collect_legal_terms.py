#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ë²•ë¥  ?©ì–´ ?˜ì§‘ ?¤í¬ë¦½íŠ¸

êµ??ë²•ë ¹?•ë³´?¼í„° OpenAPIë¥??œìš©?˜ì—¬ ë²•ë¥  ?©ì–´ë¥??˜ì§‘?©ë‹ˆ??

?¬ìš©ë²?
1. ?˜ê²½ë³€???¤ì •:
   export LAW_OPEN_API_OC="your_email@example.com"

2. ?¤í¬ë¦½íŠ¸ ?¤í–‰:
   python scripts/legal_term/collect_legal_terms.py

ê¸°ëŠ¥:
- ë²•ë¥  ?©ì–´ ?˜ì§‘ ë°??€??
- ì²´í¬?¬ì¸?¸ë? ?µí•œ ì¤‘ë‹¨/?¬ê°œ ì§€??
"""

import os
import sys
import logging
import argparse
import signal
import time
from pathlib import Path
from datetime import datetime

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# ?„ì¬ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê? (?ë? importë¥??„í•´)
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# python-dotenvë¥??¬ìš©?˜ì—¬ .env ?Œì¼ ë¡œë“œ
try:
    from dotenv import load_dotenv
    # ?„ë¡œ?íŠ¸ ë£¨íŠ¸?ì„œ .env ?Œì¼ ë¡œë“œ
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[OK] ?˜ê²½ë³€??ë¡œë“œ ?„ë£Œ: {env_path}")
    else:
        print(f"[WARN] .env ?Œì¼??ì°¾ì„ ???†ìŠµ?ˆë‹¤: {env_path}")
        # ?„ì¬ ?”ë ‰? ë¦¬?ì„œ???œë„
        current_env = Path('.env')
        if current_env.exists():
            load_dotenv(current_env)
            print(f"[OK] ?„ì¬ ?”ë ‰? ë¦¬?ì„œ ?˜ê²½ë³€??ë¡œë“œ ?„ë£Œ: {current_env.absolute()}")
except ImportError:
    print("[ERROR] python-dotenvê°€ ?¤ì¹˜?˜ì? ?Šì•˜?µë‹ˆ?? pip install python-dotenvë¡??¤ì¹˜?˜ì„¸??")

from term_collector import LegalTermCollector

# ë¡œê¹… ?¤ì • (ê°„ëµ??ë²„ì „, UTF-8 ?¸ì½”??
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/legal_term_collection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def collect_legal_terms(max_terms: int = None, collection_type: str = "all", use_mock_data: bool = False, 
                       resume: bool = True, memory_limit_mb: int = None, target_year: int = None) -> bool:
    """ë²•ë¥  ?©ì–´ ?˜ì§‘ ë°??¬ì „ êµ¬ì¶• (ë©”ëª¨ë¦?ìµœì ??ë°?ì²´í¬?¬ì¸??ì§€??"""
    if max_terms is None:
        logger.info(f"ë²•ë¥  ?©ì–´ ?˜ì§‘ ?œì‘ - ?€?? {collection_type}, ëª¨ë“œ: ë¬´ì œ??(ë©”ëª¨ë¦?ìµœì ??")
    else:
        logger.info(f"ë²•ë¥  ?©ì–´ ?˜ì§‘ ?œì‘ - ?€?? {collection_type}, ìµœë?: {max_terms}ê°?(ë©”ëª¨ë¦?ìµœì ??")
    
    try:
        # 1. ë©”ëª¨ë¦??¤ì • ?ì„±
        from term_collector import MemoryConfig
        memory_config = MemoryConfig()
        
        if memory_limit_mb:
            memory_config.max_memory_mb = memory_limit_mb
        
        # 2. ?˜ì§‘ê¸?ì´ˆê¸°??(ë©”ëª¨ë¦?ìµœì ??
        collector = LegalTermCollector(memory_config=memory_config)
        
        # 3. ë©”ëª¨ë¦??¤ì • ìµœì ??
        collector.optimize_memory_settings()
        
        # 4. ì§„í–‰ ?íƒœ ?•ì¸
        progress_status = collector.get_progress_status()
        if progress_status['status'] != 'not_started':
            logger.info(f"ê¸°ì¡´ ?˜ì§‘ ?¸ì…˜ ë°œê²¬ - ì§„í–‰ë¥? {progress_status['progress_percent']:.1f}%")
        
        # 5. ?˜ì§‘ ?€?…ì— ?°ë¥¸ ?©ì–´ ?˜ì§‘
        if collection_type == "all":
            success = collector.collect_all_terms(max_terms, use_mock_data, resume)
        elif collection_type == "categories":
            categories = ["ë¯¼ì‚¬ë²?, "?•ì‚¬ë²?, "?ì‚¬ë²?, "?¸ë™ë²?, "?‰ì •ë²?, "?˜ê²½ë²?, "?Œë¹„?ë²•", "ì§€?ì¬?°ê¶Œë²?, "ê¸ˆìœµë²?]
            success = collector.collect_terms_by_categories(categories, max_terms // len(categories), resume)
        elif collection_type == "keywords":
            keywords = ["ê³„ì•½", "?í•´ë°°ìƒ", "ë¶ˆë²•?‰ìœ„", "ì±„ê¶Œ", "ì±„ë¬´", "?Œë©¸?œíš¨", "ì·¨ì†Œê¶?, "?´ì œê¶?, "?€ë¦¬ê¶Œ", "?€?œê¶Œ"]
            success = collector.collect_terms_by_keywords(keywords, max_terms // len(keywords), resume)
        elif collection_type == "year":
            if not target_year:
                logger.error("?°ë„ë³??˜ì§‘???„í•´?œëŠ” --target-year ?µì…˜???„ìš”?©ë‹ˆ??")
                return False
            success = collector.collect_terms_by_year(target_year, max_terms, resume)
        else:
            logger.error(f"ì§€?í•˜ì§€ ?ŠëŠ” ?˜ì§‘ ?€?? {collection_type}")
            return False
        
        if not success:
            logger.error("?©ì–´ ?˜ì§‘ ?¤íŒ¨")
            return False
        
        
        # 7. ?¬ì „ ?€??
        collector.save_dictionary()
        
        
        
        
        logger.info("ë²•ë¥  ?©ì–´ ?˜ì§‘ ?„ë£Œ")
        return True
        
    except Exception as e:
        logger.error(f"ë²•ë¥  ?©ì–´ ?˜ì§‘ ?¤íŒ¨: {e}")
        return False








def main():
    """ë©”ì¸ ?¨ìˆ˜ (ë©”ëª¨ë¦?ìµœì ??ë°?ì²´í¬?¬ì¸??ì§€??"""
    # Graceful shutdown ?¤ì •
    shutdown_requested = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        signal_name = signal.Signals(signum).name
        logger.info(f"ë©”ì¸ ?„ë¡œ?¸ìŠ¤?ì„œ ?œê·¸??{signal_name}({signum}) ?˜ì‹  - graceful shutdown ?œì‘")
        shutdown_requested = True
    
    # ?œê·¸???¸ë“¤???±ë¡
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # ëª…ë ¹???¸ìˆ˜ ?Œì‹±
    parser = argparse.ArgumentParser(description='ë²•ë¥  ?©ì–´ ?˜ì§‘ ?¤í¬ë¦½íŠ¸ (ë©”ëª¨ë¦?ìµœì ??ë°?ì²´í¬?¬ì¸??ì§€??')
    parser.add_argument('--max-terms', type=int, default=None, help='ìµœë? ?˜ì§‘ ?©ì–´ ??(ê¸°ë³¸ê°? ë¬´ì œ?? 0 ?ëŠ” ?Œìˆ˜ ?…ë ¥ ??ë¬´ì œ??ëª¨ë“œ)')
    parser.add_argument('--collection-type', choices=['all', 'categories', 'keywords', 'year'], default='all', 
                       help='?˜ì§‘ ?€??(all: ?„ì²´, categories: ì¹´í…Œê³ ë¦¬ë³? keywords: ?¤ì›Œ?œë³„, year: ì§€?•ì—°??')
    parser.add_argument('--use-mock-data', action='store_true', help='API ?€??ëª¨ì˜ ?°ì´???¬ìš©')
    parser.add_argument('--no-resume', action='store_true', help='ì²´í¬?¬ì¸??ë¬´ì‹œ?˜ê³  ?ˆë¡œ ?œì‘')
    parser.add_argument('--memory-limit', type=int, help='ë©”ëª¨ë¦??¬ìš©???œí•œ (MB)')
    parser.add_argument('--clear-checkpoint', action='store_true', help='ê¸°ì¡´ ì²´í¬?¬ì¸???? œ')
    parser.add_argument('--clear-dictionary', action='store_true', help='ê¸°ì¡´ ?¬ì „ ?°ì´???„ì „ ?? œ ???ˆë¡œ ?œì‘')
    parser.add_argument('--status', action='store_true', help='?„ì¬ ?˜ì§‘ ?íƒœ ?•ì¸')
    parser.add_argument('--target-year', type=int, help='?˜ì§‘???€???°ë„ (?? 2024)')
    parser.add_argument('--search-detail', type=str, help='?¹ì • ë²•ë ¹?©ì–´???ì„¸ì¡°íšŒ (?? --search-detail "ê³„ì•½")')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("ë²•ë¥  ?©ì–´ ?˜ì§‘ ?¤í¬ë¦½íŠ¸ ?œì‘ (ë©”ëª¨ë¦?ìµœì ??")
    logger.info("=" * 60)
    
    # ?˜ê²½ ë³€???•ì¸
    if not os.getenv("LAW_OPEN_API_OC"):
        logger.error("LAW_OPEN_API_OC ?˜ê²½ë³€?˜ê? ?¤ì •?˜ì? ?Šì•˜?µë‹ˆ??")
        logger.error("?¤ìŒê³?ê°™ì´ ?˜ê²½ë³€?˜ë? ?¤ì •?´ì£¼?¸ìš”:")
        logger.error("export LAW_OPEN_API_OC='your_email@example.com'")
        return False
    
    logger.info(f"API ???•ì¸: {os.getenv('LAW_OPEN_API_OC')}")
    logger.info(f"?˜ì§‘ ?€?? {args.collection_type}")
    if args.max_terms is None or args.max_terms <= 0:
        logger.info("ìµœë? ?©ì–´ ?? ë¬´ì œ??ëª¨ë“œ")
    else:
        logger.info(f"ìµœë? ?©ì–´ ?? {args.max_terms}")
    logger.info(f"ì²´í¬?¬ì¸???¬ê°œ: {'?„ë‹ˆ?? if args.no_resume else '??}")
    if args.memory_limit:
        logger.info(f"ë©”ëª¨ë¦??œí•œ: {args.memory_limit}MB")
    if args.target_year:
        logger.info(f"?€???°ë„: {args.target_year}??)
    
    # ë¡œê·¸ ?”ë ‰? ë¦¬ ?ì„±
    Path("logs").mkdir(exist_ok=True)
    Path("data/raw/legal_terms").mkdir(parents=True, exist_ok=True)
    
    try:
        start_time = datetime.now()
        
        # ì²´í¬?¬ì¸???? œ ?”ì²­
        if args.clear_checkpoint:
            logger.info("ê¸°ì¡´ ì²´í¬?¬ì¸???? œ ì¤?..")
            from term_collector import LegalTermCollector
            collector = LegalTermCollector()
            collector.clear_checkpoint()
            logger.info("ì²´í¬?¬ì¸???? œ ?„ë£Œ")
        
        # ?¬ì „ ?°ì´???„ì „ ?? œ ?”ì²­
        if args.clear_dictionary:
            logger.info("ê¸°ì¡´ ?¬ì „ ?°ì´???„ì „ ?? œ ì¤?..")
            
            # ê¸°ë³¸ ?Œì¼???? œ
            dictionary_path = Path("data/raw/legal_terms/legal_term_dictionary.json")
            checkpoint_path = Path("data/raw/legal_terms/checkpoint.json")
            
            if dictionary_path.exists():
                dictionary_path.unlink()
                logger.info("?¬ì „ ?Œì¼ ?? œ ?„ë£Œ")
            
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info("ì²´í¬?¬ì¸???Œì¼ ?? œ ?„ë£Œ")
            
            # ?¸ì…˜ ?´ë”???? œ
            import shutil
            session_folders = list(Path("data/raw/legal_terms").glob("session_*"))
            for session_folder in session_folders:
                if session_folder.is_dir():
                    shutil.rmtree(session_folder)
                    logger.info(f"?¸ì…˜ ?´ë” ?? œ ?„ë£Œ: {session_folder}")
            
            # ê¸°í? ?Œì¼???? œ
            other_files = list(Path("data/raw/legal_terms").glob("legal_terms_*.json"))
            other_files.extend(list(Path("data/raw/legal_terms").glob("checkpoint_*.json")))
            
            for file_path in other_files:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"?Œì¼ ?? œ ?„ë£Œ: {file_path}")
            
            logger.info("ëª¨ë“  ?°ì´???? œ ?„ë£Œ - ?ˆë¡œ ?œì‘?©ë‹ˆ??)
            return True
        
        # ?íƒœ ?•ì¸ ?”ì²­
        if args.status:
            logger.info("?„ì¬ ?˜ì§‘ ?íƒœ ?•ì¸ ì¤?..")
            from term_collector import LegalTermCollector, MemoryConfig
            memory_config = MemoryConfig()
            collector = LegalTermCollector(memory_config=memory_config)
            progress_status = collector.get_progress_status()
            memory_status = collector.get_memory_status()
            
            logger.info("=" * 40)
            logger.info("?˜ì§‘ ?íƒœ ?•ë³´")
            logger.info("=" * 40)
            logger.info(f"?íƒœ: {progress_status['status']}")
            if progress_status['status'] != 'not_started':
                logger.info(f"ì§„í–‰ë¥? {progress_status['progress_percent']:.1f}%")
                logger.info(f"?˜ì§‘???©ì–´: {progress_status['collected_count']}/{progress_status['total_target']}ê°?)
                logger.info(f"?¸ì…˜ ID: {progress_status['session_id']}")
                logger.info(f"?ëŸ¬ ë°œìƒ: {progress_status['errors_count']}??)
                logger.info(f"?¬ê°œ ê°€?? {'?? if progress_status['can_resume'] else '?„ë‹ˆ??}")
            
            logger.info("=" * 40)
            logger.info("ë©”ëª¨ë¦??íƒœ ?•ë³´")
            logger.info("=" * 40)
            logger.info(f"?„ì¬ ë©”ëª¨ë¦? {memory_status['current_memory_mb']:.1f}MB")
            logger.info(f"ìµœë? ë©”ëª¨ë¦? {memory_status['peak_memory_mb']:.1f}MB")
            logger.info(f"ë©”ëª¨ë¦??œê³„: {memory_status['memory_limit_mb']:.1f}MB")
            logger.info(f"?¬ìš©ë¥? {memory_status['memory_usage_percent']:.1f}%")
            logger.info(f"ê°€ë¹„ì? ì»¬ë ‰?? {memory_status['gc_runs']}??)
            logger.info(f"ì²´í¬?¬ì¸???€?? {memory_status['checkpoints_saved']}??)
            logger.info("=" * 40)
            return True
        
        # ?ì„¸ì¡°íšŒ ?”ì²­
        if args.search_detail:
            logger.info(f"ë²•ë ¹?©ì–´ ?ì„¸ì¡°íšŒ ?œì‘: {args.search_detail}")
            from source.data.legal_term_collection_api import LegalTermCollectionAPI
            from source.utils.config import Config
            
            config = Config()
            api_client = LegalTermCollectionAPI(config)
            
            detail_info = api_client.get_term_detail(args.search_detail)
            
            if detail_info:
                logger.info("=" * 60)
                logger.info(f"ë²•ë ¹?©ì–´ ?ì„¸ì¡°íšŒ ê²°ê³¼: {args.search_detail}")
                logger.info("=" * 60)
                
                for key, value in detail_info.items():
                    if isinstance(value, list):
                        logger.info(f"{key}: {', '.join(map(str, value))}")
                    else:
                        logger.info(f"{key}: {value}")
                
                logger.info("=" * 60)
                logger.info("ë²•ë ¹?©ì–´ ?ì„¸ì¡°íšŒ ?„ë£Œ")
                return True
            else:
                logger.warning(f"ë²•ë ¹?©ì–´ ?ì„¸ì¡°íšŒ ê²°ê³¼ ?†ìŒ: {args.search_detail}")
                return False
        
        # ë²•ë¥  ?©ì–´ ?˜ì§‘
        logger.info("ë²•ë¥  ?©ì–´ ?˜ì§‘ ?œì‘")
        resume = not args.no_resume
        
        # shutdown ?”ì²­ ?•ì¸
        if shutdown_requested:
            logger.info("ì¢…ë£Œ ?”ì²­?¼ë¡œ ?¸í•œ ?˜ì§‘ ì¤‘ë‹¨")
            return False
        
        # ë¬´ì œ??ëª¨ë“œ ì²˜ë¦¬
        max_terms = args.max_terms
        if max_terms is None or max_terms <= 0:
            max_terms = None  # ë¬´ì œ??ëª¨ë“œ
            logger.info("ë¬´ì œ??ëª¨ë“œë¡??˜ì§‘???œì‘?©ë‹ˆ??)
        
        if not collect_legal_terms(max_terms, args.collection_type, args.use_mock_data, 
                                 resume, args.memory_limit, args.target_year):
            logger.error("ë²•ë¥  ?©ì–´ ?˜ì§‘ ?¤íŒ¨")
            return False
        
        # ?˜ì§‘ ?„ë£Œ ??shutdown ?”ì²­ ?•ì¸
        if shutdown_requested:
            logger.info("?˜ì§‘ ?„ë£Œ ??ì¢…ë£Œ ?”ì²­ ?•ì¸")
            return True
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("ë²•ë¥  ?©ì–´ ?˜ì§‘ ?„ë£Œ")
        logger.info(f"ì´??Œìš” ?œê°„: {duration.total_seconds():.2f}ì´?)
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"?¤í¬ë¦½íŠ¸ ?¤í–‰ ?¤íŒ¨: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)