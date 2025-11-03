#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?‰ì •?¬íŒë¡€ ?˜ì§‘ê¸??´ë˜??(collect_precedents.py êµ¬ì¡° ì°¸ê³ )
"""

import json
import time
import random
import signal
import atexit
import hashlib
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from contextlib import contextmanager

import sys
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient, LawOpenAPIConfig
from scripts.administrative_appeal.administrative_appeal_models import (
    CollectionStats, AdministrativeAppealData, CollectionStatus, AppealType,
    ADMINISTRATIVE_APPEAL_KEYWORDS, PRIORITY_KEYWORDS, KEYWORD_TARGET_COUNTS, DEFAULT_KEYWORD_COUNT,
    APPEAL_TYPE_KEYWORDS, FALLBACK_KEYWORDS
)
from scripts.administrative_appeal.administrative_appeal_logger import setup_logging

logger = setup_logging()


class AdministrativeAppealCollector:
    """?‰ì •?¬íŒë¡€ ?˜ì§‘ ?´ë˜??(ê°œì„ ??ë²„ì „)"""
    
    def __init__(self, config: LawOpenAPIConfig, output_dir: Optional[Path] = None):
        """
        ?‰ì •?¬íŒë¡€ ?˜ì§‘ê¸?ì´ˆê¸°??
        
        Args:
            config: API ?¤ì • ê°ì²´
            output_dir: ì¶œë ¥ ?”ë ‰? ë¦¬ (ê¸°ë³¸ê°? data/raw/administrative_appeals)
        """
        self.client = LawOpenAPIClient(config)
        self.output_dir = output_dir or Path("data/raw/administrative_appeals")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ?°ì´??ê´€ë¦?
        self.collected_appeals: Set[str] = set()  # ì¤‘ë³µ ë°©ì?
        self.processed_keywords: Set[str] = set()  # ì²˜ë¦¬???¤ì›Œ??ì¶”ì 
        self.pending_appeals: List[AdministrativeAppealData] = []  # ?„ì‹œ ?€?¥ì†Œ
        
        # ?¤ì •
        self.batch_size = 30  # ë°°ì¹˜ ?€???¬ê¸°
        self.max_retries = 3  # ìµœë? ?¬ì‹œ???Ÿìˆ˜
        self.retry_delay = 5  # ?¬ì‹œ??ê°„ê²© (ì´?
        self.api_delay_range = (1.0, 3.0)  # API ?”ì²­ ê°?ì§€??ë²”ìœ„
        
        # ?µê³„ ë°??íƒœ
        self.stats = CollectionStats()
        self.stats.total_keywords = len(ADMINISTRATIVE_APPEAL_KEYWORDS)
        self.checkpoint_file: Optional[Path] = None
        self.resume_mode = False
        
        # ?ëŸ¬ ì²˜ë¦¬
        self.error_count = 0
        self.max_errors = 50  # ìµœë? ?ˆìš© ?ëŸ¬ ??
        
        # Graceful shutdown ê´€??
        self.shutdown_requested = False
        self.shutdown_reason = None
        self._setup_signal_handlers()
        
        # ê¸°ì¡´ ?˜ì§‘???°ì´??ë¡œë“œ
        self._load_existing_data()
        
        # ì²´í¬?¬ì¸???Œì¼ ?•ì¸ ë°?ë³µêµ¬
        self._check_and_resume_from_checkpoint()
        
        logger.info(f"?‰ì •?¬íŒë¡€ ?˜ì§‘ê¸?ì´ˆê¸°???„ë£Œ - ëª©í‘œ: {self.stats.target_count}ê±?)
    
    def _setup_signal_handlers(self):
        """?œê·¸???¸ë“¤???¤ì • (Graceful shutdown)"""
        def signal_handler(signum, frame):
            """?œê·¸???¸ë“¤??""
            signal_name = signal.Signals(signum).name
            logger.warning(f"?œê·¸??{signal_name} ({signum}) ?˜ì‹ ?? Graceful shutdown ?œì‘...")
            self.shutdown_requested = True
            self.shutdown_reason = f"Signal {signal_name} ({signum})"
        
        # SIGINT (Ctrl+C), SIGTERM ì²˜ë¦¬
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Windows?ì„œ SIGBREAK ì²˜ë¦¬
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
        
        # ?„ë¡œê·¸ë¨ ì¢…ë£Œ ???•ë¦¬ ?¨ìˆ˜ ?±ë¡
        atexit.register(self._cleanup_on_exit)
    
    def _cleanup_on_exit(self):
        """?„ë¡œê·¸ë¨ ì¢…ë£Œ ???•ë¦¬ ?‘ì—…"""
        if self.pending_appeals:
            logger.info("?„ë¡œê·¸ë¨ ì¢…ë£Œ ???„ì‹œ ?°ì´???€??ì¤?..")
            self._save_batch_appeals()
        
        if self.checkpoint_file:
            logger.info("ìµœì¢… ì²´í¬?¬ì¸???€??ì¤?..")
            self._save_checkpoint(self.checkpoint_file)
    
    def _check_shutdown_requested(self) -> bool:
        """ì¢…ë£Œ ?”ì²­ ?•ì¸"""
        return self.shutdown_requested
    
    def _request_shutdown(self, reason: str):
        """ì¢…ë£Œ ?”ì²­"""
        self.shutdown_requested = True
        self.shutdown_reason = reason
        logger.warning(f"ì¢…ë£Œ ?”ì²­?? {reason}")
    
    def _load_existing_data(self):
        """ê¸°ì¡´ ?˜ì§‘???°ì´??ë¡œë“œ?˜ì—¬ ì¤‘ë³µ ë°©ì?"""
        logger.info("ê¸°ì¡´ ?˜ì§‘???°ì´???•ì¸ ì¤?..")
        
        loaded_count = 0
        error_count = 0
        
        # ?¤ì–‘???Œì¼ ?¨í„´?ì„œ ?°ì´??ë¡œë“œ
        file_patterns = [
            "administrative_appeal_*.json",
            "batch_*.json", 
            "checkpoints/**/*.json",
            "*.json"
        ]
        
        for pattern in file_patterns:
            files = list(self.output_dir.glob(pattern))
            for file_path in files:
                try:
                    loaded_count += self._load_appeals_from_file(file_path)
                except Exception as e:
                    error_count += 1
                    logger.debug(f"?Œì¼ ë¡œë“œ ?¤íŒ¨ {file_path}: {e}")
        
        logger.info(f"ê¸°ì¡´ ?°ì´??ë¡œë“œ ?„ë£Œ: {loaded_count:,}ê±? ?¤ë¥˜: {error_count:,}ê±?)
        self.stats.collected_count = len(self.collected_appeals)
        logger.info(f"ì¤‘ë³µ ë°©ì?ë¥??„í•œ ?¬íŒë¡€ ID {len(self.collected_appeals):,}ê°?ë¡œë“œ??)
    
    def _load_appeals_from_file(self, file_path: Path) -> int:
        """?Œì¼?ì„œ ?‰ì •?¬íŒë¡€ ?°ì´??ë¡œë“œ"""
        loaded_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # ?¤ì–‘???°ì´??êµ¬ì¡° ì²˜ë¦¬
        appeals = []
        
        if isinstance(data, dict):
            if 'appeals' in data:
                appeals = data['appeals']
            elif 'basic_info' in data:
                appeals = [data]
            elif 'by_category' in data:
                for category_data in data['by_category'].values():
                    appeals.extend(category_data)
        elif isinstance(data, list):
            appeals = data
        
        # ?¬íŒë¡€ ID ì¶”ì¶œ
        for appeal in appeals:
            if isinstance(appeal, dict):
                appeal_id = appeal.get('?ë??¼ë ¨ë²ˆí˜¸') or appeal.get('appeal_id')
                if appeal_id:
                    self.collected_appeals.add(str(appeal_id))
                    loaded_count += 1
        
        return loaded_count
    
    def _is_duplicate_appeal(self, appeal: Dict[str, Any]) -> bool:
        """?‰ì •?¬íŒë¡€ ì¤‘ë³µ ?¬ë? ?•ì¸"""
        appeal_id = appeal.get('?ë??¼ë ¨ë²ˆí˜¸') or appeal.get('appeal_id')
        
        if appeal_id and str(appeal_id).strip() != '':
            if str(appeal_id) in self.collected_appeals:
                logger.debug(f"?¬íŒë¡€?¼ë ¨ë²ˆí˜¸ë¡?ì¤‘ë³µ ?•ì¸: {appeal_id}")
                return True
        
        return False
    
    def _mark_appeal_collected(self, appeal: Dict[str, Any]):
        """?‰ì •?¬íŒë¡€ë¥??˜ì§‘?¨ìœ¼ë¡??œì‹œ"""
        appeal_id = appeal.get('?ë??¼ë ¨ë²ˆí˜¸') or appeal.get('appeal_id')
        
        if appeal_id and str(appeal_id).strip() != '':
            self.collected_appeals.add(str(appeal_id))
            logger.debug(f"?¬íŒë¡€?¼ë ¨ë²ˆí˜¸ë¡??€?? {appeal_id}")
    
    def _validate_appeal_data(self, appeal: Dict[str, Any]) -> bool:
        """?‰ì •?¬íŒë¡€ ?°ì´??ê²€ì¦?""
        appeal_id = appeal.get('?ë??¼ë ¨ë²ˆí˜¸') or appeal.get('appeal_id')
        case_name = appeal.get('?¬ê±´ëª?)
        
        if not appeal_id or str(appeal_id).strip() == '':
            if not case_name:
                logger.warning(f"?‰ì •?¬íŒë¡€ ?ë³„ ?•ë³´ ë¶€ì¡?- ?¬íŒë¡€ID: {appeal_id}, ?¬ê±´ëª? {case_name}")
                return False
            logger.debug(f"?¬íŒë¡€?¼ë ¨ë²ˆí˜¸ ?†ìŒ, ?¬ê±´ëª…ìœ¼ë¡??€ì²? {case_name}")
        elif not case_name:
            logger.warning(f"?¬ê±´ëª…ì´ ?†ìŠµ?ˆë‹¤: {appeal}")
            return False
        
        return True
    
    def _create_appeal_data(self, raw_data: Dict[str, Any]) -> Optional[AdministrativeAppealData]:
        """?ì‹œ ?°ì´?°ì—??AdministrativeAppealData ê°ì²´ ?ì„±"""
        try:
            # ?°ì´??ê²€ì¦?
            if not self._validate_appeal_data(raw_data):
                return None
            
            # ?¬íŒë¡€ ID ì¶”ì¶œ
            appeal_id = raw_data.get('?ë??¼ë ¨ë²ˆí˜¸') or raw_data.get('appeal_id')
            
            # ?¬íŒë¡€ IDê°€ ?†ëŠ” ê²½ìš° ?€ì²?ID ?ì„±
            if not appeal_id or str(appeal_id).strip() == '':
                case_name = raw_data.get('?¬ê±´ëª?, '')
                if case_name:
                    appeal_id = f"appeal_{case_name}"
                else:
                    appeal_id = f"appeal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                logger.debug(f"?€ì²?ID ?ì„±: {appeal_id}")
            
            # ?¬íŒ ? í˜• ë¶„ë¥˜
            appeal_type = self.classify_appeal_type(raw_data)
            
            # AdministrativeAppealData ê°ì²´ ?ì„±
            appeal_data = AdministrativeAppealData(
                appeal_id=str(appeal_id),
                case_name=raw_data.get('?¬ê±´ëª?, ''),
                case_number=raw_data.get('?¬ê±´ë²ˆí˜¸', ''),
                appeal_type=appeal_type,
                decision_date=raw_data.get('?ê²°?¼ì', '') or raw_data.get('? ê³ ?¼ì', ''),
                raw_data=raw_data
            )
            
            return appeal_data
            
        except Exception as e:
            logger.error(f"AdministrativeAppealData ?ì„± ?¤íŒ¨: {e}")
            logger.error(f"?ì‹œ ?°ì´?? {raw_data}")
            return None
    
    def classify_appeal_type(self, appeal: Dict[str, Any]) -> AppealType:
        """?‰ì •?¬íŒë¡€ ? í˜• ë¶„ë¥˜"""
        case_name = appeal.get('?¬ê±´ëª?, '').lower()
        case_content = appeal.get('?ì‹œ?¬í•­', '') + ' ' + appeal.get('?ê²°?”ì?', '')
        case_content = case_content.lower()
        
        # ?¤ì›Œ??ê¸°ë°˜ ë¶„ë¥˜
        for appeal_type, keywords in APPEAL_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in case_name or keyword in case_content:
                    return AppealType(appeal_type)
        
        return AppealType.OTHER
    
    def _random_delay(self, min_seconds: Optional[float] = None, max_seconds: Optional[float] = None):
        """API ?”ì²­ ê°??œë¤ ì§€??""
        min_delay = min_seconds or self.api_delay_range[0]
        max_delay = max_seconds or self.api_delay_range[1]
        delay = random.uniform(min_delay, max_delay)
        logger.debug(f"API ?”ì²­ ê°?{delay:.2f}ì´??€ê¸?..")
        time.sleep(delay)
    
    @contextmanager
    def _api_request_with_retry(self, operation_name: str):
        """API ?”ì²­ ?¬ì‹œ??ì»¨í…?¤íŠ¸ ë§¤ë‹ˆ?€"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                self.stats.api_requests_made += 1
                yield
                return
            except Exception as e:
                last_exception = e
                self.stats.api_errors += 1
                self.error_count += 1
                
                if attempt < self.max_retries - 1:
                    logger.warning(f"{operation_name} ?¤íŒ¨ (?œë„ {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))  # ì§€??ë°±ì˜¤??
                else:
                    logger.error(f"{operation_name} ìµœì¢… ?¤íŒ¨: {e}")
                    break
        
        # ëª¨ë“  ?¬ì‹œ?„ê? ?¤íŒ¨??ê²½ìš° ?ˆì™¸ ë°œìƒ
        if last_exception:
            raise last_exception
    
    def collect_all_appeals(self, target_count: int = 1000):
        """ëª¨ë“  ?‰ì •?¬íŒë¡€ ?˜ì§‘ (ê°œì„ ??ë²„ì „)"""
        self.stats.target_count = target_count
        self.stats.status = CollectionStatus.IN_PROGRESS
        
        # ì²´í¬?¬ì¸???Œì¼ ?¤ì •
        checkpoint_file = self._setup_checkpoint_file()
        
        try:
            logger.info(f"?‰ì •?¬íŒë¡€ ?˜ì§‘ ?œì‘ - ëª©í‘œ: {target_count}ê±?)
            logger.info("Graceful shutdown ì§€?? Ctrl+C ?ëŠ” SIGTERM?¼ë¡œ ?ˆì „?˜ê²Œ ì¤‘ë‹¨ ê°€??)
            logger.info("ì¤‘ë‹¨ ???„ì¬ê¹Œì? ?˜ì§‘???°ì´?°ê? ?ë™?¼ë¡œ ?€?¥ë©?ˆë‹¤")
            
            # ?¤ì›Œ?œë³„ ê²€???¤í–‰
            self._collect_by_keywords(target_count, checkpoint_file)
            
            # ëª©í‘œ ?¬ì„±?˜ì? ëª»í•œ ê²½ìš° ë°±ì—… ?¤ì›Œ???¬ìš©
            if self.stats.collected_count < target_count:
                remaining_count = target_count - self.stats.collected_count
                logger.info(f"ëª©í‘œ ?¬ì„± ?¤íŒ¨. ë°±ì—… ?¤ì›Œ?œë¡œ {remaining_count}ê±?ì¶”ê? ?˜ì§‘ ?œë„")
                self._collect_with_fallback_keywords(remaining_count, checkpoint_file)
            
            # ì¢…ë£Œ ?”ì²­ ?•ì¸
            if self._check_shutdown_requested():
                logger.warning(f"?˜ì§‘??ì¤‘ë‹¨?˜ì—ˆ?µë‹ˆ?? {self.shutdown_reason}")
                self.stats.status = CollectionStatus.CANCELLED
                self.stats.end_time = datetime.now()
                self._save_final_checkpoint(checkpoint_file)
                return
            
            # ìµœì¢… ?µê³„ ì¶œë ¥
            self._print_final_stats()
            
            # ì²´í¬?¬ì¸???Œì¼ ?•ë¦¬
            self._cleanup_checkpoint_file(checkpoint_file)
            
            self.stats.status = CollectionStatus.COMPLETED
            self.stats.end_time = datetime.now()
            
        except KeyboardInterrupt:
            logger.warning("?¬ìš©?ì— ?˜í•´ ?˜ì§‘??ì¤‘ë‹¨?˜ì—ˆ?µë‹ˆ??")
            self.stats.status = CollectionStatus.CANCELLED
            self.stats.end_time = datetime.now()
            self._save_final_checkpoint(checkpoint_file)
            return
        except Exception as e:
            logger.error(f"?‰ì •?¬íŒë¡€ ?˜ì§‘ ì¤??¤ë¥˜ ë°œìƒ: {e}")
            self.stats.status = CollectionStatus.FAILED
            self.stats.end_time = datetime.now()
            self._save_final_checkpoint(checkpoint_file)
            raise
    
    def _setup_checkpoint_file(self) -> Path:
        """ì²´í¬?¬ì¸???Œì¼ ?¤ì •"""
        if self.resume_mode and self.checkpoint_file:
            return self.checkpoint_file
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return self.output_dir / f"collection_checkpoint_{timestamp}.json"
    
    def _collect_by_keywords(self, target_count: int, checkpoint_file: Path):
        """?°ì„ ?œìœ„ ê¸°ë°˜ ?¤ì›Œ?œë³„ ?‰ì •?¬íŒë¡€ ?˜ì§‘"""
        # ?°ì„ ?œìœ„ ?¤ì›Œ?œì? ?¼ë°˜ ?¤ì›Œ??ë¶„ë¦¬
        priority_keywords = [kw for kw in PRIORITY_KEYWORDS if kw in ADMINISTRATIVE_APPEAL_KEYWORDS]
        remaining_keywords = [kw for kw in ADMINISTRATIVE_APPEAL_KEYWORDS if kw not in PRIORITY_KEYWORDS]
        
        # ?„ì²´ ?¤ì›Œ??ëª©ë¡ (?°ì„ ?œìœ„ ë¨¼ì?, ?˜ë¨¸ì§€ ?˜ì¤‘??
        ordered_keywords = priority_keywords + remaining_keywords
        
        # ?´ë? ì²˜ë¦¬???¤ì›Œ???œì™¸
        unprocessed_keywords = [kw for kw in ordered_keywords if kw not in self.processed_keywords]
        
        logger.info(f"?°ì„ ?œìœ„ ê¸°ë°˜ ?¤ì›Œ???˜ì§‘ ?œì‘")
        logger.info(f"1?œìœ„ ?¤ì›Œ?? {len(priority_keywords)}ê°?(?°ì„  ?˜ì§‘)")
        logger.info(f"2?œìœ„ ?¤ì›Œ?? {len(remaining_keywords)}ê°?(ì¶”ê? ?˜ì§‘)")
        logger.info(f"ì´??¤ì›Œ?? {len(ordered_keywords)}ê°?)
        logger.info(f"?´ë? ì²˜ë¦¬???¤ì›Œ?? {len(self.processed_keywords)}ê°?)
        logger.info(f"ì²˜ë¦¬ ?€ê¸??¤ì›Œ?? {len(unprocessed_keywords)}ê°?)
        
        if not unprocessed_keywords:
            logger.info("ëª¨ë“  ?¤ì›Œ?œê? ?´ë? ì²˜ë¦¬?˜ì—ˆ?µë‹ˆ??")
            return
        
        # ì§„í–‰ ?í™© ì¶”ì 
        total_keywords = len(unprocessed_keywords)
        logger.info(f"ì´?{total_keywords}ê°?ë¯¸ì²˜ë¦??¤ì›Œ??ì²˜ë¦¬ ?œì‘")
        
        for i, keyword in enumerate(unprocessed_keywords):
            # ì¢…ë£Œ ?”ì²­ ?•ì¸
            if self._check_shutdown_requested():
                logger.warning(f"ì¢…ë£Œ ?”ì²­?¼ë¡œ ?¤ì›Œ??ê²€??ì¤‘ë‹¨: {self.shutdown_reason}")
                break
            
            if self.stats.collected_count >= target_count:
                logger.info(f"ëª©í‘œ ?˜ëŸ‰ {target_count:,}ê±??¬ì„±?¼ë¡œ ?¤ì›Œ??ê²€??ì¤‘ë‹¨")
                break
            
            try:
                # ?¤ì›Œ?œë³„ ëª©í‘œ ê±´ìˆ˜ ê²°ì •
                if keyword in KEYWORD_TARGET_COUNTS:
                    keyword_target = KEYWORD_TARGET_COUNTS[keyword]
                    priority_level = "?°ì„ ?œìœ„"
                else:
                    keyword_target = DEFAULT_KEYWORD_COUNT
                    priority_level = "?¼ë°˜"
                
                # ì§„í–‰ ?í™© ë¡œê¹…
                progress_percent = ((i + 1) / total_keywords) * 100
                logger.info(f"[{i+1}/{total_keywords}] ({progress_percent:.1f}%) ?¤ì›Œ??'{keyword}' ì²˜ë¦¬ ?œì‘ ({priority_level}, ëª©í‘œ: {keyword_target}ê±?")
                
                if i > 0:
                    self._random_delay()
                
                # ?¤ì›Œ?œë³„ ?˜ì§‘
                appeals = self.collect_appeals_by_keyword(keyword, keyword_target)
                
                # ?µê³„ ?…ë°?´íŠ¸
                self.stats.keywords_processed = len(self.processed_keywords)
                
                # ì²´í¬?¬ì¸???€??(ë§??¤ì›Œ?œë§ˆ??
                self._save_checkpoint(checkpoint_file)
                
                # ?¤ì›Œ???„ë£Œ ë¡œê¹…
                logger.info(f"?¤ì›Œ??'{keyword}' ?„ë£Œ ({priority_level}, ëª©í‘œ: {keyword_target}ê±?. ?„ì : {self.stats.collected_count:,}/{target_count:,}ê±?)
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                if self._check_api_limits():
                    break
                    
            except KeyboardInterrupt:
                logger.warning(f"?¤ì›Œ??'{keyword}' ê²€?‰ì´ ì¤‘ë‹¨?˜ì—ˆ?µë‹ˆ??")
                break
            except Exception as e:
                logger.error(f"?¤ì›Œ??'{keyword}' ê²€???¤íŒ¨: {e}")
                continue
    
    def collect_appeals_by_keyword(self, keyword: str, max_count: int = 30) -> List[AdministrativeAppealData]:
        """?¤ì›Œ?œë¡œ ?‰ì •?¬íŒë¡€ ê²€??ë°??˜ì§‘"""
        # ?´ë? ì²˜ë¦¬???¤ì›Œ?œì¸ì§€ ?•ì¸
        if keyword in self.processed_keywords:
            logger.info(f"?¤ì›Œ??'{keyword}'???´ë? ì²˜ë¦¬?˜ì—ˆ?µë‹ˆ?? ê±´ë„ˆ?ë‹ˆ??")
            return []
        
        logger.info(f"?¤ì›Œ??'{keyword}'ë¡??‰ì •?¬íŒë¡€ ê²€???œì‘ (ëª©í‘œ: {max_count}ê±?")
        
        appeals = []
        page = 1
        consecutive_empty_pages = 0
        max_empty_pages = 3
        
        while len(appeals) < max_count and consecutive_empty_pages < max_empty_pages:
            # ì¢…ë£Œ ?”ì²­ ?•ì¸
            if self._check_shutdown_requested():
                logger.warning(f"ì¢…ë£Œ ?”ì²­?¼ë¡œ '{keyword}' ê²€??ì¤‘ë‹¨: {self.shutdown_reason}")
                break
            
            try:
                # API ?”ì²­ ê°??œë¤ ì§€??
                if page > 1:
                    self._random_delay()
                
                # ì§„í–‰ ?í™© ë¡œê¹…
                logger.debug(f"?¤ì›Œ??'{keyword}' ?˜ì´ì§€ {page} ê²€??ì¤?..")
                
                # API ?”ì²­ ?¤í–‰
                try:
                    with self._api_request_with_retry(f"?¤ì›Œ??'{keyword}' ê²€??):
                        results = self.client.get_administrative_appeal_list(
                            query=keyword,
                            display=100,
                            page=page
                        )
                except Exception as api_error:
                    logger.error(f"API ?”ì²­ ?¤íŒ¨: {api_error}")
                    consecutive_empty_pages += 1
                    page += 1
                    continue
                
                if not results:
                    consecutive_empty_pages += 1
                    logger.debug(f"?¤ì›Œ??'{keyword}' ?˜ì´ì§€ {page}?ì„œ ê²°ê³¼ ?†ìŒ (?°ì† ë¹??˜ì´ì§€: {consecutive_empty_pages})")
                    page += 1
                    continue
                else:
                    consecutive_empty_pages = 0
                
                # ê²°ê³¼ ì²˜ë¦¬
                new_count, duplicate_count = self._process_search_results(results, appeals, max_count)
                
                # ?˜ì´ì§€ë³?ê²°ê³¼ ë¡œê¹…
                logger.debug(f"?˜ì´ì§€ {page}: {new_count}ê±?? ê·œ, {duplicate_count}ê±?ì¤‘ë³µ (?„ì : {len(appeals)}/{max_count}ê±?")
                
                page += 1
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                if self._check_api_limits():
                    break
                    
            except KeyboardInterrupt:
                logger.warning(f"?¤ì›Œ??'{keyword}' ê²€?‰ì´ ì¤‘ë‹¨?˜ì—ˆ?µë‹ˆ??")
                break
            except Exception as e:
                logger.error(f"?¤ì›Œ??'{keyword}' ê²€??ì¤??¤ë¥˜: {e}")
                self.stats.failed_count += 1
                break
        
        # ?˜ì§‘???¬íŒë¡€ë¥??„ì‹œ ?€?¥ì†Œ??ì¶”ê?
        for appeal in appeals:
            self.pending_appeals.append(appeal)
            self.stats.collected_count += 1
        
        # ?¤ì›Œ??ì²˜ë¦¬ ?„ë£Œ ?œì‹œ
        self.processed_keywords.add(keyword)
        
        logger.info(f"?¤ì›Œ??'{keyword}' ?˜ì§‘ ?„ë£Œ: {len(appeals)}ê±?)
        return appeals
    
    def _process_search_results(self, results: List[Dict[str, Any]], appeals: List[AdministrativeAppealData], 
                              max_count: int) -> Tuple[int, int]:
        """ê²€??ê²°ê³¼ ì²˜ë¦¬"""
        new_count = 0
        duplicate_count = 0
        
        for result in results:
            # ì¢…ë£Œ ?”ì²­ ?•ì¸
            if self._check_shutdown_requested():
                logger.warning(f"ì¢…ë£Œ ?”ì²­?¼ë¡œ ê²°ê³¼ ì²˜ë¦¬ ì¤‘ë‹¨: {self.shutdown_reason}")
                break
            
            # ì¤‘ë³µ ?•ì¸
            if self._is_duplicate_appeal(result):
                duplicate_count += 1
                self.stats.duplicate_count += 1
                continue
            
            # AdministrativeAppealData ê°ì²´ ?ì„±
            appeal_data = self._create_appeal_data(result)
            if not appeal_data:
                self.stats.failed_count += 1
                continue
            
            # ? ê·œ ?¬íŒë¡€ ì¶”ê?
            appeals.append(appeal_data)
            self._mark_appeal_collected(result)
            new_count += 1
            
            if len(appeals) >= max_count:
                break
        
        return new_count, duplicate_count
    
    def _save_batch_appeals(self):
        """ë°°ì¹˜ ?¨ìœ„ë¡??‰ì •?¬íŒë¡€ ?€??""
        if not self.pending_appeals:
            return
        
        try:
            # ? í˜•ë³„ë¡œ ê·¸ë£¹??
            by_type = {}
            for appeal in self.pending_appeals:
                appeal_type = appeal.appeal_type.value
                if appeal_type not in by_type:
                    by_type[appeal_type] = []
                by_type[appeal_type].append(appeal)
            
            # ? í˜•ë³??Œì¼ ?€??
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            saved_files = []
            
            for appeal_type, appeals in by_type.items():
                # ?ˆì „???Œì¼ëª??ì„±
                safe_type = appeal_type.replace('_', '-')
                filename = f"batch_{safe_type}_{len(appeals)}ê±?{timestamp}.json"
                filepath = self.output_dir / filename
                
                batch_data = {
                    'metadata': {
                        'appeal_type': appeal_type,
                        'count': len(appeals),
                        'saved_at': datetime.now().isoformat(),
                        'batch_id': timestamp
                    },
                    'appeals': [a.raw_data for a in appeals]
                }
                
                with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
                    json.dump(batch_data, f, ensure_ascii=False, indent=2)
                
                saved_files.append(filepath)
                logger.info(f"ë°°ì¹˜ ?€???„ë£Œ: {appeal_type} ? í˜• {len(appeals):,}ê±?-> {filename}")
            
            # ?µê³„ ?…ë°?´íŠ¸
            self.stats.saved_count += len(self.pending_appeals)
            
            # ?„ì‹œ ?€?¥ì†Œ ì´ˆê¸°??
            self.pending_appeals = []
            
            logger.info(f"ë°°ì¹˜ ?€???„ë£Œ: ì´?{len(saved_files):,}ê°??Œì¼")
            
        except Exception as e:
            logger.error(f"ë°°ì¹˜ ?€???¤íŒ¨: {e}")
            logger.error(traceback.format_exc())
    
    def _check_api_limits(self) -> bool:
        """API ?”ì²­ ?œí•œ ?•ì¸"""
        try:
            stats = self.client.get_request_stats()
            remaining = stats.get('remaining_requests', 0)
            if remaining < 100:
                logger.warning(f"API ?”ì²­ ?œë„ê°€ ë¶€ì¡±í•©?ˆë‹¤. ?¨ì? ?”ì²­: {remaining}??)
                return True
        except Exception as e:
            logger.warning(f"API ?”ì²­ ?œí•œ ?•ì¸ ?¤íŒ¨: {e}")
        return False
    
    def _print_final_stats(self):
        """ìµœì¢… ?µê³„ ì¶œë ¥"""
        logger.info("=" * 60)
        logger.info("?˜ì§‘ ?„ë£Œ ?µê³„")
        logger.info("=" * 60)
        logger.info(f"ëª©í‘œ ?˜ì§‘ ê±´ìˆ˜: {self.stats.target_count:,}ê±?)
        logger.info(f"?¤ì œ ?˜ì§‘ ê±´ìˆ˜: {self.stats.collected_count:,}ê±?)
        logger.info(f"ì¤‘ë³µ ?œì™¸ ê±´ìˆ˜: {self.stats.duplicate_count:,}ê±?)
        logger.info(f"?¤íŒ¨ ê±´ìˆ˜: {self.stats.failed_count:,}ê±?)
        logger.info(f"ì²˜ë¦¬???¤ì›Œ?? {len(self.processed_keywords):,}ê°?)
        logger.info(f"?€?¥ëœ ë°°ì¹˜ ?? {self.stats.saved_count:,}ê±?)
        logger.info(f"API ?”ì²­ ?? {self.stats.api_requests_made:,}??)
        logger.info(f"API ?¤ë¥˜ ?? {self.stats.api_errors:,}??)
        logger.info(f"?±ê³µë¥? {self.stats.success_rate:.1f}%")
        if self.stats.duration:
            logger.info(f"?Œìš” ?œê°„: {self.stats.duration}")
        logger.info("=" * 60)
    
    def _cleanup_checkpoint_file(self, checkpoint_file: Path):
        """ì²´í¬?¬ì¸???Œì¼ ?•ë¦¬"""
        if checkpoint_file and checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                logger.info("ì²´í¬?¬ì¸???Œì¼ ?•ë¦¬ ?„ë£Œ")
            except Exception as e:
                logger.warning(f"ì²´í¬?¬ì¸???Œì¼ ?•ë¦¬ ?¤íŒ¨: {e}")
    
    def _save_final_checkpoint(self, checkpoint_file: Path):
        """ìµœì¢… ì²´í¬?¬ì¸???€??""
        try:
            self._save_checkpoint(checkpoint_file)
            logger.info(f"?„ì¬ê¹Œì? ?˜ì§‘???°ì´?°ëŠ” {checkpoint_file}???€?¥ë˜?ˆìŠµ?ˆë‹¤.")
            logger.info("?˜ì¤‘???¤ì‹œ ?¤í–‰?˜ë©´ ?´ì–´??ê³„ì†?????ˆìŠµ?ˆë‹¤.")
        except Exception as e:
            logger.error(f"ìµœì¢… ì²´í¬?¬ì¸???€???¤íŒ¨: {e}")
    
    def _save_checkpoint(self, checkpoint_file: Path):
        """ì§„í–‰ ?í™© ì²´í¬?¬ì¸???€??""
        try:
            checkpoint_data = {
                'stats': {
                    'start_time': self.stats.start_time.isoformat(),
                    'end_time': self.stats.end_time.isoformat() if self.stats.end_time else None,
                    'target_count': self.stats.target_count,
                    'collected_count': self.stats.collected_count,
                    'saved_count': self.stats.saved_count,
                    'duplicate_count': self.stats.duplicate_count,
                    'failed_count': self.stats.failed_count,
                    'keywords_processed': self.stats.keywords_processed,
                    'total_keywords': self.stats.total_keywords,
                    'api_requests_made': self.stats.api_requests_made,
                    'api_errors': self.stats.api_errors,
                    'status': self.stats.status.value,
                    'processed_keywords': list(self.processed_keywords),
                    'collected_appeals_count': len(self.collected_appeals)
                },
                'appeals': [a.raw_data for a in self.pending_appeals],
                'saved_at': datetime.now().isoformat(),
                'resume_info': {
                    'can_resume': True,
                    'last_keyword_processed': list(self.processed_keywords)[-1] if self.processed_keywords else None,
                    'progress_percentage': (self.stats.collected_count / self.stats.target_count) * 100 if self.stats.target_count > 0 else 0
                },
                'shutdown_info': {
                    'shutdown_requested': self.shutdown_requested,
                    'shutdown_reason': self.shutdown_reason,
                    'graceful_shutdown_supported': True
                }
            }
            
            with open(checkpoint_file, 'w', encoding='utf-8', newline='\n') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"ì²´í¬?¬ì¸???€???„ë£Œ: {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"ì²´í¬?¬ì¸???€???¤íŒ¨: {e}")
            logger.error(traceback.format_exc())
    
    def _check_and_resume_from_checkpoint(self):
        """ì²´í¬?¬ì¸???Œì¼ ?•ì¸ ë°?ë³µêµ¬"""
        logger.info("ì²´í¬?¬ì¸???Œì¼ ?•ì¸ ì¤?..")
        
        # ì²´í¬?¬ì¸???Œì¼ ì°¾ê¸°
        checkpoint_files = list(self.output_dir.glob("collection_checkpoint_*.json"))
        
        if not checkpoint_files:
            logger.info("ì²´í¬?¬ì¸???Œì¼???†ìŠµ?ˆë‹¤. ?ˆë¡œ ?œì‘?©ë‹ˆ??")
            return
        
        # ê°€??ìµœê·¼ ì²´í¬?¬ì¸???Œì¼ ? íƒ
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        self.checkpoint_file = latest_checkpoint
        
        logger.info(f"ì²´í¬?¬ì¸???Œì¼ ë°œê²¬: {latest_checkpoint.name}")
        
        try:
            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # ì²´í¬?¬ì¸???°ì´??ë³µêµ¬
            self._restore_from_checkpoint(checkpoint_data)
            
            self.resume_mode = True
            self.stats.status = CollectionStatus.IN_PROGRESS
            
            logger.info("=" * 60)
            logger.info("?´ì „ ?‘ì—… ë³µêµ¬ ?„ë£Œ")
            logger.info("=" * 60)
            logger.info(f"ë³µêµ¬???˜ì§‘ ê±´ìˆ˜: {self.stats.collected_count:,}ê±?)
            logger.info(f"ì²˜ë¦¬???¤ì›Œ?? {len(self.processed_keywords):,}ê°?)
            logger.info(f"?€?¥ëœ ë°°ì¹˜ ?? {self.stats.saved_count:,}ê±?)
            logger.info(f"ì¤‘ë³µ ?œì™¸ ê±´ìˆ˜: {self.stats.duplicate_count:,}ê±?)
            logger.info(f"API ?”ì²­ ?? {self.stats.api_requests_made:,}??)
            logger.info("=" * 60)
            
            # ?ë™?¼ë¡œ ê³„ì† ì§„í–‰
            logger.info("?´ì „ ?‘ì—…???´ì–´??ì§„í–‰?©ë‹ˆ??")
            
        except Exception as e:
            logger.error(f"ì²´í¬?¬ì¸???Œì¼ ë³µêµ¬ ?¤íŒ¨: {e}")
            logger.info("?ˆë¡œ ?œì‘?©ë‹ˆ??")
            self.resume_mode = False
            self.checkpoint_file = None
    
    def _restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """ì²´í¬?¬ì¸???°ì´?°ì—???íƒœ ë³µêµ¬"""
        stats_data = checkpoint_data.get('stats', {})
        appeals = checkpoint_data.get('appeals', [])
        
        # ?µê³„ ë³µêµ¬
        self.stats.collected_count = stats_data.get('collected_count', 0)
        self.stats.saved_count = stats_data.get('saved_count', 0)
        self.stats.duplicate_count = stats_data.get('duplicate_count', 0)
        self.stats.failed_count = stats_data.get('failed_count', 0)
        self.stats.keywords_processed = stats_data.get('keywords_processed', 0)
        self.stats.api_requests_made = stats_data.get('api_requests_made', 0)
        self.stats.api_errors = stats_data.get('api_errors', 0)
        
        # ì²˜ë¦¬???¤ì›Œ??ë³µêµ¬
        processed_keywords = stats_data.get('processed_keywords', [])
        self.processed_keywords = set(processed_keywords)
        
        # ?˜ì§‘???¬íŒë¡€ ID ë³µêµ¬
        for appeal in appeals:
            if isinstance(appeal, dict):
                appeal_id = appeal.get('?ë??¼ë ¨ë²ˆí˜¸') or appeal.get('appeal_id')
                if appeal_id:
                    self.collected_appeals.add(str(appeal_id))
    
    def _collect_with_fallback_keywords(self, remaining_count: int, checkpoint_file: Path):
        """ë°±ì—… ?¤ì›Œ?œë¡œ ì¶”ê? ?˜ì§‘"""
        logger.info(f"ë°±ì—… ?¤ì›Œ???˜ì§‘ ?œì‘ - ëª©í‘œ: {remaining_count}ê±?)
        
        # ë°±ì—… ?¤ì›Œ??ì¤??„ì§ ì²˜ë¦¬?˜ì? ?Šì? ê²ƒë“¤ë§?? íƒ
        unprocessed_fallback = [kw for kw in FALLBACK_KEYWORDS if kw not in self.processed_keywords]
        
        if not unprocessed_fallback:
            logger.info("ëª¨ë“  ë°±ì—… ?¤ì›Œ?œê? ?´ë? ì²˜ë¦¬?˜ì—ˆ?µë‹ˆ??")
            return
        
        logger.info(f"ë°±ì—… ?¤ì›Œ??{len(unprocessed_fallback)}ê°œë¡œ ì¶”ê? ?˜ì§‘ ?œë„")
        
        for i, keyword in enumerate(unprocessed_fallback):
            if self.stats.collected_count >= self.stats.target_count:
                logger.info(f"ëª©í‘œ ?˜ëŸ‰ {self.stats.target_count:,}ê±??¬ì„±?¼ë¡œ ë°±ì—… ?¤ì›Œ???˜ì§‘ ì¤‘ë‹¨")
                break
            
            if self._check_shutdown_requested():
                logger.warning(f"ì¢…ë£Œ ?”ì²­?¼ë¡œ ë°±ì—… ?¤ì›Œ???˜ì§‘ ì¤‘ë‹¨: {self.shutdown_reason}")
                break
            
            try:
                # ë°±ì—… ?¤ì›Œ?œëŠ” ê°ê° 5ê±´ì”© ?˜ì§‘
                keyword_target = min(5, remaining_count)
                remaining_count -= keyword_target
                
                logger.info(f"ë°±ì—… ?¤ì›Œ??'{keyword}' ì²˜ë¦¬ ?œì‘ (ëª©í‘œ: {keyword_target}ê±?")
                
                # ?¬íŒë¡€ ?˜ì§‘
                appeals = self.collect_appeals_by_keyword(keyword, keyword_target)
                
                # ë°°ì¹˜ ?€??
                if len(self.pending_appeals) >= self.batch_size:
                    self._save_batch_appeals()
                
                # ì²´í¬?¬ì¸???€??
                self._save_checkpoint(checkpoint_file)
                
                # ì§„í–‰ ?í™© ë¡œê¹…
                progress_percent = (self.stats.collected_count / self.stats.target_count) * 100
                logger.info(f"ë°±ì—… ?¤ì›Œ??'{keyword}' ?„ë£Œ: {len(appeals)}ê±??˜ì§‘ (ì´?{self.stats.collected_count:,}ê±? {progress_percent:.1f}%)")
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                if self._check_api_limits():
                    logger.warning("API ?”ì²­ ?œí•œ???„ë‹¬?˜ì—¬ ë°±ì—… ?¤ì›Œ???˜ì§‘ ì¤‘ë‹¨")
                    break
                    
            except Exception as e:
                logger.error(f"ë°±ì—… ?¤ì›Œ??'{keyword}' ?˜ì§‘ ì¤??¤ë¥˜: {e}")
                self.stats.failed_count += 1
                continue
        
        logger.info(f"ë°±ì—… ?¤ì›Œ???˜ì§‘ ?„ë£Œ - ì´?{self.stats.collected_count:,}ê±??˜ì§‘")
            
        # ë°°ì¹˜ ?€??(?„ì‹œ ?€?¥ì†Œê°€ ê°€??ì°?ê²½ìš°)
        if len(self.pending_appeals) >= self.batch_size:
            self._save_batch_appeals()
