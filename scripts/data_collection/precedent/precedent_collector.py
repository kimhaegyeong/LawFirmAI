#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?ë? ?˜ì§‘ê¸??´ë˜??
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
# tqdm ?œê±° - ë¡œê·¸ ê¸°ë°˜ ì§„í–‰ ?í™© ?œì‹œë¡??€ì²?

import sys
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient, LawOpenAPIConfig
from scripts.precedent.precedent_models import (
    CollectionStats, PrecedentData, CollectionStatus, PrecedentCategory,
    SEARCH_KEYWORDS, PRIORITY_KEYWORDS, KEYWORD_TARGET_COUNTS, DEFAULT_KEYWORD_COUNT,
    COURT_CODES, CASE_TYPE_CODES
)
from scripts.precedent.precedent_logger import setup_logging

logger = setup_logging()


class PrecedentCollector:
    """?ë? ?˜ì§‘ ?´ë˜??(ê°œì„ ??ë²„ì „)"""
    
    def __init__(self, config: LawOpenAPIConfig, output_dir: Optional[Path] = None):
        """
        ?ë? ?˜ì§‘ê¸?ì´ˆê¸°??
        
        Args:
            config: API ?¤ì • ê°ì²´
            output_dir: ì¶œë ¥ ?”ë ‰? ë¦¬ (ê¸°ë³¸ê°? data/raw/precedents)
        """
        self.client = LawOpenAPIClient(config)
        self.output_dir = output_dir or Path("data/raw/precedents")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ?°ì´??ê´€ë¦?
        self.collected_precedents: Set[str] = set()  # ì¤‘ë³µ ë°©ì? (?ë??¼ë ¨ë²ˆí˜¸)
        self.processed_keywords: Set[str] = set()  # ì²˜ë¦¬???¤ì›Œ??ì¶”ì 
        self.pending_precedents: List[PrecedentData] = []  # ?„ì‹œ ?€?¥ì†Œ
        
        # ?¤ì •
        self.batch_size = 100  # ë°°ì¹˜ ?€???¬ê¸° (ì¦ê?)
        self.max_retries = 3  # ìµœë? ?¬ì‹œ???Ÿìˆ˜
        self.retry_delay = 5  # ?¬ì‹œ??ê°„ê²© (ì´?
        self.api_delay_range = (1.0, 3.0)  # API ?”ì²­ ê°?ì§€??ë²”ìœ„
        
        # ?µê³„ ë°??íƒœ
        self.stats = CollectionStats()
        self.stats.total_keywords = len(SEARCH_KEYWORDS)
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
        
        logger.info(f"?ë? ?˜ì§‘ê¸?ì´ˆê¸°???„ë£Œ - ëª©í‘œ: {self.stats.target_count}ê±?)
    
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
        if self.pending_precedents:
            logger.info("?„ë¡œê·¸ë¨ ì¢…ë£Œ ???„ì‹œ ?°ì´???€??ì¤?..")
            self._save_batch_precedents()
        
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
        """ê¸°ì¡´ ?˜ì§‘???°ì´??ë¡œë“œ?˜ì—¬ ì¤‘ë³µ ë°©ì? (ê°•í™”??ë²„ì „)"""
        logger.info("ê¸°ì¡´ ?˜ì§‘???°ì´???•ì¸ ì¤?..")
        
        loaded_count = 0
        error_count = 0
        
        # ?¤ì–‘???Œì¼ ?¨í„´?ì„œ ?°ì´??ë¡œë“œ
        file_patterns = [
            "precedent_*.json",
            "batch_*.json", 
            "checkpoints/**/*.json",
            "*.json"
        ]
        
        for pattern in file_patterns:
            files = list(self.output_dir.glob(pattern))
            for file_path in files:
                try:
                    loaded_count += self._load_precedents_from_file(file_path)
                except Exception as e:
                    error_count += 1
                    logger.debug(f"?Œì¼ ë¡œë“œ ?¤íŒ¨ {file_path}: {e}")
        
        # ì²´í¬?¬ì¸???Œì¼?ì„œ??ì¤‘ë³µ ?°ì´??ë¡œë“œ
        checkpoint_files = list(self.output_dir.glob("collection_checkpoint_*.json"))
        for checkpoint_file in checkpoint_files:
            try:
                loaded_count += self._load_checkpoint_data(checkpoint_file)
            except Exception as e:
                error_count += 1
                logger.debug(f"ì²´í¬?¬ì¸??ë¡œë“œ ?¤íŒ¨ {checkpoint_file}: {e}")
        
        logger.info(f"ê¸°ì¡´ ?°ì´??ë¡œë“œ ?„ë£Œ: {loaded_count:,}ê±? ?¤ë¥˜: {error_count:,}ê±?)
        self.stats.collected_count = len(self.collected_precedents)
        logger.info(f"ì¤‘ë³µ ë°©ì?ë¥??„í•œ ?ë? ID {len(self.collected_precedents):,}ê°?ë¡œë“œ??)
    
    def _load_precedents_from_file(self, file_path: Path) -> int:
        """?Œì¼?ì„œ ?ë? ?°ì´??ë¡œë“œ"""
        loaded_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # ?¤ì–‘???°ì´??êµ¬ì¡° ì²˜ë¦¬
        precedents = []
        
        if isinstance(data, dict):
            if 'precedents' in data:
                precedents = data['precedents']
            elif 'basic_info' in data:
                precedents = [data]
            elif 'by_category' in data:
                for category_data in data['by_category'].values():
                    precedents.extend(category_data)
        elif isinstance(data, list):
            precedents = data
        
        # ?ë? ID ì¶”ì¶œ
        for precedent in precedents:
            if isinstance(precedent, dict):
                precedent_id = precedent.get('?ë??¼ë ¨ë²ˆí˜¸') or precedent.get('precedent_id')
                if precedent_id:
                    self.collected_precedents.add(str(precedent_id))
                    loaded_count += 1
        
        return loaded_count
    
    def _load_checkpoint_data(self, checkpoint_file: Path) -> int:
        """ì²´í¬?¬ì¸???Œì¼?ì„œ ?ë? ?°ì´??ë¡œë“œ"""
        loaded_count = 0
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # ì²´í¬?¬ì¸?¸ì—???ë? ?°ì´??ì¶”ì¶œ
            precedents = checkpoint_data.get('precedents', [])
            
            for precedent in precedents:
                if isinstance(precedent, dict):
                    # ?¤ì–‘???„ë“œëª…ìœ¼ë¡??ë? ID ?•ì¸
                    precedent_id = (
                        precedent.get('?ë??¼ë ¨ë²ˆí˜¸') or 
                        precedent.get('precedent_id') or 
                        precedent.get('prec id') or
                        precedent.get('id')
                    )
                    
                    if precedent_id:
                        self.collected_precedents.add(str(precedent_id))
                        loaded_count += 1
                    else:
                        # ?€ì²??ë³„???¬ìš©
                        case_number = precedent.get('?¬ê±´ë²ˆí˜¸', '')
                        case_name = precedent.get('?¬ê±´ëª?, '')
                        decision_date = precedent.get('? ê³ ?¼ì', '')
                        
                        if case_number and case_name:
                            alternative_id = f"{case_number}_{case_name}_{decision_date}"
                            self.collected_precedents.add(alternative_id)
                            loaded_count += 1
                        elif case_number:
                            alternative_id = f"case_{case_number}_{decision_date}"
                            self.collected_precedents.add(alternative_id)
                            loaded_count += 1
                        elif case_name and decision_date:
                            alternative_id = f"name_{case_name}_{decision_date}"
                            self.collected_precedents.add(alternative_id)
                            loaded_count += 1
            
        except Exception as e:
            logger.debug(f"ì²´í¬?¬ì¸???Œì¼ ë¡œë“œ ?¤íŒ¨ {checkpoint_file}: {e}")
        
        return loaded_count
    
    def _check_and_resume_from_checkpoint(self):
        """ì²´í¬?¬ì¸???Œì¼ ?•ì¸ ë°?ë³µêµ¬ (ê°œì„ ??ë²„ì „)"""
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
        precedents = checkpoint_data.get('precedents', [])
        
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
        
        # ?˜ì§‘???ë? ID ë³µêµ¬
        for precedent in precedents:
            if isinstance(precedent, dict):
                precedent_id = precedent.get('?ë??¼ë ¨ë²ˆí˜¸') or precedent.get('precedent_id')
                if precedent_id:
                    self.collected_precedents.add(str(precedent_id))
    
    def _is_duplicate_precedent(self, precedent: Dict[str, Any]) -> bool:
        """?ë? ì¤‘ë³µ ?¬ë? ?•ì¸ (ê°•í™”??ë²„ì „)"""
        # ?¤ì–‘???„ë“œëª…ìœ¼ë¡??ë? ID ?•ì¸
        precedent_id = (
            precedent.get('?ë??¼ë ¨ë²ˆí˜¸') or 
            precedent.get('precedent_id') or 
            precedent.get('prec id') or
            precedent.get('id')
        )
        
        # 1ì°? ?ë??¼ë ¨ë²ˆí˜¸ë¡??•ì¸
        if precedent_id and str(precedent_id).strip() != '':
            if str(precedent_id) in self.collected_precedents:
                logger.debug(f"?ë??¼ë ¨ë²ˆí˜¸ë¡?ì¤‘ë³µ ?•ì¸: {precedent_id}")
                return True
        
        # 2ì°? ?€ì²??ë³„?ë¡œ ?•ì¸
        case_number = precedent.get('?¬ê±´ë²ˆí˜¸', '')
        case_name = precedent.get('?¬ê±´ëª?, '')
        decision_date = precedent.get('? ê³ ?¼ì', '')
        
        # ?¬ê±´ë²ˆí˜¸ + ?¬ê±´ëª?+ ? ê³ ?¼ì ì¡°í•©
        if case_number and case_name and decision_date:
            alternative_id = f"{case_number}_{case_name}_{decision_date}"
            if alternative_id in self.collected_precedents:
                logger.debug(f"?€ì²?IDë¡?ì¤‘ë³µ ?•ì¸: {alternative_id}")
                return True
        
        # ?¬ê±´ë²ˆí˜¸ + ?¬ê±´ëª?ì¡°í•©
        if case_number and case_name:
            alternative_id = f"{case_number}_{case_name}"
            if alternative_id in self.collected_precedents:
                logger.debug(f"?¬ê±´ë²ˆí˜¸+?¬ê±´ëª…ìœ¼ë¡?ì¤‘ë³µ ?•ì¸: {alternative_id}")
                return True
        
        # ?¬ê±´ë²ˆí˜¸ë§??ˆëŠ” ê²½ìš°
        if case_number:
            alternative_id = f"case_{case_number}_{decision_date}" if decision_date else f"case_{case_number}"
            if alternative_id in self.collected_precedents:
                logger.debug(f"?¬ê±´ë²ˆí˜¸ë§Œìœ¼ë¡?ì¤‘ë³µ ?•ì¸: {alternative_id}")
                return True
        
        # ?¬ê±´ëª?+ ? ê³ ?¼ì ì¡°í•©
        if case_name and decision_date:
            alternative_id = f"name_{case_name}_{decision_date}"
            if alternative_id in self.collected_precedents:
                logger.debug(f"?¬ê±´ëª?? ê³ ?¼ìë¡?ì¤‘ë³µ ?•ì¸: {alternative_id}")
                return True
        
        # ëª¨ë“  ?ë³„?ê? ?†ëŠ” ê²½ìš° ì¤‘ë³µ?¼ë¡œ ì²˜ë¦¬
        if not precedent_id and not case_number and not case_name:
            precedent_str = str(precedent)
            logger.warning(f"?ë? ?ë³„?ê? ?†ì–´ ì¤‘ë³µ?¼ë¡œ ì²˜ë¦¬?©ë‹ˆ?? ?°ì´?? {precedent_str[:200]}{'...' if len(precedent_str) > 200 else ''}")
            return True
        
        return False
    
    def _mark_precedent_collected(self, precedent: Dict[str, Any]):
        """?ë?ë¥??˜ì§‘?¨ìœ¼ë¡??œì‹œ (ê°•í™”??ë²„ì „)"""
        # ?¤ì–‘???„ë“œëª…ìœ¼ë¡??ë? ID ?•ì¸
        precedent_id = (
            precedent.get('?ë??¼ë ¨ë²ˆí˜¸') or 
            precedent.get('precedent_id') or 
            precedent.get('prec id') or
            precedent.get('id')
        )
        
        case_number = precedent.get('?¬ê±´ë²ˆí˜¸', '')
        case_name = precedent.get('?¬ê±´ëª?, '')
        decision_date = precedent.get('? ê³ ?¼ì', '')
        
        # 1ì°? ?ë??¼ë ¨ë²ˆí˜¸ë¡??€??
        if precedent_id and str(precedent_id).strip() != '':
            self.collected_precedents.add(str(precedent_id))
            logger.debug(f"?ë??¼ë ¨ë²ˆí˜¸ë¡??€?? {precedent_id}")
        
        # 2ì°? ?€ì²??ë³„?ë¡œ???€??(ì¤‘ë³µ ë°©ì? ê°•í™”)
        if case_number and case_name and decision_date:
            alternative_id = f"{case_number}_{case_name}_{decision_date}"
            self.collected_precedents.add(alternative_id)
            logger.debug(f"?€ì²?IDë¡??€?? {alternative_id}")
        elif case_number and case_name:
            alternative_id = f"{case_number}_{case_name}"
            self.collected_precedents.add(alternative_id)
            logger.debug(f"?¬ê±´ë²ˆí˜¸+?¬ê±´ëª…ìœ¼ë¡??€?? {alternative_id}")
        elif case_number:
            alternative_id = f"case_{case_number}_{decision_date}" if decision_date else f"case_{case_number}"
            self.collected_precedents.add(alternative_id)
            logger.debug(f"?¬ê±´ë²ˆí˜¸ë¡??€?? {alternative_id}")
        elif case_name and decision_date:
            alternative_id = f"name_{case_name}_{decision_date}"
            self.collected_precedents.add(alternative_id)
            logger.debug(f"?¬ê±´ëª?? ê³ ?¼ìë¡??€?? {alternative_id}")
    
    def _validate_precedent_data(self, precedent: Dict[str, Any]) -> bool:
        """?ë? ?°ì´??ê²€ì¦?(ê°œì„ ??ë²„ì „)"""
        # ?ë? ID ?•ì¸ (?¤ì–‘???„ë“œëª?ì§€??
        precedent_id = (
            precedent.get('?ë??¼ë ¨ë²ˆí˜¸') or 
            precedent.get('precedent_id') or 
            precedent.get('prec id') or
            precedent.get('id')
        )
        
        # ?¬ê±´ëª??•ì¸
        case_name = precedent.get('?¬ê±´ëª?)
        case_number = precedent.get('?¬ê±´ë²ˆí˜¸', '')
        
        # ?ë? IDê°€ ?†ëŠ” ê²½ìš° ?¬ê±´ë²ˆí˜¸ë¡??€ì²?ê²€ì¦?
        if not precedent_id or str(precedent_id).strip() == '':
            if not case_number:
                logger.warning(f"?ë? ?ë³„ ?•ë³´ ë¶€ì¡?- ?ë?ID: {precedent_id}, ?¬ê±´ë²ˆí˜¸: {case_number}")
                return False
            logger.debug(f"?ë??¼ë ¨ë²ˆí˜¸ ?†ìŒ, ?¬ê±´ë²ˆí˜¸ë¡??€ì²? {case_number}")
        elif not case_name and not case_number:
            logger.warning(f"?¬ê±´ëª…ê³¼ ?¬ê±´ë²ˆí˜¸ê°€ ëª¨ë‘ ?†ìŠµ?ˆë‹¤: {precedent}")
            return False
        
        # ? ì§œ ?•ì‹ ê²€ì¦?(?¬ëŸ¬ ? ì§œ ?„ë“œ ?•ì¸)
        date_fields = ['?ê²°?¼ì', '? ê³ ?¼ì', 'decision_date']
        for field in date_fields:
            date_value = precedent.get(field)
            if date_value and str(date_value).strip():
                try:
                    # ?¤ì–‘??? ì§œ ?•ì‹ ì§€??
                    date_str = str(date_value)
                    if len(date_str) == 8:  # YYYYMMDD
                        datetime.strptime(date_str, '%Y%m%d')
                    elif len(date_str) == 10:  # YYYY-MM-DD
                        datetime.strptime(date_str, '%Y-%m-%d')
                    elif '.' in date_str and len(date_str) == 10:  # YYYY.MM.DD
                        datetime.strptime(date_str, '%Y.%m.%d')
                    else:
                        logger.debug(f"ì§€?í•˜ì§€ ?ŠëŠ” ? ì§œ ?•ì‹ ({field}): {date_value}")
                except ValueError:
                    logger.debug(f"? ì§œ ?•ì‹ ë³€???¤íŒ¨ ({field}): {date_value}")
                    # ? ì§œ ?•ì‹ ?¤ë¥˜???”ë²„ê·??ˆë²¨ë¡?ë³€ê²?(?ˆë¬´ ë§ì? ê²½ê³  ë°©ì?)
        
        return True
    
    def _create_precedent_data(self, raw_data: Dict[str, Any]) -> Optional[PrecedentData]:
        """?ì‹œ ?°ì´?°ì—??PrecedentData ê°ì²´ ?ì„± (ê°œì„ ??ë²„ì „)"""
        try:
            # ?°ì´??ê²€ì¦?
            if not self._validate_precedent_data(raw_data):
                return None
            
            # ?ë? ID ì¶”ì¶œ (?¤ì–‘???„ë“œëª?ì§€??
            precedent_id = (
                raw_data.get('?ë??¼ë ¨ë²ˆí˜¸') or 
                raw_data.get('precedent_id') or 
                raw_data.get('prec id') or
                raw_data.get('id')
            )
            
            # ?ë? IDê°€ ?†ëŠ” ê²½ìš° ?€ì²?ID ?ì„±
            if not precedent_id or str(precedent_id).strip() == '':
                case_number = raw_data.get('?¬ê±´ë²ˆí˜¸', '')
                case_name = raw_data.get('?¬ê±´ëª?, '')
                if case_name:
                    precedent_id = f"{case_number}_{case_name}"
                else:
                    precedent_id = f"case_{case_number}"
                logger.debug(f"?€ì²?ID ?ì„±: {precedent_id}")
            
            # ì¹´í…Œê³ ë¦¬ ë¶„ë¥˜
            category = self.categorize_precedent(raw_data)
            
            # PrecedentData ê°ì²´ ?ì„±
            precedent_data = PrecedentData(
                precedent_id=str(precedent_id),
                case_name=raw_data.get('?¬ê±´ëª?, ''),
                case_number=raw_data.get('?¬ê±´ë²ˆí˜¸', ''),
                court=COURT_CODES.get(raw_data.get('ë²•ì›ì½”ë“œ', ''), ''),
                case_type=CASE_TYPE_CODES.get(raw_data.get('?¬ê±´? í˜•ì½”ë“œ', ''), ''),
                decision_date=raw_data.get('?ê²°?¼ì', '') or raw_data.get('? ê³ ?¼ì', ''),
                category=category,
                raw_data=raw_data
            )
            
            return precedent_data
            
        except Exception as e:
            logger.error(f"PrecedentData ?ì„± ?¤íŒ¨: {e}")
            logger.error(f"?ì‹œ ?°ì´?? {raw_data}")
            return None
    
    def _random_delay(self, min_seconds: Optional[float] = None, max_seconds: Optional[float] = None):
        """API ?”ì²­ ê°??œë¤ ì§€??(ê°œì„ ??ë²„ì „)"""
        min_delay = min_seconds or self.api_delay_range[0]
        max_delay = max_seconds or self.api_delay_range[1]
        delay = random.uniform(min_delay, max_delay)
        logger.debug(f"API ?”ì²­ ê°?{delay:.2f}ì´??€ê¸?..")
        time.sleep(delay)
    
    @contextmanager
    def _api_request_with_retry(self, operation_name: str):
        """API ?”ì²­ ?¬ì‹œ??ì»¨í…?¤íŠ¸ ë§¤ë‹ˆ?€ (ê°œì„ ??ë²„ì „)"""
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
    
    def save_precedent_data(self, precedent_data: Dict[str, Any], filename: str):
        """?ë? ?°ì´?°ë? ?Œì¼ë¡??€??(?œêµ­??ì²˜ë¦¬ ê°œì„ )"""
        filepath = self.output_dir / filename
        
        try:
            # ?Œì¼ëª…ì— ?œêµ­?´ê? ?¬í•¨??ê²½ìš° ?ˆì „?˜ê²Œ ì²˜ë¦¬
            safe_filename = filename.encode('utf-8').decode('utf-8')
            filepath = self.output_dir / safe_filename
            
            with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
                json.dump(precedent_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"?ë? ?°ì´???€?? {filepath}")
        except UnicodeEncodeError as e:
            logger.error(f"?Œì¼ëª??¸ì½”???¤ë¥˜: {e}")
            # ?ˆì „???Œì¼ëª…ìœ¼ë¡??¬ì‹œ??
            safe_filename = f"precedent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.output_dir / safe_filename
            with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
                json.dump(precedent_data, f, ensure_ascii=False, indent=2)
            logger.info(f"?ˆì „???Œì¼ëª…ìœ¼ë¡??€?? {filepath}")
        except Exception as e:
            logger.error(f"?ë? ?°ì´???€???¤íŒ¨: {e}")
    
    def collect_precedents_by_keyword(self, keyword: str, max_count: int = 100) -> List[PrecedentData]:
        """?¤ì›Œ?œë¡œ ?ë? ê²€??ë°??˜ì§‘ (ê°œì„ ??ë²„ì „)"""
        # ?´ë? ì²˜ë¦¬???¤ì›Œ?œì¸ì§€ ?•ì¸
        if keyword in self.processed_keywords:
            logger.info(f"?¤ì›Œ??'{keyword}'???´ë? ì²˜ë¦¬?˜ì—ˆ?µë‹ˆ?? ê±´ë„ˆ?ë‹ˆ??")
            return []
        
        logger.info(f"?¤ì›Œ??'{keyword}'ë¡??ë? ê²€???œì‘ (ëª©í‘œ: {max_count}ê±?")
        
        precedents = []
        
        # ?¤ì¤‘ ê²€???„ëµ ?ìš©
        search_strategies = [
            {"name": "ê¸°ë³¸ê²€??, "params": {"search": 1, "sort": "ddes"}},
            {"name": "? ì§œë²”ìœ„ê²€??, "params": {"search": 1, "sort": "ddes", "from_date": "20200101", "to_date": "20251231"}},
            {"name": "2025?„ê???, "params": {"search": 1, "sort": "ddes", "from_date": "20250101", "to_date": "20251231"}},
            {"name": "?€ë²•ì›ê²€??, "params": {"search": 1, "sort": "ddes", "court": "01"}},
            {"name": "ê³ ë“±ë²•ì›ê²€??, "params": {"search": 1, "sort": "ddes", "court": "02"}},
            {"name": "ì§€ë°©ë²•?ê???, "params": {"search": 1, "sort": "ddes", "court": "03"}},
        ]
        
        for strategy in search_strategies:
            if len(precedents) >= max_count:
                break
                
            strategy_name = strategy["name"]
            strategy_params = strategy["params"]
            remaining_count = max_count - len(precedents)
            
            logger.info(f"?¤ì›Œ??'{keyword}' - {strategy_name} ?„ëµ ?ìš© (?¨ì? ëª©í‘œ: {remaining_count}ê±?")
            
            strategy_precedents = self._search_with_strategy(
                keyword, remaining_count, strategy_name, strategy_params
            )
            
            precedents.extend(strategy_precedents)
            logger.info(f"{strategy_name} ?„ëµ?¼ë¡œ {len(strategy_precedents)}ê±?ì¶”ê? (ì´?{len(precedents)}ê±?")
    
    def _search_with_strategy(self, keyword: str, max_count: int, strategy_name: str, params: Dict[str, Any]) -> List[PrecedentData]:
        """?¹ì • ?„ëµ?¼ë¡œ ?ë? ê²€??""
        precedents = []
        page = 1
        consecutive_empty_pages = 0
        max_empty_pages = 2  # ?„ëµë³„ë¡œ?????ì? ?˜ì´ì§€ë¡??œí•œ
        
        while len(precedents) < max_count and consecutive_empty_pages < max_empty_pages:
            # ì¢…ë£Œ ?”ì²­ ?•ì¸
            if self._check_shutdown_requested():
                logger.warning(f"ì¢…ë£Œ ?”ì²­?¼ë¡œ '{keyword}' {strategy_name} ì¤‘ë‹¨: {self.shutdown_reason}")
                break
            
            try:
                # API ?”ì²­ ê°??œë¤ ì§€??
                if page > 1:
                    self._random_delay()
                
                # ì§„í–‰ ?í™© ë¡œê¹…
                logger.debug(f"?¤ì›Œ??'{keyword}' - {strategy_name} ?˜ì´ì§€ {page} ê²€??ì¤?..")
                
                # API ?”ì²­ ?¤í–‰
                try:
                    with self._api_request_with_retry(f"?¤ì›Œ??'{keyword}' {strategy_name} ê²€??):
                        results = self.client.get_precedent_list(
                            query=keyword,
                            display=100,
                            page=page,
                            **params
                        )
                except Exception as api_error:
                    logger.error(f"API ?”ì²­ ?¤íŒ¨: {api_error}")
                    consecutive_empty_pages += 1
                    page += 1
                    continue
                
                if not results:
                    consecutive_empty_pages += 1
                    logger.debug(f"?¤ì›Œ??'{keyword}' - {strategy_name} ?˜ì´ì§€ {page}?ì„œ ê²°ê³¼ ?†ìŒ (?°ì† ë¹??˜ì´ì§€: {consecutive_empty_pages})")
                    page += 1
                    continue
                else:
                    consecutive_empty_pages = 0
                
                # ê²°ê³¼ ì²˜ë¦¬
                new_count, duplicate_count = self._process_search_results(results, precedents, max_count)
                
                # ?˜ì´ì§€ë³?ê²°ê³¼ ë¡œê¹…
                logger.debug(f"{strategy_name} ?˜ì´ì§€ {page}: {new_count}ê±?? ê·œ, {duplicate_count}ê±?ì¤‘ë³µ (?„ì : {len(precedents)}/{max_count}ê±?")
                
                page += 1
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                if self._check_api_limits():
                    break
                    
            except KeyboardInterrupt:
                logger.warning(f"?¤ì›Œ??'{keyword}' {strategy_name} ê²€?‰ì´ ì¤‘ë‹¨?˜ì—ˆ?µë‹ˆ??")
                break
            except Exception as e:
                logger.error(f"?¤ì›Œ??'{keyword}' {strategy_name} ê²€??ì¤??¤ë¥˜: {e}")
                self.stats.failed_count += 1
                # ?¤ë¥˜ê°€ ë°œìƒ?´ë„ ?¤ìŒ ?„ëµ?¼ë¡œ ê³„ì† ì§„í–‰
                break
        
        # ?˜ì§‘???ë?ë¥??„ì‹œ ?€?¥ì†Œ??ì¶”ê?
        for precedent in precedents:
            self.pending_precedents.append(precedent)
            self.stats.collected_count += 1
        
        # ?¤ì›Œ??ì²˜ë¦¬ ?„ë£Œ ?œì‹œ
        self.processed_keywords.add(keyword)
        
        logger.info(f"?¤ì›Œ??'{keyword}' ?˜ì§‘ ?„ë£Œ: {len(precedents)}ê±?)
        return precedents
    
    def _collect_with_fallback_keywords(self, remaining_count: int, checkpoint_file: Path):
        """ë°±ì—… ?¤ì›Œ?œë¡œ ì¶”ê? ?˜ì§‘"""
        from scripts.precedent.precedent_models import FALLBACK_KEYWORDS
        
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
                # ë°±ì—… ?¤ì›Œ?œëŠ” ê°ê° 20ê±´ì”© ?˜ì§‘
                keyword_target = min(20, remaining_count)
                remaining_count -= keyword_target
                
                logger.info(f"ë°±ì—… ?¤ì›Œ??'{keyword}' ì²˜ë¦¬ ?œì‘ (ëª©í‘œ: {keyword_target}ê±?")
                
                # ?ë? ?˜ì§‘
                precedents = self.collect_precedents_by_keyword(keyword, keyword_target)
                
                # ë°°ì¹˜ ?€??
                if len(self.pending_precedents) >= self.batch_size:
                    self._save_batch_precedents()
                
                # ì²´í¬?¬ì¸???€??
                self._save_checkpoint(checkpoint_file)
                
                # ì§„í–‰ ?í™© ë¡œê¹…
                progress_percent = (self.stats.collected_count / self.stats.target_count) * 100
                logger.info(f"ë°±ì—… ?¤ì›Œ??'{keyword}' ?„ë£Œ: {len(precedents)}ê±??˜ì§‘ (ì´?{self.stats.collected_count:,}ê±? {progress_percent:.1f}%)")
                
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
        if len(self.pending_precedents) >= self.batch_size:
            self._save_batch_precedents()
        
        logger.info(f"?¤ì›Œ??'{keyword}' ?˜ì§‘ ?„ë£Œ: {len(precedents)}ê±?)
        return precedents
    
    def _process_search_results(self, results: List[Dict[str, Any]], precedents: List[PrecedentData], 
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
            if self._is_duplicate_precedent(result):
                duplicate_count += 1
                self.stats.duplicate_count += 1
                continue
            
            # PrecedentData ê°ì²´ ?ì„±
            precedent_data = self._create_precedent_data(result)
            if not precedent_data:
                self.stats.failed_count += 1
                continue
            
            # ? ê·œ ?ë? ì¶”ê?
            precedents.append(precedent_data)
            self._mark_precedent_collected(result)
            new_count += 1
            
            if len(precedents) >= max_count:
                break
        
        return new_count, duplicate_count
    
    def _save_batch_precedents(self):
        """ë°°ì¹˜ ?¨ìœ„ë¡??ë? ?€??""
        if not self.pending_precedents:
            return
        
        try:
            # ì¹´í…Œê³ ë¦¬ë³„ë¡œ ê·¸ë£¹??
            by_category = {}
            for precedent in self.pending_precedents:
                category = precedent.category.value
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(precedent)
            
            # ì¹´í…Œê³ ë¦¬ë³??Œì¼ ?€??
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            saved_files = []
            
            for category, precedents in by_category.items():
                # ?ˆì „???Œì¼ëª??ì„± (?œêµ­??ì²˜ë¦¬)
                safe_category = category.replace('_', '-')  # ?¸ë”?¤ì½”?´ë? ?˜ì´?ˆìœ¼ë¡?ë³€ê²?
                filename = f"batch_{safe_category}_{len(precedents)}ê±?{timestamp}.json"
                filepath = self.output_dir / filename
                
                batch_data = {
                    'metadata': {
                        'category': category,
                        'count': len(precedents),
                        'saved_at': datetime.now().isoformat(),
                        'batch_id': timestamp
                    },
                    'precedents': [p.raw_data for p in precedents]
                }
                
                try:
                    with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
                        json.dump(batch_data, f, ensure_ascii=False, indent=2)
                except UnicodeEncodeError:
                    # ?Œì¼ëª??¸ì½”???¤ë¥˜ ???ˆì „???Œì¼ëª??¬ìš©
                    safe_filename = f"batch_{safe_category}_{len(precedents)}ê±?{timestamp}.json"
                    filepath = self.output_dir / safe_filename
                    with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
                        json.dump(batch_data, f, ensure_ascii=False, indent=2)
                
                saved_files.append(filepath)
                logger.info(f"ë°°ì¹˜ ?€???„ë£Œ: {category} ì¹´í…Œê³ ë¦¬ {len(precedents):,}ê±?-> {filename}")
            
            # ?µê³„ ?…ë°?´íŠ¸
            self.stats.saved_count += len(self.pending_precedents)
            
            # ?„ì‹œ ?€?¥ì†Œ ì´ˆê¸°??
            self.pending_precedents = []
            
            logger.info(f"ë°°ì¹˜ ?€???„ë£Œ: ì´?{len(saved_files):,}ê°??Œì¼")
            
        except Exception as e:
            logger.error(f"ë°°ì¹˜ ?€???¤íŒ¨: {e}")
            logger.error(traceback.format_exc())
    
    def categorize_precedent(self, precedent: Dict[str, Any]) -> PrecedentCategory:
        """?ë? ì¹´í…Œê³ ë¦¬ ë¶„ë¥˜ (ê°œì„ ??ë²„ì „)"""
        case_type_code = precedent.get('?¬ê±´? í˜•ì½”ë“œ', '')
        case_name = precedent.get('?¬ê±´ëª?, '').lower()
        
        # ?¤ì›Œ??ê¸°ë°˜ ë¶„ë¥˜ (?°ì„ ?œìœ„ ??
        category_keywords = {
            PrecedentCategory.CIVIL_CONTRACT: [
                'ê³„ì•½', '?í•´ë°°ìƒ', 'ë¶ˆë²•?‰ìœ„', 'ì±„ë¬´', 'ì±„ê¶Œ', 'ê³„ì•½?´ì?', '?„ì•½ê¸?
            ],
            PrecedentCategory.CIVIL_PROPERTY: [
                '?Œìœ ê¶?, '?ìœ ê¶?, 'ë¬¼ê¶Œ', '?€?¹ê¶Œ', 'ì§ˆê¶Œ', '? ì¹˜ê¶?, 'ì§€?ê¶Œ', '?„ì„¸ê¶?
            ],
            PrecedentCategory.CIVIL_FAMILY: [
                '?ì†', '?¼ì¸', '?´í˜¼', 'ì¹œìê´€ê³?, '?‘ìœ¡', 'ë¶€??, 'ê°€ì¡?, 'ì¹œì¡±'
            ],
            PrecedentCategory.CRIMINAL: [
                '?ˆë„', 'ê°•ë„', '?¬ê¸°', '?¡ë ¹', 'ë°°ì„', 'ê°•ê°„', 'ê°•ì œì¶”í–‰', '?´ì¸', '?í•´',
                'êµí†µ?¬ê³ ', '?Œì£¼?´ì „', 'ë§ˆì•½', '?„ë°•', '??–‰', '?‘ë°•', 'ê°•ìš”'
            ],
            PrecedentCategory.ADMINISTRATIVE: [
                '?‰ì •ì²˜ë¶„', '?ˆê?', '?¸ê?', '? ê³ ', '? ì²­', '?´ì˜? ì²­', '?‰ì •?¬íŒ', '?‰ì •?Œì†¡',
                'êµ?„¸', 'ì§€ë°©ì„¸', 'ë¶€?™ì‚°?±ê¸°', 'ê±´ì¶•?ˆê?', '?˜ê²½?í–¥?‰ê?'
            ],
            PrecedentCategory.COMMERCIAL: [
                '?Œì‚¬', 'ì£¼ì‹', 'ì£¼ì£¼', '?´ì‚¬', 'ê°ì‚¬', '?©ë³‘', 'ë¶„í• ', '?´ì‚°', 'ì²?‚°',
                '?´ìŒ', '?˜í‘œ', 'ë³´í—˜', '?´ìƒ', '??³µ', '?´ì†¡'
            ],
            PrecedentCategory.LABOR: [
                'ê·¼ë¡œê³„ì•½', '?„ê¸ˆ', 'ê·¼ë¡œ?œê°„', '?´ê²Œ?œê°„', '?´ì¼', '?°ì°¨? ê¸‰?´ê?', '?´ê³ ',
                '?´ì§ê¸?, '?°ì—…?¬í•´', '?°ì—…?ˆì „', '?¸ë™ì¡°í•©', '?¨ì²´êµì„­', '?Œì—…'
            ],
            PrecedentCategory.INTELLECTUAL_PROPERTY: [
                '?¹í—ˆ', '?¤ìš©? ì•ˆ', '?”ì??, '?í‘œ', '?€?‘ê¶Œ', '?ì—…ë¹„ë?', 'ë¶€?•ê²½??,
                '?¼ì´? ìŠ¤', 'ê¸°ìˆ ?´ì „', 'ë°œëª…', 'ì°½ì‘ë¬?
            ],
            PrecedentCategory.CONSUMER: [
                '?Œë¹„??, 'ê³„ì•½', '?½ê?', '?œì‹œê´‘ê³ ', '? ë?ê±°ë˜', 'ë°©ë¬¸?ë§¤', '?¤ë‹¨ê³„íŒë§?,
                '?µì‹ ?ë§¤', '?„ì?ê±°??, 'ê°œì¸?•ë³´', '?•ë³´ê³µê°œ'
            ],
            PrecedentCategory.ENVIRONMENT: [
                '?˜ê²½', '?€ê¸?, '?˜ì§ˆ', '? ì–‘', '?ŒìŒ', 'ì§„ë™', '?…ì·¨', '?ê¸°ë¬?,
                '?˜ê²½?í–¥?‰ê?', '?˜ê²½?¤ì—¼', '?íƒœê³?, '?ì—°?˜ê²½'
            ]
        }
        
        # ?¤ì›Œ??ë§¤ì¹­?¼ë¡œ ì¹´í…Œê³ ë¦¬ ê²°ì •
        for category, keywords in category_keywords.items():
            if any(keyword in case_name for keyword in keywords):
                return category
        
        # ?¬ê±´? í˜•ì½”ë“œ ê¸°ë°˜ ë¶„ë¥˜
        case_type_mapping = {
            '01': PrecedentCategory.CIVIL_CONTRACT,  # ë¯¼ì‚¬
            '02': PrecedentCategory.CRIMINAL,        # ?•ì‚¬
            '03': PrecedentCategory.ADMINISTRATIVE,  # ?‰ì •
            '04': PrecedentCategory.CIVIL_FAMILY,    # ê°€??
            '05': PrecedentCategory.OTHER            # ?¹ë³„ë²?
        }
        
        return case_type_mapping.get(case_type_code, PrecedentCategory.OTHER)
    
    def collect_all_precedents(self, target_count: int = 5000):
        """ëª¨ë“  ?ë? ?˜ì§‘ (ê°œì„ ??ë²„ì „)"""
        self.stats.target_count = target_count
        self.stats.status = CollectionStatus.IN_PROGRESS
        
        # ì²´í¬?¬ì¸???Œì¼ ?¤ì •
        checkpoint_file = self._setup_checkpoint_file()
        
        try:
            logger.info(f"?ë? ?˜ì§‘ ?œì‘ - ëª©í‘œ: {target_count}ê±?)
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
            logger.error(f"?ë? ?˜ì§‘ ì¤??¤ë¥˜ ë°œìƒ: {e}")
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
        """?°ì„ ?œìœ„ ê¸°ë°˜ ?¤ì›Œ?œë³„ ?ë? ?˜ì§‘ (?´ë? ì²˜ë¦¬???¤ì›Œ??ê±´ë„ˆ?°ê¸°)"""
        # ?°ì„ ?œìœ„ ?¤ì›Œ?œì? ?¼ë°˜ ?¤ì›Œ??ë¶„ë¦¬
        priority_keywords = [kw for kw in PRIORITY_KEYWORDS if kw in SEARCH_KEYWORDS]
        remaining_keywords = [kw for kw in SEARCH_KEYWORDS if kw not in PRIORITY_KEYWORDS]
        
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
        
        # ì§„í–‰ ?í™© ì¶”ì  (tqdm ?€??ë¡œê·¸ ê¸°ë°˜)
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
                precedents = self.collect_precedents_by_keyword(keyword, keyword_target)
                
                # ?µê³„ ?…ë°?´íŠ¸
                self.stats.keywords_processed = len(self.processed_keywords)
                
                # ì²´í¬?¬ì¸???€??(ë§??¤ì›Œ?œë§ˆ??
                self._save_checkpoint(checkpoint_file)
                
                # ?¤ì›Œ???„ë£Œ ë¡œê¹… (?°ì„ ?œìœ„ ?•ë³´ ?¬í•¨)
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
        """ìµœì¢… ?µê³„ ì¶œë ¥ (?«ì ?¬ë§·??ê°œì„ )"""
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
        """ì§„í–‰ ?í™© ì²´í¬?¬ì¸???€??(ê°œì„ ??ë²„ì „)"""
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
                    'collected_precedents_count': len(self.collected_precedents)
                },
                'precedents': [p.raw_data for p in self.pending_precedents],
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
