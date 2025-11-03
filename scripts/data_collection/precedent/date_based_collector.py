#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
? ì§œ ê¸°ë°˜ ?ë? ?˜ì§‘ê¸?

??ëª¨ë“ˆ?€ ? ì§œë³„ë¡œ ì²´ê³„?ì¸ ?ë? ?˜ì§‘???˜í–‰?©ë‹ˆ??
- ?°ë„ë³? ë¶„ê¸°ë³? ?”ë³„, ì£¼ë³„ ?˜ì§‘ ?„ëµ
- ? ê³ ?¼ì ?´ë¦¼ì°¨ìˆœ ìµœì ??
- ?´ë”ë³?raw ?°ì´???€??êµ¬ì¡°
- ì¤‘ë³µ ë°©ì? ë°?ì²´í¬?¬ì¸??ì§€??
"""

import json
import time
import random
import hashlib
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

import sys
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient, LawOpenAPIConfig
from scripts.precedent.precedent_models import (
    CollectionStats, PrecedentData, CollectionStatus, PrecedentCategory,
    COURT_CODES, CASE_TYPE_CODES
)
import logging

logger = logging.getLogger(__name__)


class DateCollectionStrategy(Enum):
    """? ì§œ ?˜ì§‘ ?„ëµ ?´ê±°??""
    YEARLY = "yearly"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    WEEKLY = "weekly"
    DAILY = "daily"


@dataclass
class DateCollectionConfig:
    """? ì§œ ?˜ì§‘ ?¤ì • ?´ë˜??""
    strategy: DateCollectionStrategy
    start_date: str
    end_date: str
    target_count: int
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: int = 5
    api_delay_range: Tuple[float, float] = (1.0, 3.0)
    output_subdir: Optional[str] = None


class DateBasedPrecedentCollector:
    """? ì§œ ê¸°ë°˜ ?ë? ?˜ì§‘ ?´ë˜??""
    
    def __init__(self, config: LawOpenAPIConfig, base_output_dir: Optional[Path] = None, 
                 include_details: bool = True):
        """
        ? ì§œ ê¸°ë°˜ ?ë? ?˜ì§‘ê¸?ì´ˆê¸°??
        
        Args:
            config: API ?¤ì • ê°ì²´
            base_output_dir: ê¸°ë³¸ ì¶œë ¥ ?”ë ‰? ë¦¬ (ê¸°ë³¸ê°? data/raw/precedents)
            include_details: ?ë?ë³¸ë¬¸ ?¬í•¨ ?¬ë? (ê¸°ë³¸ê°? True)
        """
        self.client = LawOpenAPIClient(config)
        self.base_output_dir = base_output_dir or Path("data/raw/precedents")
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.include_details = include_details  # ?ë?ë³¸ë¬¸ ?˜ì§‘ ?¬ë?
        
        # ?°ì´??ê´€ë¦?(ë©”ëª¨ë¦?ìµœì ??
        self.collected_precedents: Set[str] = set()
        self.processed_date_ranges: Set[str] = set()
        self.pending_precedents: List[PrecedentData] = []
        self.max_memory_precedents = 50000  # ìµœë? ë©”ëª¨ë¦?ë³´ê? ê±´ìˆ˜
        
        # ?µê³„ ë°??íƒœ
        self.stats = CollectionStats()
        self.stats.status = CollectionStatus.PENDING
        
        # ?ëŸ¬ ì²˜ë¦¬
        self.error_count = 0
        self.max_errors = 50
        
        # ?œê°„ ?¸í„°ë²??¤ì • (ê¸°ë³¸ê°?
        self.request_interval_base = 2.0  # ê¸°ë³¸ ê°„ê²©
        self.request_interval_range = 2.0  # ê°„ê²© ë²”ìœ„
        
        # ê¸°ì¡´ ?˜ì§‘???°ì´??ë¡œë“œ
        self._load_existing_data()
        
        logger.info(f"? ì§œ ê¸°ë°˜ ?ë? ?˜ì§‘ê¸?ì´ˆê¸°???„ë£Œ (?ë?ë³¸ë¬¸ ?¬í•¨: {include_details})")
    
    def set_request_interval(self, base_interval: float, interval_range: float):
        """API ?”ì²­ ê°„ê²© ?¤ì •"""
        self.request_interval_base = base_interval
        self.request_interval_range = interval_range
        logger.info(f"?±ï¸ ?”ì²­ ê°„ê²© ?¤ì •: {base_interval:.1f} Â± {interval_range:.1f}ì´?)
    
    def _load_existing_data(self, target_year: Optional[int] = None):
        """ê¸°ì¡´ ?˜ì§‘???°ì´??ë¡œë“œ?˜ì—¬ ì¤‘ë³µ ë°©ì? (ë©”ëª¨ë¦?ìµœì ??"""
        logger.info("ê¸°ì¡´ ?˜ì§‘???°ì´???•ì¸ ì¤?..")
        
        loaded_count = 0
        error_count = 0
        
        # ëª¨ë“  ?˜ìœ„ ?”ë ‰? ë¦¬?ì„œ ?°ì´??ë¡œë“œ (ìµœì‹  ?´ë” ?°ì„ )
        subdirs = sorted([d for d in self.base_output_dir.iterdir() if d.is_dir()], 
                        key=lambda x: x.name, reverse=True)
        
        for subdir in subdirs:
            if len(self.collected_precedents) >= self.max_memory_precedents:
                logger.info(f"? ï¸ ë©”ëª¨ë¦??œê³„ ?„ë‹¬: {len(self.collected_precedents):,}ê±? ì¶”ê? ë¡œë“œ ì¤‘ë‹¨")
                break
                
                for file_path in subdir.glob("*.json"):
                    try:
                        loaded_count += self._load_precedents_from_file(file_path, target_year)
                        
                        # ë©”ëª¨ë¦??¬ìš©??ì²´í¬
                        if len(self.collected_precedents) >= self.max_memory_precedents:
                            logger.info(f"? ï¸ ë©”ëª¨ë¦??œê³„ ?„ë‹¬: {len(self.collected_precedents):,}ê±? ì¶”ê? ë¡œë“œ ì¤‘ë‹¨")
                            break
                            
                    except Exception as e:
                        error_count += 1
                        logger.debug(f"?Œì¼ ë¡œë“œ ?¤íŒ¨ {file_path}: {e}")
            
            if len(self.collected_precedents) >= self.max_memory_precedents:
                break
        
        logger.info(f"ê¸°ì¡´ ?°ì´??ë¡œë“œ ?„ë£Œ: {loaded_count:,}ê±? ?¤ë¥˜: {error_count:,}ê±?)
        self.stats.collected_count = len(self.collected_precedents)
        logger.info(f"ì¤‘ë³µ ë°©ì?ë¥??„í•œ ?ë? ID {len(self.collected_precedents):,}ê°?ë¡œë“œ??(ë©”ëª¨ë¦?ìµœì ??")
    
    def _load_precedents_from_file(self, file_path: Path, target_year: Optional[int] = None) -> int:
        """?Œì¼?ì„œ ?ë? ?°ì´??ë¡œë“œ (?¹ì • ?°ë„ ?„í„°ë§?"""
        loaded_count = 0
        
        try:
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
            
            # ?ë? ID ì¶”ì¶œ (?¹ì • ?°ë„ ?„í„°ë§?
            for precedent in precedents:
                if isinstance(precedent, dict):
                    # ?¹ì • ?°ë„ê°€ ì§€?•ëœ ê²½ìš° ?´ë‹¹ ?°ë„???ë?ë§?ë¡œë“œ
                    if target_year:
                        decision_date = precedent.get('? ê³ ?¼ì', '') or precedent.get('?ê²°?¼ì', '')
                        if decision_date:
                            try:
                                # ? ì§œ ?Œì‹± (YYYY.MM.DD ?•ì‹)
                                if '.' in decision_date:
                                    date_parts = decision_date.split('.')
                                    if len(date_parts) >= 1:
                                        precedent_year = int(date_parts[0])
                                        if precedent_year != target_year:
                                            continue  # ?¤ë¥¸ ?°ë„??ê±´ë„ˆ?°ê¸°
                                else:
                                    continue  # ? ì§œ ?•ì‹???˜ëª»??ê²½ìš° ê±´ë„ˆ?°ê¸°
                            except (ValueError, IndexError):
                                continue  # ? ì§œ ?Œì‹± ?¤ë¥˜ ??ê±´ë„ˆ?°ê¸°
                        else:
                            continue  # ? ì§œê°€ ?†ëŠ” ê²½ìš° ê±´ë„ˆ?°ê¸°
                    
                    precedent_id = precedent.get('?ë??¼ë ¨ë²ˆí˜¸') or precedent.get('precedent_id')
                    if precedent_id:
                        self.collected_precedents.add(str(precedent_id))
                        loaded_count += 1
        
        except Exception as e:
            logger.debug(f"?Œì¼ ë¡œë“œ ?¤íŒ¨ {file_path}: {e}")
        
        return loaded_count
    
    def _create_output_subdir(self, strategy: DateCollectionStrategy, date_range: str) -> Path:
        """ì¶œë ¥ ?˜ìœ„ ?”ë ‰? ë¦¬ ?ì„±"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # ?„ëµë³??”ë ‰? ë¦¬ êµ¬ì¡°
        if strategy == DateCollectionStrategy.YEARLY:
            subdir_name = f"yearly_{date_range}_{timestamp}"
        elif strategy == DateCollectionStrategy.QUARTERLY:
            subdir_name = f"quarterly_{date_range}_{timestamp}"
        elif strategy == DateCollectionStrategy.MONTHLY:
            subdir_name = f"monthly_{date_range}_{timestamp}"
        elif strategy == DateCollectionStrategy.WEEKLY:
            subdir_name = f"weekly_{date_range}_{timestamp}"
        elif strategy == DateCollectionStrategy.DAILY:
            subdir_name = f"daily_{date_range}_{timestamp}"
        else:
            subdir_name = f"date_based_{date_range}_{timestamp}"
        
        output_dir = self.base_output_dir / subdir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def _is_duplicate_precedent(self, precedent: Dict[str, Any]) -> bool:
        """?ë? ì¤‘ë³µ ?¬ë? ?•ì¸ (ê°œì„ ??ë¡œì§)"""
        precedent_id = (
            precedent.get('?ë??¼ë ¨ë²ˆí˜¸') or 
            precedent.get('precedent_id') or 
            precedent.get('prec id') or
            precedent.get('id')
        )
        
        # ?ë??¼ë ¨ë²ˆí˜¸ë¡?ì¤‘ë³µ ?•ì¸
        if precedent_id and str(precedent_id) in self.collected_precedents:
            return True
        
        # ?€ì²??ë³„?ë¡œ ?•ì¸ (???„ê²©??ì¡°ê±´)
        case_number = precedent.get('?¬ê±´ë²ˆí˜¸', '')
        case_name = precedent.get('?¬ê±´ëª?, '')
        decision_date = precedent.get('? ê³ ?¼ì', '')
        
        # ?¬ê±´ë²ˆí˜¸, ?¬ê±´ëª? ? ê³ ?¼ìê°€ ëª¨ë‘ ?¼ì¹˜?˜ëŠ” ê²½ìš°ë§?ì¤‘ë³µ?¼ë¡œ ì²˜ë¦¬
        if case_number and case_name and decision_date:
            alternative_id = f"{case_number}_{case_name}_{decision_date}"
            if alternative_id in self.collected_precedents:
                return True
        
        return False
    
    def _mark_precedent_collected(self, precedent: Dict[str, Any]):
        """?ë?ë¥??˜ì§‘?¨ìœ¼ë¡??œì‹œ (ë©”ëª¨ë¦?ìµœì ??"""
        precedent_id = (
            precedent.get('?ë??¼ë ¨ë²ˆí˜¸') or 
            precedent.get('precedent_id') or 
            precedent.get('prec id') or
            precedent.get('id')
        )
        
        if precedent_id:
            self.collected_precedents.add(str(precedent_id))
        
        # ?€ì²??ë³„?ë¡œ???€??
        case_number = precedent.get('?¬ê±´ë²ˆí˜¸', '')
        case_name = precedent.get('?¬ê±´ëª?, '')
        decision_date = precedent.get('? ê³ ?¼ì', '')
        
        if case_number and case_name and decision_date:
            alternative_id = f"{case_number}_{case_name}_{decision_date}"
            self.collected_precedents.add(alternative_id)
        
        # ë©”ëª¨ë¦??¬ìš©??ì²´í¬ ë°?ìµœì ??
        self._check_memory_usage()
    
    def _check_memory_usage(self):
        """ë©”ëª¨ë¦??¬ìš©??ì²´í¬ ë°?ìµœì ??""
        if len(self.collected_precedents) > self.max_memory_precedents:
            logger.warning(f"? ï¸ ë©”ëª¨ë¦??¬ìš©??ì´ˆê³¼: {len(self.collected_precedents):,}ê±?> {self.max_memory_precedents:,}ê±?)
            logger.info("?”„ ë©”ëª¨ë¦?ìµœì ?”ë? ?„í•´ ì¤‘ë³µ ?°ì´???¼ë? ?•ë¦¬ ì¤?..")
            
            # ?¤ë˜???°ì´???¼ë? ?œê±° (ìµœì‹  ?°ì´???°ì„  ë³´ì¡´)
            items_to_remove = len(self.collected_precedents) - self.max_memory_precedents
            items_list = list(self.collected_precedents)
            
            # ?ë??¼ë ¨ë²ˆí˜¸??ë³´ì¡´?˜ê³  ?€ì²??ë³„?ë????œê±°
            precedent_ids = [item for item in items_list if item.isdigit()]
            alternative_ids = [item for item in items_list if not item.isdigit()]
            
            # ?€ì²??ë³„?ë????œê±°
            removed_count = 0
            for alt_id in alternative_ids[:items_to_remove]:
                self.collected_precedents.discard(alt_id)
                removed_count += 1
            
            # ?¬ì „??ì´ˆê³¼?˜ë©´ ?ë??¼ë ¨ë²ˆí˜¸???œê±°
            if len(self.collected_precedents) > self.max_memory_precedents:
                remaining_to_remove = len(self.collected_precedents) - self.max_memory_precedents
                for prec_id in precedent_ids[:remaining_to_remove]:
                    self.collected_precedents.discard(prec_id)
                    removed_count += 1
            
            logger.info(f"??ë©”ëª¨ë¦?ìµœì ???„ë£Œ: {removed_count:,}ê±??œê±°, ?„ì¬ {len(self.collected_precedents):,}ê±?ë³´ê?")
    
    def _create_precedent_data(self, raw_data: Dict[str, Any]) -> Optional[PrecedentData]:
        """?ì‹œ ?°ì´?°ì—??PrecedentData ê°ì²´ ?ì„±"""
        try:
            # ?ë? ID ì¶”ì¶œ
            precedent_id = (
                raw_data.get('?ë??¼ë ¨ë²ˆí˜¸') or 
                raw_data.get('precedent_id') or 
                raw_data.get('prec id') or
                raw_data.get('id')
            )
            
            if not precedent_id:
                case_number = raw_data.get('?¬ê±´ë²ˆí˜¸', '')
                case_name = raw_data.get('?¬ê±´ëª?, '')
                if case_name:
                    precedent_id = f"{case_number}_{case_name}"
                else:
                    precedent_id = f"case_{case_number}"
            
            # ì¹´í…Œê³ ë¦¬ ë¶„ë¥˜
            category = self._categorize_precedent(raw_data)
            
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
            return None
    
    def _create_precedent_data_with_detail(self, raw_data: Dict[str, Any]) -> Optional[PrecedentData]:
        """?ì‹œ ?°ì´?°ì—??PrecedentData ê°ì²´ ?ì„± (?ë?ë³¸ë¬¸ ?¬í•¨)"""
        try:
            # ê¸°ë³¸ PrecedentData ?ì„±
            precedent_data = self._create_precedent_data(raw_data)
            if not precedent_data:
                return None
            
            # ?ë?ë³¸ë¬¸ ?˜ì§‘ (?¬ì‹œ??ë©”ì»¤?ˆì¦˜ ?¬í•¨)
            precedent_id = raw_data.get('?ë??¼ë ¨ë²ˆí˜¸')
            if precedent_id:
                logger.info(f"?” ?ë?ë³¸ë¬¸ ?˜ì§‘ ?œì‘: {raw_data.get('?¬ê±´ëª?, 'Unknown')} (ID: {precedent_id})")
                detail_info = self._collect_precedent_detail_with_retry(precedent_id)
                precedent_data.detail_info = detail_info
                logger.info(f"???ë?ë³¸ë¬¸ ?˜ì§‘ ?„ë£Œ: {raw_data.get('?¬ê±´ëª?, 'Unknown')} (ID: {precedent_id})")
            else:
                logger.warning(f"? ï¸ ?ë??¼ë ¨ë²ˆí˜¸ê°€ ?†ì–´ ?ë?ë³¸ë¬¸ ?˜ì§‘ ë¶ˆê?: {raw_data.get('?¬ê±´ëª?, 'Unknown')}")
                precedent_data.detail_info = {}
            
            return precedent_data
            
        except Exception as e:
            logger.error(f"PrecedentData ?ì„± ?¤íŒ¨ (?ë?ë³¸ë¬¸ ?¬í•¨): {e}")
            return None
    
    def _collect_precedent_detail_with_retry(self, precedent_id: str, max_retries: int = 3) -> Dict[str, Any]:
        """?ë?ë³¸ë¬¸ ?˜ì§‘ (?¬ì‹œ??ë©”ì»¤?ˆì¦˜ ?¬í•¨)"""
        import time
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"?ë?ë³¸ë¬¸ ?˜ì§‘ ?œë„ {attempt + 1}/{max_retries}: {precedent_id}")
                
                # API ?”ì²­ ê°?ì§€??(API ë¶€??ë°©ì?)
                if attempt > 0:
                    delay = min(2 ** attempt, 10)  # ì§€??ë°±ì˜¤?? ìµœë? 10ì´?
                    logger.info(f"API ?¬ì‹œ????{delay}ì´??€ê¸?..")
                    time.sleep(delay)
                else:
                    # ì²?ë²ˆì§¸ ?”ì²­??0.5ì´?ì§€??(API ë¶€??ë°©ì?, ?ë„ ê°œì„ )
                    time.sleep(0.5)
                
                detail_response = self.client.get_precedent_detail(precedent_id=precedent_id)
                
                if detail_response:
                    # ?ë?ë³¸ë¬¸ ?•ë³´ ì¶”ì¶œ
                    detail_info = self._extract_precedent_detail(detail_response)
                    logger.info(f"???ë?ë³¸ë¬¸ ?˜ì§‘ ?„ë£Œ: {precedent_id}")
                    return detail_info
                else:
                    logger.warning(f"?ë?ë³¸ë¬¸ API ?‘ë‹µ ?†ìŒ (?œë„ {attempt + 1}/{max_retries}): {precedent_id}")
                    
            except Exception as e:
                logger.warning(f"?ë?ë³¸ë¬¸ ?˜ì§‘ ?¤ë¥˜ (?œë„ {attempt + 1}/{max_retries}): {precedent_id} - {e}")
                
                # ë§ˆì?ë§??œë„ê°€ ?„ë‹ˆë©?ê³„ì†
                if attempt < max_retries - 1:
                    continue
                else:
                    logger.error(f"?ë?ë³¸ë¬¸ ?˜ì§‘ ìµœì¢… ?¤íŒ¨: {precedent_id} - {e}")
        
        # ëª¨ë“  ?œë„ ?¤íŒ¨ ??ë¹??•ì…”?ˆë¦¬ ë°˜í™˜
        logger.error(f"?ë?ë³¸ë¬¸ ?˜ì§‘ ?„ì „ ?¤íŒ¨: {precedent_id} (ìµœë? ?¬ì‹œ???Ÿìˆ˜ ì´ˆê³¼)")
        return {'?¤ë¥˜': f'?ë?ë³¸ë¬¸ ?˜ì§‘ ?¤íŒ¨ (?¬ì‹œ??{max_retries}??ì´ˆê³¼)', '?ë??¼ë ¨ë²ˆí˜¸': precedent_id}
    
    def _extract_precedent_detail(self, detail_response: Dict[str, Any]) -> Dict[str, Any]:
        """?ë? ?ì„¸ ?‘ë‹µ?ì„œ ?•ë³´ ì¶”ì¶œ"""
        try:
            extracted_info = {}
            
            # ?¤ì–‘??API ?‘ë‹µ êµ¬ì¡° ì²˜ë¦¬
            if 'PrecService' in detail_response:
                prec_service = detail_response['PrecService']
                logger.debug(f"PrecService êµ¬ì¡° ë°œê²¬: {type(prec_service)}")
                
                # ?¤ì œ API ?‘ë‹µ êµ¬ì¡°??ë§ê²Œ ?˜ì •
                # PrecService??ì§ì ‘ ?ë? ?•ë³´ê°€ ?ˆìŒ (ë°°ì—´???„ë‹˜)
                if isinstance(prec_service, dict):
                    # ì£¼ìš” ?•ë³´ ì¶”ì¶œ
                    extracted_info = {
                        '?ë??¼ë ¨ë²ˆí˜¸': prec_service.get('?ë??•ë³´?¼ë ¨ë²ˆí˜¸', ''),
                        '?¬ê±´ëª?: prec_service.get('?¬ê±´ëª?, ''),
                        '?¬ê±´ë²ˆí˜¸': prec_service.get('?¬ê±´ë²ˆí˜¸', ''),
                        'ë²•ì›ëª?: prec_service.get('ë²•ì›ëª?, ''),
                        '? ê³ ?¼ì': prec_service.get('? ê³ ?¼ì', ''),
                        '?ê²°? í˜•': prec_service.get('?ê²°? í˜•', ''),
                        '?¬ê±´? í˜•': prec_service.get('?¬ê±´ì¢…ë¥˜ëª?, ''),
                        '?ê²°?”ì?': prec_service.get('?ê²°?”ì?', ''),
                        '?ì‹œ?¬í•­': prec_service.get('?ì‹œ?¬í•­', ''),
                        'ì°¸ì¡°ì¡°ë¬¸': prec_service.get('ì°¸ì¡°ì¡°ë¬¸', ''),
                        'ì°¸ì¡°?ë?': prec_service.get('ì°¸ì¡°?ë?', ''),
                        '?ë??´ìš©': prec_service.get('?ë??´ìš©', ''),
                        '?¬ê±´ì¢…ë¥˜ì½”ë“œ': prec_service.get('?¬ê±´ì¢…ë¥˜ì½”ë“œ', ''),
                        'ë²•ì›ì¢…ë¥˜ì½”ë“œ': prec_service.get('ë²•ì›ì¢…ë¥˜ì½”ë“œ', ''),
                        '? ê³ ': prec_service.get('? ê³ ', '')
                    }
                    
                    # ì¶”ê? ë©”í??°ì´??(ì¤‘ë³µ ?œê±°)
                    extracted_info['?˜ì§‘?¼ì‹œ'] = datetime.now().isoformat()
                    # API_?‘ë‹µ_?ë³¸?€ ?€?¥í•˜ì§€ ?ŠìŒ (ì¤‘ë³µ ?°ì´??ë°©ì?)
                    
                    logger.info(f"?ë?ë³¸ë¬¸ ?•ë³´ ì¶”ì¶œ ?±ê³µ: {extracted_info.get('?¬ê±´ëª?, 'Unknown')}")
                else:
                    logger.warning(f"PrecServiceê°€ ?•ì…”?ˆë¦¬ê°€ ?„ë‹˜: {type(prec_service)}")
                    
            elif 'Law' in detail_response:
                # Law êµ¬ì¡° ì²˜ë¦¬ (êµ?„¸ì²??ë? ??
                law_data = detail_response['Law']
                logger.debug(f"Law êµ¬ì¡° ë°œê²¬ (êµ?„¸ì²??ë?): {type(law_data)}")
                
                if isinstance(law_data, dict):
                    # Law êµ¬ì¡°?ì„œ ?•ë³´ ì¶”ì¶œ ?œë„
                    extracted_info = {
                        '?ë??¼ë ¨ë²ˆí˜¸': law_data.get('?ë??•ë³´?¼ë ¨ë²ˆí˜¸', ''),
                        '?¬ê±´ëª?: law_data.get('?¬ê±´ëª?, ''),
                        '?¬ê±´ë²ˆí˜¸': law_data.get('?¬ê±´ë²ˆí˜¸', ''),
                        'ë²•ì›ëª?: law_data.get('ë²•ì›ëª?, ''),
                        '? ê³ ?¼ì': law_data.get('? ê³ ?¼ì', ''),
                        '?ê²°? í˜•': law_data.get('?ê²°? í˜•', ''),
                        '?¬ê±´? í˜•': law_data.get('?¬ê±´ì¢…ë¥˜ëª?, ''),
                        '?ê²°?”ì?': law_data.get('?ê²°?”ì?', ''),
                        '?ì‹œ?¬í•­': law_data.get('?ì‹œ?¬í•­', ''),
                        'ì°¸ì¡°ì¡°ë¬¸': law_data.get('ì°¸ì¡°ì¡°ë¬¸', ''),
                        'ì°¸ì¡°?ë?': law_data.get('ì°¸ì¡°?ë?', ''),
                        '?ë??´ìš©': law_data.get('?ë??´ìš©', ''),
                        '?¬ê±´ì¢…ë¥˜ì½”ë“œ': law_data.get('?¬ê±´ì¢…ë¥˜ì½”ë“œ', ''),
                        'ë²•ì›ì¢…ë¥˜ì½”ë“œ': law_data.get('ë²•ì›ì¢…ë¥˜ì½”ë“œ', ''),
                        '? ê³ ': law_data.get('? ê³ ', ''),
                        '?°ì´?°ì¶œì²?: 'êµ?„¸ì²?
                    }
                    
                    # ì¶”ê? ë©”í??°ì´??(ì¤‘ë³µ ?œê±°)
                    extracted_info['?˜ì§‘?¼ì‹œ'] = datetime.now().isoformat()
                    # API_?‘ë‹µ_?ë³¸?€ ?€?¥í•˜ì§€ ?ŠìŒ (ì¤‘ë³µ ?°ì´??ë°©ì?)
                    
                    logger.info(f"?ë?ë³¸ë¬¸ ?•ë³´ ì¶”ì¶œ ?±ê³µ (êµ?„¸ì²??ë?): {extracted_info.get('?¬ê±´ëª?, 'Unknown')}")
                    
                elif isinstance(law_data, str):
                    extracted_info = {
                        '?ë??¼ë ¨ë²ˆí˜¸': '',
                        '?¬ê±´ëª?: '',
                        '?¬ê±´ë²ˆí˜¸': '',
                        'ë²•ì›ëª?: '',
                        '? ê³ ?¼ì': '',
                        '?ê²°? í˜•': '',
                        '?¬ê±´? í˜•': '',
                        '?ê²°?”ì?': '',
                        '?ì‹œ?¬í•­': '',
                        'ì°¸ì¡°ì¡°ë¬¸': '',
                        'ì°¸ì¡°?ë?': '',
                        '?ë??´ìš©': law_data[:1000] + '...' if len(law_data) > 1000 else law_data,  # HTML ?´ìš© ?¼ë? ?€??
                        '?¬ê±´ì¢…ë¥˜ì½”ë“œ': '',
                        'ë²•ì›ì¢…ë¥˜ì½”ë“œ': '',
                        '? ê³ ': '',
                        '?°ì´?°ì¶œì²?: 'êµ?„¸ì²?(HTML ?•íƒœ)',
                        '?˜ì§‘?¼ì‹œ': datetime.now().isoformat(),
                        '?¤ë¥˜': 'HTML ?•íƒœ???‘ë‹µ?¼ë¡œ JSON ?Œì‹± ë¶ˆê?'
                    }
                    
                    logger.info(f"?ë?ë³¸ë¬¸ ?•ë³´ ì¶”ì¶œ ?±ê³µ (êµ?„¸ì²?HTML): ê¸¸ì´ {len(law_data)}")
                    
                else:
                    logger.warning(f"Lawê°€ ?ˆìƒì¹?ëª»í•œ ?€?? {type(law_data)}")
                    extracted_info = {
                        '?˜ì§‘?¼ì‹œ': datetime.now().isoformat(),
                        '?¤ë¥˜': f"Law ?€???¤ë¥˜: {type(law_data)}"
                    }
                    
            else:
                logger.warning(f"?????†ëŠ” ?‘ë‹µ êµ¬ì¡°: {list(detail_response.keys())}")
                # ?„ì²´ ?‘ë‹µ??ê·¸ë?ë¡??€??(ì¤‘ë³µ ?œê±°)
                extracted_info = {
                    '?˜ì§‘?¼ì‹œ': datetime.now().isoformat(),
                    '?¤ë¥˜': f"?????†ëŠ” ?‘ë‹µ êµ¬ì¡°: {list(detail_response.keys())}"
                }
                        
            return extracted_info
            
        except Exception as e:
            logger.error(f"?ë? ?ì„¸ ?•ë³´ ì¶”ì¶œ ?¤íŒ¨: {e}")
            return {'?¤ë¥˜': str(e)}
    
    def _categorize_precedent(self, precedent: Dict[str, Any]) -> PrecedentCategory:
        """?ë? ì¹´í…Œê³ ë¦¬ ë¶„ë¥˜ (?¬ê±´? í˜•ì½”ë“œ ê¸°ë°˜)"""
        case_type_code = precedent.get('?¬ê±´? í˜•ì½”ë“œ', '')
        
        # ?¬ê±´? í˜•ì½”ë“œ ê¸°ë°˜ ë¶„ë¥˜
        case_type_mapping = {
            '01': PrecedentCategory.CIVIL_CONTRACT,
            '02': PrecedentCategory.CRIMINAL,
            '03': PrecedentCategory.ADMINISTRATIVE,
            '04': PrecedentCategory.CIVIL_FAMILY,
            '05': PrecedentCategory.OTHER
        }
        
        return case_type_mapping.get(case_type_code, PrecedentCategory.OTHER)
    
    def _random_delay(self, min_seconds: float = None, max_seconds: float = None):
        """API ?”ì²­ ê°??œë¤ ì§€??- ?¬ìš©???¤ì • ê°„ê²© ?¬ìš©"""
        if min_seconds is None:
            min_seconds = max(0.1, self.request_interval_base - self.request_interval_range)
        if max_seconds is None:
            max_seconds = self.request_interval_base + self.request_interval_range
        
        delay = random.uniform(min_seconds, max_seconds)
        logger.debug(f"API ?”ì²­ ê°?{delay:.2f}ì´??€ê¸?..")
        time.sleep(delay)
    
    def _collect_by_date_range(self, start_date: str, end_date: str, max_count: int, 
                              output_dir: Path, base_params: Dict[str, Any]) -> List[PrecedentData]:
        """? ì§œ ë²”ìœ„ë¡??ë? ?˜ì§‘"""
        precedents = []
        page = 1
        consecutive_empty_pages = 0
        max_empty_pages = 3
        total_collected = 0
        total_duplicates = 0
        total_errors = 0
        
        # ë¬´ì œ??ëª¨ë“œ ?•ì¸
        unlimited_mode = max_count >= 999999999
        
        if unlimited_mode:
            logger.info(f"?“… ? ì§œ ë²”ìœ„ {start_date} ~ {end_date} ?˜ì§‘ ?œì‘ (ë¬´ì œ??ëª¨ë“œ)")
        else:
            logger.info(f"?“… ? ì§œ ë²”ìœ„ {start_date} ~ {end_date} ?˜ì§‘ ?œì‘ (ëª©í‘œ: {max_count:,}ê±?")
        
        logger.info("=" * 80)
        
        while (unlimited_mode or len(precedents) < max_count) and consecutive_empty_pages < max_empty_pages:
            try:
                # API ?Œë¼ë¯¸í„° êµ¬ì„± (?¬ë°”ë¥?prncYd ?Œë¼ë¯¸í„° ?¬ìš©)
                params = base_params.copy()
                params.update({
                    "from_date": start_date,
                    "to_date": end_date,
                    "display": 20,  # ë°°ì¹˜ ?¬ê¸°ë¥?100?ì„œ 20?¼ë¡œ ì¤„ì„ (?ë?ë³¸ë¬¸ ?˜ì§‘ ???ë„ ê°œì„ )
                    "page": page
                })
                
                # API ?”ì²­ ê°?ì§€??
                if page > 1:
                    self._random_delay()
                
                # API ?¸ì¶œ
                logger.info(f"?” ?˜ì´ì§€ {page} ?”ì²­ ì¤?.. (? ì§œ ?„í„°ë§? {start_date}~{end_date})")
                results = self.client.get_precedent_list(**params)
                
                if not results:
                    consecutive_empty_pages += 1
                    logger.warning(f"? ï¸  ?˜ì´ì§€ {page}: ê²°ê³¼ ?†ìŒ (?°ì† ë¹??˜ì´ì§€: {consecutive_empty_pages}/{max_empty_pages})")
                    page += 1
                    continue
                else:
                    consecutive_empty_pages = 0
                
                # ê²°ê³¼ ì²˜ë¦¬
                new_count = 0
                duplicate_count = 0
                page_precedents = []
                
                for result in results:
                    # ë¬´ì œ??ëª¨ë“œê°€ ?„ë‹Œ ê²½ìš°?ë§Œ ê±´ìˆ˜ ?œí•œ ?•ì¸
                    if not unlimited_mode and len(precedents) >= max_count:
                        break
                    
                    # ì¤‘ë³µ ?•ì¸ (ê°œì„ ??ë¡œì§)
                    if self._is_duplicate_precedent(result):
                        duplicate_count += 1
                        self.stats.duplicate_count += 1
                        continue
                    
                    # PrecedentData ê°ì²´ ?ì„± (?ë?ë³¸ë¬¸ ?¬í•¨ ?¬ë????°ë¼)
                    if self.include_details:
                        logger.info(f"?“„ ?ë?ë³¸ë¬¸ ?˜ì§‘ ì¤? {result.get('?¬ê±´ëª?, 'Unknown')} (ID: {result.get('?ë??¼ë ¨ë²ˆí˜¸', 'N/A')})")
                        precedent_data = self._create_precedent_data_with_detail(result)
                    else:
                        precedent_data = self._create_precedent_data(result)
                    
                    if not precedent_data:
                        self.stats.failed_count += 1
                        total_errors += 1
                        continue
                    
                    # ? ê·œ ?ë? ì¶”ê?
                    precedents.append(precedent_data)
                    page_precedents.append(precedent_data)
                    self._mark_precedent_collected(result)
                    new_count += 1
                
                # ?˜ì´ì§€ë³?ì¦‰ì‹œ ?€??
                if page_precedents:
                    self._save_page_precedents(page_precedents, output_dir, page, start_date, end_date)
                
                # ?µê³„ ?…ë°?´íŠ¸
                total_collected += new_count
                total_duplicates += duplicate_count
                
                # ì§„í–‰?í™© ë¡œê·¸
                progress_percent = (len(precedents) / max_count * 100) if not unlimited_mode else 0
                logger.info(f"???˜ì´ì§€ {page} ?„ë£Œ: ? ê·œ {new_count:,}ê±? ì¤‘ë³µ {duplicate_count:,}ê±?)
                logger.info(f"?“Š ?„ì  ?„í™©: ì´?{len(precedents):,}ê±??˜ì§‘ (? ê·œ: {total_collected:,}, ì¤‘ë³µ: {total_duplicates:,}, ?¤ë¥˜: {total_errors:,})")
                if not unlimited_mode:
                    logger.info(f"?“ˆ ì§„í–‰ë¥? {progress_percent:.1f}% ({len(precedents):,}/{max_count:,}ê±?")
                logger.info("-" * 60)
                
                page += 1
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                if self._check_api_limits():
                    break
                
            except Exception as e:
                total_errors += 1
                self.error_count += 1
                logger.error(f"???˜ì´ì§€ {page} ?¤ë¥˜ ë°œìƒ: {e}")
                logger.error(f"?”„ ?¤ë¥˜ ì¹´ìš´?? {self.error_count}/{self.max_errors}")
                
                if self.error_count >= self.max_errors:
                    logger.error("?›‘ ìµœë? ?¤ë¥˜ ?˜ì— ?„ë‹¬?˜ì—¬ ?˜ì§‘ ì¤‘ë‹¨")
                    break
                
                # ?¤ë¥˜ ë°œìƒ ?œì—???„ì¬ê¹Œì? ?˜ì§‘???°ì´???€??
                if precedents:
                    logger.info(f"?’¾ ?¤ë¥˜ ë°œìƒ ?„ê¹Œì§€ ?˜ì§‘??{len(precedents):,}ê±??€??ì¤?..")
                    self._save_page_precedents(precedents, output_dir, page, start_date, end_date)
                
                continue
        
        # ìµœì¢… ?µê³„ ë¡œê·¸
        logger.info("=" * 80)
        logger.info(f"?‰ ? ì§œ ë²”ìœ„ {start_date} ~ {end_date} ?˜ì§‘ ?„ë£Œ!")
        logger.info(f"?“Š ìµœì¢… ?µê³„: ì´?{len(precedents):,}ê±??˜ì§‘ (? ê·œ: {total_collected:,}, ì¤‘ë³µ: {total_duplicates:,}, ?¤ë¥˜: {total_errors:,})")
        logger.info("=" * 80)
        
        return precedents
    
    def _save_page_precedents(self, precedents: List[PrecedentData], output_dir: Path, 
                             page: int, start_date: str, end_date: str):
        """?˜ì´ì§€ë³??ë? ì¦‰ì‹œ ?€??(?ë??¼ë ¨ë²ˆí˜¸ ê¸°ì?)"""
        if not precedents:
            return
        
        try:
            # ì¹´í…Œê³ ë¦¬ë³„ë¡œ ê·¸ë£¹??
            by_category = {}
            for precedent in precedents:
                category = precedent.category.value
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(precedent)
            
            # ì¹´í…Œê³ ë¦¬ë³??Œì¼ ?€??
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            saved_files = []
            
            for category, category_precedents in by_category.items():
                # ?ë??¼ë ¨ë²ˆí˜¸ ë²”ìœ„ë¡??Œì¼ëª??ì„±
                serial_numbers = [p.raw_data.get('?ë??¼ë ¨ë²ˆí˜¸', '') for p in category_precedents]
                serial_numbers = [s for s in serial_numbers if s]  # ë¹?ê°??œê±°
                
                if serial_numbers:
                    # ?ë??¼ë ¨ë²ˆí˜¸ ?•ë ¬
                    serial_numbers.sort()
                    start_serial = serial_numbers[0]
                    end_serial = serial_numbers[-1]
                    
                    # ?Œì¼ëª??ì„± (ì¹´í…Œê³ ë¦¬ ?œê±°)
                    filename = f"page_{page:03d}_{start_serial}-{end_serial}_{len(category_precedents)}ê±?{timestamp}.json"
                else:
                    # ?ë??¼ë ¨ë²ˆí˜¸ê°€ ?†ëŠ” ê²½ìš° ?€ì²??Œì¼ëª?
                    filename = f"page_{page:03d}_{len(category_precedents)}ê±?{timestamp}.json"
                
                filepath = output_dir / filename
                
                # ?ë? ?°ì´??êµ¬ì„± (ê¸°ë³¸ ?•ë³´ + ?ì„¸ ?•ë³´)
                precedents_data = []
                for p in category_precedents:
                    precedent_data = p.raw_data.copy()
                    if p.detail_info:
                        precedent_data['detail_info'] = p.detail_info
                    precedents_data.append(precedent_data)
                
                batch_data = {
                    'metadata': {
                        'page': page,
                        'category': category,
                        'count': len(category_precedents),
                        'date_range': f"{start_date}~{end_date}",
                        'saved_at': datetime.now().isoformat(),
                        'batch_id': f"page_{page}_{timestamp}",
                        'serial_number_range': f"{start_serial}-{end_serial}" if serial_numbers else None,
                        'include_details': self.include_details
                    },
                    'precedents': precedents_data
                }
                
                with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
                    json.dump(batch_data, f, ensure_ascii=False, indent=2)
                
                saved_files.append(filepath)
                logger.info(f"?’¾ ?˜ì´ì§€ {page} ?€???„ë£Œ: {category} ì¹´í…Œê³ ë¦¬ {len(category_precedents):,}ê±?-> {filename}")
            
            # ?µê³„ ?…ë°?´íŠ¸
            self.stats.saved_count += len(precedents)
            
            logger.info(f"?“ ?˜ì´ì§€ {page} ?€???„ë£Œ: ì´?{len(saved_files):,}ê°??Œì¼")
            
        except Exception as e:
            logger.error(f"???˜ì´ì§€ {page} ?€???¤íŒ¨: {e}")
            logger.error(traceback.format_exc())
    
    def _save_batch_precedents(self, output_dir: Path):
        """ë°°ì¹˜ ?¨ìœ„ë¡??ë? ?€??(?ë??¼ë ¨ë²ˆí˜¸ ê¸°ì?)"""
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
                # ?ë??¼ë ¨ë²ˆí˜¸ ë²”ìœ„ë¡??Œì¼ëª??ì„±
                serial_numbers = [p.raw_data.get('?ë??¼ë ¨ë²ˆí˜¸', '') for p in precedents]
                serial_numbers = [s for s in serial_numbers if s]  # ë¹?ê°??œê±°
                
                if serial_numbers:
                    # ?ë??¼ë ¨ë²ˆí˜¸ ?•ë ¬
                    serial_numbers.sort()
                    start_serial = serial_numbers[0]
                    end_serial = serial_numbers[-1]
                    
                    # ?ˆì „???Œì¼ëª??ì„±
                    safe_category = category.replace('_', '-')
                    filename = f"batch_{safe_category}_{start_serial}-{end_serial}_{len(precedents)}ê±?{timestamp}.json"
                else:
                    # ?ë??¼ë ¨ë²ˆí˜¸ê°€ ?†ëŠ” ê²½ìš° ?€ì²??Œì¼ëª?
                    safe_category = category.replace('_', '-')
                    filename = f"batch_{safe_category}_{len(precedents)}ê±?{timestamp}.json"
                
                filepath = output_dir / filename
                
                batch_data = {
                    'metadata': {
                        'category': category,
                        'count': len(precedents),
                        'saved_at': datetime.now().isoformat(),
                        'batch_id': timestamp,
                        'serial_number_range': f"{start_serial}-{end_serial}" if serial_numbers else None
                    },
                    'precedents': [p.raw_data for p in precedents]
                }
                
                with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
                    json.dump(batch_data, f, ensure_ascii=False, indent=2)
                
                saved_files.append(filepath)
                logger.info(f"?’¾ ë°°ì¹˜ ?€???„ë£Œ: {category} ì¹´í…Œê³ ë¦¬ {len(precedents):,}ê±?-> {filename}")
            
            # ?µê³„ ?…ë°?´íŠ¸
            self.stats.saved_count += len(self.pending_precedents)
            
            # ?„ì‹œ ?€?¥ì†Œ ì´ˆê¸°??
            self.pending_precedents = []
            
            logger.info(f"?“ ë°°ì¹˜ ?€???„ë£Œ: ì´?{len(saved_files):,}ê°??Œì¼")
            
        except Exception as e:
            logger.error(f"??ë°°ì¹˜ ?€???¤íŒ¨: {e}")
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
    
    def collect_by_yearly_strategy(self, years: List[int], target_per_year: int) -> Dict[str, Any]:
        """?°ë„ë³??˜ì§‘ ?„ëµ"""
        logger.info(f"?°ë„ë³??˜ì§‘ ?œì‘: {years}?? ?°ê°„ ëª©í‘œ {target_per_year}ê±?)
        
        all_precedents = []
        collection_summary = {
            'strategy': 'yearly',
            'years': years,
            'target_per_year': target_per_year,
            'collected_by_year': {},
            'total_collected': 0,
            'start_time': datetime.now().isoformat()
        }
        
        for year in years:
            if self.stats.collected_count >= self.stats.target_count:
                break
            
            year_str = str(year)
            start_date = f"{year_str}0101"
            end_date = f"{year_str}1231"
            
            # ?´ë‹¹ ?°ë„??ê¸°ì¡´ ?°ì´?°ë§Œ ì¤‘ë³µ ?•ì¸?˜ë„ë¡??˜ì§‘ê¸??¬ì´ˆê¸°í™”
            logger.info(f"?”„ {year}??ì¤‘ë³µ ?•ì¸???„í•œ ?°ì´??ë¡œë“œ ì¤?..")
            self.collected_precedents.clear()  # ê¸°ì¡´ ì¤‘ë³µ ?°ì´??ì´ˆê¸°??
            self._load_existing_data(target_year=year)  # ?´ë‹¹ ?°ë„ë§?ë¡œë“œ
            
            # ì¶œë ¥ ?”ë ‰? ë¦¬ ?ì„±
            output_dir = self._create_output_subdir(DateCollectionStrategy.YEARLY, year_str)
            
            logger.info(f"?“Š {year}???ë? ?˜ì§‘ ì¤?.. (ëª©í‘œ: {target_per_year}ê±?")
            
            year_precedents = self._collect_by_date_range(
                start_date, end_date, target_per_year, output_dir,
                {"search": 1, "sort": "ddes"}
            )
            
            all_precedents.extend(year_precedents)
            collection_summary['collected_by_year'][year_str] = len(year_precedents)
            
            logger.info(f"??{year}???„ë£Œ: {len(year_precedents)}ê±??˜ì§‘ (?„ì : {len(all_precedents)}ê±?")
        
        # ìµœì¢… ë°°ì¹˜ ?€??
        if self.pending_precedents:
            final_output_dir = self._create_output_subdir(DateCollectionStrategy.YEARLY, "final")
            self._save_batch_precedents(final_output_dir)
        
        collection_summary['total_collected'] = len(all_precedents)
        collection_summary['end_time'] = datetime.now().isoformat()
        
        # ?˜ì§‘ ?”ì•½ ?€??
        summary_file = self.base_output_dir / f"yearly_collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(collection_summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"?°ë„ë³??˜ì§‘ ?„ë£Œ: ì´?{len(all_precedents)}ê±?)
        return collection_summary
    
    def collect_by_quarterly_strategy(self, quarters: List[Tuple[str, str, str]], target_per_quarter: int) -> Dict[str, Any]:
        """ë¶„ê¸°ë³??˜ì§‘ ?„ëµ"""
        logger.info(f"ë¶„ê¸°ë³??˜ì§‘ ?œì‘: {len(quarters)}ê°?ë¶„ê¸°, ë¶„ê¸°??ëª©í‘œ {target_per_quarter}ê±?)
        
        all_precedents = []
        collection_summary = {
            'strategy': 'quarterly',
            'quarters': [q[0] for q in quarters],
            'target_per_quarter': target_per_quarter,
            'collected_by_quarter': {},
            'total_collected': 0,
            'start_time': datetime.now().isoformat()
        }
        
        for quarter_name, start_date, end_date in quarters:
            if self.stats.collected_count >= self.stats.target_count:
                break
            
            # ì¶œë ¥ ?”ë ‰? ë¦¬ ?ì„±
            output_dir = self._create_output_subdir(DateCollectionStrategy.QUARTERLY, quarter_name)
            
            logger.info(f"?“Š {quarter_name} ?ë? ?˜ì§‘ ì¤?.. (ëª©í‘œ: {target_per_quarter}ê±?")
            
            quarter_precedents = self._collect_by_date_range(
                start_date, end_date, target_per_quarter, output_dir,
                {"search": 1, "sort": "ddes"}
            )
            
            all_precedents.extend(quarter_precedents)
            collection_summary['collected_by_quarter'][quarter_name] = len(quarter_precedents)
            
            logger.info(f"??{quarter_name} ?„ë£Œ: {len(quarter_precedents)}ê±??˜ì§‘ (?„ì : {len(all_precedents)}ê±?")
        
        # ìµœì¢… ë°°ì¹˜ ?€??
        if self.pending_precedents:
            final_output_dir = self._create_output_subdir(DateCollectionStrategy.QUARTERLY, "final")
            self._save_batch_precedents(final_output_dir)
        
        collection_summary['total_collected'] = len(all_precedents)
        collection_summary['end_time'] = datetime.now().isoformat()
        
        # ?˜ì§‘ ?”ì•½ ?€??
        summary_file = self.base_output_dir / f"quarterly_collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(collection_summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"ë¶„ê¸°ë³??˜ì§‘ ?„ë£Œ: ì´?{len(all_precedents)}ê±?)
        return collection_summary
    
    def collect_by_monthly_strategy(self, months: List[Tuple[str, str, str]], target_per_month: int) -> Dict[str, Any]:
        """?”ë³„ ?˜ì§‘ ?„ëµ"""
        logger.info(f"?”ë³„ ?˜ì§‘ ?œì‘: {len(months)}ê°??? ?”ê°„ ëª©í‘œ {target_per_month}ê±?)
        
        all_precedents = []
        collection_summary = {
            'strategy': 'monthly',
            'months': [m[0] for m in months],
            'target_per_month': target_per_month,
            'collected_by_month': {},
            'total_collected': 0,
            'start_time': datetime.now().isoformat()
        }
        
        for month_name, start_date, end_date in months:
            if self.stats.collected_count >= self.stats.target_count:
                break
            
            # ì¶œë ¥ ?”ë ‰? ë¦¬ ?ì„±
            output_dir = self._create_output_subdir(DateCollectionStrategy.MONTHLY, month_name)
            
            logger.info(f"?“Š {month_name} ?ë? ?˜ì§‘ ì¤?.. (ëª©í‘œ: {target_per_month}ê±?")
            
            month_precedents = self._collect_by_date_range(
                start_date, end_date, target_per_month, output_dir,
                {"search": 1, "sort": "ddes"}
            )
            
            all_precedents.extend(month_precedents)
            collection_summary['collected_by_month'][month_name] = len(month_precedents)
            
            logger.info(f"??{month_name} ?„ë£Œ: {len(month_precedents)}ê±??˜ì§‘ (?„ì : {len(all_precedents)}ê±?")
        
        # ìµœì¢… ë°°ì¹˜ ?€??
        if self.pending_precedents:
            final_output_dir = self._create_output_subdir(DateCollectionStrategy.MONTHLY, "final")
            self._save_batch_precedents(final_output_dir)
        
        collection_summary['total_collected'] = len(all_precedents)
        collection_summary['end_time'] = datetime.now().isoformat()
        
        # ?˜ì§‘ ?”ì•½ ?€??
        summary_file = self.base_output_dir / f"monthly_collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(collection_summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"?”ë³„ ?˜ì§‘ ?„ë£Œ: ì´?{len(all_precedents)}ê±?)
        return collection_summary
    
    def collect_by_weekly_strategy(self, weeks: List[Tuple[str, str, str]], target_per_week: int) -> Dict[str, Any]:
        """ì£¼ë³„ ?˜ì§‘ ?„ëµ"""
        logger.info(f"ì£¼ë³„ ?˜ì§‘ ?œì‘: {len(weeks)}ê°?ì£? ì£¼ê°„ ëª©í‘œ {target_per_week}ê±?)
        
        all_precedents = []
        collection_summary = {
            'strategy': 'weekly',
            'weeks': [w[0] for w in weeks],
            'target_per_week': target_per_week,
            'collected_by_week': {},
            'total_collected': 0,
            'start_time': datetime.now().isoformat()
        }
        
        for week_name, start_date, end_date in weeks:
            if self.stats.collected_count >= self.stats.target_count:
                break
            
            # ì¶œë ¥ ?”ë ‰? ë¦¬ ?ì„±
            output_dir = self._create_output_subdir(DateCollectionStrategy.WEEKLY, week_name)
            
            logger.info(f"?“Š {week_name} ?ë? ?˜ì§‘ ì¤?.. (ëª©í‘œ: {target_per_week}ê±?")
            
            week_precedents = self._collect_by_date_range(
                start_date, end_date, target_per_week, output_dir,
                {"search": 1, "sort": "ddes"}
            )
            
            all_precedents.extend(week_precedents)
            collection_summary['collected_by_week'][week_name] = len(week_precedents)
            
            logger.info(f"??{week_name} ?„ë£Œ: {len(week_precedents)}ê±??˜ì§‘ (?„ì : {len(all_precedents)}ê±?")
        
        # ìµœì¢… ë°°ì¹˜ ?€??
        if self.pending_precedents:
            final_output_dir = self._create_output_subdir(DateCollectionStrategy.WEEKLY, "final")
            self._save_batch_precedents(final_output_dir)
        
        collection_summary['total_collected'] = len(all_precedents)
        collection_summary['end_time'] = datetime.now().isoformat()
        
        # ?˜ì§‘ ?”ì•½ ?€??
        summary_file = self.base_output_dir / f"weekly_collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(collection_summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"ì£¼ë³„ ?˜ì§‘ ?„ë£Œ: ì´?{len(all_precedents)}ê±?)
        return collection_summary
    
    def generate_date_ranges(self, strategy: DateCollectionStrategy, count: int) -> List[Tuple[str, str, str]]:
        """? ì§œ ë²”ìœ„ ?ì„±"""
        ranges = []
        current = datetime.now()
        
        if strategy == DateCollectionStrategy.YEARLY:
            for i in range(count):
                year = current.year - i
                ranges.append((f"{year}??, f"{year}0101", f"{year}1231"))
        
        elif strategy == DateCollectionStrategy.QUARTERLY:
            for i in range(count):
                target_date = current - timedelta(days=90*i)
                year = target_date.year
                quarter = (target_date.month - 1) // 3 + 1
                
                if quarter == 1:
                    start_date = f"{year}0101"
                    end_date = f"{year}0331"
                elif quarter == 2:
                    start_date = f"{year}0401"
                    end_date = f"{year}0630"
                elif quarter == 3:
                    start_date = f"{year}0701"
                    end_date = f"{year}0930"
                else:
                    start_date = f"{year}1001"
                    end_date = f"{year}1231"
                
                ranges.append((f"{year}Q{quarter}", start_date, end_date))
        
        elif strategy == DateCollectionStrategy.MONTHLY:
            for i in range(count):
                target_date = current - timedelta(days=30*i)
                year = target_date.year
                month = target_date.month
                
                start_date = datetime(year, month, 1)
                if month == 12:
                    end_date = datetime(year+1, 1, 1) - timedelta(days=1)
                else:
                    end_date = datetime(year, month+1, 1) - timedelta(days=1)
                
                ranges.append((
                    f"{year}??month:02d}??,
                    start_date.strftime('%Y%m%d'),
                    end_date.strftime('%Y%m%d')
                ))
        
        elif strategy == DateCollectionStrategy.WEEKLY:
            for i in range(count):
                target_date = current - timedelta(weeks=i)
                start_of_week = target_date - timedelta(days=target_date.weekday())
                end_of_week = start_of_week + timedelta(days=6)
                
                ranges.append((
                    f"{start_of_week.strftime('%Y%m%d')}ì£?,
                    start_of_week.strftime('%Y%m%d'),
                    end_of_week.strftime('%Y%m%d')
                ))
        
        return ranges
