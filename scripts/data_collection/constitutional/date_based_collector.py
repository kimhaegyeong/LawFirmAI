#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?Œì¬ê²°ì •ë¡€ ? ì§œ ê¸°ë°˜ ?˜ì§‘ê¸?

??ëª¨ë“ˆ?€ ? ì§œë³„ë¡œ ì²´ê³„?ì¸ ?Œì¬ê²°ì •ë¡€ ?˜ì§‘???˜í–‰?©ë‹ˆ??
- ?°ë„ë³? ë¶„ê¸°ë³? ?”ë³„, ì£¼ë³„ ?˜ì§‘ ?„ëµ
- ê²°ì •?¼ì ?´ë¦¼ì°¨ìˆœ ìµœì ??
- ?´ë”ë³?raw ?°ì´???€??êµ¬ì¡°
- ì¤‘ë³µ ë°©ì? ë°?ì²´í¬?¬ì¸??ì§€??
"""

import json
import time
import random
import hashlib
import traceback
import gc
import psutil
import os
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


@dataclass
class ConstitutionalDecisionData:
    """?Œì¬ê²°ì •ë¡€ ?°ì´???´ë˜??- ëª©ë¡ ?°ì´???´ë???ë³¸ë¬¸ ?°ì´???¬í•¨"""
    # ëª©ë¡ ì¡°íšŒ API ?‘ë‹µ (ê¸°ë³¸ ?•ë³´)
    id: str  # ê²€?‰ê²°ê³¼ë²ˆ??
    ?¬ê±´ë²ˆí˜¸: str
    ì¢…êµ­?¼ì: str
    ?Œì¬ê²°ì •ë¡€?¼ë ¨ë²ˆí˜¸: str
    ?¬ê±´ëª? str
    ?Œì¬ê²°ì •ë¡€?ì„¸ë§í¬: str
    
    # ?ì„¸ ì¡°íšŒ API ?‘ë‹µ (ë³¸ë¬¸ ?°ì´?? - ëª©ë¡ ?°ì´???´ë????¬í•¨
    ?¬ê±´ì¢…ë¥˜ëª? Optional[str] = None
    ?ì‹œ?¬í•­: Optional[str] = None
    ê²°ì •?”ì?: Optional[str] = None
    ?„ë¬¸: Optional[str] = None
    ì°¸ì¡°ì¡°ë¬¸: Optional[str] = None
    ì°¸ì¡°?ë?: Optional[str] = None
    ?¬íŒ?€?ì¡°ë¬? Optional[str] = None
    
    # ë©”í??°ì´??
    document_type: str = "constitutional_decision"
    collected_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CollectionStats:
    """?˜ì§‘ ?µê³„ ?´ë˜??""
    total_collected: int = 0
    total_duplicates: int = 0
    total_errors: int = 0
    api_requests_made: int = 0
    api_errors: int = 0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    status: str = "PENDING"
    target_count: int = 0
    retry_delay: int = 5
    collected_decisions: Set[str] = field(default_factory=set)


class DateBasedConstitutionalCollector:
    """? ì§œ ê¸°ë°˜ ?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?´ë˜??""
    
    def __init__(self, config: LawOpenAPIConfig, base_output_dir: Optional[Path] = None):
        """
        ? ì§œ ê¸°ë°˜ ?Œì¬ê²°ì •ë¡€ ?˜ì§‘ê¸?ì´ˆê¸°??
        
        Args:
            config: API ?¤ì • ê°ì²´
            base_output_dir: ê¸°ë³¸ ì¶œë ¥ ?”ë ‰? ë¦¬ (ê¸°ë³¸ê°? data/raw/constitutional_decisions)
        """
        self.client = LawOpenAPIClient(config)
        self.base_output_dir = base_output_dir or Path("data/raw/constitutional_decisions")
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # ?°ì´??ê´€ë¦?(ë©”ëª¨ë¦?ìµœì ??
        self.collected_decisions: Set[str] = set()
        self.processed_date_ranges: Set[str] = set()
        self.pending_decisions: List[ConstitutionalDecisionData] = []
        self.max_memory_decisions = 10000  # ìµœë? ë©”ëª¨ë¦?ë³´ê? ê±´ìˆ˜
        
        # ?µê³„ ë°??íƒœ
        self.stats = CollectionStats()
        
        # ?ëŸ¬ ì²˜ë¦¬
        self.error_count = 0
        self.max_errors = 50
        
        # ê¸°ì¡´ ?˜ì§‘???°ì´??ë¡œë“œ
        self._load_existing_data()
        
        # ?œê°„ ?¸í„°ë²??¤ì • (ê¸°ë³¸ê°?
        self.request_interval_base = 2.0  # ê¸°ë³¸ ê°„ê²©
        self.request_interval_range = 2.0  # ê°„ê²© ë²”ìœ„
        
        # ì²´í¬?¬ì¸???¬ê°œ ëª¨ë“œ (ê¸°ë³¸ê°? False)
        self.resume_mode = False
        
        logger.info("? ì§œ ê¸°ë°˜ ?Œì¬ê²°ì •ë¡€ ?˜ì§‘ê¸?ì´ˆê¸°???„ë£Œ")
    
    def _monitor_memory_usage(self):
        """ë©”ëª¨ë¦??¬ìš©??ëª¨ë‹ˆ?°ë§"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            if memory_mb > 1000:  # 1GB ?´ìƒ ?¬ìš© ??ê²½ê³ 
                logger.warning(f"ë©”ëª¨ë¦??¬ìš©?‰ì´ ?’ìŠµ?ˆë‹¤: {memory_mb:.1f}MB")
                self._cleanup_memory()
                
            return memory_mb
        except Exception as e:
            logger.debug(f"ë©”ëª¨ë¦?ëª¨ë‹ˆ?°ë§ ?¤ë¥˜: {e}")
            return 0
    
    def _cleanup_memory(self):
        """ë©”ëª¨ë¦??•ë¦¬"""
        try:
            # ê°€ë¹„ì? ì»¬ë ‰??ê°•ì œ ?¤í–‰
            collected = gc.collect()
            logger.debug(f"ê°€ë¹„ì? ì»¬ë ‰???„ë£Œ: {collected}ê°?ê°ì²´ ?•ë¦¬")
            
            # ?€?©ëŸ‰ ?°ì´??êµ¬ì¡° ?•ë¦¬
            if len(self.collected_decisions) > self.max_memory_decisions:
                # ?¤ë˜???°ì´???¼ë? ?œê±° (ìµœê·¼ 50%ë§?? ì?)
                sorted_decisions = sorted(self.collected_decisions)
                keep_count = len(sorted_decisions) // 2
                self.collected_decisions = set(sorted_decisions[-keep_count:])
                logger.info(f"ë©”ëª¨ë¦??•ë¦¬: ?˜ì§‘??ê²°ì •ë¡€ {len(sorted_decisions) - keep_count}ê°??œê±°")
            
            # ?€ê¸?ì¤‘ì¸ ?°ì´???•ë¦¬
            if len(self.pending_decisions) > 1000:
                self.pending_decisions = self.pending_decisions[-500:]  # ìµœê·¼ 500ê°œë§Œ ? ì?
                logger.info("ë©”ëª¨ë¦??•ë¦¬: ?€ê¸?ì¤‘ì¸ ê²°ì •ë¡€ ?°ì´???•ë¦¬")
                
        except Exception as e:
            logger.error(f"ë©”ëª¨ë¦??•ë¦¬ ì¤??¤ë¥˜: {e}")
    
    def _check_memory_and_cleanup(self):
        """ë©”ëª¨ë¦?ì²´í¬ ë°??•ë¦¬"""
        memory_mb = self._monitor_memory_usage()
        
        # ë©”ëª¨ë¦??¬ìš©?‰ì´ ?’ìœ¼ë©??•ë¦¬
        if memory_mb > 800:  # 800MB ?´ìƒ
            self._cleanup_memory()
            
        return memory_mb
    
    def set_request_interval(self, base_interval: float, interval_range: float):
        """API ?”ì²­ ê°„ê²© ?¤ì •"""
        self.request_interval_base = base_interval
        self.request_interval_range = interval_range
        logger.info(f"?±ï¸ ?”ì²­ ê°„ê²© ?¤ì •: {base_interval:.1f} Â± {interval_range:.1f}ì´?)
    
    def enable_resume_mode(self):
        """ì²´í¬?¬ì¸???¬ê°œ ëª¨ë“œ ?œì„±??""
        self.resume_mode = True
        logger.info("?”„ ì²´í¬?¬ì¸???¬ê°œ ëª¨ë“œê°€ ?œì„±?”ë˜?ˆìŠµ?ˆë‹¤")
    
    def _load_existing_data(self, target_year: Optional[int] = None):
        """ê¸°ì¡´ ?˜ì§‘???°ì´??ë¡œë“œ"""
        try:
            if target_year:
                # ?¹ì • ?°ë„ ?°ì´?°ë§Œ ë¡œë“œ
                pattern = f"yearly_{target_year}_*"
                existing_dirs = list(self.base_output_dir.glob(pattern))
            else:
                # ëª¨ë“  ê¸°ì¡´ ?°ì´??ë¡œë“œ
                existing_dirs = list(self.base_output_dir.glob("*"))
            
            for dir_path in existing_dirs:
                if dir_path.is_dir():
                    json_files = list(dir_path.glob("page_*.json"))
                    for json_file in json_files:
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                decisions = data.get('decisions', [])
                                for decision in decisions:
                                    decision_id = decision.get('?Œì¬ê²°ì •ë¡€?¼ë ¨ë²ˆí˜¸')
                                    if decision_id:
                                        self.collected_decisions.add(decision_id)
                        except Exception as e:
                            logger.warning(f"ê¸°ì¡´ ?°ì´??ë¡œë“œ ?¤íŒ¨: {json_file} - {e}")
            
            logger.info(f"ê¸°ì¡´ ?˜ì§‘???Œì¬ê²°ì •ë¡€: {len(self.collected_decisions)}ê±?)
            
        except Exception as e:
            logger.error(f"ê¸°ì¡´ ?°ì´??ë¡œë“œ ì¤??¤ë¥˜: {e}")
    
    def _create_output_directory(self, strategy: DateCollectionStrategy, 
                               year: Optional[int] = None, 
                               quarter: Optional[int] = None,
                               month: Optional[int] = None,
                               week_start: Optional[str] = None) -> Path:
        """ì¶œë ¥ ?”ë ‰? ë¦¬ ?ì„± (ì²´í¬?¬ì¸?¸ê? ?ˆìœ¼ë©?ê¸°ì¡´ ?”ë ‰? ë¦¬ ?¬ìš©)"""
        
        # ì²´í¬?¬ì¸?¸ê? ?ˆëŠ” ê¸°ì¡´ ?”ë ‰? ë¦¬ ì°¾ê¸°
        if self.resume_mode:
            existing_dir = self._find_existing_directory(strategy, year, quarter, month, week_start)
            if existing_dir:
                logger.info(f"?”„ ê¸°ì¡´ ?”ë ‰? ë¦¬ ?¬ìš©: {existing_dir}")
                return existing_dir
        
        # ?ˆë¡œ???”ë ‰? ë¦¬ ?ì„±
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if strategy == DateCollectionStrategy.YEARLY:
            if year:
                dir_name = f"yearly_{year}_{timestamp}"
            else:
                dir_name = f"yearly_collection_{timestamp}"
        elif strategy == DateCollectionStrategy.QUARTERLY:
            if year and quarter:
                dir_name = f"quarterly_{year}Q{quarter}_{timestamp}"
            else:
                dir_name = f"quarterly_collection_{timestamp}"
        elif strategy == DateCollectionStrategy.MONTHLY:
            if year and month:
                dir_name = f"monthly_{year}??month}??{timestamp}"
            else:
                dir_name = f"monthly_collection_{timestamp}"
        elif strategy == DateCollectionStrategy.WEEKLY:
            if week_start:
                dir_name = f"weekly_{week_start}ì£?{timestamp}"
            else:
                dir_name = f"weekly_collection_{timestamp}"
        else:
            dir_name = f"daily_collection_{timestamp}"
        
        output_dir = self.base_output_dir / dir_name
        
        # ?”ë ‰? ë¦¬ ?ì„± ê°•í™”
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"ì¶œë ¥ ?”ë ‰? ë¦¬ ?ì„±: {output_dir}")
        except Exception as e:
            logger.error(f"?”ë ‰? ë¦¬ ?ì„± ?¤íŒ¨: {output_dir} - {e}")
            raise
        
        return output_dir
    
    def _find_existing_directory(self, strategy: DateCollectionStrategy, 
                               year: Optional[int] = None, 
                               quarter: Optional[int] = None,
                               month: Optional[int] = None,
                               week_start: Optional[str] = None) -> Optional[Path]:
        """ì²´í¬?¬ì¸?¸ê? ?ˆëŠ” ê¸°ì¡´ ?”ë ‰? ë¦¬ ì°¾ê¸°"""
        try:
            # ?¨í„´ ?ì„±
            if strategy == DateCollectionStrategy.YEARLY and year:
                pattern = f"yearly_{year}_*"
            elif strategy == DateCollectionStrategy.QUARTERLY and year and quarter:
                pattern = f"quarterly_{year}Q{quarter}_*"
            elif strategy == DateCollectionStrategy.MONTHLY and year and month:
                pattern = f"monthly_{year}??month}??*"
            elif strategy == DateCollectionStrategy.WEEKLY and week_start:
                pattern = f"weekly_{week_start}ì£?*"
            else:
                return None
            
            # ?´ë‹¹ ?¨í„´???”ë ‰? ë¦¬??ì°¾ê¸°
            matching_dirs = list(self.base_output_dir.glob(pattern))
            
            # ì²´í¬?¬ì¸???Œì¼???ˆëŠ” ?”ë ‰? ë¦¬ ì°¾ê¸°
            for dir_path in sorted(matching_dirs, reverse=True):  # ìµœì‹ ?œìœ¼ë¡??•ë ¬
                checkpoint_file = dir_path / "checkpoint.json"
                if checkpoint_file.exists():
                    logger.info(f"?“‹ ì²´í¬?¬ì¸??ë°œê²¬: {checkpoint_file}")
                    return dir_path
            
            logger.info(f"?“‹ ì²´í¬?¬ì¸?¸ê? ?ˆëŠ” ?”ë ‰? ë¦¬ë¥?ì°¾ì? ëª»í–ˆ?µë‹ˆ?? ?¨í„´: {pattern}")
            return None
            
        except Exception as e:
            logger.warning(f"ê¸°ì¡´ ?”ë ‰? ë¦¬ ì°¾ê¸° ?¤íŒ¨: {e}")
            return None
    
    def _save_batch(self, decisions: List[ConstitutionalDecisionData], 
                   output_dir: Path, page_num: int, 
                   category: str = "constitutional") -> bool:
        """ë°°ì¹˜ ?°ì´???€??""
        try:
            if not decisions:
                return True
            
            # ì¶œë ¥ ?”ë ‰? ë¦¬ ?•ì¸ ë°??ì„±
            if not output_dir.exists():
                logger.warning(f"ì¶œë ¥ ?”ë ‰? ë¦¬ê°€ ì¡´ì¬?˜ì? ?ŠìŠµ?ˆë‹¤: {output_dir}")
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"ì¶œë ¥ ?”ë ‰? ë¦¬ ?¬ìƒ?? {output_dir}")
            
            # ?Œì¼ëª??ì„±
            start_id = decisions[0].?Œì¬ê²°ì •ë¡€?¼ë ¨ë²ˆí˜¸
            end_id = decisions[-1].?Œì¬ê²°ì •ë¡€?¼ë ¨ë²ˆí˜¸
            count = len(decisions)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"page_{page_num:03d}_{category}_{start_id}-{end_id}_{count}ê±?{timestamp}.json"
            file_path = output_dir / filename
            
            # ?°ì´??êµ¬ì¡°??
            batch_data = {
                "metadata": {
                    "category": category,
                    "page": page_num,
                    "count": count,
                    "start_id": start_id,
                    "end_id": end_id,
                    "collected_at": timestamp,
                    "strategy": "date_based"
                },
                "decisions": [decision.__dict__ for decision in decisions]
            }
            
            # JSON ?Œì¼ ?€??
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"??ë°°ì¹˜ ?€???„ë£Œ: {filename} ({count}ê±?")
            
            # ì²´í¬?¬ì¸???€??(ì§„í–‰ ?í™© ê¸°ë¡)
            self._save_checkpoint(output_dir, page_num, collected_count=len(decisions))
            
            return True
            
        except Exception as e:
            logger.error(f"ë°°ì¹˜ ?€???¤íŒ¨: {e}")
            return False
    
    def _save_checkpoint(self, output_dir: Path, page_num: int, collected_count: int):
        """ì²´í¬?¬ì¸???€??(ì§„í–‰ ?í™© ê¸°ë¡)"""
        try:
            checkpoint_data = {
                "checkpoint_info": {
                    "last_page": page_num,
                    "collected_count": collected_count,
                    "timestamp": datetime.now().isoformat(),
                    "status": "in_progress"
                }
            }
            
            checkpoint_file = output_dir / "checkpoint.json"
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.warning(f"ì²´í¬?¬ì¸???€???¤íŒ¨: {e}")
    
    def _load_checkpoint(self, output_dir: Path) -> dict:
        """ì²´í¬?¬ì¸??ë¡œë“œ (ì¤‘ë‹¨???˜ì§‘ ?¬ê°œ)"""
        try:
            checkpoint_file = output_dir / "checkpoint.json"
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                logger.info(f"?“‹ ì²´í¬?¬ì¸??ë°œê²¬: ?˜ì´ì§€ {checkpoint_data['checkpoint_info']['last_page']}, ?˜ì§‘??ê±´ìˆ˜ {checkpoint_data['checkpoint_info']['collected_count']}")
                return checkpoint_data
        except Exception as e:
            logger.warning(f"ì²´í¬?¬ì¸??ë¡œë“œ ?¤íŒ¨: {e}")
        
        return None
    
    def _save_summary(self, output_dir: Path, strategy: DateCollectionStrategy, 
                     total_collected: int, total_duplicates: int, 
                     total_errors: int, duration: timedelta):
        """?˜ì§‘ ?”ì•½ ?€??""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_data = {
                "collection_info": {
                    "strategy": strategy.value,
                    "start_time": self.stats.start_time,
                    "end_time": self.stats.end_time,
                    "duration_seconds": duration.total_seconds(),
                    "duration_str": str(duration)
                },
                "statistics": {
                    "total_collected": total_collected,
                    "total_duplicates": total_duplicates,
                    "total_errors": total_errors,
                    "api_requests_made": self.stats.api_requests_made,
                    "api_errors": self.stats.api_errors,
                    "success_rate": (total_collected / (total_collected + total_errors)) * 100 if (total_collected + total_errors) > 0 else 0
                },
                "collected_decisions": list(self.collected_decisions),
                "metadata": {
                    "collected_at": timestamp,
                    "output_directory": str(output_dir),
                    "total_files": len(list(output_dir.glob("page_*.json")))
                }
            }
            
            summary_file = output_dir / f"{strategy.value}_collection_summary_{timestamp}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"?“Š ?˜ì§‘ ?”ì•½ ?€?? {summary_file.name}")
            
        except Exception as e:
            logger.error(f"?˜ì§‘ ?”ì•½ ?€???¤íŒ¨: {e}")
    
    def collect_by_year(self, year: int, target_count: Optional[int] = None, 
                       unlimited: bool = False, use_final_date: bool = False) -> bool:
        """?¹ì • ?°ë„ ?Œì¬ê²°ì •ë¡€ ?˜ì§‘"""
        try:
            date_type = "ì¢…êµ­?¼ì" if use_final_date else "? ê³ ?¼ì"
            logger.info(f"?—“ï¸?{year}???Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?œì‘ ({date_type} ê¸°ì?)")
            
            # ì¶œë ¥ ?”ë ‰? ë¦¬ ?ì„±
            output_dir = self._create_output_directory(DateCollectionStrategy.YEARLY, year=year)
            
            # ì²´í¬?¬ì¸???•ì¸ (ì¤‘ë‹¨???˜ì§‘ ?¬ê°œ)
            checkpoint = None
            start_page = 1
            if self.resume_mode:
                checkpoint = self._load_checkpoint(output_dir)
                if checkpoint:
                    start_page = checkpoint['checkpoint_info']['last_page'] + 1
                    logger.info(f"?”„ ì¤‘ë‹¨???˜ì§‘ ?¬ê°œ: ?˜ì´ì§€ {start_page}ë¶€???œì‘")
                else:
                    logger.info("?“‹ ì²´í¬?¬ì¸?¸ê? ?†ìŠµ?ˆë‹¤. ì²˜ìŒë¶€???œì‘?©ë‹ˆ??")
            else:
                logger.info("?†• ?ˆë¡œ???˜ì§‘???œì‘?©ë‹ˆ??")
            
            # ê¸°ì¡´ ?°ì´??ë¡œë“œ (?¹ì • ?°ë„ë§?
            self._load_existing_data(target_year=year)
            
            # ? ì§œ ë²”ìœ„ ?¤ì •
            start_date = f"{year}0101"
            end_date = f"{year}1231"
            
            # ëª©í‘œ ê±´ìˆ˜ ?¤ì •
            if unlimited:
                target_count = 999999  # ë¬´ì œ??
            elif target_count is None:
                target_count = 2000  # ê¸°ë³¸ê°?
            
            self.stats.target_count = target_count
            self.stats.start_time = datetime.now().isoformat()
            
            logger.info(f"?“… ?˜ì§‘ ê¸°ê°„: {start_date} ~ {end_date} ({date_type} ê¸°ì?)")
            logger.info(f"?¯ ëª©í‘œ ê±´ìˆ˜: {target_count:,}ê±?)
            logger.info(f"?“ ì¶œë ¥ ?”ë ‰? ë¦¬: {output_dir}")
            
            # ?˜ì§‘ ?¤í–‰
            success = self._collect_decisions_by_date_range(
                start_date=start_date,
                end_date=end_date,
                target_count=target_count,
                output_dir=output_dir,
                category=f"{year}??,
                use_final_date=use_final_date,
                start_page=start_page
            )
            
            # ?˜ì§‘ ?„ë£Œ ì²˜ë¦¬
            self.stats.end_time = datetime.now().isoformat()
            duration = datetime.fromisoformat(self.stats.end_time) - datetime.fromisoformat(self.stats.start_time)
            
            # ?”ì•½ ?€??
            self._save_summary(
                output_dir=output_dir,
                strategy=DateCollectionStrategy.YEARLY,
                total_collected=self.stats.total_collected,
                total_duplicates=self.stats.total_duplicates,
                total_errors=self.stats.total_errors,
                duration=duration
            )
            
            logger.info(f"??{year}???Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?„ë£Œ")
            logger.info(f"?“Š ?˜ì§‘ ê²°ê³¼: {self.stats.total_collected:,}ê±??˜ì§‘, {self.stats.total_duplicates:,}ê±?ì¤‘ë³µ, {self.stats.total_errors:,}ê±??¤ë¥˜")
            logger.info(f"?±ï¸ ?Œìš” ?œê°„: {duration}")
            
            return success
            
        except Exception as e:
            logger.error(f"{year}???Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?¤íŒ¨: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def collect_by_quarter(self, year: int, quarter: int, target_count: int = 500) -> bool:
        """?¹ì • ë¶„ê¸° ?Œì¬ê²°ì •ë¡€ ?˜ì§‘"""
        try:
            logger.info(f"?—“ï¸?{year}??{quarter}ë¶„ê¸° ?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?œì‘")
            
            # ì¶œë ¥ ?”ë ‰? ë¦¬ ?ì„±
            output_dir = self._create_output_directory(DateCollectionStrategy.QUARTERLY, year=year, quarter=quarter)
            
            # ì²´í¬?¬ì¸???•ì¸ (ì¤‘ë‹¨???˜ì§‘ ?¬ê°œ)
            checkpoint = None
            start_page = 1
            if self.resume_mode:
                checkpoint = self._load_checkpoint(output_dir)
                if checkpoint:
                    start_page = checkpoint['checkpoint_info']['last_page'] + 1
                    logger.info(f"?”„ ì¤‘ë‹¨???˜ì§‘ ?¬ê°œ: ?˜ì´ì§€ {start_page}ë¶€???œì‘")
                else:
                    logger.info("?“‹ ì²´í¬?¬ì¸?¸ê? ?†ìŠµ?ˆë‹¤. ì²˜ìŒë¶€???œì‘?©ë‹ˆ??")
            else:
                logger.info("?†• ?ˆë¡œ???˜ì§‘???œì‘?©ë‹ˆ??")
            
            # ë¶„ê¸°ë³?? ì§œ ë²”ìœ„ ?¤ì •
            quarter_months = {
                1: (1, 3),    # 1ë¶„ê¸°: 1-3??
                2: (4, 6),    # 2ë¶„ê¸°: 4-6??
                3: (7, 9),    # 3ë¶„ê¸°: 7-9??
                4: (10, 12)   # 4ë¶„ê¸°: 10-12??
            }
            
            start_month, end_month = quarter_months[quarter]
            start_date = f"{year}{start_month:02d}01"
            end_date = f"{year}{end_month:02d}31"
            
            self.stats.target_count = target_count
            self.stats.start_time = datetime.now().isoformat()
            
            logger.info(f"?“… ?˜ì§‘ ê¸°ê°„: {start_date} ~ {end_date}")
            logger.info(f"?¯ ëª©í‘œ ê±´ìˆ˜: {target_count:,}ê±?)
            logger.info(f"?“ ì¶œë ¥ ?”ë ‰? ë¦¬: {output_dir}")
            
            # ?˜ì§‘ ?¤í–‰
            success = self._collect_decisions_by_date_range(
                start_date=start_date,
                end_date=end_date,
                target_count=target_count,
                output_dir=output_dir,
                category=f"{year}??quarter}ë¶„ê¸°",
                start_page=start_page
            )
            
            # ?˜ì§‘ ?„ë£Œ ì²˜ë¦¬
            self.stats.end_time = datetime.now().isoformat()
            duration = datetime.fromisoformat(self.stats.end_time) - datetime.fromisoformat(self.stats.start_time)
            
            # ?”ì•½ ?€??
            self._save_summary(
                output_dir=output_dir,
                strategy=DateCollectionStrategy.QUARTERLY,
                total_collected=self.stats.total_collected,
                total_duplicates=self.stats.total_duplicates,
                total_errors=self.stats.total_errors,
                duration=duration
            )
            
            logger.info(f"??{year}??{quarter}ë¶„ê¸° ?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?„ë£Œ")
            logger.info(f"?“Š ?˜ì§‘ ê²°ê³¼: {self.stats.total_collected:,}ê±??˜ì§‘, {self.stats.total_duplicates:,}ê±?ì¤‘ë³µ, {self.stats.total_errors:,}ê±??¤ë¥˜")
            logger.info(f"?±ï¸ ?Œìš” ?œê°„: {duration}")
            
            return success
            
        except Exception as e:
            logger.error(f"{year}??{quarter}ë¶„ê¸° ?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?¤íŒ¨: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def collect_by_month(self, year: int, month: int, target_count: int = 200) -> bool:
        """?¹ì • ???Œì¬ê²°ì •ë¡€ ?˜ì§‘"""
        try:
            logger.info(f"?—“ï¸?{year}??{month}???Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?œì‘")
            
            # ì¶œë ¥ ?”ë ‰? ë¦¬ ?ì„±
            output_dir = self._create_output_directory(DateCollectionStrategy.MONTHLY, year=year, month=month)
            
            # ì²´í¬?¬ì¸???•ì¸ (ì¤‘ë‹¨???˜ì§‘ ?¬ê°œ)
            checkpoint = None
            start_page = 1
            if self.resume_mode:
                checkpoint = self._load_checkpoint(output_dir)
                if checkpoint:
                    start_page = checkpoint['checkpoint_info']['last_page'] + 1
                    logger.info(f"?”„ ì¤‘ë‹¨???˜ì§‘ ?¬ê°œ: ?˜ì´ì§€ {start_page}ë¶€???œì‘")
                else:
                    logger.info("?“‹ ì²´í¬?¬ì¸?¸ê? ?†ìŠµ?ˆë‹¤. ì²˜ìŒë¶€???œì‘?©ë‹ˆ??")
            else:
                logger.info("?†• ?ˆë¡œ???˜ì§‘???œì‘?©ë‹ˆ??")
            
            # ?”ë³„ ? ì§œ ë²”ìœ„ ?¤ì •
            start_date = f"{year}{month:02d}01"
            # ?”ë§ ? ì§œ ê³„ì‚°
            if month == 12:
                end_date = f"{year}1231"
            else:
                next_month = month + 1
                next_year = year if next_month <= 12 else year + 1
                if next_month > 12:
                    next_month = 1
                end_date = f"{next_year}{next_month:02d}01"
                # ?˜ë£¨ ?„ìœ¼ë¡??¤ì •
                end_date_obj = datetime.strptime(end_date, "%Y%m%d") - timedelta(days=1)
                end_date = end_date_obj.strftime("%Y%m%d")
            
            self.stats.target_count = target_count
            self.stats.start_time = datetime.now().isoformat()
            
            logger.info(f"?“… ?˜ì§‘ ê¸°ê°„: {start_date} ~ {end_date}")
            logger.info(f"?¯ ëª©í‘œ ê±´ìˆ˜: {target_count:,}ê±?)
            logger.info(f"?“ ì¶œë ¥ ?”ë ‰? ë¦¬: {output_dir}")
            
            # ?˜ì§‘ ?¤í–‰
            success = self._collect_decisions_by_date_range(
                start_date=start_date,
                end_date=end_date,
                target_count=target_count,
                output_dir=output_dir,
                category=f"{year}??month}??,
                start_page=start_page
            )
            
            # ?˜ì§‘ ?„ë£Œ ì²˜ë¦¬
            self.stats.end_time = datetime.now().isoformat()
            duration = datetime.fromisoformat(self.stats.end_time) - datetime.fromisoformat(self.stats.start_time)
            
            # ?”ì•½ ?€??
            self._save_summary(
                output_dir=output_dir,
                strategy=DateCollectionStrategy.MONTHLY,
                total_collected=self.stats.total_collected,
                total_duplicates=self.stats.total_duplicates,
                total_errors=self.stats.total_errors,
                duration=duration
            )
            
            logger.info(f"??{year}??{month}???Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?„ë£Œ")
            logger.info(f"?“Š ?˜ì§‘ ê²°ê³¼: {self.stats.total_collected:,}ê±??˜ì§‘, {self.stats.total_duplicates:,}ê±?ì¤‘ë³µ, {self.stats.total_errors:,}ê±??¤ë¥˜")
            logger.info(f"?±ï¸ ?Œìš” ?œê°„: {duration}")
            
            return success
            
        except Exception as e:
            logger.error(f"{year}??{month}???Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?¤íŒ¨: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _collect_decisions_by_date_range(self, start_date: str, end_date: str, 
                                       target_count: int, output_dir: Path, 
                                       category: str, use_final_date: bool = False, 
                                       start_page: int = 1) -> bool:
        """? ì§œ ë²”ìœ„ë³??Œì¬ê²°ì •ë¡€ ?˜ì§‘ (? ê³ ?¼ì ?´ë¦¼ì°¨ìˆœ)"""
        try:
            page = start_page
            collected_count = 0
            batch_decisions = []
            date_type = "ì¢…êµ­?¼ì" if use_final_date else "? ê³ ?¼ì"
            
            while collected_count < target_count:
                try:
                    # ë©”ëª¨ë¦?ì²´í¬ ë°??•ë¦¬ (ë§?10?˜ì´ì§€ë§ˆë‹¤)
                    if page % 10 == 0:
                        memory_mb = self._check_memory_and_cleanup()
                        logger.info(f"?§  ë©”ëª¨ë¦??¬ìš©?? {memory_mb:.1f}MB")
                    
                    # API ?”ì²­ ì§€??- ?¬ìš©???¤ì • ê°„ê²© ?¬ìš©
                    min_interval = max(0.1, self.request_interval_base - self.request_interval_range)
                    max_interval = self.request_interval_base + self.request_interval_range
                    delay = random.uniform(min_interval, max_interval)
                    time.sleep(delay)
                    
                    # ?Œì¬ê²°ì •ë¡€ ëª©ë¡ ì¡°íšŒ (? ê³ ?¼ì ?´ë¦¼ì°¨ìˆœ)
                    logger.info(f"?“„ ?˜ì´ì§€ {page} ì¡°íšŒ ì¤?.. (?˜ì§‘??ê±´ìˆ˜: {collected_count:,}/{target_count:,}) [{date_type} ê¸°ì?]")
                    
                    # ì¢…êµ­?¼ì ê¸°ì? ?˜ì§‘??ê²½ìš° edYd ?Œë¼ë¯¸í„° ?¬ìš©
                    if use_final_date:
                        # ì¢…êµ­?¼ì ê¸°ê°„ ê²€??(edYd: YYYYMMDD-YYYYMMDD ?•ì‹)
                        edYd_range = f"{start_date}-{end_date}"
                        results = self.client.get_constitutional_list(
                            query=None,  # ?¤ì›Œ???†ì´ ? ì§œ ë²”ìœ„ë¡œë§Œ ê²€??
                            display=100,  # ?˜ì´ì§€??ìµœë? ê±´ìˆ˜
                            page=page,
                            search=1,  # ê²€??ë²”ìœ„
                            sort="efdes",  # ì¢…êµ­?¼ì ?´ë¦¼ì°¨ìˆœ ?•ë ¬
                            edYd=edYd_range  # ì¢…êµ­?¼ì ê¸°ê°„ ê²€??
                        )
                    else:
                        # ? ê³ ?¼ì ê¸°ì? ?˜ì§‘ (ê¸°ì¡´ ë°©ì‹)
                        results = self.client.get_constitutional_list(
                            query=None,  # ?¤ì›Œ???†ì´ ? ì§œ ë²”ìœ„ë¡œë§Œ ê²€??
                            display=100,  # ?˜ì´ì§€??ìµœë? ê±´ìˆ˜
                            page=page,
                            from_date=start_date,
                            to_date=end_date,
                            search=1,  # ê²€??ë²”ìœ„
                            sort="ddes"  # ? ê³ ?¼ì ?´ë¦¼ì°¨ìˆœ ?•ë ¬
                        )
                    
                    if not results:
                        logger.info(f"?“„ ?˜ì´ì§€ {page}: ???´ìƒ ?°ì´?°ê? ?†ìŠµ?ˆë‹¤.")
                        break
                    
                    new_decisions = 0
                    for result in results:
                        if collected_count >= target_count:
                            break
                        
                        # ?Œì¬ê²°ì •ë¡€ ID ?•ì¸
                        decision_id = result.get('?Œì¬ê²°ì •ë¡€?¼ë ¨ë²ˆí˜¸')
                        if not decision_id:
                            continue
                        
                        # ì¤‘ë³µ ?•ì¸
                        if decision_id in self.collected_decisions:
                            self.stats.total_duplicates += 1
                            continue
                        
                        # ?ì„¸ ?•ë³´ ì¡°íšŒ
                        try:
                            detail = self.client.get_constitutional_detail(decision_id)
                            
                            # ConstitutionalDecisionData ê°ì²´ ?ì„± (ëª©ë¡ ?°ì´???´ë???ë³¸ë¬¸ ?°ì´???¬í•¨)
                            decision_data = ConstitutionalDecisionData(
                                # ëª©ë¡ ì¡°íšŒ API ?‘ë‹µ (ê¸°ë³¸ ?•ë³´)
                                id=result.get('id', ''),
                                ?¬ê±´ë²ˆí˜¸=result.get('?¬ê±´ë²ˆí˜¸', ''),
                                ì¢…êµ­?¼ì=result.get('ì¢…êµ­?¼ì', ''),
                                ?Œì¬ê²°ì •ë¡€?¼ë ¨ë²ˆí˜¸=decision_id,
                                ?¬ê±´ëª?result.get('?¬ê±´ëª?, ''),
                                ?Œì¬ê²°ì •ë¡€?ì„¸ë§í¬=result.get('?Œì¬ê²°ì •ë¡€?ì„¸ë§í¬', ''),
                                
                                # ?ì„¸ ì¡°íšŒ API ?‘ë‹µ (ë³¸ë¬¸ ?°ì´??
                                ?¬ê±´ì¢…ë¥˜ëª?detail.get('?¬ê±´ì¢…ë¥˜ëª?) if detail else None,
                                ?ì‹œ?¬í•­=detail.get('?ì‹œ?¬í•­') if detail else None,
                                ê²°ì •?”ì?=detail.get('ê²°ì •?”ì?') if detail else None,
                                ?„ë¬¸=detail.get('?„ë¬¸') if detail else None,
                                ì°¸ì¡°ì¡°ë¬¸=detail.get('ì°¸ì¡°ì¡°ë¬¸') if detail else None,
                                ì°¸ì¡°?ë?=detail.get('ì°¸ì¡°?ë?') if detail else None,
                                ?¬íŒ?€?ì¡°ë¬?detail.get('?¬íŒ?€?ì¡°ë¬?) if detail else None,
                                
                                # ë©”í??°ì´??
                                document_type="constitutional_decision",
                                collected_at=datetime.now().isoformat()
                            )
                            
                            batch_decisions.append(decision_data)
                            self.collected_decisions.add(decision_id)
                            collected_count += 1
                            new_decisions += 1
                            
                            logger.info(f"???ˆë¡œ???Œì¬ê²°ì •ë¡€ ?˜ì§‘: {decision_data.?¬ê±´ëª? (ID: {decision_id})")
                            
                            # ë°°ì¹˜ ?¨ìœ„ë¡?ì¤‘ê°„ ?€??(10ê±´ë§ˆ??
                            if len(batch_decisions) >= 10:
                                self._save_batch(batch_decisions, output_dir, page, category)
                                batch_decisions = []  # ë°°ì¹˜ ì´ˆê¸°??
                            
                        except Exception as e:
                            logger.error(f"?ì„¸ ?•ë³´ ì¡°íšŒ ?¤íŒ¨: {decision_id} - {e}")
                            # ?ì„¸ ?•ë³´ ì¡°íšŒ ?¤íŒ¨?´ë„ ê¸°ë³¸ ?•ë³´???€??
                            decision_data = ConstitutionalDecisionData(
                                # ëª©ë¡ ì¡°íšŒ API ?‘ë‹µ (ê¸°ë³¸ ?•ë³´)
                                id=result.get('id', ''),
                                ?¬ê±´ë²ˆí˜¸=result.get('?¬ê±´ë²ˆí˜¸', ''),
                                ì¢…êµ­?¼ì=result.get('ì¢…êµ­?¼ì', ''),
                                ?Œì¬ê²°ì •ë¡€?¼ë ¨ë²ˆí˜¸=decision_id,
                                ?¬ê±´ëª?result.get('?¬ê±´ëª?, ''),
                                ?Œì¬ê²°ì •ë¡€?ì„¸ë§í¬=result.get('?Œì¬ê²°ì •ë¡€?ì„¸ë§í¬', ''),
                                
                                # ?ì„¸ ì¡°íšŒ ?¤íŒ¨ë¡?ë³¸ë¬¸ ?°ì´?°ëŠ” None
                                ?¬ê±´ì¢…ë¥˜ëª?None,
                                ?ì‹œ?¬í•­=None,
                                ê²°ì •?”ì?=None,
                                ?„ë¬¸=None,
                                ì°¸ì¡°ì¡°ë¬¸=None,
                                ì°¸ì¡°?ë?=None,
                                ?¬íŒ?€?ì¡°ë¬?None,
                                
                                # ë©”í??°ì´??
                                document_type="constitutional_decision",
                                collected_at=datetime.now().isoformat()
                            )
                            
                            batch_decisions.append(decision_data)
                            self.collected_decisions.add(decision_id)
                            collected_count += 1
                            new_decisions += 1
                            
                            logger.warning(f"? ï¸ ê¸°ë³¸ ?•ë³´ë§??˜ì§‘: {decision_data.?¬ê±´ëª? (ID: {decision_id})")
                            
                            # ë°°ì¹˜ ?¨ìœ„ë¡?ì¤‘ê°„ ?€??(10ê±´ë§ˆ??
                            if len(batch_decisions) >= 10:
                                self._save_batch(batch_decisions, output_dir, page, category)
                                batch_decisions = []  # ë°°ì¹˜ ì´ˆê¸°??
                            self.stats.total_errors += 1
                    
                    # ë°°ì¹˜ ?€??(100ê±´ë§ˆ?? - ì¶”ê? ?ˆì „?¥ì¹˜
                    if len(batch_decisions) >= 100:
                        self._save_batch(batch_decisions, output_dir, page, category)
                        batch_decisions = []
                    
                    logger.info(f"?“„ ?˜ì´ì§€ {page} ?„ë£Œ: {new_decisions}ê±´ì˜ ?ˆë¡œ??ê²°ì •ë¡€ ?˜ì§‘")
                    logger.info(f"   ?“Š ?„ì  ?˜ì§‘: {collected_count:,}/{target_count:,}ê±?({collected_count/target_count*100:.1f}%)")
                    
                    page += 1
                    self.stats.api_requests_made += 1
                    
                    # API ?”ì²­ ?œí•œ ?•ì¸
                    stats = self.client.get_request_stats()
                    if stats['remaining_requests'] < 10:
                        logger.warning("API ?”ì²­ ?œë„ê°€ ê±°ì˜ ?Œì§„?˜ì—ˆ?µë‹ˆ??")
                        break
                    
                except Exception as e:
                    logger.error(f"?˜ì´ì§€ {page} ì²˜ë¦¬ ì¤??¤ë¥˜: {e}")
                    self.stats.api_errors += 1
                    self.error_count += 1
                    
                    if self.error_count >= self.max_errors:
                        logger.error(f"ìµœë? ?¤ë¥˜ ?Ÿìˆ˜({self.max_errors})???„ë‹¬?ˆìŠµ?ˆë‹¤.")
                        break
                    
                    # ?¬ì‹œ??ì§€??
                    time.sleep(self.stats.retry_delay)
                    continue
            
            # ?¨ì? ë°°ì¹˜ ?€??
            if batch_decisions:
                self._save_batch(batch_decisions, output_dir, page, category)
            
            self.stats.total_collected = collected_count
            logger.info(f"?¯ ?˜ì§‘ ?„ë£Œ: {collected_count:,}ê±??˜ì§‘")
            
            return True
            
        except Exception as e:
            logger.error(f"? ì§œ ë²”ìœ„ë³??˜ì§‘ ?¤íŒ¨: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def collect_multiple_years(self, start_year: int, end_year: int, 
                              target_per_year: int = 2000) -> bool:
        """?¬ëŸ¬ ?°ë„ ?Œì¬ê²°ì •ë¡€ ?˜ì§‘"""
        try:
            logger.info(f"?—“ï¸?{start_year}??~ {end_year}???Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?œì‘")
            
            total_success = True
            for year in range(start_year, end_year + 1):
                logger.info(f"?“… {year}???˜ì§‘ ?œì‘...")
                success = self.collect_by_year(year, target_per_year)
                if not success:
                    logger.warning(f"? ï¸ {year}???˜ì§‘ ?¤íŒ¨")
                    total_success = False
                else:
                    logger.info(f"??{year}???˜ì§‘ ?„ë£Œ")
                
                # ?°ë„ ê°?ì§€??
                time.sleep(5)
            
            logger.info(f"?¯ ?¤ì¤‘ ?°ë„ ?˜ì§‘ ?„ë£Œ: {start_year}??~ {end_year}??)
            return total_success
            
        except Exception as e:
            logger.error(f"?¤ì¤‘ ?°ë„ ?˜ì§‘ ?¤íŒ¨: {e}")
            logger.error(traceback.format_exc())
            return False
