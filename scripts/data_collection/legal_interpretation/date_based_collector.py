#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ë²•ë ¹?´ì„ë¡€ ? ì§œ ê¸°ë°˜ ?˜ì§‘ê¸?

??ëª¨ë“ˆ?€ ? ì§œë³„ë¡œ ì²´ê³„?ì¸ ë²•ë ¹?´ì„ë¡€ ?˜ì§‘???˜í–‰?©ë‹ˆ??
- ?°ë„ë³? ë¶„ê¸°ë³? ?”ë³„ ?˜ì§‘ ?„ëµ
- ?´ì„?¼ì ?´ë¦¼ì°¨ìˆœ ìµœì ??
- ?´ë”ë³?raw ?°ì´???€??êµ¬ì¡°
- ì¤‘ë³µ ë°©ì? ë°?ì²´í¬?¬ì¸??ì§€??
"""

import sys
import os
import json
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

# source ëª¨ë“ˆ ê²½ë¡œ ì¶”ê?
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'source'))

from data.law_open_api_client import LawOpenAPIClient
import logging

logger = logging.getLogger(__name__)


class DateCollectionStrategy(Enum):
    """? ì§œ ?˜ì§‘ ?„ëµ"""
    YEARLY = "yearly"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"


@dataclass
class CollectionConfig:
    """?˜ì§‘ ?¤ì • ?´ë˜??""
    base_output_dir: Path = Path("data/raw/legal_interpretations")
    max_retries: int = 3
    retry_delay: int = 5
    api_delay_range: Tuple[float, float] = (1.0, 3.0)
    output_subdir: Optional[str] = None


@dataclass
class LegalInterpretationData:
    """ë²•ë ¹?´ì„ë¡€ ?°ì´???´ë˜??- ëª©ë¡ ?°ì´???´ë???ë³¸ë¬¸ ?°ì´???¬í•¨"""
    # ëª©ë¡ ì¡°íšŒ API ?‘ë‹µ (ê¸°ë³¸ ?•ë³´)
    id: str  # ê²€?‰ê²°ê³¼ë²ˆ??
    ë²•ë ¹?´ì„ë¡€?¼ë ¨ë²ˆí˜¸: str
    ?ˆê±´ëª? str
    ?ˆê±´ë²ˆí˜¸: str
    ì§ˆì˜ê¸°ê?ì½”ë“œ: str
    ì§ˆì˜ê¸°ê?ëª? str
    ?Œì‹ ê¸°ê?ì½”ë“œ: str
    ?Œì‹ ê¸°ê?ëª? str
    ?Œì‹ ?¼ì: str
    ë²•ë ¹?´ì„ë¡€?ì„¸ë§í¬: str
    
    # ?ì„¸ ì¡°íšŒ API ?‘ë‹µ (ë³¸ë¬¸ ?°ì´??
    ?´ì„?¼ì: Optional[str] = None
    ?´ì„ê¸°ê?ì½”ë“œ: Optional[str] = None
    ?´ì„ê¸°ê?ëª? Optional[str] = None
    ê´€ë¦¬ê¸°ê´€ì½”ë“œ: Optional[str] = None
    ?±ë¡?¼ì‹œ: Optional[str] = None
    ì§ˆì˜?”ì?: Optional[str] = None
    ?Œë‹µ: Optional[str] = None
    ?´ìœ : Optional[str] = None
    
    # ë©”í??°ì´??
    document_type: str = "legal_interpretation"
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


class DateBasedLegalInterpretationCollector:
    """? ì§œ ê¸°ë°˜ ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ ?´ë˜??""
    
    def __init__(self, config: CollectionConfig = None):
        self.config = config or CollectionConfig()
        # ê°„ë‹¨???¤ì •?¼ë¡œ API ?´ë¼?´ì–¸???ì„±
        class SimpleConfig:
            def __init__(self):
                self.law_open_api_oc = os.getenv('LAW_OPEN_API_OC', '{OC}')
                self.oc = os.getenv('LAW_OPEN_API_OC', '{OC}')  # API ?´ë¼?´ì–¸?¸ê? oc ?ì„±??ì°¾ìŒ
                self.base_url = 'https://www.law.go.kr/DRF/'
                self.rate_limit = 1000
                self.request_timeout = 30
                self.connect_timeout = 30
                self.timeout = 30
                self.max_retries = 3
                self.retry_delay = 5
                self.retry_delay_base = 1
                self.retry_delay_max = 60
                self.law_firm_ai_api_key = os.getenv('LAW_FIRM_AI_API_KEY', 'your-api-key-here')
                self.api_host = '0.0.0.0'
                self.api_port = 8000
                self.debug = False
                self.database_url = 'sqlite:///./data/lawfirm.db'
                self.database_path = './data/lawfirm.db'
                self.model_path = './models'
                self.device = 'cpu'
                self.model_cache_dir = './model_cache'
                self.chroma_db_path = './data/chroma_db'
                self.embedding_model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        
        api_config = SimpleConfig()
        self.client = LawOpenAPIClient(api_config)
        self.stats = CollectionStats()
        self.collected_decisions: Set[str] = set()
        
        # ì¶œë ¥ ?”ë ‰? ë¦¬ ?ì„±
        self.config.base_output_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_output_directory(self, strategy: DateCollectionStrategy, 
                               year: int = None, quarter: int = None, 
                               month: int = None) -> Path:
        """ì¶œë ¥ ?”ë ‰? ë¦¬ ?ì„±"""
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
                dir_name = f"monthly_{year}{month:02d}_{timestamp}"
            else:
                dir_name = f"monthly_collection_{timestamp}"
        else:
            dir_name = f"daily_collection_{timestamp}"
        
        output_dir = self.config.base_output_dir / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _save_batch(self, interpretations: List[LegalInterpretationData], 
                   output_dir: Path, page_num: int, 
                   category: str = "legal_interpretation") -> bool:
        """ë°°ì¹˜ ?°ì´???€??""
        try:
            if not interpretations:
                return True
            
            # ?Œì¼ëª??ì„±
            start_id = interpretations[0].ë²•ë ¹?´ì„ë¡€?¼ë ¨ë²ˆí˜¸
            end_id = interpretations[-1].ë²•ë ¹?´ì„ë¡€?¼ë ¨ë²ˆí˜¸
            count = len(interpretations)
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
                "interpretations": [interpretation.__dict__ for interpretation in interpretations]
            }
            
            # JSON ?Œì¼ ?€??
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"??ë°°ì¹˜ ?€???„ë£Œ: {filename} ({count}ê±?")
            
            # ì²´í¬?¬ì¸???€??(ì§„í–‰ ?í™© ê¸°ë¡)
            self._save_checkpoint(output_dir, page_num, collected_count=len(interpretations))
            
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
                    "success_rate": (total_collected / (total_collected + total_errors) * 100) if (total_collected + total_errors) > 0 else 0
                },
                "collected_interpretations": list(self.collected_decisions),
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
                       unlimited: bool = False, use_interpretation_date: bool = True) -> bool:
        """?¹ì • ?°ë„ ë²•ë ¹?´ì„ë¡€ ?˜ì§‘"""
        try:
            date_type = "?´ì„?¼ì" if use_interpretation_date else "?Œì‹ ?¼ì"
            logger.info(f"?—“ï¸?{year}??ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ ?œì‘ ({date_type} ê¸°ì?)")
            
            # ì¶œë ¥ ?”ë ‰? ë¦¬ ?ì„±
            output_dir = self._create_output_directory(DateCollectionStrategy.YEARLY, year=year)
            
            # ì²´í¬?¬ì¸???•ì¸ (ì¤‘ë‹¨???˜ì§‘ ?¬ê°œ)
            checkpoint = self._load_checkpoint(output_dir)
            start_page = 1
            if checkpoint:
                start_page = checkpoint['checkpoint_info']['last_page'] + 1
                logger.info(f"?”„ ì¤‘ë‹¨???˜ì§‘ ?¬ê°œ: ?˜ì´ì§€ {start_page}ë¶€???œì‘")
            
            # ê¸°ì¡´ ?°ì´??ë¡œë“œ (?¹ì • ?°ë„ë§?
            self._load_existing_data(target_year=year)
            
            # ? ì§œ ë²”ìœ„ ?¤ì •
            start_date = f"{year}0101"
            end_date = f"{year}1231"
            
            # ëª©í‘œ ê±´ìˆ˜ ?¤ì •
            if unlimited:
                target_count = 2000  # ë¬´ì œ?œì´ì§€ë§??ˆì „???„í•´ ?œí•œ
            elif target_count is None:
                target_count = 1000  # ê¸°ë³¸ê°?
            
            self.stats.start_time = datetime.now().isoformat()
            self.stats.target_count = target_count
            self.stats.status = "RUNNING"
            
            logger.info(f"?“… ?˜ì§‘ ê¸°ê°„: {start_date} ~ {end_date} ({date_type} ê¸°ì?)")
            logger.info(f"?¯ ëª©í‘œ ê±´ìˆ˜: {target_count:,}ê±?)
            logger.info(f"?“ ì¶œë ¥ ?”ë ‰? ë¦¬: {output_dir}")
            
            # ?˜ì§‘ ?¤í–‰
            success = self._collect_interpretations_by_date_range(
                start_date=start_date,
                end_date=end_date,
                target_count=target_count,
                output_dir=output_dir,
                category=f"{year}??,
                use_interpretation_date=use_interpretation_date,
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
            
            logger.info(f"??{year}??ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ ?„ë£Œ")
            logger.info(f"?“Š ?˜ì§‘ ê²°ê³¼: {self.stats.total_collected:,}ê±??˜ì§‘, {self.stats.total_duplicates:,}ê±?ì¤‘ë³µ, {self.stats.total_errors:,}ê±??¤ë¥˜")
            logger.info(f"?±ï¸ ?Œìš” ?œê°„: {duration}")
            
            return success
            
        except Exception as e:
            logger.error(f"{year}??ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ ?¤íŒ¨: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def collect_by_quarter(self, year: int, quarter: int, target_count: int = 500) -> bool:
        """?¹ì • ë¶„ê¸° ë²•ë ¹?´ì„ë¡€ ?˜ì§‘"""
        try:
            logger.info(f"?—“ï¸?{year}??{quarter}ë¶„ê¸° ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ ?œì‘")
            
            # ì¶œë ¥ ?”ë ‰? ë¦¬ ?ì„±
            output_dir = self._create_output_directory(DateCollectionStrategy.QUARTERLY, year=year, quarter=quarter)
            
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
            
            self.stats.start_time = datetime.now().isoformat()
            self.stats.target_count = target_count
            self.stats.status = "RUNNING"
            
            logger.info(f"?“… ?˜ì§‘ ê¸°ê°„: {start_date} ~ {end_date}")
            logger.info(f"?¯ ëª©í‘œ ê±´ìˆ˜: {target_count:,}ê±?)
            logger.info(f"?“ ì¶œë ¥ ?”ë ‰? ë¦¬: {output_dir}")
            
            # ?˜ì§‘ ?¤í–‰
            success = self._collect_interpretations_by_date_range(
                start_date=start_date,
                end_date=end_date,
                target_count=target_count,
                output_dir=output_dir,
                category=f"{year}??quarter}ë¶„ê¸°",
                start_page=1
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
            
            logger.info(f"??{year}??{quarter}ë¶„ê¸° ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ ?„ë£Œ")
            logger.info(f"?“Š ?˜ì§‘ ê²°ê³¼: {self.stats.total_collected:,}ê±??˜ì§‘, {self.stats.total_duplicates:,}ê±?ì¤‘ë³µ, {self.stats.total_errors:,}ê±??¤ë¥˜")
            logger.info(f"?±ï¸ ?Œìš” ?œê°„: {duration}")
            
            return success
            
        except Exception as e:
            logger.error(f"{year}??{quarter}ë¶„ê¸° ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ ?¤íŒ¨: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def collect_by_month(self, year: int, month: int, target_count: int = 200) -> bool:
        """?¹ì • ??ë²•ë ¹?´ì„ë¡€ ?˜ì§‘"""
        try:
            logger.info(f"?—“ï¸?{year}??{month}??ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ ?œì‘")
            
            # ì¶œë ¥ ?”ë ‰? ë¦¬ ?ì„±
            output_dir = self._create_output_directory(DateCollectionStrategy.MONTHLY, year=year, month=month)
            
            # ?”ë³„ ? ì§œ ë²”ìœ„ ?¤ì •
            start_date = f"{year}{month:02d}01"
            # ?”ë§ ? ì§œ ê³„ì‚°
            if month == 12:
                end_date = f"{year}1231"
            else:
                next_month = month + 1
                end_date = f"{year}{next_month:02d}01"
            
            self.stats.start_time = datetime.now().isoformat()
            self.stats.target_count = target_count
            self.stats.status = "RUNNING"
            
            logger.info(f"?“… ?˜ì§‘ ê¸°ê°„: {start_date} ~ {end_date}")
            logger.info(f"?¯ ëª©í‘œ ê±´ìˆ˜: {target_count:,}ê±?)
            logger.info(f"?“ ì¶œë ¥ ?”ë ‰? ë¦¬: {output_dir}")
            
            # ?˜ì§‘ ?¤í–‰
            success = self._collect_interpretations_by_date_range(
                start_date=start_date,
                end_date=end_date,
                target_count=target_count,
                output_dir=output_dir,
                category=f"{year}??month}??,
                start_page=1
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
            
            logger.info(f"??{year}??{month}??ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ ?„ë£Œ")
            logger.info(f"?“Š ?˜ì§‘ ê²°ê³¼: {self.stats.total_collected:,}ê±??˜ì§‘, {self.stats.total_duplicates:,}ê±?ì¤‘ë³µ, {self.stats.total_errors:,}ê±??¤ë¥˜")
            logger.info(f"?±ï¸ ?Œìš” ?œê°„: {duration}")
            
            return success
            
        except Exception as e:
            logger.error(f"{year}??{month}??ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ ?¤íŒ¨: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _collect_interpretations_by_date_range(self, start_date: str, end_date: str, 
                                             target_count: int, output_dir: Path, 
                                             category: str, use_interpretation_date: bool = True, 
                                             start_page: int = 1) -> bool:
        """? ì§œ ë²”ìœ„ë³?ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ (?´ì„?¼ì ?´ë¦¼ì°¨ìˆœ)"""
        try:
            page = start_page
            collected_count = 0
            batch_interpretations = []
            date_type = "?´ì„?¼ì" if use_interpretation_date else "?Œì‹ ?¼ì"
            
            while collected_count < target_count:
                try:
                    logger.info(f"?“„ ?˜ì´ì§€ {page} ì²˜ë¦¬ ì¤?..")
                    
                    # API ?¸ì¶œ
                    if use_interpretation_date:
                        # ?´ì„?¼ì ê¸°ì? ê²€??
                        interpretations = self.client.get_legal_interpretation_list(
                            page=page,
                            display=100,
                            explYd=f"{start_date}~{end_date}",
                            sort="ddes"  # ?´ì„?¼ì ?´ë¦¼ì°¨ìˆœ
                        )
                    else:
                        # ?Œì‹ ?¼ì ê¸°ì? ê²€??
                        interpretations = self.client.get_legal_interpretation_list(
                            page=page,
                            display=100,
                            regYd=f"{start_date}~{end_date}",
                            sort="ddes"  # ?Œì‹ ?¼ì ?´ë¦¼ì°¨ìˆœ
                        )
                    
                    if not interpretations:
                        logger.info(f"?“„ ?˜ì´ì§€ {page}: ???´ìƒ ?°ì´?°ê? ?†ìŠµ?ˆë‹¤.")
                        break
                    
                    self.stats.api_requests_made += 1
                    new_interpretations = 0
                    
                    for result in interpretations:
                        interpretation_id = result.get('ë²•ë ¹?´ì„ë¡€?¼ë ¨ë²ˆí˜¸', '')
                        
                        if not interpretation_id:
                            continue
                        
                        # ì¤‘ë³µ ?•ì¸
                        if interpretation_id in self.collected_decisions:
                            self.stats.total_duplicates += 1
                            continue
                        
                        # ?ì„¸ ?•ë³´ ì¡°íšŒ
                        try:
                            detail = self.client.get_legal_interpretation_detail(interpretation_id)
                            
                            # LegalInterpretationData ê°ì²´ ?ì„± (ëª©ë¡ ?°ì´???´ë???ë³¸ë¬¸ ?°ì´???¬í•¨)
                            interpretation_data = LegalInterpretationData(
                                # ëª©ë¡ ì¡°íšŒ API ?‘ë‹µ (ê¸°ë³¸ ?•ë³´)
                                id=result.get('id', ''),
                                ë²•ë ¹?´ì„ë¡€?¼ë ¨ë²ˆí˜¸=interpretation_id,
                                ?ˆê±´ëª?result.get('?ˆê±´ëª?, ''),
                                ?ˆê±´ë²ˆí˜¸=result.get('?ˆê±´ë²ˆí˜¸', ''),
                                ì§ˆì˜ê¸°ê?ì½”ë“œ=result.get('ì§ˆì˜ê¸°ê?ì½”ë“œ', ''),
                                ì§ˆì˜ê¸°ê?ëª?result.get('ì§ˆì˜ê¸°ê?ëª?, ''),
                                ?Œì‹ ê¸°ê?ì½”ë“œ=result.get('?Œì‹ ê¸°ê?ì½”ë“œ', ''),
                                ?Œì‹ ê¸°ê?ëª?result.get('?Œì‹ ê¸°ê?ëª?, ''),
                                ?Œì‹ ?¼ì=result.get('?Œì‹ ?¼ì', ''),
                                ë²•ë ¹?´ì„ë¡€?ì„¸ë§í¬=result.get('ë²•ë ¹?´ì„ë¡€?ì„¸ë§í¬', ''),
                                
                                # ?ì„¸ ì¡°íšŒ API ?‘ë‹µ (ë³¸ë¬¸ ?°ì´??
                                ?´ì„?¼ì=detail.get('?´ì„?¼ì') if detail else None,
                                ?´ì„ê¸°ê?ì½”ë“œ=detail.get('?´ì„ê¸°ê?ì½”ë“œ') if detail else None,
                                ?´ì„ê¸°ê?ëª?detail.get('?´ì„ê¸°ê?ëª?) if detail else None,
                                ê´€ë¦¬ê¸°ê´€ì½”ë“œ=detail.get('ê´€ë¦¬ê¸°ê´€ì½”ë“œ') if detail else None,
                                ?±ë¡?¼ì‹œ=detail.get('?±ë¡?¼ì‹œ') if detail else None,
                                ì§ˆì˜?”ì?=detail.get('ì§ˆì˜?”ì?') if detail else None,
                                ?Œë‹µ=detail.get('?Œë‹µ') if detail else None,
                                ?´ìœ =detail.get('?´ìœ ') if detail else None,
                                
                                # ë©”í??°ì´??
                                document_type="legal_interpretation",
                                collected_at=datetime.now().isoformat()
                            )
                            
                            batch_interpretations.append(interpretation_data)
                            self.collected_decisions.add(interpretation_id)
                            collected_count += 1
                            new_interpretations += 1
                            
                            logger.info(f"???ˆë¡œ??ë²•ë ¹?´ì„ë¡€ ?˜ì§‘: {interpretation_data.?ˆê±´ëª? (ID: {interpretation_id})")
                            
                            # ë°°ì¹˜ ?¨ìœ„ë¡?ì¤‘ê°„ ?€??(10ê±´ë§ˆ??
                            if len(batch_interpretations) >= 10:
                                self._save_batch(batch_interpretations, output_dir, page, category)
                                batch_interpretations = []  # ë°°ì¹˜ ì´ˆê¸°??
                            
                        except Exception as e:
                            logger.error(f"?ì„¸ ?•ë³´ ì¡°íšŒ ?¤íŒ¨: {interpretation_id} - {e}")
                            # ?ì„¸ ?•ë³´ ì¡°íšŒ ?¤íŒ¨?´ë„ ê¸°ë³¸ ?•ë³´???€??
                            interpretation_data = LegalInterpretationData(
                                # ëª©ë¡ ì¡°íšŒ API ?‘ë‹µ (ê¸°ë³¸ ?•ë³´)
                                id=result.get('id', ''),
                                ë²•ë ¹?´ì„ë¡€?¼ë ¨ë²ˆí˜¸=interpretation_id,
                                ?ˆê±´ëª?result.get('?ˆê±´ëª?, ''),
                                ?ˆê±´ë²ˆí˜¸=result.get('?ˆê±´ë²ˆí˜¸', ''),
                                ì§ˆì˜ê¸°ê?ì½”ë“œ=result.get('ì§ˆì˜ê¸°ê?ì½”ë“œ', ''),
                                ì§ˆì˜ê¸°ê?ëª?result.get('ì§ˆì˜ê¸°ê?ëª?, ''),
                                ?Œì‹ ê¸°ê?ì½”ë“œ=result.get('?Œì‹ ê¸°ê?ì½”ë“œ', ''),
                                ?Œì‹ ê¸°ê?ëª?result.get('?Œì‹ ê¸°ê?ëª?, ''),
                                ?Œì‹ ?¼ì=result.get('?Œì‹ ?¼ì', ''),
                                ë²•ë ¹?´ì„ë¡€?ì„¸ë§í¬=result.get('ë²•ë ¹?´ì„ë¡€?ì„¸ë§í¬', ''),
                                
                                # ?ì„¸ ì¡°íšŒ ?¤íŒ¨ë¡?ë³¸ë¬¸ ?°ì´?°ëŠ” None
                                ?´ì„?¼ì=None,
                                ?´ì„ê¸°ê?ì½”ë“œ=None,
                                ?´ì„ê¸°ê?ëª?None,
                                ê´€ë¦¬ê¸°ê´€ì½”ë“œ=None,
                                ?±ë¡?¼ì‹œ=None,
                                ì§ˆì˜?”ì?=None,
                                ?Œë‹µ=None,
                                ?´ìœ =None,
                                
                                # ë©”í??°ì´??
                                document_type="legal_interpretation",
                                collected_at=datetime.now().isoformat()
                            )
                            
                            batch_interpretations.append(interpretation_data)
                            self.collected_decisions.add(interpretation_id)
                            collected_count += 1
                            new_interpretations += 1
                            
                            logger.warning(f"? ï¸ ê¸°ë³¸ ?•ë³´ë§??˜ì§‘: {interpretation_data.?ˆê±´ëª? (ID: {interpretation_id})")
                            
                            # ë°°ì¹˜ ?¨ìœ„ë¡?ì¤‘ê°„ ?€??(10ê±´ë§ˆ??
                            if len(batch_interpretations) >= 10:
                                self._save_batch(batch_interpretations, output_dir, page, category)
                                batch_interpretations = []  # ë°°ì¹˜ ì´ˆê¸°??
                            self.stats.total_errors += 1
                    
                    # ë°°ì¹˜ ?€??(100ê±´ë§ˆ?? - ì¶”ê? ?ˆì „?¥ì¹˜
                    if len(batch_interpretations) >= 100:
                        self._save_batch(batch_interpretations, output_dir, page, category)
                        batch_interpretations = []
                    
                    logger.info(f"?“„ ?˜ì´ì§€ {page} ?„ë£Œ: {new_interpretations}ê±´ì˜ ?ˆë¡œ???´ì„ë¡€ ?˜ì§‘")
                    logger.info(f"   ?“Š ?„ì  ?˜ì§‘: {collected_count:,}/{target_count:,}ê±?({collected_count/target_count*100:.1f}%)")
                    
                    page += 1
                    
                    # API ?¸ì¶œ ê°„ê²© ì¡°ì ˆ
                    time.sleep(1.0)
                    
                except Exception as e:
                    logger.error(f"?˜ì´ì§€ {page} ì²˜ë¦¬ ì¤??¤ë¥˜: {e}")
                    self.stats.total_errors += 1
                    
                    # ?¬ì‹œ??ë¡œì§
                    if self.stats.total_errors < self.config.max_retries:
                        logger.info(f"?¬ì‹œ??ì¤?.. ({self.stats.total_errors}/{self.config.max_retries})")
                        time.sleep(self.stats.retry_delay)
                        continue
                    else:
                        logger.error(f"ìµœë? ?¬ì‹œ???Ÿìˆ˜ ì´ˆê³¼. ?˜ì§‘??ì¤‘ë‹¨?©ë‹ˆ??")
                        break
            
            # ?¨ì? ë°°ì¹˜ ?€??
            if batch_interpretations:
                self._save_batch(batch_interpretations, output_dir, page, category)
            
            logger.info(f"??? ì§œ ë²”ìœ„ë³??˜ì§‘ ?„ë£Œ: {collected_count:,}ê±??˜ì§‘")
            return True
            
        except Exception as e:
            logger.error(f"? ì§œ ë²”ìœ„ë³??˜ì§‘ ?¤íŒ¨: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _load_existing_data(self, target_year: int = None):
        """ê¸°ì¡´ ?˜ì§‘???°ì´??ë¡œë“œ (ì¤‘ë³µ ë°©ì?)"""
        try:
            # ê¸°ì¡´ ?°ì´???”ë ‰? ë¦¬ ?¤ìº”
            for year_dir in self.config.base_output_dir.glob("yearly_*"):
                if target_year and str(target_year) not in str(year_dir):
                    continue
                
                # JSON ?Œì¼?¤ì—???´ë? ?˜ì§‘??ID ì¶”ì¶œ
                for json_file in year_dir.glob("page_*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        if 'interpretations' in data:
                            for interpretation in data['interpretations']:
                                interpretation_id = interpretation.get('ë²•ë ¹?´ì„ë¡€?¼ë ¨ë²ˆí˜¸', '')
                                if interpretation_id:
                                    self.collected_decisions.add(interpretation_id)
                    except Exception as e:
                        logger.warning(f"ê¸°ì¡´ ?°ì´??ë¡œë“œ ?¤íŒ¨: {json_file} - {e}")
            
            logger.info(f"?“‹ ê¸°ì¡´ ?˜ì§‘??ë²•ë ¹?´ì„ë¡€: {len(self.collected_decisions):,}ê±?)
            
        except Exception as e:
            logger.warning(f"ê¸°ì¡´ ?°ì´??ë¡œë“œ ?¤íŒ¨: {e}")


if __name__ == "__main__":
    # ?ŒìŠ¤???¤í–‰
    collector = DateBasedLegalInterpretationCollector()
    
    # 2025??ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ ?ŒìŠ¤??(10ê±?
    success = collector.collect_by_year(2025, target_count=10, use_interpretation_date=True)
    
    if success:
        print("??ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ ?ŒìŠ¤???±ê³µ")
    else:
        print("??ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ ?ŒìŠ¤???¤íŒ¨")
