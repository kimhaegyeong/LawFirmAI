# -*- coding: utf-8 -*-
"""
ë²•ë¥  ?©ì–´ ?˜ì§‘ê¸?(ë©”ëª¨ë¦?ìµœì ??ë°?ì²´í¬?¬ì¸??ì§€??

êµ??ë²•ë ¹?•ë³´?¼í„° OpenAPIë¥??œìš©?˜ì—¬ ë²•ë¥  ?©ì–´ë¥??˜ì§‘?˜ê³  ê´€ë¦¬í•©?ˆë‹¤.
- ë©”ëª¨ë¦??¨ìœ¨?ì¸ ë°°ì¹˜ ì²˜ë¦¬
- ì²´í¬?¬ì¸???œìŠ¤?œìœ¼ë¡?ì¤‘ë‹¨ ???¬ê°œ ê°€??
- ?¤ì‹œê°?ì§„í–‰ë¥?ëª¨ë‹ˆ?°ë§
- ë©”ëª¨ë¦??¬ìš©??ì¶”ì  ë°?ìµœì ??
"""

import os
import sys
import json
import gc
import psutil
import logging
import signal
import atexit
import glob
import time
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.legal_term_collection_api import LegalTermCollectionAPI, TermCollectionConfig
from source.data.legal_term_dictionary import LegalTermDictionary

logger = logging.getLogger(__name__)


@dataclass
class CheckpointData:
    """ì²´í¬?¬ì¸???°ì´??êµ¬ì¡°"""
    session_id: str
    start_time: str
    last_update: str
    total_target: int
    collected_count: int
    processed_categories: List[str] = field(default_factory=list)
    processed_keywords: List[str] = field(default_factory=list)
    current_category: Optional[str] = None
    current_keyword: Optional[str] = None
    memory_usage_mb: float = 0.0
    api_requests_made: int = 0
    errors_count: int = 0
    can_resume: bool = True
    # ?˜ì´ì§€?¤ì´???•ë³´ ì¶”ê?
    current_page: int = 1
    page_size: int = 100
    consecutive_empty_pages: int = 0
    last_page_terms_count: int = 0


@dataclass
class MemoryConfig:
    """ë©”ëª¨ë¦?ê´€ë¦??¤ì •"""
    max_memory_mb: int = 2048  # ìµœë? ë©”ëª¨ë¦??¬ìš©??(MB)
    batch_size: int = 10  # ë°°ì¹˜ ?¬ê¸° (10ê°œì”© ì²˜ë¦¬)
    checkpoint_interval: int = 2  # ì²´í¬?¬ì¸???€??ê°„ê²© (2ê°?ë°°ì¹˜ë§ˆë‹¤ ?€??- ???ì£¼ ?€??
    gc_threshold: int = 500  # ê°€ë¹„ì? ì»¬ë ‰???„ê³„ê°?
    memory_check_interval: int = 10  # ë©”ëª¨ë¦?ì²´í¬ ê°„ê²©
    batch_delay_min: float = 1.0  # ë°°ì¹˜ ê°?ìµœì†Œ ì§€???œê°„ (ì´?
    batch_delay_max: float = 3.0  # ë°°ì¹˜ ê°?ìµœë? ì§€???œê°„ (ì´?


class LegalTermCollector:
    """ë²•ë¥  ?©ì–´ ?˜ì§‘ê¸??´ë˜??(ë©”ëª¨ë¦?ìµœì ??ë°?ì²´í¬?¬ì¸??ì§€??"""
    
    def __init__(self, config: TermCollectionConfig = None, memory_config: MemoryConfig = None):
        """?˜ì§‘ê¸?ì´ˆê¸°??""
        self.config = config or self._create_default_config()
        self.memory_config = memory_config or MemoryConfig()
        self.api_client = LegalTermCollectionAPI(self.config)
        self.dictionary = LegalTermDictionary()
        
        # ì²´í¬?¬ì¸??ê´€??
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_file = Path("data/raw/legal_terms/checkpoint.json")
        self.checkpoint_data = None
        
        # ?™ì  ?Œì¼ëª??ì„± (?¸ì…˜ë³?êµ¬ë¶„)
        self.dynamic_file_prefix = None
        self.dictionary_file = None
        self.shutdown_requested = False
        
        # ë©”ëª¨ë¦?ê´€ë¦?
        self.process = psutil.Process()
        self.memory_check_counter = 0
        self.gc_counter = 0
        
        # ê¸°ë³¸ ?µê³„
        self.stats = {
            'total_collected': 0,
            'start_time': None,
            'end_time': None
        }
        
        # ?œê·¸???¸ë“¤???±ë¡
        self._setup_signal_handlers()
        
        # ì¢…ë£Œ ???•ë¦¬ ?‘ì—… ?±ë¡
        atexit.register(self._cleanup_on_exit)
        
        logger.info("LegalTermCollector initialized with memory optimization")
    
    def _create_default_config(self) -> TermCollectionConfig:
        """ê¸°ë³¸ ?¤ì • ?ì„±"""
        config = TermCollectionConfig()
        config.batch_size = 10  # ê¸°ë³¸ ë°°ì¹˜ ?¬ê¸°
        config.delay_between_requests = 0.05
        config.max_retries = 3
        return config
    
    def _setup_signal_handlers(self):
        """?œê·¸???¸ë“¤???¤ì • (graceful shutdown ì§€??"""
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            logger.info(f"?œê·¸??{signal_name}({signum}) ?˜ì‹  - graceful shutdown ?œì‘")
            
            # ì¤‘ë³µ ì¢…ë£Œ ?”ì²­ ë°©ì?
            if self.shutdown_requested:
                logger.warning("?´ë? ì¢…ë£Œ ?”ì²­??ì§„í–‰ ì¤‘ì…?ˆë‹¤. ê°•ì œ ì¢…ë£Œ?©ë‹ˆ??")
                sys.exit(1)
            
            self.shutdown_requested = True
            
            # ì§„í–‰ ?í™© ë¡œê¹…
            if self.checkpoint_data:
                progress = (self.checkpoint_data.collected_count / self.checkpoint_data.total_target * 100) if self.checkpoint_data.total_target > 0 else 0
                logger.info(f"?„ì¬ ì§„í–‰ë¥? {progress:.1f}% ({self.checkpoint_data.collected_count}/{self.checkpoint_data.total_target})")
            
            # ì²´í¬?¬ì¸???€??
            try:
                self._save_checkpoint()
                logger.info("ì²´í¬?¬ì¸???€???„ë£Œ")
            except Exception as e:
                logger.error(f"ì²´í¬?¬ì¸???€???¤íŒ¨: {e}")
            
            # ?•ë¦¬ ?‘ì—… ?˜í–‰
            self._perform_cleanup()
            
            logger.info("graceful shutdown ?„ë£Œ")
            sys.exit(0)
        
        # ?¤ì–‘???œê·¸?ì— ?€???¸ë“¤???±ë¡
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)    # ì¢…ë£Œ ?”ì²­
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal_handler)  # ?°ë????°ê²° ?Šê?
        if hasattr(signal, 'SIGQUIT'):
            signal.signal(signal.SIGQUIT, signal_handler)  # ì¢…ë£Œ + ì½”ì–´ ?¤í”„
    
    def _perform_cleanup(self):
        """?•ë¦¬ ?‘ì—… ?˜í–‰ (graceful shutdown)"""
        logger.info("?•ë¦¬ ?‘ì—… ?œì‘...")
        
        try:
            # 1. ì§„í–‰ ì¤‘ì¸ ?‘ì—… ?„ë£Œ ?€ê¸?
            logger.info("ì§„í–‰ ì¤‘ì¸ ?‘ì—… ?„ë£Œ ?€ê¸?ì¤?..")
            time.sleep(1)  # ?„ì¬ ì§„í–‰ ì¤‘ì¸ ?‘ì—…???„ë£Œ???œê°„ ?œê³µ
            
            # 2. ë©”ëª¨ë¦??•ë¦¬
            logger.info("ë©”ëª¨ë¦??•ë¦¬ ì¤?..")
            self._force_garbage_collection()
            
            # 3. ?µê³„ ?•ë³´ ë¡œê¹…
            if self.checkpoint_data:
                logger.info("=" * 50)
                logger.info("?˜ì§‘ ?¸ì…˜ ?”ì•½:")
                logger.info(f"  ?¸ì…˜ ID: {self.checkpoint_data.session_id}")
                logger.info(f"  ?˜ì§‘???©ì–´: {self.checkpoint_data.collected_count}ê°?)
                logger.info(f"  ëª©í‘œ ?©ì–´: {self.checkpoint_data.total_target}ê°?)
                logger.info(f"  ì§„í–‰ë¥? {self.checkpoint_data.collected_count/self.checkpoint_data.total_target*100:.1f}%")
                logger.info(f"  API ?”ì²­ ?? {self.checkpoint_data.api_requests_made}")
                logger.info(f"  ?ëŸ¬ ?? {self.checkpoint_data.errors_count}")
                logger.info(f"  ë©”ëª¨ë¦??¬ìš©?? {self.checkpoint_data.memory_usage_mb:.1f}MB")
                logger.info("=" * 50)
            
            # 4. ë¦¬ì†Œ???•ë¦¬
            logger.info("ë¦¬ì†Œ???•ë¦¬ ì¤?..")
            if hasattr(self, 'api_client') and hasattr(self.api_client, 'session'):
                self.api_client.session.close()
            
            # 5. ?„ì‹œ ?Œì¼ ?•ë¦¬ (?„ìš”??ê²½ìš°)
            self._cleanup_temp_files()
            
            logger.info("?•ë¦¬ ?‘ì—… ?„ë£Œ")
            
        except Exception as e:
            logger.error(f"?•ë¦¬ ?‘ì—… ì¤??¤ë¥˜ ë°œìƒ: {e}")
    
    def _cleanup_on_exit(self):
        """ì¢…ë£Œ ???•ë¦¬ ?‘ì—… (atexit ?¸ë“¤??"""
        try:
            self._perform_cleanup()
        except Exception as e:
            logger.error(f"ì¢…ë£Œ ???•ë¦¬ ?‘ì—… ?¤íŒ¨: {e}")
    
    def _cleanup_temp_files(self):
        """?„ì‹œ ?Œì¼ ?•ë¦¬"""
        try:
            # ?„ì‹œ ?Œì¼???ˆë‹¤ë©??•ë¦¬
            temp_patterns = [
                "data/raw/legal_terms/temp_*.json",
                "data/raw/legal_terms/*.tmp",
                "data/raw/legal_terms/session_*/temp_*.json",
                "data/raw/legal_terms/session_*/*.tmp",
                "logs/temp_*.log"
            ]
            
            for pattern in temp_patterns:
                temp_files = glob.glob(pattern)
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                        logger.debug(f"?„ì‹œ ?Œì¼ ?? œ: {temp_file}")
                    except OSError:
                        pass  # ?Œì¼???´ë? ?? œ?˜ì—ˆê±°ë‚˜ ?‘ê·¼?????†ìŒ
                        
        except Exception as e:
            logger.debug(f"?„ì‹œ ?Œì¼ ?•ë¦¬ ì¤??¤ë¥˜ (ë¬´ì‹œ??: {e}")
    
    def _get_memory_usage_mb(self) -> float:
        """?„ì¬ ë©”ëª¨ë¦??¬ìš©??ì¡°íšŒ (MB)"""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # MB ?¨ìœ„
        except Exception as e:
            logger.warning(f"ë©”ëª¨ë¦??¬ìš©??ì¡°íšŒ ?¤íŒ¨: {e}")
            return 0.0
    
    def _check_memory_limit(self) -> bool:
        """ë©”ëª¨ë¦??¬ìš©???œí•œ ?•ì¸"""
        current_memory = self._get_memory_usage_mb()
        
        
        # ë©”ëª¨ë¦?ì²´í¬ ì¹´ìš´??ì¦ê?
        self.memory_check_counter += 1
        
        # ë©”ëª¨ë¦??œí•œ ì´ˆê³¼ ??ê°€ë¹„ì? ì»¬ë ‰???¤í–‰
        if current_memory > self.memory_config.max_memory_mb:
            logger.warning(f"ë©”ëª¨ë¦??¬ìš©??ì´ˆê³¼: {current_memory:.1f}MB > {self.memory_config.max_memory_mb}MB")
            self._force_garbage_collection()
            return False
        
        return True
    
    def _force_garbage_collection(self):
        """ê°•ì œ ê°€ë¹„ì? ì»¬ë ‰???¤í–‰"""
        gc.collect()
    
    def _save_checkpoint(self) -> bool:
        """ì²´í¬?¬ì¸???€??(ë¹„ì •??ì¢…ë£Œ ë°©ì?)"""
        try:
            if not self.checkpoint_data:
                return False
            
            # ?„ì¬ ?íƒœ ?…ë°?´íŠ¸
            self.checkpoint_data.last_update = datetime.now().isoformat()
            self.checkpoint_data.collected_count = len(self.dictionary.terms)
            self.checkpoint_data.memory_usage_mb = self._get_memory_usage_mb()
            self.checkpoint_data.api_requests_made = self.api_client.request_count
            
            # ì²´í¬?¬ì¸???Œì¼ ?€??(?ì???°ê¸°)
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            
            # ?„ì‹œ ?Œì¼??ë¨¼ì? ?°ê³  ?˜ì¤‘???´ë¦„ ë³€ê²?(?ì???°ê¸°)
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.checkpoint_data.__dict__, f, ensure_ascii=False, indent=2)
                
                # ?„ì‹œ ?Œì¼???¤ì œ ?Œì¼ë¡??´ë¦„ ë³€ê²?(?ì???°ì‚°)
                temp_file.replace(self.checkpoint_file)
                
                logger.debug(f"ì²´í¬?¬ì¸???€???„ë£Œ: {self.checkpoint_data.collected_count}ê°??©ì–´")
                return True
                
            except Exception as e:
                # ?„ì‹œ ?Œì¼ ?•ë¦¬
                if temp_file.exists():
                    temp_file.unlink()
                raise e
            
        except Exception as e:
            logger.error(f"ì²´í¬?¬ì¸???€???¤íŒ¨: {e}")
            return False
    
    def _load_checkpoint(self) -> Optional[CheckpointData]:
        """ì²´í¬?¬ì¸??ë¡œë“œ (ê¸°ì¡´ ?¸ì…˜ ?°ì„  ê²€?? ë¹„ì •??ì¢…ë£Œ ë³µêµ¬ ì§€??"""
        try:
            # 1. ?„ì¬ ?¤ì •??ì²´í¬?¬ì¸???Œì¼ ?•ì¸
            if self.checkpoint_file.exists():
                checkpoint_data = self._load_checkpoint_file(self.checkpoint_file)
                if checkpoint_data:
                    logger.info(f"?„ì¬ ?¸ì…˜ ì²´í¬?¬ì¸??ë¡œë“œ ?„ë£Œ: {checkpoint_data.collected_count}ê°??©ì–´ ?˜ì§‘??)
                    return checkpoint_data
            
            # 2. ëª¨ë“  ?¸ì…˜ ?´ë”?ì„œ ì²´í¬?¬ì¸???Œì¼ ê²€??(ë¹„ì •??ì¢…ë£Œ ë³µêµ¬)
            session_folders = list(Path("data/raw/legal_terms").glob("session_*"))
            if not session_folders:
                logger.info("ê¸°ì¡´ ?¸ì…˜ ?´ë”ê°€ ?†ìŠµ?ˆë‹¤.")
                return None
            
            # ì²´í¬?¬ì¸???Œì¼?¤ì„ ?˜ì§‘?˜ê³  ?•ë ¬
            all_checkpoints = []
            for session_folder in session_folders:
                checkpoint_files = list(session_folder.glob("checkpoint_*.json"))
                for checkpoint_file in checkpoint_files:
                    try:
                        checkpoint_data = self._load_checkpoint_file(checkpoint_file)
                        if checkpoint_data:
                            # ì²´í¬?¬ì¸???Œì¼???˜ì • ?œê°„??ê³ ë ¤
                            file_mtime = checkpoint_file.stat().st_mtime
                            all_checkpoints.append((checkpoint_data, checkpoint_file, file_mtime))
                    except Exception as e:
                        logger.debug(f"ì²´í¬?¬ì¸???Œì¼ ?½ê¸° ?¤íŒ¨ ({checkpoint_file}): {e}")
                        continue
            
            if not all_checkpoints:
                logger.info("?¬ìš© ê°€?¥í•œ ì²´í¬?¬ì¸???Œì¼???†ìŠµ?ˆë‹¤.")
                return None
            
            # ê°€??ìµœê·¼ ì²´í¬?¬ì¸??? íƒ (?œê°„???•ë ¬)
            all_checkpoints.sort(key=lambda x: x[2], reverse=True)  # ?Œì¼ ?˜ì • ?œê°„ ê¸°ì?
            
            # ?„ì¬ ?°ë„?€ ë§¤ì¹­?˜ëŠ” ì²´í¬?¬ì¸???°ì„  ? íƒ
            current_year = self._extract_current_year()
            matching_checkpoints = []
            other_checkpoints = []
            
            for checkpoint_data, checkpoint_file, file_mtime in all_checkpoints:
                folder_year = self._extract_folder_year(checkpoint_file.parent.name)
                if current_year and folder_year and current_year == folder_year:
                    matching_checkpoints.append((checkpoint_data, checkpoint_file, file_mtime))
                else:
                    other_checkpoints.append((checkpoint_data, checkpoint_file, file_mtime))
            
            # ?°ì„ ?œìœ„: ê°™ì? ?°ë„ > ?¤ë¥¸ ?°ë„ > ê°€??ìµœê·¼
            selected_checkpoint = None
            if matching_checkpoints:
                selected_checkpoint = matching_checkpoints[0]  # ê°™ì? ?°ë„ ì¤?ê°€??ìµœê·¼
                logger.info(f"ê°™ì? ?°ë„({current_year}) ì²´í¬?¬ì¸??ë°œê²¬")
            elif other_checkpoints:
                selected_checkpoint = other_checkpoints[0]  # ?¤ë¥¸ ?°ë„ ì¤?ê°€??ìµœê·¼
                logger.info("?¤ë¥¸ ?°ë„ ì²´í¬?¬ì¸??ë°œê²¬")
            
            if selected_checkpoint:
                checkpoint_data, checkpoint_file, file_mtime = selected_checkpoint
                
                # ?„ì¬ ?¸ì…˜?¼ë¡œ ?¤ì •
                self.checkpoint_file = checkpoint_file
                self.session_id = checkpoint_data.session_id
                
                # ê¸°ì¡´ ?¬ì „ ?Œì¼??ë¡œë“œ
                self._load_existing_dictionary_files(checkpoint_file.parent)
                
                logger.info(f"?”„ ê¸°ì¡´ ?¸ì…˜ ì²´í¬?¬ì¸??ë¡œë“œ ?„ë£Œ: {checkpoint_data.collected_count}ê°??©ì–´ ?˜ì§‘??)
                logger.info(f"?“ ?¬ìš©??ì²´í¬?¬ì¸?? {checkpoint_file}")
                logger.info(f"?†” ?ë³¸ ?¸ì…˜ ID: {checkpoint_data.session_id}")
                logger.info(f"?†” ?„ì¬ ?¸ì…˜ ID: {self.session_id}")
                logger.info(f"??ì²´í¬?¬ì¸???˜ì • ?œê°„: {datetime.fromtimestamp(file_mtime)}")
                logger.info(f"?“Š ëª©í‘œ ?©ì–´ ?? {checkpoint_data.total_target}ê°?)
                logger.info(f"?“Š ì§„í–‰ë¥? {checkpoint_data.collected_count/checkpoint_data.total_target*100:.1f}%")
                
                return checkpoint_data
            
            logger.info("?¬ìš© ê°€?¥í•œ ì²´í¬?¬ì¸???Œì¼???†ìŠµ?ˆë‹¤.")
            return None
            
        except Exception as e:
            logger.error(f"ì²´í¬?¬ì¸??ë¡œë“œ ?¤íŒ¨: {e}")
            return None
    
    def _load_checkpoint_file(self, checkpoint_file: Path) -> Optional[CheckpointData]:
        """ì²´í¬?¬ì¸???Œì¼ ë¡œë“œ ë°?ê²€ì¦?""
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_dict = json.load(f)
            
            # ?„ìˆ˜ ?„ë“œ ê²€ì¦?
            required_fields = ['session_id', 'collected_count', 'total_target']
            for field in required_fields:
                if field not in checkpoint_dict:
                    logger.warning(f"ì²´í¬?¬ì¸???Œì¼???„ìˆ˜ ?„ë“œ '{field}'ê°€ ?†ìŠµ?ˆë‹¤: {checkpoint_file}")
                    return None
            
            checkpoint_data = CheckpointData(**checkpoint_dict)
            
            # ?°ì´??ë¬´ê²°??ê²€ì¦?
            if checkpoint_data.collected_count < 0:
                logger.warning(f"ì²´í¬?¬ì¸???°ì´??ë¬´ê²°???¤ë¥˜ (?Œìˆ˜ ?˜ì§‘ ??: {checkpoint_file}")
                return None
            
            if checkpoint_data.total_target <= 0:
                logger.warning(f"ì²´í¬?¬ì¸???°ì´??ë¬´ê²°???¤ë¥˜ (?˜ëª»??ëª©í‘œ ??: {checkpoint_file}")
                return None
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"ì²´í¬?¬ì¸???Œì¼ ë¡œë“œ ?¤íŒ¨ ({checkpoint_file}): {e}")
            return None
    
    def _extract_current_year(self) -> Optional[str]:
        """?„ì¬ ?¤ì •?ì„œ ?°ë„ ì¶”ì¶œ"""
        try:
            if hasattr(self, 'dynamic_file_prefix') and self.dynamic_file_prefix:
                parts = self.dynamic_file_prefix.split('_')
                if 'year' in parts:
                    year_index = parts.index('year')
                    if year_index + 1 < len(parts):
                        return parts[year_index + 1]
        except Exception:
            pass
        return None
    
    def _extract_folder_year(self, folder_name: str) -> Optional[str]:
        """?´ë”ëª…ì—???°ë„ ì¶”ì¶œ"""
        try:
            if 'year_' in folder_name:
                year_part = folder_name.split('year_')[1]
                return year_part
        except Exception:
            pass
        return None
    
    def _load_existing_dictionary_files(self, session_folder: Path):
        """ê¸°ì¡´ ?¬ì „ ?Œì¼??ë¡œë“œ"""
        try:
            # ë©”ì¸ ?¬ì „ ?Œì¼ ì°¾ê¸°
            dictionary_files = list(session_folder.glob("legal_terms_*.json"))
            if dictionary_files:
                # ê°€??ìµœê·¼ ?¬ì „ ?Œì¼ ?¬ìš©
                latest_dictionary = max(dictionary_files, key=lambda f: f.stat().st_mtime)
                self.dictionary_file = latest_dictionary
                logger.info(f"ê¸°ì¡´ ?¬ì „ ?Œì¼ ë¡œë“œ: {latest_dictionary}")
                
                # ?¬ì „ ?°ì´??ë¡œë“œ
                if self.dictionary_file.exists():
                    self.dictionary.load_terms_from_file(str(self.dictionary_file))
                    logger.info(f"ê¸°ì¡´ ?¬ì „ ?°ì´??ë¡œë“œ ?„ë£Œ: {len(self.dictionary.terms)}ê°??©ì–´")
            else:
                logger.info("ê¸°ì¡´ ?¬ì „ ?Œì¼???†ìŠµ?ˆë‹¤.")
                
        except Exception as e:
            logger.error(f"ê¸°ì¡´ ?¬ì „ ?Œì¼ ë¡œë“œ ?¤íŒ¨: {e}")
    
    def _merge_existing_data(self, target_year: int):
        """ê¸°ì¡´ ?°ì´??ë³‘í•© (ê°™ì? ?°ë„???¤ë¥¸ ?¸ì…˜?ì„œ)"""
        try:
            # ê°™ì? ?°ë„???¤ë¥¸ ?¸ì…˜ ?´ë”??ì°¾ê¸°
            session_folders = list(Path("data/raw/legal_terms").glob(f"session_*_year_{target_year}"))
            
            merged_count = 0
            for session_folder in session_folders:
                # ?„ì¬ ?¸ì…˜ ?´ë”???œì™¸
                if session_folder == self.dictionary_file.parent:
                    continue
                
                # ?´ë‹¹ ?¸ì…˜???¬ì „ ?Œì¼??ë¡œë“œ
                dictionary_files = list(session_folder.glob("legal_terms_*.json"))
                for dict_file in dictionary_files:
                    try:
                        # ?„ì‹œ ?¬ì „ ê°ì²´ë¡?ë¡œë“œ
                        temp_dict = LegalTermDictionary()
                        temp_dict.load_terms_from_file(str(dict_file))
                        
                        # ?„ì¬ ?¬ì „??ë³‘í•©
                        for term_id, term_data in temp_dict.terms.items():
                            if term_id not in self.dictionary.terms:
                                self.dictionary.add_term(term_data)
                                merged_count += 1
                        
                        logger.info(f"?¸ì…˜ ë³‘í•© ?„ë£Œ: {dict_file.name} ({len(temp_dict.terms)}ê°??©ì–´)")
                        
                    except Exception as e:
                        logger.debug(f"?¸ì…˜ ë³‘í•© ?¤íŒ¨ ({dict_file}): {e}")
                        continue
            
            if merged_count > 0:
                logger.info(f"ê¸°ì¡´ ?°ì´??ë³‘í•© ?„ë£Œ: {merged_count}ê°??©ì–´ ì¶”ê?")
                self.checkpoint_data.collected_count = len(self.dictionary.terms)
            else:
                logger.info("ë³‘í•©??ê¸°ì¡´ ?°ì´?°ê? ?†ìŠµ?ˆë‹¤.")
                
        except Exception as e:
            logger.error(f"ê¸°ì¡´ ?°ì´??ë³‘í•© ?¤íŒ¨: {e}")
    
    def _create_checkpoint_data(self, total_target: int) -> CheckpointData:
        """ì²´í¬?¬ì¸???°ì´???ì„±"""
        return CheckpointData(
            session_id=self.session_id,
            start_time=datetime.now().isoformat(),
            last_update=datetime.now().isoformat(),
            total_target=total_target,
            collected_count=0,
            memory_usage_mb=self._get_memory_usage_mb(),
            api_requests_made=0,
            errors_count=0,
            can_resume=True
        )
    
    def collect_all_terms(self, max_terms: int = 5000, use_mock_data: bool = False, resume: bool = True) -> bool:
        """ëª¨ë“  ë²•ë¥  ?©ì–´ ?˜ì§‘ (ë©”ëª¨ë¦?ìµœì ??ë°?ì²´í¬?¬ì¸??ì§€??"""
        logger.info(f"?„ì²´ ë²•ë¥  ?©ì–´ ?˜ì§‘ ?œì‘ - ìµœë? {max_terms}ê°?(ë©”ëª¨ë¦?ìµœì ??")
        
        # ?™ì  ?Œì¼ëª??¤ì •
        self._setup_dynamic_filenames("all")
        
        # ì²´í¬?¬ì¸??ë¡œë“œ ?ëŠ” ?ì„±
        if resume:
            self.checkpoint_data = self._load_checkpoint()
        
        if not self.checkpoint_data:
            self.checkpoint_data = self._create_checkpoint_data(max_terms)
            logger.info("?ˆë¡œ???˜ì§‘ ?¸ì…˜ ?œì‘")
        else:
            # ê¸°ì¡´ ?¬ì „ ë¡œë“œ ???¤ì œ ?©ì–´ ?˜ë¡œ ?™ê¸°??
            self.load_dictionary()
            actual_count = len(self.dictionary.terms)
            if actual_count != self.checkpoint_data.collected_count:
                logger.info(f"ì²´í¬?¬ì¸?¸ì? ?¤ì œ ?¬ì „ ?©ì–´ ??ë¶ˆì¼ì¹?- ì²´í¬?¬ì¸?? {self.checkpoint_data.collected_count}ê°? ?¤ì œ: {actual_count}ê°?)
                logger.info(f"?¤ì œ ?¬ì „ ?©ì–´ ?˜ë¡œ ?™ê¸°?? {actual_count}ê°?)
                self.checkpoint_data.collected_count = actual_count
            else:
                logger.info(f"ê¸°ì¡´ ?˜ì§‘ ?¸ì…˜ ?¬ê°œ: {self.checkpoint_data.collected_count}ê°??©ì–´ ?˜ì§‘??)
        
        self.stats['start_time'] = datetime.now()
        
        try:
            # ë°°ì¹˜ ?¨ìœ„ë¡??©ì–´ ?˜ì§‘
            success = self._collect_terms_in_batches(max_terms, use_mock_data)
            
            self.stats['end_time'] = datetime.now()
            self._log_collection_summary()
            
            return success
            
        except Exception as e:
            logger.error(f"?©ì–´ ?˜ì§‘ ?¤íŒ¨: {e}")
            self.checkpoint_data.errors_count += 1
            self._save_checkpoint()
            return False
    
    def _collect_terms_in_batches(self, max_terms: int, use_mock_data: bool) -> bool:
        """ë°°ì¹˜ ?¨ìœ„ë¡??©ì–´ ?˜ì§‘"""
        batch_size = self.memory_config.batch_size
        collected_count = self.checkpoint_data.collected_count
        remaining_terms = max_terms - collected_count
        
        logger.info(f"ë°°ì¹˜ ?˜ì§‘ ?œì‘ - ë°°ì¹˜ ?¬ê¸°: {batch_size}, ?¨ì? ?©ì–´: {remaining_terms}ê°?)
        
        batch_count = 0
        while collected_count < max_terms and not self.shutdown_requested:
            # ë©”ëª¨ë¦?ì²´í¬
            if not self._check_memory_limit():
                logger.warning("ë©”ëª¨ë¦??œí•œ?¼ë¡œ ?¸í•œ ?¼ì‹œ ì¤‘ë‹¨")
                break
            
            # ?„ì¬ ë°°ì¹˜ ?¬ê¸° ê³„ì‚°
            current_batch_size = min(batch_size, remaining_terms)
            
            try:
                # ë°°ì¹˜ ?¨ìœ„ë¡??©ì–´ ?˜ì§‘
                batch_terms = self.api_client.collect_legal_terms(
                    max_terms=current_batch_size, 
                    use_mock_data=use_mock_data
                )
                
                if not batch_terms:
                    logger.info("???´ìƒ ?˜ì§‘???©ì–´ê°€ ?†ìŠµ?ˆë‹¤.")
                    break
                
                # ?¬ì „??ë°°ì¹˜ ?©ì–´ ì¶”ê?
                batch_success = self._add_terms_to_dictionary_batch(batch_terms)
                
                if batch_success:
                    # ?¤ì œë¡??¬ì „???€?¥ëœ ?©ì–´ ?˜ë¡œ ?…ë°?´íŠ¸
                    collected_count = len(self.dictionary.terms)
                    self.checkpoint_data.collected_count = collected_count
                    remaining_terms = max_terms - collected_count
                    
                    logger.info(f"ë°°ì¹˜ {batch_count + 1} ?„ë£Œ - ?˜ì§‘???©ì–´: {collected_count}/{max_terms}ê°?)
                    
                    # ì²´í¬?¬ì¸???€??
                    if (batch_count + 1) % self.memory_config.checkpoint_interval == 0:
                        self._save_checkpoint()
                
                batch_count += 1
                
                # ë°°ì¹˜ ê°??œë¤ ì§€??(2~5ì´?
                if not self.shutdown_requested:
                    delay = random.uniform(self.memory_config.batch_delay_min, self.memory_config.batch_delay_max)
                    logger.info(f"ë°°ì¹˜ ê°?ì§€?? {delay:.1f}ì´?)
                    time.sleep(delay)
                
                # ê°€ë¹„ì? ì»¬ë ‰???„ê³„ê°?ì²´í¬
                if batch_count % self.memory_config.gc_threshold == 0:
                    self._force_garbage_collection()
                
            except Exception as e:
                logger.error(f"ë°°ì¹˜ {batch_count + 1} ?˜ì§‘ ?¤íŒ¨: {e}")
                self.checkpoint_data.errors_count += 1
                
                # ?°ì† ?ëŸ¬ ??ì¤‘ë‹¨
                if self.checkpoint_data.errors_count > 10:
                    logger.error("?°ì† ?ëŸ¬ë¡??¸í•œ ?˜ì§‘ ì¤‘ë‹¨")
                    break
        
        # ìµœì¢… ì²´í¬?¬ì¸???€??(?¤ì œ ?¬ì „ ?©ì–´ ?˜ë¡œ ?…ë°?´íŠ¸)
        self.checkpoint_data.collected_count = len(self.dictionary.terms)
        self._save_checkpoint()
        
        logger.info(f"ë°°ì¹˜ ?˜ì§‘ ?„ë£Œ - ì´?{collected_count}ê°??©ì–´ ?˜ì§‘")
        return collected_count > 0
    
    def collect_terms_by_categories(self, categories: List[str], max_terms_per_category: int = 500, resume: bool = True) -> bool:
        """ì¹´í…Œê³ ë¦¬ë³??©ì–´ ?˜ì§‘ (ë©”ëª¨ë¦?ìµœì ??ë°?ì²´í¬?¬ì¸??ì§€??"""
        logger.info(f"ì¹´í…Œê³ ë¦¬ë³??©ì–´ ?˜ì§‘ ?œì‘ - {len(categories)}ê°?ì¹´í…Œê³ ë¦¬ (ë©”ëª¨ë¦?ìµœì ??")
        
        # ?™ì  ?Œì¼ëª??¤ì •
        self._setup_dynamic_filenames("categories")
        
        # ì²´í¬?¬ì¸??ë¡œë“œ ?ëŠ” ?ì„±
        if resume:
            self.checkpoint_data = self._load_checkpoint()
        
        if not self.checkpoint_data:
            total_target = len(categories) * max_terms_per_category
            self.checkpoint_data = self._create_checkpoint_data(total_target)
            logger.info("?ˆë¡œ??ì¹´í…Œê³ ë¦¬ë³??˜ì§‘ ?¸ì…˜ ?œì‘")
        else:
            logger.info(f"ê¸°ì¡´ ì¹´í…Œê³ ë¦¬ë³??˜ì§‘ ?¸ì…˜ ?¬ê°œ: {self.checkpoint_data.collected_count}ê°??©ì–´ ?˜ì§‘??)
        
        self.stats['start_time'] = datetime.now()
        
        try:
            # ì¹´í…Œê³ ë¦¬ë³?ë°°ì¹˜ ?˜ì§‘
            success = self._collect_categories_in_batches(categories, max_terms_per_category)
            
            self.stats['end_time'] = datetime.now()
            self._log_collection_summary()
            
            return success
            
        except Exception as e:
            logger.error(f"ì¹´í…Œê³ ë¦¬ë³??©ì–´ ?˜ì§‘ ?¤íŒ¨: {e}")
            self.checkpoint_data.errors_count += 1
            self._save_checkpoint()
            return False
    
    def _collect_categories_in_batches(self, categories: List[str], max_terms_per_category: int) -> bool:
        """ì¹´í…Œê³ ë¦¬ë³?ë°°ì¹˜ ?˜ì§‘"""
        batch_size = self.memory_config.batch_size
        processed_categories = set(self.checkpoint_data.processed_categories)
        
        logger.info(f"ì¹´í…Œê³ ë¦¬ë³?ë°°ì¹˜ ?˜ì§‘ ?œì‘ - ë°°ì¹˜ ?¬ê¸°: {batch_size}")
        
        for category in categories:
            if self.shutdown_requested:
                logger.info("ì¤‘ë‹¨ ?”ì²­?¼ë¡œ ?¸í•œ ?˜ì§‘ ì¤‘ë‹¨")
                break
            
            if category in processed_categories:
                logger.info(f"ì¹´í…Œê³ ë¦¬ '{category}' ?´ë? ì²˜ë¦¬??- ê±´ë„ˆ?°ê¸°")
                continue
            
            logger.info(f"ì¹´í…Œê³ ë¦¬ '{category}' ?˜ì§‘ ?œì‘...")
            
            try:
                # ì¹´í…Œê³ ë¦¬ë³?ë°°ì¹˜ ?˜ì§‘
                category_success = self._collect_single_category_batches(category, max_terms_per_category, batch_size)
                
                if category_success:
                    self.checkpoint_data.processed_categories.append(category)
                    logger.info(f"ì¹´í…Œê³ ë¦¬ '{category}' ?„ë£Œ")
                else:
                    logger.warning(f"ì¹´í…Œê³ ë¦¬ '{category}' ?˜ì§‘ ?¤íŒ¨")
                
                # ì²´í¬?¬ì¸???€??
                self._save_checkpoint()
                
                # ë©”ëª¨ë¦?ì²´í¬ ë°?ê°€ë¹„ì? ì»¬ë ‰??
                if not self._check_memory_limit():
                    logger.warning("ë©”ëª¨ë¦??œí•œ?¼ë¡œ ?¸í•œ ?¼ì‹œ ì¤‘ë‹¨")
                    break
                
            except Exception as e:
                logger.error(f"ì¹´í…Œê³ ë¦¬ '{category}' ?˜ì§‘ ì¤??¤ë¥˜: {e}")
                self.checkpoint_data.errors_count += 1
        
        logger.info(f"ì¹´í…Œê³ ë¦¬ë³?ë°°ì¹˜ ?˜ì§‘ ?„ë£Œ - ì´?{self.checkpoint_data.collected_count}ê°??©ì–´ ?˜ì§‘")
        return self.checkpoint_data.collected_count > 0
    
    def _collect_single_category_batches(self, category: str, max_terms: int, batch_size: int) -> bool:
        """?¨ì¼ ì¹´í…Œê³ ë¦¬ ë°°ì¹˜ ?˜ì§‘"""
        collected_count = 0
        batch_count = 0
        
        while collected_count < max_terms and not self.shutdown_requested:
            # ë©”ëª¨ë¦?ì²´í¬
            if not self._check_memory_limit():
                break
            
            # ?„ì¬ ë°°ì¹˜ ?¬ê¸° ê³„ì‚°
            current_batch_size = min(batch_size, max_terms - collected_count)
            
            try:
                # ë°°ì¹˜ ?¨ìœ„ë¡??©ì–´ ?˜ì§‘
                batch_terms = self.api_client.collect_legal_terms(category, current_batch_size)
                
                if not batch_terms:
                    logger.info(f"ì¹´í…Œê³ ë¦¬ '{category}'?ì„œ ???´ìƒ ?˜ì§‘???©ì–´ê°€ ?†ìŠµ?ˆë‹¤.")
                    break
                
                # ?¬ì „??ë°°ì¹˜ ?©ì–´ ì¶”ê?
                batch_success = self._add_terms_to_dictionary_batch(batch_terms)
                
                if batch_success:
                    # ?¤ì œë¡??¬ì „???€?¥ëœ ?©ì–´ ?˜ë¡œ ?…ë°?´íŠ¸
                    collected_count = len(self.dictionary.terms)
                    self.checkpoint_data.collected_count = collected_count
                    
                    logger.info(f"ì¹´í…Œê³ ë¦¬ '{category}' ë°°ì¹˜ {batch_count + 1} ?„ë£Œ - ?˜ì§‘???©ì–´: {collected_count}/{max_terms}ê°?)
                
                batch_count += 1
                
                # ë°°ì¹˜ ê°??œë¤ ì§€??(2~5ì´?
                if not self.shutdown_requested:
                    delay = random.uniform(self.memory_config.batch_delay_min, self.memory_config.batch_delay_max)
                    logger.info(f"ì¹´í…Œê³ ë¦¬ '{category}' ë°°ì¹˜ ê°?ì§€?? {delay:.1f}ì´?)
                    time.sleep(delay)
                
                # ê°€ë¹„ì? ì»¬ë ‰???„ê³„ê°?ì²´í¬
                if batch_count % self.memory_config.gc_threshold == 0:
                    self._force_garbage_collection()
                
            except Exception as e:
                logger.error(f"ì¹´í…Œê³ ë¦¬ '{category}' ë°°ì¹˜ {batch_count + 1} ?˜ì§‘ ?¤íŒ¨: {e}")
                self.checkpoint_data.errors_count += 1
                break
        
        return collected_count > 0
    
    def collect_terms_by_keywords(self, keywords: List[str], max_terms_per_keyword: int = 100, resume: bool = True) -> bool:
        """?¤ì›Œ?œë³„ ?©ì–´ ?˜ì§‘ (ë©”ëª¨ë¦?ìµœì ??ë°?ì²´í¬?¬ì¸??ì§€??"""
        logger.info(f"?¤ì›Œ?œë³„ ?©ì–´ ?˜ì§‘ ?œì‘ - {len(keywords)}ê°??¤ì›Œ??(ë©”ëª¨ë¦?ìµœì ??")
        
        # ?™ì  ?Œì¼ëª??¤ì •
        self._setup_dynamic_filenames("keywords")
        
        # ì²´í¬?¬ì¸??ë¡œë“œ ?ëŠ” ?ì„±
        if resume:
            self.checkpoint_data = self._load_checkpoint()
        
        if not self.checkpoint_data:
            total_target = len(keywords) * max_terms_per_keyword
            self.checkpoint_data = self._create_checkpoint_data(total_target)
            logger.info("?ˆë¡œ???¤ì›Œ?œë³„ ?˜ì§‘ ?¸ì…˜ ?œì‘")
        else:
            logger.info(f"ê¸°ì¡´ ?¤ì›Œ?œë³„ ?˜ì§‘ ?¸ì…˜ ?¬ê°œ: {self.checkpoint_data.collected_count}ê°??©ì–´ ?˜ì§‘??)
        
        self.stats['start_time'] = datetime.now()
        
        try:
            # ?¤ì›Œ?œë³„ ë°°ì¹˜ ?˜ì§‘
            success = self._collect_keywords_in_batches(keywords, max_terms_per_keyword)
            
            self.stats['end_time'] = datetime.now()
            self._log_collection_summary()
            
            return success
            
        except Exception as e:
            logger.error(f"?¤ì›Œ?œë³„ ?©ì–´ ?˜ì§‘ ?¤íŒ¨: {e}")
            self.checkpoint_data.errors_count += 1
            self._save_checkpoint()
            return False
    
    def _collect_keywords_in_batches(self, keywords: List[str], max_terms_per_keyword: int) -> bool:
        """?¤ì›Œ?œë³„ ë°°ì¹˜ ?˜ì§‘"""
        batch_size = self.memory_config.batch_size
        processed_keywords = set(self.checkpoint_data.processed_keywords)
        
        logger.info(f"?¤ì›Œ?œë³„ ë°°ì¹˜ ?˜ì§‘ ?œì‘ - ë°°ì¹˜ ?¬ê¸°: {batch_size}")
        
        for keyword in keywords:
            if self.shutdown_requested:
                logger.info("ì¤‘ë‹¨ ?”ì²­?¼ë¡œ ?¸í•œ ?˜ì§‘ ì¤‘ë‹¨")
                break
            
            if keyword in processed_keywords:
                logger.info(f"?¤ì›Œ??'{keyword}' ?´ë? ì²˜ë¦¬??- ê±´ë„ˆ?°ê¸°")
                continue
            
            logger.info(f"?¤ì›Œ??'{keyword}' ?˜ì§‘ ?œì‘...")
            
            try:
                # ?¤ì›Œ?œë³„ ë°°ì¹˜ ?˜ì§‘
                keyword_success = self._collect_single_keyword_batches(keyword, max_terms_per_keyword, batch_size)
                
                if keyword_success:
                    self.checkpoint_data.processed_keywords.append(keyword)
                    logger.info(f"?¤ì›Œ??'{keyword}' ?„ë£Œ")
                else:
                    logger.warning(f"?¤ì›Œ??'{keyword}' ?˜ì§‘ ?¤íŒ¨")
                
                # ì²´í¬?¬ì¸???€??
                self._save_checkpoint()
                
                # ë©”ëª¨ë¦?ì²´í¬ ë°?ê°€ë¹„ì? ì»¬ë ‰??
                if not self._check_memory_limit():
                    logger.warning("ë©”ëª¨ë¦??œí•œ?¼ë¡œ ?¸í•œ ?¼ì‹œ ì¤‘ë‹¨")
                    break
                
            except Exception as e:
                logger.error(f"?¤ì›Œ??'{keyword}' ?˜ì§‘ ì¤??¤ë¥˜: {e}")
                self.checkpoint_data.errors_count += 1
        
        logger.info(f"?¤ì›Œ?œë³„ ë°°ì¹˜ ?˜ì§‘ ?„ë£Œ - ì´?{self.checkpoint_data.collected_count}ê°??©ì–´ ?˜ì§‘")
        return self.checkpoint_data.collected_count > 0
    
    def _collect_single_keyword_batches(self, keyword: str, max_terms: int, batch_size: int) -> bool:
        """?¨ì¼ ?¤ì›Œ??ë°°ì¹˜ ?˜ì§‘"""
        collected_count = 0
        batch_count = 0
        
        while collected_count < max_terms and not self.shutdown_requested:
            # ë©”ëª¨ë¦?ì²´í¬
            if not self._check_memory_limit():
                break
            
            # ?„ì¬ ë°°ì¹˜ ?¬ê¸° ê³„ì‚°
            current_batch_size = min(batch_size, max_terms - collected_count)
            
            try:
                # ë°°ì¹˜ ?¨ìœ„ë¡??©ì–´ ?˜ì§‘
                batch_terms = self.api_client.collect_legal_terms(keyword, current_batch_size)
                
                if not batch_terms:
                    logger.info(f"?¤ì›Œ??'{keyword}'?ì„œ ???´ìƒ ?˜ì§‘???©ì–´ê°€ ?†ìŠµ?ˆë‹¤.")
                    break
                
                # ?¬ì „??ë°°ì¹˜ ?©ì–´ ì¶”ê?
                batch_success = self._add_terms_to_dictionary_batch(batch_terms)
                
                if batch_success:
                    # ?¤ì œë¡??¬ì „???€?¥ëœ ?©ì–´ ?˜ë¡œ ?…ë°?´íŠ¸
                    collected_count = len(self.dictionary.terms)
                    self.checkpoint_data.collected_count = collected_count
                    
                    logger.info(f"?¤ì›Œ??'{keyword}' ë°°ì¹˜ {batch_count + 1} ?„ë£Œ - ?˜ì§‘???©ì–´: {collected_count}/{max_terms}ê°?)
                
                batch_count += 1
                
                # ë°°ì¹˜ ê°??œë¤ ì§€??(2~5ì´?
                if not self.shutdown_requested:
                    delay = random.uniform(self.memory_config.batch_delay_min, self.memory_config.batch_delay_max)
                    logger.info(f"?¤ì›Œ??'{keyword}' ë°°ì¹˜ ê°?ì§€?? {delay:.1f}ì´?)
                    time.sleep(delay)
                
                # ê°€ë¹„ì? ì»¬ë ‰???„ê³„ê°?ì²´í¬
                if batch_count % self.memory_config.gc_threshold == 0:
                    self._force_garbage_collection()
                
            except Exception as e:
                logger.error(f"?¤ì›Œ??'{keyword}' ë°°ì¹˜ {batch_count + 1} ?˜ì§‘ ?¤íŒ¨: {e}")
                self.checkpoint_data.errors_count += 1
                break
        
        return collected_count > 0
    
    def _setup_dynamic_filenames(self, collection_type: str, target_year: int = None):
        """?™ì  ?Œì¼ëª??¤ì • (?¸ì…˜ë³?êµ¬ë¶„)"""
        try:
            # ?Œì¼ëª??‘ë‘???ì„±
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if target_year:
                self.dynamic_file_prefix = f"{timestamp}_{collection_type}_{target_year}"
            else:
                self.dynamic_file_prefix = f"{timestamp}_{collection_type}"
            
            # ?¸ì…˜ë³??´ë” ?ì„±
            session_folder = Path(f"data/raw/legal_terms/session_{self.dynamic_file_prefix}")
            session_folder.mkdir(parents=True, exist_ok=True)
            
            # ?Œì¼ ê²½ë¡œ ?¤ì • (?¸ì…˜ ?´ë” ?´ë?)
            self.dictionary_file = session_folder / f"legal_terms_{self.dynamic_file_prefix}.json"
            self.checkpoint_file = session_folder / f"checkpoint_{self.dynamic_file_prefix}.json"
            
            logger.info(f"?™ì  ?Œì¼ëª??¤ì • ?„ë£Œ:")
            logger.info(f"  ?¸ì…˜ ?´ë”: {session_folder}")
            logger.info(f"  ?¬ì „ ?Œì¼: {self.dictionary_file}")
            logger.info(f"  ì²´í¬?¬ì¸???Œì¼: {self.checkpoint_file}")
            
        except Exception as e:
            logger.error(f"?™ì  ?Œì¼ëª??¤ì • ?¤íŒ¨: {e}")
            # ê¸°ë³¸ ?Œì¼ëª…ìœ¼ë¡??´ë°±
            self.dictionary_file = Path("data/raw/legal_terms/legal_term_dictionary.json")
            self.checkpoint_file = Path("data/raw/legal_terms/checkpoint.json")
    
    def collect_terms_by_year(self, year: int, max_terms: int = None, resume: bool = True) -> bool:
        """ì§€???°ë„ ?©ì–´ ?˜ì§‘ (ë©”ëª¨ë¦?ìµœì ??ë°?ì²´í¬?¬ì¸??ì§€??"""
        # ë¬´ì œ???˜ì§‘ ëª¨ë“œ (max_termsê°€ None?´ê±°??ë§¤ìš° ??ê°’ì¸ ê²½ìš°)
        if max_terms is None:
            max_terms = 999999999  # ê±°ì˜ ë¬´ì œ??
            logger.info(f"{year}???©ì–´ ?˜ì§‘ ?œì‘ - ë¬´ì œ??ëª¨ë“œ (ë©”ëª¨ë¦?ìµœì ??")
        else:
            logger.info(f"{year}???©ì–´ ?˜ì§‘ ?œì‘ - ìµœë? {max_terms}ê°?(ë©”ëª¨ë¦?ìµœì ??")
        
        # ?™ì  ?Œì¼ëª??¤ì •
        self._setup_dynamic_filenames("year", year)
        
        # ì²´í¬?¬ì¸??ë¡œë“œ ?ëŠ” ?ì„±
        if resume:
            self.checkpoint_data = self._load_checkpoint()
        
        if not self.checkpoint_data:
            self.checkpoint_data = self._create_checkpoint_data(max_terms)
            logger.info(f"?†• ?ˆë¡œ??{year}???˜ì§‘ ?¸ì…˜ ?œì‘")
            logger.info(f"?“ ?¸ì…˜ ?´ë”: {self.dictionary_file.parent}")
            logger.info(f"?“ ?¬ì „ ?Œì¼: {self.dictionary_file}")
            logger.info(f"?“ ì²´í¬?¬ì¸???Œì¼: {self.checkpoint_file}")
            logger.info(f"?“Š ëª©í‘œ ?©ì–´ ?? {max_terms}ê°?)
        else:
            # ê¸°ì¡´ ì²´í¬?¬ì¸?¸ì˜ ëª©í‘œ ?©ì–´ ???…ë°?´íŠ¸ (ë¬´ì œ??ëª¨ë“œ??ê²½ìš°)
            if max_terms == 999999999 or self.checkpoint_data.total_target < max_terms:
                logger.info(f"ëª©í‘œ ?©ì–´ ???…ë°?´íŠ¸: {self.checkpoint_data.total_target} ??{max_terms}")
                self.checkpoint_data.total_target = max_terms
            
            # ê¸°ì¡´ ?°ì´??ë³‘í•©
            self._merge_existing_data(year)
            
            # ê¸°ì¡´ ?¬ì „ ë¡œë“œ ???¤ì œ ?©ì–´ ?˜ë¡œ ?™ê¸°??
            self.load_dictionary()
            actual_count = len(self.dictionary.terms)
            if actual_count != self.checkpoint_data.collected_count:
                logger.info(f"ì²´í¬?¬ì¸?¸ì? ?¤ì œ ?¬ì „ ?©ì–´ ??ë¶ˆì¼ì¹?- ì²´í¬?¬ì¸?? {self.checkpoint_data.collected_count}ê°? ?¤ì œ: {actual_count}ê°?)
                logger.info(f"?¤ì œ ?¬ì „ ?©ì–´ ?˜ë¡œ ?™ê¸°?? {actual_count}ê°?)
                self.checkpoint_data.collected_count = actual_count
            else:
                logger.info(f"ê¸°ì¡´ {year}???˜ì§‘ ?¸ì…˜ ?¬ê°œ: {self.checkpoint_data.collected_count}ê°??©ì–´ ?˜ì§‘??)
        
        self.stats['start_time'] = datetime.now()
        
        try:
            # ?¨ì¼ ?°ë„ ë°°ì¹˜ ?˜ì§‘
            success = self._collect_single_year_batches(year, max_terms)
            
            self.stats['end_time'] = datetime.now()
            self._log_collection_summary()
            
            return success
            
        except Exception as e:
            logger.error(f"{year}???©ì–´ ?˜ì§‘ ?¤íŒ¨: {e}")
            self.checkpoint_data.errors_count += 1
            self._save_checkpoint()
            return False
    
    
    def _collect_single_year_batches(self, year: int, max_terms: int) -> bool:
        """?¨ì¼ ?°ë„ ë°°ì¹˜ ?˜ì§‘"""
        batch_size = self.memory_config.batch_size
        collected_count = self.checkpoint_data.collected_count
        
        # ë¬´ì œ??ëª¨ë“œ??ê²½ìš° ?¨ì? ?©ì–´ ê³„ì‚° ?ëµ
        if max_terms == 999999999:
            remaining_terms = "ë¬´ì œ??
            logger.info(f"{year}??ë°°ì¹˜ ?˜ì§‘ ?œì‘ - ë°°ì¹˜ ?¬ê¸°: {batch_size}, ëª¨ë“œ: ë¬´ì œ??)
        else:
            remaining_terms = max_terms - collected_count
            logger.info(f"{year}??ë°°ì¹˜ ?˜ì§‘ ?œì‘ - ë°°ì¹˜ ?¬ê¸°: {batch_size}, ?¨ì? ?©ì–´: {remaining_terms}ê°?)
        
        # ?°ë„ë³?? ì§œ ë²”ìœ„ ?¤ì •
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        batch_count = 0
        consecutive_empty_batches = 0  # ?°ì†?¼ë¡œ ë¹?ë°°ì¹˜ê°€ ?˜ì˜¨ ?Ÿìˆ˜
        max_empty_batches = 5  # ìµœë? ?°ì† ë¹?ë°°ì¹˜ ?ˆìš© ?Ÿìˆ˜
        
        while (max_terms == 999999999 or collected_count < max_terms) and not self.shutdown_requested:
            # shutdown ?”ì²­ ?•ì¸ (???ì£¼ ì²´í¬)
            if self.shutdown_requested:
                logger.info("ì¢…ë£Œ ?”ì²­?¼ë¡œ ?¸í•œ ?˜ì§‘ ì¤‘ë‹¨")
                break
            
            # ë©”ëª¨ë¦?ì²´í¬
            if not self._check_memory_limit():
                logger.warning("ë©”ëª¨ë¦??œí•œ?¼ë¡œ ?¸í•œ ?¼ì‹œ ì¤‘ë‹¨")
                break
            
            # ?„ì¬ ë°°ì¹˜ ?¬ê¸° ê³„ì‚° (ë¬´ì œ??ëª¨ë“œ??ê²½ìš° ë°°ì¹˜ ?¬ê¸° ê·¸ë?ë¡??¬ìš©)
            if max_terms == 999999999:
                current_batch_size = batch_size
            else:
                current_batch_size = min(batch_size, remaining_terms)
            
            try:
                # ë°°ì¹˜ ?¨ìœ„ë¡??©ì–´ ?˜ì§‘ (? ì§œ ë²”ìœ„ ?¬í•¨)
                batch_terms = self.api_client.collect_legal_terms(
                    max_terms=current_batch_size,
                    start_date=start_date,
                    end_date=end_date,
                    checkpoint_data=self.checkpoint_data.__dict__ if self.checkpoint_data else None,
                    session_folder=str(self.dictionary_file.parent) if self.dictionary_file else None
                )
                
                # shutdown ?”ì²­ ?¬í™•??(API ?¸ì¶œ ??
                if self.shutdown_requested:
                    logger.info("API ?¸ì¶œ ??ì¢…ë£Œ ?”ì²­ ?•ì¸ - ?˜ì§‘ ì¤‘ë‹¨")
                    break
                
                if not batch_terms:
                    consecutive_empty_batches += 1
                    logger.info(f"{year}??ë°°ì¹˜ {batch_count + 1} - ?˜ì§‘???©ì–´ ?†ìŒ (?°ì† ë¹?ë°°ì¹˜: {consecutive_empty_batches}/{max_empty_batches})")
                    
                    # ?°ì†?¼ë¡œ ë¹?ë°°ì¹˜ê°€ ë§ì´ ?˜ì˜¤ë©?ì¤‘ë‹¨
                    if consecutive_empty_batches >= max_empty_batches:
                        logger.info(f"{year}?„ì—???°ì† {max_empty_batches}??ë¹?ë°°ì¹˜ë¡??¸í•œ ?˜ì§‘ ì¤‘ë‹¨")
                        break
                    
                    # ë¹?ë°°ì¹˜ ?„ì—??ì§€???ìš©
                    if not self.shutdown_requested:
                        delay = random.uniform(self.memory_config.batch_delay_min, self.memory_config.batch_delay_max)
                        logger.info(f"ë¹?ë°°ì¹˜ ??ì§€?? {delay:.1f}ì´?)
                        time.sleep(delay)
                    
                    batch_count += 1
                    continue
                else:
                    # ?©ì–´ê°€ ?˜ì§‘?˜ë©´ ?°ì† ë¹?ë°°ì¹˜ ì¹´ìš´??ë¦¬ì…‹
                    consecutive_empty_batches = 0
                
                # ?¬ì „??ë°°ì¹˜ ?©ì–´ ì¶”ê?
                batch_success = self._add_terms_to_dictionary_batch(batch_terms)
                
                if batch_success:
                    # ?¤ì œë¡??¬ì „???€?¥ëœ ?©ì–´ ?˜ë¡œ ?…ë°?´íŠ¸ (ê°€???•í™•??ë°©ë²•)
                    collected_count = len(self.dictionary.terms)
                    self.checkpoint_data.collected_count = collected_count
                    
                    # ë¬´ì œ??ëª¨ë“œê°€ ?„ë‹Œ ê²½ìš°?ë§Œ ?¨ì? ?©ì–´ ê³„ì‚°
                    if max_terms != 999999999:
                        remaining_terms = max_terms - collected_count
                        logger.info(f"??{year}??ë°°ì¹˜ {batch_count + 1} ?„ë£Œ - ?˜ì§‘???©ì–´: {collected_count}/{max_terms}ê°?)
                    else:
                        logger.info(f"??{year}??ë°°ì¹˜ {batch_count + 1} ?„ë£Œ - ?˜ì§‘???©ì–´: {collected_count}ê°?(ë¬´ì œ??ëª¨ë“œ)")
                    
                    # ?Œì¼ ?ì„± ?í™© ë¡œê·¸
                    logger.info(f"?“ ?„ì¬ ?¬ì „ ?Œì¼: {self.dictionary_file}")
                    logger.info(f"?“ ?„ì¬ ì²´í¬?¬ì¸?? {self.checkpoint_file}")
                    logger.info(f"?“Š ?¬ì „???€?¥ëœ ?©ì–´ ?? {len(self.dictionary.terms)}ê°?)
                    
                    
                    # ì²´í¬?¬ì¸???€??ë°??˜ì´ì§€ ?Œì¼ ì¦‰ì‹œ ?€??
                    if (batch_count + 1) % self.memory_config.checkpoint_interval == 0:
                        logger.info(f"?’¾ ì²´í¬?¬ì¸???€??ì¤?.. (ë°°ì¹˜ {batch_count + 1}ë§ˆë‹¤ ?€??")
                        self._save_checkpoint()
                        logger.info(f"?’¾ ?˜ì´ì§€ ?Œì¼ ?€??ì¤?..")
                        self._save_page_files_immediately()
                        logger.info(f"???Œì¼ ?€???„ë£Œ!")
                
                batch_count += 1
                
                # ë°°ì¹˜ ê°??œë¤ ì§€??(2~5ì´?
                if not self.shutdown_requested:
                    delay = random.uniform(self.memory_config.batch_delay_min, self.memory_config.batch_delay_max)
                    logger.info(f"??{year}??ë°°ì¹˜ ê°?ì§€?? {delay:.1f}ì´?)
                    time.sleep(delay)
                
                # ê°€ë¹„ì? ì»¬ë ‰???„ê³„ê°?ì²´í¬
                if batch_count % self.memory_config.gc_threshold == 0:
                    self._force_garbage_collection()
                
            except Exception as e:
                logger.error(f"{year}??ë°°ì¹˜ {batch_count + 1} ?˜ì§‘ ?¤íŒ¨: {e}")
                self.checkpoint_data.errors_count += 1
                
                # ?°ì† ?ëŸ¬ ??ì¤‘ë‹¨
                if self.checkpoint_data.errors_count > 10:
                    logger.error("?°ì† ?ëŸ¬ë¡??¸í•œ ?˜ì§‘ ì¤‘ë‹¨")
                    break
        
        # ìµœì¢… ì²´í¬?¬ì¸???€??(?¤ì œ ?¬ì „ ?©ì–´ ?˜ë¡œ ?…ë°?´íŠ¸)
        logger.info(f"?’¾ ìµœì¢… ì²´í¬?¬ì¸???€??ì¤?..")
        self.checkpoint_data.collected_count = len(self.dictionary.terms)
        self._save_checkpoint()
        logger.info(f"?’¾ ìµœì¢… ?˜ì´ì§€ ?Œì¼ ?€??ì¤?..")
        self._save_page_files_immediately()
        
        logger.info(f"?‰ {year}??ë°°ì¹˜ ?˜ì§‘ ?„ë£Œ - ì´?{collected_count}ê°??©ì–´ ?˜ì§‘")
        logger.info(f"?“ ìµœì¢… ?¬ì „ ?Œì¼: {self.dictionary_file}")
        logger.info(f"?“ ìµœì¢… ì²´í¬?¬ì¸?? {self.checkpoint_file}")
        logger.info(f"?“Š ìµœì¢… ?¬ì „ ?©ì–´ ?? {len(self.dictionary.terms)}ê°?)
        return collected_count > 0
    
    def _add_terms_to_dictionary_batch(self, terms: List[Dict[str, Any]]) -> bool:
        """ë°°ì¹˜ ?¨ìœ„ë¡??¬ì „???©ì–´ ì¶”ê? (ë©”ëª¨ë¦?ìµœì ??"""
        
        success_count = 0
        fail_count = 0
        category_stats = {}
        
        for term_data in terms:
            try:
                if self.dictionary.add_term(term_data):
                    success_count += 1
                    
                    # ì¹´í…Œê³ ë¦¬ë³??µê³„ ?˜ì§‘
                    category = term_data.get('category', 'ê¸°í?')
                    category_stats[category] = category_stats.get(category, 0) + 1
                else:
                    fail_count += 1
                    
            except Exception as e:
                logger.error(f"?©ì–´ ì¶”ê? ?¤íŒ¨: {e}")
                fail_count += 1
        
        # ê¸°ë³¸ ?µê³„ ?…ë°?´íŠ¸
        self.stats['total_collected'] = len(self.dictionary.terms)
        
        return success_count > 0
    
    def _add_terms_to_dictionary(self, terms: List[Dict[str, Any]]) -> bool:
        """?˜ì§‘???©ì–´ë¥??¬ì „??ì¶”ê?"""
        logger.info(f"?¬ì „??{len(terms)}ê°??©ì–´ ì¶”ê? ì¤?..")
        
        success_count = 0
        fail_count = 0
        category_stats = {}
        
        for i, term_data in enumerate(terms):
            try:
                if self.dictionary.add_term(term_data):
                    success_count += 1
                    
                    # ì¹´í…Œê³ ë¦¬ë³??µê³„ ?˜ì§‘
                    category = term_data.get('category', 'ê¸°í?')
                    category_stats[category] = category_stats.get(category, 0) + 1
                else:
                    fail_count += 1
                
                # ì§„í–‰ë¥?ë¡œê·¸ (100ê°œë§ˆ??
                if (i + 1) % 100 == 0:
                    logger.info(f"?©ì–´ ì¶”ê? ì§„í–‰: {i + 1}/{len(terms)} ({success_count}ê°??±ê³µ, {fail_count}ê°??¤íŒ¨)")
                    
            except Exception as e:
                logger.error(f"?©ì–´ ì¶”ê? ?¤íŒ¨: {e}")
                fail_count += 1
        
        # ê¸°ë³¸ ?µê³„ ?…ë°?´íŠ¸
        self.stats['total_collected'] = len(self.dictionary.terms)
        
        logger.info(f"?¬ì „ ì¶”ê? ?„ë£Œ - {success_count}ê°??±ê³µ, {fail_count}ê°??¤íŒ¨")
        
        # ì¹´í…Œê³ ë¦¬ë³??µê³„ ì¶œë ¥
        logger.info("ì¹´í…Œê³ ë¦¬ë³??©ì–´ ?˜ì§‘ ?µê³„:")
        for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {category}: {count}ê°?)
        
        return success_count > 0
    
    def _log_collection_summary(self):
        """?˜ì§‘ ?”ì•½ ë¡œê·¸ ì¶œë ¥"""
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            logger.info(f"?˜ì§‘ ?„ë£Œ - ì´??©ì–´: {len(self.dictionary.terms)}ê°? ?Œìš” ?œê°„: {duration.total_seconds():.2f}ì´?)
        else:
            logger.info(f"?˜ì§‘ ?„ë£Œ - ì´??©ì–´: {len(self.dictionary.terms)}ê°?)
    
    
    def get_progress_status(self) -> Dict[str, Any]:
        """ì§„í–‰ ?íƒœ ì¡°íšŒ"""
        if not self.checkpoint_data:
            return {'status': 'not_started'}
        
        progress_percent = (self.checkpoint_data.collected_count / self.checkpoint_data.total_target) * 100
        
        return {
            'status': 'in_progress' if not self.shutdown_requested else 'paused',
            'collected_count': len(self.dictionary.terms),
            'total_target': self.checkpoint_data.total_target
        }
    
    def clear_checkpoint(self) -> bool:
        """ì²´í¬?¬ì¸???? œ"""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                logger.info("ì²´í¬?¬ì¸???? œ ?„ë£Œ")
                return True
            else:
                logger.info("ì²´í¬?¬ì¸???Œì¼??ì¡´ì¬?˜ì? ?ŠìŠµ?ˆë‹¤.")
                return True
        except Exception as e:
            logger.error(f"ì²´í¬?¬ì¸???? œ ?¤íŒ¨: {e}")
            return False
    
    def optimize_memory_settings(self, available_memory_mb: int = None) -> bool:
        """ë©”ëª¨ë¦??¤ì • ìµœì ??""
        try:
            if available_memory_mb is None:
                # ?œìŠ¤??ë©”ëª¨ë¦??•ë³´ ì¡°íšŒ
                memory_info = psutil.virtual_memory()
                available_memory_mb = memory_info.available / 1024 / 1024
            
            # ?ˆì „??ë©”ëª¨ë¦??¬ìš©???¤ì • (?„ì²´??70%)
            safe_memory_mb = int(available_memory_mb * 0.7)
            
            # ë°°ì¹˜ ?¬ê¸° ?™ì  ì¡°ì •
            if safe_memory_mb > 4096:  # 4GB ?´ìƒ
                batch_size = 100
            elif safe_memory_mb > 2048:  # 2GB ?´ìƒ
                batch_size = 50
            else:  # 2GB ë¯¸ë§Œ
                batch_size = 25
            
            # ë©”ëª¨ë¦??¤ì • ?…ë°?´íŠ¸
            self.memory_config.max_memory_mb = safe_memory_mb
            self.memory_config.batch_size = batch_size
            
            logger.info(f"ë©”ëª¨ë¦??¤ì • ìµœì ???„ë£Œ:")
            logger.info(f"  ?¬ìš© ê°€??ë©”ëª¨ë¦? {available_memory_mb:.1f}MB")
            logger.info(f"  ?ˆì „ ë©”ëª¨ë¦??œê³„: {safe_memory_mb:.1f}MB")
            logger.info(f"  ë°°ì¹˜ ?¬ê¸°: {batch_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"ë©”ëª¨ë¦??¤ì • ìµœì ???¤íŒ¨: {e}")
            return False
    
    def save_dictionary(self, file_path: str = None) -> bool:
        """?¬ì „ ?€??(ë°°ì¹˜ë³?ë¶„ë¦¬ ?Œì¼ ì§€??"""
        try:
            if file_path is None:
                # ?™ì  ?Œì¼ëª…ì´ ?¤ì •?˜ì–´ ?ˆìœ¼ë©??¬ìš©, ?„ë‹ˆë©?ê¸°ë³¸ ?Œì¼ëª??¬ìš©
                if self.dictionary_file:
                    file_path = str(self.dictionary_file)
                else:
                    file_path = "data/raw/legal_terms/legal_term_dictionary.json"
            
            # ë°°ì¹˜ë³?ë¶„ë¦¬ ?Œì¼ë¡??€??
            return self._save_dictionary_batch_separated(file_path)
            
        except Exception as e:
            logger.error(f"?¬ì „ ?€??ì¤??¤ë¥˜: {e}")
            return False
    
    def _save_dictionary_batch_separated(self, base_file_path: str) -> bool:
        """ë°°ì¹˜ë³„ë¡œ ë¶„ë¦¬???Œì¼ë¡??¬ì „ ?€??(ì¤‘ë³µ ë°©ì?)"""
        try:
            base_path = Path(base_file_path)
            session_folder = base_path.parent
            
            # ê¸°ì¡´ ë©”ì¸ ?¬ì „ ?Œì¼ ?•ì¸
            if base_path.exists():
                logger.info(f"ë©”ì¸ ?¬ì „ ?Œì¼ {base_path.name}???´ë? ì¡´ì¬??- ê±´ë„ˆ?°ê¸°")
            else:
                # 1. ?„ì²´ ?¬ì „ ?Œì¼ ?€??(ê¸°ì¡´ ë°©ì‹)
                self.dictionary.dictionary_path = base_path
                success = self.dictionary.save_dictionary()
                
                if not success:
                    return False
            
            # 2. ?˜ì´ì§€ë³„ë¡œ ë¶„ë¦¬ ?€??(ê¸°ì¡´ ?˜ì´ì§€ ?Œì¼?¤ì´ ?ˆìœ¼ë©?ê·¸ë?ë¡?? ì?)
            existing_page_files = list(session_folder.glob("legal_terms_page_*.json"))
            if existing_page_files:
                logger.info(f"ê¸°ì¡´ ?˜ì´ì§€ ?Œì¼ {len(existing_page_files)}ê°?ë°œê²¬ - ì¶”ê? ?€???ëµ")
            else:
                # ?˜ì´ì§€ë³„ë¡œ ë¶„ë¦¬ ?€??
                self._save_batch_files(session_folder)
            
            logger.info(f"ë°°ì¹˜ë³?ë¶„ë¦¬ ?¬ì „ ?€???„ë£Œ: {base_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"ë°°ì¹˜ë³?ë¶„ë¦¬ ?¬ì „ ?€???¤íŒ¨: {e}")
            return False
    
    def _save_batch_files(self, session_folder: Path):
        """API ?˜ì´ì§€ë³„ë¡œ ?Œì¼ ë¶„ë¦¬ ?€??(ì¤‘ë³µ ë°©ì?)"""
        try:
            # API ?˜ì´ì§€ ?¬ê¸° ?¤ì • (ê¸°ë³¸ 100ê°?- API ê¸°ë³¸ê°?
            page_size = 100
            
            # ê¸°ì¡´ ?˜ì´ì§€ ?Œì¼???•ì¸
            existing_page_files = list(session_folder.glob("legal_terms_page_*.json"))
            existing_page_numbers = set()
            
            for file_path in existing_page_files:
                try:
                    # ?Œì¼ëª…ì—???˜ì´ì§€ ë²ˆí˜¸ ì¶”ì¶œ (?? legal_terms_page_001.json -> 1)
                    page_num = int(file_path.stem.split('_')[-1])
                    existing_page_numbers.add(page_num)
                except (ValueError, IndexError):
                    continue
            
            if existing_page_numbers:
                logger.info(f"ê¸°ì¡´ ?˜ì´ì§€ ?Œì¼ {len(existing_page_numbers)}ê°?ë°œê²¬: {sorted(existing_page_numbers)}")
            
            # ?©ì–´?¤ì„ ?˜ì´ì§€ë³„ë¡œ ê·¸ë£¹??
            terms_list = list(self.dictionary.terms.items())
            total_terms = len(terms_list)
            
            if total_terms == 0:
                logger.info("?€?¥í•  ?©ì–´ê°€ ?†ìŠµ?ˆë‹¤.")
                return
            
            # ?˜ì´ì§€ ?Œì¼ ?€??
            page_count = 0
            for i in range(0, total_terms, page_size):
                page_terms = dict(terms_list[i:i + page_size])
                page_count += 1
                
                # ?˜ì´ì§€ ?Œì¼ëª??ì„±
                page_file = session_folder / f"legal_terms_page_{page_count:03d}.json"
                
                # ê¸°ì¡´ ?Œì¼???ˆëŠ”ì§€ ?•ì¸
                if page_file.exists():
                    logger.info(f"?˜ì´ì§€ ?Œì¼ {page_file.name}???´ë? ì¡´ì¬??- ê±´ë„ˆ?°ê¸°")
                    continue
                
                # ?˜ì´ì§€ ?°ì´??êµ¬ì„±
                page_data = {
                    "metadata": {
                        "page_number": page_count,
                        "page_size": len(page_terms),
                        "total_pages": (total_terms + page_size - 1) // page_size,
                        "saved_at": datetime.now().isoformat(),
                        "session_id": self.session_id,
                        "api_page_size": page_size
                    },
                    "terms": page_terms
                }
                
                # ?˜ì´ì§€ ?Œì¼ ?€??
                with open(page_file, 'w', encoding='utf-8') as f:
                    json.dump(page_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"?“„ ?˜ì´ì§€ ?Œì¼ ?€???„ë£Œ: {page_file.name} ({len(page_terms)}ê°??©ì–´)")
            
            logger.info(f"?“Š ì´?{page_count}ê°??˜ì´ì§€ ?Œì¼ ?ì„± ?„ë£Œ")
            
        except Exception as e:
            logger.error(f"?˜ì´ì§€ ?Œì¼ ?€???¤íŒ¨: {e}")
    
    def _save_page_files_immediately(self):
        """ì²´í¬?¬ì¸?¸ë§ˆ??ì¦‰ì‹œ ?˜ì´ì§€ ?Œì¼ ?€??""
        try:
            if not self.dictionary_file:
                logger.warning("?¬ì „ ?Œì¼ ê²½ë¡œê°€ ?¤ì •?˜ì? ?Šì•˜?µë‹ˆ??")
                return
            
            session_folder = self.dictionary_file.parent
            logger.info(f"?“ ?¸ì…˜ ?´ë”: {session_folder}")
            self._save_batch_files(session_folder)
            logger.info("??ì²´í¬?¬ì¸?¸ë§ˆ???˜ì´ì§€ ?Œì¼ ?€???„ë£Œ")
            
        except Exception as e:
            logger.error(f"??ì¦‰ì‹œ ?˜ì´ì§€ ?Œì¼ ?€???¤íŒ¨: {e}")
    
    def load_dictionary(self, file_path: str = None) -> bool:
        """?¬ì „ ë¡œë“œ (?˜ì´ì§€ ?Œì¼ ì§€??"""
        try:
            if file_path is None:
                file_path = "data/raw/legal_terms/legal_term_dictionary.json"
            
            # ?˜ì´ì§€ ?Œì¼?¤ë„ ?¨ê»˜ ë¡œë“œ
            return self._load_dictionary_with_pages(file_path)
            
        except Exception as e:
            logger.error(f"?¬ì „ ë¡œë“œ ì¤??¤ë¥˜: {e}")
            return False
    
    def _load_dictionary_with_pages(self, base_file_path: str) -> bool:
        """?˜ì´ì§€ ?Œì¼?¤ì„ ?¬í•¨?˜ì—¬ ?¬ì „ ë¡œë“œ"""
        try:
            base_path = Path(base_file_path)
            session_folder = base_path.parent
            
            # 1. ê¸°ë³¸ ?¬ì „ ?Œì¼ ë¡œë“œ
            success = self.dictionary.load_terms_from_file(base_file_path)
            if success:
                logger.info(f"ê¸°ë³¸ ?¬ì „ ë¡œë“œ ?„ë£Œ: {base_file_path}")
            
            # 2. ?˜ì´ì§€ ?Œì¼??ë¡œë“œ
            self._load_page_files(session_folder)
            
            return success
            
        except Exception as e:
            logger.error(f"?˜ì´ì§€ ?Œì¼ ?¬í•¨ ?¬ì „ ë¡œë“œ ?¤íŒ¨: {e}")
            return False
    
    def _load_page_files(self, session_folder: Path):
        """?˜ì´ì§€ ?Œì¼??ë¡œë“œ"""
        try:
            page_files = list(session_folder.glob("legal_terms_page_*.json"))
            if not page_files:
                logger.info("?˜ì´ì§€ ?Œì¼???†ìŠµ?ˆë‹¤.")
                return
            
            # ?˜ì´ì§€ ë²ˆí˜¸ ?œìœ¼ë¡??•ë ¬
            page_files.sort(key=lambda f: int(f.stem.split('_')[-1]))
            
            loaded_count = 0
            for page_file in page_files:
                try:
                    with open(page_file, 'r', encoding='utf-8') as f:
                        page_data = json.load(f)
                    
                    # ?˜ì´ì§€ ?°ì´?°ì—???©ì–´??ì¶”ì¶œ
                    terms = page_data.get('terms', {})
                    page_number = page_data.get('metadata', {}).get('page_number', 0)
                    
                    for term_id, term_data in terms.items():
                        if self.dictionary.add_term(term_data):
                            loaded_count += 1
                    
                    logger.info(f"?˜ì´ì§€ ?Œì¼ ë¡œë“œ: {page_file.name} (?˜ì´ì§€ #{page_number}, {len(terms)}ê°??©ì–´)")
                    
                except Exception as e:
                    logger.debug(f"?˜ì´ì§€ ?Œì¼ ë¡œë“œ ?¤íŒ¨ ({page_file}): {e}")
                    continue
            
            if loaded_count > 0:
                logger.info(f"?˜ì´ì§€ ?Œì¼ ë¡œë“œ ?„ë£Œ: {loaded_count}ê°??©ì–´ ì¶”ê?")
                
        except Exception as e:
            logger.error(f"?˜ì´ì§€ ?Œì¼ ë¡œë“œ ?¤íŒ¨: {e}")
    
    
    
