#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?Œì¬ê²°ì •ë¡€ ?˜ì§‘ê¸??´ë˜??

êµ??ë²•ë ¹?•ë³´?¼í„° LAW OPEN APIë¥??¬ìš©?˜ì—¬ ?Œì¬ê²°ì •ë¡€ë¥??˜ì§‘?©ë‹ˆ??
- ìµœê·¼ 5?„ê°„ ?Œì¬ê²°ì •ë¡€ 1,000ê±??˜ì§‘
- ?Œë²•?¬íŒ??ê²°ì •ë¡€???ì„¸ ?´ìš© ?˜ì§‘
- ê²°ì •? í˜•ë³?ë¶„ë¥˜ (?„í—Œ, ?©í—Œ, ê°í•˜, ê¸°ê° ??
- ?¥ìƒ???ëŸ¬ ì²˜ë¦¬, ?±ëŠ¥ ìµœì ?? ëª¨ë‹ˆ?°ë§ ê¸°ëŠ¥
"""

import os
import sys
import json
import logging
import signal
import atexit
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient, LawOpenAPIConfig

# ?Œë²• ê´€??ê²€???¤ì›Œ??(?°ì„ ?œìœ„ë³?
CONSTITUTIONAL_KEYWORDS = [
    # ìµœê³  ?°ì„ ?œìœ„ (ê°?100ê±?
    "?Œë²•?Œì›", "?„í—Œë²•ë¥ ?¬íŒ", "?„í•µ?¬íŒ", "ê¶Œí•œ?ì˜?¬íŒ", "?•ë‹¹?´ì‚°?¬íŒ",
    
    # ê³ ìš°? ìˆœ??(ê°?50ê±?
    "?ëª…ê¶?, "? ì²´???ìœ ", "?¬ìƒ?œì˜ ?ìœ ", "?‘ì‹¬???ìœ ", "ì¢…êµ???ìœ ",
    "?¸ë¡ ???ìœ ", "ì¶œíŒ???ìœ ", "ì§‘íšŒ???ìœ ", "ê²°ì‚¬???ìœ ", "?¬ì‚°ê¶?,
    "ì§ì—…? íƒ???ìœ ", "ê±°ì£¼?´ì „???ìœ ", "ì°¸ì •ê¶?, "êµìœ¡??ë°›ì„ ê¶Œë¦¬",
    "ê·¼ë¡œ??ê¶Œë¦¬", "?˜ê²½ê¶?, "ë³´ê±´ê¶?, "ì£¼ê±°ê¶?, "ë¬¸í™”ë¥??¥ìœ ??ê¶Œë¦¬",
    
    # ì¤‘ìš°? ìˆœ??(ê°?30ê±?
    "?™ë¬¸???ìœ ", "?ˆìˆ ???ìœ ", "?ì¡´ê¶?, "ê·¼ë¡œ3ê¶?, "ë³µì?ë¥?ë°›ì„ ê¶Œë¦¬",
    "ë²•ë¥ ???˜í•˜ì§€ ?„ë‹ˆ?˜ê³ ??ì²˜ë²Œë°›ì? ?„ë‹ˆ??ê¶Œë¦¬", "ë¬´ì£„ì¶”ì •???ì¹™",
    "ì§„ìˆ ê±°ë?ê¶?, "ë³€?¸ì¸??ì¡°ë ¥??ë°›ì„ ê¶Œë¦¬", "? ì†???¬íŒ??ë°›ì„ ê¶Œë¦¬",
    "ê³µê°œ?¬íŒ??ë°›ì„ ê¶Œë¦¬", "êµ?šŒ", "?•ë?", "ë²•ì›", "?Œë²•?¬íŒ??,
    "? ê±°ê´€ë¦¬ìœ„?íšŒ", "ê°ì‚¬??, "?€?µë ¹", "êµ?¬´ì´ë¦¬", "êµ?¬´?„ì›",
    "êµ?šŒ?˜ì›", "?€ë²•ì›??, "?Œë²•?¬íŒ?Œì¥", "ê¶Œë¦¬êµ¬ì œ???Œë²•?Œì›",
    "ê·œë²”?µì œ???Œë²•?Œì›", "?Œë²•?¬íŒ?Œì˜ ê´€??, "ê¸°ë³¸ê¶??œí•œ", "ë²•ë¥ ? ë³´",
    "ê³¼ì‰ê¸ˆì????ì¹™", "ë³¸ì§ˆ???´ìš© ì¹¨í•´ ê¸ˆì?", "ë¹„ë????ì¹™",
    "ëª…í™•?±ì˜ ?ì¹™", "?ì •?±ì˜ ?ì¹™", "êµ?????˜ë¬´", "ê¸°ë³¸ê¶?ë³´ì¥ ?˜ë¬´",
    "ìµœì†Œ?œì˜ ?í™œ ë³´ì¥", "êµìœ¡?œë„ ?•ë¦½", "ê·¼ë¡œì¡°ê±´??ê¸°ì?", "?˜ê²½ë³´ì „",
    "ë¬¸í™”ì§„í¥", "ë³µì?ì¦ì§„"
]

# ê²°ì •? í˜• ë¶„ë¥˜ ?¤ì›Œ??
DECISION_TYPE_KEYWORDS = {
    "?„í—Œ": ["?„í—Œ", "?„í—Œê²°ì •", "?Œë²•???„ë°˜"],
    "?©í—Œ": ["?©í—Œ", "?©í—Œê²°ì •", "?Œë²•???©ì¹˜"],
    "ê°í•˜": ["ê°í•˜", "ê°í•˜ê²°ì •", "ê°í•˜?ê²°"],
    "ê¸°ê°": ["ê¸°ê°", "ê¸°ê°ê²°ì •", "ê¸°ê°?ê²°"],
    "?¸ìš©": ["?¸ìš©", "?¸ìš©ê²°ì •", "?¸ìš©?ê²°"],
    "?¼ë??¸ìš©": ["?¼ë??¸ìš©", "?¼ë??¸ìš©ê²°ì •"],
    "?¼ë?ê¸°ê°": ["?¼ë?ê¸°ê°", "?¼ë?ê¸°ê°ê²°ì •"]
}

# ?¤ì›Œ?œë³„ ?°ì„ ?œìœ„ ë°?ëª©í‘œ ê±´ìˆ˜
KEYWORD_PRIORITIES = {
    # ìµœê³  ?°ì„ ?œìœ„ (100ê±?
    "?Œë²•?Œì›": 100, "?„í—Œë²•ë¥ ?¬íŒ": 100, "?„í•µ?¬íŒ": 100, 
    "ê¶Œí•œ?ì˜?¬íŒ": 100, "?•ë‹¹?´ì‚°?¬íŒ": 100,
    
    # ê³ ìš°? ìˆœ??(50ê±?
    "?ëª…ê¶?: 50, "? ì²´???ìœ ": 50, "?¬ìƒ?œì˜ ?ìœ ": 50, "?‘ì‹¬???ìœ ": 50,
    "ì¢…êµ???ìœ ": 50, "?¸ë¡ ???ìœ ": 50, "ì¶œíŒ???ìœ ": 50, "ì§‘íšŒ???ìœ ": 50,
    "ê²°ì‚¬???ìœ ": 50, "?¬ì‚°ê¶?: 50, "ì§ì—…? íƒ???ìœ ": 50, "ê±°ì£¼?´ì „???ìœ ": 50,
    "ì°¸ì •ê¶?: 50, "êµìœ¡??ë°›ì„ ê¶Œë¦¬": 50, "ê·¼ë¡œ??ê¶Œë¦¬": 50, "?˜ê²½ê¶?: 50,
    "ë³´ê±´ê¶?: 50, "ì£¼ê±°ê¶?: 50, "ë¬¸í™”ë¥??¥ìœ ??ê¶Œë¦¬": 50,
    
    # ì¤‘ìš°? ìˆœ??(30ê±?
    "?™ë¬¸???ìœ ": 30, "?ˆìˆ ???ìœ ": 30, "?ì¡´ê¶?: 30, "ê·¼ë¡œ3ê¶?: 30,
    "ë³µì?ë¥?ë°›ì„ ê¶Œë¦¬": 30, "ë²•ë¥ ???˜í•˜ì§€ ?„ë‹ˆ?˜ê³ ??ì²˜ë²Œë°›ì? ?„ë‹ˆ??ê¶Œë¦¬": 30,
    "ë¬´ì£„ì¶”ì •???ì¹™": 30, "ì§„ìˆ ê±°ë?ê¶?: 30, "ë³€?¸ì¸??ì¡°ë ¥??ë°›ì„ ê¶Œë¦¬": 30,
    "? ì†???¬íŒ??ë°›ì„ ê¶Œë¦¬": 30, "ê³µê°œ?¬íŒ??ë°›ì„ ê¶Œë¦¬": 30,
    "êµ?šŒ": 30, "?•ë?": 30, "ë²•ì›": 30, "?Œë²•?¬íŒ??: 30,
    "? ê±°ê´€ë¦¬ìœ„?íšŒ": 30, "ê°ì‚¬??: 30, "?€?µë ¹": 30, "êµ?¬´ì´ë¦¬": 30,
    "êµ?¬´?„ì›": 30, "êµ?šŒ?˜ì›": 30, "?€ë²•ì›??: 30, "?Œë²•?¬íŒ?Œì¥": 30,
    "ê¶Œë¦¬êµ¬ì œ???Œë²•?Œì›": 30, "ê·œë²”?µì œ???Œë²•?Œì›": 30, "?Œë²•?¬íŒ?Œì˜ ê´€??: 30,
    "ê¸°ë³¸ê¶??œí•œ": 30, "ë²•ë¥ ? ë³´": 30, "ê³¼ì‰ê¸ˆì????ì¹™": 30,
    "ë³¸ì§ˆ???´ìš© ì¹¨í•´ ê¸ˆì?": 30, "ë¹„ë????ì¹™": 30, "ëª…í™•?±ì˜ ?ì¹™": 30,
    "?ì •?±ì˜ ?ì¹™": 30, "êµ?????˜ë¬´": 30, "ê¸°ë³¸ê¶?ë³´ì¥ ?˜ë¬´": 30,
    "ìµœì†Œ?œì˜ ?í™œ ë³´ì¥": 30, "êµìœ¡?œë„ ?•ë¦½": 30, "ê·¼ë¡œì¡°ê±´??ê¸°ì?": 30,
    "?˜ê²½ë³´ì „": 30, "ë¬¸í™”ì§„í¥": 30, "ë³µì?ì¦ì§„": 30
}

# ê¸°ë³¸ ëª©í‘œ ê±´ìˆ˜ (?°ì„ ?œìœ„ê°€ ?†ëŠ” ?¤ì›Œ??
DEFAULT_TARGET_COUNT = 15


class ConstitutionalDecisionCollector:
    """?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?´ë˜??""
    
    def __init__(self, config: LawOpenAPIConfig):
        self.client = LawOpenAPIClient(config)
        self.output_dir = Path("data/raw/constitutional_decisions")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ?˜ì§‘ ?íƒœ ê´€ë¦?
        self.collected_decisions = set()  # ì¤‘ë³µ ë°©ì?
        self.detailed_decisions = []
        self.current_batch = []
        self.batch_size = 50  # ë°°ì¹˜ ?¬ê¸°
        
        # ?µê³„ ?•ë³´
        self.stats = {
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'status': 'running',
            'target_count': 0,
            'collected_count': 0,
            'duplicate_count': 0,
            'failed_count': 0,
            'keywords_processed': 0,
            'total_keywords': len(CONSTITUTIONAL_KEYWORDS),
            'api_requests_made': 0,
            'api_errors': 0,
            'last_keyword_processed': None
        }
        
        # Graceful shutdown ê´€??ë³€??
        self.shutdown_requested = False
        self.checkpoint_file = None
        self.resume_info = {
            'progress_percentage': 0.0,
            'last_keyword_processed': None,
            'can_resume': False
        }
        
        # ?œê·¸???¸ë“¤???±ë¡
        self._setup_signal_handlers()
        
        # ì¢…ë£Œ ???•ë¦¬ ?‘ì—… ?±ë¡
        atexit.register(self._cleanup_on_exit)
    
    def _setup_signal_handlers(self):
        """?œê·¸???¸ë“¤???¤ì •"""
        def signal_handler(signum, frame):
            logger = logging.getLogger(__name__)
            logger.info(f"?œê·¸??{signum} ?˜ì‹ . Graceful shutdown ?œì‘...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # ì¢…ë£Œ ? í˜¸
    
    def _cleanup_on_exit(self):
        """?„ë¡œê·¸ë¨ ì¢…ë£Œ ???•ë¦¬ ?‘ì—…"""
        if self.detailed_decisions or self.current_batch:
            logger = logging.getLogger(__name__)
            logger.info("?˜ì§‘???°ì´?°ë? ?€??ì¤?..")
            self._save_checkpoint()
            logger.info(f"ì´?{len(self.detailed_decisions)}ê±´ì˜ ?°ì´?°ê? ?€?¥ë˜?ˆìŠµ?ˆë‹¤.")
    
    def _save_checkpoint(self):
        """ì²´í¬?¬ì¸???€??""
        if not self.checkpoint_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.checkpoint_file = self.output_dir / f"collection_checkpoint_{timestamp}.json"
        
        # ?„ì¬ ë°°ì¹˜ë¥??ì„¸ ?°ì´?°ì— ì¶”ê?
        if self.current_batch:
            self.detailed_decisions.extend(self.current_batch)
            self.current_batch = []
        
        # ì§„í–‰ë¥?ê³„ì‚°
        if self.stats['total_keywords'] > 0:
            self.resume_info['progress_percentage'] = (
                self.stats['keywords_processed'] / self.stats['total_keywords'] * 100
            )
        
        checkpoint_data = {
            'stats': self.stats,
            'resume_info': self.resume_info,
            'shutdown_info': {
                'graceful_shutdown_supported': True,
                'shutdown_requested': self.shutdown_requested,
                'shutdown_reason': 'User interrupt' if self.shutdown_requested else None
            },
            'detailed_decisions': self.detailed_decisions,
            'collected_decisions': list(self.collected_decisions)
        }
        
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            logger = logging.getLogger(__name__)
            logger.debug(f"ì²´í¬?¬ì¸???€?? {self.checkpoint_file}")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"ì²´í¬?¬ì¸???€???¤íŒ¨: {e}")
    
    def _load_checkpoint(self):
        """ì²´í¬?¬ì¸??ë¡œë“œ"""
        checkpoint_files = list(self.output_dir.glob("collection_checkpoint_*.json"))
        if not checkpoint_files:
            return False
        
        # ê°€??ìµœê·¼ ì²´í¬?¬ì¸???Œì¼ ë¡œë“œ
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # ?íƒœ ë³µì›
            self.stats = checkpoint_data.get('stats', self.stats)
            self.resume_info = checkpoint_data.get('resume_info', self.resume_info)
            self.detailed_decisions = checkpoint_data.get('detailed_decisions', [])
            self.collected_decisions = set(checkpoint_data.get('collected_decisions', []))
            self.checkpoint_file = latest_checkpoint
            
            logger = logging.getLogger(__name__)
            logger.info(f"ì²´í¬?¬ì¸??ë¡œë“œ ?„ë£Œ: {len(self.detailed_decisions)}ê±´ì˜ ?ì„¸ ?°ì´??ë³µì›")
            return True
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"ì²´í¬?¬ì¸??ë¡œë“œ ?¤íŒ¨: {e}")
            return False
    
    def _check_shutdown(self):
        """ì¢…ë£Œ ?”ì²­ ?•ì¸"""
        if self.shutdown_requested:
            logger = logging.getLogger(__name__)
            logger.info("ì¢…ë£Œ ?”ì²­??ê°ì??˜ì—ˆ?µë‹ˆ?? ?„ì¬ ?‘ì—…???„ë£Œ????ì¢…ë£Œ?©ë‹ˆ??")
            return True
        return False
    
    def collect_decisions_by_keyword(self, keyword: str, max_count: int = 50) -> List[Dict[str, Any]]:
        """?¤ì›Œ?œë¡œ ?Œì¬ê²°ì •ë¡€ ê²€??ë°??˜ì§‘"""
        logger = logging.getLogger(__name__)
        logger.info(f"?¤ì›Œ??'{keyword}'ë¡??Œì¬ê²°ì •ë¡€ ê²€???œì‘ (ëª©í‘œ: {max_count}ê±?...")
        
        decisions = []
        page = 1
        
        while len(decisions) < max_count:
            # ì¢…ë£Œ ?”ì²­ ?•ì¸
            if self._check_shutdown():
                break
                
            try:
                # ?˜ì´ì§€ë³?ì§„í–‰ë¥??œì‹œ
                page_progress = (page - 1) * 20
                logger.info(f"?“„ ?˜ì´ì§€ {page} ?”ì²­ ì¤?.. (?„ì¬ ?˜ì§‘: {len(decisions)}/{max_count}ê±? ì§„í–‰ë¥? {len(decisions)/max_count*100:.1f}%)")
                
                # queryê°€ ?ˆëŠ” ê²½ìš°?ë§Œ ê²€?? ?†ìœ¼ë©??„ì²´ ëª©ë¡ ì¡°íšŒ (? ê³ ?¼ì ?´ë¦¼ì°¨ìˆœ)
                if keyword and keyword.strip():
                    logger.debug(f"?” ?¤ì›Œ??'{keyword}'ë¡?ê²€???”ì²­ (? ê³ ?¼ì ?´ë¦¼ì°¨ìˆœ)")
                    results = self.client.get_constitutional_list(
                        query=keyword,
                        display=20,  # ?‘ì? ë°°ì¹˜ ?¬ê¸°ë¡??œì‘
                        page=page,
                        search=1  # ? ê³ ?¼ì ?´ë¦¼ì°¨ìˆœ ?•ë ¬ (ìµœì‹ ??
                    )
                else:
                    logger.debug("?“‹ ?„ì²´ ëª©ë¡ ì¡°íšŒ ?”ì²­ (? ê³ ?¼ì ?´ë¦¼ì°¨ìˆœ)")
                    results = self.client.get_constitutional_list(
                        display=20,  # ?‘ì? ë°°ì¹˜ ?¬ê¸°ë¡??œì‘
                        page=page,
                        search=1  # ? ê³ ?¼ì ?´ë¦¼ì°¨ìˆœ ?•ë ¬ (ìµœì‹ ??
                    )
                
                logger.info(f"?“Š API ?‘ë‹µ ê²°ê³¼: {len(results) if results else 0}ê±?)
                
                if not results:
                    logger.info("???´ìƒ ê²°ê³¼ê°€ ?†ì–´??ê²€?‰ì„ ì¤‘ë‹¨?©ë‹ˆ??")
                    break
                
                new_decisions = 0
                for result in results:
                    # ì¢…ë£Œ ?”ì²­ ?•ì¸
                    if self._check_shutdown():
                        break
                        
                    # ?Œì¬ê²°ì •ë¡€ ID ?•ì¸ (API ?‘ë‹µ êµ¬ì¡°???°ë¼)
                    decision_id = result.get('?Œì¬ê²°ì •ë¡€?¼ë ¨ë²ˆí˜¸') or result.get('ID') or result.get('id')
                    if decision_id and decision_id not in self.collected_decisions:
                        decisions.append(result)
                        self.collected_decisions.add(decision_id)
                        self.stats['collected_count'] += 1
                        new_decisions += 1
                        
                        logger.info(f"???ˆë¡œ???Œì¬ê²°ì •ë¡€ ?˜ì§‘: {result.get('?¬ê±´ëª?, 'Unknown')} (ID: {decision_id})")
                        logger.info(f"   ?“ˆ ?„ì¬ ì§„í–‰ë¥? {len(decisions)}/{max_count}ê±?({len(decisions)/max_count*100:.1f}%)")
                        
                        if len(decisions) >= max_count:
                            logger.info(f"?¯ ëª©í‘œ ?˜ëŸ‰ {max_count}ê±´ì— ?„ë‹¬?ˆìŠµ?ˆë‹¤!")
                            break
                    else:
                        self.stats['duplicate_count'] += 1
                        logger.debug(f"ì¤‘ë³µ???Œì¬ê²°ì •ë¡€ ê±´ë„ˆ?°ê¸°: {decision_id}")
                
                logger.info(f"?“„ ?˜ì´ì§€ {page} ?„ë£Œ: {new_decisions}ê±´ì˜ ?ˆë¡œ??ê²°ì •ë¡€ ?˜ì§‘")
                logger.info(f"   ?“Š ?„ì  ?˜ì§‘: {len(decisions)}/{max_count}ê±?({len(decisions)/max_count*100:.1f}%)")
                
                page += 1
                self.stats['api_requests_made'] += 1
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API ?”ì²­ ?œë„ê°€ ê±°ì˜ ?Œì§„?˜ì—ˆ?µë‹ˆ??")
                    break
                    
            except Exception as e:
                logger.error(f"?¤ì›Œ??'{keyword}' ê²€??ì¤??¤ë¥˜: {e}")
                self.stats['api_errors'] += 1
                break
        
        logger.info(f"?¤ì›Œ??'{keyword}'ë¡?ì´?{len(decisions)}ê±??˜ì§‘ ?„ë£Œ")
        return decisions
    
    def collect_decisions_by_date_range(self, start_date: str, end_date: str, max_count: int = 1000) -> List[Dict[str, Any]]:
        """? ì§œ ë²”ìœ„ë¡??Œì¬ê²°ì •ë¡€ ê²€??ë°??˜ì§‘"""
        logger = logging.getLogger(__name__)
        logger.info(f"?“… ? ì§œ ë²”ìœ„ {start_date} ~ {end_date}ë¡??Œì¬ê²°ì •ë¡€ ê²€???œì‘ (ëª©í‘œ: {max_count:,}ê±?...")
        
        decisions = []
        page = 1
        total_pages_estimated = max_count // 100 + 1  # ?€?µì ???˜ì´ì§€ ??ì¶”ì •
        
        while len(decisions) < max_count:
            # ì¢…ë£Œ ?”ì²­ ?•ì¸
            if self._check_shutdown():
                break
                
            try:
                # ?˜ì´ì§€ë³?ì§„í–‰ë¥??œì‹œ
                progress = (page - 1) / total_pages_estimated * 100 if total_pages_estimated > 0 else 0
                logger.info(f"?“„ ?˜ì´ì§€ {page} ?”ì²­ ì¤?.. (?„ì¬ ?˜ì§‘: {len(decisions):,}/{max_count:,}ê±? ì§„í–‰ë¥? {len(decisions)/max_count*100:.1f}%)")
                
                results = self.client.get_constitutional_list(
                    display=100,
                    page=page,
                    from_date=start_date,
                    to_date=end_date,
                    search=1  # ?¬ê±´ëª?ê²€??
                )
                
                logger.info(f"?“Š API ?‘ë‹µ ê²°ê³¼: {len(results) if results else 0}ê±?)
                
                if not results:
                    logger.info("???´ìƒ ê²°ê³¼ê°€ ?†ì–´??ê²€?‰ì„ ì¤‘ë‹¨?©ë‹ˆ??")
                    break
                
                new_decisions = 0
                for result in results:
                    # ì¢…ë£Œ ?”ì²­ ?•ì¸
                    if self._check_shutdown():
                        break
                        
                    # ?Œì¬ê²°ì •ë¡€ ID ?•ì¸ (API ?‘ë‹µ êµ¬ì¡°???°ë¼)
                    decision_id = result.get('?Œì¬ê²°ì •ë¡€?¼ë ¨ë²ˆí˜¸') or result.get('ID') or result.get('id')
                    if decision_id and decision_id not in self.collected_decisions:
                        decisions.append(result)
                        self.collected_decisions.add(decision_id)
                        self.stats['collected_count'] += 1
                        new_decisions += 1
                        
                        # 5ê±´ë§ˆ???˜ì§‘ ?„í™© ?œì‹œ (???ì£¼)
                        if len(decisions) % 5 == 0:
                            logger.info(f"??{len(decisions):,}ê±??˜ì§‘ ?„ë£Œ (ì§„í–‰ë¥? {len(decisions)/max_count*100:.1f}%)")
                        
                        if len(decisions) >= max_count:
                            logger.info(f"?¯ ëª©í‘œ ?˜ëŸ‰ {max_count:,}ê±´ì— ?„ë‹¬?ˆìŠµ?ˆë‹¤!")
                            break
                    else:
                        self.stats['duplicate_count'] += 1
                
                logger.info(f"?“„ ?˜ì´ì§€ {page} ?„ë£Œ: {new_decisions}ê±´ì˜ ?ˆë¡œ??ê²°ì •ë¡€ ?˜ì§‘")
                logger.info(f"   ?“Š ?„ì  ?˜ì§‘: {len(decisions):,}/{max_count:,}ê±?({len(decisions)/max_count*100:.1f}%)")
                logger.info(f"   ?±ï¸  ?ˆìƒ ?¨ì? ?˜ì´ì§€: {max(0, (max_count - len(decisions)) // 100)}?˜ì´ì§€")
                
                page += 1
                self.stats['api_requests_made'] += 1
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API ?”ì²­ ?œë„ê°€ ê±°ì˜ ?Œì§„?˜ì—ˆ?µë‹ˆ??")
                    break
                    
            except Exception as e:
                logger.error(f"? ì§œ ë²”ìœ„ ê²€??ì¤??¤ë¥˜: {e}")
                self.stats['api_errors'] += 1
                break
        
        logger.info("=" * 60)
        logger.info(f"?“… ? ì§œ ë²”ìœ„ ?˜ì§‘ ?„ë£Œ!")
        logger.info(f"?“Š ìµœì¢… ?˜ì§‘ ê²°ê³¼: {len(decisions):,}ê±?)
        logger.info(f"?“„ ì²˜ë¦¬???˜ì´ì§€: {page-1}?˜ì´ì§€")
        logger.info(f"?Œ API ?”ì²­ ?? {self.stats['api_requests_made']:,}??)
        logger.info(f"??ì¤‘ë³µ ?œì™¸: {self.stats['duplicate_count']:,}ê±?)
        logger.info("=" * 60)
        return decisions
    
    def collect_decision_details(self, decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """?Œì¬ê²°ì •ë¡€ ?ì„¸ ?•ë³´ ?˜ì§‘"""
        # ?Œì¬ê²°ì •ë¡€ ID ?•ì¸ (API ?‘ë‹µ êµ¬ì¡°???°ë¼)
        decision_id = decision.get('?Œì¬ê²°ì •ë¡€?¼ë ¨ë²ˆí˜¸') or decision.get('ID') or decision.get('id')
        if not decision_id:
            logger = logging.getLogger(__name__)
            logger.warning(f"?Œì¬ê²°ì •ë¡€ IDë¥?ì°¾ì„ ???†ìŠµ?ˆë‹¤: {decision}")
            return None
        
        try:
            detail = self.client.get_constitutional_detail(constitutional_id=decision_id)
            if detail:
                # ê¸°ë³¸ ?•ë³´?€ ?ì„¸ ?•ë³´ ê²°í•©
                combined_data = {
                    'basic_info': decision,
                    'detail_info': detail,
                    'collected_at': datetime.now().isoformat()
                }
                return combined_data
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"?Œì¬ê²°ì •ë¡€ {decision_id} ?ì„¸ ?•ë³´ ?˜ì§‘ ?¤íŒ¨: {e}")
            self.stats['api_errors'] += 1
        
        return None
    
    def classify_decision_type(self, decision: Dict[str, Any]) -> str:
        """?Œì¬ê²°ì •ë¡€ ? í˜• ë¶„ë¥˜"""
        case_name = decision.get('?¬ê±´ëª?, '').lower()
        decision_text = decision.get('?ì‹œ?¬í•­', '') + ' ' + decision.get('?ê²°?”ì?', '')
        decision_text = decision_text.lower()
        
        # ê²°ì •? í˜•ë³??¤ì›Œ??ë§¤ì¹­
        for decision_type, keywords in DECISION_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in case_name or keyword in decision_text:
                    return decision_type
        
        # ê¸°ë³¸ê¶Œë³„ ë¶„ë¥˜
        if any(keyword in case_name for keyword in ["?ëª…ê¶?, "? ì²´???ìœ "]):
            return "ê¸°ë³¸ê¶??ëª…? ì²´"
        elif any(keyword in case_name for keyword in ["?¬ìƒ??, "?‘ì‹¬", "ì¢…êµ"]):
            return "ê¸°ë³¸ê¶??¬ìƒ??
        elif any(keyword in case_name for keyword in ["?¸ë¡ ", "ì¶œíŒ", "ì§‘íšŒ", "ê²°ì‚¬"]):
            return "ê¸°ë³¸ê¶??œí˜„"
        elif any(keyword in case_name for keyword in ["?¬ì‚°ê¶?, "ì§ì—…? íƒ"]):
            return "ê¸°ë³¸ê¶?ê²½ì œ"
        elif any(keyword in case_name for keyword in ["êµìœ¡", "ê·¼ë¡œ", "?˜ê²½"]):
            return "ê¸°ë³¸ê¶??¬íšŒ"
        elif any(keyword in case_name for keyword in ["?Œë²•?Œì›", "?„í—Œë²•ë¥ "]):
            return "?Œë²•?¬íŒ"
        else:
            return "ê¸°í?"
    
    def save_batch_data(self, batch_data: List[Dict[str, Any]], category: str):
        """ë°°ì¹˜ ?°ì´?°ë? ?Œì¼ë¡??€??""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"batch_{category}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        batch_info = {
            'metadata': {
                'category': category,
                'count': len(batch_data),
                'timestamp': timestamp,
                'batch_size': self.batch_size
            },
            'data': batch_data
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(batch_info, f, ensure_ascii=False, indent=2)
            logger = logging.getLogger(__name__)
            logger.debug(f"ë°°ì¹˜ ?°ì´???€?? {filepath}")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"ë°°ì¹˜ ?°ì´???€???¤íŒ¨: {e}")
    
    def collect_all_decisions(self, target_count: int = 1000, resume: bool = True, keyword_mode: bool = True):
        """ëª¨ë“  ?Œì¬ê²°ì •ë¡€ ?˜ì§‘"""
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info(f"?? ?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?œì‘!")
        logger.info(f"?¯ ëª©í‘œ ?˜ëŸ‰: {target_count:,}ê±?)
        if keyword_mode:
            logger.info(f"?“ ê²€??ë°©ì‹: ?¤ì›Œ??ê¸°ë°˜ ({len(CONSTITUTIONAL_KEYWORDS)}ê°??¤ì›Œ??")
        else:
            logger.info(f"?“ ê²€??ë°©ì‹: ?„ì²´ ?°ì´???˜ì§‘ (?¤ì›Œ??ë¬´ê?)")
        logger.info(f"???œì‘ ?œê°„: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        self.stats['target_count'] = target_count
        
        # ê¸°ì¡´ ì²´í¬?¬ì¸??ë³µì›
        if resume:
            if self._load_checkpoint():
                logger.info(f"ê¸°ì¡´ ì§„í–‰ ?í™© ë³µì›: {len(self.detailed_decisions)}ê±?)
                self.resume_info['can_resume'] = True
        
        all_decisions = []
        
        if keyword_mode:
            # 1. ?°ì„ ?œìœ„ ?¤ì›Œ?œë³„ ê²€??
            logger.info(f"ì´?{len(CONSTITUTIONAL_KEYWORDS)}ê°??¤ì›Œ?œë¡œ ê²€???œì‘")
            
            for i, keyword in enumerate(CONSTITUTIONAL_KEYWORDS):
                # ?´ì „???„ë£Œ???¤ì›Œ?œëŠ” ê±´ë„ˆ?°ê¸°
                if resume and i < self.stats['keywords_processed']:
                    logger.info(f"?¤ì›Œ??'{keyword}' ê±´ë„ˆ?°ê¸° (?´ë? ì²˜ë¦¬??")
                    continue
                    
                if len(all_decisions) >= target_count:
                    logger.info(f"ëª©í‘œ ?˜ëŸ‰ {target_count}ê±´ì— ?„ë‹¬?˜ì—¬ ?¤ì›Œ??ê²€?‰ì„ ì¤‘ë‹¨?©ë‹ˆ??")
                    break
                
                # ì¢…ë£Œ ?”ì²­ ?•ì¸
                if self._check_shutdown():
                    break
                    
                try:
                    # ?¤ì›Œ?œë³„ ëª©í‘œ ê±´ìˆ˜ ?¤ì •
                    max_count = KEYWORD_PRIORITIES.get(keyword, DEFAULT_TARGET_COUNT)
                    max_count = min(max_count, target_count - len(all_decisions))
                    
                    # ì§„í–‰ë¥?ê³„ì‚°
                    progress = (i + 1) / len(CONSTITUTIONAL_KEYWORDS) * 100
                    remaining_keywords = len(CONSTITUTIONAL_KEYWORDS) - i - 1
                    
                    logger.info(f"?“Š ì§„í–‰ë¥? {progress:.1f}% ({i+1}/{len(CONSTITUTIONAL_KEYWORDS)}) - ?¤ì›Œ??'{keyword}' ì²˜ë¦¬ ?œì‘")
                    logger.info(f"?¯ ëª©í‘œ: {max_count}ê±? ?°ì„ ?œìœ„: {KEYWORD_PRIORITIES.get(keyword, 'ê¸°ë³¸')}, ?¨ì? ?¤ì›Œ?? {remaining_keywords}ê°?)
                    
                    decisions = self.collect_decisions_by_keyword(keyword, max_count)
                    all_decisions.extend(decisions)
                    
                    self.stats['keywords_processed'] += 1
                    self.stats['last_keyword_processed'] = keyword
                    
                    # ?ì„¸???„ë£Œ ?•ë³´
                    completion_rate = len(all_decisions) / target_count * 100 if target_count > 0 else 0
                    logger.info(f"???¤ì›Œ??'{keyword}' ?„ë£Œ!")
                    logger.info(f"   ?“ˆ ?˜ì§‘: {len(decisions)}ê±?| ?„ì : {len(all_decisions)}ê±?| ëª©í‘œ ?€ë¹? {completion_rate:.1f}%")
                    logger.info(f"   ?±ï¸  ?¨ì? ?¤ì›Œ?? {remaining_keywords}ê°?)
                    
                    # ì§„í–‰ ?í™© ?€??(10ê°??¤ì›Œ?œë§ˆ??
                    if self.stats['keywords_processed'] % 10 == 0:
                        logger.info("ì§„í–‰ ?í™©??ì²´í¬?¬ì¸?¸ì— ?€?¥í•©?ˆë‹¤.")
                        self._save_checkpoint()
                    
                    # API ?”ì²­ ?œí•œ ?•ì¸
                    stats = self.client.get_request_stats()
                    if stats['remaining_requests'] < 100:
                        logger.warning("API ?”ì²­ ?œë„ê°€ ë¶€ì¡±í•©?ˆë‹¤.")
                        break
                        
                except Exception as e:
                    logger.error(f"?¤ì›Œ??'{keyword}' ê²€???¤íŒ¨: {e}")
                    self.stats['api_errors'] += 1
                    continue
        else:
            # ?¤ì›Œ???†ì´ ?„ì²´ ?°ì´???˜ì§‘
            logger.info("?” ?¤ì›Œ???†ì´ ?„ì²´ ?Œì¬ê²°ì •ë¡€ ?˜ì§‘ ?œì‘")
            logger.info(f"?“… ?˜ì§‘ ê¸°ê°„: ìµœê·¼ 5??(2020??~ ?„ì¬)")
            logger.info(f"?¯ ëª©í‘œ ?˜ëŸ‰: {target_count:,}ê±?)
            
            # ìµœê·¼ 5?„ê°„ ?°ì´???˜ì§‘
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y%m%d')
            
            logger.info(f"?“Š ?˜ì§‘ ê¸°ê°„: {start_date} ~ {end_date}")
            all_decisions = self.collect_decisions_by_date_range(
                start_date, end_date, target_count
            )
        
        # 2. ? ì§œ ë²”ìœ„ë³?ê²€??(ìµœê·¼ 5?? - ?¤ì›Œ??ëª¨ë“œ?ì„œë§?
        if keyword_mode and len(all_decisions) < target_count and not self._check_shutdown():
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y%m%d')
            
            remaining_count = target_count - len(all_decisions)
            date_decisions = self.collect_decisions_by_date_range(
                start_date, end_date, remaining_count
            )
            all_decisions.extend(date_decisions)
        
        logger.info("=" * 60)
        logger.info(f"?“‹ 1?¨ê³„ ?„ë£Œ: ì´?{len(all_decisions)}ê±´ì˜ ?Œì¬ê²°ì •ë¡€ ëª©ë¡ ?˜ì§‘ ?„ë£Œ")
        logger.info("=" * 60)
        
        # 3. ê°??Œì¬ê²°ì •ë¡€???ì„¸ ?•ë³´ ?˜ì§‘
        logger.info(f"?” 2?¨ê³„ ?œì‘: ?ì„¸ ?•ë³´ ?˜ì§‘ ({len(all_decisions)}ê±?")
        for i, decision in enumerate(all_decisions):
            if i >= target_count:
                break
            
            # ì¢…ë£Œ ?”ì²­ ?•ì¸
            if self._check_shutdown():
                break
                
            try:
                detail = self.collect_decision_details(decision)
                if detail:
                    # ê²°ì •? í˜• ë¶„ë¥˜
                    decision_type = self.classify_decision_type(decision)
                    detail['decision_type'] = decision_type
                    
                    self.current_batch.append(detail)
                    
                    # ë°°ì¹˜ ?¬ê¸°???„ë‹¬?˜ë©´ ?€??
                    if len(self.current_batch) >= self.batch_size:
                        self.save_batch_data(self.current_batch, f"constitutional_decisions_{i//self.batch_size}")
                        self.detailed_decisions.extend(self.current_batch)
                        self.current_batch = []
                
                # ì§„í–‰ë¥?ë¡œê·¸ ë°?ì²´í¬?¬ì¸???€??
                if (i + 1) % 50 == 0:  # 50ê±´ë§ˆ??ì²´í¬?¬ì¸???€??
                    progress = (i + 1) / len(all_decisions) * 100
                    logger.info(f"?“Š ?ì„¸ ?•ë³´ ?˜ì§‘ ì§„í–‰ë¥? {i + 1:,}/{len(all_decisions):,} ({progress:.1f}%)")
                    self._save_checkpoint()
                elif (i + 1) % 10 == 0:  # 10ê±´ë§ˆ??ê°„ë‹¨??ì§„í–‰ë¥??œì‹œ
                    progress = (i + 1) / len(all_decisions) * 100
                    logger.info(f"???ì„¸ ?•ë³´ ?˜ì§‘ ì§„í–‰ë¥? {i + 1:,}/{len(all_decisions):,} ({progress:.1f}%)")
                elif (i + 1) % 5 == 0:  # 5ê±´ë§ˆ??ê°„ë‹¨??ì§„í–‰ë¥??œì‹œ (?¤ì›Œ???†ì´ ?˜ì§‘ ??
                    progress = (i + 1) / len(all_decisions) * 100
                    logger.info(f"?” ?ì„¸ ?•ë³´ ?˜ì§‘: {i + 1:,}/{len(all_decisions):,} ({progress:.1f}%)")
                elif (i + 1) % 2 == 0:  # 2ê±´ë§ˆ??ê°„ë‹¨??ì§„í–‰ë¥??œì‹œ (ë§¤ìš° ?ì£¼)
                    progress = (i + 1) / len(all_decisions) * 100
                    logger.info(f"???ì„¸ ?•ë³´ ?˜ì§‘: {i + 1:,}/{len(all_decisions):,} ({progress:.1f}%)")
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API ?”ì²­ ?œë„ê°€ ê±°ì˜ ?Œì§„?˜ì—ˆ?µë‹ˆ??")
                    break
                    
            except Exception as e:
                logger.error(f"?Œì¬ê²°ì •ë¡€ {i} ?ì„¸ ?•ë³´ ?˜ì§‘ ?¤íŒ¨: {e}")
                self.stats['failed_count'] += 1
                continue
        
        # ë§ˆì?ë§?ë°°ì¹˜ ?€??
        if self.current_batch:
            self.save_batch_data(self.current_batch, f"constitutional_decisions_final")
            self.detailed_decisions.extend(self.current_batch)
            self.current_batch = []
        
        # ì¢…ë£Œ ?”ì²­???ˆì—ˆ?”ì? ?•ì¸
        if self._check_shutdown():
            logger.info("=" * 60)
            logger.info("? ï¸ ?¬ìš©???”ì²­???˜í•´ ?˜ì§‘??ì¤‘ë‹¨?˜ì—ˆ?µë‹ˆ??")
            logger.info(f"?“Š ?„ì¬ê¹Œì? {len(self.detailed_decisions)}ê±´ì˜ ?ì„¸ ?°ì´?°ê? ?˜ì§‘?˜ì—ˆ?µë‹ˆ??")
            logger.info("=" * 60)
            self.stats['status'] = 'interrupted'
        else:
            logger.info("=" * 60)
            logger.info("?‰ ?Œì¬ê²°ì •ë¡€ ?˜ì§‘???±ê³µ?ìœ¼ë¡??„ë£Œ?˜ì—ˆ?µë‹ˆ??")
            logger.info(f"?“Š ìµœì¢… ?˜ì§‘ ê²°ê³¼: {len(self.detailed_decisions)}ê±?)
            logger.info(f"???„ë£Œ ?œê°„: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 60)
            self.stats['status'] = 'completed'
        
        # ìµœì¢… ?µê³„ ?…ë°?´íŠ¸
        self.stats['end_time'] = datetime.now().isoformat()
        self.stats['collected_count'] = len(self.detailed_decisions)
        
        # 4. ?˜ì§‘ ê²°ê³¼ ?”ì•½ ?ì„±
        self.generate_collection_summary()
        
        # ?„ë£Œ ??ì²´í¬?¬ì¸???Œì¼ ?•ë¦¬
        self._cleanup_checkpoint_files()
    
    def _cleanup_checkpoint_files(self):
        """ì²´í¬?¬ì¸???Œì¼ ?•ë¦¬"""
        try:
            if self.checkpoint_file and self.checkpoint_file.exists():
                # ?„ë£Œ??ê²½ìš°?ë§Œ ì²´í¬?¬ì¸???Œì¼ ?? œ
                if self.stats['status'] == 'completed':
                    self.checkpoint_file.unlink()
                    logger = logging.getLogger(__name__)
                    logger.debug("?„ë£Œ ??ì²´í¬?¬ì¸???Œì¼ ?? œ")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"ì²´í¬?¬ì¸???Œì¼ ?•ë¦¬ ?¤íŒ¨: {e}")
    
    def reset_collection(self):
        """?˜ì§‘ ?íƒœ ì´ˆê¸°??""
        try:
            # ì²´í¬?¬ì¸???Œì¼???? œ
            checkpoint_files = list(self.output_dir.glob("collection_checkpoint_*.json"))
            for file_path in checkpoint_files:
                file_path.unlink()
            
            # ?íƒœ ì´ˆê¸°??
            self.detailed_decisions = []
            self.current_batch = []
            self.collected_decisions = set()
            self.stats = {
                'start_time': datetime.now().isoformat(),
                'end_time': None,
                'status': 'running',
                'target_count': 0,
                'collected_count': 0,
                'duplicate_count': 0,
                'failed_count': 0,
                'keywords_processed': 0,
                'total_keywords': len(CONSTITUTIONAL_KEYWORDS),
                'api_requests_made': 0,
                'api_errors': 0,
                'last_keyword_processed': None
            }
            self.resume_info = {
                'progress_percentage': 0.0,
                'last_keyword_processed': None,
                'can_resume': False
            }
            
            logger = logging.getLogger(__name__)
            logger.info("?˜ì§‘ ?íƒœê°€ ì´ˆê¸°?”ë˜?ˆìŠµ?ˆë‹¤.")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"?˜ì§‘ ?íƒœ ì´ˆê¸°???¤íŒ¨: {e}")
    
    def get_collection_status(self):
        """?„ì¬ ?˜ì§‘ ?íƒœ ë°˜í™˜"""
        return {
            'stats': self.stats,
            'resume_info': self.resume_info,
            'checkpoint_file': str(self.checkpoint_file) if self.checkpoint_file else None,
            'output_directory': str(self.output_dir)
        }
    
    def generate_collection_summary(self):
        """?˜ì§‘ ê²°ê³¼ ?”ì•½ ?ì„±"""
        # ê²°ì •? í˜•ë³??µê³„
        decision_type_stats = {}
        
        for decision in self.detailed_decisions:
            decision_type = decision.get('decision_type', 'ê¸°í?')
            decision_type_stats[decision_type] = decision_type_stats.get(decision_type, 0) + 1
        
        summary = {
            'collection_date': datetime.now().isoformat(),
            'total_decisions': len(self.detailed_decisions),
            'decision_type_distribution': decision_type_stats,
            'api_stats': self.client.get_request_stats(),
            'collection_stats': self.stats
        }
        
        summary_file = self.output_dir / f"collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger = logging.getLogger(__name__)
            logger.info(f"?˜ì§‘ ê²°ê³¼ ?”ì•½ ?€?? {summary_file}")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"?˜ì§‘ ê²°ê³¼ ?”ì•½ ?€???¤íŒ¨: {e}")
