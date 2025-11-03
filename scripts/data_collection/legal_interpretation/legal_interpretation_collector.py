#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ê¸??´ë˜??(collect_precedents.py êµ¬ì¡° ì°¸ê³ )
"""

import json
import time
import random
import signal
import atexit
import hashlib
import traceback
import gc  # ê°€ë¹„ì? ì»¬ë ‰??
import re  # ?•ê·œ?œí˜„??
from bs4 import BeautifulSoup  # HTML ?Œì‹±
try:
    import psutil  # ë©”ëª¨ë¦?ëª¨ë‹ˆ?°ë§
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil???¤ì¹˜?˜ì? ?Šì•˜?µë‹ˆ?? ë©”ëª¨ë¦?ëª¨ë‹ˆ?°ë§??ë¹„í™œ?±í™”?©ë‹ˆ??")
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
from scripts.legal_interpretation.legal_interpretation_models import (
    CollectionStats, LegalInterpretationData, CollectionStatus, InterpretationCategory, MinistryType,
    INTERPRETATION_KEYWORDS, PRIORITY_KEYWORDS, KEYWORD_TARGET_COUNTS, DEFAULT_KEYWORD_COUNT,
    MINISTRY_KEYWORDS, INTERPRETATION_TOPIC_KEYWORDS, FALLBACK_KEYWORDS
)
from scripts.legal_interpretation.legal_interpretation_logger import setup_logging

logger = setup_logging()


class LegalInterpretationCollector:
    """ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ ?´ë˜??(ê°œì„ ??ë²„ì „)"""
    
    def __init__(self, config: LawOpenAPIConfig, output_dir: Optional[Path] = None):
        """
        ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ê¸?ì´ˆê¸°??
        
        Args:
            config: API ?¤ì • ê°ì²´
            output_dir: ì¶œë ¥ ?”ë ‰? ë¦¬ (ê¸°ë³¸ê°? data/raw/legal_interpretations)
        """
        self.client = LawOpenAPIClient(config)
        self.output_dir = output_dir or Path("data/raw/legal_interpretations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ë°°ì¹˜ ?€???”ë ‰? ë¦¬
        self.batch_dir = self.output_dir / "batches"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        
        # ?°ì´??ê´€ë¦?
        self.collected_interpretations: Set[str] = set()  # ì¤‘ë³µ ë°©ì?
        self.processed_keywords: Set[str] = set()  # ì²˜ë¦¬???¤ì›Œ??ì¶”ì 
        self.pending_interpretations: List[LegalInterpretationData] = []  # ?„ì‹œ ?€?¥ì†Œ
        
        # ?¤ì •
        self.batch_size = 5  # ë°°ì¹˜ ?€???¬ê¸° (?ì„¸ ?•ë³´ ?˜ì§‘?¼ë¡œ ?¸í•œ ë©”ëª¨ë¦?ìµœì ??
        self.max_retries = 3  # ìµœë? ?¬ì‹œ???Ÿìˆ˜
        self.retry_delay = 5  # ?¬ì‹œ??ê°„ê²© (ì´?
        self.api_delay_range = (1.0, 3.0)  # API ?”ì²­ ê°?ì§€??ë²”ìœ„
        
        # ë©”ëª¨ë¦?ê´€ë¦??¤ì •
        self.memory_check_interval = 50  # ë©”ëª¨ë¦?ì²´í¬ ê°„ê²© (API ?”ì²­ ??
        self.max_memory_usage = 80  # ìµœë? ë©”ëª¨ë¦??¬ìš©ë¥?(%)
        self.api_request_count = 0  # API ?”ì²­ ì¹´ìš´??
        
        # ë°°ì¹˜ ê´€ë¦?
        self.batch_counter = 0  # ë°°ì¹˜ ì¹´ìš´??
        
        # ?µê³„ ë°??íƒœ
        self.stats = CollectionStats()
        self.stats.total_keywords = len(INTERPRETATION_KEYWORDS)
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
        
        logger.info(f"ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ê¸?ì´ˆê¸°???„ë£Œ - ëª©í‘œ: {self.stats.target_count}ê±?)
    
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
        if self.pending_interpretations:
            logger.info("?„ë¡œê·¸ë¨ ì¢…ë£Œ ???„ì‹œ ?°ì´???€??ì¤?..")
            self._save_batch_interpretations()
        
        if self.checkpoint_file:
            logger.info("ìµœì¢… ì²´í¬?¬ì¸???€??ì¤?..")
            self._save_checkpoint(self.checkpoint_file)
    
    def _check_shutdown_requested(self) -> bool:
        """ì¢…ë£Œ ?”ì²­ ?•ì¸"""
        return self.shutdown_requested
    
    def _get_next_batch_number(self) -> int:
        """?¤ìŒ ë°°ì¹˜ ë²ˆí˜¸ ?ì„±"""
        self.batch_counter += 1
        return self.batch_counter
    
    def _extract_content_from_json(self, json_data: Dict[str, Any]) -> Tuple[str, str]:
        """JSON?ì„œ ì§ˆì˜?´ìš©ê³??Œì‹ ?´ìš© ì¶”ì¶œ"""
        try:
            question_content = ""
            answer_content = ""
            
            # ExpcService ?„ë“œ ?´ë? ?•ì¸
            expc_service = json_data.get('ExpcService', {})
            if isinstance(expc_service, dict):
                # ExpcService ?´ë??ì„œ ì§ˆì˜?”ì??€ ?Œë‹µ ì°¾ê¸°
                question_content = expc_service.get('ì§ˆì˜?”ì?', '').strip()
                answer_content = expc_service.get('?Œë‹µ', '').strip()
                
                # ì¶©ë¶„??ê¸¸ì´ê°€ ?ˆìœ¼ë©??¬ìš©
                if len(question_content) > 10 and len(answer_content) > 10:
                    return question_content, answer_content
            
            # ìµœìƒ???ˆë²¨?ì„œ???•ì¸
            # ?¤ì–‘???„ë“œëª…ìœ¼ë¡?ì§ˆì˜?´ìš© ì°¾ê¸°
            question_fields = [
                'ì§ˆì˜?”ì?', 'ì§ˆì˜?´ìš©', 'ì§ˆì˜ ?´ìš©', 'ì§ˆì˜?¬í•­', 'ì§ˆì˜ ?¬í•­',
                'question', 'ì§ˆë¬¸', 'ë¬¸ì˜?´ìš©', 'ë¬¸ì˜ ?´ìš©',
                'ì§ˆì˜', 'ë¬¸ì˜', '?”ì²­?´ìš©', '?”ì²­ ?´ìš©'
            ]
            
            for field in question_fields:
                if field in json_data and json_data[field]:
                    question_content = str(json_data[field]).strip()
                    if len(question_content) > 10:  # ì¶©ë¶„??ê¸¸ì´???´ìš©ë§?
                        break
            
            # ?¤ì–‘???„ë“œëª…ìœ¼ë¡??Œì‹ ?´ìš© ì°¾ê¸°
            answer_fields = [
                '?Œë‹µ', '?Œì‹ ?´ìš©', '?Œì‹  ?´ìš©', '?µë??´ìš©', '?µë? ?´ìš©',
                'answer', 'reply', '?´ì„?´ìš©', '?´ì„ ?´ìš©',
                '?Œì‹ ', '?µë?', '?´ì„', 'ê²°ë¡ '
            ]
            
            for field in answer_fields:
                if field in json_data and json_data[field]:
                    answer_content = str(json_data[field]).strip()
                    if len(answer_content) > 10:  # ì¶©ë¶„??ê¸¸ì´???´ìš©ë§?
                        break
            
            # ì¤‘ì²©??ê°ì²´?ì„œ ì°¾ê¸°
            if not question_content or not answer_content:
                for key, value in json_data.items():
                    if isinstance(value, dict):
                        sub_question, sub_answer = self._extract_content_from_json(value)
                        if sub_question and not question_content:
                            question_content = sub_question
                        if sub_answer and not answer_content:
                            answer_content = sub_answer
            
            # ìµœì¢… ?•ë¦¬
            question_content = self._clean_text(question_content)
            answer_content = self._clean_text(answer_content)
            
            return question_content, answer_content
            
        except Exception as e:
            logger.error(f"JSON ë³¸ë¬¸ ì¶”ì¶œ ?¤íŒ¨: {e}")
            return "", ""
    
    def _extract_content_from_html(self, html_content: str) -> Tuple[str, str]:
        """HTML?ì„œ ì§ˆì˜?´ìš©ê³??Œì‹ ?´ìš© ì¶”ì¶œ"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            question_content = ""
            answer_content = ""
            
            # ì§ˆì˜?´ìš© ì¶”ì¶œ (?¤ì–‘???¨í„´ ?œë„)
            question_patterns = [
                'ì§ˆì˜?´ìš©', 'ì§ˆì˜ ?´ìš©', 'ì§ˆì˜?¬í•­', 'ì§ˆì˜ ?¬í•­',
                'question', 'ì§ˆë¬¸', 'ë¬¸ì˜?´ìš©', 'ë¬¸ì˜ ?´ìš©'
            ]
            
            for pattern in question_patterns:
                # ?ìŠ¤?¸ë¡œ ?¨í„´ ê²€??
                question_elements = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
                for element in question_elements:
                    parent = element.parent
                    if parent:
                        # ?¤ìŒ ?•ì œ ?”ì†Œ?¤ì—???´ìš© ì¶”ì¶œ
                        content = self._extract_text_from_element(parent)
                        if content and len(content) > 50:  # ì¶©ë¶„??ê¸¸ì´???´ìš©ë§?
                            question_content = content
                            break
                if question_content:
                    break
            
            # ?Œì‹ ?´ìš© ì¶”ì¶œ (?¤ì–‘???¨í„´ ?œë„)
            answer_patterns = [
                '?Œì‹ ?´ìš©', '?Œì‹  ?´ìš©', '?µë??´ìš©', '?µë? ?´ìš©',
                'answer', 'reply', '?´ì„?´ìš©', '?´ì„ ?´ìš©'
            ]
            
            for pattern in answer_patterns:
                # ?ìŠ¤?¸ë¡œ ?¨í„´ ê²€??
                answer_elements = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
                for element in answer_elements:
                    parent = element.parent
                    if parent:
                        # ?¤ìŒ ?•ì œ ?”ì†Œ?¤ì—???´ìš© ì¶”ì¶œ
                        content = self._extract_text_from_element(parent)
                        if content and len(content) > 50:  # ì¶©ë¶„??ê¸¸ì´???´ìš©ë§?
                            answer_content = content
                            break
                if answer_content:
                    break
            
            # ?¨í„´?¼ë¡œ ì°¾ì? ëª»í•œ ê²½ìš° ?Œì´ë¸?êµ¬ì¡°?ì„œ ì¶”ì¶œ ?œë„
            if not question_content or not answer_content:
                tables = soup.find_all('table')
                for table in tables:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) >= 2:
                            label = cells[0].get_text(strip=True)
                            content = cells[1].get_text(strip=True)
                            
                            if any(pattern in label for pattern in question_patterns):
                                question_content = content
                            elif any(pattern in label for pattern in answer_patterns):
                                answer_content = content
            
            # ìµœì¢… ?•ë¦¬
            question_content = self._clean_text(question_content)
            answer_content = self._clean_text(answer_content)
            
            return question_content, answer_content
            
        except Exception as e:
            logger.error(f"HTML ë³¸ë¬¸ ì¶”ì¶œ ?¤íŒ¨: {e}")
            return "", ""
    
    def _extract_text_from_element(self, element) -> str:
        """?”ì†Œ?ì„œ ?ìŠ¤??ì¶”ì¶œ"""
        try:
            # ?”ì†Œ??ëª¨ë“  ?ìŠ¤??ì¶”ì¶œ
            text = element.get_text(separator=' ', strip=True)
            
            # ?¤ìŒ ?•ì œ ?”ì†Œ?¤ë„ ?•ì¸
            next_sibling = element.find_next_sibling()
            if next_sibling:
                next_text = next_sibling.get_text(separator=' ', strip=True)
                if next_text and len(next_text) > len(text):
                    text = next_text
            
            return text
        except:
            return ""
    
    def _clean_text(self, text: str) -> str:
        """?ìŠ¤???•ë¦¬"""
        if not text:
            return ""
        
        # ë¶ˆí•„?”í•œ ê³µë°± ?œê±°
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # ?¹ìˆ˜ ë¬¸ì ?•ë¦¬
        text = re.sub(r'[^\w\sê°€-??,!?()\[\]{}:";]', '', text)
        
        return text
    
    def _collect_interpretation_detail(self, interpretation: Dict[str, Any]) -> Dict[str, Any]:
        """ë²•ë ¹?´ì„ë¡€ ?ì„¸ ?•ë³´ ?˜ì§‘ (ë³¸ë¬¸ ?¬í•¨)"""
        try:
            # ê¸°ë³¸ ?•ë³´ ë³µì‚¬
            detailed_data = interpretation.copy()
            
            # ?´ì„ë¡€ ID ì¶”ì¶œ (?ì„¸ ì¡°íšŒ?©ìœ¼ë¡œëŠ” ë²•ë ¹?´ì„ë¡€?¼ë ¨ë²ˆí˜¸ ?¬ìš©)
            interpretation_id = (interpretation.get('ë²•ë ¹?´ì„ë¡€?¼ë ¨ë²ˆí˜¸') or
                               interpretation.get('?¼ë ¨ë²ˆí˜¸') or
                               interpretation.get('id'))
            
            if not interpretation_id:
                logger.warning("?´ì„ë¡€ IDê°€ ?†ì–´ ?ì„¸ ?•ë³´ë¥??˜ì§‘?????†ìŠµ?ˆë‹¤")
                return detailed_data
            
            logger.debug(f"?´ì„ë¡€ ?ì„¸ ?•ë³´ ?˜ì§‘ ì¤? ID={interpretation_id}")
            
            # JSON ?•ì‹?¼ë¡œ ?ì„¸ ?•ë³´ ì¡°íšŒ (ë³¸ë¬¸ ì¶”ì¶œ??
            detail_json = self.client.get_interpretation_detail(str(interpretation_id))
            if detail_json and isinstance(detail_json, dict):
                # JSON ?ì„¸ ?•ë³´ë¥?ê¸°ë³¸ ?°ì´?°ì— ë³‘í•©
                detailed_data.update(detail_json)
                
                # JSON êµ¬ì¡° ë¡œê¹… (?”ë²„ê¹…ìš©)
                logger.debug(f"JSON ?‘ë‹µ ?„ë“œ: {list(detail_json.keys())}")
                
                # JSON?ì„œ ë³¸ë¬¸ ?´ìš© ì¶”ì¶œ
                question_content, answer_content = self._extract_content_from_json(detail_json)
                detailed_data['question_content'] = question_content
                detailed_data['answer_content'] = answer_content
                
                logger.debug(f"JSON ?ì„¸ ?•ë³´ ?˜ì§‘ ?„ë£Œ: ID={interpretation_id}")
                logger.debug(f"ì§ˆì˜?´ìš© ê¸¸ì´: {len(question_content)}, ?Œì‹ ?´ìš© ê¸¸ì´: {len(answer_content)}")
                
                # ì§ˆì˜?´ìš©ê³??Œì‹ ?´ìš©???†ìœ¼ë©?JSON êµ¬ì¡° ì¶œë ¥
                if not question_content and not answer_content:
                    logger.debug(f"ë³¸ë¬¸ ì¶”ì¶œ ?¤íŒ¨ - JSON êµ¬ì¡°: {str(detail_json)[:500]}...")
            else:
                logger.debug(f"JSON ?ì„¸ ?•ë³´ ?˜ì§‘ ?¤íŒ¨: ID={interpretation_id}")
            
            # HTML ?•ì‹?€ ?ëŸ¬ ?˜ì´ì§€ê°€ ë°˜í™˜?˜ë?ë¡?ê±´ë„ˆ?°ê¸°
            # detail_html = self.client.get_interpretation_detail_html(str(interpretation_id))
            # if detail_html and not detail_html.strip().startswith('<!DOCTYPE html'):
            #     detailed_data['html_content'] = detail_html
            #     logger.debug(f"HTML ?ì„¸ ?•ë³´ ?˜ì§‘ ?„ë£Œ: ID={interpretation_id}")
            # else:
            #     logger.debug(f"HTML ?ì„¸ ?•ë³´ ?˜ì§‘ ?¤íŒ¨: ID={interpretation_id}")
            
            # ê¸°ë³¸ ?•ë³´???ì„¸ ë§í¬ ì¶”ê?
            detail_url = interpretation.get('ë²•ë ¹?´ì„ë¡€?ì„¸ë§í¬', '')
            if detail_url:
                detailed_data['detail_url'] = f"http://www.law.go.kr{detail_url}"
            
            # ì§ˆì˜ê¸°ê?, ?Œì‹ ê¸°ê? ?•ë³´ ì¶”ê?
            detailed_data['inquiry_agency'] = interpretation.get('ì§ˆì˜ê¸°ê?ëª?, '')
            detailed_data['reply_agency'] = interpretation.get('?Œì‹ ê¸°ê?ëª?, '')
            
            # API ?”ì²­ ê°?ì§€??
            self._random_delay()
            
            return detailed_data
            
        except Exception as e:
            logger.error(f"?´ì„ë¡€ ?ì„¸ ?•ë³´ ?˜ì§‘ ?¤íŒ¨ (ID: {interpretation_id}): {e}")
            return interpretation  # ?¤íŒ¨ ??ê¸°ë³¸ ?•ë³´ë§?ë°˜í™˜
    
    def _check_memory_usage(self) -> bool:
        """ë©”ëª¨ë¦??¬ìš©???•ì¸ ë°?ê´€ë¦?""
        if not PSUTIL_AVAILABLE:
            # psutil???†ëŠ” ê²½ìš° ê°€ë¹„ì? ì»¬ë ‰?˜ë§Œ ?¤í–‰
            if len(self.pending_interpretations) > self.batch_size:
                logger.info("psutil ?†ì´ ë°°ì¹˜ ?€???¤í–‰")
                self._save_batch_interpretations()
                return True
            return False
            
        try:
            # ?„ì¬ ?„ë¡œ?¸ìŠ¤??ë©”ëª¨ë¦??¬ìš©???•ì¸
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # ë©”ëª¨ë¦??¬ìš©ë¥ ì´ ?„ê³„ê°’ì„ ì´ˆê³¼??ê²½ìš°
            if memory_percent > self.max_memory_usage:
                logger.warning(f"ë©”ëª¨ë¦??¬ìš©ë¥ ì´ ?’ìŠµ?ˆë‹¤: {memory_percent:.1f}% (?„ê³„ê°? {self.max_memory_usage}%)")
                logger.warning(f"RSS ë©”ëª¨ë¦? {memory_info.rss / 1024 / 1024:.1f}MB")
                
                # ê°€ë¹„ì? ì»¬ë ‰??ê°•ì œ ?¤í–‰
                logger.info("ê°€ë¹„ì? ì»¬ë ‰???¤í–‰ ì¤?..")
                collected = gc.collect()
                logger.info(f"ê°€ë¹„ì? ì»¬ë ‰???„ë£Œ: {collected}ê°?ê°ì²´ ?•ë¦¬")
                
                # ë©”ëª¨ë¦??¬ìš©???¬í™•??
                memory_percent_after = process.memory_percent()
                logger.info(f"ê°€ë¹„ì? ì»¬ë ‰????ë©”ëª¨ë¦??¬ìš©ë¥? {memory_percent_after:.1f}%")
                
                # ?¬ì „???’ì? ê²½ìš° ë°°ì¹˜ ?€??ê°•ì œ ?¤í–‰
                if memory_percent_after > self.max_memory_usage and len(self.pending_interpretations) > 0:
                    logger.warning("ë©”ëª¨ë¦??¬ìš©ë¥ ì´ ?¬ì „???’ì•„ ë°°ì¹˜ ?€?¥ì„ ê°•ì œ ?¤í–‰?©ë‹ˆ??)
                    self._save_batch_interpretations()
                    return True
                
            return False
            
        except Exception as e:
            logger.error(f"ë©”ëª¨ë¦??¬ìš©???•ì¸ ì¤??¤ë¥˜: {e}")
            return False
    
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
            "legal_interpretation_*.json",
            "batch_*.json", 
            "checkpoints/**/*.json",
            "*.json"
        ]
        
        for pattern in file_patterns:
            files = list(self.output_dir.glob(pattern))
            for file_path in files:
                try:
                    loaded_count += self._load_interpretations_from_file(file_path)
                except Exception as e:
                    error_count += 1
                    logger.debug(f"?Œì¼ ë¡œë“œ ?¤íŒ¨ {file_path}: {e}")
        
        logger.info(f"ê¸°ì¡´ ?°ì´??ë¡œë“œ ?„ë£Œ: {loaded_count:,}ê±? ?¤ë¥˜: {error_count:,}ê±?)
        self.stats.collected_count = len(self.collected_interpretations)
        logger.info(f"ì¤‘ë³µ ë°©ì?ë¥??„í•œ ?´ì„ë¡€ ID {len(self.collected_interpretations):,}ê°?ë¡œë“œ??)
    
    def _load_interpretations_from_file(self, file_path: Path) -> int:
        """?Œì¼?ì„œ ë²•ë ¹?´ì„ë¡€ ?°ì´??ë¡œë“œ"""
        loaded_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # ?¤ì–‘???°ì´??êµ¬ì¡° ì²˜ë¦¬
        interpretations = []
        
        if isinstance(data, dict):
            if 'interpretations' in data:
                interpretations = data['interpretations']
            elif 'basic_info' in data:
                interpretations = [data]
            elif 'by_category' in data:
                for category_data in data['by_category'].values():
                    interpretations.extend(category_data)
        elif isinstance(data, list):
            interpretations = data
        
        # ?´ì„ë¡€ ID ì¶”ì¶œ
        for interpretation in interpretations:
            if isinstance(interpretation, dict):
                interpretation_id = interpretation.get('?ë??¼ë ¨ë²ˆí˜¸') or interpretation.get('interpretation_id')
                if interpretation_id:
                    self.collected_interpretations.add(str(interpretation_id))
                    loaded_count += 1
        
        return loaded_count
    
    def _is_duplicate_interpretation(self, interpretation: Dict[str, Any]) -> bool:
        """ë²•ë ¹?´ì„ë¡€ ì¤‘ë³µ ?¬ë? ?•ì¸"""
        interpretation_id = (interpretation.get('id') or
                           interpretation.get('ë²•ë ¹?´ì„ë¡€?¼ë ¨ë²ˆí˜¸') or
                           interpretation.get('?ë??¼ë ¨ë²ˆí˜¸') or 
                           interpretation.get('interpretation_id') or
                           interpretation.get('?¼ë ¨ë²ˆí˜¸'))
        
        if interpretation_id and str(interpretation_id).strip() != '':
            if str(interpretation_id) in self.collected_interpretations:
                logger.debug(f"?´ì„ë¡€?¼ë ¨ë²ˆí˜¸ë¡?ì¤‘ë³µ ?•ì¸: {interpretation_id}")
                return True
        
        return False
    
    def _mark_interpretation_collected(self, interpretation: Dict[str, Any]):
        """ë²•ë ¹?´ì„ë¡€ë¥??˜ì§‘?¨ìœ¼ë¡??œì‹œ"""
        interpretation_id = (interpretation.get('id') or
                           interpretation.get('ë²•ë ¹?´ì„ë¡€?¼ë ¨ë²ˆí˜¸') or
                           interpretation.get('?ë??¼ë ¨ë²ˆí˜¸') or 
                           interpretation.get('interpretation_id') or
                           interpretation.get('?¼ë ¨ë²ˆí˜¸'))
        
        if interpretation_id and str(interpretation_id).strip() != '':
            self.collected_interpretations.add(str(interpretation_id))
            logger.debug(f"?´ì„ë¡€?¼ë ¨ë²ˆí˜¸ë¡??€?? {interpretation_id}")
    
    def _validate_interpretation_data(self, interpretation: Dict[str, Any]) -> bool:
        """ë²•ë ¹?´ì„ë¡€ ?°ì´??ê²€ì¦?""
        # ë²•ë ¹?´ì„ë¡€ API ?‘ë‹µ ?„ë“œëª??•ì¸ (?¤ì œ API ?‘ë‹µ ê¸°ì?)
        interpretation_id = (interpretation.get('id') or
                           interpretation.get('ë²•ë ¹?´ì„ë¡€?¼ë ¨ë²ˆí˜¸') or
                           interpretation.get('?ë??¼ë ¨ë²ˆí˜¸') or 
                           interpretation.get('interpretation_id') or
                           interpretation.get('?¼ë ¨ë²ˆí˜¸'))
        case_name = (interpretation.get(' ?ˆê±´ëª?) or  # ?¤ì œ API ?‘ë‹µ ?„ë“œëª?(?ì— ê³µë°±)
                    interpretation.get('?ˆê±´ëª?) or
                    interpretation.get('?¬ê±´ëª?) or 
                    interpretation.get('case_name') or
                    interpretation.get('?œëª©') or
                    interpretation.get('title'))
        
        if not interpretation_id or str(interpretation_id).strip() == '':
            if not case_name:
                logger.warning(f"ë²•ë ¹?´ì„ë¡€ ?ë³„ ?•ë³´ ë¶€ì¡?- ?´ì„ë¡€ID: {interpretation_id}, ?¬ê±´ëª? {case_name}")
                logger.warning(f"?¬ìš© ê°€?¥í•œ ?„ë“œ: {list(interpretation.keys())}")
                logger.warning(f"?„ì²´ ?°ì´?? {str(interpretation)[:200]}...")
                return False
            logger.debug(f"?´ì„ë¡€?¼ë ¨ë²ˆí˜¸ ?†ìŒ, ?¬ê±´ëª…ìœ¼ë¡??€ì²? {case_name}")
        elif not case_name:
            logger.warning(f"?¬ê±´ëª…ì´ ?†ìŠµ?ˆë‹¤: {str(interpretation)[:200]}...")
            return False
        
        return True
    
    def _create_interpretation_data(self, raw_data: Dict[str, Any]) -> Optional[LegalInterpretationData]:
        """?ì‹œ ?°ì´?°ì—??LegalInterpretationData ê°ì²´ ?ì„±"""
        try:
            # ?°ì´??ê²€ì¦?
            if not self._validate_interpretation_data(raw_data):
                return None
            
            # ?´ì„ë¡€ ID ì¶”ì¶œ (?¤ì œ API ?‘ë‹µ ?„ë“œëª??°ì„ )
            interpretation_id = (raw_data.get('id') or
                               raw_data.get('ë²•ë ¹?´ì„ë¡€?¼ë ¨ë²ˆí˜¸') or
                               raw_data.get('?ë??¼ë ¨ë²ˆí˜¸') or 
                               raw_data.get('interpretation_id') or
                               raw_data.get('?¼ë ¨ë²ˆí˜¸'))
            
            # ?´ì„ë¡€ IDê°€ ?†ëŠ” ê²½ìš° ?€ì²?ID ?ì„±
            if not interpretation_id or str(interpretation_id).strip() == '':
                case_name = (raw_data.get(' ?ˆê±´ëª?, '') or  # ?¤ì œ API ?‘ë‹µ ?„ë“œëª?(?ì— ê³µë°±)
                            raw_data.get('?ˆê±´ëª?, '') or
                            raw_data.get('?¬ê±´ëª?, '') or 
                            raw_data.get('case_name', '') or
                            raw_data.get('?œëª©', '') or
                            raw_data.get('title', ''))
                if case_name:
                    interpretation_id = f"interpretation_{case_name}"
                else:
                    interpretation_id = f"interpretation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                logger.debug(f"?€ì²?ID ?ì„±: {interpretation_id}")
            
            # ì¹´í…Œê³ ë¦¬ ë¶„ë¥˜
            category = self.categorize_interpretation(raw_data)
            
            # ì£¼ì œ ë¶„ë¥˜
            topic = self.classify_interpretation_topic(raw_data)
            
            # ë¶€ì²?ë¶„ë¥˜
            ministry = self.classify_interpretation_ministry(raw_data)
            
            # LegalInterpretationData ê°ì²´ ?ì„± (?¤ì œ API ?‘ë‹µ ?„ë“œëª??¬ìš©)
            interpretation_data = LegalInterpretationData(
                interpretation_id=str(interpretation_id),
                case_name=(raw_data.get(' ?ˆê±´ëª?, '') or  # ?¤ì œ API ?‘ë‹µ ?„ë“œëª?(?ì— ê³µë°±)
                          raw_data.get('?ˆê±´ëª?, '') or
                          raw_data.get('?¬ê±´ëª?, '') or 
                          raw_data.get('case_name', '') or
                          raw_data.get('?œëª©', '') or
                          raw_data.get('title', '')),
                case_number=(raw_data.get('?ˆê±´ë²ˆí˜¸', '') or
                           raw_data.get('?¬ê±´ë²ˆí˜¸', '') or 
                           raw_data.get('case_number', '') or
                           raw_data.get('ë²ˆí˜¸', '')),
                ministry=ministry,
                decision_date=(raw_data.get('?Œì‹ ?¼ì', '') or
                             raw_data.get('?ê²°?¼ì', '') or 
                             raw_data.get('? ê³ ?¼ì', '') or
                             raw_data.get('decision_date', '') or
                             raw_data.get('?¼ì', '')),
                category=category,
                topic=topic,
                raw_data=raw_data,
                # ?ì„¸ ?•ë³´ ?„ë“œ??
                question_content=raw_data.get('ì§ˆì˜?´ìš©', '') or raw_data.get('question', ''),
                answer_content=raw_data.get('?Œì‹ ?´ìš©', '') or raw_data.get('answer', ''),
                related_laws=raw_data.get('ê´€?¨ë²•??, '') or raw_data.get('related_laws', ''),
                html_content=raw_data.get('html_content', ''),
                detail_url=raw_data.get('ë²•ë ¹?´ì„ë¡€?ì„¸ë§í¬', '') or raw_data.get('detail_url', ''),
                inquiry_agency=raw_data.get('ì§ˆì˜ê¸°ê?ëª?, '') or raw_data.get('inquiry_agency', ''),
                reply_agency=raw_data.get('?Œì‹ ê¸°ê?ëª?, '') or raw_data.get('reply_agency', '')
            )
            
            return interpretation_data
            
        except Exception as e:
            logger.error(f"LegalInterpretationData ?ì„± ?¤íŒ¨: {e}")
            logger.error(f"?ì‹œ ?°ì´?? {raw_data}")
            return None
    
    def categorize_interpretation(self, interpretation: Dict[str, Any]) -> InterpretationCategory:
        """ë²•ë ¹?´ì„ë¡€ ì¹´í…Œê³ ë¦¬ ë¶„ë¥˜"""
        case_name = interpretation.get('?¬ê±´ëª?, '').lower()
        case_content = interpretation.get('?ì‹œ?¬í•­', '') + ' ' + interpretation.get('?ê²°?”ì?', '')
        case_content = case_content.lower()
        
        # ?¤ì›Œ??ê¸°ë°˜ ë¶„ë¥˜
        category_keywords = {
            InterpretationCategory.ADMINISTRATIVE: [
                '?‰ì •ì²˜ë¶„', '?ˆê?', '?¸ê?', '? ê³ ', '?‰ì •?¬íŒ', '?‰ì •?Œì†¡', 'êµ?„¸', 'ì§€ë°©ì„¸'
            ],
            InterpretationCategory.CIVIL: [
                'ê³„ì•½', '?í•´ë°°ìƒ', '?Œìœ ê¶?, 'ë¬¼ê¶Œ', 'ì±„ê¶Œ', '?ì†', '?¼ì¸', '?´í˜¼'
            ],
            InterpretationCategory.CRIMINAL: [
                '?ˆë„', 'ê°•ë„', '?¬ê¸°', '?´ì¸', '?í•´', 'êµí†µ?¬ê³ ', '?Œì£¼?´ì „'
            ],
            InterpretationCategory.COMMERCIAL: [
                '?Œì‚¬', 'ì£¼ì‹', '?´ìŒ', '?˜í‘œ', 'ë³´í—˜', '?í–‰??
            ],
            InterpretationCategory.LABOR: [
                'ê·¼ë¡œê³„ì•½', '?„ê¸ˆ', '?´ê³ ', '?¸ë™ì¡°í•©', '?°ì—…?¬í•´'
            ],
            InterpretationCategory.INTELLECTUAL_PROPERTY: [
                '?¹í—ˆ', '?€?‘ê¶Œ', '?í‘œ', '?ì—…ë¹„ë?', 'ë¶€?•ê²½??
            ],
            InterpretationCategory.CONSUMER: [
                '?Œë¹„??, 'ê³„ì•½', '?½ê?', '?œì‹œê´‘ê³ ', '? ë?ê±°ë˜'
            ],
            InterpretationCategory.ENVIRONMENT: [
                '?˜ê²½', '?€ê¸?, '?˜ì§ˆ', '?ê¸°ë¬?, '?˜ê²½?í–¥?‰ê?'
            ],
            InterpretationCategory.FINANCE: [
                'ê¸ˆìœµ', '?€??, 'ë³´í—˜', 'ì¦ê¶Œ', '?ë³¸?œì¥'
            ],
            InterpretationCategory.INFORMATION_TECHNOLOGY: [
                '?•ë³´?µì‹ ', '?„ìê±°ë˜', 'ê°œì¸?•ë³´', '?¬ì´ë²„ë³´??
            ]
        }
        
        # ?¤ì›Œ??ë§¤ì¹­?¼ë¡œ ì¹´í…Œê³ ë¦¬ ê²°ì •
        for category, keywords in category_keywords.items():
            if any(keyword in case_name or keyword in case_content for keyword in keywords):
                return category
        
        return InterpretationCategory.OTHER
    
    def classify_interpretation_topic(self, interpretation: Dict[str, Any]) -> str:
        """ë²•ë ¹?´ì„ë¡€ ì£¼ì œ ë¶„ë¥˜"""
        case_name = interpretation.get('?¬ê±´ëª?, '').lower()
        case_content = interpretation.get('?ì‹œ?¬í•­', '') + ' ' + interpretation.get('?ê²°?”ì?', '')
        case_content = case_content.lower()
        
        # ì£¼ì œë³??¤ì›Œ??ë§¤ì¹­
        for topic, keywords in INTERPRETATION_TOPIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in case_name or keyword in case_content:
                    return topic
        
        return "ê¸°í?"
    
    def classify_interpretation_ministry(self, interpretation: Dict[str, Any]) -> str:
        """ë²•ë ¹?´ì„ë¡€ ë¶€ì²?ë¶„ë¥˜"""
        case_name = interpretation.get('?¬ê±´ëª?, '').lower()
        case_content = interpretation.get('?ì‹œ?¬í•­', '') + ' ' + interpretation.get('?ê²°?”ì?', '')
        case_content = case_content.lower()
        
        # ë¶€ì²˜ë³„ ?¤ì›Œ??ë§¤ì¹­
        for ministry, keywords in MINISTRY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in case_name or keyword in case_content:
                    return ministry
        
        return "ê¸°í?"
    
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
    
    def collect_all_interpretations(self, target_count: int = 2000):
        """ëª¨ë“  ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ (ìµœì‹  ?ë? ?°ì„  ?˜ì§‘)"""
        self.stats.target_count = target_count
        self.stats.status = CollectionStatus.IN_PROGRESS
        
        # ì²´í¬?¬ì¸???Œì¼ ?¤ì •
        checkpoint_file = self._setup_checkpoint_file()
        
        try:
            logger.info(f"ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ ?œì‘ - ëª©í‘œ: {target_count}ê±?)
            logger.info("ìµœì‹  ?ë?ë¶€???˜ì§‘?˜ëŠ” ë°©ì‹?¼ë¡œ ë³€ê²½ë¨")
            logger.info("Graceful shutdown ì§€?? Ctrl+C ?ëŠ” SIGTERM?¼ë¡œ ?ˆì „?˜ê²Œ ì¤‘ë‹¨ ê°€??)
            logger.info("ì¤‘ë‹¨ ???„ì¬ê¹Œì? ?˜ì§‘???°ì´?°ê? ?ë™?¼ë¡œ ?€?¥ë©?ˆë‹¤")
            
            # ìµœì‹  ?ë?ë¶€???˜ì§‘ (?¤ì›Œ??ê¸°ë°˜???„ë‹Œ ? ì§œ ê¸°ë°˜)
            self._collect_by_date_range(target_count, checkpoint_file)
            
            # ëª©í‘œ ?¬ì„±?˜ì? ëª»í•œ ê²½ìš° ì¶”ê? ?˜ì§‘ ?œë„
            if self.stats.collected_count < target_count:
                remaining_count = target_count - self.stats.collected_count
                logger.info(f"ëª©í‘œ ?¬ì„± ?¤íŒ¨. ì¶”ê? ?˜ì§‘?¼ë¡œ {remaining_count}ê±????˜ì§‘ ?œë„")
                self._collect_additional_interpretations(remaining_count, checkpoint_file)
            
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
            logger.error(f"ë²•ë ¹?´ì„ë¡€ ?˜ì§‘ ì¤??¤ë¥˜ ë°œìƒ: {e}")
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
    
    def _collect_by_date_range(self, target_count: int, checkpoint_file: Path):
        """? ì§œ ë²”ìœ„ ê¸°ë°˜?¼ë¡œ ìµœì‹  ?ë?ë¶€???˜ì§‘"""
        logger.info("ìµœì‹  ?ë?ë¶€??? ì§œ ê¸°ë°˜ ?˜ì§‘ ?œì‘")
        
        # ìµœê·¼ 3?„ê°„???°ì´?°ë????˜ì§‘
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)  # 3????
        
        logger.info(f"?˜ì§‘ ê¸°ê°„: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        # ? ì§œë³„ë¡œ ?˜ì§‘ (ìµœì‹ ë¶€????ˆœ?¼ë¡œ)
        current_date = end_date
        collected_count = 0
        
        while current_date >= start_date and collected_count < target_count:
            if self._check_shutdown_requested():
                logger.warning(f"ì¢…ë£Œ ?”ì²­?¼ë¡œ ? ì§œ ê¸°ë°˜ ?˜ì§‘ ì¤‘ë‹¨: {self.shutdown_reason}")
                break
            
            # ?´ë‹¹ ? ì§œ???ë? ?˜ì§‘
            date_str = current_date.strftime('%Y%m%d')
            remaining_count = target_count - collected_count
            
            logger.info(f"? ì§œ {date_str} ?ë? ?˜ì§‘ ì¤?.. (ëª©í‘œ: {remaining_count}ê±?")
            
            try:
                # ?´ë‹¹ ? ì§œ???ë? ê²€??
                start_time = time.time()
                interpretations = self.collect_interpretations_by_date(date_str, min(100, remaining_count))
                elapsed_time = time.time() - start_time
                
                if interpretations:
                    # ?˜ì§‘???´ì„ë¡€ë¥??„ì‹œ ?€?¥ì†Œ??ì¶”ê?
                    for interpretation in interpretations:
                        self.pending_interpretations.append(interpretation)
                        self.stats.collected_count += 1
                        collected_count += 1
                        
                        if collected_count >= target_count:
                            break
                    
                    # ë°°ì¹˜ ?€??
                    if len(self.pending_interpretations) >= self.batch_size:
                        logger.info(f"ë°°ì¹˜ ?¬ê¸° ?„ë‹¬ë¡??€???¤í–‰ (?€ê¸?ì¤? {len(self.pending_interpretations)}ê±?")
                        self._save_batch_interpretations()
                    
                    progress_percent = (collected_count / target_count) * 100
                    logger.info(f"? ì§œ {date_str}: {len(interpretations)}ê±??˜ì§‘ (?„ì : {collected_count}/{target_count}ê±? {progress_percent:.1f}%, ?Œìš”?œê°„: {elapsed_time:.1f}ì´?")
                else:
                    logger.debug(f"? ì§œ {date_str}: ?˜ì§‘???ë? ?†ìŒ (?Œìš”?œê°„: {elapsed_time:.1f}ì´?")
                
                # ì²´í¬?¬ì¸???€??
                self._save_checkpoint(checkpoint_file)
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                if self._check_api_limits():
                    logger.warning("API ?”ì²­ ?œí•œ???„ë‹¬?˜ì—¬ ? ì§œ ê¸°ë°˜ ?˜ì§‘ ì¤‘ë‹¨")
                    break
                
            except Exception as e:
                logger.error(f"? ì§œ {date_str} ?˜ì§‘ ì¤??¤ë¥˜: {e}")
                continue
            
            # ?¤ìŒ ? ë¡œ ?´ë™
            current_date -= timedelta(days=1)
            
            # API ?”ì²­ ê°?ì§€??
            self._random_delay()
        
        logger.info(f"? ì§œ ê¸°ë°˜ ?˜ì§‘ ?„ë£Œ: {collected_count}ê±??˜ì§‘")
    
    def collect_interpretations_by_date(self, date_str: str, max_count: int = 100) -> List[LegalInterpretationData]:
        """?¹ì • ? ì§œ??ë²•ë ¹?´ì„ë¡€ ê²€??ë°??˜ì§‘"""
        logger.debug(f"? ì§œ {date_str}ë¡?ë²•ë ¹?´ì„ë¡€ ê²€???œì‘ (ëª©í‘œ: {max_count}ê±?")
        
        interpretations = []
        page = 1
        consecutive_empty_pages = 0
        max_empty_pages = 3
        
        while len(interpretations) < max_count and consecutive_empty_pages < max_empty_pages:
            # ì¢…ë£Œ ?”ì²­ ?•ì¸
            if self._check_shutdown_requested():
                logger.warning(f"ì¢…ë£Œ ?”ì²­?¼ë¡œ ? ì§œ {date_str} ê²€??ì¤‘ë‹¨: {self.shutdown_reason}")
                break
            
            try:
                # API ?”ì²­ ê°??œë¤ ì§€??
                if page > 1:
                    self._random_delay()
                
                # ë©”ëª¨ë¦??¬ìš©??ì²´í¬
                self.api_request_count += 1
                if self.api_request_count % self.memory_check_interval == 0:
                    if self._check_memory_usage():
                        logger.info("ë©”ëª¨ë¦?ìµœì ?”ë¡œ ?¸í•œ ë°°ì¹˜ ?€???„ë£Œ")
                
                # API ?”ì²­ ?¤í–‰
                try:
                    with self._api_request_with_retry(f"? ì§œ {date_str} ê²€??):
                        results = self.client.get_interpretation_list(
                            display=100,
                            page=page,
                            from_date=date_str,
                            to_date=date_str
                        )
                except Exception as api_error:
                    logger.error(f"API ?”ì²­ ?¤íŒ¨: {api_error}")
                    consecutive_empty_pages += 1
                    page += 1
                    continue
                
                if not results or len(results) == 0:
                    consecutive_empty_pages += 1
                    logger.debug(f"? ì§œ {date_str} ?˜ì´ì§€ {page}?ì„œ ê²°ê³¼ ?†ìŒ (?°ì† ë¹??˜ì´ì§€: {consecutive_empty_pages})")
                    page += 1
                    continue
                else:
                    consecutive_empty_pages = 0
                    logger.info(f"? ì§œ {date_str} ?˜ì´ì§€ {page}: {len(results)}ê±?ë°œê²¬")
                    
                    # ì²?ë²ˆì§¸ ê²°ê³¼??êµ¬ì¡° ?•ì¸
                    if results and len(results) > 0:
                        first_result = results[0]
                        logger.debug(f"ì²?ë²ˆì§¸ ê²°ê³¼ êµ¬ì¡°: {list(first_result.keys()) if isinstance(first_result, dict) else type(first_result)}")
                        if isinstance(first_result, dict):
                            logger.debug(f"ì²?ë²ˆì§¸ ê²°ê³¼ ?˜í”Œ: {str(first_result)[:200]}...")
                
                # ê²°ê³¼ ì²˜ë¦¬
                new_count, duplicate_count = self._process_search_results(results, interpretations, max_count)
                
                # ?˜ì´ì§€ë³?ê²°ê³¼ ë¡œê¹…
                logger.debug(f"?˜ì´ì§€ {page}: {new_count}ê±?? ê·œ, {duplicate_count}ê±?ì¤‘ë³µ (?„ì : {len(interpretations)}/{max_count}ê±?")
                
                page += 1
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                if self._check_api_limits():
                    break
                    
            except KeyboardInterrupt:
                logger.warning(f"? ì§œ {date_str} ê²€?‰ì´ ì¤‘ë‹¨?˜ì—ˆ?µë‹ˆ??")
                break
            except Exception as e:
                logger.error(f"? ì§œ {date_str} ê²€??ì¤??¤ë¥˜: {e}")
                self.stats.failed_count += 1
                break
        
        # ?˜ì§‘???´ì„ë¡€ë¥??„ì‹œ ?€?¥ì†Œ??ì¶”ê?
        for interpretation in interpretations:
            self.pending_interpretations.append(interpretation)
            self.stats.collected_count += 1
        
        logger.info(f"? ì§œ {date_str} ?˜ì§‘ ?„ë£Œ: {len(interpretations)}ê±?)
        return interpretations
    
    def _collect_additional_interpretations(self, remaining_count: int, checkpoint_file: Path):
        """ì¶”ê? ?˜ì§‘ (?¤ì›Œ??ê¸°ë°˜)"""
        logger.info(f"ì¶”ê? ?˜ì§‘ ?œì‘ - ëª©í‘œ: {remaining_count}ê±?)
        
        # ?°ì„ ?œìœ„ ?¤ì›Œ?œë¡œ ì¶”ê? ?˜ì§‘
        priority_keywords = [kw for kw in PRIORITY_KEYWORDS if kw in INTERPRETATION_KEYWORDS]
        
        for keyword in priority_keywords:
            if self.stats.collected_count >= self.stats.target_count:
                logger.info(f"ëª©í‘œ ?˜ëŸ‰ {self.stats.target_count:,}ê±??¬ì„±?¼ë¡œ ì¶”ê? ?˜ì§‘ ì¤‘ë‹¨")
                break
            
            if self._check_shutdown_requested():
                logger.warning(f"ì¢…ë£Œ ?”ì²­?¼ë¡œ ì¶”ê? ?˜ì§‘ ì¤‘ë‹¨: {self.shutdown_reason}")
                break
            
            try:
                # ?¤ì›Œ?œë³„ ëª©í‘œ ê±´ìˆ˜ (ì¶”ê? ?˜ì§‘?€ ?ê²Œ)
                keyword_target = min(10, remaining_count)
                remaining_count -= keyword_target
                
                logger.info(f"ì¶”ê? ?¤ì›Œ??'{keyword}' ì²˜ë¦¬ ?œì‘ (ëª©í‘œ: {keyword_target}ê±?")
                
                # ?´ì„ë¡€ ?˜ì§‘
                interpretations = self.collect_interpretations_by_keyword(keyword, keyword_target)
                
                # ë°°ì¹˜ ?€??
                if len(self.pending_interpretations) >= self.batch_size:
                    self._save_batch_interpretations()
                
                # ì²´í¬?¬ì¸???€??
                self._save_checkpoint(checkpoint_file)
                
                # ì§„í–‰ ?í™© ë¡œê¹…
                progress_percent = (self.stats.collected_count / self.stats.target_count) * 100
                logger.info(f"ì¶”ê? ?¤ì›Œ??'{keyword}' ?„ë£Œ: {len(interpretations)}ê±??˜ì§‘ (ì´?{self.stats.collected_count:,}ê±? {progress_percent:.1f}%)")
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                if self._check_api_limits():
                    logger.warning("API ?”ì²­ ?œí•œ???„ë‹¬?˜ì—¬ ì¶”ê? ?˜ì§‘ ì¤‘ë‹¨")
                    break
                    
            except Exception as e:
                logger.error(f"ì¶”ê? ?¤ì›Œ??'{keyword}' ?˜ì§‘ ì¤??¤ë¥˜: {e}")
                self.stats.failed_count += 1
                continue
        
        logger.info(f"ì¶”ê? ?˜ì§‘ ?„ë£Œ - ì´?{self.stats.collected_count:,}ê±??˜ì§‘")
    
    def _collect_by_keywords(self, target_count: int, checkpoint_file: Path):
        """?°ì„ ?œìœ„ ê¸°ë°˜ ?¤ì›Œ?œë³„ ë²•ë ¹?´ì„ë¡€ ?˜ì§‘"""
        # ?°ì„ ?œìœ„ ?¤ì›Œ?œì? ?¼ë°˜ ?¤ì›Œ??ë¶„ë¦¬
        priority_keywords = [kw for kw in PRIORITY_KEYWORDS if kw in INTERPRETATION_KEYWORDS]
        remaining_keywords = [kw for kw in INTERPRETATION_KEYWORDS if kw not in PRIORITY_KEYWORDS]
        
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
                interpretations = self.collect_interpretations_by_keyword(keyword, keyword_target)
                
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
    
    def collect_interpretations_by_keyword(self, keyword: str, max_count: int = 50) -> List[LegalInterpretationData]:
        """?¤ì›Œ?œë¡œ ë²•ë ¹?´ì„ë¡€ ê²€??ë°??˜ì§‘"""
        # ?´ë? ì²˜ë¦¬???¤ì›Œ?œì¸ì§€ ?•ì¸
        if keyword in self.processed_keywords:
            logger.info(f"?¤ì›Œ??'{keyword}'???´ë? ì²˜ë¦¬?˜ì—ˆ?µë‹ˆ?? ê±´ë„ˆ?ë‹ˆ??")
            return []
        
        logger.info(f"?¤ì›Œ??'{keyword}'ë¡?ë²•ë ¹?´ì„ë¡€ ê²€???œì‘ (ëª©í‘œ: {max_count}ê±?")
        
        interpretations = []
        page = 1
        consecutive_empty_pages = 0
        max_empty_pages = 3
        
        while len(interpretations) < max_count and consecutive_empty_pages < max_empty_pages:
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
                        results = self.client.get_interpretation_list(
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
                new_count, duplicate_count = self._process_search_results(results, interpretations, max_count)
                
                # ?˜ì´ì§€ë³?ê²°ê³¼ ë¡œê¹…
                logger.debug(f"?˜ì´ì§€ {page}: {new_count}ê±?? ê·œ, {duplicate_count}ê±?ì¤‘ë³µ (?„ì : {len(interpretations)}/{max_count}ê±?")
                
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
        
        # ?˜ì§‘???´ì„ë¡€ë¥??„ì‹œ ?€?¥ì†Œ??ì¶”ê?
        for interpretation in interpretations:
            self.pending_interpretations.append(interpretation)
            self.stats.collected_count += 1
        
        # ?¤ì›Œ??ì²˜ë¦¬ ?„ë£Œ ?œì‹œ
        self.processed_keywords.add(keyword)
        
        logger.info(f"?¤ì›Œ??'{keyword}' ?˜ì§‘ ?„ë£Œ: {len(interpretations)}ê±?)
        return interpretations
    
    def _process_search_results(self, results: List[Dict[str, Any]], interpretations: List[LegalInterpretationData], 
                              max_count: int) -> Tuple[int, int]:
        """ê²€??ê²°ê³¼ ì²˜ë¦¬"""
        new_count = 0
        duplicate_count = 0
        
        for result in results:
            # ì¢…ë£Œ ?”ì²­ ?•ì¸
            if self._check_shutdown_requested():
                logger.warning(f"ì¢…ë£Œ ?”ì²­?¼ë¡œ ê²°ê³¼ ì²˜ë¦¬ ì¤‘ë‹¨: {self.shutdown_reason}")
                break
            
            # resultê°€ ë©”í??°ì´?°ì¸ ê²½ìš° expc ë°°ì—´?ì„œ ?¤ì œ ?°ì´??ì¶”ì¶œ
            if 'expc' in result and isinstance(result['expc'], list):
                expc_items = result['expc']
                logger.debug(f"expc ë°°ì—´?ì„œ {len(expc_items)}ê°???ª© ì²˜ë¦¬")
                
                for expc_item in expc_items:
                    if self._check_shutdown_requested():
                        break
                    
                    # ì¤‘ë³µ ?•ì¸
                    if self._is_duplicate_interpretation(expc_item):
                        duplicate_count += 1
                        self.stats.duplicate_count += 1
                        continue
                    
                    # ?ì„¸ ?•ë³´ ?˜ì§‘
                    logger.debug(f"?´ì„ë¡€ {new_count + 1} ?ì„¸ ?•ë³´ ?˜ì§‘ ì¤?..")
                    detailed_data = self._collect_interpretation_detail(expc_item)
                    
                    # LegalInterpretationData ê°ì²´ ?ì„±
                    interpretation_data = self._create_interpretation_data(detailed_data)
                    if not interpretation_data:
                        logger.warning(f"?´ì„ë¡€ ?°ì´???ì„± ?¤íŒ¨")
                        self.stats.failed_count += 1
                        continue
                    
                    # ? ê·œ ?´ì„ë¡€ ì¶”ê?
                    interpretations.append(interpretation_data)
                    self._mark_interpretation_collected(expc_item)
                    new_count += 1
                    
                    # ì§„í–‰ ?í™© ë¡œê¹… (5ê±´ë§ˆ??
                    if new_count % 5 == 0:
                        logger.info(f"ì§„í–‰ ?í™©: {new_count}ê±?ì²˜ë¦¬ ?„ë£Œ (?„ì : {len(interpretations)}/{max_count}ê±?")
                    
                    if len(interpretations) >= max_count:
                        logger.info(f"ëª©í‘œ ?˜ëŸ‰ {max_count}ê±??¬ì„±?¼ë¡œ ì²˜ë¦¬ ì¤‘ë‹¨")
                        break
            else:
                # resultê°€ ì§ì ‘ ë²•ë ¹?´ì„ë¡€ ?°ì´?°ì¸ ê²½ìš° (ê¸°ì¡´ ë¡œì§)
                # ì¤‘ë³µ ?•ì¸
                if self._is_duplicate_interpretation(result):
                    duplicate_count += 1
                    self.stats.duplicate_count += 1
                    continue
                
                # LegalInterpretationData ê°ì²´ ?ì„±
                interpretation_data = self._create_interpretation_data(result)
                if not interpretation_data:
                    self.stats.failed_count += 1
                    continue
                
                # ? ê·œ ?´ì„ë¡€ ì¶”ê?
                interpretations.append(interpretation_data)
                self._mark_interpretation_collected(result)
                new_count += 1
                
                if len(interpretations) >= max_count:
                    break
        
        return new_count, duplicate_count
    
    def _save_batch_interpretations(self):
        """ë°°ì¹˜ ?¨ìœ„ë¡?ë²•ë ¹?´ì„ë¡€ ?€??""
        if not self.pending_interpretations:
            return
        
        try:
            # ì¹´í…Œê³ ë¦¬ë³„ë¡œ ê·¸ë£¹??
            by_category = {}
            for interpretation in self.pending_interpretations:
                category = interpretation.category.value
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(interpretation)
            
            # ë°°ì¹˜ ?Œì¼ ?€??(ì¹´í…Œê³ ë¦¬ë³„ë¡œ ë¶„ë¦¬)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            batch_number = self._get_next_batch_number()
            saved_files = []
            
            for category, interpretations in by_category.items():
                # ?ˆì „???Œì¼ëª??ì„±
                safe_category = category.replace('_', '-')
                filename = f"batch_{batch_number:03d}_{safe_category}_{len(interpretations)}ê±?{timestamp}.json"
                filepath = self.batch_dir / filename
                
                batch_data = {
                    'metadata': {
                        'category': category,
                        'count': len(interpretations),
                        'saved_at': datetime.now().isoformat(),
                        'batch_id': timestamp,
                        'batch_number': batch_number,
                        'total_collected': self.stats.collected_count,
                        'api_requests': self.api_request_count,
                        'collection_progress': {
                            'target_count': self.stats.target_count,
                            'completion_percentage': (self.stats.collected_count / self.stats.target_count * 100) if self.stats.target_count > 0 else 0
                        }
                    },
                    'interpretations': [i.raw_data for i in interpretations]
                }
                
                with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
                    json.dump(batch_data, f, ensure_ascii=False, indent=2)
                
                saved_files.append(filepath)
                logger.info(f"?“ ë°°ì¹˜ ?€???„ë£Œ: {category} ì¹´í…Œê³ ë¦¬ {len(interpretations):,}ê±?-> {filename}")
            
            # ?µê³„ ?…ë°?´íŠ¸
            self.stats.saved_count += len(self.pending_interpretations)
            
            # ?„ì‹œ ?€?¥ì†Œ ì´ˆê¸°??
            self.pending_interpretations = []
            
            # ë°°ì¹˜ ?€???”ì•½ ë¡œê¹…
            total_saved = sum(len(interpretations) for interpretations in by_category.values())
            logger.info(f"??ë°°ì¹˜ ?€???„ë£Œ: ì´?{len(saved_files):,}ê°??Œì¼, {total_saved:,}ê±??€??)
            logger.info(f"?“Š ?„ì  ?˜ì§‘: {self.stats.collected_count:,}ê±?/ {self.stats.target_count:,}ê±?({self.stats.collected_count/self.stats.target_count*100:.1f}%)")
            logger.info(f"?“‚ ?€???„ì¹˜: {self.batch_dir}")
            
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
                    'collected_interpretations_count': len(self.collected_interpretations)
                },
                'interpretations': [i.raw_data for i in self.pending_interpretations],
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
        interpretations = checkpoint_data.get('interpretations', [])
        
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
        
        # ?˜ì§‘???´ì„ë¡€ ID ë³µêµ¬
        for interpretation in interpretations:
            if isinstance(interpretation, dict):
                interpretation_id = interpretation.get('?ë??¼ë ¨ë²ˆí˜¸') or interpretation.get('interpretation_id')
                if interpretation_id:
                    self.collected_interpretations.add(str(interpretation_id))
    
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
                # ë°±ì—… ?¤ì›Œ?œëŠ” ê°ê° 10ê±´ì”© ?˜ì§‘
                keyword_target = min(10, remaining_count)
                remaining_count -= keyword_target
                
                logger.info(f"ë°±ì—… ?¤ì›Œ??'{keyword}' ì²˜ë¦¬ ?œì‘ (ëª©í‘œ: {keyword_target}ê±?")
                
                # ?´ì„ë¡€ ?˜ì§‘
                interpretations = self.collect_interpretations_by_keyword(keyword, keyword_target)
                
                # ë°°ì¹˜ ?€??
                if len(self.pending_interpretations) >= self.batch_size:
                    self._save_batch_interpretations()
                
                # ì²´í¬?¬ì¸???€??
                self._save_checkpoint(checkpoint_file)
                
                # ì§„í–‰ ?í™© ë¡œê¹…
                progress_percent = (self.stats.collected_count / self.stats.target_count) * 100
                logger.info(f"ë°±ì—… ?¤ì›Œ??'{keyword}' ?„ë£Œ: {len(interpretations)}ê±??˜ì§‘ (ì´?{self.stats.collected_count:,}ê±? {progress_percent:.1f}%)")
                
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
        if len(self.pending_interpretations) >= self.batch_size:
            self._save_batch_interpretations()
