#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?‰ì •?¬íŒë¡€ ?˜ì§‘ ?¤í¬ë¦½íŠ¸

êµ??ë²•ë ¹?•ë³´?¼í„° LAW OPEN APIë¥??¬ìš©?˜ì—¬ ?‰ì •?¬íŒë¡€ë¥??˜ì§‘?©ë‹ˆ??
- ìµœê·¼ 3?„ê°„ ?‰ì •?¬íŒë¡€ 1,000ê±??˜ì§‘
- ?¬íŒ ? í˜•ë³?ë¶„ë¥˜ ë°?ë©”í??°ì´???•ì œ
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient, LawOpenAPIConfig

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/collect_administrative_appeals.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ?‰ì •?¬íŒ ê´€??ê²€???¤ì›Œ??
ADMINISTRATIVE_APPEAL_KEYWORDS = [
    # ?‰ì •ì²˜ë¶„ ê´€??
    "?‰ì •ì²˜ë¶„", "?ˆê?", "?¸ê?", "? ê³ ", "? ì²­", "?´ì˜? ì²­", "ì·¨ì†Œì²˜ë¶„", "?•ì?ì²˜ë¶„",
    "ê³¼íƒœë£?, "ê³¼ì§•ê¸?, "ë¶€ê³¼ì²˜ë¶?, "ì§•ê³„ì²˜ë¶„", "ë©´í—ˆì·¨ì†Œ", "?ˆê?ì·¨ì†Œ",
    
    # êµ?„¸ ê´€??
    "êµ?„¸", "ì§€ë°©ì„¸", "?¸ë¬´ì¡°ì‚¬", "ê°€?°ì„¸", "ê°€?°ê¸ˆ", "ì²´ë‚©ì²˜ë¶„", "?•ë¥˜ì²˜ë¶„",
    "?¸ë¬´??, "êµ?„¸ì²?, "ì§€ë°©ì„¸ì²?, "?¸ë¬´ì¡°ì •", "?¸ë¬´?¬ì‚¬",
    
    # ê±´ì¶• ê´€??
    "ê±´ì¶•?ˆê?", "ê±´ì¶•? ê³ ", "ê±´ì¶•ë²?, "ê±´ì¶•ë¬?, "ê±´ì¶•ê³„íš", "ê±´ì¶•?¬ì˜",
    "?©ë„ë³€ê²?, "ì¦ì¶•", "ê°œì¶•", "?¬ê±´ì¶?, "ì² ê±°ëª…ë ¹",
    
    # ?˜ê²½ ê´€??
    "?˜ê²½?í–¥?‰ê?", "?˜ê²½?¤ì—¼", "?€ê¸°ì˜¤??, "?˜ì§ˆ?¤ì—¼", "?ŒìŒì§„ë™", "?…ì·¨",
    "?ê¸°ë¬?, "?ê¸°ë¬¼ì²˜ë¦?, "?˜ê²½?í–¥?‰ê???, "?˜ê²½?í–¥?‰ê??¬ì˜",
    
    # ?„ì‹œê³„íš ê´€??
    "?„ì‹œê³„íš", "?„ì‹œê³„íš?œì„¤", "?„ì‹œê³„íš?¬ì—…", "?„ì‹œê³„íšë³€ê²?, "?„ì‹œê³„íšê²°ì •",
    "ê°œë°œ?‰ìœ„?ˆê?", "ê°œë°œ?‰ìœ„? ê³ ", "ê°œë°œ?œí•œêµ¬ì—­", "?„ì‹œê³„íšêµ¬ì—­",
    
    # êµí†µ ê´€??
    "êµí†µ", "êµí†µ?í–¥?‰ê?", "êµí†µê³„íš", "êµí†µ?œì„¤", "?„ë¡œ", "êµëŸ‰", "?°ë„",
    "êµí†µ?¬ê³ ", "êµí†µ?„ë°˜", "êµí†µ?•ë¦¬", "êµí†µ? í˜¸",
    
    # ë³´ê±´ë³µì? ê´€??
    "ë³´ê±´", "ë³µì?", "?˜ë£Œ", "?˜ë£Œê¸°ê?", "?˜ë£Œê¸°ê¸°", "?˜ë£Œ??, "?˜ë£Œë²?,
    "?¬íšŒë³´ì¥", "êµ???°ê¸ˆ", "ê±´ê°•ë³´í—˜", "?°ì—…?¬í•´ë³´ìƒë³´í—˜",
    
    # êµìœ¡ ê´€??
    "êµìœ¡", "?™êµ", "êµìœ¡ê¸°ê?", "êµìœ¡ë²?, "êµìœ¡ê³¼ì •", "êµìœ¡?œì„¤", "êµìœ¡?œì„¤ê¸°ì?",
    "?¬ë¦½?™êµ", "?¬ë¦½?™êµë²?, "êµìœ¡ê°?, "êµìœ¡?„ì›??,
    
    # ?¸ë™ ê´€??
    "?¸ë™", "ê³ ìš©", "ê·¼ë¡œ", "ê·¼ë¡œê¸°ì?ë²?, "?°ì—…?ˆì „ë³´ê±´ë²?, "?°ì—…?¬í•´",
    "?¸ë™ì¡°í•©", "?¨ì²´êµì„­", "?Œì—…", "?Œê²¬ê·¼ë¡œ", "ê¸°ê°„?œê·¼ë¡?,
    
    # ê¸ˆìœµ ê´€??
    "ê¸ˆìœµ", "ê¸ˆìœµê°ë…", "ê¸ˆìœµê¸°ê?", "ê¸ˆìœµ?í’ˆ", "ê¸ˆìœµê±°ë˜", "ê¸ˆìœµ?¬ì",
    "?€??, "ë³´í—˜", "ì¦ê¶Œ", "?ë³¸?œì¥", "ê¸ˆìœµ?¬ì?…ë²•"
]

# ?¬íŒ ? í˜•ë³?ë¶„ë¥˜ ?¤ì›Œ??
APPEAL_TYPE_KEYWORDS = {
    "?ˆê??¸ê?": ["?ˆê?", "?¸ê?", "ë©´í—ˆ", "?±ë¡", "? ê³ "],
    "ì²˜ë¶„ì·¨ì†Œ": ["ì²˜ë¶„", "ì·¨ì†Œ", "?•ì?", "ì² íšŒ", "ë¬´íš¨"],
    "ë¶€ê³¼ì²˜ë¶?: ["ë¶€ê³?, "ê³¼íƒœë£?, "ê³¼ì§•ê¸?, "ê°€?°ì„¸", "ê°€?°ê¸ˆ"],
    "ì§•ê³„ì²˜ë¶„": ["ì§•ê³„", "?´ì„", "?Œë©´", "?•ì§", "ê°ë´‰"],
    "?¸ë¬´ì²˜ë¶„": ["êµ?„¸", "ì§€ë°©ì„¸", "?¸ë¬´ì¡°ì‚¬", "ì²´ë‚©ì²˜ë¶„", "?•ë¥˜"],
    "ê±´ì¶•ì²˜ë¶„": ["ê±´ì¶•", "ê±´ì¶•?ˆê?", "ê±´ì¶•? ê³ ", "?©ë„ë³€ê²?, "ì² ê±°ëª…ë ¹"],
    "?˜ê²½ì²˜ë¶„": ["?˜ê²½", "?˜ê²½?í–¥?‰ê?", "?˜ê²½?¤ì—¼", "?ê¸°ë¬?, "?ŒìŒì§„ë™"],
    "?„ì‹œê³„íš": ["?„ì‹œê³„íš", "ê°œë°œ?‰ìœ„", "ê°œë°œ?œí•œêµ¬ì—­", "?„ì‹œê³„íš?œì„¤"],
    "êµí†µì²˜ë¶„": ["êµí†µ", "êµí†µ?í–¥?‰ê?", "êµí†µ?¬ê³ ", "êµí†µ?„ë°˜", "êµí†µ?•ë¦¬"],
    "ë³´ê±´ë³µì?": ["ë³´ê±´", "ë³µì?", "?˜ë£Œ", "?¬íšŒë³´ì¥", "êµ???°ê¸ˆ"],
    "êµìœ¡ì²˜ë¶„": ["êµìœ¡", "?™êµ", "êµìœ¡ê¸°ê?", "?¬ë¦½?™êµ", "êµìœ¡ë²?],
    "?¸ë™ì²˜ë¶„": ["?¸ë™", "ê³ ìš©", "ê·¼ë¡œ", "?°ì—…?¬í•´", "?¸ë™ì¡°í•©"],
    "ê¸ˆìœµì²˜ë¶„": ["ê¸ˆìœµ", "ê¸ˆìœµê°ë…", "ê¸ˆìœµê¸°ê?", "ê¸ˆìœµ?í’ˆ", "ê¸ˆìœµê±°ë˜"]
}


class AdministrativeAppealCollector:
    """?‰ì •?¬íŒë¡€ ?˜ì§‘ ?´ë˜??""
    
    def __init__(self, config: LawOpenAPIConfig):
        self.client = LawOpenAPIClient(config)
        self.output_dir = Path("data/raw/administrative_appeals")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collected_appeals = set()  # ì¤‘ë³µ ë°©ì?
        
    def collect_appeals_by_keyword(self, keyword: str, max_count: int = 50) -> List[Dict[str, Any]]:
        """?¤ì›Œ?œë¡œ ?‰ì •?¬íŒë¡€ ê²€??ë°??˜ì§‘"""
        logger.info(f"?¤ì›Œ??'{keyword}'ë¡??‰ì •?¬íŒë¡€ ê²€???œì‘...")
        
        appeals = []
        page = 1
        
        while len(appeals) < max_count:
            try:
                results = self.client.get_administrative_appeal_list(
                    query=keyword,
                    display=100,
                    page=page
                )
                
                if not results:
                    break
                
                for result in results:
                    appeal_id = result.get('?ë??¼ë ¨ë²ˆí˜¸')
                    if appeal_id and appeal_id not in self.collected_appeals:
                        appeals.append(result)
                        self.collected_appeals.add(appeal_id)
                        
                        if len(appeals) >= max_count:
                            break
                
                page += 1
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API ?”ì²­ ?œë„ê°€ ê±°ì˜ ?Œì§„?˜ì—ˆ?µë‹ˆ??")
                    break
                    
            except Exception as e:
                logger.error(f"?¤ì›Œ??'{keyword}' ê²€??ì¤??¤ë¥˜: {e}")
                break
        
        logger.info(f"?¤ì›Œ??'{keyword}'ë¡?{len(appeals)}ê±??˜ì§‘")
        return appeals
    
    def collect_appeals_by_date_range(self, start_date: str, end_date: str, max_count: int = 1000) -> List[Dict[str, Any]]:
        """? ì§œ ë²”ìœ„ë¡??‰ì •?¬íŒë¡€ ê²€??ë°??˜ì§‘"""
        logger.info(f"? ì§œ ë²”ìœ„ {start_date} ~ {end_date}ë¡??‰ì •?¬íŒë¡€ ê²€???œì‘...")
        
        appeals = []
        page = 1
        
        while len(appeals) < max_count:
            try:
                results = self.client.get_administrative_appeal_list(
                    display=100,
                    page=page,
                    from_date=start_date,
                    to_date=end_date
                )
                
                if not results:
                    break
                
                for result in results:
                    appeal_id = result.get('?ë??¼ë ¨ë²ˆí˜¸')
                    if appeal_id and appeal_id not in self.collected_appeals:
                        appeals.append(result)
                        self.collected_appeals.add(appeal_id)
                        
                        if len(appeals) >= max_count:
                            break
                
                page += 1
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API ?”ì²­ ?œë„ê°€ ê±°ì˜ ?Œì§„?˜ì—ˆ?µë‹ˆ??")
                    break
                    
            except Exception as e:
                logger.error(f"? ì§œ ë²”ìœ„ ê²€??ì¤??¤ë¥˜: {e}")
                break
        
        logger.info(f"? ì§œ ë²”ìœ„ë¡?{len(appeals)}ê±??˜ì§‘")
        return appeals
    
    def collect_appeal_details(self, appeal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """?‰ì •?¬íŒë¡€ ?ì„¸ ?•ë³´ ?˜ì§‘"""
        appeal_id = appeal.get('?ë??¼ë ¨ë²ˆí˜¸')
        if not appeal_id:
            return None
        
        try:
            detail = self.client.get_administrative_appeal_detail(appeal_id=appeal_id)
            if detail:
                # ê¸°ë³¸ ?•ë³´?€ ?ì„¸ ?•ë³´ ê²°í•©
                combined_data = {
                    'basic_info': appeal,
                    'detail_info': detail,
                    'collected_at': datetime.now().isoformat()
                }
                return combined_data
        except Exception as e:
            logger.error(f"?‰ì •?¬íŒë¡€ {appeal_id} ?ì„¸ ?•ë³´ ?˜ì§‘ ?¤íŒ¨: {e}")
        
        return None
    
    def classify_appeal_type(self, appeal: Dict[str, Any]) -> str:
        """?‰ì •?¬íŒë¡€ ? í˜• ë¶„ë¥˜"""
        case_name = appeal.get('?¬ê±´ëª?, '').lower()
        case_content = appeal.get('?ì‹œ?¬í•­', '') + ' ' + appeal.get('?ê²°?”ì?', '')
        case_content = case_content.lower()
        
        # ?¬íŒ ? í˜•ë³??¤ì›Œ??ë§¤ì¹­
        for appeal_type, keywords in APPEAL_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in case_name or keyword in case_content:
                    return appeal_type
        
        return "ê¸°í?"
    
    def save_appeal_data(self, appeal_data: Dict[str, Any], filename: str):
        """?‰ì •?¬íŒë¡€ ?°ì´?°ë? ?Œì¼ë¡??€??""
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(appeal_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"?‰ì •?¬íŒë¡€ ?°ì´???€?? {filepath}")
        except Exception as e:
            logger.error(f"?‰ì •?¬íŒë¡€ ?°ì´???€???¤íŒ¨: {e}")
    
    def collect_all_appeals(self, target_count: int = 1000):
        """ëª¨ë“  ?‰ì •?¬íŒë¡€ ?˜ì§‘"""
        logger.info(f"?‰ì •?¬íŒë¡€ ?˜ì§‘ ?œì‘ (ëª©í‘œ: {target_count}ê±?...")
        
        all_appeals = []
        
        # 1. ?¤ì›Œ?œë³„ ê²€??(ê°??¤ì›Œ?œë‹¹ ìµœë? 30ê±?
        max_per_keyword = min(30, target_count // len(ADMINISTRATIVE_APPEAL_KEYWORDS))
        
        for i, keyword in enumerate(ADMINISTRATIVE_APPEAL_KEYWORDS):
            if len(all_appeals) >= target_count:
                break
                
            try:
                appeals = self.collect_appeals_by_keyword(keyword, max_per_keyword)
                all_appeals.extend(appeals)
                logger.info(f"?¤ì›Œ??'{keyword}' ?„ë£Œ. ?„ì : {len(all_appeals)}ê±?)
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 100:
                    logger.warning("API ?”ì²­ ?œë„ê°€ ë¶€ì¡±í•©?ˆë‹¤.")
                    break
                    
            except Exception as e:
                logger.error(f"?¤ì›Œ??'{keyword}' ê²€???¤íŒ¨: {e}")
                continue
        
        # 2. ? ì§œ ë²”ìœ„ë³?ê²€??(ìµœê·¼ 3??
        if len(all_appeals) < target_count:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y%m%d')
            
            remaining_count = target_count - len(all_appeals)
            date_appeals = self.collect_appeals_by_date_range(
                start_date, end_date, remaining_count
            )
            all_appeals.extend(date_appeals)
        
        logger.info(f"ì´?{len(all_appeals)}ê±´ì˜ ?‰ì •?¬íŒë¡€ ëª©ë¡ ?˜ì§‘ ?„ë£Œ")
        
        # 3. ê°??‰ì •?¬íŒë¡€???ì„¸ ?•ë³´ ?˜ì§‘
        detailed_appeals = []
        for i, appeal in enumerate(all_appeals):
            if i >= target_count:
                break
                
            try:
                detail = self.collect_appeal_details(appeal)
                if detail:
                    # ?¬íŒ ? í˜• ë¶„ë¥˜
                    appeal_type = self.classify_appeal_type(appeal)
                    detail['appeal_type'] = appeal_type
                    
                    detailed_appeals.append(detail)
                    
                    # ê°œë³„ ?Œì¼ë¡??€??
                    appeal_id = appeal.get('?ë??¼ë ¨ë²ˆí˜¸', f'unknown_{i}')
                    filename = f"administrative_appeal_{appeal_id}_{datetime.now().strftime('%Y%m%d')}.json"
                    self.save_appeal_data(detail, filename)
                
                # ì§„í–‰ë¥?ë¡œê·¸
                if (i + 1) % 100 == 0:
                    logger.info(f"?ì„¸ ?•ë³´ ?˜ì§‘ ì§„í–‰ë¥? {i + 1}/{len(all_appeals)}")
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API ?”ì²­ ?œë„ê°€ ê±°ì˜ ?Œì§„?˜ì—ˆ?µë‹ˆ??")
                    break
                    
            except Exception as e:
                logger.error(f"?‰ì •?¬íŒë¡€ {i} ?ì„¸ ?•ë³´ ?˜ì§‘ ?¤íŒ¨: {e}")
                continue
        
        logger.info(f"?‰ì •?¬íŒë¡€ ?ì„¸ ?•ë³´ ?˜ì§‘ ?„ë£Œ: {len(detailed_appeals)}ê±?)
        
        # ?˜ì§‘ ê²°ê³¼ ?”ì•½ ?ì„±
        self.generate_collection_summary(detailed_appeals)
    
    def generate_collection_summary(self, appeals: List[Dict[str, Any]]):
        """?˜ì§‘ ê²°ê³¼ ?”ì•½ ?ì„±"""
        # ?¬íŒ ? í˜•ë³??µê³„
        appeal_type_stats = {}
        
        for appeal in appeals:
            appeal_type = appeal.get('appeal_type', 'ê¸°í?')
            appeal_type_stats[appeal_type] = appeal_type_stats.get(appeal_type, 0) + 1
        
        summary = {
            'collection_date': datetime.now().isoformat(),
            'total_appeals': len(appeals),
            'appeal_type_distribution': appeal_type_stats,
            'api_stats': self.client.get_request_stats()
        }
        
        summary_file = self.output_dir / f"collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info(f"?˜ì§‘ ê²°ê³¼ ?”ì•½ ?€?? {summary_file}")
        except Exception as e:
            logger.error(f"?˜ì§‘ ê²°ê³¼ ?”ì•½ ?€???¤íŒ¨: {e}")


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    # ?˜ê²½ë³€???•ì¸
    oc = os.getenv("LAW_OPEN_API_OC")
    if not oc:
        logger.error("LAW_OPEN_API_OC ?˜ê²½ë³€?˜ê? ?¤ì •?˜ì? ?Šì•˜?µë‹ˆ??")
        logger.info("?¬ìš©ë²? LAW_OPEN_API_OC=your_email_id python collect_administrative_appeals.py")
        return
    
    # ë¡œê·¸ ?”ë ‰? ë¦¬ ?ì„±
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # API ?¤ì •
    config = LawOpenAPIConfig(oc=oc)
    
    # ?‰ì •?¬íŒë¡€ ?˜ì§‘ ?¤í–‰
    collector = AdministrativeAppealCollector(config)
    collector.collect_all_appeals(target_count=1000)


if __name__ == "__main__":
    main()
