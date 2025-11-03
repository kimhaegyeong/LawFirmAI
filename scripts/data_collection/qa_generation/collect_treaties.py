#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ì¡°ì•½ ?˜ì§‘ ?¤í¬ë¦½íŠ¸

êµ??ë²•ë ¹?•ë³´?¼í„° LAW OPEN APIë¥??¬ìš©?˜ì—¬ ì¡°ì•½???˜ì§‘?©ë‹ˆ??
- ì£¼ìš” ì¡°ì•½ 100ê±??˜ì§‘
- ì¡°ì•½ ? í˜•ë³?ë¶„ë¥˜ ë°?ë©”í??°ì´???•ì œ
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
        logging.FileHandler('logs/collect_treaties.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ì¡°ì•½ ê´€??ê²€???¤ì›Œ??
TREATY_KEYWORDS = [
    # ê²½ì œ?µìƒ ê´€??
    "?ìœ ë¬´ì—­?‘ì •", "FTA", "ê²½ì œ?‘ë ¥", "?¬ìë³´ì¥", "?´ì¤‘ê³¼ì„¸ë°©ì?", "ê´€??,
    "ë¬´ì—­", "?˜ì¶œ??, "ê²½ì œ?µìƒ", "?í˜¸?ì¡°", "ê²½ì œ?‘ë ¥?‘ì •",
    
    # ?¸êµ?ˆë³´ ê´€??
    "?¸êµ", "?ˆë³´", "ë°©ìœ„", "êµ°ì‚¬", "êµ°ì‚¬?‘ë ¥", "?•ë³´êµí™˜", "ë²”ì£„?¸ì¸??,
    "?¬ë²•ê³µì¡°", "?•ì‚¬?¬ë²•ê³µì¡°", "ë§ˆì•½", "?ŒëŸ¬", "êµ? œë²”ì£„",
    
    # ?˜ê²½ ê´€??
    "?˜ê²½", "ê¸°í›„ë³€??, "?¨ì‹¤ê°€??, "?¤ì¡´ì¸?, "?ë¬¼?¤ì–‘??, "?´ì–‘?˜ê²½",
    "?€ê¸°ì˜¤??, "?˜ì§ˆ?¤ì—¼", "?ê¸°ë¬?, "?˜ê²½ë³´í˜¸", "ì§€?ê??¥ë°œ??,
    
    # ?¸ê¶Œ ê´€??
    "?¸ê¶Œ", "?¸ê¶Œë³´í˜¸", "?„ë™ê¶Œë¦¬", "?¬ì„±ê¶Œë¦¬", "?¥ì• ?¸ê¶Œë¦?, "?œë?",
    "?´ì£¼", "?¸ì‹ ë§¤ë§¤", "ê°•ì œ?¸ë™", "ì°¨ë³„ê¸ˆì?", "?‰ë“±ê¶?,
    
    # êµìœ¡ë¬¸í™” ê´€??
    "êµìœ¡", "ë¬¸í™”", "ê³¼í•™ê¸°ìˆ ", "?°êµ¬ê°œë°œ", "?™ìˆ êµë¥˜", "ë¬¸í™”êµë¥˜",
    "êµìœ¡?‘ë ¥", "ê³¼í•™ê¸°ìˆ ?‘ë ¥", "?°êµ¬?‘ë ¥", "ê¸°ìˆ ?´ì „", "ì§€?ì¬?°ê¶Œ",
    
    # ë³´ê±´?˜ë£Œ ê´€??
    "ë³´ê±´", "?˜ë£Œ", "ê³µì¤‘ë³´ê±´", "ì§ˆë³‘?ˆë°©", "?˜ë£Œê¸°ìˆ ", "?˜ë£Œ?‘ë ¥",
    "ë³´ê±´?‘ë ¥", "?˜ë£Œì§?, "?˜ë£Œê¸°ê¸°", "?˜ì•½??, "?˜ë£Œ?•ë³´",
    
    # êµí†µ?µì‹  ê´€??
    "êµí†µ", "?µì‹ ", "??³µ", "?´ìš´", "?¡ìƒêµí†µ", "?„ì?µì‹ ", "?•ë³´?µì‹ ",
    "??³µ?‘ì •", "?´ìš´?‘ì •", "êµí†µ?‘ë ¥", "?µì‹ ?‘ë ¥", "?”ì??¸í˜‘??,
    
    # ?ì—…?í’ˆ ê´€??
    "?ì—…", "?í’ˆ", "ì¶•ì‚°", "?˜ì‚°", "?ì—…?‘ë ¥", "?í’ˆ?ˆì „", "?ì‚°ë¬?,
    "ì¶•ì‚°ë¬?, "?˜ì‚°ë¬?, "?ì—…ê¸°ìˆ ", "?í’ˆê¸°ìˆ ", "?ì—…êµì—­",
    
    # ?ë„ˆì§€?ì› ê´€??
    "?ë„ˆì§€", "?ì›", "?ìœ ", "ê°€??, "?ì??, "?¬ìƒ?ë„ˆì§€", "?ë„ˆì§€?‘ë ¥",
    "?ì›?‘ë ¥", "?ë„ˆì§€?ˆë³´", "?ì›?ˆë³´", "?ë„ˆì§€?¨ìœ¨", "? ì¬?ì—?ˆì?",
    
    # ?¬íšŒë³´ì¥ ê´€??
    "?¬íšŒë³´ì¥", "ë³µì?", "?¸ë™", "ê³ ìš©", "?¬íšŒë³´í—˜", "êµ???°ê¸ˆ", "ê±´ê°•ë³´í—˜",
    "?°ì—…?¬í•´ë³´ìƒë³´í—˜", "ê³ ìš©ë³´í—˜", "?¬íšŒë³´ì¥?‘ë ¥", "ë³µì??‘ë ¥", "?¸ë™?‘ë ¥"
]

# ì¡°ì•½ ? í˜•ë³?ë¶„ë¥˜ ?¤ì›Œ??
TREATY_TYPE_KEYWORDS = {
    "ê²½ì œ?µìƒ": ["?ìœ ë¬´ì—­?‘ì •", "FTA", "ê²½ì œ?‘ë ¥", "?¬ìë³´ì¥", "?´ì¤‘ê³¼ì„¸ë°©ì?", "ê´€??, "ë¬´ì—­"],
    "?¸êµ?ˆë³´": ["?¸êµ", "?ˆë³´", "ë°©ìœ„", "êµ°ì‚¬", "êµ°ì‚¬?‘ë ¥", "?•ë³´êµí™˜", "ë²”ì£„?¸ì¸??, "?¬ë²•ê³µì¡°"],
    "?˜ê²½": ["?˜ê²½", "ê¸°í›„ë³€??, "?¨ì‹¤ê°€??, "?¤ì¡´ì¸?, "?ë¬¼?¤ì–‘??, "?´ì–‘?˜ê²½", "?€ê¸°ì˜¤??],
    "?¸ê¶Œ": ["?¸ê¶Œ", "?¸ê¶Œë³´í˜¸", "?„ë™ê¶Œë¦¬", "?¬ì„±ê¶Œë¦¬", "?¥ì• ?¸ê¶Œë¦?, "?œë?", "?´ì£¼", "?¸ì‹ ë§¤ë§¤"],
    "êµìœ¡ë¬¸í™”": ["êµìœ¡", "ë¬¸í™”", "ê³¼í•™ê¸°ìˆ ", "?°êµ¬ê°œë°œ", "?™ìˆ êµë¥˜", "ë¬¸í™”êµë¥˜", "êµìœ¡?‘ë ¥"],
    "ë³´ê±´?˜ë£Œ": ["ë³´ê±´", "?˜ë£Œ", "ê³µì¤‘ë³´ê±´", "ì§ˆë³‘?ˆë°©", "?˜ë£Œê¸°ìˆ ", "?˜ë£Œ?‘ë ¥", "ë³´ê±´?‘ë ¥"],
    "êµí†µ?µì‹ ": ["êµí†µ", "?µì‹ ", "??³µ", "?´ìš´", "?¡ìƒêµí†µ", "?„ì?µì‹ ", "?•ë³´?µì‹ "],
    "?ì—…?í’ˆ": ["?ì—…", "?í’ˆ", "ì¶•ì‚°", "?˜ì‚°", "?ì—…?‘ë ¥", "?í’ˆ?ˆì „", "?ì‚°ë¬?],
    "?ë„ˆì§€?ì›": ["?ë„ˆì§€", "?ì›", "?ìœ ", "ê°€??, "?ì??, "?¬ìƒ?ë„ˆì§€", "?ë„ˆì§€?‘ë ¥"],
    "?¬íšŒë³´ì¥": ["?¬íšŒë³´ì¥", "ë³µì?", "?¸ë™", "ê³ ìš©", "?¬íšŒë³´í—˜", "êµ???°ê¸ˆ", "ê±´ê°•ë³´í—˜"]
}


class TreatyCollector:
    """ì¡°ì•½ ?˜ì§‘ ?´ë˜??""
    
    def __init__(self, config: LawOpenAPIConfig):
        self.client = LawOpenAPIClient(config)
        self.output_dir = Path("data/raw/treaties")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collected_treaties = set()  # ì¤‘ë³µ ë°©ì?
        
    def collect_treaties_by_keyword(self, keyword: str, max_count: int = 10) -> List[Dict[str, Any]]:
        """?¤ì›Œ?œë¡œ ì¡°ì•½ ê²€??ë°??˜ì§‘"""
        logger.info(f"?¤ì›Œ??'{keyword}'ë¡?ì¡°ì•½ ê²€???œì‘...")
        
        treaties = []
        page = 1
        
        while len(treaties) < max_count:
            try:
                results = self.client.get_treaty_list(
                    query=keyword,
                    display=100,
                    page=page
                )
                
                if not results:
                    break
                
                for result in results:
                    treaty_id = result.get('?ë??¼ë ¨ë²ˆí˜¸')
                    if treaty_id and treaty_id not in self.collected_treaties:
                        treaties.append(result)
                        self.collected_treaties.add(treaty_id)
                        
                        if len(treaties) >= max_count:
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
        
        logger.info(f"?¤ì›Œ??'{keyword}'ë¡?{len(treaties)}ê±??˜ì§‘")
        return treaties
    
    def collect_treaties_by_date_range(self, start_date: str, end_date: str, max_count: int = 100) -> List[Dict[str, Any]]:
        """? ì§œ ë²”ìœ„ë¡?ì¡°ì•½ ê²€??ë°??˜ì§‘"""
        logger.info(f"? ì§œ ë²”ìœ„ {start_date} ~ {end_date}ë¡?ì¡°ì•½ ê²€???œì‘...")
        
        treaties = []
        page = 1
        
        while len(treaties) < max_count:
            try:
                results = self.client.get_treaty_list(
                    display=100,
                    page=page,
                    from_date=start_date,
                    to_date=end_date
                )
                
                if not results:
                    break
                
                for result in results:
                    treaty_id = result.get('?ë??¼ë ¨ë²ˆí˜¸')
                    if treaty_id and treaty_id not in self.collected_treaties:
                        treaties.append(result)
                        self.collected_treaties.add(treaty_id)
                        
                        if len(treaties) >= max_count:
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
        
        logger.info(f"? ì§œ ë²”ìœ„ë¡?{len(treaties)}ê±??˜ì§‘")
        return treaties
    
    def collect_treaty_details(self, treaty: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """ì¡°ì•½ ?ì„¸ ?•ë³´ ?˜ì§‘"""
        treaty_id = treaty.get('?ë??¼ë ¨ë²ˆí˜¸')
        if not treaty_id:
            return None
        
        try:
            detail = self.client.get_treaty_detail(treaty_id=treaty_id)
            if detail:
                # ê¸°ë³¸ ?•ë³´?€ ?ì„¸ ?•ë³´ ê²°í•©
                combined_data = {
                    'basic_info': treaty,
                    'detail_info': detail,
                    'collected_at': datetime.now().isoformat()
                }
                return combined_data
        except Exception as e:
            logger.error(f"ì¡°ì•½ {treaty_id} ?ì„¸ ?•ë³´ ?˜ì§‘ ?¤íŒ¨: {e}")
        
        return None
    
    def classify_treaty_type(self, treaty: Dict[str, Any]) -> str:
        """ì¡°ì•½ ? í˜• ë¶„ë¥˜"""
        case_name = treaty.get('?¬ê±´ëª?, '').lower()
        case_content = treaty.get('?ì‹œ?¬í•­', '') + ' ' + treaty.get('?ê²°?”ì?', '')
        case_content = case_content.lower()
        
        # ì¡°ì•½ ? í˜•ë³??¤ì›Œ??ë§¤ì¹­
        for treaty_type, keywords in TREATY_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in case_name or keyword in case_content:
                    return treaty_type
        
        return "ê¸°í?"
    
    def save_treaty_data(self, treaty_data: Dict[str, Any], filename: str):
        """ì¡°ì•½ ?°ì´?°ë? ?Œì¼ë¡??€??""
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(treaty_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"ì¡°ì•½ ?°ì´???€?? {filepath}")
        except Exception as e:
            logger.error(f"ì¡°ì•½ ?°ì´???€???¤íŒ¨: {e}")
    
    def collect_all_treaties(self, target_count: int = 100):
        """ëª¨ë“  ì¡°ì•½ ?˜ì§‘"""
        logger.info(f"ì¡°ì•½ ?˜ì§‘ ?œì‘ (ëª©í‘œ: {target_count}ê±?...")
        
        all_treaties = []
        
        # 1. ?¤ì›Œ?œë³„ ê²€??(ê°??¤ì›Œ?œë‹¹ ìµœë? 5ê±?
        max_per_keyword = min(5, target_count // len(TREATY_KEYWORDS))
        
        for i, keyword in enumerate(TREATY_KEYWORDS):
            if len(all_treaties) >= target_count:
                break
                
            try:
                treaties = self.collect_treaties_by_keyword(keyword, max_per_keyword)
                all_treaties.extend(treaties)
                logger.info(f"?¤ì›Œ??'{keyword}' ?„ë£Œ. ?„ì : {len(all_treaties)}ê±?)
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 100:
                    logger.warning("API ?”ì²­ ?œë„ê°€ ë¶€ì¡±í•©?ˆë‹¤.")
                    break
                    
            except Exception as e:
                logger.error(f"?¤ì›Œ??'{keyword}' ê²€???¤íŒ¨: {e}")
                continue
        
        # 2. ? ì§œ ë²”ìœ„ë³?ê²€??(ìµœê·¼ 10??
        if len(all_treaties) < target_count:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y%m%d')
            
            remaining_count = target_count - len(all_treaties)
            date_treaties = self.collect_treaties_by_date_range(
                start_date, end_date, remaining_count
            )
            all_treaties.extend(date_treaties)
        
        logger.info(f"ì´?{len(all_treaties)}ê±´ì˜ ì¡°ì•½ ëª©ë¡ ?˜ì§‘ ?„ë£Œ")
        
        # 3. ê°?ì¡°ì•½???ì„¸ ?•ë³´ ?˜ì§‘
        detailed_treaties = []
        for i, treaty in enumerate(all_treaties):
            if i >= target_count:
                break
                
            try:
                detail = self.collect_treaty_details(treaty)
                if detail:
                    # ì¡°ì•½ ? í˜• ë¶„ë¥˜
                    treaty_type = self.classify_treaty_type(treaty)
                    detail['treaty_type'] = treaty_type
                    
                    detailed_treaties.append(detail)
                    
                    # ê°œë³„ ?Œì¼ë¡??€??
                    treaty_id = treaty.get('?ë??¼ë ¨ë²ˆí˜¸', f'unknown_{i}')
                    filename = f"treaty_{treaty_id}_{datetime.now().strftime('%Y%m%d')}.json"
                    self.save_treaty_data(detail, filename)
                
                # ì§„í–‰ë¥?ë¡œê·¸
                if (i + 1) % 10 == 0:
                    logger.info(f"?ì„¸ ?•ë³´ ?˜ì§‘ ì§„í–‰ë¥? {i + 1}/{len(all_treaties)}")
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API ?”ì²­ ?œë„ê°€ ê±°ì˜ ?Œì§„?˜ì—ˆ?µë‹ˆ??")
                    break
                    
            except Exception as e:
                logger.error(f"ì¡°ì•½ {i} ?ì„¸ ?•ë³´ ?˜ì§‘ ?¤íŒ¨: {e}")
                continue
        
        logger.info(f"ì¡°ì•½ ?ì„¸ ?•ë³´ ?˜ì§‘ ?„ë£Œ: {len(detailed_treaties)}ê±?)
        
        # ?˜ì§‘ ê²°ê³¼ ?”ì•½ ?ì„±
        self.generate_collection_summary(detailed_treaties)
    
    def generate_collection_summary(self, treaties: List[Dict[str, Any]]):
        """?˜ì§‘ ê²°ê³¼ ?”ì•½ ?ì„±"""
        # ì¡°ì•½ ? í˜•ë³??µê³„
        treaty_type_stats = {}
        
        for treaty in treaties:
            treaty_type = treaty.get('treaty_type', 'ê¸°í?')
            treaty_type_stats[treaty_type] = treaty_type_stats.get(treaty_type, 0) + 1
        
        summary = {
            'collection_date': datetime.now().isoformat(),
            'total_treaties': len(treaties),
            'treaty_type_distribution': treaty_type_stats,
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
        logger.info("?¬ìš©ë²? LAW_OPEN_API_OC=your_email_id python collect_treaties.py")
        return
    
    # ë¡œê·¸ ?”ë ‰? ë¦¬ ?ì„±
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # API ?¤ì •
    config = LawOpenAPIConfig(oc=oc)
    
    # ì¡°ì•½ ?˜ì§‘ ?¤í–‰
    collector = TreatyCollector(config)
    collector.collect_all_treaties(target_count=100)


if __name__ == "__main__":
    main()
