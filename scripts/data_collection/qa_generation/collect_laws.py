#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ë²•ë ¹ ?˜ì§‘ ?¤í¬ë¦½íŠ¸

êµ??ë²•ë ¹?•ë³´?¼í„° LAW OPEN APIë¥??¬ìš©?˜ì—¬ ì£¼ìš” ë²•ë ¹ 20ê°œë? ?˜ì§‘?©ë‹ˆ??
- ?„í–‰ë²•ë ¹(?œí–‰?? ê¸°ì??¼ë¡œ ?˜ì§‘
- ëª¨ë“  ì¡°ë¬¸ ë°?ê°œì •?´ë ¥ ?¬í•¨
- ë²•ë ¹ ì²´ê³„?? ? êµ¬ë²?ë¹„êµ, ?ë¬¸ë²•ë ¹ ??ë¶€ê°€?œë¹„???¬í•¨

?¬ìš©ë²?
    # ëª¨ë“  ë²•ë ¹ ?˜ì§‘
    python collect_laws.py
    
    # ?¹ì • ë²•ë ¹ë§??˜ì§‘
    python collect_laws.py --names ë¯¼ë²• ?ë²• ?•ë²•
    
    # ?˜ì§‘ ê°€?¥í•œ ë²•ë ¹ ëª©ë¡ ?•ì¸
    python collect_laws.py --list
    
    # ?˜ê²½ë³€???¤ì •
    set LAW_OPEN_API_OC=your_email_id
"""

import os
import sys
import json
import logging
import time
import random
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# .env ?Œì¼ ë¡œë”©
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"???˜ê²½ë³€??ë¡œë“œ ?„ë£Œ: {env_path}")
    else:
        print(f"? ï¸ .env ?Œì¼??ì°¾ì„ ???†ìŠµ?ˆë‹¤: {env_path}")
except ImportError:
    print("??python-dotenvê°€ ?¤ì¹˜?˜ì? ?Šì•˜?µë‹ˆ?? pip install python-dotenvë¡??¤ì¹˜?˜ì„¸??")
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
        logging.FileHandler('logs/collect_laws.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ì£¼ìš” ë²•ë ¹ 20ê°?ëª©ë¡ (ê²€?‰ëœ ID ?¬í•¨)
MAJOR_LAWS = [
    # ê¸°ë³¸ë²?(5ê°?
    {"name": "ë¯¼ë²•", "id": "001706", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "ê¸°ë³¸ë²?},
    {"name": "?ë²•", "id": "001702", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "ê¸°ë³¸ë²?},
    {"name": "?•ë²•", "id": "001692", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "ê¸°ë³¸ë²?},
    {"name": "ë¯¼ì‚¬?Œì†¡ë²?, "id": "001268", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "ê¸°ë³¸ë²?},
    {"name": "?•ì‚¬?Œì†¡ë²?, "id": "013873", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "ê¸°ë³¸ë²?},
    
    # ?¹ë³„ë²?(5ê°?
    {"name": "ê·¼ë¡œê¸°ì?ë²?, "id": "001872", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "?¹ë³„ë²?},
    {"name": "ë¶€?™ì‚°?±ê¸°ë²?, "id": "001697", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "?¹ë³„ë²?},
    {"name": "ê¸ˆìœµ?¤ëª…ê±°ë˜ ë°?ë¹„ë?ë³´ì¥??ê´€??ë²•ë¥ ", "id": "000549", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "?¹ë³„ë²?},
    {"name": "?€?‘ê¶Œë²?, "id": "000798", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "?¹ë³„ë²?},
    {"name": "ê°œì¸?•ë³´ ë³´í˜¸ë²?, "id": "011357", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "?¹ë³„ë²?},
    
    # ?‰ì •ë²?(5ê°?
    {"name": "?‰ì •?Œì†¡ë²?, "id": "001218", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "?‰ì •ë²?},
    {"name": "êµ?„¸ê¸°ë³¸ë²?, "id": "001586", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "?‰ì •ë²?},
    {"name": "ê±´ì¶•ë²?, "id": "001823", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "?‰ì •ë²?},
    {"name": "?‰ì •?ˆì°¨ë²?, "id": "001362", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "?‰ì •ë²?},
    {"name": "ê³µê³µê¸°ê????•ë³´ê³µê°œ??ê´€??ë²•ë¥ ", "id": "001357", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "?‰ì •ë²?},
    
    # ?¬íšŒë²?(5ê°?
    {"name": "êµ??ê¸°ì´ˆ?í™œë³´ì¥ë²?, "id": "001973", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "?¬íšŒë²?},
    {"name": "?˜ë£Œë²?, "id": "001788", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "?¬íšŒë²?},
    {"name": "êµìœ¡ê¸°ë³¸ë²?, "id": "000901", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "?¬íšŒë²?},
    {"name": "?˜ê²½?•ì±…ê¸°ë³¸ë²?, "id": "000173", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "?¬íšŒë²?},
    {"name": "?Œë¹„?ê¸°ë³¸ë²•", "id": "001589", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "?¬íšŒë²?}
]


class LawCollector:
    """ë²•ë ¹ ?˜ì§‘ ?´ë˜??""
    
    def __init__(self, config: LawOpenAPIConfig, min_delay: float = 1.0, max_delay: float = 3.0):
        self.client = LawOpenAPIClient(config)
        self.output_dir = Path("data/raw/laws")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_delay = min_delay  # ìµœì†Œ ì§€???œê°„ (ì´?
        self.max_delay = max_delay  # ìµœë? ì§€???œê°„ (ì´?
    
    def _get_random_delay(self) -> float:
        """3~7ì´??¬ì´???œë¤??ì§€???œê°„ ë°˜í™˜"""
        return random.uniform(self.min_delay, self.max_delay)
        
    def find_law_ids(self) -> List[Dict[str, Any]]:
        """ë²•ë ¹ëª…ìœ¼ë¡?ë²•ë ¹ ID ì°¾ê¸° (?´ë? IDê°€ ?ˆìœ¼ë©?ê±´ë„ˆ?°ê¸°)"""
        logger.info("ë²•ë ¹ ID ê²€???œì‘...")
        
        laws_with_ids = []
        for i, law in enumerate(MAJOR_LAWS):
            # ?´ë? IDê°€ ?ˆëŠ” ê²½ìš° API ?¸ì¶œ ê±´ë„ˆ?°ê¸°
            if law.get('id'):
                logger.info(f"'{law['name']}' ID ?´ë? ì¡´ì¬: {law['id']} (API ?¸ì¶œ ê±´ë„ˆ?°ê¸°)")
                laws_with_ids.append(law)
                continue
            
            logger.info(f"'{law['name']}' ê²€??ì¤?.. ({i+1}/{len(MAJOR_LAWS)})")
            
            # API ?¸ì¶œ ê°??œë¤ ì§€???œê°„ ?ìš©
            if i > 0:
                delay = self._get_random_delay()
                logger.info(f"API ?¸ì¶œ ê°?{delay:.1f}ì´??€ê¸?..")
                time.sleep(delay)
            
            # ë²•ë ¹ëª…ìœ¼ë¡?ê²€??
            results = self.client.search_law_list_effective(query=law['name'], display=100)
            
            if results:
                # ?•í™•??ë²•ë ¹ëª?ë§¤ì¹­
                matched_law = None
                for result in results:
                    law_name = result.get('ë²•ë ¹ëª…í•œê¸€', '')
                    if law['name'] in law_name or law_name in law['name']:
                        matched_law = result
                        break
                
                if matched_law:
                    law['id'] = matched_law.get('ë²•ë ¹ID')
                    law['mst'] = matched_law.get('ë²•ë ¹?¼ë ¨ë²ˆí˜¸')
                    law['effective_date'] = matched_law.get('?œí–‰?¼ì')
                    law['promulgation_date'] = matched_law.get('ê³µí¬?¼ì')
                    law['ministry'] = matched_law.get('?Œê?ë¶€ì²˜ëª…')
                    laws_with_ids.append(law)
                    logger.info(f"'{law['name']}' ID ì°¾ìŒ: {law['id']}")
                else:
                    logger.warning(f"'{law['name']}' IDë¥?ì°¾ì„ ???†ìŠµ?ˆë‹¤.")
            else:
                logger.warning(f"'{law['name']}' ê²€??ê²°ê³¼ê°€ ?†ìŠµ?ˆë‹¤.")
        
        logger.info(f"ì´?{len(laws_with_ids)}ê°?ë²•ë ¹??IDë¥?ì°¾ì•˜?µë‹ˆ??")
        return laws_with_ids
    
    def collect_law_details(self, law: Dict[str, Any]) -> Dict[str, Any]:
        """ê°œë³„ ë²•ë ¹???ì„¸ ?•ë³´ ?˜ì§‘"""
        logger.info(f"'{law['name']}' ?ì„¸ ?•ë³´ ?˜ì§‘ ?œì‘...")
        
        law_data = {
            'basic_info': law,
            'current_text': None,
            'history': [],
            'articles': [],
            'collected_at': datetime.now().isoformat()
        }
        
        try:
            # 1. ?„í–‰ë²•ë ¹ ë³¸ë¬¸ ?˜ì§‘
            if law['id']:
                current_text = self.client.get_law_detail_effective(law_id=law['id'])
                if current_text:
                    law_data['current_text'] = current_text
                    logger.info(f"'{law['name']}' ?„í–‰ë²•ë ¹ ë³¸ë¬¸ ?˜ì§‘ ?„ë£Œ")
            
            # 2. ë²•ë ¹ ?°í˜ ?˜ì§‘
            if law['id']:
                history_list = self.client.get_law_history_list(law['id'], display=100)
                if history_list:
                    law_data['history'] = history_list
                    logger.info(f"'{law['name']}' ?°í˜ {len(history_list)}ê±??˜ì§‘ ?„ë£Œ")
            
            # 3. ì¡°ë¬¸ë³??ì„¸ ?•ë³´ ?˜ì§‘ (MST ?¬ìš©) - ? íƒ???˜ì§‘
            if law.get('mst') and law.get('effective_date'):
                logger.info(f"'{law['name']}' ì¡°ë¬¸ë³??ì„¸ ?•ë³´ ?˜ì§‘ ?œì‘...")
                # ì£¼ìš” ì¡°ë¬¸???˜ì§‘ (1ì¡°ë???10ì¡°ê¹Œì§€ë¡??œí•œ)
                for article_num in range(1, 11):
                    try:
                        # ì¡°ë¬¸ ?˜ì§‘ ê°??œë¤ ì§€???œê°„ ?ìš©
                        if article_num > 1:
                            delay = self._get_random_delay()
                            logger.debug(f"ì¡°ë¬¸ {article_num} ?˜ì§‘ ??{delay:.1f}ì´??€ê¸?..")
                            time.sleep(delay)
                        
                        jo = f"{article_num:04d}00"  # 6?ë¦¬ ì¡°ë²ˆ???•ì‹
                        article_detail = self.client.get_law_detail_effective(
                            mst=law['mst'], 
                            ef_yd=law['effective_date'], 
                            jo=jo
                        )
                        
                        # ?‘ë‹µ êµ¬ì¡° ?•ì¸ ë°?ì²˜ë¦¬
                        if article_detail:
                            # LawSearch êµ¬ì¡° ?•ì¸
                            if 'LawSearch' in article_detail:
                                law_search = article_detail['LawSearch']
                                if 'law' in law_search and law_search['law']:
                                    law_data['articles'].append({
                                        'article_number': article_num,
                                        'content': article_detail
                                    })
                                    logger.debug(f"ì¡°ë¬¸ {article_num} ?˜ì§‘ ?±ê³µ")
                                else:
                                    logger.debug(f"ì¡°ë¬¸ {article_num} ?´ìš© ?†ìŒ")
                            # ê¸°ì¡´ êµ¬ì¡° ?•ì¸
                            elif article_detail.get('response', {}).get('body', {}).get('items', {}).get('item'):
                                law_data['articles'].append({
                                    'article_number': article_num,
                                    'content': article_detail
                                })
                                logger.debug(f"ì¡°ë¬¸ {article_num} ?˜ì§‘ ?±ê³µ")
                            else:
                                logger.debug(f"ì¡°ë¬¸ {article_num} ?´ìš© ?†ìŒ")
                        else:
                            logger.debug(f"ì¡°ë¬¸ {article_num} ?‘ë‹µ ?†ìŒ")
                            
                    except Exception as e:
                        logger.warning(f"ì¡°ë¬¸ {article_num} ?˜ì§‘ ?¤íŒ¨: {e}")
                        # ì¡°ë¬¸ ?˜ì§‘ ?¤íŒ¨?´ë„ ê³„ì† ì§„í–‰
                        continue
                
                logger.info(f"'{law['name']}' ì¡°ë¬¸ ?˜ì§‘ ?„ë£Œ: {len(law_data['articles'])}ê°?)
            
            logger.info(f"'{law['name']}' ?ì„¸ ?•ë³´ ?˜ì§‘ ?„ë£Œ")
            
        except Exception as e:
            logger.error(f"'{law['name']}' ?˜ì§‘ ì¤??¤ë¥˜: {e}")
        
        return law_data
    
    def save_law_data(self, law_data: Dict[str, Any], law_name: str):
        """ë²•ë ¹ ?°ì´?°ë? ?Œì¼ë¡??€??""
        # ?Œì¼ëª…ì—???¹ìˆ˜ë¬¸ì ?œê±°
        safe_name = law_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        filename = f"{safe_name}_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(law_data, f, ensure_ascii=False, indent=2)
            logger.info(f"'{law_name}' ?°ì´???€???„ë£Œ: {filepath}")
        except Exception as e:
            logger.error(f"'{law_name}' ?°ì´???€???¤íŒ¨: {e}")
    
    def collect_all_laws(self, target_law_names: List[str] = None):
        """ëª¨ë“  ì£¼ìš” ë²•ë ¹ ?˜ì§‘ (?¹ì • ë²•ë ¹ ì§€??ê°€??"""
        logger.info("ë²•ë ¹ ?˜ì§‘ ?œì‘...")
        
        # 1. ë²•ë ¹ ID ì°¾ê¸°
        laws_with_ids = self.find_law_ids()
        
        if not laws_with_ids:
            logger.error("?˜ì§‘??ë²•ë ¹???†ìŠµ?ˆë‹¤.")
            return
        
        # 2. ?¹ì • ë²•ë ¹ë§??„í„°ë§?(ì§€?•ëœ ê²½ìš°)
        if target_law_names:
            filtered_laws = []
            for law in laws_with_ids:
                if law['name'] in target_law_names:
                    filtered_laws.append(law)
                else:
                    logger.info(f"'{law['name']}' ê±´ë„ˆ?°ê¸° (ì§€?•ë˜ì§€ ?ŠìŒ)")
            
            if not filtered_laws:
                logger.error(f"ì§€?•ëœ ë²•ë ¹??ì°¾ì„ ???†ìŠµ?ˆë‹¤: {target_law_names}")
                return
            
            laws_with_ids = filtered_laws
            logger.info(f"ì§€?•ëœ {len(laws_with_ids)}ê°?ë²•ë ¹ë§??˜ì§‘: {[law['name'] for law in laws_with_ids]}")
        
        # 3. ê°?ë²•ë ¹???ì„¸ ?•ë³´ ?˜ì§‘
        collected_count = 0
        for i, law in enumerate(laws_with_ids):
            try:
                logger.info(f"ë²•ë ¹ ?ì„¸ ?•ë³´ ?˜ì§‘ ì¤?.. ({i+1}/{len(laws_with_ids)})")
                
                # ë²•ë ¹ ê°??œë¤ ì§€???œê°„ ?ìš© (??ê¸?ì§€??
                if i > 0:
                    delay = self._get_random_delay() * 1.5  # ë²•ë ¹ ê°„ì—??1.5ë°???ê¸?ì§€??
                    logger.info(f"?¤ìŒ ë²•ë ¹ ?˜ì§‘ ??{delay:.1f}ì´??€ê¸?..")
                    time.sleep(delay)
                
                law_data = self.collect_law_details(law)
                self.save_law_data(law_data, law['name'])
                collected_count += 1
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API ?”ì²­ ?œë„ê°€ ê±°ì˜ ?Œì§„?˜ì—ˆ?µë‹ˆ??")
                    break
                    
            except Exception as e:
                logger.error(f"'{law['name']}' ?˜ì§‘ ?¤íŒ¨: {e}")
                continue
        
        logger.info(f"ë²•ë ¹ ?˜ì§‘ ?„ë£Œ: {collected_count}/{len(laws_with_ids)}ê°?)
        
        # 4. ?˜ì§‘ ê²°ê³¼ ?”ì•½ ?ì„±
        self.generate_collection_summary(laws_with_ids, collected_count)
    
    def generate_collection_summary(self, laws_with_ids: List[Dict[str, Any]], collected_count: int):
        """?˜ì§‘ ê²°ê³¼ ?”ì•½ ?ì„±"""
        summary = {
            'collection_date': datetime.now().isoformat(),
            'total_laws': len(MAJOR_LAWS),
            'found_laws': len(laws_with_ids),
            'collected_laws': collected_count,
            'laws_details': laws_with_ids,
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
    # ëª…ë ¹???¸ì ?Œì‹±
    parser = argparse.ArgumentParser(description='ë²•ë ¹ ?°ì´???˜ì§‘ ?¤í¬ë¦½íŠ¸')
    parser.add_argument('--names', '-n', nargs='+', help='?˜ì§‘??ë²•ë ¹ëª?(?¬ëŸ¬ ê°?ì§€??ê°€??')
    parser.add_argument('--list', '-l', action='store_true', help='?˜ì§‘ ê°€?¥í•œ ë²•ë ¹ ëª©ë¡ ì¶œë ¥')
    args = parser.parse_args()
    
    # ?˜ì§‘ ê°€?¥í•œ ë²•ë ¹ ëª©ë¡ ì¶œë ¥
    if args.list:
        print("?˜ì§‘ ê°€?¥í•œ ë²•ë ¹ ëª©ë¡:")
        print("=" * 50)
        for i, law in enumerate(MAJOR_LAWS, 1):
            print(f"{i:2d}. {law['name']} ({law['category']})")
        return
    
    # ?˜ê²½ë³€???•ì¸
    oc = os.getenv("LAW_OPEN_API_OC")
    if not oc:
        logger.error("LAW_OPEN_API_OC ?˜ê²½ë³€?˜ê? ?¤ì •?˜ì? ?Šì•˜?µë‹ˆ??")
        logger.info("?¬ìš©ë²? LAW_OPEN_API_OC=your_email_id python collect_laws.py")
        return
    
    # ë¡œê·¸ ?”ë ‰? ë¦¬ ?ì„±
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # API ?¤ì •
    config = LawOpenAPIConfig(oc=oc)
    
    # ì§€???œê°„ ?¤ì • (ê¸°ë³¸ê°? 3~7ì´? ?˜ê²½ë³€?˜ë¡œ ì¡°ì • ê°€??
    min_delay = float(os.getenv("API_MIN_DELAY", "3.0"))
    max_delay = float(os.getenv("API_MAX_DELAY", "7.0"))
    logger.info(f"API ?¸ì¶œ ê°??œë¤ ì§€???œê°„: {min_delay}~{max_delay}ì´?)
    
    # ë²•ë ¹ ?˜ì§‘ ?¤í–‰
    collector = LawCollector(config, min_delay=min_delay, max_delay=max_delay)
    
    if args.names:
        logger.info(f"ì§€?•ëœ ë²•ë ¹ë§??˜ì§‘: {args.names}")
        collector.collect_all_laws(target_law_names=args.names)
    else:
        logger.info("ëª¨ë“  ë²•ë ¹ ?˜ì§‘")
        collector.collect_all_laws()


if __name__ == "__main__":
    main()