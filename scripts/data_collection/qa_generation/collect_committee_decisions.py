#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?„ì›?Œê²°?•ë¬¸ ?˜ì§‘ ?¤í¬ë¦½íŠ¸

êµ??ë²•ë ¹?•ë³´?¼í„° LAW OPEN APIë¥??¬ìš©?˜ì—¬ ?„ì›?Œê²°?•ë¬¸???˜ì§‘?©ë‹ˆ??
- ì£¼ìš” ?„ì›?Œë³„ ê²°ì •ë¬?500ê±??˜ì§‘
- ?„ì›?Œë³„ ë¶„ë¥˜ ë°?ë©”í??°ì´???•ì œ
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
        logging.FileHandler('logs/collect_committee_decisions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ?„ì›?Œë³„ ?¤ì›Œ??
COMMITTEE_KEYWORDS = {
    "êµ? •ê°ì‚¬?„ì›??: ["êµ? •ê°ì‚¬", "ê°ì‚¬", "?•ë?ê°ì‚¬", "êµ? •ê°ì‚¬?„ì›??],
    "?ˆì‚°ê²°ì‚°?¹ë³„?„ì›??: ["?ˆì‚°", "ê²°ì‚°", "?ˆì‚°??, "ê²°ì‚°??, "?ˆì‚°ê²°ì‚°"],
    "ë²•ì œ?¬ë²•?„ì›??: ["ë²•ì œ", "?¬ë²•", "ë²•ë¥ ", "ë²•ì•ˆ", "ë²•ì œ?¬ë²•"],
    "ê¸°íš?¬ì •?„ì›??: ["ê¸°íš", "?¬ì •", "ê²½ì œ", "?•ì±…", "ê¸°íš?¬ì •"],
    "ê³¼í•™ê¸°ìˆ ?•ë³´?µì‹ ?„ì›??: ["ê³¼í•™", "ê¸°ìˆ ", "?•ë³´?µì‹ ", "?”ì???, "ICT"],
    "?‰ì •?ˆì „?„ì›??: ["?‰ì •", "?ˆì „", "ì§€ë°©ìì¹?, "ê³µë¬´??, "?‰ì •?ˆì „"],
    "ë¬¸í™”ì²´ìœ¡ê´€ê´‘ìœ„?íšŒ": ["ë¬¸í™”", "ì²´ìœ¡", "ê´€ê´?, "?ˆìˆ ", "?¤í¬ì¸?],
    "?ë¦¼ì¶•ì‚°?í’ˆ?´ì–‘?˜ì‚°?„ì›??: ["?ì—…", "ì¶•ì‚°", "?í’ˆ", "?´ì–‘", "?˜ì‚°"],
    "?°ì—…?µìƒ?ì›ì¤‘ì†Œë²¤ì²˜ê¸°ì—…?„ì›??: ["?°ì—…", "?µìƒ", "?ì›", "ì¤‘ì†Œê¸°ì—…", "ë²¤ì²˜"],
    "ë³´ê±´ë³µì??„ì›??: ["ë³´ê±´", "ë³µì?", "?˜ë£Œ", "ê±´ê°•", "?¬íšŒë³´ì¥"],
    "?˜ê²½?¸ë™?„ì›??: ["?˜ê²½", "?¸ë™", "ê³ ìš©", "?°ì—…?ˆì „", "?˜ê²½?¸ë™"]
}

# ?„ì›??ì½”ë“œ ë§¤í•‘
COMMITTEE_CODES = {
    "êµ? •ê°ì‚¬?„ì›??: "audit",
    "?ˆì‚°ê²°ì‚°?¹ë³„?„ì›??: "budget", 
    "ë²•ì œ?¬ë²•?„ì›??: "legis",
    "ê¸°íš?¬ì •?„ì›??: "plan",
    "ê³¼í•™ê¸°ìˆ ?•ë³´?µì‹ ?„ì›??: "scitech",
    "?‰ì •?ˆì „?„ì›??: "admin",
    "ë¬¸í™”ì²´ìœ¡ê´€ê´‘ìœ„?íšŒ": "culture",
    "?ë¦¼ì¶•ì‚°?í’ˆ?´ì–‘?˜ì‚°?„ì›??: "agri",
    "?°ì—…?µìƒ?ì›ì¤‘ì†Œë²¤ì²˜ê¸°ì—…?„ì›??: "industry",
    "ë³´ê±´ë³µì??„ì›??: "welfare",
    "?˜ê²½?¸ë™?„ì›??: "envlabor"
}


class CommitteeDecisionCollector:
    """?„ì›?Œê²°?•ë¬¸ ?˜ì§‘ ?´ë˜??""
    
    def __init__(self, config: LawOpenAPIConfig):
        self.client = LawOpenAPIClient(config)
        self.output_dir = Path("data/raw/committee_decisions")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collected_decisions = set()  # ì¤‘ë³µ ë°©ì?
        
    def collect_decisions_by_committee(self, committee: str, max_count: int = 50) -> List[Dict[str, Any]]:
        """?„ì›?Œë³„ ê²°ì •ë¬?ê²€??ë°??˜ì§‘"""
        logger.info(f"?„ì›??'{committee}' ê²°ì •ë¬?ê²€???œì‘...")
        
        committee_code = COMMITTEE_CODES.get(committee)
        if not committee_code:
            logger.error(f"ì§€?í•˜ì§€ ?ŠëŠ” ?„ì›?? {committee}")
            return []
        
        decisions = []
        page = 1
        
        while len(decisions) < max_count:
            try:
                results = self.client.get_committee_decision_list(
                    committee=committee_code,
                    display=100,
                    page=page
                )
                
                if not results:
                    break
                
                for result in results:
                    decision_id = result.get('?ë??¼ë ¨ë²ˆí˜¸')
                    if decision_id and decision_id not in self.collected_decisions:
                        decisions.append(result)
                        self.collected_decisions.add(decision_id)
                        
                        if len(decisions) >= max_count:
                            break
                
                page += 1
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API ?”ì²­ ?œë„ê°€ ê±°ì˜ ?Œì§„?˜ì—ˆ?µë‹ˆ??")
                    break
                    
            except Exception as e:
                logger.error(f"?„ì›??'{committee}' ê²€??ì¤??¤ë¥˜: {e}")
                break
        
        logger.info(f"?„ì›??'{committee}'ë¡?{len(decisions)}ê±??˜ì§‘")
        return decisions
    
    def collect_decision_details(self, decision: Dict[str, Any], committee: str) -> Optional[Dict[str, Any]]:
        """?„ì›?Œê²°?•ë¬¸ ?ì„¸ ?•ë³´ ?˜ì§‘"""
        decision_id = decision.get('?ë??¼ë ¨ë²ˆí˜¸')
        if not decision_id:
            return None
        
        committee_code = COMMITTEE_CODES.get(committee)
        if not committee_code:
            return None
        
        try:
            detail = self.client.get_committee_decision_detail(
                committee=committee_code, 
                decision_id=decision_id
            )
            if detail:
                # ê¸°ë³¸ ?•ë³´?€ ?ì„¸ ?•ë³´ ê²°í•©
                combined_data = {
                    'basic_info': decision,
                    'detail_info': detail,
                    'committee': committee,
                    'collected_at': datetime.now().isoformat()
                }
                return combined_data
        except Exception as e:
            logger.error(f"?„ì›?Œê²°?•ë¬¸ {decision_id} ?ì„¸ ?•ë³´ ?˜ì§‘ ?¤íŒ¨: {e}")
        
        return None
    
    def save_decision_data(self, decision_data: Dict[str, Any], filename: str):
        """?„ì›?Œê²°?•ë¬¸ ?°ì´?°ë? ?Œì¼ë¡??€??""
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(decision_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"?„ì›?Œê²°?•ë¬¸ ?°ì´???€?? {filepath}")
        except Exception as e:
            logger.error(f"?„ì›?Œê²°?•ë¬¸ ?°ì´???€???¤íŒ¨: {e}")
    
    def collect_all_committee_decisions(self, target_count: int = 500):
        """ëª¨ë“  ?„ì›?Œê²°?•ë¬¸ ?˜ì§‘"""
        logger.info(f"?„ì›?Œê²°?•ë¬¸ ?˜ì§‘ ?œì‘ (ëª©í‘œ: {target_count}ê±?...")
        
        all_decisions = []
        max_per_committee = target_count // len(COMMITTEE_KEYWORDS)
        
        for committee in COMMITTEE_KEYWORDS.keys():
            if len(all_decisions) >= target_count:
                break
                
            try:
                decisions = self.collect_decisions_by_committee(committee, max_per_committee)
                all_decisions.extend(decisions)
                logger.info(f"?„ì›??'{committee}' ?„ë£Œ. ?„ì : {len(all_decisions)}ê±?)
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 100:
                    logger.warning("API ?”ì²­ ?œë„ê°€ ë¶€ì¡±í•©?ˆë‹¤.")
                    break
                    
            except Exception as e:
                logger.error(f"?„ì›??'{committee}' ê²€???¤íŒ¨: {e}")
                continue
        
        logger.info(f"ì´?{len(all_decisions)}ê±´ì˜ ?„ì›?Œê²°?•ë¬¸ ëª©ë¡ ?˜ì§‘ ?„ë£Œ")
        
        # ê°??„ì›?Œê²°?•ë¬¸???ì„¸ ?•ë³´ ?˜ì§‘
        detailed_decisions = []
        for i, decision in enumerate(all_decisions):
            if i >= target_count:
                break
                
            try:
                # ?„ì›???•ë³´ ì¶”ì¶œ (ê¸°ë³¸ ?•ë³´?ì„œ)
                committee = decision.get('?Œê?ë¶€ì²˜ëª…', 'ê¸°í?')
                if committee not in COMMITTEE_KEYWORDS:
                    committee = 'ê¸°í?'
                
                detail = self.collect_decision_details(decision, committee)
                if detail:
                    detailed_decisions.append(detail)
                    
                    # ê°œë³„ ?Œì¼ë¡??€??
                    decision_id = decision.get('?ë??¼ë ¨ë²ˆí˜¸', f'unknown_{i}')
                    filename = f"committee_decision_{decision_id}_{datetime.now().strftime('%Y%m%d')}.json"
                    self.save_decision_data(detail, filename)
                
                # ì§„í–‰ë¥?ë¡œê·¸
                if (i + 1) % 50 == 0:
                    logger.info(f"?ì„¸ ?•ë³´ ?˜ì§‘ ì§„í–‰ë¥? {i + 1}/{len(all_decisions)}")
                
                # API ?”ì²­ ?œí•œ ?•ì¸
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API ?”ì²­ ?œë„ê°€ ê±°ì˜ ?Œì§„?˜ì—ˆ?µë‹ˆ??")
                    break
                    
            except Exception as e:
                logger.error(f"?„ì›?Œê²°?•ë¬¸ {i} ?ì„¸ ?•ë³´ ?˜ì§‘ ?¤íŒ¨: {e}")
                continue
        
        logger.info(f"?„ì›?Œê²°?•ë¬¸ ?ì„¸ ?•ë³´ ?˜ì§‘ ?„ë£Œ: {len(detailed_decisions)}ê±?)
        
        # ?˜ì§‘ ê²°ê³¼ ?”ì•½ ?ì„±
        self.generate_collection_summary(detailed_decisions)
    
    def generate_collection_summary(self, decisions: List[Dict[str, Any]]):
        """?˜ì§‘ ê²°ê³¼ ?”ì•½ ?ì„±"""
        # ?„ì›?Œë³„ ?µê³„
        committee_stats = {}
        
        for decision in decisions:
            committee = decision.get('committee', 'ê¸°í?')
            committee_stats[committee] = committee_stats.get(committee, 0) + 1
        
        summary = {
            'collection_date': datetime.now().isoformat(),
            'total_decisions': len(decisions),
            'committee_distribution': committee_stats,
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
        logger.info("?¬ìš©ë²? LAW_OPEN_API_OC=your_email_id python collect_committee_decisions.py")
        return
    
    # ë¡œê·¸ ?”ë ‰? ë¦¬ ?ì„±
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # API ?¤ì •
    config = LawOpenAPIConfig(oc=oc)
    
    # ?„ì›?Œê²°?•ë¬¸ ?˜ì§‘ ?¤í–‰
    collector = CommitteeDecisionCollector(config)
    collector.collect_all_committee_decisions(target_count=500)


if __name__ == "__main__":
    main()
