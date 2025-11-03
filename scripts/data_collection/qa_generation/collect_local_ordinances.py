#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?ì¹˜ë²•ê·œ ?°ì´???˜ì§‘ ?¤í¬ë¦½íŠ¸ (êµ??ë²•ë ¹?•ë³´?¼í„° OpenAPI ê¸°ë°˜)

???¤í¬ë¦½íŠ¸??êµ??ë²•ë ¹?•ë³´?¼í„°??OpenAPIë¥??µí•´ ?ì¹˜ë²•ê·œ ?°ì´?°ë? ?˜ì§‘?©ë‹ˆ??
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient, LawOpenAPIConfig
from source.data.data_processor import DataProcessor

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/local_ordinance_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LocalOrdinanceCollector:
    """?ì¹˜ë²•ê·œ ?°ì´???˜ì§‘ ?´ë˜??""
    
    def __init__(self):
        self.config = APIConfig()
        self.client = LawOpenAPIClient(self.config)
        self.data_processor = DataProcessor()
        
        # ?˜ì§‘ ëª©í‘œ ?¤ì •
        self.target_ordinances = 500  # ?ì¹˜ë²•ê·œ 500ê±?
        
        # ?°ì´???€???”ë ‰? ë¦¬ ?ì„±
        self.raw_data_dir = Path("data/raw/local_ordinances")
        self.processed_data_dir = Path("data/processed/local_ordinances")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_local_ordinances(self, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """?ì¹˜ë²•ê·œ ?°ì´???˜ì§‘"""
        logger.info("?ì¹˜ë²•ê·œ ?°ì´???˜ì§‘ ?œì‘")
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y%m%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        
        all_ordinances = []
        page = 1
        per_page = 50
        
        while len(all_ordinances) < self.target_ordinances:
            logger.info(f"?ì¹˜ë²•ê·œ ?˜ì§‘ ì¤?.. (?˜ì´ì§€ {page}, ?„ì¬ {len(all_ordinances)}ê±?")
            
            response = self.client.get_local_ordinance_list(page=page, per_page=per_page)
            
            if not response:
                logger.error(f"?˜ì´ì§€ {page} ?˜ì§‘ ?¤íŒ¨")
                break
            
            # ?‘ë‹µ?ì„œ ?ì¹˜ë²•ê·œ ëª©ë¡ ì¶”ì¶œ
            ordinances = response.get('localOrdinanceList', {}).get('localOrdinance', [])
            if not ordinances:
                logger.info("???´ìƒ ?˜ì§‘???ì¹˜ë²•ê·œê°€ ?†ìŠµ?ˆë‹¤.")
                break
            
            # ?¨ì¼ ?ì¹˜ë²•ê·œ??ê²½ìš° ë¦¬ìŠ¤?¸ë¡œ ë³€??
            if isinstance(ordinances, dict):
                ordinances = [ordinances]
            
            # ê°??ì¹˜ë²•ê·œ???ì„¸ ?•ë³´ ?˜ì§‘
            for ordinance in ordinances:
                if len(all_ordinances) >= self.target_ordinances:
                    break
                
                ordinance_id = ordinance.get('id')
                if ordinance_id:
                    detail = self.client.get_local_ordinance_detail(ordinance_id)
                    if detail:
                        detail['category'] = 'local_ordinance'
                        all_ordinances.append(detail)
                        
                        # ?ë³¸ ?°ì´???€??
                        self._save_raw_data(detail, f"local_ordinance_{ordinance_id}")
            
            page += 1
            
            # API ?”ì²­ ?œí•œ ?•ì¸
            stats = self.client.get_request_stats()
            if stats['remaining_requests'] <= 10:
                logger.warning("API ?”ì²­ ?œë„??ê·¼ì ‘?ˆìŠµ?ˆë‹¤. ?˜ì§‘??ì¤‘ë‹¨?©ë‹ˆ??")
                break
        
        logger.info(f"?ì¹˜ë²•ê·œ {len(all_ordinances)}ê±??˜ì§‘ ?„ë£Œ")
        return all_ordinances
    
    def _save_raw_data(self, data: Dict[str, Any], filename: str):
        """?ë³¸ ?°ì´???€??""
        file_path = self.raw_data_dir / f"{filename}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"?ë³¸ ?°ì´???€?? {file_path}")
    
    def process_collected_data(self, ordinances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """?˜ì§‘???°ì´???„ì²˜ë¦?""
        logger.info("?˜ì§‘???ì¹˜ë²•ê·œ ?°ì´???„ì²˜ë¦??œì‘")
        
        processed_ordinances = []
        
        for ordinance in ordinances:
            try:
                # ?°ì´???•ì œ ë°?êµ¬ì¡°??
                processed_ordinance = self.data_processor.process_local_ordinance_data(ordinance)
                processed_ordinances.append(processed_ordinance)
                
            except Exception as e:
                logger.error(f"?ì¹˜ë²•ê·œ ?°ì´???„ì²˜ë¦??¤íŒ¨: {e}")
                continue
        
        # ?„ì²˜ë¦¬ëœ ?°ì´???€??
        self._save_processed_data(processed_ordinances)
        
        logger.info(f"?ì¹˜ë²•ê·œ ?°ì´??{len(processed_ordinances)}ê±??„ì²˜ë¦??„ë£Œ")
        return processed_ordinances
    
    def _save_processed_data(self, data: List[Dict[str, Any]]):
        """?„ì²˜ë¦¬ëœ ?°ì´???€??""
        file_path = self.processed_data_dir / "processed_local_ordinances.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"?„ì²˜ë¦¬ëœ ?°ì´???€?? {file_path}")
    
    def generate_collection_report(self, ordinances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """?˜ì§‘ ê²°ê³¼ ë³´ê³ ???ì„±"""
        report = {
            "collection_date": datetime.now().isoformat(),
            "total_ordinances": len(ordinances),
            "api_requests_used": self.client.get_request_stats()['request_count'],
            "collection_summary": {
                "successful_collections": len([o for o in ordinances if o.get('status') == 'success']),
                "failed_collections": len([o for o in ordinances if o.get('status') == 'failed']),
            },
            "target_achievement": f"{len(ordinances)}/{self.target_ordinances}",
            "completion_rate": f"{(len(ordinances) / self.target_ordinances) * 100:.1f}%"
        }
        
        # ë³´ê³ ???€??
        report_path = Path("docs/local_ordinance_collection_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# ?ì¹˜ë²•ê·œ ?°ì´???˜ì§‘ ë³´ê³ ??n\n")
            f.write(f"**?˜ì§‘ ?¼ì‹œ**: {report['collection_date']}\n")
            f.write(f"**?˜ì§‘???ì¹˜ë²•ê·œ ??*: {report['total_ordinances']}ê±?n")
            f.write(f"**API ?”ì²­ ??*: {report['api_requests_used']}??n")
            f.write(f"**ëª©í‘œ ?¬ì„±ë¥?*: {report['completion_rate']}\n\n")
            f.write(f"## ?˜ì§‘ ê²°ê³¼ ?”ì•½\n")
            f.write(f"- ?±ê³µ: {report['collection_summary']['successful_collections']}ê±?n")
            f.write(f"- ?¤íŒ¨: {report['collection_summary']['failed_collections']}ê±?n")
            f.write(f"- ëª©í‘œ: {report['target_achievement']}\n")
        
        logger.info(f"?˜ì§‘ ë³´ê³ ???ì„±: {report_path}")
        return report


def main():
    """ë©”ì¸ ?¤í–‰ ?¨ìˆ˜"""
    logger.info("?ì¹˜ë²•ê·œ ?°ì´???˜ì§‘ ?¤í¬ë¦½íŠ¸ ?œì‘")
    
    try:
        # ?˜ì§‘ê¸?ì´ˆê¸°??
        collector = LocalOrdinanceCollector()
        
        # ?ì¹˜ë²•ê·œ ?˜ì§‘
        ordinances = collector.collect_local_ordinances()
        
        # ?°ì´???„ì²˜ë¦?
        processed_ordinances = collector.process_collected_data(ordinances)
        
        # ?˜ì§‘ ë³´ê³ ???ì„±
        report = collector.generate_collection_report(processed_ordinances)
        
        logger.info("?ì¹˜ë²•ê·œ ?°ì´???˜ì§‘ ?„ë£Œ")
        logger.info(f"?˜ì§‘???ì¹˜ë²•ê·œ ?? {len(processed_ordinances)}ê±?)
        logger.info(f"API ?”ì²­ ?? {report['api_requests_used']}??)
        logger.info(f"ëª©í‘œ ?¬ì„±ë¥? {report['completion_rate']}")
        
    except Exception as e:
        logger.error(f"?ì¹˜ë²•ê·œ ?°ì´???˜ì§‘ ì¤??¤ë¥˜ ë°œìƒ: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
