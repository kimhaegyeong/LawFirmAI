#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?‰ì •ê·œì¹™ ?°ì´???˜ì§‘ ?¤í¬ë¦½íŠ¸ (êµ??ë²•ë ¹?•ë³´?¼í„° OpenAPI ê¸°ë°˜)

???¤í¬ë¦½íŠ¸??êµ??ë²•ë ¹?•ë³´?¼í„°??OpenAPIë¥??µí•´ ?‰ì •ê·œì¹™ ?°ì´?°ë? ?˜ì§‘?©ë‹ˆ??
"""

import os
import sys
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ?”ë ‰? ë¦¬ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient, LawOpenAPIConfig
from source.data.data_processor import LegalDataProcessor

# ë¡œê¹… ?¤ì •
def setup_logging():
    """ë¡œê¹… ?¤ì • ?¨ìˆ˜"""
    # logs ?”ë ‰? ë¦¬ ?ì„±
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # ë¡œê·¸ ?¬ë§· ?¤ì •
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    
    # ?Œì¼ ?¸ë“¤???¤ì •
    file_handler = logging.FileHandler(
        logs_dir / 'administrative_rule_collection.log',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # ì½˜ì†” ?¸ë“¤???¤ì •
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # ë£¨íŠ¸ ë¡œê±° ?¤ì •
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

logger = setup_logging()


class AdministrativeRuleCollector:
    """?‰ì •ê·œì¹™ ?°ì´???˜ì§‘ ?´ë˜??""
    
    def __init__(self):
        self.config = LawOpenAPIConfig()
        self.client = LawOpenAPIClient(self.config)
        self.data_processor = LegalDataProcessor()
        
        # ?˜ì§‘ ëª©í‘œ ?¤ì •
        self.target_rules = 1000  # ?‰ì •ê·œì¹™ 1,000ê±?
        
        # ?°ì´???€???”ë ‰? ë¦¬ ?ì„±
        self.raw_data_dir = Path("data/raw/administrative_rules")
        self.processed_data_dir = Path("data/processed/administrative_rules")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_administrative_rules(self, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """?‰ì •ê·œì¹™ ?°ì´???˜ì§‘"""
        logger.info("=" * 60)
        logger.info("?‰ì •ê·œì¹™ ?°ì´???˜ì§‘ ?œì‘")
        logger.info(f"ëª©í‘œ ?˜ì§‘ ê±´ìˆ˜: {self.target_rules:,}ê±?)
        logger.info(f"?œì‘ ?œê°„: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y%m%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        
        logger.info(f"?˜ì§‘ ê¸°ê°„: {start_date} ~ {end_date}")
        
        all_rules = []
        page = 1
        display = 50
        failed_requests = 0
        successful_requests = 0
        start_time = datetime.now()
        
        while len(all_rules) < self.target_rules:
            logger.info(f"?“„ ?˜ì´ì§€ {page} ?˜ì§‘ ?œì‘ (?„ì¬ ?˜ì§‘: {len(all_rules):,}/{self.target_rules:,}ê±?")
            
            try:
                response = self.client.get_administrative_rule_list(page=page, display=display)
                successful_requests += 1
                
                if not response:
                    logger.error(f"???˜ì´ì§€ {page} API ?‘ë‹µ??ë¹„ì–´?ˆìŠµ?ˆë‹¤.")
                    failed_requests += 1
                    if failed_requests >= 3:
                        logger.error("?°ì† 3???¤íŒ¨ë¡??˜ì§‘??ì¤‘ë‹¨?©ë‹ˆ??")
                        break
                    page += 1
                    continue
                
                # ?‘ë‹µ?ì„œ ?‰ì •ê·œì¹™ ëª©ë¡ ì¶”ì¶œ
                if isinstance(response, list) and len(response) > 0:
                    # API ?‘ë‹µ??ë¦¬ìŠ¤???•íƒœ??ê²½ìš°
                    api_response = response[0]
                    if 'AdmRulSearch' in api_response:
                        search_result = api_response['AdmRulSearch']
                        rules = search_result.get('admrul', [])
                        total_count = search_result.get('totalCnt', '0')
                        try:
                            total_count_int = int(total_count)
                            logger.info(f"?“Š ì´??‰ì •ê·œì¹™ ?? {total_count_int:,}ê±?)
                        except (ValueError, TypeError):
                            logger.info(f"?“Š ì´??‰ì •ê·œì¹™ ?? {total_count}ê±?)
                    else:
                        rules = []
                else:
                    rules = []
                
                if not rules:
                    logger.info("?“­ ???´ìƒ ?˜ì§‘???‰ì •ê·œì¹™???†ìŠµ?ˆë‹¤.")
                    break
                
                # ?¨ì¼ ê·œì¹™??ê²½ìš° ë¦¬ìŠ¤?¸ë¡œ ë³€??
                if isinstance(rules, dict):
                    rules = [rules]
                
                logger.info(f"?“‹ ?˜ì´ì§€ {page}?ì„œ {len(rules)}ê°?ê·œì¹™ ë°œê²¬")
                
                # ê°?ê·œì¹™???ì„¸ ?•ë³´ ?˜ì§‘
                page_success_count = 0
                for i, rule in enumerate(rules, 1):
                    if len(all_rules) >= self.target_rules:
                        logger.info(f"?¯ ëª©í‘œ ?˜ì§‘ ê±´ìˆ˜ {self.target_rules:,}ê±??¬ì„±!")
                        break
                    
                    rule_id = rule.get('id')
                    rule_name = rule.get('name', 'Unknown')
                    
                    if rule_id:
                        logger.debug(f"  ?“ ê·œì¹™ {i}/{len(rules)}: {rule_name} (ID: {rule_id}) ?ì„¸ ?•ë³´ ?˜ì§‘ ì¤?..")
                        
                        try:
                            detail = self.client.get_administrative_rule_detail(rule_id)
                            if detail:
                                detail['category'] = 'administrative_rule'
                                all_rules.append(detail)
                                page_success_count += 1
                                
                                # ?ë³¸ ?°ì´???€??
                                self._save_raw_data(detail, f"administrative_rule_{rule_id}")
                                
                                logger.debug(f"  ??ê·œì¹™ {rule_name} ?˜ì§‘ ?„ë£Œ")
                            else:
                                logger.warning(f"  ? ï¸ ê·œì¹™ {rule_name} (ID: {rule_id}) ?ì„¸ ?•ë³´ ?˜ì§‘ ?¤íŒ¨")
                        except Exception as e:
                            logger.error(f"  ??ê·œì¹™ {rule_name} (ID: {rule_id}) ?˜ì§‘ ì¤??¤ë¥˜: {e}")
                    else:
                        logger.warning(f"  ? ï¸ ê·œì¹™ {i}/{len(rules)}: IDê°€ ?†ìŠµ?ˆë‹¤.")
                
                logger.info(f"?“Š ?˜ì´ì§€ {page} ?˜ì§‘ ê²°ê³¼: {page_success_count}ê±??±ê³µ")
                
                # ì§„í–‰ë¥?ê³„ì‚°
                progress = (len(all_rules) / self.target_rules) * 100
                elapsed_time = datetime.now() - start_time
                estimated_total_time = elapsed_time * (self.target_rules / len(all_rules)) if all_rules else None
                remaining_time = estimated_total_time - elapsed_time if estimated_total_time else None
                
                logger.info(f"?“ˆ ì§„í–‰ë¥? {progress:.1f}% ({len(all_rules):,}/{self.target_rules:,}ê±?")
                logger.info(f"?±ï¸ ê²½ê³¼ ?œê°„: {elapsed_time}")
                if remaining_time:
                    logger.info(f"???ˆìƒ ?¨ì? ?œê°„: {remaining_time}")
                
            except Exception as e:
                logger.error(f"???˜ì´ì§€ {page} ?˜ì§‘ ì¤??ˆì™¸ ë°œìƒ: {e}")
                failed_requests += 1
                if failed_requests >= 3:
                    logger.error("?°ì† 3???¤íŒ¨ë¡??˜ì§‘??ì¤‘ë‹¨?©ë‹ˆ??")
                    break
            
            page += 1
            
            # API ?”ì²­ ?œí•œ ?•ì¸
            try:
                stats = self.client.get_request_stats()
                remaining = stats.get('remaining_requests', 0)
                logger.debug(f"?”¢ API ?”ì²­ ?”ì—¬?? {remaining}??)
                
                if remaining <= 10:
                    logger.warning("? ï¸ API ?”ì²­ ?œë„??ê·¼ì ‘?ˆìŠµ?ˆë‹¤. ?˜ì§‘??ì¤‘ë‹¨?©ë‹ˆ??")
                    break
            except Exception as e:
                logger.warning(f"? ï¸ API ?”ì²­ ?µê³„ ?•ì¸ ?¤íŒ¨: {e}")
            
            # ?”ì²­ ê°??€ê¸?(API ë¶€??ë°©ì?)
            time.sleep(0.5)
        
        # ìµœì¢… ?˜ì§‘ ê²°ê³¼ ë¡œê¹…
        total_time = datetime.now() - start_time
        logger.info("=" * 60)
        logger.info("?‰ì •ê·œì¹™ ?°ì´???˜ì§‘ ?„ë£Œ")
        logger.info(f"?“Š ìµœì¢… ?˜ì§‘ ê²°ê³¼:")
        logger.info(f"  - ?˜ì§‘??ê·œì¹™ ?? {len(all_rules):,}ê±?)
        logger.info(f"  - ëª©í‘œ ?€ë¹??¬ì„±ë¥? {(len(all_rules) / self.target_rules) * 100:.1f}%")
        logger.info(f"  - ?±ê³µ??API ?”ì²­: {successful_requests}??)
        logger.info(f"  - ?¤íŒ¨??API ?”ì²­: {failed_requests}??)
        logger.info(f"  - ì´??Œìš” ?œê°„: {total_time}")
        logger.info(f"  - ?‰ê·  ?˜ì§‘ ?ë„: {len(all_rules) / total_time.total_seconds() * 60:.1f}ê±?ë¶?)
        logger.info("=" * 60)
        
        return all_rules
    
    def _save_raw_data(self, data: Dict[str, Any], filename: str):
        """?ë³¸ ?°ì´???€??""
        file_path = self.raw_data_dir / f"{filename}.json"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"?’¾ ?ë³¸ ?°ì´???€???„ë£Œ: {file_path}")
        except Exception as e:
            logger.error(f"???ë³¸ ?°ì´???€???¤íŒ¨ ({file_path}): {e}")
            raise
    
    def process_collected_data(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """?˜ì§‘???°ì´???„ì²˜ë¦?""
        logger.info("=" * 60)
        logger.info("?‰ì •ê·œì¹™ ?°ì´???„ì²˜ë¦??œì‘")
        logger.info(f"?„ì²˜ë¦??€?? {len(rules):,}ê±?)
        logger.info("=" * 60)
        
        processed_rules = []
        failed_count = 0
        start_time = datetime.now()
        
        for i, rule in enumerate(rules, 1):
            try:
                # ?°ì´???•ì œ ë°?êµ¬ì¡°??
                processed_rule = self.data_processor.process_administrative_rule_data(rule)
                processed_rules.append(processed_rule)
                
                if i % 100 == 0:
                    logger.info(f"?“Š ?„ì²˜ë¦?ì§„í–‰ë¥? {i:,}/{len(rules):,}ê±?({i/len(rules)*100:.1f}%)")
                
            except Exception as e:
                logger.error(f"???‰ì •ê·œì¹™ ?°ì´???„ì²˜ë¦??¤íŒ¨ (ê·œì¹™ {i}): {e}")
                failed_count += 1
                continue
        
        # ?„ì²˜ë¦¬ëœ ?°ì´???€??
        self._save_processed_data(processed_rules)
        
        total_time = datetime.now() - start_time
        success_rate = ((len(rules) - failed_count) / len(rules)) * 100 if rules else 0
        
        logger.info("=" * 60)
        logger.info("?‰ì •ê·œì¹™ ?°ì´???„ì²˜ë¦??„ë£Œ")
        logger.info(f"?“Š ?„ì²˜ë¦?ê²°ê³¼:")
        logger.info(f"  - ?±ê³µ: {len(processed_rules):,}ê±?)
        logger.info(f"  - ?¤íŒ¨: {failed_count:,}ê±?)
        logger.info(f"  - ?±ê³µë¥? {success_rate:.1f}%")
        logger.info(f"  - ?Œìš” ?œê°„: {total_time}")
        logger.info("=" * 60)
        
        return processed_rules
    
    def _save_processed_data(self, data: List[Dict[str, Any]]):
        """?„ì²˜ë¦¬ëœ ?°ì´???€??""
        file_path = self.processed_data_dir / "processed_administrative_rules.json"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB ?¨ìœ„
            logger.info(f"?’¾ ?„ì²˜ë¦¬ëœ ?°ì´???€???„ë£Œ: {file_path}")
            logger.info(f"?“ ?Œì¼ ?¬ê¸°: {file_size:.2f} MB")
        except Exception as e:
            logger.error(f"???„ì²˜ë¦¬ëœ ?°ì´???€???¤íŒ¨ ({file_path}): {e}")
            raise
    
    def generate_collection_report(self, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """?˜ì§‘ ê²°ê³¼ ë³´ê³ ???ì„±"""
        logger.info("?“‹ ?˜ì§‘ ê²°ê³¼ ë³´ê³ ???ì„± ì¤?..")
        
        try:
            api_stats = self.client.get_request_stats()
            api_requests_used = api_stats.get('request_count', 0)
        except Exception as e:
            logger.warning(f"? ï¸ API ?µê³„ ì¡°íšŒ ?¤íŒ¨: {e}")
            api_requests_used = 0
        
        report = {
            "collection_date": datetime.now().isoformat(),
            "total_rules": len(rules),
            "api_requests_used": api_requests_used,
            "collection_summary": {
                "successful_collections": len([r for r in rules if r.get('status') == 'success']),
                "failed_collections": len([r for r in rules if r.get('status') == 'failed']),
            },
            "target_achievement": f"{len(rules)}/{self.target_rules}",
            "completion_rate": f"{(len(rules) / self.target_rules) * 100:.1f}%"
        }
        
        # ë³´ê³ ???€??
        report_path = Path("docs/administrative_rule_collection_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# ?‰ì •ê·œì¹™ ?°ì´???˜ì§‘ ë³´ê³ ??n\n")
                f.write(f"**?˜ì§‘ ?¼ì‹œ**: {report['collection_date']}\n")
                f.write(f"**?˜ì§‘??ê·œì¹™ ??*: {report['total_rules']:,}ê±?n")
                f.write(f"**API ?”ì²­ ??*: {report['api_requests_used']:,}??n")
                f.write(f"**ëª©í‘œ ?¬ì„±ë¥?*: {report['completion_rate']}\n\n")
                f.write(f"## ?˜ì§‘ ê²°ê³¼ ?”ì•½\n")
                f.write(f"- ?±ê³µ: {report['collection_summary']['successful_collections']:,}ê±?n")
                f.write(f"- ?¤íŒ¨: {report['collection_summary']['failed_collections']:,}ê±?n")
                f.write(f"- ëª©í‘œ: {report['target_achievement']}\n\n")
                f.write(f"## ?ì„¸ ?µê³„\n")
                f.write(f"- ?˜ì§‘ ?œì‘ ?œê°„: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- ?˜ì§‘ ?„ë£Œ ?œê°„: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- ?‰ê·  ?˜ì§‘ ?ë„: {len(rules) / max(api_requests_used, 1) * 60:.1f}ê±?ë¶?n")
            
            logger.info(f"?“„ ?˜ì§‘ ë³´ê³ ???ì„± ?„ë£Œ: {report_path}")
        except Exception as e:
            logger.error(f"???˜ì§‘ ë³´ê³ ???ì„± ?¤íŒ¨: {e}")
        
        return report


def main():
    """ë©”ì¸ ?¤í–‰ ?¨ìˆ˜"""
    logger.info("?? ?‰ì •ê·œì¹™ ?°ì´???˜ì§‘ ?¤í¬ë¦½íŠ¸ ?œì‘")
    logger.info(f"?“… ?¤í–‰ ?œê°„: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # ?˜ê²½ë³€???¤ì •
        # if not os.getenv("LAW_OPEN_API_OC"):
        #     os.environ["LAW_OPEN_API_OC"] = "OC"
        #     logger.info("?”§ ?˜ê²½ë³€??LAW_OPEN_API_OC ?¤ì •: OC")
        
        # ?˜ì§‘ê¸?ì´ˆê¸°??
        logger.info("?”§ ?˜ì§‘ê¸?ì´ˆê¸°??ì¤?..")
        collector = AdministrativeRuleCollector()
        logger.info("???˜ì§‘ê¸?ì´ˆê¸°???„ë£Œ")
        
        # ?‰ì •ê·œì¹™ ?˜ì§‘
        logger.info("?“¥ ?‰ì •ê·œì¹™ ?°ì´???˜ì§‘ ?œì‘")
        rules = collector.collect_administrative_rules()
        
        # ?°ì´???„ì²˜ë¦?
        logger.info("?”„ ?°ì´???„ì²˜ë¦??œì‘")
        processed_rules = collector.process_collected_data(rules)
        
        # ?˜ì§‘ ë³´ê³ ???ì„±
        logger.info("?“Š ?˜ì§‘ ë³´ê³ ???ì„± ?œì‘")
        report = collector.generate_collection_report(processed_rules)
        
        # ìµœì¢… ê²°ê³¼ ì¶œë ¥
        logger.info("=" * 60)
        logger.info("?‰ ?‰ì •ê·œì¹™ ?°ì´???˜ì§‘ ?„ë£Œ!")
        logger.info(f"?“Š ìµœì¢… ê²°ê³¼:")
        logger.info(f"  - ?˜ì§‘??ê·œì¹™ ?? {len(processed_rules):,}ê±?)
        logger.info(f"  - API ?”ì²­ ?? {report['api_requests_used']:,}??)
        logger.info(f"  - ëª©í‘œ ?¬ì„±ë¥? {report['completion_rate']}")
        logger.info(f"  - ?ë³¸ ?°ì´???€???„ì¹˜: {collector.raw_data_dir}")
        logger.info(f"  - ?„ì²˜ë¦??°ì´???€???„ì¹˜: {collector.processed_data_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"???‰ì •ê·œì¹™ ?°ì´???˜ì§‘ ì¤??¤ë¥˜ ë°œìƒ: {e}")
        logger.error(f"?¤ë¥˜ ?ì„¸: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
