#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?∞Ïù¥???òÏßë ?ÑÏö© ?§ÌÅ¨Î¶ΩÌä∏ (JSON ?Ä??

Íµ??Î≤ïÎ†π?ïÎ≥¥?ºÌÑ∞ OpenAPI?êÏÑú Î≤ïÎ•† ?∞Ïù¥?∞Î? ?òÏßë?òÏó¨ JSON ?åÏùºÎ°??Ä?•Ìï©?àÎã§.
Î≤°ÌÑ∞DB Íµ¨Ï∂ï?Ä Î≥ÑÎèÑ???§ÌÅ¨Î¶ΩÌä∏?êÏÑú Ï≤òÎ¶¨?©Îãà??

?¨Ïö©Î≤?
    python scripts/collect_data_only.py --mode collect --oc your_email_id
    python scripts/collect_data_only.py --mode laws --oc your_email_id --query "ÎØºÎ≤ï"
    python scripts/collect_data_only.py --mode precedents --oc your_email_id --query "Í≥ÑÏïΩ ?¥Ï?"
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# ?ÑÎ°ú?ùÌä∏ Î£®Ìä∏ ?îÎ†â?†Î¶¨Î•?Python Í≤ΩÎ°ú??Ï∂îÍ?
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient, LawOpenAPIConfig
from source.utils.logger import get_logger

logger = get_logger(__name__)


class DataCollector:
    """?∞Ïù¥???òÏßë ?ÑÏö© ?¥Îûò??""
    
    def __init__(self, oc: str, base_url: str = "http://www.law.go.kr/DRF"):
        """Ï¥àÍ∏∞??""
        self.oc = oc
        self.config = LawOpenAPIConfig(oc=oc, base_url=base_url)
        self.client = LawOpenAPIClient(self.config)
        self.output_dir = Path("./data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ?òÏßë ?µÍ≥Ñ
        self.collection_stats = {
            'start_time': datetime.now().isoformat(),
            'laws_collected': 0,
            'precedents_collected': 0,
            'constitutional_decisions_collected': 0,
            'legal_interpretations_collected': 0,
            'administrative_rules_collected': 0,
            'local_ordinances_collected': 0,
            'committee_decisions_collected': 0,
            'administrative_appeals_collected': 0,
            'treaties_collected': 0,
            'total_documents': 0,
            'errors': []
        }
    
    def _save_data(self, data: List[Dict], doc_type: str, query: str = None) -> str:
        """?∞Ïù¥?∞Î? JSON ?åÏùºÎ°??Ä??""
        if not data:
            return None
        
        timestamp = int(time.time())
        query_suffix = f"_{query.replace(' ', '_')}" if query else ""
        filename = f"{doc_type}{query_suffix}_{timestamp}.json"
        file_path = self.output_dir / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Saved {len(data)} {doc_type} documents to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save {doc_type} data: {e}")
            return None
    
    def collect_laws(self, query: str = "ÎØºÎ≤ï", display: int = 100) -> bool:
        """Î≤ïÎ†π ?∞Ïù¥???òÏßë"""
        try:
            logger.info(f"Collecting laws for query: {query}")
            
            # Î≤ïÎ†π Î™©Î°ù Ï°∞Ìöå
            law_list = self.client.search_law_list_effective(query=query, display=display)
            if not law_list:
                logger.warning(f"No laws found for query: {query}")
                return False
            
            detailed_laws = []
            for law_summary in law_list:
                try:
                    law_id = law_summary.get('Î≤ïÎ†πID')
                    if law_id:
                        # Î≤ïÎ†π ?ÅÏÑ∏ ?ïÎ≥¥ Ï°∞Ìöå
                        law_detail = self.client.get_law_detail_effective(law_id=law_id)
                        if law_detail:
                            # ?µÌï© ?∞Ïù¥???ùÏÑ±
                            integrated_law = {
                                **law_summary,
                                **law_detail,
                                "document_type": "law",
                                "collected_at": datetime.now().isoformat()
                            }
                            detailed_laws.append(integrated_law)
                            
                except Exception as e:
                    logger.error(f"Error collecting law {law_summary.get('Î≤ïÎ†πID', 'unknown')}: {e}")
                    continue
            
            if detailed_laws:
                file_path = self._save_data(detailed_laws, "laws", query)
                if file_path:
                    self.collection_stats['laws_collected'] += len(detailed_laws)
                    self.collection_stats['total_documents'] += len(detailed_laws)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error collecting laws: {e}")
            self.collection_stats['errors'].append(f"Laws collection error: {e}")
            return False
    
    def collect_precedents(self, query: str = "Í≥ÑÏïΩ ?¥Ï?", display: int = 100) -> bool:
        """?êÎ? ?∞Ïù¥???òÏßë"""
        try:
            logger.info(f"Collecting precedents for query: {query}")
            
            # ?êÎ? Î™©Î°ù Ï°∞Ìöå
            prec_list = self.client.search_precedent_list(query=query, display=display)
            if not prec_list:
                logger.warning(f"No precedents found for query: {query}")
                return False
            
            detailed_precedents = []
            for prec_summary in prec_list:
                try:
                    prec_id = prec_summary.get('?êÎ??ºÎ†®Î≤àÌò∏')
                    if prec_id:
                        # ?êÎ? ?ÅÏÑ∏ ?ïÎ≥¥ Ï°∞Ìöå
                        prec_detail = self.client.get_precedent_detail(precedent_id=prec_id)
                        if prec_detail:
                            # ?µÌï© ?∞Ïù¥???ùÏÑ±
                            integrated_prec = {
                                **prec_summary,
                                **prec_detail,
                                "document_type": "precedent",
                                "collected_at": datetime.now().isoformat()
                            }
                            detailed_precedents.append(integrated_prec)
                            
                except Exception as e:
                    logger.error(f"Error collecting precedent {prec_summary.get('?êÎ??ºÎ†®Î≤àÌò∏', 'unknown')}: {e}")
                    continue
            
            if detailed_precedents:
                file_path = self._save_data(detailed_precedents, "precedents", query)
                if file_path:
                    self.collection_stats['precedents_collected'] += len(detailed_precedents)
                    self.collection_stats['total_documents'] += len(detailed_precedents)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error collecting precedents: {e}")
            self.collection_stats['errors'].append(f"Precedents collection error: {e}")
            return False
    
    def collect_constitutional_decisions(self, query: str = "?åÎ≤ï", display: int = 100) -> bool:
        """?åÏû¨Í≤∞Ï†ïÎ°Ä ?∞Ïù¥???òÏßë"""
        try:
            logger.info(f"Collecting constitutional decisions for query: {query}")
            
            # ?åÏû¨Í≤∞Ï†ïÎ°Ä Î™©Î°ù Ï°∞Ìöå
            const_list = self.client.search_constitutional_decision_list(query=query, display=display)
            if not const_list:
                logger.warning(f"No constitutional decisions found for query: {query}")
                return False
            
            detailed_decisions = []
            for const_summary in const_list:
                try:
                    const_id = const_summary.get('?åÏû¨Í≤∞Ï†ïÎ°Ä?ºÎ†®Î≤àÌò∏')
                    if const_id:
                        # ?åÏû¨Í≤∞Ï†ïÎ°Ä ?ÅÏÑ∏ ?ïÎ≥¥ Ï°∞Ìöå
                        const_detail = self.client.get_constitutional_decision_detail(const_id=const_id)
                        if const_detail:
                            # ?µÌï© ?∞Ïù¥???ùÏÑ±
                            integrated_const = {
                                **const_summary,
                                **const_detail,
                                "document_type": "constitutional_decision",
                                "collected_at": datetime.now().isoformat()
                            }
                            detailed_decisions.append(integrated_const)
                            
                except Exception as e:
                    logger.error(f"Error collecting constitutional decision {const_summary.get('?åÏû¨Í≤∞Ï†ïÎ°Ä?ºÎ†®Î≤àÌò∏', 'unknown')}: {e}")
                    continue
            
            if detailed_decisions:
                file_path = self._save_data(detailed_decisions, "constitutional_decisions", query)
                if file_path:
                    self.collection_stats['constitutional_decisions_collected'] += len(detailed_decisions)
                    self.collection_stats['total_documents'] += len(detailed_decisions)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error collecting constitutional decisions: {e}")
            self.collection_stats['errors'].append(f"Constitutional decisions collection error: {e}")
            return False
    
    def collect_legal_interpretations(self, query: str = "Î≤ïÎ†π?¥ÏÑù", display: int = 100) -> bool:
        """Î≤ïÎ†π?¥ÏÑùÎ°Ä ?∞Ïù¥???òÏßë"""
        try:
            logger.info(f"Collecting legal interpretations for query: {query}")
            
            # Î≤ïÎ†π?¥ÏÑùÎ°Ä Î™©Î°ù Ï°∞Ìöå
            interp_list = self.client.search_legal_interpretation_list(query=query, display=display)
            if not interp_list:
                logger.warning(f"No legal interpretations found for query: {query}")
                return False
            
            detailed_interpretations = []
            for interp_summary in interp_list:
                try:
                    interp_id = interp_summary.get('Î≤ïÎ†π?¥ÏÑùÎ°Ä?ºÎ†®Î≤àÌò∏')
                    if interp_id:
                        # Î≤ïÎ†π?¥ÏÑùÎ°Ä ?ÅÏÑ∏ ?ïÎ≥¥ Ï°∞Ìöå
                        interp_detail = self.client.get_legal_interpretation_detail(interp_id=interp_id)
                        if interp_detail:
                            # ?µÌï© ?∞Ïù¥???ùÏÑ±
                            integrated_interp = {
                                **interp_summary,
                                **interp_detail,
                                "document_type": "legal_interpretation",
                                "collected_at": datetime.now().isoformat()
                            }
                            detailed_interpretations.append(integrated_interp)
                            
                except Exception as e:
                    logger.error(f"Error collecting legal interpretation {interp_summary.get('Î≤ïÎ†π?¥ÏÑùÎ°Ä?ºÎ†®Î≤àÌò∏', 'unknown')}: {e}")
                    continue
            
            if detailed_interpretations:
                file_path = self._save_data(detailed_interpretations, "legal_interpretations", query)
                if file_path:
                    self.collection_stats['legal_interpretations_collected'] += len(detailed_interpretations)
                    self.collection_stats['total_documents'] += len(detailed_interpretations)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error collecting legal interpretations: {e}")
            self.collection_stats['errors'].append(f"Legal interpretations collection error: {e}")
            return False
    
    def collect_administrative_rules(self, query: str = "?âÏ†ïÍ∑úÏπô", display: int = 100) -> bool:
        """?âÏ†ïÍ∑úÏπô ?∞Ïù¥???òÏßë"""
        try:
            logger.info(f"Collecting administrative rules for query: {query}")
            
            # ?âÏ†ïÍ∑úÏπô Î™©Î°ù Ï°∞Ìöå
            admin_list = self.client.search_administrative_rule_list(query=query, display=display)
            if not admin_list:
                logger.warning(f"No administrative rules found for query: {query}")
                return False
            
            detailed_rules = []
            for admin_summary in admin_list:
                try:
                    admin_id = admin_summary.get('?âÏ†ïÍ∑úÏπô?ºÎ†®Î≤àÌò∏')
                    if admin_id:
                        # ?âÏ†ïÍ∑úÏπô ?ÅÏÑ∏ ?ïÎ≥¥ Ï°∞Ìöå
                        admin_detail = self.client.get_administrative_rule_detail(rule_id=admin_id)
                        if admin_detail:
                            # ?µÌï© ?∞Ïù¥???ùÏÑ±
                            integrated_admin = {
                                **admin_summary,
                                **admin_detail,
                                "document_type": "administrative_rule",
                                "collected_at": datetime.now().isoformat()
                            }
                            detailed_rules.append(integrated_admin)
                            
                except Exception as e:
                    logger.error(f"Error collecting administrative rule {admin_summary.get('?âÏ†ïÍ∑úÏπô?ºÎ†®Î≤àÌò∏', 'unknown')}: {e}")
                    continue
            
            if detailed_rules:
                file_path = self._save_data(detailed_rules, "administrative_rules", query)
                if file_path:
                    self.collection_stats['administrative_rules_collected'] += len(detailed_rules)
                    self.collection_stats['total_documents'] += len(detailed_rules)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error collecting administrative rules: {e}")
            self.collection_stats['errors'].append(f"Administrative rules collection error: {e}")
            return False
    
    def collect_local_ordinances(self, query: str = "?êÏπòÎ≤ïÍ∑ú", display: int = 100) -> bool:
        """?êÏπòÎ≤ïÍ∑ú ?∞Ïù¥???òÏßë"""
        try:
            logger.info(f"Collecting local ordinances for query: {query}")
            
            # ?êÏπòÎ≤ïÍ∑ú Î™©Î°ù Ï°∞Ìöå
            local_list = self.client.search_local_ordinance_list(query=query, display=display)
            if not local_list:
                logger.warning(f"No local ordinances found for query: {query}")
                return False
            
            detailed_ordinances = []
            for local_summary in local_list:
                try:
                    local_id = local_summary.get('?êÏπòÎ≤ïÍ∑ú?ºÎ†®Î≤àÌò∏')
                    if local_id:
                        # ?êÏπòÎ≤ïÍ∑ú ?ÅÏÑ∏ ?ïÎ≥¥ Ï°∞Ìöå
                        local_detail = self.client.get_local_ordinance_detail(ordinance_id=local_id)
                        if local_detail:
                            # ?µÌï© ?∞Ïù¥???ùÏÑ±
                            integrated_local = {
                                **local_summary,
                                **local_detail,
                                "document_type": "local_ordinance",
                                "collected_at": datetime.now().isoformat()
                            }
                            detailed_ordinances.append(integrated_local)
                            
                except Exception as e:
                    logger.error(f"Error collecting local ordinance {local_summary.get('?êÏπòÎ≤ïÍ∑ú?ºÎ†®Î≤àÌò∏', 'unknown')}: {e}")
                    continue
            
            if detailed_ordinances:
                file_path = self._save_data(detailed_ordinances, "local_ordinances", query)
                if file_path:
                    self.collection_stats['local_ordinances_collected'] += len(detailed_ordinances)
                    self.collection_stats['total_documents'] += len(detailed_ordinances)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error collecting local ordinances: {e}")
            self.collection_stats['errors'].append(f"Local ordinances collection error: {e}")
            return False
    
    def collect_all_data(self) -> bool:
        """Î™®Îì† ?∞Ïù¥???òÏßë"""
        try:
            logger.info("Starting comprehensive data collection...")
            
            success_count = 0
            total_tasks = 6
            
            # 1. Î≤ïÎ†π ?òÏßë
            if self.collect_laws(query="ÎØºÎ≤ï", display=50):
                success_count += 1
            
            # 2. ?êÎ? ?òÏßë
            if self.collect_precedents(query="?êÌï¥Î∞∞ÏÉÅ", display=50):
                success_count += 1
            
            # 3. ?åÏû¨Í≤∞Ï†ïÎ°Ä ?òÏßë
            if self.collect_constitutional_decisions(query="?åÎ≤ï", display=50):
                success_count += 1
            
            # 4. Î≤ïÎ†π?¥ÏÑùÎ°Ä ?òÏßë
            if self.collect_legal_interpretations(query="Î≤ïÎ†π?¥ÏÑù", display=50):
                success_count += 1
            
            # 5. ?âÏ†ïÍ∑úÏπô ?òÏßë
            if self.collect_administrative_rules(query="?âÏ†ïÍ∑úÏπô", display=50):
                success_count += 1
            
            # 6. ?êÏπòÎ≤ïÍ∑ú ?òÏßë
            if self.collect_local_ordinances(query="?êÏπòÎ≤ïÍ∑ú", display=50):
                success_count += 1
            
            # ÏµúÏ¢Ö ?µÍ≥Ñ ?ùÏÑ±
            self._generate_collection_report()
            
            logger.info(f"Data collection completed: {success_count}/{total_tasks} tasks successful")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error in comprehensive data collection: {e}")
            return False
    
    def _generate_collection_report(self):
        """?òÏßë Î≥¥Í≥†???ùÏÑ±"""
        try:
            self.collection_stats['end_time'] = datetime.now().isoformat()
            self.collection_stats['total_duration'] = (
                datetime.fromisoformat(self.collection_stats['end_time']) - 
                datetime.fromisoformat(self.collection_stats['start_time'])
            ).total_seconds()
            
            # API ?¨Ïö© ?µÍ≥Ñ
            if self.client:
                api_stats = self.client.get_request_stats()
                self.collection_stats['api_stats'] = api_stats
            
            # Î≥¥Í≥†???Ä??
            report_file = self.output_dir / "collection_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.collection_stats, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Collection report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating collection report: {e}")


def main():
    """Î©îÏù∏ ?®Ïàò"""
    parser = argparse.ArgumentParser(description="LawFirmAI Data Collection Script (JSON only)")
    parser.add_argument("--mode", type=str, choices=["collect", "laws", "precedents", "constitutional", "interpretations", "administrative", "local", "multiple"], 
                        default="collect", help="Collection mode")
    parser.add_argument("--oc", type=str, required=True, help="OC parameter for Law OpenAPI (user email ID)")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--display", type=int, default=100, help="Number of items to display per API call")
    parser.add_argument("--types", type=str, nargs="+", 
                        choices=["laws", "precedents", "constitutional", "interpretations", "administrative", "local"],
                        help="Specific data types to collect (use with --mode multiple)")
    
    args = parser.parse_args()
    
    # Î°úÍ∑∏ ?îÎ†â?†Î¶¨ ?ùÏÑ±
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # ?∞Ïù¥???òÏßë ?§Ìñâ
    collector = DataCollector(args.oc)
    
    success = False
    if args.mode == "collect":
        success = collector.collect_all_data()
    elif args.mode == "multiple":
        # ?¨Îü¨ ?Ä???òÏßë
        if not args.types:
            logger.error("--types parameter is required when using --mode multiple")
            logger.info("Example: python scripts/collect_data_only.py --mode multiple --oc your_email_id --types laws precedents")
            return
        
        success_count = 0
        for data_type in args.types:
            logger.info(f"Collecting {data_type} data...")
            
            # Í∏∞Î≥∏ ÏøºÎ¶¨ ?§Ï†ï
            default_queries = {
                "laws": "ÎØºÎ≤ï",
                "precedents": "Í≥ÑÏïΩ ?¥Ï?",
                "constitutional": "?åÎ≤ï",
                "interpretations": "Î≤ïÎ†π?¥ÏÑù",
                "administrative": "?âÏ†ïÍ∑úÏπô",
                "local": "?êÏπòÎ≤ïÍ∑ú"
            }
            
            search_query = args.query or default_queries.get(data_type, data_type)
            
            if data_type == "laws":
                type_success = collector.collect_laws(query=search_query, display=args.display)
            elif data_type == "precedents":
                type_success = collector.collect_precedents(query=search_query, display=args.display)
            elif data_type == "constitutional":
                type_success = collector.collect_constitutional_decisions(query=search_query, display=args.display)
            elif data_type == "interpretations":
                type_success = collector.collect_legal_interpretations(query=search_query, display=args.display)
            elif data_type == "administrative":
                type_success = collector.collect_administrative_rules(query=search_query, display=args.display)
            elif data_type == "local":
                type_success = collector.collect_local_ordinances(query=search_query, display=args.display)
            else:
                logger.error(f"Unknown data type: {data_type}")
                continue
            
            if type_success:
                success_count += 1
                logger.info(f"{data_type} collection completed successfully!")
            else:
                logger.error(f"{data_type} collection failed!")
        
        success = success_count > 0
        logger.info(f"Multiple types collection completed: {success_count}/{len(args.types)} types successful")
        
    elif args.mode == "laws":
        success = collector.collect_laws(query=args.query or "ÎØºÎ≤ï", display=args.display)
    elif args.mode == "precedents":
        success = collector.collect_precedents(query=args.query or "Í≥ÑÏïΩ ?¥Ï?", display=args.display)
    elif args.mode == "constitutional":
        success = collector.collect_constitutional_decisions(query=args.query or "?åÎ≤ï", display=args.display)
    elif args.mode == "interpretations":
        success = collector.collect_legal_interpretations(query=args.query or "Î≤ïÎ†π?¥ÏÑù", display=args.display)
    elif args.mode == "administrative":
        success = collector.collect_administrative_rules(query=args.query or "?âÏ†ïÍ∑úÏπô", display=args.display)
    elif args.mode == "local":
        success = collector.collect_local_ordinances(query=args.query or "?êÏπòÎ≤ïÍ∑ú", display=args.display)
    
    if success:
        logger.info("Data collection completed successfully!")
    else:
        logger.error("Data collection failed!")


if __name__ == "__main__":
    main()