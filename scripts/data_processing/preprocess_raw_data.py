#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raw ?°ì´???„ì²˜ë¦?ë©”ì¸ ?¤í¬ë¦½íŠ¸

êµ??ë²•ë ¹?•ë³´?¼í„° OpenAPIë¥??µí•´ ?˜ì§‘??raw ?°ì´?°ë? ?„ì²˜ë¦¬í•˜??
ë²¡í„° ?°ì´?°ë² ?´ìŠ¤ êµ¬ì¶•ê³?RAG ?œìŠ¤?œì— ?í•©???•íƒœë¡?ë³€?˜í•©?ˆë‹¤.
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.data_processor import LegalDataProcessor
from source.data.legal_term_normalizer import LegalTermNormalizer

class RawDataPreprocessingPipeline:
    """Raw ?°ì´???„ì²˜ë¦??µí•© ?Œì´?„ë¼??""
    
    def __init__(self, enable_term_normalization=True):
        """?„ì²˜ë¦??Œì´?„ë¼??ì´ˆê¸°??""
        self.processor = LegalDataProcessor(enable_term_normalization)
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(exist_ok=True)
        
        # ë¡œê¹… ?¤ì •
        self.setup_logging()
        
        # ?µê³„ ì´ˆê¸°??
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "by_type": {}
        }
        
        # ?°ì´??? í˜•ë³??°ì„ ?œìœ„
        self.preprocessing_priority = {
            "laws": 1,           # ë²•ë ¹ (ìµœìš°??
            "precedents": 2,     # ?ë? (High)
            "constitutional_decisions": 3,  # ?Œì¬ê²°ì •ë¡€ (Medium)
            "legal_interpretations": 4,     # ë²•ë ¹?´ì„ë¡€ (Medium)
            "legal_terms": 5,    # ë²•ë¥  ?©ì–´ (Medium)
            "administrative_rules": 6,      # ?‰ì •ê·œì¹™ (Low)
            "local_ordinances": 7,          # ?ì¹˜ë²•ê·œ (Low)
            "committee_decisions": 8,       # ?„ì›?Œê²°?•ë¬¸ (Low)
            "treaties": 9        # ì¡°ì•½ (Low)
        }
    
    def setup_logging(self):
        """ë¡œê¹… ?¤ì •"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    f'logs/preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_full_preprocessing(self):
        """?„ì²´ ?„ì²˜ë¦??Œì´?„ë¼???¤í–‰"""
        self.logger.info("=== Raw ?°ì´???„ì²˜ë¦??œì‘ ===")
        
        start_time = datetime.now()
        
        try:
            # Phase 1: ?µì‹¬ ?°ì´???„ì²˜ë¦?
            self.logger.info("Phase 1: ?µì‹¬ ?°ì´???„ì²˜ë¦??œì‘")
            self.process_laws()
            self.process_precedents()
            
            # Phase 2: ?•ì¥ ?°ì´???„ì²˜ë¦?
            self.logger.info("Phase 2: ?•ì¥ ?°ì´???„ì²˜ë¦??œì‘")
            self.process_constitutional_decisions()
            self.process_legal_interpretations()
            self.process_legal_terms()
            
            # Phase 3: ?ˆì§ˆ ê²€ì¦?ë°??µí•©
            self.logger.info("Phase 3: ?ˆì§ˆ ê²€ì¦?ë°??µí•©")
            self.validate_processed_data()
            self.consolidate_results()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info(f"=== ?„ì²˜ë¦??„ë£Œ (?Œìš”?œê°„: {duration}) ===")
            self.print_statistics()
            
        except Exception as e:
            self.logger.error(f"?„ì²˜ë¦?ì¤??¤ë¥˜ ë°œìƒ: {e}")
            raise
    
    def process_laws(self):
        """ë²•ë ¹ ?°ì´???„ì²˜ë¦?""
        self.logger.info("ë²•ë ¹ ?°ì´???„ì²˜ë¦??œì‘")
        
        law_files = list(Path("data/raw/laws").glob("*.json"))
        processed_laws = []
        
        for law_file in law_files:
            try:
                self.logger.info(f"ì²˜ë¦¬ ì¤? {law_file}")
                with open(law_file, 'r', encoding='utf-8') as f:
                    law_data = json.load(f)
                
                # ê¸°ë³¸ ?•ë³´ ?•ì¸
                basic_info = law_data.get('basic_info', {})
                self.logger.info(f"  - ë²•ë ¹ëª? {basic_info.get('name', 'N/A')}")
                self.logger.info(f"  - ID: {basic_info.get('id', 'N/A')}")
                
                processed_law = self.processor.process_law_data(law_data)
                
                # ì²˜ë¦¬ ê²°ê³¼ ?•ì¸
                if processed_law.get('status') == 'success':
                    content_length = len(processed_law.get('full_content', ''))
                    chunks_count = len(processed_law.get('chunks', []))
                    self.logger.info(f"  - ì²˜ë¦¬ ?±ê³µ: ?´ìš© ê¸¸ì´ {content_length}?? ì²?¬ {chunks_count}ê°?)
                    processed_laws.append(processed_law)
                    self.stats['successful'] += 1
                else:
                    self.stats['failed'] += 1
                    self.logger.warning(f"ë²•ë ¹ ?„ì²˜ë¦??¤íŒ¨: {law_file} - {processed_law.get('error', 'Unknown error')}")
                
                self.stats['total_processed'] += 1
                
            except Exception as e:
                self.logger.error(f"ë²•ë ¹ ?„ì²˜ë¦??¤íŒ¨ {law_file}: {e}")
                self.stats['failed'] += 1
                self.stats['total_processed'] += 1
        
        # ê²°ê³¼ ?€??
        self.save_processed_data(processed_laws, "laws")
        self.stats['by_type']['laws'] = len(processed_laws)
        
        self.logger.info(f"ë²•ë ¹ ?°ì´???„ì²˜ë¦??„ë£Œ: {len(processed_laws)}ê°?)
    
    def process_precedents(self):
        """?ë? ?°ì´???„ì²˜ë¦?""
        self.logger.info("?ë? ?°ì´???„ì²˜ë¦??œì‘")
        
        precedent_dirs = list(Path("data/raw/precedents").glob("yearly_*"))
        all_processed_precedents = []
        
        for precedent_dir in precedent_dirs:
            json_files = list(precedent_dir.glob("*.json"))
            
            for json_file in json_files:
                try:
                    self.logger.info(f"ì²˜ë¦¬ ì¤? {json_file}")
                    with open(json_file, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                    
                    # ?ë? ?°ì´?°ëŠ” precedents ë°°ì—´ ?ˆì— ?ˆìŒ
                    if isinstance(file_data, dict) and 'precedents' in file_data:
                        precedents_list = file_data['precedents']
                        self.logger.info(f"  - ?ë? ?? {len(precedents_list)}ê°?)
                        
                        processed_precedents = self.processor.process_batch(
                            precedents_list, 'precedent'
                        )
                        
                        # ì²˜ë¦¬ ê²°ê³¼ ë¡œê¹…
                        success_count = len([p for p in processed_precedents if p.get('status') == 'success'])
                        self.logger.info(f"  - ì²˜ë¦¬ ?±ê³µ: {success_count}/{len(precedents_list)}ê°?)
                        
                    elif isinstance(file_data, list):
                        # ì§ì ‘ ë°°ì—´ ?•íƒœ??ê²½ìš°
                        processed_precedents = self.processor.process_batch(
                            file_data, 'precedent'
                        )
                    else:
                        # ?¨ì¼ ?ë? ?°ì´??ì²˜ë¦¬
                        processed_precedents = [self.processor.process_precedent_data(file_data)]
                    
                    # ?±ê³µ??ê²ƒë§Œ ì¶”ê?
                    successful_precedents = [p for p in processed_precedents if p.get('status') == 'success']
                    all_processed_precedents.extend(successful_precedents)
                    
                    self.stats['total_processed'] += len(processed_precedents)
                    self.stats['successful'] += len(successful_precedents)
                    self.stats['failed'] += len(processed_precedents) - len(successful_precedents)
                    
                except Exception as e:
                    self.logger.error(f"?ë? ?„ì²˜ë¦??¤íŒ¨ {json_file}: {e}")
                    self.stats['failed'] += 1
                    self.stats['total_processed'] += 1
        
        # ê²°ê³¼ ?€??
        self.save_processed_data(all_processed_precedents, "precedents")
        self.stats['by_type']['precedents'] = len(all_processed_precedents)
        
        self.logger.info(f"?ë? ?°ì´???„ì²˜ë¦??„ë£Œ: {len(all_processed_precedents)}ê°?)
    
    def process_constitutional_decisions(self):
        """?Œì¬ê²°ì •ë¡€ ?°ì´???„ì²˜ë¦?""
        self.logger.info("?Œì¬ê²°ì •ë¡€ ?°ì´???„ì²˜ë¦??œì‘")
        
        constitutional_dirs = list(Path("data/raw/constitutional_decisions").glob("yearly_*"))
        all_processed_decisions = []
        
        for constitutional_dir in constitutional_dirs:
            json_files = list(constitutional_dir.glob("*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        decision_data = json.load(f)
                    
                    if isinstance(decision_data, list):
                        processed_decisions = self.processor.process_batch(
                            decision_data, 'constitutional_decision'
                        )
                    else:
                        processed_decisions = [self.processor.process_constitutional_decision_data(decision_data)]
                    
                    # ?±ê³µ??ê²ƒë§Œ ì¶”ê?
                    successful_decisions = [p for p in processed_decisions if p.get('status') == 'success']
                    all_processed_decisions.extend(successful_decisions)
                    
                    self.stats['total_processed'] += len(processed_decisions)
                    self.stats['successful'] += len(successful_decisions)
                    self.stats['failed'] += len(processed_decisions) - len(successful_decisions)
                    
                except Exception as e:
                    self.logger.error(f"?Œì¬ê²°ì •ë¡€ ?„ì²˜ë¦??¤íŒ¨ {json_file}: {e}")
                    self.stats['failed'] += 1
                    self.stats['total_processed'] += 1
        
        # ê²°ê³¼ ?€??
        self.save_processed_data(all_processed_decisions, "constitutional_decisions")
        self.stats['by_type']['constitutional_decisions'] = len(all_processed_decisions)
        
        self.logger.info(f"?Œì¬ê²°ì •ë¡€ ?°ì´???„ì²˜ë¦??„ë£Œ: {len(all_processed_decisions)}ê°?)
    
    def process_legal_interpretations(self):
        """ë²•ë ¹?´ì„ë¡€ ?°ì´???„ì²˜ë¦?""
        self.logger.info("ë²•ë ¹?´ì„ë¡€ ?°ì´???„ì²˜ë¦??œì‘")
        
        interpretation_dirs = list(Path("data/raw/legal_interpretations").glob("yearly_*"))
        all_processed_interpretations = []
        
        for interpretation_dir in interpretation_dirs:
            json_files = list(interpretation_dir.glob("*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        interpretation_data = json.load(f)
                    
                    if isinstance(interpretation_data, list):
                        processed_interpretations = self.processor.process_batch(
                            interpretation_data, 'legal_interpretation'
                        )
                    else:
                        processed_interpretations = [self.processor.process_legal_interpretation_data(interpretation_data)]
                    
                    # ?±ê³µ??ê²ƒë§Œ ì¶”ê?
                    successful_interpretations = [p for p in processed_interpretations if p.get('status') == 'success']
                    all_processed_interpretations.extend(successful_interpretations)
                    
                    self.stats['total_processed'] += len(processed_interpretations)
                    self.stats['successful'] += len(successful_interpretations)
                    self.stats['failed'] += len(processed_interpretations) - len(successful_interpretations)
                    
                except Exception as e:
                    self.logger.error(f"ë²•ë ¹?´ì„ë¡€ ?„ì²˜ë¦??¤íŒ¨ {json_file}: {e}")
                    self.stats['failed'] += 1
                    self.stats['total_processed'] += 1
        
        # ê²°ê³¼ ?€??
        self.save_processed_data(all_processed_interpretations, "legal_interpretations")
        self.stats['by_type']['legal_interpretations'] = len(all_processed_interpretations)
        
        self.logger.info(f"ë²•ë ¹?´ì„ë¡€ ?°ì´???„ì²˜ë¦??„ë£Œ: {len(all_processed_interpretations)}ê°?)
    
    def process_legal_terms(self):
        """ë²•ë¥  ?©ì–´ ?°ì´???„ì²˜ë¦?""
        self.logger.info("ë²•ë¥  ?©ì–´ ?°ì´???„ì²˜ë¦??œì‘")
        
        term_dirs = list(Path("data/raw/legal_terms").glob("session_*"))
        all_processed_terms = []
        
        for term_dir in term_dirs:
            json_files = list(term_dir.glob("*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        term_data = json.load(f)
                    
                    # ?©ì–´ ?°ì´?°ëŠ” ?¹ë³„??ì²˜ë¦¬ê°€ ?„ìš”?????ˆìŒ
                    processed_terms = self.process_legal_term_data(term_data)
                    all_processed_terms.extend(processed_terms)
                    
                    self.stats['total_processed'] += len(processed_terms)
                    self.stats['successful'] += len(processed_terms)
                    
                except Exception as e:
                    self.logger.error(f"ë²•ë¥  ?©ì–´ ?„ì²˜ë¦??¤íŒ¨ {json_file}: {e}")
                    self.stats['failed'] += 1
                    self.stats['total_processed'] += 1
        
        # ê²°ê³¼ ?€??
        self.save_processed_data(all_processed_terms, "legal_terms")
        self.stats['by_type']['legal_terms'] = len(all_processed_terms)
        
        self.logger.info(f"ë²•ë¥  ?©ì–´ ?°ì´???„ì²˜ë¦??„ë£Œ: {len(all_processed_terms)}ê°?)
    
    def process_legal_term_data(self, term_data):
        """ë²•ë¥  ?©ì–´ ?°ì´??ì²˜ë¦¬"""
        processed_terms = []
        
        if isinstance(term_data, dict) and 'terms' in term_data:
            for term in term_data['terms']:
                processed_term = {
                    'id': term.get('term_sequence_number', ''),
                    'term_name_korean': term.get('term_name_korean', ''),
                    'term_name_chinese': term.get('term_name_chinese', ''),
                    'definition': term.get('definition', ''),
                    'source': term.get('source', ''),
                    'category': 'legal_term',
                    'status': 'success',
                    'processed_at': datetime.now().isoformat()
                }
                processed_terms.append(processed_term)
        elif isinstance(term_data, list):
            for term in term_data:
                processed_term = {
                    'id': term.get('term_sequence_number', ''),
                    'term_name_korean': term.get('term_name_korean', ''),
                    'term_name_chinese': term.get('term_name_chinese', ''),
                    'definition': term.get('definition', ''),
                    'source': term.get('source', ''),
                    'category': 'legal_term',
                    'status': 'success',
                    'processed_at': datetime.now().isoformat()
                }
                processed_terms.append(processed_term)
        
        return processed_terms
    
    def save_processed_data(self, data, data_type):
        """?„ì²˜ë¦¬ëœ ?°ì´???€??""
        output_dir = self.output_dir / data_type
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"{data_type}_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"{data_type} ?°ì´???€???„ë£Œ: {output_file}")
    
    def validate_processed_data(self):
        """?„ì²˜ë¦¬ëœ ?°ì´??ê²€ì¦?""
        self.logger.info("?„ì²˜ë¦¬ëœ ?°ì´??ê²€ì¦??œì‘")
        
        validation_results = {}
        
        for data_type in self.stats['by_type'].keys():
            validation_results[data_type] = self.validate_data_type(data_type)
        
        # ê²€ì¦?ê²°ê³¼ ?€??
        validation_file = self.output_dir / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"?°ì´??ê²€ì¦??„ë£Œ: {validation_file}")
    
    def validate_data_type(self, data_type):
        """?¹ì • ?°ì´??? í˜• ê²€ì¦?""
        data_dir = self.output_dir / data_type
        if not data_dir.exists():
            return {
                "total_documents": 0,
                "validation_passed": False,
                "issues": ["?°ì´???”ë ‰? ë¦¬ê°€ ì¡´ì¬?˜ì? ?ŠìŒ"]
            }
        
        json_files = list(data_dir.glob("*.json"))
        total_documents = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    total_documents += len(data)
                else:
                    total_documents += 1
            except Exception as e:
                self.logger.error(f"ê²€ì¦?ì¤??¤ë¥˜ {json_file}: {e}")
        
        return {
            "total_documents": total_documents,
            "validation_passed": total_documents > 0,
            "issues": [] if total_documents > 0 else ["ë¬¸ì„œê°€ ?†ìŒ"]
        }
    
    def consolidate_results(self):
        """ê²°ê³¼ ?µí•©"""
        self.logger.info("?„ì²˜ë¦?ê²°ê³¼ ?µí•© ?œì‘")
        
        # ?µí•© ?¸ë±???ì„±
        consolidated_index = {
            "metadata": {
                "total_processed": self.stats['total_processed'],
                "successful": self.stats['successful'],
                "failed": self.stats['failed'],
                "by_type": self.stats['by_type'],
                "processed_at": datetime.now().isoformat()
            },
            "data_types": list(self.stats['by_type'].keys()),
            "file_locations": {}
        }
        
        # ?Œì¼ ?„ì¹˜ ?•ë³´ ì¶”ê?
        for data_type in self.stats['by_type'].keys():
            data_dir = self.output_dir / data_type
            if data_dir.exists():
                files = list(data_dir.glob("*.json"))
                consolidated_index["file_locations"][data_type] = [str(f) for f in files]
        
        # ?µí•© ?¸ë±???€??
        index_file = self.output_dir / "consolidated_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_index, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"ê²°ê³¼ ?µí•© ?„ë£Œ: {index_file}")
    
    def print_statistics(self):
        """?µê³„ ì¶œë ¥"""
        self.logger.info("=== ?„ì²˜ë¦??µê³„ ===")
        self.logger.info(f"ì´?ì²˜ë¦¬: {self.stats['total_processed']}ê°?)
        self.logger.info(f"?±ê³µ: {self.stats['successful']}ê°?)
        self.logger.info(f"?¤íŒ¨: {self.stats['failed']}ê°?)
        
        if self.stats['total_processed'] > 0:
            success_rate = self.stats['successful'] / self.stats['total_processed'] * 100
            self.logger.info(f"?±ê³µë¥? {success_rate:.2f}%")
        
        self.logger.info("=== ?°ì´??? í˜•ë³??µê³„ ===")
        for data_type, count in self.stats['by_type'].items():
            self.logger.info(f"{data_type}: {count}ê°?)
    
    def process_specific_type(self, data_type):
        """?¹ì • ?°ì´??? í˜•ë§?ì²˜ë¦¬"""
        if data_type == "laws":
            self.process_laws()
        elif data_type == "precedents":
            self.process_precedents()
        elif data_type == "constitutional":
            self.process_constitutional_decisions()
        elif data_type == "interpretations":
            self.process_legal_interpretations()
        elif data_type == "terms":
            self.process_legal_terms()
        else:
            self.logger.error(f"?????†ëŠ” ?°ì´??? í˜•: {data_type}")
    
    def dry_run(self, data_type):
        """?œë¼?´ëŸ° ëª¨ë“œ - ?¤ì œ ì²˜ë¦¬ ?†ì´ ê³„íšë§?ì¶œë ¥"""
        self.logger.info("=== ?œë¼?´ëŸ° ëª¨ë“œ ===")
        
        if data_type == "all":
            data_types = ["laws", "precedents", "constitutional", "interpretations", "terms"]
        else:
            data_types = [data_type]
        
        for dt in data_types:
            self.logger.info(f"ì²˜ë¦¬ ?ˆì •: {dt}")
            
            if dt == "laws":
                law_files = list(Path("data/raw/laws").glob("*.json"))
                self.logger.info(f"  - ë²•ë ¹ ?Œì¼: {len(law_files)}ê°?)
            elif dt == "precedents":
                precedent_dirs = list(Path("data/raw/precedents").glob("yearly_*"))
                total_files = sum(len(list(d.glob("*.json"))) for d in precedent_dirs)
                self.logger.info(f"  - ?ë? ?´ë”: {len(precedent_dirs)}ê°?)
                self.logger.info(f"  - ?ë? ?Œì¼: {total_files}ê°?)
            elif dt == "constitutional":
                constitutional_dirs = list(Path("data/raw/constitutional_decisions").glob("yearly_*"))
                total_files = sum(len(list(d.glob("*.json"))) for d in constitutional_dirs)
                self.logger.info(f"  - ?Œì¬ê²°ì •ë¡€ ?´ë”: {len(constitutional_dirs)}ê°?)
                self.logger.info(f"  - ?Œì¬ê²°ì •ë¡€ ?Œì¼: {total_files}ê°?)
            elif dt == "interpretations":
                interpretation_dirs = list(Path("data/raw/legal_interpretations").glob("yearly_*"))
                total_files = sum(len(list(d.glob("*.json"))) for d in interpretation_dirs)
                self.logger.info(f"  - ë²•ë ¹?´ì„ë¡€ ?´ë”: {len(interpretation_dirs)}ê°?)
                self.logger.info(f"  - ë²•ë ¹?´ì„ë¡€ ?Œì¼: {total_files}ê°?)
            elif dt == "terms":
                term_dirs = list(Path("data/raw/legal_terms").glob("session_*"))
                total_files = sum(len(list(d.glob("*.json"))) for d in term_dirs)
                self.logger.info(f"  - ë²•ë¥  ?©ì–´ ?´ë”: {len(term_dirs)}ê°?)
                self.logger.info(f"  - ë²•ë¥  ?©ì–´ ?Œì¼: {total_files}ê°?)

def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Raw ?°ì´???„ì²˜ë¦?)
    parser.add_argument("--data-type", default="all",
                       choices=["laws", "precedents", "constitutional", "interpretations", "terms", "all"],
                       help="?„ì²˜ë¦¬í•  ?°ì´??? í˜•")
    parser.add_argument("--enable-normalization", action="store_true", default=True,
                       help="ë²•ë¥  ?©ì–´ ?•ê·œ???œì„±??)
    parser.add_argument("--dry-run", action="store_true",
                       help="?¤ì œ ì²˜ë¦¬ ?†ì´ ê³„íšë§?ì¶œë ¥")
    
    args = parser.parse_args()
    
    pipeline = RawDataPreprocessingPipeline(args.enable_normalization)
    
    if args.dry_run:
        pipeline.dry_run(args.data_type)
    else:
        if args.data_type == "all":
            pipeline.run_full_preprocessing()
        else:
            pipeline.process_specific_type(args.data_type)

if __name__ == "__main__":
    main()
