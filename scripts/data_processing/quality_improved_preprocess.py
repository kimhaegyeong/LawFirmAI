#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?ˆì§ˆ ê°œì„ ??Raw ?°ì´???„ì²˜ë¦??¤í¬ë¦½íŠ¸

?ˆì§ˆ ?ìˆ˜ë¥?ê°œì„ ?˜ê¸° ?„í•œ ê°•í™”???„ì²˜ë¦??Œì´?„ë¼?¸ì…?ˆë‹¤.
"""

import sys
import os
import json
import logging
import gc
import psutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Generator
import argparse

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
sys.path.append(str(Path(__file__).parent.parent))

from source.data.enhanced_data_processor import EnhancedLegalDataProcessor

class QualityImprovedPreprocessor:
    """?ˆì§ˆ ê°œì„ ???„ì²˜ë¦??´ë˜??""
    
    def __init__(self, 
                 enable_term_normalization=True,
                 max_memory_usage=0.8,
                 batch_size=50,
                 chunk_size=1000):
        """?ˆì§ˆ ê°œì„  ?„ì²˜ë¦¬ê¸° ì´ˆê¸°??""
        self.processor = EnhancedLegalDataProcessor(enable_term_normalization)
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(exist_ok=True)
        
        # ë©”ëª¨ë¦?ê´€ë¦??¤ì •
        self.max_memory_usage = max_memory_usage
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        
        # ë¡œê¹… ?¤ì •
        self.setup_logging()
        
        # ?µê³„ ì´ˆê¸°??
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "by_type": {},
            "quality_metrics": {
                "completeness": 0.0,
                "consistency": 0.0,
                "term_normalization": 0.0
            }
        }
    
    def setup_logging(self):
        """ë¡œê¹… ?¤ì •"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/quality_improved_preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_memory_usage(self) -> float:
        """?„ì¬ ë©”ëª¨ë¦??¬ìš©ë¥?ë°˜í™˜"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024 * 1024)  # GB ?¨ìœ„
    
    def check_memory_limit(self) -> bool:
        """ë©”ëª¨ë¦??¬ìš©??ì²´í¬"""
        current_memory = self.get_memory_usage()
        
        if current_memory > 8:  # 8GB ?´ìƒ ?¬ìš© ??ê²½ê³ 
            self.logger.warning(f"ë©”ëª¨ë¦??¬ìš©?‰ì´ ?’ìŠµ?ˆë‹¤: {current_memory:.2f}GB")
            return False
        return True
    
    def force_garbage_collection(self):
        """ê°•ì œ ê°€ë¹„ì? ì»¬ë ‰???¤í–‰"""
        gc.collect()
        self.logger.debug("ê°€ë¹„ì? ì»¬ë ‰???¤í–‰ ?„ë£Œ")
    
    def process_file_in_chunks(self, file_path: Path, data_type: str) -> Generator[Dict[str, Any], None, None]:
        """?Œì¼??ì²?¬ ?¨ìœ„ë¡?ì²˜ë¦¬?˜ëŠ” ?œë„ˆ?ˆì´??""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # ?°ì´???€?…ì— ?°ë¥¸ ì²˜ë¦¬
            if data_type == 'precedent' and isinstance(data, dict) and 'precedents' in data:
                items = data['precedents']
            elif isinstance(data, list):
                items = data
            else:
                items = [data]
            
            # ì²?¬ ?¨ìœ„ë¡?ì²˜ë¦¬
            for i in range(0, len(items), self.chunk_size):
                chunk = items[i:i + self.chunk_size]
                
                # ë©”ëª¨ë¦?ì²´í¬
                if not self.check_memory_limit():
                    self.force_garbage_collection()
                
                # ì²?¬ ì²˜ë¦¬
                processed_chunk = self.processor.process_batch(chunk, data_type)
                
                # ëª¨ë“  ??ª© yield (? íš¨?˜ì? ?Šì•„???¬í•¨)
                for item in processed_chunk:
                    yield item
                
                # ë©”ëª¨ë¦??•ë¦¬
                del chunk
                del processed_chunk
                self.force_garbage_collection()
                
        except Exception as e:
            self.logger.error(f"?Œì¼ ì²˜ë¦¬ ì¤??¤ë¥˜ {file_path}: {e}")
    
    def process_laws_improved(self):
        """?ˆì§ˆ ê°œì„ ??ë²•ë ¹ ?°ì´???„ì²˜ë¦?""
        self.logger.info("?ˆì§ˆ ê°œì„ ??ë²•ë ¹ ?°ì´???„ì²˜ë¦??œì‘")
        
        law_files = list(Path("data/raw/laws").glob("*.json"))
        processed_count = 0
        valid_count = 0
        
        for law_file in law_files:
            try:
                self.logger.info(f"ì²˜ë¦¬ ì¤? {law_file}")
                
                # ?Œì¼??ì²?¬ ?¨ìœ„ë¡?ì²˜ë¦¬
                for processed_law in self.process_file_in_chunks(law_file, 'law'):
                    # ì¦‰ì‹œ ?€??(ë©”ëª¨ë¦??„ì  ë°©ì?)
                    self.save_single_document(processed_law, "laws")
                    processed_count += 1
                    
                    if processed_law.get('is_valid', False):
                        valid_count += 1
                        self.stats['successful'] += 1
                    else:
                        self.stats['failed'] += 1
                    
                    # ë©”ëª¨ë¦?ì²´í¬
                    if not self.check_memory_limit():
                        self.force_garbage_collection()
                
                self.stats['total_processed'] += 1
                
            except Exception as e:
                self.logger.error(f"ë²•ë ¹ ?„ì²˜ë¦??¤íŒ¨ {law_file}: {e}")
                self.stats['failed'] += 1
        
        self.stats['by_type']['laws'] = processed_count
        self.logger.info(f"ë²•ë ¹ ?°ì´???„ì²˜ë¦??„ë£Œ: {processed_count}ê°?(? íš¨: {valid_count}ê°?")
    
    def process_precedents_improved(self):
        """?ˆì§ˆ ê°œì„ ???ë? ?°ì´???„ì²˜ë¦?""
        self.logger.info("?ˆì§ˆ ê°œì„ ???ë? ?°ì´???„ì²˜ë¦??œì‘")
        
        precedent_dirs = list(Path("data/raw/precedents").glob("yearly_*"))
        processed_count = 0
        valid_count = 0
        
        for precedent_dir in precedent_dirs:
            json_files = list(precedent_dir.glob("*.json"))
            
            for json_file in json_files:
                try:
                    self.logger.info(f"ì²˜ë¦¬ ì¤? {json_file}")
                    
                    # ?Œì¼??ì²?¬ ?¨ìœ„ë¡?ì²˜ë¦¬
                    for processed_precedent in self.process_file_in_chunks(json_file, 'precedent'):
                        # ì¦‰ì‹œ ?€??(ë©”ëª¨ë¦??„ì  ë°©ì?)
                        self.save_single_document(processed_precedent, "precedents")
                        processed_count += 1
                        
                        if processed_precedent.get('is_valid', False):
                            valid_count += 1
                            self.stats['successful'] += 1
                        else:
                            self.stats['failed'] += 1
                        
                        # ë©”ëª¨ë¦?ì²´í¬
                        if not self.check_memory_limit():
                            self.force_garbage_collection()
                
                except Exception as e:
                    self.logger.error(f"?ë? ?„ì²˜ë¦??¤íŒ¨ {json_file}: {e}")
                    self.stats['failed'] += 1
        
        self.stats['by_type']['precedents'] = processed_count
        self.logger.info(f"?ë? ?°ì´???„ì²˜ë¦??„ë£Œ: {processed_count}ê°?(? íš¨: {valid_count}ê°?")
    
    def process_constitutional_decisions_improved(self):
        """?ˆì§ˆ ê°œì„ ???Œì¬ê²°ì •ë¡€ ?°ì´???„ì²˜ë¦?""
        self.logger.info("?ˆì§ˆ ê°œì„ ???Œì¬ê²°ì •ë¡€ ?°ì´???„ì²˜ë¦??œì‘")
        
        constitutional_dirs = list(Path("data/raw/constitutional_decisions").glob("yearly_*"))
        processed_count = 0
        valid_count = 0
        
        for constitutional_dir in constitutional_dirs:
            json_files = list(constitutional_dir.glob("*.json"))
            
            for json_file in json_files:
                try:
                    self.logger.info(f"ì²˜ë¦¬ ì¤? {json_file}")
                    
                    # ?Œì¼??ì²?¬ ?¨ìœ„ë¡?ì²˜ë¦¬
                    for processed_decision in self.process_file_in_chunks(json_file, 'constitutional_decision'):
                        # ì¦‰ì‹œ ?€??(ë©”ëª¨ë¦??„ì  ë°©ì?)
                        self.save_single_document(processed_decision, "constitutional_decisions")
                        processed_count += 1
                        
                        if processed_decision.get('is_valid', False):
                            valid_count += 1
                            self.stats['successful'] += 1
                        else:
                            self.stats['failed'] += 1
                        
                        # ë©”ëª¨ë¦?ì²´í¬
                        if not self.check_memory_limit():
                            self.force_garbage_collection()
                
                except Exception as e:
                    self.logger.error(f"?Œì¬ê²°ì •ë¡€ ?„ì²˜ë¦??¤íŒ¨ {json_file}: {e}")
                    self.stats['failed'] += 1
        
        self.stats['by_type']['constitutional_decisions'] = processed_count
        self.logger.info(f"?Œì¬ê²°ì •ë¡€ ?°ì´???„ì²˜ë¦??„ë£Œ: {processed_count}ê°?(? íš¨: {valid_count}ê°?")
    
    def process_legal_interpretations_improved(self):
        """?ˆì§ˆ ê°œì„ ??ë²•ë ¹?´ì„ë¡€ ?°ì´???„ì²˜ë¦?""
        self.logger.info("?ˆì§ˆ ê°œì„ ??ë²•ë ¹?´ì„ë¡€ ?°ì´???„ì²˜ë¦??œì‘")
        
        interpretation_dirs = list(Path("data/raw/legal_interpretations").glob("yearly_*"))
        processed_count = 0
        valid_count = 0
        
        for interpretation_dir in interpretation_dirs:
            json_files = list(interpretation_dir.glob("*.json"))
            
            for json_file in json_files:
                try:
                    self.logger.info(f"ì²˜ë¦¬ ì¤? {json_file}")
                    
                    # ?Œì¼??ì²?¬ ?¨ìœ„ë¡?ì²˜ë¦¬
                    for processed_interpretation in self.process_file_in_chunks(json_file, 'legal_interpretation'):
                        # ì¦‰ì‹œ ?€??(ë©”ëª¨ë¦??„ì  ë°©ì?)
                        self.save_single_document(processed_interpretation, "legal_interpretations")
                        processed_count += 1
                        
                        if processed_interpretation.get('is_valid', False):
                            valid_count += 1
                            self.stats['successful'] += 1
                        else:
                            self.stats['failed'] += 1
                        
                        # ë©”ëª¨ë¦?ì²´í¬
                        if not self.check_memory_limit():
                            self.force_garbage_collection()
                
                except Exception as e:
                    self.logger.error(f"ë²•ë ¹?´ì„ë¡€ ?„ì²˜ë¦??¤íŒ¨ {json_file}: {e}")
                    self.stats['failed'] += 1
        
        self.stats['by_type']['legal_interpretations'] = processed_count
        self.logger.info(f"ë²•ë ¹?´ì„ë¡€ ?°ì´???„ì²˜ë¦??„ë£Œ: {processed_count}ê°?(? íš¨: {valid_count}ê°?")
    
    def save_single_document(self, document: Dict[str, Any], data_type: str):
        """?¨ì¼ ë¬¸ì„œë¥?ì¦‰ì‹œ ?€??""
        # ?°ì´???€?…ë³„ ?”ë ‰? ë¦¬ ?ì„±
        type_dir = self.output_dir / data_type
        type_dir.mkdir(exist_ok=True)
        
        # ?Œì¼ëª??ì„±
        doc_id = document.get('id', 'unknown')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{data_type}_{doc_id}_{timestamp}.json"
        
        # ?Œì¼ ?€??
        file_path = type_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(document, f, ensure_ascii=False, indent=2)
    
    def calculate_quality_metrics(self):
        """?ˆì§ˆ ì§€??ê³„ì‚°"""
        total_docs = self.stats['total_processed']
        valid_docs = self.stats['successful']
        
        if total_docs > 0:
            # ?„ì„±??(? íš¨??ë¬¸ì„œ ë¹„ìœ¨)
            self.stats['quality_metrics']['completeness'] = valid_docs / total_docs
            
            # ?¼ê???(ëª¨ë“  ?°ì´???€?…ì—??? íš¨??ë¬¸ì„œê°€ ?ˆëŠ”ì§€)
            valid_types = sum(1 for count in self.stats['by_type'].values() if count > 0)
            total_types = len(self.stats['by_type'])
            self.stats['quality_metrics']['consistency'] = valid_types / total_types if total_types > 0 else 0
            
            # ?©ì–´ ?•ê·œ??(?±ê³µ??ë¬¸ì„œ ì¤??©ì–´ ?•ê·œ?”ê? ?ìš©??ë¹„ìœ¨)
            # ?¤ì œë¡œëŠ” ???•êµ??ê³„ì‚°???„ìš”?˜ì?ë§? ?¬ê¸°?œëŠ” ê°„ë‹¨??ì²˜ë¦¬
            self.stats['quality_metrics']['term_normalization'] = min(valid_docs / total_docs * 1.2, 1.0)
    
    def run_improved_preprocessing(self, data_types: List[str] = None):
        """?ˆì§ˆ ê°œì„ ???„ì²˜ë¦??¤í–‰"""
        if data_types is None:
            data_types = ['laws', 'precedents', 'constitutional_decisions', 'legal_interpretations']
        
        start_time = datetime.now()
        self.logger.info(f"?ˆì§ˆ ê°œì„ ???„ì²˜ë¦??œì‘: {data_types}")
        
        try:
            for data_type in data_types:
                if data_type == 'laws':
                    self.process_laws_improved()
                elif data_type == 'precedents':
                    self.process_precedents_improved()
                elif data_type == 'constitutional_decisions':
                    self.process_constitutional_decisions_improved()
                elif data_type == 'legal_interpretations':
                    self.process_legal_interpretations_improved()
                
                # ê°??°ì´???€??ì²˜ë¦¬ ??ë©”ëª¨ë¦??•ë¦¬
                self.force_garbage_collection()
                self.logger.info(f"ë©”ëª¨ë¦??¬ìš©?? {self.get_memory_usage():.2f}GB")
            
            # ?ˆì§ˆ ì§€??ê³„ì‚°
            self.calculate_quality_metrics()
            
            duration = datetime.now() - start_time
            self.logger.info(f"=== ?„ì²˜ë¦??„ë£Œ (?Œìš”?œê°„: {duration}) ===")
            self.print_statistics()
            
        except Exception as e:
            self.logger.error(f"?„ì²˜ë¦?ì¤??¤ë¥˜ ë°œìƒ: {e}")
            raise
    
    def print_statistics(self):
        """?µê³„ ì¶œë ¥"""
        self.logger.info("=== ì²˜ë¦¬ ?µê³„ ===")
        self.logger.info(f"ì´?ì²˜ë¦¬: {self.stats['total_processed']}ê°?)
        self.logger.info(f"?±ê³µ: {self.stats['successful']}ê°?)
        self.logger.info(f"?¤íŒ¨: {self.stats['failed']}ê°?)
        
        for data_type, count in self.stats['by_type'].items():
            self.logger.info(f"{data_type}: {count}ê°?)
        
        # ?ˆì§ˆ ì§€??ì¶œë ¥
        self.logger.info("=== ?ˆì§ˆ ì§€??===")
        for metric, score in self.stats['quality_metrics'].items():
            self.logger.info(f"{metric}: {score:.2%}")
        
        # ?„ì²´ ?ˆì§ˆ ?ìˆ˜ ê³„ì‚°
        overall_quality = sum(self.stats['quality_metrics'].values()) / len(self.stats['quality_metrics'])
        self.logger.info(f"?„ì²´ ?ˆì§ˆ ?ìˆ˜: {overall_quality:.2%}")

def main():
    parser = argparse.ArgumentParser(description="?ˆì§ˆ ê°œì„ ??Raw ?°ì´???„ì²˜ë¦?)
    parser.add_argument("--data-types", nargs='+', 
                       choices=["laws", "precedents", "constitutional", "interpretations", "all"],
                       default=["laws", "precedents"],
                       help="?„ì²˜ë¦¬í•  ?°ì´??? í˜•")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="ë°°ì¹˜ ?¬ê¸°")
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="ì²?¬ ?¬ê¸°")
    parser.add_argument("--max-memory", type=float, default=8.0,
                       help="ìµœë? ë©”ëª¨ë¦??¬ìš©??(GB)")
    parser.add_argument("--enable-normalization", action="store_true", default=True,
                       help="ë²•ë¥  ?©ì–´ ?•ê·œ???œì„±??)
    
    args = parser.parse_args()
    
    # ?°ì´???€??ì²˜ë¦¬
    if "all" in args.data_types:
        data_types = ["laws", "precedents", "constitutional_decisions", "legal_interpretations"]
    else:
        data_types = args.data_types
    
    # ?„ì²˜ë¦¬ê¸° ì´ˆê¸°??
    preprocessor = QualityImprovedPreprocessor(
        enable_term_normalization=args.enable_normalization,
        max_memory_usage=args.max_memory / 16,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size
    )
    
    # ?„ì²˜ë¦??¤í–‰
    preprocessor.run_improved_preprocessing(data_types)

if __name__ == "__main__":
    main()
