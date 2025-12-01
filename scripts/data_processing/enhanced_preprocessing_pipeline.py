#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Preprocessing Pipeline

?µí•©???„ì²˜ë¦??Œì´?„ë¼?¸ìœ¼ë¡?ë²•ë ¹ê³??ë? ?°ì´?°ë? ?¨ìœ¨?ìœ¼ë¡?ì²˜ë¦¬?©ë‹ˆ??
- ë©”ëª¨ë¦?ìµœì ??
- ë³‘ë ¬ ì²˜ë¦¬
- ?ˆì§ˆ ê²€ì¦?
- ?¤ë¥˜ ë³µêµ¬
- ì§„í–‰ ?í™© ì¶”ì 
"""

import os
import sys
import json
import logging
import hashlib
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import argparse
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import gc
import psutil

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from lawfirm_langgraph.core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2 as DatabaseManager
from scripts.data_collection.common.checkpoint_manager import CheckpointManager


@dataclass
class ProcessingConfig:
    """ì²˜ë¦¬ ?¤ì • ?°ì´???´ë˜??""
    # ê¸°ë³¸ ?¤ì •
    max_workers: int = min(multiprocessing.cpu_count(), 8)
    batch_size: int = 100
    max_memory_gb: float = 8.0
    
    # ?„ì²˜ë¦??¤ì •
    enable_legal_analysis: bool = True
    enable_quality_validation: bool = True
    enable_duplicate_detection: bool = True
    
    # ì¶œë ¥ ?¤ì •
    output_format: str = "json"  # json, parquet, csv
    compress_output: bool = False
    
    # ?¤ë¥˜ ì²˜ë¦¬
    max_retries: int = 3
    retry_delay: float = 1.0
    continue_on_error: bool = True


@dataclass
class ProcessingResult:
    """ì²˜ë¦¬ ê²°ê³¼ ?°ì´???´ë˜??""
    success: bool
    processed_files: List[Path] = field(default_factory=list)
    failed_files: List[Path] = field(default_factory=list)
    total_records: int = 0
    processing_time: float = 0.0
    memory_peak_mb: float = 0.0
    errors: List[str] = field(default_factory=list)
    quality_issues: List[Dict[str, Any]] = field(default_factory=list)
    duplicates_found: int = 0


class MemoryMonitor:
    """ë©”ëª¨ë¦??¬ìš©??ëª¨ë‹ˆ?°ë§ ?´ë˜??""
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_gb = max_memory_gb
        self.peak_memory_mb = 0.0
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> float:
        """?„ì¬ ë©”ëª¨ë¦??¬ìš©?‰ì„ MB ?¨ìœ„ë¡?ë°˜í™˜"""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        return memory_mb
    
    def check_memory_limit(self) -> bool:
        """ë©”ëª¨ë¦??œê³„ ?•ì¸"""
        current_mb = self.get_memory_usage()
        return current_mb < (self.max_memory_gb * 1024)
    
    def force_gc(self):
        """ê°•ì œ ê°€ë¹„ì? ì»¬ë ‰???¤í–‰"""
        gc.collect()


class QualityValidator:
    """?°ì´???ˆì§ˆ ê²€ì¦??´ë˜??""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_law_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ë²•ë ¹ ?°ì´???ˆì§ˆ ê²€ì¦?""
        issues = []
        
        # ?„ìˆ˜ ?„ë“œ ê²€ì¦?
        required_fields = ['law_id', 'law_name', 'articles']
        for field in required_fields:
            if field not in data or not data[field]:
                issues.append({
                    'type': 'missing_field',
                    'field': field,
                    'severity': 'high',
                    'message': f'Missing required field: {field}'
                })
        
        # ì¡°ë¬¸ ?°ì´??ê²€ì¦?
        if 'articles' in data and isinstance(data['articles'], list):
            for i, article in enumerate(data['articles']):
                if not isinstance(article, dict):
                    issues.append({
                        'type': 'invalid_article',
                        'index': i,
                        'severity': 'medium',
                        'message': f'Article {i} is not a valid dictionary'
                    })
                elif 'article_number' not in article or 'content' not in article:
                    issues.append({
                        'type': 'incomplete_article',
                        'index': i,
                        'severity': 'high',
                        'message': f'Article {i} missing required fields'
                    })
        
        return issues
    
    def validate_precedent_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """?ë? ?°ì´???ˆì§ˆ ê²€ì¦?""
        issues = []
        
        # ?„ìˆ˜ ?„ë“œ ê²€ì¦?
        required_fields = ['case_id', 'case_name', 'case_number', 'decision_date']
        for field in required_fields:
            if field not in data or not data[field]:
                issues.append({
                    'type': 'missing_field',
                    'field': field,
                    'severity': 'high',
                    'message': f'Missing required field: {field}'
                })
        
        # ? ì§œ ?•ì‹ ê²€ì¦?
        if 'decision_date' in data:
            try:
                datetime.strptime(data['decision_date'], '%Y-%m-%d')
            except ValueError:
                issues.append({
                    'type': 'invalid_date',
                    'field': 'decision_date',
                    'severity': 'medium',
                    'message': f'Invalid date format: {data["decision_date"]}'
                })
        
        return issues


class DuplicateDetector:
    """ì¤‘ë³µ ?°ì´???ì? ?´ë˜??""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.seen_hashes = set()
    
    def calculate_content_hash(self, data: Dict[str, Any]) -> str:
        """?°ì´???´ìš©???´ì‹œê°?ê³„ì‚°"""
        # ?µì‹¬ ?„ë“œë§Œìœ¼ë¡??´ì‹œ ê³„ì‚°
        if 'law_id' in data:
            # ë²•ë ¹ ?°ì´??
            key_fields = ['law_id', 'law_name']
        else:
            # ?ë? ?°ì´??
            key_fields = ['case_id', 'case_name', 'case_number']
        
        content = ''.join(str(data.get(field, '')) for field in key_fields)
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, data: Dict[str, Any]) -> bool:
        """ì¤‘ë³µ ?°ì´???¬ë? ?•ì¸"""
        content_hash = self.calculate_content_hash(data)
        
        if content_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(content_hash)
        return False


class EnhancedPreprocessingPipeline:
    """?¥ìƒ???„ì²˜ë¦??Œì´?„ë¼??""
    
    def __init__(self, 
                 config: ProcessingConfig = None,
                 checkpoint_manager: CheckpointManager = None,
                 db_manager: DatabaseManager = None):
        """
        ?„ì²˜ë¦??Œì´?„ë¼??ì´ˆê¸°??
        
        Args:
            config: ì²˜ë¦¬ ?¤ì •
            checkpoint_manager: ì²´í¬?¬ì¸??ê´€ë¦¬ì
            db_manager: ?°ì´?°ë² ?´ìŠ¤ ê´€ë¦¬ì
        """
        self.config = config or ProcessingConfig()
        self.checkpoint_manager = checkpoint_manager
        self.db_manager = db_manager
        
        # ë¡œê¹… ?¤ì •
        self.logger = logging.getLogger(__name__)
        
        # ëª¨ë‹ˆ?°ë§ ë°?ê²€ì¦?ì»´í¬?ŒíŠ¸
        self.memory_monitor = MemoryMonitor(self.config.max_memory_gb)
        self.quality_validator = QualityValidator()
        self.duplicate_detector = DuplicateDetector()
        
        # ?µê³„
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_records': 0,
            'duplicates_found': 0,
            'quality_issues': 0,
            'start_time': None,
            'end_time': None
        }
    
    def process_law_files(self, 
                         input_paths: List[Path], 
                         output_dir: Path) -> ProcessingResult:
        """ë²•ë ¹ ?Œì¼??ì²˜ë¦¬"""
        self.logger.info(f"Processing {len(input_paths)} law files...")
        
        result = ProcessingResult(success=True)
        result.processing_time = datetime.now()
        
        # ë°°ì¹˜ë³?ì²˜ë¦¬
        for i in range(0, len(input_paths), self.config.batch_size):
            batch = input_paths[i:i + self.config.batch_size]
            
            # ë©”ëª¨ë¦??•ì¸
            if not self.memory_monitor.check_memory_limit():
                self.logger.warning("Memory limit reached, forcing garbage collection")
                self.memory_monitor.force_gc()
            
            # ë°°ì¹˜ ì²˜ë¦¬
            batch_result = self._process_law_batch(batch, output_dir)
            
            # ê²°ê³¼ ë³‘í•©
            result.processed_files.extend(batch_result.processed_files)
            result.failed_files.extend(batch_result.failed_files)
            result.total_records += batch_result.total_records
            result.quality_issues.extend(batch_result.quality_issues)
            result.duplicates_found += batch_result.duplicates_found
            
            if not batch_result.success:
                result.success = False
                result.errors.extend(batch_result.errors)
        
        result.processing_time = (datetime.now() - result.processing_time).total_seconds()
        result.memory_peak_mb = self.memory_monitor.peak_memory_mb
        
        return result
    
    def process_precedent_files(self, 
                               input_paths: List[Path], 
                               output_dir: Path) -> ProcessingResult:
        """?ë? ?Œì¼??ì²˜ë¦¬"""
        self.logger.info(f"Processing {len(input_paths)} precedent files...")
        
        result = ProcessingResult(success=True)
        result.processing_time = datetime.now()
        
        # ë°°ì¹˜ë³?ì²˜ë¦¬
        for i in range(0, len(input_paths), self.config.batch_size):
            batch = input_paths[i:i + self.config.batch_size]
            
            # ë©”ëª¨ë¦??•ì¸
            if not self.memory_monitor.check_memory_limit():
                self.logger.warning("Memory limit reached, forcing garbage collection")
                self.memory_monitor.force_gc()
            
            # ë°°ì¹˜ ì²˜ë¦¬
            batch_result = self._process_precedent_batch(batch, output_dir)
            
            # ê²°ê³¼ ë³‘í•©
            result.processed_files.extend(batch_result.processed_files)
            result.failed_files.extend(batch_result.failed_files)
            result.total_records += batch_result.total_records
            result.quality_issues.extend(batch_result.quality_issues)
            result.duplicates_found += batch_result.duplicates_found
            
            if not batch_result.success:
                result.success = False
                result.errors.extend(batch_result.errors)
        
        result.processing_time = (datetime.now() - result.processing_time).total_seconds()
        result.memory_peak_mb = self.memory_monitor.peak_memory_mb
        
        return result
    
    def _process_law_batch(self, 
                          batch: List[Path], 
                          output_dir: Path) -> ProcessingResult:
        """ë²•ë ¹ ë°°ì¹˜ ì²˜ë¦¬"""
        result = ProcessingResult(success=True)
        
        for file_path in tqdm(batch, desc="Processing law batch"):
            try:
                # ?Œì¼ ?½ê¸°
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # ì¤‘ë³µ ?•ì¸
                if self.duplicate_detector.is_duplicate(data):
                    result.duplicates_found += 1
                    continue
                
                # ?ˆì§ˆ ê²€ì¦?
                if self.config.enable_quality_validation:
                    quality_issues = self.quality_validator.validate_law_data(data)
                    if quality_issues:
                        result.quality_issues.extend(quality_issues)
                        result.quality_issues += len(quality_issues)
                
                # ?„ì²˜ë¦?(ê¸°ì¡´ ?Œì„œ ?¬ìš©)
                processed_data = self._preprocess_law_data(data)
                
                # ì¶œë ¥ ?Œì¼ ?€??
                output_file = output_dir / f"{file_path.stem}_processed.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                
                result.processed_files.append(output_file)
                result.total_records += len(processed_data.get('articles', []))
                
            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                self.logger.error(error_msg)
                result.failed_files.append(file_path)
                result.errors.append(error_msg)
                
                if not self.config.continue_on_error:
                    result.success = False
                    break
        
        return result
    
    def _process_precedent_batch(self, 
                                batch: List[Path], 
                                output_dir: Path) -> ProcessingResult:
        """?ë? ë°°ì¹˜ ì²˜ë¦¬"""
        result = ProcessingResult(success=True)
        
        for file_path in tqdm(batch, desc="Processing precedent batch"):
            try:
                # ?Œì¼ ?½ê¸°
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # ?°ì´??êµ¬ì¡° ?•ì¸ ë°?ì²˜ë¦¬
                if isinstance(data, dict) and 'items' in data:
                    # items ë°°ì—´ ì²˜ë¦¬
                    items = data.get('items', [])
                    if not isinstance(items, list):
                        self.logger.warning(f"Items is not a list in {file_path}")
                        continue
                    
                    processed_items = []
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        
                        # ì¤‘ë³µ ?•ì¸
                        if self.duplicate_detector.is_duplicate(item):
                            result.duplicates_found += 1
                            continue
                        
                        # ?ˆì§ˆ ê²€ì¦?
                        if self.config.enable_quality_validation:
                            quality_issues = self.quality_validator.validate_precedent_data(item)
                            if quality_issues:
                                result.quality_issues.extend(quality_issues)
                        
                        # ?„ì²˜ë¦?
                        processed_item = self._preprocess_precedent_data(item)
                        processed_items.append(processed_item)
                    
                    # ì¶œë ¥ ?Œì¼ ?€??
                    output_file = output_dir / f"{file_path.stem}_processed.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'metadata': data.get('metadata', {}),
                            'items': processed_items
                        }, f, ensure_ascii=False, indent=2)
                    
                    result.processed_files.append(output_file)
                    result.total_records += len(processed_items)
                
                elif isinstance(data, list):
                    # ì§ì ‘ ë°°ì—´??ê²½ìš°
                    processed_items = []
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        
                        # ì¤‘ë³µ ?•ì¸
                        if self.duplicate_detector.is_duplicate(item):
                            result.duplicates_found += 1
                            continue
                        
                        # ?ˆì§ˆ ê²€ì¦?
                        if self.config.enable_quality_validation:
                            quality_issues = self.quality_validator.validate_precedent_data(item)
                            if quality_issues:
                                result.quality_issues.extend(quality_issues)
                        
                        # ?„ì²˜ë¦?
                        processed_item = self._preprocess_precedent_data(item)
                        processed_items.append(processed_item)
                    
                    # ì¶œë ¥ ?Œì¼ ?€??
                    output_file = output_dir / f"{file_path.stem}_processed.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(processed_items, f, ensure_ascii=False, indent=2)
                    
                    result.processed_files.append(output_file)
                    result.total_records += len(processed_items)
                
                else:
                    self.logger.warning(f"Unexpected data structure in {file_path}")
                    continue
                
            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                self.logger.error(error_msg)
                result.failed_files.append(file_path)
                result.errors.append(error_msg)
                
                if not self.config.continue_on_error:
                    result.success = False
                    break
        
        return result
    
    def _preprocess_law_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """ë²•ë ¹ ?°ì´???„ì²˜ë¦?""
        # ê¸°ë³¸ ?„ì²˜ë¦?ë¡œì§ (ê¸°ì¡´ ?Œì„œ ?œìš©)
        processed = {
            'law_id': data.get('law_id', ''),
            'law_name': data.get('law_name', ''),
            'law_type': data.get('law_type', ''),
            'enactment_date': data.get('enactment_date', ''),
            'articles': []
        }
        
        # ì¡°ë¬¸ ì²˜ë¦¬
        if 'articles' in data and isinstance(data['articles'], list):
            for article in data['articles']:
                if isinstance(article, dict):
                    processed_article = {
                        'article_number': article.get('article_number', ''),
                        'title': article.get('title', ''),
                        'content': article.get('content', ''),
                        'searchable_text': self._generate_searchable_text(article.get('content', ''))
                    }
                    processed['articles'].append(processed_article)
        
        return processed
    
    def _preprocess_precedent_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """?ë? ?°ì´???„ì²˜ë¦?""
        # ê¸°ë³¸ ?„ì²˜ë¦?ë¡œì§
        processed = {
            'case_id': data.get('case_id', ''),
            'case_name': data.get('case_name', ''),
            'case_number': data.get('case_number', ''),
            'decision_date': data.get('decision_date', ''),
            'court': data.get('court', ''),
            'category': data.get('category', ''),
            'field': data.get('field', ''),
            'detail_url': data.get('detail_url', ''),
            'full_text': data.get('precedent_content', ''),
            'searchable_text': self._generate_searchable_text(data.get('precedent_content', '')),
            'created_at': datetime.now().isoformat()
        }
        
        return processed
    
    def _generate_searchable_text(self, content: str) -> str:
        """ê²€??ê°€?¥í•œ ?ìŠ¤???ì„±"""
        if not content:
            return ""
        
        # ê¸°ë³¸ ?•ê·œ??
        import re
        # HTML ?œê·¸ ?œê±°
        content = re.sub(r'<[^>]+>', '', content)
        # ?°ì† ê³µë°± ?œê±°
        content = re.sub(r'\s+', ' ', content)
        # ?ë’¤ ê³µë°± ?œê±°
        content = content.strip()
        
        return content
    
    def generate_report(self, result: ProcessingResult) -> Dict[str, Any]:
        """ì²˜ë¦¬ ê²°ê³¼ ë³´ê³ ???ì„±"""
        return {
            'summary': {
                'success': result.success,
                'processed_files': len(result.processed_files),
                'failed_files': len(result.failed_files),
                'total_records': result.total_records,
                'processing_time_seconds': result.processing_time,
                'memory_peak_mb': result.memory_peak_mb,
                'duplicates_found': result.duplicates_found,
                'quality_issues': len(result.quality_issues)
            },
            'errors': result.errors,
            'quality_issues': result.quality_issues[:10],  # ?ìœ„ 10ê°œë§Œ
            'failed_files': [str(f) for f in result.failed_files[:10]]  # ?ìœ„ 10ê°œë§Œ
        }


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    parser = argparse.ArgumentParser(description='Enhanced Preprocessing Pipeline')
    parser.add_argument('--input', required=True, help='Input directory or file pattern')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--data-type', choices=['law', 'precedent'], required=True, help='Data type')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of workers')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    parser.add_argument('--max-memory-gb', type=float, default=8.0, help='Maximum memory usage in GB')
    parser.add_argument('--enable-quality-validation', action='store_true', help='Enable quality validation')
    parser.add_argument('--enable-duplicate-detection', action='store_true', help='Enable duplicate detection')
    
    args = parser.parse_args()
    
    # ë¡œê¹… ?¤ì •
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ?¤ì • ?ì„±
    config = ProcessingConfig(
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        max_memory_gb=args.max_memory_gb,
        enable_quality_validation=args.enable_quality_validation,
        enable_duplicate_detection=args.enable_duplicate_detection
    )
    
    # ?Œì´?„ë¼??ì´ˆê¸°??
    pipeline = EnhancedPreprocessingPipeline(config=config)
    
    # ?…ë ¥ ?Œì¼ ?˜ì§‘
    input_path = Path(args.input)
    if input_path.is_file():
        input_files = [input_path]
    else:
        input_files = list(input_path.rglob('*.json'))
    
    # ì¶œë ¥ ?”ë ‰? ë¦¬ ?ì„±
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ì²˜ë¦¬ ?¤í–‰
    if args.data_type == 'law':
        result = pipeline.process_law_files(input_files, output_dir)
    else:
        result = pipeline.process_precedent_files(input_files, output_dir)
    
    # ë³´ê³ ???ì„± ë°??€??
    report = pipeline.generate_report(result)
    report_file = output_dir / 'processing_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # ê²°ê³¼ ì¶œë ¥
    print(f"\n=== Processing Complete ===")
    print(f"Success: {result.success}")
    print(f"Processed files: {len(result.processed_files)}")
    print(f"Failed files: {len(result.failed_files)}")
    print(f"Total records: {result.total_records}")
    print(f"Processing time: {result.processing_time:.2f} seconds")
    print(f"Memory peak: {result.memory_peak_mb:.2f} MB")
    print(f"Duplicates found: {result.duplicates_found}")
    print(f"Quality issues: {len(result.quality_issues)}")
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
