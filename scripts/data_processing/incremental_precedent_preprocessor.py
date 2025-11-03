#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ì¦ë¶„ ?ë? ?„ì²˜ë¦??„ë¡œ?¸ì„œ

ë¯¸ì²˜ë¦??ë? ?Œì¼ë§?? ë³„?˜ì—¬ ?„ì²˜ë¦¬í•˜??ì¦ë¶„ ì²˜ë¦¬ ?œìŠ¤?œìž…?ˆë‹¤.
ê¸°ì¡´ PrecedentPreprocessorë¥??¬ì‚¬?©í•˜ê³?ì²´í¬?¬ì¸???œìŠ¤?œì„ ?µí•©?©ë‹ˆ??
"""

import os
import sys
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import argparse
from dataclasses import dataclass
from tqdm import tqdm

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.data_processing.precedent_preprocessor import PrecedentPreprocessor
from source.data.database import DatabaseManager
from scripts.data_collection.common.checkpoint_manager import CheckpointManager
from scripts.data_processing.auto_data_detector import AutoDataDetector


@dataclass
class ProcessingResult:
    """ì²˜ë¦¬ ê²°ê³¼ ?°ì´???´ëž˜??""
    success: bool
    processed_files: List[Path]
    failed_files: List[Path]
    total_records: int
    processing_time: float
    error_messages: List[str]


class IncrementalPrecedentPreprocessor:
    """ì¦ë¶„ ?ë? ?„ì²˜ë¦??„ë¡œ?¸ì„œ ?´ëž˜??""
    
    def __init__(self, 
                 raw_data_base_path: str = "data/raw/assembly",
                 processed_data_base_path: str = "data/processed/assembly",
                 processing_version: str = "1.0",
                 checkpoint_manager: CheckpointManager = None,
                 db_manager: DatabaseManager = None,
                 enable_term_normalization: bool = True,
                 batch_size: int = 100):
        """
        ì¦ë¶„ ?ë? ?„ì²˜ë¦??„ë¡œ?¸ì„œ ì´ˆê¸°??
        
        Args:
            raw_data_base_path: ?ë³¸ ?°ì´??ê¸°ë³¸ ê²½ë¡œ
            processed_data_base_path: ?„ì²˜ë¦¬ëœ ?°ì´??ê¸°ë³¸ ê²½ë¡œ
            processing_version: ì²˜ë¦¬ ë²„ì „
            checkpoint_manager: ì²´í¬?¬ì¸??ê´€ë¦¬ìž
            db_manager: ?°ì´?°ë² ?´ìŠ¤ ê´€ë¦¬ìž
            enable_term_normalization: ë²•ë¥  ?©ì–´ ?•ê·œ???œì„±??
            batch_size: ë°°ì¹˜ ì²˜ë¦¬ ?¬ê¸°
        """
        self.raw_data_base_path = Path(raw_data_base_path)
        self.processed_data_base_path = Path(processed_data_base_path)
        self.processed_data_base_path.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_manager = checkpoint_manager
        self.db_manager = db_manager or DatabaseManager()
        self.batch_size = batch_size
        self.processing_version = processing_version
        
        # ?ë? ?„ì²˜ë¦¬ê¸° ì´ˆê¸°??
        self.preprocessor = PrecedentPreprocessor(enable_term_normalization)
        
        # ?ë™ ?°ì´??ê°ì?ê¸?ì´ˆê¸°??
        self.auto_detector = AutoDataDetector(raw_data_base_path)
        
        # ì¶œë ¥ ?”ë ‰? ë¦¬ ?¤ì •
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ë¡œê¹… ?¤ì •
        self.logger = logging.getLogger(__name__)
        
        # ì²˜ë¦¬ ?µê³„
        self.stats = {
            'total_scanned_files': 0,
            'new_files_to_process': 0,
            'successfully_processed': 0,
            'failed_to_process': 0,
            'skipped_already_processed': 0,
            'errors': []
        }
        
        self.logger.info("IncrementalPrecedentPreprocessor initialized")
    
    def process_new_data_only(self, category: str = "civil") -> Dict[str, Any]:
        """
        ?ˆë¡œ ì¶”ê????ë? ?°ì´?°ë§Œ ê°ì??˜ì—¬ ?„ì²˜ë¦?
        
        Args:
            category: ì²˜ë¦¬???¹ì • ì¹´í…Œê³ ë¦¬ (civil, criminal, family)
            
        Returns:
            Dict[str, Any]: ì²˜ë¦¬ ê²°ê³¼ ?µê³„
        """
        self.logger.info(f"Starting incremental precedent preprocessing for category: {category}")
        start_time = datetime.now()
        
        # ?°ì´???€??ê²°ì •
        data_type = f"precedent_{category}"
        
        # ?ˆë¡œ???Œì¼ ê°ì?
        new_files_by_type = self.auto_detector.detect_new_data_sources(
            str(self.raw_data_base_path / "precedent"), 
            data_type
        )
        
        files_to_process = new_files_by_type.get(data_type, [])
        
        self.stats['total_scanned_files'] = sum(len(f) for f in new_files_by_type.values())
        self.stats['new_files_to_process'] = len(files_to_process)
        
        if not files_to_process:
            self.logger.info("No new precedent files to preprocess.")
            self.stats['end_time'] = datetime.now().isoformat()
            self.stats['duration'] = (datetime.now() - start_time).total_seconds()
            return self.stats

        self.logger.info(f"Found {len(files_to_process)} new precedent files for preprocessing.")

        for file_path in files_to_process:
            raw_file_path_str = str(file_path)
            file_hash = self.auto_detector.get_file_hash(file_path)
            
            # ?´ë? ì²˜ë¦¬???Œì¼?¸ì? ?¤ì‹œ ?•ì¸ (ê²½ìŸ ì¡°ê±´ ë°©ì?)
            if self.db_manager.is_file_processed(raw_file_path_str):
                self.stats['skipped_already_processed'] += 1
                self.logger.info(f"Skipping already processed file: {file_path}")
                continue

            try:
                # ?°ì´??ë¡œë“œ
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # ?„ì²˜ë¦??˜í–‰ - ?ë? ?°ì´??êµ¬ì¡°??ë§žê²Œ ë³€??
                processed_data = self._process_assembly_precedent_data(raw_data, category)
                
                # ì¶œë ¥ ê²½ë¡œ ?¤ì • (?? data/processed/assembly/precedent/civil/20251016/ml_enhanced_...)
                relative_path = file_path.relative_to(self.raw_data_base_path / "precedent")
                output_subdir = self.processed_data_base_path / "precedent" / category / relative_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)
                
                output_file_name = f"ml_enhanced_{file_path.stem}.json"
                output_file_path = output_subdir / output_file_name
                
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                
                # ì²˜ë¦¬ ?´ë ¥ ê¸°ë¡
                record_count = len(processed_data.get('cases', [])) if isinstance(processed_data, dict) else 1
                self.db_manager.mark_file_as_processed(
                    raw_file_path_str, file_hash, data_type, 
                    record_count=record_count, processing_version=self.processing_version
                )
                self.stats['successfully_processed'] += 1
                self.logger.info(f"Successfully preprocessed and marked as processed: {file_path}")

            except Exception as e:
                error_msg = f"Failed to preprocess file {file_path}: {e}"
                self.logger.error(error_msg)
                self.stats['failed_to_process'] += 1
                self.stats['errors'].append(error_msg)
                self.db_manager.mark_file_as_processed(
                    raw_file_path_str, file_hash, data_type, 
                    error_message=error_msg,
                    processing_version=self.processing_version
                )
                
        self.stats['end_time'] = datetime.now().isoformat()
        self.stats['duration'] = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Incremental precedent preprocessing completed. Stats: {self.stats}")
        return self.stats
    
    def _process_assembly_precedent_data(self, raw_data: Dict[str, Any], category: str) -> Dict[str, Any]:
        """
        êµ?šŒ ?ë? ?°ì´??êµ¬ì¡°??ë§žê²Œ ?„ì²˜ë¦?
        
        Args:
            raw_data: ?ë³¸ ?°ì´??(metadata, items êµ¬ì¡°)
            category: ?ë? ì¹´í…Œê³ ë¦¬
            
        Returns:
            Dict[str, Any]: ?„ì²˜ë¦¬ëœ ?°ì´??
        """
        try:
            # PrecedentPreprocessorë¥??¬ìš©?˜ì—¬ ì²˜ë¦¬
            processed_data = self.preprocessor.process_precedent_data(raw_data)
            
            # ì¹´í…Œê³ ë¦¬ ?•ë³´ ì¶”ê?
            if isinstance(processed_data, dict) and 'metadata' in processed_data:
                processed_data['metadata']['category'] = category
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing assembly precedent data: {e}")
            return {'cases': []}
    
    def process_new_files_only(self, files: List[Path], 
                              category: str = "civil") -> ProcessingResult:
        """
        ?ˆë¡œ???Œì¼ë§?ì²˜ë¦¬
        
        Args:
            files: ì²˜ë¦¬???Œì¼ ëª©ë¡
            category: ?ë? ì¹´í…Œê³ ë¦¬
        
        Returns:
            ProcessingResult: ì²˜ë¦¬ ê²°ê³¼
        """
        self.logger.info(f"Starting processing for {len(files)} new precedent files of category: {category}")
        start_time = datetime.now()
        
        processed_files = []
        failed_files = []
        total_records = 0
        error_messages = []
        
        # ì²´í¬?¬ì¸?¸ì—???¬ê°œ?????ˆëŠ”ì§€ ?•ì¸
        checkpoint_data = self._load_checkpoint()
        if checkpoint_data:
            self.logger.info("Resuming from checkpoint...")
            start_index = checkpoint_data.get('current_file_index', 0)
            processed_files.extend(checkpoint_data.get('processed_files', []))
            failed_files.extend(checkpoint_data.get('failed_files', []))
        else:
            start_index = 0
        
        # ?Œì¼ë³?ì²˜ë¦¬
        for i, file_path in enumerate(tqdm(files[start_index:], 
                                          desc="Processing precedent files",
                                          initial=start_index,
                                          total=len(files))):
            try:
                # ?Œì¼ ?´ì‹œ ê³„ì‚°
                file_hash = self._calculate_file_hash(file_path)
                
                # ?´ë? ì²˜ë¦¬???Œì¼?¸ì? ?•ì¸
                if self.db_manager.is_file_processed(str(file_path)):
                    self.logger.debug(f"File already processed: {file_path}")
                    continue
                
                # ?¨ì¼ ?Œì¼ ì²˜ë¦¬
                file_result = self._process_single_file(file_path, category)
                
                if file_result['success']:
                    processed_files.append(file_path)
                    total_records += file_result['record_count']
                    self.db_manager.mark_file_as_processed(
                        str(file_path), file_hash, f"precedent_{category}", 
                        record_count=file_result['record_count'], 
                        processing_version=self.processing_version
                    )
                else:
                    failed_files.append(file_path)
                    error_messages.append(file_result['error'])
                    self.db_manager.mark_file_as_processed(
                        str(file_path), file_hash, f"precedent_{category}", 
                        error_message=file_result['error'],
                        processing_version=self.processing_version
                    )
                
                # ì²´í¬?¬ì¸???€??
                self._save_checkpoint(i + start_index + 1, processed_files, failed_files)
                
            except Exception as e:
                error_msg = f"Error processing file {file_path}: {e}"
                self.logger.error(error_msg)
                failed_files.append(file_path)
                error_messages.append(error_msg)
                self.db_manager.mark_file_as_processed(
                    str(file_path), self._calculate_file_hash(file_path), f"precedent_{category}", 
                    error_message=error_msg,
                    processing_version=self.processing_version
                )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = ProcessingResult(
            success=len(failed_files) == 0,
            processed_files=processed_files,
            failed_files=failed_files,
            total_records=total_records,
            processing_time=duration,
            error_messages=error_messages
        )
        
        self.logger.info(f"Processing completed: {len(processed_files)} success, "
                        f"{len(failed_files)} failed, {total_records} total records")
        
        return result
    
    def _process_single_file(self, file_path: Path, category: str) -> Dict[str, Any]:
        """
        ?¨ì¼ ?Œì¼ ì²˜ë¦¬
        
        Args:
            file_path: ì²˜ë¦¬???Œì¼ ê²½ë¡œ
            category: ?ë? ì¹´í…Œê³ ë¦¬
        
        Returns:
            Dict[str, Any]: ì²˜ë¦¬ ê²°ê³¼ (?±ê³µ ?¬ë?, ?ˆì½”???? ?ëŸ¬ ë©”ì‹œì§€)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            processed_data = self._process_assembly_precedent_data(raw_data, category)
            
            # ì²˜ë¦¬???°ì´?°ë? ?„ì‹œ ?Œì¼ë¡??€?¥í•˜ê±°ë‚˜, ë©”ëª¨ë¦¬ì—???¤ìŒ ?¨ê³„ë¡??„ë‹¬
            record_count = len(processed_data.get('cases', [])) if isinstance(processed_data, dict) else 1
            
            return {'success': True, 'record_count': record_count, 'error': None}
        except Exception as e:
            return {'success': False, 'record_count': 0, 'error': str(e)}

    def _calculate_file_hash(self, file_path: Path) -> str:
        """?Œì¼ ?´ìš©??SHA256 ?´ì‹œë¥?ê³„ì‚°?˜ì—¬ ë°˜í™˜"""
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def _save_checkpoint(self, current_file_index: int, 
                         processed_files: List[Path], 
                         failed_files: List[Path]):
        """
        ?„ìž¬ ì²˜ë¦¬ ?íƒœë¥?ì²´í¬?¬ì¸?¸ë¡œ ?€??
        """
        if self.checkpoint_manager:
            checkpoint_data = {
                'current_file_index': current_file_index,
                'processed_files': [str(p) for p in processed_files],
                'failed_files': [str(p) for p in failed_files],
                'timestamp': datetime.now().isoformat()
            }
            self.checkpoint_manager.save_checkpoint('incremental_precedent_preprocessing', checkpoint_data)
            self.logger.debug(f"Checkpoint saved at index: {current_file_index}")

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        ì²´í¬?¬ì¸??ë¡œë“œ
        """
        if self.checkpoint_manager:
            return self.checkpoint_manager.load_checkpoint('incremental_precedent_preprocessing')
        return None


def main():
    parser = argparse.ArgumentParser(description="ì¦ë¶„ ?ë? ?°ì´???„ì²˜ë¦??„ë¡œ?¸ì„œ")
    parser.add_argument('--input-files', nargs='*', type=Path,
                        help='ì²˜ë¦¬???Œì¼ ëª©ë¡')
    parser.add_argument('--category', default='civil',
                        choices=['civil', 'criminal', 'family'],
                        help='?ë? ì¹´í…Œê³ ë¦¬')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='ë°°ì¹˜ ì²˜ë¦¬ ?¬ê¸°')
    parser.add_argument('--checkpoint-dir', default='data/checkpoints',
                        help='ì²´í¬?¬ì¸???”ë ‰? ë¦¬')
    parser.add_argument('--resume', action='store_true',
                        help='ì²´í¬?¬ì¸?¸ì—???¬ê°œ')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='?ì„¸ ë¡œê·¸ ì¶œë ¥')
    
    args = parser.parse_args()
    
    # ë¡œê¹… ?¤ì •
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # ì²´í¬?¬ì¸??ê´€ë¦¬ìž ì´ˆê¸°??
        checkpoint_manager = CheckpointManager(args.checkpoint_dir)
        
        # ì¦ë¶„ ?ë? ?„ì²˜ë¦??„ë¡œ?¸ì„œ ì´ˆê¸°??
        preprocessor = IncrementalPrecedentPreprocessor(
            checkpoint_manager=checkpoint_manager,
            batch_size=args.batch_size
        )
        
        # ?Œì¼ ì²˜ë¦¬ ?¤í–‰
        if args.input_files:
            result = preprocessor.process_new_files_only(args.input_files, args.category)
            # ê²°ê³¼ ì¶œë ¥
            logging.info("Processing Results:")
            logging.info(f"  Success: {len(result.processed_files)} files")
            logging.info(f"  Failed: {len(result.failed_files)} files")
            logging.info(f"  Total records: {result.total_records}")
            logging.info(f"  Processing time: {result.processing_time:.2f} seconds")
            
            if result.error_messages:
                logging.error("Error messages:")
                for error in result.error_messages:
                    logging.error(f"  - {error}")
            
            return result.success
        else:
            # ?ë™?¼ë¡œ ???Œì¼ ê°ì??˜ì—¬ ì²˜ë¦¬
            stats = preprocessor.process_new_data_only(args.category)
            
            # ê²°ê³¼ ì¶œë ¥
            logging.info("Processing Results:")
            logging.info(f"  Total scanned files: {stats['total_scanned_files']}")
            logging.info(f"  New files to process: {stats['new_files_to_process']}")
            logging.info(f"  Successfully processed: {stats['successfully_processed']}")
            logging.info(f"  Failed to process: {stats['failed_to_process']}")
            logging.info(f"  Skipped already processed: {stats['skipped_already_processed']}")
            logging.info(f"  Duration: {stats['duration']:.2f} seconds")
            
            if stats['errors']:
                logging.error("Error messages:")
                for error in stats['errors']:
                    logging.error(f"  - {error}")
            
            return stats['successfully_processed'] > 0
            
    except Exception as e:
        logging.error(f"Error in incremental precedent preprocessing: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
