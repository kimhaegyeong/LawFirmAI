#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ì¦ë¶„ ?„ì²˜ë¦??„ë¡œ?¸ì„œ

ë¯¸ì²˜ë¦??Œì¼ë§?? ë³„?˜ì—¬ ?„ì²˜ë¦¬í•˜??ì¦ë¶„ ì²˜ë¦¬ ?œìŠ¤?œì…?ˆë‹¤.
ê¸°ì¡´ LegalDataProcessorë¥??¬ì‚¬?©í•˜ê³?ì²´í¬?¬ì¸???œìŠ¤?œì„ ?µí•©?©ë‹ˆ??
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

from source.data.data_processor import LegalDataProcessor
from lawfirm_langgraph.core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2 as DatabaseManager
from scripts.data_collection.common.checkpoint_manager import CheckpointManager
from scripts.data_processing.auto_data_detector import AutoDataDetector


@dataclass
class ProcessingResult:
    """ì²˜ë¦¬ ê²°ê³¼ ?°ì´???´ë˜??""
    success: bool
    processed_files: List[Path]
    failed_files: List[Path]
    total_records: int
    processing_time: float
    error_messages: List[str]


class IncrementalPreprocessor:
    """ì¦ë¶„ ?„ì²˜ë¦??„ë¡œ?¸ì„œ ?´ë˜??""
    
    def __init__(self, 
                 raw_data_base_path: str = "data/raw/assembly",
                 processed_data_base_path: str = "data/processed/assembly",
                 processing_version: str = "1.0",
                 checkpoint_manager: CheckpointManager = None,
                 db_manager: DatabaseManager = None,
                 enable_term_normalization: bool = True,
                 batch_size: int = 100):
        """
        ì¦ë¶„ ?„ì²˜ë¦??„ë¡œ?¸ì„œ ì´ˆê¸°??
        
        Args:
            raw_data_base_path: ?ë³¸ ?°ì´??ê¸°ë³¸ ê²½ë¡œ
            processed_data_base_path: ?„ì²˜ë¦¬ëœ ?°ì´??ê¸°ë³¸ ê²½ë¡œ
            processing_version: ì²˜ë¦¬ ë²„ì „
            checkpoint_manager: ì²´í¬?¬ì¸??ê´€ë¦¬ì
            db_manager: ?°ì´?°ë² ?´ìŠ¤ ê´€ë¦¬ì
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
        
        # ê¸°ì¡´ LegalDataProcessor ì´ˆê¸°??
        self.processor = LegalDataProcessor(enable_term_normalization)
        
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
        
        self.logger.info("IncrementalPreprocessor initialized")
    
    def process_new_files_only(self, files: List[Path], 
                              data_type: str = "law_only") -> ProcessingResult:
        """
        ?ˆë¡œ???Œì¼ë§?ì²˜ë¦¬
        
        Args:
            files: ì²˜ë¦¬???Œì¼ ëª©ë¡
            data_type: ?°ì´??? í˜•
        
        Returns:
            ProcessingResult: ì²˜ë¦¬ ê²°ê³¼
        """
        self.logger.info(f"Starting incremental processing of {len(files)} files")
        self.stats['start_time'] = datetime.now()
        
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
                                          desc="Processing files",
                                          initial=start_index,
                                          total=len(files))):
            try:
                # ?Œì¼ ?´ì‹œ ê³„ì‚°
                file_hash = self._calculate_file_hash(file_path)
                
                # ?´ë? ì²˜ë¦¬???Œì¼?¸ì? ?•ì¸
                if self.db_manager.is_file_processed(str(file_path)):
                    self.logger.debug(f"File already processed: {file_path}")
                    continue
                
                # ?Œì¼ ì²˜ë¦¬
                self.logger.info(f"Processing file {i+1}/{len(files)}: {file_path.name}")
                
                result = self._process_single_file(file_path, data_type)
                
                if result['success']:
                    processed_files.append(file_path)
                    total_records += result['record_count']
                    
                    # DB??ì²˜ë¦¬ ?„ë£Œ ê¸°ë¡
                    self.db_manager.mark_file_as_processed(
                        file_path=str(file_path),
                        file_hash=file_hash,
                        data_type=data_type,
                        record_count=result['record_count'],
                        processing_version="1.0"
                    )
                    
                    self.logger.info(f"Successfully processed: {file_path.name} "
                                   f"({result['record_count']} records)")
                else:
                    failed_files.append(file_path)
                    error_messages.append(f"{file_path}: {result['error']}")
                    
                    # DB???¤íŒ¨ ê¸°ë¡
                    self.db_manager.mark_file_as_processed(
                        file_path=str(file_path),
                        file_hash=file_hash,
                        data_type=data_type,
                        record_count=0,
                        processing_version="1.0",
                        error_message=result['error']
                    )
                    
                    self.logger.error(f"Failed to process: {file_path.name} - {result['error']}")
                
                # ë°°ì¹˜ ?¨ìœ„ë¡?ì²´í¬?¬ì¸???€??
                if (i + 1) % self.batch_size == 0:
                    self._save_checkpoint({
                        'current_file_index': i + 1,
                        'processed_files': [str(f) for f in processed_files],
                        'failed_files': [str(f) for f in failed_files],
                        'total_records': total_records,
                        'data_type': data_type
                    })
                    self.logger.info(f"Checkpoint saved at file {i + 1}")
                
            except Exception as e:
                error_msg = f"Unexpected error processing {file_path}: {e}"
                error_messages.append(error_msg)
                failed_files.append(file_path)
                self.logger.error(error_msg)
        
        # ìµœì¢… ?µê³„ ê³„ì‚°
        self.stats['end_time'] = datetime.now()
        self.stats['processing_time'] = (
            self.stats['end_time'] - self.stats['start_time']
        ).total_seconds()
        self.stats['total_files'] = len(files)
        self.stats['processed_files'] = len(processed_files)
        self.stats['failed_files'] = len(failed_files)
        self.stats['total_records'] = total_records
        
        # ì²´í¬?¬ì¸???•ë¦¬
        self._cleanup_checkpoint()
        
        # ê²°ê³¼ ?ì„±
        result = ProcessingResult(
            success=len(failed_files) == 0,
            processed_files=processed_files,
            failed_files=failed_files,
            total_records=total_records,
            processing_time=self.stats['processing_time'],
            error_messages=error_messages
        )
        
        self.logger.info(f"Processing completed: {len(processed_files)} success, "
                        f"{len(failed_files)} failed, {total_records} total records")
        
        return result
    
    def process_new_data_only(self, data_type: str = "law_only") -> Dict[str, Any]:
        """
        ?ˆë¡œ ì¶”ê????°ì´?°ë§Œ ê°ì??˜ì—¬ ?„ì²˜ë¦?
        
        Args:
            data_type: ì²˜ë¦¬???¹ì • ?°ì´??? í˜• (?? 'law_only'). 'all'?´ë©´ ëª¨ë“  ? í˜• ì²˜ë¦¬.
            
        Returns:
            Dict[str, Any]: ì²˜ë¦¬ ê²°ê³¼ ?µê³„
        """
        self.logger.info(f"Starting incremental preprocessing for data type: {data_type}")
        start_time = datetime.now()
        
        new_files_by_type = self.auto_detector.detect_new_data_sources(str(self.raw_data_base_path / "law_only"), data_type)
        
        files_to_process = []
        if data_type == "all":
            for files in new_files_by_type.values():
                files_to_process.extend(files)
        elif data_type in new_files_by_type:
            files_to_process = new_files_by_type[data_type]
        else:
            self.logger.warning(f"Data type '{data_type}' not recognized or no new files found.")
            self.stats['end_time'] = datetime.now().isoformat()
            self.stats['duration'] = (datetime.now() - start_time).total_seconds()
            return self.stats

        self.stats['total_scanned_files'] = sum(len(f) for f in new_files_by_type.values())
        self.stats['new_files_to_process'] = len(files_to_process)
        
        if not files_to_process:
            self.logger.info("No new files to preprocess.")
            self.stats['end_time'] = datetime.now().isoformat()
            self.stats['duration'] = (datetime.now() - start_time).total_seconds()
            return self.stats

        self.logger.info(f"Found {len(files_to_process)} new files for preprocessing.")

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
                
                # ?„ì²˜ë¦??˜í–‰ - ?ë³¸ ?°ì´??êµ¬ì¡°??ë§ê²Œ ë³€??
                processed_data = self._process_assembly_law_data(raw_data)
                
                # ì¶œë ¥ ê²½ë¡œ ?¤ì • (?? data/processed/assembly/law_only/20251016/ml_enhanced_...)
                relative_path = file_path.relative_to(self.raw_data_base_path)
                output_subdir = self.processed_data_base_path / relative_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)
                
                output_file_name = f"ml_enhanced_{file_path.stem}.json"
                output_file_path = output_subdir / output_file_name
                
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                
                # ì²˜ë¦¬ ?´ë ¥ ê¸°ë¡
                record_count = len(processed_data.get('laws', [])) if isinstance(processed_data, dict) else 1
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
        self.logger.info(f"Incremental preprocessing completed. Stats: {self.stats}")
        return self.stats
    
    def _process_assembly_law_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        êµ?šŒ ë²•ë¥  ?°ì´??êµ¬ì¡°??ë§ê²Œ ?„ì²˜ë¦?
        
        Args:
            raw_data: ?ë³¸ ?°ì´??(metadata, items êµ¬ì¡°)
            
        Returns:
            Dict[str, Any]: ?„ì²˜ë¦¬ëœ ?°ì´??
        """
        try:
            metadata = raw_data.get('metadata', {})
            items = raw_data.get('items', [])
            
            processed_laws = []
            
            for item in items:
                # ê°?ë²•ë¥  ??ª©??ì²˜ë¦¬
                processed_law = {
                    'id': item.get('cont_id', ''),
                    'law_name': item.get('law_name', ''),
                    'law_id': item.get('cont_id', ''),
                    'mst': None,
                    'effective_date': None,
                    'promulgation_date': None,
                    'ministry': None,
                    'category': metadata.get('data_type', 'law_only'),
                    'status': 'success',
                    'processed_at': datetime.now().isoformat(),
                    'articles': self._extract_articles(item.get('law_content', '')),
                    'full_content': item.get('law_content', ''),
                    'cleaned_content': self._clean_content(item.get('law_content', '')),
                    'chunks': [],
                    'article_chunks': [],
                    'entities': {},
                    'data_quality': {
                        'parsing_quality_score': 0.8,  # ê¸°ë³¸ê°?
                        'content_length': len(item.get('law_content', '')),
                        'article_count': len(self._extract_articles(item.get('law_content', '')))
                    }
                }
                
                processed_laws.append(processed_law)
            
            return {'laws': processed_laws}
            
        except Exception as e:
            self.logger.error(f"Error processing assembly law data: {e}")
            return {'laws': []}
    
    def _extract_articles(self, content: str) -> List[Dict[str, Any]]:
        """
        ë²•ë¥  ?´ìš©?ì„œ ì¡°ë¬¸ ì¶”ì¶œ
        
        Args:
            content: ë²•ë¥  ?´ìš©
            
        Returns:
            List[Dict[str, Any]]: ì¶”ì¶œ??ì¡°ë¬¸ ëª©ë¡
        """
        articles = []
        
        if not content:
            return articles
        
        # ê°„ë‹¨??ì¡°ë¬¸ ì¶”ì¶œ ë¡œì§ (?¤ì œë¡œëŠ” ???•êµ???Œì‹±???„ìš”)
        import re
        
        # ì¡°ë¬¸ ?¨í„´ ì°¾ê¸° (?? "??ì¡?, "??ì¡? ??
        article_pattern = r'??\d+)ì¡?s*([^??*?)(?=??d+ì¡?$)'
        matches = re.findall(article_pattern, content, re.DOTALL)
        
        for i, (article_num, article_content) in enumerate(matches):
            article = {
                'article_number': int(article_num),
                'article_title': f"??article_num}ì¡?,
                'article_content': article_content.strip(),
                'is_supplementary': False,
                'ml_confidence_score': 0.8,
                'parsing_method': 'regex'
            }
            articles.append(article)
        
        return articles
    
    def _clean_content(self, content: str) -> str:
        """
        ?´ìš© ?•ë¦¬
        
        Args:
            content: ?ë³¸ ?´ìš©
            
        Returns:
            str: ?•ë¦¬???´ìš©
        """
        if not content:
            return ""
        
        # ê¸°ë³¸?ì¸ ?•ë¦¬ (HTML ?œê·¸ ?œê±°, ê³µë°± ?•ë¦¬ ??
        import re
        
        # HTML ?œê·¸ ?œê±°
        cleaned = re.sub(r'<[^>]+>', '', content)
        
        # ?°ì†??ê³µë°±???˜ë‚˜ë¡?
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def _process_single_file(self, file_path: Path, data_type: str) -> Dict[str, Any]:
        """
        ?¨ì¼ ?Œì¼ ì²˜ë¦¬
        
        Args:
            file_path: ì²˜ë¦¬???Œì¼ ê²½ë¡œ
            data_type: ?°ì´??? í˜•
        
        Returns:
            Dict[str, Any]: ì²˜ë¦¬ ê²°ê³¼
        """
        try:
            # ?Œì¼ ?½ê¸°
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # ?°ì´??? í˜•ë³?ì²˜ë¦¬
            if data_type == 'law_only':
                return self._process_law_only_file(file_path, raw_data)
            elif data_type == 'precedents':
                return self._process_precedents_file(file_path, raw_data)
            else:
                return self._process_generic_file(file_path, raw_data)
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'record_count': 0
            }
    
    def _process_law_only_file(self, file_path: Path, raw_data: Dict) -> Dict[str, Any]:
        """
        law_only ?Œì¼ ì²˜ë¦¬
        
        Args:
            file_path: ?Œì¼ ê²½ë¡œ
            raw_data: ?ë³¸ ?°ì´??
        
        Returns:
            Dict[str, Any]: ì²˜ë¦¬ ê²°ê³¼
        """
        try:
            # ê¸°ì¡´ LegalDataProcessor ?¬ìš©
            processed_data = self.data_processor.process_law_data(raw_data)
            
            if not processed_data or 'laws' not in processed_data:
                return {
                    'success': False,
                    'error': 'No laws found in processed data',
                    'record_count': 0
                }
            
            # ì¶œë ¥ ?Œì¼ ê²½ë¡œ ?ì„±
            output_path = self._get_output_path(file_path, 'law_only')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # ì²˜ë¦¬???°ì´???€??
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            record_count = len(processed_data['laws'])
            
            return {
                'success': True,
                'error': None,
                'record_count': record_count,
                'output_path': output_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'record_count': 0
            }
    
    def _process_precedents_file(self, file_path: Path, raw_data: Dict) -> Dict[str, Any]:
        """
        precedents ?Œì¼ ì²˜ë¦¬
        
        Args:
            file_path: ?Œì¼ ê²½ë¡œ
            raw_data: ?ë³¸ ?°ì´??
        
        Returns:
            Dict[str, Any]: ì²˜ë¦¬ ê²°ê³¼
        """
        try:
            # ?ë? ?°ì´??ì²˜ë¦¬ (ê¸°ë³¸ êµ¬ì¡° ? ì?)
            processed_data = {
                'metadata': raw_data.get('metadata', {}),
                'precedents': raw_data.get('items', [])
            }
            
            # ì¶œë ¥ ?Œì¼ ê²½ë¡œ ?ì„±
            output_path = self._get_output_path(file_path, 'precedents')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # ì²˜ë¦¬???°ì´???€??
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            record_count = len(processed_data['precedents'])
            
            return {
                'success': True,
                'error': None,
                'record_count': record_count,
                'output_path': output_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'record_count': 0
            }
    
    def _process_generic_file(self, file_path: Path, raw_data: Dict) -> Dict[str, Any]:
        """
        ?¼ë°˜ ?Œì¼ ì²˜ë¦¬
        
        Args:
            file_path: ?Œì¼ ê²½ë¡œ
            raw_data: ?ë³¸ ?°ì´??
        
        Returns:
            Dict[str, Any]: ì²˜ë¦¬ ê²°ê³¼
        """
        try:
            # ê¸°ë³¸ êµ¬ì¡° ? ì?
            processed_data = raw_data
            
            # ì¶œë ¥ ?Œì¼ ê²½ë¡œ ?ì„±
            output_path = self._get_output_path(file_path, 'generic')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # ì²˜ë¦¬???°ì´???€??
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            # ?ˆì½”????ê³„ì‚°
            record_count = 0
            if isinstance(raw_data, dict) and 'items' in raw_data:
                record_count = len(raw_data['items'])
            
            return {
                'success': True,
                'error': None,
                'record_count': record_count,
                'output_path': output_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'record_count': 0
            }
    
    def _get_output_path(self, input_path: Path, data_type: str) -> Path:
        """
        ì¶œë ¥ ?Œì¼ ê²½ë¡œ ?ì„±
        
        Args:
            input_path: ?…ë ¥ ?Œì¼ ê²½ë¡œ
            data_type: ?°ì´??? í˜•
        
        Returns:
            Path: ì¶œë ¥ ?Œì¼ ê²½ë¡œ
        """
        # ? ì§œ ?´ë” ì¶”ì¶œ
        date_folder = input_path.parent.name
        
        # ì¶œë ¥ ê²½ë¡œ êµ¬ì„±
        output_path = self.output_dir / "assembly" / data_type / date_folder
        
        # ?Œì¼ëª?ë³€ê²?(?ë³¸ ?Œì¼ëª?? ì??˜ë˜ ?‘ë‘??ì¶”ê?)
        filename = f"processed_{input_path.name}"
        
        return output_path / filename
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        ?Œì¼ ?´ì‹œ ê³„ì‚°
        
        Args:
            file_path: ?Œì¼ ê²½ë¡œ
        
        Returns:
            str: ?Œì¼ ?´ì‹œê°?
        """
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def _save_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """
        ì²´í¬?¬ì¸???€??
        
        Args:
            checkpoint_data: ì²´í¬?¬ì¸???°ì´??
        
        Returns:
            bool: ?€???±ê³µ ?¬ë?
        """
        if not self.checkpoint_manager:
            return False
        
        try:
            checkpoint_data['timestamp'] = datetime.now().isoformat()
            checkpoint_data['stage'] = 'preprocessing'
            return self.checkpoint_manager.save_checkpoint(checkpoint_data)
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            return False
    
    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        ì²´í¬?¬ì¸??ë¡œë“œ
        
        Returns:
            Optional[Dict]: ì²´í¬?¬ì¸???°ì´???ëŠ” None
        """
        if not self.checkpoint_manager:
            return None
        
        try:
            checkpoint_data = self.checkpoint_manager.load_checkpoint()
            if checkpoint_data and checkpoint_data.get('stage') == 'preprocessing':
                return checkpoint_data
            return None
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return None
    
    def _cleanup_checkpoint(self) -> bool:
        """
        ì²´í¬?¬ì¸???•ë¦¬
        
        Returns:
            bool: ?•ë¦¬ ?±ê³µ ?¬ë?
        """
        if not self.checkpoint_manager:
            return False
        
        try:
            # ì²´í¬?¬ì¸???Œì¼ ?? œ
            checkpoint_file = self.checkpoint_manager.checkpoint_file
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                self.logger.info("Checkpoint cleaned up")
            return True
        except Exception as e:
            self.logger.error(f"Error cleaning up checkpoint: {e}")
            return False
    
    def resume_from_checkpoint(self) -> bool:
        """
        ì²´í¬?¬ì¸?¸ì—???¬ê°œ
        
        Returns:
            bool: ?¬ê°œ ê°€???¬ë?
        """
        checkpoint_data = self._load_checkpoint()
        return checkpoint_data is not None
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        ì²˜ë¦¬ ?µê³„ ì¡°íšŒ
        
        Returns:
            Dict[str, Any]: ì²˜ë¦¬ ?µê³„
        """
        return self.stats.copy()


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    parser = argparse.ArgumentParser(description='ì¦ë¶„ ?„ì²˜ë¦??„ë¡œ?¸ì„œ')
    parser.add_argument('--input-files', nargs='+', type=Path,
                       help='ì²˜ë¦¬???Œì¼ ëª©ë¡')
    parser.add_argument('--data-type', default='law_only',
                       choices=['law_only', 'precedents', 'constitutional'],
                       help='?°ì´??? í˜•')
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
        # ì²´í¬?¬ì¸??ê´€ë¦¬ì ì´ˆê¸°??
        checkpoint_manager = CheckpointManager(args.checkpoint_dir)
        
        # ì¦ë¶„ ?„ì²˜ë¦??„ë¡œ?¸ì„œ ì´ˆê¸°??
        preprocessor = IncrementalPreprocessor(
            checkpoint_manager=checkpoint_manager,
            batch_size=args.batch_size
        )
        
        # ì²´í¬?¬ì¸?¸ì—???¬ê°œ ?•ì¸
        if args.resume and not preprocessor.resume_from_checkpoint():
            logging.warning("No checkpoint found, starting from beginning")
        
        # ?Œì¼ ì²˜ë¦¬ ?¤í–‰
        if args.input_files:
            result = preprocessor.process_new_files_only(args.input_files, args.data_type)
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
            stats = preprocessor.process_new_data_only(args.data_type)
            
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
        logging.error(f"Error in incremental preprocessing: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
