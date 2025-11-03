#!/usr/bin/env python3
"""
Assembly Law Data Preprocessing Script

This script processes raw Assembly law data into clean, structured,
searchable format for database storage and vector embedding.

Usage:
  python preprocess_laws.py --input data/raw/assembly/law/20251010 --output data/processed/assembly/law
  python preprocess_laws.py --validate  # Validation only
  python preprocess_laws.py --help  # Show help
"""

import argparse
import json
import logging
import re
import sys
import gc
import os
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Generator

# Try to import psutil, fallback if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Add parsers module to path
sys.path.append(str(Path(__file__).parent / 'parsers'))

from parsers import (
    LawHTMLParser,
    ArticleParser,
    MetadataExtractor,
    TextNormalizer
    # SearchableTextGenerator ?œê±°??
)
from parsers.improved_article_parser import ImprovedArticleParser
from ml_enhanced_parser import MLEnhancedArticleParser

# Import hybrid parser
try:
    sys.path.append(str(Path(__file__).parent.parent / 'quality'))
    from hybrid_parser import HybridArticleParser
    HYBRID_PARSER_AVAILABLE = True
except ImportError:
    HYBRID_PARSER_AVAILABLE = False
    logger.warning("Hybrid parser not available. Using individual parsers.")

# Import new legal analysis components
try:
    from parsers.version_detector import DataVersionDetector
    from parsers.version_parsers import VersionParserRegistry
    from parsers.comprehensive_legal_analyzer import ComprehensiveLegalAnalyzer
    LEGAL_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Legal analysis components not available: {e}")
    LEGAL_ANALYSIS_AVAILABLE = False

# Setup logging
def setup_logging():
    """ë¡œê¹… ?¤ì •"""
    # logs ?”ë ‰? ë¦¬ ?ì„±
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / 'preprocessing.log', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


def get_system_memory_info() -> Dict[str, float]:
    """Get system memory information"""
    if PSUTIL_AVAILABLE:
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent
        }
    else:
        return {
            'total_gb': 0.0,
            'available_gb': 0.0,
            'used_gb': 0.0,
            'percent': 0.0
        }


def check_memory_safety(threshold_percent: float = 90.0) -> bool:
    """Check if memory usage is safe (below threshold)"""
    if not PSUTIL_AVAILABLE:
        return True  # If psutil not available, assume safe
    
    memory_info = get_system_memory_info()
    if memory_info['percent'] >= threshold_percent:
        logger.critical(f"CRITICAL: System memory usage is {memory_info['percent']:.1f}% (threshold: {threshold_percent}%)")
        logger.critical(f"Total: {memory_info['total_gb']:.1f}GB, Used: {memory_info['used_gb']:.1f}GB, Available: {memory_info['available_gb']:.1f}GB")
        return False
    return True


def simple_memory_check():
    """Simple memory check for sequential processing"""
    if PSUTIL_AVAILABLE:
        memory_info = psutil.virtual_memory()
        if memory_info.percent > 90.0:
            logger.warning(f"High memory usage: {memory_info.percent:.1f}%")
            return False
    return True


def simple_garbage_collection():
    """Simple garbage collection"""
    collected = gc.collect()
    if collected > 0:
        logger.debug(f"Garbage collection: collected {collected} objects")


def is_file_already_processed(input_file: Path, output_dir: Path) -> bool:
    """
    Check if a file has already been processed by looking for output files
    
    Args:
        input_file (Path): Input JSON file path
        output_dir (Path): Output directory path
        
    Returns:
        bool: True if file is already processed, False otherwise
    """
    try:
        # Extract law name from input file (e.g., "law_page_001_181503.json" -> "law_page_001_181503")
        law_name = input_file.stem
        
        # Check if individual law files exist
        individual_files_pattern = output_dir / f"{law_name}_*.json"
        individual_files = list(output_dir.glob(f"{law_name}_*.json"))
        
        if individual_files:
            logger.info(f"File {input_file.name} already processed - found {len(individual_files)} output files")
            return True
            
        return False
        
    except Exception as e:
        logger.warning(f"Error checking if file is processed: {e}")
        return False


class ProcessingManager:
    """
    ?„ì²˜ë¦??íƒœë¥?ê´€ë¦¬í•˜???´ëž˜??
    - ?°ì´?°ë² ?´ìŠ¤ ê¸°ë°˜?¼ë¡œ ì²˜ë¦¬ ?íƒœ ì¶”ì 
    - ì²´í¬?¬ì„ ?¬ìš©???Œì¼ ë³€ê²?ê°ì?
    - ?¬ì‹œ??ê¸°ëŠ¥ ì§€??
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.db_path = output_dir / "processing_status.db"
        self.init_db()
    
    def init_db(self):
        """?°ì´?°ë² ?´ìŠ¤ ì´ˆê¸°??""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_file TEXT NOT NULL,
                output_dir TEXT NOT NULL,
                status TEXT NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_checksum TEXT,
                file_size INTEGER,
                laws_processed INTEGER DEFAULT 0,
                processing_time_seconds REAL,
                error_message TEXT,
                UNIQUE(input_file, output_dir)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_status ON processing_status(status)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_input_file ON processing_status(input_file)
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Processing status database initialized: {self.db_path}")
    
    def calculate_checksum(self, file_path: Path) -> str:
        """?Œì¼ ì²´í¬??ê³„ì‚° (MD5)"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum for {file_path}: {e}")
            return ""
    
    def is_processed(self, input_file: Path) -> bool:
        """
        ?Œì¼???´ë? ì²˜ë¦¬?˜ì—ˆ?”ì? ?•ì¸
        - ?°ì´?°ë² ?´ìŠ¤?ì„œ ?íƒœ ?•ì¸
        - ì²´í¬?¬ìœ¼ë¡??Œì¼ ë³€ê²?ê°ì?
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT status, file_checksum, file_size FROM processing_status 
            WHERE input_file = ? AND output_dir = ?
        ''', (str(input_file), str(self.output_dir)))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return False
        
        status, db_checksum, db_size = result
        
        # ?Œì¼??ì¡´ìž¬?˜ì? ?Šìœ¼ë©??¬ì²˜ë¦?
        if not input_file.exists():
            logger.warning(f"Input file not found: {input_file}")
            return False
        
        # ?Œì¼ ?¬ê¸°ê°€ ë³€ê²½ë˜?ˆìœ¼ë©??¬ì²˜ë¦?
        current_size = input_file.stat().st_size
        if db_size != current_size:
            logger.info(f"File size changed for {input_file.name}: {db_size} -> {current_size}, reprocessing")
            return False
        
        # ì²´í¬?¬ì´ ë³€ê²½ë˜?ˆìœ¼ë©??¬ì²˜ë¦?
        current_checksum = self.calculate_checksum(input_file)
        if db_checksum != current_checksum:
            logger.info(f"File checksum changed for {input_file.name}, reprocessing")
            return False
        
        # ?íƒœê°€ 'completed'??ê²½ìš°ë§?ì²˜ë¦¬ ?„ë£Œë¡?ê°„ì£¼
        if status == 'completed':
            logger.info(f"File {input_file.name} already processed successfully")
            return True
        
        # ?¤íŒ¨ ?íƒœ??ê²½ìš° ?¬ì²˜ë¦?
        if status == 'failed':
            logger.info(f"File {input_file.name} previously failed, will retry")
            return False
        
        return False
    
    def mark_processing(self, input_file: Path):
        """?Œì¼ ì²˜ë¦¬ ?œìž‘ ?œì‹œ"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        checksum = self.calculate_checksum(input_file)
        file_size = input_file.stat().st_size
        
        cursor.execute('''
            INSERT OR REPLACE INTO processing_status 
            (input_file, output_dir, status, file_checksum, file_size, processed_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            str(input_file), str(self.output_dir), 'processing',
            checksum, file_size, datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def mark_completed(self, input_file: Path, laws_processed: int, processing_time: float):
        """?Œì¼ ì²˜ë¦¬ ?„ë£Œ ?œì‹œ"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        checksum = self.calculate_checksum(input_file)
        file_size = input_file.stat().st_size
        
        cursor.execute('''
            INSERT OR REPLACE INTO processing_status 
            (input_file, output_dir, status, file_checksum, file_size, 
             laws_processed, processing_time_seconds, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(input_file), str(self.output_dir), 'completed',
            checksum, file_size, laws_processed, processing_time,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def mark_failed(self, input_file: Path, error_message: str):
        """?Œì¼ ì²˜ë¦¬ ?¤íŒ¨ ?œì‹œ"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        checksum = self.calculate_checksum(input_file)
        file_size = input_file.stat().st_size if input_file.exists() else 0
        
        cursor.execute('''
            INSERT OR REPLACE INTO processing_status 
            (input_file, output_dir, status, file_checksum, file_size, 
             error_message, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(input_file), str(self.output_dir), 'failed',
            checksum, file_size, error_message[:500],  # Limit error message length
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_summary(self) -> Dict[str, Any]:
        """ì²˜ë¦¬ ?íƒœ ?”ì•½ ?•ë³´ ë°˜í™˜"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT status, COUNT(*) as count, 
                   SUM(laws_processed) as total_laws,
                   SUM(processing_time_seconds) as total_time
            FROM processing_status 
            GROUP BY status
        ''')
        
        results = cursor.fetchall()
        
        summary = {
            'by_status': {},
            'total_files': 0,
            'total_laws': 0,
            'total_time': 0.0
        }
        
        for status, count, total_laws, total_time in results:
            summary['by_status'][status] = {
                'count': count,
                'total_laws': total_laws or 0,
                'total_time': total_time or 0.0
            }
            summary['total_files'] += count
            summary['total_laws'] += (total_laws or 0)
            summary['total_time'] += (total_time or 0.0)
        
        conn.close()
        
        return summary
    
    def get_failed_files(self) -> List[Dict[str, Any]]:
        """?¤íŒ¨???Œì¼ ëª©ë¡ ë°˜í™˜"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT input_file, error_message, processed_at
            FROM processing_status 
            WHERE status = 'failed'
            ORDER BY processed_at DESC
        ''')
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
    
    def reset_failed(self):
        """?¤íŒ¨???Œì¼?¤ì„ ?¬ì²˜ë¦¬í•  ???ˆë„ë¡??íƒœ ì´ˆê¸°??""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM processing_status WHERE status = 'failed'
        ''')
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(f"Reset {deleted_count} failed file(s) for reprocessing")
        return deleted_count


def simple_memory_monitor():
    """Optimized memory monitoring for speed"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > 800:  # 800MB threshold (lowered for better memory management)
            logger.info(f"Memory usage: {memory_mb:.1f}MB - triggering cleanup")
            simple_garbage_collection()
            
        if memory_mb > 1500:  # 1.5GB threshold (lowered for better memory management)
            logger.warning(f"High memory usage: {memory_mb:.1f}MB - forcing cleanup")
            simple_garbage_collection()
            gc.collect()  # ê°•ì œ ê°€ë¹„ì? ì»¬ë ‰??
            
        if memory_mb > 2000:  # 2GB threshold (lowered for better memory management)
            logger.error(f"Critical memory usage: {memory_mb:.1f}MB")
            return False
    return True


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage information"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    else:
        # Fallback when psutil is not available
        return {
            'rss_mb': 0.0,
            'vms_mb': 0.0,
            'percent': 0.0
        }


def simple_log_memory(stage: str):
    """Simple memory logging for sequential processing"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage at {stage}: {memory_mb:.1f}MB")


class LawPreprocessor:
    """Main preprocessing class for Assembly law data with legal analysis support and memory optimization"""
    
    def __init__(self, enable_legal_analysis: bool = True, max_memory_mb: int = 2048, max_memory_gb: float = 10.0, processing_manager: Optional['ProcessingManager'] = None):
        """Initialize the preprocessor with all parsers and optional legal analysis"""
        self.html_parser = LawHTMLParser()
        
        # Initialize hybrid parser if available, otherwise fall back to individual parsers
        if HYBRID_PARSER_AVAILABLE:
            ml_model_path = "models/article_classifier.pkl"
            self.article_parser = HybridArticleParser(ml_model_path=ml_model_path)
            logger.info("Using hybrid article parser")
        else:
            # Fallback to individual parsers
            ml_model_path = "models/article_classifier.pkl"
            if Path(ml_model_path).exists():
                self.article_parser = MLEnhancedArticleParser(ml_model_path=ml_model_path)
                logger.info("Using ML-enhanced article parser")
            else:
                self.article_parser = ImprovedArticleParser()
                logger.info("Using rule-based article parser")
        
        self.metadata_extractor = MetadataExtractor()
        self.text_normalizer = TextNormalizer()
        # self.searchable_text_generator = SearchableTextGenerator()  # ?œê±°??
        
        # Memory management
        self.max_memory_mb = max_memory_mb
        self.max_memory_gb = max_memory_gb
        self.initial_memory = get_memory_usage()
        
        # Processing manager for status tracking
        self.processing_manager = processing_manager
        
        # Legal analysis components (optional)
        self.enable_legal_analysis = enable_legal_analysis and LEGAL_ANALYSIS_AVAILABLE
        
        if self.enable_legal_analysis:
            self.version_detector = DataVersionDetector()
            self.version_registry = VersionParserRegistry()
            self.comprehensive_analyzer = ComprehensiveLegalAnalyzer()
            logger.info("Legal analysis components initialized")
        else:
            logger.info("Legal analysis disabled or components not available")
        
        # Processing statistics
        self.stats = {
            'total_files_processed': 0,
            'total_laws_processed': 0,
            'successful_laws': 0,
            'failed_laws': 0,
            'processing_errors': [],
            'memory_usage': [],
            'start_time': None,
            'end_time': None
        }
        
        # Log initial memory usage
        simple_log_memory("preprocessor_init")
    
    def _check_memory_limit(self) -> bool:
        """Simple memory check for sequential processing"""
        return simple_memory_check()
    
    def _force_garbage_collection(self):
        """Simple garbage collection for sequential processing"""
        simple_garbage_collection()
        simple_memory_monitor()
        
        # Log memory usage
        memory = get_memory_usage()
        self.stats['memory_usage'].append({
            'timestamp': datetime.now().isoformat(),
            'rss_mb': memory['rss_mb'],
            'vms_mb': memory['vms_mb'],
            'percent': memory['percent']
        })
        
        logger.debug(f"Memory after cleanup: RSS={memory['rss_mb']:.1f}MB, VMS={memory['vms_mb']:.1f}MB")
    
    def _load_law_file(self, input_file: Path) -> List[Dict[str, Any]]:
        """Load law file and return all laws with memory optimization"""
        try:
            file_size = input_file.stat().st_size
            logger.info(f"Loading file {input_file.name} (size: {file_size / 1024 / 1024:.1f}MB)")
            
            # Check memory before loading
            simple_memory_check()  # Simple memory check for file loading
            
            with open(input_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            laws = raw_data.get('laws', [])
            if not laws and isinstance(raw_data, list):
                laws = raw_data
            
            # Clean up raw_data immediately
            del raw_data
            simple_garbage_collection()
            
            logger.info(f"Loaded {len(laws)} laws from {input_file.name}")
            return laws
                    
        except Exception as e:
            logger.error(f"Error loading file {input_file}: {e}")
            # Force cleanup on error
            simple_garbage_collection()
            raise
    
    def preprocess_law_file(self, input_file: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Preprocess a single law file
        
        Args:
            input_file (Path): Input JSON file path
            output_dir (Path): Output directory path
            
        Returns:
            Dict[str, Any]: Processing results
        """
        start_time = datetime.now()
        
        try:
            # Mark file as processing in database
            if self.processing_manager:
                self.processing_manager.mark_processing(input_file)
            
            # Process laws with batch optimization
            file_stats = {
                'file_name': input_file.name,
                'total_laws': 0,
                'processed_laws': 0,
                'errors': []
            }
            
            logger.info(f"Processing file: {input_file}")
            # Memory logging only for large files
            if file_stats['total_laws'] > 5:
                simple_log_memory(f"file_start_{input_file.name}")
            
            # Check memory limit before processing
            if not self._check_memory_limit():
                self._force_garbage_collection()
            
            # Load all laws from file
            laws = self._load_law_file(input_file)
            file_stats['total_laws'] = len(laws)
            
            # Batch processing for speed optimization
            batch_size = 5  # Process 5 laws at a time
            batch_files = []  # Store file paths for batch operations
            
            # Process all laws in the file with aggressive memory management
            for i, law_data in enumerate(laws):
                try:
                    # Check memory safety every 5 laws (reduced frequency for speed)
                    if (i + 1) % 5 == 0:
                        simple_memory_check()
                    
                    processed_law = self._process_single_law(law_data)
                    if processed_law:
                        # Save individual law file immediately to free memory
                        saved_file = self._save_individual_law(processed_law, output_dir)
                        batch_files.append(saved_file)
                        
                        # Remove from memory immediately after saving
                        del processed_law
                        
                        file_stats['processed_laws'] += 1
                        self.stats['successful_laws'] += 1
                        
                        # Batch cleanup for speed optimization
                        if len(batch_files) >= batch_size:
                            # Clear batch files list
                            batch_files.clear()
                            
                            # Force garbage collection every batch
                            self._force_garbage_collection()
                        
                        # Clear law_data reference every 50 laws (reduced frequency for speed)
                        if (i + 1) % 50 == 0:
                            del law_data
                            simple_garbage_collection()
                        
                    else:
                        self.stats['failed_laws'] += 1
                        file_stats['errors'].append(f"Failed to process law: {law_data.get('law_name', 'Unknown')}")
                        
                except Exception as e:
                    error_msg = f"Error processing law {law_data.get('law_name', 'Unknown')}: {str(e)}"
                    logger.error(error_msg)
                    self.stats['failed_laws'] += 1
                    self.stats['processing_errors'].append(error_msg)
                    file_stats['errors'].append(error_msg)
                    # Force cleanup on error
                    simple_garbage_collection()
                
                # Progress logging every 50 laws (reduced frequency for speed)
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(laws)} laws from {input_file.name}")
                
                # Clean up law_data reference periodically
                if (i + 1) % 100 == 0:
                    del law_data  # This will be reassigned in next iteration
            
            # Create metadata file with file statistics (no processed_laws list needed)
            if file_stats['processed_laws'] > 0:
                batch_name = input_file.stem
                self._create_metadata_file_from_stats(file_stats, output_dir, batch_name)
                logger.info(f"Processed {file_stats['processed_laws']} laws from {input_file.name}")
            
            # Final cleanup
            self._force_garbage_collection()
            
            self.stats['total_files_processed'] += 1
            self.stats['total_laws_processed'] += file_stats['total_laws']
            
            # Mark file as completed in database
            if self.processing_manager:
                processing_time = (datetime.now() - start_time).total_seconds()
                self.processing_manager.mark_completed(input_file, file_stats['processed_laws'], processing_time)
            
            # Memory logging only for large files
            if file_stats['total_laws'] > 5:
                simple_log_memory(f"file_end_{input_file.name}")
            return file_stats
            
        except Exception as e:
            error_msg = f"Error processing file {input_file}: {str(e)}"
            logger.error(error_msg)
            self.stats['processing_errors'].append(error_msg)
            
            # Mark file as failed in database
            if self.processing_manager:
                self.processing_manager.mark_failed(input_file, error_msg)
            
            return {'file_name': input_file.name, 'error': error_msg}
    
    def _process_single_law(self, law_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single law through all parsers
        
        Args:
            law_data (Dict[str, Any]): Raw law data
            
        Returns:
            Optional[Dict[str, Any]]: Processed law data
        """
        try:
            # Extract basic information
            law_id = f"assembly_law_{law_data.get('row_number', 'unknown')}"
            
            # 1. Version detection and parsing (if legal analysis enabled)
            if self.enable_legal_analysis:
                version = self.version_detector.detect_version(law_data)
                version_parser = self.version_registry.get_parser(version)
                version_parsed = version_parser.parse(law_data)
                law_data.update(version_parsed)
                parsing_version = version
                version_confidence = self.version_detector.get_confidence(law_data, version)
            else:
                parsing_version = 'legacy'
                version_confidence = 1.0
            
            # Clean law content by removing JavaScript and HTML artifacts
            law_content = self._clean_law_content(law_data.get('law_content', ''))
            html_content = law_data.get('content_html', '')
            
            # Use hybrid parser if available, otherwise use individual parser
            if HYBRID_PARSER_AVAILABLE and isinstance(self.article_parser, HybridArticleParser):
                # Use hybrid parser with validation
                parsed_result = self.article_parser.parse_law_content(law_content, law_data.get('law_name', ''))
                
                # Extract final result from hybrid parser
                if parsed_result.get('final_result'):
                    final_result = parsed_result['final_result']
                    articles = final_result.get('all_articles', [])
                    quality_score = final_result.get('quality_score', 0.0)
                    auto_corrected = final_result.get('auto_corrected', False)
                    manual_review_required = parsed_result.get('manual_review_required', False)
                else:
                    articles = []
                    quality_score = 0.0
                    auto_corrected = False
                    manual_review_required = True
            else:
                # Fallback to individual parser
                parsed_result = self.article_parser.parse_law(law_content)
                articles = parsed_result.get('all_articles', []) if isinstance(parsed_result, dict) else []
                quality_score = 0.0
                auto_corrected = False
                manual_review_required = False
            
            # Extract metadata
            metadata = self.metadata_extractor.extract(law_data)
            
            # Normalize text
            clean_text = self.text_normalizer.normalize(law_content)
            
            # Comprehensive legal analysis (if enabled)
            legal_analysis = {}
            if self.enable_legal_analysis:
                legal_analysis = self.comprehensive_analyzer.analyze_law_comprehensively(law_data)
            
            # Combine all processed data (ìµœì ?”ëœ êµ¬ì¡°ë¡??˜ì •)
            processed_law = {
                # Basic identification (?„ìˆ˜ ?„ë“œë§?
                'law_id': law_id,
                'law_name': law_data.get('law_name', ''),
                'law_type': law_data.get('law_type', ''),
                'category': law_data.get('category', ''),
                'promulgation_number': law_data.get('promulgation_number', ''),
                'promulgation_date': law_data.get('promulgation_date', ''),
                'enforcement_date': law_data.get('enforcement_date', ''),
                'amendment_type': law_data.get('amendment_type', ''),
                'ministry': law_data.get('ministry', ''),
                
                # Parsed content (?•ì¶•??êµ¬ì¡°)
                'articles': articles,
                
                # Quality information
                'quality_score': quality_score,
                'auto_corrected': auto_corrected,
                'manual_review_required': manual_review_required,
                'parsing_method': 'hybrid' if HYBRID_PARSER_AVAILABLE else 'individual',
                'parsing_version': parsing_version,
                'version_confidence': version_confidence,
                
                # Processing metadata
                'processing_timestamp': datetime.now().isoformat(),
                'processing_version': '2.0_hybrid'
            }
            
            return processed_law
            
        except Exception as e:
            logger.error(f"Error processing single law: {e}")
            return None
    
    # _generate_compressed_search_text ë©”ì„œ???œê±°??- ???´ìƒ ?¬ìš©?˜ì? ?ŠìŒ
    # _compress_legal_text ë©”ì„œ???œê±°??- ???´ìƒ ?¬ìš©?˜ì? ?ŠìŒ
    
    def _clean_law_content(self, content: str) -> str:
        """
        Clean law content by removing JavaScript, HTML artifacts, and other unwanted elements
        
        Args:
            content (str): Original law content
            
        Returns:
            str: Cleaned law content
        """
        if not content:
            return ""
        
        import re
        
        # Remove JavaScript code blocks
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove JavaScript function calls and event handlers
        content = re.sub(r'javascript:[^"\']*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'onclick="[^"]*"', '', content, flags=re.IGNORECASE)
        content = re.sub(r'onload="[^"]*"', '', content, flags=re.IGNORECASE)
        content = re.sub(r'onchange="[^"]*"', '', content, flags=re.IGNORECASE)
        
        # Remove HTML tags but preserve content
        content = re.sub(r'<[^>]+>', '', content)
        
        # Remove specific HTML attributes
        content = re.sub(r'href="[^"]*"', '', content)
        content = re.sub(r'class="[^"]*"', '', content)
        content = re.sub(r'id="[^"]*"', '', content)
        content = re.sub(r'style="[^"]*"', '', content)
        content = re.sub(r'title="[^"]*"', '', content)
        
        # Remove specific patterns that are artifacts
        content = re.sub(r'joHistCall\([^)]*\)', '', content)
        content = re.sub(r'contId:[^,}]*', '', content)
        content = re.sub(r'contSid:[^,}]*', '', content)
        content = re.sub(r'tocAno:[^,}]*', '', content)
        content = re.sub(r'promDt:[^,}]*', '', content)
        content = re.sub(r'basicDt:[^,}]*', '', content)
        content = re.sub(r'viewGb:[^,}]*', '', content)
        content = re.sub(r'hanTranceYn:[^,}]*', '', content)
        content = re.sub(r'bonContId:[^,}]*', '', content)
        content = re.sub(r'bonContSid:[^,}]*', '', content)
        content = re.sub(r'contNm:[^,}]*', '', content)
        content = re.sub(r'contQuery:[^,}]*', '', content)
        
        # Remove HTML entities
        content = re.sub(r'&[a-zA-Z0-9#]+;', '', content)
        
        # Remove control characters (both actual and escaped)
        # Actual control characters
        content = content.replace('\n', ' ')  # Replace actual newline with space
        content = content.replace('\t', ' ')  # Replace actual tab with space
        content = content.replace('\r', ' ')  # Replace actual carriage return with space
        content = content.replace('\f', ' ')  # Replace form feed with space
        content = content.replace('\v', ' ')  # Replace vertical tab with space
        
        # Escaped control characters
        content = content.replace('\\n', ' ')  # Replace escaped newline with space
        content = content.replace('\\t', ' ')  # Replace escaped tab with space
        content = content.replace('\\r', ' ')  # Replace escaped carriage return with space
        content = content.replace('\\"', '"')  # Replace escaped quotes
        content = content.replace("\\'", "'")  # Replace escaped single quotes
        content = content.replace('\\\\', '\\')  # Replace escaped backslashes
        
        # Remove other control characters (ASCII 0-31 except space)
        import string
        control_chars = ''.join(chr(i) for i in range(32) if chr(i) not in string.whitespace)
        for char in control_chars:
            content = content.replace(char, ' ')
        
        # Remove excessive whitespace and clean up
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        # Remove any remaining artifacts
        content = re.sub(r'[{}]', '', content)
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return content
    
    # _calculate_enhanced_data_quality ë©”ì„œ???œê±°??- ?¬ìš©?˜ì? ?ŠìŒ
    # _calculate_metadata_completeness ë©”ì„œ???œê±°??- ?¬ìš©?˜ì? ?ŠìŒ
    
    def _get_output_file_path(self, input_file: Path, output_dir: Path) -> Path:
        """
        Get output file path for processed data
        
        Args:
            input_file (Path): Input file path
            output_dir (Path): Output directory
            
        Returns:
            Path: Output file path
        """
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        input_name = input_file.stem
        if input_name.startswith('law_page_'):
            output_name = input_name.replace('law_page_', 'processed_law_')
        else:
            output_name = f"processed_{input_name}"
        
        return output_dir / f"{output_name}.json"
    
    def _save_processed_data(self, processed_laws: List[Dict], output_file: Path):
        """
        Save processed data to file
        
        Args:
            processed_laws (List[Dict]): Processed law data
            output_file (Path): Output file path
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_laws, f, ensure_ascii=False, separators=(',', ':'))
    
    def _save_individual_law(self, processed_law: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """
        Save individual law to separate file
        
        Args:
            processed_law (Dict[str, Any]): Processed law data
            output_dir (Path): Output directory path
            
        Returns:
            Optional[Path]: Path to saved file or None if failed
        """
        try:
            # Create safe filename from law name
            law_name = processed_law.get('law_name', 'unknown_law')
            law_id = processed_law.get('law_id', 'unknown_id')
            
            # Clean filename
            safe_name = re.sub(r'[^\w\-_\.]', '_', law_name)
            safe_name = safe_name[:50]  # Limit length
            
            # Create filename
            filename = f"{safe_name}_{law_id}.json"
            output_file = output_dir / filename
            
            # Save individual law (?•ì¶•???•ì‹)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_law, f, ensure_ascii=False, separators=(',', ':'))
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving individual law {law_name}: {e}")
            return None
    
    def _create_metadata_file_from_stats(self, file_stats: Dict[str, Any], output_dir: Path, batch_name: str) -> Optional[Path]:
        """
        Create metadata file from file statistics (memory optimized)
        
        Args:
            file_stats (Dict[str, Any]): File processing statistics
            output_dir (Path): Output directory path
            batch_name (str): Batch name for metadata file
            
        Returns:
            Optional[Path]: Path to metadata file or None if failed
        """
        try:
            metadata_file = output_dir / f"metadata_{batch_name}.json"
            
            # Create lightweight metadata
            metadata = {
                'batch_name': batch_name,
                'processing_date': datetime.now().isoformat(),
                'file_stats': file_stats,
                'total_laws': file_stats['total_laws'],
                'processed_laws': file_stats['processed_laws'],
                'errors': file_stats['errors']
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Created metadata file: {metadata_file}")
            return metadata_file
            
        except Exception as e:
            logger.error(f"Error creating metadata file: {e}")
            return None

    def _create_metadata_file(self, processed_laws: List[Dict[str, Any]], 
                            individual_files: List[Path], output_dir: Path, 
                            batch_name: str) -> Optional[Path]:
        """
        Create metadata file for the batch
        
        Args:
            processed_laws (List[Dict[str, Any]]): List of processed laws
            individual_files (List[Path]): List of individual file paths
            output_dir (Path): Output directory path
            batch_name (str): Batch name
            
        Returns:
            Optional[Path]: Path to metadata file or None if failed
        """
        try:
            metadata = {
                'batch_name': batch_name,
                'processing_date': datetime.now().isoformat(),
                'total_laws': len(processed_laws),
                'individual_files': [str(f) for f in individual_files],
                'law_summary': []
            }
            
            # Create law summary
            for law in processed_laws:
                law_summary = {
                    'law_id': law.get('law_id'),
                    'law_name': law.get('law_name'),
                    'law_type': law.get('law_type'),
                    'hierarchy_type': law.get('hierarchy_type'),
                    'primary_field': law.get('primary_field'),
                    'comprehensive_score': law.get('comprehensive_score'),
                    'article_count': len(law.get('articles', [])),
                    'word_count': law.get('word_count', 0),
                    'char_count': law.get('char_count', 0),
                    'data_quality': law.get('data_quality', {}).get('quality_score', 0.0)
                }
                metadata['law_summary'].append(law_summary)
            
            # Save metadata file
            metadata_file = output_dir / f"metadata_{batch_name}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Created metadata file: {metadata_file}")
            return metadata_file
            
        except Exception as e:
            logger.error(f"Error creating metadata file: {e}")
            return None
    
    
    def preprocess_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Preprocess all law files in a directory with date-based folder structure (sequential processing)
        
        Args:
            input_dir (Path): Input directory path
            output_dir (Path): Output directory path
            
        Returns:
            Dict[str, Any]: Processing results
        """
        self.stats['start_time'] = datetime.now()
        
        # Create date-based output directory
        current_date = datetime.now().strftime("%Y%m%d")
        date_output_dir = output_dir / current_date
        date_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ProcessingManager for this batch
        processing_manager = ProcessingManager(date_output_dir)
        self.processing_manager = processing_manager
        
        # Find only law JSON files (files starting with 'law')
        json_files = list(input_dir.glob('law*.json'))
        total_files = len(json_files)
        
        # Filter out already processed files using ProcessingManager
        unprocessed_files = []
        skipped_files = []
        
        logger.info(f"Checking processing status for {total_files} law JSON files...")
        for json_file in json_files:
            if processing_manager.is_processed(json_file):
                skipped_files.append(json_file.name)
            else:
                unprocessed_files.append(json_file)
        
        logger.info(f"Found {total_files} law JSON files")
        logger.info(f"Skipping {len(skipped_files)} already processed files")
        logger.info(f"Processing {len(unprocessed_files)} unprocessed files")
        
        if skipped_files:
            logger.info(f"Skipped files (sample): {', '.join(skipped_files[:5])}{'...' if len(skipped_files) > 5 else ''}")
        
        logger.info(f"Starting sequential preprocessing of directory: {input_dir}")
        logger.info(f"Date-based output directory: {date_output_dir}")
        
        file_results = []
        processed_count = 0
        success_count = 0
        error_count = 0
        
        for i, json_file in enumerate(unprocessed_files, 1):
            try:
                logger.info(f"[{i}/{len(unprocessed_files)}] Processing file: {json_file.name}")
                start_time = datetime.now()
                
                file_result = self.preprocess_law_file(json_file, date_output_dir)
                file_results.append(file_result)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                processed_count += 1
                
                if 'error' in file_result:
                    error_count += 1
                    logger.error(f"[{i}/{total_files}] Error processing {json_file.name}: {file_result['error']}")
                else:
                    success_count += 1
                    laws_count = file_result.get('processed_laws', 0)
                    logger.info(f"[{i}/{total_files}] Successfully processed {json_file.name} - {laws_count} laws in {processing_time:.2f}s")
                
                # Progress update every 10 files
                if i % 10 == 0:
                    logger.info(f"Progress: {i}/{total_files} files processed ({success_count} success, {error_count} errors)")
                
            except Exception as e:
                error_msg = f"Error processing file {json_file}: {str(e)}"
                logger.error(f"[{i}/{total_files}] {error_msg}")
                file_results.append({'file_name': json_file.name, 'error': error_msg})
                error_count += 1
        
        self.stats['end_time'] = datetime.now()
        total_time = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        logger.info(f"Sequential preprocessing completed!")
        logger.info(f"Total files: {total_files}")
        logger.info(f"Successfully processed: {success_count}")
        logger.info(f"Errors: {error_count}")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Average time per file: {total_time/total_files:.2f} seconds")
        
        # Generate summary report
        summary = self._generate_summary_report(file_results)
        summary['output_directory'] = str(date_output_dir)
        summary['processing_date'] = current_date
        summary['total_files'] = total_files
        summary['success_count'] = success_count
        summary['error_count'] = error_count
        summary['total_processing_time'] = total_time
        summary['processing_mode'] = 'sequential'
        
        return summary
    
    def _generate_summary_report(self, file_results: List[Dict]) -> Dict[str, Any]:
        """
        Generate summary report
        
        Args:
            file_results (List[Dict]): File processing results
            
        Returns:
            Dict[str, Any]: Summary report
        """
        total_files = len(file_results)
        successful_files = sum(1 for result in file_results if 'error' not in result)
        failed_files = total_files - successful_files
        
        processing_time = None
        if self.stats['start_time'] and self.stats['end_time']:
            processing_time = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        return {
            'processing_summary': {
                'total_files': total_files,
                'successful_files': successful_files,
                'failed_files': failed_files,
                'total_laws_processed': self.stats['total_laws_processed'],
                'successful_laws': self.stats['successful_laws'],
                'failed_laws': self.stats['failed_laws'],
                'processing_time_seconds': processing_time,
                'processing_date': datetime.now().isoformat()
            },
            'file_results': file_results,
            'errors': self.stats['processing_errors']
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Preprocess Assembly law data')
    parser.add_argument('--input', type=str, help='Input directory path')
    parser.add_argument('--output', type=str, help='Output directory path')
    parser.add_argument('--validate', action='store_true', help='Run validation only')
    parser.add_argument('--enable-legal-analysis', action='store_true', default=True,
                       help='Enable comprehensive legal analysis (default: True)')
    parser.add_argument('--disable-legal-analysis', action='store_true',
                       help='Disable comprehensive legal analysis')
    parser.add_argument('--max-memory', type=int, default=1024, help='Maximum memory usage in MB (default: 1024)')
    parser.add_argument('--max-memory-gb', type=float, default=10.0, help='Maximum system memory usage in GB (default: 10.0)')
    parser.add_argument('--memory-threshold', type=float, default=70.0, help='System memory threshold percentage for forced exit (default: 70.0)')
    parser.add_argument('--reset-failed', action='store_true', help='Reset failed files for reprocessing')
    parser.add_argument('--show-summary', action='store_true', help='Show processing summary and exit')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Check initial system memory
    if PSUTIL_AVAILABLE:
        memory_info = get_system_memory_info()
        logger.info(f"System memory: {memory_info['total_gb']:.1f}GB total, {memory_info['available_gb']:.1f}GB available ({memory_info['percent']:.1f}% used)")
        
        # Check if we have enough memory to proceed
        if memory_info['percent'] > args.memory_threshold:
            logger.critical(f"System memory usage ({memory_info['percent']:.1f}%) exceeds threshold ({args.memory_threshold}%)")
            logger.critical("Not enough memory to proceed safely")
            sys.exit(1)
    
    if args.validate:
        logger.info("Running validation mode")
        # TODO: Implement validation logic
        return
    
    # Handle --show-summary or --reset-failed options
    if args.show_summary or args.reset_failed:
        if not args.output:
            logger.error("--output directory is required for --show-summary or --reset-failed")
            return
        
        output_dir = Path(args.output)
        current_date = datetime.now().strftime("%Y%m%d")
        date_output_dir = output_dir / current_date
        
        if not date_output_dir.exists():
            logger.error(f"Output directory does not exist: {date_output_dir}")
            return
        
        processing_manager = ProcessingManager(date_output_dir)
        
        if args.show_summary:
            logger.info("=" * 80)
            logger.info("Processing Status Summary")
            logger.info("=" * 80)
            summary = processing_manager.get_summary()
            logger.info(f"Total files tracked: {summary['total_files']}")
            logger.info(f"Total laws processed: {summary['total_laws']}")
            logger.info(f"Total processing time: {summary['total_time']:.2f} seconds")
            logger.info("")
            for status, info in summary['by_status'].items():
                logger.info(f"{status.capitalize()}:")
                logger.info(f"  Files: {info['count']}")
                logger.info(f"  Laws: {info['total_laws']}")
                logger.info(f"  Time: {info['total_time']:.2f} seconds")
            logger.info("=" * 80)
            
            # Show failed files if any
            failed_files = processing_manager.get_failed_files()
            if failed_files:
                logger.info("")
                logger.info("Failed Files:")
                for failed in failed_files[:10]:  # Show up to 10 failed files
                    logger.info(f"  - {failed['input_file']}")
                    logger.info(f"    Error: {failed['error_message'][:100]}...")
                if len(failed_files) > 10:
                    logger.info(f"  ... and {len(failed_files) - 10} more")
            return
        
        if args.reset_failed:
            count = processing_manager.reset_failed()
            logger.info(f"Reset {count} failed file(s) for reprocessing")
            return
    
    if not args.input or not args.output:
        parser.print_help()
        return
    
    # Convert to Path objects
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Determine if legal analysis should be enabled
    enable_legal_analysis = args.enable_legal_analysis and not args.disable_legal_analysis
    
    logger.info(f"Starting sequential preprocessing with legal analysis: {enable_legal_analysis}")
    
    # Create preprocessor with memory optimization
    preprocessor = LawPreprocessor(enable_legal_analysis=enable_legal_analysis, max_memory_mb=args.max_memory, max_memory_gb=args.max_memory_gb)
    
    try:
        summary = preprocessor.preprocess_directory(input_dir, output_dir)
        
        # Save summary report
        summary_file = output_dir / 'preprocessing_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Preprocessing completed. Summary saved to {summary_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Processing mode: SEQUENTIAL")
        print(f"Total files processed: {summary.get('total_files', summary['processing_summary']['total_files'])}")
        print(f"Successful files: {summary.get('success_count', summary['processing_summary']['successful_files'])}")
        print(f"Failed files: {summary.get('error_count', summary['processing_summary']['failed_files'])}")
        print(f"Total laws processed: {summary['processing_summary']['total_laws_processed']}")
        print(f"Successful laws: {summary['processing_summary']['successful_laws']}")
        print(f"Failed laws: {summary['processing_summary']['failed_laws']}")
        
        processing_time = summary.get('total_processing_time', summary['processing_summary'].get('processing_time_seconds'))
        if processing_time:
            print(f"Processing time: {processing_time:.2f} seconds")
            if summary.get('total_files', 0) > 0:
                avg_time = processing_time / summary.get('total_files', 1)
                print(f"Average time per file: {avg_time:.2f} seconds")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
