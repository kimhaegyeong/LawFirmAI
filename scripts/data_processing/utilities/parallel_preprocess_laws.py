#!/usr/bin/env python3
"""
Parallel Assembly Law Data Preprocessing Script

This script processes raw Assembly law data using parallel processing
for significantly improved performance.

Usage:
  python parallel_preprocess_laws.py --input data/raw/assembly/law/20251010 --output data/processed/assembly/law
  python parallel_preprocess_laws.py --workers 8 --disable-legal-analysis
  python parallel_preprocess_laws.py --help
"""

import argparse
import json
import logging
import sys
import os
import gc
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

# Add parsers module to path
sys.path.append(str(Path(__file__).parent / 'parsers'))

from parsers import (
    LawHTMLParser,
    ArticleParser,
    MetadataExtractor,
    TextNormalizer
)
from parsers.improved_article_parser import ImprovedArticleParser
from ml_enhanced_parser import MLEnhancedArticleParser

# Import legal analysis components
try:
    from parsers.version_detector import DataVersionDetector
    from parsers.version_parsers import VersionParserRegistry
    from parsers.comprehensive_legal_analyzer import ComprehensiveLegalAnalyzer
    LEGAL_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Legal analysis components not available: {e}")
    LEGAL_ANALYSIS_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/parallel_preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)


def get_optimal_worker_count() -> int:
    """Get optimal number of workers based on system resources"""
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Conservative approach: use 75% of CPU cores, but not more than 8
    optimal_workers = min(int(cpu_count * 0.75), 8)
    
    # If memory is limited (< 8GB), reduce workers
    if memory_gb < 8:
        optimal_workers = min(optimal_workers, 4)
    
    logger.info(f"System: {cpu_count} CPU cores, {memory_gb:.1f}GB RAM")
    logger.info(f"Optimal workers: {optimal_workers}")
    
    return optimal_workers


class ParallelLawPreprocessor:
    """Parallel preprocessing class for Assembly law data"""
    
    def __init__(self, enable_legal_analysis: bool = False, max_workers: int = None):
        """Initialize the parallel preprocessor"""
        self.enable_legal_analysis = enable_legal_analysis and LEGAL_ANALYSIS_AVAILABLE
        self.max_workers = max_workers or get_optimal_worker_count()
        
        # Initialize parsers (will be re-initialized in each worker process)
        self.parsers_initialized = False
        
        logger.info(f"Parallel preprocessor initialized with {self.max_workers} workers")
        logger.info(f"Legal analysis: {'enabled' if self.enable_legal_analysis else 'disabled'}")
    
    def _initialize_parsers(self):
        """Initialize parsers for current process"""
        if self.parsers_initialized:
            return
            
        self.html_parser = LawHTMLParser()
        
        # ML 강화 파서 사용 (모델이 있으면)
        ml_model_path = "models/article_classifier.pkl"
        if Path(ml_model_path).exists():
            self.article_parser = MLEnhancedArticleParser(ml_model_path=ml_model_path)
            logger.debug("Using ML-enhanced article parser")
        else:
            self.article_parser = ImprovedArticleParser()
            logger.debug("Using rule-based article parser")
            
        self.metadata_extractor = MetadataExtractor()
        self.text_normalizer = TextNormalizer()
        
        # Legal analysis components (optional)
        if self.enable_legal_analysis:
            self.version_detector = DataVersionDetector()
            self.version_registry = VersionParserRegistry()
            self.comprehensive_analyzer = ComprehensiveLegalAnalyzer()
            logger.debug("Legal analysis components initialized")
        
        self.parsers_initialized = True
    
    def _process_single_law(self, law_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single law through all parsers"""
        try:
            # Initialize parsers if not done
            self._initialize_parsers()
            
            # Extract metadata
            metadata = self.metadata_extractor.extract_metadata(law_data)
            
            # Parse HTML content
            parsed_content = self.html_parser.parse_html(law_data.get('law_content', ''))
            
            # Parse articles using ML-enhanced parser
            articles = self.article_parser.parse_articles(parsed_content)
            
            # Normalize text
            normalized_content = self.text_normalizer.normalize_text(parsed_content)
            
            # Legal analysis (if enabled)
            legal_analysis = None
            if self.enable_legal_analysis:
                legal_analysis = self.comprehensive_analyzer.analyze_law(
                    law_data, parsed_content, articles
                )
            
            # Create processed law data
            processed_law = {
                'law_id': law_data.get('law_id'),
                'law_name': law_data.get('law_name'),
                'law_number': law_data.get('law_number'),
                'metadata': metadata,
                'parsed_content': parsed_content,
                'normalized_content': normalized_content,
                'articles': articles,
                'legal_analysis': legal_analysis,
                'processing_timestamp': datetime.now().isoformat(),
                'processing_version': 'parallel_v1.0'
            }
            
            return processed_law
            
        except Exception as e:
            logger.error(f"Error processing law {law_data.get('law_name', 'Unknown')}: {str(e)}")
            return None
    
    def _process_law_file_worker(self, args: tuple) -> Dict[str, Any]:
        """Worker function for processing a single law file"""
        input_file, output_dir, enable_legal_analysis = args
        
        # Create a new preprocessor instance for this worker
        worker_preprocessor = ParallelLawPreprocessor(enable_legal_analysis=enable_legal_analysis)
        
        start_time = datetime.now()
        file_stats = {
            'file_name': input_file.name,
            'total_laws': 0,
            'processed_laws': 0,
            'errors': [],
            'processing_time': 0
        }
        
        try:
            logger.info(f"[Worker] Processing file: {input_file}")
            
            # Load laws from file
            with open(input_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            laws = raw_data.get('laws', [])
            file_stats['total_laws'] = len(laws)
            
            if not laws:
                logger.warning(f"No laws found in {input_file}")
                return file_stats
            
            # Process all laws in the file
            processed_laws = []
            for i, law_data in enumerate(laws):
                try:
                    processed_law = worker_preprocessor._process_single_law(law_data)
                    if processed_law:
                        processed_laws.append(processed_law)
                        file_stats['processed_laws'] += 1
                    
                    # Log progress every 5 laws
                    if (i + 1) % 5 == 0:
                        logger.info(f"[Worker] Processed {i + 1}/{len(laws)} laws from {input_file.name}")
                        
                except Exception as e:
                    error_msg = f"Error processing law {i+1}: {str(e)}"
                    file_stats['errors'].append(error_msg)
                    logger.error(f"[Worker] {error_msg}")
            
            # Save processed laws
            if processed_laws:
                output_file = output_dir / f"processed_{input_file.stem}.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'source_file': input_file.name,
                        'processing_timestamp': datetime.now().isoformat(),
                        'total_laws': file_stats['total_laws'],
                        'processed_laws': file_stats['processed_laws'],
                        'laws': processed_laws
                    }, f, ensure_ascii=False, indent=2)
                
                logger.info(f"[Worker] Saved {len(processed_laws)} processed laws to {output_file}")
            
            # Calculate processing time
            end_time = datetime.now()
            file_stats['processing_time'] = (end_time - start_time).total_seconds()
            
            logger.info(f"[Worker] Completed {input_file.name}: {file_stats['processed_laws']}/{file_stats['total_laws']} laws in {file_stats['processing_time']:.2f}s")
            
            return file_stats
            
        except Exception as e:
            error_msg = f"Failed to process file {input_file}: {str(e)}"
            file_stats['errors'].append(error_msg)
            logger.error(f"[Worker] {error_msg}")
            return file_stats
    
    def preprocess_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Preprocess all law files in a directory using parallel processing"""
        start_time = datetime.now()
        
        # Find all JSON files
        json_files = list(input_dir.glob('*.json'))
        if not json_files:
            logger.error(f"No JSON files found in {input_dir}")
            return {'success': False, 'error': 'No JSON files found'}
        
        logger.info(f"Found {len(json_files)} files to process")
        logger.info(f"Using {self.max_workers} parallel workers")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare arguments for worker processes
        worker_args = [
            (json_file, output_dir, self.enable_legal_analysis)
            for json_file in json_files
        ]
        
        # Process files in parallel
        results = []
        successful_files = 0
        failed_files = 0
        total_laws = 0
        processed_laws = 0
        
        logger.info("Starting parallel processing...")
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_law_file_worker, args): args[0]
                for args in worker_args
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['errors']:
                        failed_files += 1
                        logger.error(f"❌ {file_path.name}: {len(result['errors'])} errors")
                    else:
                        successful_files += 1
                        logger.info(f"✅ {file_path.name}: {result['processed_laws']}/{result['total_laws']} laws")
                    
                    total_laws += result['total_laws']
                    processed_laws += result['processed_laws']
                    
                except Exception as e:
                    logger.error(f"❌ {file_path.name}: Exception - {str(e)}")
                    failed_files += 1
        
        # Calculate final statistics
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        success_rate = (successful_files / len(json_files)) * 100 if json_files else 0
        processing_rate = processed_laws / total_time if total_time > 0 else 0
        
        summary = {
            'success': True,
            'total_files': len(json_files),
            'successful_files': successful_files,
            'failed_files': failed_files,
            'total_laws': total_laws,
            'processed_laws': processed_laws,
            'success_rate': success_rate,
            'total_time': total_time,
            'processing_rate': processing_rate,
            'workers_used': self.max_workers,
            'results': results
        }
        
        logger.info("=" * 60)
        logger.info("PARALLEL PROCESSING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total files: {summary['total_files']}")
        logger.info(f"Successful: {summary['successful_files']}")
        logger.info(f"Failed: {summary['failed_files']}")
        logger.info(f"Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"Total laws: {summary['total_laws']}")
        logger.info(f"Processed laws: {summary['processed_laws']}")
        logger.info(f"Total time: {summary['total_time']:.2f} seconds")
        logger.info(f"Processing rate: {summary['processing_rate']:.2f} laws/second")
        logger.info(f"Workers used: {summary['workers_used']}")
        
        return summary


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Parallel preprocess Assembly law data')
    parser.add_argument('--input', type=str, help='Input directory path')
    parser.add_argument('--output', type=str, help='Output directory path')
    parser.add_argument('--workers', type=int, help='Number of parallel workers (default: auto-detect)')
    parser.add_argument('--enable-legal-analysis', action='store_true', default=False,
                       help='Enable comprehensive legal analysis (default: False)')
    parser.add_argument('--disable-legal-analysis', action='store_true',
                       help='Disable comprehensive legal analysis')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
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
    
    logger.info(f"Starting parallel preprocessing with legal analysis: {enable_legal_analysis}")
    
    # Create parallel preprocessor
    preprocessor = ParallelLawPreprocessor(
        enable_legal_analysis=enable_legal_analysis,
        max_workers=args.workers
    )
    
    try:
        summary = preprocessor.preprocess_directory(input_dir, output_dir)
        
        if summary['success']:
            logger.info("Parallel preprocessing completed successfully!")
            return 0
        else:
            logger.error("Parallel preprocessing failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())