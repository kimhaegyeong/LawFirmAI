#!/usr/bin/env python3
"""
Fast Assembly Law Data Preprocessing Script

Ultra-fast preprocessing with minimal memory usage and maximum performance.
Optimized for speed over comprehensive analysis.

Usage:
  python fast_preprocess_laws.py --input data/raw/assembly/law/20251010 --output data/processed/assembly/law
  python fast_preprocess_laws.py --workers 12 --batch-size 20
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

from parsers import LawHTMLParser, MetadataExtractor, TextNormalizer
from parsers.improved_article_parser import ImprovedArticleParser

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/fast_preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)


class FastLawPreprocessor:
    """Ultra-fast preprocessing class with minimal overhead"""
    
    def __init__(self, batch_size: int = 20):
        """Initialize fast preprocessor"""
        self.batch_size = batch_size
        self.html_parser = LawHTMLParser()
        self.article_parser = ImprovedArticleParser()  # Use faster rule-based parser
        self.metadata_extractor = MetadataExtractor()
        self.text_normalizer = TextNormalizer()
        
        logger.info(f"Fast preprocessor initialized (batch_size: {batch_size})")
    
    def _process_law_batch(self, laws_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of laws efficiently"""
        processed_laws = []
        
        for law_data in laws_batch:
            try:
                # Minimal processing for speed
                metadata = self.metadata_extractor.extract_metadata(law_data)
                parsed_content = self.html_parser.parse_html(law_data.get('law_content', ''))
                articles = self.article_parser.parse_articles(parsed_content)
                normalized_content = self.text_normalizer.normalize_text(parsed_content)
                
                processed_law = {
                    'law_id': law_data.get('law_id'),
                    'law_name': law_data.get('law_name'),
                    'law_number': law_data.get('law_number'),
                    'metadata': metadata,
                    'parsed_content': parsed_content,
                    'normalized_content': normalized_content,
                    'articles': articles,
                    'processing_timestamp': datetime.now().isoformat(),
                    'processing_version': 'fast_v1.0'
                }
                
                processed_laws.append(processed_law)
                
            except Exception as e:
                logger.error(f"Error processing law {law_data.get('law_name', 'Unknown')}: {str(e)}")
                continue
        
        return processed_laws
    
    def _process_file_worker(self, args: tuple) -> Dict[str, Any]:
        """Worker function for processing a single file"""
        input_file, output_dir, batch_size = args
        
        start_time = datetime.now()
        file_stats = {
            'file_name': input_file.name,
            'total_laws': 0,
            'processed_laws': 0,
            'processing_time': 0
        }
        
        try:
            logger.info(f"[Worker] Processing: {input_file.name}")
            
            # Load file
            with open(input_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            laws = raw_data.get('laws', [])
            file_stats['total_laws'] = len(laws)
            
            if not laws:
                return file_stats
            
            # Create worker preprocessor
            worker_preprocessor = FastLawPreprocessor(batch_size=batch_size)
            
            # Process laws in batches
            all_processed_laws = []
            for i in range(0, len(laws), batch_size):
                batch = laws[i:i + batch_size]
                processed_batch = worker_preprocessor._process_law_batch(batch)
                all_processed_laws.extend(processed_batch)
                
                # Force garbage collection after each batch
                gc.collect()
            
            file_stats['processed_laws'] = len(all_processed_laws)
            
            # Save results
            if all_processed_laws:
                output_file = output_dir / f"fast_{input_file.stem}.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'source_file': input_file.name,
                        'processing_timestamp': datetime.now().isoformat(),
                        'total_laws': file_stats['total_laws'],
                        'processed_laws': file_stats['processed_laws'],
                        'laws': all_processed_laws
                    }, f, ensure_ascii=False, indent=2)
            
            # Calculate time
            end_time = datetime.now()
            file_stats['processing_time'] = (end_time - start_time).total_seconds()
            
            logger.info(f"[Worker] ✅ {input_file.name}: {file_stats['processed_laws']}/{file_stats['total_laws']} laws in {file_stats['processing_time']:.2f}s")
            
            return file_stats
            
        except Exception as e:
            logger.error(f"[Worker] ❌ {input_file.name}: {str(e)}")
            return file_stats
    
    def preprocess_directory(self, input_dir: Path, output_dir: Path, max_workers: int = None) -> Dict[str, Any]:
        """Preprocess directory with maximum speed"""
        start_time = datetime.now()
        
        # Find files
        json_files = list(input_dir.glob('*.json'))
        if not json_files:
            logger.error(f"No JSON files found in {input_dir}")
            return {'success': False, 'error': 'No JSON files found'}
        
        # Auto-detect workers if not specified
        if max_workers is None:
            cpu_count = mp.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            max_workers = min(int(cpu_count * 0.9), 12)  # Use 90% of CPU cores, max 12
            if memory_gb < 8:
                max_workers = min(max_workers, 6)
        
        logger.info(f"🚀 FAST PREPROCESSING STARTED")
        logger.info(f"📁 Files: {len(json_files)}")
        logger.info(f"⚡ Workers: {max_workers}")
        logger.info(f"📦 Batch size: {self.batch_size}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare worker arguments
        worker_args = [
            (json_file, output_dir, self.batch_size)
            for json_file in json_files
        ]
        
        # Process in parallel
        results = []
        successful_files = 0
        total_laws = 0
        processed_laws = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_file_worker, args): args[0]
                for args in worker_args
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['processed_laws'] > 0:
                        successful_files += 1
                    
                    total_laws += result['total_laws']
                    processed_laws += result['processed_laws']
                    
                except Exception as e:
                    logger.error(f"❌ {file_path.name}: {str(e)}")
        
        # Calculate statistics
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        success_rate = (successful_files / len(json_files)) * 100 if json_files else 0
        processing_rate = processed_laws / total_time if total_time > 0 else 0
        
        summary = {
            'success': True,
            'total_files': len(json_files),
            'successful_files': successful_files,
            'total_laws': total_laws,
            'processed_laws': processed_laws,
            'success_rate': success_rate,
            'total_time': total_time,
            'processing_rate': processing_rate,
            'workers_used': max_workers,
            'batch_size': self.batch_size
        }
        
        # Print results
        logger.info("=" * 60)
        logger.info("🚀 FAST PREPROCESSING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"📁 Total files: {summary['total_files']}")
        logger.info(f"✅ Successful: {summary['successful_files']}")
        logger.info(f"📊 Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"📜 Total laws: {summary['total_laws']}")
        logger.info(f"⚡ Processed laws: {summary['processed_laws']}")
        logger.info(f"⏱️  Total time: {summary['total_time']:.2f} seconds")
        logger.info(f"🚀 Processing rate: {summary['processing_rate']:.2f} laws/second")
        logger.info(f"👥 Workers used: {summary['workers_used']}")
        logger.info(f"📦 Batch size: {summary['batch_size']}")
        
        return summary


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fast preprocess Assembly law data')
    parser.add_argument('--input', type=str, help='Input directory path')
    parser.add_argument('--output', type=str, help='Output directory path')
    parser.add_argument('--workers', type=int, help='Number of parallel workers (default: auto-detect)')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size for processing (default: 20)')
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
    
    logger.info(f"🚀 Starting FAST preprocessing")
    logger.info(f"📁 Input: {input_dir}")
    logger.info(f"📁 Output: {output_dir}")
    
    # Create fast preprocessor
    preprocessor = FastLawPreprocessor(batch_size=args.batch_size)
    
    try:
        summary = preprocessor.preprocess_directory(
            input_dir, 
            output_dir, 
            max_workers=args.workers
        )
        
        if summary['success']:
            logger.info("🎉 Fast preprocessing completed successfully!")
            return 0
        else:
            logger.error("❌ Fast preprocessing failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("⚠️ Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"💥 Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
