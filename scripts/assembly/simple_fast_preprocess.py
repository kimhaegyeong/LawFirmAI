#!/usr/bin/env python3
"""
Simple Fast Preprocessing Script

Minimal preprocessing with maximum speed and compatibility.
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/simple_preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)


def get_optimal_worker_count() -> int:
    """Get optimal number of workers based on system resources"""
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Conservative approach: use 70% of CPU cores, but not more than 6
    optimal_workers = min(int(cpu_count * 0.7), 6)
    
    # If memory is limited (< 8GB), reduce workers
    if memory_gb < 8:
        optimal_workers = min(optimal_workers, 4)
    
    logger.info(f"System: {cpu_count} CPU cores, {memory_gb:.1f}GB RAM")
    logger.info(f"Optimal workers: {optimal_workers}")
    
    return optimal_workers


def simple_text_cleanup(text: str) -> str:
    """Simple text cleanup without complex parsing"""
    if not text:
        return ""
    
    # Basic cleanup
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')
    text = text.replace('\t', ' ')
    
    # Remove excessive whitespace
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def process_single_law(law_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single law with minimal overhead"""
    try:
        # Extract basic metadata
        metadata = {
            'law_id': law_data.get('law_id'),
            'law_name': law_data.get('law_name'),
            'law_number': law_data.get('law_number'),
            'law_type': law_data.get('law_type'),
            'enactment_date': law_data.get('enactment_date'),
            'amendment_date': law_data.get('amendment_date'),
            'status': law_data.get('status'),
            'source_url': law_data.get('source_url')
        }
        
        # Get law content
        law_content = law_data.get('law_content', '')
        
        # Simple text cleanup
        cleaned_content = simple_text_cleanup(law_content)
        
        # Create processed law data
        processed_law = {
            'law_id': law_data.get('law_id'),
            'law_name': law_data.get('law_name'),
            'law_number': law_data.get('law_number'),
            'metadata': metadata,
            'original_content': law_content,
            'cleaned_content': cleaned_content,
            'processing_timestamp': datetime.now().isoformat(),
            'processing_version': 'simple_v1.0'
        }
        
        return processed_law
        
    except Exception as e:
        logger.error(f"Error processing law {law_data.get('law_name', 'Unknown')}: {str(e)}")
        return None


def process_file_worker(args: tuple) -> Dict[str, Any]:
    """Worker function for processing a single law file"""
    input_file, output_dir = args
    
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
        
        # Process all laws in the file
        processed_laws = []
        for i, law_data in enumerate(laws):
            try:
                processed_law = process_single_law(law_data)
                if processed_law:
                    processed_laws.append(processed_law)
                    file_stats['processed_laws'] += 1
                
                # Log progress every 50 laws
                if (i + 1) % 50 == 0:
                    logger.info(f"[Worker] Processed {i + 1}/{len(laws)} laws from {input_file.name}")
                    
            except Exception as e:
                logger.error(f"[Worker] Error processing law {i+1}: {str(e)}")
                continue
        
        # Save results
        if processed_laws:
            output_file = output_dir / f"simple_{input_file.stem}.json"
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
        
        # Calculate time
        end_time = datetime.now()
        file_stats['processing_time'] = (end_time - start_time).total_seconds()
        
        logger.info(f"[Worker] Completed {input_file.name}: {file_stats['processed_laws']}/{file_stats['total_laws']} laws in {file_stats['processing_time']:.2f}s")
        
        return file_stats
        
    except Exception as e:
        logger.error(f"[Worker] Failed to process file {input_file}: {str(e)}")
        return file_stats


def preprocess_directory(input_dir: Path, output_dir: Path, max_workers: int = None) -> Dict[str, Any]:
    """Preprocess directory with optimized parallel processing"""
    start_time = datetime.now()
    
    # Find all JSON files
    json_files = list(input_dir.glob('*.json'))
    if not json_files:
        logger.error(f"No JSON files found in {input_dir}")
        return {'success': False, 'error': 'No JSON files found'}
    
    # Auto-detect workers if not specified
    if max_workers is None:
        max_workers = get_optimal_worker_count()
    
    logger.info("=" * 60)
    logger.info("SIMPLE FAST PREPROCESSING STARTED")
    logger.info("=" * 60)
    logger.info(f"Files: {len(json_files)}")
    logger.info(f"Workers: {max_workers}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare worker arguments
    worker_args = [
        (json_file, output_dir)
        for json_file in json_files
    ]
    
    # Process files in parallel
    results = []
    successful_files = 0
    total_laws = 0
    processed_laws = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_file_worker, args): args[0]
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
                logger.error(f"Failed to process {file_path.name}: {str(e)}")
    
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
        'workers_used': max_workers
    }
    
    # Print results
    logger.info("=" * 60)
    logger.info("SIMPLE FAST PREPROCESSING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total files: {summary['total_files']}")
    logger.info(f"Successful: {summary['successful_files']}")
    logger.info(f"Success rate: {summary['success_rate']:.1f}%")
    logger.info(f"Total laws: {summary['total_laws']}")
    logger.info(f"Processed laws: {summary['processed_laws']}")
    logger.info(f"Total time: {summary['total_time']:.2f} seconds")
    logger.info(f"Processing rate: {summary['processing_rate']:.2f} laws/second")
    logger.info(f"Workers used: {summary['workers_used']}")
    
    return summary


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simple fast preprocess Assembly law data')
    parser.add_argument('--input', type=str, help='Input directory path')
    parser.add_argument('--output', type=str, help='Output directory path')
    parser.add_argument('--workers', type=int, help='Number of parallel workers (default: auto-detect)')
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
    
    logger.info(f"Starting simple fast preprocessing")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    
    try:
        summary = preprocess_directory(
            input_dir, 
            output_dir, 
            max_workers=args.workers
        )
        
        if summary['success']:
            logger.info("Simple fast preprocessing completed successfully!")
            return 0
        else:
            logger.error("Simple fast preprocessing failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
