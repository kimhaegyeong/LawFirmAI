#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Duplicate Detection Pipeline

This script provides a comprehensive duplicate detection pipeline that can be run
independently to scan all processed data for duplicates and generate reports.

Usage:
    python run_duplicate_detection.py --input data/processed --output reports/
    python run_duplicate_detection.py --input data/processed --auto-resolve --strategy quality_based
    python run_duplicate_detection.py --help
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import quality modules
try:
    sys.path.append(str(project_root / 'scripts' / 'data_processing' / 'quality'))
    from duplicate_detector import AdvancedDuplicateDetector, detect_duplicates_comprehensive
    from duplicate_resolver import IntelligentDuplicateResolver, ResolutionResult
    from data_quality_validator import DataQualityValidator
    QUALITY_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Error importing quality modules: {e}")
    QUALITY_MODULES_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/duplicate_detection.log')
    ]
)
logger = logging.getLogger(__name__)


class DuplicateDetectionPipeline:
    """Standalone duplicate detection pipeline"""
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the pipeline
        
        Args:
            input_dir: Input directory containing processed data
            output_dir: Output directory for reports
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        if QUALITY_MODULES_AVAILABLE:
            self.detector = AdvancedDuplicateDetector()
            self.resolver = IntelligentDuplicateResolver()
            self.quality_validator = DataQualityValidator()
        else:
            logger.error("Quality modules not available. Pipeline cannot run.")
            sys.exit(1)
        
        # Pipeline statistics
        self.stats = {
            'files_processed': 0,
            'laws_processed': 0,
            'file_duplicates_found': 0,
            'content_duplicates_found': 0,
            'semantic_duplicates_found': 0,
            'groups_resolved': 0,
            'start_time': None,
            'end_time': None
        }
    
    def run_detection(self, auto_resolve: bool = False, resolution_strategy: str = 'quality_based') -> Dict[str, Any]:
        """
        Run the complete duplicate detection pipeline
        
        Args:
            auto_resolve: Whether to automatically resolve duplicates
            resolution_strategy: Strategy for resolving duplicates
            
        Returns:
            Dict[str, Any]: Pipeline results
        """
        try:
            self.stats['start_time'] = datetime.now().isoformat()
            logger.info("Starting duplicate detection pipeline")
            
            # Step 1: Collect all processed data
            processed_data = self._collect_processed_data()
            
            # Step 2: Detect file-level duplicates
            file_duplicates = self._detect_file_duplicates(processed_data['files'])
            
            # Step 3: Detect content-level duplicates
            content_duplicates = self._detect_content_duplicates(processed_data['laws'])
            
            # Step 4: Detect semantic duplicates
            semantic_duplicates = self._detect_semantic_duplicates(processed_data['laws'])
            
            # Step 5: Resolve duplicates if requested
            resolution_results = []
            if auto_resolve:
                resolution_results = self._resolve_duplicates(
                    file_duplicates + content_duplicates + semantic_duplicates,
                    resolution_strategy
                )
            
            # Step 6: Generate reports
            reports = self._generate_reports(
                file_duplicates, content_duplicates, semantic_duplicates, resolution_results
            )
            
            self.stats['end_time'] = datetime.now().isoformat()
            logger.info("Duplicate detection pipeline completed")
            
            return {
                'statistics': self.stats,
                'file_duplicates': file_duplicates,
                'content_duplicates': content_duplicates,
                'semantic_duplicates': semantic_duplicates,
                'resolution_results': resolution_results,
                'reports': reports
            }
            
        except Exception as e:
            logger.error(f"Error in duplicate detection pipeline: {e}")
            return {'error': str(e)}
    
    def _collect_processed_data(self) -> Dict[str, List[Any]]:
        """Collect all processed data from input directory"""
        logger.info(f"Collecting processed data from {self.input_dir}")
        
        files = []
        laws = []
        
        # Find all JSON files in the input directory
        for json_file in self.input_dir.rglob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Add file metadata
                file_info = {
                    'path': str(json_file),
                    'name': json_file.name,
                    'size': json_file.stat().st_size,
                    'modified': json_file.stat().st_mtime
                }
                files.append(file_info)
                
                # Process law data
                if isinstance(data, dict):
                    # Single law
                    if 'law_name' in data or 'articles' in data:
                        laws.append(data)
                elif isinstance(data, list):
                    # Multiple laws
                    for item in data:
                        if isinstance(item, dict) and ('law_name' in item or 'articles' in item):
                            laws.append(item)
                
                self.stats['files_processed'] += 1
                
            except Exception as e:
                logger.warning(f"Error processing file {json_file}: {e}")
        
        self.stats['laws_processed'] = len(laws)
        logger.info(f"Collected {len(files)} files and {len(laws)} laws")
        
        return {'files': files, 'laws': laws}
    
    def _detect_file_duplicates(self, files: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Detect file-level duplicates"""
        logger.info("Detecting file-level duplicates")
        
        try:
            # Convert file info to Path objects for detector
            file_paths = [Path(f['path']) for f in files]
            
            duplicate_groups = self.detector.detect_file_level_duplicates(file_paths)
            
            # Convert back to file info format
            file_duplicates = []
            for group in duplicate_groups:
                file_group = []
                for file_path in group:
                    # Find corresponding file info
                    for file_info in files:
                        if Path(file_info['path']) == file_path:
                            file_group.append(file_info)
                            break
                if file_group:
                    file_duplicates.append(file_group)
            
            self.stats['file_duplicates_found'] = len(file_duplicates)
            logger.info(f"Found {len(file_duplicates)} file-level duplicate groups")
            
            return file_duplicates
            
        except Exception as e:
            logger.error(f"Error detecting file duplicates: {e}")
            return []
    
    def _detect_content_duplicates(self, laws: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Detect content-level duplicates"""
        logger.info("Detecting content-level duplicates")
        
        try:
            content_duplicates = self.detector.detect_content_level_duplicates(laws)
            
            self.stats['content_duplicates_found'] = len(content_duplicates)
            logger.info(f"Found {len(content_duplicates)} content-level duplicate groups")
            
            return content_duplicates
            
        except Exception as e:
            logger.error(f"Error detecting content duplicates: {e}")
            return []
    
    def _detect_semantic_duplicates(self, laws: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Detect semantic duplicates"""
        logger.info("Detecting semantic duplicates")
        
        try:
            semantic_duplicates = self.detector.detect_semantic_duplicates(laws)
            
            self.stats['semantic_duplicates_found'] = len(semantic_duplicates)
            logger.info(f"Found {len(semantic_duplicates)} semantic duplicate groups")
            
            return semantic_duplicates
            
        except Exception as e:
            logger.error(f"Error detecting semantic duplicates: {e}")
            return []
    
    def _resolve_duplicates(self, duplicate_groups: List[List[Any]], strategy: str) -> List[ResolutionResult]:
        """Resolve duplicates using specified strategy"""
        logger.info(f"Resolving duplicates using {strategy} strategy")
        
        try:
            resolution_results = self.resolver.resolve_duplicates(duplicate_groups, strategy)
            
            self.stats['groups_resolved'] = len(resolution_results)
            logger.info(f"Resolved {len(resolution_results)} duplicate groups")
            
            return resolution_results
            
        except Exception as e:
            logger.error(f"Error resolving duplicates: {e}")
            return []
    
    def _generate_reports(self, file_duplicates: List, content_duplicates: List, 
                        semantic_duplicates: List, resolution_results: List) -> Dict[str, str]:
        """Generate comprehensive reports"""
        logger.info("Generating reports")
        
        reports = {}
        
        try:
            # Generate duplicate detection report
            detection_report = {
                'summary': {
                    'file_duplicates': len(file_duplicates),
                    'content_duplicates': len(content_duplicates),
                    'semantic_duplicates': len(semantic_duplicates),
                    'total_duplicate_groups': len(file_duplicates) + len(content_duplicates) + len(semantic_duplicates),
                    'generated_at': datetime.now().isoformat()
                },
                'file_duplicates': file_duplicates,
                'content_duplicates': content_duplicates,
                'semantic_duplicates': semantic_duplicates,
                'statistics': self.stats
            }
            
            detection_report_path = self.output_dir / 'duplicate_detection_report.json'
            with open(detection_report_path, 'w', encoding='utf-8') as f:
                json.dump(detection_report, f, ensure_ascii=False, indent=2)
            reports['detection_report'] = str(detection_report_path)
            
            # Generate resolution report if available
            if resolution_results:
                resolution_report_path = self.output_dir / 'duplicate_resolution_report.json'
                self.resolver.export_resolution_report(resolution_results, str(resolution_report_path))
                reports['resolution_report'] = str(resolution_report_path)
            
            # Generate summary report
            summary_report = self._generate_summary_report(detection_report, resolution_results)
            summary_report_path = self.output_dir / 'duplicate_summary_report.json'
            with open(summary_report_path, 'w', encoding='utf-8') as f:
                json.dump(summary_report, f, ensure_ascii=False, indent=2)
            reports['summary_report'] = str(summary_report_path)
            
            logger.info(f"Generated {len(reports)} reports")
            return reports
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            return {}
    
    def _generate_summary_report(self, detection_report: Dict, resolution_results: List) -> Dict[str, Any]:
        """Generate a summary report"""
        total_groups = detection_report['summary']['total_duplicate_groups']
        resolved_groups = len(resolution_results)
        
        # Calculate quality metrics
        quality_metrics = {
            'duplicate_rate': total_groups / max(self.stats['laws_processed'], 1),
            'resolution_rate': resolved_groups / max(total_groups, 1),
            'average_confidence': 0.0
        }
        
        if resolution_results:
            quality_metrics['average_confidence'] = sum(r.confidence_score for r in resolution_results) / len(resolution_results)
        
        return {
            'execution_summary': {
                'start_time': self.stats['start_time'],
                'end_time': self.stats['end_time'],
                'duration_minutes': self._calculate_duration(),
                'files_processed': self.stats['files_processed'],
                'laws_processed': self.stats['laws_processed']
            },
            'duplicate_summary': detection_report['summary'],
            'quality_metrics': quality_metrics,
            'recommendations': self._generate_recommendations(quality_metrics)
        }
    
    def _calculate_duration(self) -> float:
        """Calculate pipeline duration in minutes"""
        if self.stats['start_time'] and self.stats['end_time']:
            start = datetime.fromisoformat(self.stats['start_time'])
            end = datetime.fromisoformat(self.stats['end_time'])
            return (end - start).total_seconds() / 60.0
        return 0.0
    
    def _generate_recommendations(self, quality_metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on quality metrics"""
        recommendations = []
        
        duplicate_rate = quality_metrics['duplicate_rate']
        resolution_rate = quality_metrics['resolution_rate']
        average_confidence = quality_metrics['average_confidence']
        
        if duplicate_rate > 0.3:
            recommendations.append("High duplicate rate detected. Consider improving data collection process.")
        
        if resolution_rate < 0.8:
            recommendations.append("Low resolution rate. Review resolution strategies and thresholds.")
        
        if average_confidence < 0.7:
            recommendations.append("Low confidence in resolutions. Consider manual review for critical data.")
        
        if duplicate_rate < 0.1 and resolution_rate > 0.9:
            recommendations.append("Excellent data quality. Consider implementing automated duplicate prevention.")
        
        return recommendations


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description='Standalone Duplicate Detection Pipeline')
    
    parser.add_argument('--input', '-i', required=True, help='Input directory containing processed data')
    parser.add_argument('--output', '-o', required=True, help='Output directory for reports')
    parser.add_argument('--auto-resolve', action='store_true', help='Automatically resolve duplicates')
    parser.add_argument('--strategy', default='quality_based', 
                       choices=['quality_based', 'completeness_based', 'recency_based', 'conservative'],
                       help='Resolution strategy to use')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if quality modules are available
    if not QUALITY_MODULES_AVAILABLE:
        print("Error: Quality modules not available. Please check installation.")
        sys.exit(1)
    
    # Run pipeline
    try:
        pipeline = DuplicateDetectionPipeline(args.input, args.output)
        results = pipeline.run_detection(args.auto_resolve, args.strategy)
        
        if 'error' in results:
            print(f"Pipeline failed: {results['error']}")
            sys.exit(1)
        
        # Print summary
        stats = results['statistics']
        print(f"\n=== Duplicate Detection Pipeline Results ===")
        print(f"Files processed: {stats['files_processed']}")
        print(f"Laws processed: {stats['laws_processed']}")
        print(f"File duplicates found: {stats['file_duplicates_found']}")
        print(f"Content duplicates found: {stats['content_duplicates_found']}")
        print(f"Semantic duplicates found: {stats['semantic_duplicates_found']}")
        
        if args.auto_resolve:
            print(f"Groups resolved: {stats['groups_resolved']}")
        
        print(f"\nReports generated:")
        for report_type, report_path in results['reports'].items():
            print(f"  {report_type}: {report_path}")
        
        print(f"\nPipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

