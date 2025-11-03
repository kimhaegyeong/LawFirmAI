#!/usr/bin/env python3
"""
Assembly Law Data Validation Script

This script validates processed Assembly law data for quality and completeness.

Usage:
  python validate_processed_laws.py --input data/processed/assembly/law
  python validate_processed_laws.py --input data/processed/assembly/law --output validation_report.json
  python validate_processed_laws.py --help
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/validation.log')
    ]
)
logger = logging.getLogger(__name__)


class LawDataValidator:
    """Validator for processed Assembly law data"""
    
    def __init__(self):
        """Initialize the validator"""
        self.validation_results = {
            'total_files_validated': 0,
            'total_laws_validated': 0,
            'validation_errors': [],
            'quality_metrics': {},
            'start_time': None,
            'end_time': None
        }
    
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate a single processed law file
        
        Args:
            file_path (Path): Path to processed law file
            
        Returns:
            Dict[str, Any]: Validation results for the file
        """
        try:
            logger.info(f"Validating file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                processed_laws = json.load(f)
            
            file_validation = {
                'file_name': file_path.name,
                'total_laws': len(processed_laws),
                'valid_laws': 0,
                'invalid_laws': 0,
                'law_validations': [],
                'file_errors': []
            }
            
            for i, law_data in enumerate(processed_laws):
                try:
                    law_validation = self._validate_single_law(law_data, i)
                    file_validation['law_validations'].append(law_validation)
                    
                    if law_validation['is_valid']:
                        file_validation['valid_laws'] += 1
                    else:
                        file_validation['invalid_laws'] += 1
                        
                except Exception as e:
                    error_msg = f"Error validating law {i}: {str(e)}"
                    logger.error(error_msg)
                    file_validation['file_errors'].append(error_msg)
                    file_validation['invalid_laws'] += 1
            
            self.validation_results['total_files_validated'] += 1
            self.validation_results['total_laws_validated'] += file_validation['total_laws']
            
            return file_validation
            
        except Exception as e:
            error_msg = f"Error validating file {file_path}: {str(e)}"
            logger.error(error_msg)
            self.validation_results['validation_errors'].append(error_msg)
            return {
                'file_name': file_path.name,
                'error': error_msg
            }
    
    def _validate_single_law(self, law_data: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Validate a single law data
        
        Args:
            law_data (Dict[str, Any]): Processed law data
            index (int): Law index in file
            
        Returns:
            Dict[str, Any]: Validation results for the law
        """
        validation = {
            'law_index': index,
            'law_id': law_data.get('law_id', 'unknown'),
            'law_name': law_data.get('law_name', 'unknown'),
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 0.0,
            'checks': {}
        }
        
        # Required field checks
        required_fields = [
            'law_id', 'law_name', 'law_type', 'category',
            'full_text', 'searchable_text', 'articles'
        ]
        
        for field in required_fields:
            if field not in law_data or not law_data[field]:
                validation['errors'].append(f"Missing required field: {field}")
                validation['is_valid'] = False
        
        # Data type checks
        validation['checks']['has_string_fields'] = self._check_string_fields(law_data)
        validation['checks']['has_list_fields'] = self._check_list_fields(law_data)
        validation['checks']['has_dict_fields'] = self._check_dict_fields(law_data)
        
        # Content quality checks
        validation['checks']['text_quality'] = self._check_text_quality(law_data)
        validation['checks']['article_structure'] = self._check_article_structure(law_data)
        validation['checks']['metadata_completeness'] = self._check_metadata_completeness(law_data)
        validation['checks']['searchable_text_quality'] = self._check_searchable_text_quality(law_data)
        
        # Calculate quality score
        validation['quality_score'] = self._calculate_quality_score(validation['checks'])
        
        # Add warnings for low quality scores
        if validation['quality_score'] < 0.7:
            validation['warnings'].append(f"Low quality score: {validation['quality_score']:.2f}")
        
        return validation
    
    def _check_string_fields(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check string field validity"""
        string_fields = ['law_name', 'law_type', 'category', 'full_text', 'searchable_text']
        results = {}
        
        for field in string_fields:
            if field in law_data:
                value = law_data[field]
                results[field] = {
                    'exists': True,
                    'is_string': isinstance(value, str),
                    'is_empty': not value.strip() if isinstance(value, str) else True,
                    'length': len(value) if isinstance(value, str) else 0
                }
            else:
                results[field] = {'exists': False}
        
        return results
    
    def _check_list_fields(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check list field validity"""
        list_fields = ['articles', 'keywords']
        results = {}
        
        for field in list_fields:
            if field in law_data:
                value = law_data[field]
                results[field] = {
                    'exists': True,
                    'is_list': isinstance(value, list),
                    'length': len(value) if isinstance(value, list) else 0,
                    'has_items': len(value) > 0 if isinstance(value, list) else False
                }
            else:
                results[field] = {'exists': False}
        
        return results
    
    def _check_dict_fields(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check dictionary field validity"""
        dict_fields = ['data_quality']
        results = {}
        
        for field in dict_fields:
            if field in law_data:
                value = law_data[field]
                results[field] = {
                    'exists': True,
                    'is_dict': isinstance(value, dict),
                    'has_keys': len(value.keys()) > 0 if isinstance(value, dict) else False
                }
            else:
                results[field] = {'exists': False}
        
        return results
    
    def _check_text_quality(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check text quality"""
        full_text = law_data.get('full_text', '')
        
        return {
            'has_text': bool(full_text),
            'text_length': len(full_text),
            'word_count': len(full_text.split()) if full_text else 0,
            'has_articles': '?? in full_text if full_text else False,
            'has_legal_terms': any(term in full_text for term in ['ë²•ë¥ ', '?œí–‰??, '?œí–‰ê·œì¹™']) if full_text else False,
            'text_quality_score': min(1.0, len(full_text) / 1000) if full_text else 0.0  # Normalize to 0-1
        }
    
    def _check_article_structure(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check article structure validity"""
        articles = law_data.get('articles', [])
        
        if not articles:
            return {'has_articles': False, 'article_count': 0, 'structure_score': 0.0}
        
        article_numbers = []
        valid_articles = 0
        
        for article in articles:
            if isinstance(article, dict):
                article_num = article.get('article_number', '')
                if article_num and article_num.startswith('??) and article_num.endswith('ì¡?):
                    article_numbers.append(article_num)
                    valid_articles += 1
        
        # Check for sequential article numbers
        sequential_score = 0.0
        if len(article_numbers) > 1:
            try:
                nums = [int(num.replace('??, '').replace('ì¡?, '')) for num in article_numbers]
                nums.sort()
                expected = list(range(nums[0], nums[0] + len(nums)))
                sequential_score = 1.0 if nums == expected else 0.5
            except:
                sequential_score = 0.0
        
        return {
            'has_articles': len(articles) > 0,
            'article_count': len(articles),
            'valid_articles': valid_articles,
            'sequential_score': sequential_score,
            'structure_score': (valid_articles / len(articles)) * sequential_score if articles else 0.0
        }
    
    def _check_metadata_completeness(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check metadata completeness"""
        metadata_fields = [
            'promulgation_number', 'promulgation_date', 'enforcement_date',
            'amendment_type', 'ministry', 'parent_law'
        ]
        
        completed_fields = sum(1 for field in metadata_fields if law_data.get(field))
        
        return {
            'total_fields': len(metadata_fields),
            'completed_fields': completed_fields,
            'completeness_score': completed_fields / len(metadata_fields)
        }
    
    def _check_searchable_text_quality(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check searchable text quality"""
        searchable_text = law_data.get('searchable_text', '')
        keywords = law_data.get('keywords', [])
        
        return {
            'has_searchable_text': bool(searchable_text),
            'searchable_length': len(searchable_text),
            'has_keywords': len(keywords) > 0,
            'keyword_count': len(keywords),
            'search_quality_score': min(1.0, len(keywords) / 20) if keywords else 0.0  # Normalize to 0-1
        }
    
    def _calculate_quality_score(self, checks: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        scores = []
        
        # Text quality score
        if 'text_quality' in checks:
            scores.append(checks['text_quality'].get('text_quality_score', 0.0))
        
        # Article structure score
        if 'article_structure' in checks:
            scores.append(checks['article_structure'].get('structure_score', 0.0))
        
        # Metadata completeness score
        if 'metadata_completeness' in checks:
            scores.append(checks['metadata_completeness'].get('completeness_score', 0.0))
        
        # Search quality score
        if 'searchable_text_quality' in checks:
            scores.append(checks['searchable_text_quality'].get('search_quality_score', 0.0))
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def validate_directory(self, input_dir: Path) -> Dict[str, Any]:
        """
        Validate all processed law files in a directory
        
        Args:
            input_dir (Path): Input directory path
            
        Returns:
            Dict[str, Any]: Validation results
        """
        self.validation_results['start_time'] = datetime.now()
        
        logger.info(f"Starting validation of directory: {input_dir}")
        
        # Find all JSON files
        json_files = list(input_dir.glob('*.json'))
        logger.info(f"Found {len(json_files)} JSON files to validate")
        
        file_validations = []
        
        for json_file in json_files:
            if json_file.name == 'preprocessing_summary.json':
                continue  # Skip summary files
            
            try:
                file_validation = self.validate_file(json_file)
                file_validations.append(file_validation)
            except Exception as e:
                error_msg = f"Error validating file {json_file}: {str(e)}"
                logger.error(error_msg)
                file_validations.append({
                    'file_name': json_file.name,
                    'error': error_msg
                })
        
        self.validation_results['end_time'] = datetime.now()
        
        # Generate validation summary
        summary = self._generate_validation_summary(file_validations)
        
        return summary
    
    def _generate_validation_summary(self, file_validations: List[Dict]) -> Dict[str, Any]:
        """Generate validation summary"""
        total_files = len(file_validations)
        successful_files = sum(1 for fv in file_validations if 'error' not in fv)
        failed_files = total_files - successful_files
        
        total_laws = sum(fv.get('total_laws', 0) for fv in file_validations if 'error' not in fv)
        valid_laws = sum(fv.get('valid_laws', 0) for fv in file_validations if 'error' not in fv)
        invalid_laws = sum(fv.get('invalid_laws', 0) for fv in file_validations if 'error' not in fv)
        
        # Calculate average quality score
        all_quality_scores = []
        for fv in file_validations:
            if 'error' not in fv and 'law_validations' in fv:
                for lv in fv['law_validations']:
                    all_quality_scores.append(lv.get('quality_score', 0.0))
        
        avg_quality_score = sum(all_quality_scores) / len(all_quality_scores) if all_quality_scores else 0.0
        
        processing_time = None
        if self.validation_results['start_time'] and self.validation_results['end_time']:
            processing_time = (self.validation_results['end_time'] - self.validation_results['start_time']).total_seconds()
        
        return {
            'validation_summary': {
                'total_files': total_files,
                'successful_files': successful_files,
                'failed_files': failed_files,
                'total_laws': total_laws,
                'valid_laws': valid_laws,
                'invalid_laws': invalid_laws,
                'average_quality_score': avg_quality_score,
                'validation_time_seconds': processing_time,
                'validation_date': datetime.now().isoformat()
            },
            'file_validations': file_validations,
            'errors': self.validation_results['validation_errors']
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Validate processed Assembly law data')
    parser.add_argument('--input', type=str, required=True, help='Input directory path')
    parser.add_argument('--output', type=str, help='Output file path for validation report')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Convert to Path objects
    input_dir = Path(args.input)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Create validator and run
    validator = LawDataValidator()
    
    try:
        validation_results = validator.validate_directory(input_dir)
        
        # Save validation report
        if args.output:
            output_file = Path(args.output)
        else:
            output_file = input_dir / 'validation_report.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Validation completed. Report saved to {output_file}")
        
        # Print summary
        summary = validation_results['validation_summary']
        print("\n" + "="*50)
        print("VALIDATION SUMMARY")
        print("="*50)
        print(f"Total files validated: {summary['total_files']}")
        print(f"Successful files: {summary['successful_files']}")
        print(f"Failed files: {summary['failed_files']}")
        print(f"Total laws validated: {summary['total_laws']}")
        print(f"Valid laws: {summary['valid_laws']}")
        print(f"Invalid laws: {summary['invalid_laws']}")
        print(f"Average quality score: {summary['average_quality_score']:.3f}")
        if summary['validation_time_seconds']:
            print(f"Validation time: {summary['validation_time_seconds']:.2f} seconds")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
