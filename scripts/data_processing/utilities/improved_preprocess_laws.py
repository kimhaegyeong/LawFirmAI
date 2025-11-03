#!/usr/bin/env python3
"""
Improved Assembly Law Data Preprocessing Script

This script uses the improved article parser to correctly process law documents
with proper separation of main articles and supplementary provisions.
"""

import argparse
import json
import logging
import re
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parsers module to path
sys.path.append(str(Path(__file__).parent / 'parsers'))

from parsers.improved_article_parser import ImprovedArticleParser

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/improved_preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)


class ImprovedLawPreprocessor:
    """Improved law preprocessor using the new parser"""
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.parser = ImprovedArticleParser()
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'validation_errors': 0
        }
    
    def process_law_file(self, file_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """
        Process a single law file using the improved parser and save each law separately
        
        Args:
            file_path (Path): Path to the law file
            output_dir (Path): Output directory for processed files
            
        Returns:
            List[Dict[str, Any]]: List of processed law data
        """
        processed_laws = []
        
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Process each law separately
            if 'laws' in raw_data and isinstance(raw_data['laws'], list):
                for i, law_data in enumerate(raw_data['laws']):
                    try:
                        # Extract individual law content
                        law_content = law_data.get('law_content', '')
                        if not law_content:
                            logger.warning(f"No content found for law {i+1} in {file_path}")
                            continue
                        
                        # Parse using improved parser
                        parsed_data = self.parser.parse_law_document(law_content)
                        
                        # Validate parsed data
                        is_valid, errors = self.parser.validate_parsed_data(parsed_data)
                        if not is_valid:
                            logger.warning(f"Validation errors in law {i+1} of {file_path}: {errors}")
                            self.stats['validation_errors'] += 1
                        
                        # Create processed law data
                        processed_law = self._create_individual_law_data(law_data, parsed_data, file_path, i)
                        
                        # Save individual law file
                        self._save_individual_law(processed_law, output_dir, file_path, i)
                        
                        processed_laws.append(processed_law)
                        self.stats['successful'] += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing law {i+1} in {file_path}: {e}")
                        self.stats['failed'] += 1
                        continue
            
            return processed_laws
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.stats['failed'] += 1
            return []
    
    def _extract_law_content(self, raw_data: Dict[str, Any]) -> str:
        """
        Extract law content from raw data
        
        Args:
            raw_data (Dict[str, Any]): Raw law data
            
        Returns:
            str: Extracted law content
        """
        # Handle the new data structure with 'laws' array
        if 'laws' in raw_data and isinstance(raw_data['laws'], list):
            laws_content = []
            for law in raw_data['laws']:
                if isinstance(law, dict) and 'law_content' in law:
                    laws_content.append(law['law_content'])
            if laws_content:
                return '\n\n'.join(laws_content)
        
        # Try different content extraction methods
        content_sources = [
            raw_data.get('current_text', {}).get('법령', {}).get('개정�?, {}).get('개정문내??),
            raw_data.get('current_text', {}).get('법령', {}).get('조문'),
            raw_data.get('content'),
            raw_data.get('text')
        ]
        
        for content_source in content_sources:
            if content_source:
                if isinstance(content_source, list):
                    if len(content_source) > 0 and isinstance(content_source[0], list):
                        # Handle nested list structure
                        return ' '.join([' '.join(item) if isinstance(item, list) else str(item) for item in content_source])
                    else:
                        return ' '.join([str(item) for item in content_source])
                elif isinstance(content_source, str):
                    return content_source
                elif isinstance(content_source, dict):
                    # Extract from dictionary structure
                    return self._extract_from_dict(content_source)
        
        return ""
    
    def _create_individual_law_data(self, law_data: Dict[str, Any], parsed_data: Dict[str, Any], file_path: Path, law_index: int) -> Dict[str, Any]:
        """
        Create processed data for individual law
        
        Args:
            law_data (Dict[str, Any]): Raw law data
            parsed_data (Dict[str, Any]): Parsed law data
            file_path (Path): Source file path
            law_index (int): Index of law in the file
            
        Returns:
            Dict[str, Any]: Processed individual law data
        """
        # Extract law metadata
        law_name = law_data.get('law_name', '').strip()
        cont_id = law_data.get('cont_id', '')
        cont_sid = law_data.get('cont_sid', '')
        
        # Create unique law ID
        law_id = f"{cont_id}_{cont_sid}" if cont_id and cont_sid else f"law_{file_path.stem}_{law_index}"
        
        # Create safe filename from law name
        safe_filename = self._create_safe_filename(law_name) if law_name else f"law_{law_index}"
        
        return {
            'law_id': law_id,
            'law_name': law_name,
            'law_type': self._extract_law_type(law_name),
            'category': self._extract_category(law_name),
            'promulgation_number': self._extract_promulgation_number(parsed_data),
            'promulgation_date': self._extract_promulgation_date(parsed_data),
            'enforcement_date': self._extract_enforcement_date(parsed_data),
            'amendment_type': self._extract_amendment_type(parsed_data),
            'ministry': self._extract_ministry(parsed_data),
            'processed_at': datetime.now().isoformat(),
            'parser_version': 'improved_v1.0',
            'source_file': str(file_path),
            'law_index': law_index,
            'safe_filename': safe_filename,
            **parsed_data
        }
    
    def _save_individual_law(self, processed_law: Dict[str, Any], output_dir: Path, file_path: Path, law_index: int):
        """
        Save individual law to separate file
        
        Args:
            processed_law (Dict[str, Any]): Processed law data
            output_dir (Path): Output directory
            file_path (Path): Source file path
            law_index (int): Index of law in the file
        """
        # Create filename
        safe_filename = processed_law.get('safe_filename', f"law_{law_index}")
        output_filename = f"{safe_filename}_{processed_law['law_id']}.json"
        output_path = output_dir / output_filename
        
        # Save the file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_law, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved individual law: {output_path}")
    
    def _create_safe_filename(self, law_name: str) -> str:
        """
        Create safe filename from law name
        
        Args:
            law_name (str): Law name
            
        Returns:
            str: Safe filename
        """
        if not law_name:
            return "unnamed_law"
        
        # Remove special characters and replace with underscores
        safe_name = re.sub(r'[^\w\s-]', '', law_name)
        safe_name = re.sub(r'[-\s]+', '_', safe_name)
        safe_name = safe_name.strip('_')
        
        # Limit length
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        
        return safe_name
    
    def _extract_law_type(self, law_name: str) -> str:
        """Extract law type from law name"""
        if not law_name:
            return ""
        
        if '�? in law_name and '?�행?? in law_name:
            return '?�행??
        elif '�? in law_name and '?�행규칙' in law_name:
            return '?�행규칙'
        elif '�? in law_name and '규칙' in law_name:
            return '규칙'
        elif '�? in law_name:
            return '법률'
        elif '조�?' in law_name:
            return '조�?'
        else:
            return '기�?'
    
    def _extract_category(self, law_name: str) -> str:
        """Extract category from law name"""
        if not law_name:
            return ""
        
        # Simple category extraction based on keywords
        categories = {
            '?�료': ['?�료', '보건', '병원', '?�사', '간호'],
            '교육': ['교육', '?�교', '?�생', '교사'],
            '?�경': ['?�경', '?�염', '?��?, '?�질'],
            '교통': ['교통', '?�로', '?�동�?, '?�전'],
            '경제': ['경제', '금융', '?�??, '?�자'],
            '?�정': ['?�정', '공무??, '?��?', '기�?']
        }
        
        for category, keywords in categories.items():
            if any(keyword in law_name for keyword in keywords):
                return category
        
        return '기�?'
    
    def _extract_promulgation_number(self, parsed_data: Dict[str, Any]) -> str:
        """Extract promulgation number from parsed data"""
        # This would need to be implemented based on the parsing logic
        return ""
    
    def _extract_promulgation_date(self, parsed_data: Dict[str, Any]) -> str:
        """Extract promulgation date from parsed data"""
        # This would need to be implemented based on the parsing logic
        return ""
    
    def _extract_enforcement_date(self, parsed_data: Dict[str, Any]) -> str:
        """Extract enforcement date from parsed data"""
        # This would need to be implemented based on the parsing logic
        return ""
    
    def _extract_amendment_type(self, parsed_data: Dict[str, Any]) -> str:
        """Extract amendment type from parsed data"""
        # This would need to be implemented based on the parsing logic
        return ""
    
    def _extract_ministry(self, parsed_data: Dict[str, Any]) -> str:
        """Extract ministry from parsed data"""
        # This would need to be implemented based on the parsing logic
        return ""
    
    def _extract_from_dict(self, data: Dict[str, Any]) -> str:
        """Extract text content from dictionary structure"""
        content_parts = []
        
        for key, value in data.items():
            if isinstance(value, str):
                content_parts.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        content_parts.append(item)
                    elif isinstance(item, dict):
                        content_parts.append(self._extract_from_dict(item))
        
        return ' '.join(content_parts)
    
    def _create_processed_law_data(self, raw_data: Dict[str, Any], 
                                 parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create processed law data structure
        
        Args:
            raw_data (Dict[str, Any]): Raw law data
            parsed_data (Dict[str, Any]): Parsed law data
            
        Returns:
            Dict[str, Any]: Processed law data
        """
        # Extract basic information
        basic_info = raw_data.get('basic_info', {})
        
        processed_law = {
            'law_id': basic_info.get('id', ''),
            'law_name': basic_info.get('name', ''),
            'law_type': basic_info.get('law_type', ''),
            'category': basic_info.get('category', ''),
            'promulgation_number': basic_info.get('promulgation_number', ''),
            'promulgation_date': basic_info.get('promulgation_date', ''),
            'enforcement_date': basic_info.get('enforcement_date', ''),
            'amendment_type': basic_info.get('amendment_type', ''),
            'ministry': basic_info.get('ministry', ''),
            'processed_at': datetime.now().isoformat(),
            'parser_version': 'improved_v1.0'
        }
        
        # Add parsed articles
        all_articles = parsed_data.get('all_articles', [])
        processed_law['articles'] = all_articles
        processed_law['total_articles'] = len(all_articles)
        
        # Separate main and supplementary articles
        processed_law['main_articles'] = parsed_data.get('main_articles', [])
        processed_law['supplementary_articles'] = parsed_data.get('supplementary_articles', [])
        
        # Add parsing status
        processed_law['parsing_status'] = parsed_data.get('parsing_status', 'unknown')
        
        # Add validation information
        is_valid, errors = self.parser.validate_parsed_data(parsed_data)
        processed_law['is_valid'] = is_valid
        processed_law['validation_errors'] = errors
        
        return processed_law
    
    def process_directory(self, input_dir: Path, output_dir: Path) -> None:
        """
        Process all law files in a directory and save each law separately
        
        Args:
            input_dir (Path): Input directory
            output_dir (Path): Output directory
        """
        logger.info(f"Processing directory: {input_dir}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all JSON files recursively
        json_files = list(input_dir.glob('**/*.json'))
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        total_laws_processed = 0
        
        for i, file_path in enumerate(json_files, 1):
            self.stats['total_processed'] += 1
            
            # Process the file
            processed_laws = self.process_law_file(file_path, output_dir)
            total_laws_processed += len(processed_laws)
            
            # Log progress
            if i % 10 == 0:
                logger.info(f"Processed {i} files")
        
        # Log final statistics
        logger.info("=== Processing Statistics ===")
        logger.info(f"Total files processed: {self.stats['total_processed']}")
        logger.info(f"Total laws processed: {total_laws_processed}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Validation errors: {self.stats['validation_errors']}")
        
        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_processed']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        logger.info("Processing completed successfully")
    
    def _log_statistics(self) -> None:
        """Log processing statistics"""
        logger.info("=== Processing Statistics ===")
        logger.info(f"Total files processed: {self.stats['total_processed']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Validation errors: {self.stats['validation_errors']}")
        
        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_processed']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Improved Assembly law data preprocessing')
    parser.add_argument('--input', type=str, required=True, help='Input directory path')
    parser.add_argument('--output', type=str, required=True, help='Output directory path')
    parser.add_argument('--validate-only', action='store_true', help='Run validation only')
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    # Initialize preprocessor
    preprocessor = ImprovedLawPreprocessor()
    
    try:
        if args.validate_only:
            logger.info("Running validation only...")
            # TODO: Implement validation-only mode
        else:
            # Process the directory
            preprocessor.process_directory(input_dir, output_dir)
        
        logger.info("Processing completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
