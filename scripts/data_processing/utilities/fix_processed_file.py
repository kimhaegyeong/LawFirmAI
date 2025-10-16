#!/usr/bin/env python3
"""
Fix existing processed law files using the improved parser

This script takes the problematic processed files and fixes them using
the improved article parser.
"""

import json
import sys
from pathlib import Path

# Add parsers module to path
sys.path.append(str(Path(__file__).parent / 'parsers'))

from parsers.improved_article_parser import ImprovedArticleParser


def fix_processed_file(input_file: Path, output_file: Path):
    """
    Fix a processed law file using the improved parser
    
    Args:
        input_file (Path): Input processed file
        output_file (Path): Output fixed file
    """
    print(f"Fixing file: {input_file}")
    
    # Read the problematic file
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Extract law content from articles
    law_content = ""
    for article in original_data.get('articles', []):
        law_content += article.get('article_content', '') + "\n\n"
    
    if not law_content.strip():
        print(f"No content found in {input_file}")
        return False
    
    # Parse with improved parser
    parser = ImprovedArticleParser()
    parsed_data = parser.parse_law_document(law_content)
    
    # Create fixed data structure
    fixed_data = {
        'law_id': original_data.get('law_id', ''),
        'law_name': original_data.get('law_name', ''),
        'law_type': original_data.get('law_type', ''),
        'category': original_data.get('category', ''),
        'promulgation_number': original_data.get('promulgation_number', ''),
        'promulgation_date': original_data.get('promulgation_date', ''),
        'enforcement_date': original_data.get('enforcement_date', ''),
        'amendment_type': original_data.get('amendment_type', ''),
        'ministry': original_data.get('ministry', ''),
        'processed_at': original_data.get('processed_at', ''),
        'parser_version': 'improved_v1.0_fixed',
        'articles': parsed_data.get('all_articles', []),
        'main_articles': parsed_data.get('main_articles', []),
        'supplementary_articles': parsed_data.get('supplementary_articles', []),
        'total_articles': parsed_data.get('total_articles', 0),
        'parsing_status': parsed_data.get('parsing_status', 'unknown')
    }
    
    # Validate the fixed data
    is_valid, errors = parser.validate_parsed_data(parsed_data)
    fixed_data['is_valid'] = is_valid
    fixed_data['validation_errors'] = errors
    
    # Save the fixed file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, ensure_ascii=False, indent=2)
    
    print(f"Fixed file saved to: {output_file}")
    print(f"Original articles: {len(original_data.get('articles', []))}")
    print(f"Fixed articles: {len(fixed_data['articles'])}")
    print(f"Main articles: {len(fixed_data['main_articles'])}")
    print(f"Supplementary articles: {len(fixed_data['supplementary_articles'])}")
    print(f"Valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    print()
    
    return True


def main():
    """Main function"""
    # Path to the problematic file
    input_file = Path("../../data/processed/assembly/law/2025101201_ui_cleaned/20251012/가축분뇨의_자원화_및_이용_촉진에_관한_규칙_assembly_law_3693.json")
    
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return 1
    
    # Create output directory
    output_dir = Path("../../data/processed/assembly/law/2025101201_ui_cleaned/20251012_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output file path
    output_file = output_dir / input_file.name
    
    # Fix the file
    success = fix_processed_file(input_file, output_file)
    
    if success:
        print("File fixing completed successfully!")
        return 0
    else:
        print("File fixing failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
