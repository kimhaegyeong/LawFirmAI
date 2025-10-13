#!/usr/bin/env python3
"""
Reprocess existing data with improved control character removal
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add the parsers directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'parsers'))

from improved_article_parser import ImprovedArticleParser

def clean_article_content(content: str) -> str:
    """Clean article content by removing control characters"""
    if not content:
        return ""
    
    # Remove control characters (both actual and escaped)
    # Actual control characters
    cleaned = content.replace('\n', ' ')  # Replace actual newline with space
    cleaned = cleaned.replace('\t', ' ')  # Replace actual tab with space
    cleaned = cleaned.replace('\r', ' ')  # Replace actual carriage return with space
    cleaned = cleaned.replace('\f', ' ')  # Replace form feed with space
    cleaned = cleaned.replace('\v', ' ')  # Replace vertical tab with space
    
    # Escaped control characters
    cleaned = cleaned.replace('\\n', ' ')  # Replace escaped newline with space
    cleaned = cleaned.replace('\\t', ' ')  # Replace escaped tab with space
    cleaned = cleaned.replace('\\r', ' ')  # Replace escaped carriage return with space
    cleaned = cleaned.replace('\\"', '"')  # Replace escaped quotes
    cleaned = cleaned.replace("\\'", "'")  # Replace escaped single quotes
    cleaned = cleaned.replace('\\\\', '\\')  # Replace escaped backslashes
    
    # Remove other control characters (ASCII 0-31 except space)
    import string
    control_chars = ''.join(chr(i) for i in range(32) if chr(i) not in string.whitespace)
    for char in control_chars:
        cleaned = cleaned.replace(char, ' ')
    
    # Normalize whitespace
    import re
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

def reprocess_article_data(article: Dict[str, Any]) -> Dict[str, Any]:
    """Reprocess a single article to remove control characters"""
    # Clean main article content
    if 'article_content' in article:
        article['article_content'] = clean_article_content(article['article_content'])
    
    # Clean sub-articles
    if 'sub_articles' in article and article['sub_articles']:
        for sub_article in article['sub_articles']:
            if 'content' in sub_article:
                sub_article['content'] = clean_article_content(sub_article['content'])
            
            # Clean sub-paragraphs
            if 'sub_paragraphs' in sub_article and sub_article['sub_paragraphs']:
                for sub_para in sub_article['sub_paragraphs']:
                    if 'content' in sub_para:
                        sub_para['content'] = clean_article_content(sub_para['content'])
    
    return article

def reprocess_law_file(input_file: Path, output_file: Path) -> bool:
    """Reprocess a single law file to remove control characters"""
    try:
        print(f"Processing: {input_file.name}")
        
        # Load the file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Track changes
        changes_made = False
        
        # Process main articles
        if 'main_articles' in data and data['main_articles']:
            for article in data['main_articles']:
                original_content = article.get('article_content', '')
                processed_article = reprocess_article_data(article)
                new_content = processed_article.get('article_content', '')
                
                if original_content != new_content:
                    changes_made = True
                    print(f"  - Cleaned article: {article.get('article_number', 'Unknown')}")
        
        # Process supplementary articles
        if 'supplementary_articles' in data and data['supplementary_articles']:
            for article in data['supplementary_articles']:
                original_content = article.get('article_content', '')
                processed_article = reprocess_article_data(article)
                new_content = processed_article.get('article_content', '')
                
                if original_content != new_content:
                    changes_made = True
                    print(f"  - Cleaned supplementary article: {article.get('article_number', 'Unknown')}")
        
        # Update parser version
        data['parser_version'] = 'improved_v1.1_control_chars_fixed'
        
        # Save the processed file
        if changes_made:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  [OK] Saved with control character fixes")
            return True
        else:
            print(f"  - No control characters found")
            return False
            
    except Exception as e:
        print(f"  [ERROR] Error processing {input_file.name}: {e}")
        return False

def main():
    """Main function to reprocess all law files"""
    print("Reprocessing Law Files with Improved Control Character Removal")
    print("=" * 60)
    
    # Define paths
    input_dir = Path("../../data/processed/assembly/law/clean_individual_laws")
    output_dir = Path("../../data/processed/assembly/law/improved_individual_laws")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files
    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} files to process")
    print()
    
    processed_count = 0
    fixed_count = 0
    
    for json_file in json_files:
        output_file = output_dir / json_file.name
        
        if reprocess_law_file(json_file, output_file):
            fixed_count += 1
        
        processed_count += 1
        
        # Progress indicator
        if processed_count % 10 == 0:
            print(f"Progress: {processed_count}/{len(json_files)} files processed")
    
    print()
    print("=" * 60)
    print(f"Processing complete!")
    print(f"Total files processed: {processed_count}")
    print(f"Files with control character fixes: {fixed_count}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
