#!/usr/bin/env python3
"""
Test the improved parser with real law data
"""

import json
import sys
import os
from pathlib import Path

# Add the parsers directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'parsers'))

from improved_article_parser import ImprovedArticleParser

def test_real_law_parsing():
    """Test parsing with real law data"""
    print("Testing Improved Parser with Real Law Data")
    print("=" * 50)
    
    # Load a real law file
    law_file = Path("../../data/raw/assembly/law/2025101201/law_page_465_110225.json")
    
    if not law_file.exists():
        print(f"Law file not found: {law_file}")
        return
    
    with open(law_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Loaded raw law file: {law_file.name}")
    print(f"Number of laws in file: {len(raw_data)}")
    
    # Get the first law
    if raw_data and 'laws' in raw_data and raw_data['laws']:
        first_law = raw_data['laws'][0]
        law_content = first_law.get('law_content', '')
        
        print(f"\nLaw name: {first_law.get('law_name', 'Unknown')}")
        print(f"Original content length: {len(law_content)}")
        print(f"Contains newlines: {'\n' in law_content}")
        print(f"Contains tabs: {'\t' in law_content}")
        
        # Test the parser
        parser = ImprovedArticleParser()
        parsed_data = parser.parse_law(law_content)
        
        print(f"\nParsing Results:")
        print(f"Total articles: {parsed_data.get('total_articles')}")
        print(f"Main articles: {len(parsed_data.get('main_articles', []))}")
        print(f"Supplementary articles: {len(parsed_data.get('supplementary_articles', []))}")
        print(f"Parsing status: {parsed_data.get('parsing_status')}")
        
        # Check for control characters in parsed results
        print(f"\nControl Character Check:")
        main_articles = parsed_data.get('main_articles', [])
        
        for i, article in enumerate(main_articles[:3]):
            content = article.get('article_content', '')
            print(f"Article {i+1} ({article.get('article_number')}):")
            print(f"  - Contains \\n: {'\\n' in content}")
            print(f"  - Contains actual newlines: {'\n' in content}")
            print(f"  - Contains \\t: {'\\t' in content}")
            print(f"  - Contains actual tabs: {'\t' in content}")
            print(f"  - Content length: {len(content)}")
            print(f"  - Preview: {content[:80]}...")
            print()

if __name__ == "__main__":
    test_real_law_parsing()
