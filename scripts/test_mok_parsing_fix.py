#!/usr/bin/env python3
"""
Test script for 목 parsing fix

This script tests the fixed 목 parsing logic to ensure that
"다" 목 items are not created without "가", "나" 목 items.
"""

import sys
import json
import logging
from pathlib import Path

# Add parsers module to path
sys.path.append(str(Path(__file__).parent / 'assembly' / 'parsers'))

from article_parser import ArticleParser

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_mok_parsing():
    """Test 목 parsing with the problematic file"""
    
    # Load the problematic file
    test_file = Path("data/processed/assembly/law/2025101201_ui_cleaned/20251012/6ㆍ25전쟁_납북피해_진상규명_및_납북피해자_명예회복에_관한_법률_시행령_assembly_law_2280.json")
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    with open(test_file, 'r', encoding='utf-8') as f:
        law_data = json.load(f)
    
    # Initialize parser
    parser = ArticleParser()
    
    # Test each article
    for article in law_data.get('articles', []):
        article_number = article.get('article_number', 'Unknown')
        article_content = article.get('article_content', '')
        
        logger.info(f"\n=== Testing Article {article_number} ===")
        logger.info(f"Content: {article_content[:200]}...")
        
        # Parse articles
        parsed_articles = parser.parse_articles(article_content, "")
        
        if parsed_articles:
            for parsed_article in parsed_articles:
                logger.info(f"Parsed article: {parsed_article.get('article_number', 'Unknown')}")
                
                # Check sub-articles
                sub_articles = parsed_article.get('sub_articles', [])
                logger.info(f"Found {len(sub_articles)} sub-articles")
                
                for sub_article in sub_articles:
                    sub_type = sub_article.get('type', 'Unknown')
                    sub_number = sub_article.get('number', 'Unknown')
                    sub_content = sub_article.get('content', '')
                    
                    logger.info(f"  - {sub_type} {sub_number}: {sub_content[:100]}...")
                    
                    # Check for 목 items
                    if sub_type == '목':
                        logger.warning(f"    WARNING: Found 목 item {sub_number} without proper sequence!")
        
        # Also test the specific 목 extraction method
        logger.info(f"\n--- Testing _extract_items_korean for Article {article_number} ---")
        items = parser._extract_items_korean(article_content)
        logger.info(f"Extracted {len(items)} items")
        
        for item in items:
            item_type = item.get('type', 'Unknown')
            item_number = item.get('number', 'Unknown')
            item_letter = item.get('letter', 'Unknown')
            item_content = item.get('content', '')
            
            logger.info(f"  - {item_type} {item_letter}. ({item_number}): {item_content[:100]}...")


if __name__ == "__main__":
    test_mok_parsing()
