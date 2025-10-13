#!/usr/bin/env python3
"""
Test the improved parser with control character removal
"""

import json
import sys
import os
from pathlib import Path

# Add the parsers directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'parsers'))

from improved_article_parser import ImprovedArticleParser

def test_control_character_removal():
    """Test control character removal"""
    print("Testing Control Character Removal")
    print("=" * 40)
    
    # Test with content that has control characters
    test_content = """제1조(목적)\n이 법은 6·25전쟁 전후 적 지역에서 비정규군 신분으로 국가를 위하여 중대하고 특수한 임무를 수행하였거나 특별한 희생을 한 것에 대해 국민적 공감대가 형성된 공로자와 그 유족에 대한 보상에 관한 사항을 규정함을 목적으로 한다.
제2조(정의)\n① 이 법에서 사용하는 용어의 뜻은 다음과 같다.
② "비정규군"이란 6·25전쟁 전후 적 지역에서 활동한 비정규군을 말한다."""
    
    parser = ImprovedArticleParser()
    parsed_data = parser.parse_law(test_content)
    
    print(f"Total articles: {parsed_data.get('total_articles')}")
    print(f"Main articles: {len(parsed_data.get('main_articles', []))}")
    print()
    
    for i, article in enumerate(parsed_data.get('main_articles', []), 1):
        print(f"Article {i}: {article.get('article_number')}")
        print(f"  Title: {article.get('article_title')}")
        print(f"  Content: {article.get('article_content', '')}")
        print(f"  Has \\n: {'\\n' in article.get('article_content', '')}")
        print(f"  Has \\t: {'\\t' in article.get('article_content', '')}")
        print()
        
        for j, sub_article in enumerate(article.get('sub_articles', []), 1):
            print(f"    {j}. {sub_article.get('type')} {sub_article.get('number')}")
            print(f"       Content: {sub_article.get('content', '')}")
            print(f"       Has \\n: {'\\n' in sub_article.get('content', '')}")
            print(f"       Has \\t: {'\\t' in sub_article.get('content', '')}")
            print()

if __name__ == "__main__":
    test_control_character_removal()
