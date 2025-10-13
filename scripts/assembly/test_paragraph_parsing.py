#!/usr/bin/env python3
"""
Detailed test of paragraph parsing
"""

import json
import sys
import os
from pathlib import Path

# Add the parsers directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'parsers'))

from improved_article_parser import ImprovedArticleParser

def test_paragraph_parsing():
    """Test paragraph parsing specifically"""
    print("Testing Paragraph Parsing")
    print("=" * 40)
    
    # Test with specific content that has paragraphs
    test_content = """제3조의2 ① 사설철도회사는 감독관청의 인가를 받아 선로의 연장 또는 개량비용에 충당하기 위하여 이 영에 의한 사항을 정한 1909년 법률 제28호에 의하여 준용하는 철도저당권법 제6조에 규정하는 제한을 넘어 철도재단을 저당으로 하는 채무를 부담할 수 있다.
② 전항의 규정에 의하여 채권을 부담하는 경우에는 이 영에 의한 채무의 총액은 사채액과 합하여 총 주식불입액의 2배를 넘을 수 없다.
③ 최종 대차대조표에 의하여 회사에 현존하는 재산이 총 주식불입액에 미치지 아니한 경우에는 전2항의 규정에 의하여 채무를 분담할 수 없다.
제4조 ① 철도재단 또는 궤도재단에 대한 강제집행은 그 재단의 소유자인 회사의 본점 소재지를 관할하는 지방법원 또는 그 지청의 관할에 전속한다.
② 철도재단 또는 궤도재단의 소유자인 회사가 조선 외에 본점을 갖는 경우에는 전항의 관할재판소는 경성지방법원으로 한다."""
    
    parser = ImprovedArticleParser()
    parsed_data = parser.parse_law(test_content)
    
    print(f"Total articles: {parsed_data.get('total_articles')}")
    print(f"Main articles: {len(parsed_data.get('main_articles', []))}")
    print()
    
    for i, article in enumerate(parsed_data.get('main_articles', []), 1):
        print(f"Article {i}: {article.get('article_number')}")
        print(f"  Title: {article.get('article_title')}")
        print(f"  Content: {article.get('article_content', '')[:100]}...")
        print(f"  Sub-articles: {len(article.get('sub_articles', []))}")
        
        for j, sub_article in enumerate(article.get('sub_articles', []), 1):
            print(f"    {j}. {sub_article.get('type')} {sub_article.get('number')}")
            print(f"       Content: {sub_article.get('content', '')[:80]}...")
        print()

if __name__ == "__main__":
    test_paragraph_parsing()
