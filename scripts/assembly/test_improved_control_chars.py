#!/usr/bin/env python3
"""
Test the improved control character removal
"""

import json
import sys
import os
from pathlib import Path

# Add the parsers directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'parsers'))

from improved_article_parser import ImprovedArticleParser

def test_control_character_removal():
    """Test control character removal with actual data"""
    print("Testing Improved Control Character Removal")
    print("=" * 50)
    
    # Test with content that has actual control characters
    test_content = """제1조(목적)
이 영은 4차 산업혁명의 총체적 변화 과정을 국가적인 방향전환의 계기로 삼아, 경제성장과 사회문제해결을 함께 추구하는 포용적 성장으로 일자리를 창출하고 국가 경쟁력을 확보하며 국민의 삶의 질을 향상시키기 위하여 4차산업혁명위원회를 설치하고, 그 구성 및 운영에 필요한 사항을 규정함을 목적으로 한다.

제2조(설치 및 기능)
① 초연결·초지능 기반의 4차 산업혁명 도래에 따른 과학기술·인공지능 및 데이터 기술 등의 기반을 확보하고, 신산업·신서비스 육성 및 사회변화 대응에 필요한 주요 정책 등에 관한 사항을 효율적으로 심의·조정하기 위하여 대통령 소속으로 4차산업혁명위원회를 둔다."""
    
    parser = ImprovedArticleParser()
    
    # Test the _clean_content method directly
    print("Testing _clean_content method:")
    print("-" * 30)
    
    cleaned_content = parser._clean_content(test_content)
    print(f"Original content length: {len(test_content)}")
    print(f"Cleaned content length: {len(cleaned_content)}")
    print(f"Contains \\n: {'\\n' in cleaned_content}")
    print(f"Contains actual newlines: {'\n' in cleaned_content}")
    print()
    
    print("Cleaned content preview:")
    print("-" * 20)
    print(cleaned_content[:200] + "..." if len(cleaned_content) > 200 else cleaned_content)
    print()
    
    # Test full parsing
    print("Testing full parsing:")
    print("-" * 20)
    
    parsed_data = parser.parse_law(test_content)
    
    print(f"Total articles: {parsed_data.get('total_articles')}")
    print(f"Main articles: {len(parsed_data.get('main_articles', []))}")
    print()
    
    for i, article in enumerate(parsed_data.get('main_articles', [])[:2]):
        print(f"Article {i+1}: {article.get('article_number')} - {article.get('article_title')}")
        content = article.get('article_content', '')
        print(f"Content length: {len(content)}")
        print(f"Contains \\n: {'\\n' in content}")
        print(f"Contains actual newlines: {'\n' in content}")
        print(f"Content preview: {content[:100]}...")
        print()

def test_with_real_data():
    """Test with real data file"""
    print("Testing with Real Data File")
    print("=" * 30)
    
    # Load a real data file
    data_file = Path("../../data/processed/assembly/law/clean_individual_laws/4차산업혁명위원회의_설치_및_운영에_관한_규정_2017082300000001_0006.json")
    
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        return
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded data file: {data.get('law_name')}")
    
    # Check for control characters in the data
    main_articles = data.get('main_articles', [])
    
    for i, article in enumerate(main_articles[:3]):
        content = article.get('article_content', '')
        print(f"\nArticle {i+1}: {article.get('article_number')}")
        print(f"Contains \\n: {'\\n' in content}")
        print(f"Contains actual newlines: {'\n' in content}")
        print(f"Contains \\t: {'\\t' in content}")
        print(f"Contains actual tabs: {'\t' in content}")
        
        # Show first 100 characters
        print(f"Content preview: {repr(content[:100])}")

if __name__ == "__main__":
    test_control_character_removal()
    print("\n" + "="*60 + "\n")
    test_with_real_data()
