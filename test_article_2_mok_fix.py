#!/usr/bin/env python3
"""
제2조 다목 파싱 오류 수정 테스트 스크립트
"""

import json
import sys
from pathlib import Path

# Adjust the path to import from scripts/assembly/parsers
sys.path.append(str(Path(__file__).parent / 'scripts' / 'assembly'))
sys.path.append(str(Path(__file__).parent / 'scripts' / 'assembly' / 'parsers'))

from parsers.article_parser import ArticleParser
from parsers.text_normalizer import TextNormalizer

def test_article_2_mok_parsing():
    """제2조의 다목 파싱 오류를 테스트합니다."""
    
    # 테스트 파일
    file_path = Path("data/processed/assembly/law/2025101201_optimized/20251012/건설근로자의_기능등급확인증_발급_등에_관한_규칙_assembly_law_3801.json")
    
    if not file_path.exists():
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=== Article 2 Mok Parsing Fix Test ===")
    print(f"Law Name: {data.get('law_name', 'Unknown')}")
    
    # Find Article 2
    article_2 = None
    for article in data.get('articles', []):
        if article.get('article_number') == '제2조':
            article_2 = article
            break
    
    if not article_2:
        print("Article 2 not found.")
        return False
    
    print(f"\nArticle 2 Title: {article_2.get('article_title', 'No title')}")
    print(f"Article 2 Content Length: {len(article_2.get('article_content', ''))}")
    
    # Check existing parsing results
    original_sub_articles = article_2.get('sub_articles', [])
    print(f"\n=== Original Parsing Results ===")
    print(f"sub_articles count: {len(original_sub_articles)}")
    
    mok_items = [item for item in original_sub_articles if item.get('type') == '목']
    ho_items = [item for item in original_sub_articles if item.get('type') == '호']
    
    print(f"Mok (목) items count: {len(mok_items)}")
    print(f"Ho (호) items count: {len(ho_items)}")
    
    if mok_items:
        print(f"\nMok (목) items:")
        for i, item in enumerate(mok_items):
            number = item.get('number', 'Unknown')
            content = item.get('content', '')[:50]
            print(f"  {i+1}. {number}: {content}...")
    
    if ho_items:
        print(f"\nHo (호) items:")
        for i, item in enumerate(ho_items):
            number = item.get('number', 'Unknown')
            content = item.get('content', '')[:50]
            print(f"  {i+1}. {number}: {content}...")
    
    # Test with new parser
    article_parser = ArticleParser()
    text_normalizer = TextNormalizer()
    
    # Test parsing with original content
    original_content = article_2.get('article_content', '')
    normalized_content = text_normalizer.normalize(original_content)
    
    print(f"\n=== Improved Parsing Test ===")
    print(f"Normalized content length: {len(normalized_content)}")
    
    # New parsing results
    parsed_articles = article_parser.parse_articles(normalized_content, '')
    
    if parsed_articles:
        test_parsed_article = parsed_articles[0]
        new_sub_articles = test_parsed_article.get('sub_articles', [])
        
        print(f"New sub_articles count: {len(new_sub_articles)}")
        
        new_mok_items = [item for item in new_sub_articles if item.get('type') == '목']
        new_ho_items = [item for item in new_sub_articles if item.get('type') == '호']
        
        print(f"New Mok (목) items count: {len(new_mok_items)}")
        print(f"New Ho (호) items count: {len(new_ho_items)}")
        
        if new_mok_items:
            print(f"\nNew Mok (목) items:")
            for i, item in enumerate(new_mok_items):
                number = item.get('number', 'Unknown')
                content = item.get('content', '')[:50]
                print(f"  {i+1}. {number}: {content}...")
        
        if new_ho_items:
            print(f"\nNew Ho (호) items:")
            for i, item in enumerate(new_ho_items):
                number = item.get('number', 'Unknown')
                content = item.get('content', '')[:50]
                print(f"  {i+1}. {number}: {content}...")
        
        # Compare results
        print(f"\n=== Result Comparison ===")
        if len(mok_items) > 0 and len(new_mok_items) == 0:
            print("SUCCESS: Invalid Mok (목) items have been removed!")
            return True
        elif len(mok_items) > 0 and len(new_mok_items) > 0:
            print("FAILED: Invalid Mok (목) items still exist.")
            return False
        else:
            print("SUCCESS: Mok (목) items have been processed correctly.")
            return True
    else:
        print("Parsing failed")
        return False

if __name__ == "__main__":
    if test_article_2_mok_parsing():
        print("\nSUCCESS: Article 2 Mok parsing fix test passed!")
    else:
        print("\nFAILED: Article 2 Mok parsing fix test failed!")
