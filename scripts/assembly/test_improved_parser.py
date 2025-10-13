#!/usr/bin/env python3
"""
Test script for the improved article parser

This script tests the improved parser with the problematic law documents
to verify that the issues have been resolved.
"""

import json
import sys
from pathlib import Path

# Add parsers module to path
sys.path.append(str(Path(__file__).parent / 'parsers'))

from parsers.improved_article_parser import ImprovedArticleParser


def test_parser_with_sample_data():
    """Test the parser with sample law content"""
    
    # Sample law content (simplified version of the problematic data)
    sample_law_content = """
    제1조(목적)
    이 규칙은 「가축분뇨의 관리 및 이용에 관한 법률」에서 위임된 가축분뇨의 자원화 및 이용 등에 관한 사항과 그 시행에 필요한 사항을 규정함을 목적으로 한다.
    
    제2조(퇴비)
    「가축분뇨의 관리 및 이용에 관한 법률」(이하 "법"이라 한다) 제2조제5호에서 "농림축산식품부령이 정하는 기준"이란 「비료관리법」 제2조제4호에 따라 고시한 비료공정규격 중 퇴비의 공정규격을 말한다.
    
    제3조(액비)
    법 제2조제6호에서 "농림축산식품부령이 정하는 기준"이란 「비료관리법」 제2조제4호에 따라 고시한 비료공정규격 중 가축분뇨발효비료(액)의 공정규격을 말한다.
    
    제4조(작목별 비료의 수요량 등 조사)
    ① 법 제7조제1항에 따라 농경지의 양분(養分) 현황을 고려하여 적정한 규모의 가축이 사육될 수 있도록 하기 위하여 농림축산식품부장관이 조사할 수 있는 사항은 다음 각 호와 같다.
    ② 제1항에 따른 조사는 매년 실시한다.
    
    제5조(축사의 이전비 등 지원 절차)
    ① 농림축산식품부장관은 제4조에 따른 조사 결과에 따라 농경지에 포함된 비료의 함량 및 비료의 공급량이 비료의 수요량을 초과하여 해당 지역의 축사를 이전하거나 철거하여야 한다고 판단되는 경우에는 관할 시·도지사 또는 시장·군수·구청장으로 하여금 그 축사의 이전비 또는 철거비 등을 산정하게 한 후 농림축산식품부장관에게 지원을 요청하도록 하여야 한다.
    
    부칙
    제1조(시행일)
    이 규칙은 공포한 날부터 시행한다.
    
    제2조(다른 법령의 개정)
    가축분뇨의 자원화 및 이용 촉진에 관한 규칙 일부를 다음과 같이 개정한다.
    """
    
    # Initialize parser
    parser = ImprovedArticleParser()
    
    # Parse the content
    print("=== Testing Improved Article Parser ===")
    print(f"Input content length: {len(sample_law_content)} characters")
    print()
    
    parsed_data = parser.parse_law_document(sample_law_content)
    
    # Display results
    print("=== Parsing Results ===")
    print(f"Parsing status: {parsed_data.get('parsing_status')}")
    print(f"Total articles: {parsed_data.get('total_articles')}")
    print(f"Main articles: {len(parsed_data.get('main_articles', []))}")
    print(f"Supplementary articles: {len(parsed_data.get('supplementary_articles', []))}")
    print()
    
    # Display main articles
    print("=== Main Articles ===")
    for i, article in enumerate(parsed_data.get('main_articles', []), 1):
        print(f"{i}. {article['article_number']} - {article['article_title']}")
        print(f"   Content length: {len(article['article_content'])} characters")
        print(f"   Sub-articles: {len(article['sub_articles'])}")
        print(f"   References: {article['references']}")
        print()
    
    # Display supplementary articles
    print("=== Supplementary Articles ===")
    for i, article in enumerate(parsed_data.get('supplementary_articles', []), 1):
        print(f"{i}. {article['article_number']} - {article['article_title']}")
        print(f"   Content length: {len(article['article_content'])} characters")
        print(f"   Sub-articles: {len(article['sub_articles'])}")
        print()
    
    # Validate the parsed data
    print("=== Validation Results ===")
    is_valid, errors = parser.validate_parsed_data(parsed_data)
    print(f"Valid: {is_valid}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("No validation errors found!")
    
    return parsed_data


def test_with_real_file():
    """Test with a real problematic file"""
    
    # Path to the problematic file
    file_path = Path("../../data/processed/assembly/law/2025101201_ui_cleaned/20251012/가축분뇨의_자원화_및_이용_촉진에_관한_규칙_assembly_law_3693.json")
    
    if not file_path.exists():
        print(f"Test file not found: {file_path}")
        return None
    
    print("=== Testing with Real File ===")
    print(f"File: {file_path}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print(f"Original articles count: {len(original_data.get('articles', []))}")
    
    # Extract law content (simplified extraction)
    law_content = ""
    for article in original_data.get('articles', []):
        law_content += article.get('article_content', '') + "\n\n"
    
    # Parse with improved parser
    parser = ImprovedArticleParser()
    parsed_data = parser.parse_law_document(law_content)
    
    print(f"Improved parser articles count: {parsed_data.get('total_articles')}")
    print(f"Main articles: {len(parsed_data.get('main_articles', []))}")
    print(f"Supplementary articles: {len(parsed_data.get('supplementary_articles', []))}")
    
    # Compare with original
    print("\n=== Comparison with Original ===")
    original_articles = original_data.get('articles', [])
    improved_articles = parsed_data.get('all_articles', [])
    
    print(f"Original: {len(original_articles)} articles")
    print(f"Improved: {len(improved_articles)} articles")
    
    # Check for duplicate article numbers in original
    original_numbers = [article.get('article_number') for article in original_articles]
    duplicate_numbers = [num for num in set(original_numbers) if original_numbers.count(num) > 1]
    
    if duplicate_numbers:
        print(f"Original has duplicate article numbers: {duplicate_numbers}")
    else:
        print("Original has no duplicate article numbers")
    
    # Check for duplicate article numbers in improved
    improved_numbers = [article.get('article_number') for article in improved_articles]
    duplicate_numbers_improved = [num for num in set(improved_numbers) if improved_numbers.count(num) > 1]
    
    if duplicate_numbers_improved:
        print(f"Improved has duplicate article numbers: {duplicate_numbers_improved}")
    else:
        print("Improved has no duplicate article numbers")
    
    return parsed_data


if __name__ == '__main__':
    print("Testing Improved Article Parser")
    print("=" * 50)
    
    # Test with sample data
    sample_result = test_parser_with_sample_data()
    
    print("\n" + "=" * 50)
    
    # Test with real file
    real_result = test_with_real_file()
    
    print("\n" + "=" * 50)
    print("Testing completed!")
