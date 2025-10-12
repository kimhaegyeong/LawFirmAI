#!/usr/bin/env python3
"""
개선된 목(目) 파싱 테스트 스크립트
"""

import json
import sys
from pathlib import Path

# Adjust the path to import from scripts/assembly/parsers
sys.path.append(str(Path(__file__).parent / 'scripts' / 'assembly'))
sys.path.append(str(Path(__file__).parent / 'scripts' / 'assembly' / 'parsers'))

from parsers.article_parser import ArticleParser
from parsers.text_normalizer import TextNormalizer

def test_improved_mok_parsing():
    """개선된 목(目) 파싱을 테스트합니다."""
    
    # 문제가 있던 파일 테스트
    file_path = Path("data/processed/assembly/law/2025101201_fixed_sub_articles/20251012/_대한민국_법원의_날_제정에_관한_규칙_assembly_law_1951.json")
    
    if not file_path.exists():
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=== 개선된 목(目) 파싱 테스트 ===")
    print(f"법률명: {data.get('law_name', 'Unknown')}")
    
    # 새로운 파서로 테스트
    article_parser = ArticleParser()
    text_normalizer = TextNormalizer()
    
    # 제1조로 테스트 (목이 없어야 하는 조문)
    test_article = None
    for article in data.get('articles', []):
        if article.get('article_number') == '제1조' and article.get('article_title') == '목적':
            test_article = article
            break
    
    if not test_article:
        print("테스트할 제1조를 찾을 수 없습니다.")
        return False
    
    print(f"\n테스트 조문: {test_article.get('article_number')} - {test_article.get('article_title')}")
    
    # 원본 내용으로 파싱 테스트
    original_content = test_article.get('article_content', '')
    print(f"원본 내용 길이: {len(original_content)}")
    
    # 정규화된 내용으로 파싱
    normalized_content = text_normalizer.normalize(original_content)
    print(f"정규화된 내용 길이: {len(normalized_content)}")
    
    # 새로운 파싱 결과
    parsed_articles = article_parser.parse_articles(normalized_content, '')
    
    # 디버깅을 위해 _extract_items_korean 직접 테스트
    print(f"\n=== _extract_items_korean 직접 테스트 ===")
    items_result = article_parser._extract_items_korean(normalized_content)
    print(f"_extract_items_korean 결과: {len(items_result)}개")
    for i, item in enumerate(items_result):
        print(f"  {i+1}. {item.get('type')} {item.get('letter')}: {item.get('content', '')[:30]}...")
    
    if parsed_articles:
        test_parsed_article = parsed_articles[0]
        new_sub_articles = test_parsed_article.get('sub_articles', [])
        
        print(f"\n=== 개선된 파싱 결과 ===")
        print(f"새로운 sub_articles 수: {len(new_sub_articles)}")
        
        # 기존 결과와 비교
        original_sub_articles = test_article.get('sub_articles', [])
        print(f"기존 sub_articles 수: {len(original_sub_articles)}")
        
        # 목(目) 항목 분석
        mok_items = [item for item in new_sub_articles if item.get('type') == '목']
        print(f"목(目) 항목 수: {len(mok_items)}")
        
        if mok_items:
            print(f"\n목(目) 항목들:")
            for i, item in enumerate(mok_items):
                letter = item.get('letter', 'Unknown')
                content = item.get('content', '')
                print(f"  {i+1}. {letter}: {content[:50]}...")
        else:
            print(f"\n목(目) 항목이 없습니다. (올바른 결과)")
        
        # 디버깅을 위해 원본 내용 확인
        print(f"\n=== 원본 내용 분석 ===")
        print(f"원본 내용에서 '다.' 패턴 찾기:")
        import re
        da_pattern = re.compile(r'다\.')
        da_matches = da_pattern.findall(original_content)
        print(f"'다.' 패턴 발견: {len(da_matches)}개")
        
        ga_pattern = re.compile(r'가\.')
        ga_matches = ga_pattern.findall(original_content)
        print(f"'가.' 패턴 발견: {len(ga_matches)}개")
        
        na_pattern = re.compile(r'나\.')
        na_matches = na_pattern.findall(original_content)
        print(f"'나.' 패턴 발견: {len(na_matches)}개")
        
        # 전체 sub_articles 표시
        print(f"\n=== 전체 sub_articles ===")
        for i, sub_article in enumerate(new_sub_articles):
            sub_type = sub_article.get('type', 'Unknown')
            number = sub_article.get('number', 'Unknown')
            content = sub_article.get('content', '')
            print(f"{i+1}. {sub_type} {number}: {content[:50]}...")
        
        return True
    else:
        print("파싱 실패")
        return False

if __name__ == "__main__":
    test_improved_mok_parsing()
