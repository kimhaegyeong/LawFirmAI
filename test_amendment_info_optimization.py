#!/usr/bin/env python3
"""
amendment_info 최적화 테스트 스크립트
"""

import json
import sys
from pathlib import Path

# Adjust the path to import from scripts/assembly/parsers
sys.path.append(str(Path(__file__).parent / 'scripts' / 'assembly'))
sys.path.append(str(Path(__file__).parent / 'scripts' / 'assembly' / 'parsers'))

from parsers.article_parser import ArticleParser
from parsers.text_normalizer import TextNormalizer

def test_amendment_info_optimization():
    """amendment_info 최적화를 테스트합니다."""
    
    # 테스트 파일
    file_path = Path("data/processed/assembly/law/2025101201_final_fixed/20251012/3ㆍ1운동_및_대한민국임시정부_수립_100주년_기념사업추진위원회의_설치_및_운영에_관한_규_assembly_law_2364.json")
    
    if not file_path.exists():
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=== amendment_info 최적화 테스트 ===")
    print(f"법률명: {data.get('law_name', 'Unknown')}")
    
    # 기존 데이터 분석
    total_sub_articles = 0
    amendment_info_count = 0
    amendment_info_size = 0
    
    for article in data.get('articles', []):
        sub_articles = article.get('sub_articles', [])
        total_sub_articles += len(sub_articles)
        
        for sub_article in sub_articles:
            if 'amendment_info' in sub_article:
                amendment_info_count += 1
                amendment_info_size += len(json.dumps(sub_article['amendment_info'], ensure_ascii=False))
    
    print(f"\n=== 기존 데이터 분석 ===")
    print(f"총 sub_articles 수: {total_sub_articles}")
    print(f"amendment_info 포함 항목 수: {amendment_info_count}")
    print(f"amendment_info 총 크기: {amendment_info_size} bytes")
    print(f"평균 amendment_info 크기: {amendment_info_size / max(amendment_info_count, 1):.1f} bytes")
    
    # 새로운 파서로 테스트
    article_parser = ArticleParser()
    text_normalizer = TextNormalizer()
    
    # 첫 번째 조문으로 테스트
    test_article = data.get('articles', [])[0] if data.get('articles') else None
    
    if not test_article:
        print("테스트할 조문을 찾을 수 없습니다.")
        return False
    
    print(f"\n테스트 조문: {test_article.get('article_number')} - {test_article.get('article_title')}")
    
    # 원본 내용으로 파싱 테스트
    original_content = test_article.get('article_content', '')
    normalized_content = text_normalizer.normalize(original_content)
    
    # 새로운 파싱 결과
    parsed_articles = article_parser.parse_articles(normalized_content, '')
    
    if parsed_articles:
        test_parsed_article = parsed_articles[0]
        new_sub_articles = test_parsed_article.get('sub_articles', [])
        
        print(f"\n=== 최적화된 파싱 결과 ===")
        print(f"새로운 sub_articles 수: {len(new_sub_articles)}")
        
        # amendment 필드 분석
        amendment_count = 0
        amendment_size = 0
        
        for sub_article in new_sub_articles:
            if 'amendment' in sub_article:
                amendment_count += 1
                amendment_size += len(json.dumps(sub_article['amendment'], ensure_ascii=False))
        
        print(f"amendment 포함 항목 수: {amendment_count}")
        print(f"amendment 총 크기: {amendment_size} bytes")
        print(f"평균 amendment 크기: {amendment_size / max(amendment_count, 1):.1f} bytes")
        
        # 크기 비교
        original_size = amendment_info_size
        new_size = amendment_size
        reduction = original_size - new_size
        reduction_percent = (reduction / max(original_size, 1)) * 100
        
        print(f"\n=== 크기 비교 ===")
        print(f"기존 크기: {original_size} bytes")
        print(f"최적화된 크기: {new_size} bytes")
        print(f"절약된 크기: {reduction} bytes")
        print(f"절약 비율: {reduction_percent:.1f}%")
        
        # 샘플 데이터 표시
        print(f"\n=== 샘플 데이터 ===")
        for i, sub_article in enumerate(new_sub_articles[:3]):
            sub_type = sub_article.get('type', 'Unknown')
            number = sub_article.get('number', 'Unknown')
            content = sub_article.get('content', '')[:50]
            amendment = sub_article.get('amendment', 'None')
            print(f"{i+1}. {sub_type} {number}: {content}... (amendment: {amendment})")
        
        return True
    else:
        print("파싱 실패")
        return False

if __name__ == "__main__":
    test_amendment_info_optimization()
