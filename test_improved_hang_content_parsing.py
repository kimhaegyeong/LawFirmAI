#!/usr/bin/env python3
"""
개선된 항(項) content 파싱 테스트 스크립트
"""

import json
import sys
from pathlib import Path

# Adjust the path to import from scripts/assembly/parsers
sys.path.append(str(Path(__file__).parent / 'scripts' / 'assembly'))
sys.path.append(str(Path(__file__).parent / 'scripts' / 'assembly' / 'parsers'))

from parsers.article_parser import ArticleParser
from parsers.text_normalizer import TextNormalizer

def test_improved_hang_content_parsing():
    """개선된 항(項) content 파싱을 테스트합니다."""
    
    # 문제가 있던 파일 테스트
    file_path = Path("data/processed/assembly/law/2025101201_final_fixed/20251012/3ㆍ1운동_및_대한민국임시정부_수립_100주년_기념사업추진위원회의_설치_및_운영에_관한_규_assembly_law_2364.json")
    
    if not file_path.exists():
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=== 개선된 항(項) content 파싱 테스트 ===")
    print(f"법률명: {data.get('law_name', 'Unknown')}")
    
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
    print(f"원본 내용 길이: {len(original_content)}")
    
    # 정규화된 내용으로 파싱
    normalized_content = text_normalizer.normalize(original_content)
    print(f"정규화된 내용 길이: {len(normalized_content)}")
    
    # 새로운 파싱 결과
    parsed_articles = article_parser.parse_articles(normalized_content, '')
    
    print(f"\n=== 파싱 디버깅 ===")
    print(f"파싱된 조문 수: {len(parsed_articles) if parsed_articles else 0}")
    
    if parsed_articles:
        test_parsed_article = parsed_articles[0]
        new_sub_articles = test_parsed_article.get('sub_articles', [])
        
        print(f"\n=== 개선된 파싱 결과 ===")
        print(f"새로운 sub_articles 수: {len(new_sub_articles)}")
        
        # 기존 결과와 비교
        original_sub_articles = test_article.get('sub_articles', [])
        print(f"기존 sub_articles 수: {len(original_sub_articles)}")
        
        # 항(項) 항목 분석
        hang_items = [item for item in new_sub_articles if item.get('type') == '항']
        print(f"항(項) 항목 수: {len(hang_items)}")
        
        print(f"\n=== 항(項) content 분석 ===")
        for i, item in enumerate(hang_items[:3]):  # 처음 3개만 표시
            number = item.get('number', 'Unknown')
            content = item.get('content', '')
            print(f"{i+1}. 항 {number}: {content[:100]}...")
            
            # 호(號) 항목이 포함되어 있는지 확인
            if re.search(r'\d+\.', content):
                print(f"   [WARNING] 호(號) 항목이 포함되어 있습니다!")
            else:
                print(f"   [OK] 호(號) 항목이 제거되었습니다.")
        
        # 호(號) 항목 분석
        ho_items = [item for item in new_sub_articles if item.get('type') == '호']
        print(f"\n호(號) 항목 수: {len(ho_items)}")
        
        if ho_items:
            print(f"\n=== 호(號) 항목들 ===")
            for i, item in enumerate(ho_items[:5]):  # 처음 5개만 표시
                number = item.get('number', 'Unknown')
                content = item.get('content', '')
                print(f"{i+1}. 호 {number}: {content[:50]}...")
        
        return True
    else:
        print("파싱 실패")
        return False

if __name__ == "__main__":
    import re
    test_improved_hang_content_parsing()
