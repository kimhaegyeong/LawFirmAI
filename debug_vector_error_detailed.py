#!/usr/bin/env python3
"""
벡터 임베딩 생성 중 발생하는 'int' object is not iterable 에러 디버깅
"""

import json
import sys
from pathlib import Path

def debug_data_structure():
    """데이터 구조 디버깅"""
    input_dir = Path("data/processed/assembly/law/20251013_ml")
    
    # 첫 번째 파일 로드 (law_page 파일만)
    json_files = [f for f in input_dir.rglob("ml_enhanced_law_page_*.json")]
    if not json_files:
        print("No law page JSON files found!")
        return
    
    first_file = json_files[0]
    print(f"Debugging file: {first_file}")
    
    with open(first_file, 'r', encoding='utf-8') as f:
        file_data = json.load(f)
    
    # 파일 구조 확인
    if isinstance(file_data, dict) and 'laws' in file_data:
        laws = file_data['laws']
    elif isinstance(file_data, list):
        laws = file_data
    else:
        laws = [file_data]
    
    print(f"Total laws in file: {len(laws)}")
    
    # 모든 법률의 모든 조문 확인
    for law_idx, law_data in enumerate(laws):
        print(f"\nLaw {law_idx + 1}: {law_data.get('law_name', 'Unknown')}")
        
        articles = law_data.get('articles', [])
        print(f"  Articles count: {len(articles)}")
        
        for art_idx, article in enumerate(articles):
            print(f"  Article {art_idx + 1}: {article.get('article_number', 'Unknown')}")
            
            # 모든 필드의 타입 확인
            for key, value in article.items():
                print(f"    {key}: {type(value)}")
                
                # 특히 sub_articles와 references 상세 확인
                if key in ['sub_articles', 'references']:
                    if isinstance(value, list):
                        print(f"      List length: {len(value)}")
                        if value:
                            print(f"      First item type: {type(value[0])}")
                    elif isinstance(value, int):
                        print(f"      WARNING: {key} is an integer: {value}")
                    else:
                        print(f"      Other type: {type(value)}")
            
            # sub_articles 내부 구조 확인
            sub_articles = article.get('sub_articles', [])
            if isinstance(sub_articles, list) and sub_articles:
                print(f"    Sub-articles details:")
                for sub_idx, sub_article in enumerate(sub_articles):
                    print(f"      Sub-article {sub_idx + 1}: {type(sub_article)}")
                    if isinstance(sub_article, dict):
                        for sub_key, sub_value in sub_article.items():
                            print(f"        {sub_key}: {type(sub_value)}")
                            if isinstance(sub_value, int):
                                print(f"          WARNING: {sub_key} is an integer: {sub_value}")

if __name__ == "__main__":
    debug_data_structure()

