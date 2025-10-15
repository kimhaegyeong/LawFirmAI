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
    
    # 첫 번째 법률의 첫 번째 조문 확인
    if laws and laws[0].get('articles'):
        first_law = laws[0]
        first_article = first_law['articles'][0]
        
        print(f"\nFirst law: {first_law.get('law_name', 'Unknown')}")
        print(f"First article: {first_article.get('article_number', 'Unknown')}")
        
        # 모든 필드의 타입 확인
        for key, value in first_article.items():
            print(f"  {key}: {type(value)} = {repr(value)[:100]}")
            
            # 특히 sub_articles와 references 상세 확인
            if key in ['sub_articles', 'references']:
                if isinstance(value, list):
                    print(f"    List length: {len(value)}")
                    if value:
                        print(f"    First item type: {type(value[0])}")
                        print(f"    First item: {repr(value[0])[:50]}")
                elif isinstance(value, int):
                    print(f"    WARNING: {key} is an integer: {value}")
                else:
                    print(f"    Other type: {type(value)}")

if __name__ == "__main__":
    debug_data_structure()
