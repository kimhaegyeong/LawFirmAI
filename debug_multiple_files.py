#!/usr/bin/env python3
"""
벡터 임베딩 생성 중 발생하는 'int' object is not iterable 에러 디버깅
"""

import json
import sys
from pathlib import Path

def debug_multiple_files():
    """여러 파일로 에러 재현 시도"""
    input_dir = Path("data/processed/assembly/law/20251013_ml")
    
    # 첫 번째 파일 로드 (law_page 파일만)
    json_files = [f for f in input_dir.rglob("ml_enhanced_law_page_*.json")]
    if not json_files:
        print("No law page JSON files found!")
        return
    
    print(f"Found {len(json_files)} files to test")
    
    # 처음 5개 파일 테스트
    for i, file_path in enumerate(json_files[:5]):
        print(f"\n=== Testing file {i+1}: {file_path.name} ===")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            
            # 파일 구조 확인
            if isinstance(file_data, dict) and 'laws' in file_data:
                laws = file_data['laws']
            elif isinstance(file_data, list):
                laws = file_data
            else:
                laws = [file_data]
            
            print(f"Total laws in file: {len(laws)}")
            
            # 각 법률의 첫 번째 조문 테스트
            for law_idx, law_data in enumerate(laws[:3]):  # 처음 3개 법률만
                print(f"\n  Law {law_idx + 1}: {law_data.get('law_name', 'Unknown')}")
                
                articles = law_data.get('articles', [])
                if not articles:
                    print("    No articles found")
                    continue
                
                # 첫 번째 조문 테스트
                first_article = articles[0]
                print(f"    Article: {first_article.get('article_number', 'Unknown')}")
                
                # sub_articles와 references 타입 확인
                sub_articles = first_article.get('sub_articles', [])
                references = first_article.get('references', [])
                
                print(f"    sub_articles: {type(sub_articles)} (length: {len(sub_articles) if isinstance(sub_articles, list) else 'N/A'})")
                print(f"    references: {type(references)} (length: {len(references) if isinstance(references, list) else 'N/A'})")
                
                # sub_articles 내부 구조 확인
                if isinstance(sub_articles, list) and sub_articles:
                    for sub_idx, sub_article in enumerate(sub_articles[:2]):  # 처음 2개만
                        print(f"      Sub-article {sub_idx + 1}: {type(sub_article)}")
                        if isinstance(sub_article, dict):
                            for key, value in sub_article.items():
                                if isinstance(value, int):
                                    print(f"        WARNING: {key} is an integer: {value}")
                                elif isinstance(value, list):
                                    print(f"        {key}: list (length: {len(value)})")
                                else:
                                    print(f"        {key}: {type(value)}")
                        else:
                            print(f"        ERROR: Sub-article is not a dict: {type(sub_article)}")
                
                # references 내부 구조 확인
                if isinstance(references, list) and references:
                    for ref_idx, ref in enumerate(references[:2]):  # 처음 2개만
                        print(f"      Reference {ref_idx + 1}: {type(ref)} = {repr(ref)[:50]}")
                        if isinstance(ref, int):
                            print(f"        WARNING: Reference is an integer: {ref}")
                
        except Exception as e:
            print(f"ERROR processing file {file_path.name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_multiple_files()