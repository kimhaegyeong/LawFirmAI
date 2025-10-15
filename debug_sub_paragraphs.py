#!/usr/bin/env python3
"""
벡터 임베딩 생성 중 발생하는 'int' object is not iterable 에러 디버깅
"""

import json
import sys
import os
from pathlib import Path

# Windows 콘솔에서 UTF-8 인코딩 설정
if os.name == 'nt':  # Windows
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

def debug_sub_paragraphs():
    """sub_paragraphs 구조 디버깅"""
    input_dir = Path("data/processed/assembly/law/20251013_ml")
    
    # 첫 번째 파일 로드 (law_page 파일만)
    json_files = [f for f in input_dir.rglob("ml_enhanced_law_page_*.json")]
    if not json_files:
        print("No law page JSON files found!")
        return
    
    print(f"Found {len(json_files)} files")
    
    # 처음 3개 파일에서 sub_paragraphs가 있는 조문 찾기
    for file_idx, file_path in enumerate(json_files[:3]):
        print(f"\n=== File {file_idx + 1}: {file_path.name} ===")
        
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
            
            # sub_paragraphs가 있는 조문 찾기
            for law_idx, law_data in enumerate(laws):
                articles = law_data.get('articles', [])
                
                for art_idx, article in enumerate(articles):
                    sub_articles = article.get('sub_articles', [])
                    
                    if isinstance(sub_articles, list) and sub_articles:
                        for sub_idx, sub_article in enumerate(sub_articles):
                            if isinstance(sub_article, dict):
                                sub_paragraphs = sub_article.get('sub_paragraphs', [])
                                
                                if isinstance(sub_paragraphs, list) and sub_paragraphs:
                                    print(f"\n  Law {law_idx + 1}, Article {art_idx + 1}, Sub-article {sub_idx + 1}")
                                    print(f"    Sub-paragraphs count: {len(sub_paragraphs)}")
                                    
                                    # 각 sub_paragraph 상세 확인
                                    for para_idx, sub_paragraph in enumerate(sub_paragraphs):
                                        print(f"      Sub-paragraph {para_idx + 1}: {type(sub_paragraph)}")
                                        
                                        if isinstance(sub_paragraph, dict):
                                            for key, value in sub_paragraph.items():
                                                print(f"        {key}: {type(value)} = {repr(value)[:50]}")
                                                
                                                # 특히 list 타입인 필드들 확인
                                                if isinstance(value, list):
                                                    print(f"          List length: {len(value)}")
                                                    if value:
                                                        print(f"          First item type: {type(value[0])}")
                                                        if isinstance(value[0], int):
                                                            print(f"          WARNING: First item is an integer: {value[0]}")
                                                
                                                # integer 타입인 필드들 확인
                                                elif isinstance(value, int):
                                                    print(f"          WARNING: {key} is an integer: {value}")
                                        
                                        elif isinstance(sub_paragraph, int):
                                            print(f"        ERROR: Sub-paragraph is an integer: {sub_paragraph}")
                                        
                                        else:
                                            print(f"        WARNING: Sub-paragraph is not a dict: {type(sub_paragraph)}")
                                    
                                    # 처음 2개만 확인
                                    if para_idx >= 1:
                                        break
                                
                                # 처음 3개 sub_article만 확인
                                if sub_idx >= 2:
                                    break
                    
                    # 처음 3개 조문만 확인
                    if art_idx >= 2:
                        break
                
                # 처음 2개 법률만 확인
                if law_idx >= 1:
                    break
                    
        except Exception as e:
            print(f"ERROR processing file {file_path.name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_sub_paragraphs()
