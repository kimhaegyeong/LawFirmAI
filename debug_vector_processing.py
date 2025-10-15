#!/usr/bin/env python3
"""
벡터 임베딩 생성 중 발생하는 'int' object is not iterable 에러 디버깅
"""

import json
import sys
from pathlib import Path

def debug_vector_processing():
    """벡터 처리 과정 디버깅"""
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
    
    # 첫 번째 법률의 첫 번째 조문으로 벡터 처리 시뮬레이션
    if laws and laws[0].get('articles'):
        first_law = laws[0]
        first_article = first_law['articles'][0]
        
        print(f"\nProcessing law: {first_law.get('law_name', 'Unknown')}")
        print(f"Processing article: {first_article.get('article_number', 'Unknown')}")
        
        # 벡터 처리 과정 시뮬레이션
        try:
            # 1. 조문 메타데이터 생성
            article_metadata = {
                'article_number': first_article.get('article_number', ''),
                'article_title': first_article.get('article_title', ''),
                'article_type': 'main',
                'is_supplementary': first_article.get('is_supplementary', False),
                'ml_confidence_score': first_article.get('ml_confidence_score'),
                'parsing_method': first_article.get('parsing_method', 'ml_enhanced'),
                'word_count': first_article.get('word_count', 0),
                'char_count': first_article.get('char_count', 0),
                'sub_articles_count': len(first_article.get('sub_articles', [])) if isinstance(first_article.get('sub_articles'), list) else 0,
                'references_count': len(first_article.get('references', [])) if isinstance(first_article.get('references'), list) else 0
            }
            print("[OK] Article metadata created successfully")
            
            # 2. 텍스트 구성
            text_parts = []
            
            # 조문 번호와 제목
            if article_metadata['article_number']:
                if article_metadata['article_title']:
                    text_parts.append(f"{article_metadata['article_number']}({article_metadata['article_title']})")
                else:
                    text_parts.append(article_metadata['article_number'])
            print("[OK] Article number and title added")
            
            # 조문 내용
            article_content = first_article.get('article_content', '')
            if article_content:
                text_parts.append(article_content)
            print("[OK] Article content added")
            
            # 하위 조문들 - 여기서 에러가 발생할 수 있음
            sub_articles = first_article.get('sub_articles', [])
            if not isinstance(sub_articles, list):
                sub_articles = []
            print(f"[OK] Sub-articles type check passed: {type(sub_articles)}")
            
            for sub_idx, sub_article in enumerate(sub_articles):
                print(f"  Processing sub-article {sub_idx + 1}: {type(sub_article)}")
                if isinstance(sub_article, dict):
                    sub_content = sub_article.get('content', '')
                    if sub_content:
                        text_parts.append(sub_content)
                    print(f"    [OK] Sub-article content added: {len(sub_content)} chars")
                else:
                    print(f"    ERROR: Sub-article is not a dict: {type(sub_article)}")
                    break
            
            # 최종 텍스트
            full_text = ' '.join(text_parts)
            print(f"[OK] Full text created: {len(full_text)} chars")
            
            # 3. 문서 생성
            document_id = f"test_article_{article_metadata['article_number']}"
            document = {
                'id': document_id,
                'text': full_text,
                'metadata': article_metadata,
                'chunks': [{
                    'id': f"{document_id}_chunk_0",
                    'text': full_text,
                    'start_pos': 0,
                    'end_pos': len(full_text),
                    'entities': first_article.get('references', [])
                }]
            }
            print("[OK] Document created successfully")
            
            # 4. 벡터 인덱스 추가 시뮬레이션
            chunks = document.get('chunks', [])
            for chunk in chunks:
                entities = chunk.get('entities', [])
                print(f"[OK] Chunk entities type: {type(entities)}")
                if isinstance(entities, list):
                    print(f"  [OK] Entities list length: {len(entities)}")
                else:
                    print(f"  ERROR: Entities is not a list: {type(entities)}")
            
            print("\n[OK] All processing steps completed successfully!")
            
        except Exception as e:
            print(f"\nERROR during processing: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_vector_processing()
