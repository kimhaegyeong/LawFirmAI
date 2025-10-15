#!/usr/bin/env python3
"""
벡터 임베딩 생성 중 발생하는 'int' object is not iterable 에러 디버깅
"""

import json
import sys
from pathlib import Path

def debug_vector_store_processing():
    """벡터 스토어 처리 과정 디버깅"""
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
            
            # 4. 벡터 인덱스 추가 시뮬레이션 - 실제 벡터 스토어 처리 과정
            chunks = document.get('chunks', [])
            texts = []
            metadatas = []
            
            for chunk in chunks:
                texts.append(chunk.get('text', ''))
                metadata = {
                    'document_id': document.get('id', ''),
                    'document_type': 'law_article',
                    'chunk_id': chunk.get('id', ''),
                    'chunk_start': chunk.get('start_pos', 0),
                    'chunk_end': chunk.get('end_pos', 0),
                    'law_name': document.get('metadata', {}).get('law_name', ''),
                    'category': document.get('metadata', {}).get('category', ''),
                    'entities': chunk.get('entities', []) if isinstance(chunk.get('entities'), list) else [],
                    # ML 강화 메타데이터 추가
                    'article_number': document.get('metadata', {}).get('article_number', ''),
                    'article_title': document.get('metadata', {}).get('article_title', ''),
                    'article_type': document.get('metadata', {}).get('article_type', ''),
                    'is_supplementary': document.get('metadata', {}).get('is_supplementary', False),
                    'ml_confidence_score': document.get('metadata', {}).get('ml_confidence_score'),
                    'parsing_method': document.get('metadata', {}).get('parsing_method', 'ml_enhanced'),
                    'parsing_quality_score': document.get('metadata', {}).get('parsing_quality_score', 0.0),
                    'ml_enhanced': document.get('metadata', {}).get('ml_enhanced', True),
                    'word_count': document.get('metadata', {}).get('word_count', 0),
                    'char_count': document.get('metadata', {}).get('char_count', 0)
                }
                metadatas.append(metadata)
                
                # 메타데이터 필드 검증
                print(f"[OK] Chunk metadata created")
                for key, value in metadata.items():
                    print(f"  {key}: {type(value)} = {repr(value)[:50]}")
                    if isinstance(value, int):
                        print(f"    WARNING: {key} is an integer: {value}")
            
            print(f"[OK] Texts list created: {len(texts)} items")
            print(f"[OK] Metadatas list created: {len(metadatas)} items")
            
            # 5. 벡터 스토어에 전달할 데이터 검증
            print("\n[OK] All processing steps completed successfully!")
            print("Data ready for vector store:")
            print(f"  Texts: {len(texts)} items")
            print(f"  Metadatas: {len(metadatas)} items")
            
        except Exception as e:
            print(f"\nERROR during processing: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_vector_store_processing()

