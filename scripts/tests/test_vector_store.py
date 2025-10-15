#!/usr/bin/env python3
"""
벡터 임베딩 생성 중 발생하는 'int' object is not iterable 에러 디버깅
"""

import json
import sys
from pathlib import Path

# Add source to path
sys.path.append(str(Path(__file__).parent.parent.parent / "source"))

from data.vector_store import LegalVectorStore

def test_vector_store():
    """벡터 스토어 테스트"""
    input_dir = Path("data/processed/assembly/law/20251013_ml")
    
    # 첫 번째 파일 로드 (law_page 파일만)
    json_files = [f for f in input_dir.rglob("ml_enhanced_law_page_*.json")]
    if not json_files:
        print("No law page JSON files found!")
        return
    
    first_file = json_files[0]
    print(f"Testing with file: {first_file}")
    
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
    
    # 벡터 스토어 초기화
    try:
        vector_store = LegalVectorStore()
        print("[OK] Vector store initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize vector store: {e}")
        return
    
    # 첫 번째 법률의 첫 번째 조문으로 테스트
    if laws and laws[0].get('articles'):
        first_law = laws[0]
        first_article = first_law['articles'][0]
        
        print(f"\nTesting with law: {first_law.get('law_name', 'Unknown')}")
        print(f"Testing with article: {first_article.get('article_number', 'Unknown')}")
        
        try:
            # 간단한 텍스트와 메타데이터 생성
            text = first_article.get('article_content', '')
            metadata = {
                'document_id': f"test_{first_article.get('article_number', 'unknown')}",
                'article_number': first_article.get('article_number', ''),
                'article_title': first_article.get('article_title', ''),
                'entities': first_article.get('references', []) if isinstance(first_article.get('references'), list) else []
            }
            
            print(f"Text length: {len(text)}")
            print(f"Metadata keys: {list(metadata.keys())}")
            
            # 벡터 스토어에 추가
            success = vector_store.add_documents([text], [metadata])
            
            if success:
                print("[OK] Document added to vector store successfully")
            else:
                print("[ERROR] Failed to add document to vector store")
                
        except Exception as e:
            print(f"[ERROR] Error during vector store processing: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_vector_store()

