#!/usr/bin/env python3
"""
간단한 임베딩 테스트 스크립트
"""

import json
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from source.data.vector_store import LegalVectorStore

def test_simple_embedding():
    """간단한 임베딩 테스트"""
    print("간단한 임베딩 테스트 시작...")
    
    # 벡터 스토어 초기화
    vector_store = LegalVectorStore(model_name="jhgan/ko-sroberta-multitask")
    
    # 테스트 문서
    test_documents = [
        {
            'id': 'test_1',
            'text': '제1조(목적) 이 법은 민사에 관한 기본사항을 정함을 목적으로 한다.',
            'metadata': {'law_name': '민법', 'article_number': '제1조'}
        },
        {
            'id': 'test_2', 
            'text': '제2조(적용범위) 이 법은 대한민국 내에서 발생하는 민사에 적용한다.',
            'metadata': {'law_name': '민법', 'article_number': '제2조'}
        }
    ]
    
    try:
        # 임베딩 생성
        print("임베딩 생성 중...")
        texts = [doc['text'] for doc in test_documents]
        metadatas = [doc['metadata'] for doc in test_documents]
        vector_store.add_documents(texts, metadatas)
        print("임베딩 생성 완료!")
        
        # 검색 테스트
        print("검색 테스트 중...")
        results = vector_store.search("민사 기본사항", top_k=2)
        print(f"검색 결과: {len(results)}개")
        
        for i, result in enumerate(results):
            print(f"결과 {i+1}: {result['metadata']['article_number']} - {result['text'][:50]}...")
        
        print("테스트 완료!")
        return True
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_embedding()
    if success:
        print("✅ 간단한 임베딩 테스트 성공!")
    else:
        print("❌ 간단한 임베딩 테스트 실패!")
