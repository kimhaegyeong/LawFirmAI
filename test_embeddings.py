#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
임베딩 검증 및 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from source.data.vector_store import LegalVectorStore
from source.data.database import DatabaseManager

def test_vector_search():
    """벡터 검색 테스트"""
    print("🔍 벡터 검색 테스트")
    print("=" * 30)
    
    try:
        # 벡터 스토어 로드
        vector_store = LegalVectorStore()
        success = vector_store.load_index("data/embeddings/legal_vector_index")
        
        if not success:
            print("❌ 벡터 인덱스 로드 실패")
            return False
        
        # 통계 정보 출력
        stats = vector_store.get_stats()
        print(f"📊 벡터 스토어 통계:")
        print(f"  - 문서 수: {stats['documents_count']}개")
        print(f"  - 모델: {stats['model_name']}")
        print(f"  - 차원: {stats['embedding_dimension']}")
        print(f"  - 인덱스 타입: {stats['index_type']}")
        
        # 테스트 쿼리들
        test_queries = [
            "계약서 작성",
            "민법 제1조",
            "판례 검색",
            "법률 해석",
            "손해배상"
        ]
        
        print(f"\n🔎 검색 테스트:")
        for query in test_queries:
            print(f"\n쿼리: '{query}'")
            results = vector_store.search(query, top_k=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    score = result['score']
                    text = result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
                    doc_type = result['metadata'].get('document_type', 'unknown')
                    print(f"  {i}. [{doc_type}] 점수: {score:.3f}")
                    print(f"     내용: {text}")
            else:
                print("  검색 결과 없음")
        
        return True
        
    except Exception as e:
        print(f"❌ 벡터 검색 테스트 실패: {e}")
        return False

def test_hybrid_search():
    """하이브리드 검색 테스트"""
    print("\n🔄 하이브리드 검색 테스트")
    print("=" * 30)
    
    try:
        # 데이터베이스와 벡터 스토어 초기화
        db_manager = DatabaseManager()
        vector_store = LegalVectorStore()
        vector_store.load_index("data/embeddings/legal_vector_index")
        
        test_query = "계약서"
        
        # 1. 벡터 검색
        print(f"1. 벡터 검색 결과:")
        vector_results = vector_store.search(test_query, top_k=3)
        for i, result in enumerate(vector_results, 1):
            print(f"   {i}. 점수: {result['score']:.3f} - {result['metadata'].get('title', 'No title')}")
        
        # 2. 정확 매칭 검색
        print(f"\n2. 정확 매칭 검색 결과:")
        exact_results, total_count = db_manager.search_exact(test_query, limit=3)
        for i, result in enumerate(exact_results, 1):
            print(f"   {i}. {result['title']} ({result['document_type']})")
        
        print(f"\n총 {total_count}개 정확 매칭 결과")
        
        return True
        
    except Exception as e:
        print(f"❌ 하이브리드 검색 테스트 실패: {e}")
        return False

def test_document_retrieval():
    """문서 조회 테스트"""
    print("\n📄 문서 조회 테스트")
    print("=" * 30)
    
    try:
        db_manager = DatabaseManager()
        
        # documents 테이블에서 샘플 문서 조회
        query = "SELECT id, document_type, title FROM documents LIMIT 5"
        documents = db_manager.execute_query(query)
        
        print("샘플 문서들:")
        for doc in documents:
            print(f"  - ID: {doc['id']}, Type: {doc['document_type']}, Title: {doc['title']}")
        
        # 특정 문서 상세 조회 테스트
        if documents:
            first_doc = documents[0]
            detailed_doc = db_manager.get_document_by_id(first_doc['id'])
            
            if detailed_doc:
                print(f"\n상세 문서 조회 테스트 (ID: {first_doc['id']}):")
                print(f"  - 제목: {detailed_doc['title']}")
                print(f"  - 타입: {detailed_doc['document_type']}")
                print(f"  - 내용 길이: {len(detailed_doc['content'])}자")
        
        return True
        
    except Exception as e:
        print(f"❌ 문서 조회 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    print("🧪 LawFirmAI 임베딩 및 검색 시스템 검증")
    print("=" * 50)
    
    # 각 테스트 실행
    tests = [
        ("벡터 검색", test_vector_search),
        ("하이브리드 검색", test_hybrid_search),
        ("문서 조회", test_document_retrieval)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    # 결과 요약
    print(f"\n{'='*50}")
    print("📋 테스트 결과 요약:")
    for test_name, success in results:
        status = "✅ 통과" if success else "❌ 실패"
        print(f"  - {test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print(f"\n🎉 모든 테스트 통과! 시스템이 정상적으로 작동합니다.")
    else:
        print(f"\n⚠️ 일부 테스트 실패. 로그를 확인해주세요.")
