#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 임베딩 검증 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from source.data.vector_store import LegalVectorStore
from source.data.database import DatabaseManager

def main():
    print("🔍 LawFirmAI 임베딩 시스템 최종 검증")
    print("=" * 40)
    
    # 1. 벡터 스토어 테스트
    print("1. 벡터 스토어 로드 테스트...")
    try:
        vector_store = LegalVectorStore()
        success = vector_store.load_index("data/embeddings/legal_vector_index.faiss")
        
        if success:
            stats = vector_store.get_stats()
            print(f"   ✅ 벡터 스토어 로드 성공")
            print(f"   📊 문서 수: {stats['documents_count']}개")
            print(f"   🤖 모델: {stats['model_name']}")
            
            # 검색 테스트
            print("\n2. 벡터 검색 테스트...")
            results = vector_store.search("계약서", top_k=3)
            print(f"   ✅ 검색 성공: {len(results)}개 결과")
            
            for i, result in enumerate(results, 1):
                score = result['score']
                title = result['metadata'].get('title', 'No title')
                doc_type = result['metadata'].get('document_type', 'unknown')
                print(f"   {i}. [{doc_type}] {title} (점수: {score:.3f})")
        else:
            print("   ❌ 벡터 스토어 로드 실패")
            return False
            
    except Exception as e:
        print(f"   ❌ 벡터 스토어 오류: {e}")
        return False
    
    # 2. 데이터베이스 테스트
    print("\n3. 데이터베이스 테스트...")
    try:
        db_manager = DatabaseManager()
        
        # 문서 수 확인
        query = "SELECT COUNT(*) as count FROM documents"
        result = db_manager.execute_query(query)
        doc_count = result[0]['count'] if result else 0
        
        print(f"   ✅ 데이터베이스 연결 성공")
        print(f"   📊 총 문서 수: {doc_count}개")
        
        # 정확 매칭 검색 테스트
        exact_results, total_count = db_manager.search_exact("계약서", limit=3)
        print(f"   ✅ 정확 매칭 검색 성공: {total_count}개 결과")
        
        for i, result in enumerate(exact_results, 1):
            print(f"   {i}. {result['title']} ({result['document_type']})")
            
    except Exception as e:
        print(f"   ❌ 데이터베이스 오류: {e}")
        return False
    
    print(f"\n🎉 모든 테스트 통과!")
    print(f"✅ SQLite 데이터: {doc_count}개 문서")
    print(f"✅ FAISS 임베딩: {stats['documents_count']}개 벡터")
    print(f"✅ 하이브리드 검색 시스템 준비 완료!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ 검증 실패")
        sys.exit(1)
