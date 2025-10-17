#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
벡터 검색 테스트 스크립트
"""

def test_vector_search():
    """벡터 검색 테스트"""
    try:
        from source.data.vector_store import LegalVectorStore
        print("✅ 벡터 스토어 임포트 성공")
        
        # 벡터 스토어 초기화
        vector_store = LegalVectorStore()
        print("✅ 벡터 스토어 초기화 성공")
        
        # 벡터 인덱스 로드
        try:
            vector_store.load_index()
            print("✅ 벡터 인덱스 로드 성공")
        except Exception as e:
            print(f"⚠️ 벡터 인덱스 로드 실패: {e}")
            # 기본 경로로 다시 시도
            try:
                vector_store.load_index("data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index.faiss")
                print("✅ 기본 경로로 벡터 인덱스 로드 성공")
            except Exception as e2:
                print(f"❌ 벡터 인덱스 로드 완전 실패: {e2}")
                return False
        
        # 벡터 검색 테스트
        test_queries = [
            "손해배상 청구 방법",
            "계약 해제 조건",
            "불법행위 책임"
        ]
        
        all_success = True
        for i, query in enumerate(test_queries, 1):
            print(f"\n검색 테스트 {i}: {query}")
            try:
                results = vector_store.search(query, top_k=3)
                print(f"✅ 검색 성공: {len(results)}개 결과")
                
                if len(results) == 0:
                    print("⚠️ 검색 결과가 없습니다.")
                    all_success = False
                else:
                    print(f"   첫 번째 결과 점수: {results[0].get('score', 'N/A')}")
                    
            except Exception as e:
                print(f"❌ 검색 실패: {e}")
                all_success = False
        
        # 벡터 스토어 통계 확인
        try:
            stats = vector_store.get_stats()
            print(f"\n✅ 벡터 스토어 통계: {stats}")
        except Exception as e:
            print(f"⚠️ 통계 조회 실패: {e}")
        
        return all_success
        
    except Exception as e:
        print(f"❌ 벡터 검색 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    print("벡터 검색 테스트 시작")
    print("=" * 50)
    
    success = test_vector_search()
    
    print("=" * 50)
    if success:
        print("🎉 벡터 검색 테스트 성공!")
    else:
        print("💥 벡터 검색 테스트 실패!")
