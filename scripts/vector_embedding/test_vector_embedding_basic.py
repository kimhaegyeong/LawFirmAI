#!/usr/bin/env python3
"""
벡터 임베딩 기본 테스트
"""

import sys
import os
from pathlib import Path
import time

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from source.data.vector_store import LegalVectorStore

def test_vector_embedding():
    """벡터 임베딩 기본 테스트"""
    print("🔍 벡터 임베딩 기본 테스트 시작...")
    
    try:
        # 벡터 스토어 초기화
        vector_store = LegalVectorStore(
            model_name="jhgan/ko-sroberta-multitask",
            dimension=768,
            index_type="flat"
        )
        
        print("✅ 벡터 스토어 초기화 성공")
        
        # 인덱스 파일 확인
        index_path = Path("data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index")
        faiss_file = index_path.with_suffix('.faiss')
        json_file = index_path.with_suffix('.json')
        
        print(f"📁 인덱스 파일 확인:")
        print(f"   - FAISS 파일: {faiss_file.exists()} ({faiss_file.stat().st_size / (1024*1024):.1f} MB)")
        print(f"   - JSON 파일: {json_file.exists()} ({json_file.stat().st_size / (1024*1024):.1f} MB)")
        
        if faiss_file.exists() and json_file.exists():
            # 인덱스 로드 시도
            try:
                vector_store.load_index(str(index_path))
                stats = vector_store.get_stats()
                
                print(f"✅ 인덱스 로드 성공")
                print(f"   - 총 문서 수: {stats.get('documents_count', 0):,}")
                print(f"   - 인덱스 크기: {stats.get('index_size', 0):,}")
                
                # 간단한 검색 테스트
                test_query = "계약서 해지 조건"
                start_time = time.time()
                results = vector_store.search(test_query, top_k=3)
                search_time = time.time() - start_time
                
                print(f"🔍 검색 테스트:")
                print(f"   - 쿼리: '{test_query}'")
                print(f"   - 결과 수: {len(results)}")
                print(f"   - 검색 시간: {search_time:.3f}초")
                
                if results:
                    best_result = results[0]
                    print(f"   - 최고 점수: {best_result.get('score', 0):.3f}")
                    print(f"   - 문서 ID: {best_result.get('metadata', {}).get('document_id', 'Unknown')}")
                
                return True
                
            except Exception as e:
                print(f"❌ 인덱스 로드 실패: {e}")
                return False
        else:
            print("❌ 인덱스 파일이 존재하지 않습니다.")
            return False
            
    except Exception as e:
        print(f"❌ 벡터 스토어 테스트 실패: {e}")
        return False

def check_file_sizes():
    """파일 크기 확인"""
    print("\n📊 생성된 파일 크기 확인:")
    
    embedding_dir = Path("data/embeddings/ml_enhanced_ko_sroberta")
    
    if embedding_dir.exists():
        for file_path in embedding_dir.iterdir():
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"   - {file_path.name}: {size_mb:.2f} MB")
    else:
        print("❌ 임베딩 디렉토리가 존재하지 않습니다.")

def main():
    """메인 테스트 함수"""
    print("🚀 벡터 임베딩 기본 테스트")
    print("=" * 40)
    
    # 파일 크기 확인
    check_file_sizes()
    
    # 벡터 임베딩 테스트
    success = test_vector_embedding()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 벡터 임베딩 테스트 성공!")
        print("✅ 벡터 임베딩이 정상적으로 생성되고 로드됩니다.")
    else:
        print("⚠️ 벡터 임베딩 테스트 실패")
        print("❌ 추가 점검이 필요합니다.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
