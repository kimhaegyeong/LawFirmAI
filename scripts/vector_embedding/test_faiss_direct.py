#!/usr/bin/env python3
"""
벡터 임베딩 직접 테스트
"""

import sys
import os
from pathlib import Path
import json
import faiss
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_direct_faiss():
    """FAISS 인덱스 직접 테스트"""
    print("FAISS 인덱스 직접 테스트...")
    
    try:
        # FAISS 인덱스 직접 로드
        faiss_file = Path("data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index.faiss")
        json_file = Path("data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index.json")
        
        if not faiss_file.exists():
            print(f"FAISS 파일이 존재하지 않습니다: {faiss_file}")
            return False
        
        if not json_file.exists():
            print(f"JSON 파일이 존재하지 않습니다: {json_file}")
            return False
        
        # FAISS 인덱스 로드
        index = faiss.read_index(str(faiss_file))
        print(f"FAISS 인덱스 로드 성공")
        print(f"   - 인덱스 크기: {index.ntotal:,}")
        print(f"   - 벡터 차원: {index.d}")
        
        # 메타데이터 로드
        with open(json_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"메타데이터 로드 성공")
        print(f"   - 모델명: {metadata.get('model_name', 'Unknown')}")
        print(f"   - 차원: {metadata.get('dimension', 0)}")
        print(f"   - 문서 수: {metadata.get('document_count', 0):,}")
        print(f"   - 생성일: {metadata.get('created_at', 'Unknown')}")
        
        # 문서 메타데이터 확인
        doc_metadata = metadata.get('document_metadata', [])
        print(f"   - 메타데이터 항목 수: {len(doc_metadata):,}")
        
        if len(doc_metadata) > 0:
            sample_metadata = doc_metadata[0]
            print(f"   - 샘플 메타데이터:")
            for key, value in sample_metadata.items():
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                print(f"     {key}: {value}")
        
        # 간단한 검색 테스트 (랜덤 벡터)
        if index.ntotal > 0:
            print(f"\n검색 테스트...")
            
            # 랜덤 쿼리 벡터 생성
            query_vector = np.random.random((1, index.d)).astype('float32')
            faiss.normalize_L2(query_vector)
            
            # 검색 실행
            scores, indices = index.search(query_vector, 5)
            
            print(f"   - 검색 결과 수: {len(indices[0])}")
            print(f"   - 최고 점수: {scores[0][0]:.3f}")
            print(f"   - 검색된 인덱스: {indices[0][:3]}")
            
            # 메타데이터와 매칭
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(doc_metadata):
                    doc_info = doc_metadata[idx]
                    print(f"   {i+1}. 점수: {score:.3f}, 법률명: {doc_info.get('law_name', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"FAISS 직접 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("FAISS 인덱스 직접 테스트")
    print("=" * 40)
    
    success = test_direct_faiss()
    
    print("\n" + "=" * 40)
    if success:
        print("FAISS 인덱스 테스트 성공!")
        print("벡터 임베딩이 정상적으로 생성되었습니다.")
        print("FAISS 인덱스와 메타데이터가 올바르게 저장되었습니다.")
    else:
        print("FAISS 인덱스 테스트 실패")
        print("추가 점검이 필요합니다.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)