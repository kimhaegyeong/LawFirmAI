#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최적화된 벡터 인덱스 생성
"""

import sys
sys.path.append('source')
from source.data.vector_store import LegalVectorStore
import time
import json
from pathlib import Path

def create_optimized_index():
    """최적화된 벡터 인덱스 생성"""
    print("=== 최적화된 벡터 인덱스 생성 ===")
    
    # 기존 인덱스 로드
    print("기존 인덱스 로드 중...")
    vector_store = LegalVectorStore(
        model_name='jhgan/ko-sroberta-multitask',
        dimension=768,
        index_type='flat'
    )
    
    if not vector_store.load_index('data/embeddings/ml_enhanced_ko_sroberta_precedents'):
        print("기존 인덱스 로드 실패")
        return False
    
    print(f"기존 인덱스 크기: {vector_store.index.ntotal:,}")
    
    # IVF 인덱스로 변환
    print("IVF 인덱스로 변환 중...")
    
    # IVF 인덱스 생성
    import faiss
    
    # nlist 값 계산 (데이터 크기에 따라)
    n_vectors = vector_store.index.ntotal
    nlist = min(1000, max(100, n_vectors // 100))  # 100-1000 사이의 값
    
    print(f"IVF 파라미터 - nlist: {nlist}")
    
    # IVF 인덱스 생성
    quantizer = faiss.IndexFlatIP(768)
    ivf_index = faiss.IndexIVFFlat(quantizer, 768, nlist)
    
    # 기존 벡터 데이터 추출
    print("벡터 데이터 추출 중...")
    vectors = vector_store.index.reconstruct_n(0, n_vectors)
    
    # IVF 인덱스 훈련
    print("IVF 인덱스 훈련 중...")
    ivf_index.train(vectors)
    
    # 벡터 추가
    print("벡터 추가 중...")
    ivf_index.add(vectors)
    
    # 새로운 벡터 스토어 생성
    print("최적화된 벡터 스토어 생성 중...")
    optimized_store = LegalVectorStore(
        model_name='jhgan/ko-sroberta-multitask',
        dimension=768,
        index_type='ivf'
    )
    
    # 인덱스 설정
    optimized_store.index = ivf_index
    optimized_store.document_texts = vector_store.document_texts.copy()
    optimized_store.document_metadata = vector_store.document_metadata.copy()
    optimized_store._index_loaded = True
    
    # 최적화된 인덱스 저장
    output_path = 'data/embeddings/optimized_ko_sroberta_precedents'
    print(f"최적화된 인덱스 저장 중: {output_path}")
    
    optimized_store.save_index(output_path)
    
    print("최적화된 인덱스 생성 완료!")
    
    # 성능 비교 테스트
    print("\n=== 성능 비교 테스트 ===")
    test_queries = [
        "손해배상 청구",
        "계약 해지",
        "특허 침해",
        "이혼 소송",
        "형사 처벌"
    ]
    
    # 기존 인덱스 성능
    print("기존 Flat 인덱스 성능:")
    flat_times = []
    for query in test_queries:
        start_time = time.time()
        results = vector_store.search(query, top_k=10)
        search_time = time.time() - start_time
        flat_times.append(search_time)
        print(f"  '{query}': {search_time:.3f}초")
    
    # 최적화된 인덱스 성능
    print("\n최적화된 IVF 인덱스 성능:")
    ivf_times = []
    for query in test_queries:
        start_time = time.time()
        results = optimized_store.search(query, top_k=10)
        search_time = time.time() - start_time
        ivf_times.append(search_time)
        print(f"  '{query}': {search_time:.3f}초")
    
    # 성능 개선율 계산
    avg_flat_time = sum(flat_times) / len(flat_times)
    avg_ivf_time = sum(ivf_times) / len(ivf_times)
    improvement = ((avg_flat_time - avg_ivf_time) / avg_flat_time) * 100
    
    print(f"\n성능 개선 결과:")
    print(f"  기존 평균 검색 시간: {avg_flat_time:.3f}초")
    print(f"  최적화된 평균 검색 시간: {avg_ivf_time:.3f}초")
    print(f"  성능 개선율: {improvement:.1f}%")
    
    return True

def create_quantized_index():
    """양자화된 인덱스 생성"""
    print("\n=== 양자화된 인덱스 생성 ===")
    
    # 기존 인덱스 로드
    vector_store = LegalVectorStore(
        model_name='jhgan/ko-sroberta-multitask',
        dimension=768,
        index_type='flat'
    )
    
    if not vector_store.load_index('data/embeddings/ml_enhanced_ko_sroberta_precedents'):
        print("기존 인덱스 로드 실패")
        return False
    
    # PQ 양자화 인덱스 생성
    import faiss
    
    print("PQ 양자화 인덱스 생성 중...")
    
    # PQ 파라미터 설정
    m = 64  # 서브벡터 수
    nbits = 8  # 각 서브벡터당 비트 수
    
    pq_index = faiss.IndexPQ(768, m, nbits)
    
    # 벡터 데이터 추출
    n_vectors = vector_store.index.ntotal
    vectors = vector_store.index.reconstruct_n(0, n_vectors)
    
    # PQ 인덱스 훈련 및 벡터 추가
    print("PQ 인덱스 훈련 중...")
    pq_index.train(vectors)
    
    print("벡터 추가 중...")
    pq_index.add(vectors)
    
    # 양자화된 벡터 스토어 생성
    quantized_store = LegalVectorStore(
        model_name='jhgan/ko-sroberta-multitask',
        dimension=768,
        index_type='pq'
    )
    
    quantized_store.index = pq_index
    quantized_store.document_texts = vector_store.document_texts.copy()
    quantized_store.document_metadata = vector_store.document_metadata.copy()
    quantized_store._index_loaded = True
    
    # 양자화된 인덱스 저장
    output_path = 'data/embeddings/quantized_ko_sroberta_precedents'
    print(f"양자화된 인덱스 저장 중: {output_path}")
    
    quantized_store.save_index(output_path)
    
    print("양자화된 인덱스 생성 완료!")
    
    # 메모리 사용량 비교
    print("\n=== 메모리 사용량 비교 ===")
    
    # 기존 인덱스 메모리 사용량
    flat_memory = vector_store.get_memory_usage()
    print(f"기존 Flat 인덱스 메모리: {flat_memory.get('total_memory_mb', 0):.1f}MB")
    
    # 양자화된 인덱스 메모리 사용량
    pq_memory = quantized_store.get_memory_usage()
    print(f"양자화된 PQ 인덱스 메모리: {pq_memory.get('total_memory_mb', 0):.1f}MB")
    
    memory_reduction = ((flat_memory.get('total_memory_mb', 0) - pq_memory.get('total_memory_mb', 0)) / 
                       flat_memory.get('total_memory_mb', 1)) * 100
    print(f"메모리 절약율: {memory_reduction:.1f}%")
    
    return True

def main():
    """메인 함수"""
    print("LawFirmAI 벡터 인덱스 최적화")
    print("=" * 50)
    
    # 1. IVF 인덱스 생성
    if create_optimized_index():
        print("\n✅ IVF 인덱스 생성 완료")
    else:
        print("\n❌ IVF 인덱스 생성 실패")
    
    # 2. 양자화된 인덱스 생성
    if create_quantized_index():
        print("\n✅ 양자화된 인덱스 생성 완료")
    else:
        print("\n❌ 양자화된 인덱스 생성 실패")
    
    print("\n=== 최적화 완료 ===")
    print("생성된 인덱스:")
    print("  - data/embeddings/optimized_ko_sroberta_precedents (IVF)")
    print("  - data/embeddings/quantized_ko_sroberta_precedents (PQ)")

if __name__ == "__main__":
    main()
