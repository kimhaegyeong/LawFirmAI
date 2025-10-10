#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 벡터 데이터베이스 재구축 스크립트
"""

import os
import sys
import json
import numpy as np
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Required library not available: {e}")
    sys.exit(1)

def rebuild_improved_vector_database():
    """개선된 벡터 데이터베이스 재구축"""
    print("Rebuilding improved vector database...")
    
    # 개선된 메타데이터 로드
    with open('data/embeddings/metadata_improved.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total documents: {len(data)}")
    
    # Sentence-BERT 모델 로드
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    
    # 텍스트 추출
    texts = [doc['text'] for doc in data]
    print(f"Generating embeddings for {len(texts)} documents...")
    
    # 배치 단위로 임베딩 생성
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=True)
        all_embeddings.append(batch_embeddings)
        print(f"Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} completed")
    
    # 모든 임베딩 결합
    embeddings = np.vstack(all_embeddings)
    print(f"Total embeddings generated: {embeddings.shape}")
    
    # FAISS 인덱스 생성
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings.astype('float32'))
    
    # 파일 저장
    print("Saving files...")
    
    # FAISS 인덱스 저장
    faiss.write_index(faiss_index, "data/embeddings/faiss_index_improved.bin")
    print("FAISS index saved: data/embeddings/faiss_index_improved.bin")
    
    # 임베딩 저장
    np.save("data/embeddings/embeddings_improved.npy", embeddings)
    print("Embeddings saved: data/embeddings/embeddings_improved.npy")
    
    # 메타데이터 저장
    with open("data/embeddings/metadata_improved.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Metadata saved: data/embeddings/metadata_improved.json")
    
    # 구축 보고서 생성
    build_report = {
        'build_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_documents': len(data),
        'embeddings_shape': embeddings.shape,
        'faiss_index_size': faiss_index.ntotal,
        'improvements': {
            'precedent_titles_improved': True,
            'court_names_added': True,
            'case_types_added': True
        }
    }
    
    with open("data/embeddings/vector_db_improved_report.json", 'w', encoding='utf-8') as f:
        json.dump(build_report, f, ensure_ascii=False, indent=2)
    print("Build report saved: data/embeddings/vector_db_improved_report.json")
    
    print("\nImproved vector database rebuild completed!")
    print(f"Index size: {faiss_index.ntotal}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    return faiss_index, embeddings, data

def test_improved_search():
    """개선된 검색 테스트"""
    print("\nTesting improved search...")
    
    # 개선된 데이터 로드
    faiss_index = faiss.read_index("data/embeddings/faiss_index_improved.bin")
    with open('data/embeddings/metadata_improved.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    
    # 테스트 쿼리
    test_queries = [
        "Supreme Court Decision",
        "District Court Decision", 
        "Civil Case",
        "Criminal Case",
        "Administrative Case"
    ]
    
    print("Search test results:")
    for query in test_queries:
        start_time = time.time()
        
        # 쿼리 임베딩 생성
        query_embedding = model.encode([query])
        
        # FAISS 검색
        distances, indices = faiss_index.search(query_embedding.astype('float32'), 3)
        
        search_time = time.time() - start_time
        
        print(f"\nQuery: '{query}' (Search time: {search_time:.4f}s)")
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(metadata):
                doc = metadata[idx]
                title = doc['metadata']['original_document']
                doc_type = doc['metadata']['data_type']
                similarity = 1.0 / (1.0 + distance)
                print(f"  {i+1}. [{doc_type}] {title} (Similarity: {similarity:.3f})")

def main():
    print("Improved Vector Database Rebuild")
    print("=" * 50)
    
    # 1. 개선된 벡터 데이터베이스 재구축
    faiss_index, embeddings, data = rebuild_improved_vector_database()
    
    # 2. 개선된 검색 테스트
    test_improved_search()
    
    print("\n" + "=" * 50)
    print("All improvements completed successfully!")

if __name__ == "__main__":
    main()
