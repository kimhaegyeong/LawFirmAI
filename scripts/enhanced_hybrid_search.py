#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
향상된 하이브리드 검색 시스템 (벡터 + SQLite)
"""

import sys
import json
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Required library not available: {e}")
    sys.exit(1)

class EnhancedHybridSearch:
    """향상된 하이브리드 검색 클래스"""
    
    def __init__(self):
        self.model = None
        self.faiss_index = None
        self.metadata = []
        self.db_path = "data/lawfirm.db"
        self.load_system()
    
    def load_system(self):
        """시스템 로드"""
        print("Loading enhanced hybrid search system...")
        
        # Sentence-BERT 모델 로드
        self.model = SentenceTransformer("jhgan/ko-sroberta-multitask")
        print("Sentence-BERT model loaded")
        
        # FAISS 인덱스 로드
        self.faiss_index = faiss.read_index("data/embeddings/faiss_index_improved.bin")
        print(f"FAISS index loaded: {self.faiss_index.ntotal} vectors")
        
        # 메타데이터 로드
        with open("data/embeddings/metadata_improved.json", 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"Metadata loaded: {len(self.metadata)} documents")
        
        print("System loaded successfully!")
    
    def vector_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """벡터 검색"""
        start_time = time.time()
        
        # 쿼리 임베딩 생성
        query_embedding = self.model.encode([query])
        
        # FAISS 검색
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
        
        # 결과 처리
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                doc = self.metadata[idx]
                results.append({
                    'rank': i + 1,
                    'id': doc['id'],
                    'title': doc['metadata']['original_document'],
                    'type': doc['metadata']['data_type'],
                    'similarity': float(1.0 / (1.0 + distance)),
                    'distance': float(distance),
                    'preview': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'],
                    'search_type': 'vector'
                })
        
        search_time = time.time() - start_time
        return results, search_time
    
    def sqlite_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """SQLite 검색 (정확한 매칭)"""
        start_time = time.time()
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 제목과 내용에서 검색
            search_query = """
                SELECT id, document_type, title, content, created_at
                FROM documents 
                WHERE title LIKE ? OR content LIKE ?
                ORDER BY 
                    CASE 
                        WHEN title LIKE ? THEN 1
                        WHEN content LIKE ? THEN 2
                        ELSE 3
                    END,
                    created_at DESC
                LIMIT ?
            """
            
            search_pattern = f"%{query}%"
            cursor.execute(search_query, (search_pattern, search_pattern, search_pattern, search_pattern, limit))
            rows = cursor.fetchall()
            
            results = []
            for i, row in enumerate(rows):
                results.append({
                    'rank': i + 1,
                    'id': row['id'],
                    'title': row['title'],
                    'type': row['document_type'],
                    'similarity': 1.0,  # 정확한 매칭이므로 최고 유사도
                    'preview': row['content'][:200] + "..." if len(row['content']) > 200 else row['content'],
                    'search_type': 'sqlite'
                })
            
            conn.close()
            search_time = time.time() - start_time
            return results, search_time
            
        except Exception as e:
            print(f"SQLite search error: {e}")
            return [], 0
    
    def hybrid_search(self, query: str, vector_k: int = 5, sqlite_k: int = 5) -> Dict[str, Any]:
        """하이브리드 검색 (벡터 + SQLite)"""
        print(f"Hybrid search for: '{query}'")
        
        # 벡터 검색
        vector_results, vector_time = self.vector_search(query, vector_k)
        
        # SQLite 검색
        sqlite_results, sqlite_time = self.sqlite_search(query, sqlite_k)
        
        # 결과 통합 및 중복 제거
        combined_results = []
        seen_ids = set()
        
        # 벡터 검색 결과 추가 (유사도 기반 정렬)
        for result in vector_results:
            if result['id'] not in seen_ids:
                combined_results.append(result)
                seen_ids.add(result['id'])
        
        # SQLite 검색 결과 추가 (정확한 매칭 우선)
        for result in sqlite_results:
            if result['id'] not in seen_ids:
                # 정확한 매칭이므로 높은 우선순위 부여
                result['similarity'] = 1.0
                combined_results.append(result)
                seen_ids.add(result['id'])
        
        # 최종 결과 정렬 (유사도 + 검색 타입)
        def sort_key(result):
            # SQLite 결과를 우선하고, 그 다음 유사도 순
            if result['search_type'] == 'sqlite':
                return (0, -result['similarity'])
            else:
                return (1, -result['similarity'])
        
        combined_results.sort(key=sort_key)
        
        # 순위 재정렬
        for i, result in enumerate(combined_results):
            result['rank'] = i + 1
        
        total_time = vector_time + sqlite_time
        
        return {
            'query': query,
            'total_results': len(combined_results),
            'vector_results': len(vector_results),
            'sqlite_results': len(sqlite_results),
            'vector_time': vector_time,
            'sqlite_time': sqlite_time,
            'total_time': total_time,
            'results': combined_results[:vector_k + sqlite_k]  # 최대 결과 수 제한
        }
    
    def test_hybrid_search(self):
        """하이브리드 검색 테스트"""
        print("\nTesting hybrid search system...")
        
        test_queries = [
            "민법",
            "Supreme Court Decision",
            "계약서",
            "형사처벌",
            "부동산 등기"
        ]
        
        for query in test_queries:
            print(f"\n--- Query: '{query}' ---")
            
            # 하이브리드 검색 실행
            result = self.hybrid_search(query, vector_k=3, sqlite_k=3)
            
            print(f"Total results: {result['total_results']}")
            print(f"Vector results: {result['vector_results']}")
            print(f"SQLite results: {result['sqlite_results']}")
            print(f"Vector time: {result['vector_time']:.4f}s")
            print(f"SQLite time: {result['sqlite_time']:.4f}s")
            print(f"Total time: {result['total_time']:.4f}s")
            
            print("Top results:")
            for i, res in enumerate(result['results'][:5]):
                print(f"  {res['rank']}. [{res['type']}] {res['title']} (Similarity: {res['similarity']:.3f}, Type: {res['search_type']})")
    
    def performance_benchmark(self):
        """성능 벤치마크"""
        print("\nPerformance benchmark...")
        
        test_queries = [
            "민법", "형법", "상법", "민사소송법", "형사소송법",
            "Supreme Court", "Civil Case", "Criminal Case", "Administrative Case",
            "계약서", "부동산", "등기", "소송", "판결"
        ]
        
        vector_times = []
        sqlite_times = []
        hybrid_times = []
        
        for query in test_queries:
            # 벡터 검색 시간 측정
            _, vector_time = self.vector_search(query, k=5)
            vector_times.append(vector_time)
            
            # SQLite 검색 시간 측정
            _, sqlite_time = self.sqlite_search(query, limit=5)
            sqlite_times.append(sqlite_time)
            
            # 하이브리드 검색 시간 측정
            result = self.hybrid_search(query, vector_k=3, sqlite_k=3)
            hybrid_times.append(result['total_time'])
        
        # 통계 계산
        avg_vector_time = sum(vector_times) / len(vector_times)
        avg_sqlite_time = sum(sqlite_times) / len(sqlite_times)
        avg_hybrid_time = sum(hybrid_times) / len(hybrid_times)
        
        print(f"Average vector search time: {avg_vector_time:.4f}s")
        print(f"Average SQLite search time: {avg_sqlite_time:.4f}s")
        print(f"Average hybrid search time: {avg_hybrid_time:.4f}s")
        print(f"Vector QPS: {1.0/avg_vector_time:.2f}")
        print(f"SQLite QPS: {1.0/avg_sqlite_time:.2f}")
        print(f"Hybrid QPS: {1.0/avg_hybrid_time:.2f}")

def main():
    print("Enhanced Hybrid Search System")
    print("=" * 50)
    
    # 하이브리드 검색 시스템 초기화
    search_system = EnhancedHybridSearch()
    
    # 하이브리드 검색 테스트
    search_system.test_hybrid_search()
    
    # 성능 벤치마크
    search_system.performance_benchmark()
    
    print("\n" + "=" * 50)
    print("Hybrid search system testing completed!")

if __name__ == "__main__":
    main()
