#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
벡터 검색 성능 테스트 스크립트

구축된 벡터DB의 검색 성능을 테스트합니다.
"""

import sys
import json
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available. Please install faiss-cpu or faiss-gpu")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("SentenceTransformers not available. Please install sentence-transformers")

from source.data.database import DatabaseManager

class VectorSearchTester:
    """벡터 검색 성능 테스트 클래스"""
    
    def __init__(self, embeddings_dir: str = "./data/embeddings"):
        """테스트 초기화"""
        self.embeddings_dir = Path(embeddings_dir)
        self.faiss_index = None
        self.metadata = []
        self.model = None
        self.db_manager = DatabaseManager()
        
        # 테스트 결과
        self.test_results = {
            'search_tests': [],
            'performance_metrics': {},
            'accuracy_tests': []
        }
    
    def load_vector_database(self):
        """벡터 데이터베이스 로드"""
        print("벡터 데이터베이스 로딩 중...")
        
        # FAISS 인덱스 로드
        if FAISS_AVAILABLE:
            index_path = self.embeddings_dir / "faiss_index.bin"
            if index_path.exists():
                self.faiss_index = faiss.read_index(str(index_path))
                print(f"FAISS 인덱스 로드 완료: {self.faiss_index.ntotal}개 벡터")
            else:
                print("FAISS 인덱스 파일을 찾을 수 없습니다.")
                return False
        
        # 메타데이터 로드
        metadata_path = self.embeddings_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"메타데이터 로드 완료: {len(self.metadata)}개 문서")
        else:
            print("메타데이터 파일을 찾을 수 없습니다.")
            return False
        
        # Sentence-BERT 모델 로드
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer("jhgan/ko-sroberta-multitask")
            print("Sentence-BERT 모델 로드 완료")
        
        return True
    
    def test_search_performance(self, query: str, k: int = 10) -> Dict[str, Any]:
        """검색 성능 테스트"""
        if not self.faiss_index or not self.model:
            return {"error": "벡터 데이터베이스가 로드되지 않았습니다."}
        
        start_time = time.time()
        
        # 쿼리 임베딩 생성
        query_embedding = self.model.encode([query])
        
        # FAISS 검색
        search_start = time.time()
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
        search_time = time.time() - search_start
        
        # 결과 처리
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                doc = self.metadata[idx]
                results.append({
                    'rank': i + 1,
                    'document_id': doc['id'],
                    'title': doc['metadata']['original_document'],
                    'data_type': doc['metadata']['data_type'],
                    'similarity_score': float(1.0 / (1.0 + distance)),  # 거리를 유사도로 변환
                    'distance': float(distance),
                    'content_preview': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
                })
        
        total_time = time.time() - start_time
        
        return {
            'query': query,
            'k': k,
            'total_time': total_time,
            'search_time': search_time,
            'embedding_time': total_time - search_time,
            'results': results,
            'num_results': len(results)
        }
    
    def run_performance_tests(self):
        """성능 테스트 실행"""
        print("\n=== 벡터 검색 성능 테스트 ===")
        
        test_queries = [
            "계약서 작성 방법",
            "민사소송 절차",
            "부동산 등기",
            "형사처벌",
            "행정처분",
            "헌법재판소 결정",
            "법령 해석",
            "판례 검색",
            "법률 용어",
            "소송 제기"
        ]
        
        total_search_time = 0
        total_queries = 0
        
        for query in test_queries:
            print(f"\n테스트 쿼리: '{query}'")
            result = self.test_search_performance(query, k=5)
            
            if 'error' not in result:
                print(f"  검색 시간: {result['search_time']:.4f}초")
                print(f"  총 시간: {result['total_time']:.4f}초")
                print(f"  결과 수: {result['num_results']}개")
                
                # 상위 3개 결과 출력
                for i, res in enumerate(result['results'][:3]):
                    print(f"    {i+1}. [{res['data_type']}] {res['title']} (유사도: {res['similarity_score']:.3f})")
                
                total_search_time += result['search_time']
                total_queries += 1
                
                self.test_results['search_tests'].append(result)
            else:
                print(f"  오류: {result['error']}")
        
        # 성능 지표 계산
        if total_queries > 0:
            avg_search_time = total_search_time / total_queries
            self.test_results['performance_metrics'] = {
                'average_search_time': avg_search_time,
                'total_queries': total_queries,
                'queries_per_second': 1.0 / avg_search_time if avg_search_time > 0 else 0
            }
            
            print(f"\n=== 성능 지표 ===")
            print(f"평균 검색 시간: {avg_search_time:.4f}초")
            print(f"초당 쿼리 수: {1.0 / avg_search_time:.2f}개")
    
    def test_accuracy(self):
        """정확도 테스트"""
        print("\n=== 벡터 검색 정확도 테스트 ===")
        
        # 법령 관련 쿼리
        law_queries = [
            ("민법", "laws"),
            ("형법", "laws"),
            ("상법", "laws"),
            ("민사소송법", "laws")
        ]
        
        # 판례 관련 쿼리
        precedent_queries = [
            ("대법원 판결", "precedents"),
            ("지방법원 판결", "precedents"),
            ("고등법원 판결", "precedents")
        ]
        
        all_queries = law_queries + precedent_queries
        correct_predictions = 0
        
        for query, expected_type in all_queries:
            result = self.test_search_performance(query, k=5)
            
            if 'error' not in result and result['results']:
                # 상위 결과의 데이터 타입 확인
                top_result_type = result['results'][0]['data_type']
                is_correct = top_result_type == expected_type
                
                if is_correct:
                    correct_predictions += 1
                
                print(f"쿼리: '{query}' -> 예상: {expected_type}, 실제: {top_result_type} {'✓' if is_correct else '✗'}")
        
        accuracy = correct_predictions / len(all_queries) if all_queries else 0
        self.test_results['accuracy_tests'] = {
            'total_queries': len(all_queries),
            'correct_predictions': correct_predictions,
            'accuracy': accuracy
        }
        
        print(f"\n정확도: {accuracy:.2%} ({correct_predictions}/{len(all_queries)})")
    
    def test_hybrid_search(self):
        """하이브리드 검색 테스트 (벡터 + SQLite)"""
        print("\n=== 하이브리드 검색 테스트 ===")
        
        test_queries = [
            "민법",
            "대법원",
            "계약"
        ]
        
        for query in test_queries:
            print(f"\n하이브리드 검색: '{query}'")
            
            # 벡터 검색
            vector_result = self.test_search_performance(query, k=3)
            
            # SQLite 검색 (정확한 매칭)
            try:
                sqlite_results = self.db_manager.search_documents(query, limit=3)
                print(f"  SQLite 검색 결과: {len(sqlite_results)}개")
                
                # 결과 비교
                if 'error' not in vector_result and vector_result['results']:
                    print(f"  벡터 검색 결과: {len(vector_result['results'])}개")
                    
                    # 상위 결과 출력
                    print("  벡터 검색 상위 결과:")
                    for i, res in enumerate(vector_result['results'][:2]):
                        print(f"    {i+1}. [{res['data_type']}] {res['title']}")
                    
                    print("  SQLite 검색 상위 결과:")
                    for i, res in enumerate(sqlite_results[:2]):
                        print(f"    {i+1}. [{res['document_type']}] {res['title']}")
                
            except Exception as e:
                print(f"  SQLite 검색 오류: {e}")
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("벡터 검색 성능 테스트 시작")
        
        if not self.load_vector_database():
            print("벡터 데이터베이스 로드 실패")
            return
        
        # 성능 테스트
        self.run_performance_tests()
        
        # 정확도 테스트
        self.test_accuracy()
        
        # 하이브리드 검색 테스트
        self.test_hybrid_search()
        
        # 결과 저장
        self.save_test_results()
        
        print("\n=== 테스트 완료 ===")
        self.print_summary()
    
    def save_test_results(self):
        """테스트 결과 저장"""
        results_path = self.embeddings_dir / "vector_search_test_results.json"
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n테스트 결과 저장 완료: {results_path}")
    
    def print_summary(self):
        """테스트 요약 출력"""
        if 'performance_metrics' in self.test_results:
            metrics = self.test_results['performance_metrics']
            print(f"평균 검색 시간: {metrics['average_search_time']:.4f}초")
            print(f"초당 쿼리 수: {metrics['queries_per_second']:.2f}개")
        
        if 'accuracy_tests' in self.test_results:
            accuracy = self.test_results['accuracy_tests']
            print(f"검색 정확도: {accuracy['accuracy']:.2%}")


def main():
    tester = VectorSearchTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
