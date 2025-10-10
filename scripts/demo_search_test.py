#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실제 사용자 시나리오 기반 검색 테스트
"""

import sys
import json
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Required library not available: {e}")
    sys.exit(1)

class DemoSearchTester:
    """실제 사용자 시나리오 기반 검색 테스트"""
    
    def __init__(self):
        self.model = None
        self.faiss_index = None
        self.metadata = []
        self.load_system()
    
    def load_system(self):
        """시스템 로드"""
        print("LawFirmAI 시스템 로딩 중...")
        
        # Sentence-BERT 모델 로드
        self.model = SentenceTransformer("jhgan/ko-sroberta-multitask")
        print("Sentence-BERT 모델 로드 완료")
        
        # FAISS 인덱스 로드
        self.faiss_index = faiss.read_index("data/embeddings/faiss_index.bin")
        print(f"FAISS 인덱스 로드 완료: {self.faiss_index.ntotal}개 벡터")
        
        # 메타데이터 로드
        with open("data/embeddings/metadata.json", 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"메타데이터 로드 완료: {len(self.metadata)}개 문서")
        
        print("시스템 로드 완료!\n")
    
    def search(self, query: str, k: int = 5):
        """검색 수행"""
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
                    'title': doc['metadata']['original_document'],
                    'type': doc['metadata']['data_type'],
                    'similarity': float(1.0 / (1.0 + distance)),
                    'preview': doc['text'][:150] + "..." if len(doc['text']) > 150 else doc['text']
                })
        
        search_time = time.time() - start_time
        return results, search_time
    
    def run_demo_scenarios(self):
        """실제 사용자 시나리오 테스트"""
        print("실제 사용자 시나리오 테스트 시작\n")
        
        scenarios = [
            {
                "title": "계약서 관련 질문",
                "query": "계약서 작성할 때 주의사항이 뭐야?",
                "expected": "계약서, 계약, 작성"
            },
            {
                "title": "민사소송 관련 질문", 
                "query": "민사소송 제기하려면 어떻게 해야 해?",
                "expected": "민사소송, 소송, 제기"
            },
            {
                "title": "부동산 관련 질문",
                "query": "부동산 등기 신청 방법 알려줘",
                "expected": "부동산, 등기, 신청"
            },
            {
                "title": "판례 검색",
                "query": "대법원 판결문 찾아줘",
                "expected": "대법원, 판결, 판례"
            },
            {
                "title": "법령 해석",
                "query": "형법 조문 해석해줘",
                "expected": "형법, 조문, 해석"
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"--- 시나리오 {i}: {scenario['title']} ---")
            print(f"질문: {scenario['query']}")
            print(f"예상 키워드: {scenario['expected']}")
            
            results, search_time = self.search(scenario['query'], k=3)
            
            print(f"검색 시간: {search_time:.4f}초")
            print("검색 결과:")
            for result in results:
                print(f"  {result['rank']}. [{result['type']}] {result['title']} (유사도: {result['similarity']:.3f})")
                print(f"      미리보기: {result['preview']}")
            
            print()
    
    def run_performance_test(self):
        """성능 테스트"""
        print("성능 테스트 시작\n")
        
        test_queries = [
            "계약서 작성",
            "민사소송 절차", 
            "부동산 등기",
            "형사처벌",
            "행정처분",
            "헌법재판소",
            "법령 해석",
            "판례 검색",
            "법률 용어",
            "소송 제기"
        ]
        
        total_time = 0
        successful_searches = 0
        
        for query in test_queries:
            try:
                results, search_time = self.search(query, k=5)
                total_time += search_time
                successful_searches += 1
                print(f"OK '{query}': {search_time:.4f}초 ({len(results)}개 결과)")
            except Exception as e:
                print(f"ERROR '{query}': 오류 - {e}")
        
        if successful_searches > 0:
            avg_time = total_time / successful_searches
            qps = 1.0 / avg_time if avg_time > 0 else 0
            
            print(f"\n성능 지표:")
            print(f"  평균 검색 시간: {avg_time:.4f}초")
            print(f"  초당 쿼리 수: {qps:.2f}개")
            print(f"  성공률: {successful_searches}/{len(test_queries)} ({successful_searches/len(test_queries)*100:.1f}%)")
    
    def run_accuracy_test(self):
        """정확도 테스트"""
        print("정확도 테스트 시작\n")
        
        # 법령 관련 쿼리
        law_queries = [
            ("민법", "laws"),
            ("형법", "laws"), 
            ("상법", "laws"),
            ("민사소송법", "laws"),
            ("형사소송법", "laws")
        ]
        
        # 판례 관련 쿼리
        precedent_queries = [
            ("대법원 판결", "precedents"),
            ("지방법원 판결", "precedents"),
            ("고등법원 판결", "precedents")
        ]
        
        all_queries = law_queries + precedent_queries
        correct_predictions = 0
        
        print("법령 검색 테스트:")
        for query, expected in law_queries:
            results, _ = self.search(query, k=1)
            if results:
                actual = results[0]['type']
                is_correct = actual == expected
                if is_correct:
                    correct_predictions += 1
                print(f"  '{query}' -> 예상: {expected}, 실제: {actual} {'OK' if is_correct else 'FAIL'}")
        
        print("\n판례 검색 테스트:")
        for query, expected in precedent_queries:
            results, _ = self.search(query, k=1)
            if results:
                actual = results[0]['type']
                is_correct = actual == expected
                if is_correct:
                    correct_predictions += 1
                print(f"  '{query}' -> 예상: {expected}, 실제: {actual} {'OK' if is_correct else 'FAIL'}")
        
        accuracy = correct_predictions / len(all_queries) if all_queries else 0
        print(f"\n정확도: {accuracy:.2%} ({correct_predictions}/{len(all_queries)})")
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("LawFirmAI 종합 테스트 시작\n")
        print("=" * 50)
        
        # 실제 사용자 시나리오 테스트
        self.run_demo_scenarios()
        
        print("=" * 50)
        
        # 성능 테스트
        self.run_performance_test()
        
        print("=" * 50)
        
        # 정확도 테스트
        self.run_accuracy_test()
        
        print("=" * 50)
        print("모든 테스트 완료!")

def main():
    tester = DemoSearchTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
