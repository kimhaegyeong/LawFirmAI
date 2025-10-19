#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
검색 성능 최적화 스크립트
"""

import sys
sys.path.append('source')
from source.data.vector_store import LegalVectorStore
import time
import json
from pathlib import Path

def analyze_search_performance():
    """검색 성능 분석"""
    print("=== 검색 성능 분석 ===")
    
    # 벡터 스토어 초기화
    vector_store = LegalVectorStore(
        model_name='jhgan/ko-sroberta-multitask',
        dimension=768,
        index_type='flat'
    )
    
    # 인덱스 로드
    if not vector_store.load_index('data/embeddings/ml_enhanced_ko_sroberta_precedents'):
        print("벡터 인덱스 로드 실패")
        return
    
    print(f"벡터 인덱스 크기: {vector_store.index.ntotal:,}")
    
    # 다양한 검색 시나리오 테스트
    test_scenarios = [
        {
            'name': '단일 키워드',
            'queries': ['손해배상', '계약', '특허', '이혼', '형사']
        },
        {
            'name': '복합 키워드',
            'queries': ['손해배상 청구', '계약 해지', '특허 침해', '이혼 소송', '형사 처벌']
        },
        {
            'name': '긴 문장',
            'queries': [
                '손해배상 청구 요건과 손해의 범위',
                '계약 해지 시 손해배상 책임',
                '특허 침해 시 법적 효과와 구제방법'
            ]
        }
    ]
    
    results = {}
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} 테스트 ---")
        scenario_results = []
        
        for query in scenario['queries']:
            # 검색 시간 측정
            start_time = time.time()
            search_results = vector_store.search(query, top_k=10)
            search_time = time.time() - start_time
            
            # 결과 분석
            if search_results:
                scores = [r.get('score', 0) for r in search_results]
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
                min_score = min(scores)
                
                print(f"  '{query}': {search_time:.3f}초, 점수 범위: {min_score:.3f}-{max_score:.3f}, 평균: {avg_score:.3f}")
                
                scenario_results.append({
                    'query': query,
                    'search_time': search_time,
                    'result_count': len(search_results),
                    'avg_score': avg_score,
                    'max_score': max_score,
                    'min_score': min_score
                })
            else:
                print(f"  '{query}': {search_time:.3f}초, 결과 없음")
                scenario_results.append({
                    'query': query,
                    'search_time': search_time,
                    'result_count': 0,
                    'avg_score': 0,
                    'max_score': 0,
                    'min_score': 0
                })
        
        # 시나리오별 평균 성능
        avg_time = sum(r['search_time'] for r in scenario_results) / len(scenario_results)
        avg_score = sum(r['avg_score'] for r in scenario_results) / len(scenario_results)
        
        print(f"  평균 검색 시간: {avg_time:.3f}초")
        print(f"  평균 점수: {avg_score:.3f}")
        
        results[scenario['name']] = {
            'queries': scenario_results,
            'avg_search_time': avg_time,
            'avg_score': avg_score
        }
    
    return results

def optimize_vector_index():
    """벡터 인덱스 최적화"""
    print("\n=== 벡터 인덱스 최적화 ===")
    
    # 현재 인덱스 정보
    vector_store = LegalVectorStore(
        model_name='jhgan/ko-sroberta-multitask',
        dimension=768,
        index_type='flat'
    )
    
    if not vector_store.load_index('data/embeddings/ml_enhanced_ko_sroberta_precedents'):
        print("벡터 인덱스 로드 실패")
        return
    
    stats = vector_store.get_stats()
    print(f"현재 인덱스 정보:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 메모리 사용량 확인
    memory_usage = vector_store.get_memory_usage()
    print(f"\n메모리 사용량:")
    for key, value in memory_usage.items():
        print(f"  {key}: {value}")
    
    # 인덱스 최적화 제안
    print(f"\n최적화 제안:")
    
    if stats['documents_count'] > 10000:
        print("  - 대용량 데이터: IVF 인덱스 사용 고려")
        print("  - 양자화(Quantization) 활성화 고려")
    
    if memory_usage.get('total_memory_mb', 0) > 1000:
        print("  - 메모리 사용량이 높음: 지연 로딩 활성화 고려")
    
    print("  - 정기적인 인덱스 재구성 권장")
    print("  - 검색 결과 캐싱 구현 고려")

def create_optimized_search_config():
    """최적화된 검색 설정 생성"""
    print("\n=== 최적화된 검색 설정 생성 ===")
    
    config = {
        "search_optimization": {
            "vector_search": {
                "default_top_k": 10,
                "max_top_k": 50,
                "score_threshold": 0.3,
                "enable_reranking": True
            },
            "hybrid_search": {
                "exact_weight": 0.3,
                "semantic_weight": 0.7,
                "diversity_threshold": 0.8,
                "max_results": 20
            },
            "performance": {
                "enable_caching": True,
                "cache_ttl": 3600,
                "batch_size": 100,
                "parallel_processing": True
            }
        },
        "index_optimization": {
            "index_type": "ivf",
            "nlist": 1000,
            "quantization": "pq",
            "enable_lazy_loading": True
        }
    }
    
    # 설정 파일 저장
    with open('optimized_search_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("최적화된 검색 설정이 'optimized_search_config.json'에 저장되었습니다.")

def main():
    """메인 함수"""
    print("LawFirmAI 검색 성능 최적화")
    print("=" * 50)
    
    # 1. 검색 성능 분석
    performance_results = analyze_search_performance()
    
    # 2. 벡터 인덱스 최적화
    optimize_vector_index()
    
    # 3. 최적화된 설정 생성
    create_optimized_search_config()
    
    # 결과 저장
    with open('search_optimization_results.json', 'w', encoding='utf-8') as f:
        json.dump(performance_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n최적화 결과가 'search_optimization_results.json'에 저장되었습니다.")
    print("\n=== 최적화 완료 ===")

if __name__ == "__main__":
    main()
