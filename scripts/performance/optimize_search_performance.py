#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
검???�능 최적???�크립트
"""

import sys
sys.path.append('source')
from source.data.vector_store import LegalVectorStore
import time
import json
from pathlib import Path

def analyze_search_performance():
    """검???�능 분석"""
    print("=== 검???�능 분석 ===")
    
    # 벡터 ?�토??초기??
    vector_store = LegalVectorStore(
        model_name='jhgan/ko-sroberta-multitask',
        dimension=768,
        index_type='flat'
    )
    
    # ?�덱??로드
    if not vector_store.load_index('data/embeddings/ml_enhanced_ko_sroberta_precedents'):
        print("벡터 ?�덱??로드 ?�패")
        return
    
    print(f"벡터 ?�덱???�기: {vector_store.index.ntotal:,}")
    
    # ?�양??검???�나리오 ?�스??
    test_scenarios = [
        {
            'name': '?�일 ?�워??,
            'queries': ['?�해배상', '계약', '?�허', '?�혼', '?�사']
        },
        {
            'name': '복합 ?�워??,
            'queries': ['?�해배상 �?��', '계약 ?��?', '?�허 침해', '?�혼 ?�송', '?�사 처벌']
        },
        {
            'name': '�?문장',
            'queries': [
                '?�해배상 �?�� ?�건�??�해??범위',
                '계약 ?��? ???�해배상 책임',
                '?�허 침해 ??법적 ?�과?� 구제방법'
            ]
        }
    ]
    
    results = {}
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ?�스??---")
        scenario_results = []
        
        for query in scenario['queries']:
            # 검???�간 측정
            start_time = time.time()
            search_results = vector_store.search(query, top_k=10)
            search_time = time.time() - start_time
            
            # 결과 분석
            if search_results:
                scores = [r.get('score', 0) for r in search_results]
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
                min_score = min(scores)
                
                print(f"  '{query}': {search_time:.3f}�? ?�수 범위: {min_score:.3f}-{max_score:.3f}, ?�균: {avg_score:.3f}")
                
                scenario_results.append({
                    'query': query,
                    'search_time': search_time,
                    'result_count': len(search_results),
                    'avg_score': avg_score,
                    'max_score': max_score,
                    'min_score': min_score
                })
            else:
                print(f"  '{query}': {search_time:.3f}�? 결과 ?�음")
                scenario_results.append({
                    'query': query,
                    'search_time': search_time,
                    'result_count': 0,
                    'avg_score': 0,
                    'max_score': 0,
                    'min_score': 0
                })
        
        # ?�나리오�??�균 ?�능
        avg_time = sum(r['search_time'] for r in scenario_results) / len(scenario_results)
        avg_score = sum(r['avg_score'] for r in scenario_results) / len(scenario_results)
        
        print(f"  ?�균 검???�간: {avg_time:.3f}�?)
        print(f"  ?�균 ?�수: {avg_score:.3f}")
        
        results[scenario['name']] = {
            'queries': scenario_results,
            'avg_search_time': avg_time,
            'avg_score': avg_score
        }
    
    return results

def optimize_vector_index():
    """벡터 ?�덱??최적??""
    print("\n=== 벡터 ?�덱??최적??===")
    
    # ?�재 ?�덱???�보
    vector_store = LegalVectorStore(
        model_name='jhgan/ko-sroberta-multitask',
        dimension=768,
        index_type='flat'
    )
    
    if not vector_store.load_index('data/embeddings/ml_enhanced_ko_sroberta_precedents'):
        print("벡터 ?�덱??로드 ?�패")
        return
    
    stats = vector_store.get_stats()
    print(f"?�재 ?�덱???�보:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 메모�??�용???�인
    memory_usage = vector_store.get_memory_usage()
    print(f"\n메모�??�용??")
    for key, value in memory_usage.items():
        print(f"  {key}: {value}")
    
    # ?�덱??최적???�안
    print(f"\n최적???�안:")
    
    if stats['documents_count'] > 10000:
        print("  - ?�?�량 ?�이?? IVF ?�덱???�용 고려")
        print("  - ?�자??Quantization) ?�성??고려")
    
    if memory_usage.get('total_memory_mb', 0) > 1000:
        print("  - 메모�??�용?�이 ?�음: 지??로딩 ?�성??고려")
    
    print("  - ?�기?�인 ?�덱???�구??권장")
    print("  - 검??결과 캐싱 구현 고려")

def create_optimized_search_config():
    """최적?�된 검???�정 ?�성"""
    print("\n=== 최적?�된 검???�정 ?�성 ===")
    
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
    
    # ?�정 ?�일 ?�??
    with open('optimized_search_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("최적?�된 검???�정??'optimized_search_config.json'???�?�되?�습?�다.")

def main():
    """메인 ?�수"""
    print("LawFirmAI 검???�능 최적??)
    print("=" * 50)
    
    # 1. 검???�능 분석
    performance_results = analyze_search_performance()
    
    # 2. 벡터 ?�덱??최적??
    optimize_vector_index()
    
    # 3. 최적?�된 ?�정 ?�성
    create_optimized_search_config()
    
    # 결과 ?�??
    with open('search_optimization_results.json', 'w', encoding='utf-8') as f:
        json.dump(performance_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n최적??결과가 'search_optimization_results.json'???�?�되?�습?�다.")
    print("\n=== 최적???�료 ===")

if __name__ == "__main__":
    main()
