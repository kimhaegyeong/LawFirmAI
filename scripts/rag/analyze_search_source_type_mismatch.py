#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
검색 시 source_type 불일치 분석 스크립트

실제 검색을 수행하여 반환되는 chunk들의 source_type을 확인하고,
요청한 타입과 실제 타입의 불일치를 분석합니다.

Usage:
    python scripts/rag/analyze_search_source_type_mismatch.py
"""

import sys
import os
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "lawfirm_langgraph"))

from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
from lawfirm_langgraph.core.utils.config import Config


def analyze_search_results_by_type():
    """타입별 검색 결과 분석"""
    print("="*80)
    print("검색 시 source_type 불일치 분석")
    print("="*80)
    
    # Config 설정
    config = Config()
    
    # SemanticSearchEngineV2 초기화
    print("\n🔄 Initializing SemanticSearchEngineV2...")
    engine = SemanticSearchEngineV2(
        db_path=config.database_path,
        use_mlflow_index=True
    )
    
    if engine.index is None:
        print("❌ Failed to load index")
        return
    
    print(f"✅ Index loaded: {engine.index.ntotal} vectors")
    
    # 테스트 쿼리
    test_query = "계약 해지 사유"
    
    # 각 타입별로 검색 수행
    source_types_to_test = [
        'statute_article',
        'case_paragraph',
        'decision_paragraph',
        'interpretation_paragraph'
    ]
    
    print(f"\n🔍 Testing search with query: '{test_query}'")
    print("="*80)
    
    all_results = {}
    
    for req_type in source_types_to_test:
        print(f"\n📋 Testing source_type: {req_type}")
        print("-" * 80)
        
        try:
            # 해당 타입으로 검색
            results = engine.search(
                query=test_query,
                k=100,  # 충분한 수의 결과 요청
                source_types=[req_type],
                similarity_threshold=0.05,  # 낮은 임계값
                min_results=1
            )
            
            if not results:
                print(f"   ❌ No results returned")
                all_results[req_type] = {
                    'requested': req_type,
                    'returned_count': 0,
                    'actual_types': {},
                    'mismatches': []
                }
                continue
            
            # 반환된 결과의 실제 source_type 확인
            actual_types = defaultdict(int)
            mismatches = []
            
            for i, result in enumerate(results[:50]):  # 처음 50개만 분석
                actual_type = (
                    result.get('type') or 
                    result.get('source_type') or 
                    result.get('metadata', {}).get('source_type', 'unknown')
                )
                actual_types[actual_type] += 1
                
                if actual_type != req_type:
                    chunk_id = result.get('chunk_id') or result.get('id') or result.get('metadata', {}).get('chunk_id')
                    mismatches.append({
                        'index': i,
                        'chunk_id': chunk_id,
                        'requested': req_type,
                        'actual': actual_type,
                        'score': result.get('score') or result.get('similarity', 0.0)
                    })
            
            print(f"   ✅ Returned {len(results)} results")
            print(f"   📊 Actual source_type distribution (first 50):")
            for actual_type, count in sorted(actual_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / min(50, len(results)) * 100)
                match_indicator = "✅" if actual_type == req_type else "❌"
                print(f"      {match_indicator} {actual_type}: {count}개 ({percentage:.1f}%)")
            
            if mismatches:
                print(f"   ⚠️  Found {len(mismatches)} mismatches (first 10):")
                for mismatch in mismatches[:10]:
                    print(f"      - Result #{mismatch['index']}: chunk_id={mismatch['chunk_id']}, "
                          f"requested={mismatch['requested']}, actual={mismatch['actual']}, "
                          f"score={mismatch['score']:.4f}")
            
            all_results[req_type] = {
                'requested': req_type,
                'returned_count': len(results),
                'actual_types': dict(actual_types),
                'mismatches': mismatches
            }
            
        except Exception as e:
            print(f"   ❌ Error during search: {e}")
            import traceback
            traceback.print_exc()
            all_results[req_type] = {
                'requested': req_type,
                'returned_count': 0,
                'actual_types': {},
                'mismatches': [],
                'error': str(e)
            }
    
    # 종합 분석
    print("\n" + "="*80)
    print("📊 종합 분석")
    print("="*80)
    
    total_mismatches = 0
    for req_type, result_data in all_results.items():
        mismatches = result_data.get('mismatches', [])
        if mismatches:
            total_mismatches += len(mismatches)
            print(f"\n❌ {req_type}: {len(mismatches)}개 불일치 발견")
            print(f"   요청: {req_type}")
            print(f"   실제 타입 분포:")
            for actual_type, count in sorted(result_data['actual_types'].items(), key=lambda x: x[1], reverse=True):
                print(f"      - {actual_type}: {count}개")
        else:
            print(f"\n✅ {req_type}: 불일치 없음")
    
    if total_mismatches > 0:
        print(f"\n⚠️  총 {total_mismatches}개의 source_type 불일치 발견")
        print("   → 필터링 로직에서 실제 타입을 확인하는 과정에 문제가 있을 수 있습니다.")
        print("   → 또는 FAISS 인덱스의 메타데이터와 DB의 source_type이 불일치할 수 있습니다.")
    else:
        print(f"\n✅ source_type 불일치 없음")
        print("   → 필터링 문제는 다른 원인일 수 있습니다 (예: 필터링 로직 자체의 문제)")


if __name__ == "__main__":
    try:
        analyze_search_results_by_type()
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

