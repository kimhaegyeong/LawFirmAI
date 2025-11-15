#!/usr/bin/env python3
"""
워크플로우 최적화 테스트 스크립트
캐싱, 병렬 처리, State 접근 최적화 테스트
"""

import os
import sys
import time
import asyncio
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    env_file = project_root / "api" / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)
except:
    pass

from lawfirm_langgraph.langgraph_core.workflow.workflow_service import LangGraphWorkflowService
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig


async def test_workflow_optimization():
    """워크플로우 최적화 테스트"""
    print("=" * 80)
    print("워크플로우 최적화 테스트")
    print("=" * 80)
    print()
    
    try:
        # 설정 로드
        config = LangGraphConfig.from_env()
        service = LangGraphWorkflowService(config)
        
        # 캐시 관리자 확인
        print("1. 캐시 관리자 확인")
        print("-" * 80)
        if hasattr(service.legal_workflow, 'cache_manager') and service.legal_workflow.cache_manager:
            print("✅ WorkflowCacheManager 초기화 완료")
            cache_stats = service.legal_workflow.cache_manager.get_stats()
            print(f"   캐시 통계:")
            print(f"   - Query Preparation: {cache_stats['query_preparation']['hits']} hits, {cache_stats['query_preparation']['misses']} misses")
            print(f"   - Query Optimization: {cache_stats['query_optimization']['hits']} hits, {cache_stats['query_optimization']['misses']} misses")
            print(f"   - Search Results: {cache_stats['search_results']['hits']} hits, {cache_stats['search_results']['misses']} misses")
        else:
            print("⚠️  WorkflowCacheManager가 초기화되지 않았습니다.")
        print()
        
        # A/B 테스트 관리자 확인
        print("2. A/B 테스트 관리자 확인")
        print("-" * 80)
        if hasattr(service, 'ab_test_manager') and service.ab_test_manager:
            print("✅ ABTestManager 초기화 완료")
            summary = service.ab_test_manager.get_summary()
            if summary:
                print(f"   실험 수: {len(summary)}")
                for exp_name, exp_info in summary.items():
                    print(f"   - {exp_name}: {exp_info['total_results']} results")
            else:
                print("   아직 실험 결과가 없습니다.")
        else:
            print("⚠️  ABTestManager가 초기화되지 않았습니다.")
            print("   (ENABLE_AB_TESTING=true로 설정하면 활성화됩니다)")
        print()
        
        # 테스트 쿼리 실행
        test_queries = [
            "고용계약기간을 단축하는 것이 가능한지",
            "임신 중인 여성근로자의 휴일대체 가능 여부",
            "사내근로복지기금을 활용한 협력업체 근로자 지원 방법"
        ]
        
        print("3. 워크플로우 실행 테스트")
        print("-" * 80)
        
        execution_times = []
        cache_hits = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n테스트 {i}/{len(test_queries)}: {query[:50]}...")
            
            start_time = time.time()
            try:
                result = await service.process_query(
                    query=query,
                    session_id=f"test_optimization_{i}",
                    enable_checkpoint=False
                )
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                
                # 캐시 히트 확인
                if hasattr(service.legal_workflow, 'cache_manager') and service.legal_workflow.cache_manager:
                    cache_stats = service.legal_workflow.cache_manager.get_stats()
                    total_hits = (cache_stats['query_preparation']['hits'] + 
                                cache_stats['query_optimization']['hits'] + 
                                cache_stats['search_results']['hits'])
                    cache_hits.append(total_hits)
                
                print(f"   ✅ 실행 완료: {execution_time:.2f}초")
                print(f"   - Answer 길이: {len(result.get('answer', ''))}")
                print(f"   - Sources: {len(result.get('sources', []))}개")
                print(f"   - Confidence: {result.get('confidence', 0.0):.2f}")
                
            except Exception as e:
                print(f"   ❌ 실행 실패: {e}")
                import traceback
                traceback.print_exc()
        
        # 성능 요약
        print("\n4. 성능 요약")
        print("-" * 80)
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            
            print(f"   실행 시간:")
            print(f"   - 평균: {avg_time:.2f}초")
            print(f"   - 최소: {min_time:.2f}초")
            print(f"   - 최대: {max_time:.2f}초")
            
            # 첫 번째 vs 두 번째 이후 비교 (캐시 효과)
            if len(execution_times) >= 2:
                first_time = execution_times[0]
                subsequent_avg = sum(execution_times[1:]) / len(execution_times[1:])
                improvement = ((first_time - subsequent_avg) / first_time) * 100
                print(f"\n   캐시 효과:")
                print(f"   - 첫 실행: {first_time:.2f}초")
                print(f"   - 이후 평균: {subsequent_avg:.2f}초")
                print(f"   - 개선율: {improvement:.2f}%")
        
        # 캐시 통계
        if hasattr(service.legal_workflow, 'cache_manager') and service.legal_workflow.cache_manager:
            print("\n5. 캐시 통계")
            print("-" * 80)
            cache_stats = service.legal_workflow.cache_manager.get_stats()
            
            print(f"   Query Preparation:")
            print(f"   - Hit Rate: {cache_stats['query_preparation']['hit_rate']:.2%}")
            print(f"   - Hits: {cache_stats['query_preparation']['hits']}")
            print(f"   - Misses: {cache_stats['query_preparation']['misses']}")
            
            print(f"\n   Query Optimization:")
            print(f"   - Hit Rate: {cache_stats['query_optimization']['hit_rate']:.2%}")
            print(f"   - Hits: {cache_stats['query_optimization']['hits']}")
            print(f"   - Misses: {cache_stats['query_optimization']['misses']}")
            
            print(f"\n   Search Results:")
            print(f"   - Hit Rate: {cache_stats['search_results']['hit_rate']:.2%}")
            print(f"   - Hits: {cache_stats['search_results']['hits']}")
            print(f"   - Misses: {cache_stats['search_results']['misses']}")
            
            overall_hit_rate = service.legal_workflow.cache_manager.get_hit_rate()
            print(f"\n   전체 캐시 히트율: {overall_hit_rate:.2%}")
        
        # A/B 테스트 결과
        if hasattr(service, 'ab_test_manager') and service.ab_test_manager:
            summary = service.ab_test_manager.get_summary()
            if summary:
                print("\n6. A/B 테스트 결과")
                print("-" * 80)
                for exp_name, exp_info in summary.items():
                    print(f"   실험: {exp_name}")
                    results = service.ab_test_manager.get_results(exp_name)
                    for variant, metrics in results.items():
                        if "execution_time" in metrics:
                            stats = metrics["execution_time"]
                            print(f"   - {variant}: 평균 {stats['mean']:.2f}초 (n={stats['count']})")
        
        print("\n" + "=" * 80)
        print("테스트 완료")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    result = asyncio.run(test_workflow_optimization())
    sys.exit(0 if result else 1)

