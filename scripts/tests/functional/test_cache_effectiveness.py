#!/usr/bin/env python3
"""
캐시 효과 테스트 스크립트
동일한 쿼리를 반복 실행하여 캐시 효과 확인
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


async def test_cache_effectiveness():
    """캐시 효과 테스트"""
    print("=" * 80)
    print("캐시 효과 테스트")
    print("=" * 80)
    print()
    
    try:
        # 설정 로드
        config = LangGraphConfig.from_env()
        service = LangGraphWorkflowService(config)
        
        # 테스트 쿼리 (동일한 쿼리를 여러 번 실행)
        test_query = "고용계약기간을 단축하는 것이 가능한지"
        num_iterations = 3
        
        print(f"테스트 쿼리: {test_query}")
        print(f"반복 횟수: {num_iterations}")
        print()
        
        execution_times = []
        cache_stats_before = None
        cache_stats_after = None
        
        for i in range(num_iterations):
            print(f"\n{'=' * 80}")
            print(f"실행 {i+1}/{num_iterations}")
            print(f"{'=' * 80}")
            
            # 캐시 통계 (첫 실행 전)
            if i == 0 and hasattr(service.legal_workflow, 'cache_manager') and service.legal_workflow.cache_manager:
                cache_stats_before = service.legal_workflow.cache_manager.get_stats()
                print(f"\n캐시 통계 (실행 전):")
                print(f"  Query Optimization - Hits: {cache_stats_before['query_optimization']['hits']}, Misses: {cache_stats_before['query_optimization']['misses']}")
                print(f"  Search Results - Hits: {cache_stats_before['search_results']['hits']}, Misses: {cache_stats_before['search_results']['misses']}")
            
            start_time = time.time()
            
            try:
                result = await service.process_query(
                    query=test_query,
                    session_id=f"test_cache_{i}",
                    enable_checkpoint=False
                )
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                
                print(f"\n✅ 실행 완료: {execution_time:.2f}초")
                print(f"   - Answer 길이: {len(result.get('answer', ''))}")
                print(f"   - Sources: {len(result.get('sources', []))}개")
                print(f"   - Confidence: {result.get('confidence', 0.0):.2f}")
                
                # 캐시 통계 (각 실행 후)
                if hasattr(service.legal_workflow, 'cache_manager') and service.legal_workflow.cache_manager:
                    cache_stats = service.legal_workflow.cache_manager.get_stats()
                    print(f"\n캐시 통계 (실행 후):")
                    print(f"  Query Optimization - Hits: {cache_stats['query_optimization']['hits']}, Misses: {cache_stats['query_optimization']['misses']}")
                    print(f"  Search Results - Hits: {cache_stats['search_results']['hits']}, Misses: {cache_stats['search_results']['misses']}")
                    
                    if i == num_iterations - 1:
                        cache_stats_after = cache_stats
                
            except Exception as e:
                print(f"❌ 실행 실패: {e}")
                import traceback
                traceback.print_exc()
        
        # 결과 요약
        print(f"\n{'=' * 80}")
        print("결과 요약")
        print(f"{'=' * 80}")
        
        if execution_times:
            print(f"\n실행 시간:")
            for i, exec_time in enumerate(execution_times, 1):
                print(f"  실행 {i}: {exec_time:.2f}초")
            
            if len(execution_times) >= 2:
                first_time = execution_times[0]
                subsequent_avg = sum(execution_times[1:]) / len(execution_times[1:])
                improvement = ((first_time - subsequent_avg) / first_time) * 100
                
                print(f"\n캐시 효과:")
                print(f"  첫 실행: {first_time:.2f}초")
                print(f"  이후 평균: {subsequent_avg:.2f}초")
                print(f"  개선율: {improvement:.2f}%")
                print(f"  절대 개선: {first_time - subsequent_avg:.2f}초")
        
        # 캐시 통계 비교
        if cache_stats_before and cache_stats_after:
            print(f"\n캐시 통계 비교:")
            print(f"  Query Optimization:")
            print(f"    - Hits 증가: {cache_stats_after['query_optimization']['hits'] - cache_stats_before['query_optimization']['hits']}")
            print(f"    - Misses 증가: {cache_stats_after['query_optimization']['misses'] - cache_stats_before['query_optimization']['misses']}")
            print(f"    - Hit Rate: {cache_stats_after['query_optimization']['hit_rate']:.2%}")
            
            print(f"  Search Results:")
            print(f"    - Hits 증가: {cache_stats_after['search_results']['hits'] - cache_stats_before['search_results']['hits']}")
            print(f"    - Misses 증가: {cache_stats_after['search_results']['misses'] - cache_stats_before['search_results']['misses']}")
            print(f"    - Hit Rate: {cache_stats_after['search_results']['hit_rate']:.2%}")
        
        print(f"\n{'=' * 80}")
        print("테스트 완료")
        print(f"{'=' * 80}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_cache_effectiveness())
    sys.exit(0 if result else 1)

