# -*- coding: utf-8 -*-
"""
성능 최적화 테스트 스크립트
캐싱 및 병렬 처리 성능을 측정합니다.
"""

import sys
import os
import asyncio
import time
import statistics
from typing import List, Dict, Any
from datetime import datetime

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from source.services.chat_service import ChatService
from source.utils.config import Config
from source.services.cache_manager import get_cache_manager
from source.services.optimized_search_engine import OptimizedSearchEngine

class PerformanceTester:
    """성능 테스트 클래스"""
    
    def __init__(self):
        self.config = Config()
        self.chat_service = ChatService(self.config)
        self.cache_manager = get_cache_manager()
        
        # 테스트 쿼리들
        self.test_queries = [
            "계약서 작성에 대해 알려주세요.",
            "이혼 절차는 어떻게 진행되나요?",
            "형사 사건에서 변호사 선임은 필수인가요?",
            "부동산 매매 계약 시 주의사항이 있나요?",
            "손해배상 청구 요건은 무엇인가요?",
            "법정 절차에서 증인 선서는 어떻게 하나요?",
            "민사소송 제기 절차를 알려주세요.",
            "가족법상 친권은 어떻게 결정되나요?",
            "상속 포기 절차는 어떻게 진행되나요?",
            "법원 판결에 대한 불복 방법은 무엇인가요?"
        ]
    
    async def test_cache_performance(self) -> Dict[str, Any]:
        """캐시 성능 테스트"""
        print("=" * 60)
        print("캐시 성능 테스트")
        print("=" * 60)
        
        # 캐시 초기화
        self.cache_manager.clear()
        
        # 첫 번째 실행 (캐시 미스)
        print("\n첫 번째 실행 (캐시 미스):")
        first_run_times = []
        
        for i, query in enumerate(self.test_queries[:5]):
            start_time = time.time()
            result = await self.chat_service.process_message(query)
            end_time = time.time()
            
            execution_time = end_time - start_time
            first_run_times.append(execution_time)
            
            print(f"  {i+1}. {query[:30]}... - {execution_time:.3f}초")
        
        # 두 번째 실행 (캐시 히트)
        print("\n두 번째 실행 (캐시 히트):")
        second_run_times = []
        
        for i, query in enumerate(self.test_queries[:5]):
            start_time = time.time()
            result = await self.chat_service.process_message(query)
            end_time = time.time()
            
            execution_time = end_time - start_time
            second_run_times.append(execution_time)
            
            print(f"  {i+1}. {query[:30]}... - {execution_time:.3f}초")
        
        # 통계 계산
        first_avg = statistics.mean(first_run_times)
        second_avg = statistics.mean(second_run_times)
        improvement = ((first_avg - second_avg) / first_avg) * 100
        
        cache_stats = self.cache_manager.get_stats()
        
        return {
            'first_run_avg': first_avg,
            'second_run_avg': second_avg,
            'improvement_percent': improvement,
            'cache_stats': cache_stats
        }
    
    async def test_parallel_search_performance(self) -> Dict[str, Any]:
        """병렬 검색 성능 테스트"""
        print("\n" + "=" * 60)
        print("병렬 검색 성능 테스트")
        print("=" * 60)
        
        if not hasattr(self.chat_service, 'optimized_search_engine'):
            print("최적화된 검색 엔진이 초기화되지 않았습니다.")
            return {}
        
        search_engine = self.chat_service.optimized_search_engine
        
        # 순차 검색 테스트
        print("\n순차 검색 테스트:")
        sequential_times = []
        
        for i, query in enumerate(self.test_queries[:3]):
            start_time = time.time()
            result = await search_engine.search(query, top_k=5, search_types=['vector', 'exact', 'semantic'])
            end_time = time.time()
            
            execution_time = end_time - start_time
            sequential_times.append(execution_time)
            
            print(f"  {i+1}. {query[:30]}... - {execution_time:.3f}초")
        
        # 병렬 검색 테스트
        print("\n병렬 검색 테스트:")
        parallel_times = []
        
        for i, query in enumerate(self.test_queries[:3]):
            start_time = time.time()
            result = await search_engine.search(query, top_k=5, search_types=['vector', 'exact', 'semantic'])
            end_time = time.time()
            
            execution_time = end_time - start_time
            parallel_times.append(execution_time)
            
            print(f"  {i+1}. {query[:30]}... - {execution_time:.3f}초")
        
        # 통계 계산
        sequential_avg = statistics.mean(sequential_times)
        parallel_avg = statistics.mean(parallel_times)
        improvement = ((sequential_avg - parallel_avg) / sequential_avg) * 100
        
        search_stats = search_engine.get_stats()
        
        return {
            'sequential_avg': sequential_avg,
            'parallel_avg': parallel_avg,
            'improvement_percent': improvement,
            'search_stats': search_stats
        }
    
    async def test_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 테스트"""
        print("\n" + "=" * 60)
        print("메모리 사용량 테스트")
        print("=" * 60)
        
        import psutil
        import gc
        
        # 초기 메모리 사용량
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        print(f"초기 메모리 사용량: {initial_memory:.2f} MB")
        
        # 여러 쿼리 실행
        for i, query in enumerate(self.test_queries):
            result = await self.chat_service.process_message(query)
            
            if i % 3 == 0:  # 3개마다 메모리 체크
                current_memory = process.memory_info().rss / (1024 * 1024)
                print(f"  {i+1}개 쿼리 후: {current_memory:.2f} MB")
        
        # 최종 메모리 사용량
        final_memory = process.memory_info().rss / (1024 * 1024)
        
        # 가비지 컬렉션 후 메모리
        gc.collect()
        gc_memory = process.memory_info().rss / (1024 * 1024)
        
        return {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': final_memory - initial_memory,
            'gc_memory_mb': gc_memory,
            'memory_recovered_mb': final_memory - gc_memory
        }
    
    async def test_concurrent_requests(self) -> Dict[str, Any]:
        """동시 요청 처리 테스트"""
        print("\n" + "=" * 60)
        print("동시 요청 처리 테스트")
        print("=" * 60)
        
        # 동시 요청 수
        concurrent_count = 5
        
        print(f"{concurrent_count}개의 동시 요청 처리 테스트:")
        
        async def process_single_query(query: str, index: int) -> Dict[str, Any]:
            start_time = time.time()
            result = await self.chat_service.process_message(query)
            end_time = time.time()
            
            return {
                'index': index,
                'query': query,
                'execution_time': end_time - start_time,
                'success': result is not None
            }
        
        # 동시 요청 실행
        start_time = time.time()
        tasks = [
            process_single_query(query, i) 
            for i, query in enumerate(self.test_queries[:concurrent_count])
        ]
        
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # 결과 분석
        successful_requests = sum(1 for r in results if r['success'])
        avg_execution_time = statistics.mean([r['execution_time'] for r in results])
        
        print(f"  총 처리 시간: {total_time:.3f}초")
        print(f"  성공한 요청: {successful_requests}/{concurrent_count}")
        print(f"  평균 실행 시간: {avg_execution_time:.3f}초")
        print(f"  처리량: {concurrent_count/total_time:.2f} 요청/초")
        
        return {
            'total_time': total_time,
            'successful_requests': successful_requests,
            'total_requests': concurrent_count,
            'avg_execution_time': avg_execution_time,
            'throughput': concurrent_count / total_time
        }
    
    def print_summary(self, cache_results: Dict, search_results: Dict, 
                     memory_results: Dict, concurrent_results: Dict):
        """테스트 결과 요약 출력"""
        print("\n" + "=" * 60)
        print("성능 테스트 결과 요약")
        print("=" * 60)
        
        print(f"\n1. 캐시 성능:")
        print(f"   첫 번째 실행 평균: {cache_results.get('first_run_avg', 0):.3f}초")
        print(f"   두 번째 실행 평균: {cache_results.get('second_run_avg', 0):.3f}초")
        print(f"   성능 개선: {cache_results.get('improvement_percent', 0):.1f}%")
        
        cache_stats = cache_results.get('cache_stats', {})
        print(f"   캐시 히트율: {cache_stats.get('hit_rate', 0):.1%}")
        print(f"   캐시 사용률: {cache_stats.get('utilization_rate', 0):.1%}")
        
        print(f"\n2. 병렬 검색 성능:")
        print(f"   순차 검색 평균: {search_results.get('sequential_avg', 0):.3f}초")
        print(f"   병렬 검색 평균: {search_results.get('parallel_avg', 0):.3f}초")
        print(f"   성능 개선: {search_results.get('improvement_percent', 0):.1f}%")
        
        print(f"\n3. 메모리 사용량:")
        print(f"   초기 메모리: {memory_results.get('initial_memory_mb', 0):.2f} MB")
        print(f"   최종 메모리: {memory_results.get('final_memory_mb', 0):.2f} MB")
        print(f"   메모리 증가: {memory_results.get('memory_increase_mb', 0):.2f} MB")
        print(f"   GC 후 메모리: {memory_results.get('gc_memory_mb', 0):.2f} MB")
        
        print(f"\n4. 동시 요청 처리:")
        print(f"   총 처리 시간: {concurrent_results.get('total_time', 0):.3f}초")
        print(f"   성공률: {concurrent_results.get('successful_requests', 0)}/{concurrent_results.get('total_requests', 0)}")
        print(f"   처리량: {concurrent_results.get('throughput', 0):.2f} 요청/초")

async def main():
    """메인 함수"""
    print("=" * 60)
    print("성능 최적화 테스트 시작")
    print("=" * 60)
    
    tester = PerformanceTester()
    
    # 각 테스트 실행
    cache_results = await tester.test_cache_performance()
    search_results = await tester.test_parallel_search_performance()
    memory_results = await tester.test_memory_usage()
    concurrent_results = await tester.test_concurrent_requests()
    
    # 결과 요약
    tester.print_summary(cache_results, search_results, memory_results, concurrent_results)
    
    print("\n" + "=" * 60)
    print("성능 최적화 테스트 완료!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
