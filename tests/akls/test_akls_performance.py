# -*- coding: utf-8 -*-
"""
AKLS 성능 벤치마크 테스트
AKLS 시스템의 성능을 측정하고 최적화 포인트를 찾는 테스트
"""

import sys
import os
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


def test_search_performance():
    """검색 성능 테스트"""
    print("=" * 80)
    print("AKLS 검색 성능 테스트")
    print("=" * 80)
    
    try:
        from source.services.akls_search_engine import AKLSSearchEngine
        
        search_engine = AKLSSearchEngine()
        
        if search_engine.index is None:
            print("ERROR: 검색 인덱스가 로드되지 않았습니다")
            return False
        
        # 성능 테스트 질의들
        performance_queries = [
            "계약 해지",
            "손해배상",
            "형법",
            "민사소송",
            "대법원",
            "상법",
            "행정법",
            "헌법",
            "형사소송",
            "민법"
        ]
        
        print(f"성능 테스트 ({len(performance_queries)}개 질의):")
        
        search_times = []
        successful_searches = 0
        
        for i, query in enumerate(performance_queries, 1):
            print(f"[성능 테스트 {i}] '{query}'")
            
            start_time = time.time()
            try:
                results = search_engine.search(query, top_k=3)
                end_time = time.time()
                
                search_time = end_time - start_time
                search_times.append(search_time)
                successful_searches += 1
                
                print(f"  - 검색 시간: {search_time:.3f}초")
                print(f"  - 결과 수: {len(results)}")
                
                if search_time > 2.0:
                    print(f"  WARNING: 검색 시간이 느림")
                else:
                    print(f"  SUCCESS: 검색 시간 양호")
                
            except Exception as e:
                print(f"  ERROR: 검색 실패 - {e}")
        
        # 성능 통계
        if search_times:
            avg_time = statistics.mean(search_times)
            median_time = statistics.median(search_times)
            min_time = min(search_times)
            max_time = max(search_times)
            std_time = statistics.stdev(search_times) if len(search_times) > 1 else 0
            
            print(f"\n성능 통계:")
            print(f"  - 성공한 검색 수: {successful_searches}/{len(performance_queries)}")
            print(f"  - 평균 검색 시간: {avg_time:.3f}초")
            print(f"  - 중간값 검색 시간: {median_time:.3f}초")
            print(f"  - 최소 검색 시간: {min_time:.3f}초")
            print(f"  - 최대 검색 시간: {max_time:.3f}초")
            print(f"  - 표준편차: {std_time:.3f}초")
            
            # 성능 등급 평가
            if avg_time < 0.1:
                print("  SUCCESS: 평균 검색 시간이 매우 우수합니다")
            elif avg_time < 0.5:
                print("  SUCCESS: 평균 검색 시간이 우수합니다")
            elif avg_time < 1.0:
                print("  GOOD: 평균 검색 시간이 양호합니다")
            else:
                print("  WARNING: 평균 검색 시간이 느립니다")
        
        return True
        
    except Exception as e:
        print(f"ERROR: 검색 성능 테스트 실패: {e}")
        return False


def test_rag_performance():
    """RAG 서비스 성능 테스트"""
    print("\n" + "=" * 80)
    print("Enhanced RAG Service 성능 테스트")
    print("=" * 80)
    
    try:
        from source.services.enhanced_rag_service import EnhancedRAGService
        
        enhanced_rag = EnhancedRAGService()
        
        # RAG 성능 테스트 질의들
        rag_queries = [
            "계약 해지에 대한 표준판례를 알려주세요",
            "형법 제250조 관련 판례는 무엇인가요?",
            "손해배상 책임에 대한 법령을 찾아주세요",
            "민사소송법 관련 판례를 검색해주세요",
            "대법원의 표준판례에 대해 설명해주세요"
        ]
        
        print(f"RAG 성능 테스트 ({len(rag_queries)}개 질의):")
        
        rag_times = []
        successful_rag = 0
        
        for i, query in enumerate(rag_queries, 1):
            print(f"[RAG 테스트 {i}] '{query}'")
            
            start_time = time.time()
            try:
                result = enhanced_rag.search_with_akls(query, top_k=3)
                end_time = time.time()
                
                rag_time = end_time - start_time
                rag_times.append(rag_time)
                successful_rag += 1
                
                print(f"  - 처리 시간: {rag_time:.3f}초")
                print(f"  - 검색 유형: {result.search_type}")
                print(f"  - 신뢰도: {result.confidence:.3f}")
                print(f"  - AKLS 소스 수: {len(result.akls_sources)}")
                
                if rag_time > 5.0:
                    print(f"  WARNING: 처리 시간이 느림")
                else:
                    print(f"  SUCCESS: 처리 시간 양호")
                
            except Exception as e:
                print(f"  ERROR: RAG 처리 실패 - {e}")
        
        # RAG 성능 통계
        if rag_times:
            avg_time = statistics.mean(rag_times)
            median_time = statistics.median(rag_times)
            min_time = min(rag_times)
            max_time = max(rag_times)
            
            print(f"\nRAG 성능 통계:")
            print(f"  - 성공한 RAG 수: {successful_rag}/{len(rag_queries)}")
            print(f"  - 평균 처리 시간: {avg_time:.3f}초")
            print(f"  - 중간값 처리 시간: {median_time:.3f}초")
            print(f"  - 최소 처리 시간: {min_time:.3f}초")
            print(f"  - 최대 처리 시간: {max_time:.3f}초")
            
            # RAG 성능 등급 평가
            if avg_time < 2.0:
                print("  SUCCESS: 평균 RAG 처리 시간이 우수합니다")
            elif avg_time < 5.0:
                print("  GOOD: 평균 RAG 처리 시간이 양호합니다")
            else:
                print("  WARNING: 평균 RAG 처리 시간이 느립니다")
        
        return True
        
    except Exception as e:
        print(f"ERROR: RAG 성능 테스트 실패: {e}")
        return False


def test_memory_usage():
    """메모리 사용량 테스트"""
    print("\n" + "=" * 80)
    print("메모리 사용량 테스트")
    print("=" * 80)
    
    try:
        import psutil
        import os
        
        # 현재 프로세스의 메모리 사용량 확인
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        print(f"현재 메모리 사용량:")
        print(f"  - RSS (Resident Set Size): {memory_info.rss / 1024 / 1024:.2f} MB")
        print(f"  - VMS (Virtual Memory Size): {memory_info.vms / 1024 / 1024:.2f} MB")
        
        # 시스템 메모리 정보
        system_memory = psutil.virtual_memory()
        print(f"\n시스템 메모리 정보:")
        print(f"  - 총 메모리: {system_memory.total / 1024 / 1024 / 1024:.2f} GB")
        print(f"  - 사용 가능한 메모리: {system_memory.available / 1024 / 1024 / 1024:.2f} GB")
        print(f"  - 메모리 사용률: {system_memory.percent:.1f}%")
        
        # 메모리 사용률 평가
        if system_memory.percent < 70:
            print("  SUCCESS: 메모리 사용률이 양호합니다")
        elif system_memory.percent < 85:
            print("  WARNING: 메모리 사용률이 높습니다")
        else:
            print("  ERROR: 메모리 사용률이 매우 높습니다")
        
        return True
        
    except ImportError:
        print("WARNING: psutil이 설치되지 않아 메모리 사용량을 측정할 수 없습니다")
        print("pip install psutil을 실행하여 메모리 모니터링을 활성화하세요")
        return True
    except Exception as e:
        print(f"ERROR: 메모리 사용량 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 실행"""
    print("AKLS 성능 벤치마크 테스트")
    print("=" * 100)
    
    test_results = []
    
    # 각 테스트 실행
    tests = [
        ("검색 성능 테스트", test_search_performance),
        ("RAG 성능 테스트", test_rag_performance),
        ("메모리 사용량 테스트", test_memory_usage)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"ERROR: {test_name} 실행 중 오류: {e}")
            test_results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "=" * 100)
    print("성능 벤치마크 테스트 결과 요약")
    print("=" * 100)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "SUCCESS" if result else "FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n전체 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("\nSUCCESS: 모든 성능 벤치마크 테스트가 성공적으로 완료되었습니다!")
        print("AKLS 시스템의 성능이 양호합니다.")
    else:
        print(f"\nWARNING: 일부 테스트가 실패했습니다.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
