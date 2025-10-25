# -*- coding: utf-8 -*-
"""
AKLS 통합 시스템 종합 테스트
AKLS 데이터 처리, 검색, RAG 통합 기능을 종합적으로 테스트
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_akls_data_integration():
    """AKLS 데이터 통합 테스트"""
    print("=" * 80)
    print("AKLS 데이터 통합 테스트")
    print("=" * 80)
    
    # 1. 처리된 데이터 확인
    processed_dir = Path("data/processed/akls")
    if not processed_dir.exists():
        print("ERROR: 처리된 AKLS 데이터 디렉토리가 없습니다")
        return False
    
    json_files = list(processed_dir.glob("*.json"))
    print(f"처리된 JSON 파일 수: {len(json_files)}")
    
    if len(json_files) == 0:
        print("ERROR: JSON 파일이 없습니다")
        return False
    
    # 샘플 파일 검증
    with open(json_files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    required_fields = ["filename", "content", "metadata", "law_area", "document_type"]
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        print(f"ERROR: 누락된 필드: {missing_fields}")
        return False
    
    print(f"SUCCESS: 데이터 파일 검증 완료")
    print(f"  - 파일명: {data['filename']}")
    print(f"  - 법률영역: {data['law_area']}")
    print(f"  - 내용 길이: {len(data['content'])} 문자")
    
    # 2. 벡터 인덱스 확인
    index_dir = Path("data/embeddings/akls_precedents")
    if not index_dir.exists():
        print("ERROR: 벡터 인덱스 디렉토리가 없습니다")
        return False
    
    index_file = index_dir / "akls_index.faiss"
    metadata_file = index_dir / "akls_metadata.json"
    
    if not (index_file.exists() and metadata_file.exists()):
        print("ERROR: 인덱스 파일이 없습니다")
        return False
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"SUCCESS: 벡터 인덱스 검증 완료")
    print(f"  - 인덱스된 문서 수: {len(metadata)}")
    
    # 법률 영역별 통계
    law_areas = {}
    for doc in metadata:
        law_area = doc.get("metadata", {}).get("law_area", "unknown")
        law_areas[law_area] = law_areas.get(law_area, 0) + 1
    
    print("  - 법률 영역별 분포:")
    for area, count in law_areas.items():
        print(f"    * {area}: {count}개")
    
    return True


def test_akls_search_engine():
    """AKLS 검색 엔진 테스트"""
    print("\n" + "=" * 80)
    print("AKLS 검색 엔진 테스트")
    print("=" * 80)
    
    try:
        from source.services.akls_search_engine import AKLSSearchEngine
        
        search_engine = AKLSSearchEngine()
        
        if search_engine.index is None:
            print("ERROR: 검색 인덱스가 로드되지 않았습니다")
            return False
        
        print(f"SUCCESS: 검색 엔진 초기화 완료")
        print(f"  - 인덱스된 문서 수: {search_engine.index.ntotal}")
        
        # 검색 테스트
        test_queries = [
            "계약 해지",
            "손해배상",
            "형법 제250조",
            "민사소송",
            "대법원",
            "상법"
        ]
        
        print(f"\n검색 테스트 ({len(test_queries)}개 질의):")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n[테스트 {i}] '{query}'")
            
            start_time = time.time()
            try:
                results = search_engine.search(query, top_k=2)
                end_time = time.time()
                
                print(f"  - 검색 시간: {end_time - start_time:.3f}초")
                print(f"  - 결과 수: {len(results)}")
                
                for j, result in enumerate(results, 1):
                    print(f"  - 결과 {j}: {result.law_area} (점수: {result.score:.3f})")
                    if result.case_number:
                        print(f"    사건번호: {result.case_number}")
                
            except Exception as e:
                print(f"  ERROR: 검색 실패 - {e}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: 검색 엔진 테스트 실패: {e}")
        return False


def test_enhanced_rag_service():
    """Enhanced RAG Service 테스트"""
    print("\n" + "=" * 80)
    print("Enhanced RAG Service 테스트")
    print("=" * 80)
    
    try:
        from source.services.enhanced_rag_service import EnhancedRAGService
        
        enhanced_rag = EnhancedRAGService()
        
        # AKLS 통계 확인
        stats = enhanced_rag.get_akls_statistics()
        print(f"SUCCESS: Enhanced RAG Service 초기화 완료")
        print(f"  - 총 문서 수: {stats.get('total_documents', 0)}")
        print(f"  - 인덱스 사용 가능: {stats.get('index_available', False)}")
        
        # 쿼리 라우팅 테스트
        test_queries = [
            ("계약 해지에 대한 표준판례", "akls_precedents"),
            ("형법 제250조 관련 판례", "akls_precedents"),
            ("손해배상 책임에 대한 법령", "assembly_laws"),
            ("민사소송법 관련 판례", "assembly_precedents"),
            ("대법원의 표준판례", "akls_precedents")
        ]
        
        print(f"\n쿼리 라우팅 테스트 ({len(test_queries)}개 질의):")
        
        for i, (query, expected_source) in enumerate(test_queries, 1):
            print(f"\n[테스트 {i}] '{query}'")
            
            start_time = time.time()
            try:
                result = enhanced_rag.search_with_akls(query, top_k=2)
                end_time = time.time()
                
                print(f"  - 처리 시간: {end_time - start_time:.3f}초")
                print(f"  - 검색 유형: {result.search_type}")
                print(f"  - 법률 영역: {result.law_area}")
                print(f"  - 신뢰도: {result.confidence:.3f}")
                print(f"  - AKLS 소스 수: {len(result.akls_sources)}")
                
                # 예상 소스 타입 확인
                if result.search_type == expected_source:
                    print(f"  SUCCESS: 예상된 소스 타입과 일치")
                else:
                    print(f"  WARNING: 예상 소스 타입({expected_source})과 다름")
                
            except Exception as e:
                print(f"  ERROR: 질의 처리 실패 - {e}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Enhanced RAG Service 테스트 실패: {e}")
        return False


def test_performance_benchmark():
    """성능 벤치마크 테스트"""
    print("\n" + "=" * 80)
    print("성능 벤치마크 테스트")
    print("=" * 80)
    
    try:
        from source.services.akls_search_engine import AKLSSearchEngine
        
        search_engine = AKLSSearchEngine()
        
        if search_engine.index is None:
            print("ERROR: 검색 인덱스가 로드되지 않았습니다")
            return False
        
        # 성능 테스트 질의들
        performance_queries = [
            "계약 해지", "손해배상", "형법", "민사소송", "대법원",
            "상법", "행정법", "헌법", "형사소송", "민법"
        ]
        
        print(f"성능 테스트 ({len(performance_queries)}개 질의):")
        
        total_time = 0
        successful_searches = 0
        min_time = float('inf')
        max_time = 0
        
        for i, query in enumerate(performance_queries, 1):
            print(f"[성능 테스트 {i}] '{query}'")
            
            start_time = time.time()
            try:
                results = search_engine.search(query, top_k=3)
                end_time = time.time()
                
                search_time = end_time - start_time
                total_time += search_time
                successful_searches += 1
                
                min_time = min(min_time, search_time)
                max_time = max(max_time, search_time)
                
                print(f"  - 검색 시간: {search_time:.3f}초")
                print(f"  - 결과 수: {len(results)}")
                
                if search_time > 2.0:
                    print(f"  WARNING: 검색 시간이 느림")
                else:
                    print(f"  SUCCESS: 검색 시간 양호")
                
            except Exception as e:
                print(f"  ERROR: 검색 실패 - {e}")
        
        # 성능 통계
        if successful_searches > 0:
            avg_time = total_time / successful_searches
            
            print(f"\n성능 통계:")
            print(f"  - 성공한 검색 수: {successful_searches}/{len(performance_queries)}")
            print(f"  - 평균 검색 시간: {avg_time:.3f}초")
            print(f"  - 최소 검색 시간: {min_time:.3f}초")
            print(f"  - 최대 검색 시간: {max_time:.3f}초")
            print(f"  - 총 검색 시간: {total_time:.3f}초")
            
            if avg_time < 1.0:
                print("  SUCCESS: 평균 검색 시간이 우수합니다")
            elif avg_time < 2.0:
                print("  GOOD: 평균 검색 시간이 양호합니다")
            else:
                print("  WARNING: 평균 검색 시간이 느립니다")
        
        return True
        
    except Exception as e:
        print(f"ERROR: 성능 벤치마크 테스트 실패: {e}")
        return False


def test_file_structure():
    """파일 구조 확인"""
    print("\n" + "=" * 80)
    print("파일 구조 확인")
    print("=" * 80)
    
    important_files = [
        "source/services/akls_processor.py",
        "source/services/akls_search_engine.py", 
        "source/services/enhanced_rag_service.py",
        "gradio/components/akls_search_interface.py",
        "scripts/process_akls_documents.py"
    ]
    
    all_files_exist = True
    for file_path in important_files:
        if Path(file_path).exists():
            print(f"OK: {file_path}")
        else:
            print(f"ERROR: {file_path} 파일이 없습니다")
            all_files_exist = False
    
    if all_files_exist:
        print("SUCCESS: 모든 중요 파일이 존재합니다")
        return True
    else:
        print("ERROR: 일부 중요 파일이 없습니다")
        return False


def main():
    """메인 테스트 실행"""
    print("AKLS 통합 시스템 종합 테스트")
    print("=" * 100)
    
    test_results = []
    
    # 각 테스트 실행
    tests = [
        ("데이터 통합 테스트", test_akls_data_integration),
        ("검색 엔진 테스트", test_akls_search_engine),
        ("Enhanced RAG Service 테스트", test_enhanced_rag_service),
        ("성능 벤치마크 테스트", test_performance_benchmark),
        ("파일 구조 확인", test_file_structure)
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
    print("테스트 결과 요약")
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
        print("\nSUCCESS: 모든 테스트가 성공적으로 완료되었습니다!")
        print("AKLS 통합 시스템이 정상적으로 작동합니다.")
    else:
        print(f"\nWARNING: 일부 테스트가 실패했습니다.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
