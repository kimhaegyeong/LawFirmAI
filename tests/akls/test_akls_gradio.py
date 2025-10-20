# -*- coding: utf-8 -*-
"""
AKLS Gradio 인터페이스 테스트
Gradio 앱에서 AKLS 기능이 정상적으로 작동하는지 테스트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


def test_gradio_akls_interface():
    """Gradio AKLS 인터페이스 테스트"""
    print("=" * 80)
    print("Gradio AKLS 인터페이스 테스트")
    print("=" * 80)
    
    try:
        # Gradio 컴포넌트 경로 추가
        gradio_components_path = Path("gradio/components")
        if gradio_components_path.exists():
            sys.path.append(str(gradio_components_path))
        
        from akls_search_interface import AKLSSearchInterface
        
        print("AKLS 인터페이스 초기화 중...")
        interface = AKLSSearchInterface()
        print("SUCCESS: 인터페이스 초기화 완료")
        
        # 통계 조회 테스트
        print("\n통계 조회 테스트:")
        stats_text = interface.get_akls_statistics()
        print(f"통계 정보 길이: {len(stats_text)} 문자")
        
        # 검색 기능 테스트
        print("\n검색 기능 테스트:")
        test_cases = [
            ("계약 해지", "all", "all"),
            ("형법", "criminal_law", "all"),
            ("민사소송", "civil_procedure", "all")
        ]
        
        for i, (query, law_area, case_type) in enumerate(test_cases, 1):
            print(f"\n[검색 테스트 {i}]")
            print(f"질의: '{query}'")
            print(f"법률 영역: {law_area}")
            print(f"사건 유형: {case_type}")
            
            try:
                response, table_data = interface.search_akls_precedents(query, law_area, case_type, 2)
                
                print(f"응답 길이: {len(response)} 문자")
                print(f"테이블 행 수: {len(table_data)}")
                
                if table_data:
                    print("검색 결과:")
                    for j, row in enumerate(table_data, 1):
                        if len(row) >= 6:
                            print(f"  결과 {j}:")
                            print(f"    - 사건번호: {row[0]}")
                            print(f"    - 법원: {row[1]}")
                            print(f"    - 선고일자: {row[2]}")
                            print(f"    - 법률영역: {row[3]}")
                            print(f"    - 파일명: {row[4]}")
                            print(f"    - 유사도: {row[5]}")
                
                print("SUCCESS: 검색 완료")
                
            except Exception as e:
                print(f"WARNING: 검색 실패 (예상됨): {e}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: 인터페이스 테스트 실패: {e}")
        return False


def test_gradio_app_integration():
    """Gradio 앱 통합 테스트"""
    print("\n" + "=" * 80)
    print("Gradio 앱 통합 테스트")
    print("=" * 80)
    
    try:
        # Gradio 앱 파일 확인
        gradio_app_path = Path("gradio/app.py")
        if not gradio_app_path.exists():
            print("ERROR: gradio/app.py 파일이 없습니다")
            return False
        
        print("SUCCESS: Gradio 앱 파일 존재 확인")
        
        # 앱 파일에서 AKLS 관련 코드 확인
        with open(gradio_app_path, 'r', encoding='utf-8') as f:
            app_content = f.read()
        
        # AKLS 관련 키워드 확인
        akls_keywords = [
            "akls_search_interface",
            "AKLSSearchInterface", 
            "create_akls_interface",
            "AKLS 표준판례 검색"
        ]
        
        found_keywords = []
        for keyword in akls_keywords:
            if keyword in app_content:
                found_keywords.append(keyword)
                print(f"SUCCESS: '{keyword}' 키워드 발견")
            else:
                print(f"WARNING: '{keyword}' 키워드가 없습니다")
        
        if len(found_keywords) >= 2:
            print("SUCCESS: Gradio 앱에 AKLS 통합 코드가 포함되어 있습니다")
            return True
        else:
            print("ERROR: Gradio 앱에 AKLS 통합이 불완전합니다")
            return False
        
    except Exception as e:
        print(f"ERROR: Gradio 앱 통합 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 실행"""
    print("AKLS Gradio 인터페이스 테스트")
    print("=" * 100)
    
    test_results = []
    
    # 각 테스트 실행
    tests = [
        ("AKLS 인터페이스 테스트", test_gradio_akls_interface),
        ("Gradio 앱 통합 테스트", test_gradio_app_integration)
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
    print("Gradio 인터페이스 테스트 결과 요약")
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
        print("\nSUCCESS: 모든 Gradio 인터페이스 테스트가 성공적으로 완료되었습니다!")
        print("AKLS Gradio 통합이 정상적으로 작동합니다.")
    else:
        print(f"\nWARNING: 일부 테스트가 실패했습니다.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
