#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LawFirmAI 테스트 실행 스크립트
테스트 카테고리별 실행 및 관리
"""

import subprocess
import sys
import argparse
import os
from pathlib import Path
from typing import List, Optional

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent
TESTS_DIR = Path(__file__).parent

# 테스트 카테고리 정의
TEST_CATEGORIES = {
    "unit": "단위 테스트",
    "integration": "통합 테스트", 
    "performance": "성능 테스트",
    "quality": "품질 테스트",
    "memory": "메모리 테스트",
    "classification": "분류 시스템 테스트",
    "legal_systems": "법률 시스템 테스트",
    "contracts": "계약 관련 테스트",
    "external_integrations": "외부 시스템 통합 테스트",
    "conversational": "대화 관련 테스트",
    "database": "데이터베이스 테스트",
    "demos": "데모 및 예제 테스트",
    "regression": "회귀 테스트"
}


def run_tests(
    category: Optional[str] = None,
    verbose: bool = False,
    coverage: bool = False,
    parallel: bool = False,
    markers: Optional[List[str]] = None
) -> int:
    """
    테스트 실행
    
    Args:
        category: 테스트 카테고리
        verbose: 상세 출력 여부
        coverage: 커버리지 측정 여부
        parallel: 병렬 실행 여부
        markers: pytest 마커 필터
    
    Returns:
        테스트 실행 결과 코드
    """
    cmd = ["python", "-m", "pytest"]
    
    # 테스트 경로 설정
    if category:
        if category not in TEST_CATEGORIES:
            print(f"❌ 잘못된 카테고리: {category}")
            print(f"사용 가능한 카테고리: {', '.join(TEST_CATEGORIES.keys())}")
            return 1
        
        test_path = TESTS_DIR / category
        if not test_path.exists():
            print(f"❌ 테스트 경로가 존재하지 않습니다: {test_path}")
            return 1
        
        cmd.append(str(test_path))
    else:
        cmd.append(str(TESTS_DIR))
    
    # 옵션 설정
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=source", "--cov-report=html", "--cov-report=term"])
    
    if parallel:
        cmd.extend(["-n", "auto"])
    
    if markers:
        for marker in markers:
            cmd.extend(["-m", marker])
    
    # 기본 옵션
    cmd.extend([
        "--tb=short",
        "--strict-markers",
        "--disable-warnings"
    ])
    
    print(f"🚀 테스트 실행 중: {' '.join(cmd)}")
    print(f"📁 작업 디렉토리: {PROJECT_ROOT}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=False
        )
        return result.returncode
    except KeyboardInterrupt:
        print("\n⏹️ 테스트 실행이 중단되었습니다.")
        return 1
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        return 1


def list_categories():
    """사용 가능한 테스트 카테고리 목록 출력"""
    print("📋 사용 가능한 테스트 카테고리:")
    print()
    
    for category, description in TEST_CATEGORIES.items():
        test_path = TESTS_DIR / category
        if test_path.exists():
            test_files = list(test_path.glob("test_*.py"))
            file_count = len(test_files)
            status = "✅" if file_count > 0 else "⚠️"
            print(f"  {status} {category:<20} - {description} ({file_count}개 파일)")
        else:
            print(f"  ❌ {category:<20} - {description} (폴더 없음)")


def run_specific_test(test_file: str):
    """특정 테스트 파일 실행"""
    test_path = TESTS_DIR / test_file
    if not test_path.exists():
        print(f"❌ 테스트 파일이 존재하지 않습니다: {test_path}")
        return 1
    
    cmd = ["python", "-m", "pytest", str(test_path), "-v"]
    
    print(f"🚀 특정 테스트 실행: {test_file}")
    
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⏹️ 테스트 실행이 중단되었습니다.")
        return 1


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="LawFirmAI 테스트 실행 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python run_tests.py                    # 전체 테스트 실행
  python run_tests.py unit               # 단위 테스트만 실행
  python run_tests.py integration -v     # 통합 테스트 상세 출력
  python run_tests.py performance --coverage  # 성능 테스트 + 커버리지
  python run_tests.py --list             # 카테고리 목록 출력
  python run_tests.py test_file.py       # 특정 테스트 파일 실행
        """
    )
    
    parser.add_argument(
        "target",
        nargs="?",
        help="테스트 카테고리 또는 특정 테스트 파일"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="상세 출력"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true", 
        help="코드 커버리지 측정"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="병렬 실행 (pytest-xdist 필요)"
    )
    
    parser.add_argument(
        "-m", "--markers",
        nargs="+",
        help="pytest 마커 필터"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="사용 가능한 카테고리 목록 출력"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_categories()
        return 0
    
    if not args.target:
        # 전체 테스트 실행
        return run_tests(verbose=args.verbose, coverage=args.coverage, 
                        parallel=args.parallel, markers=args.markers)
    
    # 특정 테스트 파일인지 확인
    if args.target.endswith(".py"):
        return run_specific_test(args.target)
    
    # 카테고리별 테스트 실행
    return run_tests(
        category=args.target,
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel,
        markers=args.markers
    )


if __name__ == "__main__":
    sys.exit(main())
