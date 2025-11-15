# -*- coding: utf-8 -*-
"""
모든 테스트 실행 스크립트
전체 테스트를 실행하고 결과를 리포트합니다
Windows 환경에서도 안정적으로 실행됩니다.
"""

import sys
import os
import subprocess
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# lawfirm_langgraph 디렉토리를 sys.path에 추가
lawfirm_langgraph_path = Path(__file__).parent.parent
sys.path.insert(0, str(lawfirm_langgraph_path))


def run_all_tests(use_subprocess=True):
    """모든 테스트 실행"""
    print("=" * 80)
    print("LawFirm LangGraph 테스트 실행")
    print("=" * 80)
    print()
    
    # 테스트 디렉토리 경로
    test_dir = Path(__file__).parent
    
    # pytest 실행 옵션
    pytest_args = [
        "-m", "pytest",
        str(test_dir),
        "-v",  # 상세 출력
        "--tb=short",  # 짧은 트레이스백
        "--color=yes",  # 컬러 출력
        "--durations=10",  # 가장 느린 10개 테스트 표시
        "-s",  # 출력 버퍼링 비활성화 (Windows 버퍼 이슈 해결)
        "--capture=no"  # 캡처 비활성화 (Windows 버퍼 이슈 해결)
    ]
    
    # 커버리지 옵션 (pytest-cov가 설치된 경우)
    import importlib.util
    if importlib.util.find_spec("pytest_cov") is not None:
        pytest_args.extend([
            "--cov=lawfirm_langgraph",
            "--cov-report=term-missing",
            "--cov-report=html",
        ])
    else:
        print("주의: pytest-cov가 설치되지 않았습니다. 커버리지 리포트를 생성하지 않습니다.")
        print("      설치하려면: pip install pytest-cov")
        print()
    
    # Windows 버퍼 이슈를 피하기 위해 subprocess 사용 권장
    if use_subprocess:
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'  # Python 출력 버퍼링 비활성화
        
        cmd = [sys.executable] + pytest_args
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        exit_code = result.returncode
    else:
        # pytest.main() 사용 (Windows에서 버퍼 이슈 발생 가능)
        import importlib.util
        pytest_spec = importlib.util.find_spec("pytest")
        if pytest_spec is None:
            print("❌ pytest가 설치되지 않았습니다!")
            return 1
        import pytest
        exit_code = pytest.main(pytest_args)
    
    print()
    print("=" * 80)
    if exit_code == 0:
        print("✅ 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print(f"❌ 일부 테스트가 실패했습니다. (exit code: {exit_code})")
    print("=" * 80)
    
    return exit_code


def run_specific_test(test_name: str, use_subprocess=True):
    """특정 테스트 실행"""
    test_dir = Path(__file__).parent
    test_file = test_dir / f"test_{test_name}.py"
    
    if not test_file.exists():
        print(f"❌ 테스트 파일을 찾을 수 없습니다: {test_file}")
        return 1
    
    pytest_args = [
        "-m", "pytest",
        str(test_file),
        "-v",
        "--tb=short",
        "--color=yes",
        "-s",  # 출력 버퍼링 비활성화
        "--capture=no"  # 캡처 비활성화
    ]
    
    if use_subprocess:
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        cmd = [sys.executable] + pytest_args
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        return result.returncode
    else:
        import importlib.util
        pytest_spec = importlib.util.find_spec("pytest")
        if pytest_spec is None:
            print("❌ pytest가 설치되지 않았습니다!")
            return 1
        import pytest
        return pytest.main(pytest_args)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 특정 테스트 실행
        test_name = sys.argv[1]
        exit_code = run_specific_test(test_name)
    else:
        # 모든 테스트 실행
        exit_code = run_all_tests()
    
    sys.exit(exit_code)

