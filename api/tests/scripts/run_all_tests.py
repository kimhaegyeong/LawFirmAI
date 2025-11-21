"""
전체 테스트 실행 스크립트
"""
import subprocess
import sys
from pathlib import Path


def main():
    """전체 테스트 실행"""
    test_dir = Path(__file__).parent.parent
    project_root = test_dir.parent.parent
    
    # pytest 실행
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_dir), "-v"],
        cwd=project_root
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

