#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
테스트 커버리지 측정 스크립트
pytest-cov를 사용하여 테스트 커버리지를 측정합니다.
Windows 환경에서도 안정적으로 실행됩니다.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_coverage():
    """테스트 커버리지 실행"""
    project_root = Path(__file__).parent.parent.parent
    lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
    
    os.chdir(project_root)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "lawfirm_langgraph/tests/",
        "--cov=lawfirm_langgraph",
        "--cov-report=html:lawfirm_langgraph/htmlcov",
        "--cov-report=term-missing",
        "--cov-report=xml:lawfirm_langgraph/coverage.xml",
        "-v",
        "--tb=short",
        "-s",  # 출력 버퍼링 비활성화 (Windows 버퍼 이슈 해결)
        "--capture=no"  # 캡처 비활성화 (Windows 버퍼 이슈 해결)
    ]
    
    print("Running coverage analysis...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'  # Python 출력 버퍼링 비활성화
    
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    if result.returncode == 0:
        print("\n✅ Coverage analysis completed successfully!")
        print(f"📊 HTML report: {lawfirm_langgraph_path / 'htmlcov' / 'index.html'}")
        print(f"📄 XML report: {lawfirm_langgraph_path / 'coverage.xml'}")
    else:
        print("\n❌ Coverage analysis failed!")
        sys.exit(result.returncode)

if __name__ == "__main__":
    run_coverage()

