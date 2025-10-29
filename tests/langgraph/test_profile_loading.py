# -*- coding: utf-8 -*-
"""
프로필 로드 기능 테스트
"""

import os
import sys
import tempfile
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.langgraph.monitoring_switch import MonitoringSwitch


def test_profile_loading():
    """프로필 로드 테스트"""
    print("="*80)
    print("프로필 로드 기능 테스트")
    print("="*80)

    # 임시 프로필 파일 생성
    profiles_dir = project_root / ".env.profiles"
    profiles_dir.mkdir(exist_ok=True)

    test_profile_content = """
# 테스트 프로필
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=test-api-key-12345
LANGCHAIN_PROJECT=test-project
LANGFUSE_ENABLED=false
ENABLE_LANGSMITH=true
"""

    test_profile_path = profiles_dir / "test_profile.env"
    with open(test_profile_path, 'w', encoding='utf-8') as f:
        f.write(test_profile_content)

    # 프로필 로드
    env_vars = MonitoringSwitch.load_profile("test_profile")

    # 검증
    assert len(env_vars) == 5, f"프로필에서 5개 환경변수를 읽어야 함: {len(env_vars)}"
    assert env_vars.get("LANGCHAIN_TRACING_V2") == "true"
    assert env_vars.get("LANGCHAIN_API_KEY") == "test-api-key-12345"
    assert env_vars.get("LANGCHAIN_PROJECT") == "test-project"
    assert env_vars.get("LANGFUSE_ENABLED") == "false"
    assert env_vars.get("ENABLE_LANGSMITH") == "true"

    print("✅ 프로필 로드 성공")
    print(f"   로드된 환경변수: {list(env_vars.keys())}")

    # 존재하지 않는 프로필 테스트
    non_existent = MonitoringSwitch.load_profile("non_existent_profile")
    assert len(non_existent) == 0, "존재하지 않는 프로필은 빈 딕셔너리 반환"
    print("✅ 존재하지 않는 프로필 처리 확인")

    # 정리
    if test_profile_path.exists():
        test_profile_path.unlink()

    return True


if __name__ == "__main__":
    try:
        result = test_profile_loading()
        print("\n" + "="*80)
        if result:
            print("🎉 프로필 로드 테스트 통과!")
        else:
            print("❌ 프로필 로드 테스트 실패")
        print("="*80)
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
