#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""precedents 활성 버전을 1로 설정"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 환경 변수 로드
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(_PROJECT_ROOT)
except ImportError:
    pass
except Exception:
    pass

from scripts.ingest.open_law.embedding.pgvector.version_manager import PgEmbeddingVersionManager
from scripts.ingest.open_law.utils import build_database_url
import os

def main():
    print("=" * 80)
    print("precedents 활성 버전을 1로 설정")
    print("=" * 80)
    
    db_url = build_database_url() or os.getenv('DATABASE_URL')
    if not db_url:
        print("❌ 데이터베이스 URL을 찾을 수 없습니다.")
        return 1
    
    print(f"\n✅ 데이터베이스 연결: {db_url.split('@')[1] if '@' in db_url else '설정됨'}")
    
    version_manager = PgEmbeddingVersionManager(db_url)
    
    # 버전 정보 확인
    print("\n1. 버전 정보 확인")
    print("-" * 80)
    version_info = version_manager.get_version_info(version=1, data_type='precedents')
    if not version_info:
        print("   ❌ Version 1 (precedents)을 찾을 수 없습니다.")
        return 1
    
    print(f"   Version ID: {version_info['id']}")
    print(f"   Version: {version_info['version']}")
    print(f"   Model: {version_info.get('model_name', 'N/A')}")
    print(f"   Data Type: precedents")
    print(f"   Is Active: {version_info['is_active']}")
    
    if version_info['is_active']:
        print(f"\n   ✅ Version 1이 이미 활성화되어 있습니다!")
        return 0
    
    # 버전 활성화
    print("\n2. 버전 활성화")
    print("-" * 80)
    success = version_manager.set_active_version(version=1, data_type='precedents')
    
    if success:
        print(f"   ✅ Version 1 활성화 완료!")
        print(f"   - 기존 활성 버전이 자동으로 비활성화되었습니다.")
    else:
        print(f"   ❌ Version 1 활성화 실패")
        return 1
    
    # 최종 확인
    print("\n3. 최종 확인")
    print("-" * 80)
    final_info = version_manager.get_version_info(version=1, data_type='precedents')
    if final_info and final_info['is_active']:
        print("   ✅ Version 활성화 확인 완료!")
        print(f"   - Version ID: {final_info['id']}")
        print(f"   - Version: {final_info['version']}")
        print(f"   - Model: {final_info.get('model_name', 'N/A')}")
        print(f"   - Data Type: precedents")
    else:
        print("   ❌ Version 활성화 확인 실패")
        return 1
    
    print("\n" + "=" * 80)
    print("✅ 완료")
    print("=" * 80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

