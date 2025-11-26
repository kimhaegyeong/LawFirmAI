#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
버전 활성화 스크립트 (범용)

사용법:
    python scripts/embedding/activate_version.py \
        --version 3 \
        --data-type statutes
"""

import os
import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[1]
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

# 모듈 임포트
try:
    from scripts.ingest.open_law.embedding.pgvector.version_manager import PgEmbeddingVersionManager
    from scripts.ingest.open_law.utils import build_database_url
except ImportError as e:
    print(f"❌ 필수 모듈을 불러올 수 없습니다: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='버전 활성화 스크립트')
    parser.add_argument(
        '--version',
        type=int,
        required=True,
        help='버전 번호'
    )
    parser.add_argument(
        '--data-type',
        choices=['statutes', 'precedents'],
        required=True,
        help='데이터 타입 (statutes 또는 precedents)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Version {args.version} ({args.data_type}) 활성화")
    print("=" * 80)
    
    # 데이터베이스 URL 확인
    db_url = build_database_url() or os.getenv('DATABASE_URL')
    if not db_url:
        print("❌ 데이터베이스 URL을 찾을 수 없습니다.")
        print("   POSTGRES_* 환경 변수 또는 DATABASE_URL을 설정해주세요.")
        return 1
    
    print(f"\n✅ 데이터베이스 연결: {db_url.split('@')[1] if '@' in db_url else '설정됨'}")
    
    # 버전 관리자 초기화
    version_manager = PgEmbeddingVersionManager(db_url)
    
    # 버전 정보 확인
    print("\n1. 버전 정보 확인")
    print("-" * 80)
    
    version_info = version_manager.get_version_info(version=args.version, data_type=args.data_type)
    if not version_info:
        print(f"   ❌ Version {args.version} ({args.data_type})을 찾을 수 없습니다.")
        return 1
    
    print(f"   Version ID: {version_info['id']}")
    print(f"   Version: {version_info['version']}")
    print(f"   Model: {version_info.get('model_name', 'N/A')}")
    print(f"   Data Type: {args.data_type}")
    print(f"   Is Active: {version_info['is_active']}")
    if version_info.get('created_at'):
        print(f"   Created At: {version_info['created_at']}")
    
    if version_info['is_active']:
        print(f"\n   ✅ Version {args.version}이 이미 활성화되어 있습니다!")
        return 0
    
    # 버전 활성화
    print("\n2. 버전 활성화")
    print("-" * 80)
    
    success = version_manager.set_active_version(version=args.version, data_type=args.data_type)
    
    if success:
        print(f"   ✅ Version {args.version} 활성화 완료!")
        print(f"   - 기존 활성 버전이 자동으로 비활성화되었습니다.")
    else:
        print(f"   ❌ Version {args.version} 활성화 실패")
        return 1
    
    # 최종 확인
    print("\n3. 최종 확인")
    print("-" * 80)
    
    final_info = version_manager.get_version_info(version=args.version, data_type=args.data_type)
    if final_info and final_info['is_active']:
        print("   ✅ Version 활성화 확인 완료!")
        print(f"   - Version ID: {final_info['id']}")
        print(f"   - Version: {final_info['version']}")
        print(f"   - Model: {final_info.get('model_name', 'N/A')}")
        print(f"   - Data Type: {args.data_type}")
    else:
        print("   ❌ Version 활성화 확인 실패")
        return 1
    
    print("\n" + "=" * 80)
    print("✅ 완료")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

