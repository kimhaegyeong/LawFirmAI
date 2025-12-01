#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 재임베딩 스크립트

사용법:
    python scripts/embedding/re_embed.py \
        --data-type statutes \
        --model jhgan/ko-sroberta-multitask \
        --auto-activate
"""

import os
import sys
import subprocess
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
    from sqlalchemy import text
except ImportError as e:
    print(f"❌ 필수 모듈을 불러올 수 없습니다: {e}")
    sys.exit(1)


def determine_next_version(version_manager: PgEmbeddingVersionManager, data_type: str) -> int:
    """다음 버전 번호 결정"""
    with version_manager.engine.connect() as conn:
        result = conn.execute(text("""
            SELECT MAX(version) as max_version
            FROM embedding_versions
            WHERE data_type = :data_type
        """), {"data_type": data_type})
        row = result.fetchone()
        max_version = row[0] if row and row[0] else 0
        return max_version + 1


def get_current_active_version(data_type: str):
    """현재 활성 버전 ID 조회"""
    try:
        from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
        engine = SemanticSearchEngineV2()
        return engine._get_active_embedding_version_id(data_type=data_type)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='통합 재임베딩 스크립트')
    parser.add_argument(
        '--data-type',
        choices=['statutes', 'precedents'],
        required=True,
        help='데이터 타입 (statutes 또는 precedents)'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='임베딩 모델 이름 (예: jhgan/ko-sroberta-multitask)'
    )
    parser.add_argument(
        '--version',
        type=int,
        default=None,
        help='버전 번호 (기본값: 자동 생성)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='배치 크기 (기본값: 100)'
    )
    parser.add_argument(
        '--auto-activate',
        action='store_true',
        help='완료 후 자동 활성화'
    )
    parser.add_argument(
        '--chunking-strategy',
        default='article',
        help='청킹 전략 (기본값: article)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"{args.data_type} 재임베딩: {args.model} 모델")
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
    
    # 현재 활성 버전 확인
    print("\n1. 현재 활성 버전 확인")
    print("-" * 80)
    current_version_id = get_current_active_version(args.data_type)
    if current_version_id:
        print(f"   현재 {args.data_type} 활성 버전 ID: {current_version_id}")
    else:
        print(f"   ⚠️ {args.data_type} 활성 버전이 없습니다")
    
    # 버전 번호 결정
    if args.version is None:
        args.version = determine_next_version(version_manager, args.data_type)
    
    print("\n2. 새 임베딩 버전 생성")
    print("-" * 80)
    print(f"   새 버전 번호: {args.version}")
    print(f"   모델: {args.model}")
    print(f"   Data Type: {args.data_type}")
    
    # 모델 차원 결정 (일반적으로 768)
    dim = 768
    
    # 새 버전 생성
    new_version_id = version_manager.get_or_create_version(
        version=args.version,
        model_name=args.model,
        dim=dim,
        data_type=args.data_type,
        chunking_strategy=args.chunking_strategy,
        description=f"{args.model} 모델로 재임베딩된 {args.data_type}",
        metadata={
            "model_name": args.model,
            "dimension": dim,
            "chunking_strategy": args.chunking_strategy,
            "previous_version": current_version_id
        },
        set_active=False
    )
    
    print(f"   ✅ 새 버전 생성 완료 (ID: {new_version_id})")
    
    # 임베딩 생성 실행
    print("\n3. 임베딩 생성 시작")
    print("-" * 80)
    
    if args.data_type == 'statutes':
        script_path = _PROJECT_ROOT / "scripts" / "ingest" / "open_law" / "embedding" / "generate_statute_embeddings.py"
    else:
        script_path = _PROJECT_ROOT / "scripts" / "ingest" / "open_law" / "embedding" / "generate_embeddings.py"
    
    if not script_path.exists():
        print(f"❌ 임베딩 생성 스크립트를 찾을 수 없습니다: {script_path}")
        return 1
    
    cmd = [
        sys.executable,
        str(script_path),
        "--model", args.model,
        "--method", "pgvector",
        "--version", str(args.version),
        "--batch-size", str(args.batch_size)
    ]
    
    if args.data_type == 'statutes':
        cmd.extend(["--chunking-strategy", args.chunking_strategy])
    
    print(f"   실행 명령어: {' '.join(cmd)}")
    print("\n   🔄 임베딩 생성 스크립트 실행 중...")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=str(_PROJECT_ROOT))
        print("\n   ✅ 임베딩 생성 완료!")
        
        # 자동 활성화
        if args.auto_activate:
            print("\n4. 새 버전 활성화")
            print("-" * 80)
            success = version_manager.set_active_version(args.version, args.data_type)
            if success:
                print(f"   ✅ 새 버전 {args.version} (ID: {new_version_id}) 활성화 완료")
                if current_version_id:
                    print(f"   ✅ 기존 버전 {current_version_id} 자동 비활성화됨")
            else:
                print("   ❌ 새 버전 활성화 실패")
                return 1
        else:
            print("\n4. 새 버전 활성화")
            print("-" * 80)
            print("   ⚠️ 임베딩이 완료되었습니다.")
            print(f"   버전 {args.version} (ID: {new_version_id})을 활성화하려면 다음 명령어를 실행하세요:")
            print(f"   python scripts/embedding/activate_version.py --version {args.version} --data-type {args.data_type}")
    
    except subprocess.CalledProcessError as e:
        print(f"\n   ❌ 임베딩 생성 실패: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n   ⚠️ 사용자에 의해 중단되었습니다.")
        return 1
    
    print("\n" + "=" * 80)
    print("✅ 스크립트 완료")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

