#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ko-Legal-SBERT 재임베딩 후 MLflow 인덱스 빌드 스크립트

재임베딩이 완료된 Ko-Legal-SBERT 버전의 FAISS 인덱스를 MLflow에 빌드하고 저장합니다.
"""

import sys
import os
from pathlib import Path
import sqlite3
from datetime import datetime

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts" / "rag"))

from build_index import build_and_save_index

def find_ko_legal_sbert_version(db_path: str = "data/lawfirm_v2.db"):
    """Ko-Legal-SBERT 버전 찾기"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.execute("""
        SELECT id, version_name, model_name, created_at
        FROM embedding_versions
        WHERE model_name LIKE '%ko-legal-sbert%' OR model_name LIKE '%Legal%'
        ORDER BY created_at DESC
        LIMIT 1
    """)
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'id': row['id'],
            'version_name': row['version_name'],
            'model_name': row['model_name'],
            'created_at': row['created_at']
        }
    return None

def main():
    """메인 함수"""
    print("=" * 60)
    print("Ko-Legal-SBERT MLflow 인덱스 빌드")
    print("=" * 60)
    
    db_path = "data/lawfirm_v2.db"
    if not Path(db_path).exists():
        print(f"❌ 데이터베이스 파일을 찾을 수 없습니다: {db_path}")
        return 1
    
    # Ko-Legal-SBERT 버전 찾기
    print("\n1. Ko-Legal-SBERT 버전 확인 중...")
    version_info = find_ko_legal_sbert_version(db_path)
    
    if not version_info:
        print("❌ Ko-Legal-SBERT 버전을 찾을 수 없습니다.")
        print("   먼저 재임베딩을 완료해주세요.")
        return 1
    
    print(f"   ✅ 버전 ID: {version_info['id']}")
    print(f"   버전 이름: {version_info['version_name']}")
    print(f"   모델: {version_info['model_name']}")
    print(f"   생성일: {version_info['created_at']}")
    
    # 버전 이름에서 청킹 전략 추출
    chunking_strategy = "standard"  # 기본값
    if "dynamic" in version_info['version_name'].lower():
        chunking_strategy = "dynamic"
    elif "standard" in version_info['version_name'].lower():
        chunking_strategy = "standard"
    
    # MLflow 버전 이름 생성
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mlflow_version_name = f"production-ko-legal-sbert-{timestamp}"
    
    print(f"\n2. MLflow 인덱스 빌드 시작...")
    print(f"   버전 이름: {mlflow_version_name}")
    print(f"   임베딩 버전 ID: {version_info['id']}")
    print(f"   청킹 전략: {chunking_strategy}")
    print(f"   모델: {version_info['model_name']}")
    
    # 인덱스 빌드 및 저장
    success = build_and_save_index(
        version_name=mlflow_version_name,
        embedding_version_id=version_info['id'],
        chunking_strategy=chunking_strategy,
        db_path=db_path,
        use_mlflow=True,
        local_backup=True,
        model_name=version_info['model_name']
    )
    
    if success:
        print(f"\n✅ MLflow 인덱스 빌드 및 저장 완료!")
        print(f"   버전 이름: {mlflow_version_name}")
        print(f"\n다음 단계:")
        print(f"1. MLflow UI에서 run ID 확인")
        print(f"2. 프로덕션 태그 설정:")
        print(f"   python scripts/rag/build_production_index.py --run-id <RUN_ID> --tag-as-production")
        return 0
    else:
        print(f"\n❌ MLflow 인덱스 빌드 실패")
        return 1

if __name__ == "__main__":
    sys.exit(main())

