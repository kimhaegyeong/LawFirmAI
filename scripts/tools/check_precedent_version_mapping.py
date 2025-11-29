#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""precedent_chunks의 embedding_version 값과 embedding_versions 테이블 매핑 확인"""

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

from lawfirm_langgraph.core.data.db_adapter import DatabaseAdapter
from scripts.ingest.open_law.utils import build_database_url
import os

def main():
    db_url = build_database_url() or os.getenv('DATABASE_URL')
    if not db_url:
        print("❌ 데이터베이스 URL을 찾을 수 없습니다.")
        return 1
    
    adapter = DatabaseAdapter(db_url)
    
    print("=" * 80)
    print("precedent_chunks embedding_version 매핑 확인")
    print("=" * 80)
    
    with adapter.get_connection_context() as conn:
        cursor = conn.cursor()
        
        # precedent_chunks의 embedding_version 값들
        print("\n1. precedent_chunks의 embedding_version 값들:")
        print("-" * 80)
        cursor.execute("""
            SELECT DISTINCT embedding_version, COUNT(*) as count
            FROM precedent_chunks 
            WHERE embedding_vector IS NOT NULL 
            GROUP BY embedding_version 
            ORDER BY embedding_version
            LIMIT 10
        """)
        rows = cursor.fetchall()
        for row in rows:
            try:
                if hasattr(row, 'keys'):
                    version = row.get('embedding_version')
                    count = row.get('count')
                else:
                    version = row[0]
                    count = row[1]
                print(f"   embedding_version={version}: {count:,}개")
            except Exception as e:
                print(f"   ⚠️ 행 파싱 오류: {e}")
        
        # embedding_versions 테이블 확인
        print("\n2. embedding_versions 테이블 (precedents):")
        print("-" * 80)
        cursor.execute("""
            SELECT id, version, data_type, is_active, model_name
            FROM embedding_versions 
            WHERE data_type = 'precedents' 
            ORDER BY version
        """)
        rows = cursor.fetchall()
        for row in rows:
            try:
                if hasattr(row, 'keys'):
                    vid = row.get('id')
                    vnum = row.get('version')
                    dtype = row.get('data_type')
                    active = row.get('is_active')
                    model = row.get('model_name', 'N/A')
                else:
                    vid = row[0]
                    vnum = row[1]
                    dtype = row[2]
                    active = row[3]
                    model = row[4] if len(row) > 4 else 'N/A'
                status = "✅ 활성" if active else "❌ 비활성"
                print(f"   ID={vid}, Version={vnum}, {status}, Model={model}")
            except Exception as e:
                print(f"   ⚠️ 행 파싱 오류: {e}")
        
        # 매핑 확인
        print("\n3. 매핑 확인:")
        print("-" * 80)
        cursor.execute("""
            SELECT id, version 
            FROM embedding_versions 
            WHERE data_type = 'precedents' AND is_active = TRUE
            LIMIT 1
        """)
        active_row = cursor.fetchone()
        if active_row:
            try:
                if hasattr(active_row, 'keys'):
                    active_id = active_row.get('id')
                    active_version = active_row.get('version')
                else:
                    active_id = active_row[0]
                    active_version = active_row[1]
                
                print(f"   활성 embedding_versions: ID={active_id}, Version={active_version}")
                
                # precedent_chunks에 해당 ID가 있는지 확인
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM precedent_chunks 
                    WHERE embedding_vector IS NOT NULL 
                    AND embedding_version = %s
                """, (active_id,))
                count_row = cursor.fetchone()
                count_id = count_row[0] if hasattr(count_row, 'keys') else count_row[0]
                
                # precedent_chunks에 해당 version이 있는지 확인
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM precedent_chunks 
                    WHERE embedding_vector IS NOT NULL 
                    AND embedding_version = %s
                """, (active_version,))
                count_row = cursor.fetchone()
                count_version = count_row[0] if hasattr(count_row, 'keys') else count_row[0]
                
                print(f"   precedent_chunks.embedding_version = {active_id} (ID): {count_id:,}개")
                print(f"   precedent_chunks.embedding_version = {active_version} (Version): {count_version:,}개")
                
                if count_id > 0:
                    print(f"   ✅ precedent_chunks는 embedding_versions.id를 사용합니다!")
                elif count_version > 0:
                    print(f"   ✅ precedent_chunks는 embedding_versions.version을 사용합니다!")
                else:
                    print(f"   ⚠️ 활성 버전과 매칭되는 데이터가 없습니다!")
            except Exception as e:
                print(f"   ⚠️ 오류: {e}")
    
    print("\n" + "=" * 80)
    return 0

if __name__ == '__main__':
    sys.exit(main())

