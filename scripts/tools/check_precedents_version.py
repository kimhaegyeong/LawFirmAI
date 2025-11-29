#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""precedents 버전 분포 확인 스크립트"""

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
    print("precedents 버전 분포 확인")
    print("=" * 80)
    
    with adapter.get_connection_context() as conn:
        cursor = conn.cursor()
        
        # precedent_chunks 버전 분포
        print("\n1. precedent_chunks 테이블 버전 분포:")
        print("-" * 80)
        cursor.execute("""
            SELECT embedding_version, COUNT(*) 
            FROM precedent_chunks 
            WHERE embedding_vector IS NOT NULL 
            GROUP BY embedding_version 
            ORDER BY embedding_version
        """)
        rows = cursor.fetchall()
        if rows:
            for row in rows:
                try:
                    if hasattr(row, 'keys'):
                        version = row.get('embedding_version')
                        count = row.get('count')
                    else:
                        version = row[0]
                        count = row[1]
                    print(f"   버전 {version}: {count:,}개")
                except (KeyError, IndexError) as e:
                    print(f"   ⚠️ 행 파싱 오류: {row}, {e}")
        else:
            print("   ⚠️ 데이터가 없습니다.")
        
        # embedding_versions 테이블 확인
        print("\n2. precedents embedding_versions:")
        print("-" * 80)
        cursor.execute("""
            SELECT id, version, data_type, is_active, model_name
            FROM embedding_versions 
            WHERE data_type = 'precedents' 
            ORDER BY version
        """)
        rows = cursor.fetchall()
        if rows:
            for row in rows:
                if hasattr(row, 'keys'):
                    vid = row['id']
                    vnum = row['version']
                    dtype = row['data_type']
                    active = row['is_active']
                    model = row.get('model_name', 'N/A')
                else:
                    vid = row[0]
                    vnum = row[1]
                    dtype = row[2]
                    active = row[3]
                    model = row[4] if len(row) > 4 else 'N/A'
                status = "✅ 활성" if active else "❌ 비활성"
                print(f"   ID={vid}, Version={vnum}, {status}, Model={model}")
        else:
            print("   ⚠️ precedents 버전이 없습니다.")
        
        # 활성 버전의 데이터 존재 여부 확인
        print("\n3. 활성 버전 데이터 확인:")
        print("-" * 80)
        cursor.execute("""
            SELECT id, version 
            FROM embedding_versions 
            WHERE data_type = 'precedents' AND is_active = TRUE
            LIMIT 1
        """)
        active_row = cursor.fetchone()
        if active_row:
            if hasattr(active_row, 'keys'):
                active_version_id = active_row['id']
                active_version_num = active_row['version']
            else:
                active_version_id = active_row[0]
                active_version_num = active_row[1]
            
            print(f"   활성 버전: ID={active_version_id}, Version={active_version_num}")
            
            # 해당 버전의 데이터 개수 확인
            cursor.execute("""
                SELECT COUNT(*) 
                FROM precedent_chunks 
                WHERE embedding_vector IS NOT NULL 
                AND embedding_version = %s
            """, (active_version_id,))
            count_row = cursor.fetchone()
            try:
                if hasattr(count_row, 'keys'):
                    count = count_row.get('count', 0)
                else:
                    count = count_row[0] if count_row else 0
            except (KeyError, IndexError, TypeError):
                count = 0
            print(f"   해당 버전의 데이터: {count:,}개")
            
            if count == 0:
                print("   ⚠️ 활성 버전에 데이터가 없습니다!")
                print("   → precedents 검색이 실패하는 원인입니다.")
        else:
            print("   ⚠️ 활성 버전이 설정되지 않았습니다.")
    
    print("\n" + "=" * 80)
    return 0

if __name__ == '__main__':
    sys.exit(main())

