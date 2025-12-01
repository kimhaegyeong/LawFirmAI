#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
임베딩 진행 상황 확인 스크립트 (범용)

사용법:
    python scripts/embedding/check_progress.py \
        --data-type statutes \
        --version 3
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

# 모듈 임포트
try:
    from scripts.ingest.open_law.utils import build_database_url
    from lawfirm_langgraph.core.data.db_adapter import DatabaseAdapter
    from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
except ImportError as e:
    print(f"❌ 필수 모듈을 불러올 수 없습니다: {e}")
    sys.exit(1)


def check_progress(data_type: str, version: int = None):
    """임베딩 진행 상황 확인"""
    db_url = build_database_url() or os.getenv('DATABASE_URL')
    if not db_url:
        print("❌ 데이터베이스 URL을 찾을 수 없습니다.")
        return 1
    
    db_adapter = DatabaseAdapter(db_url)
    
    print("=" * 80)
    print(f"{data_type} 임베딩 진행 상황 확인")
    if version:
        print(f"Version: {version}")
    print("=" * 80)
    
    with db_adapter.get_connection_context() as conn:
        cursor = conn.cursor()
        
        if data_type == 'statutes':
            # 전체 법령 조문 수 확인
            cursor.execute("""
                SELECT COUNT(DISTINCT article_id) as total
                FROM statute_embeddings
            """)
            row = cursor.fetchone()
            total_articles = row[0] if isinstance(row, tuple) else (row.get('total', 0) if hasattr(row, 'get') else 0)
            
            # 버전별 임베딩 통계
            if version:
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM statute_embeddings
                    WHERE embedding_version = :version
                    AND embedding_vector IS NOT NULL
                """, {"version": version})
                row = cursor.fetchone()
                version_count = row[0] if isinstance(row, tuple) else (row.get('count', 0) if hasattr(row, 'get') else 0)
                
                print(f"\n📊 전체 법령 조문 수: {total_articles:,}개")
                print(f"📊 Version {version} 임베딩 완료: {version_count:,}개")
                
                if total_articles > 0:
                    progress = (version_count / total_articles) * 100
                    print(f"📊 진행률: {progress:.1f}% ({version_count:,}/{total_articles:,})")
            else:
                # 모든 버전 통계
                cursor.execute("""
                    SELECT 
                        embedding_version,
                        COUNT(*) as count,
                        COUNT(CASE WHEN embedding_vector IS NOT NULL THEN 1 END) as with_vector
                    FROM statute_embeddings
                    GROUP BY embedding_version
                    ORDER BY embedding_version DESC
                """)
                
                rows = cursor.fetchall()
                print(f"\n📊 전체 법령 조문 수: {total_articles:,}개")
                print("\n버전별 임베딩 통계:")
                for row in rows:
                    v = row[0] if isinstance(row, tuple) else row.get('embedding_version')
                    total = row[1] if isinstance(row, tuple) else row.get('count')
                    with_vector = row[2] if isinstance(row, tuple) else row.get('with_vector')
                    
                    status = "✅ 완료" if with_vector == total else f"⏳ 진행 중 ({with_vector}/{total})"
                    print(f"  Version {v}: {with_vector:,}개 벡터 {status}")
        
        elif data_type == 'precedents':
            # 전체 판례 청크 수 확인
            cursor.execute("""
                SELECT COUNT(DISTINCT precedent_content_id) as total
                FROM precedent_chunks
            """)
            row = cursor.fetchone()
            total_precedents = row[0] if isinstance(row, tuple) else (row.get('total', 0) if hasattr(row, 'get') else 0)
            
            # 버전별 임베딩 통계
            if version:
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM precedent_chunks
                    WHERE embedding_version = :version
                    AND embedding_vector IS NOT NULL
                """, {"version": version})
                row = cursor.fetchone()
                version_count = row[0] if isinstance(row, tuple) else (row.get('count', 0) if hasattr(row, 'get') else 0)
                
                print(f"\n📊 전체 판례 청크 수: {total_precedents:,}개")
                print(f"📊 Version {version} 임베딩 완료: {version_count:,}개")
                
                if total_precedents > 0:
                    progress = (version_count / total_precedents) * 100
                    print(f"📊 진행률: {progress:.1f}% ({version_count:,}/{total_precedents:,})")
            else:
                # 모든 버전 통계
                cursor.execute("""
                    SELECT 
                        embedding_version,
                        COUNT(*) as count,
                        COUNT(CASE WHEN embedding_vector IS NOT NULL THEN 1 END) as with_vector
                    FROM precedent_chunks
                    GROUP BY embedding_version
                    ORDER BY embedding_version DESC
                """)
                
                rows = cursor.fetchall()
                print(f"\n📊 전체 판례 청크 수: {total_precedents:,}개")
                print("\n버전별 임베딩 통계:")
                for row in rows:
                    v = row[0] if isinstance(row, tuple) else row.get('embedding_version')
                    total = row[1] if isinstance(row, tuple) else row.get('count')
                    with_vector = row[2] if isinstance(row, tuple) else row.get('with_vector')
                    
                    status = "✅ 완료" if with_vector == total else f"⏳ 진행 중 ({with_vector}/{total})"
                    print(f"  Version {v}: {with_vector:,}개 벡터 {status}")
        
        # 활성 버전 확인
        print("\n" + "-" * 80)
        print("활성 버전 정보:")
        print("-" * 80)
        
        try:
            engine = SemanticSearchEngineV2()
            active_version_id = engine._get_active_embedding_version_id(data_type=data_type)
            if active_version_id:
                print(f"✅ {data_type} 활성 버전 ID: {active_version_id}")
            else:
                print(f"⚠️ {data_type} 활성 버전 없음")
        except Exception as e:
            print(f"⚠️ 활성 버전 확인 실패: {e}")
        
        print("\n" + "=" * 80)
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='임베딩 진행 상황 확인')
    parser.add_argument(
        '--data-type',
        choices=['statutes', 'precedents'],
        required=True,
        help='데이터 타입 (statutes 또는 precedents)'
    )
    parser.add_argument(
        '--version',
        type=int,
        default=None,
        help='버전 번호 (기본값: 모든 버전)'
    )
    
    args = parser.parse_args()
    
    return check_progress(args.data_type, args.version)


if __name__ == "__main__":
    sys.exit(main())

