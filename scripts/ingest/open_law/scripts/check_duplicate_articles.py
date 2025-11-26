#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
statutes_articles 테이블 중복 데이터 검토 스크립트
PostgreSQL 데이터베이스의 statutes_articles 테이블에서 중복된 데이터를 확인합니다.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from sqlalchemy import create_engine, text

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
# scripts/ingest/open_law/scripts/check_duplicate_articles.py -> 프로젝트 루트
_PROJECT_ROOT = _CURRENT_FILE.parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# .env 파일 로드 (utils/env_loader.py 사용)
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(_PROJECT_ROOT)
except ImportError:
    try:
        from dotenv import load_dotenv
        scripts_env = _PROJECT_ROOT / "scripts" / ".env"
        if scripts_env.exists():
            load_dotenv(dotenv_path=str(scripts_env), override=True)
        root_env = _PROJECT_ROOT / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=str(root_env), override=False)
    except ImportError:
        pass

# 공통 유틸리티 임포트
from scripts.ingest.open_law.utils import build_database_url

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_duplicate_articles(conn) -> Dict[str, Any]:
    """statutes_articles 테이블의 중복 데이터 확인"""
    
    results = {
        'total_count': 0,
        'duplicate_groups': [],
        'duplicate_count': 0,
        'unique_count': 0
    }
    
    # 전체 레코드 수
    result = conn.execute(text("SELECT COUNT(*) FROM statutes_articles"))
    results['total_count'] = result.fetchone()[0]
    
    # 중복 확인 기준: statute_id, article_no, article_title, article_content, clause_no
    duplicate_query = text("""
        SELECT 
            statute_id,
            article_no,
            article_title,
            article_content,
            clause_no,
            COUNT(*) as duplicate_count,
            MIN(id) as min_id,
            MAX(id) as max_id,
            MIN(collected_at) as first_collected,
            MAX(collected_at) as last_collected,
            ARRAY_AGG(id ORDER BY id) as all_ids
        FROM statutes_articles
        GROUP BY statute_id, article_no, article_title, article_content, clause_no
        HAVING COUNT(*) > 1
        ORDER BY duplicate_count DESC, statute_id, article_no
    """)
    
    result = conn.execute(duplicate_query)
    duplicate_groups = result.fetchall()
    
    results['duplicate_groups'] = [
        {
            'statute_id': row[0],
            'article_no': row[1],
            'article_title': row[2] or '',
            'article_content': row[3] or '',
            'clause_no': row[4] or '',
            'duplicate_count': row[5],
            'min_id': row[6],
            'max_id': row[7],
            'first_collected': row[8],
            'last_collected': row[9],
            'all_ids': row[10]
        }
        for row in duplicate_groups
    ]
    
    results['duplicate_count'] = sum(group['duplicate_count'] for group in results['duplicate_groups'])
    results['unique_count'] = results['total_count'] - results['duplicate_count'] + len(results['duplicate_groups'])
    
    return results


def get_duplicate_details(conn, statute_id: int, article_no: str, article_title: str = None,
                          article_content: str = None, clause_no: str = None) -> List[Dict[str, Any]]:
    """특정 조문의 중복 레코드 상세 정보 조회"""
    
    query = text("""
        SELECT 
            sa.id,
            sa.statute_id,
            s.law_name_kr,
            sa.article_no,
            sa.article_title,
            sa.clause_no,
            sa.item_no,
            sa.sub_item_no,
            LEFT(sa.article_content, 100) as content_preview,
            sa.effective_date,
            sa.collected_at
        FROM statutes_articles sa
        JOIN statutes s ON sa.statute_id = s.id
        WHERE sa.statute_id = :statute_id
          AND sa.article_no = :article_no
          AND COALESCE(sa.article_title, '') = COALESCE(:article_title, '')
          AND sa.article_content = :article_content
          AND COALESCE(sa.clause_no, '') = COALESCE(:clause_no, '')
        ORDER BY sa.id
    """)
    
    result = conn.execute(query, {
        'statute_id': statute_id,
        'article_no': article_no,
        'article_title': article_title,
        'article_content': article_content,
        'clause_no': clause_no
    })
    
    return [
        {
            'id': row[0],
            'statute_id': row[1],
            'law_name_kr': row[2],
            'article_no': row[3],
            'article_title': row[4],
            'clause_no': row[5] or '',
            'item_no': row[6] or '',
            'sub_item_no': row[7] or '',
            'content_preview': row[8],
            'effective_date': row[9],
            'collected_at': row[10]
        }
        for row in result.fetchall()
    ]


def generate_cleanup_sql(duplicate_groups: List[Dict[str, Any]], keep_oldest: bool = True) -> str:
    """중복 데이터 제거를 위한 SQL 생성"""
    
    sql_lines = ["-- 중복 데이터 제거 SQL", "-- 주의: 실행 전에 백업을 권장합니다", ""]
    
    for group in duplicate_groups:
        ids = group['all_ids']
        if len(ids) <= 1:
            continue
        
        # 가장 오래된 것(또는 가장 최신 것)을 제외하고 나머지 삭제
        if keep_oldest:
            ids_to_delete = ids[1:]  # 첫 번째(가장 오래된) ID를 제외
        else:
            ids_to_delete = ids[:-1]  # 마지막(가장 최신) ID를 제외
        
        ids_str = ', '.join(map(str, ids_to_delete))
        title_preview = (group['article_title'][:30] + '...') if group['article_title'] and len(group['article_title']) > 30 else (group['article_title'] or '')
        sql_lines.append(f"-- 법령ID: {group['statute_id']}, 조문: {group['article_no']}, 항: {group['clause_no'] or '(없음)'}, 중복 {group['duplicate_count']}개")
        if title_preview:
            sql_lines.append(f"-- 제목: {title_preview}")
        sql_lines.append(f"DELETE FROM statutes_articles WHERE id IN ({ids_str});")
        sql_lines.append("")
    
    return "\n".join(sql_lines)


def main():
    parser = argparse.ArgumentParser(description='statutes_articles 테이블 중복 데이터 검토')
    parser.add_argument(
        '--db',
        default=build_database_url(),
        help='PostgreSQL 데이터베이스 URL (환경변수: DATABASE_URL 또는 개별 POSTGRES_* 변수)'
    )
    parser.add_argument(
        '--detail',
        action='store_true',
        help='중복된 레코드의 상세 정보 출력'
    )
    parser.add_argument(
        '--generate-sql',
        action='store_true',
        help='중복 데이터 제거를 위한 SQL 생성'
    )
    parser.add_argument(
        '--keep-oldest',
        action='store_true',
        default=True,
        help='제거 SQL 생성 시 가장 오래된 레코드 유지 (기본값: True)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=20,
        help='상세 정보 출력 시 최대 개수 (기본값: 20)'
    )
    
    args = parser.parse_args()
    
    if not args.db:
        logger.error("--db 인자 또는 DATABASE_URL 환경변수가 필요합니다.")
        return
    
    # 데이터베이스 연결
    engine = create_engine(
        args.db,
        pool_pre_ping=True,
        echo=False
    )
    
    print("=" * 80)
    print("statutes_articles 테이블 중복 데이터 검토")
    print("=" * 80)
    print()
    
    with engine.connect() as conn:
        results = check_duplicate_articles(conn)
        
        # 기본 통계 출력
        print("📊 통계")
        print("-" * 80)
        print(f"전체 레코드 수: {results['total_count']:,}개")
        print(f"고유 레코드 수: {results['unique_count']:,}개")
        print(f"중복 그룹 수: {len(results['duplicate_groups'])}개")
        print(f"중복 레코드 수: {results['duplicate_count']:,}개")
        print()
        
        if results['duplicate_groups']:
            print("⚠️  중복 데이터 발견")
            print("-" * 80)
            print(f"중복된 조문 그룹: {len(results['duplicate_groups'])}개")
            print()
            
            # 상위 중복 그룹 출력
            print("🔍 상위 중복 그룹 (상위 10개):")
            print("-" * 80)
            for i, group in enumerate(results['duplicate_groups'][:10], 1):
                title_preview = (group['article_title'][:30] + '...') if group['article_title'] and len(group['article_title']) > 30 else (group['article_title'] or '(없음)')
                content_preview = (group['article_content'][:50] + '...') if group['article_content'] and len(group['article_content']) > 50 else (group['article_content'] or '(없음)')
                print(f"{i}. 법령ID: {group['statute_id']}, 조문: {group['article_no']}, "
                      f"항: {group['clause_no'] or '(없음)'}")
                print(f"   제목: {title_preview}")
                print(f"   내용 미리보기: {content_preview}")
                print(f"   중복 횟수: {group['duplicate_count']}회")
                print(f"   레코드 ID: {group['all_ids']}")
                print(f"   최초 수집: {group['first_collected']}")
                print(f"   최종 수집: {group['last_collected']}")
                print()
            
            if len(results['duplicate_groups']) > 10:
                print(f"... 외 {len(results['duplicate_groups']) - 10}개 그룹")
                print()
            
            # 상세 정보 출력
            if args.detail:
                print("📋 중복 레코드 상세 정보:")
                print("-" * 80)
                detail_count = 0
                for group in results['duplicate_groups']:
                    if detail_count >= args.limit:
                        break
                    
                    details = get_duplicate_details(
                        conn,
                        group['statute_id'],
                        group['article_no'],
                        group['article_title'],
                        group['article_content'],
                        group['clause_no']
                    )
                    
                    print(f"\n법령: {details[0]['law_name_kr'] if details else 'N/A'}")
                    print(f"조문: {group['article_no']}")
                    for detail in details:
                        print(f"  - ID: {detail['id']}, 수집일시: {detail['collected_at']}")
                        print(f"    제목: {detail['article_title'] or '(없음)'}")
                        print(f"    내용 미리보기: {detail['content_preview']}...")
                    detail_count += 1
                    print()
            
            # SQL 생성
            if args.generate_sql:
                print("💾 중복 데이터 제거 SQL 생성:")
                print("-" * 80)
                cleanup_sql = generate_cleanup_sql(results['duplicate_groups'], args.keep_oldest)
                print(cleanup_sql)
                
                # SQL 파일 저장
                sql_file = Path(_PROJECT_ROOT) / "scripts" / "ingest" / "open_law" / "scripts" / "cleanup_duplicate_articles.sql"
                sql_file.parent.mkdir(parents=True, exist_ok=True)
                with open(sql_file, 'w', encoding='utf-8') as f:
                    f.write(cleanup_sql)
                print(f"\n✅ SQL 파일 저장: {sql_file}")
        else:
            print("✅ 중복 데이터 없음")
            print()
        
        print("=" * 80)


if __name__ == '__main__':
    main()

