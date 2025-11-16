# -*- coding: utf-8 -*-
"""
데이터 정합성 개선 스크립트
- text_chunks 테이블의 embedding_version_id NULL 값 업데이트
- 메타데이터 필드 보완
"""

import sys
import os
import sqlite3
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_active_version_id(conn: sqlite3.Connection) -> Optional[int]:
    """활성 버전 ID 조회"""
    try:
        cursor = conn.execute("""
            SELECT id FROM embedding_versions
            WHERE is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        if row:
            return row['id']
        return None
    except Exception as e:
        logger.error(f"Failed to get active version ID: {e}")
        return None


def fix_embedding_version_ids(db_path: str, dry_run: bool = True) -> Dict[str, Any]:
    """
    text_chunks 테이블의 embedding_version_id NULL 값 업데이트
    
    Args:
        db_path: 데이터베이스 경로
        dry_run: 실제로 업데이트하지 않고 통계만 조회
        
    Returns:
        업데이트 통계
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        cursor = conn.cursor()
        
        # 현재 상태 확인
        cursor.execute("""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(CASE WHEN embedding_version_id IS NULL THEN 1 END) as null_version_chunks,
                COUNT(CASE WHEN embedding_version_id IS NOT NULL THEN 1 END) as assigned_chunks
            FROM text_chunks
        """)
        stats = dict(cursor.fetchone())
        
        logger.info("="*80)
        logger.info("데이터 정합성 개선: embedding_version_id")
        logger.info("="*80)
        logger.info(f"전체 청크 수: {stats['total_chunks']}")
        logger.info(f"버전 ID 할당된 청크: {stats['assigned_chunks']}")
        logger.info(f"버전 ID NULL인 청크: {stats['null_version_chunks']}")
        
        if stats['null_version_chunks'] == 0:
            logger.info("✅ 모든 청크에 버전 ID가 할당되어 있습니다.")
            return stats
        
        # 활성 버전 ID 조회
        active_version_id = get_active_version_id(conn)
        if not active_version_id:
            logger.error("❌ 활성 버전을 찾을 수 없습니다.")
            return stats
        
        logger.info(f"활성 버전 ID: {active_version_id}")
        
        if dry_run:
            logger.info(f"\n[DRY RUN] {stats['null_version_chunks']}개 청크에 버전 ID {active_version_id}를 할당할 예정입니다.")
            return stats
        
        # 버전 ID 할당
        logger.info("\n" + "="*80)
        logger.info("버전 ID 할당 중...")
        logger.info("="*80)
        
        cursor.execute("""
            UPDATE text_chunks
            SET embedding_version_id = ?
            WHERE embedding_version_id IS NULL
        """, (active_version_id,))
        
        updated_count = cursor.rowcount
        conn.commit()
        
        logger.info(f"✅ {updated_count}개 청크에 버전 ID {active_version_id} 할당 완료")
        
        # 최종 상태 확인
        cursor.execute("""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(CASE WHEN embedding_version_id = ? THEN 1 END) as version_chunks,
                COUNT(CASE WHEN embedding_version_id IS NULL THEN 1 END) as null_version_chunks
            FROM text_chunks
        """, (active_version_id,))
        final_stats = dict(cursor.fetchone())
        
        logger.info(f"\n최종 상태:")
        logger.info(f"전체 청크 수: {final_stats['total_chunks']}")
        logger.info(f"버전 ID {active_version_id} 청크: {final_stats['version_chunks']}")
        logger.info(f"버전 ID NULL인 청크: {final_stats['null_version_chunks']}")
        
        return final_stats
        
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ 버전 ID 할당 실패: {e}", exc_info=True)
        return stats
    finally:
        conn.close()


def fix_metadata_fields(db_path: str, dry_run: bool = True) -> Dict[str, Any]:
    """
    메타데이터 필드 보완 (모든 source_type의 메타데이터)
    
    Args:
        db_path: 데이터베이스 경로
        dry_run: 실제로 업데이트하지 않고 통계만 조회
        
    Returns:
        업데이트 통계
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        cursor = conn.cursor()
        total_updated = 0
        
        logger.info("\n" + "="*80)
        logger.info("메타데이터 필드 보완")
        logger.info("="*80)
        
        # 1. case_paragraph 타입 처리
        cursor.execute("""
            SELECT 
                COUNT(*) as total_case_chunks,
                COUNT(CASE WHEN tc.meta IS NULL OR tc.meta = '' THEN 1 END) as missing_metadata
            FROM text_chunks tc
            WHERE tc.source_type = 'case_paragraph'
        """)
        case_stats = dict(cursor.fetchone())
        
        logger.info(f"\n[case_paragraph]")
        logger.info(f"  전체 청크 수: {case_stats['total_case_chunks']}")
        logger.info(f"  메타데이터 누락 청크: {case_stats['missing_metadata']}")
        
        if case_stats['missing_metadata'] > 0:
            cursor.execute("""
                SELECT 
                    tc.id as chunk_id,
                    tc.source_id,
                    c.doc_id,
                    c.casenames,
                    c.court
                FROM text_chunks tc
                LEFT JOIN case_paragraphs cp ON tc.source_id = cp.id
                LEFT JOIN cases c ON cp.case_id = c.id
                WHERE tc.source_type = 'case_paragraph'
                AND (tc.meta IS NULL OR tc.meta = '')
                AND cp.id IS NOT NULL
                AND c.id IS NOT NULL
            """)
            
            case_rows = cursor.fetchall()
            logger.info(f"  메타데이터 보완 가능한 청크: {len(case_rows)}개")
            
            if not dry_run:
                case_updated = 0
                for row in case_rows:
                    try:
                        import json
                        metadata = {
                            'doc_id': row['doc_id'],
                            'casenames': row['casenames'],
                            'court': row['court']
                        }
                        metadata_json = json.dumps(metadata, ensure_ascii=False)
                        
                        cursor.execute("""
                            UPDATE text_chunks
                            SET meta = ?
                            WHERE id = ?
                        """, (metadata_json, row['chunk_id']))
                        
                        case_updated += 1
                    except Exception as e:
                        logger.debug(f"Failed to update metadata for chunk_id={row['chunk_id']}: {e}")
                
                total_updated += case_updated
                logger.info(f"  ✅ {case_updated}개 청크의 메타데이터 보완 완료")
            else:
                if case_rows:
                    logger.info("  샘플:")
                    for row in case_rows[:3]:
                        logger.info(f"    청크 ID: {row['chunk_id']}, doc_id: {row['doc_id']}, casenames: {row['casenames']}")
        
        # 2. decision_paragraph 타입 처리
        cursor.execute("""
            SELECT 
                COUNT(*) as total_decision_chunks,
                COUNT(CASE WHEN tc.meta IS NULL OR tc.meta = '' THEN 1 END) as missing_metadata
            FROM text_chunks tc
            WHERE tc.source_type = 'decision_paragraph'
        """)
        decision_stats = dict(cursor.fetchone())
        
        logger.info(f"\n[decision_paragraph]")
        logger.info(f"  전체 청크 수: {decision_stats['total_decision_chunks']}")
        logger.info(f"  메타데이터 누락 청크: {decision_stats['missing_metadata']}")
        
        if decision_stats['missing_metadata'] > 0:
            cursor.execute("""
                SELECT 
                    tc.id as chunk_id,
                    tc.source_id,
                    d.org,
                    d.doc_id
                FROM text_chunks tc
                LEFT JOIN decision_paragraphs dp ON tc.source_id = dp.id
                LEFT JOIN decisions d ON dp.decision_id = d.id
                WHERE tc.source_type = 'decision_paragraph'
                AND (tc.meta IS NULL OR tc.meta = '')
                AND dp.id IS NOT NULL
                AND d.id IS NOT NULL
            """)
            
            decision_rows = cursor.fetchall()
            logger.info(f"  메타데이터 보완 가능한 청크: {len(decision_rows)}개")
            
            if not dry_run:
                decision_updated = 0
                for row in decision_rows:
                    try:
                        import json
                        metadata = {
                            'org': row['org'],
                            'doc_id': row['doc_id']
                        }
                        metadata_json = json.dumps(metadata, ensure_ascii=False)
                        
                        cursor.execute("""
                            UPDATE text_chunks
                            SET meta = ?
                            WHERE id = ?
                        """, (metadata_json, row['chunk_id']))
                        
                        decision_updated += 1
                    except Exception as e:
                        logger.debug(f"Failed to update metadata for chunk_id={row['chunk_id']}: {e}")
                
                total_updated += decision_updated
                logger.info(f"  ✅ {decision_updated}개 청크의 메타데이터 보완 완료")
            else:
                if decision_rows:
                    logger.info("  샘플:")
                    for row in decision_rows[:3]:
                        logger.info(f"    청크 ID: {row['chunk_id']}, doc_id: {row['doc_id']}, org: {row['org']}")
        
        # 3. statute_article 타입 처리
        cursor.execute("""
            SELECT 
                COUNT(*) as total_statute_chunks,
                COUNT(CASE WHEN tc.meta IS NULL OR tc.meta = '' THEN 1 END) as missing_metadata
            FROM text_chunks tc
            WHERE tc.source_type = 'statute_article'
        """)
        statute_stats = dict(cursor.fetchone())
        
        logger.info(f"\n[statute_article]")
        logger.info(f"  전체 청크 수: {statute_stats['total_statute_chunks']}")
        logger.info(f"  메타데이터 누락 청크: {statute_stats['missing_metadata']}")
        
        if statute_stats['missing_metadata'] > 0:
            cursor.execute("""
                SELECT 
                    tc.id as chunk_id,
                    tc.source_id,
                    s.name as statute_name,
                    sa.article_no
                FROM text_chunks tc
                LEFT JOIN statute_articles sa ON tc.source_id = sa.id
                LEFT JOIN statutes s ON sa.statute_id = s.id
                WHERE tc.source_type = 'statute_article'
                AND (tc.meta IS NULL OR tc.meta = '')
                AND sa.id IS NOT NULL
                AND s.id IS NOT NULL
            """)
            
            statute_rows = cursor.fetchall()
            logger.info(f"  메타데이터 보완 가능한 청크: {len(statute_rows)}개")
            
            if not dry_run:
                statute_updated = 0
                for row in statute_rows:
                    try:
                        import json
                        metadata = {
                            'statute_name': row['statute_name'],
                            'law_name': row['statute_name'],
                            'article_no': row['article_no'],
                            'article_number': row['article_no']
                        }
                        metadata_json = json.dumps(metadata, ensure_ascii=False)
                        
                        cursor.execute("""
                            UPDATE text_chunks
                            SET meta = ?
                            WHERE id = ?
                        """, (metadata_json, row['chunk_id']))
                        
                        statute_updated += 1
                    except Exception as e:
                        logger.debug(f"Failed to update metadata for chunk_id={row['chunk_id']}: {e}")
                
                total_updated += statute_updated
                logger.info(f"  ✅ {statute_updated}개 청크의 메타데이터 보완 완료")
            else:
                if statute_rows:
                    logger.info("  샘플:")
                    for row in statute_rows[:3]:
                        logger.info(f"    청크 ID: {row['chunk_id']}, statute_name: {row['statute_name']}, article_no: {row['article_no']}")
        
        # 4. interpretation_paragraph 타입 처리
        cursor.execute("""
            SELECT 
                COUNT(*) as total_interpretation_chunks,
                COUNT(CASE WHEN tc.meta IS NULL OR tc.meta = '' THEN 1 END) as missing_metadata
            FROM text_chunks tc
            WHERE tc.source_type = 'interpretation_paragraph'
        """)
        interpretation_stats = dict(cursor.fetchone())
        
        logger.info(f"\n[interpretation_paragraph]")
        logger.info(f"  전체 청크 수: {interpretation_stats['total_interpretation_chunks']}")
        logger.info(f"  메타데이터 누락 청크: {interpretation_stats['missing_metadata']}")
        
        if interpretation_stats['missing_metadata'] > 0:
            cursor.execute("""
                SELECT 
                    tc.id as chunk_id,
                    tc.source_id,
                    i.org,
                    i.doc_id,
                    i.title
                FROM text_chunks tc
                LEFT JOIN interpretation_paragraphs ip ON tc.source_id = ip.id
                LEFT JOIN interpretations i ON ip.interpretation_id = i.id
                WHERE tc.source_type = 'interpretation_paragraph'
                AND (tc.meta IS NULL OR tc.meta = '')
                AND ip.id IS NOT NULL
                AND i.id IS NOT NULL
            """)
            
            interpretation_rows = cursor.fetchall()
            logger.info(f"  메타데이터 보완 가능한 청크: {len(interpretation_rows)}개")
            
            if not dry_run:
                interpretation_updated = 0
                for row in interpretation_rows:
                    try:
                        import json
                        metadata = {
                            'org': row['org'],
                            'doc_id': row['doc_id'],
                            'title': row['title']
                        }
                        metadata_json = json.dumps(metadata, ensure_ascii=False)
                        
                        cursor.execute("""
                            UPDATE text_chunks
                            SET meta = ?
                            WHERE id = ?
                        """, (metadata_json, row['chunk_id']))
                        
                        interpretation_updated += 1
                    except Exception as e:
                        logger.debug(f"Failed to update metadata for chunk_id={row['chunk_id']}: {e}")
                
                total_updated += interpretation_updated
                logger.info(f"  ✅ {interpretation_updated}개 청크의 메타데이터 보완 완료")
            else:
                if interpretation_rows:
                    logger.info("  샘플:")
                    for row in interpretation_rows[:3]:
                        logger.info(f"    청크 ID: {row['chunk_id']}, doc_id: {row['doc_id']}, org: {row['org']}, title: {row['title']}")
        
        conn.commit()
        
        if total_updated > 0:
            logger.info(f"\n✅ 총 {total_updated}개 청크의 메타데이터 보완 완료")
        
        return {
            'case_updated': case_stats.get('missing_metadata', 0) if dry_run else 0,
            'decision_updated': decision_stats.get('missing_metadata', 0) if dry_run else 0,
            'statute_updated': statute_stats.get('missing_metadata', 0) if dry_run else 0,
            'interpretation_updated': interpretation_stats.get('missing_metadata', 0) if dry_run else 0,
            'total_updated': total_updated
        }
        
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ 메타데이터 보완 실패: {e}", exc_info=True)
        return {'total_updated': 0, 'error': str(e)}
    finally:
        conn.close()


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='데이터 정합성 개선 스크립트')
    parser.add_argument('--db-path', type=str, default=None, help='데이터베이스 경로')
    parser.add_argument('--dry-run', action='store_true', help='실제로 업데이트하지 않고 통계만 조회')
    parser.add_argument('--fix-versions', action='store_true', help='embedding_version_id NULL 값 수정')
    parser.add_argument('--fix-metadata', action='store_true', help='메타데이터 필드 보완')
    
    args = parser.parse_args()
    
    # 데이터베이스 경로 결정
    if args.db_path:
        db_path = args.db_path
    else:
        db_path = project_root / "data" / "lawfirm_v2.db"
    
    if not Path(db_path).exists():
        logger.error(f"❌ 데이터베이스 파일을 찾을 수 없습니다: {db_path}")
        return
    
    logger.info(f"데이터베이스: {db_path}")
    logger.info(f"DRY RUN 모드: {args.dry_run}")
    
    # embedding_version_id 수정
    if args.fix_versions or (not args.fix_metadata):
        fix_embedding_version_ids(str(db_path), dry_run=args.dry_run)
    
    # 메타데이터 보완
    if args.fix_metadata or (not args.fix_versions):
        fix_metadata_fields(str(db_path), dry_run=args.dry_run)
    
    logger.info("\n" + "="*80)
    logger.info("데이터 정합성 개선 완료")
    logger.info("="*80)


if __name__ == "__main__":
    main()

