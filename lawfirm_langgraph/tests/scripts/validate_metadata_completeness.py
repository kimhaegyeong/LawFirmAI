# -*- coding: utf-8 -*-
"""
메타데이터 완전성 검증 스크립트
- 모든 source_type의 메타데이터 완전성 검증
- 데이터베이스 전체 통계 수집
"""

import sys
import os
import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_metadata_completeness(db_path: str) -> Dict[str, Any]:
    """
    모든 source_type의 메타데이터 완전성 검증
    
    Args:
        db_path: 데이터베이스 경로
        
    Returns:
        검증 결과 통계
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        cursor = conn.cursor()
        
        logger.info("="*80)
        logger.info("메타데이터 완전성 검증")
        logger.info("="*80)
        
        results = {}
        source_types = ['case_paragraph', 'decision_paragraph', 'statute_article', 'interpretation_paragraph']
        
        for source_type in source_types:
            logger.info(f"\n[{source_type}]")
            
            # 전체 청크 수 및 메타데이터 누락 수
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(CASE WHEN meta IS NULL OR meta = '' THEN 1 END) as missing_metadata,
                    COUNT(CASE WHEN meta IS NOT NULL AND meta != '' THEN 1 END) as has_metadata
                FROM text_chunks
                WHERE source_type = ?
            """, (source_type,))
            
            stats = dict(cursor.fetchone())
            total = stats['total_chunks']
            missing = stats['missing_metadata']
            has_meta = stats['has_metadata']
            
            logger.info(f"  전체 청크 수: {total:,}")
            logger.info(f"  메타데이터 있음: {has_meta:,} ({has_meta/total*100:.1f}%)")
            logger.info(f"  메타데이터 누락: {missing:,} ({missing/total*100:.1f}%)")
            
            # 메타데이터가 있는 청크의 필드 완전성 검증
            if has_meta > 0:
                cursor.execute("""
                    SELECT id, meta
                    FROM text_chunks
                    WHERE source_type = ? AND meta IS NOT NULL AND meta != ''
                    LIMIT 1000
                """, (source_type,))
                
                rows = cursor.fetchall()
                field_stats = defaultdict(int)
                valid_count = 0
                invalid_count = 0
                
                required_fields = {
                    'case_paragraph': ['doc_id', 'casenames', 'court'],
                    'decision_paragraph': ['doc_id', 'org'],
                    'statute_article': ['statute_name', 'article_no'],
                    'interpretation_paragraph': ['doc_id', 'org', 'title']
                }
                
                required = required_fields.get(source_type, [])
                
                for row in rows:
                    try:
                        meta_json = json.loads(row['meta'])
                        has_all = True
                        
                        for field in required:
                            if field in meta_json and meta_json[field]:
                                field_stats[f'has_{field}'] += 1
                            else:
                                has_all = False
                        
                        if has_all:
                            valid_count += 1
                        else:
                            invalid_count += 1
                    except Exception as e:
                        invalid_count += 1
                        logger.debug(f"Failed to parse meta for chunk_id={row['id']}: {e}")
                
                logger.info(f"\n  메타데이터 필드 완전성 (샘플 {len(rows)}개):")
                for field in required:
                    count = field_stats.get(f'has_{field}', 0)
                    percentage = count / len(rows) * 100 if rows else 0
                    logger.info(f"    {field}: {count}/{len(rows)} ({percentage:.1f}%)")
                
                logger.info(f"  완전한 메타데이터: {valid_count}/{len(rows)} ({valid_count/len(rows)*100:.1f}%)")
                logger.info(f"  불완전한 메타데이터: {invalid_count}/{len(rows)} ({invalid_count/len(rows)*100:.1f}%)")
            
            results[source_type] = {
                'total': total,
                'has_metadata': has_meta,
                'missing_metadata': missing,
                'completeness_rate': (has_meta / total * 100) if total > 0 else 0
            }
        
        # 전체 통계
        logger.info("\n" + "="*80)
        logger.info("전체 통계")
        logger.info("="*80)
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(CASE WHEN meta IS NULL OR meta = '' THEN 1 END) as missing_metadata,
                COUNT(CASE WHEN meta IS NOT NULL AND meta != '' THEN 1 END) as has_metadata
            FROM text_chunks
        """)
        
        overall_stats = dict(cursor.fetchone())
        total_all = overall_stats['total_chunks']
        missing_all = overall_stats['missing_metadata']
        has_meta_all = overall_stats['has_metadata']
        
        logger.info(f"전체 청크 수: {total_all:,}")
        logger.info(f"메타데이터 있음: {has_meta_all:,} ({has_meta_all/total_all*100:.1f}%)")
        logger.info(f"메타데이터 누락: {missing_all:,} ({missing_all/total_all*100:.1f}%)")
        
        results['overall'] = {
            'total': total_all,
            'has_metadata': has_meta_all,
            'missing_metadata': missing_all,
            'completeness_rate': (has_meta_all / total_all * 100) if total_all > 0 else 0
        }
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 검증 실패: {e}", exc_info=True)
        return {}
    finally:
        conn.close()


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='메타데이터 완전성 검증 스크립트')
    parser.add_argument('--db-path', type=str, default=None, help='데이터베이스 경로')
    
    args = parser.parse_args()
    
    # 데이터베이스 경로 결정
    if args.db_path:
        db_path = args.db_path
    else:
        db_path = project_root / "data" / "lawfirm_v2.db"
    
    if not Path(db_path).exists():
        logger.error(f"❌ 데이터베이스 파일을 찾을 수 없습니다: {db_path}")
        return 1
    
    logger.info(f"데이터베이스: {db_path}")
    
    results = validate_metadata_completeness(str(db_path))
    
    logger.info("\n" + "="*80)
    logger.info("검증 완료")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

