#!/usr/bin/env python3
"""
동의어 데이터베이스 SQLite → PostgreSQL 마이그레이션 스크립트
"""

import os
import sys
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from lawfirm_langgraph.core.data.db_adapter import DatabaseAdapter
    from lawfirm_langgraph.core.utils.logger import get_logger
    from lawfirm_langgraph.config.app_config import Config
except ImportError:
    print("Error: Required modules not found. Make sure you're running from the project root.")
    sys.exit(1)

logger = get_logger(__name__)


def get_sqlite_data(sqlite_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """SQLite 데이터베이스에서 모든 데이터 읽기"""
    if not os.path.exists(sqlite_path):
        logger.warning(f"SQLite 파일이 존재하지 않습니다: {sqlite_path}")
        return {
            "synonyms": [],
            "synonym_usage_stats": [],
            "synonym_quality_metrics": []
        }
    
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    data = {}
    
    # synonyms 테이블 데이터
    try:
        cursor.execute("SELECT * FROM synonyms")
        rows = cursor.fetchall()
        data["synonyms"] = [dict(row) for row in rows]
        logger.info(f"synonyms 테이블에서 {len(data['synonyms'])}개 레코드 읽기 완료")
    except sqlite3.OperationalError as e:
        logger.warning(f"synonyms 테이블이 없거나 읽기 실패: {e}")
        data["synonyms"] = []
    
    # synonym_usage_stats 테이블 데이터
    try:
        cursor.execute("SELECT * FROM synonym_usage_stats")
        rows = cursor.fetchall()
        data["synonym_usage_stats"] = [dict(row) for row in rows]
        logger.info(f"synonym_usage_stats 테이블에서 {len(data['synonym_usage_stats'])}개 레코드 읽기 완료")
    except sqlite3.OperationalError as e:
        logger.warning(f"synonym_usage_stats 테이블이 없거나 읽기 실패: {e}")
        data["synonym_usage_stats"] = []
    
    # synonym_quality_metrics 테이블 데이터
    try:
        cursor.execute("SELECT * FROM synonym_quality_metrics")
        rows = cursor.fetchall()
        data["synonym_quality_metrics"] = [dict(row) for row in rows]
        logger.info(f"synonym_quality_metrics 테이블에서 {len(data['synonym_quality_metrics'])}개 레코드 읽기 완료")
    except sqlite3.OperationalError as e:
        logger.warning(f"synonym_quality_metrics 테이블이 없거나 읽기 실패: {e}")
        data["synonym_quality_metrics"] = []
    
    conn.close()
    return data


def migrate_to_postgresql(sqlite_path: str, database_url: str) -> bool:
    """SQLite 데이터를 PostgreSQL로 마이그레이션"""
    logger.info("=== 동의어 데이터베이스 마이그레이션 시작 ===")
    logger.info(f"SQLite 경로: {sqlite_path}")
    logger.info(f"PostgreSQL URL: {database_url[:50]}...")
    
    # SQLite 데이터 읽기
    logger.info("SQLite 데이터 읽기 중...")
    sqlite_data = get_sqlite_data(sqlite_path)
    
    total_records = (
        len(sqlite_data["synonyms"]) +
        len(sqlite_data["synonym_usage_stats"]) +
        len(sqlite_data["synonym_quality_metrics"])
    )
    
    if total_records == 0:
        logger.warning("마이그레이션할 데이터가 없습니다.")
        return True
    
    logger.info(f"총 {total_records}개 레코드 마이그레이션 예정")
    
    # PostgreSQL 연결
    try:
        db_adapter = DatabaseAdapter(database_url)
        with db_adapter.get_connection_context() as conn:
            cursor = conn.cursor()
        
            # 테이블 생성 (synonym_database.py의 _initialize_database와 동일한 스키마)
            logger.info("PostgreSQL 테이블 생성 중...")
            
            # synonyms 테이블
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS synonyms (
                id SERIAL PRIMARY KEY,
                keyword VARCHAR(255) NOT NULL,
                synonym VARCHAR(255) NOT NULL,
                domain VARCHAR(100) DEFAULT 'general',
                context VARCHAR(100) DEFAULT 'general',
                confidence DOUBLE PRECISION DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                user_rating DOUBLE PRECISION DEFAULT 0.0,
                source VARCHAR(100) DEFAULT 'unknown',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                UNIQUE(keyword, synonym, domain, context)
            )
        ''')
        
            # synonym_usage_stats 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS synonym_usage_stats (
                    id SERIAL PRIMARY KEY,
                    synonym_id INTEGER REFERENCES synonyms(id) ON DELETE CASCADE,
                    usage_date DATE,
                    usage_count INTEGER DEFAULT 0,
                    success_rate DOUBLE PRECISION DEFAULT 0.0,
                    UNIQUE(synonym_id, usage_date)
                )
            ''')
            
            # synonym_quality_metrics 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS synonym_quality_metrics (
                    id SERIAL PRIMARY KEY,
                    synonym_id INTEGER REFERENCES synonyms(id) ON DELETE CASCADE,
                    semantic_similarity DOUBLE PRECISION DEFAULT 0.0,
                    context_relevance DOUBLE PRECISION DEFAULT 0.0,
                    domain_relevance DOUBLE PRECISION DEFAULT 0.0,
                    user_feedback_score DOUBLE PRECISION DEFAULT 0.0,
                    overall_score DOUBLE PRECISION DEFAULT 0.0,
                    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 인덱스 생성
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_synonyms_keyword ON synonyms(keyword)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_synonyms_domain ON synonyms(domain)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_synonyms_usage ON synonyms(usage_count DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_synonyms_active ON synonyms(is_active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_synonyms_confidence ON synonyms(confidence DESC)')
            
            conn.commit()
            logger.info("PostgreSQL 테이블 생성 완료")
            
            # 데이터 마이그레이션
            logger.info("데이터 마이그레이션 시작...")
            
            # synonyms 테이블 마이그레이션
            if sqlite_data["synonyms"]:
                logger.info(f"synonyms 테이블 {len(sqlite_data['synonyms'])}개 레코드 마이그레이션 중...")
                synonyms_inserted = 0
                synonyms_skipped = 0
                
                for row in sqlite_data["synonyms"]:
                    try:
                        cursor.execute('''
                            INSERT INTO synonyms 
                            (keyword, synonym, domain, context, confidence, usage_count, 
                             user_rating, source, created_at, last_used, is_active)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (keyword, synonym, domain, context) 
                            DO UPDATE SET
                                confidence = EXCLUDED.confidence,
                                usage_count = EXCLUDED.usage_count,
                                user_rating = EXCLUDED.user_rating,
                                source = EXCLUDED.source,
                                last_used = EXCLUDED.last_used,
                                is_active = EXCLUDED.is_active
                        ''', (
                            row.get('keyword'),
                            row.get('synonym'),
                            row.get('domain', 'general'),
                            row.get('context', 'general'),
                            row.get('confidence', 0.0),
                            row.get('usage_count', 0),
                            row.get('user_rating', 0.0),
                            row.get('source', 'unknown'),
                            row.get('created_at'),
                            row.get('last_used'),
                            bool(row.get('is_active', True))
                        ))
                        synonyms_inserted += 1
                    except Exception as e:
                        logger.warning(f"synonyms 레코드 마이그레이션 실패: {e}, 레코드: {row}")
                        synonyms_skipped += 1
                
                conn.commit()
                logger.info(f"synonyms 테이블 마이그레이션 완료: {synonyms_inserted}개 삽입, {synonyms_skipped}개 건너뜀")
            
            # synonym_id 매핑 생성 (synonym_usage_stats와 synonym_quality_metrics에서 사용)
            logger.info("synonym_id 매핑 생성 중...")
            cursor.execute("SELECT id, keyword, synonym, domain, context FROM synonyms")
            synonym_mapping = {}
            for row in cursor.fetchall():
                key = (row['keyword'], row['synonym'], row['domain'], row['context'])  # (keyword, synonym, domain, context)
                synonym_mapping[key] = row['id']  # id
            
            logger.info(f"synonym_id 매핑 생성 완료: {len(synonym_mapping)}개")
            
            # synonym_usage_stats 테이블 마이그레이션
            if sqlite_data["synonym_usage_stats"]:
                logger.info(f"synonym_usage_stats 테이블 {len(sqlite_data['synonym_usage_stats'])}개 레코드 마이그레이션 중...")
                stats_inserted = 0
                stats_skipped = 0
                
                for row in sqlite_data["synonym_usage_stats"]:
                    # SQLite의 synonym_id로 PostgreSQL의 id 찾기
                    # SQLite의 id를 사용하여 매핑 (id가 직접 매칭되는 경우)
                    sqlite_synonym_id = row.get('synonym_id')
                    
                    # SQLite의 synonyms 테이블에서 해당 id의 레코드 찾기
                    sqlite_conn = sqlite3.connect(sqlite_path)
                    sqlite_conn.row_factory = sqlite3.Row
                    sqlite_cursor = sqlite_conn.cursor()
                    sqlite_cursor.execute("SELECT keyword, synonym, domain, context FROM synonyms WHERE id = ?", (sqlite_synonym_id,))
                    sqlite_row = sqlite_cursor.fetchone()
                    sqlite_conn.close()
                    
                    if sqlite_row:
                        key = (sqlite_row['keyword'], sqlite_row['synonym'], sqlite_row['domain'], sqlite_row['context'])
                        postgres_synonym_id = synonym_mapping.get(key)
                        
                        if postgres_synonym_id:
                            try:
                                cursor.execute('''
                                    INSERT INTO synonym_usage_stats 
                                    (synonym_id, usage_date, usage_count, success_rate)
                                    VALUES (%s, %s, %s, %s)
                                    ON CONFLICT (synonym_id, usage_date) 
                                    DO UPDATE SET
                                        usage_count = EXCLUDED.usage_count,
                                        success_rate = EXCLUDED.success_rate
                                ''', (
                                    postgres_synonym_id,
                                    row.get('usage_date'),
                                    row.get('usage_count', 0),
                                    row.get('success_rate', 0.0)
                                ))
                                stats_inserted += 1
                            except Exception as e:
                                logger.warning(f"synonym_usage_stats 레코드 마이그레이션 실패: {e}")
                                stats_skipped += 1
                        else:
                            logger.warning(f"synonym_id 매핑을 찾을 수 없음: {key}")
                            stats_skipped += 1
                    else:
                        logger.warning(f"SQLite에서 synonym_id {sqlite_synonym_id}를 찾을 수 없음")
                        stats_skipped += 1
                
                conn.commit()
                logger.info(f"synonym_usage_stats 테이블 마이그레이션 완료: {stats_inserted}개 삽입, {stats_skipped}개 건너뜀")
            
            # synonym_quality_metrics 테이블 마이그레이션
            if sqlite_data["synonym_quality_metrics"]:
                logger.info(f"synonym_quality_metrics 테이블 {len(sqlite_data['synonym_quality_metrics'])}개 레코드 마이그레이션 중...")
                metrics_inserted = 0
                metrics_skipped = 0
                
                for row in sqlite_data["synonym_quality_metrics"]:
                    sqlite_synonym_id = row.get('synonym_id')
                    
                    # SQLite의 synonyms 테이블에서 해당 id의 레코드 찾기
                    sqlite_conn = sqlite3.connect(sqlite_path)
                    sqlite_conn.row_factory = sqlite3.Row
                    sqlite_cursor = sqlite_conn.cursor()
                    sqlite_cursor.execute("SELECT keyword, synonym, domain, context FROM synonyms WHERE id = ?", (sqlite_synonym_id,))
                    sqlite_row = sqlite_cursor.fetchone()
                    sqlite_conn.close()
                    
                    if sqlite_row:
                        key = (sqlite_row['keyword'], sqlite_row['synonym'], sqlite_row['domain'], sqlite_row['context'])
                        postgres_synonym_id = synonym_mapping.get(key)
                        
                        if postgres_synonym_id:
                            try:
                                cursor.execute('''
                                    INSERT INTO synonym_quality_metrics 
                                    (synonym_id, semantic_similarity, context_relevance, 
                                     domain_relevance, user_feedback_score, overall_score, evaluated_at)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                                ''', (
                                    postgres_synonym_id,
                                    row.get('semantic_similarity', 0.0),
                                    row.get('context_relevance', 0.0),
                                    row.get('domain_relevance', 0.0),
                                    row.get('user_feedback_score', 0.0),
                                    row.get('overall_score', 0.0),
                                    row.get('evaluated_at')
                                ))
                                metrics_inserted += 1
                            except Exception as e:
                                logger.warning(f"synonym_quality_metrics 레코드 마이그레이션 실패: {e}")
                                metrics_skipped += 1
                        else:
                            logger.warning(f"synonym_id 매핑을 찾을 수 없음: {key}")
                            metrics_skipped += 1
                    else:
                        logger.warning(f"SQLite에서 synonym_id {sqlite_synonym_id}를 찾을 수 없음")
                        metrics_skipped += 1
                
                conn.commit()
                logger.info(f"synonym_quality_metrics 테이블 마이그레이션 완료: {metrics_inserted}개 삽입, {metrics_skipped}개 건너뜀")
            
            # 마이그레이션 결과 확인
            cursor.execute("SELECT COUNT(*) as count FROM synonyms")
            postgres_count = cursor.fetchone()['count']
            
            logger.info("=== 마이그레이션 완료 ===")
            logger.info(f"PostgreSQL synonyms 테이블 레코드 수: {postgres_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"마이그레이션 실패: {e}", exc_info=True)
        return False


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="동의어 데이터베이스 SQLite → PostgreSQL 마이그레이션")
    parser.add_argument(
        "--sqlite-path",
        type=str,
        default="data/synonym_database.db",
        help="SQLite 데이터베이스 파일 경로 (기본값: data/synonym_database.db)"
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="PostgreSQL 데이터베이스 URL (기본값: 환경 변수에서 가져옴)"
    )
    
    args = parser.parse_args()
    
    # PostgreSQL URL 가져오기
    if args.database_url:
        database_url = args.database_url
    else:
        config = Config()
        database_url = config.database_url
    
    if not database_url:
        logger.error("PostgreSQL 데이터베이스 URL이 설정되지 않았습니다.")
        logger.error("환경 변수 DATABASE_URL 또는 POSTGRES_* 변수를 설정하거나 --database-url 옵션을 사용하세요.")
        sys.exit(1)
    
    # 마이그레이션 실행
    success = migrate_to_postgresql(args.sqlite_path, database_url)
    
    if success:
        logger.info("마이그레이션이 성공적으로 완료되었습니다.")
        sys.exit(0)
    else:
        logger.error("마이그레이션이 실패했습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()

