#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Legal Terms Database Migration Script
base_legal_terms 데이터베이스에서 메인 데이터베이스로 마이그레이션
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from source.data.database import DatabaseManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseLegalTermsMigrator:
    """Base Legal Terms 마이그레이션 클래스"""
    
    def __init__(self):
        """마이그레이션 클래스 초기화"""
        self.source_db_path = Path("data/database/base_legal_terms.db")
        self.target_db_path = Path("data/lawfirm.db")
        
        # 소스 데이터베이스 연결
        self.source_conn = None
        # 타겟 데이터베이스 매니저
        self.target_db = DatabaseManager(str(self.target_db_path))
        
        logger.info(f"마이그레이션 초기화 완료")
        logger.info(f"소스 DB: {self.source_db_path}")
        logger.info(f"타겟 DB: {self.target_db_path}")
    
    def connect_source_database(self):
        """소스 데이터베이스 연결"""
        try:
            if not self.source_db_path.exists():
                raise FileNotFoundError(f"소스 데이터베이스가 존재하지 않습니다: {self.source_db_path}")
            
            self.source_conn = sqlite3.connect(str(self.source_db_path))
            self.source_conn.row_factory = sqlite3.Row
            logger.info("소스 데이터베이스 연결 성공")
            
        except Exception as e:
            logger.error(f"소스 데이터베이스 연결 실패: {e}")
            raise
    
    def close_source_database(self):
        """소스 데이터베이스 연결 종료"""
        if self.source_conn:
            self.source_conn.close()
            logger.info("소스 데이터베이스 연결 종료")
    
    def get_source_data_count(self) -> int:
        """소스 데이터베이스의 레코드 수 조회"""
        try:
            cursor = self.source_conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM base_legal_term_lists")
            result = cursor.fetchone()
            return result['count'] if result else 0
            
        except Exception as e:
            logger.error(f"소스 데이터 개수 조회 실패: {e}")
            return 0
    
    def get_source_data_batch(self, offset: int, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """소스 데이터베이스에서 배치로 데이터 조회"""
        try:
            cursor = self.source_conn.cursor()
            cursor.execute("""
                SELECT 법령용어ID, 법령용어명, 동음이의어존재여부, 비고, 
                       용어간관계링크, 조문간관계링크, 수집일시
                FROM base_legal_term_lists 
                ORDER BY id
                LIMIT ? OFFSET ?
            """, (batch_size, offset))
            
            results = []
            for row in cursor.fetchall():
                results.append(dict(row))
            
            return results
            
        except Exception as e:
            logger.error(f"소스 데이터 배치 조회 실패: {e}")
            return []
    
    def migrate_data(self, batch_size: int = 1000) -> Dict[str, Any]:
        """데이터 마이그레이션 실행"""
        logger.info("데이터 마이그레이션 시작")
        
        try:
            # 소스 데이터베이스 연결
            self.connect_source_database()
            
            # 소스 데이터 개수 확인
            total_count = self.get_source_data_count()
            logger.info(f"마이그레이션할 총 레코드 수: {total_count:,}개")
            
            if total_count == 0:
                logger.warning("마이그레이션할 데이터가 없습니다")
                return {"success": False, "message": "마이그레이션할 데이터가 없습니다"}
            
            # 타겟 데이터베이스의 기존 데이터 확인
            existing_count = self.target_db.get_base_legal_terms_count()
            logger.info(f"타겟 데이터베이스 기존 레코드 수: {existing_count:,}개")
            
            # 마이그레이션 통계
            stats = {
                "total_source": total_count,
                "existing_target": existing_count,
                "processed": 0,
                "inserted": 0,
                "updated": 0,
                "errors": 0,
                "start_time": datetime.now(),
                "end_time": None
            }
            
            # 배치별로 데이터 마이그레이션
            offset = 0
            while offset < total_count:
                logger.info(f"배치 처리 중: {offset:,} ~ {min(offset + batch_size, total_count):,}")
                
                # 소스에서 데이터 조회
                batch_data = self.get_source_data_batch(offset, batch_size)
                
                if not batch_data:
                    logger.warning(f"배치 데이터가 비어있습니다: offset={offset}")
                    break
                
                # 타겟 데이터베이스에 삽입
                try:
                    inserted_count = self.target_db.insert_base_legal_terms_batch(batch_data)
                    stats["processed"] += len(batch_data)
                    stats["inserted"] += inserted_count
                    
                    logger.info(f"배치 처리 완료: {len(batch_data)}개 조회, {inserted_count}개 삽입")
                    
                except Exception as e:
                    logger.error(f"배치 삽입 실패: {e}")
                    stats["errors"] += len(batch_data)
                
                offset += batch_size
            
            # 마이그레이션 완료
            stats["end_time"] = datetime.now()
            stats["duration"] = (stats["end_time"] - stats["start_time"]).total_seconds()
            
            # 최종 결과 확인
            final_count = self.target_db.get_base_legal_terms_count()
            stats["final_target"] = final_count
            
            logger.info("데이터 마이그레이션 완료")
            logger.info(f"마이그레이션 통계: {stats}")
            
            return {
                "success": True,
                "stats": stats,
                "message": f"마이그레이션 완료: {stats['processed']:,}개 처리, {final_count:,}개 최종 저장"
            }
            
        except Exception as e:
            logger.error(f"데이터 마이그레이션 실패: {e}")
            return {"success": False, "message": f"마이그레이션 실패: {e}"}
        
        finally:
            # 소스 데이터베이스 연결 종료
            self.close_source_database()
    
    def verify_migration(self) -> Dict[str, Any]:
        """마이그레이션 결과 검증"""
        logger.info("마이그레이션 결과 검증 시작")
        
        try:
            # 소스 데이터베이스 연결
            self.connect_source_database()
            
            # 소스 데이터 개수
            source_count = self.get_source_data_count()
            
            # 타겟 데이터 개수
            target_count = self.target_db.get_base_legal_terms_count()
            
            # 샘플 데이터 비교
            source_sample = self.get_source_data_batch(0, 10)
            target_sample = self.target_db.execute_query(
                "SELECT * FROM base_legal_term_lists ORDER BY id LIMIT 10"
            )
            
            verification_result = {
                "source_count": source_count,
                "target_count": target_count,
                "count_match": source_count == target_count,
                "source_sample": source_sample,
                "target_sample": target_sample,
                "sample_match": len(source_sample) == len(target_sample)
            }
            
            logger.info(f"검증 결과: {verification_result}")
            
            return verification_result
            
        except Exception as e:
            logger.error(f"마이그레이션 검증 실패: {e}")
            return {"error": str(e)}
        
        finally:
            self.close_source_database()


def main():
    """메인 함수"""
    print("=" * 60)
    print("Base Legal Terms Database Migration")
    print("=" * 60)
    
    migrator = BaseLegalTermsMigrator()
    
    try:
        # 마이그레이션 실행
        result = migrator.migrate_data(batch_size=1000)
        
        if result["success"]:
            print(f"✅ 마이그레이션 성공: {result['message']}")
            
            # 검증 실행
            verification = migrator.verify_migration()
            
            if verification.get("count_match", False):
                print("✅ 데이터 개수 일치 확인")
            else:
                print(f"⚠️ 데이터 개수 불일치: 소스 {verification.get('source_count', 0)}, 타겟 {verification.get('target_count', 0)}")
            
            print(f"📊 마이그레이션 통계:")
            stats = result.get("stats", {})
            print(f"   - 소스 데이터: {stats.get('total_source', 0):,}개")
            print(f"   - 처리된 데이터: {stats.get('processed', 0):,}개")
            print(f"   - 삽입된 데이터: {stats.get('inserted', 0):,}개")
            print(f"   - 오류 발생: {stats.get('errors', 0):,}개")
            print(f"   - 소요 시간: {stats.get('duration', 0):.2f}초")
            
        else:
            print(f"❌ 마이그레이션 실패: {result['message']}")
            return 1
    
    except Exception as e:
        print(f"❌ 마이그레이션 중 오류 발생: {e}")
        return 1
    
    print("=" * 60)
    print("마이그레이션 완료")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
