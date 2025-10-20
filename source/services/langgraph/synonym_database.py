#!/usr/bin/env python3
"""
동의어 데이터베이스 관리 시스템
SQLite를 사용한 동의어 저장, 관리, 최적화 시스템
"""

import sqlite3
import json
import os
import time
from typing import List, Dict, Set, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from pathlib import Path

@dataclass
class SynonymRecord:
    """동의어 레코드"""
    keyword: str
    synonym: str
    domain: str
    context: str
    confidence: float
    usage_count: int = 0
    user_rating: float = 0.0
    source: str = "unknown"
    created_at: str = ""
    last_used: str = ""
    is_active: bool = True

class SynonymDatabase:
    """동의어 데이터베이스 관리 클래스"""
    
    def __init__(self, db_path: str = "data/synonym_database.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
    
    def _initialize_database(self):
        """데이터베이스 초기화"""
        # 디렉토리 생성
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        conn.text_factory = str  # UTF-8 텍스트 처리
        cursor = conn.cursor()
        
        # 동의어 테이블 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS synonyms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL,
                synonym TEXT NOT NULL,
                domain TEXT DEFAULT 'general',
                context TEXT DEFAULT 'general',
                confidence REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                user_rating REAL DEFAULT 0.0,
                source TEXT DEFAULT 'unknown',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                UNIQUE(keyword, synonym, domain, context)
            )
        ''')
        
        # 동의어 사용 통계 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS synonym_usage_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                synonym_id INTEGER REFERENCES synonyms(id),
                usage_date DATE,
                usage_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                UNIQUE(synonym_id, usage_date)
            )
        ''')
        
        # 동의어 품질 평가 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS synonym_quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                synonym_id INTEGER REFERENCES synonyms(id),
                semantic_similarity REAL DEFAULT 0.0,
                context_relevance REAL DEFAULT 0.0,
                domain_relevance REAL DEFAULT 0.0,
                user_feedback_score REAL DEFAULT 0.0,
                overall_score REAL DEFAULT 0.0,
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
        conn.close()
        
        self.logger.info(f"동의어 데이터베이스 초기화 완료: {self.db_path}")
    
    def save_synonym(self, synonym_record: SynonymRecord) -> bool:
        """동의어 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.text_factory = str  # UTF-8 텍스트 처리
            cursor = conn.cursor()
            
            # 현재 시간 설정
            current_time = datetime.now().isoformat()
            if not synonym_record.created_at:
                synonym_record.created_at = current_time
            
            # INSERT 또는 UPDATE
            cursor.execute('''
                INSERT OR REPLACE INTO synonyms 
                (keyword, synonym, domain, context, confidence, usage_count, 
                 user_rating, source, created_at, last_used, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                synonym_record.keyword,
                synonym_record.synonym,
                synonym_record.domain,
                synonym_record.context,
                synonym_record.confidence,
                synonym_record.usage_count,
                synonym_record.user_rating,
                synonym_record.source,
                synonym_record.created_at,
                synonym_record.last_used or current_time,
                synonym_record.is_active
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"동의어 저장 완료: {synonym_record.keyword} -> {synonym_record.synonym}")
            return True
            
        except Exception as e:
            self.logger.error(f"동의어 저장 실패: {e}")
            return False
    
    def save_multiple_synonyms(self, synonym_records: List[SynonymRecord]) -> int:
        """여러 동의어 일괄 저장"""
        success_count = 0
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.text_factory = str  # UTF-8 텍스트 처리
            cursor = conn.cursor()
            
            current_time = datetime.now().isoformat()
            
            for record in synonym_records:
                try:
                    if not record.created_at:
                        record.created_at = current_time
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO synonyms 
                        (keyword, synonym, domain, context, confidence, usage_count, 
                         user_rating, source, created_at, last_used, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        record.keyword,
                        record.synonym,
                        record.domain,
                        record.context,
                        record.confidence,
                        record.usage_count,
                        record.user_rating,
                        record.source,
                        record.created_at,
                        record.last_used or current_time,
                        record.is_active
                    ))
                    success_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"동의어 저장 실패: {record.keyword} -> {record.synonym}, {e}")
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"일괄 저장 완료: {success_count}/{len(synonym_records)}")
            return success_count
            
        except Exception as e:
            self.logger.error(f"일괄 저장 실패: {e}")
            return success_count
    
    def get_synonyms(self, keyword: str, domain: str = None, 
                    context: str = None, limit: int = None) -> List[SynonymRecord]:
        """동의어 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.text_factory = str  # UTF-8 텍스트 처리
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = '''
                SELECT * FROM synonyms 
                WHERE keyword = ? AND is_active = TRUE
            '''
            params = [keyword]
            
            if domain:
                query += ' AND domain = ?'
                params.append(domain)
            
            if context:
                query += ' AND context = ?'
                params.append(context)
            
            query += ' ORDER BY confidence DESC, usage_count DESC'
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            self.logger.info(f"동의어 조회 쿼리: {query}")
            self.logger.info(f"동의어 조회 파라미터: {params}")
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            self.logger.info(f"조회된 행 수: {len(rows)}")
            
            synonyms = []
            for row in rows:
                synonym = SynonymRecord(
                    keyword=row['keyword'],
                    synonym=row['synonym'],
                    domain=row['domain'],
                    context=row['context'],
                    confidence=row['confidence'],
                    usage_count=row['usage_count'],
                    user_rating=row['user_rating'],
                    source=row['source'],
                    created_at=row['created_at'],
                    last_used=row['last_used'],
                    is_active=bool(row['is_active'])
                )
                synonyms.append(synonym)
                self.logger.info(f"동의어 추가: {synonym.keyword} -> {synonym.synonym}")
            
            conn.close()
            return synonyms
            
        except Exception as e:
            self.logger.error(f"동의어 조회 실패: {e}")
            return []
    
    def update_usage_count(self, keyword: str, synonym: str, 
                          domain: str = None, context: str = None) -> bool:
        """사용 횟수 업데이트"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.text_factory = str  # UTF-8 텍스트 처리
            cursor = conn.cursor()
            
            query = '''
                UPDATE synonyms 
                SET usage_count = usage_count + 1, last_used = ?
                WHERE keyword = ? AND synonym = ?
            '''
            params = [datetime.now().isoformat(), keyword, synonym]
            
            if domain:
                query += ' AND domain = ?'
                params.append(domain)
            
            if context:
                query += ' AND context = ?'
                params.append(context)
            
            cursor.execute(query, params)
            conn.commit()
            conn.close()
            
            return cursor.rowcount > 0
            
        except Exception as e:
            self.logger.error(f"사용 횟수 업데이트 실패: {e}")
            return False
    
    def update_user_rating(self, keyword: str, synonym: str, rating: float,
                          domain: str = None, context: str = None) -> bool:
        """사용자 평점 업데이트"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.text_factory = str  # UTF-8 텍스트 처리
            cursor = conn.cursor()
            
            # 기존 평점과 새 평점의 평균 계산
            cursor.execute('''
                SELECT user_rating, usage_count FROM synonyms 
                WHERE keyword = ? AND synonym = ?
            ''', (keyword, synonym))
            
            row = cursor.fetchone()
            if row:
                old_rating, usage_count = row
                if old_rating > 0:
                    new_rating = (old_rating * usage_count + rating) / (usage_count + 1)
                else:
                    new_rating = rating
                
                query = '''
                    UPDATE synonyms 
                    SET user_rating = ?, usage_count = usage_count + 1
                    WHERE keyword = ? AND synonym = ?
                '''
                params = [new_rating, keyword, synonym]
                
                if domain:
                    query += ' AND domain = ?'
                    params.append(domain)
                
                if context:
                    query += ' AND context = ?'
                    params.append(context)
                
                cursor.execute(query, params)
                conn.commit()
                conn.close()
                
                return cursor.rowcount > 0
            
            conn.close()
            return False
            
        except Exception as e:
            self.logger.error(f"사용자 평점 업데이트 실패: {e}")
            return False
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """데이터베이스 통계 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.text_factory = str  # UTF-8 텍스트 처리
            cursor = conn.cursor()
            
            # 전체 동의어 수
            cursor.execute('SELECT COUNT(*) FROM synonyms WHERE is_active = TRUE')
            total_synonyms = cursor.fetchone()[0]
            
            # 키워드 수
            cursor.execute('SELECT COUNT(DISTINCT keyword) FROM synonyms WHERE is_active = TRUE')
            total_keywords = cursor.fetchone()[0]
            
            # 도메인별 통계
            cursor.execute('''
                SELECT domain, COUNT(*) as count 
                FROM synonyms 
                WHERE is_active = TRUE 
                GROUP BY domain 
                ORDER BY count DESC
            ''')
            domain_stats = dict(cursor.fetchall())
            
            # 평균 신뢰도
            cursor.execute('SELECT AVG(confidence) FROM synonyms WHERE is_active = TRUE')
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            # 평균 사용 횟수
            cursor.execute('SELECT AVG(usage_count) FROM synonyms WHERE is_active = TRUE')
            avg_usage = cursor.fetchone()[0] or 0.0
            
            # 최근 생성된 동의어 (7일)
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute('''
                SELECT COUNT(*) FROM synonyms 
                WHERE created_at >= ? AND is_active = TRUE
            ''', (week_ago,))
            recent_synonyms = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total_synonyms": total_synonyms,
                "total_keywords": total_keywords,
                "domain_statistics": domain_stats,
                "average_confidence": round(avg_confidence, 3),
                "average_usage_count": round(avg_usage, 2),
                "recent_synonyms_7days": recent_synonyms,
                "database_size_mb": round(os.path.getsize(self.db_path) / (1024 * 1024), 2)
            }
            
        except Exception as e:
            self.logger.error(f"통계 조회 실패: {e}")
            return {}
    
    def cleanup_unused_synonyms(self, days_threshold: int = 30) -> int:
        """사용하지 않는 동의어 정리"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.text_factory = str  # UTF-8 텍스트 처리
            cursor = conn.cursor()
            
            # 지정된 일수 이상 사용되지 않은 동의어 비활성화
            cutoff_date = (datetime.now() - timedelta(days=days_threshold)).isoformat()
            
            cursor.execute('''
                UPDATE synonyms 
                SET is_active = FALSE 
                WHERE last_used < ? AND usage_count < 5
            ''', (cutoff_date,))
            
            affected_rows = cursor.rowcount
            conn.commit()
            conn.close()
            
            self.logger.info(f"사용하지 않는 동의어 정리 완료: {affected_rows}개")
            return affected_rows
            
        except Exception as e:
            self.logger.error(f"동의어 정리 실패: {e}")
            return 0
    
    def export_synonyms(self, output_file: str = None) -> str:
        """동의어 데이터 내보내기"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"data/synonym_export_{timestamp}.json"
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM synonyms WHERE is_active = TRUE')
            rows = cursor.fetchall()
            
            synonyms_data = []
            for row in rows:
                synonyms_data.append(dict(row))
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_synonyms": len(synonyms_data),
                "synonyms": synonyms_data
            }
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            conn.close()
            
            self.logger.info(f"동의어 데이터 내보내기 완료: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"동의어 데이터 내보내기 실패: {e}")
            return ""

# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 데이터베이스 초기화
    db = SynonymDatabase()
    
    # 테스트 동의어 저장
    test_synonyms = [
        SynonymRecord(
            keyword="계약서",
            synonym="계약문서",
            domain="민사법",
            context="계약_관련_맥락",
            confidence=0.95,
            source="gemini_api"
        ),
        SynonymRecord(
            keyword="계약서",
            synonym="계약장",
            domain="민사법",
            context="계약_관련_맥락",
            confidence=0.90,
            source="gemini_api"
        ),
        SynonymRecord(
            keyword="아파트",
            synonym="공동주택",
            domain="부동산법",
            context="부동산_거래_맥락",
            confidence=0.88,
            source="gemini_api"
        )
    ]
    
    # 동의어 저장
    saved_count = db.save_multiple_synonyms(test_synonyms)
    print(f"저장된 동의어: {saved_count}개")
    
    # 동의어 조회
    synonyms = db.get_synonyms("계약서", domain="민사법")
    print(f"조회된 동의어: {len(synonyms)}개")
    for syn in synonyms:
        print(f"  - {syn.synonym} (신뢰도: {syn.confidence})")
    
    # 통계 조회
    stats = db.get_database_statistics()
    print(f"\n데이터베이스 통계:")
    print(f"  총 동의어: {stats.get('total_synonyms', 0)}개")
    print(f"  총 키워드: {stats.get('total_keywords', 0)}개")
    print(f"  평균 신뢰도: {stats.get('average_confidence', 0)}")
    print(f"  데이터베이스 크기: {stats.get('database_size_mb', 0)}MB")
