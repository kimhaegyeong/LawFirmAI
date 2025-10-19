#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터베이스 기반 키워드 관리 시스템
SQLite를 사용하여 질문 유형별 키워드를 효율적으로 관리
"""

import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseKeywordManager:
    """데이터베이스 기반 키워드 관리자"""
    
    def __init__(self, db_path: str = "data/question_keywords.db"):
        self.db_path = db_path
        self._ensure_db_directory()
        self._init_database()
        logger.info(f"DatabaseKeywordManager initialized with DB: {db_path}")
    
    def _ensure_db_directory(self):
        """데이터베이스 디렉토리 생성"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """데이터베이스 초기화 및 테이블 생성"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 키워드 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS keywords (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_type TEXT NOT NULL,
                    keyword TEXT NOT NULL,
                    weight_level TEXT NOT NULL CHECK(weight_level IN ('high', 'medium', 'low')),
                    weight_value REAL NOT NULL DEFAULT 1.0,
                    category TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    UNIQUE(question_type, keyword)
                )
            ''')
            
            # 패턴 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_type TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    pattern_type TEXT NOT NULL CHECK(pattern_type IN ('regex', 'keyword', 'phrase')),
                    priority INTEGER DEFAULT 1,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # 질문 유형 메타데이터 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS question_types (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type_name TEXT UNIQUE NOT NULL,
                    display_name TEXT NOT NULL,
                    description TEXT,
                    parent_type TEXT,
                    priority INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # 키워드 통계 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS keyword_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    keyword_id INTEGER NOT NULL,
                    match_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    last_matched_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (keyword_id) REFERENCES keywords (id)
                )
            ''')
            
            # 인덱스 생성
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_keywords_type_weight ON keywords(question_type, weight_level)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON keywords(keyword)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(question_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_question_types_name ON question_types(type_name)')
            
            conn.commit()
            logger.info("Database tables and indexes created successfully")
    
    def add_keyword(self, question_type: str, keyword: str, weight_level: str, 
                   weight_value: float = None, category: str = None, 
                   description: str = None) -> bool:
        """키워드 추가"""
        try:
            # 가중치 값 설정
            if weight_value is None:
                weight_value = self._get_default_weight(weight_level)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO keywords 
                    (question_type, keyword, weight_level, weight_value, category, description, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (question_type, keyword, weight_level, weight_value, category, description))
                
                conn.commit()
                logger.info(f"Keyword added: {question_type} - {keyword} ({weight_level})")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Failed to add keyword: {e}")
            return False
    
    def add_keywords_batch(self, keywords_data: List[Dict[str, Any]]) -> int:
        """키워드 일괄 추가"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                success_count = 0
                for data in keywords_data:
                    try:
                        cursor.execute('''
                            INSERT OR REPLACE INTO keywords 
                            (question_type, keyword, weight_level, weight_value, category, description, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        ''', (
                            data['question_type'],
                            data['keyword'],
                            data['weight_level'],
                            data.get('weight_value', self._get_default_weight(data['weight_level'])),
                            data.get('category'),
                            data.get('description')
                        ))
                        success_count += 1
                    except sqlite3.Error as e:
                        logger.error(f"Failed to add keyword {data}: {e}")
                
                conn.commit()
                logger.info(f"Batch added {success_count}/{len(keywords_data)} keywords")
                return success_count
                
        except sqlite3.Error as e:
            logger.error(f"Failed to add keywords batch: {e}")
            return 0
    
    def get_keywords_for_type(self, question_type: str, weight_level: str = None, 
                            limit: int = 100) -> List[Dict[str, Any]]:
        """질문 유형별 키워드 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = '''
                    SELECT id, keyword, weight_level, weight_value, category, description, 
                           created_at, updated_at, is_active
                    FROM keywords 
                    WHERE question_type = ? AND is_active = 1
                '''
                params = [question_type]
                
                if weight_level:
                    query += ' AND weight_level = ?'
                    params.append(weight_level)
                
                query += ' ORDER BY weight_value DESC, keyword LIMIT ?'
                params.append(limit)
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                return [dict(row) for row in results]
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get keywords for type {question_type}: {e}")
            return []
    
    def search_keywords(self, query: str, question_type: str = None, 
                       weight_level: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """키워드 검색"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                sql_query = '''
                    SELECT id, question_type, keyword, weight_level, weight_value, 
                           category, description, created_at, updated_at
                    FROM keywords 
                    WHERE keyword LIKE ? AND is_active = 1
                '''
                params = [f'%{query}%']
                
                if question_type:
                    sql_query += ' AND question_type = ?'
                    params.append(question_type)
                
                if weight_level:
                    sql_query += ' AND weight_level = ?'
                    params.append(weight_level)
                
                sql_query += ' ORDER BY weight_value DESC, keyword LIMIT 50'
                
                cursor.execute(sql_query, params)
                results = cursor.fetchall()
                
                return [dict(row) for row in results]
                
        except sqlite3.Error as e:
            logger.error(f"Failed to search keywords: {e}")
            return []
    
    def add_pattern(self, question_type: str, pattern: str, pattern_type: str = 'regex',
                   priority: int = 1, description: str = None) -> bool:
        """패턴 추가"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO patterns 
                    (question_type, pattern, pattern_type, priority, description, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (question_type, pattern, pattern_type, priority, description))
                
                conn.commit()
                logger.info(f"Pattern added: {question_type} - {pattern}")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Failed to add pattern: {e}")
            return False
    
    def get_patterns_for_type(self, question_type: str) -> List[Dict[str, Any]]:
        """질문 유형별 패턴 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, pattern, pattern_type, priority, description, created_at, updated_at
                    FROM patterns 
                    WHERE question_type = ? AND is_active = 1
                    ORDER BY priority DESC, pattern
                ''', (question_type,))
                
                results = cursor.fetchall()
                return [dict(row) for row in results]
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get patterns for type {question_type}: {e}")
            return []
    
    def register_question_type(self, type_name: str, display_name: str, 
                            description: str = None, parent_type: str = None,
                            priority: int = 1) -> bool:
        """질문 유형 등록"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO question_types 
                    (type_name, display_name, description, parent_type, priority, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (type_name, display_name, description, parent_type, priority))
                
                conn.commit()
                logger.info(f"Question type registered: {type_name}")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Failed to register question type: {e}")
            return False
    
    def get_all_question_types(self) -> List[Dict[str, Any]]:
        """모든 질문 유형 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT type_name, display_name, description, parent_type, priority, 
                           created_at, updated_at
                    FROM question_types 
                    WHERE is_active = 1
                    ORDER BY priority DESC, type_name
                ''')
                
                results = cursor.fetchall()
                return [dict(row) for row in results]
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get question types: {e}")
            return []
    
    def update_keyword_stats(self, keyword_id: int, success: bool = True):
        """키워드 통계 업데이트"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 통계 레코드 확인/생성
                cursor.execute('''
                    INSERT OR IGNORE INTO keyword_stats (keyword_id, match_count, success_count)
                    VALUES (?, 0, 0)
                ''', (keyword_id,))
                
                # 통계 업데이트
                if success:
                    cursor.execute('''
                        UPDATE keyword_stats 
                        SET match_count = match_count + 1, 
                            success_count = success_count + 1,
                            last_matched_at = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE keyword_id = ?
                    ''', (keyword_id,))
                else:
                    cursor.execute('''
                        UPDATE keyword_stats 
                        SET match_count = match_count + 1,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE keyword_id = ?
                    ''', (keyword_id,))
                
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"Failed to update keyword stats: {e}")
    
    def get_keyword_statistics(self, question_type: str = None) -> Dict[str, Any]:
        """키워드 통계 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 기본 통계
                if question_type:
                    cursor.execute('''
                        SELECT COUNT(*) as total_keywords,
                               COUNT(CASE WHEN weight_level = 'high' THEN 1 END) as high_weight_count,
                               COUNT(CASE WHEN weight_level = 'medium' THEN 1 END) as medium_weight_count,
                               COUNT(CASE WHEN weight_level = 'low' THEN 1 END) as low_weight_count
                        FROM keywords 
                        WHERE question_type = ? AND is_active = 1
                    ''', (question_type,))
                else:
                    cursor.execute('''
                        SELECT COUNT(*) as total_keywords,
                               COUNT(CASE WHEN weight_level = 'high' THEN 1 END) as high_weight_count,
                               COUNT(CASE WHEN weight_level = 'medium' THEN 1 END) as medium_weight_count,
                               COUNT(CASE WHEN weight_level = 'low' THEN 1 END) as low_weight_count
                        FROM keywords 
                        WHERE is_active = 1
                    ''')
                
                result = cursor.fetchone()
                stats = {
                    'total_keywords': result[0],
                    'high_weight_count': result[1],
                    'medium_weight_count': result[2],
                    'low_weight_count': result[3]
                }
                
                # 성능 통계
                cursor.execute('''
                    SELECT AVG(ks.success_count * 1.0 / ks.match_count) as avg_success_rate,
                           COUNT(*) as keywords_with_stats
                    FROM keyword_stats ks
                    JOIN keywords k ON ks.keyword_id = k.id
                    WHERE ks.match_count > 0
                ''')
                
                perf_stats = cursor.fetchone()
                if perf_stats:
                    stats.update({
                        'avg_success_rate': perf_stats[0] or 0,
                        'keywords_with_stats': perf_stats[1] or 0
                    })
                
                return stats
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get keyword statistics: {e}")
            return {}
    
    def delete_keyword(self, keyword_id: int) -> bool:
        """키워드 삭제 (소프트 삭제)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE keywords 
                    SET is_active = 0, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (keyword_id,))
                
                conn.commit()
                logger.info(f"Keyword {keyword_id} deactivated")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Failed to delete keyword: {e}")
            return False
    
    def _get_default_weight(self, weight_level: str) -> float:
        """가중치 레벨별 기본값 반환"""
        weights = {
            'high': 3.0,
            'medium': 2.0,
            'low': 1.0
        }
        return weights.get(weight_level, 1.0)
    
    def export_keywords_to_json(self, output_file: str, question_type: str = None):
        """키워드를 JSON 파일로 내보내기"""
        try:
            keywords_data = {}
            
            if question_type:
                keywords = self.get_keywords_for_type(question_type)
                keywords_data[question_type] = keywords
            else:
                question_types = self.get_all_question_types()
                for qt in question_types:
                    keywords = self.get_keywords_for_type(qt['type_name'])
                    keywords_data[qt['type_name']] = keywords
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(keywords_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Keywords exported to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to export keywords: {e}")
    
    def import_keywords_from_json(self, input_file: str) -> int:
        """JSON 파일에서 키워드 가져오기"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                keywords_data = json.load(f)
            
            total_imported = 0
            
            for question_type, keywords in keywords_data.items():
                # 질문 유형 등록
                self.register_question_type(question_type, question_type)
                
                # 키워드 추가
                batch_data = []
                for keyword_info in keywords:
                    if isinstance(keyword_info, dict):
                        batch_data.append({
                            'question_type': question_type,
                            'keyword': keyword_info.get('keyword', ''),
                            'weight_level': keyword_info.get('weight_level', 'medium'),
                            'weight_value': keyword_info.get('weight_value'),
                            'category': keyword_info.get('category'),
                            'description': keyword_info.get('description')
                        })
                    else:
                        # 단순 문자열인 경우
                        batch_data.append({
                            'question_type': question_type,
                            'keyword': str(keyword_info),
                            'weight_level': 'medium'
                        })
                
                imported_count = self.add_keywords_batch(batch_data)
                total_imported += imported_count
            
            logger.info(f"Imported {total_imported} keywords from {input_file}")
            return total_imported
            
        except Exception as e:
            logger.error(f"Failed to import keywords: {e}")
            return 0


# 전역 인스턴스
db_keyword_manager = DatabaseKeywordManager()
