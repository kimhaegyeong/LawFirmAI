#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
데이터베이스 기반 템플릿 관리 시스템
답변 구조 템플릿을 데이터베이스에서 동적으로 관리
"""

import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class TemplateDatabaseManager:
    """템플릿 데이터베이스 관리자"""
    
    def __init__(self, db_path: str = "data/templates.db"):
        self.db_path = db_path
        self._ensure_db_directory()
        self._init_database()
        logger.info(f"TemplateDatabaseManager initialized with DB: {db_path}")
    
    def _ensure_db_directory(self):
        """데이터베이스 디렉토리 생성"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """데이터베이스 초기화 및 테이블 생성"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 답변 템플릿 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS answer_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_type TEXT NOT NULL,
                    template_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    priority INTEGER DEFAULT 1,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(question_type, template_name)
                )
            ''')
            
            # 템플릿 섹션 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS template_sections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    template_id INTEGER NOT NULL,
                    section_name TEXT NOT NULL,
                    priority TEXT NOT NULL CHECK(priority IN ('high', 'medium', 'low')),
                    template_text TEXT NOT NULL,
                    content_guide TEXT,
                    legal_citations BOOLEAN DEFAULT 0,
                    section_order INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (template_id) REFERENCES answer_templates (id)
                )
            ''')
            
            # 품질 지표 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quality_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    indicator_type TEXT NOT NULL,
                    keyword TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    description TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(indicator_type, keyword)
                )
            ''')
            
            # 질문 유형 설정 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS question_type_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_type TEXT UNIQUE NOT NULL,
                    display_name TEXT NOT NULL,
                    law_names TEXT,  -- JSON 배열로 저장
                    question_words TEXT,  -- JSON 배열로 저장
                    special_keywords TEXT,  -- JSON 배열로 저장
                    bonus_score REAL DEFAULT 0.0,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 충돌 해결 규칙 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conflict_resolution_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conflict_type TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    keywords TEXT NOT NULL,  -- JSON 배열로 저장
                    bonus_score REAL DEFAULT 2.0,
                    priority INTEGER DEFAULT 1,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 인덱스 생성
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_templates_type ON answer_templates(question_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sections_template ON template_sections(template_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sections_priority ON template_sections(priority)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_quality_type ON quality_indicators(indicator_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_conflict_type ON conflict_resolution_rules(conflict_type)')
            
            conn.commit()
            logger.info("Template database tables and indexes created successfully")
    
    def add_template(self, question_type: str, template_name: str, title: str, 
                    description: str = None, priority: int = 1) -> int:
        """템플릿 추가"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO answer_templates 
                    (question_type, template_name, title, description, priority, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (question_type, template_name, title, description, priority))
                
                template_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Template added: {question_type} - {template_name}")
                return template_id
                
        except sqlite3.Error as e:
            logger.error(f"Failed to add template: {e}")
            return -1
    
    def add_template_section(self, template_id: int, section_name: str, priority: str,
                           template_text: str, content_guide: str = None,
                           legal_citations: bool = False, section_order: int = 0) -> bool:
        """템플릿 섹션 추가"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO template_sections 
                    (template_id, section_name, priority, template_text, content_guide, 
                     legal_citations, section_order, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (template_id, section_name, priority, template_text, content_guide,
                      legal_citations, section_order))
                
                conn.commit()
                logger.info(f"Template section added: {section_name}")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Failed to add template section: {e}")
            return False
    
    def get_template(self, question_type: str) -> Optional[Dict[str, Any]]:
        """질문 유형별 템플릿 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # 템플릿 기본 정보 조회
                cursor.execute('''
                    SELECT id, template_name, title, description, priority
                    FROM answer_templates 
                    WHERE question_type = ? AND is_active = 1
                    ORDER BY priority DESC, id
                    LIMIT 1
                ''', (question_type,))
                
                template_row = cursor.fetchone()
                if not template_row:
                    return None
                
                template = dict(template_row)
                
                # 섹션 정보 조회
                cursor.execute('''
                    SELECT section_name, priority, template_text, content_guide, 
                           legal_citations, section_order
                    FROM template_sections 
                    WHERE template_id = ? AND is_active = 1
                    ORDER BY section_order, priority DESC
                ''', (template['id'],))
                
                sections = []
                for section_row in cursor.fetchall():
                    section = dict(section_row)
                    sections.append({
                        "name": section['section_name'],
                        "priority": section['priority'],
                        "template": section['template_text'],
                        "content_guide": section['content_guide'],
                        "legal_citations": bool(section['legal_citations'])
                    })
                
                template['sections'] = sections
                return template
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get template for {question_type}: {e}")
            return None
    
    def get_all_templates(self) -> Dict[str, Dict[str, Any]]:
        """모든 템플릿 조회"""
        try:
            templates = {}
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # 모든 템플릿 조회
                cursor.execute('''
                    SELECT question_type, template_name, title, description, priority
                    FROM answer_templates 
                    WHERE is_active = 1
                    ORDER BY question_type, priority DESC
                ''')
                
                for template_row in cursor.fetchall():
                    template = dict(template_row)
                    question_type = template['question_type']
                    
                    if question_type not in templates:
                        templates[question_type] = {
                            "title": template['title'],
                            "sections": []
                        }
                
                # 각 템플릿의 섹션 조회
                for question_type in templates.keys():
                    template_id = self._get_template_id(question_type)
                    if template_id:
                        sections = self._get_template_sections(template_id)
                        templates[question_type]['sections'] = sections
            
            return templates
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get all templates: {e}")
            return {}
    
    def add_quality_indicator(self, indicator_type: str, keyword: str, 
                            weight: float = 1.0, description: str = None) -> bool:
        """품질 지표 추가"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO quality_indicators 
                    (indicator_type, keyword, weight, description, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (indicator_type, keyword, weight, description))
                
                conn.commit()
                logger.info(f"Quality indicator added: {indicator_type} - {keyword}")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Failed to add quality indicator: {e}")
            return False
    
    def get_quality_indicators(self) -> Dict[str, List[str]]:
        """품질 지표 조회"""
        try:
            indicators = {}
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT indicator_type, keyword, weight
                    FROM quality_indicators 
                    WHERE is_active = 1
                    ORDER BY indicator_type, weight DESC
                ''')
                
                for row in cursor.fetchall():
                    indicator_type = row['indicator_type']
                    keyword = row['keyword']
                    
                    if indicator_type not in indicators:
                        indicators[indicator_type] = []
                    indicators[indicator_type].append(keyword)
            
            return indicators
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get quality indicators: {e}")
            return {}
    
    def add_question_type_config(self, question_type: str, display_name: str,
                               law_names: List[str] = None, question_words: List[str] = None,
                               special_keywords: List[str] = None, bonus_score: float = 0.0) -> bool:
        """질문 유형 설정 추가"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO question_type_configs 
                    (question_type, display_name, law_names, question_words, 
                     special_keywords, bonus_score, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (question_type, display_name, 
                      json.dumps(law_names) if law_names else None,
                      json.dumps(question_words) if question_words else None,
                      json.dumps(special_keywords) if special_keywords else None,
                      bonus_score))
                
                conn.commit()
                logger.info(f"Question type config added: {question_type}")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Failed to add question type config: {e}")
            return False
    
    def get_question_type_config(self, question_type: str) -> Optional[Dict[str, Any]]:
        """질문 유형 설정 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT question_type, display_name, law_names, question_words, 
                           special_keywords, bonus_score
                    FROM question_type_configs 
                    WHERE question_type = ? AND is_active = 1
                ''', (question_type,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                config = dict(row)
                
                # JSON 배열 파싱
                config['law_names'] = json.loads(config['law_names']) if config['law_names'] else []
                config['question_words'] = json.loads(config['question_words']) if config['question_words'] else []
                config['special_keywords'] = json.loads(config['special_keywords']) if config['special_keywords'] else []
                
                return config
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get question type config: {e}")
            return None
    
    def add_conflict_resolution_rule(self, conflict_type: str, target_type: str,
                                   keywords: List[str], bonus_score: float = 2.0,
                                   priority: int = 1) -> bool:
        """충돌 해결 규칙 추가"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO conflict_resolution_rules 
                    (conflict_type, target_type, keywords, bonus_score, priority, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (conflict_type, target_type, json.dumps(keywords), bonus_score, priority))
                
                conn.commit()
                logger.info(f"Conflict resolution rule added: {conflict_type}")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Failed to add conflict resolution rule: {e}")
            return False
    
    def get_conflict_resolution_rules(self) -> Dict[str, Dict[str, Any]]:
        """충돌 해결 규칙 조회"""
        try:
            rules = {}
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT conflict_type, target_type, keywords, bonus_score, priority
                    FROM conflict_resolution_rules 
                    WHERE is_active = 1
                    ORDER BY priority DESC, conflict_type
                ''')
                
                for row in cursor.fetchall():
                    conflict_type = row['conflict_type']
                    rules[conflict_type] = {
                        'target_type': row['target_type'],
                        'keywords': json.loads(row['keywords']),
                        'bonus_score': row['bonus_score'],
                        'priority': row['priority']
                    }
            
            return rules
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get conflict resolution rules: {e}")
            return {}
    
    def _get_template_id(self, question_type: str) -> Optional[int]:
        """템플릿 ID 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id FROM answer_templates 
                    WHERE question_type = ? AND is_active = 1
                    ORDER BY priority DESC, id LIMIT 1
                ''', (question_type,))
                
                result = cursor.fetchone()
                return result[0] if result else None
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get template ID: {e}")
            return None
    
    def _get_template_sections(self, template_id: int) -> List[Dict[str, Any]]:
        """템플릿 섹션 조회"""
        try:
            sections = []
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT section_name, priority, template_text, content_guide, 
                           legal_citations, section_order
                    FROM template_sections 
                    WHERE template_id = ? AND is_active = 1
                    ORDER BY section_order, priority DESC
                ''', (template_id,))
                
                for row in cursor.fetchall():
                    sections.append({
                        "name": row['section_name'],
                        "priority": row['priority'],
                        "template": row['template_text'],
                        "content_guide": row['content_guide'],
                        "legal_citations": bool(row['legal_citations'])
                    })
            
            return sections
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get template sections: {e}")
            return []
    
    def update_template(self, template_id: int, **kwargs) -> bool:
        """템플릿 업데이트"""
        try:
            if not kwargs:
                return False
            
            set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
            values = list(kwargs.values()) + [template_id]
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    UPDATE answer_templates 
                    SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', values)
                
                conn.commit()
                logger.info(f"Template {template_id} updated")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Failed to update template: {e}")
            return False
    
    def delete_template(self, template_id: int) -> bool:
        """템플릿 삭제 (소프트 삭제)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE answer_templates 
                    SET is_active = 0, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (template_id,))
                
                conn.commit()
                logger.info(f"Template {template_id} deactivated")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Failed to delete template: {e}")
            return False
    
    def get_template_statistics(self) -> Dict[str, Any]:
        """템플릿 통계 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 기본 통계
                cursor.execute('''
                    SELECT COUNT(*) as total_templates,
                           COUNT(CASE WHEN is_active = 1 THEN 1 END) as active_templates
                    FROM answer_templates
                ''')
                
                result = cursor.fetchone()
                stats = {
                    'total_templates': result[0],
                    'active_templates': result[1]
                }
                
                # 섹션 통계
                cursor.execute('''
                    SELECT COUNT(*) as total_sections,
                           COUNT(CASE WHEN is_active = 1 THEN 1 END) as active_sections
                    FROM template_sections
                ''')
                
                section_result = cursor.fetchone()
                stats.update({
                    'total_sections': section_result[0],
                    'active_sections': section_result[1]
                })
                
                # 질문 유형별 통계
                cursor.execute('''
                    SELECT question_type, COUNT(*) as template_count
                    FROM answer_templates 
                    WHERE is_active = 1
                    GROUP BY question_type
                ''')
                
                type_stats = {}
                for row in cursor.fetchall():
                    type_stats[row[0]] = row[1]
                
                stats['by_question_type'] = type_stats
                
                return stats
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get template statistics: {e}")
            return {}


# 전역 인스턴스
template_db_manager = TemplateDatabaseManager()
