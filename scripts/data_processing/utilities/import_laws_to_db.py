#!/usr/bin/env python3
"""
Assembly Law Database Import Script

This script imports processed Assembly law data into the database
and creates full-text search indices.

Usage:
  python import_laws_to_db.py --input data/processed/assembly/law
  python import_laws_to_db.py --input data/processed/assembly/law --db-path data/lawfirm.db
  python import_laws_to_db.py --help
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add source module to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from source.data.database import DatabaseManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/database_import.log')
    ]
)
logger = logging.getLogger(__name__)


class AssemblyLawImporter:
    """Database importer for processed Assembly law data"""
    
    def __init__(self, db_path: str = "data/lawfirm.db"):
        """
        Initialize the importer
        
        Args:
            db_path (str): Database file path
        """
        self.db_path = db_path
        self.db_manager = DatabaseManager(db_path)
        
        # Import statistics
        self.import_stats = {
            'total_files_processed': 0,
            'total_laws_imported': 0,
            'successful_imports': 0,
            'successful_updates': 0,
            'skipped_imports': 0,
            'failed_imports': 0,
            'import_errors': [],
            'start_time': None,
            'end_time': None
        }
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables for Assembly law data"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create assembly_laws table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS assembly_laws (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        law_id TEXT UNIQUE NOT NULL,
                        source TEXT NOT NULL DEFAULT 'assembly',
                        law_name TEXT NOT NULL,
                        law_type TEXT,
                        category TEXT,
                        row_number TEXT,
                        
                        -- Promulgation information
                        promulgation_number TEXT,
                        promulgation_date TEXT,
                        enforcement_date TEXT,
                        amendment_type TEXT,
                        
                        -- Extracted metadata
                        ministry TEXT,
                        parent_law TEXT,
                        related_laws TEXT,  -- JSON array
                        
                        -- Content
                        full_text TEXT NOT NULL,
                        summary TEXT,
                        
                        -- HTML content
                        html_clean_text TEXT,
                        content_html TEXT,
                        
                        -- Original data
                        raw_content TEXT,
                        detail_url TEXT,
                        cont_id TEXT,
                        cont_sid TEXT,
                        collected_at TEXT,
                        
                        -- Processing metadata
                        processed_at TEXT NOT NULL,
                        processing_version TEXT,
                        data_quality TEXT,  -- JSON object
                        
                        -- Timestamps
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create assembly_articles table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS assembly_articles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        law_id TEXT NOT NULL,
                        article_number TEXT NOT NULL,
                        article_title TEXT,
                        article_content TEXT NOT NULL,
                        sub_articles TEXT,  -- JSON array
                        law_references TEXT,  -- JSON array
                        word_count INTEGER,
                        char_count INTEGER,
                        
                        -- Timestamps
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        FOREIGN KEY (law_id) REFERENCES assembly_laws (law_id)
                    )
                ''')
                
                # Create full-text search indices
                cursor.execute('''
                    CREATE VIRTUAL TABLE IF NOT EXISTS assembly_laws_fts USING fts5(
                        law_name,
                        full_text,
                        summary,
                        content='assembly_laws',
                        content_rowid='id'
                    )
                ''')
                
                cursor.execute('''
                    CREATE VIRTUAL TABLE IF NOT EXISTS assembly_articles_fts USING fts5(
                        article_number,
                        article_title,
                        article_content,
                        content='assembly_articles',
                        content_rowid='id'
                    )
                ''')
                
                # Create regular indices
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_assembly_laws_law_id ON assembly_laws (law_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_assembly_laws_law_type ON assembly_laws (law_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_assembly_laws_category ON assembly_laws (category)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_assembly_laws_ministry ON assembly_laws (ministry)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_assembly_laws_enforcement_date ON assembly_laws (enforcement_date)')
                
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_assembly_articles_law_id ON assembly_articles (law_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_assembly_articles_article_number ON assembly_articles (article_number)')
                
                conn.commit()
                logger.info("Database tables created successfully")
                
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def import_file(self, file_path: Path, incremental: bool = False) -> Dict[str, Any]:
        """
        Import a single processed law file
        
        Args:
            file_path (Path): Path to processed law file
            incremental (bool): Whether to use incremental mode
            
        Returns:
            Dict[str, Any]: Import results
        """
        try:
            logger.info(f"Importing file: {file_path} (incremental: {incremental})")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            
            # ML 강화 데이터 구조 처리
            if isinstance(file_data, dict) and 'laws' in file_data:
                processed_laws = file_data['laws']
            elif isinstance(file_data, list):
                processed_laws = file_data
            else:
                processed_laws = [file_data]
            
            file_results = {
                'file_name': file_path.name,
                'total_laws': len(processed_laws),
                'imported_laws': 0,
                'updated_laws': 0,
                'failed_laws': 0,
                'skipped_laws': 0,
                'errors': []
            }
            
            for law_data in processed_laws:
                try:
                    if incremental:
                        result = self._import_single_law_incremental(law_data)
                        if result['action'] == 'inserted':
                            file_results['imported_laws'] += 1
                            self.import_stats['successful_imports'] += 1
                        elif result['action'] == 'updated':
                            file_results['updated_laws'] += 1
                            self.import_stats['successful_updates'] += 1
                        elif result['action'] == 'skipped':
                            file_results['skipped_laws'] += 1
                            self.import_stats['skipped_imports'] += 1
                        else:
                            file_results['failed_laws'] += 1
                            self.import_stats['failed_imports'] += 1
                    else:
                        success = self._import_single_law(law_data)
                        if success:
                            file_results['imported_laws'] += 1
                            self.import_stats['successful_imports'] += 1
                        else:
                            file_results['failed_laws'] += 1
                            self.import_stats['failed_imports'] += 1
                        
                except Exception as e:
                    error_msg = f"Error importing law {law_data.get('law_name', 'Unknown')}: {str(e)}"
                    logger.error(error_msg)
                    file_results['errors'].append(error_msg)
                    file_results['failed_laws'] += 1
                    self.import_stats['failed_imports'] += 1
                    self.import_stats['import_errors'].append(error_msg)
            
            self.import_stats['total_files_processed'] += 1
            self.import_stats['total_laws_imported'] += file_results['total_laws']
            
            return file_results
            
        except Exception as e:
            error_msg = f"Error importing file {file_path}: {str(e)}"
            logger.error(error_msg)
            self.import_stats['import_errors'].append(error_msg)
            return {
                'file_name': file_path.name,
                'error': error_msg
            }
    
    def _import_single_law(self, law_data: Dict[str, Any]) -> bool:
        """
        Import a single law into the database
        
        Args:
            law_data (Dict[str, Any]): Processed law data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare law data for insertion
                law_record = self._prepare_law_record(law_data)
                
                # Insert law record
                cursor.execute('''
                    INSERT OR REPLACE INTO assembly_laws (
                        law_id, source, law_name, law_type, category, row_number,
                        promulgation_number, promulgation_date, enforcement_date, amendment_type,
                        ministry, parent_law, related_laws,
                        full_text, searchable_text, keywords, summary,
                        html_clean_text, content_html,
                        raw_content, detail_url, cont_id, cont_sid, collected_at,
                        processed_at, processing_version, data_quality,
                        ml_enhanced, parsing_quality_score, article_count, supplementary_count, control_characters_removed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', law_record)
                
                # Insert articles
                articles = law_data.get('articles', [])
                actual_law_id = law_record[0]  # _prepare_law_record에서 생성된 law_id 사용
                for article in articles:
                    article_record = self._prepare_article_record(actual_law_id, article)
                    cursor.execute('''
                        INSERT OR REPLACE INTO assembly_articles (
                        law_id, article_number, article_title, article_content,
                        sub_articles, law_references, word_count, char_count,
                        is_supplementary, ml_confidence_score, parsing_method, article_type
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', article_record)
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error importing single law: {e}")
            return False
    
    def _generate_law_id(self, law_data: Dict[str, Any]) -> str:
        """Generate law ID from law data"""
        # 기존 law_id가 있으면 사용
        if law_data.get('law_id'):
            return law_data['law_id']
        
        # law_name을 기반으로 ID 생성
        law_name = law_data.get('law_name', 'unknown')
        # 공백을 언더스코어로 변경하고 특수문자 제거
        clean_name = law_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        return f"ml_enhanced_{clean_name}"

    def _import_single_law_incremental(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import a single law in incremental mode (check for existing, update if needed)
        
        Args:
            law_data (Dict[str, Any]): Processed law data
            
        Returns:
            Dict[str, Any]: Import result with action type
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # 법률 ID 확인
                law_id = self._generate_law_id(law_data)
                
                # 기존 법률 존재 여부 확인
                cursor.execute('SELECT law_id FROM assembly_laws WHERE law_id = ?', (law_id,))
                existing_law = cursor.fetchone()
                
                if existing_law:
                    # 기존 법률과 비교하여 업데이트 필요 여부 확인
                    needs_update = self._check_if_update_needed(cursor, law_id, law_data)
                    
                    if needs_update:
                        # 업데이트 수행
                        success = self._update_existing_law(cursor, law_id, law_data)
                        if success:
                            conn.commit()
                            return {'action': 'updated', 'law_id': law_id}
                        else:
                            return {'action': 'failed', 'law_id': law_id, 'error': 'Update failed'}
                    else:
                        # 업데이트 불필요 (스킵)
                        return {'action': 'skipped', 'law_id': law_id, 'reason': 'No changes needed'}
                else:
                    # 새로운 법률 삽입
                    success = self._insert_new_law(cursor, law_data)
                    if success:
                        conn.commit()
                        return {'action': 'inserted', 'law_id': law_id}
                    else:
                        return {'action': 'failed', 'law_id': law_id, 'error': 'Insert failed'}
                
        except Exception as e:
            logger.error(f"Error in incremental import: {e}")
            return {'action': 'failed', 'error': str(e)}
    
    def _extract_full_text(self, law_data: Dict[str, Any]) -> str:
        """Extract full text from law data"""
        # 여러 필드에서 텍스트 추출 시도
        full_text = law_data.get('full_text', '')
        if not full_text:
            full_text = law_data.get('full_content', '')
        if not full_text:
            full_text = law_data.get('cleaned_content', '')
        if not full_text:
            # articles에서 텍스트 조합
            articles = law_data.get('articles', [])
            if articles:
                full_text = '\n'.join([article.get('article_content', '') for article in articles])
        
        return full_text or ''

    def _check_if_update_needed(self, cursor, law_id: str, law_data: Dict[str, Any]) -> bool:
        """
        기존 법률과 새 데이터를 비교하여 업데이트 필요 여부 확인
        
        Args:
            cursor: 데이터베이스 커서
            law_id: 법률 ID
            law_data: 새로운 법률 데이터
        
        Returns:
            bool: 업데이트 필요 여부
        """
        try:
            # 기존 법률 데이터 조회
            cursor.execute('''
                SELECT law_name, full_text, processed_at, processing_version
                FROM assembly_laws 
                WHERE law_id = ?
            ''', (law_id,))
            
            existing = cursor.fetchone()
            if not existing:
                return True  # 존재하지 않으면 삽입 필요
            
            # 주요 필드 비교
            new_law_name = law_data.get('law_name', '')
            new_full_text = self._extract_full_text(law_data)
            
            if (existing[0] != new_law_name or 
                existing[1] != new_full_text):
                return True
            
            # 처리 버전 비교 (새로운 버전이면 업데이트)
            new_version = law_data.get('processing_version', '1.0')
            if existing[3] != new_version:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking update need: {e}")
            return True  # 에러 시 업데이트 수행
    
    def _update_existing_law(self, cursor, law_id: str, law_data: Dict[str, Any]) -> bool:
        """
        기존 법률 업데이트
        
        Args:
            cursor: 데이터베이스 커서
            law_id: 법률 ID
            law_data: 새로운 법률 데이터
        
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            # 법률 레코드 준비
            law_record = self._prepare_law_record(law_data)
            
            # 기존 법률 업데이트
            cursor.execute('''
                UPDATE assembly_laws SET
                    source = ?, law_name = ?, law_type = ?, category = ?, row_number = ?,
                    promulgation_number = ?, promulgation_date = ?, enforcement_date = ?, amendment_type = ?,
                    ministry = ?, parent_law = ?, related_laws = ?,
                    full_text = ?, searchable_text = ?, keywords = ?, summary = ?,
                    html_clean_text = ?, content_html = ?,
                    raw_content = ?, detail_url = ?, cont_id = ?, cont_sid = ?, collected_at = ?,
                    processed_at = ?, processing_version = ?, data_quality = ?,
                    ml_enhanced = ?, parsing_quality_score = ?, article_count = ?, supplementary_count = ?, control_characters_removed = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE law_id = ?
            ''', law_record[1:] + (law_id,))  # law_id 제외하고 업데이트
            
            # 기존 조문 삭제
            cursor.execute('DELETE FROM assembly_articles WHERE law_id = ?', (law_id,))
            
            # 새로운 조문 삽입
            articles = law_data.get('articles', [])
            for article in articles:
                article_record = self._prepare_article_record(law_id, article)
                cursor.execute('''
                    INSERT INTO assembly_articles (
                        law_id, article_number, article_title, article_content,
                        sub_articles, law_references, word_count, char_count,
                        is_supplementary, ml_confidence_score, parsing_method, article_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', article_record)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating existing law: {e}")
            return False
    
    def _insert_new_law(self, cursor, law_data: Dict[str, Any]) -> bool:
        """
        새로운 법률 삽입
        
        Args:
            cursor: 데이터베이스 커서
            law_data: 법률 데이터
        
        Returns:
            bool: 삽입 성공 여부
        """
        try:
            # 법률 레코드 준비
            law_record = self._prepare_law_record(law_data)
            
            # 법률 삽입
            cursor.execute('''
                INSERT INTO assembly_laws (
                    law_id, source, law_name, law_type, category, row_number,
                    promulgation_number, promulgation_date, enforcement_date, amendment_type,
                    ministry, parent_law, related_laws,
                    full_text, searchable_text, keywords, summary,
                    html_clean_text, content_html,
                    raw_content, detail_url, cont_id, cont_sid, collected_at,
                    processed_at, processing_version, data_quality,
                    ml_enhanced, parsing_quality_score, article_count, supplementary_count, control_characters_removed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', law_record)
            
            # 조문 삽입
            articles = law_data.get('articles', [])
            law_id = law_record[0]
            for article in articles:
                article_record = self._prepare_article_record(law_id, article)
                cursor.execute('''
                    INSERT INTO assembly_articles (
                        law_id, article_number, article_title, article_content,
                        sub_articles, law_references, word_count, char_count,
                        is_supplementary, ml_confidence_score, parsing_method, article_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', article_record)
            
            return True
            
        except Exception as e:
            logger.error(f"Error inserting new law: {e}")
            return False
    
    def _prepare_law_record(self, law_data: Dict[str, Any]) -> Tuple:
        """Prepare law record for database insertion"""
        # ML 강화 필드 추출
        ml_enhanced = law_data.get('ml_enhanced', False)
        parsing_quality_score = law_data.get('data_quality', {}).get('parsing_quality_score', 0.0)
        
        # 본칙/부칙 카운트
        articles = law_data.get('articles', [])
        main_articles = [a for a in articles if not a.get('is_supplementary', False)]
        supp_articles = [a for a in articles if a.get('is_supplementary', False)]
        
        return (
            law_data.get('law_id') or f"ml_enhanced_{law_data.get('law_name', 'unknown').replace(' ', '_')}",
            law_data.get('source', 'assembly'),
            law_data.get('law_name', ''),
            law_data.get('law_type', ''),
            law_data.get('category', ''),
            law_data.get('row_number', ''),
            law_data.get('promulgation_number', ''),
            law_data.get('promulgation_date', ''),
            law_data.get('enforcement_date', ''),
            law_data.get('amendment_type', ''),
            law_data.get('ministry', ''),
            law_data.get('parent_law', ''),
            json.dumps(law_data.get('related_laws', []), ensure_ascii=False),
            law_data.get('full_text', ''),
            law_data.get('searchable_text', law_data.get('full_text', '')),
            json.dumps(law_data.get('keywords', []), ensure_ascii=False),
            law_data.get('summary', ''),
            law_data.get('html_clean_text', ''),
            law_data.get('content_html', ''),
            law_data.get('raw_content', ''),
            law_data.get('detail_url', ''),
            law_data.get('cont_id', ''),
            law_data.get('cont_sid', ''),
            law_data.get('collected_at', ''),
            law_data.get('processed_at', ''),
            law_data.get('processing_version', ''),
            json.dumps(law_data.get('data_quality', {}), ensure_ascii=False),
            # ML 강화 필드
            ml_enhanced,
            parsing_quality_score,
            len(main_articles),
            len(supp_articles),
            True  # control_characters_removed
        )
    
    def _prepare_article_record(self, law_id: str, article: Dict[str, Any]) -> Tuple:
        """Prepare article record for database insertion"""
        return (
            law_id,
            article.get('article_number', ''),
            article.get('article_title', ''),
            article.get('article_content', ''),
            json.dumps(article.get('sub_articles', []), ensure_ascii=False),
            json.dumps(article.get('references', []), ensure_ascii=False),
            article.get('word_count', 0),
            article.get('char_count', 0),
            # ML 강화 필드
            article.get('is_supplementary', False),
            article.get('ml_confidence_score'),
            article.get('parsing_method', 'rule_based'),
            article.get('article_type', 'main' if not article.get('is_supplementary', False) else 'supplementary')
        )
    
    def import_directory(self, input_dir: Path, incremental: bool = False) -> Dict[str, Any]:
        """
        Import all processed law files in a directory
        
        Args:
            input_dir (Path): Input directory path
            
        Returns:
            Dict[str, Any]: Import results
        """
        self.import_stats['start_time'] = datetime.now()
        
        logger.info(f"Starting import of directory: {input_dir}")
        
        # Find all JSON files
        json_files = list(input_dir.glob('*.json'))
        logger.info(f"Found {len(json_files)} JSON files to import")
        
        file_results = []
        
        for json_file in json_files:
            if json_file.name in ['preprocessing_summary.json', 'validation_report.json']:
                continue  # Skip summary files
            
            try:
                file_result = self.import_file(json_file, incremental=incremental)
                file_results.append(file_result)
            except Exception as e:
                error_msg = f"Error importing file {json_file}: {str(e)}"
                logger.error(error_msg)
                file_results.append({
                    'file_name': json_file.name,
                    'error': error_msg
                })
        
        self.import_stats['end_time'] = datetime.now()
        
        # Update FTS indices
        self._update_fts_indices()
        
        # Generate import summary
        summary = self._generate_import_summary(file_results)
        
        return summary
    
    def _update_fts_indices(self):
        """Update full-text search indices"""
        try:
            logger.info("Updating full-text search indices...")
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Update FTS indices
                cursor.execute('INSERT INTO assembly_laws_fts(assembly_laws_fts) VALUES("rebuild")')
                cursor.execute('INSERT INTO assembly_articles_fts(assembly_articles_fts) VALUES("rebuild")')
                
                conn.commit()
                logger.info("FTS indices updated successfully")
                
        except Exception as e:
            logger.error(f"Error updating FTS indices: {e}")
    
    def _generate_import_summary(self, file_results: List[Dict]) -> Dict[str, Any]:
        """Generate import summary"""
        total_files = len(file_results)
        successful_files = sum(1 for fr in file_results if 'error' not in fr)
        failed_files = total_files - successful_files
        
        total_laws = sum(fr.get('total_laws', 0) for fr in file_results if 'error' not in fr)
        imported_laws = sum(fr.get('imported_laws', 0) for fr in file_results if 'error' not in fr)
        updated_laws = sum(fr.get('updated_laws', 0) for fr in file_results if 'error' not in fr)
        skipped_laws = sum(fr.get('skipped_laws', 0) for fr in file_results if 'error' not in fr)
        failed_laws = sum(fr.get('failed_laws', 0) for fr in file_results if 'error' not in fr)
        
        processing_time = None
        if self.import_stats['start_time'] and self.import_stats['end_time']:
            processing_time = (self.import_stats['end_time'] - self.import_stats['start_time']).total_seconds()
        
        return {
            'import_summary': {
                'total_files': total_files,
                'successful_files': successful_files,
                'failed_files': failed_files,
                'total_laws': total_laws,
                'imported_laws': imported_laws,
                'updated_laws': updated_laws,
                'skipped_laws': skipped_laws,
                'failed_laws': failed_laws,
                'import_time_seconds': processing_time,
                'import_date': datetime.now().isoformat(),
                'database_path': self.db_path
            },
            'file_results': file_results,
            'errors': self.import_stats['import_errors']
        }
    
    def generate_statistics_report(self) -> Dict[str, Any]:
        """Generate database statistics report"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get law statistics
                cursor.execute('SELECT COUNT(*) FROM assembly_laws')
                total_laws = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(DISTINCT law_type) FROM assembly_laws WHERE law_type IS NOT NULL')
                law_types = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(DISTINCT ministry) FROM assembly_laws WHERE ministry IS NOT NULL')
                ministries = cursor.fetchone()[0]
                
                # Get article statistics
                cursor.execute('SELECT COUNT(*) FROM assembly_articles')
                total_articles = cursor.fetchone()[0]
                
                # Get FTS statistics
                cursor.execute('SELECT COUNT(*) FROM assembly_laws_fts')
                fts_laws = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM assembly_articles_fts')
                fts_articles = cursor.fetchone()[0]
                
                return {
                    'database_statistics': {
                        'total_laws': total_laws,
                        'law_types': law_types,
                        'ministries': ministries,
                        'total_articles': total_articles,
                        'fts_laws': fts_laws,
                        'fts_articles': fts_articles,
                        'report_date': datetime.now().isoformat()
                    }
                }
                
        except Exception as e:
            logger.error(f"Error generating statistics report: {e}")
            return {}


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Import processed Assembly law data to database')
    parser.add_argument('--input', type=str, required=True, help='Input directory path')
    parser.add_argument('--db-path', type=str, default='data/lawfirm.db', help='Database file path')
    parser.add_argument('--output', type=str, help='Output file path for import report')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--replace-existing', action='store_true',
                       help='Replace existing data instead of updating')
    parser.add_argument('--incremental', action='store_true',
                       help='Use incremental import mode (check for existing, update if needed)')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Convert to Path objects
    input_dir = Path(args.input)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Create importer and run
    importer = AssemblyLawImporter(args.db_path)
    
    # Clear existing data if replace-existing is specified
    if args.replace_existing:
        logger.info("Clearing existing data...")
        with importer.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM assembly_articles')
            cursor.execute('DELETE FROM assembly_laws')
            cursor.execute('DELETE FROM assembly_laws_fts')
            cursor.execute('DELETE FROM assembly_articles_fts')
            conn.commit()
        logger.info("Existing data cleared")
    
    try:
        import_results = importer.import_directory(input_dir, incremental=args.incremental)
        
        # Generate statistics report
        stats_report = importer.generate_statistics_report()
        
        # Combine results
        final_report = {
            **import_results,
            **stats_report
        }
        
        # Save import report
        if args.output:
            output_file = Path(args.output)
        else:
            output_file = input_dir / 'import_report.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Import completed. Report saved to {output_file}")
        
        # Print summary
        summary = import_results['import_summary']
        print("\n" + "="*50)
        print("DATABASE IMPORT SUMMARY")
        print("="*50)
        print(f"Total files processed: {summary['total_files']}")
        print(f"Successful files: {summary['successful_files']}")
        print(f"Failed files: {summary['failed_files']}")
        print(f"Total laws: {summary['total_laws']}")
        print(f"Imported laws: {summary['imported_laws']}")
        if args.incremental:
            print(f"Updated laws: {summary.get('updated_laws', 0)}")
            print(f"Skipped laws: {summary.get('skipped_laws', 0)}")
        print(f"Failed laws: {summary['failed_laws']}")
        if summary['import_time_seconds']:
            print(f"Import time: {summary['import_time_seconds']:.2f} seconds")
        print(f"Database path: {summary['database_path']}")
        print("="*50)
        
        # Print statistics
        if 'database_statistics' in stats_report:
            stats = stats_report['database_statistics']
            print("\nDATABASE STATISTICS")
            print("="*30)
            print(f"Total laws in database: {stats['total_laws']}")
            print(f"Law types: {stats['law_types']}")
            print(f"Ministries: {stats['ministries']}")
            print(f"Total articles: {stats['total_articles']}")
            print(f"FTS laws: {stats['fts_laws']}")
            print(f"FTS articles: {stats['fts_articles']}")
            print("="*30)
        
    except Exception as e:
        logger.error(f"Import failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
