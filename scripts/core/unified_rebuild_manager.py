#!/usr/bin/env python3
"""
통합 데이터베이스 재구축 매니저
다양한 rebuild 전략을 지원하는 통합 시스템
"""

import sys
import os
from pathlib import Path
import json
import logging
from datetime import datetime
import sqlite3
from typing import List, Dict, Any, Optional, Union
import re
import hashlib
import uuid
import shutil
from enum import Enum
from dataclasses import dataclass

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Raw 데이터 파서를 위한 라이브러리
import html
from bs4 import BeautifulSoup


class RebuildMode(Enum):
    """재구축 모드"""
    FULL = "full"           # 전체 재구축 (조문 처리 포함)
    REAL = "real"           # 실제 데이터 재구축
    SIMPLE = "simple"       # 간단한 재구축
    INCREMENTAL = "incremental"  # 증분 재구축
    QUALITY_FIX = "quality_fix"  # 품질 개선 전용


@dataclass
class RebuildConfig:
    """재구축 설정"""
    mode: RebuildMode
    db_path: str = "data/lawfirm.db"
    raw_data_dir: str = "data/raw"
    backup_enabled: bool = True
    log_level: str = "INFO"
    batch_size: int = 100
    include_articles: bool = True
    quality_check: bool = True
    quality_fix_enabled: bool = True  # 품질 개선 기능 활성화


class UnifiedRebuildManager:
    """통합 데이터베이스 재구축 매니저"""
    
    def __init__(self, config: RebuildConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
        
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(f"rebuild_manager_{self.config.mode.value}")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # 핸들러가 이미 있으면 제거
        if logger.handlers:
            logger.handlers.clear()
            
        # 파일 핸들러
        log_file = f"logs/unified_rebuild_{self.config.mode.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def rebuild_database(self) -> Dict[str, Any]:
        """데이터베이스 재구축 실행"""
        self.start_time = datetime.now()
        self.logger.info(f"Starting database rebuild in {self.config.mode.value} mode")
        
        results = {
            'mode': self.config.mode.value,
            'start_time': self.start_time.isoformat(),
            'phases': {},
            'errors': [],
            'summary': {}
        }
        
        try:
            # Phase 1: 백업 및 정리
            results['phases']['backup'] = self._backup_and_cleanup()
            
            # Phase 2: 데이터베이스 구축 (테이블 생성)
            results['phases']['database_build'] = self._build_database()
            
            # Phase 3: 데이터 처리
            results['phases']['data_processing'] = self._process_data()
            
            # Phase 4: 품질 개선 (활성화된 경우)
            if self.config.quality_fix_enabled and self.config.mode != RebuildMode.QUALITY_FIX:
                results['phases']['quality_improvement'] = self._improve_data_quality()
            
            # Phase 5: 검증
            results['phases']['validation'] = self._validate_results()
            
            # 완료
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            results['end_time'] = end_time.isoformat()
            results['duration_seconds'] = duration
            results['summary'] = {
                'processed_count': self.processed_count,
                'error_count': self.error_count,
                'success_rate': (self.processed_count / (self.processed_count + self.error_count)) * 100 if (self.processed_count + self.error_count) > 0 else 0
            }
            
            self.logger.info(f"Database rebuild completed successfully in {duration:.2f} seconds")
            return results
            
        except Exception as e:
            self.logger.error(f"Database rebuild failed: {e}", exc_info=True)
            results['errors'].append(str(e))
            return results
    
    def _backup_and_cleanup(self) -> Dict[str, Any]:
        """백업 및 정리"""
        self.logger.info("Phase 1: Backup and cleanup")
        
        phase_results = {
            'backup_created': False,
            'database_cleared': False,
            'errors': []
        }
        
        try:
            # 데이터베이스 백업
            if self.config.backup_enabled:
                backup_path = self._create_backup()
                if backup_path:
                    phase_results['backup_created'] = True
                    phase_results['backup_path'] = str(backup_path)
                    self.logger.info(f"Database backed up to: {backup_path}")
            
            # 데이터베이스 정리
            self._clear_database()
            phase_results['database_cleared'] = True
            self.logger.info("Database cleared successfully")
            
        except Exception as e:
            error_msg = f"Backup and cleanup failed: {e}"
            self.logger.error(error_msg)
            phase_results['errors'].append(error_msg)
            
        return phase_results
    
    def _process_data(self) -> Dict[str, Any]:
        """데이터 처리"""
        self.logger.info("Phase 2: Data processing")
        
        phase_results = {
            'processed_files': 0,
            'total_laws': 0,
            'total_articles': 0,
            'errors': []
        }
        
        try:
            raw_path = Path(self.config.raw_data_dir)
            if not raw_path.exists():
                raise FileNotFoundError(f"Raw data directory not found: {self.config.raw_data_dir}")
            
            # 모드별 데이터 처리
            if self.config.mode == RebuildMode.FULL:
                phase_results = self._process_full_mode(raw_path)
            elif self.config.mode == RebuildMode.REAL:
                phase_results = self._process_real_mode(raw_path)
            elif self.config.mode == RebuildMode.SIMPLE:
                phase_results = self._process_simple_mode(raw_path)
            elif self.config.mode == RebuildMode.INCREMENTAL:
                phase_results = self._process_incremental_mode(raw_path)
            
            self.logger.info(f"Data processing completed: {phase_results['total_laws']} laws, {phase_results['total_articles']} articles")
            
        except Exception as e:
            error_msg = f"Data processing failed: {e}"
            self.logger.error(error_msg)
            phase_results['errors'].append(error_msg)
            
        return phase_results
    
    def _process_full_mode(self, raw_path: Path) -> Dict[str, Any]:
        """전체 모드 데이터 처리 (조문 처리 포함)"""
        results = {
            'processed_files': 0,
            'total_laws': 0,
            'total_articles': 0,
            'errors': []
        }
        
        parser = RawDataParser(self.config.include_articles)
        
        for file_path in raw_path.glob("**/*.json"):
            try:
                self.logger.info(f"Processing file: {file_path.name}")
                
                parsed_data = parser.parse_raw_file(file_path)
                laws = parsed_data.get('laws', [])
                articles = parsed_data.get('articles', [])
                
                # 데이터베이스에 저장
                self._save_laws_to_db(laws)
                if self.config.include_articles:
                    self._save_articles_to_db(articles)
                
                results['processed_files'] += 1
                results['total_laws'] += len(laws)
                results['total_articles'] += len(articles)
                
                self.processed_count += len(laws)
                
            except Exception as e:
                error_msg = f"Error processing {file_path.name}: {e}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
                self.error_count += 1
        
        return results
    
    def _process_real_mode(self, raw_path: Path) -> Dict[str, Any]:
        """실제 데이터 모드 처리"""
        results = {
            'processed_files': 0,
            'total_laws': 0,
            'total_articles': 0,
            'errors': []
        }
        
        processor = RealDataProcessor(self.config.db_path)
        
        for file_path in raw_path.glob("**/*.json"):
            try:
                self.logger.info(f"Processing file: {file_path.name}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                laws_data = processor._extract_laws_from_raw_data(raw_data, file_path)
                
                for law_data in laws_data:
                    processor._save_law_to_database(law_data)
                    results['total_laws'] += 1
                    results['total_articles'] += len(law_data.get('articles', []))
                
                results['processed_files'] += 1
                self.processed_count += len(laws_data)
                
            except Exception as e:
                error_msg = f"Error processing {file_path.name}: {e}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
                self.error_count += 1
        
        return results
    
    def _process_simple_mode(self, raw_path: Path) -> Dict[str, Any]:
        """간단한 모드 데이터 처리"""
        results = {
            'processed_files': 0,
            'total_laws': 0,
            'total_articles': 0,
            'errors': []
        }
        
        # 간단한 처리 로직
        for file_path in raw_path.glob("**/*.json"):
            try:
                self.logger.info(f"Processing file: {file_path.name}")
                
                # 파일 크기 확인 (너무 큰 파일은 건너뛰기)
                if file_path.stat().st_size > 50 * 1024 * 1024:  # 50MB 제한
                    self.logger.warning(f"Skipping large file: {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.1f}MB)")
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 기본적인 법률 데이터만 추출
                laws = self._extract_basic_laws(data)
                
                # 배치 처리로 데이터베이스에 저장
                if laws:
                    self._save_laws_to_db(laws)  # 배치 저장 사용
                    results['total_laws'] += len(laws)
                
                results['processed_files'] += 1
                self.processed_count += len(laws)
                
            except KeyboardInterrupt:
                self.logger.warning(f"Processing interrupted by user at file: {file_path.name}")
                raise
            except json.JSONDecodeError as e:
                error_msg = f"JSON decode error in {file_path.name}: {e}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
                self.error_count += 1
            except Exception as e:
                error_msg = f"Error processing {file_path.name}: {e}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
                self.error_count += 1
        
        return results
    
    def _process_incremental_mode(self, raw_path: Path) -> Dict[str, Any]:
        """증분 모드 데이터 처리"""
        results = {
            'processed_files': 0,
            'total_laws': 0,
            'total_articles': 0,
            'errors': []
        }
        
        # 증분 처리 로직 (기존 데이터와 비교하여 변경된 부분만 처리)
        self.logger.info("Incremental mode processing - checking for changes")
        
        # TODO: 증분 처리 로직 구현
        # 현재는 전체 처리와 동일하게 처리
        return self._process_simple_mode(raw_path)
    
    def _build_database(self) -> Dict[str, Any]:
        """데이터베이스 구축"""
        self.logger.info("Phase 3: Database build")
        
        phase_results = {
            'tables_created': False,
            'indexes_created': False,
            'errors': []
        }
        
        try:
            # 테이블 생성
            self._create_tables()
            phase_results['tables_created'] = True
            
            # 인덱스 생성
            self._create_indexes()
            phase_results['indexes_created'] = True
            
            self.logger.info("Database build completed successfully")
            
        except Exception as e:
            error_msg = f"Database build failed: {e}"
            self.logger.error(error_msg)
            phase_results['errors'].append(error_msg)
            
        return phase_results
    
    def _validate_results(self) -> Dict[str, Any]:
        """결과 검증"""
        self.logger.info("Phase 4: Validation")
        
        phase_results = {
            'validation_passed': False,
            'data_quality_score': 0.0,
            'errors': []
        }
        
        try:
            if self.config.quality_check:
                quality_score = self._check_data_quality()
                phase_results['data_quality_score'] = quality_score
                
                if quality_score >= 0.8:
                    phase_results['validation_passed'] = True
                    self.logger.info(f"Validation passed with quality score: {quality_score:.2f}")
                else:
                    self.logger.warning(f"Validation failed with quality score: {quality_score:.2f}")
            else:
                phase_results['validation_passed'] = True
                self.logger.info("Validation skipped (quality check disabled)")
            
        except Exception as e:
            error_msg = f"Validation failed: {e}"
            self.logger.error(error_msg)
            phase_results['errors'].append(error_msg)
            
        return phase_results
    
    def _create_backup(self) -> Optional[Path]:
        """데이터베이스 백업 생성"""
        try:
            db_path = Path(self.config.db_path)
            if not db_path.exists():
                return None
                
            backup_dir = Path("data/backups")
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = backup_dir / f"lawfirm_backup_{timestamp}.db"
            
            shutil.copy2(db_path, backup_path)
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            return None
    
    def _clear_database(self):
        """데이터베이스 정리"""
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                
                # 테이블 삭제
                cursor.execute("DROP TABLE IF EXISTS laws")
                cursor.execute("DROP TABLE IF EXISTS articles")
                cursor.execute("DROP TABLE IF EXISTS precedents")
                cursor.execute("DROP TABLE IF EXISTS legal_terms")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Database clear failed: {e}")
            raise
    
    def _create_tables(self):
        """테이블 생성"""
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.cursor()
            
            # 법률 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS laws (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    law_id TEXT UNIQUE NOT NULL,
                    cont_id TEXT,
                    cont_sid TEXT,
                    law_name TEXT NOT NULL,
                    law_content TEXT,
                    content_html TEXT,
                    clean_text TEXT,
                    law_type TEXT,
                    law_field TEXT,
                    effective_date TEXT,
                    collected_at TEXT,
                    quality_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 조문 테이블
            if self.config.include_articles:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS articles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_id TEXT UNIQUE NOT NULL,
                        law_id TEXT NOT NULL,
                        article_number TEXT,
                        article_title TEXT,
                        article_content TEXT,
                        article_type TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (law_id) REFERENCES laws (law_id)
                    )
                """)
            
            conn.commit()
    
    def _create_indexes(self):
        """인덱스 생성"""
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.cursor()
            
            # 법률 테이블 인덱스
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_laws_law_id ON laws (law_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_laws_law_name ON laws (law_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_laws_law_type ON laws (law_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_laws_law_field ON laws (law_field)")
            
            # 조문 테이블 인덱스
            if self.config.include_articles:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_law_id ON articles (law_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_article_number ON articles (article_number)")
            
            conn.commit()
    
    def _save_laws_to_db(self, laws: List[Dict[str, Any]]):
        """법률 데이터를 데이터베이스에 저장"""
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.cursor()
            
            for law in laws:
                cursor.execute("""
                    INSERT OR REPLACE INTO laws (
                        law_id, cont_id, cont_sid, law_name, law_content,
                        content_html, clean_text, law_type, law_field,
                        effective_date, collected_at, quality_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    law.get('law_id'),
                    law.get('cont_id'),
                    law.get('cont_sid'),
                    law.get('law_name'),
                    law.get('law_content'),
                    law.get('content_html'),
                    law.get('clean_text'),
                    law.get('law_type'),
                    law.get('law_field'),
                    law.get('effective_date'),
                    law.get('collected_at'),
                    law.get('quality_score')
                ))
            
            conn.commit()
    
    def _save_articles_to_db(self, articles: List[Dict[str, Any]]):
        """조문 데이터를 데이터베이스에 저장"""
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.cursor()
            
            for article in articles:
                cursor.execute("""
                    INSERT OR REPLACE INTO articles (
                        article_id, law_id, article_number, article_title,
                        article_content, article_type
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    article.get('article_id'),
                    article.get('law_id'),
                    article.get('article_number'),
                    article.get('article_title'),
                    article.get('article_content'),
                    article.get('article_type')
                ))
            
            conn.commit()
    
    def _save_simple_law_to_db(self, law: Dict[str, Any]):
        """간단한 법률 데이터 저장"""
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO laws (
                    law_id, law_name, law_content, clean_text
                ) VALUES (?, ?, ?, ?)
            """, (
                law.get('law_id'),
                law.get('law_name'),
                law.get('law_content'),
                law.get('clean_text')
            ))
            
            conn.commit()
    
    def _extract_basic_laws(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """기본적인 법률 데이터 추출 (안전한 처리)"""
        laws = []
        
        try:
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and 'items' in data:
                items = data['items']
            else:
                items = [data]
            
            # 아이템 수가 너무 많으면 제한
            if len(items) > 1000:
                self.logger.warning(f"Too many items ({len(items)}), limiting to 1000")
                items = items[:1000]
            
            for item in items:
                if not isinstance(item, dict):
                    continue
                    
                try:
                    law = {
                        'law_id': item.get('cont_id', ''),
                        'law_name': item.get('law_name', ''),
                        'law_content': item.get('law_content', ''),
                        'clean_text': self._extract_text_from_html(item.get('content_html', ''))
                    }
                    
                    # 필수 필드 검증
                    if law['law_id'] and law['law_name']:
                        laws.append(law)
                        
                except Exception as e:
                    self.logger.warning(f"Error processing item: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error in _extract_basic_laws: {e}")
        
        return laws
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """HTML에서 텍스트 추출 (안전한 파싱)"""
        if not html_content or not isinstance(html_content, str):
            return ""
        
        try:
            # HTML 내용이 너무 크면 잘라내기 (메모리 보호)
            if len(html_content) > 1000000:  # 1MB 제한
                html_content = html_content[:1000000]
                self.logger.warning("HTML content truncated due to size limit")
            
            # BeautifulSoup 파싱 시도
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            
            # 결과가 너무 길면 잘라내기
            if len(text) > 50000:  # 50KB 제한
                text = text[:50000]
                self.logger.warning("Extracted text truncated due to length limit")
            
            return text
            
        except KeyboardInterrupt:
            # 사용자가 중단한 경우
            self.logger.warning("HTML parsing interrupted by user")
            raise
        except Exception as e:
            # 기타 파싱 에러의 경우 원본 HTML 반환
            self.logger.warning(f"HTML parsing failed, returning original content: {e}")
            return html_content
    
    def _check_data_quality(self) -> float:
        """데이터 품질 검사"""
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                
                # 전체 레코드 수
                cursor.execute("SELECT COUNT(*) FROM laws")
                total_laws = cursor.fetchone()[0]
                
                # 품질 점수 계산
                cursor.execute("SELECT AVG(quality_score) FROM laws WHERE quality_score IS NOT NULL")
                avg_quality = cursor.fetchone()[0] or 0.0
                
                # 기본적인 품질 검사
                cursor.execute("SELECT COUNT(*) FROM laws WHERE law_name IS NULL OR law_name = ''")
                empty_names = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM laws WHERE law_content IS NULL OR law_content = ''")
                empty_content = cursor.fetchone()[0]
                
                # 품질 점수 계산 (0.0 ~ 1.0)
                quality_score = avg_quality
                if total_laws > 0:
                    quality_score *= (1.0 - (empty_names + empty_content) / (total_laws * 2))
                
                return max(0.0, min(1.0, quality_score))
                
        except Exception as e:
            self.logger.error(f"Quality check failed: {e}")
            return 0.0


class RawDataParser:
    """Raw 데이터 파서 클래스"""
    
    def __init__(self, include_articles: bool = True):
        self.include_articles = include_articles
        self.processed_count = 0
        self.error_count = 0
        
    def parse_raw_file(self, file_path: Path) -> Dict[str, List[Dict[str, Any]]]:
        """Raw JSON 파일을 파싱하여 법률 및 조문 데이터 추출"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            laws = []
            articles = []
            
            # metadata 확인
            metadata = data.get('metadata', {})
            
            # items 배열 처리
            for item in data.get('items', []):
                try:
                    law_data = self._parse_law_item(item, metadata)
                    if law_data:
                        laws.append(law_data)
                        
                        # 조문 추출
                        if self.include_articles:
                            law_articles = self._extract_articles_from_law(item, law_data['law_id'])
                            articles.extend(law_articles)
                        
                except Exception as e:
                    self.error_count += 1
            
            self.processed_count += len(laws)
            
            return {
                'laws': laws,
                'articles': articles
            }
            
        except Exception as e:
            self.error_count += 1
            return {'laws': [], 'articles': []}
    
    def _parse_law_item(self, item: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """개별 법률 항목 파싱"""
        try:
            # 기본 정보 추출
            cont_id = item.get('cont_id', '')
            cont_sid = item.get('cont_sid', '')
            law_name = item.get('law_name', '')
            law_content = item.get('law_content', '')
            content_html = item.get('content_html', '')
            collected_at = item.get('collected_at', '')
            
            # HTML에서 텍스트 추출
            clean_text = self._extract_text_from_html(content_html)
            
            # 법률 정보 파싱
            law_info = self._parse_law_info(law_content, law_name)
            
            # 데이터 품질 평가
            quality_score = self._calculate_quality_score(item, clean_text)
            
            # 데이터베이스에 저장할 형태로 변환
            law_data = {
                'law_id': cont_id,
                'cont_id': cont_id,
                'cont_sid': cont_sid,
                'law_name': law_name,
                'law_content': law_content,
                'content_html': content_html,
                'clean_text': clean_text,
                'law_type': law_info.get('type', ''),
                'law_field': law_info.get('field', ''),
                'effective_date': law_info.get('effective_date', ''),
                'collected_at': collected_at,
                'quality_score': quality_score
            }
            
            return law_data
            
        except Exception as e:
            return None
    
    def _extract_articles_from_law(self, item: Dict[str, Any], law_id: str) -> List[Dict[str, Any]]:
        """법률에서 조문 추출"""
        articles = []
        
        try:
            content_html = item.get('content_html', '')
            if not content_html:
                return articles
            
            soup = BeautifulSoup(content_html, 'html.parser')
            
            # 조문 패턴 찾기
            article_patterns = [
                r'제(\d+)조',
                r'제(\d+)조\s*\([^)]*\)',
                r'제(\d+)조\s*\([^)]*\)\s*\([^)]*\)'
            ]
            
            for pattern in article_patterns:
                matches = re.finditer(pattern, content_html)
                for match in matches:
                    article_number = match.group(1)
                    
                    # 조문 내용 추출
                    article_content = self._extract_article_content(soup, article_number)
                    
                    article = {
                        'article_id': f"{law_id}_article_{article_number}",
                        'law_id': law_id,
                        'article_number': article_number,
                        'article_title': f"제{article_number}조",
                        'article_content': article_content,
                        'article_type': 'article'
                    }
                    
                    articles.append(article)
            
        except Exception as e:
            pass
        
        return articles
    
    def _parse_law_info(self, law_content: str, law_name: str) -> Dict[str, str]:
        """법률 정보 파싱"""
        info = {
            'type': '',
            'field': '',
            'effective_date': ''
        }
        
        # 법률 유형 추출
        if '법' in law_name:
            info['type'] = 'law'
        elif '시행령' in law_name:
            info['type'] = 'enforcement_decree'
        elif '시행규칙' in law_name:
            info['type'] = 'enforcement_rule'
        
        # 법률 분야 추출
        if any(field in law_name for field in ['민법', '상법', '형법']):
            info['field'] = 'civil_criminal'
        elif any(field in law_name for field in ['노동', '근로']):
            info['field'] = 'labor'
        elif any(field in law_name for field in ['부동산', '토지']):
            info['field'] = 'real_estate'
        
        return info
    
    def _calculate_quality_score(self, item: Dict[str, Any], clean_text: str) -> float:
        """데이터 품질 점수 계산"""
        score = 0.0
        
        # 기본 정보 존재 여부
        if item.get('law_name'):
            score += 0.3
        if item.get('law_content'):
            score += 0.3
        if clean_text and len(clean_text) > 100:
            score += 0.2
        
        # HTML 내용 존재 여부
        if item.get('content_html'):
            score += 0.2
        
        return min(1.0, score)
    
    def _extract_article_content(self, soup: BeautifulSoup, article_number: str) -> str:
        """조문 내용 추출"""
        try:
            # 조문 번호로 텍스트 찾기
            text = soup.get_text()
            pattern = rf'제{article_number}조[^제]*'
            match = re.search(pattern, text)
            
            if match:
                return match.group(0).strip()
            
            return f"제{article_number}조"
            
        except Exception:
            return f"제{article_number}조"


    def _improve_data_quality(self) -> Dict[str, Any]:
        """데이터 품질 개선"""
        self.logger.info("Phase 4: Data quality improvement")
        
        phase_results = {
            'quality_improved': False,
            'articles_processed': 0,
            'articles_updated': 0,
            'articles_deleted': 0,
            'errors': []
        }
        
        try:
            if self.config.mode == RebuildMode.QUALITY_FIX:
                # 품질 개선 전용 모드
                phase_results = self._fix_assembly_articles_quality()
            else:
                # 일반 모드에서 품질 개선
                phase_results = self._fix_assembly_articles_quality()
            
            phase_results['quality_improved'] = True
            self.logger.info(f"Quality improvement completed: {phase_results['articles_updated']} updated, {phase_results['articles_deleted']} deleted")
            
        except Exception as e:
            error_msg = f"Quality improvement failed: {e}"
            self.logger.error(error_msg)
            phase_results['errors'].append(error_msg)
            
        return phase_results
    
    def _fix_assembly_articles_quality(self) -> Dict[str, Any]:
        """assembly_articles 품질 개선"""
        results = {
            'articles_processed': 0,
            'articles_updated': 0,
            'articles_deleted': 0,
            'errors': []
        }
        
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                
                # assembly_articles 테이블 존재 확인
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='assembly_articles'")
                if not cursor.fetchone():
                    self.logger.warning("assembly_articles table not found, skipping quality improvement")
                    return results
                
                # 전체 개수 확인
                cursor.execute('SELECT COUNT(*) FROM assembly_articles')
                total_count = cursor.fetchone()[0]
                self.logger.info(f"Total assembly articles to process: {total_count:,}")
                
                if total_count == 0:
                    return results
                
                # 모든 데이터 가져오기
                cursor.execute('SELECT id, article_content FROM assembly_articles')
                all_rows = cursor.fetchall()
                
                processed = 0
                updated = 0
                deleted = 0
                
                batch_size = self.config.batch_size
                batch_updates = []
                batch_deletes = []
                
                for row_id, content in all_rows:
                    processed += 1
                    
                    # 내용 정리
                    cleaned_content = self._clean_article_content(content)
                    
                    if self._is_valid_article_content(cleaned_content):
                        if cleaned_content != content:
                            batch_updates.append((cleaned_content, row_id))
                            updated += 1
                    else:
                        # 유효하지 않은 내용은 삭제
                        batch_deletes.append(row_id)
                        deleted += 1
                    
                    # 배치 처리
                    if len(batch_updates) >= batch_size or len(batch_deletes) >= batch_size:
                        if batch_updates:
                            cursor.executemany(
                                'UPDATE assembly_articles SET article_content = ? WHERE id = ?',
                                batch_updates
                            )
                            batch_updates = []
                        
                        if batch_deletes:
                            cursor.executemany(
                                'DELETE FROM assembly_articles WHERE id = ?',
                                [(row_id,) for row_id in batch_deletes]
                            )
                            batch_deletes = []
                        
                        conn.commit()
                        
                        if processed % 10000 == 0:
                            self.logger.info(f"Processed: {processed:,} / {total_count:,} ({processed/total_count*100:.1f}%)")
                
                # 남은 배치 처리
                if batch_updates:
                    cursor.executemany(
                        'UPDATE assembly_articles SET article_content = ? WHERE id = ?',
                        batch_updates
                    )
                
                if batch_deletes:
                    cursor.executemany(
                        'DELETE FROM assembly_articles WHERE id = ?',
                        [(row_id,) for row_id in batch_deletes]
                    )
                
                conn.commit()
                
                results['articles_processed'] = processed
                results['articles_updated'] = updated
                results['articles_deleted'] = deleted
                
                self.logger.info(f"Assembly articles quality improvement completed:")
                self.logger.info(f"  - Processed: {processed:,}")
                self.logger.info(f"  - Updated: {updated:,}")
                self.logger.info(f"  - Deleted: {deleted:,}")
                
        except Exception as e:
            error_msg = f"Assembly articles quality improvement failed: {e}"
            self.logger.error(error_msg)
            results['errors'].append(error_msg)
            
        return results
    
    def _clean_html_tags(self, content: str) -> str:
        """HTML 태그 제거 및 정리"""
        if not content:
            return content
            
        # HTML 태그 제거
        content = re.sub(r'<[^>]+>', '', content)
        
        # HTML 엔티티 디코딩
        html_entities = {
            '&lt;': '<',
            '&gt;': '>',
            '&amp;': '&',
            '&quot;': '"',
            '&apos;': "'",
            '&nbsp;': ' ',
            '&hellip;': '...',
            '&mdash;': '—',
            '&ndash;': '–'
        }
        
        for entity, char in html_entities.items():
            content = content.replace(entity, char)
            
        return content.strip()
    
    def _clean_article_content(self, content: str) -> str:
        """조문 내용 정리"""
        if not content:
            return content
            
        # HTML 태그 제거
        content = self._clean_html_tags(content)
        
        # 불필요한 공백 정리
        content = re.sub(r'\s+', ' ', content)
        
        # 특수 문자 정리 (한글, 영문, 숫자, 기본 문장부호만 유지)
        content = re.sub(r'[^\w\s가-힣.,;:!?()[\]{}"\'<>/\\-]', '', content)
        
        return content.strip()
    
    def _is_valid_article_content(self, content: str) -> bool:
        """유효한 조문 내용인지 확인"""
        if not content or len(content.strip()) < 10:
            return False
            
        # 너무 짧거나 의미없는 내용 필터링
        invalid_patterns = [
            r'^[,\s]*$',  # 공백이나 쉼표만
            r'^[^\w가-힣]*$',  # 한글/영문이 없는 경우
            r'^[0-9\s,.-]*$',  # 숫자와 기호만
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, content):
                return False
                
        return True


class RealDataProcessor:
    """실제 데이터 처리기 (기존 real_data_rebuild.py에서 가져옴)"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.processed_count = 0
        self.error_count = 0
    
    def _extract_laws_from_raw_data(self, raw_data: Dict[str, Any], file_path: Path) -> List[Dict[str, Any]]:
        """Raw 데이터에서 법률 정보 추출"""
        laws = []
        
        # Raw 데이터 구조 분석
        if isinstance(raw_data, list):
            law_items = raw_data
        elif isinstance(raw_data, dict) and 'laws' in raw_data:
            law_items = raw_data['laws']
        else:
            law_items = [raw_data]
        
        for law_item in law_items:
            try:
                law_data = self._process_law_item(law_item)
                if law_data:
                    laws.append(law_data)
            except Exception as e:
                self.error_count += 1
        
        return laws
    
    def _process_law_item(self, law_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """법률 항목 처리"""
        try:
            return {
                'law_id': law_item.get('cont_id', ''),
                'law_name': law_item.get('law_name', ''),
                'law_content': law_item.get('law_content', ''),
                'articles': []  # 간단한 처리에서는 조문 제외
            }
        except Exception:
            return None
    
    def _save_law_to_database(self, law_data: Dict[str, Any]):
        """법률 데이터를 데이터베이스에 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO laws (
                        law_id, law_name, law_content
                    ) VALUES (?, ?, ?)
                """, (
                    law_data.get('law_id'),
                    law_data.get('law_name'),
                    law_data.get('law_content')
                ))
                
                conn.commit()
                
        except Exception as e:
            self.error_count += 1


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Database Rebuild Manager')
    parser.add_argument('--mode', choices=['full', 'real', 'simple', 'incremental', 'quality_fix'], 
                       default='simple', help='Rebuild mode')
    parser.add_argument('--db-path', default='data/lawfirm.db', help='Database path')
    parser.add_argument('--raw-dir', default='data/raw', help='Raw data directory')
    parser.add_argument('--no-backup', action='store_true', help='Disable backup')
    parser.add_argument('--no-articles', action='store_true', help='Disable article processing')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Log level')
    
    args = parser.parse_args()
    
    # 설정 생성
    config = RebuildConfig(
        mode=RebuildMode(args.mode),
        db_path=args.db_path,
        raw_data_dir=args.raw_dir,
        backup_enabled=not args.no_backup,
        log_level=args.log_level,
        include_articles=not args.no_articles
    )
    
    # 매니저 생성 및 실행
    manager = UnifiedRebuildManager(config)
    results = manager.rebuild_database()
    
    # 결과 출력
    print(f"\n=== Rebuild Results ===")
    print(f"Mode: {results['mode']}")
    print(f"Duration: {results.get('duration_seconds', 0):.2f} seconds")
    print(f"Processed: {results['summary'].get('processed_count', 0)}")
    print(f"Errors: {results['summary'].get('error_count', 0)}")
    print(f"Success Rate: {results['summary'].get('success_rate', 0):.1f}%")
    
    if results['errors']:
        print(f"\nErrors:")
        for error in results['errors']:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
