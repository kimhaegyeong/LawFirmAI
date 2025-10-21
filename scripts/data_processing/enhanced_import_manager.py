#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
향상된 데이터베이스 적재 시스템
전처리된 데이터를 고품질로 데이터베이스에 적재합니다.
"""

import os
import sys
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_import.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    """임포트 결과 데이터 클래스"""
    imported_laws: int = 0
    imported_articles: int = 0
    imported_cases: int = 0
    imported_sections: int = 0
    quality_improvements: int = 0
    errors: List[str] = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class EnhancedImportManager:
    """향상된 데이터 임포트 매니저"""
    
    def __init__(self, db_path: str = "data/lawfirm.db"):
        self.db_path = db_path
        self.quality_threshold = 0.7  # 품질 임계값
        
        # 통계
        self.stats = {
            'total_files_processed': 0,
            'total_records_imported': 0,
            'quality_improvements': 0,
            'errors': []
        }
    
    @contextmanager
    def get_connection(self):
        """데이터베이스 연결 컨텍스트 매니저"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def import_processed_laws(self, processed_dir: str) -> ImportResult:
        """
        전처리된 법률 데이터 임포트
        
        Args:
            processed_dir (str): 전처리된 데이터 디렉토리
            
        Returns:
            ImportResult: 임포트 결과
        """
        logger.info(f"💾 Importing processed law data from: {processed_dir}")
        start_time = datetime.now()
        
        result = ImportResult()
        processed_path = Path(processed_dir)
        
        if not processed_path.exists():
            error_msg = f"Processed directory not found: {processed_dir}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            return result
        
        # JSON 파일 처리
        json_files = list(processed_path.glob("**/*.json"))
        logger.info(f"Found {len(json_files)} processed files to import")
        
        for file_path in json_files:
            try:
                file_result = self._import_single_law_file(file_path)
                result.imported_laws += file_result['imported_laws']
                result.imported_articles += file_result['imported_articles']
                result.quality_improvements += file_result['quality_improvements']
                result.errors.extend(file_result['errors'])
                
            except Exception as e:
                error_msg = f"Error importing {file_path}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)
        
        # 처리 시간 계산
        end_time = datetime.now()
        result.processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ Law import completed:")
        logger.info(f"  - Imported laws: {result.imported_laws}")
        logger.info(f"  - Imported articles: {result.imported_articles}")
        logger.info(f"  - Quality improvements: {result.quality_improvements}")
        logger.info(f"  - Processing time: {result.processing_time:.2f} seconds")
        
        return result
    
    def _import_single_law_file(self, file_path: Path) -> Dict[str, Any]:
        """단일 법률 파일 임포트"""
        try:
            # 전처리된 데이터 로드
            with open(file_path, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
            
            imported_laws = 0
            imported_articles = 0
            quality_improvements = 0
            errors = []
            
            # 법률 데이터 임포트
            laws = processed_data.get('laws', [])
            for law in laws:
                try:
                    success = self._import_single_law_enhanced(law)
                    if success:
                        imported_laws += 1
                        imported_articles += len(law.get('articles', []))
                        
                        # 품질 개선 확인
                        if law.get('parsing_quality_score', 0) > self.quality_threshold:
                            quality_improvements += 1
                    
                except Exception as e:
                    error_msg = f"Error importing law {law.get('law_name', 'Unknown')}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            return {
                'imported_laws': imported_laws,
                'imported_articles': imported_articles,
                'quality_improvements': quality_improvements,
                'errors': errors
            }
            
        except Exception as e:
            error_msg = f"Error loading file {file_path}: {e}"
            logger.error(error_msg)
            return {
                'imported_laws': 0,
                'imported_articles': 0,
                'quality_improvements': 0,
                'errors': [error_msg]
            }
    
    def _import_single_law_enhanced(self, law_data: Dict[str, Any]) -> bool:
        """향상된 단일 법률 임포트"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 법률 기본 정보 삽입
                law_record = self._prepare_enhanced_law_record(law_data)
                cursor.execute('''
                    INSERT INTO assembly_laws (
                        law_id, source, law_name, law_type, category, row_number,
                        promulgation_number, promulgation_date, enforcement_date, amendment_type,
                        ministry, parent_law, related_laws,
                        full_text, searchable_text, keywords, summary,
                        html_clean_text, content_html,
                        raw_content, detail_url, cont_id, cont_sid, collected_at,
                        processed_at, processing_version, data_quality,
                        ml_enhanced, parsing_quality_score, article_count, supplementary_count,
                        control_characters_removed, law_name_hash, content_hash, quality_score,
                        duplicate_group_id, is_primary_version, version_number,
                        parsing_method, auto_corrected, manual_review_required, migration_timestamp,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', law_record)
                
                # 조문 정보 삽입
                for article in law_data.get('articles', []):
                    article_record = self._prepare_enhanced_article_record(
                        law_data['law_id'], article
                    )
                    cursor.execute('''
                        INSERT INTO assembly_articles (
                            law_id, article_number, article_title, article_content,
                            sub_articles, law_references, word_count, char_count,
                            is_supplementary, ml_confidence_score, parsing_method, article_type,
                            parsing_quality_score, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', article_record)
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error importing law {law_data.get('law_name', 'Unknown')}: {e}")
            return False
    
    def _prepare_enhanced_law_record(self, law_data: Dict[str, Any]) -> Tuple:
        """향상된 법률 레코드 준비"""
        now = datetime.now().isoformat()
        
        return (
            law_data.get('law_id', ''),
            'assembly',
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
            law_data.get('related_laws', ''),
            law_data.get('full_text', ''),
            law_data.get('searchable_text', ''),
            law_data.get('keywords', ''),
            law_data.get('summary', ''),
            law_data.get('html_clean_text', ''),
            law_data.get('content_html', ''),
            law_data.get('raw_content', ''),
            law_data.get('detail_url', ''),
            law_data.get('cont_id', ''),
            law_data.get('cont_sid', ''),
            law_data.get('collected_at', ''),
            now,
            law_data.get('processing_version', '2.0'),
            law_data.get('data_quality', ''),
            law_data.get('ml_enhanced', False),
            law_data.get('parsing_quality_score', 0.0),
            law_data.get('article_count', 0),
            law_data.get('supplementary_count', 0),
            law_data.get('control_characters_removed', False),
            law_data.get('law_name_hash', ''),
            law_data.get('content_hash', ''),
            law_data.get('quality_score', 0.0),
            law_data.get('duplicate_group_id', ''),
            law_data.get('is_primary_version', True),
            law_data.get('version_number', 1),
            law_data.get('parsing_method', 'enhanced'),
            law_data.get('auto_corrected', False),
            law_data.get('manual_review_required', False),
            law_data.get('migration_timestamp', ''),
            now,
            now
        )
    
    def _prepare_enhanced_article_record(self, law_id: str, article_data: Dict[str, Any]) -> Tuple:
        """향상된 조문 레코드 준비"""
        now = datetime.now().isoformat()
        
        return (
            law_id,
            article_data.get('article_number', ''),
            article_data.get('article_title', ''),
            article_data.get('article_content', ''),
            article_data.get('sub_articles', ''),
            article_data.get('law_references', ''),
            article_data.get('word_count', 0),
            article_data.get('char_count', 0),
            article_data.get('is_supplementary', False),
            article_data.get('ml_confidence_score', 0.0),
            article_data.get('parsing_method', 'enhanced'),
            article_data.get('article_type', ''),
            article_data.get('parsing_quality_score', 0.0),
            now,
            now
        )
    
    def import_processed_precedents(self, processed_dir: str) -> ImportResult:
        """
        전처리된 판례 데이터 임포트
        
        Args:
            processed_dir (str): 전처리된 데이터 디렉토리
            
        Returns:
            ImportResult: 임포트 결과
        """
        logger.info(f"💾 Importing processed precedent data from: {processed_dir}")
        start_time = datetime.now()
        
        result = ImportResult()
        processed_path = Path(processed_dir)
        
        if not processed_path.exists():
            error_msg = f"Processed directory not found: {processed_dir}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            return result
        
        # 카테고리별 처리
        categories = ['civil', 'criminal', 'family', 'administrative']
        
        for category in categories:
            category_path = processed_path / category
            if category_path.exists():
                category_result = self._import_precedent_category(category_path, category)
                result.imported_cases += category_result['imported_cases']
                result.errors.extend(category_result['errors'])
        
        # 처리 시간 계산
        end_time = datetime.now()
        result.processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ Precedent import completed:")
        logger.info(f"  - Imported cases: {result.imported_cases}")
        logger.info(f"  - Processing time: {result.processing_time:.2f} seconds")
        
        return result
    
    def _import_precedent_category(self, category_path: Path, category: str) -> Dict[str, Any]:
        """카테고리별 판례 임포트"""
        imported_cases = 0
        errors = []
        
        json_files = list(category_path.glob("**/*.json"))
        logger.info(f"Importing {len(json_files)} files for category: {category}")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)
                
                cases = processed_data.get('cases', [])
                for case in cases:
                    try:
                        success = self._import_single_precedent_case(case)
                        if success:
                            imported_cases += 1
                    except Exception as e:
                        error_msg = f"Error importing case {case.get('case_name', 'Unknown')}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                
            except Exception as e:
                error_msg = f"Error loading precedent file {file_path}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        return {
            'imported_cases': imported_cases,
            'errors': errors
        }
    
    def _import_single_precedent_case(self, case_data: Dict[str, Any]) -> bool:
        """단일 판례 사건 임포트"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 판례 사건 정보 삽입
                case_record = (
                    case_data.get('case_id', ''),
                    case_data.get('field', ''),
                    case_data.get('case_name', ''),
                    case_data.get('case_number', ''),
                    case_data.get('decision_date', ''),
                    case_data.get('court', ''),
                    case_data.get('detail_url', ''),
                    case_data.get('full_text', ''),
                    case_data.get('searchable_text', ''),
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                )
                
                cursor.execute('''
                    INSERT INTO precedent_cases (
                        case_id, category, case_name, case_number, decision_date,
                        field, court, detail_url, full_text, searchable_text,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', case_record)
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error importing precedent case: {e}")
            return False
    
    def create_fts_indices(self) -> bool:
        """FTS 인덱스 생성"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                logger.info("🔍 Creating FTS indices...")
                
                # assembly_laws FTS 인덱스
                try:
                    cursor.execute('''
                        CREATE VIRTUAL TABLE IF NOT EXISTS assembly_laws_fts USING fts5(
                            law_name,
                            full_text,
                            summary,
                            content='assembly_laws',
                            content_rowid='id'
                        )
                    ''')
                    logger.info("Created assembly_laws_fts index")
                except sqlite3.OperationalError as e:
                    logger.warning(f"assembly_laws_fts index already exists or error: {e}")
                
                # assembly_articles FTS 인덱스
                try:
                    cursor.execute('''
                        CREATE VIRTUAL TABLE IF NOT EXISTS assembly_articles_fts USING fts5(
                            article_number,
                            article_title,
                            article_content,
                            content='assembly_articles',
                            content_rowid='id'
                        )
                    ''')
                    logger.info("Created assembly_articles_fts index")
                except sqlite3.OperationalError as e:
                    logger.warning(f"assembly_articles_fts index already exists or error: {e}")
                
                # precedent_cases FTS 인덱스
                try:
                    cursor.execute('''
                        CREATE VIRTUAL TABLE IF NOT EXISTS fts_precedent_cases USING fts5(
                            case_name,
                            case_number,
                            full_text,
                            content='precedent_cases',
                            content_rowid='id'
                        )
                    ''')
                    logger.info("Created fts_precedent_cases index")
                except sqlite3.OperationalError as e:
                    logger.warning(f"fts_precedent_cases index already exists or error: {e}")
                
                conn.commit()
                logger.info("✅ FTS indices created successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error creating FTS indices: {e}")
            return False
    
    def optimize_database(self) -> bool:
        """데이터베이스 최적화"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                logger.info("⚡ Optimizing database...")
                
                # VACUUM 실행
                cursor.execute('VACUUM')
                logger.info("Database vacuumed")
                
                # ANALYZE 실행
                cursor.execute('ANALYZE')
                logger.info("Database analyzed")
                
                # 인덱스 재구성
                cursor.execute('REINDEX')
                logger.info("Indexes rebuilt")
                
                conn.commit()
                logger.info("✅ Database optimization completed")
                return True
                
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            return False


def main():
    """메인 함수"""
    logger.info("🚀 Starting enhanced database import...")
    
    # 임포트 매니저 초기화
    import_manager = EnhancedImportManager()
    
    # 법률 데이터 임포트
    logger.info("\n📋 Phase 1: Importing law data...")
    law_result = import_manager.import_processed_laws("data/processed/assembly/law_only")
    
    # 판례 데이터 임포트
    logger.info("\n📋 Phase 2: Importing precedent data...")
    precedent_result = import_manager.import_processed_precedents("data/processed/assembly/precedent")
    
    # FTS 인덱스 생성
    logger.info("\n📋 Phase 3: Creating FTS indices...")
    fts_success = import_manager.create_fts_indices()
    
    # 데이터베이스 최적화
    logger.info("\n📋 Phase 4: Optimizing database...")
    optimize_success = import_manager.optimize_database()
    
    # 결과 리포트 생성
    total_result = ImportResult(
        imported_laws=law_result.imported_laws,
        imported_articles=law_result.imported_articles,
        imported_cases=precedent_result.imported_cases,
        quality_improvements=law_result.quality_improvements,
        errors=law_result.errors + precedent_result.errors,
        processing_time=law_result.processing_time + precedent_result.processing_time
    )
    
    # 결과 저장
    result_data = {
        'law_import': {
            'imported_laws': law_result.imported_laws,
            'imported_articles': law_result.imported_articles,
            'quality_improvements': law_result.quality_improvements,
            'errors': law_result.errors
        },
        'precedent_import': {
            'imported_cases': precedent_result.imported_cases,
            'errors': precedent_result.errors
        },
        'database_optimization': {
            'fts_indices_created': fts_success,
            'database_optimized': optimize_success
        },
        'total_import': {
            'imported_laws': total_result.imported_laws,
            'imported_articles': total_result.imported_articles,
            'imported_cases': total_result.imported_cases,
            'quality_improvements': total_result.quality_improvements,
            'processing_time': total_result.processing_time,
            'total_errors': len(total_result.errors)
        }
    }
    
    # 리포트 저장
    with open("data/import_report.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n📊 Detailed report saved to: data/import_report.json")
    logger.info("✅ Enhanced database import completed successfully!")
    
    return total_result


if __name__ == "__main__":
    result = main()
    if result.errors:
        print(f"\n⚠️ Import completed with {len(result.errors)} errors")
        print("Check logs for details.")
    else:
        print("\n🎉 Database import completed successfully!")
        print("You can now proceed with quality validation.")
