#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
처리된 데이터를 데이터베이스에 로드하는 스크립트
"""

import os
import sys
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/load_processed_data.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_processed_laws(db_path: str = "data/lawfirm.db", processed_dir: str = "data/processed/assembly/law_only/20251016"):
    """처리된 법률 데이터를 데이터베이스에 로드"""
    
    logger.info(f"Loading processed laws from: {processed_dir}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    loaded_count = 0
    error_count = 0
    
    try:
        # 처리된 파일들 찾기
        processed_files = list(Path(processed_dir).glob("ml_enhanced_law_only_page_*.json"))
        logger.info(f"Found {len(processed_files)} processed files")
        
        for file_path in processed_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'laws' not in data:
                    continue
                
                for law in data['laws']:
                    try:
                        # assembly_laws 테이블에 삽입
                        insert_law_sql = """
                        INSERT INTO assembly_laws (
                            law_id, source, law_name, law_type, category,
                            promulgation_date, enforcement_date, ministry,
                            full_text, searchable_text, keywords, summary,
                            html_clean_text, content_html, raw_content,
                            processed_at, processing_version, data_quality,
                            created_at, updated_at, ml_enhanced,
                            parsing_quality_score, article_count,
                            control_characters_removed, law_name_hash,
                            content_hash, quality_score, duplicate_group_id,
                            is_primary_version, version_number, parsing_method,
                            auto_corrected, manual_review_required
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """
                        
                        # 데이터 준비
                        law_id = law.get('law_id', '')
                        law_name = law.get('law_name', '')
                        
                        # articles에서 전체 텍스트 생성
                        full_text = ""
                        article_count = 0
                        if 'articles' in law:
                            article_count = len(law['articles'])
                            for article in law['articles']:
                                full_text += f"{article.get('article_title', '')} {article.get('article_content', '')}\n"
                        
                        # 해시 생성
                        law_name_hash = str(hash(law_name)) if law_name else ""
                        content_hash = str(hash(full_text)) if full_text else ""
                        
                        values = (
                            law_id,  # law_id
                            'processed',  # source
                            law_name,  # law_name
                            'law',  # law_type
                            law.get('category', 'law_only'),  # category
                            law.get('promulgation_date'),  # promulgation_date
                            law.get('effective_date'),  # enforcement_date
                            law.get('ministry'),  # ministry
                            full_text,  # full_text
                            full_text,  # searchable_text
                            '',  # keywords
                            '',  # summary
                            full_text,  # html_clean_text
                            '',  # content_html
                            json.dumps(law, ensure_ascii=False),  # raw_content
                            datetime.now().isoformat(),  # processed_at
                            '1.0',  # processing_version
                            'good',  # data_quality
                            datetime.now().isoformat(),  # created_at
                            datetime.now().isoformat(),  # updated_at
                            True,  # ml_enhanced
                            0.8,  # parsing_quality_score
                            article_count,  # article_count
                            True,  # control_characters_removed
                            law_name_hash,  # law_name_hash
                            content_hash,  # content_hash
                            0.8,  # quality_score
                            '',  # duplicate_group_id
                            True,  # is_primary_version
                            1,  # version_number
                            'ml_enhanced',  # parsing_method
                            False,  # auto_corrected
                            False  # manual_review_required
                        )
                        
                        cursor.execute(insert_law_sql, values)
                        loaded_count += 1
                        
                        # assembly_articles 테이블에 조문 삽입
                        if 'articles' in law:
                            for article in law['articles']:
                                insert_article_sql = """
                                INSERT INTO assembly_articles (
                                    law_id, article_number, article_title, article_content,
                                    is_supplementary, ml_confidence_score, parsing_method,
                                    created_at, updated_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """
                                
                                article_values = (
                                    law_id,  # law_id
                                    article.get('article_number', 0),  # article_number
                                    article.get('article_title', ''),  # article_title
                                    article.get('article_content', ''),  # article_content
                                    article.get('is_supplementary', False),  # is_supplementary
                                    article.get('ml_confidence_score', 0.8),  # ml_confidence_score
                                    article.get('parsing_method', 'regex'),  # parsing_method
                                    datetime.now().isoformat(),  # created_at
                                    datetime.now().isoformat()  # updated_at
                                )
                                
                                cursor.execute(insert_article_sql, article_values)
                        
                    except Exception as e:
                        logger.error(f"Error inserting law {law.get('law_id', 'unknown')}: {e}")
                        error_count += 1
                        continue
                
                logger.info(f"Processed file: {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                error_count += 1
                continue
        
        conn.commit()
        logger.info(f"Successfully loaded {loaded_count} laws, {error_count} errors")
        
    except Exception as e:
        logger.error(f"Database error: {e}")
        conn.rollback()
    finally:
        conn.close()


def main():
    """메인 함수"""
    logger.info("Starting processed data loading...")
    
    try:
        load_processed_laws()
        logger.info("Processed data loading completed successfully!")
        
    except Exception as e:
        logger.error(f"Process failed: {e}")


if __name__ == "__main__":
    main()
