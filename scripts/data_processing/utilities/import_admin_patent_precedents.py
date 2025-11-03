#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
행정 및 특허 판례 데이터 임포트 스크립트
"""

import os
import sys
import json
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

try:
    from core.data.database import DatabaseManager
except ImportError:
    # 직접 import 시도
    sys.path.append('source')
    from data.database import DatabaseManager

logger = logging.getLogger(__name__)

def import_precedent_data(input_dir: str, category: str, incremental: bool = True):
    """판례 데이터를 데이터베이스에 임포트"""
    
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    db_manager = DatabaseManager()
    
    # 처리된 파일 추적
    processed_files = set()
    if incremental:
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT file_path FROM processed_files WHERE data_type = ?", (f'precedent_{category}',))
                processed_files = {row[0] for row in cursor.fetchall()}
        except Exception as e:
            logger.warning(f"Could not load processed files: {e}")
    
    total_files = 0
    successful_files = 0
    failed_files = 0
    total_cases = 0
    
    logger.info(f"Starting import of {category} precedents from {input_dir}")
    
    # JSON 파일 처리
    for file_path in input_path.glob('*.json'):
        if 'precedent_' not in file_path.name or 'summary' in file_path.name:
            continue
            
        total_files += 1
        
        if incremental and str(file_path) in processed_files:
            logger.info(f"Skipping already processed file: {file_path.name}")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'items' not in data or not data['items']:
                logger.warning(f"No items found in {file_path.name}")
                continue
            
            cases_imported = 0
            for item in data['items']:
                try:
                    # 판례 데이터 추출
                    case_data = {
                        'case_id': item.get('case_number', ''),
                        'case_name': item.get('case_name', ''),
                        'case_number': item.get('case_number', ''),
                        'decision_date': item.get('decision_date', ''),
                        'field': item.get('field', ''),
                        'court': item.get('court', ''),
                        'category': category,
                        'detail_url': item.get('detail_url', ''),
                        'full_text': item.get('precedent_content', ''),
                        'searchable_text': item.get('precedent_content', ''),
                        'collected_at': item.get('collected_at', '')
                    }
                    
                    # 데이터베이스에 삽입
                    with db_manager.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT OR REPLACE INTO precedent_cases 
                            (case_id, case_name, case_number, decision_date, field, court, category, detail_url, full_text, searchable_text, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            case_data['case_id'],
                            case_data['case_name'],
                            case_data['case_number'],
                            case_data['decision_date'],
                            case_data['field'],
                            case_data['court'],
                            case_data['category'],
                            case_data['detail_url'],
                            case_data['full_text'],
                            case_data['searchable_text'],
                            case_data['collected_at']
                        ))
                        
                        case_id = cursor.lastrowid
                        
                        # 구조화된 내용에서 조항 정보 추출
                        if item.get('structured_content'):
                            structured = item['structured_content']
                            if isinstance(structured, dict):
                                # 판례 조항 처리
                                if 'sections' in structured:
                                    for section in structured['sections']:
                                        cursor.execute("""
                                            INSERT OR REPLACE INTO precedent_sections 
                                            (case_id, section_number, section_title, section_content, section_type)
                                            VALUES (?, ?, ?, ?, ?)
                                        """, (
                                            case_id,
                                            section.get('number', ''),
                                            section.get('title', ''),
                                            section.get('content', ''),
                                            section.get('type', '')
                                        ))
                                
                                # 당사자 정보 처리
                                if 'parties' in structured:
                                    for party in structured['parties']:
                                        cursor.execute("""
                                            INSERT OR REPLACE INTO precedent_parties 
                                            (case_id, party_name, party_type, party_role)
                                            VALUES (?, ?, ?, ?)
                                        """, (
                                            case_id,
                                            party.get('name', ''),
                                            party.get('type', ''),
                                            party.get('role', '')
                                        ))
                        
                        conn.commit()
                        cases_imported += 1
                        
                except Exception as e:
                    logger.error(f"Error processing case in {file_path.name}: {e}")
                    continue
            
            # 처리된 파일 기록
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO processed_files 
                    (file_path, data_type, processed_at, processing_status, record_count)
                    VALUES (?, ?, datetime('now'), 'success', ?)
                """, (str(file_path), f'precedent_{category}', cases_imported))
                conn.commit()
            
            successful_files += 1
            total_cases += cases_imported
            logger.info(f"Processed {file_path.name}: {cases_imported} cases")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")
            failed_files += 1
            
            # 실패한 파일 기록
            try:
                with db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO processed_files 
                        (file_path, data_type, processed_at, processing_status, error_message)
                        VALUES (?, ?, datetime('now'), 'failed', ?)
                    """, (str(file_path), f'precedent_{category}', str(e)))
                    conn.commit()
            except:
                pass
    
    logger.info(f"Import completed: {successful_files}/{total_files} files, {total_cases} cases")
    return {
        'total_files': total_files,
        'successful_files': successful_files,
        'failed_files': failed_files,
        'total_cases': total_cases
    }

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='행정 및 특허 판례 데이터 임포트')
    parser.add_argument('--input', required=True, help='입력 디렉토리')
    parser.add_argument('--category', required=True, choices=['administrative', 'patent'], help='카테고리')
    parser.add_argument('--incremental', action='store_true', help='증분 임포트')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    result = import_precedent_data(args.input, args.category, args.incremental)
    
    print(f"\n=== IMPORT SUMMARY ===")
    print(f"Total files: {result['total_files']}")
    print(f"Successful: {result['successful_files']}")
    print(f"Failed: {result['failed_files']}")
    print(f"Total cases: {result['total_cases']}")

if __name__ == "__main__":
    main()
