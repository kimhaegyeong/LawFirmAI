#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
판례 데이터 DB 임포터

전처리된 판례 데이터를 SQLite 데이터베이스에 임포트하는 스크립트입니다.
증분 모드를 지원하여 기존 데이터와 중복을 방지합니다.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from core.data.database import DatabaseManager

logger = logging.getLogger(__name__)


class PrecedentDataImporter:
    """판례 데이터 임포터 클래스"""
    
    def __init__(self, db_path: str = "data/lawfirm.db"):
        """
        판례 데이터 임포터 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        self.db_manager = DatabaseManager(db_path)
        self.import_stats = {
            'total_files_processed': 0,
            'total_cases_imported': 0,
            'successful_imports': 0,
            'successful_updates': 0,
            'skipped_imports': 0,
            'failed_imports': 0,
            'import_errors': []
        }
        
        logger.info("PrecedentDataImporter initialized")
    
    def import_file(self, file_path: Path, incremental: bool = False) -> Dict[str, Any]:
        """
        단일 전처리된 판례 파일 임포트
        
        Args:
            file_path: 임포트할 파일 경로
            incremental: 증분 모드 여부
            
        Returns:
            Dict[str, Any]: 임포트 결과
        """
        try:
            logger.info(f"Importing file: {file_path} (incremental: {incremental})")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            
            # 판례 데이터 구조 처리
            if isinstance(file_data, dict) and 'cases' in file_data:
                processed_cases = file_data['cases']
            elif isinstance(file_data, list):
                processed_cases = file_data
            else:
                processed_cases = [file_data]
            
            file_results = {
                'file_name': file_path.name,
                'total_cases': len(processed_cases),
                'imported_cases': 0,
                'updated_cases': 0,
                'failed_cases': 0,
                'skipped_cases': 0,
                'errors': []
            }
            
            for case_data in processed_cases:
                try:
                    if incremental:
                        result = self._import_single_case_incremental(case_data)
                        if result['action'] == 'inserted':
                            file_results['imported_cases'] += 1
                            self.import_stats['successful_imports'] += 1
                        elif result['action'] == 'updated':
                            file_results['updated_cases'] += 1
                            self.import_stats['successful_updates'] += 1
                        elif result['action'] == 'skipped':
                            file_results['skipped_cases'] += 1
                            self.import_stats['skipped_imports'] += 1
                        else:
                            file_results['failed_cases'] += 1
                            self.import_stats['failed_imports'] += 1
                    else:
                        success = self._import_single_case(case_data)
                        if success:
                            file_results['imported_cases'] += 1
                            self.import_stats['successful_imports'] += 1
                        else:
                            file_results['failed_cases'] += 1
                            self.import_stats['failed_imports'] += 1
                        
                except Exception as e:
                    error_msg = f"Error importing case {case_data.get('case_name', 'Unknown')}: {str(e)}"
                    logger.error(error_msg)
                    file_results['errors'].append(error_msg)
                    file_results['failed_cases'] += 1
                    self.import_stats['failed_imports'] += 1
                    self.import_stats['import_errors'].append(error_msg)
            
            self.import_stats['total_files_processed'] += 1
            self.import_stats['total_cases_imported'] += file_results['total_cases']
            
            return file_results
            
        except Exception as e:
            error_msg = f"Error importing file {file_path}: {str(e)}"
            logger.error(error_msg)
            self.import_stats['import_errors'].append(error_msg)
            return {
                'file_name': file_path.name,
                'error': error_msg
            }
    
    def import_directory(self, directory_path: Path, incremental: bool = False) -> Dict[str, Any]:
        """
        디렉토리 내 모든 전처리된 판례 파일 임포트
        
        Args:
            directory_path: 임포트할 디렉토리 경로
            incremental: 증분 모드 여부
            
        Returns:
            Dict[str, Any]: 임포트 결과 요약
        """
        logger.info(f"Importing directory: {directory_path} (incremental: {incremental})")
        
        if not directory_path.exists():
            error_msg = f"Directory does not exist: {directory_path}"
            logger.error(error_msg)
            return {'error': error_msg}
        
        # JSON 파일 찾기
        json_files = list(directory_path.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in directory: {directory_path}")
            return {'message': 'No JSON files found'}
        
        logger.info(f"Found {len(json_files)} JSON files to import")
        
        file_results = []
        for file_path in json_files:
            result = self.import_file(file_path, incremental)
            file_results.append(result)
        
        # 결과 요약 생성
        summary = self._generate_import_summary(file_results)
        
        return {
            'directory_path': str(directory_path),
            'total_files': len(json_files),
            'file_results': file_results,
            'import_summary': summary
        }
    
    def _import_single_case(self, case_data: Dict[str, Any]) -> bool:
        """
        단일 판례 케이스 임포트
        
        Args:
            case_data: 판례 케이스 데이터
            
        Returns:
            bool: 임포트 성공 여부
        """
        try:
            case_id = case_data.get('case_id')
            if not case_id:
                logger.error("Case ID is missing")
                return False
            
            # 케이스 기본 정보 저장
            self._insert_case(case_data)
            
            # 섹션 정보 저장
            sections = case_data.get('sections', [])
            for section in sections:
                self._insert_section(case_id, section)
            
            # 당사자 정보 저장
            parties = case_data.get('parties', [])
            for party in parties:
                self._insert_party(case_id, party)
            
            # FTS 인덱스 업데이트
            self._update_fts_indices(case_id, case_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error importing single case: {e}")
            return False
    
    def _import_single_case_incremental(self, case_data: Dict[str, Any]) -> Dict[str, str]:
        """
        증분 모드로 단일 판례 케이스 임포트
        
        Args:
            case_data: 판례 케이스 데이터
            
        Returns:
            Dict[str, str]: 임포트 결과 (action: inserted/updated/skipped/failed)
        """
        try:
            case_id = case_data.get('case_id')
            if not case_id:
                return {'action': 'failed', 'reason': 'Case ID is missing'}
            
            # 기존 케이스 확인
            existing_case = self._check_existing_case(case_id)
            
            if existing_case:
                # 업데이트 필요성 확인
                if self._check_if_update_needed(existing_case, case_data):
                    self._update_existing_case(case_id, case_data)
                    return {'action': 'updated', 'reason': 'Case updated'}
                else:
                    return {'action': 'skipped', 'reason': 'No changes needed'}
            else:
                # 새 케이스 삽입
                self._insert_case(case_data)
                
                # 섹션 정보 저장
                sections = case_data.get('sections', [])
                for section in sections:
                    self._insert_section(case_id, section)
                
                # 당사자 정보 저장
                parties = case_data.get('parties', [])
                for party in parties:
                    self._insert_party(case_id, party)
                
                # FTS 인덱스 업데이트
                self._update_fts_indices(case_id, case_data)
                
                return {'action': 'inserted', 'reason': 'New case inserted'}
            
        except Exception as e:
            logger.error(f"Error importing single case incrementally: {e}")
            return {'action': 'failed', 'reason': str(e)}
    
    def _insert_case(self, case_data: Dict[str, Any]):
        """케이스 기본 정보 삽입"""
        query = """
            INSERT OR REPLACE INTO precedent_cases 
            (case_id, category, case_name, case_number, decision_date, field, court, 
             detail_url, full_text, searchable_text, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """
        params = (
            case_data.get('case_id'),
            case_data.get('category'),
            case_data.get('case_name'),
            case_data.get('case_number'),
            case_data.get('decision_date'),
            case_data.get('field'),
            case_data.get('court'),
            case_data.get('detail_url'),
            case_data.get('full_text'),
            case_data.get('searchable_text')
        )
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
    
    def _insert_section(self, case_id: str, section_data: Dict[str, Any]):
        """섹션 정보 삽입"""
        section_id = f"{case_id}_{section_data.get('section_type')}"
        
        query = """
            INSERT OR REPLACE INTO precedent_sections 
            (section_id, case_id, section_type, section_type_korean, section_content, 
             section_length, has_content)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            section_id,
            case_id,
            section_data.get('section_type'),
            section_data.get('section_type_korean'),
            section_data.get('section_content'),
            section_data.get('section_length', 0),
            section_data.get('has_content', False)
        )
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
    
    def _insert_party(self, case_id: str, party_data: Dict[str, Any]):
        """당사자 정보 삽입"""
        query = """
            INSERT INTO precedent_parties 
            (case_id, party_type, party_type_korean, party_content, party_length)
            VALUES (?, ?, ?, ?, ?)
        """
        params = (
            case_id,
            party_data.get('party_type'),
            party_data.get('party_type_korean'),
            party_data.get('party_content'),
            party_data.get('party_length', 0)
        )
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
    
    def _check_existing_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """기존 케이스 확인"""
        query = "SELECT * FROM precedent_cases WHERE case_id = ?"
        results = self.db_manager.execute_query(query, (case_id,))
        return results[0] if results else None
    
    def _check_if_update_needed(self, existing_case: Dict[str, Any], new_case_data: Dict[str, Any]) -> bool:
        """업데이트 필요성 확인"""
        # 간단한 해시 비교로 변경 여부 확인
        existing_hash = hash(str(existing_case.get('full_text', '')))
        new_hash = hash(str(new_case_data.get('full_text', '')))
        
        return existing_hash != new_hash
    
    def _update_existing_case(self, case_id: str, case_data: Dict[str, Any]):
        """기존 케이스 업데이트"""
        # 기존 섹션과 당사자 정보 삭제
        self._delete_case_sections(case_id)
        self._delete_case_parties(case_id)
        
        # 케이스 정보 업데이트
        self._insert_case(case_data)
        
        # 섹션과 당사자 정보 재삽입
        sections = case_data.get('sections', [])
        for section in sections:
            self._insert_section(case_id, section)
        
        parties = case_data.get('parties', [])
        for party in parties:
            self._insert_party(case_id, party)
        
        # FTS 인덱스 업데이트
        self._update_fts_indices(case_id, case_data)
    
    def _delete_case_sections(self, case_id: str):
        """케이스의 섹션 정보 삭제"""
        query = "DELETE FROM precedent_sections WHERE case_id = ?"
        self.db_manager.execute_update(query, (case_id,))
    
    def _delete_case_parties(self, case_id: str):
        """케이스의 당사자 정보 삭제"""
        query = "DELETE FROM precedent_parties WHERE case_id = ?"
        self.db_manager.execute_update(query, (case_id,))
    
    def _update_fts_indices(self, case_id: str, case_data: Dict[str, Any]):
        """FTS 인덱스 업데이트"""
        try:
            # FTS 케이스 테이블 업데이트
            fts_query = """
                INSERT OR REPLACE INTO fts_precedent_cases 
                (case_id, case_name, case_number, full_text, searchable_text)
                VALUES (?, ?, ?, ?, ?)
            """
            fts_params = (
                case_id,
                case_data.get('case_name'),
                case_data.get('case_number'),
                case_data.get('full_text'),
                case_data.get('searchable_text')
            )
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(fts_query, fts_params)
                conn.commit()
            
            # FTS 섹션 테이블 업데이트
            sections = case_data.get('sections', [])
            for section in sections:
                section_id = f"{case_id}_{section.get('section_type')}"
                section_fts_query = """
                    INSERT OR REPLACE INTO fts_precedent_sections 
                    (section_id, case_id, section_content)
                    VALUES (?, ?, ?)
                """
                section_fts_params = (
                    section_id,
                    case_id,
                    section.get('section_content')
                )
                
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(section_fts_query, section_fts_params)
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error updating FTS indices: {e}")
    
    def _generate_import_summary(self, file_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """임포트 결과 요약 생성"""
        total_cases = sum(r.get('total_cases', 0) for r in file_results)
        imported_cases = sum(r.get('imported_cases', 0) for r in file_results)
        updated_cases = sum(r.get('updated_cases', 0) for r in file_results)
        skipped_cases = sum(r.get('skipped_cases', 0) for r in file_results)
        failed_cases = sum(r.get('failed_cases', 0) for r in file_results)
        
        return {
            'total_cases': total_cases,
            'imported_cases': imported_cases,
            'updated_cases': updated_cases,
            'skipped_cases': skipped_cases,
            'failed_cases': failed_cases,
            'success_rate': (imported_cases + updated_cases) / total_cases if total_cases > 0 else 0
        }


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="판례 데이터 DB 임포터")
    parser.add_argument('--input', required=True, help='입력 파일 또는 디렉토리 경로')
    parser.add_argument('--db-path', default='data/lawfirm.db', help='데이터베이스 파일 경로')
    parser.add_argument('--incremental', action='store_true', help='증분 모드 활성화')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 로그 출력')
    
    args = parser.parse_args()
    
    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 임포터 초기화
        importer = PrecedentDataImporter(args.db_path)
        
        input_path = Path(args.input)
        
        if input_path.is_file():
            # 단일 파일 임포트
            result = importer.import_file(input_path, args.incremental)
            logger.info(f"File import completed: {result}")
        elif input_path.is_dir():
            # 디렉토리 임포트
            result = importer.import_directory(input_path, args.incremental)
            
            # 결과 출력
            summary = result.get('import_summary', {})
            logger.info("Import Results:")
            logger.info(f"  Total cases: {summary.get('total_cases', 0)}")
            logger.info(f"  Imported cases: {summary.get('imported_cases', 0)}")
            logger.info(f"  Updated cases: {summary.get('updated_cases', 0)}")
            logger.info(f"  Skipped cases: {summary.get('skipped_cases', 0)}")
            logger.info(f"  Failed cases: {summary.get('failed_cases', 0)}")
            logger.info(f"  Success rate: {summary.get('success_rate', 0):.2%}")
        else:
            logger.error(f"Input path does not exist: {input_path}")
            return False
        
        # 전체 통계 출력
        stats = importer.import_stats
        logger.info("Overall Import Statistics:")
        logger.info(f"  Total files processed: {stats['total_files_processed']}")
        logger.info(f"  Total cases imported: {stats['total_cases_imported']}")
        logger.info(f"  Successful imports: {stats['successful_imports']}")
        logger.info(f"  Successful updates: {stats['successful_updates']}")
        logger.info(f"  Skipped imports: {stats['skipped_imports']}")
        logger.info(f"  Failed imports: {stats['failed_imports']}")
        
        if stats['import_errors']:
            logger.error("Import errors:")
            for error in stats['import_errors']:
                logger.error(f"  - {error}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
