#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?ë? ?°ì´??DB ?„í¬??

?„ì²˜ë¦¬ëœ ?ë? ?°ì´?°ë? SQLite ?°ì´?°ë² ?´ìŠ¤???„í¬?¸í•˜???¤í¬ë¦½íŠ¸?…ë‹ˆ??
ì¦ë¶„ ëª¨ë“œë¥?ì§€?í•˜??ê¸°ì¡´ ?°ì´?°ì? ì¤‘ë³µ??ë°©ì??©ë‹ˆ??
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from lawfirm_langgraph.core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2 as DatabaseManager

logger = logging.getLogger(__name__)


class PrecedentDataImporter:
    """?ë? ?°ì´???„í¬???´ë˜??""
    
    def __init__(self, db_path: str = "data/lawfirm.db"):
        """
        ?ë? ?°ì´???„í¬??ì´ˆê¸°??
        
        Args:
            db_path: ?°ì´?°ë² ?´ìŠ¤ ?Œì¼ ê²½ë¡œ
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
        ?¨ì¼ ?„ì²˜ë¦¬ëœ ?ë? ?Œì¼ ?„í¬??
        
        Args:
            file_path: ?„í¬?¸í•  ?Œì¼ ê²½ë¡œ
            incremental: ì¦ë¶„ ëª¨ë“œ ?¬ë?
            
        Returns:
            Dict[str, Any]: ?„í¬??ê²°ê³¼
        """
        try:
            logger.info(f"Importing file: {file_path} (incremental: {incremental})")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            
            # ?ë? ?°ì´??êµ¬ì¡° ì²˜ë¦¬
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
        ?”ë ‰? ë¦¬ ??ëª¨ë“  ?„ì²˜ë¦¬ëœ ?ë? ?Œì¼ ?„í¬??
        
        Args:
            directory_path: ?„í¬?¸í•  ?”ë ‰? ë¦¬ ê²½ë¡œ
            incremental: ì¦ë¶„ ëª¨ë“œ ?¬ë?
            
        Returns:
            Dict[str, Any]: ?„í¬??ê²°ê³¼ ?”ì•½
        """
        logger.info(f"Importing directory: {directory_path} (incremental: {incremental})")
        
        if not directory_path.exists():
            error_msg = f"Directory does not exist: {directory_path}"
            logger.error(error_msg)
            return {'error': error_msg}
        
        # JSON ?Œì¼ ì°¾ê¸°
        json_files = list(directory_path.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in directory: {directory_path}")
            return {'message': 'No JSON files found'}
        
        logger.info(f"Found {len(json_files)} JSON files to import")
        
        file_results = []
        for file_path in json_files:
            result = self.import_file(file_path, incremental)
            file_results.append(result)
        
        # ê²°ê³¼ ?”ì•½ ?ì„±
        summary = self._generate_import_summary(file_results)
        
        return {
            'directory_path': str(directory_path),
            'total_files': len(json_files),
            'file_results': file_results,
            'import_summary': summary
        }
    
    def _import_single_case(self, case_data: Dict[str, Any]) -> bool:
        """
        ?¨ì¼ ?ë? ì¼€?´ìŠ¤ ?„í¬??
        
        Args:
            case_data: ?ë? ì¼€?´ìŠ¤ ?°ì´??
            
        Returns:
            bool: ?„í¬???±ê³µ ?¬ë?
        """
        try:
            case_id = case_data.get('case_id')
            if not case_id:
                logger.error("Case ID is missing")
                return False
            
            # ì¼€?´ìŠ¤ ê¸°ë³¸ ?•ë³´ ?€??
            self._insert_case(case_data)
            
            # ?¹ì…˜ ?•ë³´ ?€??
            sections = case_data.get('sections', [])
            for section in sections:
                self._insert_section(case_id, section)
            
            # ?¹ì‚¬???•ë³´ ?€??
            parties = case_data.get('parties', [])
            for party in parties:
                self._insert_party(case_id, party)
            
            # FTS ?¸ë±???…ë°?´íŠ¸
            self._update_fts_indices(case_id, case_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error importing single case: {e}")
            return False
    
    def _import_single_case_incremental(self, case_data: Dict[str, Any]) -> Dict[str, str]:
        """
        ì¦ë¶„ ëª¨ë“œë¡??¨ì¼ ?ë? ì¼€?´ìŠ¤ ?„í¬??
        
        Args:
            case_data: ?ë? ì¼€?´ìŠ¤ ?°ì´??
            
        Returns:
            Dict[str, str]: ?„í¬??ê²°ê³¼ (action: inserted/updated/skipped/failed)
        """
        try:
            case_id = case_data.get('case_id')
            if not case_id:
                return {'action': 'failed', 'reason': 'Case ID is missing'}
            
            # ê¸°ì¡´ ì¼€?´ìŠ¤ ?•ì¸
            existing_case = self._check_existing_case(case_id)
            
            if existing_case:
                # ?…ë°?´íŠ¸ ?„ìš”???•ì¸
                if self._check_if_update_needed(existing_case, case_data):
                    self._update_existing_case(case_id, case_data)
                    return {'action': 'updated', 'reason': 'Case updated'}
                else:
                    return {'action': 'skipped', 'reason': 'No changes needed'}
            else:
                # ??ì¼€?´ìŠ¤ ?½ì…
                self._insert_case(case_data)
                
                # ?¹ì…˜ ?•ë³´ ?€??
                sections = case_data.get('sections', [])
                for section in sections:
                    self._insert_section(case_id, section)
                
                # ?¹ì‚¬???•ë³´ ?€??
                parties = case_data.get('parties', [])
                for party in parties:
                    self._insert_party(case_id, party)
                
                # FTS ?¸ë±???…ë°?´íŠ¸
                self._update_fts_indices(case_id, case_data)
                
                return {'action': 'inserted', 'reason': 'New case inserted'}
            
        except Exception as e:
            logger.error(f"Error importing single case incrementally: {e}")
            return {'action': 'failed', 'reason': str(e)}
    
    def _insert_case(self, case_data: Dict[str, Any]):
        """ì¼€?´ìŠ¤ ê¸°ë³¸ ?•ë³´ ?½ì…"""
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
        """?¹ì…˜ ?•ë³´ ?½ì…"""
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
        """?¹ì‚¬???•ë³´ ?½ì…"""
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
        """ê¸°ì¡´ ì¼€?´ìŠ¤ ?•ì¸"""
        query = "SELECT * FROM precedent_cases WHERE case_id = ?"
        results = self.db_manager.execute_query(query, (case_id,))
        return results[0] if results else None
    
    def _check_if_update_needed(self, existing_case: Dict[str, Any], new_case_data: Dict[str, Any]) -> bool:
        """?…ë°?´íŠ¸ ?„ìš”???•ì¸"""
        # ê°„ë‹¨???´ì‹œ ë¹„êµë¡?ë³€ê²??¬ë? ?•ì¸
        existing_hash = hash(str(existing_case.get('full_text', '')))
        new_hash = hash(str(new_case_data.get('full_text', '')))
        
        return existing_hash != new_hash
    
    def _update_existing_case(self, case_id: str, case_data: Dict[str, Any]):
        """ê¸°ì¡´ ì¼€?´ìŠ¤ ?…ë°?´íŠ¸"""
        # ê¸°ì¡´ ?¹ì…˜ê³??¹ì‚¬???•ë³´ ?? œ
        self._delete_case_sections(case_id)
        self._delete_case_parties(case_id)
        
        # ì¼€?´ìŠ¤ ?•ë³´ ?…ë°?´íŠ¸
        self._insert_case(case_data)
        
        # ?¹ì…˜ê³??¹ì‚¬???•ë³´ ?¬ì‚½??
        sections = case_data.get('sections', [])
        for section in sections:
            self._insert_section(case_id, section)
        
        parties = case_data.get('parties', [])
        for party in parties:
            self._insert_party(case_id, party)
        
        # FTS ?¸ë±???…ë°?´íŠ¸
        self._update_fts_indices(case_id, case_data)
    
    def _delete_case_sections(self, case_id: str):
        """ì¼€?´ìŠ¤???¹ì…˜ ?•ë³´ ?? œ"""
        query = "DELETE FROM precedent_sections WHERE case_id = ?"
        self.db_manager.execute_update(query, (case_id,))
    
    def _delete_case_parties(self, case_id: str):
        """ì¼€?´ìŠ¤???¹ì‚¬???•ë³´ ?? œ"""
        query = "DELETE FROM precedent_parties WHERE case_id = ?"
        self.db_manager.execute_update(query, (case_id,))
    
    def _update_fts_indices(self, case_id: str, case_data: Dict[str, Any]):
        """FTS ?¸ë±???…ë°?´íŠ¸"""
        try:
            # FTS ì¼€?´ìŠ¤ ?Œì´ë¸??…ë°?´íŠ¸
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
            
            # FTS ?¹ì…˜ ?Œì´ë¸??…ë°?´íŠ¸
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
        """?„í¬??ê²°ê³¼ ?”ì•½ ?ì„±"""
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
    """ë©”ì¸ ?¨ìˆ˜"""
    parser = argparse.ArgumentParser(description="?ë? ?°ì´??DB ?„í¬??)
    parser.add_argument('--input', required=True, help='?…ë ¥ ?Œì¼ ?ëŠ” ?”ë ‰? ë¦¬ ê²½ë¡œ')
    parser.add_argument('--db-path', default='data/lawfirm.db', help='?°ì´?°ë² ?´ìŠ¤ ?Œì¼ ê²½ë¡œ')
    parser.add_argument('--incremental', action='store_true', help='ì¦ë¶„ ëª¨ë“œ ?œì„±??)
    parser.add_argument('--verbose', '-v', action='store_true', help='?ì„¸ ë¡œê·¸ ì¶œë ¥')
    
    args = parser.parse_args()
    
    # ë¡œê¹… ?¤ì •
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # ?„í¬??ì´ˆê¸°??
        importer = PrecedentDataImporter(args.db_path)
        
        input_path = Path(args.input)
        
        if input_path.is_file():
            # ?¨ì¼ ?Œì¼ ?„í¬??
            result = importer.import_file(input_path, args.incremental)
            logger.info(f"File import completed: {result}")
        elif input_path.is_dir():
            # ?”ë ‰? ë¦¬ ?„í¬??
            result = importer.import_directory(input_path, args.incremental)
            
            # ê²°ê³¼ ì¶œë ¥
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
        
        # ?„ì²´ ?µê³„ ì¶œë ¥
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
