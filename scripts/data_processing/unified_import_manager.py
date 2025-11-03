#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Import Manager

ëª¨ë“  ?°ì´???„í¬???‘ì—…???µí•© ê´€ë¦¬í•˜??ë§¤ë‹ˆ?€ ?´ë˜?¤ì…?ˆë‹¤.
- ë²•ë ¹ ?°ì´???„í¬??
- ?ë? ?°ì´???„í¬??
- ?°ì´??ê²€ì¦?
- ì¤‘ë³µ ì²˜ë¦¬
- ?¤ë¥˜ ë³µêµ¬
"""

import os
import sys
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import argparse
from dataclasses import dataclass, field

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.database import DatabaseManager


@dataclass
class ImportTask:
    """?„í¬???‘ì—… ?°ì´???´ë˜??""
    task_id: str
    data_type: str  # 'law' or 'precedent'
    input_path: Path
    category: Optional[str] = None
    priority: int = 1  # 1=high, 2=medium, 3=low
    status: str = 'pending'  # pending, processing, completed, failed
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    records_imported: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class UnifiedImportManager:
    """?µí•© ?„í¬??ë§¤ë‹ˆ?€"""
    
    def __init__(self, db_path: str = "data/lawfirm.db"):
        """
        ?„í¬??ë§¤ë‹ˆ?€ ì´ˆê¸°??
        
        Args:
            db_path: ?°ì´?°ë² ?´ìŠ¤ ê²½ë¡œ
        """
        self.db_manager = DatabaseManager(db_path)
        
        # ë¡œê¹… ?¤ì •
        self.logger = logging.getLogger(__name__)
        
        # ?‘ì—… ??
        self.task_queue: List[ImportTask] = []
        self.completed_tasks: List[ImportTask] = []
        self.failed_tasks: List[ImportTask] = []
    
    def add_law_import_task(self, 
                           input_path: Union[str, Path],
                           priority: int = 1) -> str:
        """ë²•ë ¹ ?„í¬???‘ì—… ì¶”ê?"""
        task_id = f"law_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.task_queue)}"
        
        task = ImportTask(
            task_id=task_id,
            data_type='law',
            input_path=Path(input_path),
            priority=priority
        )
        
        self.task_queue.append(task)
        self.logger.info(f"Added law import task: {task_id}")
        return task_id
    
    def add_precedent_import_task(self, 
                                 input_path: Union[str, Path],
                                 category: str,
                                 priority: int = 1) -> str:
        """?ë? ?„í¬???‘ì—… ì¶”ê?"""
        task_id = f"precedent_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.task_queue)}"
        
        task = ImportTask(
            task_id=task_id,
            data_type='precedent',
            input_path=Path(input_path),
            category=category,
            priority=priority
        )
        
        self.task_queue.append(task)
        self.logger.info(f"Added precedent import task: {task_id}")
        return task_id
    
    def process_next_task(self) -> bool:
        """?¤ìŒ ?‘ì—… ì²˜ë¦¬"""
        if not self.task_queue:
            return False
        
        # ?°ì„ ?œìœ„ë³„ë¡œ ?•ë ¬
        self.task_queue.sort(key=lambda x: x.priority)
        task = self.task_queue.pop(0)
        
        self.logger.info(f"Processing task: {task.task_id}")
        task.status = 'processing'
        task.started_at = datetime.now()
        
        try:
            # ?…ë ¥ ?Œì¼ ?˜ì§‘
            if task.input_path.is_file():
                input_files = [task.input_path]
            else:
                input_files = list(task.input_path.rglob('*.json'))
            
            if not input_files:
                raise ValueError(f"No JSON files found in {task.input_path}")
            
            # ?„í¬???¤í–‰
            if task.data_type == 'law':
                records_imported = self._import_law_files(input_files)
            else:
                records_imported = self._import_precedent_files(input_files, task.category)
            
            task.records_imported = records_imported
            task.status = 'completed'
            task.completed_at = datetime.now()
            self.completed_tasks.append(task)
            self.logger.info(f"Task completed successfully: {task.task_id} - {records_imported} records imported")
            
            return True
            
        except Exception as e:
            task.status = 'failed'
            task.error_message = str(e)
            task.completed_at = datetime.now()
            self.failed_tasks.append(task)
            self.logger.error(f"Task failed with exception: {task.task_id} - {e}")
            return True
    
    def _import_law_files(self, input_files: List[Path]) -> int:
        """ë²•ë ¹ ?Œì¼???„í¬??""
        total_records = 0
        
        for file_path in input_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # ë²•ë ¹ ?°ì´??ì²˜ë¦¬
                if isinstance(data, dict) and 'articles' in data:
                    # ?¨ì¼ ë²•ë ¹
                    records = self._import_single_law(data, file_path)
                    total_records += records
                elif isinstance(data, list):
                    # ë²•ë ¹ ë°°ì—´
                    for law_data in data:
                        if isinstance(law_data, dict):
                            records = self._import_single_law(law_data, file_path)
                            total_records += records
                
            except Exception as e:
                self.logger.error(f"Error importing law file {file_path}: {e}")
                continue
        
        return total_records
    
    def _import_single_law(self, law_data: Dict[str, Any], file_path: Path) -> int:
        """?¨ì¼ ë²•ë ¹ ?„í¬??""
        try:
            # ë²•ë ¹ ê¸°ë³¸ ?•ë³´
            law_id = law_data.get('law_id', '')
            law_name = law_data.get('law_name', '')
            law_type = law_data.get('law_type', '')
            enactment_date = law_data.get('enactment_date', '')
            
            # ë²•ë ¹ ?Œì´ë¸”ì— ?½ì…
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO laws 
                    (law_id, law_name, law_type, enactment_date, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (law_id, law_name, law_type, enactment_date, datetime.now().isoformat()))
                
                law_db_id = cursor.lastrowid
                
                # ì¡°ë¬¸ ì²˜ë¦¬
                articles = law_data.get('articles', [])
                for article in articles:
                    if isinstance(article, dict):
                        article_number = article.get('article_number', '')
                        title = article.get('title', '')
                        content = article.get('content', '')
                        searchable_text = article.get('searchable_text', content)
                        
                        cursor.execute("""
                            INSERT OR REPLACE INTO law_articles 
                            (law_id, article_number, title, content, searchable_text, created_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (law_id, article_number, title, content, searchable_text, datetime.now().isoformat()))
                
                conn.commit()
                return len(articles)
        
        except Exception as e:
            self.logger.error(f"Error importing single law: {e}")
            return 0
    
    def _import_precedent_files(self, input_files: List[Path], category: str) -> int:
        """?ë? ?Œì¼???„í¬??""
        total_records = 0
        
        for file_path in input_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # ?ë? ?°ì´??ì²˜ë¦¬
                if isinstance(data, dict) and 'items' in data:
                    # items ë°°ì—´ ì²˜ë¦¬
                    items = data.get('items', [])
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                records = self._import_single_precedent(item, file_path, category)
                                total_records += records
                elif isinstance(data, list):
                    # ì§ì ‘ ë°°ì—´
                    for item in data:
                        if isinstance(item, dict):
                            records = self._import_single_precedent(item, file_path, category)
                            total_records += records
                
            except Exception as e:
                self.logger.error(f"Error importing precedent file {file_path}: {e}")
                continue
        
        return total_records
    
    def _import_single_precedent(self, precedent_data: Dict[str, Any], file_path: Path, category: str) -> int:
        """?¨ì¼ ?ë? ?„í¬??""
        try:
            # ?ë? ê¸°ë³¸ ?•ë³´
            case_id = precedent_data.get('case_id', '')
            case_name = precedent_data.get('case_name', '')
            case_number = precedent_data.get('case_number', '')
            decision_date = precedent_data.get('decision_date', '')
            court = precedent_data.get('court', '')
            field = precedent_data.get('field', '')
            detail_url = precedent_data.get('detail_url', '')
            full_text = precedent_data.get('precedent_content', precedent_data.get('full_text', ''))
            searchable_text = precedent_data.get('searchable_text', full_text)
            
            # ?ë? ?Œì´ë¸”ì— ?½ì…
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO precedent_cases 
                    (case_id, case_name, case_number, decision_date, field, court, category, detail_url, full_text, searchable_text, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (case_id, case_name, case_number, decision_date, field, court, category, detail_url, full_text, searchable_text, datetime.now().isoformat()))
                
                conn.commit()
                return 1
        
        except Exception as e:
            self.logger.error(f"Error importing single precedent: {e}")
            return 0
    
    def process_all_tasks(self) -> Dict[str, Any]:
        """ëª¨ë“  ?‘ì—… ì²˜ë¦¬"""
        self.logger.info(f"Processing {len(self.task_queue)} tasks...")
        
        start_time = datetime.now()
        processed_count = 0
        total_records = 0
        
        while self.task_queue:
            success = self.process_next_task()
            if success:
                processed_count += 1
                # ?„ë£Œ???‘ì—…???ˆì½”????ì¶”ê?
                if self.completed_tasks:
                    total_records += self.completed_tasks[-1].records_imported
            else:
                break
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # ê²°ê³¼ ?”ì•½
        summary = {
            'total_tasks': processed_count,
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'total_records_imported': total_records,
            'processing_time_seconds': processing_time,
            'completed_task_ids': [t.task_id for t in self.completed_tasks],
            'failed_task_ids': [t.task_id for t in self.failed_tasks]
        }
        
        self.logger.info(f"Import complete: {summary}")
        return summary
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """?‘ì—… ?íƒœ ì¡°íšŒ"""
        # ?€ê¸?ì¤‘ì¸ ?‘ì—…?ì„œ ì°¾ê¸°
        for task in self.task_queue:
            if task.task_id == task_id:
                return {
                    'task_id': task.task_id,
                    'status': task.status,
                    'data_type': task.data_type,
                    'category': task.category,
                    'created_at': task.created_at.isoformat(),
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'error_message': task.error_message,
                    'records_imported': task.records_imported
                }
        
        # ?„ë£Œ???‘ì—…?ì„œ ì°¾ê¸°
        for task in self.completed_tasks + self.failed_tasks:
            if task.task_id == task_id:
                return {
                    'task_id': task.task_id,
                    'status': task.status,
                    'data_type': task.data_type,
                    'category': task.category,
                    'created_at': task.created_at.isoformat(),
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'error_message': task.error_message,
                    'records_imported': task.records_imported
                }
        
        return None
    
    def get_all_tasks_status(self) -> Dict[str, Any]:
        """ëª¨ë“  ?‘ì—… ?íƒœ ì¡°íšŒ"""
        return {
            'pending_tasks': len(self.task_queue),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'task_details': {
                'pending': [
                    {
                        'task_id': t.task_id,
                        'data_type': t.data_type,
                        'category': t.category,
                        'priority': t.priority,
                        'created_at': t.created_at.isoformat()
                    } for t in self.task_queue
                ],
                'completed': [
                    {
                        'task_id': t.task_id,
                        'data_type': t.data_type,
                        'category': t.category,
                        'records_imported': t.records_imported,
                        'completed_at': t.completed_at.isoformat() if t.completed_at else None
                    } for t in self.completed_tasks
                ],
                'failed': [
                    {
                        'task_id': t.task_id,
                        'data_type': t.data_type,
                        'category': t.category,
                        'error_message': t.error_message,
                        'completed_at': t.completed_at.isoformat() if t.completed_at else None
                    } for t in self.failed_tasks
                ]
            }
        }


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    parser = argparse.ArgumentParser(description='Unified Import Manager')
    parser.add_argument('--mode', choices=['law', 'precedent', 'status'], required=True, help='Operation mode')
    parser.add_argument('--input', required=True, help='Input path')
    parser.add_argument('--category', help='Category (for precedent mode)')
    parser.add_argument('--priority', type=int, default=1, help='Priority (1=high, 2=medium, 3=low)')
    parser.add_argument('--db-path', default='data/lawfirm.db', help='Database path')
    
    args = parser.parse_args()
    
    # ë¡œê¹… ?¤ì •
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ë§¤ë‹ˆ?€ ì´ˆê¸°??
    manager = UnifiedImportManager(db_path=args.db_path)
    
    if args.mode == 'law':
        # ë²•ë ¹ ?„í¬??
        task_id = manager.add_law_import_task(args.input, args.priority)
        print(f"Added law import task: {task_id}")
        
        summary = manager.process_all_tasks()
        print(f"Import complete: {summary}")
    
    elif args.mode == 'precedent':
        # ?ë? ?„í¬??
        if not args.category:
            print("Error: --category is required for precedent mode")
            return
        
        task_id = manager.add_precedent_import_task(args.input, args.category, args.priority)
        print(f"Added precedent import task: {task_id}")
        
        summary = manager.process_all_tasks()
        print(f"Import complete: {summary}")
    
    elif args.mode == 'status':
        # ?íƒœ ì¡°íšŒ
        status = manager.get_all_tasks_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
