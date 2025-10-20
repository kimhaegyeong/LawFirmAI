#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Import Manager

모든 데이터 임포트 작업을 통합 관리하는 매니저 클래스입니다.
- 법령 데이터 임포트
- 판례 데이터 임포트
- 데이터 검증
- 중복 처리
- 오류 복구
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

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.database import DatabaseManager


@dataclass
class ImportTask:
    """임포트 작업 데이터 클래스"""
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
    """통합 임포트 매니저"""
    
    def __init__(self, db_path: str = "data/lawfirm.db"):
        """
        임포트 매니저 초기화
        
        Args:
            db_path: 데이터베이스 경로
        """
        self.db_manager = DatabaseManager(db_path)
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 작업 큐
        self.task_queue: List[ImportTask] = []
        self.completed_tasks: List[ImportTask] = []
        self.failed_tasks: List[ImportTask] = []
    
    def add_law_import_task(self, 
                           input_path: Union[str, Path],
                           priority: int = 1) -> str:
        """법령 임포트 작업 추가"""
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
        """판례 임포트 작업 추가"""
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
        """다음 작업 처리"""
        if not self.task_queue:
            return False
        
        # 우선순위별로 정렬
        self.task_queue.sort(key=lambda x: x.priority)
        task = self.task_queue.pop(0)
        
        self.logger.info(f"Processing task: {task.task_id}")
        task.status = 'processing'
        task.started_at = datetime.now()
        
        try:
            # 입력 파일 수집
            if task.input_path.is_file():
                input_files = [task.input_path]
            else:
                input_files = list(task.input_path.rglob('*.json'))
            
            if not input_files:
                raise ValueError(f"No JSON files found in {task.input_path}")
            
            # 임포트 실행
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
        """법령 파일들 임포트"""
        total_records = 0
        
        for file_path in input_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 법령 데이터 처리
                if isinstance(data, dict) and 'articles' in data:
                    # 단일 법령
                    records = self._import_single_law(data, file_path)
                    total_records += records
                elif isinstance(data, list):
                    # 법령 배열
                    for law_data in data:
                        if isinstance(law_data, dict):
                            records = self._import_single_law(law_data, file_path)
                            total_records += records
                
            except Exception as e:
                self.logger.error(f"Error importing law file {file_path}: {e}")
                continue
        
        return total_records
    
    def _import_single_law(self, law_data: Dict[str, Any], file_path: Path) -> int:
        """단일 법령 임포트"""
        try:
            # 법령 기본 정보
            law_id = law_data.get('law_id', '')
            law_name = law_data.get('law_name', '')
            law_type = law_data.get('law_type', '')
            enactment_date = law_data.get('enactment_date', '')
            
            # 법령 테이블에 삽입
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO laws 
                    (law_id, law_name, law_type, enactment_date, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (law_id, law_name, law_type, enactment_date, datetime.now().isoformat()))
                
                law_db_id = cursor.lastrowid
                
                # 조문 처리
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
        """판례 파일들 임포트"""
        total_records = 0
        
        for file_path in input_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 판례 데이터 처리
                if isinstance(data, dict) and 'items' in data:
                    # items 배열 처리
                    items = data.get('items', [])
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                records = self._import_single_precedent(item, file_path, category)
                                total_records += records
                elif isinstance(data, list):
                    # 직접 배열
                    for item in data:
                        if isinstance(item, dict):
                            records = self._import_single_precedent(item, file_path, category)
                            total_records += records
                
            except Exception as e:
                self.logger.error(f"Error importing precedent file {file_path}: {e}")
                continue
        
        return total_records
    
    def _import_single_precedent(self, precedent_data: Dict[str, Any], file_path: Path, category: str) -> int:
        """단일 판례 임포트"""
        try:
            # 판례 기본 정보
            case_id = precedent_data.get('case_id', '')
            case_name = precedent_data.get('case_name', '')
            case_number = precedent_data.get('case_number', '')
            decision_date = precedent_data.get('decision_date', '')
            court = precedent_data.get('court', '')
            field = precedent_data.get('field', '')
            detail_url = precedent_data.get('detail_url', '')
            full_text = precedent_data.get('precedent_content', precedent_data.get('full_text', ''))
            searchable_text = precedent_data.get('searchable_text', full_text)
            
            # 판례 테이블에 삽입
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
        """모든 작업 처리"""
        self.logger.info(f"Processing {len(self.task_queue)} tasks...")
        
        start_time = datetime.now()
        processed_count = 0
        total_records = 0
        
        while self.task_queue:
            success = self.process_next_task()
            if success:
                processed_count += 1
                # 완료된 작업의 레코드 수 추가
                if self.completed_tasks:
                    total_records += self.completed_tasks[-1].records_imported
            else:
                break
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # 결과 요약
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
        """작업 상태 조회"""
        # 대기 중인 작업에서 찾기
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
        
        # 완료된 작업에서 찾기
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
        """모든 작업 상태 조회"""
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
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Unified Import Manager')
    parser.add_argument('--mode', choices=['law', 'precedent', 'status'], required=True, help='Operation mode')
    parser.add_argument('--input', required=True, help='Input path')
    parser.add_argument('--category', help='Category (for precedent mode)')
    parser.add_argument('--priority', type=int, default=1, help='Priority (1=high, 2=medium, 3=low)')
    parser.add_argument('--db-path', default='data/lawfirm.db', help='Database path')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 매니저 초기화
    manager = UnifiedImportManager(db_path=args.db_path)
    
    if args.mode == 'law':
        # 법령 임포트
        task_id = manager.add_law_import_task(args.input, args.priority)
        print(f"Added law import task: {task_id}")
        
        summary = manager.process_all_tasks()
        print(f"Import complete: {summary}")
    
    elif args.mode == 'precedent':
        # 판례 임포트
        if not args.category:
            print("Error: --category is required for precedent mode")
            return
        
        task_id = manager.add_precedent_import_task(args.input, args.category, args.priority)
        print(f"Added precedent import task: {task_id}")
        
        summary = manager.process_all_tasks()
        print(f"Import complete: {summary}")
    
    elif args.mode == 'status':
        # 상태 조회
        status = manager.get_all_tasks_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
