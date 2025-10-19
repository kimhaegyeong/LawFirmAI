#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Preprocessing Manager

모든 전처리 작업을 통합 관리하는 매니저 클래스입니다.
- 법령 전처리
- 판례 전처리
- 품질 검증
- 벡터 임베딩
- 데이터베이스 저장
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import argparse
from dataclasses import dataclass

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.database import DatabaseManager
from scripts.data_collection.common.checkpoint_manager import CheckpointManager
from scripts.data_processing.enhanced_preprocessing_pipeline import EnhancedPreprocessingPipeline, ProcessingConfig
from scripts.data_processing.auto_data_detector import AutoDataDetector


@dataclass
class PreprocessingTask:
    """전처리 작업 데이터 클래스"""
    task_id: str
    data_type: str  # 'law' or 'precedent'
    input_path: Path
    output_path: Path
    category: Optional[str] = None
    priority: int = 1  # 1=high, 2=medium, 3=low
    status: str = 'pending'  # pending, processing, completed, failed
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class UnifiedPreprocessingManager:
    """통합 전처리 매니저"""
    
    def __init__(self, 
                 config: ProcessingConfig = None,
                 checkpoint_dir: str = "data/checkpoints",
                 db_path: str = "data/lawfirm.db"):
        """
        전처리 매니저 초기화
        
        Args:
            config: 처리 설정
            checkpoint_dir: 체크포인트 디렉토리
            db_path: 데이터베이스 경로
        """
        self.config = config or ProcessingConfig()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 컴포넌트 초기화
        self.db_manager = DatabaseManager(db_path)
        self.checkpoint_manager = CheckpointManager(str(self.checkpoint_dir))
        self.pipeline = EnhancedPreprocessingPipeline(
            config=self.config,
            checkpoint_manager=self.checkpoint_manager,
            db_manager=self.db_manager
        )
        self.data_detector = AutoDataDetector("data/raw/assembly", self.db_manager)
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 작업 큐
        self.task_queue: List[PreprocessingTask] = []
        self.completed_tasks: List[PreprocessingTask] = []
        self.failed_tasks: List[PreprocessingTask] = []
    
    def add_law_processing_task(self, 
                               input_path: Union[str, Path], 
                               output_path: Union[str, Path],
                               category: str = None,
                               priority: int = 1) -> str:
        """법령 전처리 작업 추가"""
        task_id = f"law_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.task_queue)}"
        
        task = PreprocessingTask(
            task_id=task_id,
            data_type='law',
            input_path=Path(input_path),
            output_path=Path(output_path),
            category=category,
            priority=priority
        )
        
        self.task_queue.append(task)
        self.logger.info(f"Added law processing task: {task_id}")
        return task_id
    
    def add_precedent_processing_task(self, 
                                     input_path: Union[str, Path], 
                                     output_path: Union[str, Path],
                                     category: str = None,
                                     priority: int = 1) -> str:
        """판례 전처리 작업 추가"""
        task_id = f"precedent_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.task_queue)}"
        
        task = PreprocessingTask(
            task_id=task_id,
            data_type='precedent',
            input_path=Path(input_path),
            output_path=Path(output_path),
            category=category,
            priority=priority
        )
        
        self.task_queue.append(task)
        self.logger.info(f"Added precedent processing task: {task_id}")
        return task_id
    
    def auto_detect_and_add_tasks(self, 
                                 base_path: str = "data/raw/assembly",
                                 output_base_path: str = "data/processed/assembly") -> List[str]:
        """자동으로 데이터를 감지하고 작업 추가"""
        self.logger.info("Auto-detecting data and adding tasks...")
        
        # 데이터 감지
        detected_data = self.data_detector.detect_new_data_sources(str(self.data_detector.raw_data_base_path))
        
        task_ids = []
        
        # 감지된 데이터를 작업으로 변환
        for data_type, files in detected_data.items():
            if not files:
                continue
            
            # 출력 경로 설정
            if data_type.startswith('precedent_'):
                category = data_type.replace('precedent_', '')
                output_path = Path(output_base_path) / 'precedent' / category
            else:
                output_path = Path(output_base_path) / data_type
            
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 작업 추가
            if data_type.startswith('precedent_'):
                task_id = self.add_precedent_processing_task(
                    input_path=files[0].parent,  # 디렉토리 경로
                    output_path=output_path,
                    category=category
                )
            else:
                task_id = self.add_law_processing_task(
                    input_path=files[0].parent,  # 디렉토리 경로
                    output_path=output_path
                )
            
            task_ids.append(task_id)
        
        self.logger.info(f"Added {len(task_ids)} tasks from auto-detection")
        return task_ids
    
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
            
            # 출력 디렉토리 생성
            task.output_path.mkdir(parents=True, exist_ok=True)
            
            # 전처리 실행
            if task.data_type == 'law':
                result = self.pipeline.process_law_files(input_files, task.output_path)
            else:
                result = self.pipeline.process_precedent_files(input_files, task.output_path)
            
            if result.success:
                task.status = 'completed'
                task.completed_at = datetime.now()
                self.completed_tasks.append(task)
                self.logger.info(f"Task completed successfully: {task.task_id}")
            else:
                task.status = 'failed'
                task.error_message = '; '.join(result.errors)
                task.completed_at = datetime.now()
                self.failed_tasks.append(task)
                self.logger.error(f"Task failed: {task.task_id} - {task.error_message}")
            
            return True
            
        except Exception as e:
            task.status = 'failed'
            task.error_message = str(e)
            task.completed_at = datetime.now()
            self.failed_tasks.append(task)
            self.logger.error(f"Task failed with exception: {task.task_id} - {e}")
            return True
    
    def process_all_tasks(self) -> Dict[str, Any]:
        """모든 작업 처리"""
        self.logger.info(f"Processing {len(self.task_queue)} tasks...")
        
        start_time = datetime.now()
        processed_count = 0
        
        while self.task_queue:
            success = self.process_next_task()
            if success:
                processed_count += 1
            else:
                break
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # 결과 요약
        summary = {
            'total_tasks': processed_count,
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'processing_time_seconds': processing_time,
            'completed_task_ids': [t.task_id for t in self.completed_tasks],
            'failed_task_ids': [t.task_id for t in self.failed_tasks]
        }
        
        self.logger.info(f"Processing complete: {summary}")
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
                    'created_at': task.created_at.isoformat(),
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'error_message': task.error_message
                }
        
        # 완료된 작업에서 찾기
        for task in self.completed_tasks + self.failed_tasks:
            if task.task_id == task_id:
                return {
                    'task_id': task.task_id,
                    'status': task.status,
                    'data_type': task.data_type,
                    'created_at': task.created_at.isoformat(),
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'error_message': task.error_message
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
                        'priority': t.priority,
                        'created_at': t.created_at.isoformat()
                    } for t in self.task_queue
                ],
                'completed': [
                    {
                        'task_id': t.task_id,
                        'data_type': t.data_type,
                        'completed_at': t.completed_at.isoformat() if t.completed_at else None
                    } for t in self.completed_tasks
                ],
                'failed': [
                    {
                        'task_id': t.task_id,
                        'data_type': t.data_type,
                        'error_message': t.error_message,
                        'completed_at': t.completed_at.isoformat() if t.completed_at else None
                    } for t in self.failed_tasks
                ]
            }
        }
    
    def save_checkpoint(self, checkpoint_file: str = None):
        """체크포인트 저장"""
        if checkpoint_file is None:
            checkpoint_file = self.checkpoint_dir / f"preprocessing_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        checkpoint_data = {
            'config': {
                'max_workers': self.config.max_workers,
                'batch_size': self.config.batch_size,
                'max_memory_gb': self.config.max_memory_gb,
                'enable_quality_validation': self.config.enable_quality_validation,
                'enable_duplicate_detection': self.config.enable_duplicate_detection
            },
            'tasks': {
                'pending': [
                    {
                        'task_id': t.task_id,
                        'data_type': t.data_type,
                        'input_path': str(t.input_path),
                        'output_path': str(t.output_path),
                        'category': t.category,
                        'priority': t.priority,
                        'status': t.status,
                        'created_at': t.created_at.isoformat()
                    } for t in self.task_queue
                ],
                'completed': [
                    {
                        'task_id': t.task_id,
                        'data_type': t.data_type,
                        'input_path': str(t.input_path),
                        'output_path': str(t.output_path),
                        'category': t.category,
                        'priority': t.priority,
                        'status': t.status,
                        'created_at': t.created_at.isoformat(),
                        'started_at': t.started_at.isoformat() if t.started_at else None,
                        'completed_at': t.completed_at.isoformat() if t.completed_at else None,
                        'error_message': t.error_message
                    } for t in self.completed_tasks
                ],
                'failed': [
                    {
                        'task_id': t.task_id,
                        'data_type': t.data_type,
                        'input_path': str(t.input_path),
                        'output_path': str(t.output_path),
                        'category': t.category,
                        'priority': t.priority,
                        'status': t.status,
                        'created_at': t.created_at.isoformat(),
                        'started_at': t.started_at.isoformat() if t.started_at else None,
                        'completed_at': t.completed_at.isoformat() if t.completed_at else None,
                        'error_message': t.error_message
                    } for t in self.failed_tasks
                ]
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Checkpoint saved to: {checkpoint_file}")
        return checkpoint_file
    
    def load_checkpoint(self, checkpoint_file: str):
        """체크포인트 로드"""
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        # 설정 복원
        config_data = checkpoint_data.get('config', {})
        self.config = ProcessingConfig(
            max_workers=config_data.get('max_workers', 4),
            batch_size=config_data.get('batch_size', 100),
            max_memory_gb=config_data.get('max_memory_gb', 8.0),
            enable_quality_validation=config_data.get('enable_quality_validation', True),
            enable_duplicate_detection=config_data.get('enable_duplicate_detection', True)
        )
        
        # 작업 복원
        tasks_data = checkpoint_data.get('tasks', {})
        
        # 대기 중인 작업
        self.task_queue = []
        for task_data in tasks_data.get('pending', []):
            task = PreprocessingTask(
                task_id=task_data['task_id'],
                data_type=task_data['data_type'],
                input_path=Path(task_data['input_path']),
                output_path=Path(task_data['output_path']),
                category=task_data.get('category'),
                priority=task_data.get('priority', 1),
                status=task_data.get('status', 'pending'),
                created_at=datetime.fromisoformat(task_data['created_at'])
            )
            self.task_queue.append(task)
        
        # 완료된 작업
        self.completed_tasks = []
        for task_data in tasks_data.get('completed', []):
            task = PreprocessingTask(
                task_id=task_data['task_id'],
                data_type=task_data['data_type'],
                input_path=Path(task_data['input_path']),
                output_path=Path(task_data['output_path']),
                category=task_data.get('category'),
                priority=task_data.get('priority', 1),
                status=task_data.get('status', 'completed'),
                created_at=datetime.fromisoformat(task_data['created_at']),
                started_at=datetime.fromisoformat(task_data['started_at']) if task_data.get('started_at') else None,
                completed_at=datetime.fromisoformat(task_data['completed_at']) if task_data.get('completed_at') else None,
                error_message=task_data.get('error_message')
            )
            self.completed_tasks.append(task)
        
        # 실패한 작업
        self.failed_tasks = []
        for task_data in tasks_data.get('failed', []):
            task = PreprocessingTask(
                task_id=task_data['task_id'],
                data_type=task_data['data_type'],
                input_path=Path(task_data['input_path']),
                output_path=Path(task_data['output_path']),
                category=task_data.get('category'),
                priority=task_data.get('priority', 1),
                status=task_data.get('status', 'failed'),
                created_at=datetime.fromisoformat(task_data['created_at']),
                started_at=datetime.fromisoformat(task_data['started_at']) if task_data.get('started_at') else None,
                completed_at=datetime.fromisoformat(task_data['completed_at']) if task_data.get('completed_at') else None,
                error_message=task_data.get('error_message')
            )
            self.failed_tasks.append(task)
        
        self.logger.info(f"Checkpoint loaded from: {checkpoint_file}")
        self.logger.info(f"Restored {len(self.task_queue)} pending, {len(self.completed_tasks)} completed, {len(self.failed_tasks)} failed tasks")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Unified Preprocessing Manager')
    parser.add_argument('--mode', choices=['auto', 'manual', 'status', 'resume'], required=True, help='Operation mode')
    parser.add_argument('--input', help='Input path (for manual mode)')
    parser.add_argument('--output', help='Output path (for manual mode)')
    parser.add_argument('--data-type', choices=['law', 'precedent'], help='Data type (for manual mode)')
    parser.add_argument('--category', help='Category (for precedent mode)')
    parser.add_argument('--checkpoint', help='Checkpoint file (for resume mode)')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of workers')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    parser.add_argument('--max-memory-gb', type=float, default=8.0, help='Maximum memory usage in GB')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 설정 생성
    config = ProcessingConfig(
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        max_memory_gb=args.max_memory_gb
    )
    
    # 매니저 초기화
    manager = UnifiedPreprocessingManager(config=config)
    
    if args.mode == 'auto':
        # 자동 감지 및 처리
        task_ids = manager.auto_detect_and_add_tasks()
        print(f"Added {len(task_ids)} tasks from auto-detection")
        
        if task_ids:
            summary = manager.process_all_tasks()
            print(f"Processing complete: {summary}")
    
    elif args.mode == 'manual':
        # 수동 작업 추가 및 처리
        if not all([args.input, args.output, args.data_type]):
            print("Error: --input, --output, and --data-type are required for manual mode")
            return
        
        if args.data_type == 'precedent' and not args.category:
            print("Error: --category is required for precedent data type")
            return
        
        if args.data_type == 'law':
            task_id = manager.add_law_processing_task(args.input, args.output)
        else:
            task_id = manager.add_precedent_processing_task(args.input, args.output, args.category)
        
        print(f"Added task: {task_id}")
        
        summary = manager.process_all_tasks()
        print(f"Processing complete: {summary}")
    
    elif args.mode == 'status':
        # 상태 조회
        status = manager.get_all_tasks_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
    
    elif args.mode == 'resume':
        # 체크포인트에서 복원
        if not args.checkpoint:
            print("Error: --checkpoint is required for resume mode")
            return
        
        manager.load_checkpoint(args.checkpoint)
        summary = manager.process_all_tasks()
        print(f"Processing complete: {summary}")


if __name__ == "__main__":
    main()
