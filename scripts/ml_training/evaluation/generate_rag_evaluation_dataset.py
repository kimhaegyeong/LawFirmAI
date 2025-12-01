#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 평가 데이터셋 통합 생성

클러스터링 기반과 Pseudo-Query 기반 Ground Truth를 통합하고,
train/val/test 분할을 수행합니다.
"""

import logging
import json
import sys
import gc
import pickle
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _process_entry_batch(args: tuple) -> List[Dict[str, Any]]:
    """배치 단위로 엔트리 처리 (멀티프로세싱용)"""
    batch, operation, min_relevant_docs = args
    
    if operation == "deduplicate":
        seen = set()
        unique_entries = []
        for entry in batch:
            if 'query' in entry:
                key = entry.get('query', '')
            elif 'query_text' in entry:
                key = entry.get('query_text', '')
            else:
                key = str(entry)
            
            if key and key not in seen:
                seen.add(key)
                unique_entries.append(entry)
        return unique_entries
    
    elif operation == "filter":
        filtered = []
        for entry in batch:
            if 'query' in entry:
                query = entry.get('query', '')
                if not query or len(query.strip()) < 5:
                    continue
            elif 'query_text' in entry:
                query = entry.get('query_text', '')
                if not query or len(query.strip()) < 10:
                    continue
            else:
                continue
            
            if 'relevant_doc_ids' in entry:
                relevant_docs = entry.get('relevant_doc_ids', [])
                if len(relevant_docs) < min_relevant_docs:
                    continue
            
            filtered.append(entry)
        return filtered
    
    return batch


def _save_dataset_worker(args: tuple) -> None:
    """데이터셋 저장 워커 (멀티프로세싱용)"""
    data, output_path = args
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class RAGEvaluationDatasetGenerator:
    """RAG 평가 데이터셋 생성 클래스"""
    
    def __init__(self, clustering_path: Optional[str] = None,
                 pseudo_queries_path: Optional[str] = None,
                 checkpoint_dir: Optional[str] = None,
                 num_workers: Optional[int] = None):
        """
        초기화
        
        Args:
            clustering_path: 클러스터링 기반 Ground Truth 파일 경로
            pseudo_queries_path: Pseudo-Query 기반 Ground Truth 파일 경로
            checkpoint_dir: 체크포인트 저장 디렉토리 경로
            num_workers: 병렬 처리에 사용할 워커 수 (None이면 CPU 코어 수 사용)
        """
        self.clustering_path = Path(clustering_path) if clustering_path else None
        self.pseudo_queries_path = Path(pseudo_queries_path) if pseudo_queries_path else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.num_workers = num_workers if num_workers is not None else max(1, cpu_count() - 1)
        
        if not self.clustering_path and not self.pseudo_queries_path:
            raise ValueError("At least one ground truth file path must be provided")
        
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized with {self.num_workers} workers")
    
    def load_ground_truth(self, file_path: Path) -> List[Dict[str, Any]]:
        """Ground Truth 파일 로드"""
        logger.info(f"Loading ground truth from: {file_path}")
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'ground_truth' in data:
            return data['ground_truth']
        elif isinstance(data, list):
            return data
        else:
            logger.warning(f"Unexpected data format in {file_path}")
            return []
    
    def deduplicate_ground_truth(self, ground_truth: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """중복 제거 (병렬 처리 지원)"""
        logger.info(f"Deduplicating ground truth entries with {self.num_workers} workers...")
        
        if len(ground_truth) < 1000 or self.num_workers == 1:
            seen = set()
            unique_entries = []
            
            for entry in ground_truth:
                if 'query' in entry:
                    key = entry.get('query', '')
                elif 'query_text' in entry:
                    key = entry.get('query_text', '')
                else:
                    key = str(entry)
                
                if key and key not in seen:
                    seen.add(key)
                    unique_entries.append(entry)
            
            logger.info(f"Deduplicated: {len(ground_truth)} -> {len(unique_entries)} entries")
            return unique_entries
        
        batch_size = max(100, len(ground_truth) // (self.num_workers * 4))
        batches = [ground_truth[i:i + batch_size] for i in range(0, len(ground_truth), batch_size)]
        
        seen = set()
        unique_entries = []
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(_process_entry_batch, (batch, "deduplicate", 1)) for batch in batches]
            
            for future in as_completed(futures):
                batch_result = future.result()
                for entry in batch_result:
                    if 'query' in entry:
                        key = entry.get('query', '')
                    elif 'query_text' in entry:
                        key = entry.get('query_text', '')
                    else:
                        key = str(entry)
                    
                    if key and key not in seen:
                        seen.add(key)
                        unique_entries.append(entry)
        
        logger.info(f"Deduplicated: {len(ground_truth)} -> {len(unique_entries)} entries")
        return unique_entries
    
    def filter_quality(self, ground_truth: List[Dict[str, Any]],
                      min_relevant_docs: int = 1) -> List[Dict[str, Any]]:
        """품질 필터링 (병렬 처리 지원)"""
        logger.info(f"Filtering ground truth by quality with {self.num_workers} workers...")
        
        if len(ground_truth) < 1000 or self.num_workers == 1:
            filtered = []
            
            for entry in ground_truth:
                if 'query' in entry:
                    query = entry.get('query', '')
                    if not query or len(query.strip()) < 5:
                        continue
                elif 'query_text' in entry:
                    query = entry.get('query_text', '')
                    if not query or len(query.strip()) < 10:
                        continue
                else:
                    continue
                
                if 'relevant_doc_ids' in entry:
                    relevant_docs = entry.get('relevant_doc_ids', [])
                    if len(relevant_docs) < min_relevant_docs:
                        continue
                
                filtered.append(entry)
            
            logger.info(f"Filtered: {len(ground_truth)} -> {len(filtered)} entries")
            return filtered
        
        batch_size = max(100, len(ground_truth) // (self.num_workers * 4))
        batches = [ground_truth[i:i + batch_size] for i in range(0, len(ground_truth), batch_size)]
        
        filtered = []
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(_process_entry_batch, (batch, "filter", min_relevant_docs)) for batch in batches]
            
            for future in as_completed(futures):
                batch_result = future.result()
                filtered.extend(batch_result)
        
        logger.info(f"Filtered: {len(ground_truth)} -> {len(filtered)} entries")
        return filtered
    
    def combine_ground_truth(self, resume_from_checkpoint: bool = True) -> List[Dict[str, Any]]:
        """두 Ground Truth 데이터셋 통합"""
        logger.info("Combining ground truth datasets...")
        
        combined = None
        checkpoint_loaded = None
        
        if resume_from_checkpoint and self.checkpoint_dir:
            checkpoint_names = [
                ("filtered_ground_truth.pkl", "filtered"),
                ("deduplicated_ground_truth.pkl", "deduplicated"),
                ("combine_ground_truth.pkl", "combined")
            ]
            
            for checkpoint_name, stage in checkpoint_names:
                checkpoint = self.load_checkpoint(checkpoint_name)
                if checkpoint and 'combined' in checkpoint:
                    logger.info(f"Resuming from checkpoint: {checkpoint_name} (stage: {stage})")
                    combined = checkpoint['combined']
                    checkpoint_loaded = stage
                    gc.collect()
                    break
        
        if combined is None:
            combined = []
            
            if self.clustering_path and self.pseudo_queries_path:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    clustering_future = executor.submit(self.load_ground_truth, self.clustering_path)
                    pseudo_queries_future = executor.submit(self.load_ground_truth, self.pseudo_queries_path)
                    
                    clustering_gt = clustering_future.result()
                    pseudo_queries_gt = pseudo_queries_future.result()
                
                for entry in clustering_gt:
                    entry['source'] = 'clustering'
                combined.extend(clustering_gt)
                logger.info(f"Loaded {len(clustering_gt)} entries from clustering")
                del clustering_gt
                gc.collect()
                
                for entry in pseudo_queries_gt:
                    entry['source'] = 'pseudo_queries'
                combined.extend(pseudo_queries_gt)
                logger.info(f"Loaded {len(pseudo_queries_gt)} entries from pseudo-queries")
                del pseudo_queries_gt
                gc.collect()
            else:
                if self.clustering_path:
                    clustering_gt = self.load_ground_truth(self.clustering_path)
                    for entry in clustering_gt:
                        entry['source'] = 'clustering'
                    combined.extend(clustering_gt)
                    logger.info(f"Loaded {len(clustering_gt)} entries from clustering")
                    del clustering_gt
                    gc.collect()
                
                if self.pseudo_queries_path:
                    pseudo_queries_gt = self.load_ground_truth(self.pseudo_queries_path)
                    for entry in pseudo_queries_gt:
                        entry['source'] = 'pseudo_queries'
                    combined.extend(pseudo_queries_gt)
                    logger.info(f"Loaded {len(pseudo_queries_gt)} entries from pseudo-queries")
                    del pseudo_queries_gt
                    gc.collect()
            
            if self.checkpoint_dir:
                self.save_checkpoint({'combined': combined}, "combine_ground_truth.pkl")
        
        if checkpoint_loaded != "deduplicated" and checkpoint_loaded != "filtered":
            combined = self.deduplicate_ground_truth(combined)
            gc.collect()
            
            if self.checkpoint_dir:
                self.save_checkpoint({'combined': combined}, "deduplicated_ground_truth.pkl")
        
        if checkpoint_loaded != "filtered":
            combined = self.filter_quality(combined)
            gc.collect()
            
            if self.checkpoint_dir:
                self.save_checkpoint({'combined': combined}, "filtered_ground_truth.pkl")
        
        logger.info(f"Combined total: {len(combined)} entries")
        return combined
    
    def split_dataset(self, ground_truth: List[Dict[str, Any]],
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     random_seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
        """
        데이터셋 분할
        
        Args:
            ground_truth: Ground Truth 데이터셋
            train_ratio: 학습 데이터 비율
            val_ratio: 검증 데이터 비율
            test_ratio: 테스트 데이터 비율
            random_seed: 랜덤 시드
        
        Returns:
            분할된 데이터셋 딕셔너리
        """
        logger.info("Splitting dataset...")
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError("Ratios must sum to 1.0")
        
        random.seed(random_seed)
        shuffled = ground_truth.copy()
        random.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data = shuffled[:n_train]
        val_data = shuffled[n_train:n_train + n_val]
        test_data = shuffled[n_train + n_val:]
        
        logger.info(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def save_dataset(self, data: List[Dict[str, Any]], output_path: Path):
        """데이터셋 저장"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved dataset to: {output_path}")
    
    def save_checkpoint(self, state: Dict[str, Any], checkpoint_name: str = "checkpoint.pkl"):
        """체크포인트 저장"""
        if not self.checkpoint_dir:
            return
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_name: str = "checkpoint.pkl") -> Optional[Dict[str, Any]]:
        """체크포인트 로드"""
        if not self.checkpoint_dir:
            return None
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'rb') as f:
                state = pickle.load(f)
            logger.info(f"Checkpoint loaded from: {checkpoint_path}")
            return state
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None
    
    def run(self, train_ratio: float = 0.7, val_ratio: float = 0.15,
           test_ratio: float = 0.15, output_path: str = "data/evaluation/rag_ground_truth_combined.json",
           random_seed: int = 42, resume_from_checkpoint: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        전체 프로세스 실행
        
        Args:
            train_ratio: 학습 데이터 비율
            val_ratio: 검증 데이터 비율
            test_ratio: 테스트 데이터 비율
            output_path: 출력 파일 경로
            random_seed: 랜덤 시드
            resume_from_checkpoint: 체크포인트에서 재개할지 여부
        
        Returns:
            분할된 데이터셋 딕셔너리
        """
        combined = self.combine_ground_truth(resume_from_checkpoint=resume_from_checkpoint)
        gc.collect()
        
        output_path_obj = Path(output_path)
        self.save_dataset(combined, output_path_obj)
        
        if self.checkpoint_dir:
            checkpoint = self.load_checkpoint("split_dataset.pkl")
            if checkpoint and resume_from_checkpoint and 'splits' in checkpoint:
                logger.info("Resuming from split checkpoint...")
                splits = checkpoint['splits']
            else:
                splits = self.split_dataset(combined, train_ratio, val_ratio, test_ratio, random_seed)
                if self.checkpoint_dir:
                    self.save_checkpoint({'splits': splits}, "split_dataset.pkl")
        else:
            splits = self.split_dataset(combined, train_ratio, val_ratio, test_ratio, random_seed)
        
        del combined
        gc.collect()
        
        base_path = output_path_obj.parent / output_path_obj.stem
        
        save_tasks = [
            (splits['train'], str(base_path.parent / f"{base_path.name}_train.json")),
            (splits['val'], str(base_path.parent / f"{base_path.name}_val.json")),
            (splits['test'], str(base_path.parent / f"{base_path.name}_test.json"))
        ]
        
        with ThreadPoolExecutor(max_workers=min(3, len(save_tasks))) as executor:
            futures = [executor.submit(_save_dataset_worker, task) for task in save_tasks]
            for future in as_completed(futures):
                future.result()
                logger.info("Dataset file saved")
        
        gc.collect()
        
        return splits


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="RAG 평가 데이터셋 통합 생성")
    parser.add_argument(
        "--clustering-path",
        type=str,
        default=None,
        help="클러스터링 기반 Ground Truth 파일 경로"
    )
    parser.add_argument(
        "--pseudo-queries-path",
        type=str,
        default=None,
        help="Pseudo-Query 기반 Ground Truth 파일 경로"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/evaluation/rag_ground_truth_combined.json",
        help="출력 파일 경로"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="학습 데이터 비율"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="검증 데이터 비율"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="테스트 데이터 비율"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="랜덤 시드"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="체크포인트 저장 디렉토리 경로"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="체크포인트에서 재개하지 않음"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="병렬 처리에 사용할 워커 수 (기본값: CPU 코어 수 - 1)"
    )
    
    args = parser.parse_args()
    
    if not args.clustering_path and not args.pseudo_queries_path:
        logger.error("At least one ground truth file path must be provided")
        sys.exit(1)
    
    try:
        generator = RAGEvaluationDatasetGenerator(
            clustering_path=args.clustering_path,
            pseudo_queries_path=args.pseudo_queries_path,
            checkpoint_dir=args.checkpoint_dir,
            num_workers=args.num_workers
        )
        
        splits = generator.run(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            output_path=args.output_path,
            random_seed=args.random_seed,
            resume_from_checkpoint=not args.no_resume
        )
        
        logger.info("Dataset generation completed successfully!")
        logger.info(f"Train: {len(splits['train'])} entries")
        logger.info(f"Val: {len(splits['val'])} entries")
        logger.info(f"Test: {len(splits['test'])} entries")
        
    except Exception as e:
        logger.error(f"Failed to generate dataset: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

