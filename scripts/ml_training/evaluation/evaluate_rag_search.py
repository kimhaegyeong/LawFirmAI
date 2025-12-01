#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 검색 평가

생성된 Ground Truth를 사용하여 RAG 검색 시스템의 성능을 평가합니다.
Recall@K, Precision@K, MRR 등의 메트릭을 계산합니다.
"""

import logging
import json
import sys
import gc
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from lawfirm_langgraph.core.data.vector_store import LegalVectorStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGSearchEvaluator:
    """RAG 검색 평가 클래스"""
    
    def __init__(self, vector_store_path: str, model_name: str = "jhgan/ko-sroberta-multitask",
                 checkpoint_dir: Optional[str] = None):
        """
        초기화
        
        Args:
            vector_store_path: 벡터 스토어 경로
            model_name: 임베딩 모델명
            checkpoint_dir: 체크포인트 저장 디렉토리 경로
        """
        self.vector_store_path = Path(vector_store_path)
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initializing RAG Search Evaluator...")
        logger.info(f"Vector store path: {self.vector_store_path}")
        logger.info(f"Model name: {self.model_name}")
        if self.checkpoint_dir:
            logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        
        self.vector_store = LegalVectorStore(
            model_name=model_name,
            dimension=768,
            index_type="flat"
        )
        
        logger.info(f"Loading vector store from: {self.vector_store_path}")
        load_start = time.time()
        
        index_file = self.vector_store_path
        if self.vector_store_path.is_dir():
            if (self.vector_store_path / "ml_enhanced_faiss_index.faiss").exists():
                index_file = self.vector_store_path / "ml_enhanced_faiss_index"
            elif (self.vector_store_path / "index.faiss").exists():
                index_file = self.vector_store_path / "index"
            elif (self.vector_store_path / "faiss_index_jhgan_ko-sroberta-multitask.index").exists():
                index_file = self.vector_store_path / "faiss_index_jhgan_ko-sroberta-multitask"
            else:
                raise ValueError(f"Could not find FAISS index file in {self.vector_store_path}")
        
        if not self.vector_store.load_index(str(index_file)):
            raise ValueError(f"Failed to load vector store from {index_file}")
        load_time = time.time() - load_start
        
        self.document_texts = self.vector_store.document_texts
        self.document_metadata = self.vector_store.document_metadata
        
        logger.info(f"Loaded {len(self.document_texts)} documents in {load_time:.2f} seconds")
        gc.collect()
    
    def calculate_recall_at_k(self, retrieved_doc_ids: List[Any], 
                            relevant_doc_ids: List[Any], k: int) -> float:
        """Recall@K 계산"""
        if not relevant_doc_ids:
            return 0.0
        
        top_k_retrieved = retrieved_doc_ids[:k]
        relevant_retrieved = len(set(top_k_retrieved) & set(relevant_doc_ids))
        
        return relevant_retrieved / len(relevant_doc_ids)
    
    def calculate_precision_at_k(self, retrieved_doc_ids: List[Any],
                                relevant_doc_ids: List[Any], k: int) -> float:
        """Precision@K 계산"""
        if k == 0:
            return 0.0
        
        top_k_retrieved = retrieved_doc_ids[:k]
        relevant_retrieved = len(set(top_k_retrieved) & set(relevant_doc_ids))
        
        return relevant_retrieved / k
    
    def calculate_mrr(self, retrieved_doc_ids: List[Any],
                     relevant_doc_ids: List[Any]) -> float:
        """MRR (Mean Reciprocal Rank) 계산"""
        if not relevant_doc_ids:
            return 0.0
        
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in relevant_doc_ids:
                return 1.0 / rank
        
        return 0.0
    
    def calculate_ndcg_at_k(self, retrieved_doc_ids: List[Any],
                           relevant_doc_ids: List[Any], k: int) -> float:
        """NDCG@K 계산"""
        if not relevant_doc_ids:
            return 0.0
        
        top_k_retrieved = retrieved_doc_ids[:k]
        
        dcg = 0.0
        for i, doc_id in enumerate(top_k_retrieved, start=1):
            if doc_id in relevant_doc_ids:
                dcg += 1.0 / np.log2(i + 1)
        
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_doc_ids), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def save_checkpoint(self, state: Dict[str, Any], checkpoint_name: str = "evaluation_checkpoint.pkl"):
        """체크포인트 저장"""
        if not self.checkpoint_dir:
            return
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"Checkpoint saved to: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_name: str = "evaluation_checkpoint.pkl") -> Optional[Dict[str, Any]]:
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
    
    def evaluate_query(self, query: str, relevant_doc_ids: List[Any],
                      top_k_list: List[int] = [5, 10, 20]) -> Dict[str, Any]:
        """
        단일 쿼리 평가
        
        Args:
            query: 검색 쿼리
            relevant_doc_ids: 관련 문서 ID 리스트
            top_k_list: 평가할 K 값 리스트
        
        Returns:
            평가 결과 딕셔너리
        """
        results = self.vector_store.search(query, top_k=max(top_k_list), enhanced=True)
        
        retrieved_doc_ids = []
        for result in results:
            metadata = result.get('metadata', {})
            doc_id = metadata.get('chunk_id')
            
            if doc_id is not None:
                try:
                    doc_id = int(doc_id)
                except (ValueError, TypeError):
                    pass
            else:
                doc_id = len(retrieved_doc_ids)
            
            retrieved_doc_ids.append(doc_id)
        
        relevant_doc_ids_int = []
        for doc_id in relevant_doc_ids:
            try:
                relevant_doc_ids_int.append(int(doc_id))
            except (ValueError, TypeError):
                relevant_doc_ids_int.append(doc_id)
        
        metrics = {}
        
        for k in top_k_list:
            metrics[f'recall@{k}'] = self.calculate_recall_at_k(retrieved_doc_ids, relevant_doc_ids_int, k)
            metrics[f'precision@{k}'] = self.calculate_precision_at_k(retrieved_doc_ids, relevant_doc_ids_int, k)
            metrics[f'ndcg@{k}'] = self.calculate_ndcg_at_k(retrieved_doc_ids, relevant_doc_ids_int, k)
        
        metrics['mrr'] = self.calculate_mrr(retrieved_doc_ids, relevant_doc_ids_int)
        metrics['num_retrieved'] = len(retrieved_doc_ids)
        metrics['num_relevant'] = len(relevant_doc_ids_int)
        
        return metrics
    
    def evaluate_dataset(self, ground_truth: List[Dict[str, Any]],
                        top_k_list: List[int] = [5, 10, 20],
                        resume_from_checkpoint: bool = True,
                        checkpoint_interval: int = 100) -> Dict[str, Any]:
        """
        전체 데이터셋 평가
        
        Args:
            ground_truth: Ground Truth 데이터셋
            top_k_list: 평가할 K 값 리스트
            resume_from_checkpoint: 체크포인트에서 재개할지 여부
            checkpoint_interval: 체크포인트 저장 간격 (쿼리 수)
        
        Returns:
            평가 결과 딕셔너리
        """
        total_queries = len(ground_truth)
        logger.info("=" * 60)
        logger.info(f"Starting evaluation of {total_queries} queries...")
        logger.info(f"Top-K values: {top_k_list}")
        logger.info(f"Checkpoint interval: {checkpoint_interval} queries")
        logger.info("=" * 60)
        
        all_metrics = []
        start_idx = 0
        
        if resume_from_checkpoint:
            checkpoint = self.load_checkpoint("evaluation_checkpoint.pkl")
            if checkpoint and 'all_metrics' in checkpoint and 'start_idx' in checkpoint:
                logger.info(f"Resuming from checkpoint at index {checkpoint['start_idx']}")
                all_metrics = checkpoint['all_metrics']
                start_idx = checkpoint['start_idx']
                logger.info(f"Loaded {len(all_metrics)} completed evaluations from checkpoint")
        
        evaluation_start = time.time()
        processed_count = 0
        error_count = 0
        last_checkpoint_time = time.time()
        
        for idx, entry in enumerate(tqdm(ground_truth[start_idx:], 
                                         desc="Evaluating queries",
                                         initial=start_idx,
                                         total=total_queries)):
            current_idx = start_idx + idx
            
            if 'query' in entry:
                query = entry['query']
            elif 'query_text' in entry:
                query = entry['query_text']
            else:
                logger.warning(f"Skipping entry {current_idx}: No query field found")
                continue
            
            if 'relevant_doc_ids' in entry:
                relevant_doc_ids = entry['relevant_doc_ids']
            else:
                logger.warning(f"Skipping entry {current_idx}: No relevant_doc_ids field found")
                continue
            
            try:
                metrics = self.evaluate_query(query, relevant_doc_ids, top_k_list)
                metrics['query'] = query
                all_metrics.append(metrics)
                processed_count += 1
                
                if (current_idx + 1) % 10 == 0:
                    elapsed = time.time() - evaluation_start
                    avg_time = elapsed / (current_idx + 1)
                    remaining = avg_time * (total_queries - current_idx - 1)
                    logger.info(
                        f"Progress: {current_idx + 1}/{total_queries} "
                        f"({(current_idx + 1) / total_queries * 100:.1f}%) | "
                        f"Processed: {processed_count} | Errors: {error_count} | "
                        f"Avg time: {avg_time:.2f}s/query | "
                        f"ETA: {remaining / 60:.1f} minutes"
                    )
                
            except Exception as e:
                error_count += 1
                logger.error(f"Failed to evaluate query {current_idx} '{query[:50]}...': {e}")
                continue
            
            if (current_idx + 1) % checkpoint_interval == 0:
                checkpoint_data = {
                    'all_metrics': all_metrics,
                    'start_idx': current_idx + 1,
                    'processed_count': processed_count,
                    'error_count': error_count,
                    'timestamp': datetime.now().isoformat()
                }
                self.save_checkpoint(checkpoint_data, "evaluation_checkpoint.pkl")
                checkpoint_time = time.time() - last_checkpoint_time
                logger.info(f"Checkpoint saved at {current_idx + 1}/{total_queries} (took {checkpoint_time:.2f}s)")
                last_checkpoint_time = time.time()
                gc.collect()
        
        total_time = time.time() - evaluation_start
        logger.info("=" * 60)
        logger.info(f"Evaluation completed in {total_time / 60:.2f} minutes")
        logger.info(f"Total processed: {processed_count}, Errors: {error_count}")
        logger.info(f"Average time per query: {total_time / processed_count if processed_count > 0 else 0:.2f} seconds")
        logger.info("=" * 60)
        
        if not all_metrics:
            logger.error("No metrics collected")
            return {}
        
        logger.info("Aggregating metrics...")
        aggregation_start = time.time()
        
        aggregated = {}
        
        for k in top_k_list:
            recalls = [m[f'recall@{k}'] for m in all_metrics]
            precisions = [m[f'precision@{k}'] for m in all_metrics]
            ndcgs = [m[f'ndcg@{k}'] for m in all_metrics]
            
            aggregated[f'recall@{k}_mean'] = np.mean(recalls)
            aggregated[f'recall@{k}_std'] = np.std(recalls)
            aggregated[f'precision@{k}_mean'] = np.mean(precisions)
            aggregated[f'precision@{k}_std'] = np.std(precisions)
            aggregated[f'ndcg@{k}_mean'] = np.mean(ndcgs)
            aggregated[f'ndcg@{k}_std'] = np.std(ndcgs)
        
        mrrs = [m['mrr'] for m in all_metrics]
        aggregated['mrr_mean'] = np.mean(mrrs)
        aggregated['mrr_std'] = np.std(mrrs)
        
        aggregated['total_queries'] = len(all_metrics)
        aggregated['processed_count'] = processed_count
        aggregated['error_count'] = error_count
        
        aggregation_time = time.time() - aggregation_start
        logger.info(f"Metrics aggregation completed in {aggregation_time:.2f} seconds")
        gc.collect()
        
        return {
            'aggregated_metrics': aggregated,
            'per_query_metrics': all_metrics
        }
    
    def run(self, ground_truth_path: str, top_k_list: List[int] = [5, 10, 20],
            resume_from_checkpoint: bool = True,
            checkpoint_interval: int = 100) -> Dict[str, Any]:
        """
        전체 프로세스 실행
        
        Args:
            ground_truth_path: Ground Truth 파일 경로
            top_k_list: 평가할 K 값 리스트
            resume_from_checkpoint: 체크포인트에서 재개할지 여부
            checkpoint_interval: 체크포인트 저장 간격 (쿼리 수)
        
        Returns:
            평가 결과 딕셔너리
        """
        logger.info(f"Loading ground truth from: {ground_truth_path}")
        load_start = time.time()
        
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'ground_truth' in data:
            ground_truth = data['ground_truth']
        elif isinstance(data, list):
            ground_truth = data
        else:
            raise ValueError("Invalid ground truth format")
        
        load_time = time.time() - load_start
        logger.info(f"Loaded {len(ground_truth)} ground truth entries in {load_time:.2f} seconds")
        gc.collect()
        
        evaluation_results = self.evaluate_dataset(
            ground_truth, 
            top_k_list,
            resume_from_checkpoint=resume_from_checkpoint,
            checkpoint_interval=checkpoint_interval
        )
        
        return evaluation_results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="RAG 검색 평가")
    parser.add_argument(
        "--ground-truth-path",
        type=str,
        required=True,
        help="Ground Truth 파일 경로"
    )
    parser.add_argument(
        "--vector-store-path",
        type=str,
        default="data/embeddings/ml_enhanced_ko_sroberta",
        help="벡터 스토어 경로"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/evaluation/evaluation_reports/rag_evaluation_report.json",
        help="출력 파일 경로"
    )
    parser.add_argument(
        "--top-k-list",
        type=str,
        default="5,10,20",
        help="평가할 K 값 리스트 (쉼표로 구분)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="jhgan/ko-sroberta-multitask",
        help="임베딩 모델명"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="체크포인트 저장 디렉토리 경로"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="체크포인트 저장 간격 (쿼리 수)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="체크포인트에서 재개하지 않음"
    )
    
    args = parser.parse_args()
    
    top_k_list = [int(k.strip()) for k in args.top_k_list.split(',')]
    
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        evaluator = RAGSearchEvaluator(
            vector_store_path=args.vector_store_path,
            model_name=args.model_name,
            checkpoint_dir=args.checkpoint_dir
        )
        
        evaluation_results = evaluator.run(
            ground_truth_path=args.ground_truth_path,
            top_k_list=top_k_list,
            resume_from_checkpoint=not args.no_resume,
            checkpoint_interval=args.checkpoint_interval
        )
        
        output_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "ground_truth_path": args.ground_truth_path,
                "vector_store_path": args.vector_store_path,
                "model_name": args.model_name,
                "top_k_list": top_k_list
            },
            "evaluation_results": evaluation_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation report saved to: {output_path}")
        
        if 'aggregated_metrics' in evaluation_results:
            metrics = evaluation_results['aggregated_metrics']
            logger.info("=" * 60)
            logger.info("Evaluation Results:")
            logger.info("=" * 60)
            for k in top_k_list:
                logger.info(f"Recall@{k}: {metrics.get(f'recall@{k}_mean', 0):.4f} ± {metrics.get(f'recall@{k}_std', 0):.4f}")
                logger.info(f"Precision@{k}: {metrics.get(f'precision@{k}_mean', 0):.4f} ± {metrics.get(f'precision@{k}_std', 0):.4f}")
                logger.info(f"NDCG@{k}: {metrics.get(f'ndcg@{k}_mean', 0):.4f} ± {metrics.get(f'ndcg@{k}_std', 0):.4f}")
            logger.info(f"MRR: {metrics.get('mrr_mean', 0):.4f} ± {metrics.get('mrr_std', 0):.4f}")
            logger.info(f"Total queries: {metrics.get('total_queries', 0)}")
            logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to evaluate: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

