#!/usr/bin/env python3
"""
RAG 검색 파라미터 튜닝 스크립트
다양한 top_k 값으로 검색 성능 비교
"""

import json
import sys
import os
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from lawfirm_langgraph.core.data.vector_store import LegalVectorStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SearchParameterTuner:
    """검색 파라미터 튜닝 클래스"""
    
    def __init__(self, vector_store_path: str, model_name: str = "jhgan/ko-sroberta-multitask"):
        logger.info("Initializing Search Parameter Tuner...")
        self.vector_store = LegalVectorStore(model_name=model_name)
        
        vector_store_path = Path(vector_store_path)
        if vector_store_path.is_dir():
            index_files = [
                vector_store_path / "ml_enhanced_faiss_index.faiss",
                vector_store_path / "index.faiss",
                vector_store_path / "faiss_index_jhgan_ko-sroberta-multitask.index"
            ]
            
            index_path = None
            for idx_file in index_files:
                if idx_file.exists():
                    index_path = idx_file.parent / idx_file.stem
                    break
            
            if index_path is None:
                raise ValueError(f"Could not find FAISS index in {vector_store_path}")
        else:
            index_path = vector_store_path
        
        logger.info(f"Loading vector store from: {index_path}")
        load_start = time.time()
        self.vector_store.load_index(str(index_path))
        load_time = time.time() - load_start
        logger.info(f"Vector store loaded in {load_time:.2f} seconds")
    
    def calculate_recall_at_k(self, retrieved: List[Any], relevant: List[Any], k: int) -> float:
        """Recall@K 계산"""
        if not relevant:
            return 0.0
        
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        if not relevant_set:
            return 0.0
        
        intersection = retrieved_k & relevant_set
        return len(intersection) / len(relevant_set)
    
    def calculate_precision_at_k(self, retrieved: List[Any], relevant: List[Any], k: int) -> float:
        """Precision@K 계산"""
        if k == 0:
            return 0.0
        
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        if not retrieved_k:
            return 0.0
        
        intersection = retrieved_k & relevant_set
        return len(intersection) / k
    
    def calculate_ndcg_at_k(self, retrieved: List[Any], relevant: List[Any], k: int) -> float:
        """NDCG@K 계산"""
        if not relevant:
            return 0.0
        
        relevant_set = set(relevant)
        if not relevant_set:
            return 0.0
        
        dcg = 0.0
        retrieved_k = retrieved[:k]
        
        for i, doc_id in enumerate(retrieved_k):
            if doc_id in relevant_set:
                dcg += 1.0 / (i + 2)
        
        idcg = sum(1.0 / (i + 2) for i in range(min(len(relevant_set), k)))
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_mrr(self, retrieved: List[Any], relevant: List[Any]) -> float:
        """MRR 계산"""
        relevant_set = set(relevant)
        if not relevant_set:
            return 0.0
        
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def evaluate_with_top_k(self, ground_truth: List[Dict[str, Any]], 
                           top_k_values: List[int]) -> Dict[str, Any]:
        """다양한 top_k 값으로 평가"""
        logger.info(f"Evaluating with top_k values: {top_k_values}")
        logger.info(f"Total queries: {len(ground_truth)}")
        
        results = {k: {'metrics': [], 'total_queries': 0, 'processed': 0} for k in top_k_values}
        
        max_k = max(top_k_values)
        
        for idx, entry in enumerate(ground_truth):
            if 'query' in entry:
                query = entry['query']
            elif 'query_text' in entry:
                query = entry['query_text']
            else:
                continue
            
            if 'relevant_doc_ids' not in entry:
                continue
            
            relevant_doc_ids = entry['relevant_doc_ids']
            relevant_doc_ids_int = []
            for doc_id in relevant_doc_ids:
                try:
                    relevant_doc_ids_int.append(int(doc_id))
                except (ValueError, TypeError):
                    relevant_doc_ids_int.append(doc_id)
            
            try:
                search_results = self.vector_store.search(query, top_k=max_k, enhanced=True)
                
                retrieved_doc_ids = []
                for result in search_results:
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
                
                for k in top_k_values:
                    results[k]['total_queries'] += 1
                    results[k]['processed'] += 1
                    
                    metrics = {
                        'recall': self.calculate_recall_at_k(retrieved_doc_ids, relevant_doc_ids_int, k),
                        'precision': self.calculate_precision_at_k(retrieved_doc_ids, relevant_doc_ids_int, k),
                        'ndcg': self.calculate_ndcg_at_k(retrieved_doc_ids, relevant_doc_ids_int, k),
                        'mrr': self.calculate_mrr(retrieved_doc_ids, relevant_doc_ids_int)
                    }
                    results[k]['metrics'].append(metrics)
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(ground_truth)} queries")
            
            except Exception as e:
                logger.warning(f"Error processing query {idx}: {e}")
                continue
        
        aggregated = {}
        for k in top_k_values:
            metrics_list = results[k]['metrics']
            if not metrics_list:
                continue
            
            aggregated[k] = {
                'recall_mean': sum(m['recall'] for m in metrics_list) / len(metrics_list),
                'recall_std': (sum((m['recall'] - sum(m['recall'] for m in metrics_list) / len(metrics_list))**2 for m in metrics_list) / len(metrics_list))**0.5,
                'precision_mean': sum(m['precision'] for m in metrics_list) / len(metrics_list),
                'precision_std': (sum((m['precision'] - sum(m['precision'] for m in metrics_list) / len(metrics_list))**2 for m in metrics_list) / len(metrics_list))**0.5,
                'ndcg_mean': sum(m['ndcg'] for m in metrics_list) / len(metrics_list),
                'ndcg_std': (sum((m['ndcg'] - sum(m['ndcg'] for m in metrics_list) / len(metrics_list))**2 for m in metrics_list) / len(metrics_list))**0.5,
                'mrr_mean': sum(m['mrr'] for m in metrics_list) / len(metrics_list),
                'mrr_std': (sum((m['mrr'] - sum(m['mrr'] for m in metrics_list) / len(metrics_list))**2 for m in metrics_list) / len(metrics_list))**0.5,
                'total_queries': results[k]['total_queries'],
                'processed': results[k]['processed']
            }
        
        return aggregated
    
    def generate_tuning_report(self, results: Dict[int, Dict[str, float]], 
                              output_path: Optional[str] = None) -> str:
        """튜닝 결과 리포트 생성"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("검색 파라미터 튜닝 결과 리포트")
        report_lines.append("=" * 80)
        report_lines.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        report_lines.append("Top-K 값별 성능 비교")
        report_lines.append("-" * 80)
        report_lines.append(f"{'K':<6} {'Recall':<20} {'Precision':<20} {'NDCG':<20} {'MRR':<20}")
        report_lines.append("-" * 80)
        
        for k in sorted(results.keys()):
            r = results[k]
            report_lines.append(
                f"{k:<6} "
                f"{r['recall_mean']:.6f}±{r['recall_std']:.6f}  "
                f"{r['precision_mean']:.6f}±{r['precision_std']:.6f}  "
                f"{r['ndcg_mean']:.6f}±{r['ndcg_std']:.6f}  "
                f"{r['mrr_mean']:.6f}±{r['mrr_std']:.6f}"
            )
        
        report_lines.append("")
        report_lines.append("최적 파라미터 추천")
        report_lines.append("-" * 80)
        
        best_recall_k = max(results.keys(), key=lambda k: results[k]['recall_mean'])
        best_precision_k = max(results.keys(), key=lambda k: results[k]['precision_mean'])
        best_ndcg_k = max(results.keys(), key=lambda k: results[k]['ndcg_mean'])
        best_mrr_k = max(results.keys(), key=lambda k: results[k]['mrr_mean'])
        
        report_lines.append(f"최고 Recall: K={best_recall_k} ({results[best_recall_k]['recall_mean']:.6f})")
        report_lines.append(f"최고 Precision: K={best_precision_k} ({results[best_precision_k]['precision_mean']:.6f})")
        report_lines.append(f"최고 NDCG: K={best_ndcg_k} ({results[best_ndcg_k]['ndcg_mean']:.6f})")
        report_lines.append(f"최고 MRR: K={best_mrr_k} ({results[best_mrr_k]['mrr_mean']:.6f})")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Tuning report saved to: {output_file}")
        
        return report_text


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG 검색 파라미터 튜닝')
    parser.add_argument(
        '--ground-truth-path',
        type=str,
        default='data/evaluation/rag_ground_truth_combined_test.json',
        help='Ground Truth 파일 경로'
    )
    parser.add_argument(
        '--vector-store-path',
        type=str,
        default='data/vector_store/v2.0.0-dynamic-dynamic-ivfpq',
        help='벡터 스토어 경로'
    )
    parser.add_argument(
        '--top-k-values',
        type=int,
        nargs='+',
        default=[3, 5, 10, 20, 30, 50],
        help='테스트할 top_k 값 리스트'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='data/evaluation/evaluation_reports/search_parameter_tuning_report.txt',
        help='출력 리포트 경로'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default='data/evaluation/evaluation_reports/search_parameter_tuning_results.json',
        help='출력 JSON 경로'
    )
    
    args = parser.parse_args()
    
    logger.info("Loading ground truth data...")
    with open(args.ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    logger.info(f"Loaded {len(ground_truth)} queries")
    
    tuner = SearchParameterTuner(args.vector_store_path)
    
    logger.info("Starting parameter tuning...")
    start_time = time.time()
    results = tuner.evaluate_with_top_k(ground_truth, args.top_k_values)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Tuning completed in {elapsed_time / 60:.2f} minutes")
    
    report_text = tuner.generate_tuning_report(results, output_path=args.output_path)
    
    output_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'ground_truth_path': args.ground_truth_path,
            'vector_store_path': args.vector_store_path,
            'top_k_values': args.top_k_values,
            'elapsed_time_seconds': elapsed_time
        },
        'results': results
    }
    
    output_file = Path(args.output_json)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Results JSON saved to: {output_file}")
    
    print("\n" + report_text)


if __name__ == "__main__":
    main()

