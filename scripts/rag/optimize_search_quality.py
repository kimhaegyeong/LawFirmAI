"""
MLflow를 활용한 RAG 검색 품질 최적화

다양한 검색 파라미터 조합을 실험하고 MLflow로 추적하여 최적의 설정을 찾습니다.
"""
import argparse
import logging
import json
import sys
import itertools
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts" / "ml_training" / "evaluation"))
sys.path.insert(0, str(project_root / "scripts" / "rag"))

from evaluate_rag_search import RAGSearchEvaluator
from mlflow_manager import MLflowFAISSManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with: pip install mlflow")


class RAGSearchQualityOptimizer:
    """RAG 검색 품질 최적화 클래스"""
    
    def __init__(
        self,
        vector_store_path: str,
        model_name: str = "jhgan/ko-sroberta-multitask",
        experiment_name: str = "rag_search_optimization"
    ):
        """
        초기화
        
        Args:
            vector_store_path: 벡터 스토어 경로
            model_name: 임베딩 모델명
            experiment_name: MLflow 실험 이름
        """
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.evaluator = None
        
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(experiment_name)
    
    def _initialize_evaluator(self):
        """평가기 초기화 (지연 로딩)"""
        if self.evaluator is None:
            self.evaluator = RAGSearchEvaluator(
                vector_store_path=self.vector_store_path,
                model_name=self.model_name
            )
    
    def define_parameter_space(self) -> Dict[str, List[Any]]:
        """
        튜닝할 파라미터 공간 정의
        
        Returns:
            Dict: 파라미터 이름과 가능한 값들의 리스트
        """
        return {
            'top_k': [5, 10, 15, 20, 30, 50],
            'similarity_threshold': [0.3, 0.5, 0.7, 0.8, 0.9],
            'use_reranking': [True, False],
            'rerank_top_n': [10, 20, 30] if True else [None],
            'query_enhancement': [True, False],
            'hybrid_search_ratio': [0.0, 0.3, 0.5, 0.7, 1.0]
        }
    
    def generate_parameter_combinations(
        self,
        parameter_space: Dict[str, List[Any]],
        max_combinations: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        파라미터 조합 생성
        
        Args:
            parameter_space: 파라미터 공간
            max_combinations: 최대 조합 수 (None이면 모든 조합)
        
        Returns:
            List[Dict]: 파라미터 조합 리스트
        """
        keys = list(parameter_space.keys())
        values = [parameter_space[key] for key in keys]
        
        all_combinations = list(itertools.product(*values))
        
        combinations = []
        for combo in all_combinations:
            params = dict(zip(keys, combo))
            
            if params['use_reranking'] is False:
                params['rerank_top_n'] = None
            
            if params['hybrid_search_ratio'] == 0.0:
                params['use_keyword_search'] = False
            else:
                params['use_keyword_search'] = True
            
            combinations.append(params)
        
        if max_combinations and len(combinations) > max_combinations:
            import random
            random.seed(42)
            combinations = random.sample(combinations, max_combinations)
        
        return combinations
    
    def evaluate_parameters(
        self,
        ground_truth_path: str,
        parameters: Dict[str, Any],
        top_k_list: List[int] = [5, 10, 20],
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        특정 파라미터 조합으로 평가
        
        Args:
            ground_truth_path: Ground Truth 파일 경로
            parameters: 검색 파라미터
            top_k_list: 평가할 K 값 리스트
            sample_size: 샘플 크기 (None이면 전체)
        
        Returns:
            Dict: 평가 결과
        """
        self._initialize_evaluator()
        
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        if sample_size and len(ground_truth) > sample_size:
            import random
            random.seed(42)
            ground_truth = random.sample(ground_truth, sample_size)
        
        logger.info(f"Evaluating with parameters: {parameters}")
        logger.info(f"Total queries: {len(ground_truth)}")
        
        all_metrics = []
        for entry in ground_truth:
            if 'query' not in entry and 'query_text' not in entry:
                continue
            if 'relevant_doc_ids' not in entry:
                continue
            
            query = entry.get('query') or entry.get('query_text')
            relevant_doc_ids = entry['relevant_doc_ids']
            
            try:
                top_k = parameters.get('top_k', 20)
                search_results = self.evaluator.vector_store.search(
                    query,
                    top_k=top_k,
                    enhanced=parameters.get('query_enhancement', False)
                )
                
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
                
                relevant_doc_ids_int = []
                for doc_id in relevant_doc_ids:
                    try:
                        relevant_doc_ids_int.append(int(doc_id))
                    except (ValueError, TypeError):
                        relevant_doc_ids_int.append(doc_id)
                
                metrics = {}
                for k in top_k_list:
                    if k <= top_k:
                        metrics[f'recall@{k}'] = self.evaluator.calculate_recall_at_k(
                            retrieved_doc_ids, relevant_doc_ids_int, k
                        )
                        metrics[f'precision@{k}'] = self.evaluator.calculate_precision_at_k(
                            retrieved_doc_ids, relevant_doc_ids_int, k
                        )
                        metrics[f'ndcg@{k}'] = self.evaluator.calculate_ndcg_at_k(
                            retrieved_doc_ids, relevant_doc_ids_int, k
                        )
                
                metrics['mrr'] = self.evaluator.calculate_mrr(
                    retrieved_doc_ids, relevant_doc_ids_int
                )
                
                all_metrics.append(metrics)
                
            except Exception as e:
                logger.warning(f"Error evaluating query: {e}")
                continue
        
        if not all_metrics:
            return {}
        
        aggregated = {}
        for k in top_k_list:
            recall_values = [m.get(f'recall@{k}', 0.0) for m in all_metrics]
            precision_values = [m.get(f'precision@{k}', 0.0) for m in all_metrics]
            ndcg_values = [m.get(f'ndcg@{k}', 0.0) for m in all_metrics]
            
            if recall_values:
                aggregated[f'recall@{k}_mean'] = sum(recall_values) / len(recall_values)
                aggregated[f'recall@{k}_std'] = (
                    sum((x - aggregated[f'recall@{k}_mean'])**2 for x in recall_values) / len(recall_values)
                ) ** 0.5
            
            if precision_values:
                aggregated[f'precision@{k}_mean'] = sum(precision_values) / len(precision_values)
                aggregated[f'precision@{k}_std'] = (
                    sum((x - aggregated[f'precision@{k}_mean'])**2 for x in precision_values) / len(precision_values)
                ) ** 0.5
            
            if ndcg_values:
                aggregated[f'ndcg@{k}_mean'] = sum(ndcg_values) / len(ndcg_values)
                aggregated[f'ndcg@{k}_std'] = (
                    sum((x - aggregated[f'ndcg@{k}_mean'])**2 for x in ndcg_values) / len(ndcg_values)
                ) ** 0.5
        
        mrr_values = [m.get('mrr', 0.0) for m in all_metrics]
        if mrr_values:
            aggregated['mrr_mean'] = sum(mrr_values) / len(mrr_values)
            aggregated['mrr_std'] = (
                sum((x - aggregated['mrr_mean'])**2 for x in mrr_values) / len(mrr_values)
            ) ** 0.5
        
        aggregated['total_queries'] = len(all_metrics)
        
        return aggregated
    
    def optimize_with_mlflow(
        self,
        ground_truth_path: str,
        parameter_space: Optional[Dict[str, List[Any]]] = None,
        max_combinations: Optional[int] = 50,
        top_k_list: List[int] = [5, 10, 20],
        sample_size: Optional[int] = None,
        primary_metric: str = "ndcg@10"
    ) -> Dict[str, Any]:
        """
        MLflow를 사용한 파라미터 최적화
        
        Args:
            ground_truth_path: Ground Truth 파일 경로
            parameter_space: 파라미터 공간 (None이면 기본값 사용)
            max_combinations: 최대 조합 수
            top_k_list: 평가할 K 값 리스트
            sample_size: 샘플 크기
            primary_metric: 최적화할 주요 메트릭
        
        Returns:
            Dict: 최적 파라미터 및 결과
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required for optimization")
        
        if parameter_space is None:
            parameter_space = self.define_parameter_space()
        
        combinations = self.generate_parameter_combinations(
            parameter_space,
            max_combinations=max_combinations
        )
        
        logger.info(f"Testing {len(combinations)} parameter combinations...")
        
        best_result = None
        best_score = -1.0
        all_results = []
        
        for idx, params in enumerate(combinations):
            run_name = f"experiment_{idx+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags({
                    "optimization_type": "rag_search_quality",
                    "vector_store_path": str(self.vector_store_path),
                    "model_name": self.model_name
                })
                
                mlflow.log_params(params)
                
                start_time = time.time()
                results = self.evaluate_parameters(
                    ground_truth_path=ground_truth_path,
                    parameters=params,
                    top_k_list=top_k_list,
                    sample_size=sample_size
                )
                elapsed_time = time.time() - start_time
                
                if results:
                    for metric_name, metric_value in results.items():
                        if isinstance(metric_value, (int, float)):
                            mlflow_metric_name = metric_name.replace('@', '_at_')
                            mlflow.log_metric(mlflow_metric_name, metric_value)
                    
                    mlflow.log_metric("evaluation_time_seconds", elapsed_time)
                    
                    if not primary_metric.endswith('_mean'):
                        primary_metric_key = f"{primary_metric}_mean"
                    else:
                        primary_metric_key = primary_metric
                    primary_score = results.get(primary_metric_key, 0.0)
                    mlflow.log_metric("primary_score", primary_score)
                    
                    result_entry = {
                        'parameters': params,
                        'results': results,
                        'primary_score': primary_score,
                        'run_id': mlflow.active_run().info.run_id
                    }
                    all_results.append(result_entry)
                    
                    if primary_score > best_score:
                        best_score = primary_score
                        best_result = result_entry
                    
                    logger.info(
                        f"[{idx+1}/{len(combinations)}] "
                        f"Primary Score: {primary_score:.4f} | "
                        f"Time: {elapsed_time:.2f}s"
                    )
        
        if best_result:
            logger.info(f"\nBest parameters found:")
            logger.info(f"Primary Score ({primary_metric}): {best_score:.4f}")
            logger.info(f"Parameters: {best_result['parameters']}")
            logger.info(f"Run ID: {best_result['run_id']}")
            
            with mlflow.start_run(run_name="best_parameters") as best_run:
                mlflow.set_tags({
                    "optimization_type": "rag_search_quality",
                    "status": "best_parameters"
                })
                mlflow.log_params(best_result['parameters'])
                for metric_name, metric_value in best_result['results'].items():
                    if isinstance(metric_value, (int, float)):
                        mlflow_metric_name = metric_name.replace('@', '_at_')
                        mlflow.log_metric(mlflow_metric_name, metric_value)
                mlflow.log_metric("primary_score", best_score)
        
        return {
            'best_result': best_result,
            'all_results': all_results,
            'total_experiments': len(combinations),
            'primary_metric': primary_metric
        }


def main():
    parser = argparse.ArgumentParser(description="Optimize RAG search quality with MLflow")
    parser.add_argument("--ground-truth-path", required=True, help="Ground truth JSON file path")
    parser.add_argument("--vector-store-path", required=True, help="Vector store path")
    parser.add_argument("--model-name", default="jhgan/ko-sroberta-multitask", help="Embedding model name")
    parser.add_argument("--experiment-name", default="rag_search_optimization", help="MLflow experiment name")
    parser.add_argument("--max-combinations", type=int, default=50, help="Maximum parameter combinations to test")
    parser.add_argument("--top-k-list", default="5,10,20", help="Comma-separated list of K values")
    parser.add_argument("--sample-size", type=int, help="Sample size for faster evaluation")
    parser.add_argument("--primary-metric", default="ndcg@10", help="Primary metric to optimize")
    parser.add_argument("--output-path", help="Output JSON file path")
    
    args = parser.parse_args()
    
    top_k_list = [int(k.strip()) for k in args.top_k_list.split(",")]
    
    optimizer = RAGSearchQualityOptimizer(
        vector_store_path=args.vector_store_path,
        model_name=args.model_name,
        experiment_name=args.experiment_name
    )
    
    results = optimizer.optimize_with_mlflow(
        ground_truth_path=args.ground_truth_path,
        max_combinations=args.max_combinations,
        top_k_list=top_k_list,
        sample_size=args.sample_size,
        primary_metric=args.primary_metric
    )
    
    if args.output_path:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {args.output_path}")
    
    logger.info("Optimization completed!")


if __name__ == "__main__":
    main()

