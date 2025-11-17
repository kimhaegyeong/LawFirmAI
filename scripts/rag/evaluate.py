"""
MLflow를 사용한 RAG 평가

기존 evaluate_rag_search.py의 RAGSearchEvaluator를 활용하여
평가 결과를 MLflow에 저장합니다.
"""
import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

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


class MLflowRAGEvaluator:
    """MLflow 통합 RAG 평가기"""
    
    def __init__(
        self,
        vector_store_path: str,
        model_name: str = "jhgan/ko-sroberta-multitask",
        checkpoint_dir: Optional[str] = None
    ):
        """
        초기화
        
        Args:
            vector_store_path: 벡터 스토어 경로
            model_name: 임베딩 모델명
            checkpoint_dir: 체크포인트 저장 디렉토리 경로
        """
        self.evaluator = RAGSearchEvaluator(
            vector_store_path=vector_store_path,
            model_name=model_name,
            checkpoint_dir=checkpoint_dir
        )
        self.vector_store_path = vector_store_path
        self.model_name = model_name
    
    def evaluate_with_mlflow(
        self,
        ground_truth_path: str,
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
        experiment_name: str = "rag_evaluation",
        top_k_list: list = [5, 10, 20],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        RAG 평가 및 MLflow에 결과 저장
        
        Args:
            ground_truth_path: Ground Truth 파일 경로
            run_id: MLflow run ID (None이면 새로 생성)
            run_name: MLflow run 이름
            experiment_name: MLflow 실험 이름
            top_k_list: 평가할 K 값 리스트
            output_path: 평가 결과 JSON 파일 저장 경로 (선택)
        
        Returns:
            Dict: 평가 결과
        """
        logger.info(f"Evaluating RAG search with MLflow...")
        logger.info(f"Ground truth: {ground_truth_path}")
        logger.info(f"Vector store: {self.vector_store_path}")
        
        evaluation_results = self.evaluator.run(
            ground_truth_path=ground_truth_path,
            top_k_list=top_k_list
        )
        
        try:
            import mlflow
            
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run(run_id=run_id, run_name=run_name) as run:
                actual_run_id = run.info.run_id
                
                mlflow.set_tags({
                    "evaluation_type": "rag_search",
                    "model_name": self.model_name,
                    "vector_store_path": str(self.vector_store_path)
                })
                
                mlflow.log_params({
                    "model_name": self.model_name,
                    "vector_store_path": str(self.vector_store_path),
                    "ground_truth_path": ground_truth_path
                })
                
                metrics = evaluation_results.get("aggregated_metrics", {})
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)
                
                for k in top_k_list:
                    for metric_type in ["recall", "precision", "ndcg"]:
                        metric_key = f"{metric_type}@{k}_mean"
                        if metric_key in metrics:
                            mlflow.log_metric(metric_key, metrics[metric_key])
                
                mlflow.log_dict(evaluation_results, "evaluation_results.json")
                
                if output_path:
                    mlflow.log_artifact(output_path, "evaluation_results")
                
                logger.info(f"Evaluation results saved to MLflow: {actual_run_id}")
                
                evaluation_results["mlflow_run_id"] = actual_run_id
                evaluation_results["mlflow_tracking_uri"] = mlflow.get_tracking_uri()
                
        except ImportError:
            logger.warning("MLflow not available. Skipping MLflow logging.")
        except Exception as e:
            logger.error(f"Failed to save to MLflow: {e}", exc_info=True)
        
        return evaluation_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG search and save to MLflow")
    parser.add_argument("--ground-truth-path", required=True, help="Ground truth JSON file path")
    parser.add_argument("--vector-store-path", required=True, help="Vector store path")
    parser.add_argument("--model-name", default="jhgan/ko-sroberta-multitask", help="Embedding model name")
    parser.add_argument("--checkpoint-dir", help="Checkpoint directory")
    parser.add_argument("--run-id", help="MLflow run ID (if resuming)")
    parser.add_argument("--run-name", help="MLflow run name")
    parser.add_argument("--experiment-name", default="rag_evaluation", help="MLflow experiment name")
    parser.add_argument("--top-k-list", default="5,10,20", help="Comma-separated list of K values")
    parser.add_argument("--output-path", help="Output JSON file path")
    parser.add_argument("--use-mlflow", action="store_true", default=True, help="Use MLflow")
    parser.add_argument("--no-mlflow", dest="use_mlflow", action="store_false", help="Don't use MLflow")
    
    args = parser.parse_args()
    
    top_k_list = [int(k.strip()) for k in args.top_k_list.split(",")]
    
    evaluator = MLflowRAGEvaluator(
        vector_store_path=args.vector_store_path,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir
    )
    
    if args.use_mlflow:
        results = evaluator.evaluate_with_mlflow(
            ground_truth_path=args.ground_truth_path,
            run_id=args.run_id,
            run_name=args.run_name,
            experiment_name=args.experiment_name,
            top_k_list=top_k_list,
            output_path=args.output_path
        )
    else:
        results = evaluator.evaluator.run(
            ground_truth_path=args.ground_truth_path,
            top_k_list=top_k_list
        )
    
    logger.info("Evaluation completed")
    logger.info(f"Metrics: {results.get('aggregated_metrics', {})}")
    
    if args.output_path:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()

