"""
자동화 최적화 파이프라인

평가 -> 최적화 -> 적용 -> 프로덕션 빌드의 전체 워크플로우를 자동화합니다.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts" / "utils"))
sys.path.insert(0, str(project_root / "scripts" / "rag"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizationPipeline:
    """자동화 최적화 파이프라인"""
    
    def __init__(
        self,
        vector_store_path: str,
        ground_truth_path: str,
        db_path: str = "data/lawfirm_v2.db",
        model_name: str = "jhgan/ko-sroberta-multitask",
        embedding_version_id: Optional[int] = None,
        chunking_strategy: Optional[str] = None
    ):
        self.vector_store_path = vector_store_path
        self.ground_truth_path = ground_truth_path
        self.db_path = db_path
        self.model_name = model_name
        self.embedding_version_id = embedding_version_id
        self.chunking_strategy = chunking_strategy
        
    def get_active_embedding_version(self) -> Dict[str, Any]:
        """활성 임베딩 버전 정보 가져오기"""
        try:
            from embedding_version_manager import EmbeddingVersionManager
            
            evm = EmbeddingVersionManager(self.db_path)
            
            if self.chunking_strategy:
                active_version = evm.get_active_version(self.chunking_strategy)
            else:
                active_version = evm.get_active_version("dynamic")
            
            if not active_version:
                raise ValueError("활성 임베딩 버전을 찾을 수 없습니다")
            
            return active_version
        except Exception as e:
            logger.error(f"임베딩 버전 정보 가져오기 실패: {e}")
            raise
    
    def run_evaluation(
        self,
        output_path: str,
        top_k_list: list = [5, 10, 20],
        checkpoint_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """RAG 평가 실행"""
        logger.info("=" * 80)
        logger.info("1단계: RAG 평가 실행")
        logger.info("=" * 80)
        
        try:
            from evaluate import MLflowRAGEvaluator
            
            evaluator = MLflowRAGEvaluator(
                vector_store_path=self.vector_store_path,
                model_name=self.model_name,
                checkpoint_dir=checkpoint_dir
            )
            
            results = evaluator.evaluate_with_mlflow(
                ground_truth_path=self.ground_truth_path,
                run_name=f"pipeline-evaluation-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                experiment_name="rag_evaluation",
                top_k_list=top_k_list,
                output_path=output_path
            )
            
            logger.info(f"평가 완료: {results.get('aggregated_metrics', {}).get('total_queries', 0)}개 쿼리")
            return results
            
        except Exception as e:
            logger.error(f"평가 실패: {e}", exc_info=True)
            raise
    
    def run_optimization(
        self,
        max_combinations: int = 20,
        sample_size: Optional[int] = None,
        primary_metric: str = "ndcg@10_mean",
        output_path: str = "data/evaluation/evaluation_reports/pipeline_optimization_results.json"
    ) -> Dict[str, Any]:
        """검색 품질 최적화 실행"""
        logger.info("=" * 80)
        logger.info("2단계: 검색 품질 최적화")
        logger.info("=" * 80)
        
        try:
            from optimize_search_quality import RAGSearchQualityOptimizer
            
            optimizer = RAGSearchQualityOptimizer(
                vector_store_path=self.vector_store_path,
                model_name=self.model_name,
                experiment_name="rag_search_optimization"
            )
            
            results = optimizer.optimize_with_mlflow(
                ground_truth_path=self.ground_truth_path,
                max_combinations=max_combinations,
                top_k_list=[5, 10, 20],
                sample_size=sample_size,
                primary_metric=primary_metric
            )
            
            if results.get("best_result"):
                best_score = results["best_result"].get("primary_score", 0.0)
                logger.info(f"최적 파라미터 발견: Primary Score = {best_score:.6f}")
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f"최적화 결과 저장: {output_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"최적화 실패: {e}", exc_info=True)
            raise
    
    def apply_optimized_params(
        self,
        optimization_results_path: str,
        output_path: str = "data/ml_config/optimized_search_params.json"
    ) -> Dict[str, Any]:
        """최적 파라미터 적용"""
        logger.info("=" * 80)
        logger.info("3단계: 최적 파라미터 적용")
        logger.info("=" * 80)
        
        try:
            from apply_optimized_params import load_optimization_results, create_search_config, save_config
            
            results = load_optimization_results(optimization_results_path)
            best_result = results["best_result"]
            best_params = best_result["parameters"]
            
            metadata = {
                "optimization_run_id": best_result.get("run_id"),
                "primary_score": best_result.get("primary_score"),
                "optimization_date": datetime.now().isoformat(),
                "total_experiments": results.get("total_experiments", 0),
                "primary_metric": results.get("primary_metric", "ndcg@10_mean")
            }
            
            config = create_search_config(best_params, metadata)
            save_config(config, output_path, backup=True)
            
            logger.info(f"최적 파라미터 저장 완료: {output_path}")
            return config
            
        except Exception as e:
            logger.error(f"파라미터 적용 실패: {e}", exc_info=True)
            raise
    
    def build_production_index(
        self,
        version_name: Optional[str] = None,
        tag_production: bool = True
    ) -> str:
        """프로덕션 인덱스 빌드"""
        logger.info("=" * 80)
        logger.info("4단계: 프로덕션 인덱스 빌드")
        logger.info("=" * 80)
        
        try:
            from build_index import build_and_save_index
            from mlflow_manager import MLflowFAISSManager
            
            if not self.embedding_version_id or not self.chunking_strategy:
                version_info = self.get_active_embedding_version()
                embedding_version_id = version_info['id']
                chunking_strategy = version_info['chunking_strategy']
            else:
                embedding_version_id = self.embedding_version_id
                chunking_strategy = self.chunking_strategy
            
            if version_name is None:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                version_name = f"production-{timestamp}"
            
            success = build_and_save_index(
                version_name=version_name,
                embedding_version_id=embedding_version_id,
                chunking_strategy=chunking_strategy,
                db_path=self.db_path,
                use_mlflow=True,
                local_backup=True,
                model_name=self.model_name
            )
            
            if not success:
                raise RuntimeError("프로덕션 인덱스 빌드 실패")
            
            if tag_production:
                import mlflow
                mlflow_manager = MLflowFAISSManager()
                runs = mlflow_manager.list_runs()
                if runs:
                    latest_run = runs[0]
                    run_id = latest_run.get("run_id")
                    if run_id:
                        try:
                            mlflow.end_run()
                            client = mlflow.tracking.MlflowClient()
                            client.set_tag(run_id, "environment", "production")
                            client.set_tag(run_id, "status", "production_ready")
                            client.set_tag(run_id, "optimized", "true")
                            client.set_tag(run_id, "production_date", datetime.now().isoformat())
                            logger.info(f"프로덕션 태그 추가 완료: {run_id}")
                        except Exception as e:
                            logger.warning(f"프로덕션 태그 추가 실패: {e}")
            
            logger.info(f"프로덕션 인덱스 빌드 완료: {version_name}")
            return version_name
            
        except Exception as e:
            logger.error(f"프로덕션 인덱스 빌드 실패: {e}", exc_info=True)
            raise
    
    def run_full_pipeline(
        self,
        skip_evaluation: bool = False,
        skip_optimization: bool = False,
        skip_apply: bool = False,
        skip_build: bool = False,
        optimization_max_combinations: int = 20,
        optimization_sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        logger.info("=" * 80)
        logger.info("자동화 최적화 파이프라인 시작")
        logger.info("=" * 80)
        
        pipeline_results = {
            "start_time": datetime.now().isoformat(),
            "steps_completed": [],
            "errors": []
        }
        
        try:
            if not skip_evaluation:
                eval_results = self.run_evaluation(
                    output_path="data/evaluation/evaluation_reports/pipeline_evaluation.json"
                )
                pipeline_results["evaluation"] = {
                    "total_queries": eval_results.get("aggregated_metrics", {}).get("total_queries", 0),
                    "ndcg@10": eval_results.get("aggregated_metrics", {}).get("ndcg@10_mean", 0.0)
                }
                pipeline_results["steps_completed"].append("evaluation")
            else:
                logger.info("평가 단계 건너뜀")
            
            optimization_results_path = None
            if not skip_optimization:
                opt_results = self.run_optimization(
                    max_combinations=optimization_max_combinations,
                    sample_size=optimization_sample_size
                )
                optimization_results_path = "data/evaluation/evaluation_reports/pipeline_optimization_results.json"
                pipeline_results["optimization"] = {
                    "best_score": opt_results.get("best_result", {}).get("primary_score", 0.0),
                    "total_experiments": opt_results.get("total_experiments", 0)
                }
                pipeline_results["steps_completed"].append("optimization")
            else:
                logger.info("최적화 단계 건너뜀")
                optimization_results_path = "data/evaluation/evaluation_reports/search_optimization_results.json"
            
            if not skip_apply and optimization_results_path:
                config = self.apply_optimized_params(optimization_results_path)
                pipeline_results["config"] = {
                    "output_path": "data/ml_config/optimized_search_params.json",
                    "parameters": config.get("search_parameters", {})
                }
                pipeline_results["steps_completed"].append("apply")
            else:
                logger.info("파라미터 적용 단계 건너뜀")
            
            if not skip_build:
                version_name = self.build_production_index()
                pipeline_results["production"] = {
                    "version_name": version_name
                }
                pipeline_results["steps_completed"].append("build")
            else:
                logger.info("프로덕션 빌드 단계 건너뜀")
            
            pipeline_results["end_time"] = datetime.now().isoformat()
            pipeline_results["status"] = "success"
            
            logger.info("=" * 80)
            logger.info("파이프라인 완료")
            logger.info("=" * 80)
            logger.info(f"완료된 단계: {', '.join(pipeline_results['steps_completed'])}")
            
        except Exception as e:
            pipeline_results["status"] = "failed"
            pipeline_results["errors"].append(str(e))
            logger.error(f"파이프라인 실패: {e}", exc_info=True)
        
        return pipeline_results


def main():
    parser = argparse.ArgumentParser(description="자동화 최적화 파이프라인")
    parser.add_argument(
        "--vector-store-path",
        required=True,
        help="벡터 스토어 경로"
    )
    parser.add_argument(
        "--ground-truth-path",
        required=True,
        help="Ground Truth 파일 경로"
    )
    parser.add_argument(
        "--db-path",
        default="data/lawfirm_v2.db",
        help="데이터베이스 경로"
    )
    parser.add_argument(
        "--model-name",
        default="jhgan/ko-sroberta-multitask",
        help="임베딩 모델명"
    )
    parser.add_argument(
        "--embedding-version-id",
        type=int,
        help="임베딩 버전 ID (None이면 활성 버전 사용)"
    )
    parser.add_argument(
        "--chunking-strategy",
        help="청킹 전략 (None이면 활성 버전 사용)"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="평가 단계 건너뛰기"
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="최적화 단계 건너뛰기"
    )
    parser.add_argument(
        "--skip-apply",
        action="store_true",
        help="파라미터 적용 단계 건너뛰기"
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="프로덕션 빌드 단계 건너뛰기"
    )
    parser.add_argument(
        "--optimization-max-combinations",
        type=int,
        default=20,
        help="최적화 최대 조합 수"
    )
    parser.add_argument(
        "--optimization-sample-size",
        type=int,
        help="최적화 샘플 크기 (None이면 전체)"
    )
    parser.add_argument(
        "--output-path",
        default="data/evaluation/evaluation_reports/pipeline_results.json",
        help="파이프라인 결과 저장 경로"
    )
    
    args = parser.parse_args()
    
    pipeline = OptimizationPipeline(
        vector_store_path=args.vector_store_path,
        ground_truth_path=args.ground_truth_path,
        db_path=args.db_path,
        model_name=args.model_name,
        embedding_version_id=args.embedding_version_id,
        chunking_strategy=args.chunking_strategy
    )
    
    results = pipeline.run_full_pipeline(
        skip_evaluation=args.skip_evaluation,
        skip_optimization=args.skip_optimization,
        skip_apply=args.skip_apply,
        skip_build=args.skip_build,
        optimization_max_combinations=args.optimization_max_combinations,
        optimization_sample_size=args.optimization_sample_size
    )
    
    if args.output_path:
        output_file = Path(args.output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"파이프라인 결과 저장: {output_file}")
    
    if results["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()

