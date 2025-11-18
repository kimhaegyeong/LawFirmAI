# -*- coding: utf-8 -*-
"""
MLflow Tracker for Search Quality
검색 품질 MLflow 추적 모듈
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with: pip install mlflow")


class SearchQualityTracker:
    """검색 품질 MLflow 추적기"""
    
    def __init__(
        self,
        experiment_name: str = "search_quality_improvement",
        tracking_uri: Optional[str] = None
    ):
        """
        초기화
        
        Args:
            experiment_name: MLflow 실험 이름
            tracking_uri: MLflow tracking URI
        """
        self.logger = logging.getLogger(__name__)
        self.experiment_name = experiment_name
        
        if not MLFLOW_AVAILABLE:
            self.logger.warning("MLflow not available. Tracking disabled.")
            return
        
        try:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            else:
                # 환경변수에서 읽기
                import os
                env_uri = os.getenv("MLFLOW_TRACKING_URI")
                if env_uri:
                    mlflow.set_tracking_uri(env_uri)
                else:
                    # 기본값: 프로젝트 루트의 mlflow/mlruns 사용
                    from pathlib import Path
                    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
                    default_mlruns = project_root / "mlflow" / "mlruns"
                    if default_mlruns.exists():
                        # Windows 경로를 file:/// 형식으로 변환
                        default_uri = f"file:///{str(default_mlruns).replace(chr(92), '/')}"
                        mlflow.set_tracking_uri(default_uri)
                    else:
                        # 환경변수나 config에서 읽기
                        from core.utils.config import Config
                        config = Config()
                        if hasattr(config, 'mlflow_tracking_uri') and config.mlflow_tracking_uri:
                            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            
            mlflow.set_experiment(experiment_name)
            self.logger.info(f"SearchQualityTracker initialized (experiment: {experiment_name})")
        except Exception as e:
            self.logger.warning(f"MLflow initialization failed: {e}")
    
    def track_search_experiment(
        self,
        query: str,
        results: List[Dict[str, Any]],
        feature_name: str,
        params: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        query_type: Optional[str] = None
    ) -> Optional[str]:
        """
        검색 실험 추적
        
        Args:
            query: 검색 쿼리
            results: 검색 결과
            feature_name: 개선 기능명
            params: 실험 파라미터
            metrics: 평가 메트릭
            query_type: 질문 유형
        
        Returns:
            Optional[str]: MLflow run ID
        """
        if not MLFLOW_AVAILABLE:
            return None
        
        try:
            run_name = f"{feature_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with mlflow.start_run(run_name=run_name):
                # 태그 설정
                mlflow.set_tags({
                    "feature": feature_name,
                    "query_type": query_type or "unknown",
                    "query_length": len(query),
                    "result_count": len(results)
                })
                
                # 파라미터 로깅
                mlflow.log_params(params)
                
                # 메트릭 계산 및 로깅
                if metrics is None:
                    metrics = self._calculate_metrics(results, query)
                
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)
                
                # 쿼리 및 결과 샘플 저장
                sample_data = {
                    "query": query,
                    "results_count": len(results),
                    "top_results": results[:5] if results else []
                }
                
                # 임시 파일로 저장
                temp_dir = Path("logs/mlflow_artifacts")
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                sample_file = temp_dir / f"{run_name}_sample.json"
                with open(sample_file, 'w', encoding='utf-8') as f:
                    json.dump(sample_data, f, ensure_ascii=False, indent=2)
                
                mlflow.log_artifact(str(sample_file), "samples")
                
                run_id = mlflow.active_run().info.run_id
                self.logger.info(f"Search experiment tracked: {run_id} ({feature_name})")
                
                return run_id
        
        except Exception as e:
            self.logger.error(f"MLflow tracking failed: {e}")
            return None
    
    def _calculate_metrics(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, float]:
        """메트릭 계산"""
        metrics = {}
        
        if not results:
            return {
                "avg_relevance": 0.0,
                "max_relevance": 0.0,
                "min_relevance": 0.0,
                "diversity_score": 0.0,
                "keyword_coverage": 0.0
            }
        
        # 관련성 점수
        relevance_scores = [
            r.get("relevance_score", r.get("similarity", 0.0))
            for r in results
        ]
        
        metrics["avg_relevance"] = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        metrics["max_relevance"] = max(relevance_scores) if relevance_scores else 0.0
        metrics["min_relevance"] = min(relevance_scores) if relevance_scores else 0.0
        
        # 다양성 점수 (간단한 계산)
        sources = [r.get("source", "") for r in results]
        unique_sources = len(set(sources))
        metrics["diversity_score"] = unique_sources / len(results) if results else 0.0
        
        # Keyword Coverage (간단한 추정)
        query_words = set(query.split())
        matched_words = 0
        for result in results[:5]:  # 상위 5개만 확인
            text = result.get("text", result.get("content", ""))
            if text:
                text_words = set(text.split())
                matched_words += len(query_words & text_words)
        
        metrics["keyword_coverage"] = matched_words / (len(query_words) * 5) if query_words else 0.0
        
        return metrics
    
    async def track_batch_experiment(
        self,
        test_queries: List[Dict[str, str]],
        feature_name: str,
        params: Dict[str, Any],
        results_func: Any
    ) -> Optional[str]:
        """
        배치 실험 추적
        
        Args:
            test_queries: 테스트 쿼리 리스트 (각각 {"query": "...", "type": "..."})
            feature_name: 개선 기능명
            params: 실험 파라미터
            results_func: 검색 결과를 반환하는 async 함수 또는 일반 함수
        
        Returns:
            Optional[str]: MLflow run ID
        """
        if not MLFLOW_AVAILABLE:
            return None
        
        try:
            run_name = f"{feature_name}_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags({
                    "feature": feature_name,
                    "test_queries_count": len(test_queries),
                    "experiment_type": "batch"
                })
                
                mlflow.log_params(params)
                
                # 각 쿼리별 결과 수집
                all_metrics = []
                
                for i, test_query in enumerate(test_queries):
                    query = test_query.get("query", "")
                    query_type = test_query.get("type", "general_question")
                    
                    try:
                        import inspect
                        if inspect.iscoroutinefunction(results_func):
                            results = await results_func(query, query_type)
                        else:
                            results = results_func(query, query_type)
                        metrics = self._calculate_metrics(results, query)
                        all_metrics.append(metrics)
                    except Exception as e:
                        self.logger.warning(f"Query {i+1} failed: {e}")
                
                # 평균 메트릭 계산
                if all_metrics:
                    avg_metrics = {}
                    for key in all_metrics[0].keys():
                        avg_metrics[f"avg_{key}"] = sum(m.get(key, 0.0) for m in all_metrics) / len(all_metrics)
                    
                    for metric_name, metric_value in avg_metrics.items():
                        mlflow.log_metric(metric_name, metric_value)
                
                run_id = mlflow.active_run().info.run_id
                self.logger.info(f"Batch experiment tracked: {run_id} ({feature_name}, {len(test_queries)} queries)")
                
                return run_id
        
        except Exception as e:
            self.logger.error(f"Batch MLflow tracking failed: {e}")
            return None

