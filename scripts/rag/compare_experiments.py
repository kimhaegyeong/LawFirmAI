"""
MLflow 실험 결과 비교 및 분석

여러 실험 결과를 비교하여 최적의 설정을 찾습니다.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts" / "rag"))

try:
    import mlflow
    import pandas as pd
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow or pandas not available")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentComparator:
    """MLflow 실험 비교 클래스"""
    
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """
        초기화
        
        Args:
            experiment_name: MLflow 실험 이름
            tracking_uri: MLflow tracking URI
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow and pandas are required")
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # 환경 변수 확인
            import os
            env_uri = os.getenv("MLFLOW_TRACKING_URI")
            if env_uri:
                mlflow.set_tracking_uri(env_uri)
            else:
                # SQLite 백엔드 사용 (FutureWarning 해결)
                default_db_path = project_root / "mlflow" / "mlflow.db"
                default_db_path.parent.mkdir(parents=True, exist_ok=True)
                default_uri = f"sqlite:///{str(default_db_path).replace(os.sep, '/')}"
                mlflow.set_tracking_uri(default_uri)
                logger.info(f"✅ Using SQLite backend: {default_uri}")
        
        self.experiment_name = experiment_name
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        self.experiment_id = experiment.experiment_id
    
    def get_all_runs(self, filter_string: Optional[str] = None) -> pd.DataFrame:
        """
        모든 runs 조회
        
        Args:
            filter_string: MLflow filter string
        
        Returns:
            pd.DataFrame: runs 데이터프레임
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=["metrics.primary_score DESC"]
        )
        return runs
    
    def find_best_run(self, metric: str = "primary_score") -> Dict[str, Any]:
        """
        최고 성능 run 찾기
        
        Args:
            metric: 비교할 메트릭 이름
        
        Returns:
            Dict: 최고 run 정보
        """
        runs = self.get_all_runs()
        
        if runs.empty:
            return {}
        
        metric_col = f"metrics.{metric}"
        if metric_col not in runs.columns:
            logger.warning(f"Metric '{metric}' not found. Available metrics: {runs.filter(regex='^metrics\\.').columns.tolist()}")
            return {}
        
        best_idx = runs[metric_col].idxmax()
        best_run = runs.loc[best_idx]
        
        return {
            'run_id': best_run['run_id'],
            'metric_value': best_run[metric_col],
            'parameters': best_run.filter(regex='^params\\.').to_dict(),
            'metrics': best_run.filter(regex='^metrics\\.').to_dict()
        }
    
    def compare_top_runs(self, top_n: int = 10, metric: str = "primary_score") -> pd.DataFrame:
        """
        상위 N개 runs 비교
        
        Args:
            top_n: 비교할 상위 runs 수
            metric: 정렬 기준 메트릭
        
        Returns:
            pd.DataFrame: 상위 runs 데이터프레임
        """
        runs = self.get_all_runs()
        
        if runs.empty:
            return pd.DataFrame()
        
        metric_col = f"metrics.{metric}"
        if metric_col not in runs.columns:
            return pd.DataFrame()
        
        top_runs = runs.nlargest(top_n, metric_col)
        
        return top_runs[['run_id', metric_col] + 
                       [col for col in top_runs.columns if col.startswith('params.') or 
                        (col.startswith('metrics.') and col != metric_col)]]
    
    def analyze_parameter_importance(self, metric: str = "primary_score") -> Dict[str, Any]:
        """
        파라미터 중요도 분석
        
        Args:
            metric: 분석할 메트릭
        
        Returns:
            Dict: 파라미터별 중요도 분석 결과
        """
        runs = self.get_all_runs()
        
        if runs.empty:
            return {}
        
        metric_col = f"metrics.{metric}"
        if metric_col not in runs.columns:
            return {}
        
        param_cols = [col for col in runs.columns if col.startswith('params.')]
        
        analysis = {}
        for param_col in param_cols:
            param_name = param_col.replace('params.', '')
            
            param_values = runs[param_col].unique()
            param_means = {}
            
            for value in param_values:
                if pd.isna(value):
                    continue
                subset = runs[runs[param_col] == value]
                if not subset.empty:
                    param_means[str(value)] = subset[metric_col].mean()
            
            if param_means:
                best_value = max(param_means.items(), key=lambda x: x[1])
                analysis[param_name] = {
                    'values': param_means,
                    'best_value': best_value[0],
                    'best_score': best_value[1],
                    'range': (min(param_means.values()), max(param_means.values()))
                }
        
        return analysis
    
    def generate_comparison_report(self, output_path: Optional[str] = None) -> str:
        """
        비교 리포트 생성
        
        Args:
            output_path: 출력 파일 경로
        
        Returns:
            str: 리포트 텍스트
        """
        best_run = self.find_best_run()
        top_runs = self.compare_top_runs(top_n=10)
        param_importance = self.analyze_parameter_importance()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MLflow 실험 결과 비교 리포트")
        report_lines.append("=" * 80)
        report_lines.append(f"실험 이름: {self.experiment_name}")
        report_lines.append("")
        
        if best_run:
            report_lines.append("1. 최고 성능 Run")
            report_lines.append("-" * 80)
            report_lines.append(f"Run ID: {best_run['run_id']}")
            report_lines.append(f"Primary Score: {best_run['metric_value']:.6f}")
            report_lines.append("\n파라미터:")
            for param, value in best_run['parameters'].items():
                param_name = param.replace('params.', '')
                report_lines.append(f"  {param_name}: {value}")
            report_lines.append("")
        
        if not top_runs.empty:
            report_lines.append("2. 상위 10개 Runs")
            report_lines.append("-" * 80)
            for idx, (_, row) in enumerate(top_runs.iterrows(), 1):
                report_lines.append(f"\n[{{idx}}] Run ID: {row['run_id']}")
                primary_score = row.get('metrics.primary_score', 'N/A')
                report_lines.append(f"  Primary Score: {primary_score}")
        
        if param_importance:
            report_lines.append("\n\n3. 파라미터 중요도 분석")
            report_lines.append("-" * 80)
            for param_name, analysis in param_importance.items():
                report_lines.append(f"\n{param_name}:")
                report_lines.append(f"  최적 값: {analysis['best_value']} (점수: {analysis['best_score']:.6f})")
                report_lines.append(f"  범위: {analysis['range'][0]:.6f} ~ {analysis['range'][1]:.6f}")
        
        report_lines.append("\n" + "=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Report saved to: {output_path}")
        
        return report_text


def main():
    parser = argparse.ArgumentParser(description="Compare MLflow experiments")
    parser.add_argument("--experiment-name", required=True, help="MLflow experiment name")
    parser.add_argument("--tracking-uri", help="MLflow tracking URI")
    parser.add_argument("--output-path", help="Output report file path")
    
    args = parser.parse_args()
    
    comparator = ExperimentComparator(
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri
    )
    
    report = comparator.generate_comparison_report(output_path=args.output_path)
    print("\n" + report)


if __name__ == "__main__":
    main()

