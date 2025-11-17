"""
최적화 결과 시각화 및 분석 리포트 생성

파라미터 조합별 성능을 시각화하고 분석 리포트를 생성합니다.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("시각화 라이브러리가 없습니다. matplotlib, pandas, seaborn을 설치하세요.")


def load_optimization_results(results_path: str) -> Dict[str, Any]:
    """최적화 결과 파일 로드"""
    results_file = Path(results_path)
    if not results_file.exists():
        raise FileNotFoundError(f"최적화 결과 파일을 찾을 수 없습니다: {results_path}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def create_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """결과를 DataFrame으로 변환"""
    all_results = results.get("all_results", [])
    
    rows = []
    for result in all_results:
        params = result.get("parameters", {})
        metrics = result.get("results", {})
        
        row = {
            "run_id": result.get("run_id", ""),
            "primary_score": result.get("primary_score", 0.0),
            **params,
            **metrics
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_parameter_analysis(df: pd.DataFrame, output_dir: Path):
    """파라미터별 성능 분석"""
    logger.info("파라미터별 성능 분석 생성 중...")
    
    param_columns = ['top_k', 'similarity_threshold', 'use_reranking', 
                     'query_enhancement', 'hybrid_search_ratio']
    
    analysis = {}
    
    for param in param_columns:
        if param not in df.columns:
            continue
        
        param_stats = df.groupby(param)['primary_score'].agg(['mean', 'std', 'count'])
        analysis[param] = param_stats.to_dict('index')
        
        logger.info(f"\n{param}별 성능:")
        for value, stats in param_stats.iterrows():
            logger.info(f"  {value}: 평균={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['count']}")
    
    analysis_file = output_dir / "parameter_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"파라미터 분석 저장: {analysis_file}")
    
    return analysis


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """시각화 생성"""
    if not VISUALIZATION_AVAILABLE:
        logger.warning("시각화 라이브러리가 없어 시각화를 건너뜁니다.")
        return
    
    logger.info("시각화 생성 중...")
    
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Primary Score 분포
    plt.figure(figsize=(10, 6))
    plt.hist(df['primary_score'], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Primary Score (NDCG@10)')
    plt.ylabel('Frequency')
    plt.title('Primary Score Distribution')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "primary_score_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Primary Score 분포 차트 저장 완료")
    
    # 2. Top-K별 성능
    if 'top_k' in df.columns:
        plt.figure(figsize=(10, 6))
        top_k_performance = df.groupby('top_k')['primary_score'].mean().sort_index()
        plt.plot(top_k_performance.index, top_k_performance.values, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Top-K')
        plt.ylabel('Mean Primary Score')
        plt.title('Performance by Top-K')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "top_k_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Top-K 성능 차트 저장 완료")
    
    # 3. Similarity Threshold별 성능
    if 'similarity_threshold' in df.columns:
        plt.figure(figsize=(10, 6))
        threshold_performance = df.groupby('similarity_threshold')['primary_score'].mean().sort_index()
        plt.plot(threshold_performance.index, threshold_performance.values, marker='s', linewidth=2, markersize=8)
        plt.xlabel('Similarity Threshold')
        plt.ylabel('Mean Primary Score')
        plt.title('Performance by Similarity Threshold')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "similarity_threshold_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Similarity Threshold 성능 차트 저장 완료")
    
    # 4. 파라미터 조합 히트맵 (상위 파라미터만)
    if 'top_k' in df.columns and 'similarity_threshold' in df.columns:
        plt.figure(figsize=(12, 8))
        pivot_table = df.pivot_table(
            values='primary_score',
            index='top_k',
            columns='similarity_threshold',
            aggfunc='mean'
        )
        sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlOrRd', cbar_kws={'label': 'Primary Score'})
        plt.title('Performance Heatmap: Top-K vs Similarity Threshold')
        plt.savefig(output_dir / "parameter_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("파라미터 히트맵 저장 완료")


def generate_report(results: Dict[str, Any], df: pd.DataFrame, output_dir: Path):
    """분석 리포트 생성"""
    logger.info("분석 리포트 생성 중...")
    
    best_result = results.get("best_result", {})
    best_params = best_result.get("parameters", {})
    best_score = best_result.get("primary_score", 0.0)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("RAG 검색 품질 최적화 결과 리포트")
    report_lines.append("=" * 80)
    report_lines.append(f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    report_lines.append("1. 최적 파라미터")
    report_lines.append("-" * 80)
    report_lines.append(f"Primary Score (NDCG@10): {best_score:.6f}")
    report_lines.append(f"Run ID: {best_result.get('run_id', 'N/A')}")
    report_lines.append("")
    report_lines.append("파라미터 값:")
    for key, value in best_params.items():
        report_lines.append(f"  - {key}: {value}")
    report_lines.append("")
    
    report_lines.append("2. 전체 실험 통계")
    report_lines.append("-" * 80)
    report_lines.append(f"총 실험 수: {results.get('total_experiments', 0)}")
    report_lines.append(f"Primary Score 평균: {df['primary_score'].mean():.6f}")
    report_lines.append(f"Primary Score 표준편차: {df['primary_score'].std():.6f}")
    report_lines.append(f"Primary Score 최소값: {df['primary_score'].min():.6f}")
    report_lines.append(f"Primary Score 최대값: {df['primary_score'].max():.6f}")
    report_lines.append("")
    
    report_lines.append("3. 파라미터별 성능 분석")
    report_lines.append("-" * 80)
    
    param_columns = ['top_k', 'similarity_threshold', 'use_reranking', 
                     'query_enhancement', 'hybrid_search_ratio']
    
    for param in param_columns:
        if param not in df.columns:
            continue
        
        param_stats = df.groupby(param)['primary_score'].agg(['mean', 'std', 'count'])
        report_lines.append(f"\n{param}:")
        for value, stats in param_stats.iterrows():
            report_lines.append(f"  {value}: 평균={stats['mean']:.6f}, std={stats['std']:.6f}, n={int(stats['count'])}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    report_file = output_dir / "optimization_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"분석 리포트 저장: {report_file}")
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description="최적화 결과 시각화 및 분석 리포트 생성")
    parser.add_argument(
        "--results-path",
        default="data/evaluation/evaluation_reports/search_optimization_results.json",
        help="최적화 결과 파일 경로"
    )
    parser.add_argument(
        "--output-dir",
        default="data/evaluation/evaluation_reports/visualizations",
        help="출력 디렉토리"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("최적화 결과 시각화 및 분석 시작")
    logger.info("=" * 80)
    
    try:
        results = load_optimization_results(args.results_path)
        df = create_dataframe(results)
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generate_parameter_analysis(df, output_dir)
        create_visualizations(df, output_dir)
        generate_report(results, df, output_dir)
        
        logger.info("=" * 80)
        logger.info("시각화 및 분석 완료")
        logger.info("=" * 80)
        logger.info(f"출력 디렉토리: {output_dir}")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

