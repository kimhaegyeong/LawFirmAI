#!/usr/bin/env python3
"""
RAG 검색 평가 결과 분석 및 비교 스크립트
Test, Val, Train 데이터셋의 평가 결과를 비교 분석
"""

import json
import sys
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGEvaluationAnalyzer:
    """RAG 평가 결과 분석 클래스"""
    
    def __init__(self, reports_dir: str = "data/evaluation/evaluation_reports"):
        self.reports_dir = Path(reports_dir)
        self.results = {}
    
    def load_evaluation_reports(self) -> Dict[str, Dict[str, Any]]:
        """평가 리포트 파일들을 로드"""
        logger.info(f"Loading evaluation reports from {self.reports_dir}")
        
        datasets = ['test', 'val', 'train']
        loaded_results = {}
        
        for dataset in datasets:
            report_path = self.reports_dir / f"rag_evaluation_report_{dataset}.json"
            if not report_path.exists():
                logger.warning(f"Report not found: {report_path}")
                continue
            
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    loaded_results[dataset] = data
                    logger.info(f"Loaded {dataset} report: {len(data.get('evaluation_results', {}).get('per_query_metrics', []))} queries")
            except Exception as e:
                logger.error(f"Failed to load {report_path}: {e}")
                continue
        
        self.results = loaded_results
        return loaded_results
    
    def extract_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """평가 결과에서 메트릭 추출"""
        metrics = {}
        agg_metrics = data.get('evaluation_results', {}).get('aggregated_metrics', {})
        
        for k in [5, 10, 20]:
            metrics[f'recall@{k}'] = agg_metrics.get(f'recall@{k}_mean', 0.0)
            metrics[f'recall@{k}_std'] = agg_metrics.get(f'recall@{k}_std', 0.0)
            metrics[f'precision@{k}'] = agg_metrics.get(f'precision@{k}_mean', 0.0)
            metrics[f'precision@{k}_std'] = agg_metrics.get(f'precision@{k}_std', 0.0)
            metrics[f'ndcg@{k}'] = agg_metrics.get(f'ndcg@{k}_mean', 0.0)
            metrics[f'ndcg@{k}_std'] = agg_metrics.get(f'ndcg@{k}_std', 0.0)
        
        metrics['mrr'] = agg_metrics.get('mrr_mean', 0.0)
        metrics['mrr_std'] = agg_metrics.get('mrr_std', 0.0)
        metrics['total_queries'] = agg_metrics.get('total_queries', 0)
        metrics['processed_count'] = agg_metrics.get('processed_count', 0)
        
        return metrics
    
    def compare_datasets(self) -> Dict[str, Any]:
        """데이터셋 간 메트릭 비교"""
        if not self.results:
            logger.error("No results loaded. Call load_evaluation_reports() first.")
            return {}
        
        comparison = {
            'summary': {},
            'detailed_comparison': {},
            'best_performers': {}
        }
        
        dataset_metrics = {}
        for dataset, data in self.results.items():
            metrics = self.extract_metrics(data)
            dataset_metrics[dataset] = metrics
            comparison['summary'][dataset] = {
                'total_queries': metrics['total_queries'],
                'processed_count': metrics['processed_count'],
                'precision@5': f"{metrics['precision@5']:.4f} ± {metrics['precision@5_std']:.4f}",
                'recall@5': f"{metrics['recall@5']:.4f} ± {metrics['recall@5_std']:.4f}",
                'ndcg@5': f"{metrics['ndcg@5']:.4f} ± {metrics['ndcg@5_std']:.4f}",
                'mrr': f"{metrics['mrr']:.4f} ± {metrics['mrr_std']:.4f}"
            }
        
        for k in [5, 10, 20]:
            comparison['detailed_comparison'][f'k={k}'] = {}
            for metric_type in ['recall', 'precision', 'ndcg']:
                metric_key = f'{metric_type}@{k}'
                values = {ds: dataset_metrics[ds][metric_key] for ds in dataset_metrics.keys()}
                comparison['detailed_comparison'][f'k={k}'][metric_type] = values
                
                best_dataset = max(values.items(), key=lambda x: x[1])
                comparison['best_performers'][metric_key] = {
                    'dataset': best_dataset[0],
                    'value': best_dataset[1]
                }
        
        mrr_values = {ds: dataset_metrics[ds]['mrr'] for ds in dataset_metrics.keys()}
        best_mrr = max(mrr_values.items(), key=lambda x: x[1])
        comparison['best_performers']['mrr'] = {
            'dataset': best_mrr[0],
            'value': best_mrr[1]
        }
        
        return comparison
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """종합 리포트 생성"""
        if not self.results:
            logger.error("No results loaded. Call load_evaluation_reports() first.")
            return ""
        
        comparison = self.compare_datasets()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("RAG 검색 평가 결과 종합 리포트")
        report_lines.append("=" * 80)
        report_lines.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        report_lines.append("1. 데이터셋 요약")
        report_lines.append("-" * 80)
        for dataset, summary in comparison['summary'].items():
            report_lines.append(f"\n[{dataset.upper()}]")
            report_lines.append(f"  총 쿼리 수: {summary['total_queries']}")
            report_lines.append(f"  처리된 쿼리 수: {summary['processed_count']}")
            report_lines.append(f"  Precision@5: {summary['precision@5']}")
            report_lines.append(f"  Recall@5: {summary['recall@5']}")
            report_lines.append(f"  NDCG@5: {summary['ndcg@5']}")
            report_lines.append(f"  MRR: {summary['mrr']}")
        
        report_lines.append("\n\n2. 상세 메트릭 비교")
        report_lines.append("-" * 80)
        for k, metrics in comparison['detailed_comparison'].items():
            report_lines.append(f"\n[{k}]")
            for metric_type, values in metrics.items():
                report_lines.append(f"  {metric_type.upper()}:")
                for dataset, value in values.items():
                    report_lines.append(f"    {dataset}: {value:.6f}")
        
        report_lines.append("\n\n3. 최고 성능 데이터셋")
        report_lines.append("-" * 80)
        for metric, info in comparison['best_performers'].items():
            report_lines.append(f"  {metric}: {info['dataset']} ({info['value']:.6f})")
        
        report_lines.append("\n\n4. 분석 및 인사이트")
        report_lines.append("-" * 80)
        
        all_metrics = {}
        for dataset in self.results.keys():
            all_metrics[dataset] = self.extract_metrics(self.results[dataset])
        
        avg_precision = sum(m['precision@5'] for m in all_metrics.values()) / len(all_metrics)
        avg_recall = sum(m['recall@5'] for m in all_metrics.values()) / len(all_metrics)
        avg_ndcg = sum(m['ndcg@5'] for m in all_metrics.values()) / len(all_metrics)
        avg_mrr = sum(m['mrr'] for m in all_metrics.values()) / len(all_metrics)
        
        report_lines.append(f"\n평균 성능:")
        report_lines.append(f"  Precision@5: {avg_precision:.6f}")
        report_lines.append(f"  Recall@5: {avg_recall:.6f}")
        report_lines.append(f"  NDCG@5: {avg_ndcg:.6f}")
        report_lines.append(f"  MRR: {avg_mrr:.6f}")
        
        report_lines.append(f"\n주요 관찰사항:")
        if avg_precision < 0.01:
            report_lines.append("  - Precision 값이 매우 낮음 (0.01 미만)")
            report_lines.append("    → 검색 결과의 정확도가 낮을 수 있음")
        if avg_recall < 0.001:
            report_lines.append("  - Recall 값이 매우 낮음 (0.001 미만)")
            report_lines.append("    → 관련 문서를 찾지 못하는 경우가 많음")
        if avg_mrr < 0.02:
            report_lines.append("  - MRR 값이 낮음 (0.02 미만)")
            report_lines.append("    → 첫 번째 관련 문서의 순위가 낮음")
        
        report_lines.append("\n개선 제안:")
        report_lines.append("  1. 임베딩 모델 개선 (더 나은 문장 임베딩 모델 사용)")
        report_lines.append("  2. 검색 파라미터 튜닝 (top_k, 검색 알고리즘)")
        report_lines.append("  3. Ground Truth 품질 개선 (더 정확한 관련 문서 라벨링)")
        report_lines.append("  4. 하이브리드 검색 (키워드 + 벡터 검색 결합)")
        
        report_lines.append("\n" + "=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Report saved to: {output_file}")
        
        return report_text
    
    def save_comparison_json(self, output_path: str):
        """비교 결과를 JSON으로 저장"""
        comparison = self.compare_datasets()
        
        output_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'datasets': list(self.results.keys())
            },
            'comparison': comparison,
            'raw_metrics': {
                dataset: self.extract_metrics(data)
                for dataset, data in self.results.items()
            }
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Comparison JSON saved to: {output_file}")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG 평가 결과 분석')
    parser.add_argument(
        '--reports-dir',
        type=str,
        default='data/evaluation/evaluation_reports',
        help='평가 리포트 디렉토리 경로'
    )
    parser.add_argument(
        '--output-report',
        type=str,
        default='data/evaluation/evaluation_reports/rag_evaluation_summary.txt',
        help='종합 리포트 출력 경로'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default='data/evaluation/evaluation_reports/rag_evaluation_comparison.json',
        help='비교 결과 JSON 출력 경로'
    )
    
    args = parser.parse_args()
    
    analyzer = RAGEvaluationAnalyzer(reports_dir=args.reports_dir)
    analyzer.load_evaluation_reports()
    
    report_text = analyzer.generate_report(output_path=args.output_report)
    analyzer.save_comparison_json(output_path=args.output_json)
    
    print("\n" + report_text)


if __name__ == "__main__":
    main()

