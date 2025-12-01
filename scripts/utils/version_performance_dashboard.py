"""
버전별 성능 통계 대시보드 생성 스크립트

FAISS 버전별 검색 성능 통계를 시각화하고 리포트를 생성합니다.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from version_performance_monitor import VersionPerformanceMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_dashboard(
    log_path: str = "data/performance_logs",
    output_path: Optional[str] = None,
    format: str = "json"
):
    """
    성능 대시보드 생성
    
    Args:
        log_path: 성능 로그 경로
        output_path: 출력 파일 경로 (None이면 콘솔 출력)
        format: 출력 형식 (json, text, markdown)
    """
    monitor = VersionPerformanceMonitor(log_path)
    versions = monitor.list_versions()
    
    if not versions:
        print("No performance data available")
        return
    
    dashboard_data = {
        "summary": {},
        "versions": {}
    }
    
    for version in versions:
        metrics = monitor.get_version_metrics(version)
        if metrics:
            dashboard_data["versions"][version] = {
                "total_queries": metrics.get("total_queries", 0),
                "avg_latency_ms": round(metrics.get("avg_latency", 0.0), 2),
                "avg_relevance": round(metrics.get("avg_relevance", 0.0), 4),
                "feedback_positive": metrics.get("feedback_positive", 0),
                "feedback_negative": metrics.get("feedback_negative", 0),
                "feedback_score": round(
                    metrics.get("feedback_positive", 0) / 
                    (metrics.get("feedback_positive", 0) + metrics.get("feedback_negative", 0))
                    if (metrics.get("feedback_positive", 0) + metrics.get("feedback_negative", 0)) > 0
                    else 0.0,
                    4
                )
            }
    
    if len(versions) > 1:
        comparisons = []
        for i, v1 in enumerate(versions):
            for v2 in versions[i+1:]:
                comparison = monitor.compare_performance(v1, v2)
                if "error" not in comparison:
                    comparisons.append(comparison)
        dashboard_data["comparisons"] = comparisons
    
    if format == "json":
        output = json.dumps(dashboard_data, indent=2, ensure_ascii=False)
    elif format == "text":
        output = format_text_dashboard(dashboard_data)
    elif format == "markdown":
        output = format_markdown_dashboard(dashboard_data)
    else:
        output = json.dumps(dashboard_data, indent=2, ensure_ascii=False)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output)
        logger.info(f"Dashboard saved to: {output_path}")
    else:
        print(output)


def format_text_dashboard(data: Dict) -> str:
    """텍스트 형식 대시보드 포맷팅"""
    lines = []
    lines.append("=" * 80)
    lines.append("Version Performance Dashboard")
    lines.append("=" * 80)
    lines.append("")
    
    if "versions" in data:
        lines.append("Version Metrics:")
        lines.append("-" * 80)
        for version, metrics in data["versions"].items():
            lines.append(f"\nVersion: {version}")
            lines.append(f"  Total Queries: {metrics['total_queries']}")
            lines.append(f"  Avg Latency: {metrics['avg_latency_ms']} ms")
            lines.append(f"  Avg Relevance: {metrics['avg_relevance']:.4f}")
            lines.append(f"  Feedback Score: {metrics['feedback_score']:.4f}")
            lines.append(f"  Positive: {metrics['feedback_positive']}, Negative: {metrics['feedback_negative']}")
    
    if "comparisons" in data and data["comparisons"]:
        lines.append("\n" + "=" * 80)
        lines.append("Version Comparisons:")
        lines.append("-" * 80)
        for comp in data["comparisons"]:
            lines.append(f"\n{comp['version1']} vs {comp['version2']}:")
            lines.append(f"  Latency Improvement: {comp['latency_improvement_percent']:.2f}%")
            lines.append(f"  Relevance Improvement: {comp['relevance_improvement_percent']:.2f}%")
            lines.append(f"  Feedback Score V1: {comp['feedback_score_v1']:.4f}")
            lines.append(f"  Feedback Score V2: {comp['feedback_score_v2']:.4f}")
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def format_markdown_dashboard(data: Dict) -> str:
    """Markdown 형식 대시보드 포맷팅"""
    lines = []
    lines.append("# Version Performance Dashboard")
    lines.append("")
    
    if "versions" in data:
        lines.append("## Version Metrics")
        lines.append("")
        lines.append("| Version | Queries | Avg Latency (ms) | Avg Relevance | Feedback Score |")
        lines.append("|---------|---------|------------------|---------------|----------------|")
        for version, metrics in data["versions"].items():
            lines.append(
                f"| {version} | {metrics['total_queries']} | "
                f"{metrics['avg_latency_ms']} | {metrics['avg_relevance']:.4f} | "
                f"{metrics['feedback_score']:.4f} |"
            )
        lines.append("")
    
    if "comparisons" in data and data["comparisons"]:
        lines.append("## Version Comparisons")
        lines.append("")
        lines.append("| Version 1 | Version 2 | Latency Improvement | Relevance Improvement |")
        lines.append("|-----------|-----------|---------------------|----------------------|")
        for comp in data["comparisons"]:
            lines.append(
                f"| {comp['version1']} | {comp['version2']} | "
                f"{comp['latency_improvement_percent']:.2f}% | "
                f"{comp['relevance_improvement_percent']:.2f}% |"
            )
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate version performance dashboard")
    parser.add_argument("--log-path", default="data/performance_logs", help="Performance log path")
    parser.add_argument("--output", help="Output file path (None for console)")
    parser.add_argument("--format", choices=["json", "text", "markdown"], default="text", help="Output format")
    
    args = parser.parse_args()
    
    generate_dashboard(
        log_path=args.log_path,
        output_path=args.output,
        format=args.format
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

