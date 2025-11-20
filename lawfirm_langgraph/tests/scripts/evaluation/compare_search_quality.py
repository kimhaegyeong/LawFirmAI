# -*- coding: utf-8 -*-
"""
검색 품질 Before/After 비교 스크립트
개선 전후 성능 비교 및 리포트 생성
"""

import sys
import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# 프로젝트 루트 경로 추가
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent.parent
lawfirm_langgraph_dir = project_root / "lawfirm_langgraph"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(lawfirm_langgraph_dir))

import logging

# UTF-8 인코딩 설정 (Windows PowerShell 호환)
_original_stdout = sys.stdout
_original_stderr = sys.stderr

if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'buffer'):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except (ValueError, AttributeError):
            pass
    if hasattr(sys.stderr, 'buffer'):
        try:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except (ValueError, AttributeError):
            pass
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# SafeStreamHandler 클래스 정의
class SafeStreamHandler(logging.StreamHandler):
    """버퍼 분리 오류를 방지하는 안전한 스트림 핸들러"""
    
    def __init__(self, stream, original_stdout_ref=None):
        super().__init__(stream)
        self._original_stdout = original_stdout_ref
    
    def _get_safe_stream(self):
        """안전한 스트림 반환"""
        streams_to_try = []
        if self.stream and hasattr(self.stream, 'write'):
            streams_to_try.append(self.stream)
        if self._original_stdout and hasattr(self._original_stdout, 'write'):
            streams_to_try.append(self._original_stdout)
        
        for stream in streams_to_try:
            try:
                if hasattr(stream, 'buffer'):
                    try:
                        buffer = stream.buffer
                        if buffer is not None:
                            return stream
                    except (ValueError, AttributeError):
                        continue
                else:
                    return stream
            except (ValueError, AttributeError):
                continue
        
        return None
    
    def emit(self, record):
        """안전한 로그 출력 (버퍼 분리 오류 방지)"""
        try:
            msg = self.format(record) + self.terminator
            safe_stream = self._get_safe_stream()
            if safe_stream is not None:
                try:
                    safe_stream.write(msg)
                    try:
                        safe_stream.flush()
                    except (ValueError, AttributeError, OSError):
                        pass
                    return
                except (ValueError, AttributeError, OSError) as e:
                    if "detached" not in str(e).lower():
                        pass
        except Exception:
            pass

# 로깅 설정
# 루트 로거 설정 (모든 하위 모듈 로거가 같은 출력 사용)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers.clear()

# SafeStreamHandler 사용
safe_handler = SafeStreamHandler(sys.stdout, _original_stdout)
safe_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
safe_handler.setFormatter(formatter)
root_logger.addHandler(safe_handler)

# 현재 모듈 로거
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import 시도 (여러 경로)
try:
    from evaluation.test_search_quality_evaluation import (
        SearchQualityEvaluator,
        TEST_QUERIES
    )
except ImportError:
    try:
        from lawfirm_langgraph.tests.scripts.evaluation.test_search_quality_evaluation import (
            SearchQualityEvaluator,
            TEST_QUERIES
        )
    except ImportError:
        sys.path.insert(0, str(current_file.parent))
        from test_search_quality_evaluation import (
            SearchQualityEvaluator,
            TEST_QUERIES
        )


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """체크포인트 파일 로드"""
    if not checkpoint_path.exists():
        return None
    
    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


def save_checkpoint(checkpoint_path: Path, data: Dict[str, Any]):
    """체크포인트 파일 저장"""
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


async def evaluate_batch_with_resume(
    evaluator: SearchQualityEvaluator,
    test_queries: List[Dict[str, str]],
    experiment_name: str,
    checkpoint_path: Path,
    output_path: Path
) -> Dict[str, Any]:
    """재개 가능한 배치 평가"""
    
    checkpoint = load_checkpoint(checkpoint_path)
    
    if checkpoint:
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        all_metrics = checkpoint.get("all_metrics", [])
        failed_queries = checkpoint.get("failed_queries", [])
        completed_queries = {m.get("query") for m in all_metrics}
        start_idx = len(all_metrics)
        
        logger.info(f"Loaded {len(all_metrics)} completed queries from checkpoint")
        logger.info(f"Resuming from query {start_idx + 1}/{len(test_queries)}")
    else:
        logger.info("Starting new evaluation (no checkpoint found)")
        all_metrics = []
        failed_queries = []
        completed_queries = set()
        start_idx = 0
    
    logger.info(f"Starting batch evaluation: {len(test_queries)} queries")
    
    for i, test_query in enumerate(test_queries[start_idx:], start=start_idx):
        query = test_query.get("query", "")
        query_type = test_query.get("type", "general_question")
        relevant_doc_ids = test_query.get("relevant_doc_ids", [])
        
        if query in completed_queries:
            logger.info(f"Skipping already completed query {i+1}/{len(test_queries)}: {query[:50]}...")
            continue
        
        logger.info(f"Evaluating query {i+1}/{len(test_queries)}: {query[:50]}...")
        
        try:
            metrics = await evaluator.evaluate_query_async(query, query_type, relevant_doc_ids)
            
            if "error" not in metrics:
                all_metrics.append(metrics)
            else:
                failed_queries.append({"query": query, "error": metrics.get("error")})
            
            checkpoint_data = {
                "experiment_name": experiment_name,
                "enable_improvements": evaluator.enable_improvements,
                "all_metrics": all_metrics,
                "failed_queries": failed_queries,
                "last_updated": datetime.now().isoformat(),
                "progress": {
                    "completed": len(all_metrics),
                    "total": len(test_queries),
                    "failed": len(failed_queries)
                }
            }
            save_checkpoint(checkpoint_path, checkpoint_data)
            
        except Exception as e:
            logger.error(f"Error evaluating query {i+1}: {e}")
            failed_queries.append({"query": query, "error": str(e)})
            checkpoint_data = {
                "experiment_name": experiment_name,
                "enable_improvements": evaluator.enable_improvements,
                "all_metrics": all_metrics,
                "failed_queries": failed_queries,
                "last_updated": datetime.now().isoformat(),
                "progress": {
                    "completed": len(all_metrics),
                    "total": len(test_queries),
                    "failed": len(failed_queries)
                }
            }
            save_checkpoint(checkpoint_path, checkpoint_data)
    
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key not in ["query", "query_type"] and isinstance(all_metrics[0][key], (int, float)):
                values = [m[key] for m in all_metrics if key in m]
                if values:
                    avg_metrics[f"avg_{key}"] = sum(values) / len(values)
                    avg_metrics[f"min_{key}"] = min(values)
                    avg_metrics[f"max_{key}"] = max(values)
        
        results = {
            "experiment_name": experiment_name,
            "enable_improvements": evaluator.enable_improvements,
            "total_queries": len(test_queries),
            "successful_queries": len(all_metrics),
            "failed_queries": len(failed_queries),
            "average_metrics": avg_metrics,
            "detailed_metrics": all_metrics,
            "failed_queries_list": failed_queries
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Checkpoint removed: {checkpoint_path}")
        
        return results
    else:
        return {
            "experiment_name": experiment_name,
            "error": "All queries failed",
            "failed_queries": failed_queries
        }


def compare_results(
    before_results: Dict[str, Any],
    after_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Before/After 결과 비교"""
    comparison = {
        "comparison_date": datetime.now().isoformat(),
        "before": {
            "experiment_name": before_results.get("experiment_name"),
            "enable_improvements": before_results.get("enable_improvements"),
            "total_queries": before_results.get("total_queries"),
            "successful_queries": before_results.get("successful_queries")
        },
        "after": {
            "experiment_name": after_results.get("experiment_name"),
            "enable_improvements": after_results.get("enable_improvements"),
            "total_queries": after_results.get("total_queries"),
            "successful_queries": after_results.get("successful_queries")
        },
        "improvements": {}
    }
    
    before_metrics = before_results.get("average_metrics", {})
    after_metrics = after_results.get("average_metrics", {})
    
    # 각 메트릭별 개선율 계산
    for metric_key in after_metrics.keys():
        if metric_key.startswith("avg_") and metric_key in before_metrics:
            before_value = before_metrics[metric_key]
            after_value = after_metrics[metric_key]
            
            if before_value > 0:
                improvement_pct = ((after_value - before_value) / before_value) * 100
            else:
                improvement_pct = 100.0 if after_value > 0 else 0.0
            
            comparison["improvements"][metric_key] = {
                "before": before_value,
                "after": after_value,
                "improvement_pct": improvement_pct,
                "improvement_abs": after_value - before_value
            }
    
    return comparison


def generate_report(comparison: Dict[str, Any], output_path: Path):
    """비교 리포트 생성"""
    report_lines = [
        "# 검색 품질 개선 효과 리포트",
        "",
        f"**비교 일시**: {comparison['comparison_date']}",
        "",
        "## 개요",
        "",
        f"- **Before**: {'개선 기능 활성화' if comparison['before']['enable_improvements'] else '개선 기능 비활성화'}",
        f"- **After**: {'개선 기능 활성화' if comparison['after']['enable_improvements'] else '개선 기능 비활성화'}",
        f"- **테스트 쿼리 수**: {comparison['before']['total_queries']}",
        f"- **성공한 쿼리 수**: {comparison['before']['successful_queries']}",
        "",
        "## 주요 메트릭 개선",
        "",
        "| 메트릭 | Before | After | 개선율 | 절대 개선 |",
        "|--------|--------|-------|--------|----------|"
    ]
    
    positive_improvements = []
    negative_improvements = []
    
    lower_is_better_metrics = ['response_time', 'response time']
    
    for metric_key, metric_data in comparison["improvements"].items():
        metric_name = metric_key.replace("avg_", "").replace("_", " ").title()
        before_val = f"{metric_data['before']:.4f}"
        after_val = f"{metric_data['after']:.4f}"
        improvement_pct = metric_data['improvement_pct']
        improvement_abs = metric_data['improvement_abs']
        
        is_lower_better = any(lower_metric in metric_name.lower() for lower_metric in lower_is_better_metrics)
        
        if is_lower_better:
            if improvement_pct < 0:
                positive_improvements.append((metric_name, -improvement_pct))
            elif improvement_pct > 0:
                negative_improvements.append((metric_name, improvement_pct))
        else:
            if improvement_pct > 0:
                positive_improvements.append((metric_name, improvement_pct))
            elif improvement_pct < 0:
                negative_improvements.append((metric_name, improvement_pct))
        
        report_lines.append(
            f"| {metric_name} | {before_val} | {after_val} | {improvement_pct:+.2f}% | {improvement_abs:+.4f} |"
        )
    
    report_lines.extend([
        "",
        "## 개선 사항 분석",
        ""
    ])
    
    if positive_improvements:
        report_lines.append("### ✅ 개선된 메트릭")
        for metric_name, improvement_pct in positive_improvements:
            report_lines.append(f"- **{metric_name}**: {improvement_pct:+.2f}%")
        report_lines.append("")
    
    if negative_improvements:
        report_lines.append("### ⚠️ 감소한 메트릭")
        for metric_name, improvement_pct in negative_improvements:
            report_lines.append(f"- **{metric_name}**: {improvement_pct:+.2f}%")
        report_lines.append("")
    
    report_lines.extend([
        "## 결론",
        ""
    ])
    
    if positive_improvements and len(positive_improvements) > len(negative_improvements):
        report_lines.append("개선 기능 적용 후 검색 품질이 전반적으로 향상되었습니다.")
    elif positive_improvements:
        report_lines.append("개선 기능 적용 후 일부 메트릭에서 향상이 확인되었습니다.")
    else:
        report_lines.append("개선 기능 적용 후 추가 최적화가 필요할 수 있습니다.")
    
    report_lines.extend([
        "",
        "## 참고사항",
        "",
        "- Precision/Recall 메트릭은 relevant_doc_ids가 없어 모두 0으로 표시됩니다.",
        "- 실제 검색 품질 평가를 위해서는 Ground Truth 데이터가 필요합니다.",
        "- 현재는 Keyword Coverage와 Relevance 메트릭으로 품질을 평가할 수 있습니다."
    ])
    
    report_content = "\n".join(report_lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"Report generated: {output_path}")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Search Quality Before/After")
    parser.add_argument(
        "--query-type",
        type=str,
        choices=["statute_article", "precedent", "procedure", "general_question", "all"],
        default="all",
        help="Query type to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/search_quality_comparison",
        help="Output directory"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--skip-step1",
        action="store_true",
        help="Skip Step 1 if before_results.json exists"
    )
    parser.add_argument(
        "--skip-step2",
        action="store_true",
        help="Skip Step 2 if after_results.json exists"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 테스트 쿼리 선택
    if args.query_type == "all":
        test_queries = []
        for qtype, queries in TEST_QUERIES.items():
            test_queries.extend([
                {"query": q, "type": qtype}
                for q in queries
            ])
    else:
        test_queries = [
            {"query": q, "type": args.query_type}
            for q in TEST_QUERIES.get(args.query_type, [])
        ]
    
    before_output = output_dir / "before_results.json"
    before_checkpoint = output_dir / "before_checkpoint.json"
    
    if args.skip_step1 and before_output.exists():
        logger.info("=" * 60)
        logger.info("Step 1: Skipping (before_results.json exists)")
        logger.info("=" * 60)
        with open(before_output, 'r', encoding='utf-8') as f:
            before_results = json.load(f)
        logger.info(f"Loaded existing results: {before_results.get('successful_queries', 0)}/{before_results.get('total_queries', 0)} queries")
    else:
        logger.info("=" * 60)
        logger.info("Step 1: Evaluating WITHOUT improvements")
        logger.info(f"Total queries to evaluate: {len(test_queries)}")
        logger.info("=" * 60)
        
        evaluator_before = SearchQualityEvaluator(enable_improvements=False)
        logger.info("Starting batch evaluation (improvements disabled)...")
        
        if args.resume or before_checkpoint.exists():
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            before_results = loop.run_until_complete(
                evaluate_batch_with_resume(
                    evaluator_before,
                    test_queries,
                    "before_improvements",
                    before_checkpoint,
                    before_output
                )
            )
        else:
            before_results = evaluator_before.evaluate_batch(
                test_queries,
                experiment_name="before_improvements"
            )
            with open(before_output, 'w', encoding='utf-8') as f:
                json.dump(before_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Step 1 completed: {before_results.get('successful_queries', 0)}/{before_results.get('total_queries', 0)} queries successful")
        logger.info(f"Before results saved: {before_output}")
    
    after_output = output_dir / "after_results.json"
    after_checkpoint = output_dir / "after_checkpoint.json"
    
    if args.skip_step2 and after_output.exists():
        logger.info("=" * 60)
        logger.info("Step 2: Skipping (after_results.json exists)")
        logger.info("=" * 60)
        with open(after_output, 'r', encoding='utf-8') as f:
            after_results = json.load(f)
        logger.info(f"Loaded existing results: {after_results.get('successful_queries', 0)}/{after_results.get('total_queries', 0)} queries")
    else:
        logger.info("=" * 60)
        logger.info("Step 2: Evaluating WITH improvements")
        logger.info(f"Total queries to evaluate: {len(test_queries)}")
        logger.info("=" * 60)
        
        evaluator_after = SearchQualityEvaluator(enable_improvements=True)
        logger.info("Starting batch evaluation (improvements enabled)...")
        
        if args.resume or after_checkpoint.exists():
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            after_results = loop.run_until_complete(
                evaluate_batch_with_resume(
                    evaluator_after,
                    test_queries,
                    "after_improvements",
                    after_checkpoint,
                    after_output
                )
            )
        else:
            after_results = evaluator_after.evaluate_batch(
                test_queries,
                experiment_name="after_improvements"
            )
            with open(after_output, 'w', encoding='utf-8') as f:
                json.dump(after_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Step 2 completed: {after_results.get('successful_queries', 0)}/{after_results.get('total_queries', 0)} queries successful")
        logger.info(f"After results saved: {after_output}")
    
    logger.info("=" * 60)
    logger.info("Step 3: Comparing results")
    logger.info("=" * 60)
    
    comparison = compare_results(before_results, after_results)
    
    comparison_output = output_dir / "comparison.json"
    with open(comparison_output, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    logger.info(f"Comparison saved: {comparison_output}")
    
    report_output = output_dir / "comparison_report.md"
    generate_report(comparison, report_output)
    
    logger.info("=" * 60)
    logger.info("Comparison completed!")
    logger.info("=" * 60)
    logger.info(f"Results directory: {output_dir}")
    logger.info(f"Report: {report_output}")
    
    # 주요 개선 사항 출력
    logger.info("\n주요 개선 사항:")
    for metric_key, metric_data in comparison["improvements"].items():
        metric_name = metric_key.replace("avg_", "").replace("_", " ").title()
        improvement_pct = metric_data['improvement_pct']
        logger.info(f"  - {metric_name}: {improvement_pct:+.2f}%")


if __name__ == "__main__":
    main()

