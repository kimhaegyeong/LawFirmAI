"""
최적 파라미터 적용 스크립트

최적화 결과에서 최적 파라미터를 추출하여 설정 파일로 저장합니다.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_optimization_results(results_path: str) -> Dict[str, Any]:
    """최적화 결과 파일 로드"""
    results_file = Path(results_path)
    if not results_file.exists():
        raise FileNotFoundError(f"최적화 결과 파일을 찾을 수 없습니다: {results_path}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'best_result' not in data or not data['best_result']:
        raise ValueError("최적화 결과에 best_result가 없습니다")
    
    return data


def create_search_config(best_params: Dict[str, Any], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """검색 설정 파일 생성"""
    config = {
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "source": "optimization_results",
        "metadata": metadata or {},
        "search_parameters": {
            "top_k": best_params.get("top_k", 15),
            "similarity_threshold": best_params.get("similarity_threshold", 0.9),
            "use_reranking": best_params.get("use_reranking", True),
            "rerank_top_n": best_params.get("rerank_top_n", 30),
            "query_enhancement": best_params.get("query_enhancement", True),
            "hybrid_search_ratio": best_params.get("hybrid_search_ratio", 1.0),
            "use_keyword_search": best_params.get("use_keyword_search", True)
        },
        "description": "MLflow 최적화를 통해 찾은 최적 검색 파라미터"
    }
    
    return config


def save_config(config: Dict[str, Any], output_path: str, backup: bool = True):
    """설정 파일 저장 (기존 파일 백업)"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if backup and output_file.exists():
        backup_path = output_file.with_suffix(f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        logger.info(f"기존 설정 파일 백업: {backup_path}")
        output_file.rename(backup_path)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"설정 파일 저장 완료: {output_file}")


def apply_to_langgraph_config(config: Dict[str, Any], langgraph_config_path: str = None):
    """LangGraph 설정 파일에 적용 (선택사항)"""
    if not langgraph_config_path:
        return
    
    langgraph_config_file = Path(langgraph_config_path)
    if not langgraph_config_file.exists():
        logger.warning(f"LangGraph 설정 파일을 찾을 수 없습니다: {langgraph_config_path}")
        return
    
    try:
        with open(langgraph_config_file, 'r', encoding='utf-8') as f:
            langgraph_config = json.load(f)
        
        search_params = config.get("search_parameters", {})
        
        if "similarity_threshold" in search_params:
            langgraph_config["similarity_threshold"] = search_params["similarity_threshold"]
        
        if "max_retrieved_docs" not in langgraph_config:
            langgraph_config["max_retrieved_docs"] = search_params.get("top_k", 15)
        else:
            langgraph_config["max_retrieved_docs"] = search_params.get("top_k", 15)
        
        with open(langgraph_config_file, 'w', encoding='utf-8') as f:
            json.dump(langgraph_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"LangGraph 설정 파일 업데이트 완료: {langgraph_config_file}")
    except Exception as e:
        logger.error(f"LangGraph 설정 파일 업데이트 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description="최적 파라미터를 설정 파일로 저장")
    parser.add_argument(
        "--results-path",
        default="data/evaluation/evaluation_reports/search_optimization_results.json",
        help="최적화 결과 파일 경로"
    )
    parser.add_argument(
        "--output-path",
        default="data/ml_config/optimized_search_params.json",
        help="출력 설정 파일 경로"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="기존 파일 백업하지 않음"
    )
    parser.add_argument(
        "--apply-to-langgraph",
        help="LangGraph 설정 파일 경로 (선택사항)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("최적 파라미터 적용 시작")
    logger.info("=" * 80)
    
    try:
        results = load_optimization_results(args.results_path)
        best_result = results["best_result"]
        best_params = best_result["parameters"]
        
        logger.info(f"최적 파라미터 발견:")
        logger.info(f"  - Primary Score: {best_result.get('primary_score', 0):.4f}")
        logger.info(f"  - Run ID: {best_result.get('run_id', 'N/A')}")
        logger.info(f"  - Parameters: {best_params}")
        
        metadata = {
            "optimization_run_id": best_result.get("run_id"),
            "primary_score": best_result.get("primary_score"),
            "optimization_date": datetime.now().isoformat(),
            "total_experiments": results.get("total_experiments", 0),
            "primary_metric": results.get("primary_metric", "ndcg@10_mean")
        }
        
        config = create_search_config(best_params, metadata)
        
        save_config(config, args.output_path, backup=not args.no_backup)
        
        if args.apply_to_langgraph:
            apply_to_langgraph_config(config, args.apply_to_langgraph)
        
        logger.info("=" * 80)
        logger.info("최적 파라미터 적용 완료")
        logger.info("=" * 80)
        logger.info(f"설정 파일: {args.output_path}")
        logger.info(f"다음 단계: 이 설정 파일을 RAG 시스템에서 로드하여 사용하세요")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

