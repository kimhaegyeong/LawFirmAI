"""
최적 파라미터 로더 유틸리티

최적화된 검색 파라미터를 로드하고 RAG 시스템에 적용할 수 있는 형태로 변환합니다.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_optimized_search_params(
    config_path: str = "data/ml_config/optimized_search_params.json"
) -> Optional[Dict[str, Any]]:
    """
    최적 검색 파라미터 로드
    
    Args:
        config_path: 설정 파일 경로
    
    Returns:
        Dict: 검색 파라미터 딕셔너리 (None if not found)
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"최적 파라미터 파일을 찾을 수 없습니다: {config_path}")
        return None
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        search_params = config.get("search_parameters", {})
        
        if not search_params:
            logger.warning("설정 파일에 search_parameters가 없습니다")
            return None
        
        logger.info(f"최적 파라미터 로드 완료: {config_path}")
        return search_params
        
    except Exception as e:
        logger.error(f"최적 파라미터 로드 실패: {e}")
        return None


def convert_to_workflow_params(optimized_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    최적화된 파라미터를 워크플로우 파라미터 형식으로 변환
    
    Args:
        optimized_params: 최적화된 파라미터 딕셔너리
    
    Returns:
        Dict: 워크플로우에서 사용할 수 있는 파라미터 형식
    """
    workflow_params = {
        "semantic_k": optimized_params.get("top_k", 15),
        "keyword_k": int(optimized_params.get("top_k", 15) * (1 - optimized_params.get("hybrid_search_ratio", 1.0))),
        "min_relevance": optimized_params.get("similarity_threshold", 0.9),
        "use_reranking": optimized_params.get("use_reranking", True),
        "rerank_top_n": optimized_params.get("rerank_top_n", 30) if optimized_params.get("use_reranking", True) else None,
        "query_enhancement": optimized_params.get("query_enhancement", True),
        "use_keyword_search": optimized_params.get("use_keyword_search", True),
        "hybrid_search_ratio": optimized_params.get("hybrid_search_ratio", 1.0)
    }
    
    return workflow_params


def get_optimized_search_params(
    config_path: str = "data/ml_config/optimized_search_params.json"
) -> Optional[Dict[str, Any]]:
    """
    최적 파라미터를 워크플로우 형식으로 로드 및 변환
    
    Args:
        config_path: 설정 파일 경로
    
    Returns:
        Dict: 워크플로우 파라미터 딕셔너리 (None if not found)
    """
    optimized_params = load_optimized_search_params(config_path)
    if not optimized_params:
        return None
    
    return convert_to_workflow_params(optimized_params)

