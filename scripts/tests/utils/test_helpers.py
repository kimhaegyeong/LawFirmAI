#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
테스트 공통 유틸리티 함수
"""

import sys
import os
import time
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List
from contextlib import contextmanager
from functools import wraps


def setup_test_path():
    """테스트를 위한 프로젝트 경로 설정"""
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    scripts_utils_path = project_root / "scripts" / "utils"
    if str(scripts_utils_path) not in sys.path:
        sys.path.insert(0, str(scripts_utils_path))
    
    return project_root


@contextmanager
def create_temp_dir(prefix: str = "test_"):
    """임시 디렉토리 생성 (컨텍스트 매니저)"""
    temp_path = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield temp_path
    finally:
        cleanup_temp_dir(temp_path)


def cleanup_temp_dir(path: Path):
    """임시 디렉토리 정리"""
    if path.exists():
        shutil.rmtree(path)


@contextmanager
def temporary_env(**env_vars):
    """임시 환경 변수 설정 (컨텍스트 매니저)"""
    original_values = {}
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key, None)
        os.environ[key] = str(value)
    
    try:
        yield
    finally:
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def run_async(coro):
    """비동기 함수를 동기적으로 실행"""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def measure_time(func: Callable) -> Callable:
    """함수 실행 시간 측정 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        if hasattr(wrapper, 'execution_times'):
            wrapper.execution_times.append(execution_time)
        else:
            wrapper.execution_times = [execution_time]
        return result
    return wrapper


def print_section(title: str, width: int = 80, char: str = "="):
    """섹션 헤더 출력"""
    print(char * width)
    print(title)
    print(char * width)


def print_test_header(test_name: str, width: int = 80):
    """테스트 헤더 출력"""
    print_section(f"테스트: {test_name}", width, "=")


def validate_search_results(results: List[Dict[str, Any]], min_count: int = 1, 
                           min_similarity: float = 0.0) -> Dict[str, Any]:
    """검색 결과 검증"""
    validation = {
        "valid": True,
        "errors": [],
        "count": len(results),
        "avg_similarity": 0.0,
        "min_similarity": 1.0,
        "max_similarity": 0.0
    }
    
    if len(results) < min_count:
        validation["valid"] = False
        validation["errors"].append(f"결과 수 부족: {len(results)} < {min_count}")
    
    similarities = []
    for result in results:
        similarity = result.get("similarity") or result.get("score") or result.get("relevance_score", 0.0)
        similarities.append(similarity)
        
        if similarity < min_similarity:
            validation["valid"] = False
            validation["errors"].append(f"유사도가 너무 낮음: {similarity:.4f} < {min_similarity:.4f}")
    
    if similarities:
        validation["avg_similarity"] = sum(similarities) / len(similarities)
        validation["min_similarity"] = min(similarities)
        validation["max_similarity"] = max(similarities)
    
    return validation


def assert_result_quality(result: Dict[str, Any], min_answer_length: int = 0,
                          min_sources: int = 0, min_confidence: float = 0.0):
    """워크플로우 결과 품질 검증"""
    errors = []
    
    answer = result.get("answer", "")
    if len(answer) < min_answer_length:
        errors.append(f"답변 길이 부족: {len(answer)} < {min_answer_length}")
    
    sources = result.get("sources", [])
    if len(sources) < min_sources:
        errors.append(f"소스 수 부족: {len(sources)} < {min_sources}")
    
    confidence = result.get("confidence", 0.0)
    if confidence < min_confidence:
        errors.append(f"신뢰도 부족: {confidence:.4f} < {min_confidence:.4f}")
    
    if errors:
        raise AssertionError(f"품질 검증 실패:\n" + "\n".join(f"  - {e}" for e in errors))


def get_cache_stats(service) -> Optional[Dict[str, Any]]:
    """워크플로우 서비스의 캐시 통계 조회"""
    if not hasattr(service, 'legal_workflow'):
        return None
    
    if not hasattr(service.legal_workflow, 'cache_manager'):
        return None
    
    cache_manager = service.legal_workflow.cache_manager
    if cache_manager is None:
        return None
    
    return cache_manager.get_stats()


def compare_cache_stats(stats_before: Dict[str, Any], stats_after: Dict[str, Any]) -> Dict[str, Any]:
    """캐시 통계 비교"""
    comparison = {}
    
    for cache_type in stats_before.keys():
        before = stats_before[cache_type]
        after = stats_after.get(cache_type, {"hits": 0, "misses": 0})
        
        comparison[cache_type] = {
            "hits_diff": after["hits"] - before["hits"],
            "misses_diff": after["misses"] - before["misses"],
            "hit_rate_before": before["hits"] / (before["hits"] + before["misses"]) if (before["hits"] + before["misses"]) > 0 else 0.0,
            "hit_rate_after": after["hits"] / (after["hits"] + after["misses"]) if (after["hits"] + after["misses"]) > 0 else 0.0
        }
    
    return comparison


def analyze_workflow_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """워크플로우 결과 분석"""
    analysis = {
        "answer_length": len(str(result.get("answer", ""))),
        "sources_count": len(result.get("sources", [])),
        "retrieved_docs_count": len(result.get("retrieved_docs", [])),
        "confidence": result.get("confidence", 0.0),
        "source_types": {},
        "avg_similarity": 0.0
    }
    
    retrieved_docs = result.get("retrieved_docs", [])
    if retrieved_docs:
        type_counts = {}
        similarities = []
        
        for doc in retrieved_docs:
            doc_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("source_type", "unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            similarity = doc.get("score") or doc.get("similarity") or doc.get("relevance_score", 0.0)
            similarities.append(similarity)
        
        analysis["source_types"] = type_counts
        if similarities:
            analysis["avg_similarity"] = sum(similarities) / len(similarities)
    
    return analysis

