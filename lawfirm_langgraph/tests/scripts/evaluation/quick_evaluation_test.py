# -*- coding: utf-8 -*-
"""
빠른 평가 테스트 스크립트
소수의 쿼리로 평가 시스템이 정상 작동하는지 확인
"""

import sys
import os
import asyncio
from pathlib import Path

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

# 프로젝트 루트 경로 추가
current_file = Path(__file__).resolve()
# lawfirm_langgraph/tests/scripts/quick_evaluation_test.py
# -> lawfirm_langgraph (프로젝트 루트)
project_root = current_file.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 환경 변수 로딩 (다른 import 전에 실행)
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(project_root)
except ImportError:
    import warnings
    warnings.warn("utils.env_loader not found. Environment variables may not be loaded properly.")

import logging
import time
from datetime import datetime

# SafeStreamHandler 클래스 정의
class SafeStreamHandler(logging.StreamHandler):
    """버퍼 분리 오류를 방지하는 안전한 스트림 핸들러"""
    
    def __init__(self, stream, original_stdout_ref=None):
        super().__init__(stream)
        self._original_stdout = original_stdout_ref
        self._fallback_stream = None
    
    def _get_safe_stream(self):
        """안전한 스트림 반환"""
        streams_to_try = []
        if self.stream and hasattr(self.stream, 'write'):
            streams_to_try.append(self.stream)
        if self._original_stdout and hasattr(self._original_stdout, 'write'):
            streams_to_try.append(self._original_stdout)
        if self._fallback_stream and hasattr(self._fallback_stream, 'write'):
            streams_to_try.append(self._fallback_stream)
        
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
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()

# SafeStreamHandler 사용
safe_handler = SafeStreamHandler(sys.stdout, _original_stdout)
safe_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
safe_handler.setFormatter(formatter)
logger.addHandler(safe_handler)

# 상대 import 사용
from evaluation.test_search_quality_evaluation import SearchQualityEvaluator, TEST_QUERIES


def format_progress_bar(current: int, total: int, width: int = 30) -> str:
    """진행 바 포맷팅"""
    filled = int(width * current / total)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}]"


async def quick_test():
    """빠른 테스트 실행"""
    logger.info("=" * 60)
    logger.info("Quick Evaluation Test")
    logger.info("=" * 60)
    
    # 소수의 테스트 쿼리 선택 (각 유형별 2개씩)
    test_queries = [
        {"query": TEST_QUERIES["statute_article"][0], "type": "statute_article"},
        {"query": TEST_QUERIES["statute_article"][1], "type": "statute_article"},
        {"query": TEST_QUERIES["precedent"][0], "type": "precedent"},
        {"query": TEST_QUERIES["precedent"][1], "type": "precedent"},
        {"query": TEST_QUERIES["procedure"][0], "type": "procedure"},
        {"query": TEST_QUERIES["general_question"][0], "type": "general_question"},
    ]
    
    total_queries = len(test_queries)
    logger.info(f"Testing with {total_queries} queries")
    logger.info("")
    
    # 개선 기능 활성화 상태로 테스트
    logger.info("Testing WITH improvements...")
    evaluator_with = SearchQualityEvaluator(enable_improvements=True)
    
    # 진행 상황 추적 변수
    start_time = time.time()
    query_times = []
    
    for i, test_query in enumerate(test_queries):
        query = test_query["query"]
        query_type = test_query["type"]
        
        # 진행률 계산
        progress_pct = ((i + 1) / total_queries) * 100
        
        # 경과 시간 및 예상 남은 시간 계산
        elapsed_time = time.time() - start_time
        if i > 0:
            avg_time_per_query = elapsed_time / i
            remaining_queries = total_queries - (i + 1)
            estimated_remaining = avg_time_per_query * remaining_queries
        else:
            estimated_remaining = 0
        
        # 진행 상황 표시
        logger.info("")
        logger.info("-" * 60)
        progress_bar = format_progress_bar(i + 1, total_queries)
        logger.info(f"{progress_bar} {i+1}/{total_queries} ({progress_pct:.1f}%)")
        logger.info(f"[경과 시간] {elapsed_time:.1f}초")
        if estimated_remaining > 0:
            logger.info(f"[예상 남은 시간] {estimated_remaining:.1f}초")
        logger.info("-" * 60)
        logger.info(f"[{i+1}/{total_queries}] Query: {query[:60]}...")
        logger.info(f"Type: {query_type}")
        
        query_start_time = time.time()
        
        try:
            metrics = await evaluator_with.evaluate_query_async(
                query=query,
                query_type=query_type
            )
            
            query_elapsed = time.time() - query_start_time
            query_times.append(query_elapsed)
            
            if "error" not in metrics:
                logger.info(f"  ✓ Success ({query_elapsed:.2f}초)")
                logger.info(f"  - Result count: {metrics.get('result_count', 0)}")
                logger.info(f"  - Response time: {metrics.get('response_time', 0):.2f}s")
                logger.info(f"  - Precision@5: {metrics.get('precision_at_5', 0):.3f}")
                logger.info(f"  - Keyword Coverage: {metrics.get('keyword_coverage', 0):.3f}")
                logger.info(f"  - Diversity Score: {metrics.get('diversity_score', 0):.3f}")
            else:
                logger.error(f"  ✗ Error: {metrics.get('error')}")
        except Exception as e:
            query_elapsed = time.time() - query_start_time
            query_times.append(query_elapsed)
            logger.error(f"  ✗ Exception: {e} ({query_elapsed:.2f}초)")
    
    # 최종 요약
    total_time = time.time() - start_time
    avg_query_time = sum(query_times) / len(query_times) if query_times else 0
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Quick test completed!")
    logger.info("=" * 60)
    logger.info(f"총 소요 시간: {total_time:.2f}초")
    logger.info(f"평균 쿼리 처리 시간: {avg_query_time:.2f}초")
    logger.info(f"처리된 쿼리 수: {len(query_times)}/{total_queries}")
    if query_times:
        logger.info(f"최소 처리 시간: {min(query_times):.2f}초")
        logger.info(f"최대 처리 시간: {max(query_times):.2f}초")
    logger.info("=" * 60)


def main():
    """메인 실행 함수"""
    try:
        asyncio.run(quick_test())
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()

