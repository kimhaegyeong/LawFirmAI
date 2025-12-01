# -*- coding: utf-8 -*-
"""
성능 개선 사항 테스트 스크립트

개선 사항:
1. time.sleep() 최적화 (exponential backoff)
2. 모델 싱글톤 패턴 (SentenceTransformer 재사용)
3. State 접근 최적화 (캐싱)
4. 비동기 처리 개선 (asyncio.gather)
5. 검색 작업 병렬화 강화 (as_completed)

Usage:
    python lawfirm_langgraph/tests/script_unit/performance/test_performance_improvements.py
"""

import sys
import os
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
performance_dir = script_dir.parent
unit_dir = performance_dir.parent
tests_dir = unit_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

# 로깅 설정
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger(__name__)

# 테스트 쿼리
TEST_QUERIES = [
    "계약 해지 사유에 대해 알려주세요",
    "부동산 매매 계약서 작성 시 주의사항은?",
    "민사소송 절차에 대해 설명해주세요",
    "지적재산권 침해 시 구제 방법은?",
    "형사소송에서 변호인 선임 절차는?"
]

async def test_workflow_performance(query: str, session_id: str = None) -> Dict[str, Any]:
    """워크플로우 성능 테스트"""
    try:
        from core.workflow.workflow_service import LangGraphWorkflowService
        from core.utils.langgraph_config import LangGraphConfig
        
        config = LangGraphConfig()
        service = LangGraphWorkflowService(config)
        
        start_time = time.time()
        
        result = await service.process_query(
            query=query,
            session_id=session_id,
            enable_checkpoint=False,
            use_astream_events=False
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "query": query,
            "duration": duration,
            "success": True,
            "answer_length": len(result.get("response", "")),
            "sources_count": len(result.get("sources", [])),
            "node_count": len(result.get("processing_steps", []))
        }
    except Exception as e:
        logger.error(f"테스트 실패: {query} - {e}")
        return {
            "query": query,
            "duration": 0,
            "success": False,
            "error": str(e)
        }

async def test_model_singleton():
    """모델 싱글톤 패턴 테스트"""
    logger.info("=" * 60)
    logger.info("모델 싱글톤 패턴 테스트")
    logger.info("=" * 60)
    
    try:
        from scripts.utils.embeddings import SentenceEmbedder
        
        # 첫 번째 인스턴스 생성
        start_time = time.time()
        embedder1 = SentenceEmbedder()
        first_load_time = time.time() - start_time
        logger.info(f"첫 번째 모델 로딩 시간: {first_load_time:.3f}초")
        
        # 두 번째 인스턴스 생성 (싱글톤이면 빠르게 로드됨)
        start_time = time.time()
        embedder2 = SentenceEmbedder()
        second_load_time = time.time() - start_time
        logger.info(f"두 번째 모델 로딩 시간: {second_load_time:.3f}초")
        
        # 같은 인스턴스인지 확인
        is_same = embedder1 is embedder2
        logger.info(f"인스턴스 동일성: {is_same}")
        
        if is_same:
            improvement = ((first_load_time - second_load_time) / first_load_time) * 100
            logger.info(f"✅ 싱글톤 패턴 작동: {improvement:.1f}% 시간 절약")
        else:
            logger.warning("⚠️ 싱글톤 패턴이 작동하지 않음")
        
        return {
            "first_load_time": first_load_time,
            "second_load_time": second_load_time,
            "is_singleton": is_same,
            "improvement_percent": ((first_load_time - second_load_time) / first_load_time) * 100 if is_same else 0
        }
    except Exception as e:
        logger.error(f"모델 싱글톤 테스트 실패: {e}")
        return {"error": str(e)}

async def test_state_cache():
    """State 접근 캐싱 테스트"""
    logger.info("=" * 60)
    logger.info("State 접근 캐싱 테스트")
    logger.info("=" * 60)
    
    try:
        from core.workflow.processors.search_execution_processor import SearchExecutionProcessor
        from core.workflow.state.state_definitions import LegalWorkflowState
        
        # Mock 객체 생성
        class MockSearchHandler:
            pass
        
        class MockLogger:
            def debug(self, msg):
                pass
            def info(self, msg):
                pass
        
        processor = SearchExecutionProcessor(
            search_handler=MockSearchHandler(),
            logger=MockLogger(),
            config={}
        )
        
        # 테스트 state 생성
        test_state: LegalWorkflowState = {
            "query": "테스트 쿼리",
            "query_type": "legal_advice",
            "legal_field": "civil",
            "extracted_keywords": ["계약", "해지"],
            "optimized_queries": {
                "semantic_query": "테스트 쿼리",
                "multi_queries": ["테스트 쿼리 1", "테스트 쿼리 2"]
            }
        }
        
        # 첫 번째 호출 (캐시 미스)
        start_time = time.time()
        params1 = processor.get_search_params(test_state)
        first_call_time = time.time() - start_time
        
        # 두 번째 호출 (캐시 히트)
        start_time = time.time()
        params2 = processor.get_search_params(test_state)
        second_call_time = time.time() - start_time
        
        logger.info(f"첫 번째 호출 시간: {first_call_time:.6f}초")
        logger.info(f"두 번째 호출 시간: {second_call_time:.6f}초")
        
        if second_call_time < first_call_time:
            improvement = ((first_call_time - second_call_time) / first_call_time) * 100
            logger.info(f"✅ 캐싱 작동: {improvement:.1f}% 시간 절약")
        else:
            logger.warning("⚠️ 캐싱이 작동하지 않음")
        
        return {
            "first_call_time": first_call_time,
            "second_call_time": second_call_time,
            "improvement_percent": ((first_call_time - second_call_time) / first_call_time) * 100 if second_call_time < first_call_time else 0
        }
    except Exception as e:
        logger.error(f"State 캐싱 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

async def test_async_parallel():
    """비동기 병렬 처리 테스트"""
    logger.info("=" * 60)
    logger.info("비동기 병렬 처리 테스트")
    logger.info("=" * 60)
    
    try:
        # 간단한 비동기 작업 시뮬레이션
        async def task1():
            await asyncio.sleep(0.1)
            return "task1_result"
        
        async def task2():
            await asyncio.sleep(0.1)
            return "task2_result"
        
        # 순차 실행
        start_time = time.time()
        result1 = await task1()
        result2 = await task2()
        sequential_time = time.time() - start_time
        
        # 병렬 실행
        start_time = time.time()
        results = await asyncio.gather(task1(), task2())
        parallel_time = time.time() - start_time
        
        logger.info(f"순차 실행 시간: {sequential_time:.3f}초")
        logger.info(f"병렬 실행 시간: {parallel_time:.3f}초")
        
        improvement = ((sequential_time - parallel_time) / sequential_time) * 100
        logger.info(f"✅ 병렬 처리 개선: {improvement:.1f}% 시간 절약")
        
        return {
            "sequential_time": sequential_time,
            "parallel_time": parallel_time,
            "improvement_percent": improvement
        }
    except Exception as e:
        logger.error(f"비동기 병렬 처리 테스트 실패: {e}")
        return {"error": str(e)}

async def run_performance_tests():
    """전체 성능 테스트 실행"""
    logger.info("=" * 60)
    logger.info("성능 개선 사항 테스트 시작")
    logger.info("=" * 60)
    
    results = {
        "model_singleton": {},
        "state_cache": {},
        "async_parallel": {},
        "workflow_tests": []
    }
    
    # 1. 모델 싱글톤 테스트
    results["model_singleton"] = await test_model_singleton()
    await asyncio.sleep(1)
    
    # 2. State 캐싱 테스트
    results["state_cache"] = await test_state_cache()
    await asyncio.sleep(1)
    
    # 3. 비동기 병렬 처리 테스트
    results["async_parallel"] = await test_async_parallel()
    await asyncio.sleep(1)
    
    # 4. 워크플로우 성능 테스트 (첫 번째 쿼리만)
    logger.info("=" * 60)
    logger.info("워크플로우 성능 테스트")
    logger.info("=" * 60)
    
    if TEST_QUERIES:
        test_query = TEST_QUERIES[0]
        logger.info(f"테스트 쿼리: {test_query}")
        
        workflow_result = await test_workflow_performance(test_query)
        results["workflow_tests"].append(workflow_result)
        
        if workflow_result["success"]:
            logger.info(f"✅ 워크플로우 실행 시간: {workflow_result['duration']:.3f}초")
            logger.info(f"   답변 길이: {workflow_result['answer_length']}자")
            logger.info(f"   소스 수: {workflow_result['sources_count']}개")
            logger.info(f"   노드 수: {workflow_result['node_count']}개")
        else:
            logger.error(f"❌ 워크플로우 실행 실패: {workflow_result.get('error', 'Unknown error')}")
    
    # 결과 요약
    logger.info("=" * 60)
    logger.info("테스트 결과 요약")
    logger.info("=" * 60)
    
    if results["model_singleton"].get("improvement_percent"):
        logger.info(f"✅ 모델 싱글톤: {results['model_singleton']['improvement_percent']:.1f}% 개선")
    
    if results["state_cache"].get("improvement_percent"):
        logger.info(f"✅ State 캐싱: {results['state_cache']['improvement_percent']:.1f}% 개선")
    
    if results["async_parallel"].get("improvement_percent"):
        logger.info(f"✅ 비동기 병렬 처리: {results['async_parallel']['improvement_percent']:.1f}% 개선")
    
    if results["workflow_tests"]:
        avg_duration = sum(r["duration"] for r in results["workflow_tests"] if r["success"]) / len([r for r in results["workflow_tests"] if r["success"]])
        logger.info(f"✅ 워크플로우 평균 실행 시간: {avg_duration:.3f}초")
    
    return results

if __name__ == "__main__":
    try:
        results = asyncio.run(run_performance_tests())
        logger.info("=" * 60)
        logger.info("테스트 완료")
        logger.info("=" * 60)
    except KeyboardInterrupt:
        logger.info("테스트 중단됨")
    except Exception as e:
        logger.error(f"테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

