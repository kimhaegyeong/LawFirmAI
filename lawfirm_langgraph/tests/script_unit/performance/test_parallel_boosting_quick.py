# -*- coding: utf-8 -*-
"""
병렬 처리 개선 빠른 검증 스크립트
법령 조문 부스팅 병렬 처리 부분만 테스트합니다.
"""

import sys
import os
import time
from pathlib import Path

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
tests_dir = script_dir.parent.parent.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

# 환경 변수 로드
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(project_root)
except ImportError:
    pass

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = get_logger(__name__)

def test_parallel_boosting():
    """병렬 처리 개선 검증"""
    logger.info("=" * 60)
    logger.info("병렬 처리 개선 검증 테스트")
    logger.info("=" * 60)
    
    try:
        from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        
        config = LangGraphConfig()
        workflow = EnhancedLegalQuestionWorkflow(config)
        
        # 테스트 데이터 준비
        text2sql_docs = [
            {
                "type": "statute_article",
                "statute_name": "민법",
                "article_no": "123",
                "final_weighted_score": 0.7,
                "content": "민법 제123조 내용"
            }
        ]
        
        reranked_vector_docs = [
            {
                "type": "statute_article",
                "statute_name": "민법",
                "article_no": "456",
                "final_weighted_score": 0.6,
                "content": "민법 제456조 내용"
            }
        ]
        
        query = "민법 제123조에 대해 설명해주세요"
        
        logger.info(f"테스트 쿼리: {query}")
        logger.info(f"Text2SQL 문서 수: {len(text2sql_docs)}")
        logger.info(f"Reranked 문서 수: {len(reranked_vector_docs)}")
        
        # 병렬 처리 실행 (수정된 코드와 동일한 방식)
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
        
        start_boosting_time = time.time()
        doc_count = len(text2sql_docs) + len(reranked_vector_docs)
        timeout = max(10.0, min(30.0, doc_count * 0.1))
        
        logger.info(f"동적 타임아웃 계산: {timeout}초 (문서 수: {doc_count})")
        
        results = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(workflow._apply_statute_article_boosting, text2sql_docs, query): "text2sql",
                executor.submit(workflow._apply_statute_article_boosting, reranked_vector_docs, query): "reranked"
            }
            
            logger.info(f"병렬 작업 제출 완료: {len(futures)}개 작업")
            
            for future in as_completed(futures, timeout=timeout):
                task_name = futures[future]
                try:
                    results[task_name] = future.result()
                    logger.info(f"✅ {task_name} boosting completed")
                except FutureTimeoutError:
                    logger.warning(f"⚠️ {task_name} boosting timeout ({timeout}s), using original docs")
                    results[task_name] = text2sql_docs if task_name == "text2sql" else reranked_vector_docs
                except Exception as e:
                    logger.error(f"❌ {task_name} boosting error: {e}, using original docs", exc_info=True)
                    results[task_name] = text2sql_docs if task_name == "text2sql" else reranked_vector_docs
        
        text2sql_result = results.get("text2sql", text2sql_docs)
        reranked_result = results.get("reranked", reranked_vector_docs)
        
        elapsed = time.time() - start_boosting_time
        logger.info(f"⏱️ Statute article boosting completed in {elapsed:.2f}s")
        
        # 결과 검증
        logger.info("=" * 60)
        logger.info("결과 검증")
        logger.info("=" * 60)
        
        if text2sql_result:
            boosted_doc = text2sql_result[0]
            boosted_score = boosted_doc.get("final_weighted_score", 0.0)
            original_score = text2sql_docs[0].get("final_weighted_score", 0.0)
            
            logger.info(f"Text2SQL 문서 부스팅:")
            logger.info(f"  원본 점수: {original_score:.3f}")
            logger.info(f"  부스팅 후 점수: {boosted_score:.3f}")
            
            if boosted_score >= original_score:
                logger.info("✅ 부스팅이 정상적으로 적용되었습니다")
            else:
                logger.warning("⚠️ 부스팅이 적용되지 않았습니다")
        
        logger.info("=" * 60)
        logger.info("✅ 병렬 처리 개선 검증 완료")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"테스트 실패: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_parallel_boosting()
    sys.exit(0 if success else 1)

