# -*- coding: utf-8 -*-
"""
분류 성능 테스트 스크립트
단일 통합 프롬프트 vs 체인 방식 비교
"""

import sys
import os
import time
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent.parent
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(lawfirm_langgraph_path))

from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
from lawfirm_langgraph.core.classification.handlers.classification_handler import ClassificationHandler
from lawfirm_langgraph.core.workflow.utils.workflow_config import WorkflowConfig
from langchain_openai import ChatOpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationPerformanceTester:
    """분류 성능 테스트 클래스"""
    
    def __init__(self):
        self.config = WorkflowConfig()
        self.llm = ChatOpenAI(
            model_name=self.config.llm_model_name,
            temperature=0.1,
            max_tokens=500
        )
        self.llm_fast = ChatOpenAI(
            model_name=self.config.llm_fast_model_name if hasattr(self.config, 'llm_fast_model_name') else self.config.llm_model_name,
            temperature=0.1,
            max_tokens=500
        )
        self.classification_handler = ClassificationHandler(
            llm=self.llm,
            llm_fast=self.llm_fast,
            logger=logger
        )
        
        # 테스트 쿼리
        self.test_queries = [
            "계약 해지 사유에 대해 알려주세요",
            "민법 제123조의 내용을 알려주세요",
            "이혼 절차는 어떻게 되나요?",
            "손해배상과 위약금의 차이는 무엇인가요?",
            "판례를 찾아주세요",
            "계약이란 무엇인가요?",
            "안녕하세요",
            "특허 침해 시 대응 방법은?",
            "임대차 계약서 작성 시 주의사항",
            "형법 제250조에 대해 설명해주세요"
        ]
    
    def test_unified_classification(self, query: str) -> Dict[str, Any]:
        """단일 통합 프롬프트 방식 테스트"""
        start_time = time.time()
        try:
            result = self.classification_handler.classify_query_and_complexity_with_llm(query)
            elapsed_time = time.time() - start_time
            
            return {
                "success": True,
                "elapsed_time": elapsed_time,
                "llm_calls": 1,
                "question_type": result[0].value if hasattr(result[0], 'value') else str(result[0]),
                "confidence": result[1],
                "complexity": result[2].value if hasattr(result[2], 'value') else str(result[2]),
                "needs_search": result[3],
                "error": None
            }
        except Exception as e:
            elapsed_time = time.time() - start_time
            return {
                "success": False,
                "elapsed_time": elapsed_time,
                "llm_calls": 1,
                "error": str(e)
            }
    
    def run_performance_test(self) -> Dict[str, Any]:
        """성능 테스트 실행"""
        results = {
            "unified": [],
            "summary": {}
        }
        
        total_time = 0
        total_llm_calls = 0
        success_count = 0
        
        logger.info("=" * 80)
        logger.info("분류 성능 테스트 시작 (단일 통합 프롬프트)")
        logger.info("=" * 80)
        
        for i, query in enumerate(self.test_queries, 1):
            logger.info(f"\n[{i}/{len(self.test_queries)}] 테스트 쿼리: {query}")
            
            result = self.test_unified_classification(query)
            results["unified"].append({
                "query": query,
                **result
            })
            
            if result["success"]:
                total_time += result["elapsed_time"]
                total_llm_calls += result["llm_calls"]
                success_count += 1
                logger.info(
                    f"✅ 성공: {result['elapsed_time']:.3f}s, "
                    f"LLM 호출: {result['llm_calls']}회, "
                    f"유형: {result['question_type']}, "
                    f"복잡도: {result['complexity']}"
                )
            else:
                logger.error(f"❌ 실패: {result['error']}")
        
        # 요약 통계
        if success_count > 0:
            avg_time = total_time / success_count
            avg_llm_calls = total_llm_calls / success_count
        else:
            avg_time = 0
            avg_llm_calls = 0
        
        results["summary"] = {
            "total_queries": len(self.test_queries),
            "success_count": success_count,
            "failure_count": len(self.test_queries) - success_count,
            "total_time": total_time,
            "avg_time": avg_time,
            "total_llm_calls": total_llm_calls,
            "avg_llm_calls": avg_llm_calls,
            "method": "unified_single_prompt"
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("테스트 결과 요약")
        logger.info("=" * 80)
        logger.info(f"총 쿼리 수: {results['summary']['total_queries']}")
        logger.info(f"성공: {results['summary']['success_count']}")
        logger.info(f"실패: {results['summary']['failure_count']}")
        logger.info(f"총 소요 시간: {results['summary']['total_time']:.3f}s")
        logger.info(f"평균 소요 시간: {results['summary']['avg_time']:.3f}s")
        logger.info(f"총 LLM 호출: {results['summary']['total_llm_calls']}회")
        logger.info(f"평균 LLM 호출: {results['summary']['avg_llm_calls']:.2f}회/쿼리")
        logger.info("=" * 80)
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_file: str = None):
        """결과 저장"""
        if output_file is None:
            output_file = f"classification_performance_{int(time.time())}.json"
        
        output_path = Path(__file__).parent / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n결과 저장: {output_path}")


def main():
    """메인 함수"""
    tester = ClassificationPerformanceTester()
    results = tester.run_performance_test()
    tester.save_results(results)
    
    return results


if __name__ == "__main__":
    main()

