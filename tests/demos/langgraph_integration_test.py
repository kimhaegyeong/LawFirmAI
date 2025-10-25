# -*- coding: utf-8 -*-
"""
LangGraph 통합 테스트
LangGraph 기반 법률 AI 시스템의 통합 테스트
"""

import asyncio
import time
import logging
from typing import Dict, Any, List
from datetime import datetime

# 테스트용 설정
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from source.utils.config import Config
from source.services.enhanced_chat_service import EnhancedChatService
from source.services.langgraph.integrated_workflow_service import IntegratedWorkflowService

logger = logging.getLogger(__name__)


class LangGraphIntegrationTest:
    """LangGraph 통합 테스트 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self.test_results = []
        
        # 테스트 케이스 정의
        self.test_cases = [
            {
                "name": "기본 법률 질문",
                "query": "계약서 작성 방법을 알려주세요",
                "expected_features": ["langgraph_enabled", "workflow_steps"],
                "timeout": 30
            },
            {
                "name": "법률 조문 질문",
                "query": "민법 제750조에 대해 설명해주세요",
                "expected_features": ["langgraph_enabled", "workflow_steps", "sources"],
                "timeout": 30
            },
            {
                "name": "복잡한 법률 질문",
                "query": "이혼 시 재산분할과 양육권에 대한 법적 절차를 자세히 알려주세요",
                "expected_features": ["langgraph_enabled", "workflow_steps", "quality_metrics"],
                "timeout": 45
            },
            {
                "name": "판례 검색 질문",
                "query": "부동산 매매계약 관련 최근 판례를 찾아주세요",
                "expected_features": ["langgraph_enabled", "workflow_steps", "precedent_references"],
                "timeout": 40
            },
            {
                "name": "계약서 검토 질문",
                "query": "근로계약서 검토 시 주의사항을 알려주세요",
                "expected_features": ["langgraph_enabled", "workflow_steps", "legal_analysis"],
                "timeout": 35
            }
        ]
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        self.logger.info("Starting LangGraph integration tests...")
        start_time = time.time()
        
        # Enhanced Chat Service 초기화
        try:
            self.chat_service = EnhancedChatService(self.config)
            self.logger.info("Enhanced Chat Service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Enhanced Chat Service: {e}")
            return {"success": False, "error": f"Service initialization failed: {e}"}
        
        # 테스트 실행
        test_results = []
        for i, test_case in enumerate(self.test_cases):
            self.logger.info(f"Running test {i+1}/{len(self.test_cases)}: {test_case['name']}")
            
            result = await self._run_single_test(test_case)
            test_results.append(result)
            
            # 테스트 간 간격
            await asyncio.sleep(1)
        
        # 결과 분석
        total_time = time.time() - start_time
        summary = self._analyze_test_results(test_results, total_time)
        
        self.logger.info(f"All tests completed in {total_time:.2f} seconds")
        return summary
    
    async def _run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """단일 테스트 실행"""
        test_start_time = time.time()
        
        try:
            # 테스트 실행
            result = await asyncio.wait_for(
                self.chat_service.process_message(
                    message=test_case["query"],
                    session_id=f"test_session_{int(time.time())}",
                    user_id=f"test_user_{int(time.time())}"
                ),
                timeout=test_case["timeout"]
            )
            
            test_duration = time.time() - test_start_time
            
            # 결과 검증
            validation_result = self._validate_test_result(result, test_case)
            
            return {
                "test_name": test_case["name"],
                "query": test_case["query"],
                "success": validation_result["success"],
                "duration": test_duration,
                "result": result,
                "validation": validation_result,
                "timestamp": datetime.now().isoformat()
            }
            
        except asyncio.TimeoutError:
            return {
                "test_name": test_case["name"],
                "query": test_case["query"],
                "success": False,
                "duration": test_case["timeout"],
                "error": "Test timeout",
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            test_duration = time.time() - test_start_time
            return {
                "test_name": test_case["name"],
                "query": test_case["query"],
                "success": False,
                "duration": test_duration,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_test_result(self, result: Dict[str, Any], test_case: Dict[str, Any]) -> Dict[str, Any]:
        """테스트 결과 검증"""
        validation_result = {
            "success": True,
            "checks": [],
            "errors": []
        }
        
        # 기본 응답 검증
        if not result.get("response"):
            validation_result["success"] = False
            validation_result["errors"].append("No response generated")
        else:
            validation_result["checks"].append("Response generated")
        
        # LangGraph 활성화 검증
        if result.get("langgraph_enabled"):
            validation_result["checks"].append("LangGraph enabled")
        else:
            validation_result["errors"].append("LangGraph not enabled")
        
        # 워크플로우 단계 검증
        workflow_steps = result.get("workflow_steps", [])
        if workflow_steps:
            validation_result["checks"].append(f"Workflow steps: {len(workflow_steps)}")
        else:
            validation_result["errors"].append("No workflow steps found")
        
        # 예상 기능 검증
        for feature in test_case.get("expected_features", []):
            if feature in result:
                validation_result["checks"].append(f"Feature '{feature}' present")
            else:
                validation_result["errors"].append(f"Expected feature '{feature}' not found")
        
        # 응답 품질 검증
        response_length = len(result.get("response", ""))
        if response_length < 50:
            validation_result["errors"].append(f"Response too short: {response_length} characters")
        else:
            validation_result["checks"].append(f"Response length: {response_length} characters")
        
        # 신뢰도 검증
        confidence = result.get("confidence", 0)
        if confidence < 0.1:
            validation_result["errors"].append(f"Low confidence: {confidence}")
        else:
            validation_result["checks"].append(f"Confidence: {confidence}")
        
        # 처리 시간 검증
        processing_time = result.get("processing_time", 0)
        if processing_time > test_case["timeout"]:
            validation_result["errors"].append(f"Processing time exceeded timeout: {processing_time}s")
        else:
            validation_result["checks"].append(f"Processing time: {processing_time:.2f}s")
        
        # 전체 성공 여부 결정
        if validation_result["errors"]:
            validation_result["success"] = False
        
        return validation_result
    
    def _analyze_test_results(self, test_results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """테스트 결과 분석"""
        total_tests = len(test_results)
        successful_tests = sum(1 for result in test_results if result.get("success", False))
        failed_tests = total_tests - successful_tests
        
        # 성능 통계
        durations = [result.get("duration", 0) for result in test_results]
        avg_duration = sum(durations) / len(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        
        # LangGraph 사용 통계
        langgraph_enabled_count = sum(1 for result in test_results 
                                    if result.get("result", {}).get("langgraph_enabled", False))
        
        # 워크플로우 단계 통계
        workflow_steps_counts = [len(result.get("result", {}).get("workflow_steps", [])) 
                               for result in test_results]
        avg_workflow_steps = sum(workflow_steps_counts) / len(workflow_steps_counts) if workflow_steps_counts else 0
        
        # 오류 분석
        errors = []
        for result in test_results:
            if not result.get("success", True):
                error_info = {
                    "test_name": result.get("test_name", "Unknown"),
                    "error": result.get("error", "Unknown error"),
                    "validation_errors": result.get("validation", {}).get("errors", [])
                }
                errors.append(error_info)
        
        # 성능 점수 계산
        performance_score = self._calculate_performance_score(test_results)
        
        summary = {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "total_execution_time": total_time
            },
            "performance_stats": {
                "average_duration": avg_duration,
                "max_duration": max_duration,
                "min_duration": min_duration,
                "performance_score": performance_score
            },
            "langgraph_stats": {
                "langgraph_enabled_count": langgraph_enabled_count,
                "langgraph_usage_rate": langgraph_enabled_count / total_tests if total_tests > 0 else 0,
                "average_workflow_steps": avg_workflow_steps
            },
            "errors": errors,
            "detailed_results": test_results,
            "recommendations": self._generate_recommendations(test_results)
        }
        
        return summary
    
    def _calculate_performance_score(self, test_results: List[Dict[str, Any]]) -> float:
        """성능 점수 계산 (0-100)"""
        if not test_results:
            return 0.0
        
        score = 100.0
        
        # 성공률 점수 (40% 가중치)
        success_rate = sum(1 for result in test_results if result.get("success", False)) / len(test_results)
        success_score = success_rate * 40
        score = min(score, success_score)
        
        # 응답 시간 점수 (30% 가중치)
        avg_duration = sum(result.get("duration", 0) for result in test_results) / len(test_results)
        if avg_duration > 30:  # 30초 이상이면 감점
            duration_score = max(0, 30 - (avg_duration - 30) * 0.5)
        else:
            duration_score = 30
        score = min(score, duration_score)
        
        # LangGraph 사용률 점수 (20% 가중치)
        langgraph_rate = sum(1 for result in test_results 
                           if result.get("result", {}).get("langgraph_enabled", False)) / len(test_results)
        langgraph_score = langgraph_rate * 20
        score = min(score, langgraph_score)
        
        # 응답 품질 점수 (10% 가중치)
        quality_score = 0
        for result in test_results:
            response_length = len(result.get("result", {}).get("response", ""))
            confidence = result.get("result", {}).get("confidence", 0)
            if response_length > 50 and confidence > 0.1:
                quality_score += 1
        quality_score = (quality_score / len(test_results)) * 10
        score = min(score, quality_score)
        
        return max(0, score)
    
    def _generate_recommendations(self, test_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """개선 권장사항 생성"""
        recommendations = []
        
        # 성공률 분석
        success_rate = sum(1 for result in test_results if result.get("success", False)) / len(test_results)
        if success_rate < 0.8:
            recommendations.append({
                "type": "reliability",
                "priority": "high",
                "title": "테스트 성공률 개선 필요",
                "description": f"현재 성공률이 {success_rate:.2%}입니다.",
                "suggestions": [
                    "오류 처리 로직 강화",
                    "입력 검증 개선",
                    "예외 상황 대응 강화"
                ]
            })
        
        # 응답 시간 분석
        avg_duration = sum(result.get("duration", 0) for result in test_results) / len(test_results)
        if avg_duration > 30:
            recommendations.append({
                "type": "performance",
                "priority": "high",
                "title": "응답 시간 최적화 필요",
                "description": f"평균 응답 시간이 {avg_duration:.2f}초입니다.",
                "suggestions": [
                    "워크플로우 최적화",
                    "캐싱 전략 개선",
                    "병렬 처리 도입"
                ]
            })
        
        # LangGraph 사용률 분석
        langgraph_rate = sum(1 for result in test_results 
                           if result.get("result", {}).get("langgraph_enabled", False)) / len(test_results)
        if langgraph_rate < 0.9:
            recommendations.append({
                "type": "integration",
                "priority": "medium",
                "title": "LangGraph 통합 개선 필요",
                "description": f"LangGraph 사용률이 {langgraph_rate:.2%}입니다.",
                "suggestions": [
                    "LangGraph 서비스 안정성 개선",
                    "폴백 메커니즘 최적화",
                    "워크플로우 구성 검토"
                ]
            })
        
        # 응답 품질 분석
        low_quality_count = 0
        for result in test_results:
            response_length = len(result.get("result", {}).get("response", ""))
            confidence = result.get("result", {}).get("confidence", 0)
            if response_length < 50 or confidence < 0.1:
                low_quality_count += 1
        
        if low_quality_count > len(test_results) * 0.2:  # 20% 이상
            recommendations.append({
                "type": "quality",
                "priority": "medium",
                "title": "응답 품질 개선 필요",
                "description": f"{low_quality_count}개 테스트에서 낮은 품질 응답이 발생했습니다.",
                "suggestions": [
                    "응답 생성 로직 개선",
                    "품질 검증 강화",
                    "템플릿 시스템 최적화"
                ]
            })
        
        # 기본 권장사항
        if not recommendations:
            recommendations.append({
                "type": "info",
                "priority": "low",
                "title": "테스트 결과 양호",
                "description": "모든 테스트가 성공적으로 통과했습니다.",
                "suggestions": [
                    "정기적인 테스트 실행 유지",
                    "새로운 기능 추가 시 테스트 케이스 확장"
                ]
            })
        
        return recommendations


async def run_langgraph_integration_test():
    """LangGraph 통합 테스트 실행 함수"""
    test_runner = LangGraphIntegrationTest()
    results = await test_runner.run_all_tests()
    
    # 결과 출력
    print("\n" + "="*70)
    print("LangGraph 통합 테스트 결과")
    print("="*70)
    
    summary = results["test_summary"]
    print(f"총 테스트: {summary['total_tests']}")
    print(f"성공: {summary['successful_tests']}")
    print(f"실패: {summary['failed_tests']}")
    print(f"성공률: {summary['success_rate']:.2%}")
    print(f"총 실행 시간: {summary['total_execution_time']:.2f}초")
    
    performance = results["performance_stats"]
    print(f"\n성능 통계:")
    print(f"평균 응답 시간: {performance['average_duration']:.2f}초")
    print(f"최대 응답 시간: {performance['max_duration']:.2f}초")
    print(f"최소 응답 시간: {performance['min_duration']:.2f}초")
    print(f"성능 점수: {performance['performance_score']:.1f}/100")
    
    langgraph_stats = results["langgraph_stats"]
    print(f"\nLangGraph 통계:")
    print(f"LangGraph 사용 횟수: {langgraph_stats['langgraph_enabled_count']}")
    print(f"LangGraph 사용률: {langgraph_stats['langgraph_usage_rate']:.2%}")
    print(f"평균 워크플로우 단계: {langgraph_stats['average_workflow_steps']:.1f}")
    
    if results["errors"]:
        print(f"\n오류 목록:")
        for error in results["errors"]:
            print(f"- {error['test_name']}: {error['error']}")
    
    print(f"\n권장사항:")
    for rec in results["recommendations"]:
        print(f"- [{rec['priority'].upper()}] {rec['title']}")
        print(f"  {rec['description']}")
        for suggestion in rec['suggestions']:
            print(f"  • {suggestion}")
        print()
    
    return results


if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(run_langgraph_integration_test())
