#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
대규모 테스트 실행 시스템
3000개의 테스트 질의를 효율적으로 실행하고 결과를 분석합니다.
"""

import sys
import os
import json
import time
import asyncio
import multiprocessing
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import traceback

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from source.services.multi_stage_validation_system import MultiStageValidationSystem
from source.services.chat_service import ChatService
from source.utils.config import Config

# --------------------
# 멀티프로세스 워커 전역 및 함수 (Windows 호환)
# --------------------
mp_multi_stage_system = None
mp_chat_service = None

def mp_init_worker(enable_chat: bool = False):
    """프로세스 워커 초기화 (모듈 전역)"""
    global mp_multi_stage_system, mp_chat_service
    try:
        # 환경 변수로 개선된 검증 시스템 사용 여부 선택
        use_improved = os.getenv("USE_IMPROVED_VALIDATION", "0") == "1"
        if use_improved:
            try:
                from source.services.improved_multi_stage_validation_system import ImprovedMultiStageValidationSystem as _VSys
                mp_multi_stage_system = _VSys()
            except Exception as _e:
                print(f"개선된 검증 시스템 초기화 실패, 기본 시스템으로 대체: {_e}")
                mp_multi_stage_system = MultiStageValidationSystem()
        else:
            mp_multi_stage_system = MultiStageValidationSystem()
        if enable_chat:
            config = Config()
            mp_chat_service = ChatService(config)
        else:
            mp_chat_service = None
    except Exception as e:
        print(f"워커 초기화 실패: {e}")
        mp_multi_stage_system = None
        mp_chat_service = None

def mp_process_query_worker(query_data: Dict[str, Any]) -> Dict[str, Any]:
    """단일 질의 처리(프로세스 워커용) - 직렬화 가능한 dict 반환"""
    from dataclasses import asdict as _asdict
    import time as _time
    global mp_multi_stage_system
    if mp_multi_stage_system is None:
        return {
            "query": query_data.get("query", ""),
            "category": query_data.get("category", ""),
            "subcategory": query_data.get("subcategory", ""),
            "expected_restricted": query_data.get("expected_restricted", False),
            "actual_restricted": False,
            "is_correct": False,
            "confidence": 0.0,
            "total_score": 0.0,
            "processing_time": 0.0,
            "error_message": "서비스 초기화 실패"
        }
    start_time = _time.time()
    try:
        validation_result = mp_multi_stage_system.validate(query_data["query"], category=query_data.get("category"))
        # 개선된/기존 검증기 모두 호환 처리
        if isinstance(validation_result, dict):
            final_decision = validation_result.get("final_decision", "restricted")
            confidence = float(validation_result.get("confidence", 0.0))
            total_score = float(validation_result.get("total_score", 0.0))
        else:
            final_decision = getattr(getattr(validation_result, "final_decision", None), "value", "restricted")
            confidence = float(getattr(validation_result, "confidence", 0.0))
            total_score = float(getattr(validation_result, "total_score", 0.0))
        actual_restricted = (final_decision == "restricted")
        is_correct = query_data["expected_restricted"] == actual_restricted
        processing_time = _time.time() - start_time
        return {
            "query": query_data["query"],
            "category": query_data["category"],
            "subcategory": query_data["subcategory"],
            "expected_restricted": query_data["expected_restricted"],
            "actual_restricted": actual_restricted,
            "is_correct": is_correct,
            "confidence": confidence,
            "total_score": total_score,
            "processing_time": processing_time,
            "error_message": None
        }
    except Exception as e:
        processing_time = _time.time() - start_time
        return {
            "query": query_data.get("query", ""),
            "category": query_data.get("category", ""),
            "subcategory": query_data.get("subcategory", ""),
            "expected_restricted": query_data.get("expected_restricted", False),
            "actual_restricted": False,
            "is_correct": False,
            "confidence": 0.0,
            "total_score": 0.0,
            "processing_time": processing_time,
            "error_message": str(e)
        }

@dataclass
class TestResult:
    """테스트 결과 데이터 클래스"""
    query: str
    category: str
    subcategory: str
    expected_restricted: bool
    actual_restricted: bool
    is_correct: bool
    confidence: float
    total_score: float
    processing_time: float
    error_message: Optional[str] = None
    stage_results: Optional[List[Dict]] = None
    chat_service_result: Optional[Dict] = None

@dataclass
class TestSummary:
    """테스트 요약 데이터 클래스"""
    total_tests: int
    correct_predictions: int
    incorrect_predictions: int
    overall_accuracy: float
    category_accuracies: Dict[str, float]
    processing_time: float
    error_count: int
    average_confidence: float
    average_score: float

class MassiveTestRunner:
    """대규모 테스트 실행기"""
    
    def __init__(self, max_workers: int = None, enable_chat: bool = False, store_details: bool = False):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.enable_chat = enable_chat
        self.store_details = store_details
        self.multi_stage_system = None
        self.chat_service = None
        self.results = []
        self.start_time = None
        self.end_time = None
        
    def initialize_services(self):
        """서비스 초기화"""
        print("🔧 서비스 초기화 중...")
        
        try:
            # 다단계 검증 시스템 초기화 (ML 통합 / 개선된 / 기본 순서)
            use_ml_integrated = os.getenv("USE_ML_INTEGRATED_VALIDATION", "0") == "1"
            use_improved = os.getenv("USE_IMPROVED_VALIDATION", "0") == "1"
            if use_ml_integrated:
                try:
                    from source.services.ml_integrated_validation_system import MLIntegratedValidationSystem as _VSys
                    self.multi_stage_system = _VSys()
                    print("  ✅ ML 통합 검증 시스템 초기화 완료")
                except Exception as _e:
                    print(f"  ⚠️ ML 통합 시스템 초기화 실패, 개선된 시스템으로 대체: {_e}")
                    use_improved = True
            if not use_ml_integrated and use_improved:
                try:
                    from source.services.improved_multi_stage_validation_system import ImprovedMultiStageValidationSystem as _VSys
                    self.multi_stage_system = _VSys()
                    print("  ✅ 개선된 다단계 검증 시스템 초기화 완료")
                except Exception as _e:
                    print(f"  ⚠️ 개선된 시스템 초기화 실패, 기본 시스템으로 대체: {_e}")
                    self.multi_stage_system = MultiStageValidationSystem()
            if not use_ml_integrated and not use_improved:
                self.multi_stage_system = MultiStageValidationSystem()
            print("  ✅ 다단계 검증 시스템 초기화 완료")
            
            # ChatService 초기화 (옵션)
            if self.enable_chat:
                config = Config()
                self.chat_service = ChatService(config)
                print("  ✅ ChatService 초기화 완료")
            
        except Exception as e:
            print(f"  ❌ 서비스 초기화 실패: {e}")
            raise
    
    def process_single_query(self, query_data: Dict[str, Any]) -> TestResult:
        """단일 질의 처리"""
        try:
            start_time = time.time()
            
            # 다단계 검증 수행
            validation_result = self.multi_stage_system.validate(query_data["query"], category=query_data.get("category"))
        
        # 개선된/기존 검증기 모두 호환 처리
            if isinstance(validation_result, dict):
                final_decision = validation_result.get("final_decision", "restricted")
                confidence = float(validation_result.get("confidence", 0.0))
                total_score = float(validation_result.get("total_score", 0.0))
            else:
                # 기존 객체형 결과
                final_decision = getattr(getattr(validation_result, "final_decision", None), "value", "restricted")
                confidence = float(getattr(validation_result, "confidence", 0.0))
                total_score = float(getattr(validation_result, "total_score", 0.0))
        
        # 실제 결과
            actual_restricted = (final_decision == "restricted")
        
        # 정확도 계산
            is_correct = query_data["expected_restricted"] == actual_restricted
            
            processing_time = time.time() - start_time
            
            # ChatService 통합 테스트 (옵션)
            chat_service_result = None
            if self.enable_chat and self.chat_service is not None:
                try:
                    chat_response = asyncio.run(self.chat_service.process_message(
                        message=query_data["query"],
                        user_id="test_user",
                        session_id="test_session"
                    ))
                    chat_service_result = {
                        "is_restricted": chat_response.get("restriction_info", {}).get("is_restricted", False),
                        "has_multi_stage_info": "multi_stage_validation" in chat_response.get("restriction_info", {}),
                        "response_length": len(chat_response.get("response", "")),
                        "success": True
                    }
                except Exception as e:
                    chat_service_result = {
                        "error": str(e),
                        "success": False
                    }
            
            # 단계별 결과 정리
            stage_results = []
            if self.store_details:
                if isinstance(validation_result, dict) and validation_result.get("stages"):
                    for stage in validation_result["stages"]:
                        # stage expected as dict in improved system
                        stage_results.append({
                            "stage": stage.get("stage"),
                            "result": stage.get("result"),
                            "score": stage.get("score"),
                            "reasoning": stage.get("reasoning")
                        })
                elif hasattr(validation_result, "stages") and validation_result.stages:
                    for stage in validation_result.stages:
                        stage_results.append({
                            "stage": stage.stage.value,
                            "result": stage.result.value,
                            "score": stage.score,
                            "reasoning": stage.reasoning
                        })
            
            return TestResult(
                query=query_data["query"],
                category=query_data["category"],
                subcategory=query_data["subcategory"],
                expected_restricted=query_data["expected_restricted"],
                actual_restricted=actual_restricted,
                is_correct=is_correct,
                confidence=confidence,
                total_score=total_score,
                processing_time=processing_time,
                stage_results=stage_results,
                chat_service_result=chat_service_result
            )
            
        except Exception as e:
            processing_time = time.time() - start_time if 'start_time' in locals() else 0
            return TestResult(
                query=query_data["query"],
                category=query_data["category"],
                subcategory=query_data["subcategory"],
                expected_restricted=query_data["expected_restricted"],
                actual_restricted=False,
                is_correct=False,
                confidence=0.0,
                total_score=0.0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def run_batch_test(self, queries: List[Dict[str, Any]], batch_size: int = 100) -> List[TestResult]:
        """배치 테스트 실행"""
        print(f"🚀 배치 테스트 시작 (총 {len(queries)}개 질의, 배치 크기: {batch_size})")
        
        all_results = []
        total_batches = (len(queries) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(queries), batch_size):
            batch_queries = queries[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            
            print(f"📦 배치 {batch_num}/{total_batches} 처리 중... ({len(batch_queries)}개 질의)")
            
            batch_start_time = time.time()
            
            # 멀티스레딩으로 배치 처리
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_query = {
                    executor.submit(self.process_single_query, query): query 
                    for query in batch_queries
                }
                
                batch_results = []
                for future in as_completed(future_to_query):
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as e:
                        query = future_to_query[future]
                        print(f"  ❌ 질의 처리 실패: {query['query'][:50]}... - {e}")
                        batch_results.append(TestResult(
                            query=query["query"],
                            category=query["category"],
                            subcategory=query["subcategory"],
                            expected_restricted=query["expected_restricted"],
                            actual_restricted=False,
                            is_correct=False,
                            confidence=0.0,
                            total_score=0.0,
                            processing_time=0.0,
                            error_message=str(e)
                        ))
            
            batch_time = time.time() - batch_start_time
            batch_correct = sum(1 for r in batch_results if r.is_correct)
            batch_accuracy = batch_correct / len(batch_results) if batch_results else 0
            
            print(f"  ✅ 배치 {batch_num} 완료: {batch_correct}/{len(batch_results)} 정확 ({batch_accuracy:.1%}, {batch_time:.2f}초)")
            
            all_results.extend(batch_results)
        
        return all_results
    
    def run_parallel_test(self, queries: List[Dict[str, Any]]) -> List[TestResult]:
        """병렬 테스트 실행"""
        print(f"🚀 병렬 테스트 시작 (총 {len(queries)}개 질의, 워커: {self.max_workers})")
        
        # 멀티프로세싱으로 병렬 처리 (Windows 호환: 모듈 수준 초기화자 사용)
        from functools import partial
        with ProcessPoolExecutor(max_workers=self.max_workers, initializer=mp_init_worker, initargs=(self.enable_chat,)) as executor:
            dict_results = list(executor.map(mp_process_query_worker, queries))
        
        # dict -> TestResult 변환
        results: List[TestResult] = []
        for d in dict_results:
            results.append(TestResult(
                query=d["query"],
                category=d["category"],
                subcategory=d["subcategory"],
                expected_restricted=d["expected_restricted"],
                actual_restricted=d["actual_restricted"],
                is_correct=d["is_correct"],
                confidence=d["confidence"],
                total_score=d["total_score"],
                processing_time=d["processing_time"],
                error_message=d.get("error_message")
            ))
        return results
    
    def run_sequential_test(self, queries: List[Dict[str, Any]]) -> List[TestResult]:
        """순차 테스트 실행"""
        print(f"🚀 순차 테스트 시작 (총 {len(queries)}개 질의)")
        
        results = []
        
        for i, query_data in enumerate(queries):
            if (i + 1) % 100 == 0:
                print(f"📊 진행률: {i + 1}/{len(queries)} ({(i + 1)/len(queries)*100:.1f}%)")
            
            result = self.process_single_query(query_data)
            results.append(result)
        
        return results
    
    def run_massive_test(self, queries: List[Dict[str, Any]], method: str = "batch", batch_size: int = 100) -> List[TestResult]:
        """대규모 테스트 실행"""
        print(f"🎯 대규모 테스트 시작 - 방법: {method}")
        print(f"📊 총 질의 수: {len(queries)}")
        
        self.start_time = time.time()
        
        # 서비스 초기화
        self.initialize_services()
        
        # 테스트 실행
        if method == "batch":
            results = self.run_batch_test(queries, batch_size=batch_size)
        elif method == "parallel":
            results = self.run_parallel_test(queries)
        elif method == "sequential":
            results = self.run_sequential_test(queries)
        else:
            raise ValueError(f"지원하지 않는 테스트 방법: {method}")
        
        self.end_time = time.time()
        self.results = results
        
        print(f"✅ 테스트 완료! 총 소요 시간: {self.end_time - self.start_time:.2f}초")
        
        return results
    
    def generate_summary(self) -> TestSummary:
        """테스트 요약 생성"""
        if not self.results:
            return TestSummary(
                total_tests=0, correct_predictions=0, incorrect_predictions=0,
                overall_accuracy=0.0, category_accuracies={}, processing_time=0.0,
                error_count=0, average_confidence=0.0, average_score=0.0
            )
        
        # 개인 법률 자문 카테고리 제외 플래그 (기본: 제외)
        exclude_personal_accuracy = os.getenv("EXCLUDE_PERSONAL_FROM_ACCURACY", "1") == "1"
        filtered_results = [r for r in self.results if not (exclude_personal_accuracy and r.category == "personal_legal_advice")]

        # 기본 통계 (필터링 적용)
        total_tests = len(filtered_results)
        correct_predictions = sum(1 for r in filtered_results if r.is_correct)
        incorrect_predictions = total_tests - correct_predictions
        overall_accuracy = correct_predictions / total_tests if total_tests > 0 else 0.0
        
        # 카테고리별 정확도
        category_stats = {}
        for result in filtered_results:
            category = result.category
            if category not in category_stats:
                category_stats[category] = {"correct": 0, "total": 0}
            
            category_stats[category]["total"] += 1
            if result.is_correct:
                category_stats[category]["correct"] += 1
        
        category_accuracies = {
            category: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            for category, stats in category_stats.items()
        }
        
        # 기타 통계
        error_count = sum(1 for r in filtered_results if r.error_message)
        average_confidence = sum(r.confidence for r in filtered_results) / total_tests if total_tests > 0 else 0.0
        average_score = sum(r.total_score for r in filtered_results) / total_tests if total_tests > 0 else 0.0
        processing_time = self.end_time - self.start_time if self.end_time and self.start_time else 0.0
        
        return TestSummary(
            total_tests=total_tests,
            correct_predictions=correct_predictions,
            incorrect_predictions=incorrect_predictions,
            overall_accuracy=overall_accuracy,
            category_accuracies=category_accuracies,
            processing_time=processing_time,
            error_count=error_count,
            average_confidence=average_confidence,
            average_score=average_score
        )
    
    def save_results(self, results: List[TestResult], summary: TestSummary, filename: str = None) -> str:
        """결과를 파일로 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results/massive_test_results_{timestamp}.json"
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 결과를 JSON 직렬화 가능한 형태로 변환
        results_data = []
        for result in results:
            result_dict = asdict(result)
            if not self.store_details:
                result_dict.pop("stage_results", None)
                result_dict.pop("chat_service_result", None)
            results_data.append(result_dict)
        
        # 요약 데이터
        summary_data = asdict(summary)
        
        # 전체 데이터
        full_data = {
            "metadata": {
                "test_run_at": datetime.now().isoformat(),
                "total_queries": len(results),
                "test_method": "massive_test",
                "processing_time": summary.processing_time
            },
            "summary": summary_data,
            "detailed_results": results_data
        }
        
        # 파일 저장
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, ensure_ascii=False, indent=2)
        
        print(f"📁 결과가 {filename}에 저장되었습니다.")
        return filename
    
    def generate_report(self, summary: TestSummary) -> str:
        """상세 보고서 생성"""
        report = []
        report.append("=" * 100)
        report.append("🎯 대규모 테스트 결과 보고서")
        report.append("=" * 100)
        
        # 전체 결과
        report.append(f"\n📊 전체 결과:")
        report.append(f"  총 테스트: {summary.total_tests:,}")
        report.append(f"  정확한 예측: {summary.correct_predictions:,}")
        report.append(f"  잘못된 예측: {summary.incorrect_predictions:,}")
        report.append(f"  전체 정확도: {summary.overall_accuracy:.1%}")
        report.append(f"  테스트 소요 시간: {summary.processing_time:.2f}초")
        report.append(f"  평균 처리 시간: {summary.processing_time/summary.total_tests*1000:.2f}ms/질의")
        report.append(f"  오류 발생: {summary.error_count}개")
        report.append(f"  평균 신뢰도: {summary.average_confidence:.2f}")
        report.append(f"  평균 점수: {summary.average_score:.2f}")
        
        # 카테고리별 상세 결과
        report.append(f"\n📋 카테고리별 정확도:")
        # 환경 플래그: 개인 카테고리 표시 여부 및 비어있는 카테고리 표시 여부
        show_personal_in_report = os.getenv("SHOW_PERSONAL_IN_REPORT", "0") == "1"
        show_empty_categories = os.getenv("SHOW_EMPTY_CATEGORIES", "0") == "1"

        # 표시할 카테고리 집합 구성
        categories_to_show = set(summary.category_accuracies.keys())

        if show_empty_categories:
            try:
                # 전체 카테고리 목록 확보 (질의 생성기의 정의 사용)
                from scripts.massive_test_query_generator import MassiveTestQueryGenerator  # 지연 임포트
                _gen = MassiveTestQueryGenerator()
                categories_to_show.update(_gen.categories.keys())
            except Exception:
                # 질의 생성기 접근 실패 시, 현재 존재하는 카테고리만 사용
                pass

        # 정렬하여 출력 (알파벳순)
        for category in sorted(categories_to_show):
            if category == "personal_legal_advice" and not show_personal_in_report:
                continue
            accuracy = summary.category_accuracies.get(category, 0.0)
            report.append(f"  {category}: {accuracy:.1%}")
        
        # 성능 분석
        report.append(f"\n📈 성능 분석:")
        
        # 민감한 카테고리들의 정확도
        sensitive_categories = ["medical_legal_advice", "criminal_case_advice", "illegal_activity_assistance"]
        sensitive_accuracies = [summary.category_accuracies.get(cat, 0.0) for cat in sensitive_categories if cat in summary.category_accuracies]
        if sensitive_accuracies:
            sensitive_avg = sum(sensitive_accuracies) / len(sensitive_accuracies)
            report.append(f"  민감한 질문 제한 정확도: {sensitive_avg:.1%}")
        
        # 일반 정보 허용 정확도
        general_categories = ["general_legal_information", "edge_cases"]
        general_accuracies = [summary.category_accuracies.get(cat, 0.0) for cat in general_categories if cat in summary.category_accuracies]
        if general_accuracies:
            general_avg = sum(general_accuracies) / len(general_accuracies)
            report.append(f"  일반 정보 허용 정확도: {general_avg:.1%}")
        
        # 처리 성능
        queries_per_second = summary.total_tests / summary.processing_time if summary.processing_time > 0 else 0
        report.append(f"  처리 성능: {queries_per_second:.1f} 질의/초")
        
        # 최종 평가
        report.append(f"\n🎯 최종 평가:")
        if summary.overall_accuracy >= 0.95:
            report.append("  🏆 우수: 시스템이 매우 잘 작동하고 있습니다.")
        elif summary.overall_accuracy >= 0.90:
            report.append("  🥇 양호: 시스템이 잘 작동하고 있지만 일부 개선이 필요합니다.")
        elif summary.overall_accuracy >= 0.80:
            report.append("  🥈 보통: 시스템이 작동하고 있지만 상당한 개선이 필요합니다.")
        else:
            report.append("  🥉 미흡: 시스템 개선이 시급합니다.")
        
        # 개선 권장사항
        report.append(f"\n💡 개선 권장사항:")
        
        if summary.overall_accuracy < 0.90:
            report.append("  - 전체 정확도가 90% 미만입니다. 시스템 튜닝이 필요합니다.")
        
        # 정확도가 낮은 카테고리 식별
        low_accuracy_categories = [
            category for category, accuracy in summary.category_accuracies.items()
            if accuracy < 0.80
        ]
        
        if low_accuracy_categories:
            report.append(f"  - 정확도가 낮은 카테고리: {', '.join(low_accuracy_categories)}")
            report.append("  - 해당 카테고리의 패턴과 로직을 재검토해야 합니다.")
        
        if summary.error_count > 0:
            report.append(f"  - {summary.error_count}개의 오류가 발생했습니다. 오류 처리 로직을 개선해야 합니다.")
        
        if summary.average_confidence < 0.7:
            report.append("  - 평균 신뢰도가 낮습니다. 모델의 확신도를 높이는 튜닝이 필요합니다.")
        
        report.append("\n" + "=" * 100)
        
        return "\n".join(report)

def load_test_queries(filename: str) -> List[Dict[str, Any]]:
    """테스트 질의 파일 로드"""
    print(f"📂 테스트 질의 로드 중: {filename}")
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = data.get("queries", [])
    print(f"✅ {len(queries)}개의 질의 로드 완료")
    
    return queries

def main():
    """메인 함수"""
    try:
        # 테스트 질의 파일 경로 (생성된 파일 사용)
        queries_file = "test_results/massive_test_queries_*.json"  # 실제 파일명으로 변경 필요
        
        # 최신 파일 찾기
        import glob
        query_files = glob.glob(queries_file)
        if not query_files:
            print("❌ 테스트 질의 파일을 찾을 수 없습니다. 먼저 질의 생성기를 실행하세요.")
            return None
        
        latest_file = max(query_files, key=os.path.getctime)
        print(f"📁 사용할 질의 파일: {latest_file}")
        
        # 질의 로드
        queries = load_test_queries(latest_file)
        
        # 테스트 실행기 초기화
        runner = MassiveTestRunner(max_workers=8)  # 워커 수 조정 가능
        
        # 테스트 실행 (배치 방식 권장)
        results = runner.run_massive_test(queries, method="batch")
        
        # 요약 생성
        summary = runner.generate_summary()
        
        # 결과 저장
        results_file = runner.save_results(results, summary)
        
        # 보고서 생성 및 출력
        report = runner.generate_report(summary)
        print("\n" + report)
        
        # 보고서 파일 저장
        report_file = results_file.replace('.json', '_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 상세 보고서가 {report_file}에 저장되었습니다.")
        
        return results, summary, report
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    main()
