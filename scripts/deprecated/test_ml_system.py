#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML 통합 검증 시스템 테스트
머신러닝 기반 패턴 학습과 자동 튜닝 기능을 테스트합니다.
"""

import sys
import os
import json
import time
import random
from typing import Dict, List, Any
from datetime import datetime, timedelta

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from source.services.ml_integrated_validation_system import MLIntegratedValidationSystem
from source.services.ml_pattern_learning_system import FeedbackData

class MLSystemTester:
    """ML 시스템 테스트"""
    
    def __init__(self):
        self.ml_system = MLIntegratedValidationSystem()
        
        # 테스트용 피드백 데이터 생성
        self.test_feedback_data = self._generate_test_feedback_data()
        
        # 테스트 쿼리들
        self.test_queries = [
            # Edge Cases (허용되어야 함)
            ("의료분쟁조정중재원은 어디에 있나요?", False),
            ("법원은 어디에 있나요?", False),
            ("형사절차 관련 일반적인 절차는 무엇인가요?", False),
            ("계약서 작성에 도움이 필요해요", False),
            ("법률 정보를 알고 싶어요", False),
            
            # 개인적 조언 (제한되어야 함)
            ("제 경우 계약서에 문제가 있는데 어떻게 해야 할까요?", True),
            ("저는 현재 소송을 진행 중인데 승소할까요?", True),
            ("내 사건에서 변호사를 고용해야 할까요?", True),
            
            # 의료법 조언 (제한되어야 함)
            ("의료사고의 과실이 있나요?", True),
            ("의료진이 잘못했나요?", True),
            ("장애등급은 몇 급인가요?", True),
        ]
    
    def _generate_test_feedback_data(self) -> List[FeedbackData]:
        """테스트용 피드백 데이터 생성"""
        feedback_data = []
        
        # Edge Cases 피드백 (대부분 정확)
        edge_cases_queries = [
            "의료분쟁조정중재원은 어디에 있나요?",
            "법원은 어디에 있나요?",
            "형사절차 관련 일반적인 절차는 무엇인가요?",
            "계약서 작성에 도움이 필요해요",
            "법률 정보를 알고 싶어요"
        ]
        
        for query in edge_cases_queries:
            feedback_data.append(FeedbackData(
                query=query,
                predicted_result="allowed",
                actual_result="allowed",
                user_feedback="correct",
                confidence=0.8,
                timestamp=datetime.now() - timedelta(days=random.randint(1, 30)),
                edge_case_type="institution_location"
            ))
        
        # 개인적 조언 피드백 (대부분 정확)
        personal_advice_queries = [
            "제 경우 계약서에 문제가 있는데 어떻게 해야 할까요?",
            "저는 현재 소송을 진행 중인데 승소할까요?",
            "내 사건에서 변호사를 고용해야 할까요?"
        ]
        
        for query in personal_advice_queries:
            feedback_data.append(FeedbackData(
                query=query,
                predicted_result="restricted",
                actual_result="restricted",
                user_feedback="correct",
                confidence=0.9,
                timestamp=datetime.now() - timedelta(days=random.randint(1, 30)),
                edge_case_type=None
            ))
        
        # 의료법 조언 피드백 (대부분 정확)
        medical_advice_queries = [
            "의료사고의 과실이 있나요?",
            "의료진이 잘못했나요?",
            "장애등급은 몇 급인가요?"
        ]
        
        for query in medical_advice_queries:
            feedback_data.append(FeedbackData(
                query=query,
                predicted_result="restricted",
                actual_result="restricted",
                user_feedback="correct",
                confidence=0.85,
                timestamp=datetime.now() - timedelta(days=random.randint(1, 30)),
                edge_case_type=None
            ))
        
        # 일부 잘못된 예측 (학습용)
        feedback_data.append(FeedbackData(
            query="법률상담를 받고 싶은데 어떻게 해야 할까요?",
            predicted_result="restricted",
            actual_result="allowed",
            user_feedback="incorrect",
            confidence=0.7,
            timestamp=datetime.now() - timedelta(days=5),
            edge_case_type="service_request"
        ))
        
        feedback_data.append(FeedbackData(
            query="제 경우 위자료는 얼마나 받을 수 있을까요?",
            predicted_result="allowed",
            actual_result="restricted",
            user_feedback="incorrect",
            confidence=0.6,
            timestamp=datetime.now() - timedelta(days=3),
            edge_case_type=None
        ))
        
        return feedback_data
    
    def test_feedback_collection(self) -> Dict[str, Any]:
        """피드백 수집 테스트"""
        print("\n[피드백 수집 테스트]")
        print("-" * 50)
        
        results = {
            "total_feedback": len(self.test_feedback_data),
            "successful_collection": 0,
            "failed_collection": 0,
            "details": []
        }
        
        for i, feedback in enumerate(self.test_feedback_data, 1):
            try:
                success = self.ml_system.ml_system.feedback_collector.collect_feedback(feedback)
                
                if success:
                    results["successful_collection"] += 1
                    status = "성공"
                else:
                    results["failed_collection"] += 1
                    status = "실패"
                
                results["details"].append({
                    "query": feedback.query[:50] + "..." if len(feedback.query) > 50 else feedback.query,
                    "feedback": feedback.user_feedback,
                    "status": status
                })
                
                print(f"  [{i:2d}] {status}: {feedback.query[:50]}{'...' if len(feedback.query) > 50 else ''}")
                
            except Exception as e:
                results["failed_collection"] += 1
                print(f"  [{i:2d}] 오류: {str(e)}")
        
        print(f"\n피드백 수집 결과: {results['successful_collection']}/{results['total_feedback']} 성공")
        return results
    
    def test_pattern_learning(self) -> Dict[str, Any]:
        """패턴 학습 테스트"""
        print("\n[패턴 학습 테스트]")
        print("-" * 50)
        
        try:
            # 피드백 데이터 로드
            feedback_data = self.ml_system.ml_system.feedback_collector.load_feedback_data(30)
            
            if len(feedback_data) < 5:
                print("학습할 피드백 데이터가 부족합니다.")
                return {"error": "insufficient_data"}
            
            print(f"학습 데이터: {len(feedback_data)}개")
            
            # 모델 학습
            learning_results = self.ml_system.ml_system.pattern_learner.train_models(feedback_data)
            
            results = {
                "learning_data_count": len(feedback_data),
                "models_trained": len(learning_results),
                "model_results": {}
            }
            
            for model_name, result in learning_results.items():
                results["model_results"][model_name] = {
                    "accuracy": result.accuracy,
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1_score": result.f1_score,
                    "training_samples": result.training_samples,
                    "validation_samples": result.validation_samples
                }
                
                print(f"  {model_name}:")
                print(f"    정확도: {result.accuracy:.3f}")
                print(f"    정밀도: {result.precision:.3f}")
                print(f"    재현율: {result.recall:.3f}")
                print(f"    F1 점수: {result.f1_score:.3f}")
                print(f"    학습 샘플: {result.training_samples}")
                print(f"    검증 샘플: {result.validation_samples}")
            
            return results
            
        except Exception as e:
            print(f"패턴 학습 테스트 오류: {e}")
            return {"error": str(e)}
    
    def test_ml_prediction(self) -> Dict[str, Any]:
        """ML 예측 테스트"""
        print("\n[ML 예측 테스트]")
        print("-" * 50)
        
        results = {
            "total_queries": len(self.test_queries),
            "predictions": [],
            "accuracy": 0.0
        }
        
        correct_predictions = 0
        
        for i, (query, expected_restricted) in enumerate(self.test_queries, 1):
            try:
                # ML 예측 수행
                ml_prediction = self.ml_system.ml_system.get_ml_prediction(
                    query=query,
                    confidence=0.8,
                    edge_case_type=None
                )
                
                # 예측 결과 해석
                predicted_restricted = ml_prediction["restricted"] > ml_prediction["allowed"]
                is_correct = predicted_restricted == expected_restricted
                
                if is_correct:
                    correct_predictions += 1
                
                results["predictions"].append({
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "expected_restricted": expected_restricted,
                    "predicted_restricted": predicted_restricted,
                    "is_correct": is_correct,
                    "ml_confidence": max(ml_prediction.values())
                })
                
                status = "정확" if is_correct else "부정확"
                print(f"  [{i:2d}] {status}: {query[:50]}{'...' if len(query) > 50 else ''}")
                print(f"      예상: {'제한' if expected_restricted else '허용'}, 예측: {'제한' if predicted_restricted else '허용'}")
                print(f"      ML 신뢰도: {max(ml_prediction.values()):.3f}")
                
            except Exception as e:
                print(f"  [{i:2d}] 오류: {str(e)}")
        
        results["accuracy"] = correct_predictions / len(self.test_queries) if self.test_queries else 0
        print(f"\nML 예측 정확도: {results['accuracy']:.1%}")
        
        return results
    
    def test_integrated_validation(self) -> Dict[str, Any]:
        """통합 검증 테스트"""
        print("\n[통합 검증 테스트]")
        print("-" * 50)
        
        results = {
            "total_queries": len(self.test_queries),
            "validations": [],
            "accuracy": 0.0,
            "ml_contributions": 0
        }
        
        correct_predictions = 0
        ml_contributions = 0
        
        for i, (query, expected_restricted) in enumerate(self.test_queries, 1):
            try:
                # 통합 검증 수행
                validation_result = self.ml_system.validate(query)
                
                # 결과 해석
                predicted_restricted = validation_result["final_decision"] == "restricted"
                is_correct = predicted_restricted == expected_restricted
                
                if is_correct:
                    correct_predictions += 1
                
                # ML 기여도 확인
                ml_contributed = "ml_prediction" in validation_result
                if ml_contributed:
                    ml_contributions += 1
                
                results["validations"].append({
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "expected_restricted": expected_restricted,
                    "predicted_restricted": predicted_restricted,
                    "is_correct": is_correct,
                    "confidence": validation_result["confidence"],
                    "ml_contributed": ml_contributed,
                    "edge_case": validation_result["edge_case_info"]["is_edge_case"]
                })
                
                status = "정확" if is_correct else "부정확"
                ml_status = "ML 기여" if ml_contributed else "기존 시스템"
                print(f"  [{i:2d}] {status} ({ml_status}): {query[:50]}{'...' if len(query) > 50 else ''}")
                print(f"      예상: {'제한' if expected_restricted else '허용'}, 예측: {'제한' if predicted_restricted else '허용'}")
                print(f"      신뢰도: {validation_result['confidence']:.3f}")
                if ml_contributed:
                    print(f"      ML 예측: {validation_result['ml_prediction']}")
                
            except Exception as e:
                print(f"  [{i:2d}] 오류: {str(e)}")
        
        results["accuracy"] = correct_predictions / len(self.test_queries) if self.test_queries else 0
        results["ml_contributions"] = ml_contributions
        
        print(f"\n통합 검증 정확도: {results['accuracy']:.1%}")
        print(f"ML 기여도: {ml_contributions}/{len(self.test_queries)} ({ml_contributions/len(self.test_queries)*100:.1f}%)")
        
        return results
    
    def test_auto_tuning(self) -> Dict[str, Any]:
        """자동 튜닝 테스트"""
        print("\n[자동 튜닝 테스트]")
        print("-" * 50)
        
        try:
            # 자동 튜닝 수행
            tuning_results = self.ml_system.auto_tune_system(days=30)
            
            results = {
                "tuning_suggestions": len(tuning_results),
                "applied_tunings": [],
                "system_status": self.ml_system.get_system_status()
            }
            
            for tuning in tuning_results:
                results["applied_tunings"].append({
                    "parameter": tuning.parameter_name,
                    "old_value": tuning.old_value,
                    "new_value": tuning.new_value,
                    "improvement": tuning.performance_improvement,
                    "confidence": tuning.confidence
                })
                
                print(f"  튜닝 적용: {tuning.parameter_name}")
                print(f"    {tuning.old_value} → {tuning.new_value}")
                print(f"    예상 개선: {tuning.performance_improvement:.1%}")
                print(f"    신뢰도: {tuning.confidence:.2f}")
            
            if not tuning_results:
                print("  튜닝 제안 없음 (현재 성능이 양호)")
            
            print(f"\n시스템 상태:")
            print(f"  ML 활성화: {results['system_status']['ml_enabled']}")
            print(f"  ML 가중치: {results['system_status']['ml_weight']:.2f}")
            print(f"  기존 시스템 가중치: {results['system_status']['traditional_weight']:.2f}")
            print(f"  ML 모델 로드: {results['system_status']['ml_models_loaded']}")
            
            return results
            
        except Exception as e:
            print(f"자동 튜닝 테스트 오류: {e}")
            return {"error": str(e)}
    
    def test_performance_monitoring(self) -> Dict[str, Any]:
        """성능 모니터링 테스트"""
        print("\n[성능 모니터링 테스트]")
        print("-" * 50)
        
        try:
            # 성능 보고서 생성
            report = self.ml_system.get_performance_report(days=30)
            
            results = {
                "report_generated": True,
                "period_days": report.get("period_days", 0),
                "total_feedback_samples": report.get("total_feedback_samples", 0),
                "performance_metrics": report.get("performance_metrics", {}),
                "ml_integration": report.get("ml_integration", {})
            }
            
            print(f"성능 보고서 생성 완료:")
            print(f"  기간: {results['period_days']}일")
            print(f"  피드백 샘플: {results['total_feedback_samples']}개")
            
            if results["performance_metrics"]:
                metrics = results["performance_metrics"]
                print(f"  전체 정확도: {metrics.get('overall_accuracy', 0):.1%}")
                print(f"  Edge Cases 정확도: {metrics.get('edge_cases_accuracy', 0):.1%}")
                print(f"  개인적 조언 정확도: {metrics.get('personal_advice_accuracy', 0):.1%}")
                print(f"  의료법 조언 정확도: {metrics.get('medical_advice_accuracy', 0):.1%}")
            
            if results["ml_integration"]:
                ml_info = results["ml_integration"]
                print(f"  ML 활성화: {ml_info.get('ml_enabled', False)}")
                print(f"  ML 가중치: {ml_info.get('ml_weight', 0):.2f}")
                print(f"  시스템 버전: {ml_info.get('system_version', 'Unknown')}")
            
            return results
            
        except Exception as e:
            print(f"성능 모니터링 테스트 오류: {e}")
            return {"error": str(e)}
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """종합 테스트 실행"""
        print("=" * 80)
        print("ML 통합 검증 시스템 종합 테스트")
        print("=" * 80)
        
        start_time = time.time()
        
        # 각 테스트 실행
        test_results = {}
        
        # 1. 피드백 수집 테스트
        test_results["feedback_collection"] = self.test_feedback_collection()
        
        # 2. 패턴 학습 테스트
        test_results["pattern_learning"] = self.test_pattern_learning()
        
        # 3. ML 예측 테스트
        test_results["ml_prediction"] = self.test_ml_prediction()
        
        # 4. 통합 검증 테스트
        test_results["integrated_validation"] = self.test_integrated_validation()
        
        # 5. 자동 튜닝 테스트
        test_results["auto_tuning"] = self.test_auto_tuning()
        
        # 6. 성능 모니터링 테스트
        test_results["performance_monitoring"] = self.test_performance_monitoring()
        
        end_time = time.time()
        
        # 전체 결과 요약
        summary = {
            "total_test_time": end_time - start_time,
            "test_results": test_results,
            "overall_status": "완료"
        }
        
        print("\n" + "=" * 80)
        print("테스트 완료 요약")
        print("=" * 80)
        print(f"총 테스트 시간: {summary['total_test_time']:.2f}초")
        print(f"전체 상태: {summary['overall_status']}")
        
        return summary

def main():
    """메인 함수"""
    try:
        tester = MLSystemTester()
        results = tester.run_comprehensive_test()
        
        # 결과를 파일로 저장
        os.makedirs("test_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 결과 저장
        with open(f"test_results/ml_system_test_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n[파일 저장 완료]")
        print(f"  - test_results/ml_system_test_{timestamp}.json")
        
        return results
        
    except Exception as e:
        print(f"[오류] 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()

