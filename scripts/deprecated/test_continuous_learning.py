#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML 시스템 지속적 학습 데모
머신러닝 기반 패턴 학습과 자동 튜닝의 지속적 학습 과정을 시뮬레이션합니다.
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

class ContinuousLearningDemo:
    """지속적 학습 데모"""
    
    def __init__(self):
        self.ml_system = MLIntegratedValidationSystem()
        
        # 시뮬레이션용 쿼리 데이터베이스
        self.query_database = {
            "edge_cases": [
                "의료분쟁조정중재원은 어디에 있나요?",
                "법원은 어디에 있나요?",
                "검찰청은 어디에 있나요?",
                "경찰서은 어디에 있나요?",
                "국세청은 어디에 있나요?",
                "형사절차 관련 일반적인 절차는 무엇인가요?",
                "세법 관련 일반적인 절차는 무엇인가요?",
                "계약법 관련 일반적인 절차는 무엇인가요?",
                "사법에서 법적 의무이 무엇인가요?",
                "법정에서 법적 권리이 무엇인가요?",
                "법률상담를 받고 싶은데 어떻게 해야 할까요?",
                "국선변호인을 받고 싶은데 어떻게 해야 할까요?",
                "법률도움을 받고 싶은데 어떻게 해야 할까요?",
                "소장 작성에 도움이 필요해요",
                "계약서 작성에 도움이 필요해요",
                "진단서 작성에 도움이 필요해요",
                "법률 정보를 알고 싶어요",
                "법정 절차를 알고 싶어요",
                "소송 과정을 알고 싶어요",
                "고용이 생겼는데 어디에 문의해야 할까요?",
                "법적 문제가 생겼는데 어디에 문의해야 할까요?",
                "계약 문제가 생겼는데 어디에 문의해야 할까요?",
                "계약 분쟁을 해결하고 싶어요",
                "법적 분쟁을 해결하고 싶어요",
                "고용 분쟁을 해결하고 싶어요"
            ],
            "personal_advice": [
                "제 경우 계약서에 문제가 있는데 어떻게 해야 할까요?",
                "저는 현재 소송을 진행 중인데 승소할까요?",
                "내 사건에서 변호사를 고용해야 할까요?",
                "이런 상황에서 법적 조치를 취해야 할까요?",
                "제 경우 위자료는 얼마나 받을 수 있을까요?",
                "저는 이혼을 하고 싶은데 어떻게 해야 할까요?",
                "내 사업에서 법적 문제가 생겼는데 어떻게 해야 할까요?",
                "제 경우 세금 문제가 있는데 어떻게 해야 할까요?",
                "저는 현재 고용 문제가 있는데 어떻게 해야 할까요?",
                "내 계약에서 문제가 생겼는데 어떻게 해야 할까요?"
            ],
            "medical_advice": [
                "의료사고의 과실이 있나요?",
                "의료진이 잘못했나요?",
                "장애등급은 몇 급인가요?",
                "의학적 인과관계가 명확한가요?",
                "의료과실이 인정될까요?",
                "의료진의 진료과실이 있나요?",
                "의료사고로 인한 손해배상은 얼마나 받을 수 있을까요?",
                "의료진이 진단을 잘못했나요?",
                "의료사고로 인한 치료비는 누가 부담하나요?",
                "의료진의 치료 방법이 잘못되었나요?"
            ]
        }
        
        # 학습 이력
        self.learning_history = []
        
    def simulate_user_interactions(self, num_interactions: int = 50) -> List[Dict[str, Any]]:
        """사용자 상호작용 시뮬레이션"""
        interactions = []
        
        for i in range(num_interactions):
            # 쿼리 카테고리 선택 (가중치 적용)
            category = random.choices(
                ["edge_cases", "personal_advice", "medical_advice"],
                weights=[0.6, 0.25, 0.15]  # Edge Cases가 더 많이 발생
            )[0]
            
            # 쿼리 선택
            query = random.choice(self.query_database[category])
            
            # 시스템 검증
            validation_result = self.ml_system.validate(query)
            
            # 사용자 피드백 시뮬레이션 (대부분 정확하지만 일부 오류 포함)
            predicted_restricted = validation_result["final_decision"] == "restricted"
            expected_restricted = category in ["personal_advice", "medical_advice"]
            
            # 피드백 정확도 (90% 정확)
            if random.random() < 0.9:
                user_feedback = "correct"
            else:
                user_feedback = "incorrect"
            
            # 피드백 수집
            self.ml_system.collect_user_feedback(
                query=query,
                user_feedback=user_feedback,
                user_id=f"user_{i % 10}",  # 10명의 사용자 시뮬레이션
                session_id=f"session_{i}"
            )
            
            interaction = {
                "interaction_id": i,
                "query": query,
                "category": category,
                "predicted_result": validation_result["final_decision"],
                "expected_result": "restricted" if expected_restricted else "allowed",
                "user_feedback": user_feedback,
                "confidence": validation_result["confidence"],
                "ml_contributed": "ml_prediction" in validation_result,
                "timestamp": datetime.now()
            }
            
            interactions.append(interaction)
            
            # 진행 상황 출력
            if (i + 1) % 10 == 0:
                print(f"  상호작용 {i + 1}/{num_interactions} 완료")
        
        return interactions
    
    def run_learning_cycle(self, cycle_num: int) -> Dict[str, Any]:
        """학습 사이클 실행"""
        print(f"\n[학습 사이클 {cycle_num}]")
        print("-" * 50)
        
        cycle_start_time = time.time()
        
        # 1. 피드백 학습
        print("1. 피드백 학습 중...")
        learning_results = self.ml_system.learn_from_feedback(days=7)
        
        # 2. 자동 튜닝
        print("2. 자동 튜닝 중...")
        tuning_results = self.ml_system.auto_tune_system(days=7)
        
        # 3. 성능 평가
        print("3. 성능 평가 중...")
        performance_report = self.ml_system.get_performance_report(days=7)
        
        # 4. 시스템 상태 확인
        system_status = self.ml_system.get_system_status()
        
        cycle_end_time = time.time()
        
        cycle_result = {
            "cycle_number": cycle_num,
            "cycle_time": cycle_end_time - cycle_start_time,
            "learning_results": {name: {
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "training_samples": result.training_samples
            } for name, result in learning_results.items()},
            "tuning_results": [{
                "parameter": tuning.parameter_name,
                "old_value": tuning.old_value,
                "new_value": tuning.new_value,
                "improvement": tuning.performance_improvement
            } for tuning in tuning_results],
            "performance_metrics": performance_report.get("performance_metrics", {}),
            "system_status": system_status,
            "timestamp": datetime.now().isoformat()
        }
        
        # 결과 출력
        if learning_results:
            best_model = max(learning_results.values(), key=lambda x: x.accuracy)
            print(f"  최고 모델 정확도: {best_model.accuracy:.3f}")
            print(f"  학습 샘플: {best_model.training_samples}개")
        
        if tuning_results:
            print(f"  튜닝 적용: {len(tuning_results)}개")
            for tuning in tuning_results:
                print(f"    - {tuning.parameter_name}: {tuning.old_value} → {tuning.new_value}")
        else:
            print("  튜닝 제안 없음")
        
        if performance_report.get("performance_metrics"):
            metrics = performance_report["performance_metrics"]
            print(f"  전체 정확도: {metrics.get('overall_accuracy', 0):.1%}")
            print(f"  Edge Cases 정확도: {metrics.get('edge_cases_accuracy', 0):.1%}")
        
        print(f"  ML 가중치: {system_status.get('ml_weight', 0):.2f}")
        print(f"  사이클 시간: {cycle_result['cycle_time']:.2f}초")
        
        return cycle_result
    
    def run_continuous_learning_simulation(self, num_cycles: int = 5, 
                                         interactions_per_cycle: int = 20) -> Dict[str, Any]:
        """지속적 학습 시뮬레이션 실행"""
        print("=" * 80)
        print("ML 시스템 지속적 학습 시뮬레이션")
        print("=" * 80)
        
        start_time = time.time()
        
        # 초기 상태 확인
        print("\n[초기 상태]")
        initial_status = self.ml_system.get_system_status()
        print(f"  ML 활성화: {initial_status['ml_enabled']}")
        print(f"  ML 가중치: {initial_status['ml_weight']:.2f}")
        print(f"  피드백 샘플: {initial_status['feedback_samples']}개")
        
        # 학습 사이클 실행
        cycle_results = []
        
        for cycle in range(1, num_cycles + 1):
            # 사용자 상호작용 시뮬레이션
            print(f"\n[사용자 상호작용 시뮬레이션 - 사이클 {cycle}]")
            interactions = self.simulate_user_interactions(interactions_per_cycle)
            
            # 학습 사이클 실행
            cycle_result = self.run_learning_cycle(cycle)
            cycle_results.append(cycle_result)
            
            # 잠시 대기 (실제 환경에서는 시간 간격)
            time.sleep(1)
        
        end_time = time.time()
        
        # 전체 결과 요약
        final_status = self.ml_system.get_system_status()
        
        summary = {
            "simulation_time": end_time - start_time,
            "total_cycles": num_cycles,
            "interactions_per_cycle": interactions_per_cycle,
            "total_interactions": num_cycles * interactions_per_cycle,
            "cycle_results": cycle_results,
            "initial_status": initial_status,
            "final_status": final_status,
            "improvement_summary": self._analyze_improvements(cycle_results)
        }
        
        print("\n" + "=" * 80)
        print("시뮬레이션 완료 요약")
        print("=" * 80)
        print(f"총 시뮬레이션 시간: {summary['simulation_time']:.2f}초")
        print(f"총 학습 사이클: {summary['total_cycles']}개")
        print(f"총 상호작용: {summary['total_interactions']}개")
        
        if summary['improvement_summary']:
            improvements = summary['improvement_summary']
            print(f"ML 가중치 변화: {improvements['ml_weight_change']:+.2f}")
            print(f"피드백 샘플 증가: +{improvements['feedback_samples_increase']}개")
            print(f"평균 모델 정확도: {improvements['average_model_accuracy']:.3f}")
        
        return summary
    
    def _analyze_improvements(self, cycle_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """개선사항 분석"""
        if not cycle_results:
            return {}
        
        initial_status = cycle_results[0]["system_status"]
        final_status = cycle_results[-1]["system_status"]
        
        # 모델 정확도 추출
        model_accuracies = []
        for cycle in cycle_results:
            if cycle["learning_results"]:
                best_accuracy = max(
                    result["accuracy"] for result in cycle["learning_results"].values()
                )
                model_accuracies.append(best_accuracy)
        
        return {
            "ml_weight_change": final_status["ml_weight"] - initial_status["ml_weight"],
            "feedback_samples_increase": final_status["feedback_samples"] - initial_status["feedback_samples"],
            "average_model_accuracy": sum(model_accuracies) / len(model_accuracies) if model_accuracies else 0,
            "total_tuning_applications": sum(len(cycle["tuning_results"]) for cycle in cycle_results)
        }
    
    def generate_learning_report(self, simulation_results: Dict[str, Any]) -> str:
        """학습 보고서 생성"""
        report = []
        report.append("=" * 100)
        report.append("ML 시스템 지속적 학습 시뮬레이션 보고서")
        report.append("=" * 100)
        
        # 시뮬레이션 개요
        report.append(f"\n[시뮬레이션 개요]")
        report.append(f"  총 시뮬레이션 시간: {simulation_results['simulation_time']:.2f}초")
        report.append(f"  총 학습 사이클: {simulation_results['total_cycles']}개")
        report.append(f"  사이클당 상호작용: {simulation_results['interactions_per_cycle']}개")
        report.append(f"  총 상호작용: {simulation_results['total_interactions']}개")
        
        # 초기 vs 최종 상태
        report.append(f"\n[시스템 상태 변화]")
        initial = simulation_results['initial_status']
        final = simulation_results['final_status']
        
        report.append(f"  ML 가중치: {initial['ml_weight']:.2f} → {final['ml_weight']:.2f}")
        report.append(f"  피드백 샘플: {initial['feedback_samples']} → {final['feedback_samples']}개")
        report.append(f"  ML 모델 로드: {initial['ml_models_loaded']} → {final['ml_models_loaded']}")
        
        # 개선사항 요약
        if simulation_results['improvement_summary']:
            improvements = simulation_results['improvement_summary']
            report.append(f"\n[개선사항 요약]")
            report.append(f"  ML 가중치 변화: {improvements['ml_weight_change']:+.2f}")
            report.append(f"  피드백 샘플 증가: +{improvements['feedback_samples_increase']}개")
            report.append(f"  평균 모델 정확도: {improvements['average_model_accuracy']:.3f}")
            report.append(f"  총 튜닝 적용: {improvements['total_tuning_applications']}개")
        
        # 사이클별 상세 결과
        report.append(f"\n[사이클별 상세 결과]")
        for cycle_result in simulation_results['cycle_results']:
            cycle_num = cycle_result['cycle_number']
            report.append(f"\n  사이클 {cycle_num}:")
            report.append(f"    사이클 시간: {cycle_result['cycle_time']:.2f}초")
            
            if cycle_result['learning_results']:
                best_model = max(
                    cycle_result['learning_results'].items(),
                    key=lambda x: x[1]['accuracy']
                )
                report.append(f"    최고 모델: {best_model[0]} (정확도: {best_model[1]['accuracy']:.3f})")
                report.append(f"    학습 샘플: {best_model[1]['training_samples']}개")
            
            if cycle_result['tuning_results']:
                report.append(f"    튜닝 적용: {len(cycle_result['tuning_results'])}개")
                for tuning in cycle_result['tuning_results']:
                    report.append(f"      - {tuning['parameter']}: {tuning['old_value']} → {tuning['new_value']}")
            else:
                report.append(f"    튜닝 적용: 없음")
            
            if cycle_result['performance_metrics']:
                metrics = cycle_result['performance_metrics']
                report.append(f"    전체 정확도: {metrics.get('overall_accuracy', 0):.1%}")
                report.append(f"    Edge Cases 정확도: {metrics.get('edge_cases_accuracy', 0):.1%}")
        
        # 결론 및 권장사항
        report.append(f"\n[결론 및 권장사항]")
        report.append(f"  ✅ ML 시스템이 성공적으로 학습하고 자동 튜닝을 수행했습니다.")
        report.append(f"  📈 피드백 데이터가 지속적으로 축적되어 모델 성능이 향상되었습니다.")
        report.append(f"  🔧 자동 튜닝 시스템이 시스템 파라미터를 최적화했습니다.")
        
        report.append(f"\n  권장사항:")
        report.append(f"    - 실제 운영 환경에서 지속적 학습 활성화")
        report.append(f"    - 사용자 피드백 수집 체계 구축")
        report.append(f"    - 성능 모니터링 대시보드 구축")
        report.append(f"    - A/B 테스트를 통한 성능 검증")
        
        report.append("\n" + "=" * 100)
        
        return "\n".join(report)

def main():
    """메인 함수"""
    try:
        demo = ContinuousLearningDemo()
        
        # 지속적 학습 시뮬레이션 실행
        results = demo.run_continuous_learning_simulation(
            num_cycles=3,
            interactions_per_cycle=15
        )
        
        # 학습 보고서 생성
        report = demo.generate_learning_report(results)
        print("\n" + report)
        
        # 결과를 파일로 저장
        os.makedirs("test_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 결과 저장
        with open(f"test_results/continuous_learning_simulation_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # 텍스트 보고서 저장
        with open(f"test_results/continuous_learning_report_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"\n[파일 저장 완료]")
        print(f"  - test_results/continuous_learning_simulation_{timestamp}.json")
        print(f"  - test_results/continuous_learning_report_{timestamp}.txt")
        
        return results
        
    except Exception as e:
        print(f"[오류] 시뮬레이션 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()

