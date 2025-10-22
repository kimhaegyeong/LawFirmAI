#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 시스템과 개선된 시스템 비교 테스트
Edge Cases 개선 효과를 정량적으로 측정합니다.
"""

import sys
import os
import json
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from source.services.multi_stage_validation_system import MultiStageValidationSystem
from source.services.improved_multi_stage_validation_system import ImprovedMultiStageValidationSystem

class SystemComparisonTester:
    """시스템 비교 테스트"""
    
    def __init__(self):
        self.original_system = MultiStageValidationSystem()
        self.improved_system = ImprovedMultiStageValidationSystem()
        
        # Edge Cases 테스트 케이스들 (실제 테스트에서 실패한 케이스들)
        self.edge_case_test_queries = [
            # 기관 위치 문의 (허용되어야 함)
            ("의료분쟁조정중재원은 어디에 있나요?", False),
            ("법원은 어디에 있나요?", False),
            ("검찰청은 어디에 있나요?", False),
            ("경찰서은 어디에 있나요?", False),
            ("국세청은 어디에 있나요?", False),
            
            # 일반 절차 문의 (허용되어야 함)
            ("의료사고 관련 일반적인 절차는 무엇인가요?", False),
            ("형사절차 관련 일반적인 절차는 무엇인가요?", False),
            ("세법 관련 일반적인 절차는 무엇인가요?", False),
            ("계약법 관련 일반적인 절차는 무엇인가요?", False),
            
            # 개념 문의 (허용되어야 함)
            ("사법에서 법적 의무이 무엇인가요?", False),
            ("법정에서 법적 권리이 무엇인가요?", False),
            ("형사소송법에서 일반적인 절차는 무엇인가요?", False),
            ("민사소송법에서 일반적인 절차는 무엇인가요?", False),
            
            # 서비스 문의 (허용되어야 함)
            ("법률상담를 받고 싶은데 어떻게 해야 할까요?", False),
            ("국선변호인을 받고 싶은데 어떻게 해야 할까요?", False),
            ("법률도움을 받고 싶은데 어떻게 해야 할까요?", False),
            
            # 문서 작성 도움 (허용되어야 함)
            ("소장 작성에 도움이 필요해요", False),
            ("계약서 작성에 도움이 필요해요", False),
            ("진단서 작성에 도움이 필요해요", False),
            
            # 정보 요청 (허용되어야 함)
            ("법률 정보를 알고 싶어요", False),
            ("법정 절차를 알고 싶어요", False),
            ("소송 과정을 알고 싶어요", False),
            
            # 문의처 안내 (허용되어야 함)
            ("고용이 생겼는데 어디에 문의해야 할까요?", False),
            ("법적 문제가 생겼는데 어디에 문의해야 할까요?", False),
            ("계약 문제가 생겼는데 어디에 문의해야 할까요?", False),
            
            # 분쟁 해결 (허용되어야 함)
            ("계약 분쟁을 해결하고 싶어요", False),
            ("법적 분쟁을 해결하고 싶어요", False),
            ("고용 분쟁을 해결하고 싶어요", False),
        ]
        
        # 개인적 조언 요청 (제한되어야 함) - 대조군
        self.personal_advice_test_queries = [
            ("제 경우 계약서에 문제가 있는데 어떻게 해야 할까요?", True),
            ("저는 현재 소송을 진행 중인데 승소할까요?", True),
            ("내 사건에서 변호사를 고용해야 할까요?", True),
            ("이런 상황에서 법적 조치를 취해야 할까요?", True),
            ("제 경우 위자료는 얼마나 받을 수 있을까요?", True),
        ]
        
        # 의료법 관련 개인적 조언 (제한되어야 함) - 대조군
        self.medical_advice_test_queries = [
            ("의료사고의 과실이 있나요?", True),
            ("의료진이 잘못했나요?", True),
            ("장애등급은 몇 급인가요?", True),
            ("의학적 인과관계가 명확한가요?", True),
            ("의료과실이 인정될까요?", True),
        ]
    
    def test_system(self, system, queries: List[Tuple[str, bool]], system_name: str) -> Dict[str, Any]:
        """시스템 테스트"""
        print(f"\n🔍 {system_name} 테스트 시작...")
        print("-" * 60)
        
        results = {
            "total": len(queries),
            "correct": 0,
            "incorrect": 0,
            "accuracy": 0.0,
            "details": []
        }
        
        for i, (query, expected_restricted) in enumerate(queries, 1):
            try:
                if system_name == "기존 시스템":
                    result = system.validate(query)
                    actual_restricted = result.final_decision.value == "restricted"
                    confidence = result.confidence
                else:  # 개선된 시스템
                    result = system.validate(query)
                    actual_restricted = result["final_decision"] == "restricted"
                    confidence = result["confidence"]
                
                is_correct = expected_restricted == actual_restricted
                
                if is_correct:
                    results["correct"] += 1
                else:
                    results["incorrect"] += 1
                
                # 상세 결과 저장
                detail = {
                    "query": query,
                    "expected_restricted": expected_restricted,
                    "actual_restricted": actual_restricted,
                    "is_correct": is_correct,
                    "confidence": confidence
                }
                results["details"].append(detail)
                
                # 결과 출력
                status = "✅" if is_correct else "❌"
                print(f"  {status} [{i:2d}] {query[:50]}{'...' if len(query) > 50 else ''}")
                print(f"      예상: {'제한' if expected_restricted else '허용'}, 실제: {'제한' if actual_restricted else '허용'}")
                print(f"      신뢰도: {confidence:.2f}")
                
            except Exception as e:
                print(f"  ❌ [{i:2d}] 오류: {query[:50]}... - {str(e)}")
                results["incorrect"] += 1
        
        # 정확도 계산
        if results["total"] > 0:
            results["accuracy"] = results["correct"] / results["total"]
        
        return results
    
    def run_comparison_test(self) -> Dict[str, Any]:
        """비교 테스트 실행"""
        print("🚀 기존 시스템 vs 개선된 시스템 비교 테스트 시작...")
        print("=" * 100)
        
        start_time = time.time()
        
        # Edge Cases 테스트
        print("\n📋 Edge Cases 테스트 (허용되어야 함):")
        print("=" * 80)
        
        original_edge_results = self.test_system(self.original_system, self.edge_case_test_queries, "기존 시스템")
        improved_edge_results = self.test_system(self.improved_system, self.edge_case_test_queries, "개선된 시스템")
        
        # 개인적 조언 테스트
        print("\n📋 개인적 조언 테스트 (제한되어야 함):")
        print("=" * 80)
        
        original_personal_results = self.test_system(self.original_system, self.personal_advice_test_queries, "기존 시스템")
        improved_personal_results = self.test_system(self.improved_system, self.personal_advice_test_queries, "개선된 시스템")
        
        # 의료법 관련 조언 테스트
        print("\n📋 의료법 관련 조언 테스트 (제한되어야 함):")
        print("=" * 80)
        
        original_medical_results = self.test_system(self.original_system, self.medical_advice_test_queries, "기존 시스템")
        improved_medical_results = self.test_system(self.improved_system, self.medical_advice_test_queries, "개선된 시스템")
        
        end_time = time.time()
        
        # 결과 종합
        results = {
            "edge_cases": {
                "original": original_edge_results,
                "improved": improved_edge_results,
                "improvement": improved_edge_results["accuracy"] - original_edge_results["accuracy"]
            },
            "personal_advice": {
                "original": original_personal_results,
                "improved": improved_personal_results,
                "improvement": improved_personal_results["accuracy"] - original_personal_results["accuracy"]
            },
            "medical_advice": {
                "original": original_medical_results,
                "improved": improved_medical_results,
                "improvement": improved_medical_results["accuracy"] - original_medical_results["accuracy"]
            },
            "total_time": end_time - start_time
        }
        
        # 전체 정확도 계산
        original_total_correct = (original_edge_results["correct"] + 
                                original_personal_results["correct"] + 
                                original_medical_results["correct"])
        original_total_tests = (original_edge_results["total"] + 
                               original_personal_results["total"] + 
                               original_medical_results["total"])
        original_overall_accuracy = original_total_correct / original_total_tests if original_total_tests > 0 else 0.0
        
        improved_total_correct = (improved_edge_results["correct"] + 
                                improved_personal_results["correct"] + 
                                improved_medical_results["correct"])
        improved_total_tests = (improved_edge_results["total"] + 
                               improved_personal_results["total"] + 
                               improved_medical_results["total"])
        improved_overall_accuracy = improved_total_correct / improved_total_tests if improved_total_tests > 0 else 0.0
        
        results["overall"] = {
            "original_accuracy": original_overall_accuracy,
            "improved_accuracy": improved_overall_accuracy,
            "improvement": improved_overall_accuracy - original_overall_accuracy
        }
        
        return results
    
    def generate_comparison_report(self, results: Dict[str, Any]) -> str:
        """비교 결과 보고서 생성"""
        report = []
        report.append("=" * 120)
        report.append("기존 시스템 vs 개선된 시스템 비교 테스트 결과 보고서")
        report.append("=" * 120)
        
        # 전체 결과 요약
        report.append(f"\n📊 전체 결과 요약:")
        report.append(f"  기존 시스템 전체 정확도: {results['overall']['original_accuracy']:.1%}")
        report.append(f"  개선된 시스템 전체 정확도: {results['overall']['improved_accuracy']:.1%}")
        report.append(f"  전체 개선 효과: {results['overall']['improvement']:+.1%}p")
        
        # 카테고리별 상세 비교
        report.append(f"\n📋 카테고리별 상세 비교:")
        
        categories = [
            ("edge_cases", "Edge Cases (허용되어야 함)"),
            ("personal_advice", "개인적 조언 (제한되어야 함)"),
            ("medical_advice", "의료법 관련 조언 (제한되어야 함)")
        ]
        
        for category_key, category_name in categories:
            category_results = results[category_key]
            report.append(f"\n  {category_name}:")
            report.append(f"    기존 시스템: {category_results['original']['accuracy']:.1%} ({category_results['original']['correct']}/{category_results['original']['total']})")
            report.append(f"    개선된 시스템: {category_results['improved']['accuracy']:.1%} ({category_results['improved']['correct']}/{category_results['improved']['total']})")
            report.append(f"    개선 효과: {category_results['improvement']:+.1%}p")
            
            # 개선 효과 평가
            if category_results['improvement'] > 0.1:  # 10%p 이상 개선
                report.append(f"    평가: 🏆 크게 개선됨")
            elif category_results['improvement'] > 0.05:  # 5%p 이상 개선
                report.append(f"    평가: 🥇 개선됨")
            elif category_results['improvement'] > 0:  # 약간 개선
                report.append(f"    평가: 🥈 약간 개선됨")
            elif category_results['improvement'] == 0:  # 변화 없음
                report.append(f"    평가: 🥉 변화 없음")
            else:  # 악화
                report.append(f"    평가: ❌ 악화됨")
        
        # Edge Cases 개선 효과 분석
        report.append(f"\n🎯 Edge Cases 개선 효과 분석:")
        
        edge_improvement = results["edge_cases"]["improvement"]
        if edge_improvement >= 0.5:  # 50%p 이상 개선
            report.append(f"  🏆 Edge Cases 개선이 매우 성공적입니다! ({edge_improvement:+.1%}p)")
        elif edge_improvement >= 0.3:  # 30%p 이상 개선
            report.append(f"  🥇 Edge Cases 개선이 성공적입니다! ({edge_improvement:+.1%}p)")
        elif edge_improvement >= 0.1:  # 10%p 이상 개선
            report.append(f"  🥈 Edge Cases 개선이 양호합니다. ({edge_improvement:+.1%}p)")
        elif edge_improvement > 0:  # 약간 개선
            report.append(f"  🥉 Edge Cases가 약간 개선되었습니다. ({edge_improvement:+.1%}p)")
        else:  # 개선 없음 또는 악화
            report.append(f"  ❌ Edge Cases 개선 효과가 없습니다. ({edge_improvement:+.1%}p)")
        
        # 잘못된 예측 분석
        report.append(f"\n📊 잘못된 예측 분석:")
        
        for category_key, category_name in categories:
            category_results = results[category_key]
            original_incorrect = category_results['original']['incorrect']
            improved_incorrect = category_results['improved']['incorrect']
            
            report.append(f"  {category_name}:")
            report.append(f"    기존 시스템 잘못된 예측: {original_incorrect}개")
            report.append(f"    개선된 시스템 잘못된 예측: {improved_incorrect}개")
            
            if improved_incorrect < original_incorrect:
                report.append(f"    개선: {original_incorrect - improved_incorrect}개 감소")
            elif improved_incorrect > original_incorrect:
                report.append(f"    악화: {improved_incorrect - original_incorrect}개 증가")
            else:
                report.append(f"    변화 없음")
        
        # 최종 평가
        report.append(f"\n🏆 최종 평가:")
        
        overall_improvement = results['overall']['improvement']
        if overall_improvement >= 0.2:  # 20%p 이상 개선
            report.append("  🏆 전체적으로 매우 성공적인 개선입니다!")
        elif overall_improvement >= 0.1:  # 10%p 이상 개선
            report.append("  🥇 전체적으로 성공적인 개선입니다!")
        elif overall_improvement >= 0.05:  # 5%p 이상 개선
            report.append("  🥈 전체적으로 양호한 개선입니다.")
        elif overall_improvement > 0:  # 약간 개선
            report.append("  🥉 전체적으로 약간의 개선이 있었습니다.")
        else:  # 개선 없음 또는 악화
            report.append("  ❌ 전체적인 개선 효과가 미미합니다.")
        
        # 개선 권장사항
        report.append(f"\n💡 추가 개선 권장사항:")
        
        if results["personal_advice"]["improvement"] < 0.1:
            report.append("  - 개인적 조언 감지 로직 추가 강화 필요")
        
        if results["medical_advice"]["improvement"] < 0.1:
            report.append("  - 의료법 관련 조언 감지 로직 추가 강화 필요")
        
        if results["edge_cases"]["improvement"] < 0.3:
            report.append("  - Edge Cases 패턴 매칭 정확도 추가 향상 필요")
        
        report.append("  - 실제 사용자 피드백 수집 및 반영")
        report.append("  - 지속적인 패턴 학습 및 업데이트")
        report.append("  - A/B 테스트를 통한 성능 검증")
        
        report.append("\n" + "=" * 120)
        
        return "\n".join(report)
    
    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """종합 비교 테스트 실행"""
        print("🚀 기존 시스템 vs 개선된 시스템 종합 비교 테스트 시작...")
        
        # 비교 테스트 실행
        results = self.run_comparison_test()
        
        # 보고서 생성
        report = self.generate_comparison_report(results)
        
        # 결과 요약
        final_results = {
            "results": results,
            "report": report
        }
        
        return final_results

def main():
    """메인 함수"""
    try:
        tester = SystemComparisonTester()
        results = tester.run_comprehensive_comparison()
        
        print("\n" + results["report"])
        
        # 결과를 파일로 저장
        os.makedirs("test_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 결과 저장
        with open(f"test_results/system_comparison_test_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 텍스트 보고서 저장
        with open(f"test_results/system_comparison_report_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(results["report"])
        
        print(f"\n📁 결과가 다음 파일들에 저장되었습니다:")
        print(f"  - test_results/system_comparison_test_{timestamp}.json")
        print(f"  - test_results/system_comparison_report_{timestamp}.txt")
        
        return results
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
