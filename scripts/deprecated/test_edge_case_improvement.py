#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 Edge Cases 처리 시스템 테스트
새로 구현한 Edge Cases 특별 처리 시스템의 성능을 테스트합니다.
"""

import sys
import os
import json
import time
from typing import Dict, List, Any
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from source.services.improved_multi_stage_validation_system import ImprovedMultiStageValidationSystem

class EdgeCaseImprovementTester:
    """Edge Cases 개선 테스트"""
    
    def __init__(self):
        self.improved_system = ImprovedMultiStageValidationSystem()
        
        # Edge Cases 테스트 케이스들 (실제 테스트에서 실패한 케이스들)
        self.edge_case_test_queries = [
            # 기관 위치 문의 (허용되어야 함)
            ("의료분쟁조정중재원은 어디에 있나요?", False, "institution_location"),
            ("법원은 어디에 있나요?", False, "institution_location"),
            ("검찰청은 어디에 있나요?", False, "institution_location"),
            ("경찰서은 어디에 있나요?", False, "institution_location"),
            ("국세청은 어디에 있나요?", False, "institution_location"),
            
            # 일반 절차 문의 (허용되어야 함)
            ("의료사고 관련 일반적인 절차는 무엇인가요?", False, "general_procedure"),
            ("형사절차 관련 일반적인 절차는 무엇인가요?", False, "general_procedure"),
            ("세법 관련 일반적인 절차는 무엇인가요?", False, "general_procedure"),
            ("계약법 관련 일반적인 절차는 무엇인가요?", False, "general_procedure"),
            
            # 개념 문의 (허용되어야 함)
            ("사법에서 법적 의무이 무엇인가요?", False, "concept_inquiry"),
            ("법정에서 법적 권리이 무엇인가요?", False, "concept_inquiry"),
            ("형사소송법에서 일반적인 절차는 무엇인가요?", False, "concept_inquiry"),
            ("민사소송법에서 일반적인 절차는 무엇인가요?", False, "concept_inquiry"),
            
            # 서비스 문의 (허용되어야 함)
            ("법률상담를 받고 싶은데 어떻게 해야 할까요?", False, "service_request"),
            ("국선변호인을 받고 싶은데 어떻게 해야 할까요?", False, "service_request"),
            ("법률도움을 받고 싶은데 어떻게 해야 할까요?", False, "service_request"),
            
            # 문서 작성 도움 (허용되어야 함)
            ("소장 작성에 도움이 필요해요", False, "document_help"),
            ("계약서 작성에 도움이 필요해요", False, "document_help"),
            ("진단서 작성에 도움이 필요해요", False, "document_help"),
            
            # 정보 요청 (허용되어야 함)
            ("법률 정보를 알고 싶어요", False, "information_request"),
            ("법정 절차를 알고 싶어요", False, "information_request"),
            ("소송 과정을 알고 싶어요", False, "information_request"),
            
            # 문의처 안내 (허용되어야 함)
            ("고용이 생겼는데 어디에 문의해야 할까요?", False, "inquiry_guidance"),
            ("법적 문제가 생겼는데 어디에 문의해야 할까요?", False, "inquiry_guidance"),
            ("계약 문제가 생겼는데 어디에 문의해야 할까요?", False, "inquiry_guidance"),
            
            # 분쟁 해결 (허용되어야 함)
            ("계약 분쟁을 해결하고 싶어요", False, "dispute_resolution"),
            ("법적 분쟁을 해결하고 싶어요", False, "dispute_resolution"),
            ("고용 분쟁을 해결하고 싶어요", False, "dispute_resolution"),
        ]
        
        # 개인적 조언 요청 (제한되어야 함) - 대조군
        self.personal_advice_test_queries = [
            ("제 경우 계약서에 문제가 있는데 어떻게 해야 할까요?", True, "personal_advice"),
            ("저는 현재 소송을 진행 중인데 승소할까요?", True, "personal_advice"),
            ("내 사건에서 변호사를 고용해야 할까요?", True, "personal_advice"),
            ("이런 상황에서 법적 조치를 취해야 할까요?", True, "personal_advice"),
            ("제 경우 위자료는 얼마나 받을 수 있을까요?", True, "personal_advice"),
        ]
        
        # 의료법 관련 개인적 조언 (제한되어야 함) - 대조군
        self.medical_advice_test_queries = [
            ("의료사고의 과실이 있나요?", True, "medical_advice"),
            ("의료진이 잘못했나요?", True, "medical_advice"),
            ("장애등급은 몇 급인가요?", True, "medical_advice"),
            ("의학적 인과관계가 명확한가요?", True, "medical_advice"),
            ("의료과실이 인정될까요?", True, "medical_advice"),
        ]
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Edge Cases 테스트"""
        print("🔍 Edge Cases 개선 테스트 시작...")
        print("=" * 80)
        
        results = {
            "edge_cases": {
                "total": len(self.edge_case_test_queries),
                "correct": 0,
                "incorrect": 0,
                "accuracy": 0.0,
                "details": []
            },
            "personal_advice": {
                "total": len(self.personal_advice_test_queries),
                "correct": 0,
                "incorrect": 0,
                "accuracy": 0.0,
                "details": []
            },
            "medical_advice": {
                "total": len(self.medical_advice_test_queries),
                "correct": 0,
                "incorrect": 0,
                "accuracy": 0.0,
                "details": []
            }
        }
        
        # Edge Cases 테스트
        print("\n📋 Edge Cases 테스트 (허용되어야 함):")
        print("-" * 50)
        
        for i, (query, expected_restricted, case_type) in enumerate(self.edge_case_test_queries, 1):
            try:
                result = self.improved_system.validate(query)
                actual_restricted = result["final_decision"] == "restricted"
                is_correct = expected_restricted == actual_restricted
                
                if is_correct:
                    results["edge_cases"]["correct"] += 1
                else:
                    results["edge_cases"]["incorrect"] += 1
                
                # 상세 결과 저장
                detail = {
                    "query": query,
                    "expected_restricted": expected_restricted,
                    "actual_restricted": actual_restricted,
                    "is_correct": is_correct,
                    "case_type": case_type,
                    "confidence": result["confidence"],
                    "edge_case_info": result["edge_case_info"],
                    "reasoning": result["reasoning"]
                }
                results["edge_cases"]["details"].append(detail)
                
                # 결과 출력
                status = "✅" if is_correct else "❌"
                print(f"  {status} [{i:2d}] {query[:50]}{'...' if len(query) > 50 else ''}")
                print(f"      예상: {'제한' if expected_restricted else '허용'}, 실제: {'제한' if actual_restricted else '허용'}")
                print(f"      Edge Case: {result['edge_case_info']['is_edge_case']}, 타입: {result['edge_case_info']['edge_case_type']}")
                print(f"      신뢰도: {result['confidence']:.2f}")
                
            except Exception as e:
                print(f"  ❌ [{i:2d}] 오류: {query[:50]}... - {str(e)}")
                results["edge_cases"]["incorrect"] += 1
        
        # 개인적 조언 테스트 (대조군)
        print("\n📋 개인적 조언 테스트 (제한되어야 함):")
        print("-" * 50)
        
        for i, (query, expected_restricted, case_type) in enumerate(self.personal_advice_test_queries, 1):
            try:
                result = self.improved_system.validate(query)
                actual_restricted = result["final_decision"] == "restricted"
                is_correct = expected_restricted == actual_restricted
                
                if is_correct:
                    results["personal_advice"]["correct"] += 1
                else:
                    results["personal_advice"]["incorrect"] += 1
                
                # 상세 결과 저장
                detail = {
                    "query": query,
                    "expected_restricted": expected_restricted,
                    "actual_restricted": actual_restricted,
                    "is_correct": is_correct,
                    "case_type": case_type,
                    "confidence": result["confidence"],
                    "edge_case_info": result["edge_case_info"],
                    "reasoning": result["reasoning"]
                }
                results["personal_advice"]["details"].append(detail)
                
                # 결과 출력
                status = "✅" if is_correct else "❌"
                print(f"  {status} [{i:2d}] {query[:50]}{'...' if len(query) > 50 else ''}")
                print(f"      예상: {'제한' if expected_restricted else '허용'}, 실제: {'제한' if actual_restricted else '허용'}")
                print(f"      Edge Case: {result['edge_case_info']['is_edge_case']}")
                print(f"      신뢰도: {result['confidence']:.2f}")
                
            except Exception as e:
                print(f"  ❌ [{i:2d}] 오류: {query[:50]}... - {str(e)}")
                results["personal_advice"]["incorrect"] += 1
        
        # 의료법 관련 조언 테스트 (대조군)
        print("\n📋 의료법 관련 조언 테스트 (제한되어야 함):")
        print("-" * 50)
        
        for i, (query, expected_restricted, case_type) in enumerate(self.medical_advice_test_queries, 1):
            try:
                result = self.improved_system.validate(query)
                actual_restricted = result["final_decision"] == "restricted"
                is_correct = expected_restricted == actual_restricted
                
                if is_correct:
                    results["medical_advice"]["correct"] += 1
                else:
                    results["medical_advice"]["incorrect"] += 1
                
                # 상세 결과 저장
                detail = {
                    "query": query,
                    "expected_restricted": expected_restricted,
                    "actual_restricted": actual_restricted,
                    "is_correct": is_correct,
                    "case_type": case_type,
                    "confidence": result["confidence"],
                    "edge_case_info": result["edge_case_info"],
                    "reasoning": result["reasoning"]
                }
                results["medical_advice"]["details"].append(detail)
                
                # 결과 출력
                status = "✅" if is_correct else "❌"
                print(f"  {status} [{i:2d}] {query[:50]}{'...' if len(query) > 50 else ''}")
                print(f"      예상: {'제한' if expected_restricted else '허용'}, 실제: {'제한' if actual_restricted else '허용'}")
                print(f"      Edge Case: {result['edge_case_info']['is_edge_case']}")
                print(f"      신뢰도: {result['confidence']:.2f}")
                
            except Exception as e:
                print(f"  ❌ [{i:2d}] 오류: {query[:50]}... - {str(e)}")
                results["medical_advice"]["incorrect"] += 1
        
        # 정확도 계산
        for category in results:
            if results[category]["total"] > 0:
                results[category]["accuracy"] = results[category]["correct"] / results[category]["total"]
        
        return results
    
    def generate_improvement_report(self, results: Dict[str, Any]) -> str:
        """개선 결과 보고서 생성"""
        report = []
        report.append("=" * 100)
        report.append("Edge Cases 개선 시스템 테스트 결과 보고서")
        report.append("=" * 100)
        
        # 전체 결과
        total_tests = sum(results[cat]["total"] for cat in results)
        total_correct = sum(results[cat]["correct"] for cat in results)
        overall_accuracy = total_correct / total_tests if total_tests > 0 else 0.0
        
        report.append(f"\n📊 전체 결과:")
        report.append(f"  총 테스트: {total_tests}")
        report.append(f"  정확한 예측: {total_correct}")
        report.append(f"  잘못된 예측: {total_tests - total_correct}")
        report.append(f"  전체 정확도: {overall_accuracy:.1%}")
        
        # 카테고리별 상세 결과
        report.append(f"\n📋 카테고리별 상세 결과:")
        
        for category, result in results.items():
            report.append(f"  {category}:")
            report.append(f"    정확도: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
            
            # 잘못된 예측들 표시
            incorrect_cases = [detail for detail in result['details'] if not detail['is_correct']]
            if incorrect_cases:
                report.append(f"    잘못된 예측 ({len(incorrect_cases)}개):")
                for case in incorrect_cases[:3]:  # 처음 3개만 표시
                    report.append(f"      - {case['query'][:50]}... (예상: {'제한' if case['expected_restricted'] else '허용'}, 실제: {'제한' if case['actual_restricted'] else '허용'})")
                if len(incorrect_cases) > 3:
                    report.append(f"      ... 외 {len(incorrect_cases) - 3}개")
        
        # Edge Cases 개선 효과 분석
        report.append(f"\n🎯 Edge Cases 개선 효과 분석:")
        
        edge_accuracy = results["edge_cases"]["accuracy"]
        personal_accuracy = results["personal_advice"]["accuracy"]
        medical_accuracy = results["medical_advice"]["accuracy"]
        
        report.append(f"  Edge Cases 정확도: {edge_accuracy:.1%}")
        report.append(f"  개인적 조언 제한 정확도: {personal_accuracy:.1%}")
        report.append(f"  의료법 조언 제한 정확도: {medical_accuracy:.1%}")
        
        # 개선 효과 평가
        if edge_accuracy >= 0.90:
            report.append(f"  ✅ Edge Cases 개선 우수: {edge_accuracy:.1%}")
        elif edge_accuracy >= 0.80:
            report.append(f"  ⚠️ Edge Cases 개선 양호: {edge_accuracy:.1%}")
        else:
            report.append(f"  ❌ Edge Cases 개선 부족: {edge_accuracy:.1%}")
        
        # Edge Case 타입별 분석
        report.append(f"\n📊 Edge Case 타입별 분석:")
        
        edge_case_types = {}
        for detail in results["edge_cases"]["details"]:
            case_type = detail["case_type"]
            if case_type not in edge_case_types:
                edge_case_types[case_type] = {"correct": 0, "total": 0}
            
            edge_case_types[case_type]["total"] += 1
            if detail["is_correct"]:
                edge_case_types[case_type]["correct"] += 1
        
        for case_type, stats in edge_case_types.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            report.append(f"  {case_type}: {accuracy:.1%} ({stats['correct']}/{stats['total']})")
        
        # 최종 평가
        report.append(f"\n🏆 최종 평가:")
        
        if overall_accuracy >= 0.95:
            report.append("  🏆 우수: Edge Cases 개선이 매우 성공적입니다.")
        elif overall_accuracy >= 0.90:
            report.append("  🥇 양호: Edge Cases 개선이 성공적입니다.")
        elif overall_accuracy >= 0.80:
            report.append("  🥈 보통: Edge Cases 개선이 어느 정도 성공적입니다.")
        else:
            report.append("  🥉 미흡: Edge Cases 개선이 부족합니다.")
        
        # 개선 권장사항
        report.append(f"\n💡 추가 개선 권장사항:")
        
        if edge_accuracy < 0.90:
            report.append("  - Edge Cases 패턴 매칭 정확도 향상 필요")
            report.append("  - 허용 키워드 목록 확장 검토")
        
        if personal_accuracy < 0.90:
            report.append("  - 개인적 조언 감지 로직 강화 필요")
        
        if medical_accuracy < 0.90:
            report.append("  - 의료법 관련 조언 감지 로직 강화 필요")
        
        report.append("  - 실제 사용자 피드백 수집 및 반영")
        report.append("  - 지속적인 패턴 학습 및 업데이트")
        
        report.append("\n" + "=" * 100)
        
        return "\n".join(report)
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """종합 테스트 실행"""
        print("🚀 Edge Cases 개선 시스템 종합 테스트 시작...")
        
        start_time = time.time()
        
        # Edge Cases 테스트 실행
        results = self.test_edge_cases()
        
        end_time = time.time()
        
        # 보고서 생성
        report = self.generate_improvement_report(results)
        
        # 결과 요약
        final_results = {
            "results": results,
            "total_time": end_time - start_time,
            "report": report
        }
        
        return final_results

def main():
    """메인 함수"""
    try:
        tester = EdgeCaseImprovementTester()
        results = tester.run_comprehensive_test()
        
        print("\n" + results["report"])
        
        # 결과를 파일로 저장
        os.makedirs("test_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 결과 저장
        with open(f"test_results/edge_case_improvement_test_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 텍스트 보고서 저장
        with open(f"test_results/edge_case_improvement_report_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(results["report"])
        
        print(f"\n📁 결과가 다음 파일들에 저장되었습니다:")
        print(f"  - test_results/edge_case_improvement_test_{timestamp}.json")
        print(f"  - test_results/edge_case_improvement_report_{timestamp}.txt")
        
        return results
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
