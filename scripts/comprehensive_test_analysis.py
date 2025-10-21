#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
대규모 테스트 결과 종합 분석 보고서
3000개 테스트 질의 결과를 종합적으로 분석하고 개선 방안을 제시합니다.
"""

import sys
import os
import json
from typing import Dict, List, Any
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class ComprehensiveTestAnalysis:
    """종합 테스트 분석 클래스"""
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.data = self._load_results()
        
    def _load_results(self) -> Dict[str, Any]:
        """결과 파일 로드"""
        with open(self.results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_comprehensive_report(self) -> str:
        """종합 분석 보고서 생성"""
        report = []
        report.append("=" * 120)
        report.append("LawFirmAI 대규모 테스트 종합 분석 보고서")
        report.append("=" * 120)
        
        # 1. 실행 개요
        report.append("\n📋 1. 테스트 실행 개요")
        report.append("-" * 60)
        metadata = self.data["metadata"]
        summary = self.data["summary"]
        
        report.append(f"  테스트 실행 시간: {metadata['test_run_at']}")
        report.append(f"  총 질의 수: {metadata['total_queries']:,}개")
        report.append(f"  총 소요 시간: {metadata['processing_time']:.2f}초")
        report.append(f"  처리 성능: {metadata['total_queries']/metadata['processing_time']:.1f} 질의/초")
        report.append(f"  평균 처리 시간: {metadata['processing_time']/metadata['total_queries']*1000:.2f}ms/질의")
        
        # 2. 전체 성능 분석
        report.append("\n📊 2. 전체 성능 분석")
        report.append("-" * 60)
        
        report.append(f"  전체 정확도: {summary['overall_accuracy']:.1%}")
        report.append(f"  정확한 예측: {summary['correct_predictions']:,}개")
        report.append(f"  잘못된 예측: {summary['incorrect_predictions']:,}개")
        report.append(f"  오류 발생: {summary['error_count']}개")
        report.append(f"  평균 신뢰도: {summary['average_confidence']:.3f}")
        report.append(f"  평균 점수: {summary['average_score']:.3f}")
        
        # 3. 카테고리별 상세 분석
        report.append("\n📋 3. 카테고리별 상세 분석")
        report.append("-" * 60)
        
        category_accuracies = summary["category_accuracies"]
        
        # 정확도 순으로 정렬
        sorted_categories = sorted(category_accuracies.items(), key=lambda x: x[1], reverse=True)
        
        for category, accuracy in sorted_categories:
            status = self._get_performance_status(accuracy)
            report.append(f"  {category}: {accuracy:.1%} {status}")
        
        # 4. 성능 등급별 분석
        report.append("\n🎯 4. 성능 등급별 분석")
        report.append("-" * 60)
        
        excellent = [cat for cat, acc in category_accuracies.items() if acc >= 0.90]
        good = [cat for cat, acc in category_accuracies.items() if 0.80 <= acc < 0.90]
        fair = [cat for cat, acc in category_accuracies.items() if 0.70 <= acc < 0.80]
        poor = [cat for cat, acc in category_accuracies.items() if acc < 0.70]
        
        report.append(f"  🏆 우수 (90% 이상): {len(excellent)}개 - {', '.join(excellent)}")
        report.append(f"  🥇 양호 (80-90%): {len(good)}개 - {', '.join(good)}")
        report.append(f"  🥈 보통 (70-80%): {len(fair)}개 - {', '.join(fair)}")
        report.append(f"  🥉 미흡 (70% 미만): {len(poor)}개 - {', '.join(poor)}")
        
        # 5. 민감도 분석
        report.append("\n🔒 5. 민감도 분석")
        report.append("-" * 60)
        
        sensitive_categories = ["personal_legal_advice", "medical_legal_advice", "criminal_case_advice", "illegal_activity_assistance"]
        general_categories = ["general_legal_information", "edge_cases"]
        
        sensitive_accuracy = sum(category_accuracies.get(cat, 0) for cat in sensitive_categories) / len(sensitive_categories)
        general_accuracy = sum(category_accuracies.get(cat, 0) for cat in general_categories) / len(general_categories)
        
        report.append(f"  민감한 질문 제한 정확도: {sensitive_accuracy:.1%}")
        report.append(f"  일반 정보 허용 정확도: {general_accuracy:.1%}")
        
        # 민감도별 상세 분석
        report.append("\n  📊 민감한 질문별 분석:")
        for cat in sensitive_categories:
            if cat in category_accuracies:
                acc = category_accuracies[cat]
                status = "✅ 우수" if acc >= 0.90 else "⚠️ 개선 필요" if acc >= 0.80 else "❌ 긴급 개선"
                report.append(f"    {cat}: {acc:.1%} {status}")
        
        report.append("\n  📊 일반 정보별 분석:")
        for cat in general_categories:
            if cat in category_accuracies:
                acc = category_accuracies[cat]
                status = "✅ 우수" if acc >= 0.80 else "⚠️ 개선 필요" if acc >= 0.70 else "❌ 긴급 개선"
                report.append(f"    {cat}: {acc:.1%} {status}")
        
        # 6. 오류 분석
        report.append("\n❌ 6. 오류 분석")
        report.append("-" * 60)
        
        if summary['error_count'] == 0:
            report.append("  ✅ 시스템 오류 없음")
        else:
            report.append(f"  ⚠️ 총 {summary['error_count']}개의 오류 발생")
        
        # 7. 신뢰도 분석
        report.append("\n📈 7. 신뢰도 분석")
        report.append("-" * 60)
        
        avg_confidence = summary['average_confidence']
        if avg_confidence >= 0.8:
            confidence_status = "✅ 높음"
        elif avg_confidence >= 0.6:
            confidence_status = "⚠️ 보통"
        else:
            confidence_status = "❌ 낮음"
        
        report.append(f"  평균 신뢰도: {avg_confidence:.3f} {confidence_status}")
        
        if avg_confidence < 0.5:
            report.append("  ⚠️ 신뢰도가 매우 낮습니다. 모델 튜닝이 필요합니다.")
        
        # 8. 개선 우선순위
        report.append("\n🎯 8. 개선 우선순위")
        report.append("-" * 60)
        
        # 정확도가 낮은 카테고리 식별
        low_accuracy_categories = [(cat, acc) for cat, acc in category_accuracies.items() if acc < 0.70]
        low_accuracy_categories.sort(key=lambda x: x[1])
        
        if low_accuracy_categories:
            report.append("  🔥 긴급 개선 필요 (70% 미만):")
            for cat, acc in low_accuracy_categories:
                report.append(f"    1. {cat}: {acc:.1%}")
        
        # 중간 정확도 카테고리
        medium_accuracy_categories = [(cat, acc) for cat, acc in category_accuracies.items() if 0.70 <= acc < 0.80]
        if medium_accuracy_categories:
            report.append("\n  ⚠️ 개선 권장 (70-80%):")
            for cat, acc in medium_accuracy_categories:
                report.append(f"    2. {cat}: {acc:.1%}")
        
        # 9. 구체적 개선 방안
        report.append("\n💡 9. 구체적 개선 방안")
        report.append("-" * 60)
        
        # 카테고리별 개선 방안
        improvement_plans = self._generate_improvement_plans(category_accuracies)
        for category, plans in improvement_plans.items():
            if plans:
                report.append(f"\n  📋 {category}:")
                for plan in plans:
                    report.append(f"    • {plan}")
        
        # 10. 시스템 최적화 방안
        report.append("\n⚡ 10. 시스템 최적화 방안")
        report.append("-" * 60)
        
        optimization_plans = self._generate_optimization_plans(summary, metadata)
        for plan in optimization_plans:
            report.append(f"  • {plan}")
        
        # 11. 최종 평가 및 권장사항
        report.append("\n🏆 11. 최종 평가 및 권장사항")
        report.append("-" * 60)
        
        overall_accuracy = summary['overall_accuracy']
        if overall_accuracy >= 0.95:
            final_grade = "🏆 우수"
            recommendation = "시스템이 매우 잘 작동하고 있습니다. 지속적인 모니터링을 유지하세요."
        elif overall_accuracy >= 0.90:
            final_grade = "🥇 양호"
            recommendation = "시스템이 잘 작동하고 있지만 일부 개선이 필요합니다."
        elif overall_accuracy >= 0.80:
            final_grade = "🥈 보통"
            recommendation = "시스템이 작동하고 있지만 상당한 개선이 필요합니다."
        else:
            final_grade = "🥉 미흡"
            recommendation = "시스템 개선이 시급합니다. 즉시 조치가 필요합니다."
        
        report.append(f"  최종 등급: {final_grade}")
        report.append(f"  전체 정확도: {overall_accuracy:.1%}")
        report.append(f"  권장사항: {recommendation}")
        
        # 12. 다음 단계
        report.append("\n🚀 12. 다음 단계")
        report.append("-" * 60)
        
        next_steps = self._generate_next_steps(category_accuracies, overall_accuracy)
        for i, step in enumerate(next_steps, 1):
            report.append(f"  {i}. {step}")
        
        report.append("\n" + "=" * 120)
        
        return "\n".join(report)
    
    def _get_performance_status(self, accuracy: float) -> str:
        """성능 상태 반환"""
        if accuracy >= 0.90:
            return "🏆 우수"
        elif accuracy >= 0.80:
            return "🥇 양호"
        elif accuracy >= 0.70:
            return "🥈 보통"
        else:
            return "🥉 미흡"
    
    def _generate_improvement_plans(self, category_accuracies: Dict[str, float]) -> Dict[str, List[str]]:
        """카테고리별 개선 방안 생성"""
        plans = {}
        
        # illegal_activity_assistance 개선 방안
        if category_accuracies.get("illegal_activity_assistance", 0) < 0.70:
            plans["illegal_activity_assistance"] = [
                "불법 행위 키워드 패턴 강화",
                "탈법 행위 감지 로직 개선",
                "의도 분석 알고리즘 정교화",
                "컨텍스트 분석 정확도 향상"
            ]
        
        # general_legal_information 개선 방안
        if category_accuracies.get("general_legal_information", 0) < 0.70:
            plans["general_legal_information"] = [
                "일반 정보 허용 패턴 확장",
                "학술적/교육적 질문 구분 로직 개선",
                "가설적 상황과 실제 상황 구분 강화",
                "정보 제공과 조언 제공 구분 정교화"
            ]
        
        # edge_cases 개선 방안
        if category_accuracies.get("edge_cases", 0) < 0.70:
            plans["edge_cases"] = [
                "애매한 케이스 처리 로직 개선",
                "최소 정보 질문 처리 방안 수립",
                "안전 우선 정책 적용",
                "사용자 의도 명확화 유도"
            ]
        
        # complex_mixed_questions 개선 방안
        if category_accuracies.get("complex_mixed_questions", 0) < 0.80:
            plans["complex_mixed_questions"] = [
                "복합 질문 분해 알고리즘 개발",
                "일반 정보와 개인 조언 구분 강화",
                "맥락 기반 의도 분석 개선",
                "단계별 검증 로직 정교화"
            ]
        
        return plans
    
    def _generate_optimization_plans(self, summary: Dict, metadata: Dict) -> List[str]:
        """시스템 최적화 방안 생성"""
        plans = []
        
        # 성능 최적화
        queries_per_second = metadata['total_queries'] / metadata['processing_time']
        if queries_per_second < 20:
            plans.append("처리 성능 향상을 위한 병렬 처리 최적화")
            plans.append("캐싱 시스템 도입으로 응답 시간 단축")
        
        # 신뢰도 최적화
        if summary['average_confidence'] < 0.5:
            plans.append("모델 신뢰도 향상을 위한 훈련 데이터 보강")
            plans.append("임계값 조정으로 의사결정 정확도 향상")
        
        # 메모리 최적화
        plans.append("메모리 사용량 최적화로 안정성 향상")
        plans.append("가비지 컬렉션 최적화")
        
        # 모니터링 강화
        plans.append("실시간 성능 모니터링 시스템 구축")
        plans.append("자동 알림 및 경고 시스템 도입")
        
        return plans
    
    def _generate_next_steps(self, category_accuracies: Dict[str, float], overall_accuracy: float) -> List[str]:
        """다음 단계 생성"""
        steps = []
        
        if overall_accuracy < 0.80:
            steps.append("긴급: 정확도가 낮은 카테고리 우선 개선")
            steps.append("핵심 알고리즘 재검토 및 튜닝")
        
        # 낮은 정확도 카테고리 개선
        low_accuracy_categories = [cat for cat, acc in category_accuracies.items() if acc < 0.70]
        if low_accuracy_categories:
            steps.append(f"우선순위 1: {', '.join(low_accuracy_categories)} 카테고리 개선")
        
        # 중간 정확도 카테고리 개선
        medium_accuracy_categories = [cat for cat, acc in category_accuracies.items() if 0.70 <= acc < 0.80]
        if medium_accuracy_categories:
            steps.append(f"우선순위 2: {', '.join(medium_accuracy_categories)} 카테고리 개선")
        
        steps.append("A/B 테스트를 통한 개선 효과 검증")
        steps.append("사용자 피드백 수집 및 반영")
        steps.append("정기적인 성능 모니터링 체계 구축")
        
        return steps

def main():
    """메인 함수"""
    try:
        # 최신 결과 파일 찾기
        import glob
        result_files = glob.glob("test_results/massive_test_results_*.json")
        # 분석 파일이 아닌 실제 결과 파일만 선택
        result_files = [f for f in result_files if not f.endswith('_analysis.json')]
        if not result_files:
            print("테스트 결과 파일을 찾을 수 없습니다.")
            return
        
        latest_file = max(result_files, key=os.path.getctime)
        print(f"분석할 파일: {latest_file}")
        
        # 종합 분석 실행
        analyzer = ComprehensiveTestAnalysis(latest_file)
        report = analyzer.generate_comprehensive_report()
        
        # 보고서 출력
        print("\n" + report)
        
        # 보고서 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"test_results/comprehensive_analysis_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n종합 분석 보고서가 {report_file}에 저장되었습니다.")
        
        return report
        
    except Exception as e:
        print(f"분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
