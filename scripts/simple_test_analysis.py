#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
대규모 테스트 결과 요약 보고서
3000개 테스트 질의 결과를 요약하여 간단한 보고서를 생성합니다.
"""

import sys
import os
import json
import glob
from datetime import datetime

def generate_simple_report():
    """간단한 보고서 생성"""
    
    # 최신 결과 파일 찾기
    result_files = glob.glob("test_results/massive_test_results_*.json")
    result_files = [f for f in result_files if not f.endswith('_analysis.json')]
    
    if not result_files:
        print("테스트 결과 파일을 찾을 수 없습니다.")
        return
    
    latest_file = max(result_files, key=os.path.getctime)
    print(f"분석할 파일: {latest_file}")
    
    # 결과 로드
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data["metadata"]
    summary = data["summary"]
    
    # 보고서 생성
    report = []
    report.append("=" * 80)
    report.append("LawFirmAI 대규모 테스트 결과 요약 보고서")
    report.append("=" * 80)
    
    # 1. 실행 개요
    report.append("\n1. 테스트 실행 개요")
    report.append("-" * 40)
    report.append(f"테스트 실행 시간: {metadata['test_run_at']}")
    report.append(f"총 질의 수: {metadata['total_queries']:,}개")
    report.append(f"총 소요 시간: {metadata['processing_time']:.2f}초")
    report.append(f"처리 성능: {metadata['total_queries']/metadata['processing_time']:.1f} 질의/초")
    report.append(f"평균 처리 시간: {metadata['processing_time']/metadata['total_queries']*1000:.2f}ms/질의")
    
    # 2. 전체 성능
    report.append("\n2. 전체 성능 분석")
    report.append("-" * 40)
    report.append(f"전체 정확도: {summary['overall_accuracy']:.1%}")
    report.append(f"정확한 예측: {summary['correct_predictions']:,}개")
    report.append(f"잘못된 예측: {summary['incorrect_predictions']:,}개")
    report.append(f"오류 발생: {summary['error_count']}개")
    report.append(f"평균 신뢰도: {summary['average_confidence']:.3f}")
    report.append(f"평균 점수: {summary['average_score']:.3f}")
    
    # 3. 카테고리별 정확도
    report.append("\n3. 카테고리별 정확도")
    report.append("-" * 40)
    category_accuracies = summary["category_accuracies"]
    
    # 정확도 순으로 정렬
    sorted_categories = sorted(category_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    for category, accuracy in sorted_categories:
        if accuracy >= 0.90:
            status = "[우수]"
        elif accuracy >= 0.80:
            status = "[양호]"
        elif accuracy >= 0.70:
            status = "[보통]"
        else:
            status = "[미흡]"
        report.append(f"{category}: {accuracy:.1%} {status}")
    
    # 4. 성능 등급별 분석
    report.append("\n4. 성능 등급별 분석")
    report.append("-" * 40)
    
    excellent = [cat for cat, acc in category_accuracies.items() if acc >= 0.90]
    good = [cat for cat, acc in category_accuracies.items() if 0.80 <= acc < 0.90]
    fair = [cat for cat, acc in category_accuracies.items() if 0.70 <= acc < 0.80]
    poor = [cat for cat, acc in category_accuracies.items() if acc < 0.70]
    
    report.append(f"우수 (90% 이상): {len(excellent)}개 - {', '.join(excellent)}")
    report.append(f"양호 (80-90%): {len(good)}개 - {', '.join(good)}")
    report.append(f"보통 (70-80%): {len(fair)}개 - {', '.join(fair)}")
    report.append(f"미흡 (70% 미만): {len(poor)}개 - {', '.join(poor)}")
    
    # 5. 민감도 분석
    report.append("\n5. 민감도 분석")
    report.append("-" * 40)
    
    sensitive_categories = ["personal_legal_advice", "medical_legal_advice", "criminal_case_advice", "illegal_activity_assistance"]
    general_categories = ["general_legal_information", "edge_cases"]
    
    sensitive_accuracy = sum(category_accuracies.get(cat, 0) for cat in sensitive_categories) / len(sensitive_categories)
    general_accuracy = sum(category_accuracies.get(cat, 0) for cat in general_categories) / len(general_categories)
    
    report.append(f"민감한 질문 제한 정확도: {sensitive_accuracy:.1%}")
    report.append(f"일반 정보 허용 정확도: {general_accuracy:.1%}")
    
    # 6. 개선 우선순위
    report.append("\n6. 개선 우선순위")
    report.append("-" * 40)
    
    # 정확도가 낮은 카테고리 식별
    low_accuracy_categories = [(cat, acc) for cat, acc in category_accuracies.items() if acc < 0.70]
    low_accuracy_categories.sort(key=lambda x: x[1])
    
    if low_accuracy_categories:
        report.append("긴급 개선 필요 (70% 미만):")
        for i, (cat, acc) in enumerate(low_accuracy_categories, 1):
            report.append(f"  {i}. {cat}: {acc:.1%}")
    
    # 중간 정확도 카테고리
    medium_accuracy_categories = [(cat, acc) for cat, acc in category_accuracies.items() if 0.70 <= acc < 0.80]
    if medium_accuracy_categories:
        report.append("\n개선 권장 (70-80%):")
        for i, (cat, acc) in enumerate(medium_accuracy_categories, 1):
            report.append(f"  {i}. {cat}: {acc:.1%}")
    
    # 7. 구체적 개선 방안
    report.append("\n7. 구체적 개선 방안")
    report.append("-" * 40)
    
    if "illegal_activity_assistance" in low_accuracy_categories:
        report.append("불법 행위 조력 감지:")
        report.append("  - 불법 행위 키워드 패턴 강화")
        report.append("  - 탈법 행위 감지 로직 개선")
        report.append("  - 의도 분석 알고리즘 정교화")
    
    if "general_legal_information" in low_accuracy_categories:
        report.append("\n일반 정보 허용:")
        report.append("  - 일반 정보 허용 패턴 확장")
        report.append("  - 학술적/교육적 질문 구분 로직 개선")
        report.append("  - 가설적 상황과 실제 상황 구분 강화")
    
    if "edge_cases" in low_accuracy_categories:
        report.append("\n애매한 케이스 처리:")
        report.append("  - 애매한 케이스 처리 로직 개선")
        report.append("  - 최소 정보 질문 처리 방안 수립")
        report.append("  - 안전 우선 정책 적용")
    
    # 8. 최종 평가
    report.append("\n8. 최종 평가")
    report.append("-" * 40)
    
    overall_accuracy = summary['overall_accuracy']
    if overall_accuracy >= 0.95:
        final_grade = "우수"
        recommendation = "시스템이 매우 잘 작동하고 있습니다. 지속적인 모니터링을 유지하세요."
    elif overall_accuracy >= 0.90:
        final_grade = "양호"
        recommendation = "시스템이 잘 작동하고 있지만 일부 개선이 필요합니다."
    elif overall_accuracy >= 0.80:
        final_grade = "보통"
        recommendation = "시스템이 작동하고 있지만 상당한 개선이 필요합니다."
    else:
        final_grade = "미흡"
        recommendation = "시스템 개선이 시급합니다. 즉시 조치가 필요합니다."
    
    report.append(f"최종 등급: {final_grade}")
    report.append(f"전체 정확도: {overall_accuracy:.1%}")
    report.append(f"권장사항: {recommendation}")
    
    # 9. 다음 단계
    report.append("\n9. 다음 단계")
    report.append("-" * 40)
    
    if overall_accuracy < 0.80:
        report.append("1. 긴급: 정확도가 낮은 카테고리 우선 개선")
        report.append("2. 핵심 알고리즘 재검토 및 튜닝")
    
    if low_accuracy_categories:
        report.append(f"3. 우선순위 1: {', '.join([cat for cat, _ in low_accuracy_categories])} 카테고리 개선")
    
    if medium_accuracy_categories:
        report.append(f"4. 우선순위 2: {', '.join([cat for cat, _ in medium_accuracy_categories])} 카테고리 개선")
    
    report.append("5. A/B 테스트를 통한 개선 효과 검증")
    report.append("6. 사용자 피드백 수집 및 반영")
    report.append("7. 정기적인 성능 모니터링 체계 구축")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)

def main():
    """메인 함수"""
    try:
        report = generate_simple_report()
        
        # 보고서 출력
        print("\n" + report)
        
        # 보고서 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"test_results/simple_analysis_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n간단 분석 보고서가 {report_file}에 저장되었습니다.")
        
        return report
        
    except Exception as e:
        print(f"분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
