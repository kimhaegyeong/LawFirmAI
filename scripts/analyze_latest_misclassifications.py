#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최신 오분류 사례 분석 스크립트 (수정된 버전)
- 91.5% 정확도 테스트 결과에서 오분류 사례 분석
- 카테고리별 오분류 패턴 식별
- 추가 개선 방안 제시
"""

import os
import json
import glob
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import re

def load_latest_test_results() -> Dict[str, Any]:
    """최신 테스트 결과 로드"""
    files = glob.glob(os.path.join("test_results", "massive_test_results_*.json"))
    files = [f for f in files if not f.endswith("_analysis.json")]
    if not files:
        raise FileNotFoundError("테스트 결과 파일이 없습니다.")
    latest = max(files, key=os.path.getctime)
    
    with open(latest, "r", encoding="utf-8") as fp:
        return json.load(fp)

def analyze_misclassifications(results: Dict[str, Any]) -> Dict[str, Any]:
    """오분류 사례 분석"""
    detailed_results = results.get("detailed_results", [])
    
    # 카테고리별 오분류 분석
    category_misclassifications = defaultdict(list)
    category_stats = defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0})
    
    # 전체 통계
    total_queries = len(detailed_results)
    correct_predictions = 0
    incorrect_predictions = 0
    
    for result in detailed_results:
        query = result.get("query", "")
        category = result.get("category", "unknown")
        expected_restricted = result.get("expected_restricted", False)
        predicted_restricted = result.get("actual_restricted", False)  # 수정: actual_restricted 사용
        
        # 통계 업데이트
        category_stats[category]["total"] += 1
        if expected_restricted == predicted_restricted:
            correct_predictions += 1
            category_stats[category]["correct"] += 1
        else:
            incorrect_predictions += 1
            category_stats[category]["incorrect"] += 1
            
            # 오분류 사례 저장
            misclassification = {
                "query": query,
                "expected_restricted": expected_restricted,
                "predicted_restricted": predicted_restricted,
                "confidence": result.get("confidence", 0.0),
                "reasoning": result.get("reasoning", []),
                "subcategory": result.get("subcategory", "unknown")
            }
            category_misclassifications[category].append(misclassification)
    
    return {
        "total_stats": {
            "total_queries": total_queries,
            "correct_predictions": correct_predictions,
            "incorrect_predictions": incorrect_predictions,
            "accuracy": correct_predictions / total_queries if total_queries > 0 else 0
        },
        "category_stats": dict(category_stats),
        "category_misclassifications": dict(category_misclassifications)
    }

def analyze_misclassification_patterns(misclassifications: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """오분류 패턴 분석"""
    patterns = {}
    
    for category, cases in misclassifications.items():
        if not cases:
            continue
            
        # 허용으로 잘못 분류된 사례 (제한되어야 하는데 허용됨)
        false_allows = [case for case in cases if case["expected_restricted"] and not case["predicted_restricted"]]
        
        # 제한으로 잘못 분류된 사례 (허용되어야 하는데 제한됨)
        false_restricts = [case for case in cases if not case["expected_restricted"] and case["predicted_restricted"]]
        
        # 키워드 패턴 분석
        false_allow_keywords = analyze_keyword_patterns(false_allows)
        false_restrict_keywords = analyze_keyword_patterns(false_restricts)
        
        # 서브카테고리별 분석
        subcategory_analysis = analyze_subcategory_patterns(cases)
        
        patterns[category] = {
            "false_allows": {
                "count": len(false_allows),
                "percentage": len(false_allows) / len(cases) * 100 if cases else 0,
                "keyword_patterns": false_allow_keywords,
                "cases": false_allows[:10]  # 상위 10개만 저장
            },
            "false_restricts": {
                "count": len(false_restricts),
                "percentage": len(false_restricts) / len(cases) * 100 if cases else 0,
                "keyword_patterns": false_restrict_keywords,
                "cases": false_restricts[:10]  # 상위 10개만 저장
            },
            "subcategory_analysis": subcategory_analysis
        }
    
    return patterns

def analyze_keyword_patterns(cases: List[Dict]) -> Dict[str, Any]:
    """키워드 패턴 분석"""
    if not cases:
        return {}
    
    # 모든 쿼리에서 공통 키워드 찾기
    all_queries = [case["query"] for case in cases]
    
    # 단어 빈도 분석
    word_freq = Counter()
    for query in all_queries:
        words = re.findall(r'\b\w+\b', query.lower())
        word_freq.update(words)
    
    # 자주 나타나는 단어들
    common_words = dict(word_freq.most_common(20))
    
    # 패턴 분석
    patterns = {
        "common_words": common_words,
        "query_lengths": [len(query) for query in all_queries],
        "avg_length": sum(len(query) for query in all_queries) / len(all_queries) if all_queries else 0
    }
    
    return patterns

def analyze_subcategory_patterns(cases: List[Dict]) -> Dict[str, Any]:
    """서브카테고리별 패턴 분석"""
    subcategory_stats = defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0})
    
    for case in cases:
        subcategory = case.get("subcategory", "unknown")
        subcategory_stats[subcategory]["total"] += 1
        subcategory_stats[subcategory]["incorrect"] += 1  # 이미 오분류된 케이스들
    
    return dict(subcategory_stats)

def generate_improvement_suggestions(patterns: Dict[str, Any]) -> Dict[str, List[str]]:
    """개선 제안 생성"""
    suggestions = {}
    
    for category, pattern_data in patterns.items():
        category_suggestions = []
        
        # 허용으로 잘못 분류된 사례 분석
        false_allows = pattern_data.get("false_allows", {})
        if false_allows.get("count", 0) > 0:
            category_suggestions.append(f"허용으로 잘못 분류된 사례: {false_allows['count']}개 ({false_allows['percentage']:.1f}%)")
            
            # 공통 키워드 기반 제안
            common_words = false_allows.get("keyword_patterns", {}).get("common_words", {})
            if common_words:
                top_words = list(common_words.keys())[:5]
                category_suggestions.append(f"   - 공통 키워드: {', '.join(top_words)}")
                category_suggestions.append(f"   - 제안: 이 키워드들을 금지 키워드에 추가하거나 패턴을 강화")
        
        # 제한으로 잘못 분류된 사례 분석
        false_restricts = pattern_data.get("false_restricts", {})
        if false_restricts.get("count", 0) > 0:
            category_suggestions.append(f"제한으로 잘못 분류된 사례: {false_restricts['count']}개 ({false_restricts['percentage']:.1f}%)")
            
            # 공통 키워드 기반 제안
            common_words = false_restricts.get("keyword_patterns", {}).get("common_words", {})
            if common_words:
                top_words = list(common_words.keys())[:5]
                category_suggestions.append(f"   - 공통 키워드: {', '.join(top_words)}")
                category_suggestions.append(f"   - 제안: 이 키워드들을 허용 키워드에 추가하거나 패턴을 완화")
        
        if category_suggestions:
            suggestions[category] = category_suggestions
    
    return suggestions

def save_analysis_report(analysis: Dict[str, Any], patterns: Dict[str, Any], suggestions: Dict[str, List[str]]):
    """분석 보고서 저장"""
    report = {
        "analysis_timestamp": os.popen("date").read().strip(),
        "total_stats": analysis["total_stats"],
        "category_stats": analysis["category_stats"],
        "misclassification_patterns": patterns,
        "improvement_suggestions": suggestions
    }
    
    # JSON 파일로 저장
    report_file = "test_results/latest_misclassification_analysis_report.json"
    with open(report_file, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    
    # 텍스트 보고서 생성
    text_report = generate_text_report(report)
    text_file = "test_results/latest_misclassification_analysis_report.txt"
    with open(text_file, "w", encoding="utf-8") as fp:
        fp.write(text_report)
    
    print(f"분석 보고서 저장 완료:")
    print(f"   - JSON: {report_file}")
    print(f"   - 텍스트: {text_file}")

def generate_text_report(report: Dict[str, Any]) -> str:
    """텍스트 보고서 생성"""
    lines = []
    lines.append("=" * 80)
    lines.append("최신 오분류 사례 분석 보고서 (91.5% 정확도)")
    lines.append("=" * 80)
    lines.append("")
    
    # 전체 통계
    total_stats = report["total_stats"]
    lines.append("전체 통계:")
    lines.append(f"  총 질의 수: {total_stats['total_queries']:,}")
    lines.append(f"  정확한 예측: {total_stats['correct_predictions']:,}")
    lines.append(f"  잘못된 예측: {total_stats['incorrect_predictions']:,}")
    lines.append(f"  전체 정확도: {total_stats['accuracy']:.1%}")
    lines.append("")
    
    # 카테고리별 통계
    lines.append("카테고리별 정확도:")
    category_stats = report["category_stats"]
    for category, stats in category_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        lines.append(f"  {category}: {accuracy:.1%} ({stats['correct']}/{stats['total']})")
    lines.append("")
    
    # 오분류 패턴 분석
    lines.append("오분류 패턴 분석:")
    patterns = report["misclassification_patterns"]
    for category, pattern_data in patterns.items():
        lines.append(f"\n{category}:")
        
        false_allows = pattern_data.get("false_allows", {})
        if false_allows.get("count", 0) > 0:
            lines.append(f"  허용으로 잘못 분류: {false_allows['count']}개 ({false_allows['percentage']:.1f}%)")
            common_words = false_allows.get("keyword_patterns", {}).get("common_words", {})
            if common_words:
                top_words = list(common_words.keys())[:5]
                lines.append(f"     공통 키워드: {', '.join(top_words)}")
        
        false_restricts = pattern_data.get("false_restricts", {})
        if false_restricts.get("count", 0) > 0:
            lines.append(f"  제한으로 잘못 분류: {false_restricts['count']}개 ({false_restricts['percentage']:.1f}%)")
            common_words = false_restricts.get("keyword_patterns", {}).get("common_words", {})
            if common_words:
                top_words = list(common_words.keys())[:5]
                lines.append(f"     공통 키워드: {', '.join(top_words)}")
    
    # 개선 제안
    lines.append("\n개선 제안:")
    suggestions = report["improvement_suggestions"]
    for category, category_suggestions in suggestions.items():
        lines.append(f"\n{category}:")
        for suggestion in category_suggestions:
            lines.append(f"  {suggestion}")
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)

def main():
    """메인 함수"""
    print("최신 오분류 사례 분석 시작...")
    
    try:
        # 최신 테스트 결과 로드
        results = load_latest_test_results()
        print(f"테스트 결과 로드 완료: {len(results.get('detailed_results', []))}개 질의")
        
        # 오분류 사례 분석
        analysis = analyze_misclassifications(results)
        print(f"오분류 사례 분석 완료")
        
        # 오분류 패턴 분석
        patterns = analyze_misclassification_patterns(analysis["category_misclassifications"])
        print(f"오분류 패턴 분석 완료")
        
        # 개선 제안 생성
        suggestions = generate_improvement_suggestions(patterns)
        print(f"개선 제안 생성 완료")
        
        # 분석 보고서 저장
        save_analysis_report(analysis, patterns, suggestions)
        
        print("\n분석 결과 요약:")
        total_stats = analysis["total_stats"]
        print(f"  전체 정확도: {total_stats['accuracy']:.1%}")
        print(f"  오분류 사례: {total_stats['incorrect_predictions']}개")
        
        # 가장 문제가 많은 카테고리 식별
        category_stats = analysis["category_stats"]
        if category_stats:
            worst_category = min(category_stats.items(), 
                               key=lambda x: x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 1)
            print(f"  가장 문제가 많은 카테고리: {worst_category[0]} ({worst_category[1]['correct']}/{worst_category[1]['total']})")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
