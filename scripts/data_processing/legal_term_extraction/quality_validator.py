# -*- coding: utf-8 -*-
"""
법률 용어 품질 검증 시스템
추출된 용어들의 품질을 검증하고 개선사항을 제안
"""

import json
import logging
from typing import Dict, List, Tuple, Set
from pathlib import Path
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class QualityValidator:
    """법률 용어 품질 검증기"""
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 품질 검증 기준
        self.quality_criteria = {
            "min_synonyms": 0,      # 최소 동의어 수
            "min_related_terms": 0,  # 최소 관련 용어 수
            "min_related_laws": 0,   # 최소 관련 법률 수
            "min_precedent_keywords": 0,  # 최소 판례 키워드 수
            "min_confidence": 0.3,   # 최소 신뢰도
            "min_frequency": 1       # 최소 빈도
        }
        
        # 품질 점수 가중치
        self.quality_weights = {
            "synonyms": 0.2,        # 동의어 20%
            "related_terms": 0.25,   # 관련 용어 25%
            "related_laws": 0.25,    # 관련 법률 25%
            "precedent_keywords": 0.15,  # 판례 키워드 15%
            "confidence": 0.15      # 신뢰도 15%
        }
        
        # 검증 결과 저장
        self.validation_results = defaultdict(dict)
        self.quality_issues = defaultdict(list)
        self.improvement_suggestions = defaultdict(list)
    
    def validate_term_quality(self, term: str, term_info: Dict) -> Tuple[float, List[str]]:
        """개별 용어 품질 검증"""
        quality_score = 0.0
        issues = []
        
        # 1. 동의어 검증
        synonyms = term_info.get("synonyms", [])
        if len(synonyms) >= self.quality_criteria["min_synonyms"]:
            quality_score += self.quality_weights["synonyms"]
        else:
            issues.append(f"동의어 부족 (현재: {len(synonyms)}, 최소: {self.quality_criteria['min_synonyms']})")
        
        # 2. 관련 용어 검증
        related_terms = term_info.get("related_terms", [])
        if len(related_terms) >= self.quality_criteria["min_related_terms"]:
            quality_score += self.quality_weights["related_terms"]
        else:
            issues.append(f"관련 용어 부족 (현재: {len(related_terms)}, 최소: {self.quality_criteria['min_related_terms']})")
        
        # 3. 관련 법률 검증
        related_laws = term_info.get("related_laws", [])
        if len(related_laws) >= self.quality_criteria["min_related_laws"]:
            quality_score += self.quality_weights["related_laws"]
        else:
            issues.append(f"관련 법률 부족 (현재: {len(related_laws)}, 최소: {self.quality_criteria['min_related_laws']})")
        
        # 4. 판례 키워드 검증
        precedent_keywords = term_info.get("precedent_keywords", [])
        if len(precedent_keywords) >= self.quality_criteria["min_precedent_keywords"]:
            quality_score += self.quality_weights["precedent_keywords"]
        else:
            issues.append(f"판례 키워드 부족 (현재: {len(precedent_keywords)}, 최소: {self.quality_criteria['min_precedent_keywords']})")
        
        # 5. 신뢰도 검증
        confidence = term_info.get("confidence", 0.0)
        if confidence >= self.quality_criteria["min_confidence"]:
            quality_score += self.quality_weights["confidence"]
        else:
            issues.append(f"신뢰도 부족 (현재: {confidence:.2f}, 최소: {self.quality_criteria['min_confidence']})")
        
        # 6. 빈도 검증
        frequency = term_info.get("frequency", 0)
        if frequency < self.quality_criteria["min_frequency"]:
            issues.append(f"빈도 부족 (현재: {frequency}, 최소: {self.quality_criteria['min_frequency']})")
        
        return quality_score, issues
    
    def validate_dictionary_quality(self, dictionary: Dict) -> Dict:
        """전체 사전 품질 검증"""
        self.logger.info("Starting dictionary quality validation")
        
        validation_summary = {
            "total_terms": len(dictionary),
            "high_quality_terms": 0,
            "medium_quality_terms": 0,
            "low_quality_terms": 0,
            "rejected_terms": 0,
            "quality_distribution": defaultdict(int),
            "common_issues": Counter(),
            "domain_quality": defaultdict(dict)
        }
        
        for term, term_info in dictionary.items():
            # 개별 용어 검증
            quality_score, issues = self.validate_term_quality(term, term_info)
            
            # 검증 결과 저장
            self.validation_results[term] = {
                "quality_score": quality_score,
                "issues": issues,
                "term_info": term_info
            }
            
            # 품질 등급 분류
            if quality_score >= 0.8:
                validation_summary["high_quality_terms"] += 1
                quality_grade = "high"
            elif quality_score >= 0.6:
                validation_summary["medium_quality_terms"] += 1
                quality_grade = "medium"
            elif quality_score >= 0.4:
                validation_summary["low_quality_terms"] += 1
                quality_grade = "low"
            else:
                validation_summary["rejected_terms"] += 1
                quality_grade = "rejected"
            
            validation_summary["quality_distribution"][quality_grade] += 1
            
            # 공통 문제점 수집
            for issue in issues:
                validation_summary["common_issues"][issue] += 1
            
            # 도메인별 품질 분석
            domain = term_info.get("domain", "기타")
            if domain not in validation_summary["domain_quality"]:
                validation_summary["domain_quality"][domain] = {
                    "total": 0, "high": 0, "medium": 0, "low": 0, "rejected": 0
                }
            
            validation_summary["domain_quality"][domain]["total"] += 1
            validation_summary["domain_quality"][domain][quality_grade] += 1
        
        return validation_summary
    
    def generate_improvement_suggestions(self, dictionary: Dict) -> Dict:
        """개선 제안 생성"""
        suggestions = {
            "overall_suggestions": [],
            "term_specific_suggestions": {},
            "domain_suggestions": defaultdict(list)
        }
        
        # 전체 개선 제안
        total_terms = len(dictionary)
        high_quality_ratio = sum(1 for result in self.validation_results.values() 
                               if result["quality_score"] >= 0.8) / total_terms
        
        if high_quality_ratio < 0.7:
            suggestions["overall_suggestions"].append(
                f"전체 용어의 고품질 비율이 낮습니다 ({high_quality_ratio:.1%}). "
                "동의어와 관련 용어를 더 많이 추가하세요."
            )
        
        # 용어별 개선 제안
        for term, result in self.validation_results.items():
            term_suggestions = []
            
            if result["quality_score"] < 0.6:
                term_info = result["term_info"]
                
                if len(term_info.get("synonyms", [])) < 2:
                    term_suggestions.append("동의어를 더 추가하세요")
                
                if len(term_info.get("related_terms", [])) < 3:
                    term_suggestions.append("관련 용어를 더 추가하세요")
                
                if len(term_info.get("related_laws", [])) < 2:
                    term_suggestions.append("관련 법률을 더 추가하세요")
                
                if len(term_info.get("precedent_keywords", [])) < 2:
                    term_suggestions.append("판례 키워드를 더 추가하세요")
            
            if term_suggestions:
                suggestions["term_specific_suggestions"][term] = term_suggestions
        
        # 도메인별 개선 제안
        domain_quality = defaultdict(list)
        for term, result in self.validation_results.items():
            domain = result["term_info"].get("domain", "기타")
            domain_quality[domain].append(result["quality_score"])
        
        for domain, scores in domain_quality.items():
            avg_score = sum(scores) / len(scores)
            if avg_score < 0.7:
                suggestions["domain_suggestions"][domain].append(
                    f"{domain} 도메인의 평균 품질이 낮습니다 ({avg_score:.2f}). "
                    "해당 도메인의 용어 정보를 보강하세요."
                )
        
        return suggestions
    
    def filter_high_quality_terms(self, dictionary: Dict, min_quality: float = 0.6) -> Dict:
        """고품질 용어만 필터링"""
        filtered_dict = {}
        
        for term, term_info in dictionary.items():
            if term in self.validation_results:
                quality_score = self.validation_results[term]["quality_score"]
                if quality_score >= min_quality:
                    filtered_dict[term] = term_info
        
        self.logger.info(f"Filtered {len(filtered_dict)} high-quality terms from {len(dictionary)} total terms")
        return filtered_dict
    
    def generate_quality_report(self, validation_summary: Dict, suggestions: Dict) -> str:
        """품질 보고서 생성"""
        report = []
        report.append("=== 법률 용어 사전 품질 검증 보고서 ===\n")
        
        # 전체 통계
        report.append("1. 전체 통계")
        report.append(f"   총 용어 수: {validation_summary['total_terms']}")
        report.append(f"   고품질 용어: {validation_summary['high_quality_terms']} ({validation_summary['high_quality_terms']/validation_summary['total_terms']:.1%})")
        report.append(f"   중품질 용어: {validation_summary['medium_quality_terms']} ({validation_summary['medium_quality_terms']/validation_summary['total_terms']:.1%})")
        report.append(f"   저품질 용어: {validation_summary['low_quality_terms']} ({validation_summary['low_quality_terms']/validation_summary['total_terms']:.1%})")
        report.append(f"   제외된 용어: {validation_summary['rejected_terms']} ({validation_summary['rejected_terms']/validation_summary['total_terms']:.1%})")
        report.append("")
        
        # 도메인별 품질
        report.append("2. 도메인별 품질")
        for domain, stats in validation_summary["domain_quality"].items():
            if stats["total"] > 0:
                high_ratio = stats["high"] / stats["total"]
                report.append(f"   {domain}: {stats['total']}개 (고품질 {high_ratio:.1%})")
        report.append("")
        
        # 공통 문제점
        report.append("3. 공통 문제점 (상위 5개)")
        for issue, count in validation_summary["common_issues"].most_common(5):
            report.append(f"   {issue}: {count}개")
        report.append("")
        
        # 개선 제안
        report.append("4. 개선 제안")
        for suggestion in suggestions["overall_suggestions"]:
            report.append(f"   - {suggestion}")
        report.append("")
        
        return "\n".join(report)
    
    def save_validation_results(self, output_path: str, validation_summary: Dict, suggestions: Dict):
        """검증 결과 저장"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            results = {
                "validation_summary": validation_summary,
                "suggestions": suggestions,
                "detailed_results": dict(self.validation_results)
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Validation results saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving validation results: {e}")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='법률 용어 품질 검증기')
    parser.add_argument('--input_file', type=str, required=True,
                       help='검증할 사전 파일 경로')
    parser.add_argument('--output_file', type=str,
                       default='data/extracted_terms/quality_validation_results.json',
                       help='검증 결과 출력 파일 경로')
    parser.add_argument('--min_quality', type=float, default=0.6,
                       help='최소 품질 기준 (기본값: 0.6)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='로그 레벨')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 품질 검증 실행
    validator = QualityValidator()
    
    # 입력 파일 로드
    with open(args.input_file, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    
    # 품질 검증
    validation_summary = validator.validate_dictionary_quality(dictionary)
    
    # 개선 제안 생성
    suggestions = validator.generate_improvement_suggestions(dictionary)
    
    # 고품질 용어 필터링
    high_quality_dict = validator.filter_high_quality_terms(dictionary, args.min_quality)
    
    # 결과 저장
    validator.save_validation_results(args.output_file, validation_summary, suggestions)
    
    # 품질 보고서 생성 및 출력
    report = validator.generate_quality_report(validation_summary, suggestions)
    print(report)
    
    # 고품질 사전 저장
    high_quality_output = args.output_file.replace('.json', '_high_quality.json')
    with open(high_quality_output, 'w', encoding='utf-8') as f:
        json.dump(high_quality_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n고품질 용어 {len(high_quality_dict)}개가 {high_quality_output}에 저장되었습니다.")


if __name__ == "__main__":
    main()
