# -*- coding: utf-8 -*-
"""
용어 검증기

수집된 법률 용어의 품질을 검증하고 개선사항을 제안합니다.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class TermValidator:
    """용어 검증기 클래스"""
    
    def __init__(self, dictionary=None):
        """검증기 초기화"""
        self.dictionary = dictionary
        self.validation_rules = self._load_validation_rules()
        
        logger.info("TermValidator initialized")
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """검증 규칙 로드"""
        return {
            'term_name': {
                'min_length': 2,
                'max_length': 50,
                'pattern': r'^[가-힣a-zA-Z0-9\s\-_]+$',
                'required': True
            },
            'definition': {
                'min_length': 10,
                'max_length': 1000,
                'required': True
            },
            'category': {
                'allowed_values': [
                    '민사법', '형사법', '상사법', '노동법', '행정법',
                    '환경법', '소비자법', '지적재산권법', '금융법', '기타'
                ],
                'required': True
            },
            'frequency': {
                'min_value': 0,
                'max_value': 10000,
                'required': True
            }
        }
    
    def validate_all_terms(self, dictionary) -> Dict[str, Any]:
        """모든 용어 검증"""
        if not dictionary:
            logger.error("사전이 제공되지 않았습니다.")
            return {'is_valid': False, 'issues': ['사전이 제공되지 않았습니다.']}
        
        self.dictionary = dictionary
        
        logger.info(f"용어 검증 시작 - {len(dictionary.terms)}개 용어")
        
        validation_results = {
            'total_terms': len(dictionary.terms),
            'valid_terms': 0,
            'invalid_terms': 0,
            'issues': [],
            'term_issues': {},
            'statistics': {},
            'recommendations': []
        }
        
        # 개별 용어 검증
        for term_id, term_data in dictionary.terms.items():
            term_validation = self.validate_single_term(term_data)
            
            if term_validation['is_valid']:
                validation_results['valid_terms'] += 1
            else:
                validation_results['invalid_terms'] += 1
                validation_results['term_issues'][term_id] = term_validation['issues']
                validation_results['issues'].extend(term_validation['issues'])
        
        # 전체 통계 생성
        validation_results['statistics'] = self._generate_validation_statistics(dictionary)
        
        # 개선 권장사항 생성
        validation_results['recommendations'] = self._generate_recommendations(validation_results)
        
        validation_results['is_valid'] = validation_results['invalid_terms'] == 0
        
        logger.info(f"용어 검증 완료 - {validation_results['valid_terms']}개 유효, {validation_results['invalid_terms']}개 무효")
        
        return validation_results
    
    def validate_single_term(self, term_data: Dict[str, Any]) -> Dict[str, Any]:
        """개별 용어 검증"""
        issues = []
        
        # 용어명 검증
        term_name = term_data.get('term_name', '')
        if not term_name:
            issues.append("용어명이 없습니다.")
        else:
            if len(term_name) < self.validation_rules['term_name']['min_length']:
                issues.append(f"용어명이 너무 짧습니다: {len(term_name)}자 (최소 {self.validation_rules['term_name']['min_length']}자)")
            
            if len(term_name) > self.validation_rules['term_name']['max_length']:
                issues.append(f"용어명이 너무 깁니다: {len(term_name)}자 (최대 {self.validation_rules['term_name']['max_length']}자)")
            
            if not re.match(self.validation_rules['term_name']['pattern'], term_name):
                issues.append(f"용어명에 유효하지 않은 문자가 포함되어 있습니다: {term_name}")
        
        # 정의 검증
        definition = term_data.get('definition', '')
        if not definition:
            issues.append("정의가 없습니다.")
        else:
            if len(definition) < self.validation_rules['definition']['min_length']:
                issues.append(f"정의가 너무 짧습니다: {len(definition)}자 (최소 {self.validation_rules['definition']['min_length']}자)")
            
            if len(definition) > self.validation_rules['definition']['max_length']:
                issues.append(f"정의가 너무 깁니다: {len(definition)}자 (최대 {self.validation_rules['definition']['max_length']}자)")
        
        # 카테고리 검증
        category = term_data.get('category', '')
        if not category:
            issues.append("카테고리가 없습니다.")
        elif category not in self.validation_rules['category']['allowed_values']:
            issues.append(f"유효하지 않은 카테고리입니다: {category}")
        
        # 빈도 검증
        frequency = term_data.get('frequency', 0)
        if not isinstance(frequency, (int, float)):
            issues.append("빈도가 숫자가 아닙니다.")
        elif frequency < self.validation_rules['frequency']['min_value']:
            issues.append(f"빈도가 너무 낮습니다: {frequency} (최소 {self.validation_rules['frequency']['min_value']})")
        elif frequency > self.validation_rules['frequency']['max_value']:
            issues.append(f"빈도가 너무 높습니다: {frequency} (최대 {self.validation_rules['frequency']['max_value']})")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'term_name': term_name,
            'category': category
        }
    
    def _generate_validation_statistics(self, dictionary) -> Dict[str, Any]:
        """검증 통계 생성"""
        stats = {
            'category_distribution': defaultdict(int),
            'frequency_distribution': defaultdict(int),
            'definition_length_distribution': defaultdict(int),
            'term_name_length_distribution': defaultdict(int),
            'duplicate_terms': [],
            'empty_definitions': 0,
            'low_frequency_terms': 0
        }
        
        term_names = []
        
        for term_data in dictionary.terms.values():
            # 카테고리 분포
            category = term_data.get('category', '기타')
            stats['category_distribution'][category] += 1
            
            # 빈도 분포
            frequency = term_data.get('frequency', 0)
            freq_range = f"{(frequency // 10) * 10}-{(frequency // 10) * 10 + 9}"
            stats['frequency_distribution'][freq_range] += 1
            
            if frequency < 5:
                stats['low_frequency_terms'] += 1
            
            # 정의 길이 분포
            definition = term_data.get('definition', '')
            if not definition:
                stats['empty_definitions'] += 1
            else:
                def_len_range = f"{(len(definition) // 50) * 50}-{(len(definition) // 50) * 50 + 49}"
                stats['definition_length_distribution'][def_len_range] += 1
            
            # 용어명 길이 분포
            term_name = term_data.get('term_name', '')
            if term_name:
                name_len_range = f"{(len(term_name) // 5) * 5}-{(len(term_name) // 5) * 5 + 4}"
                stats['term_name_length_distribution'][name_len_range] += 1
                term_names.append(term_name)
        
        # 중복 용어 찾기
        term_name_counts = Counter(term_names)
        stats['duplicate_terms'] = [name for name, count in term_name_counts.items() if count > 1]
        
        # 딕셔너리로 변환
        for key in ['category_distribution', 'frequency_distribution', 'definition_length_distribution', 'term_name_length_distribution']:
            stats[key] = dict(stats[key])
        
        return stats
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        # 중복 용어 정리 권장
        if validation_results['statistics']['duplicate_terms']:
            recommendations.append(f"중복된 용어명 {len(validation_results['statistics']['duplicate_terms'])}개를 정리하세요: {validation_results['statistics']['duplicate_terms'][:5]}")
        
        # 빈 정의 정리 권장
        if validation_results['statistics']['empty_definitions'] > 0:
            recommendations.append(f"정의가 없는 용어 {validation_results['statistics']['empty_definitions']}개를 정리하세요.")
        
        # 낮은 빈도 용어 검토 권장
        if validation_results['statistics']['low_frequency_terms'] > 0:
            recommendations.append(f"빈도가 낮은 용어 {validation_results['statistics']['low_frequency_terms']}개를 검토하세요.")
        
        # 카테고리 불균형 권장
        category_dist = validation_results['statistics']['category_distribution']
        if category_dist:
            max_category = max(category_dist, key=category_dist.get)
            min_category = min(category_dist, key=category_dist.get)
            if category_dist[max_category] > category_dist[min_category] * 3:
                recommendations.append(f"카테고리 분포가 불균형합니다. '{min_category}' 카테고리 용어를 추가하세요.")
        
        return recommendations
    
    def validate_synonym_groups(self, dictionary) -> Dict[str, Any]:
        """동의어 그룹 검증"""
        if not dictionary:
            return {'is_valid': False, 'issues': ['사전이 제공되지 않았습니다.']}
        
        issues = []
        group_stats = {
            'total_groups': len(dictionary.synonym_groups),
            'groups_with_issues': 0,
            'empty_groups': 0,
            'invalid_confidence': 0
        }
        
        for group_id, group_data in dictionary.synonym_groups.items():
            group_issues = []
            
            # 표준 용어 검증
            standard_term = group_data.get('standard_term', '')
            if not standard_term:
                group_issues.append("표준 용어가 없습니다.")
            elif not dictionary.get_term(standard_term):
                group_issues.append(f"표준 용어 '{standard_term}'가 사전에 없습니다.")
            
            # 변형어 검증
            variants = group_data.get('variants', [])
            if not variants:
                group_issues.append("변형어가 없습니다.")
                group_stats['empty_groups'] += 1
            else:
                for variant in variants:
                    if not dictionary.get_term(variant):
                        group_issues.append(f"변형어 '{variant}'가 사전에 없습니다.")
            
            # 신뢰도 검증
            confidence = group_data.get('confidence', 0)
            if not 0 <= confidence <= 1:
                group_issues.append(f"신뢰도가 유효하지 않습니다: {confidence}")
                group_stats['invalid_confidence'] += 1
            
            if group_issues:
                issues.extend([f"{group_id}: {issue}" for issue in group_issues])
                group_stats['groups_with_issues'] += 1
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'statistics': group_stats
        }
    
    def suggest_term_improvements(self, term_data: Dict[str, Any]) -> List[str]:
        """개별 용어 개선 제안"""
        suggestions = []
        
        term_name = term_data.get('term_name', '')
        definition = term_data.get('definition', '')
        category = term_data.get('category', '')
        
        # 용어명 개선 제안
        if term_name:
            if len(term_name) < 3:
                suggestions.append("용어명을 더 구체적으로 만드세요.")
            
            if ' ' in term_name:
                suggestions.append("용어명에서 공백을 제거하는 것을 고려하세요.")
            
            if not re.match(r'^[가-힣]+', term_name):
                suggestions.append("용어명을 한글로 시작하도록 하세요.")
        
        # 정의 개선 제안
        if definition:
            if len(definition) < 20:
                suggestions.append("정의를 더 자세히 작성하세요.")
            
            if definition.endswith('.'):
                suggestions.append("정의에서 마침표를 제거하는 것을 고려하세요.")
            
            if '등' in definition and len(definition) < 50:
                suggestions.append("'등'을 사용할 때는 구체적인 예시를 추가하세요.")
        
        # 카테고리 개선 제안
        if category == '기타':
            suggestions.append("더 구체적인 카테고리를 지정하세요.")
        
        return suggestions
    
    def generate_quality_report(self, dictionary) -> Dict[str, Any]:
        """품질 보고서 생성"""
        if not dictionary:
            return {'error': '사전이 제공되지 않았습니다.'}
        
        # 전체 검증
        validation_results = self.validate_all_terms(dictionary)
        synonym_validation = self.validate_synonym_groups(dictionary)
        
        # 품질 점수 계산
        total_terms = validation_results['total_terms']
        valid_terms = validation_results['valid_terms']
        quality_score = (valid_terms / total_terms * 100) if total_terms > 0 else 0
        
        # 보고서 생성
        report = {
            'generated_at': datetime.now().isoformat(),
            'quality_score': round(quality_score, 2),
            'total_terms': total_terms,
            'valid_terms': valid_terms,
            'invalid_terms': validation_results['invalid_terms'],
            'validation_results': validation_results,
            'synonym_validation': synonym_validation,
            'statistics': validation_results['statistics'],
            'recommendations': validation_results['recommendations']
        }
        
        return report
    
    def export_validation_report(self, dictionary, file_path: str) -> bool:
        """검증 보고서 내보내기"""
        try:
            import json
            
            report = self.generate_quality_report(dictionary)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"검증 보고서 내보내기 완료: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"검증 보고서 내보내기 실패: {e}")
            return False


def main():
    """메인 함수 - 검증기 테스트"""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 테스트용 사전 생성
    from source.data.legal_term_dictionary import LegalTermDictionary
    
    dictionary = LegalTermDictionary()
    
    # 테스트 용어 추가
    test_terms = [
        {
            'term_id': 'T001',
            'term_name': '계약',
            'definition': '당사자 간 의사표시의 합치',
            'category': '민사법',
            'frequency': 100
        },
        {
            'term_id': 'T002',
            'term_name': '계약서',
            'definition': '계약의 내용을 문서로 작성한 것',
            'category': '민사법',
            'frequency': 80
        },
        {
            'term_id': 'T003',
            'term_name': '손해배상',
            'definition': '불법행위로 인한 손해의 배상',
            'category': '민사법',
            'frequency': 90
        }
    ]
    
    for term in test_terms:
        dictionary.add_term(term)
    
    # 검증기 초기화
    validator = TermValidator(dictionary)
    
    # 전체 검증
    validation_results = validator.validate_all_terms(dictionary)
    print(f"검증 결과: {validation_results}")
    
    # 품질 보고서 생성
    report = validator.generate_quality_report(dictionary)
    print(f"품질 보고서: {report}")


if __name__ == "__main__":
    main()
