"""
동의어 관리자

법률 용어의 동의어 그룹을 관리하고 자동으로 생성합니다.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class SynonymManager:
    """동의어 관리자 클래스"""
    
    def __init__(self, dictionary=None):
        """동의어 관리자 초기화"""
        self.dictionary = dictionary
        self.synonym_patterns = self._load_synonym_patterns()
        
        logger.info("SynonymManager initialized")
    
    def _load_synonym_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """동의어 패턴 로드"""
        return {
            'contract_terms': [
                {
                    'standard': '계약',
                    'patterns': [r'계약서', r'계약관계', r'계약체결'],
                    'confidence': 0.95
                }
            ],
            'damage_terms': [
                {
                    'standard': '손해배상',
                    'patterns': [r'손해보상', r'손해배상책임'],
                    'confidence': 0.90
                },
                {
                    'standard': '손해',
                    'patterns': [r'피해', r'손실'],
                    'confidence': 0.90
                },
                {
                    'standard': '배상',
                    'patterns': [r'보상', r'배상금'],
                    'confidence': 0.85
                }
            ],
            'legal_terms': [
                {
                    'standard': '법률',
                    'patterns': [r'법령', r'법규'],
                    'confidence': 0.85
                },
                {
                    'standard': '조문',
                    'patterns': [r'법조문', r'조항'],
                    'confidence': 0.90
                }
            ],
            'court_terms': [
                {
                    'standard': '법원',
                    'patterns': [r'재판소', r'법정'],
                    'confidence': 0.80
                }
            ],
            'case_terms': [
                {
                    'standard': '사건',
                    'patterns': [r'사건번호', r'사건명'],
                    'confidence': 0.80
                }
            ],
            'party_terms': [
                {
                    'standard': '당사자',
                    'patterns': [r'계약당사자', r'계약자'],
                    'confidence': 0.90
                }
            ],
            'tort_terms': [
                {
                    'standard': '불법행위',
                    'patterns': [r'불법행위책임'],
                    'confidence': 0.85
                }
            ]
        }
    
    def create_synonym_groups_from_patterns(self, dictionary) -> Dict[str, Any]:
        """패턴 기반 동의어 그룹 생성"""
        if not dictionary:
            logger.error("사전이 제공되지 않았습니다.")
            return {'created': 0, 'skipped': 0, 'errors': 0}
        
        self.dictionary = dictionary
        results = {'created': 0, 'skipped': 0, 'errors': 0}
        
        logger.info("패턴 기반 동의어 그룹 생성 시작")
        
        for category, patterns in self.synonym_patterns.items():
            for pattern_data in patterns:
                try:
                    result = self._create_group_from_pattern(pattern_data)
                    if result['success']:
                        results['created'] += 1
                        logger.info(f"동의어 그룹 생성: {pattern_data['standard']} -> {result['variants']}")
                    else:
                        results['skipped'] += 1
                        logger.warning(f"동의어 그룹 건너뜀: {pattern_data['standard']} - {result['reason']}")
                except Exception as e:
                    results['errors'] += 1
                    logger.error(f"동의어 그룹 생성 실패 ({pattern_data['standard']}): {e}")
        
        logger.info(f"패턴 기반 동의어 그룹 생성 완료 - {results['created']}개 생성, {results['skipped']}개 건너뜀, {results['errors']}개 오류")
        return results
    
    def _create_group_from_pattern(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """개별 패턴에서 동의어 그룹 생성"""
        standard_term = pattern_data['standard']
        patterns = pattern_data['patterns']
        confidence = pattern_data['confidence']
        
        # 표준 용어가 사전에 있는지 확인
        if not self.dictionary.get_term(standard_term):
            return {
                'success': False,
                'reason': f"표준 용어 '{standard_term}'가 사전에 없습니다."
            }
        
        # 패턴에 매칭되는 용어들 찾기
        matching_terms = []
        for pattern in patterns:
            matches = self._find_terms_by_pattern(pattern)
            matching_terms.extend(matches)
        
        # 중복 제거
        matching_terms = list(set(matching_terms))
        
        if not matching_terms:
            return {
                'success': False,
                'reason': f"표준 용어 '{standard_term}'에 대한 매칭 용어가 없습니다."
            }
        
        # 동의어 그룹 생성
        group_id = f"{standard_term}_group_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        success = self.dictionary.create_synonym_group(
            group_id,
            standard_term,
            matching_terms,
            confidence
        )
        
        if success:
            return {
                'success': True,
                'variants': matching_terms,
                'group_id': group_id
            }
        else:
            return {
                'success': False,
                'reason': "사전에 동의어 그룹 추가 실패"
            }
    
    def _find_terms_by_pattern(self, pattern: str) -> List[str]:
        """패턴에 매칭되는 용어들 찾기"""
        matching_terms = []
        
        for term_name in self.dictionary.term_index.keys():
            if re.search(pattern, term_name, re.IGNORECASE):
                matching_terms.append(term_name)
        
        return matching_terms
    
    def create_synonym_groups_from_frequency(self, dictionary, min_frequency: int = 5) -> Dict[str, Any]:
        """빈도 기반 동의어 그룹 생성"""
        if not dictionary:
            logger.error("사전이 제공되지 않았습니다.")
            return {'created': 0, 'skipped': 0, 'errors': 0}
        
        self.dictionary = dictionary
        results = {'created': 0, 'skipped': 0, 'errors': 0}
        
        logger.info(f"빈도 기반 동의어 그룹 생성 시작 (최소 빈도: {min_frequency})")
        
        # 빈도가 높은 용어들 찾기
        frequent_terms = [
            term for term, freq in dictionary.frequency_index.items()
            if freq >= min_frequency
        ]
        
        # 유사한 용어들 그룹화
        term_groups = self._group_similar_terms(frequent_terms)
        
        for group in term_groups:
            if len(group) < 2:  # 최소 2개 이상의 용어가 있어야 그룹 생성
                continue
            
            try:
                # 가장 빈도가 높은 용어를 표준 용어로 선택
                standard_term = max(group, key=lambda x: dictionary.frequency_index.get(x, 0))
                variants = [term for term in group if term != standard_term]
                
                group_id = f"freq_group_{standard_term}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                confidence = 0.7  # 빈도 기반이므로 낮은 신뢰도
                
                success = dictionary.create_synonym_group(
                    group_id,
                    standard_term,
                    variants,
                    confidence
                )
                
                if success:
                    results['created'] += 1
                    logger.info(f"빈도 기반 동의어 그룹 생성: {standard_term} -> {variants}")
                else:
                    results['skipped'] += 1
                    logger.warning(f"빈도 기반 동의어 그룹 건너뜀: {standard_term}")
                    
            except Exception as e:
                results['errors'] += 1
                logger.error(f"빈도 기반 동의어 그룹 생성 실패: {e}")
        
        logger.info(f"빈도 기반 동의어 그룹 생성 완료 - {results['created']}개 생성, {results['skipped']}개 건너뜀, {results['errors']}개 오류")
        return results
    
    def _group_similar_terms(self, terms: List[str]) -> List[List[str]]:
        """유사한 용어들을 그룹화"""
        groups = []
        used_terms = set()
        
        for term in terms:
            if term in used_terms:
                continue
            
            # 현재 용어와 유사한 용어들 찾기
            similar_terms = [term]
            
            for other_term in terms:
                if other_term == term or other_term in used_terms:
                    continue
                
                # 간단한 유사도 계산 (문자열 포함 관계)
                if self._calculate_similarity(term, other_term) > 0.7:
                    similar_terms.append(other_term)
                    used_terms.add(other_term)
            
            if len(similar_terms) > 1:
                groups.append(similar_terms)
            
            used_terms.add(term)
        
        return groups
    
    def _calculate_similarity(self, term1: str, term2: str) -> float:
        """두 용어 간의 유사도 계산"""
        # 간단한 문자열 유사도 계산
        if term1 in term2 or term2 in term1:
            return 0.8
        
        # 공통 문자 비율 계산
        common_chars = set(term1) & set(term2)
        total_chars = set(term1) | set(term2)
        
        if len(total_chars) == 0:
            return 0.0
        
        return len(common_chars) / len(total_chars)
    
    def create_manual_synonym_groups(self, dictionary, synonym_definitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """수동 정의된 동의어 그룹 생성"""
        if not dictionary:
            logger.error("사전이 제공되지 않았습니다.")
            return {'created': 0, 'skipped': 0, 'errors': 0}
        
        self.dictionary = dictionary
        results = {'created': 0, 'skipped': 0, 'errors': 0}
        
        logger.info(f"수동 정의 동의어 그룹 생성 시작 - {len(synonym_definitions)}개 그룹")
        
        for group_data in synonym_definitions:
            try:
                standard_term = group_data['standard_term']
                variants = group_data['variants']
                confidence = group_data.get('confidence', 0.9)
                group_id = group_data.get('group_id', f"manual_{standard_term}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                # 표준 용어가 사전에 있는지 확인
                if not dictionary.get_term(standard_term):
                    results['skipped'] += 1
                    logger.warning(f"표준 용어 '{standard_term}'가 사전에 없습니다.")
                    continue
                
                # 실제로 존재하는 변형어만 필터링
                existing_variants = []
                for variant in variants:
                    if dictionary.get_term(variant):
                        existing_variants.append(variant)
                
                if not existing_variants:
                    results['skipped'] += 1
                    logger.warning(f"표준 용어 '{standard_term}'의 변형어가 사전에 없습니다.")
                    continue
                
                # 동의어 그룹 생성
                success = dictionary.create_synonym_group(
                    group_id,
                    standard_term,
                    existing_variants,
                    confidence
                )
                
                if success:
                    results['created'] += 1
                    logger.info(f"수동 동의어 그룹 생성: {group_id} ({standard_term} -> {existing_variants})")
                else:
                    results['skipped'] += 1
                    logger.warning(f"수동 동의어 그룹 생성 실패: {group_id}")
                    
            except Exception as e:
                results['errors'] += 1
                logger.error(f"수동 동의어 그룹 생성 실패 ({group_data.get('group_id', 'unknown')}): {e}")
        
        logger.info(f"수동 정의 동의어 그룹 생성 완료 - {results['created']}개 생성, {results['skipped']}개 건너뜀, {results['errors']}개 오류")
        return results
    
    def get_synonym_statistics(self, dictionary) -> Dict[str, Any]:
        """동의어 그룹 통계 조회"""
        if not dictionary:
            return {}
        
        total_groups = len(dictionary.synonym_groups)
        total_variants = sum(len(group['variants']) for group in dictionary.synonym_groups.values())
        
        # 신뢰도별 분포
        confidence_distribution = defaultdict(int)
        for group in dictionary.synonym_groups.values():
            confidence = group.get('confidence', 0)
            confidence_range = f"{int(confidence * 10) * 10}%"
            confidence_distribution[confidence_range] += 1
        
        return {
            'total_groups': total_groups,
            'total_variants': total_variants,
            'average_variants_per_group': total_variants / total_groups if total_groups > 0 else 0,
            'confidence_distribution': dict(confidence_distribution)
        }
    
    def validate_synonym_groups(self, dictionary) -> Dict[str, Any]:
        """동의어 그룹 유효성 검사"""
        if not dictionary:
            return {'is_valid': False, 'issues': ['사전이 제공되지 않았습니다.']}
        
        issues = []
        
        for group_id, group_data in dictionary.synonym_groups.items():
            # 표준 용어가 사전에 있는지 확인
            standard_term = group_data.get('standard_term', '')
            if not dictionary.get_term(standard_term):
                issues.append(f"동의어 그룹 {group_id}의 표준 용어 '{standard_term}'가 사전에 없습니다.")
            
            # 변형어들이 사전에 있는지 확인
            variants = group_data.get('variants', [])
            for variant in variants:
                if not dictionary.get_term(variant):
                    issues.append(f"동의어 그룹 {group_id}의 변형어 '{variant}'가 사전에 없습니다.")
            
            # 신뢰도 범위 확인
            confidence = group_data.get('confidence', 0)
            if not 0 <= confidence <= 1:
                issues.append(f"동의어 그룹 {group_id}의 신뢰도가 유효하지 않습니다: {confidence}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'total_issues': len(issues)
        }


def main():
    """메인 함수 - 동의어 관리 테스트"""
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
        {'term_id': 'T001', 'term_name': '계약', 'definition': '계약의 정의', 'category': '민사법', 'frequency': 100},
        {'term_id': 'T002', 'term_name': '계약서', 'definition': '계약서의 정의', 'category': '민사법', 'frequency': 80},
        {'term_id': 'T003', 'term_name': '손해배상', 'definition': '손해배상의 정의', 'category': '민사법', 'frequency': 90},
        {'term_id': 'T004', 'term_name': '손해보상', 'definition': '손해보상의 정의', 'category': '민사법', 'frequency': 70},
    ]
    
    for term in test_terms:
        dictionary.add_term(term)
    
    # 동의어 관리자 초기화
    synonym_manager = SynonymManager(dictionary)
    
    # 패턴 기반 동의어 그룹 생성
    results = synonym_manager.create_synonym_groups_from_patterns(dictionary)
    print(f"패턴 기반 결과: {results}")
    
    # 통계 조회
    stats = synonym_manager.get_synonym_statistics(dictionary)
    print(f"동의어 통계: {stats}")
    
    # 유효성 검사
    validation = synonym_manager.validate_synonym_groups(dictionary)
    print(f"유효성 검사: {validation}")


if __name__ == "__main__":
    main()
