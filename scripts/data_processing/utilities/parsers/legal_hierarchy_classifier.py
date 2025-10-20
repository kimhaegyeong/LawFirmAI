"""
Legal Hierarchy Classifier for Korean Law Data

This module classifies Korean law data based on the legal hierarchy structure
including Constitution, Laws, Presidential Decrees, Ministry Ordinances, etc.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class LegalHierarchyClassifier:
    """법률 위계 구조 기반 분류기"""
    
    def __init__(self):
        """Initialize hierarchy classifier with Korean legal system patterns"""
        self.hierarchy_patterns = {
            'constitution': {
                'keywords': ['헌법', '헌법재판소', '기본권', '국가기관', '헌법재판'],
                'patterns': [r'헌법\s*제\d+조', r'헌법재판소', r'기본권', r'국가기관'],
                'level': 1,
                'weight': 1.0
            },
            'law': {
                'keywords': ['법률', '국회', '제정', '공포', '법'],
                'patterns': [r'법률\s*제\d+호', r'국회에서\s*제정', r'[가-힣]+법\s*제\d+조'],
                'level': 2,
                'weight': 1.0
            },
            'presidential_decree': {
                'keywords': ['대통령령', '시행령'],
                'patterns': [r'대통령령\s*제\d+호', r'시행령'],
                'level': 3,
                'weight': 1.0
            },
            'prime_minister_decree': {
                'keywords': ['총리령'],
                'patterns': [r'총리령\s*제\d+호'],
                'level': 4,
                'weight': 1.0
            },
            'ministry_ordinance': {
                'keywords': ['부령', '시행규칙'],
                'patterns': [r'[가-힣]+부령\s*제\d+호', r'시행규칙'],
                'level': 5,
                'weight': 1.0
            },
            'local_ordinance': {
                'keywords': ['조례', '규칙', '시·도', '시장', '군수', '구청장'],
                'patterns': [r'[가-힣]+시\s*조례', r'[가-힣]+도\s*조례', r'[가-힣]+군\s*조례'],
                'level': 6,
                'weight': 1.0
            }
        }
        
        # 법률명 패턴 (더 정확한 분류를 위해)
        self.law_name_patterns = {
            'constitution': re.compile(r'.*헌법.*'),
            'law': re.compile(r'.*법$'),
            'presidential_decree': re.compile(r'.*시행령$'),
            'ministry_ordinance': re.compile(r'.*시행규칙$|.*부령$'),
            'local_ordinance': re.compile(r'.*조례$|.*규칙$')
        }
    
    def classify_law_hierarchy(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        법률 위계 분류
        
        Args:
            law_data (Dict[str, Any]): Law data dictionary
            
        Returns:
            Dict[str, Any]: Hierarchy classification results
        """
        try:
            law_name = law_data.get('law_name', '')
            promulgation_number = law_data.get('promulgation_number', '')
            law_content = law_data.get('law_content', '')
            
            hierarchy_info = {
                'hierarchy_level': 0,
                'hierarchy_type': 'unknown',
                'hierarchy_confidence': 0.0,
                'parent_laws': [],
                'subordinate_laws': [],
                'related_hierarchy': [],
                'classification_method': 'unknown',
                'classification_timestamp': datetime.now().isoformat()
            }
            
            # 각 방법별 분류 수행
            name_classification = self._classify_by_name(law_name)
            promulgation_classification = self._classify_by_promulgation(promulgation_number)
            content_classification = self._classify_by_content(law_content)
            
            # 종합 분류
            final_classification = self._combine_classifications(
                name_classification, promulgation_classification, content_classification
            )
            
            hierarchy_info.update(final_classification)
            
            # 관련 법률 분석
            hierarchy_info['parent_laws'] = self._extract_parent_laws(law_content)
            hierarchy_info['subordinate_laws'] = self._extract_subordinate_laws(law_content)
            hierarchy_info['related_hierarchy'] = self._extract_related_hierarchy(law_content)
            
            return hierarchy_info
            
        except Exception as e:
            logger.error(f"Error classifying law hierarchy: {e}")
            return {
                'hierarchy_level': 0,
                'hierarchy_type': 'unknown',
                'hierarchy_confidence': 0.0,
                'error': str(e)
            }
    
    def _classify_by_name(self, law_name: str) -> Dict[str, Any]:
        """법률명 기반 분류"""
        if not law_name:
            return {'hierarchy_type': 'unknown', 'confidence': 0.0, 'method': 'name'}
        
        # 법률명 패턴 매칭
        for hierarchy_type, pattern in self.law_name_patterns.items():
            if pattern.match(law_name):
                hierarchy_info = self.hierarchy_patterns.get(hierarchy_type, {})
                return {
                    'hierarchy_type': hierarchy_type,
                    'hierarchy_level': hierarchy_info.get('level', 0),
                    'confidence': 0.9,
                    'method': 'name_pattern'
                }
        
        # 키워드 기반 분류
        for hierarchy_type, patterns in self.hierarchy_patterns.items():
            for keyword in patterns['keywords']:
                if keyword in law_name:
                    return {
                        'hierarchy_type': hierarchy_type,
                        'hierarchy_level': patterns['level'],
                        'confidence': 0.8,
                        'method': 'name_keyword'
                    }
        
        return {'hierarchy_type': 'unknown', 'confidence': 0.0, 'method': 'name'}
    
    def _classify_by_promulgation(self, promulgation_number: str) -> Dict[str, Any]:
        """공포번호 기반 분류"""
        if not promulgation_number:
            return {'hierarchy_type': 'unknown', 'confidence': 0.0, 'method': 'promulgation'}
        
        for hierarchy_type, patterns in self.hierarchy_patterns.items():
            for pattern in patterns['patterns']:
                if re.search(pattern, promulgation_number):
                    return {
                        'hierarchy_type': hierarchy_type,
                        'hierarchy_level': patterns['level'],
                        'confidence': 0.95,
                        'method': 'promulgation_pattern'
                    }
        
        return {'hierarchy_type': 'unknown', 'confidence': 0.0, 'method': 'promulgation'}
    
    def _classify_by_content(self, law_content: str) -> Dict[str, Any]:
        """내용 기반 분류"""
        if not law_content:
            return {'hierarchy_type': 'unknown', 'confidence': 0.0, 'method': 'content'}
        
        hierarchy_scores = {}
        
        for hierarchy_type, patterns in self.hierarchy_patterns.items():
            score = 0
            matched_patterns = []
            
            # 키워드 매칭
            for keyword in patterns['keywords']:
                count = law_content.count(keyword)
                score += count
                if count > 0:
                    matched_patterns.append(f"keyword:{keyword}")
            
            # 패턴 매칭
            for pattern in patterns['patterns']:
                matches = re.findall(pattern, law_content)
                score += len(matches) * 2  # 패턴 매치에 더 높은 가중치
                if matches:
                    matched_patterns.append(f"pattern:{pattern}")
            
            hierarchy_scores[hierarchy_type] = {
                'score': score,
                'matched_patterns': matched_patterns
            }
        
        # 가장 높은 점수의 위계 선택
        if hierarchy_scores:
            best_type = max(hierarchy_scores, key=lambda x: hierarchy_scores[x]['score'])
            best_score = hierarchy_scores[best_type]['score']
            
            if best_score > 0:
                return {
                    'hierarchy_type': best_type,
                    'hierarchy_level': self.hierarchy_patterns[best_type]['level'],
                    'confidence': min(0.7, best_score / 10),
                    'method': 'content_analysis',
                    'matched_patterns': hierarchy_scores[best_type]['matched_patterns']
                }
        
        return {'hierarchy_type': 'unknown', 'confidence': 0.0, 'method': 'content'}
    
    def _combine_classifications(self, name_class: Dict, promulgation_class: Dict, 
                               content_class: Dict) -> Dict[str, Any]:
        """분류 결과 종합"""
        classifications = [name_class, promulgation_class, content_class]
        
        # 유효한 분류만 필터링
        valid_classifications = [c for c in classifications if c['hierarchy_type'] != 'unknown']
        
        if not valid_classifications:
            return {
                'hierarchy_type': 'unknown',
                'hierarchy_level': 0,
                'hierarchy_confidence': 0.0,
                'classification_method': 'combined'
            }
        
        # 가중 평균으로 최종 분류 결정
        hierarchy_votes = {}
        
        for classification in valid_classifications:
            hierarchy_type = classification['hierarchy_type']
            confidence = classification['confidence']
            
            if hierarchy_type not in hierarchy_votes:
                hierarchy_votes[hierarchy_type] = {
                    'total_confidence': 0.0,
                    'count': 0,
                    'methods': []
                }
            
            hierarchy_votes[hierarchy_type]['total_confidence'] += confidence
            hierarchy_votes[hierarchy_type]['count'] += 1
            hierarchy_votes[hierarchy_type]['methods'].append(classification.get('method', 'unknown'))
        
        # 가장 높은 평균 신뢰도의 위계 선택
        best_hierarchy = max(hierarchy_votes, 
                           key=lambda x: hierarchy_votes[x]['total_confidence'] / hierarchy_votes[x]['count'])
        
        best_info = hierarchy_votes[best_hierarchy]
        hierarchy_info = self.hierarchy_patterns[best_hierarchy]
        
        return {
            'hierarchy_type': best_hierarchy,
            'hierarchy_level': hierarchy_info['level'],
            'hierarchy_confidence': best_info['total_confidence'] / best_info['count'],
            'classification_method': 'combined',
            'classification_methods': best_info['methods']
        }
    
    def _extract_parent_laws(self, law_content: str) -> List[Dict[str, Any]]:
        """상위법 추출"""
        parent_laws = []
        
        # 법률명 추출 패턴
        law_patterns = [
            r'「([^」]+법)」',  # 따옴표로 둘러싸인 법률명
            r'([가-힣]+법)\s*제\d+조',  # 법률명 + 조문
            r'([가-힣]+법)\s*및',  # 법률명 + 및
        ]
        
        for pattern in law_patterns:
            matches = re.findall(pattern, law_content)
            for match in matches:
                law_name = match if isinstance(match, str) else match[0]
                
                # 상위법 여부 판단
                if self._is_parent_law(law_name):
                    parent_laws.append({
                        'law_name': law_name,
                        'reference_type': 'parent_law',
                        'hierarchy_level': 2,  # 법률은 일반적으로 레벨 2
                        'extraction_method': 'pattern_matching'
                    })
        
        # 중복 제거
        unique_parent_laws = []
        seen_names = set()
        
        for parent_law in parent_laws:
            if parent_law['law_name'] not in seen_names:
                unique_parent_laws.append(parent_law)
                seen_names.add(parent_law['law_name'])
        
        return unique_parent_laws
    
    def _extract_subordinate_laws(self, law_content: str) -> List[Dict[str, Any]]:
        """하위법 추출"""
        subordinate_laws = []
        
        # 하위법 패턴
        subordinate_patterns = [
            r'([가-힣]+시행령)',  # 시행령
            r'([가-힣]+시행규칙)',  # 시행규칙
            r'([가-힣]+부령)',  # 부령
        ]
        
        for pattern in subordinate_patterns:
            matches = re.findall(pattern, law_content)
            for match in matches:
                law_name = match if isinstance(match, str) else match[0]
                
                subordinate_laws.append({
                    'law_name': law_name,
                    'reference_type': 'subordinate_law',
                    'hierarchy_level': 3 if '시행령' in law_name else 5,
                    'extraction_method': 'pattern_matching'
                })
        
        return subordinate_laws
    
    def _extract_related_hierarchy(self, law_content: str) -> List[Dict[str, Any]]:
        """관련 위계 법률 추출"""
        related_hierarchy = []
        
        # 관련 법률 패턴
        related_patterns = [
            r'([가-힣]+법)\s*및\s*([가-힣]+법)',  # A법 및 B법
            r'([가-힣]+법)\s*또는\s*([가-힣]+법)',  # A법 또는 B법
            r'([가-힣]+법)\s*,\s*([가-힣]+법)',  # A법, B법
        ]
        
        for pattern in related_patterns:
            matches = re.findall(pattern, law_content)
            for match in matches:
                if isinstance(match, tuple):
                    for law_name in match:
                        if law_name:
                            related_hierarchy.append({
                                'law_name': law_name,
                                'reference_type': 'related_hierarchy',
                                'hierarchy_level': 2,  # 관련 법률은 일반적으로 법률 레벨
                                'extraction_method': 'pattern_matching'
                            })
        
        return related_hierarchy
    
    def _is_parent_law(self, law_name: str) -> bool:
        """상위법 여부 판단"""
        # 시행령, 시행규칙, 부령 등은 하위법
        subordinate_indicators = ['시행령', '시행규칙', '부령', '대통령령', '총리령']
        
        for indicator in subordinate_indicators:
            if indicator in law_name:
                return False
        
        # 법률명이 '법'으로 끝나는 경우 상위법으로 간주
        return law_name.endswith('법')
    
    def get_hierarchy_info(self, hierarchy_type: str) -> Dict[str, Any]:
        """위계 정보 반환"""
        if hierarchy_type not in self.hierarchy_patterns:
            return {}
        
        patterns = self.hierarchy_patterns[hierarchy_type]
        
        return {
            'hierarchy_type': hierarchy_type,
            'hierarchy_level': patterns['level'],
            'keywords': patterns['keywords'],
            'patterns': patterns['patterns'],
            'weight': patterns['weight']
        }
    
    def get_supported_hierarchies(self) -> List[str]:
        """지원되는 위계 목록 반환"""
        return list(self.hierarchy_patterns.keys())
    
    def validate_hierarchy_classification(self, law_data: Dict[str, Any], 
                                        hierarchy_type: str) -> Dict[str, Any]:
        """위계 분류 검증"""
        classification_result = self.classify_law_hierarchy(law_data)
        
        return {
            'is_valid': classification_result['hierarchy_type'] == hierarchy_type,
            'confidence': classification_result['hierarchy_confidence'],
            'actual_type': classification_result['hierarchy_type'],
            'expected_type': hierarchy_type,
            'validation_timestamp': datetime.now().isoformat()
        }
