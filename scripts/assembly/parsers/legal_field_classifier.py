"""
Legal Field Classifier for Korean Law Data

This module classifies Korean law data based on legal fields
such as Constitutional Law, Civil Law, Criminal Law, Commercial Law, etc.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class LegalFieldClassifier:
    """법률 분야 분류기"""
    
    def __init__(self):
        """Initialize field classifier with Korean legal field patterns"""
        self.field_patterns = {
            'constitutional_law': {
                'keywords': ['헌법', '기본권', '국가기관', '헌법재판소', '위헌', '헌법재판', '국민의 권리', '국가의 의무'],
                'patterns': [r'헌법\s*제\d+조', r'기본권', r'국가기관', r'헌법재판소', r'위헌법률심판'],
                'subfields': ['constitutional_rights', 'state_organization', 'constitutional_review'],
                'weight': 1.0
            },
            'civil_law': {
                'keywords': ['민법', '재산권', '계약', '가족', '상속', '채권', '물권', '손해배상', '불법행위', '채무'],
                'patterns': [r'민법\s*제\d+조', r'계약', r'재산권', r'가족관계', r'상속', r'채권', r'물권'],
                'subfields': ['property_law', 'contract_law', 'family_law', 'inheritance_law', 'tort_law'],
                'weight': 1.0
            },
            'criminal_law': {
                'keywords': ['형법', '범죄', '형벌', '벌금', '징역', '금고', '사형', '자유형', '형사처벌'],
                'patterns': [r'형법\s*제\d+조', r'범죄', r'형벌', r'벌금', r'징역', r'금고'],
                'subfields': ['general_criminal_law', 'special_criminal_law', 'criminal_procedure'],
                'weight': 1.0
            },
            'commercial_law': {
                'keywords': ['상법', '회사', '상거래', '어음', '수표', '보험', '해상', '항공', '상행위'],
                'patterns': [r'상법\s*제\d+조', r'회사법', r'상거래', r'어음법', r'수표법', r'보험법'],
                'subfields': ['company_law', 'commercial_transaction', 'insurance_law', 'maritime_law', 'aviation_law'],
                'weight': 1.0
            },
            'administrative_law': {
                'keywords': ['행정법', '행정처분', '허가', '승인', '신고', '행정절차', '행정소송', '행정심판'],
                'patterns': [r'행정법', r'허가', r'승인', r'신고', r'행정절차', r'행정소송'],
                'subfields': ['administrative_procedure', 'administrative_disposition', 'administrative_litigation'],
                'weight': 1.0
            },
            'labor_law': {
                'keywords': ['노동법', '근로', '임금', '근로시간', '산업안전', '고용', '근로기준', '산업재해'],
                'patterns': [r'노동법', r'근로기준법', r'임금', r'근로시간', r'산업안전보건법'],
                'subfields': ['labor_standards', 'industrial_safety', 'employment_security', 'workers_compensation'],
                'weight': 1.0
            },
            'economic_law': {
                'keywords': ['경제법', '공정거래', '독점규제', '소비자', '금융', '증권', '자본시장', '경쟁법'],
                'patterns': [r'공정거래법', r'독점규제', r'소비자보호', r'금융법', r'증권법'],
                'subfields': ['fair_trade', 'consumer_protection', 'financial_law', 'securities_law', 'competition_law'],
                'weight': 1.0
            },
            'procedural_law': {
                'keywords': ['소송법', '민사소송', '형사소송', '행정소송', '재판', '소송절차', '법원', '판사'],
                'patterns': [r'소송법', r'민사소송법', r'형사소송법', r'행정소송법', r'재판'],
                'subfields': ['civil_procedure', 'criminal_procedure', 'administrative_procedure'],
                'weight': 1.0
            },
            'tax_law': {
                'keywords': ['세법', '소득세', '법인세', '부가가치세', '상속세', '증여세', '국세', '지방세'],
                'patterns': [r'세법', r'소득세법', r'법인세법', r'부가가치세법', r'상속세법'],
                'subfields': ['income_tax', 'corporate_tax', 'value_added_tax', 'inheritance_tax'],
                'weight': 1.0
            },
            'environmental_law': {
                'keywords': ['환경법', '대기환경', '수질환경', '폐기물', '환경영향평가', '환경오염'],
                'patterns': [r'환경법', r'대기환경보전법', r'수질환경보전법', r'폐기물관리법'],
                'subfields': ['air_quality', 'water_quality', 'waste_management', 'environmental_assessment'],
                'weight': 1.0
            },
            'intellectual_property_law': {
                'keywords': ['지적재산권', '특허', '상표', '저작권', '디자인', '영업비밀', '특허법'],
                'patterns': [r'특허법', r'상표법', r'저작권법', r'디자인보호법', r'지적재산권'],
                'subfields': ['patent_law', 'trademark_law', 'copyright_law', 'design_law'],
                'weight': 1.0
            },
            'health_law': {
                'keywords': ['의료법', '보건법', '의료기관', '의료인', '의료행위', '보건복지'],
                'patterns': [r'의료법', r'보건법', r'의료기관', r'의료인', r'보건복지'],
                'subfields': ['medical_law', 'public_health', 'healthcare_institutions', 'medical_practitioners'],
                'weight': 1.0
            }
        }
        
        # 법률명 기반 분야 매핑
        self.law_name_field_mapping = {
            '민법': 'civil_law',
            '형법': 'criminal_law',
            '상법': 'commercial_law',
            '헌법': 'constitutional_law',
            '행정법': 'administrative_law',
            '노동법': 'labor_law',
            '근로기준법': 'labor_law',
            '공정거래법': 'economic_law',
            '소비자보호법': 'economic_law',
            '소송법': 'procedural_law',
            '민사소송법': 'procedural_law',
            '형사소송법': 'procedural_law',
            '행정소송법': 'procedural_law',
            '세법': 'tax_law',
            '소득세법': 'tax_law',
            '법인세법': 'tax_law',
            '환경법': 'environmental_law',
            '특허법': 'intellectual_property_law',
            '상표법': 'intellectual_property_law',
            '저작권법': 'intellectual_property_law',
            '의료법': 'health_law',
            '보건법': 'health_law'
        }
    
    def classify_legal_field(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        법률 분야 분류
        
        Args:
            law_data (Dict[str, Any]): Law data dictionary
            
        Returns:
            Dict[str, Any]: Field classification results
        """
        try:
            law_name = law_data.get('law_name', '')
            law_content = law_data.get('law_content', '')
            
            field_info = {
                'primary_field': 'unknown',
                'secondary_fields': [],
                'subfields': [],
                'field_confidence': 0.0,
                'field_keywords': [],
                'related_laws': [],
                'classification_method': 'unknown',
                'classification_timestamp': datetime.now().isoformat()
            }
            
            # 각 방법별 분류 수행
            name_classification = self._classify_by_name(law_name)
            content_classification = self._classify_by_content(law_content)
            
            # 종합 분류
            final_classification = self._combine_classifications(
                name_classification, content_classification
            )
            
            field_info.update(final_classification)
            
            # 관련 법률 분석
            field_info['related_laws'] = self._extract_related_laws(law_content)
            
            return field_info
            
        except Exception as e:
            logger.error(f"Error classifying legal field: {e}")
            return {
                'primary_field': 'unknown',
                'field_confidence': 0.0,
                'error': str(e)
            }
    
    def _classify_by_name(self, law_name: str) -> Dict[str, Any]:
        """법률명 기반 분류"""
        if not law_name:
            return {'primary_field': 'unknown', 'confidence': 0.0, 'method': 'name'}
        
        # 직접 매핑 확인
        for law_keyword, field in self.law_name_field_mapping.items():
            if law_keyword in law_name:
                field_info = self.field_patterns.get(field, {})
                return {
                    'primary_field': field,
                    'confidence': 0.95,
                    'method': 'name_mapping',
                    'matched_keyword': law_keyword,
                    'subfields': field_info.get('subfields', [])
                }
        
        # 패턴 기반 분류
        for field, patterns in self.field_patterns.items():
            for keyword in patterns['keywords']:
                if keyword in law_name:
                    return {
                        'primary_field': field,
                        'confidence': 0.8,
                        'method': 'name_keyword',
                        'matched_keyword': keyword,
                        'subfields': patterns['subfields']
                    }
        
        return {'primary_field': 'unknown', 'confidence': 0.0, 'method': 'name'}
    
    def _classify_by_content(self, law_content: str) -> Dict[str, Any]:
        """내용 기반 분류"""
        if not law_content:
            return {'primary_field': 'unknown', 'confidence': 0.0, 'method': 'content'}
        
        field_scores = {}
        field_keywords = {}
        
        for field, patterns in self.field_patterns.items():
            score = 0
            matched_keywords = []
            
            # 키워드 매칭
            for keyword in patterns['keywords']:
                count = law_content.count(keyword)
                score += count
                if count > 0:
                    matched_keywords.append(f"{keyword}({count})")
            
            # 패턴 매칭
            for pattern in patterns['patterns']:
                matches = re.findall(pattern, law_content)
                score += len(matches) * 2  # 패턴 매치에 더 높은 가중치
                if matches:
                    matched_keywords.append(f"pattern:{pattern}({len(matches)})")
            
            if score > 0:
                field_scores[field] = score
                field_keywords[field] = matched_keywords
        
        # 가장 높은 점수의 분야 선택
        if field_scores:
            best_field = max(field_scores, key=field_scores.get)
            best_score = field_scores[best_field]
            
            return {
                'primary_field': best_field,
                'confidence': min(0.7, best_score / 10),
                'method': 'content_analysis',
                'field_keywords': field_keywords[best_field],
                'subfields': self.field_patterns[best_field]['subfields']
            }
        
        return {'primary_field': 'unknown', 'confidence': 0.0, 'method': 'content'}
    
    def _combine_classifications(self, name_class: Dict, content_class: Dict) -> Dict[str, Any]:
        """분류 결과 종합"""
        classifications = [name_class, content_class]
        
        # 유효한 분류만 필터링
        valid_classifications = [c for c in classifications if c['primary_field'] != 'unknown']
        
        if not valid_classifications:
            return {
                'primary_field': 'unknown',
                'field_confidence': 0.0,
                'classification_method': 'combined'
            }
        
        # 가중 평균으로 최종 분야 결정
        field_votes = {}
        
        for classification in valid_classifications:
            field = classification['primary_field']
            confidence = classification['confidence']
            
            if field not in field_votes:
                field_votes[field] = {
                    'total_confidence': 0.0,
                    'count': 0,
                    'methods': [],
                    'keywords': []
                }
            
            field_votes[field]['total_confidence'] += confidence
            field_votes[field]['count'] += 1
            field_votes[field]['methods'].append(classification.get('method', 'unknown'))
            
            # 키워드 수집
            if 'field_keywords' in classification:
                field_votes[field]['keywords'].extend(classification['field_keywords'])
        
        # 가장 높은 평균 신뢰도의 분야 선택
        best_field = max(field_votes, 
                        key=lambda x: field_votes[x]['total_confidence'] / field_votes[x]['count'])
        
        best_info = field_votes[best_field]
        field_info = self.field_patterns[best_field]
        
        # 보조 분야들 결정
        secondary_fields = []
        sorted_fields = sorted(field_votes.items(), 
                             key=lambda x: x[1]['total_confidence'] / x[1]['count'], 
                             reverse=True)
        
        for field, info in sorted_fields[1:3]:  # 상위 2개 보조 분야
            if info['total_confidence'] / info['count'] > 0.3:  # 최소 신뢰도 임계값
                secondary_fields.append(field)
        
        return {
            'primary_field': best_field,
            'secondary_fields': secondary_fields,
            'field_confidence': best_info['total_confidence'] / best_info['count'],
            'classification_method': 'combined',
            'classification_methods': best_info['methods'],
            'field_keywords': best_info['keywords'],
            'subfields': field_info['subfields']
        }
    
    def _extract_related_laws(self, law_content: str) -> List[Dict[str, Any]]:
        """관련 법률 추출"""
        related_laws = []
        
        # 관련 법률 패턴
        related_patterns = [
            r'「([^」]+법)」',  # 따옴표로 둘러싸인 법률명
            r'([가-힣]+법)\s*제\d+조',  # 법률명 + 조문
            r'([가-힣]+법)\s*및',  # 법률명 + 및
            r'([가-힣]+법)\s*또는',  # 법률명 + 또는
        ]
        
        for pattern in related_patterns:
            matches = re.findall(pattern, law_content)
            for match in matches:
                law_name = match if isinstance(match, str) else match[0]
                
                # 분야 매핑 확인
                field = self.law_name_field_mapping.get(law_name, 'unknown')
                
                related_laws.append({
                    'law_name': law_name,
                    'field': field,
                    'extraction_method': 'pattern_matching'
                })
        
        # 중복 제거
        unique_related_laws = []
        seen_names = set()
        
        for related_law in related_laws:
            if related_law['law_name'] not in seen_names:
                unique_related_laws.append(related_law)
                seen_names.add(related_law['law_name'])
        
        return unique_related_laws
    
    def get_field_info(self, field: str) -> Dict[str, Any]:
        """분야 정보 반환"""
        if field not in self.field_patterns:
            return {}
        
        patterns = self.field_patterns[field]
        
        return {
            'field': field,
            'keywords': patterns['keywords'],
            'patterns': patterns['patterns'],
            'subfields': patterns['subfields'],
            'weight': patterns['weight']
        }
    
    def get_supported_fields(self) -> List[str]:
        """지원되는 분야 목록 반환"""
        return list(self.field_patterns.keys())
    
    def get_subfields(self, field: str) -> List[str]:
        """특정 분야의 하위 분야 목록 반환"""
        if field not in self.field_patterns:
            return []
        
        return self.field_patterns[field]['subfields']
    
    def validate_field_classification(self, law_data: Dict[str, Any], 
                                    expected_field: str) -> Dict[str, Any]:
        """분야 분류 검증"""
        classification_result = self.classify_legal_field(law_data)
        
        return {
            'is_valid': classification_result['primary_field'] == expected_field,
            'confidence': classification_result['field_confidence'],
            'actual_field': classification_result['primary_field'],
            'expected_field': expected_field,
            'secondary_fields': classification_result.get('secondary_fields', []),
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def get_field_statistics(self, law_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """분야별 통계 생성"""
        field_counts = {}
        field_confidences = {}
        
        for law_data in law_data_list:
            classification = self.classify_legal_field(law_data)
            field = classification['primary_field']
            confidence = classification['field_confidence']
            
            if field not in field_counts:
                field_counts[field] = 0
                field_confidences[field] = []
            
            field_counts[field] += 1
            field_confidences[field].append(confidence)
        
        # 평균 신뢰도 계산
        field_stats = {}
        for field, counts in field_counts.items():
            confidences = field_confidences[field]
            field_stats[field] = {
                'count': counts,
                'percentage': counts / len(law_data_list) * 100,
                'average_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
                'min_confidence': min(confidences) if confidences else 0.0,
                'max_confidence': max(confidences) if confidences else 0.0
            }
        
        return {
            'total_laws': len(law_data_list),
            'field_statistics': field_stats,
            'generated_at': datetime.now().isoformat()
        }
