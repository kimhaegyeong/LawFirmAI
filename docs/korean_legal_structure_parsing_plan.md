# 대한민국 법률 구조 기반 데이터 파싱 계획

## Overview

대한민국의 법률 위계 구조와 체계를 반영하여 Assembly 법률 데이터를 체계적으로 파싱하고 분류하는 시스템을 구축합니다.

## 1. 법률 위계 구조 기반 파싱

### 1.1 법률 위계 분류기

**File:** `scripts/assembly/parsers/legal_hierarchy_classifier.py`

```python
class LegalHierarchyClassifier:
    """법률 위계 구조 기반 분류기"""
    
    def __init__(self):
        self.hierarchy_patterns = {
            'constitution': {
                'keywords': ['헌법', '헌법재판소', '기본권', '국가기관'],
                'patterns': [r'헌법\s*제\d+조', r'헌법재판소'],
                'level': 1
            },
            'law': {
                'keywords': ['법률', '국회', '제정', '공포'],
                'patterns': [r'법률\s*제\d+호', r'국회에서\s*제정'],
                'level': 2
            },
            'presidential_decree': {
                'keywords': ['대통령령', '시행령'],
                'patterns': [r'대통령령\s*제\d+호', r'시행령'],
                'level': 3
            },
            'prime_minister_decree': {
                'keywords': ['총리령'],
                'patterns': [r'총리령\s*제\d+호'],
                'level': 4
            },
            'ministry_ordinance': {
                'keywords': ['부령', '시행규칙'],
                'patterns': [r'[가-힣]+부령\s*제\d+호', r'시행규칙'],
                'level': 5
            },
            'local_ordinance': {
                'keywords': ['조례', '규칙', '시·도', '시장', '군수'],
                'patterns': [r'[가-힣]+시\s*조례', r'[가-힣]+도\s*조례'],
                'level': 6
            }
        }
    
    def classify_law_hierarchy(self, law_data: dict) -> dict:
        """법률 위계 분류"""
        law_name = law_data.get('law_name', '')
        promulgation_number = law_data.get('promulgation_number', '')
        law_content = law_data.get('law_content', '')
        
        hierarchy_info = {
            'hierarchy_level': 0,
            'hierarchy_type': 'unknown',
            'hierarchy_confidence': 0.0,
            'parent_laws': [],
            'subordinate_laws': [],
            'related_hierarchy': []
        }
        
        # 법률명 기반 분류
        name_classification = self._classify_by_name(law_name)
        
        # 공포번호 기반 분류
        promulgation_classification = self._classify_by_promulgation(promulgation_number)
        
        # 내용 기반 분류
        content_classification = self._classify_by_content(law_content)
        
        # 종합 분류
        final_classification = self._combine_classifications(
            name_classification, promulgation_classification, content_classification
        )
        
        hierarchy_info.update(final_classification)
        
        return hierarchy_info
    
    def _classify_by_name(self, law_name: str) -> dict:
        """법률명 기반 분류"""
        for hierarchy_type, patterns in self.hierarchy_patterns.items():
            for keyword in patterns['keywords']:
                if keyword in law_name:
                    return {
                        'hierarchy_type': hierarchy_type,
                        'hierarchy_level': patterns['level'],
                        'confidence': 0.8
                    }
        return {'hierarchy_type': 'unknown', 'confidence': 0.0}
    
    def _classify_by_promulgation(self, promulgation_number: str) -> dict:
        """공포번호 기반 분류"""
        for hierarchy_type, patterns in self.hierarchy_patterns.items():
            for pattern in patterns['patterns']:
                if re.search(pattern, promulgation_number):
                    return {
                        'hierarchy_type': hierarchy_type,
                        'hierarchy_level': patterns['level'],
                        'confidence': 0.9
                    }
        return {'hierarchy_type': 'unknown', 'confidence': 0.0}
    
    def _classify_by_content(self, law_content: str) -> dict:
        """내용 기반 분류"""
        hierarchy_scores = {}
        
        for hierarchy_type, patterns in self.hierarchy_patterns.items():
            score = 0
            for keyword in patterns['keywords']:
                score += law_content.count(keyword)
            
            for pattern in patterns['patterns']:
                matches = len(re.findall(pattern, law_content))
                score += matches * 2  # 패턴 매치에 더 높은 가중치
            
            hierarchy_scores[hierarchy_type] = score
        
        if hierarchy_scores:
            best_type = max(hierarchy_scores, key=hierarchy_scores.get)
            if hierarchy_scores[best_type] > 0:
                return {
                    'hierarchy_type': best_type,
                    'hierarchy_level': self.hierarchy_patterns[best_type]['level'],
                    'confidence': min(0.7, hierarchy_scores[best_type] / 10)
                }
        
        return {'hierarchy_type': 'unknown', 'confidence': 0.0}
```

### 1.2 법률 분야 분류기

**File:** `scripts/assembly/parsers/legal_field_classifier.py`

```python
class LegalFieldClassifier:
    """법률 분야 분류기"""
    
    def __init__(self):
        self.field_patterns = {
            'constitutional_law': {
                'keywords': ['헌법', '기본권', '국가기관', '헌법재판소', '위헌'],
                'patterns': [r'헌법\s*제\d+조', r'기본권', r'국가기관'],
                'subfields': ['constitutional_rights', 'state_organization', 'constitutional_review']
            },
            'civil_law': {
                'keywords': ['민법', '재산권', '계약', '가족', '상속', '채권', '물권'],
                'patterns': [r'민법\s*제\d+조', r'계약', r'재산권'],
                'subfields': ['property_law', 'contract_law', 'family_law', 'inheritance_law']
            },
            'criminal_law': {
                'keywords': ['형법', '범죄', '형벌', '벌금', '징역', '금고'],
                'patterns': [r'형법\s*제\d+조', r'범죄', r'형벌'],
                'subfields': ['general_criminal_law', 'special_criminal_law', 'criminal_procedure']
            },
            'commercial_law': {
                'keywords': ['상법', '회사', '상거래', '어음', '수표', '보험', '해상'],
                'patterns': [r'상법\s*제\d+조', r'회사법', r'상거래'],
                'subfields': ['company_law', 'commercial_transaction', 'insurance_law', 'maritime_law']
            },
            'administrative_law': {
                'keywords': ['행정법', '행정처분', '허가', '승인', '신고', '행정절차'],
                'patterns': [r'행정법', r'허가', r'승인'],
                'subfields': ['administrative_procedure', 'administrative_disposition', 'administrative_litigation']
            },
            'labor_law': {
                'keywords': ['노동법', '근로', '임금', '근로시간', '산업안전', '고용'],
                'patterns': [r'노동법', r'근로기준법', r'임금'],
                'subfields': ['labor_standards', 'industrial_safety', 'employment_security']
            },
            'economic_law': {
                'keywords': ['경제법', '공정거래', '독점규제', '소비자', '금융', '증권'],
                'patterns': [r'공정거래법', r'독점규제', r'소비자보호'],
                'subfields': ['fair_trade', 'consumer_protection', 'financial_law', 'securities_law']
            },
            'procedural_law': {
                'keywords': ['소송법', '민사소송', '형사소송', '행정소송', '재판', '소송절차'],
                'patterns': [r'소송법', r'민사소송법', r'형사소송법'],
                'subfields': ['civil_procedure', 'criminal_procedure', 'administrative_procedure']
            }
        }
    
    def classify_legal_field(self, law_data: dict) -> dict:
        """법률 분야 분류"""
        law_name = law_data.get('law_name', '')
        law_content = law_data.get('law_content', '')
        
        field_info = {
            'primary_field': 'unknown',
            'secondary_fields': [],
            'subfields': [],
            'field_confidence': 0.0,
            'field_keywords': [],
            'related_laws': []
        }
        
        # 법률명과 내용을 결합하여 분석
        combined_text = f"{law_name} {law_content}"
        
        field_scores = {}
        field_keywords = {}
        
        for field, patterns in self.field_patterns.items():
            score = 0
            matched_keywords = []
            
            # 키워드 매칭
            for keyword in patterns['keywords']:
                if keyword in combined_text:
                    score += 1
                    matched_keywords.append(keyword)
            
            # 패턴 매칭
            for pattern in patterns['patterns']:
                matches = re.findall(pattern, combined_text)
                score += len(matches) * 2
                matched_keywords.extend(matches)
            
            field_scores[field] = score
            field_keywords[field] = matched_keywords
        
        # 주요 분야 결정
        if field_scores:
            primary_field = max(field_scores, key=field_scores.get)
            if field_scores[primary_field] > 0:
                field_info['primary_field'] = primary_field
                field_info['field_confidence'] = min(0.9, field_scores[primary_field] / 10)
                field_info['field_keywords'] = field_keywords[primary_field]
                field_info['subfields'] = self.field_patterns[primary_field]['subfields']
                
                # 보조 분야들
                sorted_fields = sorted(field_scores.items(), key=lambda x: x[1], reverse=True)
                field_info['secondary_fields'] = [
                    field for field, score in sorted_fields[1:3] if score > 0
                ]
        
        return field_info
```

## 2. 법률 구조 분석기

### 2.1 법률 구조 파서

**File:** `scripts/assembly/parsers/legal_structure_parser.py`

```python
class LegalStructureParser:
    """법률 구조 분석기"""
    
    def __init__(self):
        self.structure_patterns = {
            'articles': re.compile(r'제(\d+)조\s*\(([^)]+)\)'),
            'paragraphs': re.compile(r'제(\d+)항'),
            'subparagraphs': re.compile(r'제(\d+)호'),
            'items': re.compile(r'제(\d+)목'),
            'numbered_items': re.compile(r'(\d+)\.'),
            'lettered_items': re.compile(r'([가-힣])\.'),
            'enforcement_clause': re.compile(r'\[시행\s+([^\]]+)\]'),
            'amendment_clause': re.compile(r'<개정\s+([^>]+)>'),
            'supplementary_provisions': re.compile(r'부칙\s*<([^>]+)>')
        }
    
    def parse_legal_structure(self, law_content: str) -> dict:
        """법률 구조 분석"""
        structure_info = {
            'total_articles': 0,
            'total_paragraphs': 0,
            'total_subparagraphs': 0,
            'articles': [],
            'enforcement_info': {},
            'amendment_history': [],
            'supplementary_provisions': [],
            'structure_complexity': 0.0
        }
        
        # 조문 분석
        articles = self._parse_articles(law_content)
        structure_info['articles'] = articles
        structure_info['total_articles'] = len(articles)
        
        # 항, 호, 목 분석
        for article in articles:
            paragraphs = self._parse_paragraphs(article['content'])
            structure_info['total_paragraphs'] += len(paragraphs)
            
            for paragraph in paragraphs:
                subparagraphs = self._parse_subparagraphs(paragraph['content'])
                structure_info['total_subparagraphs'] += len(subparagraphs)
        
        # 시행 조항 분석
        structure_info['enforcement_info'] = self._parse_enforcement_clause(law_content)
        
        # 개정 이력 분석
        structure_info['amendment_history'] = self._parse_amendment_history(law_content)
        
        # 부칙 분석
        structure_info['supplementary_provisions'] = self._parse_supplementary_provisions(law_content)
        
        # 구조 복잡도 계산
        structure_info['structure_complexity'] = self._calculate_complexity(structure_info)
        
        return structure_info
    
    def _parse_articles(self, content: str) -> list:
        """조문 파싱"""
        articles = []
        article_matches = self.structure_patterns['articles'].findall(content)
        
        for match in article_matches:
            article_num = match[0]
            article_title = match[1]
            
            # 조문 내용 추출
            article_pattern = f'제{article_num}조\s*\\([^)]+\\)'
            article_match = re.search(article_pattern, content)
            
            if article_match:
                start_pos = article_match.end()
                # 다음 조문까지의 내용 추출
                next_article_pattern = f'제{int(article_num)+1}조'
                next_match = re.search(next_article_pattern, content)
                
                if next_match:
                    end_pos = next_match.start()
                else:
                    end_pos = len(content)
                
                article_content = content[start_pos:end_pos].strip()
                
                articles.append({
                    'article_number': f'제{article_num}조',
                    'article_title': article_title,
                    'article_content': article_content,
                    'paragraphs': self._parse_paragraphs(article_content)
                })
        
        return articles
    
    def _parse_paragraphs(self, content: str) -> list:
        """항 파싱"""
        paragraphs = []
        paragraph_matches = self.structure_patterns['paragraphs'].findall(content)
        
        for match in paragraph_matches:
            para_num = match
            paragraph_pattern = f'제{para_num}항'
            para_match = re.search(paragraph_pattern, content)
            
            if para_match:
                start_pos = para_match.end()
                # 다음 항까지의 내용 추출
                next_para_pattern = f'제{int(para_num)+1}항'
                next_match = re.search(next_para_pattern, content)
                
                if next_match:
                    end_pos = next_match.start()
                else:
                    end_pos = len(content)
                
                paragraph_content = content[start_pos:end_pos].strip()
                
                paragraphs.append({
                    'paragraph_number': f'제{para_num}항',
                    'paragraph_content': paragraph_content,
                    'subparagraphs': self._parse_subparagraphs(paragraph_content)
                })
        
        return paragraphs
    
    def _parse_subparagraphs(self, content: str) -> list:
        """호 파싱"""
        subparagraphs = []
        subpara_matches = self.structure_patterns['subparagraphs'].findall(content)
        
        for match in subpara_matches:
            subpara_num = match
            subpara_pattern = f'제{subpara_num}호'
            subpara_match = re.search(subpara_pattern, content)
            
            if subpara_match:
                start_pos = subpara_match.end()
                # 다음 호까지의 내용 추출
                next_subpara_pattern = f'제{int(subpara_num)+1}호'
                next_match = re.search(next_subpara_pattern, content)
                
                if next_match:
                    end_pos = next_match.start()
                else:
                    end_pos = len(content)
                
                subpara_content = content[start_pos:end_pos].strip()
                
                subparagraphs.append({
                    'subparagraph_number': f'제{subpara_num}호',
                    'subparagraph_content': subpara_content
                })
        
        return subparagraphs
    
    def _parse_enforcement_clause(self, content: str) -> dict:
        """시행 조항 파싱"""
        enforcement_match = self.structure_patterns['enforcement_clause'].search(content)
        
        if enforcement_match:
            enforcement_text = enforcement_match.group(1)
            return {
                'enforcement_date': enforcement_text,
                'enforcement_text': f'[시행 {enforcement_text}]',
                'parsed_date': self._parse_date(enforcement_text)
            }
        
        return {}
    
    def _parse_amendment_history(self, content: str) -> list:
        """개정 이력 파싱"""
        amendments = []
        amendment_matches = self.structure_patterns['amendment_clause'].findall(content)
        
        for match in amendment_matches:
            amendment_text = match
            amendments.append({
                'amendment_text': f'<개정 {amendment_text}>',
                'amendment_info': amendment_text,
                'parsed_date': self._parse_date(amendment_text)
            })
        
        return amendments
    
    def _parse_supplementary_provisions(self, content: str) -> list:
        """부칙 파싱"""
        provisions = []
        provision_matches = self.structure_patterns['supplementary_provisions'].findall(content)
        
        for match in provision_matches:
            provision_text = match
            provisions.append({
                'provision_text': f'부칙 <{provision_text}>',
                'provision_info': provision_text
            })
        
        return provisions
    
    def _calculate_complexity(self, structure_info: dict) -> float:
        """구조 복잡도 계산"""
        total_elements = (
            structure_info['total_articles'] +
            structure_info['total_paragraphs'] +
            structure_info['total_subparagraphs']
        )
        
        # 복잡도 점수 (0-1)
        if total_elements == 0:
            return 0.0
        elif total_elements < 10:
            return 0.2
        elif total_elements < 50:
            return 0.5
        elif total_elements < 100:
            return 0.7
        else:
            return 1.0
```

## 3. 법률 관계 분석기

### 3.1 법률 참조 분석기

**File:** `scripts/assembly/parsers/legal_reference_analyzer.py`

```python
class LegalReferenceAnalyzer:
    """법률 참조 분석기"""
    
    def __init__(self):
        self.reference_patterns = {
            'parent_law': re.compile(r'「([^」]+)」'),
            'same_law': re.compile(r'(같은\s+법|동법|이\s+법)'),
            'related_law': re.compile(r'([가-힣]+법)\s*제\d+조'),
            'subordinate_law': re.compile(r'([가-힣]+령|시행규칙)'),
            'procedural_law': re.compile(r'([가-힣]+소송법|소송법)')
        }
        
        self.law_hierarchy_map = {
            '법률': 2,
            '시행령': 3,
            '시행규칙': 5,
            '부령': 5,
            '대통령령': 3,
            '총리령': 4
        }
    
    def analyze_legal_references(self, law_content: str) -> dict:
        """법률 참조 분석"""
        references = {
            'parent_laws': [],
            'related_laws': [],
            'subordinate_laws': [],
            'procedural_laws': [],
            'reference_network': {},
            'hierarchy_relationships': []
        }
        
        # 각 유형별 참조 분석
        references['parent_laws'] = self._extract_parent_laws(law_content)
        references['related_laws'] = self._extract_related_laws(law_content)
        references['subordinate_laws'] = self._extract_subordinate_laws(law_content)
        references['procedural_laws'] = self._extract_procedural_laws(law_content)
        
        # 참조 네트워크 구축
        references['reference_network'] = self._build_reference_network(references)
        
        # 위계 관계 분석
        references['hierarchy_relationships'] = self._analyze_hierarchy_relationships(references)
        
        return references
    
    def _extract_parent_laws(self, content: str) -> list:
        """상위법 추출"""
        parent_laws = []
        matches = self.reference_patterns['parent_law'].findall(content)
        
        for match in matches:
            if '법' in match and not any(x in match for x in ['시행', '부령', '대통령령']):
                parent_laws.append({
                    'law_name': match,
                    'reference_type': 'parent_law',
                    'hierarchy_level': self.law_hierarchy_map.get('법률', 2)
                })
        
        return parent_laws
    
    def _extract_related_laws(self, content: str) -> list:
        """관련법 추출"""
        related_laws = []
        matches = self.reference_patterns['related_law'].findall(content)
        
        for match in matches:
            related_laws.append({
                'law_name': match,
                'reference_type': 'related_law',
                'hierarchy_level': self.law_hierarchy_map.get('법률', 2)
            })
        
        return related_laws
    
    def _extract_subordinate_laws(self, content: str) -> list:
        """하위법 추출"""
        subordinate_laws = []
        matches = self.reference_patterns['subordinate_law'].findall(content)
        
        for match in matches:
            law_type = '시행령' if '시행령' in match else '시행규칙'
            subordinate_laws.append({
                'law_name': match,
                'reference_type': 'subordinate_law',
                'hierarchy_level': self.law_hierarchy_map.get(law_type, 3)
            })
        
        return subordinate_laws
    
    def _extract_procedural_laws(self, content: str) -> list:
        """절차법 추출"""
        procedural_laws = []
        matches = self.reference_patterns['procedural_law'].findall(content)
        
        for match in matches:
            procedural_laws.append({
                'law_name': match,
                'reference_type': 'procedural_law',
                'hierarchy_level': self.law_hierarchy_map.get('법률', 2)
            })
        
        return procedural_laws
    
    def _build_reference_network(self, references: dict) -> dict:
        """참조 네트워크 구축"""
        network = {
            'nodes': [],
            'edges': [],
            'network_density': 0.0
        }
        
        all_laws = []
        for ref_type, laws in references.items():
            if isinstance(laws, list):
                all_laws.extend(laws)
        
        # 노드 생성
        for law in all_laws:
            network['nodes'].append({
                'id': law['law_name'],
                'type': law['reference_type'],
                'hierarchy_level': law['hierarchy_level']
            })
        
        # 엣지 생성 (위계 관계 기반)
        for i, law1 in enumerate(all_laws):
            for j, law2 in enumerate(all_laws):
                if i != j:
                    if law1['hierarchy_level'] > law2['hierarchy_level']:
                        network['edges'].append({
                            'source': law1['law_name'],
                            'target': law2['law_name'],
                            'relationship': 'subordinate_to'
                        })
        
        # 네트워크 밀도 계산
        total_possible_edges = len(all_laws) * (len(all_laws) - 1)
        network['network_density'] = len(network['edges']) / total_possible_edges if total_possible_edges > 0 else 0.0
        
        return network
    
    def _analyze_hierarchy_relationships(self, references: dict) -> list:
        """위계 관계 분석"""
        relationships = []
        
        # 상위법-하위법 관계
        for parent in references['parent_laws']:
            for subordinate in references['subordinate_laws']:
                relationships.append({
                    'source': parent['law_name'],
                    'target': subordinate['law_name'],
                    'relationship_type': 'parent_subordinate',
                    'hierarchy_difference': parent['hierarchy_level'] - subordinate['hierarchy_level']
                })
        
        # 관련법 간 관계
        for i, law1 in enumerate(references['related_laws']):
            for j, law2 in enumerate(references['related_laws']):
                if i != j:
                    relationships.append({
                        'source': law1['law_name'],
                        'target': law2['law_name'],
                        'relationship_type': 'related',
                        'hierarchy_difference': 0
                    })
        
        return relationships
```

## 4. 통합 법률 분석기

### 4.1 통합 분석기

**File:** `scripts/assembly/parsers/comprehensive_legal_analyzer.py`

```python
class ComprehensiveLegalAnalyzer:
    """통합 법률 분석기"""
    
    def __init__(self):
        self.hierarchy_classifier = LegalHierarchyClassifier()
        self.field_classifier = LegalFieldClassifier()
        self.structure_parser = LegalStructureParser()
        self.reference_analyzer = LegalReferenceAnalyzer()
    
    def analyze_law_comprehensively(self, law_data: dict) -> dict:
        """종합 법률 분석"""
        analysis_result = {
            'law_id': law_data.get('law_id', ''),
            'law_name': law_data.get('law_name', ''),
            'analysis_timestamp': datetime.now().isoformat(),
            
            # 위계 구조 분석
            'hierarchy_analysis': self.hierarchy_classifier.classify_law_hierarchy(law_data),
            
            # 분야 분류
            'field_classification': self.field_classifier.classify_legal_field(law_data),
            
            # 구조 분석
            'structure_analysis': self.structure_parser.parse_legal_structure(
                law_data.get('law_content', '')
            ),
            
            # 참조 분석
            'reference_analysis': self.reference_analyzer.analyze_legal_references(
                law_data.get('law_content', '')
            ),
            
            # 종합 평가
            'comprehensive_score': 0.0,
            'analysis_quality': 'unknown'
        }
        
        # 종합 점수 계산
        analysis_result['comprehensive_score'] = self._calculate_comprehensive_score(analysis_result)
        
        # 분석 품질 평가
        analysis_result['analysis_quality'] = self._evaluate_analysis_quality(analysis_result)
        
        return analysis_result
    
    def _calculate_comprehensive_score(self, analysis_result: dict) -> float:
        """종합 점수 계산"""
        scores = []
        
        # 위계 분류 점수
        hierarchy_score = analysis_result['hierarchy_analysis'].get('hierarchy_confidence', 0.0)
        scores.append(hierarchy_score)
        
        # 분야 분류 점수
        field_score = analysis_result['field_classification'].get('field_confidence', 0.0)
        scores.append(field_score)
        
        # 구조 분석 점수
        structure_score = analysis_result['structure_analysis'].get('structure_complexity', 0.0)
        scores.append(structure_score)
        
        # 참조 분석 점수
        reference_score = analysis_result['reference_analysis'].get('reference_network', {}).get('network_density', 0.0)
        scores.append(reference_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _evaluate_analysis_quality(self, analysis_result: dict) -> str:
        """분석 품질 평가"""
        score = analysis_result['comprehensive_score']
        
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'fair'
        else:
            return 'poor'
```

## 5. 업데이트된 전처리 파이프라인

### 5.1 버전 인식 전처리기 업데이트

**File:** `scripts/assembly/preprocess_laws.py`

```python
class VersionAwareLawPreprocessor:
    """버전 인식 법률 데이터 전처리기 (법률 구조 분석 포함)"""
    
    def __init__(self):
        # 기존 컴포넌트들
        self.version_detector = DataVersionDetector()
        self.version_registry = VersionParserRegistry()
        
        # 법률 구조 분석 컴포넌트들
        self.comprehensive_analyzer = ComprehensiveLegalAnalyzer()
        
        # 기존 파서들
        self.html_parsers = {
            'v1.0': LawHTMLParserV1_0(),
            'v1.1': LawHTMLParserV1_1(),
            'v1.2': LawHTMLParserV1_2()
        }
        
        self.article_parsers = {
            'v1.0': ArticleParserV1_0(),
            'v1.1': ArticleParserV1_1(),
            'v1.2': ArticleParserV1_2()
        }
        
        # 공통 파서들
        self.metadata_extractor = MetadataExtractor()
        self.text_normalizer = TextNormalizer()
        self.searchable_text_generator = SearchableTextGenerator()
    
    def _process_single_law_with_version(self, law_data: Dict[str, Any], 
                                       version: str, version_parser, 
                                       html_parser, article_parser) -> Optional[Dict[str, Any]]:
        """버전별 단일 법률 처리 (법률 구조 분석 포함)"""
        try:
            # 버전별 기본 파싱
            version_parsed = version_parser.parse(law_data)
            
            # HTML 파싱
            html_content = version_parsed.get('content_html', '')
            html_parsed = html_parser.parse_html(html_content) if html_content else {
                'clean_text': '',
                'articles': [],
                'metadata': {}
            }
            
            # Article 파싱
            law_content = version_parsed.get('law_content', '')
            articles = article_parser.parse_articles(law_content)
            
            # 공통 메타데이터 추출
            metadata = self.metadata_extractor.extract(law_data)
            
            # 텍스트 정규화
            clean_text = self.text_normalizer.normalize(law_content)
            
            # 검색 가능한 텍스트 생성
            searchable_data = self.searchable_text_generator.generate(clean_text, articles)
            
            # 종합 법률 분석
            comprehensive_analysis = self.comprehensive_analyzer.analyze_law_comprehensively(law_data)
            
            # 최종 결과 조합
            processed_law = {
                # 버전 정보
                'parsing_version': version,
                'version_metadata': {
                    'detected_at': datetime.now().isoformat(),
                    'version_confidence': self.version_detector.get_confidence(law_data, version)
                },
                
                # 기본 식별 정보
                'law_id': version_parsed.get('law_id', ''),
                'source': 'assembly',
                
                # 버전별 파싱된 데이터
                **version_parsed,
                
                # 추출된 메타데이터
                **metadata,
                
                # 파싱된 콘텐츠
                'articles': articles,
                'full_text': clean_text,
                'searchable_text': searchable_data.get('full_text', ''),
                'keywords': searchable_data.get('keywords', []),
                'summary': searchable_data.get('summary', ''),
                
                # HTML 파싱 데이터
                'html_clean_text': html_parsed.get('clean_text', ''),
                'html_articles': html_parsed.get('articles', []),
                'html_metadata': html_parsed.get('metadata', {}),
                
                # 법률 구조 분석 결과
                'legal_analysis': comprehensive_analysis,
                
                # 원본 데이터 (참조용)
                'raw_content': law_content,
                'content_html': html_content,
                
                # 처리 메타데이터
                'processed_at': datetime.now().isoformat(),
                'processing_version': '2.1',  # 법률 구조 분석 포함 버전
                'data_quality': self._calculate_enhanced_data_quality(
                    articles, metadata, searchable_data, comprehensive_analysis, version
                )
            }
            
            return processed_law
            
        except Exception as e:
            logger.error(f"Error processing single law with version {version}: {e}")
            return None
    
    def _calculate_enhanced_data_quality(self, articles: List[Dict], metadata: Dict, 
                                       searchable_data: Dict, comprehensive_analysis: Dict, 
                                       version: str) -> Dict[str, Any]:
        """향상된 데이터 품질 계산 (법률 구조 분석 포함)"""
        return {
            'article_count': len(articles),
            'has_html': bool(searchable_data.get('full_text')),
            'has_articles': len(articles) > 0,
            'has_metadata': bool(metadata),
            'has_keywords': len(searchable_data.get('keywords', [])) > 0,
            
            # 법률 구조 분석 품질
            'hierarchy_classification_quality': comprehensive_analysis['hierarchy_analysis'].get('hierarchy_confidence', 0.0),
            'field_classification_quality': comprehensive_analysis['field_classification'].get('field_confidence', 0.0),
            'structure_analysis_quality': comprehensive_analysis['structure_analysis'].get('structure_complexity', 0.0),
            'reference_analysis_quality': comprehensive_analysis['reference_analysis'].get('reference_network', {}).get('network_density', 0.0),
            
            # 종합 품질 점수
            'comprehensive_analysis_score': comprehensive_analysis.get('comprehensive_score', 0.0),
            'analysis_quality': comprehensive_analysis.get('analysis_quality', 'unknown'),
            
            # 버전별 품질
            'version_compatibility': self.version_detector.validate_data_compatibility({'test': 'data'}, version).get('compatible', False),
            
            # 전체 완성도 점수
            'completeness_score': self._calculate_completeness_score(articles, metadata, searchable_data, comprehensive_analysis)
        }
    
    def _calculate_completeness_score(self, articles: List[Dict], metadata: Dict, 
                                    searchable_data: Dict, comprehensive_analysis: Dict) -> float:
        """완성도 점수 계산"""
        factors = [
            1.0 if len(articles) > 0 else 0.0,  # Has articles
            1.0 if bool(metadata) else 0.0,     # Has metadata
            1.0 if bool(searchable_data.get('full_text')) else 0.0,  # Has searchable text
            1.0 if len(searchable_data.get('keywords', [])) > 0 else 0.0,  # Has keywords
            1.0 if comprehensive_analysis.get('comprehensive_score', 0.0) > 0.5 else 0.0,  # Good analysis
        ]
        
        return sum(factors) / len(factors)
```

## 6. 데이터베이스 스키마 업데이트

### 6.1 법률 구조 분석 테이블

**File:** `scripts/assembly/import_laws_to_db.py`

```python
def _create_enhanced_tables(self):
    """향상된 데이터베이스 테이블 생성 (법률 구조 분석 포함)"""
    try:
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # assembly_laws 테이블 (법률 구조 분석 필드 추가)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS assembly_laws (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    law_id TEXT UNIQUE NOT NULL,
                    source TEXT NOT NULL DEFAULT 'assembly',
                    
                    -- 버전 정보
                    parsing_version TEXT NOT NULL,
                    version_confidence REAL,
                    migration_history TEXT,  -- JSON array
                    
                    -- 기본 정보
                    law_name TEXT NOT NULL,
                    law_type TEXT,
                    category TEXT,
                    row_number TEXT,
                    
                    -- 공포 정보
                    promulgation_number TEXT,
                    promulgation_date TEXT,
                    enforcement_date TEXT,
                    amendment_type TEXT,
                    
                    -- 법률 위계 정보
                    hierarchy_level INTEGER,
                    hierarchy_type TEXT,
                    hierarchy_confidence REAL,
                    
                    -- 법률 분야 정보
                    primary_field TEXT,
                    secondary_fields TEXT,  -- JSON array
                    subfields TEXT,  -- JSON array
                    field_confidence REAL,
                    
                    -- 구조 분석 정보
                    total_articles INTEGER,
                    total_paragraphs INTEGER,
                    total_subparagraphs INTEGER,
                    structure_complexity REAL,
                    
                    -- 참조 분석 정보
                    parent_laws TEXT,  -- JSON array
                    related_laws TEXT,  -- JSON array
                    subordinate_laws TEXT,  -- JSON array
                    reference_network_density REAL,
                    
                    -- 종합 분석 점수
                    comprehensive_analysis_score REAL,
                    analysis_quality TEXT,
                    
                    -- 추출된 메타데이터
                    ministry TEXT,
                    parent_law TEXT,
                    related_laws_legacy TEXT,  -- JSON array (기존 필드와 구분)
                    
                    -- 콘텐츠
                    full_text TEXT NOT NULL,
                    searchable_text TEXT NOT NULL,
                    keywords TEXT,  -- JSON array
                    summary TEXT,
                    
                    -- HTML 콘텐츠
                    html_clean_text TEXT,
                    content_html TEXT,
                    
                    -- 원본 데이터
                    raw_content TEXT,
                    detail_url TEXT,
                    cont_id TEXT,
                    cont_sid TEXT,
                    collected_at TEXT,
                    
                    -- 처리 메타데이터
                    processed_at TEXT NOT NULL,
                    processing_version TEXT,
                    data_quality TEXT,  -- JSON object
                    
                    -- 타임스탬프
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 법률 위계별 통계 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS assembly_hierarchy_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hierarchy_type TEXT NOT NULL,
                    hierarchy_level INTEGER NOT NULL,
                    total_laws INTEGER DEFAULT 0,
                    average_complexity REAL DEFAULT 0.0,
                    average_analysis_score REAL DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(hierarchy_type, hierarchy_level)
                )
            ''')
            
            # 법률 분야별 통계 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS assembly_field_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    field_name TEXT NOT NULL,
                    total_laws INTEGER DEFAULT 0,
                    average_complexity REAL DEFAULT 0.0,
                    average_analysis_score REAL DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(field_name)
                )
            ''')
            
            # 법률 참조 관계 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS assembly_law_references (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_law_id TEXT NOT NULL,
                    target_law_name TEXT NOT NULL,
                    reference_type TEXT NOT NULL,
                    hierarchy_level INTEGER,
                    relationship_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_law_id) REFERENCES assembly_laws (law_id)
                )
            ''')
            
            # 향상된 인덱스
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_assembly_laws_hierarchy ON assembly_laws (hierarchy_type, hierarchy_level)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_assembly_laws_field ON assembly_laws (primary_field)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_assembly_laws_analysis_score ON assembly_laws (comprehensive_analysis_score)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_assembly_laws_analysis_quality ON assembly_laws (analysis_quality)')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_assembly_references_source ON assembly_law_references (source_law_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_assembly_references_target ON assembly_law_references (target_law_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_assembly_references_type ON assembly_law_references (reference_type)')
            
            conn.commit()
            logger.info("Enhanced database tables created successfully")
            
    except Exception as e:
        logger.error(f"Error creating enhanced database tables: {e}")
        raise
```

## 7. 사용 예시

### 7.1 법률 구조 분석 실행

```bash
# 법률 구조 분석 포함 전처리
python scripts/assembly/preprocess_laws.py --input data/raw/assembly/law/20251010 --output data/processed/assembly/law --enable-legal-analysis

# 법률 위계별 통계 생성
python scripts/assembly/legal_analytics.py --generate-hierarchy-stats

# 법률 분야별 통계 생성
python scripts/assembly/legal_analytics.py --generate-field-stats

# 법률 참조 네트워크 분석
python scripts/assembly/legal_analytics.py --analyze-reference-network
```

### 7.2 분석 결과 활용

```python
# 법률 위계별 검색
SELECT * FROM assembly_laws 
WHERE hierarchy_type = 'law' AND hierarchy_level = 2
ORDER BY comprehensive_analysis_score DESC;

# 특정 분야 법률 검색
SELECT * FROM assembly_laws 
WHERE primary_field = 'civil_law' 
AND field_confidence > 0.8
ORDER BY structure_complexity DESC;

# 법률 참조 관계 분석
SELECT 
    source_law_id,
    target_law_name,
    reference_type,
    COUNT(*) as reference_count
FROM assembly_law_references 
GROUP BY source_law_id, reference_type
ORDER BY reference_count DESC;
```

## 8. 기대 효과

1. **체계적인 법률 분류**: 대한민국 법률 위계 구조에 따른 정확한 분류
2. **법률 관계 분석**: 상위법-하위법, 관련법 간의 관계 파악
3. **구조적 검색**: 법률 위계, 분야별 검색 기능
4. **품질 향상**: 법률 구조 분석을 통한 데이터 품질 개선
5. **확장성**: 새로운 법률 분야나 위계 추가 시 쉽게 확장 가능

이 계획을 통해 대한민국 법률 구조에 맞는 체계적이고 정확한 데이터 파싱 시스템을 구축할 수 있습니다.
