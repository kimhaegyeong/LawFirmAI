# 법률 용어 정규화 전략 문서

## 📋 개요

본 문서는 LawFirmAI 프로젝트에서 국가법령정보센터 OpenAPI의 법령 용어 관련 API를 활용하여 법률 용어 정규화를 수행하는 전략을 제시합니다.

## 🎯 정규화의 목적

- **검색 정확도 향상**: 동일한 의미의 다양한 표현을 통일된 용어로 정규화
- **의미적 일관성 보장**: 법률 문서 간 용어 사용의 일관성 확보
- **RAG 성능 개선**: 벡터 검색 시 관련성 높은 문서 검색
- **사용자 경험 향상**: 다양한 표현으로 검색해도 정확한 결과 제공

## 🔧 API 활용 전략

### 1. 법령 용어 목록 조회 API 활용

#### 1.1 API 정보
- **엔드포인트**: `/lawSearch.do?target=lsTrm`
- **목적**: 법령 용어 목록 조회
- **활용 방안**: 표준 용어 사전 구축

#### 1.2 수집 전략
```python
# 법령 용어 목록 수집 전략
class LegalTermCollectionStrategy:
    def __init__(self):
        self.collection_priority = {
            "high": [
                "민사법", "형사법", "상사법", "노동법", "행정법"
            ],
            "medium": [
                "환경법", "소비자법", "지적재산권법", "금융법"
            ],
            "low": [
                "기타 특수법", "시행령", "시행규칙"
            ]
        }
    
    def collect_terms_by_category(self, category: str) -> List[Dict]:
        """카테고리별 용어 수집"""
        # 1. 해당 카테고리의 법령 목록 조회
        # 2. 각 법령별 용어 목록 수집
        # 3. 용어 중복 제거 및 정리
        pass
```

#### 1.3 수집 데이터 구조
```json
{
  "term_id": "T001",
  "term_name": "계약",
  "category": "민사법",
  "law_references": [
    {
      "law_name": "민법",
      "article": "제105조",
      "definition": "계약의 정의"
    }
  ],
  "variants": ["계약서", "계약관계", "계약체결"],
  "related_terms": ["채권", "채무", "이행"],
  "frequency": 1250
}
```

### 2. 법령 용어 본문 조회 API 활용

#### 2.1 API 정보
- **엔드포인트**: `/lawService.do?target=lsTrm`
- **목적**: 특정 용어의 상세 정의 및 해석 조회
- **활용 방안**: 용어 정의 및 동의어 추출

#### 2.2 용어 정의 수집 전략
```python
# 용어 정의 수집 전략
class TermDefinitionStrategy:
    def __init__(self):
        self.definition_sources = {
            "primary": "법령 조문 내 정의",
            "secondary": "법령해석례",
            "tertiary": "판례 해석"
        }
    
    def collect_term_definitions(self, term_list: List[str]) -> Dict[str, Dict]:
        """용어별 정의 수집"""
        definitions = {}
        
        for term in term_list:
            # 1. 법령 조문에서 정의 추출
            law_definition = self._extract_from_law_articles(term)
            
            # 2. 법령해석례에서 정의 추출
            interpretation_definition = self._extract_from_interpretations(term)
            
            # 3. 판례에서 정의 추출
            precedent_definition = self._extract_from_precedents(term)
            
            definitions[term] = {
                "law_definition": law_definition,
                "interpretation_definition": interpretation_definition,
                "precedent_definition": precedent_definition,
                "consensus_definition": self._create_consensus_definition(
                    law_definition, interpretation_definition, precedent_definition
                )
            }
        
        return definitions
```

## 🏗️ 정규화 시스템 아키텍처

### 1. 다층 정규화 구조

```
Level 1: 기본 정규화 (Basic Normalization)
├── 공백 및 특수문자 정리
├── 대소문자 통일
└── 불필요한 문자 제거

Level 2: 법률 용어 표준화 (Legal Term Standardization)
├── API 수집 용어 사전 기반 매핑
├── 동의어 그룹화
└── 표준 용어로 통일

Level 3: 의미적 정규화 (Semantic Normalization)
├── 의미적 동의어 그룹 매핑
├── 법률 영역별 용어 분류
└── 맥락 기반 용어 해석

Level 4: 구조적 정규화 (Structural Normalization)
├── 법률 구조 요소 정규화
├── 조문 번호 표준화
└── 사건번호 형식 통일
```

### 2. 정규화 파이프라인

```python
# 정규화 파이프라인 구조
class LegalTermNormalizationPipeline:
    def __init__(self):
        self.stages = [
            "data_collection",      # API에서 용어 수집
            "term_standardization", # 표준 용어로 변환
            "semantic_mapping",     # 의미적 매핑
            "quality_validation",   # 품질 검증
            "indexing"             # 인덱싱 및 저장
        ]
    
    def process(self, text: str) -> Dict[str, Any]:
        """정규화 파이프라인 실행"""
        result = {
            "original_text": text,
            "normalized_text": text,
            "normalization_steps": [],
            "term_mappings": {},
            "confidence_scores": {}
        }
        
        for stage in self.stages:
            result = self._execute_stage(stage, result)
        
        return result
```

## 📚 용어 사전 구축 전략

### 1. 계층적 용어 사전 구조

```python
# 용어 사전 구조
class LegalTermDictionary:
    def __init__(self):
        self.term_hierarchy = {
            "level_1": {
                "민사법": {
                    "계약": {
                        "variants": ["계약서", "계약관계", "계약체결"],
                        "definition": "당사자 간 의사표시의 합치",
                        "related_terms": ["채권", "채무", "이행"],
                        "law_references": ["민법 제105조", "민법 제106조"]
                    }
                }
            },
            "level_2": {
                "형사법": {
                    "절도": {
                        "variants": ["절도죄", "절도행위"],
                        "definition": "타인의 재물을 절취하는 행위",
                        "related_terms": ["강도", "사기", "횡령"],
                        "law_references": ["형법 제329조"]
                    }
                }
            }
        }
```

### 2. 동의어 그룹 매핑

```python
# 동의어 그룹 매핑 전략
class SynonymMappingStrategy:
    def __init__(self):
        self.synonym_groups = {
            "contract_terms": {
                "standard": "계약",
                "variants": ["계약서", "계약관계", "계약체결", "계약당사자"],
                "confidence": 0.95
            },
            "damage_terms": {
                "standard": "손해배상",
                "variants": ["손해", "배상", "손해배상책임", "손해보상"],
                "confidence": 0.90
            }
        }
    
    def map_synonyms(self, term: str) -> str:
        """동의어 매핑"""
        for group_name, group_data in self.synonym_groups.items():
            if term in group_data["variants"]:
                return group_data["standard"]
        return term
```

## 🔍 품질 관리 전략

### 1. 정규화 품질 지표

```python
# 품질 지표 정의
class NormalizationQualityMetrics:
    def __init__(self):
        self.metrics = {
            "consistency_score": {
                "description": "용어 일관성 점수",
                "target": 0.90,
                "weight": 0.3
            },
            "completeness_score": {
                "description": "정규화 완성도 점수",
                "target": 0.95,
                "weight": 0.25
            },
            "accuracy_score": {
                "description": "정확도 점수",
                "target": 0.85,
                "weight": 0.25
            },
            "coverage_score": {
                "description": "용어 커버리지 점수",
                "target": 0.80,
                "weight": 0.2
            }
        }
```

### 2. 검증 프로세스

```python
# 검증 프로세스
class ValidationProcess:
    def __init__(self):
        self.validation_stages = [
            "automated_validation",  # 자동 검증
            "expert_review",        # 전문가 검토
            "user_feedback",        # 사용자 피드백
            "continuous_monitoring" # 지속적 모니터링
        ]
    
    def validate_normalization(self, result: Dict) -> Dict[str, Any]:
        """정규화 결과 검증"""
        validation_result = {
            "is_valid": True,
            "issues": [],
            "recommendations": [],
            "quality_score": 0.0
        }
        
        # 각 검증 단계 실행
        for stage in self.validation_stages:
            stage_result = self._execute_validation_stage(stage, result)
            validation_result.update(stage_result)
        
        return validation_result
```

## 🚀 구현 로드맵

### Phase 1: 기반 구축 (1주)
- [ ] 법령 용어 목록 조회 API 연동
- [ ] 법령 용어 본문 조회 API 연동
- [ ] 기본 용어 사전 구축 (1,000개 용어)

### Phase 2: 정규화 시스템 개발 (2주)
- [ ] 다층 정규화 파이프라인 구현
- [ ] 동의어 그룹 매핑 시스템 구축
- [ ] 품질 검증 시스템 개발

### Phase 3: 통합 및 최적화 (1주)
- [ ] 기존 데이터 처리 시스템과 통합
- [ ] 성능 최적화
- [ ] 사용자 피드백 수집 및 개선

## 📊 성공 지표

### 기술적 지표
- **용어 정규화 정확도**: 90% 이상
- **용어 커버리지**: 80% 이상
- **처리 속도**: 1,000개 용어/분 이상
- **메모리 사용량**: 2GB 이하

### 품질 지표
- **일관성 점수**: 0.90 이상
- **완성도 점수**: 0.95 이상
- **사용자 만족도**: 4.0/5.0 이상

## 🔄 지속적 개선 전략

### 1. 데이터 기반 개선
- 사용자 검색 패턴 분석
- 정규화 실패 사례 수집
- 용어 사용 빈도 분석

### 2. 전문가 검토
- 법률 전문가 정기 검토
- 용어 정의 정확성 검증
- 새로운 용어 추가

### 3. 시스템 최적화
- 정규화 알고리즘 개선
- 성능 최적화
- 메모리 사용량 최적화

## 📁 관련 파일

### 구현 파일
- `source/data/legal_term_normalizer.py` - 법률 용어 정규화 핵심 모듈
- `source/data/legal_term_dictionary.py` - 법률 용어 사전 관리
- `source/data/term_collection_api.py` - API 연동 모듈
- `scripts/collect_legal_terms.py` - 용어 수집 스크립트

### 데이터 파일
- `data/legal_terms/` - 수집된 법률 용어 데이터
- `data/term_mappings/` - 용어 매핑 데이터
- `data/normalization_rules/` - 정규화 규칙 데이터

### 테스트 파일
- `tests/test_legal_term_normalizer.py` - 정규화 모듈 테스트
- `tests/test_term_dictionary.py` - 용어 사전 테스트
- `tests/test_term_collection_api.py` - API 연동 테스트

---

*본 문서는 LawFirmAI 프로젝트의 법률 용어 정규화 전략을 정의하며, 프로젝트 진행에 따라 지속적으로 업데이트됩니다.*
