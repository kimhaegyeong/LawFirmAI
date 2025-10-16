# 법률 용어 정규화 시스템 사용 가이드

## 📋 개요

법률 용어 정규화 시스템은 국가법령정보센터 OpenAPI를 활용하여 법률 용어를 수집하고, 다층 정규화 파이프라인을 통해 일관된 용어로 변환하는 시스템입니다.

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 환경 변수 설정
export LAW_OPEN_API_OC="your_email@example.com"

# 로그 디렉토리 생성
mkdir -p logs
mkdir -p data/legal_terms
```

### 2. 용어 수집 및 사전 구축

```bash
# 법률 용어 수집 스크립트 실행
python scripts/collect_legal_terms.py
```

### 3. 기본 사용법

```python
from source.data.legal_term_normalizer import LegalTermNormalizer

# 정규화기 초기화
normalizer = LegalTermNormalizer()

# 텍스트 정규화
text = "계약서에 명시된 손해배상 조항을 검토해야 합니다."
result = normalizer.normalize_text(text)

print(f"원본: {result['original_text']}")
print(f"정규화: {result['normalized_text']}")
print(f"용어 매핑: {result['term_mappings']}")
print(f"신뢰도: {result['confidence_scores']}")
```

## 🏗️ 시스템 아키텍처

### 모듈 구조

```
source/data/
├── legal_term_collection_api.py    # API 연동 모듈
├── legal_term_dictionary.py        # 용어 사전 관리
├── legal_term_normalizer.py        # 정규화 파이프라인
└── data_processor.py               # 통합 데이터 처리

scripts/
└── collect_legal_terms.py          # 용어 수집 스크립트
```

### 데이터 흐름

```
API 수집 → 용어 사전 구축 → 정규화 파이프라인 → 통합 데이터 처리
    ↓           ↓              ↓              ↓
용어 수집    사전 관리      다층 정규화     기존 시스템 통합
```

## 🔧 상세 사용법

### 1. 용어 수집 API

```python
from source.data.legal_term_collection_api import LegalTermCollectionAPI, TermCollectionConfig

# API 클라이언트 초기화
config = TermCollectionConfig()
api_client = LegalTermCollectionAPI(config)

# 카테고리별 용어 수집
categories = ["민사법", "형사법", "상사법"]
all_terms = api_client.collect_terms_by_category(categories, max_terms_per_category=200)

# 용어 정의 수집
term_ids = ["T001", "T002", "T003"]
definitions = api_client.collect_term_definitions(term_ids)
```

### 2. 용어 사전 관리

```python
from source.data.legal_term_dictionary import LegalTermDictionary

# 사전 초기화
dictionary = LegalTermDictionary()

# 용어 추가
term_data = {
    'term_id': 'T001',
    'term_name': '계약',
    'definition': '당사자 간 의사표시의 합치',
    'category': '민사법',
    'law_references': ['민법 제105조'],
    'related_terms': ['채권', '채무', '이행'],
    'frequency': 100
}
dictionary.add_term(term_data)

# 동의어 그룹 생성
dictionary.create_synonym_group(
    'contract_group',
    '계약',
    ['계약서', '계약관계', '계약체결'],
    0.95
)

# 용어 검색
results = dictionary.search_terms('계약', category='민사법', limit=10)

# 용어 정규화
normalized_term, confidence = dictionary.normalize_term('계약서')
print(f"정규화 결과: {normalized_term} (신뢰도: {confidence})")
```

### 3. 정규화 파이프라인

```python
from source.data.legal_term_normalizer import LegalTermNormalizer

# 정규화기 초기화
normalizer = LegalTermNormalizer()

# 단일 텍스트 정규화
text = "불법행위로 인한 손해보상 청구권이 인정됩니다."
result = normalizer.normalize_text(text, context="precedent_case")

# 배치 정규화
texts = [
    "계약서에 명시된 손해배상 조항을 검토해야 합니다.",
    "민법 제105조에 따른 계약의 효력에 대해 논의하겠습니다.",
    "채권자와 채무자 간의 계약관계가 성립되었습니다."
]
contexts = ["contract_review", "law_discussion", "legal_analysis"]

results = normalizer.batch_normalize(texts, contexts)

# 정규화 통계
stats = normalizer.get_normalization_statistics()
print(f"정규화 통계: {stats}")
```

### 4. 기존 데이터 처리 시스템 통합

```python
from source.data.data_processor import LegalDataProcessor

# 정규화 기능 활성화된 데이터 프로세서
processor = LegalDataProcessor(enable_term_normalization=True)

# 법령 데이터 처리 (자동으로 용어 정규화 적용)
law_data = {...}  # API에서 수집한 법령 데이터
processed_law = processor.process_law_data(law_data)

# 판례 데이터 처리 (자동으로 용어 정규화 적용)
precedent_data = {...}  # API에서 수집한 판례 데이터
processed_precedent = processor.process_precedent_data(precedent_data)
```

## 📊 정규화 레벨

### Level 1: 기본 정규화
- HTML 태그 제거
- 공백 정규화
- 따옴표 정규화

### Level 2: 법률 용어 표준화
- API 수집 용어 사전 기반 매핑
- 동의어 그룹 매핑
- 신뢰도 기반 용어 선택

### Level 3: 의미적 정규화
- 의미적 동의어 그룹 매핑
- 법률 영역별 용어 분류
- 맥락 기반 용어 해석

### Level 4: 구조적 정규화
- 조문 번호 정규화 (제X조)
- 법률명 정규화 (민법, 상법 등)
- 사건번호 정규화
- 날짜 형식 정규화

## 🎯 품질 관리

### 품질 지표

```python
# 정규화 결과 품질 확인
result = normalizer.normalize_text(text)
validation = result['validation']

print(f"유효성: {validation['is_valid']}")
print(f"품질 점수: {validation['quality_score']:.2f}")
print(f"이슈: {validation['issues']}")
```

### 품질 기준

- **정규화 성공률**: 70% 이상
- **용어 일관성**: 90% 이상
- **처리 속도**: 1,000개 용어/분 이상
- **메모리 사용량**: 2GB 이하

## 🔍 고급 기능

### 1. 커스텀 정규화 규칙

```python
# 정규화 규칙 수정
normalizer.normalization_rules['term_standardization']['confidence_threshold'] = 0.8
normalizer.normalization_rules['legal_structure']['normalize_article_numbers'] = True
```

### 2. 용어 빈도 업데이트

```python
# 텍스트에서 용어 빈도 자동 업데이트
normalizer.update_term_frequency("계약서에 명시된 손해배상 조항...")
```

### 3. 통계 및 모니터링

```python
# 정규화 통계 조회
stats = normalizer.get_normalization_statistics()
print(f"총 처리 건수: {stats['total_processed']}")
print(f"성공률: {stats['success_rate']:.2f}")
print(f"실패률: {stats['failure_rate']:.2f}")

# 통계 저장
normalizer.save_statistics("data/normalization_stats.json")
```

## 🚨 문제 해결

### 일반적인 문제

1. **API 연결 실패**
   ```bash
   # 환경 변수 확인
   echo $LAW_OPEN_API_OC
   
   # API 키 재설정
   export LAW_OPEN_API_OC="your_email@example.com"
   ```

2. **용어 사전 로드 실패**
   ```python
   # 사전 파일 경로 확인
   dictionary = LegalTermDictionary("data/legal_terms/legal_term_dictionary.json")
   
   # 사전 유효성 검사
   validation = dictionary.validate_dictionary()
   print(f"유효성: {validation['is_valid']}")
   ```

3. **정규화 성능 저하**
   ```python
   # 정규화 기능 비활성화
   processor = LegalDataProcessor(enable_term_normalization=False)
   
   # 배치 크기 조정
   results = normalizer.batch_normalize(texts, batch_size=50)
   ```

### 로그 확인

```bash
# 용어 수집 로그
tail -f logs/legal_term_collection.log

# 정규화 로그
tail -f logs/data_processing.log
```

## 📈 성능 최적화

### 1. 메모리 최적화

```python
# 대용량 데이터 처리 시 배치 크기 조정
config = TermCollectionConfig()
config.batch_size = 50  # 기본값: 100
api_client = LegalTermCollectionAPI(config)
```

### 2. 처리 속도 개선

```python
# 정규화 규칙 최적화
normalizer.normalization_rules['term_standardization']['fallback_to_similar'] = False
normalizer.normalization_rules['legal_structure']['normalize_dates'] = False
```

### 3. 캐싱 활용

```python
# 용어 사전 캐싱
dictionary = LegalTermDictionary()
# 사전은 자동으로 메모리에 캐싱됨
```

## 🔄 지속적 개선

### 1. 용어 사전 업데이트

```python
# 새로운 용어 추가
new_terms = api_client.collect_legal_terms("새로운_카테고리", max_terms=100)
for term in new_terms:
    dictionary.add_term(term)

# 사전 저장
dictionary.save_dictionary()
```

### 2. 동의어 그룹 확장

```python
# 새로운 동의어 그룹 추가
dictionary.create_synonym_group(
    'new_group',
    '표준용어',
    ['변형1', '변형2', '변형3'],
    0.9
)
```

### 3. 품질 모니터링

```python
# 정기적인 품질 검사
validation = dictionary.validate_dictionary()
if not validation['is_valid']:
    print(f"사전 품질 이슈: {validation['issues']}")
```

## 📚 참고 자료

- [법률 용어 정규화 전략 문서](legal_term_normalization_strategy.md)
- [텍스트 청킹 전략 문서](text_chunking_strategy.md)
- [국가법령정보센터 OpenAPI 가이드](https://open.law.go.kr/LSO/openApi/guideList.do)

---

*본 가이드는 LawFirmAI 프로젝트의 법률 용어 정규화 시스템 사용법을 설명합니다. 추가 문의사항이 있으시면 개발팀에 연락해주세요.*
