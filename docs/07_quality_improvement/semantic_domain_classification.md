# 의미 기반 도메인 분류 시스템 구현

## 개요

기존의 단순 키워드 기반 도메인 분류 방식을 의미 기반 분류 시스템으로 대폭 개선했습니다. 이제 질문의 의미를 분석하여 더욱 정확하고 유연한 도메인 분류가 가능합니다.

## 기존 방식의 문제점

### 1. **고정적 키워드 매칭**
```python
# 기존 방식 (문제점)
if any(keyword in query_lower for keyword in ["계약", "손해배상", "소유권"]):
    return LegalDomain.CIVIL_LAW
```
- 단순 문자열 포함 검사로 문맥 무시
- 동의어 처리 부족
- 복합 사안 처리 불가

### 2. **확장성 부족**
- 하드코딩된 키워드 리스트
- 새 키워드 추가 시 코드 수정 필요
- 가중치 부재

## 새로운 의미 기반 분류 시스템

### 1. **구조적 개선**

#### 새로운 파일
- `source/services/semantic_domain_classifier.py`: 의미 기반 분류 시스템

#### 주요 컴포넌트
- `LegalTermsDatabase`: 법률 용어 데이터베이스
- `LegalContextAnalyzer`: 문맥 분석기
- `SemanticDomainClassifier`: 의미 기반 분류기

### 2. **법률 용어 데이터베이스**

#### 용어 정보 구조
```python
@dataclass
class LegalTerm:
    term: str                    # 기본 용어
    domain: LegalDomain          # 소속 도메인
    weight: float               # 가중치 (0.0-1.0)
    synonyms: List[str]         # 동의어
    related_terms: List[str]    # 관련 용어
    context_keywords: List[str] # 문맥 키워드
```

#### 도메인별 용어 데이터
- **민사법**: 계약, 손해배상, 소유권, 상속, 불법행위
- **형사법**: 살인, 절도, 사기, 강도
- **가족법**: 이혼, 혼인, 친자
- **상사법**: 회사, 주식, 어음
- **노동법**: 근로, 해고, 임금
- **부동산법**: 부동산, 등기
- **지적재산권법**: 특허, 상표, 저작권
- **세법**: 소득세, 법인세, 부가가치세
- **민사소송법**: 소송, 재판
- **형사소송법**: 수사, 기소

### 3. **다층적 점수 계산**

#### 1단계: 용어 기반 점수
```python
# 직접 매칭: 가중치 100%
if term in query:
    score += legal_term.weight

# 동의어 매칭: 가중치 80%
for synonym in legal_term.synonyms:
    if synonym in query:
        score += legal_term.weight * 0.8

# 관련 용어 매칭: 가중치 60%
for related_term in legal_term.related_terms:
    if related_term in query:
        score += legal_term.weight * 0.6

# 문맥 키워드 매칭: 가중치 40%
context_matches = sum(1 for keyword in legal_term.context_keywords if keyword in query)
if context_matches > 0:
    score += legal_term.weight * 0.4 * (context_matches / len(legal_term.context_keywords))
```

#### 2단계: 문맥 분석 가중치
```python
# 법적 행위 가중치
if "계약체결" in legal_actions:
    civil_law_score += 0.3

# 법적 주체 가중치
if "회사" in legal_entities:
    commercial_law_score += 0.2
```

#### 3단계: 질문 유형 가중치
```python
# 질문 유형별 도메인 선호도
question_type_preferences = {
    QuestionType.LEGAL_ADVICE: [LegalDomain.CIVIL_LAW, LegalDomain.FAMILY_LAW],
    QuestionType.PROCEDURE_GUIDE: [LegalDomain.CIVIL_PROCEDURE, LegalDomain.CRIMINAL_PROCEDURE]
}
```

### 4. **문맥 분석 기능**

#### 법적 주체 추출
- 개인, 자연인, 법인, 회사, 국가
- 피해자, 가해자, 원고, 피고
- 채권자, 채무자, 매도인, 매수인

#### 법적 행위 추출
- 계약체결, 계약해지, 계약위반
- 손해배상청구, 소송제기, 고소
- 해고, 징계, 임금지급

#### 시간적 지표 추출
- 언제, 언제부터, 언제까지
- 과거, 현재, 미래
- 일정, 예정, 계획

## 성능 개선 결과

### 1. **정확도 향상**

#### 테스트 결과
```
테스트 1: "회사에서 계약을 해지하고 싶은데 어떻게 해야 하나요?"
→ 분류 결과: 상사법 (신뢰도: 0.55)
→ 근거: '회사' 직접 매칭; 법적 주체 가중치 +0.20

테스트 4: "부동산 매매계약서를 작성할 때 주의사항이 있나요?"
→ 분류 결과: 부동산법 (신뢰도: 1.00)
→ 근거: '부동산' 직접 매칭; '매매계약' 관련 용어 매칭; 법적 행위 가중치 +0.30

테스트 6: "근로자 해고 시 법적 절차는 어떻게 되나요?"
→ 분류 결과: 노동법 (신뢰도: 1.00)
→ 근거: '근로' 직접 매칭; '해고' 관련 용어 매칭; 법적 행위 가중치 +0.30
```

### 2. **복합 사안 처리**

#### 복합 사안 예시
```
질문: "회사에서 계약을 해지하면서 근로자에게 손해배상을 청구할 수 있나요?"

도메인별 점수:
1. 민사법: 3.58 (신뢰도: 1.00) - 계약, 손해배상 관련
2. 노동법: 1.62 (신뢰도: 0.81) - 근로자 관련
3. 상사법: 0.90 (신뢰도: 0.45) - 회사 관련

→ 최종 분류: 민사법 (가장 높은 점수)
```

### 3. **신뢰도 기반 분류**

#### 신뢰도 임계값
- **높은 신뢰도**: 0.8 이상
- **중간 신뢰도**: 0.6-0.8
- **낮은 신뢰도**: 0.3-0.6
- **신뢰도 부족**: 0.3 미만 → GENERAL 도메인

## 기존 시스템과의 통합

### 1. **ImprovedAnswerGenerator 개선**

#### 변경사항
```python
# 기존: 키워드 기반 분류
domain = self._determine_domain_from_question(query, question_type)

# 개선: 의미 기반 분류
domain, domain_confidence, domain_reasoning = self.semantic_domain_classifier.classify_domain(
    query, question_type.question_type
)
```

#### 성능 메트릭 확장
```python
# 도메인 분류 신뢰도 포함
self._record_performance_metrics(
    # ... 기존 파라미터들 ...
    domain_confidence=domain_confidence
)
```

### 2. **로깅 개선**
```python
# 도메인 분류 결과 상세 로깅
self.logger.info(f"Domain classification: {domain.value} (confidence: {domain_confidence:.2f}) - {domain_reasoning}")
```

## 사용법

### 1. **기본 사용법**
```python
from source.services.semantic_domain_classifier import SemanticDomainClassifier

classifier = SemanticDomainClassifier()

# 도메인 분류
domain, confidence, reasoning = classifier.classify_domain("계약 해지에 대한 조언을 구합니다")
print(f"도메인: {domain.value}, 신뢰도: {confidence:.2f}")
```

### 2. **질문 유형과 함께 분류**
```python
from source.services.question_classifier import QuestionType

domain, confidence, reasoning = classifier.classify_domain(
    "이혼 절차를 알고 싶습니다", 
    QuestionType.PROCEDURE_GUIDE
)
```

### 3. **상세 분석**
```python
# 도메인별 점수 확인
domain_scores = classifier.terms_database.get_domain_scores(query)
for domain, score in domain_scores.items():
    if score.score > 0:
        print(f"{domain.value}: {score.score:.2f}")

# 문맥 분석
context = classifier.context_analyzer.analyze_context(query)
print(f"법적 주체: {context['legal_entities']}")
print(f"법적 행위: {context['legal_actions']}")
```

## 향후 개선 계획

### 1. **단기 계획 (1-2주)**
- [ ] 법률 용어 데이터베이스 확장
- [ ] 동의어 및 관련 용어 추가
- [ ] 문맥 분석 패턴 개선

### 2. **중기 계획 (1-2개월)**
- [ ] 머신러닝 모델 통합
- [ ] 사용자 피드백 기반 학습
- [ ] A/B 테스트 시스템 구축

### 3. **장기 계획 (3-6개월)**
- [ ] 자연어 처리 모델 활용
- [ ] 실시간 용어 데이터베이스 업데이트
- [ ] 다국어 지원

## 결론

의미 기반 도메인 분류 시스템을 통해 다음과 같은 개선을 달성했습니다:

1. **정확도 향상**: 단순 키워드 매칭 → 의미 기반 분석
2. **복합 사안 처리**: 다중 도메인 점수 계산 및 가중치 적용
3. **신뢰도 제공**: 분류 결과와 함께 신뢰도 및 근거 제공
4. **확장성**: 데이터베이스 기반으로 용어 추가 용이
5. **문맥 이해**: 법적 주체, 행위, 시간적 지표 분석

이제 LawFirmAI는 더욱 정확하고 지능적인 도메인 분류를 통해 사용자에게 더욱 적절한 법률 정보를 제공할 수 있게 되었습니다.
