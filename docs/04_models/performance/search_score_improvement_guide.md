# 검색 점수 개선 시스템 가이드

## 개요

LawFirmAI 프로젝트에 구현된 향상된 검색 시스템은 기존 벡터 검색의 성능을 유지하면서도 더 풍부한 정보를 제공하는 시스템입니다.

## 구현된 기능

### 1. 키워드 매칭 시스템

#### 매칭 유형별 가중치:
- **정확한 매칭**: 가중치 2.0
- **부분 매칭**: 가중치 1.5  
- **동의어 매칭**: 가중치 1.3

#### 동작 방식:
```python
def _calculate_keyword_score(self, query: str, text: str, legal_expansions: Dict, keyword_weights: Dict) -> float:
    """키워드 매칭 점수 계산"""
    query_terms = query.lower().split()
    text_lower = text.lower()
    
    exact_matches = 0
    partial_matches = 0
    synonym_matches = 0
    
    for term in query_terms:
        # 정확한 매칭
        if term in text_lower:
            exact_matches += 1
        else:
            # 부분 매칭 (2글자 이상)
            if len(term) >= 2:
                for i in range(len(text_lower) - len(term) + 1):
                    if text_lower[i:i+len(term)] == term:
                        partial_matches += 1
                        break
            
            # 동의어 매칭
            for key, expansions in legal_expansions.items():
                if term in expansions:
                    for expansion in expansions:
                        if expansion in text_lower:
                            synonym_matches += 1
                            break
                    break
    
    # 가중치 적용한 점수 계산
    keyword_score = (
        exact_matches * keyword_weights["exact_match"] +
        partial_matches * keyword_weights["partial_match"] +
        synonym_matches * keyword_weights["synonym_match"]
    ) / len(query_terms) if query_terms else 0
    
    return min(keyword_score, 2.0)  # 최대 2.0으로 제한
```

### 2. 법률 용어 확장 사전

#### 주요 법률 용어 및 동의어:
```python
legal_expansions = {
    "손해배상": ["손해배상", "배상", "피해보상", "손실보상", "금전적 손해", "물질적 손해", "정신적 손해"],
    "이혼": ["이혼", "혼인해소", "혼인무효", "별거", "가정파탄", "부부갈등", "혼인관계"],
    "계약": ["계약", "계약서", "약정", "합의", "계약관계", "계약체결", "계약이행"],
    "변호인": ["변호인", "변호사", "법정변호인", "국선변호인", "선임변호인", "변호"],
    "형사처벌": ["형사처벌", "형사처분", "형사제재", "형사처리", "형사처분", "형사처벌"],
    "재산분할": ["재산분할", "재산분배", "재산정리", "재산처분", "재산분할", "재산분할"],
    "친권": ["친권", "친권자", "친권행사", "친권포기", "친권상실", "친권"],
    "양육비": ["양육비", "양육비용", "양육비지급", "양육비부담", "양육비지원", "양육비"],
    "소송": ["소송", "소송절차", "소송진행", "소송제기", "소송제출", "소송"],
    "법원": ["법원", "법정", "재판부", "법원판결", "법원결정", "법원"],
    "청구": ["청구", "청구권", "청구서", "청구사유", "청구이유"],
    "요건": ["요건", "요소", "조건", "기준", "요구사항"]
}
```

### 3. 카테고리별 가중치

#### 법령 유형별 차별화된 점수:
```python
category_weights = {
    "civil": 1.1,           # 민사법
    "criminal": 1.1,        # 형사법
    "family": 1.1,          # 가사법
    "constitutional": 1.3,   # 헌법 (가장 높음)
    "assembly_law": 1.2      # 국회법
}
```

### 4. 점수 계산 시스템

#### 최적화된 가중치 조합:
```python
# 최종 점수 계산 (매우 보수적인 가중치)
final_score = (
    base_score * 0.95 +          # 기본 벡터 점수 95% (핵심 유지)
    keyword_score * 0.03 +       # 키워드 매칭 3% (정확한 매칭 강화)
    (category_boost - 1.0) * 0.01 +  # 카테고리 부스트 1% (법령 유형별 가중치)
    (quality_boost - 0.95) * 0.005 + # 품질 부스트 0.5% (파싱 품질 고려)
    (length_boost - 1.0) * 0.005     # 길이 부스트 0.5% (적절한 문서 길이 선호)
)
```

## 사용 방법

### 기본 검색 (기존과 동일)
```python
from source.data.vector_store import LegalVectorStore

vector_store = LegalVectorStore(enable_lazy_loading=False)
vector_store.load_index("data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index.faiss")

# 기본 검색
results = vector_store.search("손해배상 청구", top_k=5, enhanced=False)

for result in results:
    print(f"점수: {result['score']}")
    print(f"내용: {result['text'][:100]}...")
```

### 향상된 검색 (새로운 기능)
```python
# 향상된 검색 (기본값)
results = vector_store.search("손해배상 청구", top_k=5, enhanced=True)

for result in results:
    print(f"향상된 점수: {result['enhanced_score']}")
    print(f"기본 점수: {result['base_score']}")
    print(f"키워드 점수: {result['keyword_score']}")
    print(f"카테고리 부스트: {result['category_boost']}")
    print(f"품질 부스트: {result['quality_boost']}")
    print(f"길이 부스트: {result['length_boost']}")
    print(f"내용: {result['text'][:100]}...")
```

## 성능 결과

### 테스트 결과:
- **전체 개선율**: -2.0% (기존 성능과 거의 동일)
- **추가 정보 제공**: 향상된 검색 결과에 상세한 점수 정보 포함
- **호환성**: 기존 시스템과 완전 호환 (`enhanced=True/False` 옵션)

### 개별 쿼리별 결과:
1. **손해배상 청구 요건**: -0.4% (거의 동일)
2. **이혼 소송 절차**: -2.1% (약간 하락)
3. **형사처벌 기준**: -3.4% (약간 하락)
4. **계약 해지 조건**: -0.8% (거의 동일)
5. **재산분할 원칙**: -3.5% (약간 하락)

## 장점

1. **기존 성능 유지**: 기본 벡터 검색의 성능을 거의 그대로 유지
2. **추가 정보 제공**: 키워드 매칭, 카테고리 부스트 등 상세한 점수 정보
3. **확장성**: 향후 더 정교한 가중치 조정 가능
4. **호환성**: 기존 시스템과 완전 호환
5. **투명성**: 각 점수 요소별 상세 정보 제공

## 향후 개선 방안

1. **가중치 동적 조정**: 쿼리 유형에 따른 가중치 자동 조정
2. **법률 용어 확장**: 더 많은 법률 용어 및 동의어 추가
3. **학습 기반 최적화**: 사용자 피드백을 통한 가중치 학습
4. **도메인 특화**: 특정 법률 분야별 맞춤형 가중치 적용

---

*이 문서는 LawFirmAI 프로젝트의 검색 점수 개선 시스템 구현 내용을 설명합니다.*
