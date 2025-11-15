# 질의 테스트 기반 개선점 분석

## 분석 방법
코드 분석을 통해 현재 구현된 필터링 및 문서 선택 로직을 검토하고 개선점을 도출했습니다.

## 분석 일시
2024년 (최신)

---

## 발견된 개선점 (번호순)

### 1. 필터링 통계 수집 및 로깅 개선

**현재 문제점**:
- 필터링된 문서 수만 로깅 (`invalid_docs_count`)
- 필터링 이유별 통계 부족
- 평균 관련도, 키워드 매칭 비율 등 메트릭 부족
- 필터링 전후 문서 수 비교 부족

**개선 방안**:
```python
filtering_stats = {
    "total_docs": len(retrieved_docs),
    "filtered_count": invalid_docs_count,
    "filtered_by": {
        "type_check": 0,
        "content_empty": 0,
        "content_too_short": 0,
        "content_quality_low": 0,
        "relevance_too_low": 0,
        "keyword_match_insufficient": 0
    },
    "valid_docs": {
        "count": len(valid_docs),
        "average_relevance": sum(d.get("relevance_score", 0) for d in valid_docs) / len(valid_docs) if valid_docs else 0,
        "average_keyword_ratio": 0.0,
        "semantic_count": sum(1 for d in valid_docs if d.get("search_type") == "semantic"),
        "keyword_count": sum(1 for d in valid_docs if d.get("search_type") == "keyword")
    }
}
```

**우선순위**: 중간

---

### 2. 상위 70% 선택 로직의 최소 문서 수 보장

**현재 문제점**:
```python
top_percentile = max(1, int(len(sorted_docs) * 0.7))
```
- 문서 수가 적을 때 문제 발생 가능
- 예: 문서가 3개만 있으면 2개만 선택 (70% = 2.1 → 2개)
- 최소 문서 수 보장 로직 부족

**개선 방안**:
```python
# 최소 문서 수 보장
min_docs_required = min(5, len(sorted_docs))  # 최소 5개 또는 전체 문서 수
top_percentile = max(min_docs_required, int(len(sorted_docs) * 0.7))
top_docs = sorted_docs[:top_percentile] if len(sorted_docs) > 1 else sorted_docs
```

**우선순위**: 중간

---

### 3. 균형 선택 로직의 문서 타입 다양성 보장

**현재 문제점**:
- `_select_balanced_documents`에서 검색 타입(semantic/keyword)만 고려
- 문서 타입(법령/판례/결정례) 다양성은 고려하지 않음
- 같은 문서 타입만 선택될 수 있음

**개선 방안**:
```python
def _select_balanced_documents(
    self,
    sorted_docs: List[Dict[str, Any]],
    max_docs: int = 10
) -> List[Dict[str, Any]]:
    # 검색 타입별 분류
    semantic_docs = [d for d in sorted_docs if d.get("search_type") == "semantic"]
    keyword_docs = [d for d in sorted_docs if d.get("search_type") == "keyword"]
    hybrid_docs = [d for d in sorted_docs if d.get("search_type") == "hybrid"]
    
    # 문서 타입별 분류 추가
    statute_docs = [d for d in sorted_docs if d.get("type") == "statute_article"]
    case_docs = [d for d in sorted_docs if d.get("type") in ["case_paragraph", "precedent"]]
    decision_docs = [d for d in sorted_docs if d.get("type") == "decision_paragraph"]
    
    # 검색 타입과 문서 타입 모두 고려하여 균형 선택
    # ...
```

**우선순위**: 중간

---

### 4. 키워드 매칭 계산 결과 캐싱

**현재 문제점**:
- 같은 문서에 대해 반복 계산 가능
- 동의어 사전 조회가 매번 수행됨
- 법률 용어 목록 추출이 매번 수행됨 (이미 캐싱됨)

**개선 방안**:
```python
# 클래스 초기화 시 캐시 추가
self._keyword_match_cache = {}  # {doc_id: {keyword: match_result}}

# 키워드 매칭 계산 시 캐시 확인
doc_id = doc.get("id") or doc.get("source", "")
cache_key = f"{doc_id}_{keyword}"
if cache_key in self._keyword_match_cache:
    return self._keyword_match_cache[cache_key]
```

**우선순위**: 낮음

---

### 5. 필터링 결과 검증 로직 추가

**현재 문제점**:
- 필터링 후 유효한 문서가 없을 때만 에러 반환
- 필터링 결과의 품질을 검증하지 않음
- 평균 관련도나 키워드 매칭 비율 확인 부족

**개선 방안**:
```python
# 필터링 후 품질 검증
if valid_docs:
    avg_relevance = sum(d.get("relevance_score", 0) for d in valid_docs) / len(valid_docs)
    avg_keyword_ratio = sum(d.get("keyword_match_ratio", 0) for d in valid_docs) / len(valid_docs)
    
    # 품질이 너무 낮으면 경고
    if avg_relevance < 0.3:
        self.logger.warning(f"Filtered documents have low average relevance: {avg_relevance:.3f}")
    if avg_keyword_ratio < 0.2:
        self.logger.warning(f"Filtered documents have low average keyword ratio: {avg_keyword_ratio:.3f}")
```

**우선순위**: 낮음

---

### 6. 문서 내용 품질 검증의 법률 용어 확인 선택적 활성화

**현재 문제점**:
- 법률 용어 확인이 기본적으로 비활성화됨
- 관련도가 낮은 문서에 대해 법률 용어 확인을 활성화할 수 있음

**개선 방안**:
```python
# 관련도가 낮은 문서에 대해 법률 용어 확인 활성화
check_legal_terms = False
if relevance_score < min_score + 0.1:  # 관련도가 임계값 근처인 경우
    check_legal_terms = True  # 법률 용어 확인으로 재검증
```

**우선순위**: 낮음

---

### 7. 키워드 추출 실패 시 처리 개선

**현재 문제점**:
```python
return keywords if keywords else [query.strip()]  # 전체 질문 반환
```
- 키워드 추출 실패 시 전체 질문을 키워드로 사용
- 이 경우 키워드 매칭이 의미 없어짐

**개선 방안**:
```python
if not keywords:
    # 키워드 추출 실패 시 키워드 매칭 검증 생략
    # 관련도 점수만으로 필터링
    self.logger.warning("Keyword extraction failed, skipping keyword matching validation")
    return []  # 빈 리스트 반환하여 키워드 매칭 검증 생략
```

**우선순위**: 낮음

---

### 8. 에러 처리 및 예외 상황 대응 강화

**현재 문제점**:
- 키워드 추출 실패 시 전체 질문을 키워드로 사용
- 점수 계산 실패 시 0.0으로 처리
- 예외 발생 시 필터링 로직이 중단될 수 있음

**개선 방안**:
```python
try:
    # 키워드 추출
    query_keywords = self._extract_keywords_from_query(query)
except Exception as e:
    self.logger.error(f"Keyword extraction failed: {e}")
    query_keywords = []  # 빈 리스트로 fallback

try:
    # 점수 계산
    relevance_score = float(doc.get("relevance_score", 0.0))
except (ValueError, TypeError) as e:
    self.logger.warning(f"Relevance score calculation failed: {e}")
    relevance_score = 0.0  # 기본값 사용
```

**우선순위**: 낮음

---

### 9. 문서 선택 로직의 관련도 절대 기준 적용

**현재 문제점**:
- 상위 70%만 선택하지만 관련도 절대 기준이 없음
- 관련도가 매우 낮은 문서도 상위 70%에 포함될 수 있음

**개선 방안**:
```python
# 관련도 절대 기준 적용
min_absolute_relevance = 0.3  # 최소 관련도
top_docs = [
    doc for doc in sorted_docs[:top_percentile]
    if doc.get("final_weighted_score", doc.get("relevance_score", 0.0)) >= min_absolute_relevance
]
```

**우선순위**: 낮음

---

### 10. 필터링 순서 최적화 추가 검토

**현재 문제점**:
- 필터링 순서는 최적화되어 있지만, 일부 검증이 중복될 수 있음
- 품질 검증이 길이 검증 후에 수행되지만, 품질 검증에서도 길이를 확인할 수 있음

**개선 방안**:
- 현재 필터링 순서는 적절함 (빠른 검증부터)
- 품질 검증 내부에서 길이 확인은 중복이지만, 명확성을 위해 유지 가능

**우선순위**: 낮음

---

## 우선순위별 정리

### 중간 우선순위 (단기 개선 권장)
1. 필터링 통계 수집 및 로깅 개선
2. 상위 70% 선택 로직의 최소 문서 수 보장
3. 균형 선택 로직의 문서 타입 다양성 보장

### 낮은 우선순위 (장기 개선 고려)
4. 키워드 매칭 계산 결과 캐싱
5. 필터링 결과 검증 로직 추가
6. 문서 내용 품질 검증의 법률 용어 확인 선택적 활성화
7. 키워드 추출 실패 시 처리 개선
8. 에러 처리 및 예외 상황 대응 강화
9. 문서 선택 로직의 관련도 절대 기준 적용
10. 필터링 순서 최적화 추가 검토

---

## 즉시 적용 가능한 개선사항

### 개선 1: 필터링 통계 수집
```python
# 필터링 통계 수집
filtering_stats = {
    "total": len(retrieved_docs),
    "filtered": invalid_docs_count,
    "filtered_by": {
        "type_check": type_check_count,
        "content_empty": content_empty_count,
        "content_too_short": content_too_short_count,
        "content_quality_low": content_quality_low_count,
        "relevance_too_low": relevance_too_low_count,
        "keyword_match_insufficient": keyword_match_insufficient_count
    },
    "valid": {
        "count": len(valid_docs),
        "avg_relevance": sum(d.get("relevance_score", 0) for d in valid_docs) / len(valid_docs) if valid_docs else 0
    }
}
```

### 개선 2: 최소 문서 수 보장
```python
# 최소 문서 수 보장
min_docs_required = min(5, len(sorted_docs))
top_percentile = max(min_docs_required, int(len(sorted_docs) * 0.7))
```

### 개선 3: 문서 타입 다양성 보장
```python
# 문서 타입별 분류 추가
statute_docs = [d for d in sorted_docs if d.get("type") == "statute_article"]
case_docs = [d for d in sorted_docs if d.get("type") in ["case_paragraph", "precedent"]]
# 각 타입에서 최소 1개씩 선택 보장
```

---

## 결론

현재 구현된 필터링 및 문서 선택 로직은 기본적인 품질 향상을 제공하지만, 
위의 10가지 개선점을 적용하면 더욱 정확하고 효율적인 참조자료 필터링이 가능할 것입니다.

특히 **필터링 통계 수집**, **최소 문서 수 보장**, **문서 타입 다양성 보장**이 가장 중요한 개선 사항으로 판단됩니다.

