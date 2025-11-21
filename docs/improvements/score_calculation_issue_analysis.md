# 점수 계산 문제 분석 보고서

## 문제 발견

### 증상
- **검색 시점 점수**: `relevance_score = 0.980` (매우 높음)
- **필터링 후 문서 수**: 3개만 남음 (94.4% 손실)
- **필터링 시점 점수**: `final_weighted_score`가 낮아서 필터링됨

### 원인 분석

#### 1. `final_weighted_score` 계산 과정에서 점수 하락

**검색 시점**:
```python
relevance_score = 0.980  # 벡터 검색 결과 (매우 높음)
```

**`final_weighted_score` 계산** (`search_result_processor.py:546-605`):
```python
# 1. 기본 점수
base_relevance = 0.980

# 2. 키워드 매칭 점수 (낮을 수 있음)
keyword_match_normalized = 0.2  # 예시

# 3. 하이브리드 점수 계산
hybrid_score = (
    0.5 * base_relevance +      # 0.5 * 0.980 = 0.49
    0.5 * keyword_match_normalized  # 0.5 * 0.2 = 0.1
) = 0.59  # 검색 점수 0.980 → 0.59로 하락!

# 4. 품질 점수 추가
quality_score = 0.3  # 예시
final_score = (
    hybrid_score * base_weight +  # 0.59 * 0.8 = 0.472
    quality_score * quality_weight  # 0.3 * 0.2 = 0.06
) = 0.532  # 더 낮아짐

# 5. 정규화 후
final_weighted_score = 0.4-0.5 정도로 낮아짐
```

**문제점**:
- 검색 점수가 0.980인데 `final_weighted_score`가 0.4-0.5로 낮아짐
- 키워드 매칭 점수가 낮으면 하이브리드 점수가 크게 하락
- 필터링 시 `final_weighted_score`를 사용하므로 높은 검색 점수가 무시됨

#### 2. 필터링 시점 점수 선택 로직

**현재 로직** (`workflow_document_processor.py:715`):
```python
score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
```

**문제점**:
- `final_weighted_score`가 있으면 그것을 우선 사용
- 하지만 `final_weighted_score`가 검색 점수보다 낮을 수 있음
- 검색 시점의 높은 점수가 무시됨

---

## 해결 방안

### 방안 1: 검색 시점 점수 우선 사용 (권장)

**로직 변경**:
```python
# 현재
score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))

# 개선
relevance_score = doc.get("relevance_score", 0.0)
final_weighted_score = doc.get("final_weighted_score", 0.0)

# 검색 시점 점수를 우선 사용하되, final_weighted_score가 더 높으면 사용
score = max(relevance_score, final_weighted_score) if final_weighted_score > 0 else relevance_score
```

**장점**:
- 검색 시점의 높은 점수를 보존
- `final_weighted_score`가 더 높으면 그것을 사용 (양방향 보호)

### 방안 2: 두 점수의 가중 평균 사용

**로직 변경**:
```python
relevance_score = doc.get("relevance_score", 0.0)
final_weighted_score = doc.get("final_weighted_score", 0.0)

# 검색 점수에 더 높은 가중치
score = 0.7 * relevance_score + 0.3 * final_weighted_score
```

**장점**:
- 두 점수를 모두 고려
- 검색 점수에 더 높은 가중치 부여

### 방안 3: `final_weighted_score` 계산 로직 개선

**문제점**:
- 키워드 매칭 점수가 낮으면 하이브리드 점수가 크게 하락
- 검색 점수가 높으면 키워드 매칭 점수 영향 완화

**개선 방안**:
```python
# 검색 점수가 높으면 키워드 매칭 점수 영향 완화
if base_relevance >= 0.8:
    # 검색 점수가 매우 높으면 키워드 매칭 점수 영향 감소
    hybrid_score = (
        0.8 * base_relevance +  # 검색 점수에 더 높은 가중치
        0.2 * keyword_match_normalized
    )
else:
    # 기존 로직
    hybrid_score = (
        0.5 * base_relevance +
        0.5 * keyword_match_normalized
    )
```

---

## 권장 해결책

**방안 1 + 방안 3 + 고정 가중치 최적화 조합**:
1. ✅ 필터링 시 검색 시점 점수 우선 사용 (완료)
2. ✅ `final_weighted_score` 계산 시 검색 점수가 높으면 키워드 매칭 점수 영향 완화 (완료)
3. ✅ 고정 가중치 최적화: `hybrid_law` 0.3:0.7 → 0.45:0.55 (완료)

**이유**:
- 검색 점수가 높다는 것은 문서가 질의와 매우 관련이 높다는 의미
- 키워드 매칭 점수가 낮아도 검색 점수가 높으면 문서는 관련성이 높음
- 두 점수 중 높은 것을 사용하여 문서 손실 방지
- 고정 가중치 조정으로 검색 점수가 낮을 때도 점수 하락 완화

