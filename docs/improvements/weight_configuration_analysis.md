# 가중치 설정 분석 및 최적화 보고서

## 수정 이력
- **2024년**: 고정 가중치 최적화
  - `hybrid_law`: 0.3:0.7 → 0.45:0.55 (검색 점수 보호)
  - `hybrid_case`: 0.7:0.3 → 0.65:0.35 (약간 조정)
  - `hybrid_general`: 0.5:0.5 유지 (적절함)

## 현재 가중치 설정

### `search_result_processor.py` (final_weighted_score 계산용)
```python
"hybrid_law": {"semantic": 0.3, "keyword": 0.7},      # 법령 조회: 키워드 강조
"hybrid_case": {"semantic": 0.7, "keyword": 0.3},   # 판례 검색: 의미적 검색 강조
"hybrid_general": {"semantic": 0.5, "keyword": 0.5}  # 일반 질문: 균형
```

### 다른 파일의 가중치 설정 비교

#### `hybrid_search_engine_v2.py` (하이브리드 검색용)
```python
"law_inquiry": {"exact": 0.6, "semantic": 0.4},      # 법령 조회: 키워드 강조
"precedent_search": {"exact": 0.4, "semantic": 0.6}, # 판례 검색: 의미적 검색 강조
"complex_question": {"exact": 0.5, "semantic": 0.5} # 복잡한 질문: 균형
```

#### `adaptive_hybrid_weights.py` (적응형 가중치)
```python
기본값: {"semantic": 0.6, "keyword": 0.4}  # 의미적 검색 강조
```

---

## 문제 분석

### 문제 1: 법령 조회 가중치 불일치

**현재 설정**:
- `search_result_processor.py`: `hybrid_law` = {"semantic": 0.3, "keyword": 0.7}
- `hybrid_search_engine_v2.py`: `law_inquiry` = {"exact": 0.6, "semantic": 0.4}

**차이점**:
- `search_result_processor.py`는 키워드 가중치가 0.7로 매우 높음
- `hybrid_search_engine_v2.py`는 키워드 가중치가 0.6으로 상대적으로 낮음

**문제점**:
- 검색 점수가 높아도 (0.980) 키워드 매칭이 낮으면 (0.2) 하이브리드 점수가 크게 하락
- 예: `0.3 * 0.980 + 0.7 * 0.2 = 0.294 + 0.14 = 0.434` (0.980 → 0.434로 하락!)

### 문제 2: 검색 점수 기반 동적 가중치와의 불일치

**현재 로직**:
- 검색 점수 >= 0.8: 동적 가중치 사용 (semantic: 0.7-0.85)
- 검색 점수 < 0.6: 고정 가중치 사용 (semantic: 0.3-0.7)

**문제점**:
- 검색 점수가 0.6-0.8 사이일 때는 중간 가중치 사용
- 하지만 검색 점수가 0.6 미만일 때는 키워드 가중치가 너무 높아서 점수 하락

---

## 법률 도메인 특성 분석

### 법령 조회 (law_inquiry)
**특성**:
- 정확한 조문 번호 매칭이 중요 (키워드 검색)
- 하지만 의미적 검색도 중요 (유사 조문, 관련 조문)
- **권장 가중치**: semantic: 0.4-0.5, keyword: 0.5-0.6

### 판례 검색 (precedent_search)
**특성**:
- 의미적 검색이 매우 중요 (유사 사례 찾기)
- 키워드 검색은 보조적 역할
- **권장 가중치**: semantic: 0.6-0.7, keyword: 0.3-0.4

### 일반 질문 (general)
**특성**:
- 균형이 중요
- **권장 가중치**: semantic: 0.5, keyword: 0.5

---

## 권장 가중치 설정

### 개선된 고정 가중치

```python
"hybrid_law": {"semantic": 0.45, "keyword": 0.55},    # 법령 조회: 약간 키워드 강조
"hybrid_case": {"semantic": 0.65, "keyword": 0.35},  # 판례 검색: 의미적 검색 강조
"hybrid_general": {"semantic": 0.5, "keyword": 0.5}   # 일반 질문: 균형
```

**변경 사항**:
- `hybrid_law`: semantic 0.3 → 0.45, keyword 0.7 → 0.55
  - 검색 점수 보호를 위해 의미적 검색 가중치 증가
  - 키워드 매칭이 낮아도 검색 점수가 높으면 점수 하락 완화

### 검색 점수 기반 동적 가중치 (이미 구현됨)

**검색 점수 >= 0.8**:
- `law_inquiry`: semantic 0.7, keyword 0.3
- `precedent_search`: semantic 0.85, keyword 0.15
- `general`: semantic 0.75, keyword 0.25

**검색 점수 0.6-0.8**:
- `law_inquiry`: semantic 0.5, keyword 0.5
- `precedent_search`: semantic 0.75, keyword 0.25
- `general`: semantic 0.65, keyword 0.35

**검색 점수 < 0.6**:
- 고정 가중치 사용 (개선된 값)

---

## 비교 분석

### 현재 설정 vs 권장 설정

| 질문 유형 | 현재 (semantic:keyword) | 권장 (semantic:keyword) | 변경 이유 |
|----------|------------------------|------------------------|----------|
| 법령 조회 | 0.3:0.7 | 0.45:0.55 | 검색 점수 보호, 키워드 매칭 낮을 때 점수 하락 완화 |
| 판례 검색 | 0.7:0.3 | 0.65:0.35 | 약간 조정 (현재 설정이 적절하지만 약간 완화) |
| 일반 질문 | 0.5:0.5 | 0.5:0.5 | 유지 (적절함) |

### 예상 효과

**법령 조회 예시**:
- 검색 점수: 0.980
- 키워드 매칭: 0.2

**현재 설정**:
```
hybrid_score = 0.3 * 0.980 + 0.7 * 0.2 = 0.294 + 0.14 = 0.434
```

**권장 설정**:
```
hybrid_score = 0.45 * 0.980 + 0.55 * 0.2 = 0.441 + 0.11 = 0.551
```

**개선 효과**: 0.434 → 0.551 (27% 증가)

---

## 결론

### 권장 사항

1. **고정 가중치 조정**:
   - `hybrid_law`: {"semantic": 0.45, "keyword": 0.55}
   - `hybrid_case`: {"semantic": 0.65, "keyword": 0.35} (약간 조정)
   - `hybrid_general`: {"semantic": 0.5, "keyword": 0.5} (유지)

2. **이유**:
   - 검색 점수가 높을 때 키워드 매칭이 낮아도 점수 하락 완화
   - 법률 도메인 특성 고려 (의미적 검색도 중요)
   - 다른 파일의 가중치 설정과 일관성 유지

3. **동적 가중치**:
   - 검색 점수 기반 동적 가중치는 이미 적절하게 구현됨
   - 고정 가중치는 검색 점수가 낮을 때 사용되므로 조정 필요

