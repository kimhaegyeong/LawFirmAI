# Rank Score 계산 가이드

## 개요

이 문서는 LawFirmAI 프로젝트에서 PostgreSQL `ts_rank_cd` 함수를 사용한 검색 결과의 `rank_score`를 `relevance_score`로 정규화하는 방법과 로직에 대해 설명합니다.

## 목차

1. [배경 및 개요](#배경-및-개요)
2. [PostgreSQL ts_rank_cd 특성](#postgresql-ts_rank_cd-특성)
3. [정규화 계수 설정](#정규화-계수-설정)
4. [rank_score 처리 로직](#rank_score-처리-로직)
5. [환경 변수 설정](#환경-변수-설정)
6. [로깅 및 디버깅](#로깅-및-디버깅)
7. [실제 사용 예시](#실제-사용-예시)
8. [문제 해결](#문제-해결)

---

## 배경 및 개요

### 문제 상황

PostgreSQL의 `ts_rank_cd` 함수는 텍스트 검색 결과의 관련성을 점수로 반환합니다. 그러나 이 점수는 다음과 같은 특성을 가집니다:

1. **점수 범위의 불일치**: `ts_rank_cd`는 일반적으로 0.0 ~ 1.0 범위의 값을 반환하지만, 때로는 1.0보다 큰 값이 나올 수 있습니다.
2. **낮은 점수 분포**: 대부분의 경우 `rank_score`는 0.01 ~ 0.1 범위의 작은 값을 가집니다.
3. **임계값 필터링의 어려움**: `similarity_threshold` (기본값 0.75)와 비교하기 위해서는 적절한 정규화가 필요합니다.

### 해결 방안

`rank_score`를 `relevance_score` (0.0 ~ 1.0 범위)로 정규화하여 `similarity_threshold`와 비교할 수 있도록 합니다.

---

## PostgreSQL ts_rank_cd 특성

### 점수 범위

PostgreSQL의 `ts_rank_cd` 함수는 다음과 같은 특성을 가집니다:

- **일반적인 범위**: 0.0 ~ 1.0
- **예외적인 경우**: 1.0보다 큰 값이 나올 수 있음 (예: 1.2, 1.9, 5.8 등)
- **실제 분포**: 대부분의 경우 0.01 ~ 0.1 범위의 작은 값

### 예시

```sql
-- PostgreSQL 쿼리 예시
SELECT 
    ts_rank_cd(to_tsvector('simple', article_content), plainto_tsquery('simple', '손해배상')) as rank_score
FROM statutes_articles
WHERE to_tsvector('simple', article_content) @@ plainto_tsquery('simple', '손해배상')
ORDER BY rank_score DESC
LIMIT 10;
```

**실제 로그에서 확인된 rank_score 값들**:
- `rank_score=0.900000` (일반적인 경우)
- `rank_score=0.600000` (일반적인 경우)
- `rank_score=1.900000` (1.0 이상인 경우)
- `rank_score=5.800000` (1.0 이상인 경우)

---

## 정규화 계수 설정

### 기본 정규화 계수

기본 정규화 계수는 **15.0**입니다. 이 값은 환경 변수 `POSTGRESQL_NORMALIZATION_COEFFICIENT`를 통해 조정할 수 있습니다.

### 정규화 계수 선택 기준

정규화 계수는 다음을 고려하여 선택합니다:

1. **rank_score 분포**: 대부분의 `rank_score`가 0.01 ~ 0.1 범위에 있으므로, 이를 0.75 이상의 `relevance_score`로 변환하기 위해서는 충분히 큰 계수가 필요합니다.
2. **임계값 유지**: `similarity_threshold` (기본값 0.75)를 유지하면서도 적절한 수의 문서가 필터링을 통과할 수 있어야 합니다.
3. **실험적 조정**: 실제 데이터에서 `rank_score` 분포를 확인하고, 상위 문서의 일정 비율(예: 20-30%)이 임계값을 통과하도록 조정합니다.

### 정규화 공식

```python
if rank_score >= 1.0:
    # 이미 높은 점수이므로 그대로 사용 (최대 1.0으로 제한)
    relevance_score = min(1.0, rank_score)
else:
    # 일반적인 경우: 정규화 계수 적용
    relevance_score = max(0.0, min(1.0, rank_score * normalization_coefficient))
```

**예시**:
- `rank_score=0.05`, `coefficient=15.0` → `relevance_score = 0.05 * 15.0 = 0.75` ✅
- `rank_score=0.10`, `coefficient=15.0` → `relevance_score = 0.10 * 15.0 = 1.0` (최대값으로 제한)
- `rank_score=1.90`, `coefficient=15.0` → `relevance_score = 1.0` (1.0 이상이므로 그대로 사용, 최대값으로 제한)

---

## rank_score 처리 로직

### 구현 위치

`lawfirm_langgraph/core/search/connectors/legal_data_connector_v2.py`의 `LegalDataConnectorV2` 클래스에서 처리됩니다.

### 처리 로직

다음과 같은 위치에서 `rank_score`를 `relevance_score`로 변환합니다:

1. **법령 검색** (`_fallback_statute_search`, `search_statutes_fts`)
2. **판례 검색** (`search_cases_fts`)
3. **판결문 검색** (`search_decisions_fts`)
4. **해석 검색** (`search_interpretations_fts`)

### 코드 구조

```python
class LegalDataConnectorV2:
    def __init__(self, db_path: Optional[str] = None):
        # 정규화 계수 초기화
        self.postgresql_normalization_coefficient = float(
            os.getenv("POSTGRESQL_NORMALIZATION_COEFFICIENT", "15.0")
        )
    
    def _process_rank_score(self, rank_score: float) -> float:
        """rank_score를 relevance_score로 변환"""
        if rank_score >= 1.0:
            # 이미 높은 점수이므로 그대로 사용 (최대 1.0으로 제한)
            relevance_score = min(1.0, rank_score)
        else:
            # 일반적인 경우: 정규화 계수 적용
            relevance_score = max(0.0, min(1.0, rank_score * self.postgresql_normalization_coefficient))
        return relevance_score
```

### 실제 구현 예시

```python
# search_cases_fts 메서드 내부
if row.get('rank_score') is not None:
    rank_score = row['rank_score']
    if self._db_adapter and self._db_adapter.db_type == 'postgresql':
        # PostgreSQL ts_rank_cd는 일반적으로 0.0 ~ 1.0 범위이지만, 때로는 더 큰 값이 나올 수 있음
        if rank_score >= 1.0:
            relevance_score = min(1.0, rank_score)
        else:
            relevance_score = max(0.0, min(1.0, rank_score * self.postgresql_normalization_coefficient))
        
        # 디버깅용 로깅
        self.logger.debug(
            f"[RANK_SCORE] query='{query[:50]}', "
            f"rank_score={rank_score:.6f}, "
            f"relevance_score={relevance_score:.4f}, "
            f"coefficient={self.postgresql_normalization_coefficient}, "
            f"type=precedent_content_fts, "
            f"normalized={'no' if rank_score >= 1.0 else 'yes'}"
        )
```

---

## 환경 변수 설정

### 환경 변수명

`POSTGRESQL_NORMALIZATION_COEFFICIENT`

### 기본값

`15.0`

### 설정 방법

#### Windows (PowerShell)

```powershell
$env:POSTGRESQL_NORMALIZATION_COEFFICIENT="15.0"
python lawfirm_langgraph/tests/runners/run_query_test.py "테스트 질의"
```

#### Linux/Mac (Bash)

```bash
export POSTGRESQL_NORMALIZATION_COEFFICIENT=15.0
python lawfirm_langgraph/tests/runners/run_query_test.py "테스트 질의"
```

#### .env 파일

```env
POSTGRESQL_NORMALIZATION_COEFFICIENT=15.0
```

### 권장 값

- **기본값**: `15.0` (대부분의 경우 적절함)
- **낮은 점수 분포**: `20.0` ~ `30.0` (더 많은 문서가 임계값을 통과하도록)
- **높은 점수 분포**: `10.0` ~ `12.0` (더 엄격한 필터링)

---

## 로깅 및 디버깅

### 로깅 레벨

`DEBUG` 레벨에서 `rank_score` 변환 과정이 로깅됩니다.

### 로그 형식

```
[RANK_SCORE] query='질의 내용', rank_score=0.900000, relevance_score=1.0000, coefficient=15.0, type=precedent_content_fts, normalized=yes
```

### 로그 필드 설명

- `query`: 검색 질의 (최대 50자)
- `rank_score`: PostgreSQL `ts_rank_cd`에서 반환된 원본 점수
- `relevance_score`: 정규화된 관련성 점수 (0.0 ~ 1.0)
- `coefficient`: 적용된 정규화 계수
- `type`: 문서 타입 (예: `precedent_content_fts`, `statute_article_fts`)
- `normalized`: 정규화 여부 (`yes` 또는 `no`)

### 로그 분석 예시

#### PowerShell

```powershell
# rank_score 분포 확인
Select-String -Pattern "RANK_SCORE" "logs/langgraph/test_langgraph_query_*.log" | 
    Select-String -Pattern "rank_score=(\d+\.\d+)" | 
    ForEach-Object { [double]($_.Matches.Groups[1].Value) } | 
    Measure-Object -Average -Maximum -Minimum

# 정규화되지 않은 rank_score 확인 (>= 1.0)
Select-String -Pattern "normalized=no" "logs/langgraph/test_langgraph_query_*.log"

# 정규화된 rank_score 확인 (< 1.0)
Select-String -Pattern "normalized=yes" "logs/langgraph/test_langgraph_query_*.log"
```

#### 로그에서 확인할 수 있는 정보

1. **정규화 계수 적용 확인**:
   ```
   PostgreSQL normalization coefficient: 15.0
   ```

2. **rank_score >= 1.0 처리**:
   ```
   rank_score=1.900000, relevance_score=1.0000, normalized=no
   ```

3. **rank_score < 1.0 처리**:
   ```
   rank_score=0.900000, relevance_score=1.0000, normalized=yes
   ```

---

## 실제 사용 예시

### 예시 1: 일반적인 rank_score (< 1.0)

**입력**:
- `rank_score = 0.05`
- `normalization_coefficient = 15.0`

**처리**:
```python
if rank_score >= 1.0:  # False
    relevance_score = min(1.0, rank_score)
else:
    relevance_score = max(0.0, min(1.0, 0.05 * 15.0))  # 0.75
```

**결과**:
- `relevance_score = 0.75`
- `normalized = yes`

### 예시 2: 높은 rank_score (>= 1.0)

**입력**:
- `rank_score = 1.90`
- `normalization_coefficient = 15.0`

**처리**:
```python
if rank_score >= 1.0:  # True
    relevance_score = min(1.0, 1.90)  # 1.0
else:
    relevance_score = max(0.0, min(1.0, rank_score * coefficient))
```

**결과**:
- `relevance_score = 1.0`
- `normalized = no`

### 예시 3: 매우 높은 rank_score

**입력**:
- `rank_score = 5.80`
- `normalization_coefficient = 15.0`

**처리**:
```python
if rank_score >= 1.0:  # True
    relevance_score = min(1.0, 5.80)  # 1.0
else:
    relevance_score = max(0.0, min(1.0, rank_score * coefficient))
```

**결과**:
- `relevance_score = 1.0`
- `normalized = no`

---

## 문제 해결

### 문제 1: 너무 많은 문서가 필터링됨

**증상**: `similarity_threshold` (0.75)를 통과하는 문서가 너무 적음

**원인**: 정규화 계수가 너무 작음

**해결 방법**:
1. 환경 변수 `POSTGRESQL_NORMALIZATION_COEFFICIENT` 값을 증가시킵니다 (예: 15.0 → 20.0)
2. 로그에서 `rank_score` 분포를 확인하고 적절한 계수를 선택합니다

```powershell
$env:POSTGRESQL_NORMALIZATION_COEFFICIENT="20.0"
```

### 문제 2: 너무 많은 문서가 통과함

**증상**: `similarity_threshold` (0.75)를 통과하는 문서가 너무 많음

**원인**: 정규화 계수가 너무 큼

**해결 방법**:
1. 환경 변수 `POSTGRESQL_NORMALIZATION_COEFFICIENT` 값을 감소시킵니다 (예: 15.0 → 12.0)
2. 로그에서 `rank_score` 분포를 확인하고 적절한 계수를 선택합니다

```powershell
$env:POSTGRESQL_NORMALIZATION_COEFFICIENT="12.0"
```

### 문제 3: rank_score >= 1.0인 경우 처리 확인

**증상**: 로그에서 `normalized=no`가 자주 나타남

**원인**: `rank_score >= 1.0`인 경우 정규화 없이 그대로 사용

**해결 방법**:
- 이는 정상적인 동작입니다. `rank_score >= 1.0`인 경우는 이미 높은 점수이므로 정규화 없이 최대값 1.0으로 제한합니다.

### 문제 4: 로그가 출력되지 않음

**증상**: `[RANK_SCORE]` 로그가 보이지 않음

**원인**: 로그 레벨이 `DEBUG`보다 높게 설정됨

**해결 방법**:
1. 환경 변수 `LOG_LEVEL`을 `DEBUG`로 설정합니다

```powershell
$env:LOG_LEVEL="DEBUG"
```

2. 또는 테스트 스크립트 실행 시 로그 레벨을 설정합니다

---

## 참고 자료

### 관련 파일

- `lawfirm_langgraph/core/search/connectors/legal_data_connector_v2.py`: `rank_score` 처리 로직 구현
- `lawfirm_langgraph/core/workflow/legal_workflow_enhanced.py`: `relevance_score` 필터링 로직

### 관련 문서

- [PostgreSQL Full-Text Search 문서](https://www.postgresql.org/docs/current/textsearch-controls.html)
- [PostgreSQL ts_rank_cd 함수 문서](https://www.postgresql.org/docs/current/textsearch-controls.html#TEXTSEARCH-RANKING)

### 변경 이력

- **2025-11-28**: 정규화 계수를 환경 변수로 설정 가능하게 개선, `rank_score >= 1.0` 처리 로직 추가
- **2025-11-28**: 정규화 계수 기본값을 10.0에서 15.0으로 증가

---

## 요약

1. **PostgreSQL `ts_rank_cd`**는 일반적으로 0.0 ~ 1.0 범위의 점수를 반환하지만, 때로는 1.0보다 큰 값이 나올 수 있습니다.
2. **정규화 계수** (기본값 15.0)를 사용하여 `rank_score`를 `relevance_score` (0.0 ~ 1.0)로 변환합니다.
3. **`rank_score >= 1.0`**인 경우는 정규화 없이 그대로 사용하되, 최대값 1.0으로 제한합니다.
4. **환경 변수** `POSTGRESQL_NORMALIZATION_COEFFICIENT`를 통해 정규화 계수를 동적으로 조정할 수 있습니다.
5. **로깅**을 통해 `rank_score` 분포와 정규화 과정을 확인할 수 있습니다.

