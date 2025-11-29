# Cross-Encoder Reranker 모델 비교 분석

## 📋 개요

법률 문서 검색 시스템에서 사용하는 Cross-Encoder reranker 모델의 성능을 비교 분석한 문서입니다.

**테스트 일시**: 2025-11-29  
**테스트 쿼리**: "계약 해지 사유에 대해 알려주세요"  
**테스트 문서**: 실제 검색 결과 사용 (법령 5개 + 판례 5개 = 총 10개)

## 🔍 비교 대상 모델

1. **Dongjin-kr/ko-reranker**
   - 한국어 특화 reranker 모델
   - HuggingFace Hub: https://huggingface.co/Dongjin-kr/ko-reranker

2. **dragonkue/bge-reranker-v2-m3-ko**
   - BAAI BGE 시리즈의 다국어 reranker 모델 (한국어 포함)
   - HuggingFace Hub: https://huggingface.co/dragonkue/bge-reranker-v2-m3-ko

## 📊 테스트 방법

### 검색 커넥터 직접 사용

실제 검색 결과를 사용하여 두 모델의 성능을 비교했습니다:

```python
# 검색 커넥터를 통해 실제 문서 검색
connector = LegalDataConnectorV2()
statute_results = connector.search_statutes_fts(query, limit=5)
precedent_results = connector.search_cases_fts(query, limit=5)
```

### 테스트 프로세스

1. **문서 검색**: FTS 검색을 통해 법령 5개, 판례 5개 검색
2. **모델 로드**: 각 모델을 로드하고 로드 시간 측정
3. **점수 계산**: 동일한 쿼리-문서 쌍에 대해 Cross-Encoder 점수 계산
4. **통계 분석**: 평균, 최대, 최소 점수 및 문서 타입별 통계 계산

## 📈 테스트 결과

### 전체 성능 비교

| 항목 | Dongjin-kr/ko-reranker | dragonkue/bge-reranker-v2-m3-ko | 차이 |
|------|------------------------|----------------------------------|------|
| **로드 시간** | 3.04초 | 2.41초 | **-20.7%** (빠름) |
| **점수 계산 시간** | 8.35초 | 7.07초 | **-15.3%** (빠름) |
| **평균 점수** | 0.0623 | 0.1253 | **+101.2%** (높음) |
| **최대 점수** | 0.5259 | 0.7352 | **+39.7%** (높음) |
| **최소 점수** | 0.0002 | 0.0000 | - |

### 법령 문서 성능 비교

| 항목 | Dongjin-kr/ko-reranker | dragonkue/bge-reranker-v2-m3-ko | 차이 |
|------|------------------------|----------------------------------|------|
| **평균 점수** | 0.1235 | 0.2506 | **+102.9%** (높음) |
| **문서 수** | 5개 | 5개 | - |

**법령 문서 점수 분포**:
- **Dongjin-kr/ko-reranker**: 0.0017 ~ 0.5259 (범위: 0.5242)
- **dragonkue/bge-reranker-v2-m3-ko**: 0.0000 ~ 0.7352 (범위: 0.7352)

### 판례 문서 성능 비교

| 항목 | Dongjin-kr/ko-reranker | dragonkue/bge-reranker-v2-m3-ko | 차이 |
|------|------------------------|----------------------------------|------|
| **평균 점수** | 0.0011 | 0.0000 | **-99.8%** (낮음) |
| **문서 수** | 5개 | 5개 | - |

**판례 문서 점수 분포**:
- **Dongjin-kr/ko-reranker**: 0.0002 ~ 0.0014 (모두 매우 낮음)
- **dragonkue/bge-reranker-v2-m3-ko**: 0.0000 (모두 0)

## 🔍 상세 분석

### 1. 법령 문서 성능

**dragonkue/bge-reranker-v2-m3-ko**가 법령 문서에서 우수한 성능을 보입니다:

- **평균 점수**: 0.2506 (Dongjin-kr/ko-reranker의 2배)
- **최대 점수**: 0.7352 (Dongjin-kr/ko-reranker의 0.5259보다 높음)
- **점수 분포**: 더 넓은 범위로 차별화 (0.0000 ~ 0.7352)

**법령 문서별 점수 비교**:

| 문서 | Dongjin-kr/ko-reranker | dragonkue/bge-reranker-v2-m3-ko |
|------|------------------------|----------------------------------|
| Doc 1 | 0.0692 | 0.3236 |
| Doc 2 | 0.5259 | **0.7352** |
| Doc 3 | 0.0030 | 0.0000 |
| Doc 4 | 0.0177 | 0.1943 |
| Doc 5 | 0.0017 | 0.0000 |

### 2. 판례 문서 성능

**두 모델 모두 판례 문서에서 매우 낮은 점수를 보입니다**:

- **Dongjin-kr/ko-reranker**: 평균 0.0011 (매우 낮음)
- **dragonkue/bge-reranker-v2-m3-ko**: 평균 0.0000 (모두 0)

**판례 문서 점수가 낮은 이유**:
1. HTML 태그 포함: 판례 문서에 `<br/>` 등의 HTML 태그가 포함되어 있음
2. 텍스트 구조: 판례 문서의 특수한 형식 (【신 청 인】, 【피신청인】 등)
3. 모델 한계: 두 모델 모두 판례 문서의 관련성을 제대로 평가하지 못함

**해결 방안**:
- 텍스트 전처리 강화 (HTML 태그 제거, 텍스트 정리)
- 점수 보정 로직 적용 (현재 구현됨)
- 판례 특화 모델 검토

### 3. 성능 (속도)

**dragonkue/bge-reranker-v2-m3-ko**가 더 빠릅니다:

- **로드 시간**: 2.41초 (Dongjin-kr/ko-reranker: 3.04초) - 20.7% 빠름
- **점수 계산 시간**: 7.07초 (Dongjin-kr/ko-reranker: 8.35초) - 15.3% 빠름

**참고**: 첫 다운로드 시에는 모델 크기(2.27GB)로 인해 시간이 더 걸릴 수 있습니다.

## 💡 결론 및 권장 사항

### 결론

1. **법령 문서**: `dragonkue/bge-reranker-v2-m3-ko`가 우수한 성능
   - 평균 점수 2배 높음
   - 최대 점수도 더 높음
   - 점수 분포가 더 넓어 차별화에 유리

2. **판례 문서**: 두 모델 모두 낮은 점수
   - 추가 개선 필요 (텍스트 전처리, 점수 보정)

3. **성능**: `dragonkue/bge-reranker-v2-m3-ko`가 더 빠름
   - 로드 시간 및 점수 계산 시간 모두 단축

### 권장 사항

#### 1. 모델 선택

**법령 문서 중심 검색**: `dragonkue/bge-reranker-v2-m3-ko` 사용 권장
- 법령 문서에서 우수한 성능
- 더 빠른 처리 속도
- 더 넓은 점수 분포로 차별화에 유리

**판례 문서 포함 검색**: 현재 점수 보정 로직 유지 필요
- 두 모델 모두 판례 문서 점수가 낮음
- 점수 보정 로직으로 영향 완화

#### 2. 개선 방안

**판례 문서 점수 개선**:
1. 텍스트 전처리 강화
   - HTML 태그 제거 (현재 구현됨)
   - 판례 문서 특수 형식 처리
   - 텍스트 정리 및 정규화

2. 점수 보정 로직 강화
   - 판례 문서의 경우 원본 점수(FTS rank_score)에 더 높은 가중치 부여
   - Cross-Encoder 점수가 매우 낮을 때 보정 적용

3. 판례 특화 모델 검토
   - 판례 문서에 특화된 reranker 모델 검토
   - 또는 판례 문서에 대한 파인튜닝 고려

#### 3. 운영 고려사항

**모델 크기**:
- `dragonkue/bge-reranker-v2-m3-ko`: 약 2.27GB (첫 다운로드 필요)
- `Dongjin-kr/ko-reranker`: 상대적으로 작음

**캐싱**:
- 두 모델 모두 HuggingFace 캐시 사용
- 첫 로드 후 캐시에서 빠르게 로드됨

**메모리 사용량**:
- 두 모델 모두 유사한 메모리 사용량 예상
- 실제 운영 환경에서 모니터링 필요

## 📝 테스트 스크립트

테스트 스크립트 위치: `lawfirm_langgraph/tests/runners/compare_reranker_models.py`

**사용법**:
```bash
python lawfirm_langgraph/tests/runners/compare_reranker_models.py "질의 내용"
```

**주요 기능**:
- 검색 커넥터를 통한 실제 문서 검색
- 두 모델의 성능 비교
- 상세 통계 및 분석 결과 출력

## 🔗 참고 자료

- [BGE Reranker V2 M3 문서](https://huggingface.co/dragonkue/bge-reranker-v2-m3-ko)
- [Dongjin-kr/ko-reranker 문서](https://huggingface.co/Dongjin-kr/ko-reranker)
- [판례 문서 점수 개선 분석](../08_features/precedent_low_score_analysis.md)

## 📅 업데이트 이력

- **2025-11-29**: 초기 테스트 및 문서 작성
  - 실제 검색 결과를 사용한 비교 테스트 완료
  - 두 모델의 성능 비교 분석 완료

