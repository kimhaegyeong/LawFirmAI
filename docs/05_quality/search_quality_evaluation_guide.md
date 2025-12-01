# 검색 품질 평가 가이드

## 개요

검색 품질 개선 기능의 효과를 측정하고 평가하기 위한 가이드입니다.

## 평가 스크립트

### 0. 빠른 테스트 스크립트

**파일**: `lawfirm_langgraph/tests/scripts/quick_evaluation_test.py`

**사용법**:

```bash
# 소수의 쿼리(6개)로 빠르게 테스트
python lawfirm_langgraph/tests/scripts/quick_evaluation_test.py
```

**용도**: 평가 시스템이 정상 작동하는지 빠르게 확인

### 1. 단일 평가 스크립트

**파일**: `lawfirm_langgraph/tests/scripts/test_search_quality_evaluation.py`

**사용법**:

```bash
# 개선 기능 활성화 상태로 평가
python lawfirm_langgraph/tests/scripts/test_search_quality_evaluation.py --enable-improvements

# 개선 기능 비활성화 상태로 평가
python lawfirm_langgraph/tests/scripts/test_search_quality_evaluation.py --disable-improvements

# 특정 질문 유형만 평가
python lawfirm_langgraph/tests/scripts/test_search_quality_evaluation.py --query-type statute_article

# 결과 저장 경로 지정
python lawfirm_langgraph/tests/scripts/test_search_quality_evaluation.py --output logs/my_evaluation.json
```

**옵션**:
- `--enable-improvements`: 개선 기능 활성화 (기본값)
- `--disable-improvements`: 개선 기능 비활성화
- `--query-type`: 평가할 질문 유형 선택
  - `statute_article`: 법령 조문 문의 (20개)
  - `precedent`: 판례 검색 (20개)
  - `procedure`: 절차 문의 (20개)
  - `general_question`: 일반 질문 (20개)
  - `all`: 전체 (80개)
- `--output`: 결과 저장 경로

### 2. Before/After 비교 스크립트

**파일**: `lawfirm_langgraph/tests/scripts/compare_search_quality.py`

**사용법**:

```bash
# 전체 비교
python lawfirm_langgraph/tests/scripts/compare_search_quality.py

# 특정 질문 유형만 비교
python lawfirm_langgraph/tests/scripts/compare_search_quality.py --query-type precedent

# 결과 저장 디렉토리 지정
python lawfirm_langgraph/tests/scripts/compare_search_quality.py --output-dir logs/my_comparison
```

**생성되는 파일**:
- `before_results.json`: 개선 전 결과
- `after_results.json`: 개선 후 결과
- `comparison.json`: 비교 결과 (JSON)
- `comparison_report.md`: 비교 리포트 (Markdown)

## 평가 메트릭

### 1. Precision@K

상위 K개 결과 중 관련 문서의 비율

```
Precision@K = (관련 문서 수) / K
```

### 2. Recall@K

관련 문서 중 검색된 비율

```
Recall@K = (검색된 관련 문서 수) / (전체 관련 문서 수)
```

### 3. NDCG@K

순위를 고려한 정규화된 누적 이득 (Normalized Discounted Cumulative Gain)

```
NDCG@K = DCG@K / IDCG@K
```

### 4. Keyword Coverage

검색 결과에 쿼리 키워드가 포함된 비율

```
Keyword Coverage = (매칭된 키워드 수) / (전체 키워드 수)
```

### 5. Diversity Score

검색 결과의 다양성 점수 (출처 및 타입 다양성)

```
Diversity Score = (출처 다양성 + 타입 다양성) / 2
```

### 6. Average Relevance

평균 관련성 점수

```
Avg Relevance = Σ(relevance_score) / N
```

### 7. Response Time

평균 응답 시간 (초)

## 테스트 쿼리 세트

### 법령 조문 문의 (20개)
- 민법, 형법, 상법 등 주요 법령의 조문 문의
- 예: "민법 제1조의 내용은 무엇인가요?"

### 판례 검색 (20개)
- 주요 법률 분야별 판례 검색
- 예: "계약 해지 사유에 대한 대법원 판례를 알려주세요"

### 절차 문의 (20개)
- 소송 절차, 행정 절차 등 절차 관련 문의
- 예: "민사소송 절차는 어떻게 진행되나요?"

### 일반 질문 (20개)
- 법률 상담, 계약, 권리 등 일반적인 법률 질문
- 예: "계약서 작성 시 주의사항은?"

## MLflow 통합

평가 결과는 자동으로 MLflow에 추적됩니다.

### 실험 이름
- `search_quality_evaluation`: 단일 평가
- `search_quality_before_improvements`: 개선 전 배치 평가
- `search_quality_after_improvements`: 개선 후 배치 평가

### 추적되는 정보
- **파라미터**: 개선 기능 활성화 여부, 테스트 쿼리 수 등
- **메트릭**: Precision@K, Recall@K, NDCG@K, Keyword Coverage, Diversity Score 등
- **태그**: 질문 유형, 쿼리 길이, 결과 수 등
- **아티팩트**: 샘플 쿼리 및 결과

### MLflow UI 확인

```bash
# MLflow UI 실행
mlflow ui

# 브라우저에서 http://localhost:5000 접속
```

## 결과 해석

### 개선 효과 판단 기준

1. **Precision@K 증가**: 상위 결과의 관련성 향상
2. **Recall@K 증가**: 관련 문서 검색률 향상
3. **NDCG@K 증가**: 순위 정확도 향상
4. **Keyword Coverage 증가**: 키워드 매칭률 향상
5. **Diversity Score 증가**: 결과 다양성 향상
6. **Response Time 감소**: 응답 속도 개선 (또는 유지)

### 예상 개선 효과

- **Precision@5**: 10-20% 향상 예상
- **Recall@10**: 15-25% 향상 예상
- **NDCG@10**: 20-30% 향상 예상
- **Keyword Coverage**: 10-15% 향상 예상
- **Diversity Score**: 15-20% 향상 예상

## 주의사항

1. **평가 시간**: 전체 80개 쿼리 평가 시 약 20-30분 소요
2. **리소스 사용**: 검색 및 평가 과정에서 CPU/메모리 사용량 증가
3. **MLflow 설정**: MLflow tracking URI가 설정되어 있어야 추적 가능
4. **관련 문서 ID**: Precision/Recall/NDCG 계산을 위해 관련 문서 ID가 필요 (현재는 추정값 사용)
5. **로깅 개선**: 로깅 에러 방지를 위해 `SafeStreamHandler`가 적용되어 있습니다

## 향후 개선

1. **Ground Truth 구축**: 실제 관련 문서 ID 수집 및 구축
2. **자동화**: CI/CD 파이프라인에 평가 통합
3. **대시보드**: 실시간 품질 모니터링 대시보드 구축
4. **A/B 테스트**: 프로덕션 환경에서 A/B 테스트 수행

