# 가중치 검증 시스템 사용 가이드

> **실험 계획서**: 상세한 실험 계획은 [`weight_validation_experiment_plan.md`](./weight_validation_experiment_plan.md)를 참조하세요.

## 개요

가중치 검증 시스템은 다양한 가중치 조합을 테스트하고 평가 메트릭을 수집하여 최적의 가중치를 찾는 도구입니다.

## 주요 기능

1. **다양한 가중치 조합 자동 생성**
   - 법령 조회, 판례 검색, 일반 질문별 가중치 조합 생성
   - 빠른 테스트 모드 지원

2. **포괄적인 평가 메트릭 수집**
   - 검색 관련 메트릭 (관련성 점수, 키워드 커버리지)
   - 문서 활용 메트릭 (검색된 문서 수, 실제 사용된 문서 수, 활용률)
   - 답변 품질 메트릭 (답변 길이, 품질 점수, 소스 포함 여부)
   - 소스 관련성 메트릭 (소스 관련성 평균, 소스 커버리지)
   - 성능 메트릭 (총 소요 시간, 검색 시간, 생성 시간)

3. **결과 분석 및 리포트 생성**
   - 설정별 평균 점수 계산
   - 최적 가중치 자동 선택
   - 질문 유형별 분석
   - JSON 및 텍스트 리포트 생성

## 사용 방법

### 기본 사용

```bash
# 모든 질문 유형에 대해 전체 테스트
python lawfirm_langgraph/tests/scripts/validate_weight_configurations.py
```

### 빠른 테스트

```bash
# 빠른 테스트 모드 (적은 조합으로 빠르게 테스트)
python lawfirm_langgraph/tests/scripts/validate_weight_configurations.py --quick
```

### 특정 질문 유형만 테스트

```bash
# 법령 조회만 테스트
python lawfirm_langgraph/tests/scripts/validate_weight_configurations.py --query-type law_inquiry

# 판례 검색만 테스트
python lawfirm_langgraph/tests/scripts/validate_weight_configurations.py --query-type precedent_search

# 일반 질문만 테스트
python lawfirm_langgraph/tests/scripts/validate_weight_configurations.py --query-type general
```

### 결과 저장 위치 지정

```bash
# 결과를 특정 디렉토리에 저장
python lawfirm_langgraph/tests/scripts/validate_weight_configurations.py --output-dir logs/custom
```

## 평가 메트릭

### 종합 점수 계산

종합 점수는 다음 가중치로 계산됩니다:

- **답변 품질**: 30%
- **문서 활용률**: 25%
- **소스 관련성**: 20%
- **검색 점수**: 15%
- **성능**: 10%

### 개별 메트릭

#### 검색 관련 메트릭
- `avg_relevance_score`: 평균 관련성 점수
- `min_relevance_score`: 최소 관련성 점수
- `max_relevance_score`: 최대 관련성 점수
- `keyword_coverage`: 키워드 커버리지

#### 문서 활용 메트릭
- `retrieved_docs_count`: 검색된 문서 수
- `used_docs_count`: 실제 사용된 문서 수
- `document_utilization_rate`: 문서 활용률 (used_docs / retrieved_docs)

#### 답변 품질 메트릭
- `answer_length`: 답변 길이 (자)
- `answer_quality_score`: 답변 품질 점수 (0-100)
- `has_sources`: 소스 포함 여부
- `source_count`: 소스 개수

#### 소스 관련성 메트릭
- `source_relevance_avg`: 소스 관련성 평균
- `source_coverage`: 소스 커버리지 (답변이 소스에 기반하는 정도)

#### 성능 메트릭
- `total_time`: 총 소요 시간 (초)
- `search_time`: 검색 시간 (초)
- `generation_time`: 생성 시간 (초)

## 결과 분석

### 리포트 파일

테스트 완료 후 다음 파일들이 생성됩니다:

1. **JSON 리포트**: `weight_validation_YYYYMMDD_HHMMSS.json`
   - 전체 결과 데이터 (JSON 형식)
   - 설정별 점수, 최적 설정, 질문 유형별 분석 포함

2. **텍스트 리포트**: `weight_validation_YYYYMMDD_HHMMSS.txt`
   - 사람이 읽기 쉬운 형식의 요약 리포트
   - 최적 설정, 상위 10개 설정, 질문 유형별 분석 포함

### 리포트 구조

#### JSON 리포트 구조

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "analysis": {
    "total_tests": 100,
    "successful_tests": 95,
    "config_scores": {
      "law_sem0.45_kw0.55": {
        "avg_score": 0.823,
        "median_score": 0.815,
        "min_score": 0.750,
        "max_score": 0.890,
        "std_dev": 0.045,
        "test_count": 15
      }
    },
    "best_config": {
      "name": "law_sem0.45_kw0.55",
      "metrics": {
        "avg_score": 0.823,
        "median_score": 0.815,
        "min_score": 0.750,
        "max_score": 0.890,
        "std_dev": 0.045,
        "test_count": 15
      }
    },
    "query_type_analysis": {
      "law_inquiry": {
        "avg_score": 0.823,
        "test_count": 15
      }
    }
  },
  "detailed_results": [...],
  "recommendations": [
    "최적 가중치 설정: law_sem0.45_kw0.55 (평균 점수: 0.823)",
    "law_inquiry 질문 유형 평균 점수: 0.823"
  ]
}
```

## 가중치 조합 생성 규칙

### 법령 조회 (law_inquiry)

**전체 테스트 모드**:
- semantic: 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6
- keyword: 1.0 - semantic

**빠른 테스트 모드**:
- semantic: 0.3, 0.4, 0.45, 0.5
- keyword: 1.0 - semantic

### 판례 검색 (precedent_search)

**전체 테스트 모드**:
- semantic: 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8
- keyword: 1.0 - semantic

**빠른 테스트 모드**:
- semantic: 0.6, 0.65, 0.7, 0.75
- keyword: 1.0 - semantic

### 일반 질문 (general)

**전체 테스트 모드**:
- semantic: 0.4, 0.45, 0.5, 0.55, 0.6
- keyword: 1.0 - semantic

**빠른 테스트 모드**:
- semantic: 0.5 (기본값만)
- keyword: 0.5

## 테스트 쿼리 세트

### 법령 조회 (law_inquiry)
- "민법 제750조 손해배상에 대해 설명해주세요"
- "계약 위약금에 대해 설명해주세요"
- "근로기준법 제60조 휴게시간에 대해 알려주세요"
- "형법 제250조 살인죄의 구성요건은 무엇인가요?"
- "상법 제170조 주식회사의 설립에 대해 설명해주세요"

### 판례 검색 (precedent_search)
- "계약 해지 관련 판례를 찾아주세요"
- "손해배상 청구 사례를 알려주세요"
- "부당해고 판례를 찾아주세요"
- "임대차 계약 해지 판례를 알려주세요"
- "계약 위약금 관련 판례를 찾아주세요"

### 일반 질문 (general)
- "법률 자문이 필요합니다"
- "계약서 작성 시 주의사항을 알려주세요"
- "법률 용어를 설명해주세요"
- "법률 절차에 대해 안내해주세요"
- "법률 상담이 필요합니다"

## 최적 가중치 적용 방법

검증 결과를 바탕으로 최적 가중치를 적용하려면:

1. 리포트에서 최적 설정 확인
2. `lawfirm_langgraph/core/search/processors/search_result_processor.py` 파일 수정
3. `weight_config` 딕셔너리의 값 업데이트

예시:
```python
self.weight_config = weight_config or {
    "hybrid_law": {"semantic": 0.45, "keyword": 0.55},  # 검증 결과 반영
    "hybrid_case": {"semantic": 0.65, "keyword": 0.35},  # 검증 결과 반영
    "hybrid_general": {"semantic": 0.5, "keyword": 0.5},
    "doc_type_boost": {"statute": 1.2, "case": 1.15},
    "quality_weight": 0.2,
    "keyword_adjustment": 1.8
}
```

## 주의사항

1. **테스트 시간**: 전체 테스트는 시간이 오래 걸릴 수 있습니다 (수십 개의 조합 × 여러 쿼리)
2. **원본 설정 백업**: 테스트 전에 원본 설정이 자동으로 백업되며, 테스트 후 복원됩니다
3. **환경 변수**: `.env` 파일이 올바르게 설정되어 있어야 합니다
4. **의존성**: `run_query_test.py`가 정상 작동해야 합니다

## 트러블슈팅

### 테스트 실패
- 로그 파일 확인: `logs/test/run_query_test_*.log`
- 환경 변수 확인: `.env` 파일 설정 확인
- 데이터베이스 연결 확인

### 메트릭 추출 실패
- `run_query_test.py`의 출력 형식이 변경되었을 수 있습니다
- `MetricsExtractor` 클래스의 정규식 패턴 확인

### 성능 문제
- `--quick` 옵션 사용하여 빠른 테스트 모드 실행
- 특정 질문 유형만 테스트 (`--query-type` 옵션)

## 향후 개선 사항

1. **자동 가중치 적용**: 검증 결과를 바탕으로 자동으로 가중치 적용
2. **실시간 모니터링**: 테스트 진행 상황 실시간 모니터링
3. **통계 분석**: 더 상세한 통계 분석 (신뢰구간, 유의성 검정 등)
4. **시각화**: 결과 시각화 (그래프, 차트 등)
5. **A/B 테스트**: 실제 사용자 환경에서의 A/B 테스트

