# LawFirmAI 테스트 가이드

LawFirmAI 프로젝트의 테스트 구조 및 실행 방법에 대한 가이드입니다.

## 목차

- [빠른 시작](#빠른-시작)
- [디렉토리 구조](#디렉토리-구조)
- [테스트 파일 목록 및 목적](#테스트-파일-목록-및-목적)
- [테스트 실행 방법](#테스트-실행-방법)
- [테스트 환경 설정](#테스트-환경-설정)
- [테스트 작성 가이드](#테스트-작성-가이드)
- [테스트 실행 우선순위](#테스트-실행-우선순위)
- [테스트 결과 확인](#테스트-결과-확인)
- [문제 해결](#문제-해결)
- [테스트 유지보수](#테스트-유지보수)
- [삭제된 파일 목록](#삭제된-파일-목록)
- [테스트 파일 통계](#테스트-파일-통계)

## 빠른 시작

```bash
# 모든 테스트 실행
pytest tests/ -v

# 특정 카테고리 테스트 실행
pytest tests/langgraph/ -v  # LangGraph 테스트만
pytest tests/search/ -v     # 검색 시스템 테스트만

# 마스터 테스트 실행
python tests/run_master_tests.py
```

## 디렉토리 구조

```
tests/
├── README.md                      # 메인 테스트 가이드 (본 문서)
├── TEST_ORGANIZATION.md           # 리다이렉트 문서 (README.md로 통합됨)
├── verify_test_structure.py       # 테스트 구조 검증 스크립트
├── migrate_test_files.py          # 파일 마이그레이션 스크립트 (유지보수용)
├── conftest.py                    # Pytest 공통 설정 및 픽스처
├── run_master_tests.py            # 마스터 테스트 실행 스크립트
│
├── langgraph/                     # LangGraph 전용 테스트 (12개 파일)
│   ├── README.md                  # LangGraph 테스트 상세 가이드
│   ├── ENV_PROFILES_EXAMPLE.md    # 환경변수 프로필 예시
│   ├── monitoring_switch.py       # 모니터링 전환 유틸리티
│   ├── test_monitoring_switch_basic.py
│   ├── test_profile_loading.py
│   ├── test_with_monitoring_switch.py
│   ├── test_langgraph.py          # 기본 LangGraph 워크플로우
│   ├── test_langgraph_state_optimization.py
│   ├── test_langgraph_multi_turn.py
│   ├── test_all_state_systems.py
│   ├── test_core_state_systems.py
│   ├── test_state_reduction_performance.py
│   └── fixtures/
│       ├── __init__.py
│       ├── monitoring_configs.py
│       └── workflow_factory.py
│
├── integration/                   # 통합 시스템 테스트 (2개)
│   ├── test_comprehensive_system.py
│   └── test_integrated_system.py
│
├── search/                        # 검색 시스템 테스트 (6개)
│   ├── test_query_classification.py
│   ├── test_query_system.py
│   ├── test_classify_question_type.py
│   ├── test_hybrid_search_integration.py
│   ├── test_hybrid_search_simple.py
│   └── test_rag_integration.py
│
├── legal/                         # 법률 시스템 테스트 (4개)
│   ├── test_legal_basis_system.py
│   ├── test_database_keyword_system.py
│   ├── test_term_integration_workflow.py
│   └── test_akls_integration.py   # AKLS 통합 (legal 디렉토리)
│
├── monitoring/                    # 모니터링 및 통합 테스트 (3개)
│   ├── test_langsmith_integration.py
│   ├── test_langfuse_integration.py
│   └── test_unified_prompt_integration.py
│
├── quality_performance/           # 품질 및 성능 테스트 (7개)
│   ├── test_quality_enhancement.py
│   ├── test_quality_improvement_workflow.py
│   ├── test_performance_benchmark.py
│   ├── test_performance_monitor_fix.py
│   ├── test_optimized_performance.py
│   ├── test_stress_system.py
│   └── test_workflow_execution.py
│
├── phase/                         # Phase별 기능 테스트 (3개)
│   ├── test_phase1_context_enhancement.py
│   ├── test_phase2_personalization_analysis.py
│   └── test_phase3_memory_quality.py
│
├── akls/                          # AKLS (법률 용어) 관련 테스트 (3개)
│   ├── test_akls_gradio.py
│   ├── test_akls_integration.py
│   └── test_akls_performance.py
│
├── unit/                          # 단위 테스트
│   ├── __init__.py
│   └── models/
│       └── __init__.py
│
└── fixtures/                      # 공통 테스트 픽스처
    └── __init__.py
```

## 테스트 파일 목록 및 목적

### 📁 Category 1: LangGraph 테스트 (최신)

LangGraph 워크플로우 및 State 관리 시스템을 테스트하는 파일들입니다.

**디렉토리:** `tests/langgraph/`

#### 모니터링 전환 관련

| 파일 | 목적 | 용도 |
|------|------|------|
| `monitoring_switch.py` | 모니터링 도구 전환 유틸리티 | LangSmith/Langfuse 전환 관리 |
| `test_monitoring_switch_basic.py` | 모니터링 전환 기본 기능 테스트 | 환경변수 설정/복원 검증 |
| `test_profile_loading.py` | 환경변수 프로필 로딩 테스트 | .env.profiles/ 파일 로딩 검증 |
| `test_with_monitoring_switch.py` | 통합 모니터링 전환 테스트 | 모든 모드 전환 시나리오 테스트 |
| `fixtures/workflow_factory.py` | 워크플로우 팩토리 | 모드별 워크플로우 인스턴스 생성/캐싱 |
| `fixtures/monitoring_configs.py` | 모니터링 설정 픽스처 | 모니터링 설정 관리 |

#### LangGraph 워크플로우 테스트

| 파일 | 목적 | 테스트 내용 |
|------|------|-------------|
| `test_langgraph.py` | 기본 LangGraph 워크플로우 테스트 | 워크플로우 기본 동작 검증 |
| `test_langgraph_state_optimization.py` | State 최적화 테스트 | State 구조 및 최적화 기능 검증 |
| `test_langgraph_multi_turn.py` | 멀티턴 대화 테스트 | 대화 히스토리 관리 검증 |
| `test_all_state_systems.py` | State 시스템 통합 테스트 | 전체 State 시스템 통합 검증 |
| `test_core_state_systems.py` | Core State 시스템 테스트 | Core State 컴포넌트 검증 |
| `test_state_reduction_performance.py` | State Reduction 성능 테스트 | 메모리 사용량 및 성능 측정 |

**상세 가이드:** [LangGraph 테스트 README](./langgraph/README.md)

---

### 📁 Category 2: Phase별 기능 테스트

개발 단계별로 구현된 기능을 테스트하는 파일들입니다.

**디렉토리:** `tests/phase/`

| 파일 | 목적 | 테스트 내용 |
|------|------|-------------|
| `test_phase1_context_enhancement.py` | Phase 1 기능 테스트 | 세션 관리, 다중 턴 처리, 컨텍스트 압축 |
| `test_phase2_personalization_analysis.py` | Phase 2 기능 테스트 | 사용자 프로필, 감정 분석, 대화 흐름 추적 |
| `test_phase3_memory_quality.py` | Phase 3 기능 테스트 | 맥락적 메모리, 대화 품질 모니터링 |

---

### 📁 Category 3: 통합 시스템 테스트

전체 시스템의 통합 동작을 검증하는 파일들입니다.

**디렉토리:** `tests/integration/`

| 파일 | 목적 | 테스트 내용 |
|------|------|-------------|
| `test_comprehensive_system.py` | 종합 시스템 통합 테스트 | 모든 컴포넌트 통합 검증 |
| `test_integrated_system.py` | 통합 시스템 테스트 | 주요 시스템 간 통합 검증 |

---

### 📁 Category 4: 검색 시스템 테스트

검색 및 검색 관련 기능을 테스트하는 파일들입니다.

**디렉토리:** `tests/search/`

| 파일 | 목적 | 테스트 내용 |
|------|------|-------------|
| `test_query_classification.py` | 질문 분류 테스트 (통합) | 질문 유형 분류 기능 검증, classify_question_type 메서드 테스트 포함 |
| `test_query_system.py` | 쿼리 시스템 테스트 | 쿼리 처리 시스템 검증 |
| `test_hybrid_search.py` | 하이브리드 검색 단위 테스트 | 하이브리드 검색 기본 기능 |
| `test_hybrid_search_integration.py` | 하이브리드 검색 통합 테스트 | 하이브리드 검색 통합 검증 |
| `test_rag_integration.py` | RAG(Retrieval-Augmented Generation) 통합 테스트 | RAG 시스템 통합 검증 |
| `test_sql_router_*.py` | SQL 라우터 테스트 | SQL 라우터 보안 및 동작 검증 |

---

### 📁 Category 5: 법률 시스템 테스트

법률 관련 기능을 테스트하는 파일들입니다.

**디렉토리:** `tests/legal/`

| 파일 | 목적 | 테스트 내용 |
|------|------|-------------|
| `test_legal_basis_system.py` | 법적 근거 시스템 테스트 | 법적 근거 검증 시스템 |
| `test_database_keyword_system.py` | 데이터베이스 키워드 시스템 테스트 | 법률 용어 데이터베이스 검색 |
| `test_term_integration_workflow.py` | 용어 통합 워크플로우 테스트 | 법률 용어 통합 처리 |
| `test_akls_integration.py` | AKLS 통합 테스트 | 법률 용어 시스템 통합 검증 |

**참고:** `test_akls_integration.py`는 여러 위치에 존재합니다:
- **`tests/legal/`**: 법률 시스템 통합 테스트 (현재 위치)
- **`tests/akls/`**: AKLS 전용 디렉토리 테스트 (별도 파일)

---

### 📁 Category 6: 품질 및 성능 테스트

시스템 품질과 성능을 테스트하는 파일들입니다.

**디렉토리:** `tests/quality_performance/`

| 파일 | 목적 | 테스트 내용 |
|------|------|-------------|
| `test_quality_enhancement.py` | 품질 향상 테스트 | 답변 품질 향상 기능 검증 |
| `test_quality_improvement_workflow.py` | 품질 개선 워크플로우 테스트 | 데이터 품질 관리 시스템 테스트 |
| `test_performance_benchmark.py` | 성능 벤치마크 테스트 | 시스템 성능 측정 |
| `test_performance_monitor_fix.py` | 성능 모니터링 수정 테스트 | 성능 모니터링 수정 검증 |
| `test_optimized_performance.py` | 최적화 성능 테스트 | 최적화 후 성능 검증 |
| `test_stress_system.py` | 스트레스 테스트 | 고부하 상황 테스트 |
| `test_workflow_execution.py` | 워크플로우 실행 테스트 | 워크플로우 실행 성능 |

**참고:** `test_state_reduction_performance.py`는 LangGraph State 시스템의 성능 테스트이므로 Category 1 (`tests/langgraph/`)에 포함되어 있습니다.

---

### 📁 Category 7: 모니터링 및 통합 테스트

외부 도구 및 통합 기능을 테스트하는 파일들입니다.

**디렉토리:** `tests/monitoring/`

| 파일 | 목적 | 테스트 내용 |
|------|------|-------------|
| `test_unified_prompt_integration.py` | 통합 프롬프트 테스트 | UnifiedPromptManager 통합 검증 |
| `test_langsmith_integration.py` | LangSmith 통합 테스트 | LangSmith 모니터링 통합 검증 |
| `test_langfuse_integration.py` | Langfuse 통합 테스트 | Langfuse 모니터링 통합 검증 |

---

### 📁 Category 8: AKLS (법률 용어) 테스트

AKLS (법률 용어 시스템) 관련 테스트입니다.

**디렉토리:** `tests/akls/`

| 파일 | 목적 | 테스트 내용 |
|------|------|-------------|
| `test_akls_gradio.py` | AKLS Gradio 테스트 | Gradio 인터페이스 테스트 |
| `test_akls_integration.py` | AKLS 통합 테스트 (디렉토리 버전) | AKLS 시스템 통합 검증 |
| `test_akls_performance.py` | AKLS 성능 테스트 | AKLS 성능 측정 |

**참고:** `test_akls_integration.py`는 여러 위치에 존재합니다:
- **`tests/akls/`**: AKLS 전용 테스트 (함수 기반 통합 테스트)
- **`tests/legal/`**: 법률 시스템 통합 테스트 (AKLS 포함)

---

## 테스트 실행 방법

### 전체 테스트 실행

```bash
# 모든 테스트 실행
pytest tests/ -v

# 특정 카테고리 테스트 실행
pytest tests/langgraph/ -v  # LangGraph 테스트만
pytest tests/akls/ -v       # AKLS 테스트만
```

### LangGraph 테스트 실행

```bash
# LangGraph 전용 테스트 (모니터링 전환)
python tests/langgraph/test_with_monitoring_switch.py

# LangGraph 전체 테스트 (모니터링 전환 포함)
pytest tests/langgraph/ -v

# 기본 LangGraph 워크플로우 테스트
pytest tests/langgraph/test_langgraph.py -v

# State 시스템 테스트
pytest tests/langgraph/test_all_state_systems.py -v
pytest tests/langgraph/test_core_state_systems.py -v
pytest tests/langgraph/test_state_reduction_performance.py -v
```

### Phase별 테스트 실행

```bash
# Phase 1 테스트
pytest tests/phase/test_phase1_context_enhancement.py -v

# Phase 2 테스트
pytest tests/phase/test_phase2_personalization_analysis.py -v

# Phase 3 테스트
pytest tests/phase/test_phase3_memory_quality.py -v

# 모든 Phase 테스트
pytest tests/phase/ -v
```

### 통합 테스트 실행

```bash
# 통합 시스템 전체 테스트
pytest tests/integration/ -v

# 종합 시스템 테스트
pytest tests/integration/test_comprehensive_system.py -v

# 통합 시스템 테스트
pytest tests/integration/test_integrated_system.py -v
```

### 검색 시스템 테스트

```bash
# 검색 시스템 전체 테스트
pytest tests/search/ -v

# 질문 분류 테스트
pytest tests/search/test_query_classification.py -v
pytest tests/search/test_classify_question_type.py -v

# 하이브리드 검색 테스트
pytest tests/search/test_hybrid_search*.py -v

# RAG 통합 테스트
pytest tests/search/test_rag_integration.py -v
```

### 법률 시스템 테스트

```bash
# 법률 시스템 전체 테스트
pytest tests/legal/ -v

# 법적 근거 시스템 테스트
pytest tests/legal/test_legal_basis_system.py -v

# 데이터베이스 키워드 시스템 테스트
pytest tests/legal/test_database_keyword_system.py -v

# 용어 통합 워크플로우 테스트
pytest tests/legal/test_term_integration_workflow.py -v
```

### 모니터링 및 통합 테스트

```bash
# 모니터링 전체 테스트
pytest tests/monitoring/ -v

# LangSmith 통합 테스트
pytest tests/monitoring/test_langsmith_integration.py -v

# Langfuse 통합 테스트
pytest tests/monitoring/test_langfuse_integration.py -v

# 통합 프롬프트 테스트
pytest tests/monitoring/test_unified_prompt_integration.py -v
```

### 품질 및 성능 테스트

```bash
# 품질 및 성능 전체 테스트
pytest tests/quality_performance/ -v

# 품질 테스트
pytest tests/quality_performance/test_quality*.py -v

# 성능 벤치마크
pytest tests/quality_performance/test_performance_benchmark.py -v

# 최적화 성능 테스트
pytest tests/quality_performance/test_optimized_performance.py -v

# 스트레스 테스트
pytest tests/quality_performance/test_stress_system.py -v
```

### 마스터 테스트 실행

```bash
# 순차적으로 모든 테스트 실행
python tests/run_master_tests.py
```

### 테스트 구조 검증

```bash
# 테스트 구조 검증
python tests/verify_test_structure.py
```

## 테스트 환경 설정

테스트 환경은 `tests/conftest.py`에서 설정됩니다:

- 프로젝트 루트 경로 자동 추가
- 환경 변수 기본 설정
- LangGraph 모니터링 전환 픽스처
- 워크플로우 팩토리 픽스처

### 환경 변수

테스트 실행 전 필요한 환경 변수는 `.env` 파일에 설정되어 있어야 합니다.

## 테스트 작성 가이드

### 새로운 테스트 추가 시

1. **위치 결정**: 테스트 목적에 맞는 디렉토리에 배치
   - LangGraph 관련 (모든 종류): `tests/langgraph/`
   - Phase 테스트: `tests/phase/`
   - 통합 테스트: `tests/integration/`
   - 검색 시스템: `tests/search/`
   - 법률 시스템: `tests/legal/`
   - 모니터링: `tests/monitoring/`
   - 품질/성능: `tests/quality_performance/`
   - AKLS 관련: `tests/akls/`

2. **명명 규칙**: `test_*.py` 형식 준수
   - 파일명: `test_*.py`
   - 테스트 함수: `test_*`
   - 테스트 클래스: `Test*`

3. **픽스처 활용**: `conftest.py`의 공통 픽스처 활용

4. **문서화**: 파일 상단에 목적과 사용법 명시
   - 파일 상단에 목적과 사용법 명시
   - 주요 테스트 함수에 docstring 추가

5. **업데이트**: 이 문서에 새 테스트 파일 추가

### LangGraph 테스트 작성

LangGraph 테스트는 `tests/langgraph/` 디렉토리를 참고하세요.

- 모니터링 전환 유틸리티 활용
- 워크플로우 팩토리 사용
- State 시스템 테스트 패턴 준수

자세한 내용은 [LangGraph 테스트 README](./langgraph/README.md)를 참조하세요.

## 테스트 실행 우선순위

### 1. 필수 테스트 (CI/CD)

- LangGraph 기본 테스트: `tests/langgraph/test_langgraph.py`
- State 시스템 테스트: `tests/langgraph/test_all_state_systems.py`, `tests/langgraph/test_core_state_systems.py`
- 통합 테스트: `tests/integration/test_comprehensive_system.py`

### 2. 개발 단계별 테스트

- Phase 1-3 테스트: `tests/phase/test_phase*.py`
- 기능별 테스트: 각 카테고리별 테스트 디렉토리

### 3. 성능 및 스트레스 테스트

- 성능 테스트: `tests/quality_performance/test_performance*.py`
- 스트레스 테스트: `tests/quality_performance/test_stress_system.py`

## 테스트 결과 확인

### Pytest 결과

```bash
# 상세 출력
pytest tests/ -v

# 커버리지 포함
pytest tests/ --cov=. --cov-report=html
```

### 마스터 테스트 결과

```bash
python tests/run_master_tests.py
# 실행 결과가 콘솔에 출력됩니다
```

## 문제 해결

### 일반적인 문제

1. **Import 오류**
   - 프로젝트 루트가 `sys.path`에 추가되어 있는지 확인
   - `conftest.py`가 올바르게 로드되는지 확인

2. **환경 변수 오류**
   - `.env` 파일이 존재하고 필요한 변수가 설정되어 있는지 확인

3. **LangGraph 컴파일 오류**
   - LangGraph 의존성이 설치되어 있는지 확인
   - 워크플로우 설정이 올바른지 확인

### 디버깅

```bash
# 특정 테스트만 실행
pytest tests/langgraph/test_langgraph.py::test_specific_function -v

# 디버그 모드
pytest tests/ -v --pdb
```

## 테스트 유지보수

### 정기 점검 사항

1. **레거시 테스트 확인**
   - 오래된 경로 참조 확인
   - 중복 테스트 확인
   - 더 이상 사용하지 않는 테스트 식별

2. **성능 테스트 업데이트**
   - 기준값(baseline) 업데이트
   - 새로운 최적화 반영

3. **통합 테스트 검증**
   - 시스템 변경사항 반영
   - 새로운 컴포넌트 통합 검증

## 테스트 파일 통계

- **전체 테스트 파일 수**: 약 47개 (2025-01 정리 기준)
  - **LangGraph 관련**: 약 21개 (`tests/langgraph/`)
    - 기본 워크플로우: `test_langgraph.py`, `test_langgraph_with_logging.py`, `test_langgraph_multi_turn.py`
    - 최적화: `test_optimized_workflow.py` (통합됨)
    - 노드 통합: `test_node_integration.py` (통합됨)
    - State 시스템: `test_all_state_systems.py`, `test_core_state_systems.py`, `test_state_*.py`
    - 모니터링: `test_monitoring_switch_basic.py`, `test_with_monitoring_switch.py`, `test_profile_loading.py`
    - 기타: `test_all_scenarios.py`, `test_*.py`
  - **Phase 테스트**: 3개 (`tests/phase/`)
  - **통합 시스템**: 2개 (`tests/integration/`)
  - **검색 시스템**: 약 8개 (`tests/search/`)
    - 질의 분류: `test_query_classification.py` (통합됨 - classify_question_type 포함)
    - 하이브리드 검색: `test_hybrid_search.py`, `test_hybrid_search_integration.py`
    - SQL 라우터: `test_sql_router_*.py`
    - RAG: `test_rag_integration.py`
    - 기타: `test_query_system.py`
  - **법률 시스템**: 3개 (`tests/legal/`)
  - **모니터링**: 7개 (`tests/monitoring/`)
  - **품질/성능**: 7개 (`tests/quality_performance/`)
  - **단위 테스트**: 2개 (`tests/unit/`)
  - **서비스**: 1개 (`tests/services/`)

## 관련 문서

- [LangGraph 테스트 가이드](./langgraph/README.md)
- [프로젝트 메인 README](../README.md)

## 업데이트 이력

- **2025-01**: 테스트 파일 정리 및 통합
  - 중복 테스트 파일 통합
    - `test_optimized_workflow_simple.py` → `test_optimized_workflow.py`에 통합
    - `test_node_integration_simple.py` → `test_node_integration.py`에 통합
    - `test_classify_question_type.py` → `test_query_classification.py`에 통합
    - `test_hybrid_search_simple.py` 삭제 (통합 버전 유지)
    - `test_moderate_query.py` 삭제 (중복)
  - 불필요한 파일 제거 및 코드 중복 제거
  - README 업데이트 및 통계 갱신

- **2025-01 (이전)**: 테스트 구조 재구성 및 문서화
  - 레거시 테스트 파일 삭제
  - 루트 레벨 테스트 파일을 카테고리별 디렉토리로 이동
  - 31개 파일을 7개 디렉토리로 재구성
  - Import 경로 자동 수정
  - 테스트 분류 체계 구축 완료
  - 실제 폴더 구조에 맞게 문서 업데이트
  - 모든 테스트 파일 목록 정리
  - 구조 검증 스크립트 추가
  - **문서 통합**: TEST_ORGANIZATION.md 내용을 README.md에 통합하여 단일 문서로 개선
