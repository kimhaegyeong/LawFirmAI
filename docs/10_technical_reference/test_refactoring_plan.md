# API 테스트 리팩토링 계획

## 📋 목차

1. [현재 상태 분석](#현재-상태-분석)
2. [리팩토링 목표](#리팩토링-목표)
3. [제안하는 구조](#제안하는-구조)
4. [구체적인 리팩토링 계획](#구체적인-리팩토링-계획)
5. [단계별 마이그레이션 계획](#단계별-마이그레이션-계획)
6. [예상 효과](#예상-효과)

---

## 현재 상태 분석

### 주요 문제점

#### 1. 중복 코드
- **30개 이상의 파일**에서 `project_root`, `sys.path` 설정이 반복됨
- `TestClient` fixture가 여러 파일에 중복 정의
- `from api.main import app` 패턴이 반복됨

#### 2. 구조적 문제
- 공통 fixture 파일(`conftest.py`)이 없음
- 단위 테스트와 통합 테스트가 혼재되어 있음
- 테스트 파일들이 모두 루트에 평평하게 배치됨
- 테스트 카테고리 구분이 명확하지 않음

#### 3. 실행 스크립트 불일치
- 여러 실행 스크립트가 서로 다른 방식으로 동작
- 일관성 없는 테스트 실행 방법

---

## 리팩토링 목표

1. **중복 코드 제거**: 공통 설정 및 fixture를 한 곳에 모음
2. **구조 명확화**: 단위/통합/E2E 테스트를 명확히 구분
3. **유지보수성 향상**: 공통 설정 변경 시 한 곳만 수정
4. **일관성 확보**: 표준화된 테스트 실행 방법

---

## 제안하는 구조

```
api/test/
├── conftest.py                    # 공통 fixture 및 설정
├── pytest.ini                     # pytest 설정
│
├── unit/                          # 단위 테스트
│   ├── __init__.py
│   ├── test_schemas_*.py          # 스키마 테스트
│   ├── test_services_*.py          # 서비스 테스트
│   ├── test_utils_*.py             # 유틸리티 테스트
│   └── test_middleware_*.py       # 미들웨어 테스트
│
├── integration/                   # 통합 테스트
│   ├── __init__.py
│   ├── test_api_*.py              # API 통합 테스트
│   ├── test_stream_*.py           # 스트리밍 테스트
│   ├── test_database_*.py         # 데이터베이스 테스트
│   └── test_security_*.py         # 보안 통합 테스트
│
├── e2e/                           # End-to-End 테스트
│   ├── __init__.py
│   ├── test_chat_flow.py          # 채팅 플로우 테스트
│   └── test_oauth_flow.py         # OAuth 플로우 테스트
│
├── fixtures/                      # 테스트 데이터 및 fixture
│   ├── __init__.py
│   ├── auth_fixtures.py           # 인증 관련 fixture
│   ├── database_fixtures.py       # DB fixture
│   └── mock_data.py               # Mock 데이터
│
├── helpers/                       # 테스트 헬퍼 함수
│   ├── __init__.py
│   ├── client_helpers.py          # 클라이언트 헬퍼
│   └── server_helpers.py          # 서버 헬퍼
│
└── scripts/                       # 테스트 실행 스크립트
    ├── run_all_tests.py           # 전체 테스트 실행
    ├── run_unit_tests.py          # 단위 테스트만
    ├── run_integration_tests.py   # 통합 테스트만
    └── run_with_server.py         # 서버와 함께 실행
```

---

## 구체적인 리팩토링 계획

### 1. 공통 conftest.py 생성

**목적**: 모든 테스트에서 공통으로 사용하는 fixture와 설정을 한 곳에 모음

**주요 내용**:
- 프로젝트 경로 설정 (한 번만)
- `TestClient` fixture
- 인증 관련 mock fixture
- Rate limit 관련 mock fixture

### 2. 테스트 헬퍼 함수 생성

**목적**: 반복되는 테스트 패턴을 헬퍼 함수로 추출

**주요 헬퍼**:
- `create_test_client()`: 테스트 클라이언트 생성
- `make_chat_request()`: 채팅 요청 헬퍼
- `make_stream_request()`: 스트리밍 요청 헬퍼
- `wait_for_server()`: 서버 대기 헬퍼
- `check_server_health()`: 서버 상태 확인

### 3. 디렉토리 재구성

**단위 테스트 (`unit/`)**:
- 스키마 검증 테스트
- 서비스 로직 테스트
- 유틸리티 함수 테스트
- 미들웨어 단위 테스트

**통합 테스트 (`integration/`)**:
- API 엔드포인트 통합 테스트
- 스트리밍 기능 통합 테스트
- 데이터베이스 통합 테스트
- 보안 기능 통합 테스트

**E2E 테스트 (`e2e/`)**:
- 전체 사용자 플로우 테스트
- OAuth 인증 플로우 테스트

### 4. 테스트 파일 리팩토링

**변경 사항**:
- 중복된 경로 설정 코드 제거
- `conftest.py`의 fixture 사용
- 헬퍼 함수 활용
- 명확한 테스트 카테고리 분류

### 5. pytest.ini 업데이트

**변경 사항**:
- `testpaths` 설정으로 테스트 경로 명시
- 마커 추가 (unit, integration, e2e, slow)

---

## 단계별 마이그레이션 계획

### Phase 1: 기반 구조 생성 ✅
- [x] `conftest.py` 생성 및 공통 fixture 이동
- [x] `helpers/` 디렉토리 생성 및 헬퍼 함수 추출
- [x] `pytest.ini` 업데이트

### Phase 2: 디렉토리 재구성 ✅
- [x] `unit/`, `integration/`, `e2e/` 디렉토리 생성
- [x] 각 디렉토리에 `__init__.py` 추가
- [x] 예시 테스트 파일을 카테고리별로 이동 및 리팩토링

### Phase 3: 테스트 파일 리팩토링 ✅
- [x] 예시 테스트 파일에서 중복 코드 제거
- [x] `conftest.py`의 fixture 사용
- [x] 헬퍼 함수 활용
- [x] 단위 테스트 파일 리팩토링 완료
  - [x] test_schemas_*.py → unit/
  - [x] test_services_*.py → unit/
  - [x] test_utils_*.py → unit/
  - [x] test_middleware_*.py → unit/
  - [x] test_config.py → unit/
- [x] 통합 테스트 파일 리팩토링 완료
  - [x] test_api_integration.py → integration/
  - [x] test_routers_*.py → integration/
  - [x] test_database_*.py → integration/
  - [x] test_security.py → integration/
  - [x] test_oauth2_*.py → integration/
- [x] E2E 테스트 파일 리팩토링
  - [x] test_chat_api_with_improvements.py → e2e/test_chat_api_flow.py
- [x] 추가 통합 테스트 파일 리팩토링
  - [x] test_stream_api.py → integration/test_stream_api.py
  - [x] test_anonymous_quota.py → integration/test_anonymous_quota.py
  - [x] test_suggested_questions.py → integration/test_suggested_questions.py
  - [x] test_integration.py → integration/test_integration_full.py
- [x] 나머지 스트리밍 테스트 파일 리팩토링
  - [x] test_stream_handler.py → integration/test_stream_handler.py
  - [x] test_stream_cache.py → unit/test_stream_cache.py
  - [x] test_stream_cache_integration.py → integration/test_stream_cache_integration.py
  - [x] test_sources_unification.py → integration/test_sources_unification.py
  - [x] test_sources_enhancement.py → integration/test_sources_enhancement.py
  - [x] test_sources_by_type_in_stream.py → integration/test_sources_by_type_in_stream.py
  - [x] test_security_validation.py → integration/test_security_validation.py
  - [ ] test_stream_simple.py (선택사항 - 스크립트 형태)

### Phase 4: 실행 스크립트 통합 ✅
- [x] `scripts/` 디렉토리 생성
- [x] 실행 스크립트 통합 및 표준화
- [ ] 배치 파일 정리 (선택사항)

---

## 예상 효과

### 중복 코드 제거
- **30개 이상의 파일**에서 중복된 경로 설정 코드 제거
- 공통 fixture 중복 정의 제거

### 유지보수성 향상
- 공통 설정 변경 시 **한 곳만 수정**하면 됨
- 테스트 구조가 명확해져 **새 테스트 작성이 쉬워짐**

### 테스트 구조 명확화
- 단위/통합/E2E 테스트가 **명확히 구분**됨
- 테스트 실행 범위를 **선택적으로 지정** 가능

### 실행 일관성
- 표준화된 테스트 실행 스크립트
- 일관된 테스트 실행 방법

---

## 주의사항

1. **기존 파일 수정 최소화**: 점진적 마이그레이션으로 기존 테스트 동작 유지
2. **프로젝트 규칙 준수**: `docs/11.cursor_rules/`의 규칙 준수
3. **테스트 동작 보장**: 리팩토링 후에도 모든 테스트가 동일하게 동작해야 함

---

## 최종 완료 상태

### ✅ Phase 1-4 모두 완료

- [x] Phase 1: 기반 구조 생성
- [x] Phase 2: 디렉토리 재구성
- [x] Phase 3: 테스트 파일 리팩토링
- [x] Phase 4: 실행 스크립트 통합

## 구현 완료 사항

### 생성된 파일

1. **`api/test/conftest.py`**: 공통 fixture 및 설정
   - `client`: TestClient fixture
   - `mock_auth_disabled`: 인증 비활성화 모킹
   - `mock_auth_enabled`: 인증 활성화 모킹
   - `mock_rate_limit_disabled`: Rate limit 비활성화 모킹
   - `mock_rate_limit_enabled`: Rate limit 활성화 모킹

2. **`api/test/helpers/client_helpers.py`**: 클라이언트 헬퍼 함수
   - `create_test_client()`: 테스트 클라이언트 생성
   - `make_chat_request()`: 채팅 요청 헬퍼
   - `make_stream_request()`: 스트리밍 요청 헬퍼

3. **`api/test/helpers/server_helpers.py`**: 서버 헬퍼 함수
   - `wait_for_server()`: 서버 대기
   - `check_server_health()`: 서버 상태 확인

4. **리팩토링된 테스트 파일**:
   
   **단위 테스트 (unit/)**:
   - `test_schemas_health.py`: 헬스체크 스키마 테스트
   - `test_schemas_session.py`: 세션 스키마 테스트
   - `test_services_answer_splitter.py`: 답변 분할 서비스 테스트
   - `test_utils_sse_formatter.py`: SSE 포맷터 유틸리티 테스트
   - `test_middleware_rate_limit.py`: Rate limit 미들웨어 테스트
   - `test_middleware_csrf.py`: CSRF 보호 미들웨어 테스트
   - `test_middleware_error_handler.py`: 에러 핸들러 미들웨어 테스트
   - `test_middleware_security_headers.py`: 보안 헤더 미들웨어 테스트
   - `test_config.py`: API 설정 테스트
   - `test_stream_cache.py`: 스트리밍 캐시 단위 테스트
   
   **통합 테스트 (integration/)**:
   - `test_api_integration.py`: API 통합 테스트
   - `test_routers_health.py`: 헬스체크 라우터 테스트
   - `test_database_connection.py`: 데이터베이스 연결 테스트
   - `test_database_models.py`: 데이터베이스 모델 테스트
   - `test_security.py`: 보안 기능 통합 테스트
   - `test_oauth2_auth.py`: OAuth2 인증 플로우 테스트
   - `test_stream_api.py`: 스트리밍 API 통합 테스트
   - `test_stream_handler.py`: StreamHandler 통합 테스트
   - `test_stream_cache_integration.py`: 스트리밍 캐시 통합 테스트
   - `test_anonymous_quota.py`: 익명 사용자 질의 제한 테스트
   - `test_suggested_questions.py`: 추천 질문 기능 테스트
   - `test_integration_full.py`: 전체 통합 테스트
   - `test_sources_unification.py`: Sources 통일 기능 테스트
   - `test_sources_enhancement.py`: Sources 데이터 개선 로직 테스트
   - `test_sources_by_type_in_stream.py`: 스트리밍 API에서 sources_by_type 포함 여부 테스트
   - `test_security_validation.py`: 보안 검증 테스트
   
   **E2E 테스트 (e2e/)**:
   - `test_chat_api_flow.py`: Chat API 전체 플로우 테스트

5. **실행 스크립트**:
   - `api/test/scripts/run_all_tests.py`: 전체 테스트 실행
   - `api/test/scripts/run_unit_tests.py`: 단위 테스트만 실행
   - `api/test/scripts/run_integration_tests.py`: 통합 테스트만 실행

6. **마이그레이션 문서**:
   - `api/test/MIGRATION_NOTES.md`: 마이그레이션 가이드 및 파일 상태

### 사용 방법

#### 새로운 테스트 파일 작성 시

```python
"""
테스트 파일 예시
"""
import pytest
from api.test.helpers.client_helpers import make_chat_request


class TestExample:
    """예시 테스트"""
    
    def test_example(self, client):
        """예시 테스트"""
        # conftest.py의 client fixture 사용
        response = make_chat_request(client, "test message")
        assert response.status_code == 200
```

#### 기존 테스트 파일 리팩토링 시

1. 중복된 경로 설정 코드 제거:
   ```python
   # ❌ 제거할 코드
   project_root = Path(__file__).parent.parent.parent
   sys.path.insert(0, str(project_root))
   ```

2. 중복된 fixture 제거:
   ```python
   # ❌ 제거할 코드
   @pytest.fixture
   def client():
       return TestClient(app)
   ```

3. conftest.py의 fixture 사용:
   ```python
   # ✅ 사용할 방법
   def test_example(self, client):  # conftest.py의 client fixture
       response = client.get("/health")
   ```

## 참고 파일

- `docs/11.cursor_rules/06_testing_rules.md`: 테스트 규칙
- `api/test/pytest.ini`: pytest 설정
- `api/test/conftest.py`: 공통 fixture
- `api/test/helpers/`: 헬퍼 함수 모듈

