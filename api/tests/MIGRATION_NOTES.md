# 테스트 파일 마이그레이션 노트

## 리팩토링 완료된 파일

다음 파일들은 새로운 구조로 리팩토링되어 `unit/`, `integration/`, `e2e/` 디렉토리로 이동되었습니다.

### 단위 테스트 (unit/)
- ✅ `test_schemas_health.py` → `unit/test_schemas_health.py`
- ✅ `test_schemas_session.py` → `unit/test_schemas_session.py`
- ✅ `test_services_answer_splitter.py` → `unit/test_services_answer_splitter.py`
- ✅ `test_utils_sse_formatter.py` → `unit/test_utils_sse_formatter.py`
- ✅ `test_middleware_rate_limit.py` → `unit/test_middleware_rate_limit.py`
- ✅ `test_middleware_csrf.py` → `unit/test_middleware_csrf.py`
- ✅ `test_middleware_error_handler.py` → `unit/test_middleware_error_handler.py`
- ✅ `test_middleware_security_headers.py` → `unit/test_middleware_security_headers.py`
- ✅ `test_config.py` → `unit/test_config.py`

### 통합 테스트 (integration/)
- ✅ `test_api_integration.py` → `integration/test_api_integration.py`
- ✅ `test_routers_health.py` → `integration/test_routers_health.py`
- ✅ `test_database_connection.py` → `integration/test_database_connection.py`
- ✅ `test_database_models.py` → `integration/test_database_models.py`
- ✅ `test_security.py` → `integration/test_security.py`
- ✅ `test_oauth2_auth.py` → `integration/test_oauth2_auth.py`
- ✅ `test_stream_api.py` → `integration/test_stream_api.py` (새로 생성)
- ✅ `test_anonymous_quota.py` → `integration/test_anonymous_quota.py`
- ✅ `test_suggested_questions.py` → `integration/test_suggested_questions.py`
- ✅ `test_integration.py` → `integration/test_integration_full.py` (새로 생성)

### E2E 테스트 (e2e/)
- ✅ `test_chat_api_with_improvements.py` → `e2e/test_chat_api_flow.py` (새로 생성)

## 기존 파일 상태

루트 디렉토리에 남아있는 파일들은 다음과 같습니다:

### 리팩토링 완료 (기존 파일은 유지 - 하위 호환성)
다음 파일들은 리팩토링되었지만, 기존 파일도 유지되어 있습니다 (하위 호환성):
- `test_api_integration.py` (리팩토링: `integration/test_api_integration.py`)
- `test_routers_health.py` (리팩토링: `integration/test_routers_health.py`)
- `test_database_connection.py` (리팩토링: `integration/test_database_connection.py`)
- `test_database_models.py` (리팩토링: `integration/test_database_models.py`)
- `test_security.py` (리팩토링: `integration/test_security.py`)
- `test_oauth2_auth.py` (리팩토링: `integration/test_oauth2_auth.py`)
- `test_schemas_health.py` (리팩토링: `unit/test_schemas_health.py`)
- `test_schemas_session.py` (리팩토링: `unit/test_schemas_session.py`)
- `test_services_answer_splitter.py` (리팩토링: `unit/test_services_answer_splitter.py`)
- `test_utils_sse_formatter.py` (리팩토링: `unit/test_utils_sse_formatter.py`)
- `test_middleware_*.py` (리팩토링: `unit/test_middleware_*.py`)
- `test_config.py` (리팩토링: `unit/test_config.py`)

### 추가로 리팩토링 완료된 파일
- ✅ `test_stream_handler.py` → `integration/test_stream_handler.py`
- ✅ `test_stream_cache.py` → `unit/test_stream_cache.py`
- ✅ `test_stream_cache_integration.py` → `integration/test_stream_cache_integration.py`
- ✅ `test_sources_unification.py` → `integration/test_sources_unification.py`
- ✅ `test_sources_enhancement.py` → `integration/test_sources_enhancement.py`
- ✅ `test_sources_by_type_in_stream.py` → `integration/test_sources_by_type_in_stream.py`
- ✅ `test_security_validation.py` → `integration/test_security_validation.py`

### 아직 리팩토링되지 않은 파일 (선택사항)
다음 파일들은 스크립트 형태이거나 특수한 용도로 사용되므로 리팩토링이 선택사항입니다:
- `test_chat_api_response.py` - 스크립트 형태 (E2E 테스트로 리팩토링 가능)
- `test_stream_simple.py` - 스크립트 형태 (통합 테스트로 리팩토링 가능)
- `test_security_manual.py` - 수동 테스트 스크립트

## 사용 방법

### 새로운 테스트 작성 시
1. 적절한 디렉토리 선택:
   - 단위 테스트: `unit/`
   - 통합 테스트: `integration/`
   - E2E 테스트: `e2e/`

2. `conftest.py`의 fixture 사용:
   ```python
   def test_example(self, client):  # conftest.py의 client fixture
       response = client.get("/health")
   ```

3. 헬퍼 함수 활용:
   ```python
   from api.test.helpers.client_helpers import make_chat_request
   response = make_chat_request(client, "test message")
   ```

### 테스트 실행
```bash
# 전체 테스트
pytest api/test

# 단위 테스트만
pytest api/test/unit

# 통합 테스트만
pytest api/test/integration

# E2E 테스트만
pytest api/test/e2e
```

## 마이그레이션 가이드

기존 테스트 파일을 리팩토링할 때:

1. 중복 코드 제거:
   - `project_root`, `sys.path` 설정 제거
   - 중복된 fixture 제거

2. `conftest.py` 사용:
   - `client` fixture는 `conftest.py`에서 제공
   - `mock_auth_disabled`, `mock_auth_enabled` 등도 `conftest.py`에서 제공

3. 헬퍼 함수 사용:
   - `make_chat_request()`: 채팅 요청
   - `make_stream_request()`: 스트리밍 요청
   - `wait_for_server()`: 서버 대기
   - `check_server_health()`: 서버 상태 확인

4. 적절한 디렉토리로 이동:
   - 단위 테스트 → `unit/`
   - 통합 테스트 → `integration/`
   - E2E 테스트 → `e2e/`

