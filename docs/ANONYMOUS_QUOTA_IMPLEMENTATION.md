# 익명 사용자 3회 질의 제한 구현 완료

## 구현 완료 항목

### 1. AnonymousQuotaService 생성
- **파일**: `api/services/anonymous_quota_service.py`
- IP 주소 기반 질의 횟수 추적
- 일일 리셋 기능 (자정 기준)
- 메모리 기반 저장 (향후 Redis로 확장 가능)

### 2. 설정 추가
- **파일**: `api/config.py`
- `anonymous_quota_enabled: bool = True`
- `anonymous_quota_limit: int = 3`
- `anonymous_quota_reset_hour: int = 0`

### 3. 인증 미들웨어 수정
- **파일**: `api/middleware/auth_middleware.py`
- `require_auth` 함수에서 익명 사용자 제한 확인
- 3회 이하: 익명 사용자로 허용
- 3회 초과: 429 에러 반환

### 4. Chat 엔드포인트 수정
- **파일**: `api/routers/chat.py`
- 익명 사용자 응답에 `X-Quota-Remaining` 헤더 추가

### 5. 프론트엔드 수정
- **파일**: `frontend/src/services/api.ts`
- 429 에러 처리 추가
- 익명 사용자 제한 초과 시 로그인 유도 메시지

### 6. 환경 변수 예제 업데이트
- **파일**: `api/.env.example`
- 익명 사용자 제한 설정 추가

## 동작 방식

1. **익명 사용자 질의 시도**
   - `require_auth`에서 IP 주소 확인
   - `AnonymousQuotaService`로 질의 가능 여부 확인
   - 3회 이하: 질의 횟수 증가 후 허용
   - 3회 초과: 429 에러 반환

2. **일일 리셋**
   - 날짜 변경 시 자동 리셋
   - 각 IP 주소별로 독립적으로 관리

3. **인증된 사용자**
   - 로그인한 사용자는 제한 없음
   - 무제한 질의 가능

## 테스트

- **파일**: `api/test/test_anonymous_quota.py`
- 익명 사용자 제한 서비스 테스트
- 엔드포인트 테스트 (선택적)

## 주의사항

- IP 기반 추적은 프록시/공유 IP 환경에서 제한적
- 메모리 기반 저장은 서버 재시작 시 카운터 리셋
- 프로덕션 환경에서는 Redis 기반 저장 권장

