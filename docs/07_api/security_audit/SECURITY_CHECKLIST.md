# FastAPI 보안 점검 계획서

## 📋 개요

이 문서는 LawFirmAI FastAPI 백엔드의 보안 점검 계획을 정리한 것입니다.
보안 체크리스트를 기반으로 현재 상태를 점검하고 개선 사항을 식별합니다.

**작성일**: 2024년
**대상**: `api/` 디렉토리 전체
**점검 기준**: FastAPI 보안 모범 사례

---

## 🔍 1. 인증(Authentication) & 인가(Authorization)

### 1.1 JWT 토큰 서명 키 관리

| 항목 | 상태 | 비고 |
|------|------|------|
| ✅ **JWT_SECRET_KEY 환경 변수 분리** | ✅ 구현됨 | `.env` 파일 사용 중, Git 노출 방지 확인 완료 |
| ✅ **Secrets Manager 사용** | ❌ 미구현 | 프로덕션 환경에서 Secrets Manager 도입 검토 |
| ✅ **Git에 노출 금지** | ✅ 구현됨 | `.gitignore`에 `.env` 포함 확인 완료 |

**현재 상태**:
- `api/services/auth_service.py`에서 `JWT_SECRET_KEY` 환경 변수 사용
- `api/config.py`에서 `jwt_secret_key` 설정 지원
- `.env` 파일 사용 중

**개선 사항**:
- [x] `.gitignore`에 `.env` 파일 포함 확인 ✅ (점검 완료)
- [ ] 프로덕션 환경에서 Secrets Manager (AWS Secrets Manager, Azure Key Vault 등) 도입 검토
- [ ] JWT_SECRET_KEY가 설정되지 않은 경우 명확한 에러 메시지 제공

**점검 방법**:
```bash
# .gitignore 확인
grep -r "\.env" .gitignore

# 환경 변수 노출 확인
git log --all --full-history -- "**/.env"
```

---

### 1.2 JWT 만료시간 설정

| 항목 | 상태 | 비고 |
|------|------|------|
| ✅ **access_token 짧게 설정 (15~30분)** | ⚠️ 개선 필요 | 현재 24시간으로 설정됨 |
| ✅ **refresh_token 길게 설정** | ❌ 미구현 | refresh_token 기능 미구현 |

**현재 상태**:
- `api/config.py`: `jwt_expiration_hours: int = 24` (기본값 24시간)
- `api/services/auth_service.py`: `expiration_hours` 사용

**개선 사항**:
- [ ] access_token 만료시간을 15~30분으로 단축 ⚠️ **즉시 조치 필요** (현재 24시간)
- [ ] refresh_token 기능 구현 (만료시간 7일~30일) ⚠️ **단기 개선 필요**
- [ ] 토큰 갱신 엔드포인트 추가

**권장 설정**:
```python
# config.py
jwt_access_token_expiration_minutes: int = 30  # 30분
jwt_refresh_token_expiration_days: int = 7     # 7일
```

---

## 📝 점검 체크리스트 요약

### 즉시 점검 필요 (High Priority)

- [x] `.gitignore`에 `.env` 파일 포함 확인 ✅ (점검 완료)
- [ ] JWT access_token 만료시간 단축 (24시간 → 30분) ⚠️ **즉시 조치 필요**
- [ ] refresh_token 기능 구현 ⚠️ **단기 개선 필요**
- [ ] 프로덕션 환경에서 CSRF 보호 활성화 ⚠️ **단기 개선 필요**
- [ ] Health Check 엔드포인트 보호 (IP 제한 또는 인증) ⚠️ **단기 개선 필요**
- [ ] 프로덕션 환경에서 API 문서 (`/docs`, `/redoc`) 비활성화 또는 보호 ⚠️ **즉시 조치 필요**
- [x] `pip-audit` 실행하여 의존성 취약점 확인 ✅ (점검 완료 - 4개 취약점 발견)
- [x] 로그에서 민감정보 마스킹 확인 ✅ (점검 완료)

### 중기 개선 사항 (Medium Priority)

- [ ] 역할(Role) 기반 접근 제어 구현 ⚠️ **중기 개선 필요**
- [x] 파일 업로드 검증 강화 (크기, 확장자, MIME 타입) ✅ (점검 완료 - 이미 강력하게 구현됨)
- [ ] Redis 기반 Rate Limiting 도입
- [ ] Gunicorn + Uvicorn Workers 조합 사용
- [x] 전역 예외 핸들러 구현 ✅ (점검 완료 - `error_handler.py` 구현됨)
- [ ] 보안 이벤트 로깅 강화

### 장기 개선 사항 (Low Priority)

- [ ] OAuth2/OpenID Connect 지원
- [ ] Secrets Manager 도입
- [ ] Sentry 등 모니터링 도구 연동
- [ ] Bandit 정적 분석 도구 도입
- [ ] Trivy 컨테이너 스캔 도구 도입

---

## 📚 참고 자료

- [FastAPI Security Documentation](https://fastapi.tiangolo.com/tutorial/security/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security.html)
- [보안 감사 가이드](./SECURITY_AUDIT.md): 의존성 보안 스캔 가이드
- [보안 감사 결과](./security_audit/AUDIT_RESULT.md): 보안 감사 결과 보고서

---

**마지막 업데이트**: 2025년 11월 10일
**다음 점검 예정일**: 2025년 12월 1일 (매월 1일)

