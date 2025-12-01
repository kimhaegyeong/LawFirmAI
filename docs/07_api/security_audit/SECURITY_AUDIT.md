# 보안 감사 가이드

## 의존성 보안 스캔

### 1. pip-audit 사용 (권장)

```bash
# pip-audit 설치
pip install pip-audit

# 보안 스캔 실행
pip-audit -r api/requirements.txt

# JSON 형식으로 출력
pip-audit -r api/requirements.txt --format json

# 취약점이 발견되면 자동으로 수정 가능한 경우 수정
pip-audit -r api/requirements.txt --fix
```

### 2. safety 사용

```bash
# safety 설치
pip install safety

# 보안 스캔 실행
safety check -r api/requirements.txt

# JSON 형식으로 출력
safety check -r api/requirements.txt --json
```

### 3. 정기적 스캔

보안 스캔은 다음 시점에 실행해야 합니다:
- 새로운 의존성 추가 시
- 주기적으로 (월 1회 권장)
- 프로덕션 배포 전

### 4. 자동화 스크립트

CI/CD 파이프라인에 보안 스캔을 추가하는 것을 권장합니다.

```yaml
# .github/workflows/security-audit.yml 예시
name: Security Audit
on:
  schedule:
    - cron: '0 0 1 * *'  # 매월 1일
  workflow_dispatch:

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install pip-audit
        run: pip install pip-audit
      - name: Run security audit
        run: pip-audit -r api/requirements.txt
```

## 알려진 취약점 대응

1. **심각도가 높은 취약점 발견 시**
   - 즉시 해당 패키지 업데이트
   - 대안 패키지 검토
   - 임시 패치 적용 고려

2. **의존성 업데이트 시 주의사항**
   - 호환성 테스트 필수
   - 변경 로그 확인
   - 단계적 업데이트 (한 번에 하나씩)

3. **업데이트 불가능한 경우**
   - 대안 패키지 검토
   - 취약점 완화 조치 적용
   - 모니터링 강화

