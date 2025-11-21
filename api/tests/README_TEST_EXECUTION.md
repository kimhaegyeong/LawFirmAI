# 테스트 실행 가이드

## Windows 환경에서 테스트 실행 방법

### ⚠️ 중요: pytest 버퍼 문제 해결

Windows 환경에서 pytest를 실행할 때 버퍼 문제가 발생할 수 있습니다. 이를 해결하기 위해 다음 방법을 사용하세요.

### 방법 1: api/tests 디렉토리에서 실행 (권장)

```powershell
# api/tests 디렉토리로 이동
cd api\tests

# pytest 직접 실행 (api/tests/pytest.ini 사용)
$env:PYTHONUNBUFFERED = "1"
pytest -v
```

### 방법 2: 프로젝트 루트에서 실행

```powershell
# 프로젝트 루트에서
$env:PYTHONUNBUFFERED = "1"
pytest api/tests -v -s --capture=no
```

### 방법 3: pytest.ini 명시적 지정

```powershell
# api/tests 디렉토리에서
cd api\tests
$env:PYTHONUNBUFFERED = "1"
pytest -c pytest.ini -v
```

## 테스트 카테고리별 실행

### 단위 테스트만
```powershell
cd api\tests
$env:PYTHONUNBUFFERED = "1"
pytest -c pytest.ini -m unit unit -v
```

### 통합 테스트만
```powershell
cd api\tests
$env:PYTHONUNBUFFERED = "1"
pytest -c pytest.ini -m integration integration -v
```

### E2E 테스트만
```powershell
cd api\tests
$env:PYTHONUNBUFFERED = "1"
pytest -c pytest.ini -m e2e e2e -v
```

## 문제 해결

### pytest 버퍼 문제 (ValueError: underlying buffer has been detached)

**해결 방법:**

1. **환경 변수 설정:**
   ```powershell
   $env:PYTHONUNBUFFERED = "1"
   ```

2. **pytest 옵션 추가:**
   ```powershell
   pytest -v -s --capture=no
   ```

3. **api/tests/pytest.ini 사용:**
   ```powershell
   cd api\tests
   pytest -c pytest.ini -v
   ```

### 커버리지 옵션 오류

프로젝트 루트의 `pytest.ini`에 커버리지 옵션이 있는데 `pytest-cov`가 설치되지 않은 경우:

```powershell
# pytest-cov 설치
pip install pytest-cov

# 또는 커버리지 옵션 없이 실행
pytest api/tests -v -s --capture=no --override-ini="addopts=-v --strict-markers --tb=short --disable-warnings -ra -s --capture=no"
```

### 설정 파일 충돌

프로젝트 루트와 `api/tests`에 각각 `pytest.ini`가 있는 경우:

- `api/tests`에서 실행하면 `api/tests/pytest.ini`가 우선 적용됩니다
- 명시적으로 지정하려면: `pytest -c pytest.ini`

## 권장 실행 방법

**가장 안정적인 방법:**

```powershell
# 1. api/tests 디렉토리로 이동
cd api\tests

# 2. 환경 변수 설정
$env:PYTHONUNBUFFERED = "1"

# 3. pytest 실행 (api/tests/pytest.ini 자동 사용)
pytest -v
```

이 방법은:
- ✅ Windows 버퍼 문제 해결 (`-s --capture=no` 포함)
- ✅ 커버리지 옵션 충돌 방지
- ✅ 올바른 테스트 경로 사용

