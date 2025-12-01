# 가상환경 사용 규칙 (Virtual Environment Rules)

## 목적

프로젝트는 운영 배포 시 각 모듈별로 독립적인 Python 가상환경을 사용하여 의존성 충돌을 방지하고 환경을 격리합니다.

## 가상환경 구조

### 유지되는 가상환경
- `/api/venv/` - FastAPI 서버용 가상환경
- `/scripts/venv/` - 스크립트 실행용 가상환경

### 제거된 가상환경
- 프로젝트 루트의 `.venv/`, `venv/`, `env/` - 사용하지 않음 (삭제됨)

## 파일 경로별 가상환경 매핑

### 1. API 관련 파일 (`/api/**/*.py`)

**가상환경**: `/api/venv/`

**활성화 방법**:
```powershell
# Windows (PowerShell)
cd api
.\venv\Scripts\Activate.ps1

# Windows (CMD)
cd api
venv\Scripts\activate.bat

# Linux/Mac
cd api
source venv/bin/activate
```

**의존성 파일**: `api/requirements.txt`

**사용 시나리오**:
- FastAPI 서버 실행 (`api/main.py`)
- API 라우터 테스트
- API 관련 유닛/통합 테스트

### 2. 스크립트 관련 파일 (`/scripts/**/*.py`)

**가상환경**: `/scripts/venv/`

**활성화 방법**:
```powershell
# Windows (PowerShell)
cd scripts
.\venv\Scripts\Activate.ps1

# Windows (CMD)
cd scripts
venv\Scripts\activate.bat

# Linux/Mac
cd scripts
source venv/bin/activate
```

**의존성 파일**: `scripts/requirements.txt`

**사용 시나리오**:
- 데이터 수집 스크립트 실행
- 데이터 전처리 스크립트 실행
- 마이그레이션 스크립트 실행
- 분석 및 모니터링 스크립트 실행

### 3. LangGraph 코어 모듈 (`/lawfirm_langgraph/**/*.py`)

**가상환경**: API 또는 Scripts 가상환경에서 사용 (공통 의존성)

**설명**:
- LangGraph 코어 모듈은 별도의 가상환경이 없음
- API나 Scripts 가상환경에서 공통으로 사용
- 두 가상환경 모두 필요한 의존성을 포함해야 함

## 실행 규칙

### Python 파일 실행 시
1. **파일 경로 확인**: 실행할 Python 파일의 경로를 확인
2. **가상환경 매핑**: 파일 경로에 따라 적절한 가상환경 선택
   - `/api/` 경로 → `/api/venv/`
   - `/scripts/` 경로 → `/scripts/venv/`
3. **가상환경 활성화**: 해당 디렉토리로 이동하여 가상환경 활성화
4. **스크립트 실행**: 활성화된 가상환경에서 스크립트 실행

### 터미널 명령 실행 시
- Python 파일을 실행하는 명령은 해당 파일의 가상환경을 사용
- 명령 실행 전 적절한 가상환경이 활성화되어 있는지 확인
- 가상환경이 활성화되지 않은 경우 자동으로 활성화하거나 경고 메시지 표시

### 테스트 실행 시
- API 테스트 (`/api/tests/`) → `/api/venv/` 사용
- Scripts 테스트 (`/scripts/tests/`) → `/scripts/venv/` 사용
- LangGraph 테스트 (`/lawfirm_langgraph/tests/`) → API 또는 Scripts 가상환경 사용

## 의존성 관리

### requirements.txt 업데이트
각 모듈의 `requirements.txt` 파일을 독립적으로 관리:

```bash
# API 가상환경에 패키지 설치 후
cd api
.\venv\Scripts\Activate.ps1
pip freeze > requirements.txt

# Scripts 가상환경에 패키지 설치 후
cd scripts
.\venv\Scripts\Activate.ps1
pip freeze > requirements.txt
```

### 공통 의존성
- LangGraph 관련 패키지는 두 가상환경 모두에 포함
- 버전 충돌 방지를 위해 동일한 버전 사용 권장

## 주의사항

### ⚠️ 금지 사항
1. **프로젝트 루트 가상환경 사용 금지**
   - 프로젝트 루트에 `.venv/`, `venv/`, `env/` 생성 금지
   - 기존 루트 가상환경은 삭제됨

2. **가상환경 혼용 금지**
   - API 스크립트를 Scripts 가상환경에서 실행 금지
   - Scripts 스크립트를 API 가상환경에서 실행 금지

3. **의존성 충돌 방지**
   - 각 가상환경의 의존성을 독립적으로 관리
   - 공통 패키지는 동일한 버전 사용

### ✅ 권장 사항
1. **가상환경 자동 활성화 스크립트 사용**
   - 각 디렉토리에 활성화 스크립트 제공 (선택사항)
   - 배치 파일이나 PowerShell 스크립트로 자동화

2. **의존성 버전 고정**
   - `requirements.txt`에 버전 명시
   - 프로덕션 환경에서 재현 가능한 빌드 보장

3. **가상환경 상태 확인**
   - 스크립트 실행 전 가상환경 활성화 상태 확인
   - 잘못된 가상환경 사용 시 경고 메시지 표시

## 예시

### API 서버 실행
```powershell
# 1. API 디렉토리로 이동
cd api

# 2. 가상환경 활성화
.\venv\Scripts\Activate.ps1

# 3. 서버 실행
python main.py
```

### 스크립트 실행
```powershell
# 1. Scripts 디렉토리로 이동
cd scripts

# 2. 가상환경 활성화
.\venv\Scripts\Activate.ps1

# 3. 스크립트 실행
python ingest/open_law/scripts/collect_statutes.py
```

### 테스트 실행

각 모듈의 가상환경에서 독립적으로 pytest를 실행합니다.

#### API 테스트 실행
```powershell
# 1. API 디렉토리로 이동
cd api

# 2. 가상환경 활성화
.\venv\Scripts\Activate.ps1

# 3. 테스트 실행
pytest tests/                    # 모든 테스트
pytest tests/unit/               # 단위 테스트만
pytest tests/integration/        # 통합 테스트만
pytest tests/e2e/               # E2E 테스트만
pytest -m "not slow"            # 느린 테스트 제외
```

#### Scripts 테스트 실행
```powershell
# 1. Scripts 디렉토리로 이동
cd scripts

# 2. 가상환경 활성화
.\venv\Scripts\Activate.ps1

# 3. 테스트 실행
pytest tests/                    # 모든 테스트
pytest tests/unit/               # 단위 테스트만
pytest tests/integration/        # 통합 테스트만
pytest tests/functional/         # 기능 테스트만
pytest -m "not slow"            # 느린 테스트 제외
```

#### LangGraph 테스트 실행
```powershell
# API 또는 Scripts 가상환경 사용
cd api  # 또는 cd scripts
.\venv\Scripts\Activate.ps1

# LangGraph 테스트 실행
cd ../lawfirm_langgraph
pytest tests/                    # 모든 테스트
pytest tests/unit/               # 단위 테스트만
pytest tests/integration/        # 통합 테스트만
```

#### pytest 설정 파일
각 모듈은 독립적인 `pytest.ini` 파일을 가지고 있습니다:
- `/api/tests/pytest.ini` - API 테스트 설정
- `/scripts/tests/pytest.ini` - Scripts 테스트 설정
- `/lawfirm_langgraph/pytest.ini` - LangGraph 테스트 설정

**주의**: 프로젝트 루트의 `pytest.ini`는 삭제되었습니다. 각 모듈의 가상환경에서 해당 모듈의 `pytest.ini`를 사용합니다.

## 문제 해결

### 가상환경이 없는 경우
```powershell
# API 가상환경 생성
cd api
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Scripts 가상환경 생성
cd scripts
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 의존성 충돌 발생 시
1. 해당 가상환경의 `requirements.txt` 확인
2. 충돌하는 패키지 버전 조정
3. 가상환경 재생성 및 의존성 재설치

---

이 규칙은 프로젝트의 모듈별 환경 격리를 보장하고, 운영 배포 시 의존성 충돌을 방지하기 위해 작성되었습니다.

