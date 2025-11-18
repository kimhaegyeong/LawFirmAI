# 가상환경 설정 가이드

## 설정 완료

프로젝트의 각 폴더별 가상환경이 자동으로 활성화되도록 설정되었습니다.

## 현재 가상환경 상태

- ✅ `api/venv`: 존재
- ✅ `scripts/venv`: 존재
- ⚠️ `lawfirm_langgraph/venv`: 없음 (api/venv 사용)

## 설정 파일

### 1. `.vscode/settings.json` (프로젝트 루트)
- Cursor/VSCode에서 Python 인터프리터 자동 감지
- 터미널에서 가상환경 자동 활성화
- 각 폴더별 Python 경로 설정

### 2. 각 폴더별 `pyrightconfig.json`
- `api/pyrightconfig.json`: api 폴더의 가상환경 사용
- `lawfirm_langgraph/pyrightconfig.json`: api 폴더의 가상환경 사용 (자체 venv 없음)
- `scripts/pyrightconfig.json`: scripts 폴더의 가상환경 사용

## 사용 방법

### Cursor에서 Python 명령 실행 시

1. **자동 활성화**: Cursor가 현재 열린 파일의 폴더에 맞는 가상환경을 자동으로 감지하고 활성화합니다.

2. **터미널에서**: 
   - `api/` 폴더의 파일을 열면 → `api/venv` 자동 활성화
   - `lawfirm_langgraph/` 폴더의 파일을 열면 → `api/venv` 자동 활성화
   - `scripts/` 폴더의 파일을 열면 → `scripts/venv` 자동 활성화

### 테스트 실행

**lawfirm_langgraph 폴더에서:**
```bash
# 방법 1: 자동 스크립트 사용
.\run_tests.bat

# 방법 2: 수동 실행 (가상환경 자동 활성화됨)
python -m pytest tests/langgraph_core -v
```

**api 폴더에서:**
```bash
# 가상환경이 자동 활성화된 상태에서
python -m pytest test/ -v
```

## 문제 해결

### 가상환경이 활성화되지 않는 경우

1. **Cursor 재시작**: 설정 변경 후 Cursor를 재시작하세요.

2. **Python 인터프리터 수동 선택**:
   - `Ctrl+Shift+P` → "Python: Select Interpreter"
   - 해당 폴더의 가상환경 선택

3. **가상환경 생성** (필요한 경우):
   ```bash
   # lawfirm_langgraph에 자체 venv 생성하려면
   cd lawfirm_langgraph
   python -m venv venv
   venv\Scripts\activate
   pip install -r ../api/requirements.txt
   ```

## 참고

- 각 폴더의 `pyrightconfig.json`은 타입 체크와 자동완성을 위한 설정입니다.
- `.vscode/settings.json`은 Cursor/VSCode의 Python 확장 기능을 위한 설정입니다.
- 두 설정이 함께 작동하여 올바른 가상환경을 자동으로 사용합니다.


