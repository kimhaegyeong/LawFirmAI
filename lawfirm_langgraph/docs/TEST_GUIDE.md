# LangGraph 테스트 가이드

## 테스트 실행 방법

LangGraph가 정상적으로 동작하는지 테스트하기 위한 여러 테스트 스크립트를 제공합니다.

### 1. 빠른 테스트 (기본 확인)

가장 빠르게 기본 기능이 동작하는지 확인합니다:

```bash
cd lawfirm_langgraph
.venv\Scripts\python.exe test_quick.py
```

**테스트 항목:**
- Python 버전 확인
- 필수 모듈 import 확인
- 설정 로딩 확인
- Graph 생성 확인
- App 생성 확인

### 2. 종합 테스트 (권장)

모든 기능을 종합적으로 테스트합니다:

```bash
cd lawfirm_langgraph
.venv\Scripts\python.exe test_workflow.py
```

**테스트 항목:**
- 모든 빠른 테스트 항목 포함
- 워크플로우 실행 테스트
- LangGraph CLI 확인
- langgraph.json 설정 확인

### 3. 결과 파일 저장 테스트

테스트 결과를 파일로 저장하여 나중에 확인할 수 있습니다:

```bash
cd lawfirm_langgraph
.venv\Scripts\python.exe test_with_output.py
```

테스트 결과는 `test_results.txt` 파일에 저장됩니다.

### 4. PowerShell 스크립트로 실행

PowerShell 스크립트를 사용하여 모든 테스트를 순차적으로 실행할 수 있습니다:

```powershell
cd lawfirm_langgraph
powershell -ExecutionPolicy Bypass -File run_tests.ps1
```

## 테스트 결과 확인

### 성공적인 테스트 결과

모든 테스트가 통과하면 다음과 같은 메시지를 볼 수 있습니다:

```
✓ 모든 기본 테스트 통과!
```

### 실패한 테스트 해결 방법

#### 1. Import 오류

**오류:**
```
✗ Graph export 모듈: No module named 'xxx'
```

**해결 방법:**
```bash
cd lawfirm_langgraph
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### 2. 설정 오류

**오류:**
```
✗ 설정 로딩 실패: ...
```

**해결 방법:**
- 상위 프로젝트의 `.env` 파일이 존재하는지 확인
- 필요한 환경 변수가 설정되어 있는지 확인

#### 3. Graph 생성 오류

**오류:**
```
✗ Graph 생성 실패: ...
```

**해결 방법:**
- LLM API 키가 올바르게 설정되어 있는지 확인
- 네트워크 연결 확인
- 의존성 패키지가 모두 설치되어 있는지 확인

#### 4. 워크플로우 실행 오류

**오류:**
```
⚠ 워크플로우 실행 실패: ...
```

**해결 방법:**
- LLM API 키 설정 확인 (GOOGLE_API_KEY 등)
- 네트워크 연결 확인
- 모델 설정 확인 (GOOGLE_MODEL 등)

이 오류는 치명적이지 않을 수 있습니다. Graph와 App 생성이 성공했다면 LangGraph Studio를 실행할 수 있습니다.

## 다음 단계

모든 테스트가 통과하면 다음 단계를 진행하세요:

### 1. LangGraph Studio 실행

```bash
cd lawfirm_langgraph
langgraph dev
```

브라우저에서 `http://localhost:8123`에 접속하여 LangGraph Studio를 사용할 수 있습니다.

### 2. 워크플로우 시각화 확인

LangGraph Studio에서 다음을 확인할 수 있습니다:
- 워크플로우 그래프 구조
- 각 노드의 입력/출력
- 상태 변환 과정
- 디버깅 정보

### 3. 실제 워크플로우 테스트

Studio에서 실제 법률 질문을 입력하여 워크플로우가 정상 동작하는지 확인합니다.

## 테스트 파일 목록

- `test_setup.py`: 기본 설정 테스트 (Import, Graph 생성, CLI 확인)
- `test_quick.py`: 빠른 기본 기능 테스트
- `test_workflow.py`: 종합 워크플로우 테스트
- `test_with_output.py`: 결과를 파일로 저장하는 테스트
- `run_tests.ps1`: 모든 테스트를 순차적으로 실행하는 PowerShell 스크립트

## 문제 해결

테스트 중 문제가 발생하면 다음을 확인하세요:

1. **가상환경 활성화**: `.venv\Scripts\Activate.ps1` 실행
2. **의존성 설치**: `pip install -r requirements.txt` 실행
3. **Python 버전**: Python 3.10 이상 필요
4. **환경 변수**: `.env` 파일이 상위 디렉토리에 있는지 확인
5. **네트워크 연결**: LLM API 사용 시 인터넷 연결 필요

추가 문제가 있으면 프로젝트 문서를 참조하세요.
