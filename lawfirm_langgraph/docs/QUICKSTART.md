# LangGraph Studio 빠른 시작 가이드

## 1. 가상환경 설정 (이미 완료됨)

```bash
cd lawfirm_langgraph
python -m venv .venv
.venv\Scripts\Activate.ps1  # PowerShell
# 또는
.venv\Scripts\activate.bat  # CMD
```

## 2. 의존성 설치 (이미 완료됨)

```bash
pip install -r requirements.txt
```

## 3. LangGraph Studio 실행

```bash
# 가상환경 활성화 후
langgraph dev
```

또는 가상환경을 활성화하지 않고:

```bash
.venv\Scripts\langgraph.exe dev
```

## 4. Studio 접속

브라우저에서 제공된 URL로 접속합니다. 일반적으로:
- `http://localhost:8123`

## 5. 그래프 선택

Studio에서 다음 그래프 중 하나를 선택합니다:
- **legal_workflow**: 워크플로우 그래프
- **workflow_service**: 컴파일된 앱

## 6. 설정 확인

설정이 올바른지 확인:

```bash
python test_setup.py
```

## 문제 해결

### 가상환경 활성화 오류

PowerShell에서 실행 정책 오류가 발생하면:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 포트 충돌

기본 포트(8123)가 사용 중이면:

```bash
langgraph dev --port 8124
```

### 그래프 로드 오류

`graph.py`가 올바르게 작동하는지 확인:

```bash
python -c "from graph import graph, app; print('OK')"
```
