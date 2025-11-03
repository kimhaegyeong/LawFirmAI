# LawFirm LangGraph 가상환경 설정 가이드

`lawfirm_langgraph` 디렉토리는 LangGraph 기반 워크플로우 실행을 위한 별도 가상환경을 사용합니다.

## 가상환경 생성 및 활성화

### Windows (PowerShell)

```powershell
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
.\.venv\Scripts\Activate.ps1

# 의존성 설치
pip install -r requirements.txt
```

### Windows (CMD)

```cmd
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
.venv\Scripts\activate.bat

# 의존성 설치
pip install -r requirements.txt
```

### Linux/macOS

```bash
# 가상환경 생성
python3 -m venv .venv

# 가상환경 활성화
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

## 가상환경 비활성화

### Windows (PowerShell)

```powershell
deactivate
```

### Windows (CMD)

```cmd
deactivate
```

### Linux/macOS

```bash
deactivate
```

## 주요 의존성

- **LangGraph v1.0**: 그래프 기반 워크플로우 실행
- **LangChain v1.0**: LLM 통합 및 RAG
- **LangChain Google GenAI**: Google Gemini 모델 통합
- **FAISS**: 벡터 검색
- **Sentence Transformers**: 한국어 임베딩 모델

## 스크립트 실행

가상환경이 활성화된 상태에서:

```powershell
# LangGraph 워크플로우 실행
langgraph dev

# Streamlit 앱 실행
streamlit run streamlit/app.py

# Python 스크립트 실행
python graph.py
```

## 문제 해결

### PowerShell 실행 정책 오류

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 패키지 설치 오류

```powershell
# pip 업그레이드
python -m pip install --upgrade pip

# 캐시 클리어 후 재설치
pip cache purge
pip install -r requirements.txt
```
