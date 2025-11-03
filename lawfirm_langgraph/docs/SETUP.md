# LangGraph Studio 설정 가이드

## 1. 의존성 설치

```bash
cd lawfirm_langgraph
pip install -r requirements.txt
```

또는 LangGraph CLI만 설치:

```bash
pip install "langgraph-cli[inmem]>=0.4.5"
```

## 2. 환경 변수 확인

상위 프로젝트의 `.env` 파일이 올바르게 설정되어 있는지 확인하세요.

필수 환경 변수:
- `GOOGLE_API_KEY`: Google Gemini API 키

선택 환경 변수:
- `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`: Langfuse 추적
- `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`: LangSmith 추적

## 3. LangGraph Studio 실행

```bash
cd lawfirm_langgraph
langgraph dev
```

브라우저에서 제공된 URL로 접속하면 Studio를 사용할 수 있습니다.

## 4. 문제 해결

### Import 오류

상위 프로젝트의 모듈을 찾을 수 없는 경우:

1. 상위 프로젝트 경로 확인
2. `.env` 파일 경로 확인 (`langgraph.json`의 `env` 설정)

### CLI 설치 오류

```bash
python -m pip install --upgrade pip
pip install "langgraph-cli[inmem]>=0.4.5"
```

### 그래프 로드 오류

`graph.py`가 올바르게 로드되는지 확인:

```bash
cd lawfirm_langgraph
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('..').resolve())); from graph import graph; print('OK')"
```

## 5. Studio 사용

1. `langgraph dev` 실행
2. 브라우저에서 제공된 URL 접속
3. 그래프 선택:
   - `legal_workflow`: 워크플로우 그래프
   - `workflow_service`: 컴파일된 앱
4. 그래프 시각화 및 디버깅
