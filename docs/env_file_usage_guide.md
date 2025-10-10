# .env 파일 사용 가이드

## 개요

LawFirmAI 프로젝트는 `.env` 파일을 사용하여 환경 변수를 관리합니다. 이를 통해 개발, 스테이징, 프로덕션 환경별로 다른 설정을 쉽게 관리할 수 있습니다.

## 설치 및 설정

### 1. python-dotenv 패키지 설치

```bash
pip install python-dotenv>=1.0.0
```

### 2. .env 파일 생성

프로젝트 루트 디렉토리에 `.env` 파일을 생성합니다:

```bash
# env.example 파일을 복사하여 .env 파일 생성
cp env.example .env
```

### 3. 환경 변수 설정

`.env` 파일을 열고 실제 값으로 수정합니다:

```bash
# LLM 설정
LLM_PROVIDER=google
LLM_MODEL=gemini-pro
GOOGLE_API_KEY=your-actual-google-api-key-here

# Langfuse 설정
LANGFUSE_ENABLED=true
LANGFUSE_SECRET_KEY=your-actual-secret-key
LANGFUSE_PUBLIC_KEY=your-actual-public-key
```

## 주요 환경 변수

### LLM 설정

| 변수명 | 설명 | 기본값 | 예시 |
|--------|------|--------|------|
| `LLM_PROVIDER` | LLM 제공자 | `openai` | `google`, `openai`, `anthropic`, `local` |
| `LLM_MODEL` | LLM 모델명 | `gpt-3.5-turbo` | `gemini-pro`, `gpt-4`, `claude-3` |
| `LLM_TEMPERATURE` | 생성 온도 | `0.7` | `0.0` ~ `1.0` |
| `LLM_MAX_TOKENS` | 최대 토큰 수 | `1000` | `512`, `2048`, `4096` |

### API 키 설정

| 변수명 | 설명 | 필수 여부 |
|--------|------|-----------|
| `OPENAI_API_KEY` | OpenAI API 키 | OpenAI 사용 시 |
| `GOOGLE_API_KEY` | Google AI API 키 | Gemini 사용 시 |
| `ANTHROPIC_API_KEY` | Anthropic API 키 | Claude 사용 시 |

### 벡터 저장소 설정

| 변수명 | 설명 | 기본값 | 예시 |
|--------|------|--------|------|
| `VECTOR_STORE_TYPE` | 벡터 저장소 타입 | `faiss` | `faiss`, `chroma`, `pinecone` |
| `VECTOR_STORE_PATH` | 벡터 저장소 경로 | `./data/embeddings/faiss_index` | `./data/embeddings/chroma_db` |
| `EMBEDDING_MODEL` | 임베딩 모델 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `text-embedding-ada-002` |

### Langfuse 설정

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `LANGFUSE_ENABLED` | Langfuse 활성화 | `true` |
| `LANGFUSE_SECRET_KEY` | Langfuse 시크릿 키 | - |
| `LANGFUSE_PUBLIC_KEY` | Langfuse 퍼블릭 키 | - |
| `LANGFUSE_HOST` | Langfuse 호스트 | `https://cloud.langfuse.com` |
| `LANGFUSE_DEBUG` | 디버그 모드 | `false` |

## 환경별 설정 예시

### 개발 환경

```bash
# .env.development
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
LANGFUSE_DEBUG=true
LLM_PROVIDER=google
LLM_MODEL=gemini-pro
GOOGLE_API_KEY=your-dev-api-key
```

### 프로덕션 환경

```bash
# .env.production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
LANGFUSE_DEBUG=false
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=your-prod-api-key
ENABLE_CACHING=true
CACHE_TTL=7200
```

## 코드에서 사용법

### 1. 자동 로드 (권장)

모든 Python 파일에서 `.env` 파일이 자동으로 로드됩니다:

```python
# source/utils/langchain_config.py
from dotenv import load_dotenv

# .env 파일 자동 로드
load_dotenv()

# 환경 변수 사용
api_key = os.getenv('GOOGLE_API_KEY')
```

### 2. 수동 로드

특정 경로의 `.env` 파일을 로드하려면:

```python
from dotenv import load_dotenv

# 특정 경로의 .env 파일 로드
load_dotenv('.env.production')

# 또는 여러 파일 로드
load_dotenv('.env')  # 기본 설정
load_dotenv('.env.local', override=True)  # 로컬 오버라이드
```

### 3. 설정 클래스 사용

```python
from source.utils.langchain_config import LangChainConfig

# 환경 변수에서 설정 로드
config = LangChainConfig.from_env()

# 설정 사용
print(f"LLM Provider: {config.llm_provider.value}")
print(f"LLM Model: {config.llm_model}")
```

## 보안 고려사항

### 1. .env 파일 보안

- `.env` 파일은 절대 Git에 커밋하지 마세요
- `.gitignore`에 `.env` 파일이 포함되어 있는지 확인하세요
- 프로덕션에서는 환경 변수를 직접 설정하거나 보안 관리 시스템을 사용하세요

### 2. API 키 관리

```bash
# .env 파일에서 API 키 설정
GOOGLE_API_KEY=your-actual-api-key-here

# 환경 변수로 직접 설정 (더 안전)
export GOOGLE_API_KEY=your-actual-api-key-here
```

## 문제 해결

### 1. .env 파일이 로드되지 않는 경우

```python
# 디버깅을 위해 환경 변수 확인
import os
from dotenv import load_dotenv

print("Before load_dotenv:", os.getenv('GOOGLE_API_KEY'))
load_dotenv()
print("After load_dotenv:", os.getenv('GOOGLE_API_KEY'))
```

### 2. 환경 변수가 None인 경우

```python
# 기본값 설정
api_key = os.getenv('GOOGLE_API_KEY', 'default-value')
if not api_key or api_key == 'default-value':
    raise ValueError("GOOGLE_API_KEY is not set")
```

### 3. 여러 환경 파일 사용

```python
# 환경별 .env 파일 로드
import os
from dotenv import load_dotenv

env = os.getenv('ENVIRONMENT', 'development')
load_dotenv(f'.env.{env}')
load_dotenv('.env')  # 기본 설정
```

## 스크립트 실행

### 1. 기본 실행

```bash
# .env 파일이 자동으로 로드됨
python scripts/demo_langchain_rag.py
```

### 2. 특정 환경으로 실행

```bash
# 환경 변수 설정 후 실행
export ENVIRONMENT=production
python scripts/demo_langchain_rag.py
```

### 3. 환경별 스크립트 실행

```bash
# 개발 환경
python scripts/test_env_variables.py

# 프로덕션 환경
ENVIRONMENT=production python scripts/test_env_variables.py
```

## 모범 사례

### 1. 환경 변수 명명 규칙

- 대문자와 언더스코어 사용: `GOOGLE_API_KEY`
- 의미 있는 이름 사용: `LLM_PROVIDER` (not `PROVIDER`)
- 네임스페이스 사용: `LANGFUSE_SECRET_KEY`

### 2. 기본값 설정

```python
# 좋은 예
llm_provider = os.getenv('LLM_PROVIDER', 'openai')
debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'

# 나쁜 예
api_key = os.getenv('API_KEY')  # None일 수 있음
```

### 3. 유효성 검사

```python
# 필수 환경 변수 검사
required_vars = ['GOOGLE_API_KEY', 'LLM_PROVIDER']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {missing_vars}")
```

## 참고 자료

- [python-dotenv 공식 문서](https://python-dotenv.readthedocs.io/)
- [12-Factor App - Config](https://12factor.net/config)
- [환경 변수 예시 파일](env.example)
