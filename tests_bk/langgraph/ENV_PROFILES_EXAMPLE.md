# 환경변수 프로필 예시

`.env.profiles/` 디렉토리에 각 모니터링 모드별 환경변수 프로필을 생성할 수 있습니다.

## 파일 구조

```
.env.profiles/
├── langsmith.env
├── langfuse.env
├── both.env
├── none.env
└── README.md
```

## 프로필 파일 예시

### langsmith.env

```bash
# LangSmith 전용 환경변수 설정
# LangSmith만 활성화하고 Langfuse는 비활성화

# LangSmith 설정 (LangChain 표준 환경변수)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key-here
LANGCHAIN_PROJECT=LawFirmAI-Test-LangSmith
# LANGCHAIN_ENDPOINT=https://api.smith.langchain.com  # 기본값 사용 시 생략 가능

# Langfuse 비활성화
LANGFUSE_ENABLED=false

# 추가 설정
ENABLE_LANGSMITH=true
```

### langfuse.env

```bash
# Langfuse 전용 환경변수 설정
# Langfuse만 활성화하고 LangSmith는 비활성화

# LangSmith 비활성화
LANGCHAIN_TRACING_V2=false

# Langfuse 설정
LANGFUSE_ENABLED=true
LANGFUSE_SECRET_KEY=your-langfuse-secret-key-here
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key-here
LANGFUSE_HOST=https://cloud.langfuse.com
# LANGFUSE_DEBUG=false  # 디버그 모드 (필요시 true)
```

### both.env

```bash
# LangSmith + Langfuse 동시 사용 환경변수 설정
# 두 모니터링 도구 모두 활성화

# LangSmith 설정
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key-here
LANGCHAIN_PROJECT=LawFirmAI-Test-Both
ENABLE_LANGSMITH=true

# Langfuse 설정
LANGFUSE_ENABLED=true
LANGFUSE_SECRET_KEY=your-langfuse-secret-key-here
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key-here
LANGFUSE_HOST=https://cloud.langfuse.com
```

### none.env

```bash
# 모니터링 비활성화 환경변수 설정
# 모든 모니터링 도구 비활성화 (최소 리소스 사용)

# LangSmith 비활성화
LANGCHAIN_TRACING_V2=false
ENABLE_LANGSMITH=false

# Langfuse 비활성화
LANGFUSE_ENABLED=false

# 참고: API 키 설정은 불필요하지만, 코드에서 참조할 수 있으므로 주석 처리
# LANGCHAIN_API_KEY=
# LANGFUSE_SECRET_KEY=
# LANGFUSE_PUBLIC_KEY=
```

## 사용 방법

### 1. 프로필 로드 (Python 코드)

```python
from tests.langgraph.monitoring_switch import MonitoringSwitch
import os

# 프로필 로드
env_vars = MonitoringSwitch.load_profile("langsmith")
for key, value in env_vars.items():
    os.environ[key] = value
```

### 2. 프로필 적용 (수동)

```bash
# 환경변수 적용
export $(cat .env.profiles/langsmith.env | grep -v '^#' | xargs)
```

## 보안 주의사항

1. **API 키 보호**: 실제 API 키를 저장하지 말고, 환경변수나 보안 저장소에서 읽어오세요.
2. **.gitignore**: `.env.profiles/` 디렉토리는 `.gitignore`에 추가 후, `.env.profiles/*.example` 파일만 버전 관리하세요.
3. **기본값 제거**: 프로덕션 환경에서는 `your-*-key-here` 같은 플레이스홀더를 실제 값으로 교체하세요.
