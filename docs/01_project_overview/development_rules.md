# LawFirmAI 개발 규칙 및 가이드라인

## 📋 문서 개요

본 문서는 LawFirmAI 프로젝트의 개발 규칙, 코딩 스타일, 운영 가이드라인을 정의합니다. Phase 1-6이 완료된 지능형 대화 시스템과 성능 최적화된 의미적 검색 시스템의 개발 가이드라인을 포함합니다.

## 📝 Git 커밋 메시지 규칙

### 커밋 메시지 형식
모든 커밋 메시지는 **영어**로 작성해야 합니다.

#### 기본 형식
```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Type (필수)
- `feat`: 새로운 기능 추가
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 코드 포맷팅, 세미콜론 누락 등 (기능 변경 없음)
- `refactor`: 코드 리팩토링
- `test`: 테스트 코드 추가/수정
- `chore`: 빌드 프로세스, 보조 도구 변경

#### Scope (선택)
- `data-collection`: 데이터 수집 관련
- `legal-terms`: 법률 용어 수집 시스템 관련 (NEW - 2025.10.24)
- `search-engine`: 검색 엔진 관련
- `database`: 데이터베이스 관련
- `chat-service`: 채팅 서비스 관련
- `api`: API 관련
- `ui`: 사용자 인터페이스 관련
- `config`: 설정 관련

#### Subject (필수)
- 50자 이내
- 명령형 (imperative mood)
- 첫 글자는 소문자
- 마침표 없음

#### Body (선택)
- 72자 이내로 줄바꿈
- 무엇을 왜 변경했는지 설명
- 어떻게 변경했는지 설명

#### Footer (선택)
- Breaking changes
- Issues 참조

#### 예시
```
feat(search-engine): implement current law search engine

- Add CurrentLawSearchEngine class
- Implement hybrid search (vector + FTS + exact matching)
- Add law name normalization and mapping
- Integrate with unified search engine

Closes #123
```

```
fix(chat-service): resolve law article pattern matching issue

- Fix regex pattern for law article detection
- Improve confidence calculation for specific articles
- Add priority processing for article search

Fixes #456
```

```
docs: update project completion report

- Add Phase 5 current law search system
- Update performance metrics
- Add technical documentation

No breaking changes
```

```
feat(legal-terms): implement legal term collection system

- Add LegalTermCollector class with alternating collection
- Implement JSON response parsing and quality filtering
- Add duplicate prevention and error handling
- Support batch processing and progress tracking

Closes #789
```

```
fix(legal-terms): resolve JSON serialization issue

- Fix LegalTermDetail object serialization using asdict()
- Resolve variable scope error for asdict import
- Prevent empty array file saving
- Improve data validation logic

Fixes #790
```

### 커밋 메시지 작성 가이드라인

#### 1. 명확성
- 변경사항을 명확하게 설명
- 왜 변경했는지 이유 포함
- 어떤 문제를 해결했는지 명시

#### 2. 일관성
- 동일한 용어 사용
- 일관된 형식 유지
- 표준화된 타입 사용

#### 3. 간결성
- 불필요한 단어 제거
- 핵심 내용만 포함
- 명확하고 간결한 표현

#### 4. 검색 가능성
- 관련 이슈 번호 포함
- 키워드 사용
- 태그 활용

## 🚀 프로세스 관리 규칙

### Gradio 서버 관리

#### 서버 시작
```bash
# Gradio 서버 시작 (LangChain 기반)
cd gradio
python simple_langchain_app.py

# 또는 최신 앱 실행 (7개 탭 구성)
python app.py
```

#### 서버 종료 (PID 기준)
**⚠️ 중요**: `taskkill /f /im python.exe` 사용 금지

**올바른 종료 방법**:

1. **PID 파일 기반 종료** (권장):
```bash
# Windows
python gradio/stop_server.py

# 또는 배치 파일 사용
gradio/stop_server.bat
```

2. **포트 기반 종료**:
```bash
# 7860 포트 사용 프로세스 확인
netstat -ano | findstr :7860

# 특정 PID 종료
taskkill /PID [PID번호] /F
```

#### PID 관리 구현 규칙

**모든 Gradio 애플리케이션은 다음 규칙을 따라야 합니다**:

1. **PID 파일 생성**:
```python
import os
import signal
import atexit
from pathlib import Path

def save_pid():
    """현재 프로세스 PID를 파일에 저장"""
    pid = os.getpid()
    pid_file = Path("gradio_server.pid")
    
    try:
        with open(pid_file, 'w') as f:
            f.write(str(pid))
        print(f"PID {pid} saved to {pid_file}")
    except Exception as e:
        print(f"Failed to save PID: {e}")

def cleanup_pid():
    """PID 파일 정리"""
    pid_file = Path("gradio_server.pid")
    if pid_file.exists():
        try:
            pid_file.unlink()
            print("PID file removed")
        except Exception as e:
            print(f"Failed to remove PID file: {e}")

# 앱 시작 시
save_pid()

# 앱 종료 시 정리
atexit.register(cleanup_pid)
signal.signal(signal.SIGINT, lambda s, f: cleanup_pid() or exit(0))
signal.signal(signal.SIGTERM, lambda s, f: cleanup_pid() or exit(0))
```

#### 금지 사항

**❌ 절대 사용하지 말 것**:
```bash
# 모든 Python 프로세스 종료 (위험!)
taskkill /f /im python.exe
```

**✅ 올바른 방법**:
```bash
# 특정 PID만 종료
taskkill /PID 12345 /F

# 또는 제공된 스크립트 사용
python gradio/stop_server.py
```

## 🚀 Phase별 개발 가이드라인

### Phase 1-3: 지능형 대화 시스템 개발

#### Phase 1: 대화 맥락 강화
```python
# 통합 세션 관리 구현 예시
from source.services.integrated_session_manager import IntegratedSessionManager

class ChatService:
    def __init__(self):
        self.session_manager = IntegratedSessionManager()
    
    def process_message(self, message: str, session_id: str):
        # 세션 컨텍스트 로드
        context = self.session_manager.get_session_context(session_id)
        
        # 다중 턴 질문 처리
        processed_message = self.session_manager.process_multi_turn(message, context)
        
        # 컨텍스트 압축
        compressed_context = self.session_manager.compress_context(context)
        
        return processed_message, compressed_context
```

#### Phase 2: 개인화 및 지능형 분석
```python
# 사용자 프로필 기반 개인화 구현 예시
from source.services.user_profile_manager import UserProfileManager
from source.services.emotion_intent_analyzer import EmotionIntentAnalyzer

class PersonalizedChatService:
    def __init__(self):
        self.profile_manager = UserProfileManager()
        self.emotion_analyzer = EmotionIntentAnalyzer()
    
    def get_personalized_response(self, message: str, user_id: str):
        # 사용자 프로필 로드
        profile = self.profile_manager.get_profile(user_id)
        
        # 감정 및 의도 분석
        emotion_result = self.emotion_analyzer.analyze(message)
        
        # 개인화된 응답 생성
        response = self.generate_response(message, profile, emotion_result)
        
        return response
```

#### Phase 3: 장기 기억 및 품질 모니터링
```python
# 맥락적 메모리 관리 구현 예시
from source.services.contextual_memory_manager import ContextualMemoryManager
from source.services.conversation_quality_monitor import ConversationQualityMonitor

class AdvancedChatService:
    def __init__(self):
        self.memory_manager = ContextualMemoryManager()
        self.quality_monitor = ConversationQualityMonitor()
    
    def process_with_memory(self, message: str, user_id: str):
        # 관련 메모리 검색
        relevant_memories = self.memory_manager.search_memories(message, user_id)
        
        # 품질 모니터링
        quality_score = self.quality_monitor.assess_quality(message, relevant_memories)
        
        # 메모리 업데이트
        self.memory_manager.update_memory(message, user_id, quality_score)
        
        return relevant_memories, quality_score
```

### Phase 5: 성능 최적화 개발

#### 통합 캐싱 시스템
```python
# 다층 캐싱 시스템 구현 예시
from source.services.integrated_cache_system import IntegratedCacheSystem

class OptimizedChatService:
    def __init__(self):
        self.cache_system = IntegratedCacheSystem()
    
    def get_cached_response(self, message: str, session_id: str):
        # 캐시 키 생성
        cache_key = self.cache_system.generate_key(message, session_id)
        
        # 다층 캐시 검색
        cached_result = self.cache_system.get(cache_key)
        
        if cached_result:
            return cached_result
        
        # 캐시 미스 시 새로 생성
        result = self.generate_response(message)
        
        # 캐시 저장
        self.cache_system.set(cache_key, result)
        
        return result
```

#### 병렬 검색 엔진
```python
# 병렬 검색 구현 예시
import asyncio
from source.services.optimized_hybrid_search_engine import OptimizedHybridSearchEngine

class ParallelSearchService:
    def __init__(self):
        self.search_engine = OptimizedHybridSearchEngine()
    
    async def parallel_search(self, query: str):
        # 정확 검색과 의미 검색을 동시 실행
        exact_task = asyncio.create_task(self.search_engine.exact_search(query))
        semantic_task = asyncio.create_task(self.search_engine.semantic_search(query))
        
        # 결과 병합
        exact_results, semantic_results = await asyncio.gather(exact_task, semantic_task)
        
        return self.search_engine.merge_results(exact_results, semantic_results)
```

### Phase 6: 의미적 검색 시스템 개발

#### FAISS 기반 벡터 검색
```python
# 의미적 검색 엔진 구현 예시
from source.services.semantic_search_engine import SemanticSearchEngine

class VectorSearchService:
    def __init__(self):
        self.semantic_engine = SemanticSearchEngine()
    
    def semantic_search(self, query: str, limit: int = 10):
        # 쿼리 벡터화
        query_vector = self.semantic_engine.encode_query(query)
        
        # FAISS 인덱스에서 검색
        scores, indices = self.semantic_engine.search(query_vector, limit)
        
        # 메타데이터와 함께 결과 반환
        results = []
        for score, idx in zip(scores, indices):
            metadata = self.semantic_engine.get_metadata(idx)
            results.append({
                'text': metadata['text'],
                'score': float(score),
                'metadata': metadata
            })
        
        return results
```

#### 다중 모델 지원
```python
# 다중 모델 관리자 구현 예시
from source.services.multi_model_manager import MultiModelManager

class MultiModelService:
    def __init__(self):
        self.model_manager = MultiModelManager()
    
    def search_with_multiple_models(self, query: str):
        results = {}
        
        # ko-sroberta-multitask 모델로 검색
        kobart_results = self.model_manager.search_with_model(
            query, model_name="ko-sroberta-multitask"
        )
        results['kobart'] = kobart_results
        
        # BGE-M3-Korean 모델로 검색
        bge_results = self.model_manager.search_with_model(
            query, model_name="BGE-M3-Korean"
        )
        results['bge'] = bge_results
        
        # 결과 통합
        return self.model_manager.merge_model_results(results)
```
```
LawFirmAI/
├── gradio/                          # Gradio 웹 애플리케이션
│   ├── simple_langchain_app.py      # 메인 LangChain 기반 앱
│   ├── app.py                       # 기본 Gradio 앱
│   ├── stop_server.py               # 서버 종료 스크립트
│   ├── requirements.txt             # Gradio 의존성
│   └── Dockerfile                   # Gradio Docker 설정
├── source/                          # 핵심 모듈
│   ├── services/                    # 비즈니스 로직 (80+ 서비스)
│   ├── data/                        # 데이터 처리
│   ├── models/                      # AI 모델
│   └── utils/                       # 유틸리티
├── data/                            # 데이터 파일
│   ├── lawfirm.db                   # SQLite 데이터베이스
│   └── embeddings/                  # 벡터 임베딩
└── docs/                            # 문서
```

### 벡터 저장소 경로 규칙
```python
# 현재 사용 중인 벡터 저장소
vector_store_paths = [
    "data/embeddings/ml_enhanced_ko_sroberta",  # ko-sroberta 벡터
    "data/embeddings/ml_enhanced_bge_m3",       # BGE-M3 벡터
]
```

## 📝 로깅 규칙

### Windows 환경 로깅 주의사항

**이모지 사용 금지** (Windows cp949 인코딩 문제):
```python
# ❌ 잘못된 예시
logger.info("🚀 Starting process...")
logger.info("✅ Process completed")

# ✅ 올바른 예시  
logger.info("Starting process...")
logger.info("Process completed")
logger.info("[OK] Process completed")
logger.info("[ERROR] Process failed")
```

### 한국어 인코딩 처리 규칙

**⚠️ 중요**: Windows 환경에서 한국어 콘솔 출력 문제 해결을 위한 규칙

#### 환경 변수 설정 (필수)
```python
# 모든 Python 파일 상단에 추가
import os
import sys

# 인코딩 설정 (최우선)
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
```

#### 안전한 콘솔 출력
```python
def safe_print(message: str):
    """안전한 콘솔 출력 (인코딩 처리)"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode('utf-8', errors='replace').decode('utf-8'))

# 사용 예시
safe_print("법률 문서 분석을 시작합니다.")
safe_print("벡터 저장소 로딩 완료")
```

### 현재 구현된 로깅 시스템
```python
# gradio/simple_langchain_app.py에서 사용 중
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/simple_langchain_gradio.log')
    ]
)
logger = logging.getLogger(__name__)

# 사용 예시
logger.info("LawFirmAI service initialized")
logger.info("Vector store loaded successfully")
logger.warning("Configuration issue detected")
logger.error("Critical error occurred")
```

## 🛡️ 보안 규칙

### 환경 변수 관리
```python
import os
from pathlib import Path

# 환경 변수 파일 로드
env_file = Path(".env")
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv()

# API 키 관리
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.warning("OpenAI API key not found, using fallback")

# 현재 사용 중인 환경 변수
required_env_vars = [
    "OPENAI_API_KEY",      # OpenAI API 키
    "GOOGLE_API_KEY",      # Google API 키 (선택사항)
    "DATABASE_URL",        # 데이터베이스 URL
    "MODEL_PATH"           # 모델 경로
]
```

## 🧪 테스트 규칙

### 현재 구현된 테스트 시스템
```python
# gradio/test_simple_query.py에서 구현
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_vector_store_loading():
    """벡터 저장소 로딩 테스트"""
    from source.data.vector_store import LegalVectorStore
    
    vector_store = LegalVectorStore("test-model")
    assert vector_store is not None

def test_gradio_app_startup():
    """Gradio 앱 시작 테스트"""
    import subprocess
    import time
    
    # 앱 시작
    process = subprocess.Popen(['python', 'gradio/simple_langchain_app.py'])
    
    # 잠시 대기
    time.sleep(5)
    
    # 프로세스 상태 확인
    assert process.poll() is None, "App should be running"
    
    # 정리
    process.terminate()
    process.wait()
```

## 📊 성능 모니터링 규칙

### 메모리 사용량 모니터링
```python
import psutil
import time

def monitor_memory():
    """메모리 사용량 모니터링"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
    
    # 메모리 사용량이 임계값을 초과하면 경고
    if memory_info.rss > 1024 * 1024 * 1024:  # 1GB
        logger.warning("High memory usage detected")
```

### 응답 시간 측정
```python
import time
from functools import wraps

def measure_time(func):
    """실행 시간 측정 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"{func.__name__} executed in {end_time - start_time:.3f}s")
        return result
    return wrapper

# 사용 예시
@measure_time
def search_documents(query):
    # 검색 로직
    pass
```

## 🔄 배포 규칙

### 현재 구현된 Docker 설정
```dockerfile
# gradio/Dockerfile (현재 구현)
FROM python:3.9-slim

WORKDIR /app

# 의존성 설치
COPY gradio/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사
COPY gradio/ ./gradio/
COPY source/ ./source/

# 비root 사용자로 실행
RUN useradd --create-home --shell /bin/bash app
USER app

# 포트 노출
EXPOSE 7860

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "gradio/simple_langchain_app.py"]
```

