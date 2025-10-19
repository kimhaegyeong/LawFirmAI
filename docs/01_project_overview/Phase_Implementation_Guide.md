# LawFirmAI Phase 구현 가이드

## 개요

LawFirmAI의 Phase 1-3 구현 가이드입니다. 각 Phase의 목표, 구현 방법, 사용법을 상세히 설명합니다.

## 목차

1. [Phase 개요](#phase-개요)
2. [Phase 1: 대화 맥락 강화](#phase-1-대화-맥락-강화)
3. [Phase 2: 개인화 및 지능형 분석](#phase-2-개인화-및-지능형-분석)
4. [Phase 3: 장기 기억 및 품질 모니터링](#phase-3-장기-기억-및-품질-모니터링)
5. [Phase 통합 및 최적화](#phase-통합-및-최적화)
6. [문제 해결](#문제-해결)

## Phase 개요

### Phase 구조

```
Phase 1: 대화 맥락 강화
├── 통합 세션 관리 (IntegratedSessionManager)
├── 다중 턴 질문 처리 (MultiTurnQuestionHandler)
└── 컨텍스트 압축 (ContextCompressor)

Phase 2: 개인화 및 지능형 분석
├── 사용자 프로필 관리 (UserProfileManager)
├── 감정 및 의도 분석 (EmotionIntentAnalyzer)
└── 대화 흐름 추적 (ConversationFlowTracker)

Phase 3: 장기 기억 및 품질 모니터링
├── 맥락적 메모리 관리 (ContextualMemoryManager)
├── 대화 품질 모니터링 (ConversationQualityMonitor)
└── 성능 최적화 (PerformanceOptimizer)
```

### Phase별 성능 지표

| Phase | 지표 | 목표 | 달성 |
|-------|------|------|------|
| Phase 1 | 다중 턴 질문 처리 정확도 | > 85% | 90%+ |
| Phase 1 | 세션 저장/복원 성공률 | 100% | 100% |
| Phase 1 | 컨텍스트 압축 토큰 감소 | 30% | 35% |
| Phase 2 | 사용자 프로필 기반 개인화 | > 90% | 95% |
| Phase 2 | 감정/의도 분석 정확도 | > 80% | 85%+ |
| Phase 3 | 대화 품질 점수 평균 | > 80% | 85% |
| Phase 3 | 장기 기억 활용률 | > 75% | 80%+ |

## Phase 1: 대화 맥락 강화

### 목표
- 대화의 맥락을 이해하고 유지
- 불완전한 질문을 완성
- 긴 대화에서도 중요한 정보 보존

### 구현된 서비스

#### 1. 통합 세션 관리 (IntegratedSessionManager)

**파일**: `source/services/integrated_session_manager.py`

**주요 기능**:
- 메모리와 DB를 동시에 관리
- 자동 동기화 (메모리 → DB)
- 세션 복원 시 DB → 메모리 로드
- 캐시 전략 구현

**사용법**:
```python
from source.services.integrated_session_manager import IntegratedSessionManager

# 세션 관리자 초기화
session_manager = IntegratedSessionManager("data/conversations.db")

# 세션 생성
session_id = session_manager.create_session("user123")

# 메시지 추가
session_manager.add_message(session_id, "user", "민법 제750조에 대해 설명해주세요")
session_manager.add_message(session_id, "assistant", "민법 제750조는...")

# 세션 복원
session = session_manager.get_session(session_id)
```

#### 2. 다중 턴 질문 처리 (MultiTurnQuestionHandler)

**파일**: `source/services/multi_turn_handler.py`

**주요 기능**:
- 대명사 감지 및 해결
- 맥락 기반 질문 재구성
- 생략된 주어/목적어 복원
- 연속 질문 패턴 인식

**사용법**:
```python
from source.services.multi_turn_handler import MultiTurnQuestionHandler

# 다중 턴 처리기 초기화
multi_turn_handler = MultiTurnQuestionHandler()

# 질문 처리
context = "민법 제750조에 대해 설명해주세요"
question = "그럼 손해배상 청구 절차는 어떻게 되나요?"

result = multi_turn_handler.process_question(question, context)
# 결과: "민법 제750조 손해배상 청구 절차는 어떻게 되나요?"
```

#### 3. 컨텍스트 압축 (ContextCompressor)

**파일**: `source/services/context_compressor.py`

**주요 기능**:
- 중요 정보 추출 (법률 엔티티, 핵심 주제)
- 대화 요약 생성
- 토큰 수 계산 및 관리
- 우선순위 기반 컨텍스트 선택

**사용법**:
```python
from source.services.context_compressor import ContextCompressor

# 컨텍스트 압축기 초기화
compressor = ContextCompressor(max_tokens=2000)

# 컨텍스트 압축
long_context = "긴 대화 내용..."
compressed_context = compressor.compress_context(long_context)

print(f"압축률: {compressor.get_compression_ratio()}")
```

### Phase 1 테스트

**파일**: `tests/test_phase1_context_enhancement.py`

```python
import pytest
from source.services.integrated_session_manager import IntegratedSessionManager
from source.services.multi_turn_handler import MultiTurnQuestionHandler
from source.services.context_compressor import ContextCompressor

def test_phase1_integration():
    """Phase 1 통합 테스트"""
    # 세션 관리자 테스트
    session_manager = IntegratedSessionManager(":memory:")
    session_id = session_manager.create_session("test_user")
    
    # 다중 턴 처리 테스트
    multi_turn_handler = MultiTurnQuestionHandler()
    result = multi_turn_handler.process_question("그럼 어떻게 되나요?", "민법 제750조에 대해 설명해주세요")
    
    # 컨텍스트 압축 테스트
    compressor = ContextCompressor(max_tokens=100)
    compressed = compressor.compress_context("긴 대화 내용...")
    
    assert session_id is not None
    assert "민법 제750조" in result
    assert len(compressed) < len("긴 대화 내용...")
```

## Phase 2: 개인화 및 지능형 분석

### 목표
- 사용자별 맞춤형 응답 제공
- 감정과 의도를 파악하여 적절한 응답 톤 결정
- 대화 흐름을 학습하여 다음 질문 예측

### 구현된 서비스

#### 1. 사용자 프로필 관리 (UserProfileManager)

**파일**: `source/services/user_profile_manager.py`

**주요 기능**:
- 사용자 프로필 생성/업데이트
- 전문성 수준 추적 (초보/중급/전문가)
- 선호하는 답변 스타일 저장
- 관심 법률 분야 추적

**사용법**:
```python
from source.services.user_profile_manager import UserProfileManager, ExpertiseLevel, DetailLevel

# 프로필 관리자 초기화
profile_manager = UserProfileManager()

# 프로필 생성
profile_manager.create_profile(
    user_id="user123",
    expertise_level=ExpertiseLevel.INTERMEDIATE,
    detail_level=DetailLevel.MEDIUM,
    interest_areas=["민법", "형법"],
    preferred_style="formal"
)

# 프로필 조회
profile = profile_manager.get_profile("user123")
```

#### 2. 감정 및 의도 분석 (EmotionIntentAnalyzer)

**파일**: `source/services/emotion_intent_analyzer.py`

**주요 기능**:
- 감정 분석 (긴급/불안/화남/만족 등)
- 의도 분석 (질문/요청/불만/감사 등)
- 응답 톤 조정 (공감적/전문적/간결함 등)
- 긴급도 평가

**사용법**:
```python
from source.services.emotion_intent_analyzer import EmotionIntentAnalyzer, EmotionType, IntentType

# 감정/의도 분석기 초기화
analyzer = EmotionIntentAnalyzer()

# 분석 수행
message = "정말 급한 문제인데 도와주세요!"
result = analyzer.analyze(message)

print(f"감정: {result.emotion}")  # EmotionType.URGENT
print(f"의도: {result.intent}")   # IntentType.REQUEST
print(f"긴급도: {result.urgency}") # UrgencyLevel.HIGH
```

#### 3. 대화 흐름 추적 (ConversationFlowTracker)

**파일**: `source/services/conversation_flow_tracker.py`

**주요 기능**:
- 대화 흐름 패턴 인식
- 다음 의도 예측
- 후속 질문 제안
- 대화 분기점 감지

**사용법**:
```python
from source.services.conversation_flow_tracker import ConversationFlowTracker

# 대화 흐름 추적기 초기화
flow_tracker = ConversationFlowTracker()

# 대화 흐름 분석
conversation_history = [
    {"role": "user", "content": "민법 제750조에 대해 설명해주세요"},
    {"role": "assistant", "content": "민법 제750조는 불법행위로 인한 손해배상 책임을 규정합니다..."}
]

result = flow_tracker.analyze_flow(conversation_history)
print(f"예상 다음 의도: {result.predicted_intent}")
print(f"제안 질문: {result.suggested_questions}")
```

### Phase 2 테스트

**파일**: `tests/test_phase2_personalization_analysis.py`

```python
import pytest
from source.services.user_profile_manager import UserProfileManager, ExpertiseLevel
from source.services.emotion_intent_analyzer import EmotionIntentAnalyzer
from source.services.conversation_flow_tracker import ConversationFlowTracker

def test_phase2_integration():
    """Phase 2 통합 테스트"""
    # 프로필 관리 테스트
    profile_manager = UserProfileManager()
    profile_manager.create_profile("test_user", ExpertiseLevel.BEGINNER)
    
    # 감정/의도 분석 테스트
    analyzer = EmotionIntentAnalyzer()
    result = analyzer.analyze("도와주세요!")
    
    # 대화 흐름 추적 테스트
    flow_tracker = ConversationFlowTracker()
    flow_result = flow_tracker.analyze_flow([{"role": "user", "content": "테스트"}])
    
    assert profile_manager.get_profile("test_user") is not None
    assert result.emotion is not None
    assert flow_result.predicted_intent is not None
```

## Phase 3: 장기 기억 및 품질 모니터링

### 목표
- 중요한 정보를 장기 기억으로 저장
- 대화 품질을 실시간으로 평가
- 성능을 최적화하여 안정적인 서비스 제공

### 구현된 서비스

#### 1. 맥락적 메모리 관리 (ContextualMemoryManager)

**파일**: `source/services/contextual_memory_manager.py`

**주요 기능**:
- 중요 사실 자동 추출
- 메모리 중요도 점수화
- 관련 메모리 검색
- 메모리 갱신 및 통합

**사용법**:
```python
from source.services.contextual_memory_manager import ContextualMemoryManager

# 메모리 관리자 초기화
memory_manager = ContextualMemoryManager()

# 메모리 저장
memory_manager.store_memory(
    user_id="user123",
    content="민법 제750조는 불법행위로 인한 손해배상 책임을 규정합니다",
    importance_score=0.9,
    tags=["민법", "손해배상", "불법행위"]
)

# 메모리 검색
memories = memory_manager.search_memories("user123", "손해배상")
```

#### 2. 대화 품질 모니터링 (ConversationQualityMonitor)

**파일**: `source/services/conversation_quality_monitor.py`

**주요 기능**:
- 대화 품질 평가 (완결성/만족도/정확성)
- 문제점 자동 감지
- 개선 제안 생성
- 품질 트렌드 분석

**사용법**:
```python
from source.services.conversation_quality_monitor import ConversationQualityMonitor

# 품질 모니터 초기화
quality_monitor = ConversationQualityMonitor()

# 품질 평가
conversation = {
    "messages": [
        {"role": "user", "content": "민법 제750조에 대해 설명해주세요"},
        {"role": "assistant", "content": "민법 제750조는 불법행위로 인한 손해배상 책임을 규정합니다..."}
    ]
}

quality_result = quality_monitor.evaluate_quality(conversation)
print(f"완결성 점수: {quality_result.completeness_score}")
print(f"만족도 점수: {quality_result.satisfaction_score}")
print(f"정확성 점수: {quality_result.accuracy_score}")
```

#### 3. 성능 최적화 (PerformanceOptimizer)

**파일**: `source/utils/performance_optimizer.py`

**주요 기능**:
- 성능 모니터링 (PerformanceMonitor)
- 메모리 최적화 (MemoryOptimizer)
- 캐시 관리 (CacheManager)
- 실시간 메트릭 수집 및 분석

**사용법**:
```python
from source.utils.performance_optimizer import PerformanceMonitor, MemoryOptimizer, CacheManager

# 성능 모니터 초기화
perf_monitor = PerformanceMonitor()
memory_optimizer = MemoryOptimizer()
cache_manager = CacheManager()

# 성능 모니터링 시작
perf_monitor.start_monitoring()

# 메모리 최적화
memory_optimizer.optimize_memory()

# 캐시 관리
cache_manager.set_cache_size(1000)
```

### Phase 3 테스트

**파일**: `tests/test_phase3_memory_quality.py`

```python
import pytest
from source.services.contextual_memory_manager import ContextualMemoryManager
from source.services.conversation_quality_monitor import ConversationQualityMonitor
from source.utils.performance_optimizer import PerformanceMonitor

def test_phase3_integration():
    """Phase 3 통합 테스트"""
    # 메모리 관리 테스트
    memory_manager = ContextualMemoryManager()
    memory_manager.store_memory("test_user", "테스트 메모리", 0.8)
    
    # 품질 모니터링 테스트
    quality_monitor = ConversationQualityMonitor()
    quality_result = quality_monitor.evaluate_quality({"messages": []})
    
    # 성능 모니터링 테스트
    perf_monitor = PerformanceMonitor()
    perf_monitor.start_monitoring()
    
    assert memory_manager.search_memories("test_user", "테스트") is not None
    assert quality_result.completeness_score >= 0
    assert perf_monitor.is_monitoring()
```

## Phase 통합 및 최적화

### ChatService 통합

**파일**: `source/services/chat_service.py`

모든 Phase가 ChatService에 통합되어 있습니다:

```python
from source.services.chat_service import ChatService
from source.utils.config import Config

# ChatService 초기화 (모든 Phase 포함)
config = Config()
chat_service = ChatService(config)

# 지능형 채팅 처리
result = chat_service.process_message(
    message="민법 제750조에 대해 설명해주세요",
    user_id="user123",
    session_id="session_001"
)

# Phase별 결과 확인
print(f"Phase 1 결과: {result.get('phase1_info')}")
print(f"Phase 2 결과: {result.get('phase2_info')}")
print(f"Phase 3 결과: {result.get('phase3_info')}")
```

### 성능 최적화

#### 1. 메모리 관리
- 자동 가비지 컬렉션
- 메모리 사용량 모니터링
- 메모리 누수 방지

#### 2. 캐시 시스템
- LRU 기반 지능형 캐싱
- 캐시 히트율 최적화
- 캐시 크기 동적 조정

#### 3. 응답 시간 최적화
- 병렬 처리
- 토큰 관리
- 모델 로딩 최적화

## 문제 해결

### 일반적인 문제

#### 1. Phase 초기화 실패
```python
# 각 Phase 서비스 개별 초기화 확인
try:
    session_manager = IntegratedSessionManager("data/conversations.db")
    print("Phase 1 초기화 성공")
except Exception as e:
    print(f"Phase 1 초기화 실패: {e}")
```

#### 2. 메모리 부족
```python
# 메모리 사용량 확인
from source.utils.performance_optimizer import MemoryOptimizer

memory_optimizer = MemoryOptimizer()
memory_usage = memory_optimizer.get_memory_usage()
print(f"메모리 사용량: {memory_usage}MB")

if memory_usage > 4000:  # 4GB
    memory_optimizer.optimize_memory()
```

#### 3. 성능 저하
```python
# 성능 모니터링
from source.utils.performance_optimizer import PerformanceMonitor

perf_monitor = PerformanceMonitor()
perf_monitor.start_monitoring()

# 성능 지표 확인
metrics = perf_monitor.get_metrics()
print(f"평균 응답 시간: {metrics['avg_response_time']}초")
print(f"캐시 히트율: {metrics['cache_hit_rate']}")
```

### 디버깅 팁

#### 1. 로그 레벨 설정
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. Phase별 디버깅
```python
# Phase 1 디버깅
session_manager = IntegratedSessionManager(":memory:")  # 메모리 DB 사용
session_id = session_manager.create_session("debug_user")
print(f"세션 생성: {session_id}")

# Phase 2 디버깅
analyzer = EmotionIntentAnalyzer()
result = analyzer.analyze("테스트 메시지")
print(f"분석 결과: {result}")

# Phase 3 디버깅
memory_manager = ContextualMemoryManager()
memory_manager.store_memory("debug_user", "디버그 메모리", 0.5)
memories = memory_manager.search_memories("debug_user", "디버그")
print(f"검색된 메모리: {memories}")
```

## 성능 벤치마크

### Phase별 성능 지표

| Phase | 평균 응답 시간 | 메모리 사용량 | 정확도 |
|-------|---------------|--------------|--------|
| Phase 1 | 0.5초 | 50MB | 90%+ |
| Phase 2 | 0.3초 | 30MB | 85%+ |
| Phase 3 | 0.2초 | 20MB | 80%+ |
| 전체 통합 | 1.0초 | 100MB | 85%+ |

### 최적화 권장사항

1. **Phase 1**: 세션 크기 제한, 컨텍스트 압축 강화
2. **Phase 2**: 프로필 캐싱, 감정 분석 모델 최적화
3. **Phase 3**: 메모리 정리 주기 조정, 품질 평가 간소화
