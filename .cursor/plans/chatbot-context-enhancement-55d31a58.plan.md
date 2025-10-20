<!-- 55d31a58-3956-45cc-97d4-c5c8829918eb 42ba7054-7b97-4593-b475-973233aca141 -->
# LawFirmAI 챗봇 대화 맥락 강화 구현 계획

## 구현 범위

모든 Phase (1, 2, 3)를 포함한 완전한 대화 맥락 강화 시스템 구축

- 기존 `ConversationStore` 클래스 확장
- Gradio 앱에 동시 통합하여 즉시 확인 가능

## Phase 1: 핵심 대화 맥락 기능 (즉시 구현)

### 1. 영구적 세션 저장소 강화

**파일**: `source/data/conversation_store.py`

현재 `ConversationStore`는 기본 기능만 구현되어 있음. 다음 기능 추가:

- 사용자별 세션 관리 (user_id 기반)
- 세션 메타데이터 확장 (device_info, location, preferences)
- 자동 백업 및 복구 기능
- 세션 검색 기능 (키워드, 날짜, 엔티티 기반)
```python
# 추가할 메서드들
def get_user_sessions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]
def search_sessions(self, query: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]
def backup_session(self, session_id: str, backup_path: str) -> bool
def restore_session(self, backup_path: str) -> Optional[str]
```


### 2. 통합 세션 관리자 생성

**새 파일**: `source/services/integrated_session_manager.py`

`ConversationManager`와 `ConversationStore`를 통합하는 새로운 관리자:

- 메모리와 DB를 동시에 관리
- 자동 동기화 (메모리 → DB)
- 세션 복원 시 DB → 메모리 로드
- 캐시 전략 구현
```python
class IntegratedSessionManager:
    def __init__(self):
        self.conversation_manager = ConversationManager()
        self.conversation_store = ConversationStore()
        self.sync_interval = 5  # 5턴마다 DB 동기화
    
    def add_turn(self, session_id: str, user_query: str, bot_response: str, 
                 question_type: Optional[str] = None) -> ConversationContext
    def get_or_create_session(self, session_id: str, user_id: Optional[str] = None) -> ConversationContext
    def sync_to_database(self, session_id: str) -> bool
    def load_from_database(self, session_id: str) -> Optional[ConversationContext]
```


### 3. 다중 턴 질문 처리기

**새 파일**: `source/services/multi_turn_handler.py`

대명사 해결 및 불완전한 질문을 완성하는 기능:

- 대명사 감지 (그것, 이것, 위의, 해당, 앞서 등)
- 맥락 기반 질문 재구성
- 생략된 주어/목적어 복원
- 연속 질문 패턴 인식
```python
class MultiTurnQuestionHandler:
    def detect_multi_turn_question(self, query: str, context: ConversationContext) -> bool
    def resolve_pronouns(self, query: str, context: ConversationContext) -> str
    def build_complete_query(self, query: str, context: ConversationContext) -> Dict[str, Any]
    def extract_reference_entities(self, query: str) -> List[str]
```


### 4. 컨텍스트 압축기

**새 파일**: `source/services/context_compressor.py`

긴 대화를 요약하여 토큰 제한 문제 해결:

- 중요 정보 추출 (법률 엔티티, 핵심 주제)
- 대화 요약 생성
- 토큰 수 계산 및 관리
- 우선순위 기반 컨텍스트 선택
```python
class ContextCompressor:
    def compress_long_conversation(self, context: ConversationContext, 
                                   max_tokens: int = 2000) -> str
    def extract_key_information(self, turns: List[ConversationTurn]) -> Dict[str, Any]
    def maintain_relevant_context(self, context: ConversationContext, 
                                  current_query: str) -> List[ConversationTurn]
    def calculate_tokens(self, text: str) -> int
```


## Phase 2: 개인화 및 지능형 분석 (중기 구현)

### 5. 사용자 프로필 관리자

**새 파일**: `source/services/user_profile_manager.py`

**DB 테이블 추가**: `conversation_store.py`에 user_profiles 테이블 추가

사용자별 선호도 및 전문성 수준 관리:

- 사용자 프로필 생성/업데이트
- 전문성 수준 추적 (초보/중급/전문가)
- 선호하는 답변 스타일 저장
- 관심 법률 분야 추적
```python
# conversation_store.py에 추가할 테이블
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    expertise_level TEXT DEFAULT 'beginner',
    preferred_detail_level TEXT DEFAULT 'medium',
    preferred_language TEXT DEFAULT 'ko',
    interest_areas TEXT,  -- JSON
    created_at TIMESTAMP,
    last_updated TIMESTAMP
)

class UserProfileManager:
    def create_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool
    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]
    def update_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool
    def track_expertise_level(self, user_id: str, question_history: List[Dict]) -> str
    def get_personalized_context(self, user_id: str, query: str) -> Dict[str, Any]
```


### 6. 감정 및 의도 분석기

**새 파일**: `source/services/emotion_intent_analyzer.py`

사용자 감정과 의도를 파악하여 적절한 응답 톤 결정:

- 감정 분석 (긴급/불안/화남/만족 등)
- 의도 분석 (질문/요청/불만/감사 등)
- 응답 톤 조정 (공감적/전문적/간결함 등)
- 긴급도 평가
```python
class EmotionIntentAnalyzer:
    def analyze_emotion(self, text: str) -> Dict[str, float]
    def analyze_intent(self, text: str, context: ConversationContext) -> Dict[str, Any]
    def get_contextual_response_tone(self, emotion: str, intent: str, 
                                     user_profile: Dict) -> str
    def assess_urgency(self, text: str, emotion: Dict) -> str  # low/medium/high/critical
```


### 7. 대화 흐름 추적기

**새 파일**: `source/services/conversation_flow_tracker.py`

대화 패턴을 학습하고 다음 질문을 예측:

- 대화 흐름 패턴 인식
- 다음 의도 예측
- 후속 질문 제안
- 대화 분기점 감지
```python
class ConversationFlowTracker:
    def track_conversation_flow(self, session_id: str, turn: ConversationTurn) -> None
    def predict_next_intent(self, context: ConversationContext) -> List[str]
    def suggest_follow_up_questions(self, context: ConversationContext) -> List[str]
    def detect_conversation_branch(self, context: ConversationContext) -> Optional[str]
    def analyze_flow_patterns(self, sessions: List[ConversationContext]) -> Dict[str, Any]
```


## Phase 3: 장기 기억 및 품질 모니터링 (장기 구현)

### 8. 맥락적 메모리 관리자

**새 파일**: `source/services/contextual_memory_manager.py`

**DB 테이블 추가**: `conversation_store.py`에 contextual_memories 테이블 추가

중요한 사실과 정보를 장기 기억으로 저장:

- 중요 사실 자동 추출
- 메모리 중요도 점수화
- 관련 메모리 검색
- 메모리 갱신 및 통합
```python
# conversation_store.py에 추가할 테이블
CREATE TABLE IF NOT EXISTS contextual_memories (
    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    user_id TEXT,
    memory_type TEXT,  -- fact/preference/case_detail
    memory_content TEXT,
    importance_score REAL,
    created_at TIMESTAMP,
    last_accessed TIMESTAMP,
    access_count INTEGER,
    related_entities TEXT  -- JSON
)

class ContextualMemoryManager:
    def store_important_facts(self, session_id: str, user_id: str, 
                              facts: Dict[str, Any]) -> bool
    def retrieve_relevant_memory(self, session_id: str, query: str, 
                                 user_id: Optional[str] = None) -> List[Dict]
    def update_memory_importance(self, memory_id: str, importance: float) -> bool
    def consolidate_memories(self, user_id: str) -> int
    def extract_facts_from_conversation(self, turn: ConversationTurn) -> List[Dict]
```


### 9. 대화 품질 모니터

**새 파일**: `source/services/conversation_quality_monitor.py`

**DB 테이블 추가**: `conversation_store.py`에 quality_metrics 테이블 추가

대화 품질을 모니터링하고 개선점 제안:

- 대화 품질 평가 (완결성/만족도/정확성)
- 문제점 자동 감지
- 개선 제안 생성
- 품질 트렌드 분석
```python
# conversation_store.py에 추가할 테이블
CREATE TABLE IF NOT EXISTS quality_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    turn_id INTEGER,
    completeness_score REAL,
    satisfaction_score REAL,
    accuracy_score REAL,
    response_time REAL,
    issues_detected TEXT,  -- JSON
    timestamp TIMESTAMP
)

class ConversationQualityMonitor:
    def assess_conversation_quality(self, context: ConversationContext) -> Dict[str, Any]
    def detect_conversation_issues(self, context: ConversationContext) -> List[str]
    def suggest_improvements(self, context: ConversationContext) -> List[str]
    def analyze_quality_trends(self, session_ids: List[str]) -> Dict[str, Any]
    def calculate_turn_quality(self, turn: ConversationTurn, context: ConversationContext) -> float
```


## Gradio 앱 통합

### 10. ChatService 업데이트

**파일**: `source/services/chat_service.py`

새로운 기능들을 `ChatService`에 통합:

```python
class ChatService:
    def __init__(self, config: Config):
        # 기존 초기화...
        
        # Phase 1 컴포넌트
        self.session_manager = IntegratedSessionManager()
        self.multi_turn_handler = MultiTurnQuestionHandler()
        self.context_compressor = ContextCompressor()
        
        # Phase 2 컴포넌트
        self.user_profile_manager = UserProfileManager()
        self.emotion_intent_analyzer = EmotionIntentAnalyzer()
        self.conversation_flow_tracker = ConversationFlowTracker()
        
        # Phase 3 컴포넌트
        self.contextual_memory_manager = ContextualMemoryManager()
        self.quality_monitor = ConversationQualityMonitor()
    
    async def process_message(self, message: str, context: Optional[str] = None, 
                             session_id: Optional[str] = None, 
                             user_id: Optional[str] = None) -> Dict[str, Any]:
        # 1. 세션 로드 또는 생성
        # 2. 사용자 프로필 로드
        # 3. 다중 턴 질문 처리
        # 4. 감정/의도 분석
        # 5. 맥락적 메모리 검색
        # 6. 질문 처리 및 답변 생성
        # 7. 컨텍스트 압축 (필요시)
        # 8. 품질 평가
        # 9. 대화 흐름 추적
        # 10. 세션 저장
```

### 11. Gradio UI 업데이트

**파일**: `gradio/app.py`

사용자 인터페이스에 새로운 기능 노출:

```python
# 추가할 UI 컴포넌트
with gr.Tabs():
    with gr.Tab("채팅"):
        # 기존 채팅 인터페이스
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox()
        
        # 새로운 컨트롤
        with gr.Accordion("고급 설정", open=False):
            user_id_input = gr.Textbox(label="사용자 ID (선택)")
            detail_level = gr.Radio(["간단", "보통", "상세"], label="답변 상세도")
            show_follow_ups = gr.Checkbox(label="후속 질문 제안 표시")
    
    with gr.Tab("대화 이력"):
        session_list = gr.Dataframe(label="세션 목록")
        load_session_btn = gr.Button("세션 불러오기")
    
    with gr.Tab("사용자 프로필"):
        expertise_level = gr.Radio(["초보", "중급", "전문가"], label="전문성 수준")
        interest_areas = gr.CheckboxGroup(["민법", "형법", "상법", ...], label="관심 분야")
        save_profile_btn = gr.Button("프로필 저장")
    
    with gr.Tab("통계 및 분석"):
        quality_metrics = gr.JSON(label="대화 품질 지표")
        conversation_insights = gr.JSON(label="대화 인사이트")
```

## 구현 순서

1. **Phase 1 기반 구조** (1-4일)

   - `ConversationStore` DB 스키마 확장
   - `IntegratedSessionManager` 구현
   - `MultiTurnQuestionHandler` 구현
   - `ContextCompressor` 구현

2. **Phase 1 통합 테스트** (1일)

   - 단위 테스트 작성
   - 통합 테스트 수행
   - Gradio 기본 통합

3. **Phase 2 개인화 기능** (2-3일)

   - `UserProfileManager` 구현
   - `EmotionIntentAnalyzer` 구현
   - `ConversationFlowTracker` 구현

4. **Phase 2 통합 테스트** (1일)

   - Phase 2 기능 테스트
   - Gradio UI 확장

5. **Phase 3 고급 기능** (2-3일)

   - `ContextualMemoryManager` 구현
   - `ConversationQualityMonitor` 구현

6. **Phase 3 통합 및 최종 테스트** (1-2일)

   - 전체 시스템 통합
   - 성능 최적화
   - 최종 Gradio UI 완성

7. **문서화 및 배포 준비** (1일)

   - API 문서 작성
   - 사용자 가이드 작성
   - 배포 설정 업데이트

## 주요 파일 목록

### 새로 생성할 파일

- `source/services/integrated_session_manager.py`
- `source/services/multi_turn_handler.py`
- `source/services/context_compressor.py`
- `source/services/user_profile_manager.py`
- `source/services/emotion_intent_analyzer.py`
- `source/services/conversation_flow_tracker.py`
- `source/services/contextual_memory_manager.py`
- `source/services/conversation_quality_monitor.py`

### 수정할 파일

- `source/data/conversation_store.py` - DB 스키마 및 메서드 확장
- `source/services/chat_service.py` - 새 기능 통합
- `source/services/conversation_manager.py` - `IntegratedSessionManager`와 호환
- `gradio/app.py` - UI 업데이트 및 기능 노출

### 테스트 파일

- `tests/test_integrated_session_manager.py`
- `tests/test_multi_turn_handler.py`
- `tests/test_context_compressor.py`
- `tests/test_user_profile_manager.py`
- `tests/test_emotion_intent_analyzer.py`
- `tests/test_conversation_flow_tracker.py`
- `tests/test_contextual_memory_manager.py`
- `tests/test_conversation_quality_monitor.py`

## 성공 지표

- 다중 턴 질문 처리 정확도 > 85%
- 세션 저장/복원 성공률 100%
- 컨텍스트 압축으로 토큰 사용량 30% 감소
- 사용자 프로필 기반 개인화 적용률 > 90%
- 대화 품질 점수 평균 > 80%
- 응답 시간 증가 < 10% (기존 대비)

### To-dos

- [ ] ConversationStore DB 스키마 확장 (user_profiles, quality_metrics, contextual_memories 테이블 추가)
- [ ] IntegratedSessionManager 구현 (메모리-DB 통합 관리)
- [ ] MultiTurnQuestionHandler 구현 (대명사 해결, 질문 재구성)
- [ ] ContextCompressor 구현 (대화 압축, 토큰 관리)
- [ ] Phase 1 단위 및 통합 테스트 작성
- [ ] Phase 1 기능 Gradio 앱에 기본 통합
- [ ] UserProfileManager 구현 (사용자 프로필, 전문성 추적)
- [ ] EmotionIntentAnalyzer 구현 (감정/의도 분석)
- [ ] ConversationFlowTracker 구현 (대화 흐름 추적, 후속 질문 제안)
- [ ] Phase 2 단위 및 통합 테스트 작성
- [ ] Gradio UI 확장 (프로필 관리, 고급 설정)
- [ ] ContextualMemoryManager 구현 (장기 기억 관리)
- [ ] ConversationQualityMonitor 구현 (품질 모니터링)
- [ ] Phase 3 단위 및 통합 테스트 작성
- [ ] ChatService에 모든 새 기능 통합
- [ ] Gradio 최종 UI 완성 (대화 이력, 통계 탭)
- [ ] 성능 최적화 및 메모리 관리
- [ ] API 문서 및 사용자 가이드 작성