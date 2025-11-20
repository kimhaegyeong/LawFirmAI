# LawFirmAI 리팩토링 계획서

## 📋 목차

1. [현재 상태 분석](#현재-상태-분석)
2. [리팩토링 목표](#리팩토링-목표)
3. [단계별 실행 계획](#단계별-실행-계획)
4. [마이그레이션 전략](#마이그레이션-전략)
5. [체크리스트](#체크리스트)
6. [예상 효과](#예상-효과)

---

## 현재 상태 분석

### 1.1 구조적 문제점

#### 거대한 파일들
- `core/services/unified_prompt_manager.py`: **2,993 lines**
- `core/generation/formatters/answer_structure_enhancer.py`: **3,243 lines**

#### 중복된 기능
- **대화 관리**: 
  - `core/services/conversation_manager.py` ↔ `core/conversation/conversation_manager.py`
  - `core/services/contextual_memory_manager.py` ↔ `core/conversation/contextual_memory_manager.py`
  - `core/services/integrated_session_manager.py` ↔ `core/conversation/integrated_session_manager.py`
  - `core/services/multi_turn_handler.py` ↔ `core/conversation/multi_turn_handler.py`

- **컨텍스트 관리**:
  - `core/services/context_manager.py` ↔ `core/agents/handlers/context_manager.py`
  - `core/services/context_builder.py` ↔ `core/agents/handlers/context_builder.py`
  - `core/services/context_compressor.py` ↔ `core/agents/handlers/context_compressor.py`

- **답변 생성**:
  - `core/services/answer_generator.py` ↔ `core/agents/handlers/answer_generator.py`
  - `core/services/answer_formatter.py` ↔ `core/agents/handlers/answer_formatter.py`

#### 폴더 구조 문제
- `core/services`: **57개 파일**이 한 폴더에 집중
- `core/agents`: 역할이 불명확 (워크플로우 전용인지, 일반 에이전트인지)
- 기능별 분리가 완전하지 않음

### 1.2 의존성 문제
- `core/services`와 `core/agents` 간 순환 의존성 가능성
- Import 경로 불일치 (`core.agents` vs `core.services`)
- Deprecation 경고가 있지만 완전히 정리되지 않음

---

## 리팩토링 목표

### 2.1 주요 목표
1. **가독성 향상**: 거대 파일을 작은 모듈로 분리
2. **중복 제거**: 중복된 기능 통합
3. **명확한 구조**: 도메인별 명확한 폴더 구조
4. **의존성 관리**: 순환 의존성 제거 및 명확한 의존성 방향
5. **유지보수성**: 코드 수정 시 영향 범위 최소화

### 2.2 최종 목표 구조

```
lawfirm_langgraph/
├── config/                    # 설정
├── core/
│   ├── workflow/             # LangGraph 워크플로우 (메인)
│   │   ├── nodes/
│   │   ├── state/
│   │   ├── edges/
│   │   ├── routes/
│   │   ├── subgraphs/
│   │   └── tools/
│   ├── search/               # 검색 시스템
│   │   ├── engines/
│   │   ├── handlers/
│   │   ├── processors/
│   │   └── optimizers/
│   ├── generation/           # 답변 생성
│   │   ├── generators/
│   │   ├── formatters/
│   │   ├── validators/
│   │   └── context/
│   ├── classification/       # 분류 시스템
│   │   ├── classifiers/
│   │   ├── handlers/
│   │   └── analyzers/
│   ├── processing/           # 데이터 처리
│   │   ├── extractors/
│   │   ├── processors/
│   │   ├── parsers/
│   │   └── integration/
│   ├── conversation/         # 대화 관리
│   │   ├── manager.py
│   │   ├── memory/
│   │   └── flow/
│   ├── services/             # 통합 서비스 (최소화)
│   │   ├── prompts/          # 프롬프트 관리
│   │   ├── chat_service.py
│   │   └── search_service.py
│   ├── agents/               # LangGraph 워크플로우 전용
│   │   ├── workflow/
│   │   ├── state/
│   │   ├── nodes/
│   │   └── tools/
│   ├── data/                 # 데이터 레이어
│   ├── shared/               # 공유 유틸리티
│   │   ├── cache/
│   │   ├── clients/
│   │   ├── monitoring/
│   │   └── utils/
│   └── utils/                # 유틸리티
└── tests/
```

---

## 단계별 실행 계획

### Phase 1: 거대 파일 분리 (우선순위: 높음)

#### 1.1 `unified_prompt_manager.py` 분리

**목표**: 2,993 lines → 여러 모듈로 분리

**새로운 구조**:
```
core/services/prompts/
├── __init__.py
├── manager.py              # UnifiedPromptManager (메인 클래스, ~200 lines)
├── loaders/
│   ├── __init__.py
│   ├── base_loader.py      # 기본 프롬프트 로더 (~300 lines)
│   ├── domain_loader.py    # 도메인 템플릿 로더 (~400 lines)
│   └── model_loader.py     # 모델 최적화 로더 (~200 lines)
├── templates/
│   ├── __init__.py
│   ├── base_templates.py   # 기본 프롬프트 템플릿 (~800 lines)
│   ├── domain_templates.py # 도메인별 템플릿 (~1000 lines)
│   └── question_templates.py # 질문 유형별 템플릿 (~300 lines)
└── optimizers/
    ├── __init__.py
    └── prompt_optimizer.py # 프롬프트 최적화 로직 (~200 lines)
```

**작업 내용**:
1. `core/services/prompts/` 디렉토리 생성
2. 프롬프트 로더 클래스 분리
3. 템플릿 정의 분리
4. 메인 매니저 클래스 리팩토링
5. 호환성을 위한 re-export 추가

#### 1.2 `answer_structure_enhancer.py` 분리

**목표**: 3,243 lines → 여러 모듈로 분리

**새로운 구조**:
```
core/generation/formatters/structure/
├── __init__.py
├── enhancer.py             # AnswerStructureEnhancer (메인, ~300 lines)
├── processors/
│   ├── __init__.py
│   ├── section_processor.py    # 섹션 처리 (~800 lines)
│   ├── citation_processor.py   # 인용 처리 (~600 lines)
│   ├── formatting_processor.py # 포맷팅 처리 (~700 lines)
│   └── validation_processor.py # 검증 처리 (~400 lines)
└── templates/
    ├── __init__.py
    └── structure_templates.py  # 구조 템플릿 (~400 lines)
```

**작업 내용**:
1. `core/generation/formatters/structure/` 디렉토리 생성
2. 프로세서 클래스 분리
3. 템플릿 정의 분리
4. 메인 enhancer 클래스 리팩토링
5. 호환성을 위한 re-export 추가

---

### Phase 2: 중복 제거 및 통합 (우선순위: 높음)

#### 2.1 대화 관리 통합

**목표**: 중복된 대화 관리 코드 통합

**새로운 구조**:
```
core/conversation/
├── __init__.py
├── manager.py              # ConversationManager (통합)
├── memory/
│   ├── __init__.py
│   ├── contextual_memory.py
│   └── session_memory.py
├── flow/
│   ├── __init__.py
│   ├── flow_tracker.py
│   └── quality_monitor.py
└── handlers/
    ├── __init__.py
    └── multi_turn_handler.py
```

**제거 대상**:
- `core/services/conversation_manager.py`
- `core/services/contextual_memory_manager.py`
- `core/services/conversation_flow_tracker.py`
- `core/services/conversation_quality_monitor.py`
- `core/services/integrated_session_manager.py`
- `core/services/multi_turn_handler.py`

#### 2.2 컨텍스트 관리 통합

**목표**: 중복된 컨텍스트 관리 코드 통합

**새로운 구조**:
```
core/generation/context/
├── __init__.py
├── manager.py              # ContextManager (통합)
├── builder.py              # ContextBuilder
├── compressor.py           # ContextCompressor
└── quality/
    ├── __init__.py
    └── enhancer.py         # ContextQualityEnhancer
```

**제거 대상**:
- `core/services/context_manager.py`
- `core/services/context_builder.py`
- `core/services/context_compressor.py`
- `core/services/context_quality_enhancer.py`
- `core/agents/handlers/context_manager.py`
- `core/agents/handlers/context_builder.py`
- `core/agents/handlers/context_compressor.py`
- `core/agents/handlers/context_quality_enhancer.py`

---

### Phase 3: `core/services` 폴더 정리 (우선순위: 중간)

#### 3.1 파일 재분류

**검색 관련** → `core/search/`:
- `exact_search_engine_v2.py` → `engines/exact_search_engine.py`
- `semantic_search_engine_v2.py` → `engines/semantic_search_engine.py`
- `hybrid_search_engine_v2.py` → `engines/hybrid_search_engine.py`
- `precedent_search_engine.py` → `engines/precedent_search_engine.py`
- `optimized_hybrid_search_engine.py` → `engines/optimized_hybrid_search_engine.py`
- `search_service.py` → `handlers/search_service.py`

**답변 생성 관련** → `core/generation/`:
- `answer_generator.py` → `generators/answer_generator.py` (통합)
- `improved_answer_generator.py` → `generators/improved_answer_generator.py`
- `answer_formatter.py` → `formatters/answer_formatter.py` (통합)
- `answer_quality_enhancer.py` → `validators/answer_quality_validator.py`

**분류 관련** → `core/classification/`:
- `question_classifier.py` → `classifiers/question_classifier.py` (통합)
- `hybrid_question_classifier.py` → `classifiers/hybrid_question_classifier.py`
- `semantic_domain_classifier.py` → `classifiers/semantic_domain_classifier.py`
- `optimized_hybrid_classifier.py` → `classifiers/optimized_hybrid_classifier.py`

**키워드/용어 관련** → `core/processing/`:
- `legal_term_extractor.py` → `extractors/legal_term_extractor.py`
- `legal_term_expander.py` → `extractors/legal_term_expander.py`
- `legal_term_validator.py` → `extractors/legal_term_validator.py`
- `multi_method_term_extractor.py` → `extractors/multi_method_term_extractor.py`
- `keyword_cache.py` → `shared/cache/keyword_cache.py`
- 기타 키워드 관련 파일들

**법률 관련** → `core/processing/legal/`:
- `legal_basis_validator.py` → `validators/legal_basis_validator.py`
- `legal_basis_integration_service.py` → `integration/legal_basis_integration.py`
- `legal_citation_enhancer.py` → `enhancers/legal_citation_enhancer.py`
- `legal_text_preprocessor.py` → `processors/legal_text_preprocessor.py`

**프롬프트 관련** → `core/services/prompts/` (Phase 1에서 생성)

**기타**:
- `gemini_client.py` → `shared/clients/gemini_client.py`
- `gemini_validation_pipeline.py` → `shared/clients/gemini_validation.py`
- `confidence_calculator.py` → `generation/validators/confidence_calculator.py`
- `document_processor.py` → `processing/processors/document_processor.py`
- `emotion_intent_analyzer.py` → `classification/analyzers/emotion_intent_analyzer.py`
- `result_merger.py` → `search/processors/result_merger.py`
- `integrated_cache_system.py` → `shared/cache/integrated_cache.py`
- `term_integration_system.py` → `processing/integration/term_integration.py`

#### 3.2 최종 `core/services` 구조

```
core/services/  # 최종적으로는 최소한의 통합 서비스만 유지
├── __init__.py
├── prompts/                 # 프롬프트 관리 (Phase 1에서 생성)
├── chat_service.py          # 통합 채팅 서비스 (유지)
└── search_service.py        # 통합 검색 서비스 (유지)
```

---

### Phase 4: `core/agents` 폴더 정리 (우선순위: 중간)

#### 4.1 agents 폴더 재정의

**목표**: LangGraph 워크플로우 전용으로 재정의

**새로운 구조**:
```
core/agents/  # LangGraph 워크플로우 전용
├── __init__.py
├── workflow/              # 워크플로우 관련
│   ├── routes.py
│   ├── utils.py
│   └── constants.py
├── state/                 # 상태 관리
│   ├── definitions.py
│   ├── helpers.py
│   ├── utils.py
│   └── reducers.py
├── nodes/                 # 노드 래퍼 및 유틸리티
│   ├── wrappers.py
│   └── specs.py
├── subgraphs/             # 서브그래프
└── tools/                 # Agentic AI Tools
```

**제거/이동 대상**:
- `handlers/` → `core/generation`, `core/search`, `core/classification`로 이동
- `extractors/` → `core/processing/extractors`로 이동
- `validators/` → `core/generation/validators`로 이동
- `parsers/` → `core/processing/parsers`로 이동
- `prompt_builders/` → `core/services/prompts/builders`로 이동
- `optimizers/` → `core/search/optimizers`, `core/generation/optimizers`로 이동

---

## 마이그레이션 전략

### 4.1 호환성 유지

**기존 코드 수정 최소화 원칙**:
- 기존 import 경로는 Deprecation 경고와 함께 re-export
- 충분한 시간(최소 2-3개월) 후 기존 경로 제거
- 점진적 마이그레이션 지원

**Re-export 예시**:
```python
# core/services/__init__.py
import warnings
from pathlib import Path

_DEPRECATED_IMPORTS = {
    'ConversationManager': 'core.conversation.manager',
    'ContextManager': 'core.generation.context.manager',
    # ... 기타
}

def __getattr__(name):
    if name in _DEPRECATED_IMPORTS:
        warnings.warn(
            f"{name}은(는) {_DEPRECATED_IMPORTS[name]}로 이동되었습니다. "
            f"새로운 경로를 사용하세요.",
            DeprecationWarning,
            stacklevel=2
        )
        # 실제 import 및 반환
        module_path = _DEPRECATED_IMPORTS[name]
        # ... import 로직
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
```

### 4.2 점진적 마이그레이션

1. **새 경로에 코드 작성**
2. **기존 경로에서 새 경로로 re-export**
3. **Deprecation 경고 추가**
4. **사용처를 새 경로로 점진적 변경**
5. **기존 경로 제거** (충분한 시간 후)

---

## 체크리스트

### Phase 1 체크리스트
- [ ] `unified_prompt_manager.py` 분리 완료
- [ ] `answer_structure_enhancer.py` 분리 완료
- [ ] 단일 책임 원칙 준수
- [ ] 순환 의존성 없음
- [ ] Import 경로 일관성
- [ ] 테스트 커버리지 유지
- [ ] 호환성 레이어 제공

### Phase 2 체크리스트
- [ ] 대화 관리 통합 완료
- [ ] 컨텍스트 관리 통합 완료
- [ ] 기능 비교 및 통합 검증
- [ ] 기존 사용처 확인
- [ ] 호환성 레이어 제공
- [ ] 테스트 업데이트

### Phase 3 체크리스트
- [ ] 파일 이동 및 재분류 완료
- [ ] Import 경로 업데이트
- [ ] Deprecation 경고 추가
- [ ] 도메인별 명확한 분리
- [ ] 의존성 방향 일관성 (상위 → 하위)
- [ ] 공통 유틸리티는 shared로 이동

### Phase 4 체크리스트
- [ ] agents 폴더 재구성 완료
- [ ] 워크플로우 전용으로 정리
- [ ] 불필요한 파일 제거
- [ ] Import 경로 업데이트

### 공통 체크리스트
- [ ] 모든 테스트 통과
- [ ] Linter 오류 없음
- [ ] 문서 업데이트
- [ ] 코드 리뷰 완료

---

## 예상 효과

### 5.1 가독성
- 거대 파일 분리로 코드 이해도 향상
- 파일당 평균 라인 수 감소 (목표: 500 lines 이하)

### 5.2 재사용성
- 기능별 모듈화로 재사용성 향상
- 명확한 인터페이스로 의존성 관리 개선

### 5.3 테스트 용이성
- 작은 단위로 테스트 작성 용이
- Mock 객체 사용 용이

### 5.4 의존성 관리
- 명확한 계층 구조로 의존성 관리 개선
- 순환 의존성 제거

### 5.5 확장성
- 새로운 기능 추가 시 적절한 위치 명확
- 코드 구조 이해도 향상

---

## 진행 상황

### Phase 1: 거대 파일 분리
- [ ] `unified_prompt_manager.py` 분리
- [ ] `answer_structure_enhancer.py` 분리

### Phase 2: 중복 제거
- [ ] 대화 관리 통합
- [ ] 컨텍스트 관리 통합

### Phase 3: services 정리
- [ ] 파일 재분류
- [ ] Import 경로 업데이트

### Phase 4: agents 정리
- [ ] 폴더 재구성
- [ ] 워크플로우 전용으로 정리

---

## 참고 사항

- **기존 코드 수정 최소화 원칙** 준수
- **점진적 리팩토링**으로 위험 최소화
- **각 단계마다 테스트 실행** 필수
- **문서화 업데이트** 필수
- **팀원과의 커뮤니케이션** 중요

---

**작성일**: 2024-12-19  
**최종 수정일**: 2024-12-19  
**버전**: 1.0

