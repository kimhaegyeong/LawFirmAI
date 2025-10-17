<!-- 0f9c520e-cde8-4420-876f-029ffb4531a0 9121c223-21ea-4aef-9b55-b1a22b585ed7 -->
# 법률 챗봇 개선 - 1단계 및 2단계 구현 계획

## 1단계: 즉시 개선 가능한 핵심 기능 구현

### Phase 1.1: 질문 유형 분류기 구현

**목표:** 사용자 질문을 분석하여 법률/판례 검색 비중을 조정

**구현 파일:**

- `source/services/question_classifier.py` (신규)
                                - 질문 유형 분류 (법률 조회, 판례 검색, 법적 조언, 절차 안내 등)
                                - 키워드 및 패턴 기반 분류
                                - 법률/판례 검색 가중치 결정

**주요 기능:**

```python
class QuestionClassifier:
    def classify_question(self, query: str) -> QuestionType:
        # "판례 찾아줘" -> QuestionType.PRECEDENT_SEARCH (precedent_weight=0.8)
        # "민법 제750조" -> QuestionType.LAW_INQUIRY (law_weight=0.8)
        # "손해배상 청구 방법" -> QuestionType.LEGAL_ADVICE (balanced)
```

### Phase 1.2: 통합 검색 엔진 개선

**목표:** 법률과 판례를 동적 가중치로 통합 검색

**수정 파일:**

- `source/services/hybrid_search_engine.py`
                                - 질문 유형별 검색 가중치 적용
                                - 법률 인덱스와 판례 인덱스 통합 검색
                                - 결과 재랭킹 로직 추가

**구현 내용:**

```python
def search_with_question_type(self, query: str, question_type: QuestionType):
    # 법률 검색
    law_results = self.search_laws(query, weight=question_type.law_weight)
    
    # 판례 검색 (민사 판례 인덱스)
    precedent_results = self.search_precedents(
        query, 
        category='civil',
        weight=question_type.precedent_weight
    )
    
    # 통합 및 재랭킹
    return self.merge_and_rerank(law_results, precedent_results)
```

### Phase 1.3: 판례 전용 검색 엔진 구현

**목표:** 판례 데이터베이스와 벡터 인덱스를 활용한 전문 검색

**신규 파일:**

- `source/services/precedent_search_engine.py`
                                - precedent_cases, precedent_sections 테이블 검색
                                - 판례 벡터 인덱스 활용 (ml_enhanced_ko_sroberta_precedents)
                                - 카테고리별 검색 (민사/형사/가사)

**주요 기능:**

```python
class PrecedentSearchEngine:
    def __init__(self):
        self.db = DatabaseManager()
        self.vector_store = LegalVectorStore(
            index_path="data/embeddings/ml_enhanced_ko_sroberta_precedents"
        )
    
    def search_precedents(self, query: str, category: str = 'civil', top_k: int = 5):
        # FTS5 검색 + 벡터 검색
        # 판시사항, 판결요지 우선 반환
```

### Phase 1.4: 질문 유형별 프롬프트 템플릿

**목표:** 각 질문 유형에 최적화된 프롬프트 생성

**수정/신규 파일:**

- `gradio/prompt_manager.py` (확장)
- `source/services/prompt_templates.py` (신규)

**구현 내용:**

```python
PROMPT_TEMPLATES = {
    "precedent_search": """당신은 판례 전문가입니다. 
    
관련 판례:
{precedent_list}

위 판례를 바탕으로 다음과 같이 답변하세요:
1. 가장 유사한 판례 3개 소개 (사건번호, 판결요지)
2. 해당 사안에의 적용 가능성
3. 실무적 시사점
""",
    
    "law_explanation": """당신은 법률 해설 전문가입니다.
    
관련 법률:
{law_articles}

위 법률을 다음 순서로 설명하세요:
1. 법률의 목적 및 취지
2. 주요 내용을 쉬운 말로 풀이
3. 실제 적용 예시
4. 주의사항
""",
    
    "legal_advice": """당신은 법률 상담 전문가입니다.
    
관련 법률 및 판례:
{context}

다음 구조로 조언하세요:
1. 상황 정리
2. 적용 가능한 법률 (조문 명시)
3. 관련 판례 (핵심 판결요지)
4. 권리 구제 방법 (단계별)
5. 필요한 증거 자료
"""
}
```

### Phase 1.5: 신뢰도 기반 답변 시스템

**목표:** 검색 결과 품질을 기반으로 답변 신뢰도 표시

**신규 파일:**

- `source/services/confidence_calculator.py`

**구현 내용:**

```python
class ConfidenceCalculator:
    def calculate_confidence(self, query: str, retrieved_docs: List[Dict], answer: str):
        # 1. 검색 결과 유사도 점수
        similarity_score = self._calc_similarity_score(retrieved_docs)
        
        # 2. 법률/판례 매칭 정확도
        matching_score = self._calc_matching_score(query, retrieved_docs)
        
        # 3. 답변 길이 및 구조 품질
        answer_quality = self._calc_answer_quality(answer)
        
        confidence = (similarity_score * 0.4 + matching_score * 0.4 + answer_quality * 0.2)
        
        return {
            "confidence": confidence,
            "reliability_level": self._get_reliability_level(confidence),
            "warnings": self._generate_warnings(confidence)
        }
    
    def _get_reliability_level(self, confidence: float):
        if confidence >= 0.8:
            return "HIGH"
        elif confidence >= 0.6:
            return "MEDIUM"
        else:
            return "LOW - 전문가 상담 권장"
```

### Phase 1.6: FastAPI 엔드포인트 통합

**목표:** 개선된 기능을 API로 제공

**수정 파일:**

- `source/api/endpoints.py`

**추가 엔드포인트:**

```python
@api_router.post("/chat/intelligent", response_model=IntelligentChatResponse)
async def intelligent_chat_endpoint(request: IntelligentChatRequest):
    # 1. 질문 분류
    question_type = question_classifier.classify(request.message)
    
    # 2. 통합 검색
    search_results = hybrid_search_engine.search_with_question_type(
        request.message, question_type
    )
    
    # 3. 프롬프트 생성
    prompt = prompt_template_manager.get_template(question_type.type)
    context = context_builder.build(search_results, question_type)
    
    # 4. 답변 생성 (Ollama)
    answer = ollama_client.generate(prompt, context)
    
    # 5. 신뢰도 계산
    confidence_info = confidence_calculator.calculate(
        request.message, search_results, answer
    )
    
    return IntelligentChatResponse(
        answer=answer,
        question_type=question_type.name,
        confidence=confidence_info,
        sources=self._format_sources(search_results),
        law_sources=[...],
        precedent_sources=[...]
    )
```

## 2단계: 단기 개선 기능 구현

### Phase 2.1: 법률 용어 확장 검색

**목표:** 법률 전문 용어와 일반 용어 간 매핑으로 검색 품질 향상

**신규 파일:**

- `source/services/legal_term_expander.py`
- `data/legal_term_dictionary.json` (법률 용어 사전)

**구현 내용:**

```python
class LegalTermExpander:
    def __init__(self):
        self.term_dict = self._load_legal_terms()
        # "손해배상" -> ["불법행위", "채무불이행", "위자료", "민법 제750조"]
        # "임대차" -> ["전세", "월세", "보증금", "주택임대차보호법"]
    
    def expand_query(self, query: str):
        # 원본 쿼리 + 관련 법률 용어 + 관련 법조문
        expanded = self._extract_and_expand_terms(query)
        return {
            "original": query,
            "expanded_terms": expanded,
            "related_laws": self._find_related_laws(expanded)
        }
```

**데이터 구조:**

```json
{
  "손해배상": {
    "synonyms": ["배상", "보상", "위자료", "손해전보"],
    "related_terms": ["불법행위", "채무불이행", "과실"],
    "related_laws": ["민법 제750조", "민법 제751조", "민법 제393조"],
    "precedent_keywords": ["손해배상청구권", "배상책임"]
  }
}
```

### Phase 2.2: 대화 맥락 관리 시스템

**목표:** 이전 대화를 기억하고 연속된 질문에 대응

**신규 파일:**

- `source/services/conversation_manager.py`
- `source/data/conversation_store.py`

**구현 내용:**

```python
class ConversationManager:
    def __init__(self):
        self.sessions = {}  # session_id -> ConversationContext
        self.db = ConversationStore()
    
    def add_turn(self, session_id: str, user_query: str, bot_response: str):
        context = self.sessions.get(session_id, ConversationContext())
        context.add_turn(user_query, bot_response)
        
        # 법률 엔티티 추출 및 저장
        entities = self._extract_legal_entities(user_query, bot_response)
        context.update_entities(entities)
        
        self.sessions[session_id] = context
    
    def get_relevant_context(self, session_id: str, current_query: str):
        context = self.sessions.get(session_id)
        if not context:
            return None
        
        # 현재 질문과 관련된 이전 대화 추출
        return context.get_relevant_turns(current_query, max_turns=3)
    
    def _extract_legal_entities(self, query: str, response: str):
        # 법률명, 조문번호, 판례번호, 법률용어 추출
        return {
            "laws": self._extract_law_names(query + response),
            "articles": self._extract_article_numbers(query + response),
            "precedents": self._extract_case_numbers(query + response),
            "legal_terms": self._extract_legal_terms(query + response)
        }

class ConversationContext:
    def __init__(self):
        self.turns = []  # [(user_query, bot_response, timestamp), ...]
        self.entities = {
            "laws": set(),
            "articles": set(),
            "precedents": set(),
            "legal_terms": set()
        }
        self.topic_stack = []  # 대화 주제 추적
```

### Phase 2.3: 답변 구조화 개선

**목표:** 일관된 형식의 구조화된 답변 제공

**신규 파일:**

- `source/services/answer_formatter.py`

**구현 내용:**

```python
class AnswerFormatter:
    def format_answer(self, 
                     raw_answer: str,
                     question_type: QuestionType,
                     sources: Dict[str, List],
                     confidence: ConfidenceInfo):
        
        if question_type == QuestionType.PRECEDENT_SEARCH:
            return self._format_precedent_answer(raw_answer, sources, confidence)
        elif question_type == QuestionType.LAW_EXPLANATION:
            return self._format_law_explanation(raw_answer, sources, confidence)
        else:
            return self._format_general_answer(raw_answer, sources, confidence)
    
    def _format_precedent_answer(self, answer: str, sources: Dict, confidence: ConfidenceInfo):
        formatted = f"""
## 관련 판례 분석

{answer}

### 📋 참고 판례

{self._format_precedent_sources(sources['precedents'])}

### ⚖️ 적용 가능한 법률

{self._format_law_sources(sources['laws'])}

### 💡 신뢰도 정보
- 신뢰도: {confidence.confidence:.1%}
- 수준: {confidence.reliability_level}
{self._format_warnings(confidence.warnings)}

---
💼 본 답변은 일반적인 법률 정보 제공을 목적으로 하며, 개별 사안에 대한 법률 자문이 아닙니다.
구체적인 법률 문제는 변호사와 직접 상담하시기 바랍니다.
"""
        return formatted
    
    def _format_precedent_sources(self, precedents: List[Dict]):
        result = []
        for i, prec in enumerate(precedents[:5], 1):
            result.append(f"""
{i}. **{prec['case_name']}** ({prec['case_number']})
   - 법원: {prec['court']}
   - 판결일: {prec['decision_date']}
   - 판결요지: {prec['summary'][:200]}...
   - 유사도: {prec['similarity']:.1%}
""")
        return "\n".join(result)
```

### Phase 2.4: 컨텍스트 윈도우 최적화

**목표:** 질문 유형에 따라 최적화된 컨텍스트 구성

**수정 파일:**

- `source/services/context_manager.py` (확장)

**구현 내용:**

```python
class ContextBuilder:
    def build_context_by_question_type(self, 
                                      question_type: QuestionType,
                                      search_results: Dict,
                                      conversation_context: Optional[ConversationContext]):
        
        if question_type == QuestionType.PRECEDENT_SEARCH:
            return self._build_precedent_context(search_results)
        elif question_type == QuestionType.LAW_EXPLANATION:
            return self._build_law_explanation_context(search_results)
        elif question_type == QuestionType.LEGAL_ADVICE:
            return self._build_advice_context(search_results, conversation_context)
    
    def _build_precedent_context(self, results: Dict):
        # 판례 중심 컨텍스트
        context = "=== 관련 판례 ===\n\n"
        
        for prec in results['precedents'][:3]:
            context += f"""
사건번호: {prec['case_number']}
법원: {prec['court']} / 판결일: {prec['decision_date']}

[판시사항]
{prec.get('judgment_summary', '')}

[판결요지]
{prec.get('judgment_gist', '')}

---
"""
        
        # 관련 법률도 간략히 포함
        if results.get('laws'):
            context += "\n=== 적용 법률 ===\n\n"
            for law in results['laws'][:2]:
                context += f"{law['law_name']} {law['article_number']}\n{law['content'][:300]}...\n\n"
        
        return context
    
    def _build_advice_context(self, results: Dict, conv_context: Optional[ConversationContext]):
        # 법률 + 판례 + 대화 맥락 통합
        context = ""
        
        # 이전 대화에서 언급된 법률/판례 우선 포함
        if conv_context:
            context += "=== 이전 대화 맥락 ===\n"
            for turn in conv_context.get_relevant_turns():
                context += f"Q: {turn['query']}\nA: {turn['response'][:200]}...\n\n"
        
        # 법률 우선, 판례는 보조
        context += "=== 적용 법률 ===\n"
        for law in results['laws'][:3]:
            context += f"{law['law_name']} {law['article_number']}\n{law['content']}\n\n"
        
        context += "=== 참고 판례 ===\n"
        for prec in results['precedents'][:2]:
            context += f"{prec['case_number']}: {prec['summary']}\n\n"
        
        return context
```

### Phase 2.5: Ollama 통합 및 응답 생성 개선

**목표:** Ollama Qwen2.5:7b 모델과 최적 통합

**신규/수정 파일:**

- `source/services/ollama_client.py` (신규)
- `source/services/answer_generator.py` (수정)

**구현 내용:**

```python
class OllamaClient:
    def __init__(self, model_name: str = "qwen2.5:7b"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
    
    def generate(self, prompt: str, context: str, temperature: float = 0.7):
        full_prompt = f"""{prompt}

컨텍스트:
{context}

위 정보를 바탕으로 전문적이고 구조화된 답변을 작성하세요."""
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": full_prompt,
                "temperature": temperature,
                "stream": False
            }
        )
        
        return response.json()['response']

class ImprovedAnswerGenerator:
    def __init__(self):
        self.ollama = OllamaClient()
        self.formatter = AnswerFormatter()
        self.confidence_calc = ConfidenceCalculator()
    
    def generate_answer(self, 
                       query: str,
                       question_type: QuestionType,
                       context: str,
                       sources: Dict):
        
        # 질문 유형별 프롬프트 선택
        prompt = PROMPT_TEMPLATES[question_type.template_key]
        
        # Ollama로 답변 생성
        raw_answer = self.ollama.generate(prompt, context)
        
        # 신뢰도 계산
        confidence = self.confidence_calc.calculate(query, sources, raw_answer)
        
        # 답변 구조화
        formatted_answer = self.formatter.format_answer(
            raw_answer, question_type, sources, confidence
        )
        
        return {
            "answer": formatted_answer,
            "raw_answer": raw_answer,
            "confidence": confidence,
            "question_type": question_type.name
        }
```

### Phase 2.6: 통합 테스트 및 엔드포인트 최종 구성

**목표:** 모든 개선 사항을 통합하여 최종 API 제공

**수정 파일:**

- `source/api/endpoints.py` (최종 통합)

**최종 엔드포인트 구조:**

```python
@api_router.post("/chat/v2", response_model=ChatV2Response)
async def chat_v2_endpoint(request: ChatV2Request):
    """개선된 법률 챗봇 API v2"""
    
    # 1. 대화 맥락 로드
    conversation_context = None
    if request.session_id:
        conversation_context = conversation_manager.get_relevant_context(
            request.session_id, request.message
        )
    
    # 2. 질문 분류
    question_type = question_classifier.classify_question(request.message)
    
    # 3. 법률 용어 확장
    expanded_query = legal_term_expander.expand_query(request.message)
    
    # 4. 통합 검색 (법률 + 판례)
    search_results = unified_search_engine.search(
        query=expanded_query,
        question_type=question_type,
        top_k=10
    )
    
    # 5. 컨텍스트 구성
    context = context_builder.build_context_by_question_type(
        question_type, search_results, conversation_context
    )
    
    # 6. 답변 생성 (Ollama)
    answer_result = improved_answer_generator.generate_answer(
        query=request.message,
        question_type=question_type,
        context=context,
        sources=search_results
    )
    
    # 7. 대화 맥락 저장
    if request.session_id:
        conversation_manager.add_turn(
            request.session_id,
            request.message,
            answer_result['answer']
        )
    
    return ChatV2Response(
        answer=answer_result['answer'],
        question_type=question_type.name,
        confidence=answer_result['confidence'],
        law_sources=search_results['laws'],
        precedent_sources=search_results['precedents'],
        conversation_context_used=conversation_context is not None,
        expanded_terms=expanded_query['expanded_terms']
    )
```

## 구현 순서 및 우선순위

### Week 1: 1단계 핵심 기능

1. Phase 1.1: 질문 분류기
2. Phase 1.3: 판례 검색 엔진
3. Phase 1.2: 통합 검색 개선

### Week 2: 1단계 완성

4. Phase 1.4: 프롬프트 템플릿
5. Phase 1.5: 신뢰도 시스템
6. Phase 1.6: API 통합

### Week 3: 2단계 시작

7. Phase 2.1: 법률 용어 확장
8. Phase 2.5: Ollama 통합

### Week 4: 2단계 완성

9. Phase 2.2: 대화 맥락 관리
10. Phase 2.3: 답변 구조화
11. Phase 2.4: 컨텍스트 최적화
12. Phase 2.6: 최종 통합

## 핵심 개선 효과

1. **검색 정확도**: 질문 유형별 가중치로 법률/판례 검색 최적화
2. **답변 품질**: 구조화된 프롬프트와 신뢰도 표시로 신뢰성 향상
3. **사용자 경험**: 대화 맥락 유지로 연속 질문 대응
4. **확장성**: 법률 용어 사전으로 지속적 검색 품질 개선

### To-dos

- [ ] 질문 유형 분류기 구현 (QuestionClassifier)
- [ ] 판례 전용 검색 엔진 구현 (PrecedentSearchEngine)
- [ ] 통합 검색 엔진 개선 (HybridSearchEngine 확장)
- [ ] 질문 유형별 프롬프트 템플릿 추가
- [ ] 신뢰도 기반 답변 시스템 구현 (ConfidenceCalculator)
- [ ] FastAPI 엔드포인트 통합 (intelligent chat endpoint)
- [ ] 법률 용어 확장 검색 시스템 구현 (LegalTermExpander)
- [ ] Ollama 클라이언트 및 답변 생성 개선
- [ ] 대화 맥락 관리 시스템 구현 (ConversationManager)
- [ ] 답변 구조화 개선 (AnswerFormatter)
- [ ] 컨텍스트 윈도우 최적화 (ContextBuilder)
- [ ] 최종 통합 및 API v2 엔드포인트 구현