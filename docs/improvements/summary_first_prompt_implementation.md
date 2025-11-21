# 요약 기반(Summary-First) 프롬프트 구현 방법

## 📋 목차

1. [개요](#개요)
2. [구현 목표](#구현-목표)
3. [설계 원칙](#설계-원칙)
4. [구현 상세](#구현-상세)
5. [코드 구조](#코드-구조)
6. [사용 예시](#사용-예시)
7. [테스트 방법](#테스트-방법)

---

## 개요

### 배경

긴 문서를 프롬프트에 그대로 포함하면 다음과 같은 문제가 발생합니다:

1. **토큰 사용량 증가**: 긴 문서는 많은 토큰을 소비하여 비용 증가
2. **LLM 집중도 저하**: 긴 문서 속에서 핵심 정보를 찾기 어려움
3. **응답 품질 저하**: 불필요한 정보로 인한 답변 정확도 감소

### 해결 방안

**Summary-First 접근법**: 긴 문서는 요약을 먼저 제공하고, 필요한 경우에만 상세 추출을 포함합니다.

```
[기존 방식]
문서 1: [전체 내용 1500자]
문서 2: [전체 내용 800자]
...

[Summary-First 방식]
[Context Summary]
- 문서 1: 핵심 쟁점 3개 요약 (200자)
- 문서 2: 관련 조문과 판례 요약 (150자)

[Detailed Extracts]
- 문서 1 관련 부분: "..." (300자)
- 문서 2 조항 핵심: "..." (200자)
```

---

## 구현 목표

### 1. 토큰 절감
- **목표**: 문서당 평균 50-70% 토큰 절감
- **방법**: 요약(100-200 토큰) + 선택적 상세 추출(200-400 토큰)

### 2. 응답 품질 향상
- **목표**: LLM의 핵심 정보 집중도 향상
- **방법**: 요약으로 전체 맥락 제공, 상세 추출로 정확한 인용

### 3. 처리 속도 개선
- **목표**: 프롬프트 처리 시간 단축
- **방법**: 프롬프트 길이 감소로 인한 처리 시간 단축

---

## 설계 원칙

### 1. 문서 길이 기반 조건부 처리

```python
# 요약 임계값
SUMMARY_THRESHOLD_LAW = 1000      # 법률 조문: 1000자 이상이면 요약
SUMMARY_THRESHOLD_CASE = 600      # 판례: 600자 이상이면 요약
SUMMARY_THRESHOLD_COMMENTARY = 400  # 해설: 400자 이상이면 요약
```

### 2. 요약 우선, 상세 추출은 선택적

- **모든 문서**: 요약 생성 (긴 문서만)
- **상세 추출**: 상위 3개 문서만 선택적으로 포함

### 3. 문서 유형별 맞춤 요약

- **법령**: 조문번호, 핵심 조항, 질문 관련성
- **판례**: 판시사항, 판결요지, 질문 관련성
- **해설**: 핵심 내용, 주요 논점, 질문 관련성

---

## 구현 상세

### 1. 요약 생성 에이전트

#### 1.1 `DocumentSummaryAgent` 클래스

```python
# lawfirm_langgraph/core/agents/handlers/document_summary_agent.py

class DocumentSummaryAgent:
    """문서 요약 생성 에이전트"""
    
    def __init__(
        self,
        llm: Optional[Any] = None,  # LLM 인스턴스 (선택적)
        llm_fast: Optional[Any] = None,  # 빠른 LLM (선택적)
        logger: Optional[logging.Logger] = None
    ):
        """요약 에이전트 초기화"""
        self.llm = llm
        self.llm_fast = llm_fast or llm
        self.logger = logger or get_logger(__name__)
        
        # 요약 임계값
        self.SUMMARY_THRESHOLD_LAW = 1000
        self.SUMMARY_THRESHOLD_CASE = 600
        self.SUMMARY_THRESHOLD_COMMENTARY = 400
        self.MAX_SUMMARY_LENGTH = 200
    
    def summarize_document(
        self,
        doc: Dict[str, Any],
        query: str,
        max_summary_length: int = 200,
        use_llm: bool = False  # LLM 사용 여부
    ) -> Dict[str, Any]:
        """
        문서 요약 생성 (Summary-First 접근법)
        
        Args:
            doc: 문서 딕셔너리
            query: 사용자 질문
            max_summary_length: 최대 요약 길이
            use_llm: LLM 사용 여부 (False면 규칙 기반)
        
        Returns:
            {
                'summary': '요약 텍스트',
                'key_points': ['핵심 포인트 1', '핵심 포인트 2', ...],
                'relevance_notes': '질문과의 연관성',
                'document_type': 'law/case/commentary',
                'original_length': 원본 문서 길이,
                'summary_length': 요약 길이
            }
        """
        doc_type = self._get_document_type(doc)
        
        if use_llm and self.llm_fast:
            return self._summarize_with_llm(doc, query, doc_type, max_summary_length)
        else:
            return self._summarize_with_rules(doc, query, doc_type, max_summary_length)
    
    def summarize_batch(
        self,
        docs: List[Dict[str, Any]],
        query: str,
        max_summary_length: int = 200,
        use_llm: bool = False
    ) -> List[Dict[str, Any]]:
        """배치 요약 생성"""
        return [
            self.summarize_document(doc, query, max_summary_length, use_llm)
            for doc in docs
        ]
```

**구현 전략**:
1. **규칙 기반 요약** (기본): 빠르고 안정적, 비용 없음
2. **LLM 기반 요약** (선택적): 품질 향상, 비용 발생
3. 문서 유형별 맞춤 요약 로직

#### 1.2 문서 유형별 요약 로직

**법령 문서 요약** (`_summarize_law`):
```python
def _summarize_law(
    self, doc: Dict[str, Any], query: str, max_length: int
) -> Dict[str, Any]:
    """법령 문서 요약"""
    law_name = doc.get("law_name", "")
    article_no = doc.get("article_no", "")
    content = doc.get("content", "")
    
    # 핵심 정보 추출
    summary_parts = []
    if law_name and article_no:
        summary_parts.append(f"{law_name} 제{article_no}조")
    
    # 핵심 조항 추출 (질문 키워드 포함 문장 우선)
    key_sentences = self._extract_key_sentences(content, query, max_sentences=3)
    summary_parts.extend(key_sentences)
    
    # 질문 관련성 분석
    relevance = self._analyze_relevance(content, query)
    
    return {
        'summary': ' '.join(summary_parts)[:max_length],
        'key_points': key_sentences,
        'relevance_notes': relevance,
        'document_type': 'law'
    }
```

**판례 문서 요약** (`_summarize_case`):
```python
def _summarize_case(
    self, doc: Dict[str, Any], query: str, max_length: int
) -> Dict[str, Any]:
    """판례 문서 요약"""
    court = doc.get("court", "")
    case_name = doc.get("case_name", "")
    content = doc.get("content", "")
    
    # 판시사항 추출
    reasoning = doc.get("case_reasoning") or self._extract_reasoning(content)
    
    # 판결요지 추출
    key_points = self._extract_judgment_points(content, query)
    
    return {
        'summary': f"{court} {case_name} 판결: {reasoning[:100]}",
        'key_points': key_points,
        'relevance_notes': self._analyze_relevance(content, query),
        'document_type': 'case'
    }
```

**해설 문서 요약** (`_summarize_commentary`):
```python
def _summarize_commentary(
    self, doc: Dict[str, Any], query: str, max_length: int
) -> Dict[str, Any]:
    """해설 문서 요약"""
    content = doc.get("content", "")
    title = doc.get("title", "")
    
    # 핵심 내용 추출 (앞부분 + 키워드 관련 부분)
    intro = content[:200]  # 앞부분
    relevant_parts = self._extract_relevant_parts(content, query, max_length=300)
    
    return {
        'summary': f"{title}: {intro}",
        'key_points': relevant_parts,
        'relevance_notes': self._analyze_relevance(content, query),
        'document_type': 'commentary'
    }
```

### 2. 프롬프트 구조 변경

#### 2.1 `UnifiedPromptManager`에서 에이전트 사용

**에이전트 초기화**:
```python
class UnifiedPromptManager:
    def __init__(self, prompts_dir: str = "streamlit/prompts"):
        # ... 기존 초기화 코드 ...
        
        # 요약 에이전트 초기화 (지연 로딩)
        self._summary_agent = None
    
    def _get_summary_agent(self) -> DocumentSummaryAgent:
        """요약 에이전트 가져오기 (지연 초기화)"""
        if self._summary_agent is None:
            from lawfirm_langgraph.core.agents.handlers.document_summary_agent import DocumentSummaryAgent
            # LLM은 필요시 주입 (선택적)
            self._summary_agent = DocumentSummaryAgent(
                llm=None,  # 필요시 주입
                llm_fast=None,  # 필요시 주입
                logger=logger
            )
        return self._summary_agent
```

#### 2.2 `_build_documents_section` 리팩토링

**기존 구조**:
```python
## 검색된 법률 문서
[문서 1] 전체 내용 (1500자)
[문서 2] 전체 내용 (800자)
...
```

**새로운 구조 (에이전트 사용)**:
```python
def _build_documents_section(
    self, sorted_docs: List[Dict[str, Any]], query: str
) -> str:
    """Summary-First 방식으로 문서 섹션 생성 (에이전트 사용)"""
    summary_agent = self._get_summary_agent()
    
    # 1. 문서 분류 (요약 필요 vs 전체 포함)
    docs_for_summary = []
    docs_for_full = []
    
    for doc in sorted_docs:
        if self._should_use_summary(doc):
            docs_for_summary.append(doc)
        else:
            docs_for_full.append(doc)
    
    # 2. 요약 생성 (에이전트 사용)
    summaries = summary_agent.summarize_batch(
        docs_for_summary,
        query,
        max_summary_length=self.MAX_SUMMARY_LENGTH,
        use_llm=False  # 규칙 기반 요약 (빠르고 안정적)
    )
    
    # 3. Summary 섹션 생성
    summary_section = self._build_summary_section(summaries, sorted_docs)
    
    # 4. Detailed Extracts 섹션 생성 (상위 3개만)
    detailed_section = self._build_detailed_section(
        docs_for_summary[:self.MAX_DETAILED_EXTRACTS],
        query
    )
    
    # 5. 전체 문서 섹션 (요약 불필요한 문서)
    full_docs_section = self._build_full_docs_section(docs_for_full, query)
    
    # 6. 통합
    return summary_section + detailed_section + full_docs_section
```

#### 2.3 문서 분류 로직

```python
def _should_use_summary(self, doc: Dict[str, Any]) -> bool:
    """문서가 요약이 필요한지 판단"""
    content = doc.get("content", "")
    doc_type = self._get_document_type(doc)
    
    thresholds = {
        'law': self.SUMMARY_THRESHOLD_LAW,
        'case': self.SUMMARY_THRESHOLD_CASE,
        'commentary': self.SUMMARY_THRESHOLD_COMMENTARY
    }
    
    threshold = thresholds.get(doc_type, 500)
    return len(content) > threshold
```

### 3. 상세 추출 로직

#### 3.1 `_extract_detailed_relevant_parts`

```python
def _extract_detailed_relevant_parts(
    self,
    doc: Dict[str, Any],
    query: str,
    max_extract_length: int = 500
) -> str:
    """
    질문과 직접 관련된 부분만 상세 추출
    
    전략:
    1. 질문 키워드 포함 문장 우선
    2. 관련 문맥 포함 (전후 2-3문장)
    3. 최대 길이 제한
    """
```

**추출 우선순위**:
1. 질문 키워드가 포함된 문장
2. 키워드 주변 문맥 (전후 2-3문장)
3. 문서의 앞부분 (개요)
4. 문서의 뒷부분 (결론)

---

## 코드 구조

### 파일 구조

```
lawfirm_langgraph/
├── core/
│   ├── agents/
│   │   └── handlers/
│   │       └── document_summary_agent.py  [신규]
│   │           ├── DocumentSummaryAgent 클래스
│   │           ├── summarize_document()
│   │           ├── summarize_batch()
│   │           ├── _summarize_with_rules()
│   │           ├── _summarize_with_llm()
│   │           ├── _summarize_law()
│   │           ├── _summarize_case()
│   │           ├── _summarize_commentary()
│   │           └── 헬퍼 메서드들
│   │
│   └── services/
│       └── unified_prompt_manager.py
│           ├── 상수 정의
│           │   ├── SUMMARY_THRESHOLD_LAW
│           │   ├── SUMMARY_THRESHOLD_CASE
│           │   └── SUMMARY_THRESHOLD_COMMENTARY
│           │
│           ├── 에이전트 관리
│           │   ├── _summary_agent (인스턴스 변수)
│           │   └── _get_summary_agent() (지연 초기화)
│           │
│           ├── 프롬프트 구조 메서드
│           │   ├── _build_documents_section() [리팩토링 - 에이전트 사용]
│           │   ├── _build_summary_section()
│           │   ├── _build_detailed_section()
│           │   └── _build_full_docs_section()
│           │
│           └── 헬퍼 메서드
│               ├── _should_use_summary()
│               ├── _get_document_type()
│               └── _extract_detailed_relevant_parts()
```

### 메서드 시그니처

#### DocumentSummaryAgent 클래스

```python
class DocumentSummaryAgent:
    """문서 요약 생성 에이전트"""
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        llm_fast: Optional[Any] = None,
        logger: Optional[logging.Logger] = None
    )
    
    def summarize_document(
        self,
        doc: Dict[str, Any],
        query: str,
        max_summary_length: int = 200,
        use_llm: bool = False
    ) -> Dict[str, Any]
    
    def summarize_batch(
        self,
        docs: List[Dict[str, Any]],
        query: str,
        max_summary_length: int = 200,
        use_llm: bool = False
    ) -> List[Dict[str, Any]]
    
    # 내부 메서드
    def _summarize_with_rules(
        self, doc: Dict[str, Any], query: str, doc_type: str, max_length: int
    ) -> Dict[str, Any]
    
    def _summarize_with_llm(
        self, doc: Dict[str, Any], query: str, doc_type: str, max_length: int
    ) -> Dict[str, Any]
    
    def _summarize_law(
        self, doc: Dict[str, Any], query: str, max_length: int
    ) -> Dict[str, Any]
    
    def _summarize_case(
        self, doc: Dict[str, Any], query: str, max_length: int
    ) -> Dict[str, Any]
    
    def _summarize_commentary(
        self, doc: Dict[str, Any], query: str, max_length: int
    ) -> Dict[str, Any]
    
    def _get_document_type(self, doc: Dict[str, Any]) -> str
    def _extract_key_sentences(self, content: str, query: str, max_sentences: int) -> List[str]
    def _analyze_relevance(self, content: str, query: str) -> str
```

#### UnifiedPromptManager 클래스

```python
class UnifiedPromptManager:
    # 상수
    SUMMARY_THRESHOLD_LAW = 1000
    SUMMARY_THRESHOLD_CASE = 600
    SUMMARY_THRESHOLD_COMMENTARY = 400
    MAX_SUMMARY_LENGTH = 200
    MAX_DETAILED_EXTRACTS = 3
    
    # 에이전트 관리
    def _get_summary_agent(self) -> DocumentSummaryAgent
    
    # 프롬프트 구조
    def _build_documents_section(
        self, sorted_docs: List[Dict[str, Any]], query: str
    ) -> str
    
    def _build_summary_section(
        self, summaries: List[Dict[str, Any]], original_docs: List[Dict[str, Any]]
    ) -> str
    
    def _build_detailed_section(
        self, docs: List[Dict[str, Any]], query: str, max_docs: int = 3
    ) -> str
    
    def _build_full_docs_section(
        self, docs: List[Dict[str, Any]], query: str
    ) -> str
    
    # 헬퍼
    def _should_use_summary(self, doc: Dict[str, Any]) -> bool
    def _get_document_type(self, doc: Dict[str, Any]) -> str
    def _extract_detailed_relevant_parts(
        self, doc: Dict[str, Any], query: str, max_extract_length: int = 500
    ) -> str
```

---

## 사용 예시

### 예시 1: 긴 법령 문서

**입력**:
```python
doc = {
    "law_name": "민법",
    "article_no": "543",
    "content": "계약 해지에 관한 긴 조문 내용... (2000자)"
}
query = "계약서 작성 시 주의할 사항은 무엇인가요?"
```

**요약 생성 (에이전트 사용)**:
```python
# UnifiedPromptManager에서 에이전트 사용
summary_agent = manager._get_summary_agent()
summary = summary_agent.summarize_document(doc, query, use_llm=False)

# 결과:
# {
#     'summary': '민법 제543조는 계약 해지 요건 및 절차를 규정합니다.',
#     'key_points': [
#         '계약 해지 요건: 채무불이행, 기간 경과 등',
#         '해지 절차: 상대방에게 통지 필요',
#         '해지 효과: 계약 관계 종료'
#     ],
#     'relevance_notes': '계약서 작성 시 해지 조항 명시 필요성과 관련',
#     'document_type': 'law',
#     'original_length': 2000,
#     'summary_length': 180
# }
```

**프롬프트 출력**:
```
### [Context Summary]

**[문서 1]** 민법 제543조 (관련도: 0.61)
- 핵심 쟁점: 계약 해지 요건 및 절차
- 관련 조항: 민법 제543조
- 질문 연관성: 계약서 작성 시 해지 조항 명시 필요성

### [Detailed Extracts]

**[문서 1]** 민법 제543조 상세 내용:
계약 해지 요건: 당사자 일방이 계약의 내용에 따르지 아니한 때에는...
[질문과 직접 관련된 부분만 추출]
```

### 예시 2: 판례 문서

**입력**:
```python
doc = {
    "court": "대법원",
    "case_name": "손해배상",
    "content": "판례 내용... (1500자)"
}
query = "계약 해지 시 손해배상 범위는?"
```

**요약 생성 (에이전트 사용)**:
```python
# UnifiedPromptManager에서 에이전트 사용
summary_agent = manager._get_summary_agent()
summary = summary_agent.summarize_document(doc, query, use_llm=False)

# 결과:
# {
#     'summary': '대법원 판결은 계약 해지 시 손해배상 범위를 명확히 합니다.',
#     'key_points': [
#         '일방적 해지 시 위약금 청구 가능',
#         '손해액 산정 기준: 실제 손해 범위',
#         '과실상계 고려 필요'
#     ],
#     'relevance_notes': '계약 해지 시 손해배상 범위와 직접 관련',
#     'document_type': 'case',
#     'original_length': 1500,
#     'summary_length': 165
# }
```

---

## 테스트 방법

### 1. 단위 테스트

```python
def test_document_summary_agent():
    """요약 에이전트 테스트"""
    from lawfirm_langgraph.core.agents.handlers.document_summary_agent import DocumentSummaryAgent
    
    agent = DocumentSummaryAgent()
    doc = {
        "law_name": "민법",
        "article_no": "543",
        "content": "긴 조문 내용..." * 100  # 2000자 이상
    }
    query = "계약 해지 요건은?"
    
    summary = agent.summarize_document(doc, query, use_llm=False)
    
    assert 'summary' in summary
    assert 'key_points' in summary
    assert 'document_type' in summary
    assert summary['document_type'] == 'law'
    assert len(summary['summary']) <= 200
    assert len(summary['key_points']) > 0
    assert summary['original_length'] > summary['summary_length']
```

```python
def test_summary_agent_integration():
    """UnifiedPromptManager와 에이전트 통합 테스트"""
    manager = UnifiedPromptManager()
    doc = {
        "law_name": "민법",
        "article_no": "543",
        "content": "긴 조문 내용..." * 100
    }
    query = "계약 해지 요건은?"
    
    # 에이전트 가져오기
    agent = manager._get_summary_agent()
    assert agent is not None
    
    # 요약 생성
    summary = agent.summarize_document(doc, query)
    assert 'summary' in summary
```

### 2. 통합 테스트

```python
def test_build_documents_section_summary_first():
    """Summary-First 프롬프트 구조 테스트"""
    manager = UnifiedPromptManager()
    docs = [
        {"content": "긴 문서 1..." * 200, "law_name": "민법", "article_no": "543"},
        {"content": "긴 문서 2..." * 150, "court": "대법원", "case_name": "판례"},
        {"content": "짧은 문서 3...", "title": "해설"}
    ]
    query = "계약 해지 요건은?"
    
    result = manager._build_documents_section(docs, query)
    
    # Summary 섹션 포함 확인
    assert "[Context Summary]" in result
    
    # Detailed Extracts 섹션 포함 확인
    assert "[Detailed Extracts]" in result
    
    # 토큰 절감 확인 (기존 대비 50% 이상)
    assert len(result) < sum(len(d.get("content", "")) for d in docs) * 0.5
```

### 3. 성능 테스트

```python
def test_token_reduction():
    """토큰 절감 효과 테스트"""
    # 기존 방식 토큰 수
    old_tokens = calculate_tokens(old_prompt)
    
    # Summary-First 방식 토큰 수
    new_tokens = calculate_tokens(new_prompt)
    
    # 50% 이상 절감 확인
    reduction_rate = (old_tokens - new_tokens) / old_tokens
    assert reduction_rate >= 0.5
```

---

## 구현 체크리스트

### Phase 1: 요약 생성 에이전트 구현
- [ ] `DocumentSummaryAgent` 클래스 생성
- [ ] `summarize_document` 메서드 구현
- [ ] `summarize_batch` 메서드 구현
- [ ] `_summarize_with_rules` 메서드 구현 (규칙 기반)
- [ ] `_summarize_with_llm` 메서드 구현 (LLM 기반, 선택적)
- [ ] `_summarize_law` 메서드 구현
- [ ] `_summarize_case` 메서드 구현
- [ ] `_summarize_commentary` 메서드 구현
- [ ] 헬퍼 메서드 구현 (`_get_document_type`, `_extract_key_sentences`, `_analyze_relevance`)

### Phase 2: UnifiedPromptManager 통합
- [ ] `_get_summary_agent` 메서드 추가 (에이전트 지연 초기화)
- [ ] 요약 임계값 상수 추가
- [ ] `_build_documents_section` 리팩토링 (에이전트 사용)
- [ ] `_build_summary_section` 메서드 추가 (에이전트 결과 사용)
- [ ] `_build_detailed_section` 메서드 추가
- [ ] `_build_full_docs_section` 메서드 추가
- [ ] `_should_use_summary` 메서드 추가

### Phase 3: 상세 추출 로직
- [ ] `_extract_detailed_relevant_parts` 메서드 추가
- [ ] 질문 키워드 기반 추출 로직
- [ ] 문맥 포함 로직

### Phase 4: 테스트 및 최적화
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 작성
- [ ] 성능 테스트
- [ ] 요약 품질 검증

---

## 참고 사항

### 주의사항

1. **요약 품질 보장**: 요약이 핵심 정보를 누락하지 않도록 주의
2. **하위 호환성**: 기존 프롬프트 구조와의 호환성 유지
3. **점진적 전환**: A/B 테스트를 통한 점진적 적용

### 향후 개선 사항

1. **LLM 기반 요약 활성화**: `use_llm=True` 옵션으로 고품질 요약 제공
2. **동적 임계값**: 문서 유형별 동적 임계값 조정
3. **요약 캐싱**: 동일 문서의 요약 결과 캐싱 (에이전트 내부)
4. **배치 최적화**: 여러 문서 요약 시 병렬 처리
5. **요약 품질 메트릭**: 요약 품질 평가 및 개선

---

**작성일**: 2025-11-21  
**작성자**: AI Assistant  
**버전**: 1.0

