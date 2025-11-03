# LangGraph 워크플로우 개선 계획서

## 📋 개요

테스트 결과 분석 보고서에서 발견된 문제점을 해결하고, 데이터베이스/벡터스토어 검색 결과 기반 답변 생성 보장을 위한 개선 계획입니다.

---

## 🔍 발견된 문제점 요약

### 1. 중간 생성 텍스트 포함 문제
- **증상**: "STEP 0: 원본 품질 평가", "질문 정보", "원본 답변" 등 중간 생성 텍스트가 최종 답변에 포함됨
- **영향**: 답변 가독성 저하, 전문성 저하
- **발생 위치**: `lawfirm_langgraph/source/services/answer_formatter.py` (또는 해당 서비스)

### 2. 검색되지 않은 내용이 답변에 포함되는 문제 (Hallucination)
- **증상**: 검색 결과에 없는 정보가 답변에 포함될 가능성
- **영향**: 잘못된 법률 정보 제공, 신뢰도 저하
- **발생 위치**: `core/services/generation/answer_generator.py`, 답변 생성 과정

### 3. 답변 길이 편차
- **증상**: 800자 ~ 3,781자까지 큰 편차
- **영향**: 사용자 경험 일관성 부족

### 4. 신뢰도 점수 편차
- **증상**: 83.15% ~ 95.00%까지 편차
- **영향**: 일관성 부족

---

## 🎯 개선 목표

1. **중간 생성 텍스트 완전 제거**: 최종 답변에서 100% 제거
2. **검색 결과 기반 검증 강화**: 검색되지 않은 내용이 답변에 포함되지 않도록 보장
3. **답변 길이 일관성 향상**: 질의 유형별 적절한 길이 유지
4. **신뢰도 계산 일관성 향상**: 질의 유형별 일관된 기준 적용

---

## 📝 상세 개선 방안

## 1. 중간 생성 텍스트 필터링 강화

### 1.1 구현 위치
- **파일**: `lawfirm_langgraph/source/services/answer_formatter.py` (또는 해당 서비스)
- **함수**: `_remove_intermediate_text()` 추가, `_validate_final_answer()` 수정

> ⚠️ **참고**: 실제 파일 경로는 프로젝트 구조에 따라 다를 수 있습니다. `lawfirm_langgraph/source/services/` 디렉토리를 확인하세요.

### 1.2 제거할 패턴
```python
# 제거할 패턴 목록
INTERMEDIATE_TEXT_PATTERNS = [
    r'^##\s*STEP\s*0.*?\n(?:.*\n)*?',  # STEP 0 섹션 전체
    r'^##\s*원본\s*품질\s*평가.*?\n(?:.*\n)*?',
    r'^##\s*질문\s*정보.*?\n(?:.*\n)*?',
    r'^##\s*원본\s*답변.*?\n(?:.*\n)*?',
    r'^\*\*질문\*\*:.*?\n',
    r'^\*\*질문\s*유형\*\*:.*?\n',
    r'^평가\s*결\s*과\s*에\s*따른\s*작업:.*?\n',
    r'^\s*•\s*\[.*?\].*?개선.*?\n',  # 체크리스트 패턴
    r'^\[.*?\].*?충분하고.*?\n',
    r'^원본\s*에\s*개선이\s*필요하면.*?\n',
]
```

### 1.3 구현 코드
```python
def _remove_intermediate_text(self, answer_text: str) -> str:
    """
    중간 생성 텍스트 제거
    
    Args:
        answer_text: 원본 답변 텍스트
        
    Returns:
        중간 텍스트가 제거된 답변
    """
    import re
    
    if not answer_text or not isinstance(answer_text, str):
        return answer_text
    
    lines = answer_text.split('\n')
    cleaned_lines = []
    skip_section = False
    skip_patterns = [
        r'^##\s*STEP\s*0',
        r'^##\s*원본\s*품질\s*평가',
        r'^##\s*질문\s*정보',
        r'^##\s*원본\s*답변',
        r'^\*\*질문\*\*:',
        r'^\*\*질문\s*유형\*\*:',
        r'^평가\s*결과',
        r'원본\s*에\s*개선이\s*필요하면',
    ]
    
    for i, line in enumerate(lines):
        # 섹션 시작 패턴 확인
        is_section_start = False
        for pattern in skip_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                skip_section = True
                is_section_start = True
                break
        
        if is_section_start:
            continue
        
        # 섹션 종료 확인 (다음 ## 헤더 또는 실제 답변 시작)
        if skip_section:
            # 다음 ## 헤더가 나오거나, 실제 답변 시작 패턴 확인
            if re.match(r'^##\s+[가-힣]', line):  # 실제 답변 섹션 시작
                skip_section = False
                # 이 줄은 포함
                if not any(re.match(p, line, re.IGNORECASE) for p in skip_patterns):
                    cleaned_lines.append(line)
            # 체크리스트 패턴 제거
            elif re.match(r'^\s*•\s*\[.*?\].*?', line):
                continue
            else:
                continue
        else:
            # 일반 텍스트 추가 (다시 체크리스트 패턴 필터링)
            if re.match(r'^\s*•\s*\[.*?\].*?', line):
                continue
            cleaned_lines.append(line)
    
    cleaned_text = '\n'.join(cleaned_lines)
    
    # 연속된 빈 줄 정리
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text.strip()
```

### 1.4 적용 위치
- `format_and_prepare_final()` 메서드에서 `_remove_metadata_sections()` 호출 이후 추가 적용

---

## 2. 검색 결과 기반 검증 강화 (Hallucination 방지)

### 2.1 구현 위치
- **파일**: `lawfirm_langgraph/source/services/quality_validators.py` (또는 해당 서비스), `lawfirm_langgraph/source/services/answer_formatter.py` (또는 해당 서비스)
- **함수**: `validate_answer_source_verification()`, `_validate_final_answer()` 추가/수정

### 2.2 검증 로직
```python
def validate_answer_source_verification(
    answer: str,
    retrieved_docs: List[Dict[str, Any]],
    query: str
) -> Dict[str, Any]:
    """
    답변의 내용이 검색된 문서에 기반하는지 검증
    
    Args:
        answer: 검증할 답변 텍스트
        retrieved_docs: 검색된 문서 목록
        query: 원본 질의
        
    Returns:
        검증 결과 딕셔너리
        {
            "is_grounded": bool,
            "grounding_score": float,
            "unverified_sections": List[str],
            "source_coverage": float,
            "needs_review": bool
        }
    """
    import re
    from difflib import SequenceMatcher
    
    if not answer or not retrieved_docs:
        return {
            "is_grounded": False,
            "grounding_score": 0.0,
            "unverified_sections": [answer] if answer else [],
            "source_coverage": 0.0,
            "needs_review": True,
            "error": "답변 또는 검색 결과가 없습니다."
        }
    
    # 1. 검색된 문서에서 모든 텍스트 추출
    source_texts = []
    for doc in retrieved_docs:
        if isinstance(doc, dict):
            content = (
                doc.get("content") or
                doc.get("text") or
                doc.get("content_text") or
                ""
            )
            if content and len(content.strip()) > 50:
                source_texts.append(content.lower())
    
    if not source_texts:
        return {
            "is_grounded": False,
            "grounding_score": 0.0,
            "unverified_sections": [],
            "source_coverage": 0.0,
            "needs_review": True,
            "error": "검색된 문서의 내용이 없습니다."
        }
    
    # 2. 답변을 문장 단위로 분리
    answer_sentences = re.split(r'[.!?。！？]\s+', answer)
    answer_sentences = [s.strip() for s in answer_sentences if len(s.strip()) > 20]
    
    # 3. 각 문장이 검색된 문서에 기반하는지 검증
    verified_sentences = []
    unverified_sentences = []
    
    for sentence in answer_sentences:
        sentence_lower = sentence.lower()
        
        # 문장의 핵심 키워드 추출 (불용어 제거)
        stopwords = {'는', '은', '이', '가', '을', '를', '에', '의', '와', '과', '로', '으로', '에서', '도', '만', '부터', '까지'}
        sentence_words = [w for w in re.findall(r'[가-힣]+', sentence_lower) if len(w) > 1 and w not in stopwords]
        
        if not sentence_words:
            continue
        
        # 각 소스 텍스트와 유사도 계산
        max_similarity = 0.0
        best_match_source = None
        
        for source_text in source_texts:
            # 키워드 매칭 점수
            matched_keywords = sum(1 for word in sentence_words if word in source_text)
            keyword_score = matched_keywords / len(sentence_words) if sentence_words else 0.0
            
            # 문장 유사도 (SequenceMatcher 사용)
            similarity = SequenceMatcher(None, sentence_lower[:100], source_text[:1000]).ratio()
            
            # 종합 점수 (키워드 매칭 + 유사도)
            combined_score = (keyword_score * 0.6) + (similarity * 0.4)
            
            if combined_score > max_similarity:
                max_similarity = combined_score
                best_match_source = source_text[:100]  # 디버깅용
        
        # 검증 기준: 30% 이상 유사하거나 핵심 키워드 50% 이상 매칭
        if max_similarity >= 0.3 or (matched_keywords / len(sentence_words) if sentence_words else 0) >= 0.5:
            verified_sentences.append({
                "sentence": sentence,
                "similarity": max_similarity,
                "source_preview": best_match_source
            })
        else:
            # 법령 인용이나 일반적인 면책 조항은 제외
            if not (re.search(r'\[법령:\s*[^\]]+\]', sentence) or 
                   re.search(r'본\s*답변은\s*일반적인', sentence) or
                   re.search(r'변호사와\s*직접\s*상담', sentence)):
                unverified_sentences.append({
                    "sentence": sentence[:100],
                    "similarity": max_similarity,
                    "keywords": sentence_words[:5]
                })
    
    # 4. 종합 검증 점수 계산
    total_sentences = len(answer_sentences)
    verified_count = len(verified_sentences)
    
    grounding_score = verified_count / total_sentences if total_sentences > 0 else 0.0
    source_coverage = len(set([s["source_preview"] for s in verified_sentences if s.get("source_preview")])) / len(source_texts) if source_texts else 0.0
    
    # 5. 검증 통과 기준: 80% 이상 문장이 검증됨
    is_grounded = grounding_score >= 0.8
    
    # 6. 신뢰도 조정 (검증되지 않은 문장이 많으면 신뢰도 감소)
    confidence_penalty = len(unverified_sentences) * 0.05  # 문장당 5% 감소
    
    return {
        "is_grounded": is_grounded,
        "grounding_score": grounding_score,
        "verified_sentences": verified_sentences[:5],  # 샘플
        "unverified_sentences": unverified_sentences,
        "unverified_count": len(unverified_sentences),
        "source_coverage": source_coverage,
        "needs_review": not is_grounded or len(unverified_sentences) > 3,
        "confidence_penalty": min(confidence_penalty, 0.3),  # 최대 30% 감소
        "total_sentences": total_sentences,
        "verified_count": verified_count
    }
```

### 2.3 적용 위치
- `format_and_prepare_final()` 메서드에서 답변 검증 단계에 추가
- `_validate_final_answer()` 메서드에 통합

### 2.4 답변 재생성 로직
```python
def _regenerate_answer_if_needed(
    self,
    state: LegalWorkflowState,
    verification_result: Dict[str, Any]
) -> LegalWorkflowState:
    """
    검증 결과에 따라 답변 재생성
    
    Args:
        state: 워크플로우 상태
        verification_result: 검증 결과
        
    Returns:
        수정된 상태
    """
    if verification_result.get("needs_review", False):
        self.logger.warning(
            f"답변 검증 실패: grounding_score={verification_result.get('grounding_score', 0):.2f}, "
            f"unverified_count={verification_result.get('unverified_count', 0)}"
        )
        
        # 신뢰도 조정
        current_confidence = state.get("confidence", 0.8)
        penalty = verification_result.get("confidence_penalty", 0.0)
        adjusted_confidence = max(0.0, current_confidence - penalty)
        state["confidence"] = adjusted_confidence
        
        # 검증되지 않은 섹션을 로그에 기록
        unverified = verification_result.get("unverified_sentences", [])
        if unverified:
            self.logger.warning(
                f"검증되지 않은 문장 {len(unverified)}개 발견. "
                f"샘플: {unverified[0].get('sentence', '')[:50]}..."
            )
    
    return state
```

---

## 3. 답변 길이 일관성 개선

### 3.1 질의 유형별 목표 길이
```python
ANSWER_LENGTH_TARGETS = {
    "simple_question": (500, 1000),      # 간단한 질의: 500-1000자
    "term_explanation": (800, 1500),     # 용어 설명: 800-1500자
    "legal_analysis": (1500, 2500),      # 법률 분석: 1500-2500자
    "complex_question": (2000, 3500),    # 복잡한 질의: 2000-3500자
    "default": (800, 2000)               # 기본값: 800-2000자
}
```

### 3.2 답변 길이 조절 로직
```python
def _adjust_answer_length(
    self,
    answer: str,
    query_type: str,
    query_complexity: str
) -> str:
    """
    답변 길이를 질의 유형에 맞게 조절
    
    Args:
        answer: 원본 답변
        query_type: 질의 유형
        query_complexity: 질의 복잡도
        
    Returns:
        조절된 답변
    """
    import re
    
    if not answer:
        return answer
    
    current_length = len(answer)
    
    # 목표 길이 결정
    if query_complexity == "simple":
        min_len, max_len = ANSWER_LENGTH_TARGETS.get("simple_question", (500, 1000))
    elif query_complexity == "complex":
        min_len, max_len = ANSWER_LENGTH_TARGETS.get("complex_question", (2000, 3500))
    else:
        targets = ANSWER_LENGTH_TARGETS.get(query_type, ANSWER_LENGTH_TARGETS["default"])
        min_len, max_len = targets
    
    # 길이가 적절한 경우 그대로 반환
    if min_len <= current_length <= max_len:
        return answer
    
    # 너무 긴 경우: 핵심 내용만 추출
    if current_length > max_len:
        # 섹션별로 분리
        sections = re.split(r'\n\n+', answer)
        
        # 각 섹션의 중요도 평가 (법령 인용, 판례 등 포함 여부)
        important_sections = []
        other_sections = []
        
        for section in sections:
            if (re.search(r'\[법령:', section) or 
                re.search(r'대법원', section) or
                re.search(r'제\s*\d+\s*조', section)):
                important_sections.append(section)
            else:
                other_sections.append(section)
        
        # 중요 섹션 우선 포함
        result = []
        current_len = 0
        
        for section in important_sections:
            if current_len + len(section) <= max_len:
                result.append(section)
                current_len += len(section)
            else:
                # 섹션 일부만 포함
                remaining = max_len - current_len
                result.append(section[:remaining] + "...")
                break
        
        # 여유가 있으면 다른 섹션도 포함
        for section in other_sections:
            if current_len + len(section) <= max_len:
                result.append(section)
                current_len += len(section)
            else:
                break
        
        return '\n\n'.join(result)
    
    # 너무 짧은 경우: 이미 최소 길이로 생성된 것이므로 그대로 반환
    # (추가 생성은 LLM 호출이 필요하므로 여기서는 하지 않음)
    return answer
```

### 3.3 적용 위치
- `format_and_prepare_final()` 메서드에서 답변 길이 조절

---

## 4. 신뢰도 계산 일관성 개선

### 4.1 질의 유형별 신뢰도 기준
```python
def _calculate_consistent_confidence(
    self,
    base_confidence: float,
    query_type: str,
    query_complexity: str,
    grounding_score: float,
    source_coverage: float
) -> float:
    """
    일관된 신뢰도 계산
    
    Args:
        base_confidence: 기본 신뢰도
        query_type: 질의 유형
        query_complexity: 질의 복잡도
        grounding_score: 검증 점수
        source_coverage: 소스 커버리지
        
    Returns:
        조정된 신뢰도
    """
    # 1. 기본 신뢰도 조정
    confidence = base_confidence
    
    # 2. 질의 복잡도에 따른 조정
    complexity_adjustments = {
        "simple": 0.05,      # 간단한 질의: +5%
        "moderate": 0.0,      # 보통: 변화 없음
        "complex": -0.05      # 복잡한 질의: -5%
    }
    confidence += complexity_adjustments.get(query_complexity, 0.0)
    
    # 3. 검증 점수에 따른 조정
    if grounding_score < 0.8:
        confidence -= (0.8 - grounding_score) * 0.3  # 최대 30% 감소
    
    # 4. 소스 커버리지에 따른 조정
    if source_coverage < 0.5:
        confidence -= (0.5 - source_coverage) * 0.2  # 최대 20% 감소
    
    # 5. 범위 제한 (0.0 ~ 1.0)
    confidence = max(0.0, min(1.0, confidence))
    
    # 6. 질의 유형별 최소 신뢰도 설정
    min_confidence_by_type = {
        "simple_question": 0.75,
        "term_explanation": 0.80,
        "legal_analysis": 0.75,
        "complex_question": 0.70
    }
    min_confidence = min_confidence_by_type.get(query_type, 0.70)
    
    # 최소 신뢰도보다 낮으면 경고
    if confidence < min_confidence:
        self.logger.warning(
            f"신뢰도가 최소 기준({min_confidence:.2%})보다 낮음: {confidence:.2%}"
        )
    
    return confidence
```

### 4.2 적용 위치
- `prepare_final_response_part()` 메서드에서 신뢰도 계산 시 적용

---

## 🔧 구현 순서

### Phase 1: 중간 텍스트 필터링 (우선순위: 높음)
1. `_remove_intermediate_text()` 함수 추가
2. `format_and_prepare_final()` 메서드에 통합
3. 테스트 및 검증

### Phase 2: 검색 결과 기반 검증 (우선순위: 높음)
1. `validate_answer_source_verification()` 함수 추가 (`quality_validators.py`)
2. `_validate_final_answer()` 메서드에 통합
3. `_regenerate_answer_if_needed()` 로직 추가
4. 테스트 및 검증

### Phase 3: 답변 길이 조절 (우선순위: 중간)
1. `_adjust_answer_length()` 함수 추가
2. `format_and_prepare_final()` 메서드에 통합
3. 테스트 및 검증

### Phase 4: 신뢰도 계산 일관성 (우선순위: 중간)
1. `_calculate_consistent_confidence()` 함수 추가
2. `prepare_final_response_part()` 메서드에 통합
3. 테스트 및 검증

---

## 📊 성공 기준

### 중간 텍스트 필터링
- ✅ "STEP 0", "원본 답변", "질문 정보" 등 패턴이 100% 제거됨
- ✅ 테스트 질의에서 중간 텍스트가 포함되지 않음

### 검색 결과 기반 검증
- ✅ 답변의 80% 이상이 검색된 문서에 기반함
- ✅ 검증되지 않은 문장이 3개 이하
- ✅ 검증 실패 시 신뢰도 자동 조정

### 답변 길이 일관성
- ✅ 질의 유형별 목표 길이 범위 내 유지율 90% 이상
- ✅ 최대 길이 초과 시 자동 축약

### 신뢰도 일관성
- ✅ 질의 유형별 신뢰도 편차 10% 이하
- ✅ 검증 점수 반영된 신뢰도 조정

---

## 🧪 테스트 계획

### 단위 테스트
1. `_remove_intermediate_text()` 테스트
2. `validate_answer_source_verification()` 테스트
3. `_adjust_answer_length()` 테스트
4. `_calculate_consistent_confidence()` 테스트

### 통합 테스트
1. 전체 워크플로우 테스트 (5가지 질의)
2. 검증 점수 확인
3. 신뢰도 조정 확인
4. 답변 품질 검증

---

## 📝 파일 수정 목록

### 수정할 파일
1. `lawfirm_langgraph/source/services/answer_formatter.py` (또는 해당 서비스)
   - `_remove_intermediate_text()` 추가
   - `_adjust_answer_length()` 추가
   - `_calculate_consistent_confidence()` 추가
   - `format_and_prepare_final()` 수정
   - `_validate_final_answer()` 수정

2. `lawfirm_langgraph/source/services/quality_validators.py` (또는 해당 서비스)
   - `validate_answer_source_verification()` 추가
   - `AnswerValidator` 클래스 확장

### 추가할 테스트 파일
1. `tests/unit/test_answer_formatter_improvements.py`
2. `tests/integration/test_source_verification.py`

---

## ⚠️ 주의사항

1. **성능 영향**: 검증 로직 추가로 처리 시간이 약간 증가할 수 있음 (예상: +2~3초)
2. **답변 품질**: 검증이 너무 엄격하면 일부 유효한 답변이 제거될 수 있음 → 임계값 조정 필요
3. **검색 결과 부족**: 검색 결과가 없는 경우에도 기본 답변은 제공해야 함 → 폴백 로직 필요

---

## 📅 예상 일정

- **Phase 1**: 1일
- **Phase 2**: 2일
- **Phase 3**: 1일
- **Phase 4**: 1일
- **통합 테스트**: 1일

**총 예상 일정**: 약 6일

---

## 🔄 롤백 계획

각 Phase는 독립적으로 구현되므로, 문제 발생 시 해당 Phase만 롤백 가능:
1. Git 브랜치에서 개별 Phase 커밋 롤백
2. 기능 플래그를 통한 개별 기능 비활성화
