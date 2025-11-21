# 참고자료 사용 검토 보고서

## 테스트 질의
**질의**: "계약 위약금에 대해 설명해주세요"

## 검토 일시
2024년 (테스트 실행 후)

## 수정 이력
- **2024년**: 문서 필터링 기준 완화 적용
  - 최소 문서 수 보장 로직 추가 (최소 3개)
  - 동적 임계값 계산 완화
  - `select_balanced_documents`에서 최소 문서 수 보장
  - 프롬프트 검증 강화

---

## 1. 검색 단계 (✅ 정상)

### 검색 결과
- **검색 수행**: ✅ 정상
- **검색된 문서 수**: 16개 (벡터 검색) + 7개 (법조문 검색) + 10개 (일반 검색) + 9개 (확장 검색) + 10개 (최종 검색)
- **검색 타입**: 
  - `case_paragraph`: 판례 문서 다수
  - `statute_article`: 법조문 문서 (민법 494, 493, 543, 533, 529조 등)
  - `약정금`, `위약금` 관련 문서 포함

### 검색 품질
- ✅ 관련성 높은 문서 검색됨 (위약금, 약정금, 계약 관련)
- ✅ 법조문 검색 성공 (민법 관련 조문)
- ✅ 판례 문서 검색 성공

---

## 2. 프롬프트 구성 단계 (⚠️ 주의 필요)

### 문서 포함 과정
1. **`build_prompt_optimized_context`** (workflow_document_processor.py)
   - `retrieved_docs`를 받아서 프롬프트에 포함할 문서 선별
   - 동적 임계값 조정으로 관련성 높은 문서 필터링
   - **문제점 발견**: 
     ```
     ⚠️ [DOCUMENT LOSS] select_balanced_documents: 1 documents lost (2 → 1, max_docs==10, loss_ratio=50.0%)
     ❌ [PROMPT BUILD] 프롬프트에 문서 내용 없음: valid_docs=1, prompt_length=1003
     ```

### 발견된 문제
1. **문서 손실 발생**
   - `select_balanced_documents`에서 문서가 손실됨 (2개 → 1개)
   - 손실률: 50.0%

2. **프롬프트에 문서 내용 부족**
   - `valid_docs=1`: 유효한 문서가 1개만 남음
   - `prompt_length=1003`: 프롬프트 길이가 짧음 (문서 내용이 거의 없음)

3. **낮은 관련성 점수**
   ```
   build_prompt_optimized_context: Low relevance ratio: 100.0%, avg_score: 0.175 (low_relevance_count: 1)
   ```
   - 평균 관련성 점수: 0.175 (매우 낮음)
   - 낮은 관련성 문서만 남음

---

## 3. 답변 생성 단계 (❌ 문제 발생)

### 답변 생성 실패
```
[PREPARE_FINAL_RESPONSE_PART] ⚠️ Answer is too short or empty: normalized_lengthh=0, raw_answer_length=0, raw_answer_preview=None, state_answer_type=None
[PREPARE_FINAL_RESPONSE_PART] ⚠️ Answer recovery failed, attempting fallback ansswer generation
```

### 원인 분석
1. **프롬프트에 문서 내용 부족**
   - 유효한 문서가 1개만 남아서 프롬프트에 충분한 컨텍스트가 없음
   - LLM이 답변을 생성할 수 있는 정보가 부족

2. **문서 필터링 기준이 너무 엄격**
   - 동적 임계값 조정이 있지만, 여전히 문서가 과도하게 필터링됨
   - 관련성 점수가 낮은 문서도 일부 포함해야 할 수 있음

---

## 4. 참고자료 추출 단계 (✅ 로직 정상, ⚠️ 데이터 부족)

### 추출 로직 검토
**`_extract_and_process_sources`** (answer_formatter.py:1561-1760)

#### 로직 흐름
1. ✅ `retrieved_docs`에서 문서 순회
2. ✅ 문서 타입별 처리 (`statute_article`, `case_paragraph` 등)
3. ✅ `UnifiedSourceFormatter`로 소스 포맷팅
4. ✅ `legal_references` 추출 (statute_article 타입)
5. ✅ `sources_detail` 생성
6. ✅ Fallback 메커니즘 (3단계)

#### 강점
- ✅ 다단계 Fallback 메커니즘으로 소스 생성 보장
- ✅ 중복 제거 로직 (`seen_sources` 사용)
- ✅ 타입별 처리 로직 명확

#### 잠재적 문제
- ⚠️ `retrieved_docs`가 비어있거나 부족하면 `sources`도 부족할 수 있음
- ⚠️ 답변이 생성되지 않으면 `sources` 추출도 의미가 제한적

---

## 5. 프롬프트 지시사항 검토 (✅ 정상)

### `UnifiedPromptManager._build_final_prompt` 검토

#### Citation 요구사항
```python
⚠️ **필수 요구사항: 법률 RAG 답변 원칙**

**원칙 1: 문서 외 내용 추론/생성 금지**
- 검색된 문서에 없는 내용은 절대 추론하거나 생성하지 마세요

**원칙 2: 문서 근거 필수 포함**
- 모든 답변은 반드시 문서 근거를 포함해야 합니다
- 인용 형식: "[문서 N]에 따르면..." 또는 "민법 제XXX조에 따르면..." [문서 N]
```

#### 답변 예시 제공
- ✅ 예시 기반 학습 방식 적용
- ✅ 문서별 근거 비교 표 형식 제공
- ✅ `[문서 N]` 형식 명확히 지시

#### 평가
- ✅ Citation 요구사항이 명확함
- ✅ 예시가 구체적이고 실용적
- ⚠️ 하지만 프롬프트에 문서가 없으면 지시사항이 무의미

---

## 6. 종합 평가

### ✅ 정상 동작하는 부분
1. **검색 시스템**: 관련 문서를 잘 검색함
2. **참고자료 추출 로직**: 다단계 Fallback으로 안정적
3. **프롬프트 지시사항**: Citation 요구사항이 명확함

### ⚠️ 개선이 필요한 부분

#### 1. 문서 필터링 기준 완화 (CRITICAL)
**문제**: 
- 문서가 과도하게 필터링되어 프롬프트에 포함되지 않음
- 관련성 점수가 낮아도 질의와 관련이 있으면 포함해야 함

**개선 방안**:
```python
# workflow_document_processor.py의 build_prompt_optimized_context
# 동적 임계값을 더 완화하거나, 최소 문서 수를 보장
min_docs_required = 3  # 최소 3개 문서는 항상 포함
if len(valid_docs) < min_docs_required:
    # 임계값을 더 낮춰서 추가 문서 포함
    dynamic_threshold = min(dynamic_threshold, min_score * 0.8)
```

#### 2. 문서 손실 방지 (CRITICAL)
**문제**:
- `select_balanced_documents`에서 문서가 손실됨 (50% 손실률)

**개선 방안**:
```python
# select_balanced_documents 함수에서
# 문서 손실이 발생하면 경고하고, 최소 문서 수를 보장
if len(selected_docs) < min_required_docs:
    # 추가 문서 포함
    remaining_docs = [d for d in all_docs if d not in selected_docs]
    selected_docs.extend(remaining_docs[:min_required_docs - len(selected_docs)])
```

#### 3. 프롬프트 검증 강화
**문제**:
- 프롬프트에 문서 내용이 없어도 답변 생성 시도

**개선 방안**:
```python
# 프롬프트 생성 전 검증
if document_count < 1 or prompt_length < 500:
    # 문서 재수집 또는 Fallback 답변 생성
    logger.warning("프롬프트에 문서가 부족합니다. 문서 재수집 시도...")
    # 재수집 로직 또는 Fallback 답변
```

#### 4. 답변 생성 실패 시 Fallback 강화
**문제**:
- 답변이 생성되지 않으면 Fallback 답변도 실패

**개선 방안**:
```python
# answer_formatter.py의 _recover_and_validate_answer
if not answer or len(answer) < MIN_ANSWER_LENGTH:
    # 최소한의 일반적인 답변 생성
    fallback_answer = f"죄송합니다. '{query}'에 대한 충분한 참고자료를 찾지 못했습니다. "
    fallback_answer += "다시 질문해 주시거나, 더 구체적인 질문을 해주시면 도움을 드릴 수 있습니다."
    return fallback_answer
```

---

## 7. 권장 사항

### 즉시 개선 (High Priority)
1. ✅ **문서 필터링 기준 완화**: 최소 문서 수 보장 (최소 3-5개)
2. ✅ **문서 손실 방지**: `select_balanced_documents`에서 손실률 모니터링 및 최소 문서 수 보장
3. ✅ **프롬프트 검증**: 문서가 없으면 재수집 또는 Fallback

### 중기 개선 (Medium Priority)
1. **관련성 점수 개선**: 키워드 매칭 가중치 증가
2. **문서 다양성 보장**: 타입별 균형 유지 (법조문, 판례, 해설 등)
3. **로깅 강화**: 각 단계에서 문서 수와 품질 로깅

### 장기 개선 (Low Priority)
1. **답변 품질 평가**: 생성된 답변의 참고자료 사용 여부 자동 검증
2. **사용자 피드백 수집**: 답변 품질과 참고자료 관련성 피드백

---

## 8. 결론

### 현재 상태
- ✅ **검색 시스템**: 정상 동작
- ⚠️ **프롬프트 구성**: 문서 필터링이 과도하여 문서 부족
- ❌ **답변 생성**: 문서 부족으로 인한 실패
- ✅ **참고자료 추출 로직**: 정상 (데이터 부족으로 활용 제한)

### 핵심 문제
**프롬프트에 문서가 충분히 포함되지 않아 답변 생성이 실패하고, 결과적으로 참고자료도 제대로 활용되지 않음**

### 해결 방향
1. ✅ 문서 필터링 기준 완화로 최소 문서 수 보장 (완료)
2. ✅ 문서 손실 방지 메커니즘 추가 (완료)
3. ✅ 프롬프트 검증 강화로 문서 부족 시 조기 감지 및 대응 (완료)

### 수정 후 테스트 결과
- **문서 수 개선**: 필터링 후 3개 문서 보장 (이전: 1개)
- **문서 손실 감소**: 25% 손실률 (이전: 50%)
- **남은 문제**: 프롬프트에 문서 내용이 포함되지 않는 문제 (프롬프트 재구성 로직 확인 필요)

---

## 부록: 코드 참조

### 주요 파일
- `lawfirm_langgraph/core/workflow/processors/workflow_document_processor.py`: 문서 필터링 및 프롬프트 구성
- `lawfirm_langgraph/core/agents/handlers/answer_formatter.py`: 참고자료 추출
- `lawfirm_langgraph/core/services/unified_prompt_manager.py`: 프롬프트 생성 및 Citation 지시사항

### 주요 함수
- `build_prompt_optimized_context()`: 프롬프트에 문서 포함
- `_extract_and_process_sources()`: 참고자료 추출
- `_build_final_prompt()`: 최종 프롬프트 생성

