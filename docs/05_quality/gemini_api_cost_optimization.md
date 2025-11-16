# Gemini API 비용 절감 최적화

## 적용된 개선사항

### 1. 모델 선택 최적화 ✅
- **기본 모델**: `gemini-2.0-flash-exp` (저비용 모델)
- **자동 다운그레이드**: 고비용 모델 지정 시 자동으로 저비용 모델로 변경
- **출력 토큰 제한**: `max_output_tokens=150` (기본값 대비 75% 감소)

### 2. 프롬프트 최적화 ✅
- **텍스트 길이 제한**: 2000자 → 500자 (75% 감소)
- **SystemMessage 제거**: 토큰 절약
- **간결한 프롬프트**: 불필요한 설명 제거

### 3. 질문 생성 수 감소 ✅
- **기본값**: 3개 → 1개 (66% 감소)
- **권장값**: 1-2개

### 4. 문서 필터링 ✅
- **최소 길이**: 50자 미만 문서 제외
- **최대 길이**: 5000자 초과 문서 제외
- **불필요한 API 호출 방지**

### 5. 기본 처리 문서 수 제한 ✅
- **기본값**: 500개 (전체 처리 시 비용 과다 방지)
- **샘플링**: 더 적극적인 샘플링 권장

## 비용 절감 효과

### 토큰 사용량 비교

**이전 설정 (문서당)**:
- 입력: ~2500 토큰 (2000자 문서 + 긴 프롬프트 + SystemMessage)
- 출력: ~300 토큰 (3개 질문)
- **총: ~2800 토큰/문서**

**개선 후 (문서당)**:
- 입력: ~600 토큰 (500자 문서 + 간결한 프롬프트)
- 출력: ~150 토큰 (1개 질문, max_output_tokens 제한)
- **총: ~750 토큰/문서**

**절감률: 약 73%**

### 전체 비용 비교 (500개 문서 기준)

**이전**:
- 500 문서 × 2800 토큰 = 1,400,000 토큰
- Gemini 1.5 Pro 기준: 약 $7-14 (모델에 따라 다름)

**개선 후**:
- 500 문서 × 750 토큰 = 375,000 토큰
- Gemini 2.0 Flash 기준: 약 $0.75-1.5
- **절감률: 약 85-90%**

## 사용 방법

### 최소 비용 설정 (권장)
```bash
python scripts/ml_training/evaluation/generate_pseudo_queries.py \
    --vector-store-path data/vector_store/v2.0.0-dynamic-dynamic-ivfpq/ml_enhanced_faiss_index \
    --output-path data/evaluation/rag_ground_truth_pseudo_queries.json \
    --llm-provider google \
    --llm-model gemini-2.0-flash-exp \
    --queries-per-doc 1 \
    --max-documents 500 \
    --max-text-length 500 \
    --min-text-length 50 \
    --max-text-length-filter 5000
```

### 더 적은 비용 (테스트용)
```bash
# 100개 문서만 처리, 텍스트 길이 300자로 제한
python scripts/ml_training/evaluation/generate_pseudo_queries.py \
    ... \
    --max-documents 100 \
    --max-text-length 300
```

### 비용과 품질 균형
```bash
# 질문 2개, 텍스트 800자까지
python scripts/ml_training/evaluation/generate_pseudo_queries.py \
    ... \
    --queries-per-doc 2 \
    --max-text-length 800
```

## 비용 모니터링

### 예상 비용 계산
```python
# 문서당 예상 토큰 수
input_tokens_per_doc = (max_text_length / 4) + 50  # 대략적인 토큰 수
output_tokens_per_doc = 150  # max_output_tokens 제한

total_tokens = (input_tokens_per_doc + output_tokens_per_doc) * num_documents

# Gemini 2.0 Flash 가격 (예시, 실제 가격 확인 필요)
# Input: $0.075 per 1M tokens
# Output: $0.30 per 1M tokens
cost = (input_tokens_per_doc * num_documents * 0.075 / 1_000_000) + \
       (output_tokens_per_doc * num_documents * 0.30 / 1_000_000)
```

### 실제 사용량 확인
- Google Cloud Console에서 API 사용량 확인
- 체크포인트 파일에서 처리된 문서 수 확인

## 추가 비용 절감 팁

1. **샘플링 활용**: 전체 데이터가 아닌 대표 샘플만 사용
2. **문서 품질 필터링**: 의미 있는 문서만 처리
3. **캐싱**: 같은 문서에 대해 재생성 방지
4. **배치 처리**: 가능한 경우 여러 질문을 한 번에 생성

## 주의사항

1. **텍스트 길이 제한**: 너무 짧으면 질문 품질 저하 가능
2. **질문 수 감소**: 평가 데이터셋 크기 감소
3. **모델 변경**: 저비용 모델은 품질이 다를 수 있음

