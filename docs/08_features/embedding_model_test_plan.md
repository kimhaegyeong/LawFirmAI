# 임베딩 모델 테스트 계획서

## 📋 목차

1. [테스트 개요](#테스트-개요)
2. [후보 모델 목록](#후보-모델-목록)
3. [테스트 환경 설정](#테스트-환경-설정)
4. [평가 지표 및 방법](#평가-지표-및-방법)
5. [테스트 데이터셋](#테스트-데이터셋)
6. [테스트 절차](#테스트-절차)
7. [결과 분석 및 선택 기준](#결과-분석-및-선택-기준)
8. [예상 일정](#예상-일정)

---

## 테스트 개요

### 목적

현재 사용 중인 `woong0322/ko-legal-sbert-finetuned` 모델의 성능을 개선하기 위해 다양한 임베딩 모델을 테스트하고, 최적의 모델을 선정합니다.

### 배경

- 현재 모델의 벡터 검색 성능이 낮음 (유사도 0.11~0.12)
- TSVECTOR 검색은 잘 작동하지만 벡터 검색의 이점을 살리지 못함
- 다양한 최신 모델들이 법률 도메인에 더 적합할 수 있음

### 테스트 범위

- **벡터 검색 성능**: 유사도 점수, 검색 정확도
- **의미적 매칭 품질**: 관련 문서 검색 능력
- **성능**: 검색 속도, 메모리 사용량
- **호환성**: 기존 시스템과의 통합 가능성

---

## 후보 모델 목록

### 현재 모델

| 모델명 | 차원 | 특화 | 상태 |
|--------|------|------|------|
| `woong0322/ko-legal-sbert-finetuned` | 768 | 법률 도메인 파인튜닝 | ✅ 기준 모델 |

### 테스트 대상 모델

| 모델명 | 차원 | 예상 개선폭 | 근거 | 우선순위 |
|--------|------|------------|------|----------|
| `BAAI/bge-m3` | 1024 | ⭐⭐⭐⭐⭐ | multi-vector+sparse 조합, 최신 기술 | 🔴 높음 |
| `BAAI/bge-large-ko-v1.5` | 1024 | ⭐⭐⭐⭐ | 의미 구분력 가장 높음, 한국어 특화 | 🔴 높음 |
| `jhgan/ko-sroberta-multitask` | 768 | ⭐⭐⭐ | SBERT보다 확실하게 더 좋음, 한국어 특화 | 🟡 중간 |
| `BAAI/bge-base-ko-v1.5` | 768 | ⭐⭐⭐ | 안정적인 일정 개선, 한국어 특화 | 🟡 중간 |
| `upstage/solar-embedding-ko` | 768 | ⭐⭐⭐⭐ | 긴 문서 대응 매우 강함, 한국어 특화 | 🟡 중간 |
| `intfloat/multilingual-e5-large` | 1024 | ⭐⭐⭐ | RAG 전용 구조, 다국어 지원 | 🟢 낮음 |

**참고**: 
- `E5-ko`는 HuggingFace에서 직접 제공되지 않을 수 있으므로 `multilingual-e5-large`로 대체
- 실제 모델명은 HuggingFace에서 확인 필요

---

## 테스트 환경 설정

### 하드웨어 요구사항

- **CPU**: 최소 8코어
- **RAM**: 최소 16GB (권장 32GB)
- **GPU**: 선택사항 (테스트 속도 향상)
- **디스크**: 최소 50GB 여유 공간

### 소프트웨어 요구사항

```python
# 필수 패키지
sentence-transformers>=2.2.0
torch>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### 테스트 스크립트 구조

```
scripts/
├── embedding_test/
│   ├── __init__.py
│   ├── model_loader.py          # 모델 로딩 유틸리티
│   ├── test_runner.py           # 테스트 실행기
│   ├── evaluator.py             # 평가 지표 계산
│   ├── dataset_loader.py        # 테스트 데이터셋 로더
│   ├── similarity_tester.py     # 유사도 테스트
│   ├── search_tester.py         # 검색 성능 테스트
│   └── report_generator.py      # 결과 리포트 생성
├── embedding_test/
│   ├── config/
│   │   └── test_config.yaml     # 테스트 설정
│   ├── data/
│   │   ├── queries.json         # 테스트 쿼리
│   │   └── documents.json       # 테스트 문서
│   └── results/
│       └── {model_name}/        # 모델별 결과
```

---

## 평가 지표 및 방법

### 1. 유사도 점수 분포 분석

**목적**: 모델이 생성하는 유사도 점수의 분포를 분석

**지표**:
- 평균 유사도
- 중앙값 유사도
- 표준편차
- 최소/최대 유사도
- 유사도 분포 히스토그램

**측정 방법**:
```python
# 샘플 쿼리-문서 쌍에 대한 유사도 측정
for query in test_queries:
    query_embedding = model.encode(query)
    for doc in test_documents:
        doc_embedding = model.encode(doc)
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append(similarity)

# 통계 계산
mean_similarity = np.mean(similarities)
median_similarity = np.median(similarities)
std_similarity = np.std(similarities)
```

### 2. 검색 정확도 (Retrieval Accuracy)

**목적**: 실제 검색 시나리오에서 관련 문서를 찾는 능력 평가

**지표**:
- **Top-K 정확도** (K=1, 3, 5, 10)
- **MRR (Mean Reciprocal Rank)**
- **NDCG@K** (Normalized Discounted Cumulative Gain)

**측정 방법**:
```python
# 각 쿼리에 대해 검색 수행
for query, relevant_docs in test_cases:
    results = search(query, top_k=10)
    
    # Top-K 정확도
    top_k_accuracy = len(set(results[:k]) & set(relevant_docs)) / k
    
    # MRR 계산
    for i, doc in enumerate(results):
        if doc in relevant_docs:
            mrr += 1.0 / (i + 1)
            break
```

### 3. 의미적 매칭 품질

**목적**: 의미적으로 관련된 문서를 얼마나 잘 찾는지 평가

**테스트 케이스**:
- **동의어 매칭**: "계약 해지" vs "계약 해제"
- **관련 개념 매칭**: "손해배상" vs "불법행위"
- **상위/하위 개념**: "계약" vs "매매계약"

**지표**:
- 관련 문서 평균 유사도
- 무관 문서 평균 유사도
- 구분력 (Discrimination): 관련-무관 유사도 차이

### 4. 성능 지표

**목적**: 실제 운영 환경에서의 성능 평가

**지표**:
- **임베딩 생성 속도**: 문서당 평균 시간
- **검색 속도**: 쿼리당 평균 시간
- **메모리 사용량**: 모델 로딩 후 메모리
- **모델 크기**: 디스크 사용량

**측정 방법**:
```python
import time
import psutil

# 임베딩 생성 속도
start_time = time.time()
embeddings = model.encode(documents)
embedding_time = (time.time() - start_time) / len(documents)

# 메모리 사용량
process = psutil.Process()
memory_mb = process.memory_info().rss / 1024 / 1024
```

### 5. 호환성 평가

**목적**: 기존 시스템과의 통합 가능성 평가

**평가 항목**:
- 벡터 차원 호환성
- API 호환성
- 기존 인덱스 재사용 가능성
- 마이그레이션 난이도

---

## 테스트 데이터셋

### 1. 쿼리 데이터셋

**구성**:
- 법률 질문 50개 (다양한 유형)
- 각 질문에 대한 관련 문서 ID 목록 (정답)

**카테고리**:
- 계약법 (10개)
- 불법행위법 (10개)
- 가족법 (10개)
- 형법 (10개)
- 기타 (10개)

**예시**:
```json
{
  "queries": [
    {
      "id": "q001",
      "text": "계약 해지 사유에 대해 알려주세요",
      "category": "계약법",
      "relevant_doc_ids": ["doc_001", "doc_002", "doc_003"]
    },
    ...
  ]
}
```

### 2. 문서 데이터셋

**구성**:
- 법령 조문 100개
- 판례 100개
- 총 200개 문서

**메타데이터**:
- 문서 ID
- 문서 타입 (statute/precedent)
- 카테고리
- 내용

**예시**:
```json
{
  "documents": [
    {
      "id": "doc_001",
      "type": "statute",
      "category": "계약법",
      "content": "민법 제543조 (계약의 해제)..."
    },
    ...
  ]
}
```

### 3. 벤치마크 데이터셋

**기존 데이터 활용**:
- 현재 데이터베이스의 샘플 문서 사용
- 실제 검색 쿼리 로그 활용

---

## 테스트 절차

### Phase 1: 환경 준비 (1일)

1. **테스트 스크립트 작성**
   ```bash
   # 스크립트 생성
   mkdir -p scripts/embedding_test
   touch scripts/embedding_test/{model_loader,test_runner,evaluator}.py
   ```

2. **테스트 데이터셋 준비**
   - 쿼리 데이터셋 생성
   - 문서 데이터셋 생성
   - 정답 레이블 생성

3. **기준 모델 테스트**
   - 현재 모델로 기준 성능 측정
   - 결과를 기준선으로 설정

### Phase 2: 모델별 테스트 (각 모델 1-2일)

#### 2.1 모델 로딩 및 검증

```python
# scripts/embedding_test/model_loader.py
from sentence_transformers import SentenceTransformer
import torch

def load_model(model_name: str, device: str = "cpu"):
    """모델 로딩"""
    try:
        model = SentenceTransformer(model_name, device=device)
        print(f"✅ Model loaded: {model_name}")
        return model
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return None

def verify_model(model, test_text: str = "테스트"):
    """모델 검증"""
    embedding = model.encode(test_text)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dtype: {embedding.dtype}")
    return embedding
```

#### 2.2 유사도 테스트

```python
# scripts/embedding_test/similarity_tester.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def test_similarity_distribution(model, queries, documents):
    """유사도 분포 테스트"""
    similarities = []
    
    for query in queries:
        query_emb = model.encode(query)
        for doc in documents:
            doc_emb = model.encode(doc)
            sim = cosine_similarity([query_emb], [doc_emb])[0][0]
            similarities.append(sim)
    
    return {
        "mean": np.mean(similarities),
        "median": np.median(similarities),
        "std": np.std(similarities),
        "min": np.min(similarities),
        "max": np.max(similarities),
        "distribution": similarities
    }
```

#### 2.3 검색 성능 테스트

```python
# scripts/embedding_test/search_tester.py
def test_search_accuracy(model, test_cases, top_k=10):
    """검색 정확도 테스트"""
    results = {
        "top_1_accuracy": [],
        "top_3_accuracy": [],
        "top_5_accuracy": [],
        "top_10_accuracy": [],
        "mrr": []
    }
    
    for query, relevant_doc_ids in test_cases:
        # 검색 수행
        query_emb = model.encode(query)
        doc_embs = model.encode([doc["content"] for doc in documents])
        
        similarities = cosine_similarity([query_emb], doc_embs)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 정확도 계산
        retrieved_ids = [documents[i]["id"] for i in top_indices]
        
        for k in [1, 3, 5, 10]:
            accuracy = len(set(retrieved_ids[:k]) & set(relevant_doc_ids)) / k
            results[f"top_{k}_accuracy"].append(accuracy)
        
        # MRR 계산
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_doc_ids:
                results["mrr"].append(1.0 / (i + 1))
                break
        else:
            results["mrr"].append(0.0)
    
    # 평균 계산
    return {k: np.mean(v) for k, v in results.items()}
```

#### 2.4 성능 테스트

```python
# scripts/embedding_test/performance_tester.py
import time
import psutil

def test_performance(model, test_documents):
    """성능 테스트"""
    # 임베딩 생성 속도
    start_time = time.time()
    embeddings = model.encode(test_documents)
    embedding_time = (time.time() - start_time) / len(test_documents)
    
    # 메모리 사용량
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    # 모델 크기
    import os
    model_path = model._modules['0'].auto_model.config.name_or_path
    # 모델 크기 계산 (대략적)
    
    return {
        "embedding_time_per_doc": embedding_time,
        "memory_usage_mb": memory_mb,
        "embedding_dimension": embeddings.shape[1]
    }
```

### Phase 3: 결과 분석 (2일)

1. **결과 수집 및 정리**
2. **모델별 성능 비교**
3. **최적 모델 선정**

---

## 결과 분석 및 선택 기준

### 평가 매트릭스

| 모델 | 유사도 평균 | Top-5 정확도 | 검색 속도 | 메모리 사용 | 종합 점수 |
|------|------------|-------------|----------|------------|----------|
| 기준 모델 | 0.12 | 0.40 | 기준 | 기준 | - |
| bge-m3 | ? | ? | ? | ? | ? |
| bge-large-ko | ? | ? | ? | ? | ? |
| ... | ... | ... | ... | ... | ... |

### 선택 기준

#### 필수 조건

1. **유사도 평균 > 0.30**
   - 현재 모델(0.12)보다 최소 2.5배 향상 필요

2. **Top-5 정확도 > 0.60**
   - 현재 모델(0.40)보다 최소 1.5배 향상 필요

3. **검색 속도 < 100ms/쿼리**
   - 실용적인 성능 보장

#### 우선순위 기준

1. **검색 정확도** (가중치: 40%)
   - Top-K 정확도
   - MRR

2. **유사도 품질** (가중치: 30%)
   - 평균 유사도
   - 의미적 구분력

3. **성능** (가중치: 20%)
   - 검색 속도
   - 메모리 사용량

4. **호환성** (가중치: 10%)
   - 기존 시스템 통합 난이도
   - 마이그레이션 비용

### 종합 점수 계산

```python
def calculate_composite_score(model_results):
    """종합 점수 계산"""
    # 정규화된 점수
    accuracy_score = model_results["top_5_accuracy"] * 0.4
    similarity_score = (model_results["mean_similarity"] / 1.0) * 0.3
    performance_score = (1.0 / model_results["search_time_ms"]) * 0.2
    compatibility_score = model_results["compatibility"] * 0.1
    
    return accuracy_score + similarity_score + performance_score + compatibility_score
```

---

## 예상 일정

### 전체 일정: 2-3주

| 단계 | 작업 | 소요 시간 | 담당 |
|------|------|----------|------|
| **Phase 1** | 환경 준비 | 1일 | 개발자 |
| **Phase 2** | 모델별 테스트 | 6-12일 | 개발자 |
| - | bge-m3 | 2일 | |
| - | bge-large-ko | 2일 | |
| - | ko-sroberta-multitask | 1일 | |
| - | bge-base-ko | 1일 | |
| - | solar-embedding-ko | 1일 | |
| - | multilingual-e5-large | 1일 | |
| **Phase 3** | 결과 분석 | 2일 | 개발자 + 검토자 |
| **Phase 4** | 최종 선정 및 문서화 | 1일 | 개발자 |

### 상세 일정

#### Week 1: 환경 준비 및 우선순위 모델 테스트

- **Day 1**: 환경 준비, 테스트 스크립트 작성
- **Day 2-3**: bge-m3 테스트
- **Day 4-5**: bge-large-ko 테스트

#### Week 2: 나머지 모델 테스트

- **Day 1**: ko-sroberta-multitask 테스트
- **Day 2**: bge-base-ko 테스트
- **Day 3**: solar-embedding-ko 테스트
- **Day 4**: multilingual-e5-large 테스트

#### Week 3: 결과 분석 및 선정

- **Day 1-2**: 결과 분석 및 비교
- **Day 3**: 최종 모델 선정 및 문서화

---

## 테스트 실행 가이드

### 1. 환경 설정

```bash
# 가상환경 생성
python -m venv venv_embedding_test
source venv_embedding_test/bin/activate  # Windows: venv_embedding_test\Scripts\activate

# 패키지 설치
pip install sentence-transformers torch numpy scikit-learn pandas matplotlib seaborn psutil
```

### 2. 테스트 데이터 준비

```bash
# 테스트 데이터 생성
python scripts/embedding_test/dataset_loader.py \
    --output_dir scripts/embedding_test/data \
    --num_queries 50 \
    --num_documents 200
```

### 3. 모델 테스트 실행

```bash
# 개별 모델 테스트
python scripts/embedding_test/test_runner.py \
    --model BAAI/bge-m3 \
    --output_dir scripts/embedding_test/results/bge-m3

# 모든 모델 일괄 테스트
python scripts/embedding_test/test_runner.py \
    --models all \
    --output_dir scripts/embedding_test/results
```

### 4. 결과 분석

```bash
# 결과 리포트 생성
python scripts/embedding_test/report_generator.py \
    --results_dir scripts/embedding_test/results \
    --output report.html
```

---

## 리스크 및 대응 방안

### 리스크 1: 모델 다운로드 실패

**대응**: 
- 모델을 미리 다운로드하여 로컬 캐시에 저장
- 대안 모델 준비

### 리스크 2: 메모리 부족

**대응**:
- 배치 처리로 메모리 사용량 제한
- GPU 사용 시 CPU 모드로 폴백

### 리스크 3: 테스트 시간 초과

**대응**:
- 샘플링을 통한 테스트 데이터 축소
- 병렬 처리로 테스트 시간 단축

### 리스크 4: 모델 호환성 문제

**대응**:
- 각 모델의 벡터 차원 확인
- 기존 인덱스 재생성 필요성 평가

---

## 다음 단계

테스트 완료 후:

1. **최적 모델 선정**
2. **프로덕션 환경 통합 계획 수립**
3. **기존 인덱스 재생성 계획**
4. **성능 모니터링 계획**

---

**작성일**: 2025-11-30  
**예상 시작일**: 2025-12-01  
**예상 완료일**: 2025-12-20

