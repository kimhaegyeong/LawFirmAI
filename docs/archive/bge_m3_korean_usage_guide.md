# BGE-M3-Korean 임베딩 모델 사용 가이드

## 개요

LawFirmAI 프로젝트에서 BGE-M3-Korean 모델을 사용하여 한국어 텍스트 임베딩을 처리할 수 있도록 설정이 완료되었습니다.

## 주요 변경사항

### 1. 의존성 추가
- `requirements.txt`에 `FlagEmbedding>=1.2.0` 추가
- BGE-M3 모델을 위한 필수 라이브러리 설치

### 2. LegalVectorStore 클래스 업데이트
- BGE-M3 모델과 Sentence-BERT 모델 모두 지원
- 자동 모델 타입 감지 (`bge-m3` 키워드 기반)
- 임베딩 차원 자동 설정 (BGE-M3: 1024, Sentence-BERT: 768)

### 3. 벡터 빌더 업데이트
- 기본 모델을 BGE-M3로 변경
- CPU 최적화 버전에서 BGE-M3 지원

## 사용법

### 1. 기본 사용법

```python
from source.data.vector_store import LegalVectorStore

# BGE-M3 모델 사용
vector_store = LegalVectorStore(
    model_name="BAAI/bge-m3",
    dimension=1024,
    index_type="flat"
)

# 텍스트 임베딩 생성
texts = ["계약서 검토 요청", "민법 제1조는 민법의 기본 원칙을 규정한다"]
embeddings = vector_store.generate_embeddings(texts)

# 문서 추가
documents = [
    {
        'text': '민법 제1조는 민법의 기본 원칙을 규정한다',
        'metadata': {
            'law_name': '민법',
            'article_number': '제1조',
            'category': '민사법'
        }
    }
]

texts = [doc['text'] for doc in documents]
metadatas = [doc['metadata'] for doc in documents]
vector_store.add_documents(texts, metadatas)

# 검색
results = vector_store.search("계약서 검토", top_k=5)
```

### 2. 벡터 빌더 사용법

```bash
# BGE-M3 모델로 벡터 임베딩 생성
python scripts/build_ml_enhanced_vector_db_cpu_optimized.py \
    --input data/processed \
    --output data/embeddings/ml_enhanced_bge_m3 \
    --batch-size 20 \
    --chunk-size 200
```

### 3. 모델 비교

| 모델 | 차원 | 속도 | 특징 |
|------|------|------|------|
| Sentence-BERT (ko-sroberta-multitask) | 768 | 빠름 | 한국어 특화 |
| BGE-M3 | 1024 | 보통 | 다국어 지원, 더 풍부한 의미 정보 |

## 성능 테스트 결과

테스트 환경: Windows 10, CPU만 사용

```
Sentence-BERT embedding time: 0.1057s, dimension: 768
BGE-M3 embedding time: 0.8658s, dimension: 1024
```

## 주요 특징

### BGE-M3의 장점
1. **다국어 지원**: 한국어뿐만 아니라 영어, 중국어 등 다양한 언어 지원
2. **높은 차원**: 1024차원으로 더 풍부한 의미 정보 제공
3. **최신 기술**: 최신 임베딩 기술 적용

### 한국어 특화 기능
1. **자동 정규화**: cosine similarity를 위한 자동 정규화
2. **메모리 최적화**: CPU 환경에서의 메모리 사용량 최적화
3. **배치 처리**: 대량 텍스트 처리 시 배치 단위로 처리

## 주의사항

1. **메모리 사용량**: BGE-M3는 더 높은 차원을 사용하므로 메모리 사용량이 증가합니다.
2. **처리 속도**: Sentence-BERT 대비 약 8배 느린 처리 속도를 보입니다.
3. **모델 크기**: BGE-M3 모델은 더 크므로 초기 다운로드 시간이 필요합니다.

## 문제 해결

### 일반적인 문제
1. **FlagEmbedding 설치 오류**: `pip install FlagEmbedding` 실행
2. **메모리 부족**: 배치 크기와 청크 크기를 줄여서 사용
3. **모델 다운로드 실패**: 인터넷 연결 확인 및 HuggingFace Hub 접근 권한 확인

### 성능 최적화
1. **배치 크기 조정**: 메모리에 따라 10-50 사이에서 조정
2. **청크 크기 조정**: 처리 속도에 따라 100-500 사이에서 조정
3. **인덱스 타입 선택**: 
   - `flat`: 정확도 우선
   - `ivf`: 속도 우선
   - `hnsw`: 균형

## 예제 코드

완전한 예제는 `test_bge_m3_korean.py` 파일을 참조하세요.

```python
# 테스트 실행
python test_bge_m3_korean.py
```

이 가이드를 통해 BGE-M3-Korean 모델을 효과적으로 사용할 수 있습니다.
