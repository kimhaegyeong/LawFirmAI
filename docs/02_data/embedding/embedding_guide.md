# 벡터 임베딩 시스템 가이드

LawFirmAI 프로젝트의 벡터 임베딩 생성, 관리 및 최적화에 대한 종합 가이드입니다.

## 📋 목차

1. [시스템 개요](#시스템-개요)
2. [모델 구성](#모델-구성)
3. [벡터화 프로세스](#벡터화-프로세스)
4. [성능 최적화](#성능-최적화)
5. [테스트 및 검증](#테스트-및-검증)
6. [문제 해결](#문제-해결)

## 🎯 시스템 개요

LawFirmAI는 법률 문서의 의미적 검색을 위해 Sentence-BERT 기반 벡터 임베딩을 사용합니다.

### 핵심 특징

- **모델**: jhgan/ko-sroberta-multitask (한국어 특화)
- **차원**: 768차원 벡터
- **인덱스**: FAISS 기반 고속 검색
- **양자화**: Float16 양자화로 메모리 50% 절약
- **지연 로딩**: 필요시에만 모델 로딩

## 🤖 모델 구성

### 현재 활성 모델

| 모델명 | 상태 | 데이터 타입 | 파일 수 | 용도 |
|--------|------|-------------|---------|------|
| `ml_enhanced_ko_sroberta` | ✅ 활성 | 법령 데이터 | 7개 | 기본 법령 검색 |
| `ml_enhanced_ko_sroberta_precedents` | ✅ 활성 | 판례 데이터 | 2개 | 판례 검색 |

### 개발보류 모델

| 모델명 | 상태 | 이유 |
|--------|------|------|
| `ml_enhanced_bge_m3` | ❌ 보류 | 메모리 사용량 과다, 현재 모델로 충분 |

## 🔄 벡터화 프로세스

### 1. 데이터 전처리

```bash
# 법령 데이터 전처리
python scripts/data_processing/preprocessing/preprocess_laws.py

# 판례 데이터 전처리 (카테고리별)
python scripts/ml_training/vector_embedding/incremental_precedent_vector_builder.py --category civil
python scripts/ml_training/vector_embedding/incremental_precedent_vector_builder.py --category criminal  
python scripts/ml_training/vector_embedding/incremental_precedent_vector_builder.py --category family
```

### 2. 벡터 임베딩 생성

벡터 임베딩은 자동으로 다음 단계를 거칩니다:

1. **텍스트 청킹**: 문서를 의미적 단위로 분할
2. **임베딩 생성**: Sentence-BERT로 벡터 변환
3. **FAISS 인덱스**: 고속 검색을 위한 인덱스 생성
4. **메타데이터 저장**: 검색 결과에 필요한 정보 저장

### 3. 파일 구조

```
data/embeddings/
├── ml_enhanced_ko_sroberta/           # 법령 데이터
│   ├── ml_enhanced_faiss_index.faiss # FAISS 인덱스
│   ├── ml_enhanced_faiss_index.json  # 인덱스 메타데이터
│   ├── ml_enhanced_stats.json        # 통계 정보
│   └── checkpoint.json               # 체크포인트
└── ml_enhanced_ko_sroberta_precedents/ # 판례 데이터
    ├── ml_enhanced_ko_sroberta_precedents.faiss
    └── ml_enhanced_ko_sroberta_precedents.json
```

## ⚡ 성능 최적화

### 메모리 최적화

1. **Float16 양자화**
   - 메모리 사용량 50% 감소
   - 성능 유지 (검색 정확도 동일)

2. **지연 로딩**
   - 필요시에만 모델 로딩
   - 초기 시작 시간 단축

3. **메모리 압축**
   - 불필요한 데이터 즉시 해제
   - 자동 가비지 컬렉션

### 검색 성능 향상

1. **하이브리드 검색**
   - 벡터 유사도 + 키워드 매칭
   - 카테고리 부스팅
   - 품질 점수 부스팅

2. **향상된 스코어링**
   - 기본 벡터 점수: 85%
   - 키워드 매칭: 10%
   - 카테고리 부스팅: 3%
   - 품질 부스팅: 2%

## 🧪 테스트 및 검증

### 벡터 검색 테스트

현재 시스템의 검색 성능:

| 테스트 쿼리 | 최고 점수 | 카테고리 | 상태 |
|-------------|-----------|----------|------|
| "계약 위반 손해배상" | 0.618 | civil | ✅ |
| "이혼 재산분할" | 0.610 | family | ✅ |
| "살인 미수" | 0.520 | criminal | ✅ |
| "교통사고 과실" | 0.685 | criminal | ✅ |
| "상속 분쟁" | 0.582 | criminal, family | ✅ |

**검색 성공률**: 100% (5/5 쿼리 성공)

### 데이터 현황

- **법령 데이터**: 4,321개 조문
- **판례 데이터**: 6,285개 텍스트 청크
- **총 벡터화된 데이터**: 10,606개 문서

## 🔧 문제 해결

### 일반적인 문제

1. **벡터 검색 결과 없음**
   ```python
   # 인덱스 강제 로드
   vector_store.load_index("data/embeddings/ml_enhanced_ko_sroberta_precedents")
   ```

2. **메모리 부족**
   - Float16 양자화 활성화 확인
   - 지연 로딩 설정 확인

3. **검색 점수 낮음**
   - 하이브리드 검색 활성화 확인
   - 키워드 매칭 가중치 조정

### 로그 확인

```bash
# 벡터화 로그 확인
tail -f logs/vector_embedding.log

# 검색 성능 로그
tail -f logs/search_performance.log
```

## 📊 성능 모니터링

### 주요 메트릭

- **메모리 사용량**: 429MB 절약 (34.3% 감소)
- **검색 속도**: 평균 0.124초
- **검색 정확도**: 100% 성공률
- **벡터 일관성**: 100% 일치

### 모니터링 도구

```python
from source.data.vector_store import LegalVectorStore

# 벡터 스토어 상태 확인
vector_store = LegalVectorStore()
vector_store.load_index("data/embeddings/ml_enhanced_ko_sroberta_precedents")

# 검색 테스트
results = vector_store.search("테스트 쿼리", top_k=5)
print(f"검색 결과: {len(results)}개")
```

## 🚀 향후 계획

1. **ko-sroberta 모델 최적화**: 메모리 최적화 후 성능 향상 검토
2. **다국어 지원**: 영어 법률 문서 지원
3. **실시간 업데이트**: 증분 벡터화 시스템 고도화
4. **성능 튜닝**: 검색 속도 및 정확도 지속 개선

---

**최종 업데이트**: 2025-10-17  
**문서 버전**: v2.0  
**작성자**: LawFirmAI 개발팀
