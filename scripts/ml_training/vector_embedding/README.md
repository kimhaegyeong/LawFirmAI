# Vector Embedding Scripts

벡터 임베딩 생성, 관리, 테스트를 담당하는 스크립트들입니다.

## 📁 파일 목록

### 벡터 임베딩 생성
- **`build_ml_enhanced_vector_db.py`** (22.0 KB) - ML 강화 벡터 DB 구축
- **`build_ml_enhanced_vector_db_optimized.py`** (15.0 KB) - 최적화된 ML 강화 벡터 DB 구축
- **`build_ml_enhanced_vector_db_cpu_optimized.py`** (25.0 KB) - CPU 최적화된 ML 강화 벡터 DB 구축
- **`build_resumable_vector_db.py`** (21.0 KB) - 재시작 가능한 벡터 DB 구축

### 벡터 DB 관리
- **`rebuild_improved_vector_db.py`** (5.3 KB) - 개선된 벡터 DB 재구축

### 테스트 스크립트
- **`test_faiss_direct.py`** (3.9 KB) - FAISS 직접 테스트
- **`test_vector_embedding_basic.py`** (4.1 KB) - 기본 벡터 임베딩 테스트

## 🚀 사용법

### 기본 벡터 임베딩 생성
```bash
# ML 강화 벡터 임베딩 생성
python scripts/vector_embedding/build_ml_enhanced_vector_db.py \
    --input data/processed/assembly/law/20251013_ml \
    --output data/embeddings/ml_enhanced

# CPU 최적화 버전 (권장)
python scripts/vector_embedding/build_ml_enhanced_vector_db_cpu_optimized.py \
    --input data/processed/assembly/law/20251013_ml \
    --output data/embeddings/ml_enhanced_ko_sroberta \
    --batch-size 20 \
    --chunk-size 200
```

### 재시작 가능한 벡터 임베딩
```bash
# 체크포인트 지원 벡터 임베딩
python scripts/vector_embedding/build_resumable_vector_db.py \
    --input data/processed/assembly/law/20251013_ml \
    --output data/embeddings/resumable \
    --resume
```

### 벡터 DB 재구축
```bash
# 개선된 벡터 DB 재구축
python scripts/vector_embedding/rebuild_improved_vector_db.py
```

### 테스트 실행
```bash
# 기본 벡터 임베딩 테스트
python scripts/vector_embedding/test_vector_embedding_basic.py

# FAISS 직접 테스트
python scripts/vector_embedding/test_faiss_direct.py
```

## 🔧 설정

### 모델 설정
- **기본 모델**: `jhgan/ko-sroberta-multitask` (768차원)
- **대안 모델**: `BAAI/bge-m3` (1024차원)
- **인덱스 타입**: `flat` (정확도 우선)

### 성능 설정
- **배치 크기**: 20 (메모리 효율성)
- **청크 크기**: 200 (처리 효율성)
- **체크포인트**: 매 10개 청크마다 저장

### 환경 변수
```bash
# 모델 경로 설정
export MODEL_PATH="models/"

# 임베딩 출력 경로
export EMBEDDING_OUTPUT="data/embeddings/"

# 로그 레벨
export LOG_LEVEL="INFO"
```

## 📊 성능 지표

### 처리 성능
| 모델 | 차원 | 처리 속도 | 메모리 사용량 |
|------|------|-----------|---------------|
| ko-sroberta-multitask | 768 | 1-2분/청크 | 190MB |
| BGE-M3 | 1024 | 6-7분/청크 | 16.5GB |

### 최적화 결과
- **처리 시간**: 5-7배 단축 (88시간 → 2시간 46분)
- **메모리 효율**: 99% 감소 (16.5GB → 190MB)
- **검색 성능**: 평균 0.015초

## 🛡️ 안전성 기능

### 체크포인트 시스템
- **자동 저장**: 매 10개 청크마다 진행 상황 저장
- **재시작 지원**: 중단된 지점부터 이어서 작업
- **진행률 추적**: 실시간 진행 상황 및 예상 완료 시간

### Graceful Shutdown
- **시그널 처리**: SIGTERM, SIGINT, SIGBREAK 지원
- **안전한 종료**: 현재 청크 완료 후 체크포인트 저장
- **데이터 무결성**: 부분 완료된 작업 보호

## 📁 출력 파일

### 생성되는 파일들
```
data/embeddings/ml_enhanced_ko_sroberta/
├── ml_enhanced_faiss_index.faiss    # FAISS 인덱스 (456.5 MB)
├── ml_enhanced_faiss_index.json    # 메타데이터 (326.7 MB)
├── ml_enhanced_stats.json          # 처리 통계
└── embedding_checkpoint.json        # 체크포인트 (완료 후 삭제)
```

### 메타데이터 구조
```json
{
  "model_name": "jhgan/ko-sroberta-multitask",
  "dimension": 768,
  "index_type": "flat",
  "document_count": 155819,
  "created_at": "2025-10-15T19:47:36.695342",
  "document_metadata": [...]
}
```

## 🔍 테스트 및 검증

### 벡터 임베딩 검증
```bash
# 기본 테스트
python scripts/vector_embedding/test_vector_embedding_basic.py

# FAISS 직접 테스트
python scripts/vector_embedding/test_faiss_direct.py

# 최종 성능 테스트
python scripts/tests/test_final_vector_embedding_performance.py
```

### 검증 항목
- ✅ FAISS 인덱스 정상 로드
- ✅ 155,819개 벡터 완전 생성
- ✅ 메타데이터 완전 저장
- ✅ 검색 기능 정상 작동
- ✅ 체크포인트 시스템 작동

## 🛠️ 트러블슈팅

### 일반적인 문제
1. **메모리 부족**: 배치 크기 및 청크 크기 조정
2. **모델 로딩 실패**: 인터넷 연결 및 모델 경로 확인
3. **인덱스 로딩 실패**: 파일 경로 및 권한 확인

### 성능 최적화
```bash
# CPU 사용량 최적화
python scripts/vector_embedding/build_ml_enhanced_vector_db_cpu_optimized.py \
    --batch-size 10 \
    --chunk-size 100

# 메모리 사용량 최적화
python scripts/vector_embedding/build_ml_enhanced_vector_db_optimized.py \
    --batch-size 5 \
    --chunk-size 50
```

---

**마지막 업데이트**: 2025-10-15  
**관리자**: LawFirmAI 개발팀
