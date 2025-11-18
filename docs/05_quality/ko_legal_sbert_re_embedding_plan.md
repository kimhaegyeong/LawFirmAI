# Ko-Legal-SBERT 재임베딩 계획

## 개요

Ko-Legal-SBERT 모델을 사용하기 위해 기존 데이터를 재임베딩하는 계획입니다.

## 사전 준비

### 1. 모델 확인 및 설정

**지원 모델**:
- ✅ `woong0322/ko-legal-sbert-finetuned` - Ko-Legal-SBERT (한국 법률 도메인 특화 모델, 확인됨)
- ⚠️ `LegalInsight/PretrainedModel` - 법률 인사이트 사전 학습 모델 (HuggingFace에서 정확한 경로 확인 필요)
- `snunlp/KR-SBERT-V40K-klueNLI-augSTS` - 기본 한국어 SBERT (기본값)

**모델 차원**: 768차원 (기본값)

**모델 설정 방법**:

#### 환경 변수 설정 (권장)
```bash
# .env 파일 또는 시스템 환경 변수
EMBEDDING_MODEL=woong0322/ko-legal-sbert-finetuned
```

#### 코드에서 직접 설정
```python
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig

config = LangGraphConfig.from_env()
config.embedding_model = "woong0322/ko-legal-sbert-finetuned"
```

**모델 로딩 테스트**:
```python
from sentence_transformers import SentenceTransformer

try:
    model = SentenceTransformer("woong0322/ko-legal-sbert-finetuned")
    print(f"Model loaded: {model.get_sentence_embedding_dimension()} dimensions")
except Exception as e:
    print(f"Model loading failed: {e}")
```

### 2. 모델 호환성 확인

**중요**: 기존 데이터베이스의 임베딩이 다른 모델로 생성된 경우, 모델 변경 시 재임베딩이 필요합니다.

- 기존 임베딩과의 호환성 확인
- 필요시 데이터 재임베딩 (이 문서의 재임베딩 단계 참조)
- FAISS 인덱스 재구축

**차원 불일치 오류 해결**:
1. 데이터베이스에서 현재 사용 중인 모델 확인
2. 새 모델의 차원 확인
3. 필요시 데이터 재임베딩

### 3. 데이터베이스 백업

**중요**: 재임베딩 전 반드시 데이터베이스 백업

```bash
# PowerShell
Copy-Item data/lawfirm_v2.db data/lawfirm_v2.db.backup_$(Get-Date -Format "yyyyMMdd_HHmmss")
```

### 4. 현재 임베딩 상태 확인

```bash
# 현재 사용 중인 모델 확인
python -c "
import sqlite3
conn = sqlite3.connect('data/lawfirm_v2.db')
cursor = conn.execute('SELECT DISTINCT model, COUNT(*) as count FROM embeddings GROUP BY model')
for row in cursor.fetchall():
    print(f'Model: {row[0]}, Count: {row[1]}')
"

# 총 청크 수 확인
python -c "
import sqlite3
conn = sqlite3.connect('data/lawfirm_v2.db')
cursor = conn.execute('SELECT COUNT(*) FROM text_chunks')
print(f'Total chunks: {cursor.fetchone()[0]}')
"
```

## 재임베딩 단계

### Phase 1: 소규모 테스트 (필수)

**목적**: 모델이 정상 작동하는지, 차원이 맞는지 확인

#### 1-1. 모델 테스트

```bash
# 환경 변수 설정
$env:EMBEDDING_MODEL="Ko-Legal-SBERT"

# 모델 로딩 테스트
python -c "
from scripts.utils.embeddings import SentenceEmbedder
embedder = SentenceEmbedder('woong0322/ko-legal-sbert-finetuned')
print(f'Model loaded: {embedder.model_name}')
print(f'Dimension: {embedder.dim}')
test_text = '민법 제1조의 내용은 무엇인가요?'
vector = embedder.encode([test_text])
print(f'Vector shape: {vector.shape}')
"
```

#### 1-2. 소규모 재임베딩 테스트

```bash
# 10개 문서만 테스트
python scripts/migrations/re_embed_existing_data.py \
    --db data/lawfirm_v2.db \
    --source-type statute_article \
    --chunking-strategy standard \
    --batch-size 64 \
    --limit 10 \
    --dry-run

# 실제 실행 (10개만)
python scripts/migrations/re_embed_existing_data.py \
    --db data/lawfirm_v2.db \
    --source-type statute_article \
    --chunking-strategy standard \
    --batch-size 64 \
    --limit 10
```

#### 1-3. 결과 검증

```bash
# 재임베딩된 청크 확인
python -c "
import sqlite3
conn = sqlite3.connect('data/lawfirm_v2.db')
cursor = conn.execute('''
    SELECT model, COUNT(*) as count 
    FROM embeddings 
    WHERE model LIKE '%ko-legal-sbert%' OR model LIKE '%Legal%'
    GROUP BY model
''')
for row in cursor.fetchall():
    print(f'Model: {row[0]}, Count: {row[1]}')
"
```

### Phase 2: 단계별 재임베딩

**전략**: 소스 타입별로 순차적으로 재임베딩하여 문제 발생 시 롤백 용이하게

#### 2-1. statute_article 재임베딩

```bash
# 환경 변수 설정
$env:EMBEDDING_MODEL="Ko-Legal-SBERT"

# 재임베딩 실행
python scripts/migrations/re_embed_existing_data_optimized.py \
    --db data/lawfirm_v2.db \
    --source-type statute_article \
    --chunking-strategy standard \
    --doc-batch-size 200 \
    --commit-interval 5
```

**예상 소요 시간**: 
- 문서 수에 따라 다름
- GPU 사용 시: 약 1-2시간 (10,000개 문서 기준)
- CPU 사용 시: 약 4-8시간 (10,000개 문서 기준)

#### 2-2. case_paragraph 재임베딩

```bash
python scripts/migrations/re_embed_existing_data_optimized.py \
    --db data/lawfirm_v2.db \
    --source-type case_paragraph \
    --chunking-strategy standard \
    --doc-batch-size 200 \
    --commit-interval 5
```

#### 2-3. decision_paragraph 재임베딩

```bash
python scripts/migrations/re_embed_existing_data_optimized.py \
    --db data/lawfirm_v2.db \
    --source-type decision_paragraph \
    --chunking-strategy standard \
    --doc-batch-size 200 \
    --commit-interval 5
```

#### 2-4. interpretation_paragraph 재임베딩

```bash
python scripts/migrations/re_embed_existing_data_optimized.py \
    --db data/lawfirm_v2.db \
    --source-type interpretation_paragraph \
    --chunking-strategy standard \
    --doc-batch-size 200 \
    --commit-interval 5
```

### Phase 3: 전체 재임베딩 (선택)

모든 소스 타입을 한 번에 재임베딩:

```bash
# 환경 변수 설정
$env:EMBEDDING_MODEL="Ko-Legal-SBERT"

# 전체 재임베딩 (최적화 버전 사용)
python scripts/migrations/re_embed_existing_data_optimized.py \
    --db data/lawfirm_v2.db \
    --chunking-strategy standard \
    --doc-batch-size 200 \
    --commit-interval 5
```

**예상 소요 시간**: 
- 전체 데이터 규모에 따라 다름
- GPU 사용 시: 약 8-24시간
- CPU 사용 시: 약 24-72시간

## 스크립트 수정 필요 사항

현재 `re_embed_existing_data.py`는 기본 모델을 사용합니다. Ko-Legal-SBERT를 사용하려면:

### 옵션 1: 환경 변수 사용 (권장)

```bash
# 환경 변수로 모델 지정
$env:EMBEDDING_MODEL="woong0322/ko-legal-sbert-finetuned"
python scripts/migrations/re_embed_existing_data_optimized.py ...
```

### 옵션 2: 스크립트 수정

`scripts/migrations/re_embed_existing_data.py`의 402번째 줄 수정:

```python
# 기존
embedder = SentenceEmbedder()

# 수정
model_name = os.getenv("EMBEDDING_MODEL", "woong0322/ko-legal-sbert-finetuned")
embedder = SentenceEmbedder(model_name)
```

## FAISS 인덱스 재구축

재임베딩 완료 후 FAISS 인덱스도 재구축해야 합니다:

```bash
# FAISS 인덱스 재구축 스크립트 확인 필요
# 또는 SemanticSearchEngineV2가 자동으로 재구축할 수 있음
```

## 검증 및 테스트

### 1. 재임베딩 결과 확인

```bash
# Ko-Legal-SBERT로 재임베딩된 청크 수 확인
python -c "
import sqlite3
conn = sqlite3.connect('data/lawfirm_v2.db')
cursor = conn.execute('''
    SELECT 
        source_type,
        COUNT(DISTINCT e.chunk_id) as embedded_chunks,
        COUNT(DISTINCT tc.id) as total_chunks
    FROM text_chunks tc
    LEFT JOIN embeddings e ON tc.id = e.chunk_id 
        AND (e.model LIKE '%ko-legal-sbert%' OR e.model LIKE '%Legal%')
    GROUP BY source_type
''')
for row in cursor.fetchall():
    print(f'{row[0]}: {row[1]}/{row[2]} chunks embedded')
"
```

### 2. 검색 품질 테스트

```bash
# Before 설정으로 검색 테스트
$env:EMBEDDING_MODEL="woong0322/ko-legal-sbert-finetuned"
$env:ENABLE_SEARCH_IMPROVEMENTS="false"

python lawfirm_langgraph/tests/scripts/test_search_quality_evaluation.py \
    --disable-improvements \
    --query-type statute_article \
    --output logs/evaluation_ko_legal_sbert.json
```

### 3. 성능 비교

기존 모델과 Ko-Legal-SBERT 성능 비교:

```bash
# 기존 모델 평가
$env:EMBEDDING_MODEL="snunlp/KR-SBERT-V40K-klueNLI-augSTS"
python lawfirm_langgraph/tests/scripts/test_search_quality_evaluation.py \
    --disable-improvements \
    --query-type statute_article \
    --output logs/evaluation_baseline.json

# Ko-Legal-SBERT 평가
$env:EMBEDDING_MODEL="woong0322/ko-legal-sbert-finetuned"
python lawfirm_langgraph/tests/scripts/test_search_quality_evaluation.py \
    --disable-improvements \
    --query-type statute_article \
    --output logs/evaluation_ko_legal_sbert.json

# 비교 리포트 생성
python lawfirm_langgraph/tests/scripts/create_comparison_from_existing.py \
    --before-file logs/evaluation_baseline.json \
    --after-file logs/evaluation_ko_legal_sbert.json \
    --output-dir logs/model_comparison
```

## 예상 일정

### 소규모 테스트 (Phase 1)
- 모델 테스트: 30분
- 소규모 재임베딩: 1-2시간
- 검증: 30분
- **총 소요 시간**: 약 2-3시간

### 단계별 재임베딩 (Phase 2)
- statute_article: 1-2시간
- case_paragraph: 2-4시간
- decision_paragraph: 1-2시간
- interpretation_paragraph: 1-2시간
- **총 소요 시간**: 약 5-10시간 (GPU 사용 시)

### 전체 재임베딩 (Phase 3)
- **총 소요 시간**: 약 8-24시간 (GPU 사용 시)

## 리스크 관리

### 1. 데이터 손실 방지
- ✅ 데이터베이스 백업 필수
- ✅ 단계별 재임베딩으로 롤백 용이하게
- ✅ Dry-run 모드로 사전 테스트

### 2. 성능 이슈
- ✅ 배치 크기 조정 (메모리 부족 시)
- ✅ 단계별 커밋으로 진행 상황 저장
- ✅ 중단 시 재개 가능 (skip_if_exists 옵션)

### 3. 모델 호환성
- ✅ 모델 차원 확인
- ✅ 소규모 테스트로 사전 검증
- ✅ 기존 모델과 병행 사용 가능 (버전 관리)

## 모니터링

### 진행 상황 모니터링

```bash
# 재임베딩 진행 상황 확인
python scripts/monitoring/monitor_re_embedding_progress.py \
    --db data/lawfirm_v2.db \
    --model "woong0322/ko-legal-sbert-finetuned"
```

### 성능 모니터링

```bash
# 재임베딩 속도 모니터링
python scripts/monitoring/monitor_re_embedding_speed.py \
    --db data/lawfirm_v2.db
```

## 롤백 계획

문제 발생 시 롤백 방법:

### 1. 데이터베이스 복원

```bash
# 백업 파일로 복원
cp data/lawfirm_v2.db.backup_YYYYMMDD_HHMMSS data/lawfirm_v2.db
```

### 2. 특정 버전 삭제

```python
# 특정 모델의 임베딩만 삭제
import sqlite3
conn = sqlite3.connect('data/lawfirm_v2.db')
conn.execute("DELETE FROM embeddings WHERE model LIKE '%ko-legal-sbert%'")
conn.commit()
```

## 체크리스트

### 사전 준비
- [ ] HuggingFace에서 정확한 모델명 확인
- [ ] 모델 차원 확인
- [ ] 데이터베이스 백업
- [ ] 현재 임베딩 상태 확인

### Phase 1: 소규모 테스트
- [ ] 모델 로딩 테스트
- [ ] 10개 문서 재임베딩 테스트
- [ ] 결과 검증
- [ ] 검색 품질 테스트

### Phase 2: 단계별 재임베딩
- [ ] statute_article 재임베딩
- [ ] case_paragraph 재임베딩
- [ ] decision_paragraph 재임베딩
- [ ] interpretation_paragraph 재임베딩
- [ ] 각 단계별 검증

### Phase 3: 전체 재임베딩 (선택)
- [ ] 전체 재임베딩 실행
- [ ] 진행 상황 모니터링
- [ ] 완료 후 검증

### 완료 후
- [ ] FAISS 인덱스 재구축
- [ ] 검색 품질 테스트
- [ ] 성능 비교 리포트 생성
- [ ] 문서 업데이트

## 참고 자료

- [재임베딩 스크립트](../../scripts/migrations/re_embed_existing_data.py)
- [최적화 재임베딩 스크립트](../../scripts/migrations/re_embed_existing_data_optimized.py)
- [임베딩 버전 관리](../../scripts/utils/embedding_version_manager.py)
- [Sentence Transformers 문서](https://www.sbert.net/)
- [HuggingFace 모델 허브](https://huggingface.co/models)

