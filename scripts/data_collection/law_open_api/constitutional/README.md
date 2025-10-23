# 헌재결정례 데이터 수집 및 활용 시스템

국가법령정보센터 OPEN API를 활용하여 헌재결정례 데이터를 선고일자 오름차순으로 100개 단위 배치로 수집하고, 데이터베이스와 벡터 저장소에 저장하는 시스템입니다.

## 📋 주요 기능

- **선고일자 오름차순 수집**: `dasc` 정렬 옵션으로 가장 오래된 결정례부터 수집
- **100개 단위 배치 처리**: 메모리 효율성을 위한 배치 단위 처리
- **상세 정보 포함 수집**: 목록 정보와 본문 정보를 모두 수집
- **데이터베이스 저장**: SQLite 데이터베이스에 구조화된 저장
- **벡터 검색 지원**: FAISS 기반 벡터 임베딩 및 유사도 검색
- **체크포인트 관리**: 중단 시 재개 가능한 체크포인트 시스템
- **FTS 검색**: 전문 검색을 위한 FTS5 테이블 지원

## 🏗️ 시스템 구조

```
scripts/data_collection/constitutional/
├── constitutional_decision_collector.py    # 헌재결정례 수집기
├── constitutional_checkpoint_manager.py    # 체크포인트 관리자
├── collect_constitutional_decisions.py    # 수집 실행 스크립트
├── test_constitutional_system.py          # 통합 테스트 스크립트
├── collect_constitutional_decisions.bat   # Windows 배치 스크립트
├── collect_constitutional_decisions.ps1   # PowerShell 스크립트
└── README.md                              # 이 파일
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 환경변수 설정
export LAW_OPEN_API_OC='your_email@example.com'

# 또는 Windows에서
set LAW_OPEN_API_OC=your_email@example.com

# 또는 PowerShell에서
$env:LAW_OPEN_API_OC='your_email@example.com'
```

### 2. 기본 수집 실행

```bash
# Python 스크립트로 실행
python scripts/data_collection/constitutional/collect_constitutional_decisions.py \
    --keyword "" \
    --max-count 1000 \
    --batch-size 100 \
    --sort-order dasc

# Windows 배치 파일로 실행
scripts/data_collection/constitutional/collect_constitutional_decisions.bat

# PowerShell 스크립트로 실행
scripts/data_collection/constitutional/collect_constitutional_decisions.ps1
```

### 3. 시스템 테스트

```bash
# 통합 테스트 실행
python scripts/data_collection/constitutional/test_constitutional_system.py
```

## 📊 API 사용법

### 헌재결정례 목록 조회

```python
from source.data.law_open_api_client import LawOpenAPIClient

client = LawOpenAPIClient()

# 헌재결정례 목록 조회 (선고일자 오름차순)
response = client.search_constitutional_decisions(
    query="헌법",
    display=100,
    page=1,
    sort="dasc"  # 선고일자 오름차순
)

# 응답 구조
# {
#   "DetcSearch": {
#     "totalCnt": 1000,
#     "detc": [
#       {
#         "헌재결정례일련번호": 12345,
#         "사건명": "사건명",
#         "사건번호": "2024헌마123",
#         "종국일자": "20241201",
#         "사건종류명": "헌법소원"
#       }
#     ]
#   }
# }
```

### 헌재결정례 상세 조회

```python
# 헌재결정례 상세 정보 조회
detail = client.get_constitutional_decision_detail(
    decision_id="12345"
)

# 응답 구조
# {
#   "헌재결정례일련번호": 12345,
#   "사건명": "사건명",
#   "판시사항": "판시사항",
#   "결정요지": "결정요지",
#   "전문": "전문 내용",
#   "참조조문": "참조조문",
#   "참조판례": "참조판례",
#   "심판대상조문": "심판대상조문"
# }
```

## 🗄️ 데이터베이스 사용법

### 헌재결정례 데이터 삽입

```python
from source.data.database import DatabaseManager

db_manager = DatabaseManager()

# 단일 결정례 삽입
decision_data = {
    '헌재결정례일련번호': 12345,
    '사건명': '사건명',
    '판시사항': '판시사항',
    '결정요지': '결정요지',
    '전문': '전문 내용'
}

success = db_manager.insert_constitutional_decision(decision_data)

# 배치 삽입
decisions = [decision_data1, decision_data2, ...]
inserted_count = db_manager.insert_constitutional_decisions_batch(decisions)
```

### 헌재결정례 검색

```python
# FTS 검색
results = db_manager.search_constitutional_decisions_fts(
    query="표현의 자유",
    limit=10
)

# 키워드 검색
results = db_manager.get_constitutional_decisions_by_keyword(
    keyword="평등권",
    limit=10
)

# 날짜 범위 검색
results = db_manager.get_constitutional_decisions_by_date_range(
    start_date="2024-01-01",
    end_date="2024-12-31",
    limit=100
)
```

## 🔍 벡터 검색 사용법

### 헌재결정례 벡터 검색

```python
from source.data.vector_store import LegalVectorStore

vector_store = LegalVectorStore()

# 벡터 검색
results = vector_store.search_constitutional_decisions(
    query="표현의 자유",
    top_k=10,
    filter_by_date="2024",  # 선택사항
    filter_by_type="헌법소원"  # 선택사항
)

# 유사 결정례 검색
similar_results = vector_store.get_constitutional_decisions_by_similarity(
    decision_id=12345,
    top_k=5
)

# 통계 조회
stats = vector_store.get_constitutional_decisions_stats()
```

## 💾 체크포인트 관리

### 체크포인트 생성 및 관리

```python
from scripts.data_collection.constitutional.constitutional_checkpoint_manager import ConstitutionalCheckpointManager

manager = ConstitutionalCheckpointManager()

# 체크포인트 생성
checkpoint_id = manager.create_checkpoint(
    collection_type="keyword",
    keyword="헌법",
    sort_order="dasc"
)

# 체크포인트 업데이트
manager.update_checkpoint(
    checkpoint_id,
    current_page=10,
    collected_count=500
)

# 체크포인트 완료
manager.complete_checkpoint(checkpoint_id)

# 체크포인트 목록 조회
checkpoints = manager.list_checkpoints(status="in_progress")
```

### 체크포인트 관리 도구

```bash
# 체크포인트 목록 조회
python scripts/data_collection/constitutional/constitutional_checkpoint_manager.py --list

# 최신 체크포인트 조회
python scripts/data_collection/constitutional/constitutional_checkpoint_manager.py --latest keyword

# 체크포인트 삭제
python scripts/data_collection/constitutional/constitutional_checkpoint_manager.py --delete checkpoint_id

# 오래된 체크포인트 정리
python scripts/data_collection/constitutional/constitutional_checkpoint_manager.py --cleanup 7
```

## 📈 수집 옵션

### 정렬 옵션

- `dasc`: 선고일자 오름차순 (기본값)
- `ddes`: 선고일자 내림차순
- `lasc`: 사건명 오름차순
- `ldes`: 사건명 내림차순
- `nasc`: 사건번호 오름차순
- `ndes`: 사건번호 내림차순
- `efasc`: 종국일자 오름차순
- `efdes`: 종국일자 내림차순

### 수집 스크립트 옵션

```bash
python scripts/data_collection/constitutional/collect_constitutional_decisions.py \
    --keyword "헌법" \                    # 검색 키워드
    --max-count 1000 \                    # 최대 수집 개수
    --batch-size 100 \                    # 배치 크기
    --sort-order dasc \                   # 정렬 순서
    --no-details \                        # 상세 정보 제외
    --no-database \                        # 데이터베이스 업데이트 제외
    --no-vectors \                         # 벡터 저장소 업데이트 제외
    --test \                               # API 연결 테스트만 실행
    --sample 50                            # 샘플 수집 (50개)
```

## 📊 데이터 구조

### 헌재결정례 테이블 스키마

```sql
CREATE TABLE constitutional_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER UNIQUE NOT NULL,           -- 헌재결정례일련번호
    decision_name TEXT NOT NULL,                   -- 사건명
    case_number TEXT,                              -- 사건번호
    case_type TEXT,                               -- 사건종류명
    case_type_code INTEGER,                       -- 사건종류코드
    court_division_code INTEGER,                  -- 재판부구분코드
    decision_date TEXT,                           -- 종국일자
    final_date TEXT,                              -- 종국일자
    summary TEXT,                                 -- 판시사항
    decision_gist TEXT,                           -- 결정요지
    full_text TEXT,                               -- 전문
    reference_articles TEXT,                      -- 참조조문
    reference_precedents TEXT,                    -- 참조판례
    target_articles TEXT,                         -- 심판대상조문
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### FTS 테이블 스키마

```sql
CREATE VIRTUAL TABLE constitutional_decisions_fts USING fts5(
    decision_name,
    summary,
    decision_gist,
    full_text,
    content='constitutional_decisions',
    content_rowid='id'
);
```

## 🔧 문제 해결

### 일반적인 문제

1. **API 연결 실패**
   - `LAW_OPEN_API_OC` 환경변수 확인
   - 네트워크 연결 상태 확인
   - API 서버 상태 확인

2. **데이터베이스 오류**
   - 데이터베이스 파일 권한 확인
   - 디스크 공간 확인
   - SQLite 버전 호환성 확인

3. **벡터 저장소 오류**
   - FAISS 설치 확인
   - 메모리 부족 확인
   - 임베딩 모델 다운로드 확인

### 로그 확인

```bash
# 로그 파일 위치
logs/constitutional_collection_YYYYMMDD_HHMMSS.log

# 실시간 로그 모니터링
tail -f logs/constitutional_collection_*.log
```

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해주세요.
