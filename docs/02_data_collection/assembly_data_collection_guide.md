# Assembly 데이터 수집 가이드

## 개요

국가법령정보센터 API 서비스 중단으로 인해 국회 법률정보시스템을 대안으로 사용하는 새로운 데이터 수집 시스템입니다. Playwright를 사용한 웹 스크래핑 방식으로 데이터를 수집합니다.

## 시스템 특징

- **웹 스크래핑**: Playwright를 사용한 브라우저 자동화
- **점진적 수집**: 중단 시 재개 가능한 체크포인트 시스템
- **시작 페이지 지정**: 특정 페이지부터 수집 시작 가능
- **페이지별 저장**: 각 페이지의 데이터를 별도 파일로 저장
- **카테고리별 수집**: 법률 분야별로 구분된 판례 수집
- **메모리 관리**: 대용량 데이터 처리 시 메모리 사용량 모니터링

## 설치 및 설정

### 1. 의존성 설치

```bash
pip install playwright
playwright install chromium
```

### 2. 환경 설정

```bash
# 환경변수 파일 복사
cp gradio/env_example.txt .env

# .env 파일 편집하여 실제 값 입력
nano .env
```

## 사용법

### 기본 법령 수집

```bash
# 기본 사용법
python scripts/data_collection/assembly/collect_laws.py --sample 100

# 시작 페이지 지정
python scripts/data_collection/assembly/collect_laws.py --sample 50 --start-page 5 --no-resume

# 특정 페이지 범위 수집
python scripts/data_collection/assembly/collect_laws.py --sample 180 --start-page 3 --no-resume

# 전체 수집
python scripts/data_collection/assembly/collect_laws.py --full
```

### 분야별 판례 수집 (2025.10.17 업데이트)

실제 국회 시스템의 카테고리 코드에 맞게 수정되었습니다:

#### 사용 가능한 카테고리

| 카테고리 | 한국어 | 코드 | 설명 |
|---------|--------|------|------|
| `civil` | 민사 | PREC00_001 | 민사 사건 |
| `criminal` | 형사 | PREC00_002 | 형사 사건 |
| `tax` | 조세 | PREC00_003 | 조세 사건 |
| `administrative` | 행정 | PREC00_004 | 행정 사건 |
| `family` | 가사 | PREC00_005 | 가사 사건 |
| `patent` | 특허 | PREC00_006 | 특허 사건 |
| `maritime` | 해사 | PREC00_009 | 해사 사건 |
| `military` | 군사 | PREC00_010 | 군사 사건 |

#### 기본 명령어

```bash
# 조세 사건 수집 (기존 family와 동일한 데이터)
python scripts/data_collection/assembly/collect_precedents_by_category.py --category tax --sample 300 --memory-limit 500 --batch-size 20 --start-page 251

# 실제 가사 사건 수집
python scripts/data_collection/assembly/collect_precedents_by_category.py --category family --sample 300 --memory-limit 500 --batch-size 20 --start-page 251

# 민사 사건 수집
python scripts/data_collection/assembly/collect_precedents_by_category.py --category civil --sample 300 --memory-limit 500 --batch-size 20 --start-page 251

# 형사 사건 수집
python scripts/data_collection/assembly/collect_precedents_by_category.py --category criminal --sample 300 --memory-limit 500 --batch-size 20 --start-page 251

# 행정 사건 수집
python scripts/data_collection/assembly/collect_precedents_by_category.py --category administrative --sample 300 --memory-limit 500 --batch-size 20 --start-page 251

# 특허 사건 수집
python scripts/data_collection/assembly/collect_precedents_by_category.py --category patent --sample 300 --memory-limit 500 --batch-size 20 --start-page 251
```

#### 모든 카테고리 동시 수집

```bash
# 모든 분야별로 동시 수집 (각 분야당 50건)
python scripts/data_collection/assembly/collect_precedents_by_category.py --all-categories --sample 50
```

#### 고급 옵션

```bash
# 메모리 제한 설정
python scripts/data_collection/assembly/collect_precedents_by_category.py --category civil --sample 100 --memory-limit 1000

# 배치 크기 조정
python scripts/data_collection/assembly/collect_precedents_by_category.py --category criminal --sample 200 --batch-size 10

# 재시도 횟수 설정
python scripts/data_collection/assembly/collect_precedents_by_category.py --category tax --sample 150 --max-retries 5

# 로그 레벨 설정
python scripts/data_collection/assembly/collect_precedents_by_category.py --category family --sample 100 --log-level DEBUG

# 체크포인트에서 재개하지 않기
python scripts/data_collection/assembly/collect_precedents_by_category.py --category civil --sample 100 --no-resume

# 특정 페이지부터 시작
python scripts/data_collection/assembly/collect_precedents_by_category.py --category criminal --sample 200 --start-page 50
```

## 데이터 마이그레이션 (2025.10.17)

기존 `family` 카테고리로 수집된 데이터가 실제로는 조세 사건이었으므로, 올바른 카테고리로 마이그레이션되었습니다:

### 마이그레이션 실행

```bash
# 마이그레이션 스크립트 실행 (완료됨)
python scripts/data_processing/migrate_family_to_tax.py
```

### 마이그레이션 결과

- **처리된 파일**: 472개
- **업데이트된 파일**: 472개
- **원본 백업**: `data/raw/assembly/precedent/20251017/family_backup_20251017_231702`
- **새 위치**: `data/raw/assembly/precedent/20251017/tax`

## 데이터 구조

### 수집된 파일 구조

```
data/raw/assembly/precedent/20251017/
├── tax/                                    # 조세 판례 (마이그레이션됨)
│   ├── collection_summary_*.json           # 수집 요약 정보
│   ├── precedent_tax_page_*.json          # 페이지별 판례 데이터
│   └── ...
├── family/                                 # 가사 판례 (실제 가사 사건)
├── civil/                                  # 민사 판례
├── criminal/                               # 형사 판례
├── administrative/                         # 행정 판례
└── patent/                                 # 특허 판례
```

### 판례 데이터 구조

```json
{
  "case_id": "사건번호",
  "case_name": "사건명",
  "court": "법원명",
  "decision_date": "판결일",
  "case_type": "사건유형",
  "category": "조세",
  "category_code": "PREC00_003",
  "content": {
    "summary": "사건 요약",
    "facts": "사실관계",
    "reasoning": "판결요지",
    "decision": "판결내용"
  },
  "metadata": {
    "collection_date": "2025-10-17T23:17:02.824089",
    "migration_note": "Migrated from family category"
  }
}
```

## 성능 최적화

### 메모리 관리

- **메모리 제한**: 기본 500MB, 필요시 조정 가능
- **배치 처리**: 페이지별로 배치 단위로 처리
- **즉시 저장**: 각 판례 처리 후 즉시 파일 저장
- **메모리 정리**: 주기적인 가비지 컬렉션 실행

### 수집 성능

- **페이지당 처리 시간**: 평균 30-60초
- **메모리 사용량**: 500MB 이하 유지
- **재시도 로직**: 네트워크 오류 시 자동 재시도
- **체크포인트**: 중단 시 마지막 페이지부터 재개

## 문제 해결

### 자주 발생하는 문제

1. **메모리 부족**
   ```bash
   # 메모리 제한 증가
   python scripts/data_collection/assembly/collect_precedents_by_category.py --category civil --sample 100 --memory-limit 1000
   ```

2. **네트워크 오류**
   ```bash
   # 재시도 횟수 증가
   python scripts/data_collection/assembly/collect_precedents_by_category.py --category criminal --sample 200 --max-retries 5
   ```

3. **브라우저 오류**
   ```bash
   # Playwright 재설치
   pip install --upgrade playwright
   playwright install chromium
   ```

### 로그 확인

```bash
# 실시간 로그 확인
tail -f logs/precedent_category_collection.log

# 특정 카테고리 로그 확인
grep "civil" logs/precedent_category_collection.log
```

## 모니터링

### 수집 진행률 확인

```bash
# 체크포인트 파일 확인
ls data/checkpoints/precedents_*/

# 수집된 파일 수 확인
find data/raw/assembly/precedent/20251017/tax -name "*.json" | wc -l
```

### 품질 검증

```bash
# 수집 요약 정보 확인
cat data/raw/assembly/precedent/20251017/tax/collection_summary_*.json | jq '.collection_info'
```

## 다음 단계

데이터 수집 완료 후:

1. **데이터 전처리**: 수집된 원본 데이터 구조화
2. **벡터 임베딩**: 판례 섹션별 벡터 임베딩 생성
3. **데이터베이스 저장**: SQLite 데이터베이스에 저장
4. **FAISS 인덱스**: 검색을 위한 벡터 인덱스 구축
5. **RAG 시스템**: 검색 증강 생성 시스템 통합

## 참고 자료

- [데이터 수집 가이드](../02_data_collection/data_collection_guide.md)
- [데이터 전처리 가이드](../03_data_processing/preprocessing_guide.md)
- [벡터 임베딩 가이드](../04_vector_embedding/embedding_guide.md)
- [RAG 시스템 가이드](../05_rag_system/rag_architecture.md)
