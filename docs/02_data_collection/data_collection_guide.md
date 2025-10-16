# LawFirmAI 데이터 수집 가이드

## 개요

이 문서는 LawFirmAI 프로젝트의 데이터 수집 시스템 사용법을 설명합니다. 데이터 수집과 벡터DB 구축이 분리되어 있어 더욱 유연하고 효율적인 데이터 처리가 가능합니다.

## 사전 준비

### 1. API 키 발급

국가법령정보센터 OpenAPI OC 파라미터를 발급받아야 합니다.

1. [국가법령정보센터 OpenAPI 가이드](https://open.law.go.kr/LSO/openApi/guideList.do) 방문
2. OC 파라미터 신청 및 발급 (사용자 이메일 ID)
3. 환경변수 설정

```bash
export LAW_OPEN_API_OC='your_email_id_here'
```

### 2. 환경 설정

```bash
# 환경변수 파일 복사
cp env.example .env

# .env 파일 편집하여 실제 값 입력
nano .env

# 또는 환경 변수 설정(윈도우)
$env:LAW_OPEN_API_OC={OC}
```

### 3. 필요한 디렉토리 생성

```bash
mkdir -p data/raw/{laws,precedents,constitutional_decisions,legal_interpretations,administrative_rules,local_ordinances}
mkdir -p data/processed/{laws,precedents,constitutional_decisions,legal_interpretations,administrative_rules,local_ordinances}
mkdir -p data/embeddings
mkdir -p logs
```

## Assembly 데이터 수집 시스템 (NEW)

국가법령정보센터 API 서비스 중단으로 인해 국회 법률정보시스템을 대안으로 사용하는 새로운 데이터 수집 시스템이 추가되었습니다.

### Assembly 시스템 특징

- **웹 스크래핑**: Playwright를 사용한 브라우저 자동화
- **점진적 수집**: 중단 시 재개 가능한 체크포인트 시스템
- **시작 페이지 지정**: 특정 페이지부터 수집 시작 가능
- **페이지별 저장**: 각 페이지의 데이터를 별도 파일로 저장

### Assembly 시스템 사용법

```bash
# 기본 사용법
python scripts/assembly/collect_laws.py --sample 100

# 시작 페이지 지정 (NEW)
python scripts/assembly/collect_laws.py --sample 50 --start-page 5 --no-resume

# 특정 페이지 범위 수집
python scripts/assembly/collect_laws.py --sample 180 --start-page 3 --no-resume

# 전체 수집
python scripts/assembly/collect_laws.py --full
```

### 자세한 사용법

Assembly 데이터 수집 시스템의 자세한 사용법은 [Assembly 데이터 수집 가이드](development/assembly_data_collection_guide.md)를 참조하세요.

## 기존 API 기반 데이터 수집 시스템

### 1. 통합 파이프라인 (권장)

#### 전체 파이프라인 실행
```bash
# 데이터 수집 + 벡터DB 구축 (전체)
python scripts/run_data_pipeline.py --mode full --oc your_email_id

# 데이터 수집만
python scripts/run_data_pipeline.py --mode collect --oc your_email_id

# 벡터DB 구축만
python scripts/run_data_pipeline.py --mode build
```

### 2. 판례 수집 (NEW - 개선된 기능)

#### 특정 연도 판례 수집
```bash
# 2025년 판례 수집 (무제한)
python scripts/precedent/collect_by_date.py --strategy yearly --year 2025 --unlimited

# 2024년 판례 수집 (무제한)
python scripts/precedent/collect_by_date.py --strategy yearly --year 2024 --unlimited

# 2023년 판례 수집 (무제한)
python scripts/precedent/collect_by_date.py --strategy yearly --year 2023 --unlimited
```

#### 기간별 판례 수집
```bash
# 연도별 수집 (최근 5년, 연간 2000건)
python scripts/precedent/collect_by_date.py --strategy yearly --target 10000

# 분기별 수집 (최근 2년, 분기당 500건)
python scripts/precedent/collect_by_date.py --strategy quarterly --target 4000

# 월별 수집 (최근 1년, 월간 200건)
python scripts/precedent/collect_by_date.py --strategy monthly --target 2400

# 주별 수집 (최근 3개월, 주간 100건)
python scripts/precedent/collect_by_date.py --strategy weekly --target 1200

# 모든 전략 순차 실행 (총 17,600건)
python scripts/precedent/collect_by_date.py --strategy all --target 20000
```

#### 고급 옵션
```bash
# 특정 연도 + 건수 제한
python scripts/precedent/collect_by_date.py --strategy yearly --year 2025 --target 5000

# 특정 연도 + 출력 디렉토리 지정
python scripts/precedent/collect_by_date.py --strategy yearly --year 2025 --unlimited --output data/custom/2025_precedents

# 특정 연도 + 드라이런 모드
python scripts/precedent/collect_by_date.py --strategy yearly --year 2025 --unlimited --dry-run

# 재시작 모드 (중단된 지점부터 재시작)
python scripts/precedent/collect_by_date.py --strategy yearly --target 5000 --resume
```

#### 주요 개선사항 (NEW)
- ✅ **페이지별 즉시 저장**: API 요청마다 즉시 파일 생성으로 데이터 손실 방지
- ✅ **판례일련번호 기준 파일명**: 파일명에 판례일련번호 범위 포함으로 추적 용이
- ✅ **실시간 진행상황**: 페이지별 상세 로그와 통계 정보 제공
- ✅ **오류 복구**: 중간 오류 발생 시에도 현재까지 수집된 데이터 자동 저장

#### 파일명 예시
```
page_001_민사-계약손해_20250001-20250050_50건_20250925_162705.json
page_002_형사_20250051-20250100_50건_20250925_162710.json
page_003_행정_20250101-20250150_50건_20250925_162715.json
```

#### 개별 데이터 타입별 수집
```bash
# 법령 데이터만 수집
python scripts/run_data_pipeline.py --mode laws --oc your_email_id --query "민법" --display 50

# 판례 데이터만 수집 (기존 방식)
python scripts/run_data_pipeline.py --mode precedents --oc your_email_id --query "계약 해지" --display 100

# 헌재결정례만 수집
python scripts/run_data_pipeline.py --mode constitutional --oc your_email_id --query "헌법" --display 50

# 법령해석례만 수집
python scripts/run_data_pipeline.py --mode interpretations --oc your_email_id --query "법령해석" --display 50

# 행정규칙만 수집
python scripts/run_data_pipeline.py --mode administrative --oc your_email_id --query "행정규칙" --display 50

# 자치법규만 수집
python scripts/run_data_pipeline.py --mode local --oc your_email_id --query "자치법규" --display 50
```

#### 여러 데이터 타입 동시 수집
```bash
# 법령, 판례, 헌재결정례 동시 수집
python scripts/run_data_pipeline.py --mode all_types --oc your_email_id --types laws precedents constitutional

# 특정 쿼리로 여러 타입 수집
python scripts/run_data_pipeline.py --mode all_types --oc your_email_id --types laws precedents --query "계약"
```

### 2. 개별 스크립트 사용

#### 데이터 수집 전용 스크립트
```bash
# 개별 타입 수집
python scripts/collect_data_only.py --mode laws --oc your_email_id --query "민법"
python scripts/collect_data_only.py --mode precedents --oc your_email_id --query "계약 해지"

# 여러 타입 동시 수집
python scripts/collect_data_only.py --mode multiple --oc your_email_id --types laws precedents constitutional

# 모든 타입 수집
python scripts/collect_data_only.py --mode collect --oc your_email_id
```

#### 벡터DB 구축 전용 스크립트
```bash
# 개별 타입 벡터DB 구축
python scripts/build_vector_db.py --mode laws
python scripts/build_vector_db.py --mode precedents

# 여러 타입 동시 벡터DB 구축
python scripts/build_vector_db.py --mode multiple --types laws precedents constitutional

# 모든 타입 벡터DB 구축
python scripts/build_vector_db.py --mode build
```

### 3. 레거시 스크립트 (호환성)

#### 개별 데이터 수집 (기존)
```bash
python scripts/collect_laws.py                    # 법령 수집
python scripts/collect_precedents.py              # 판례 수집
python scripts/collect_constitutional_decisions.py # 헌재결정례 수집
python scripts/collect_legal_interpretations.py   # 법령해석례 수집
python scripts/collect_administrative_rules.py    # 행정규칙 수집
python scripts/collect_local_ordinances.py        # 자치법규 수집
```

#### 통합 데이터 수집 (기존)
```bash
python scripts/collect_all_data.py                # 통합 데이터 수집
```

### 4. 데이터 품질 검증

수집된 데이터의 품질을 검증합니다:

```bash
python scripts/validate_data_quality.py
```

## 수집 목표

| 데이터 유형 | 목표 수량 | 설명 | 수집 스크립트 |
|------------|----------|------|-------------|
| 법령 | 1,000개 | 주요 법령 (민법, 상법, 형법 등) | `--mode laws` |
| 판례 | 5,000건 | 대법원, 고등법원 판례 | `--mode precedents` |
| 헌재결정례 | 1,000건 | 헌법재판소 결정례 | `--mode constitutional` |
| 법령해석례 | 2,000건 | 법령해석 관련 사례 | `--mode interpretations` |
| 행정규칙 | 1,000건 | 주요 부처별 행정규칙 | `--mode administrative` |
| 자치법규 | 500건 | 주요 지자체별 자치법규 | `--mode local` |
| 위원회결정문 | 500건 | 주요 위원회별 결정문 | `--mode committee` |
| 행정심판례 | 1,000건 | 최근 3년간 행정심판례 | `--mode administrative_appeals` |
| 조약 | 100건 | 주요 조약 | `--mode treaties` |

## 파이프라인 모드별 사용법

### 1. 전체 파이프라인 (`--mode full`)
```bash
# 모든 데이터 타입 수집 + 벡터DB 구축
python scripts/run_data_pipeline.py --mode full --oc your_email_id
```

### 2. 데이터 수집만 (`--mode collect`)
```bash
# 모든 데이터 타입 수집 (JSON 저장)
python scripts/run_data_pipeline.py --mode collect --oc your_email_id
```

### 3. 벡터DB 구축만 (`--mode build`)
```bash
# 수집된 JSON 파일을 기반으로 벡터DB 구축
python scripts/run_data_pipeline.py --mode build
```

### 4. 개별 데이터 타입 수집
```bash
# 특정 데이터 타입만 수집
python scripts/run_data_pipeline.py --mode laws --oc your_email_id --query "민법" --display 100
python scripts/run_data_pipeline.py --mode precedents --oc your_email_id --query "계약 해지" --display 200
```

### 5. 여러 데이터 타입 동시 수집 (`--mode all_types`)
```bash
# 지정된 여러 타입 동시 수집
python scripts/run_data_pipeline.py --mode all_types --oc your_email_id --types laws precedents constitutional

# 특정 쿼리로 여러 타입 수집
python scripts/run_data_pipeline.py --mode all_types --oc your_email_id --types laws precedents --query "계약"
```

## 데이터 구조

### 법령 데이터
```json
{
  "law_id": "법률 제12345호",
  "law_name": "민법",
  "effective_date": "2020.01.01",
  "promulgation_date": "2019.12.31",
  "law_type": "법률",
  "content": "법령 본문...",
  "category": "basic",
  "status": "success"
}
```

### 판례 데이터
```json
{
  "precedent_id": "2020다12345",
  "case_name": "사건명",
  "court": "대법원",
  "decision_date": "2020.01.01",
  "case_type": "민사",
  "content": "판시사항...",
  "reasoning": "판결요지...",
  "category": "precedent",
  "status": "success"
}
```

### 법령용어 데이터
```json
{
  "term_id": "TERM001",
  "term_name": "계약",
  "definition": "당사자 간의 의사표시의 합치에 의하여 성립하는 법률행위",
  "related_law": "민법",
  "example": "매매계약, 임대차계약 등",
  "category": "legal_term",
  "status": "success"
}
```

## 품질 관리

### 검증 기준

1. **완성도**: 95% 이상 (필수 필드 누락 없음)
2. **정확도**: 98% 이상 (API 응답과 일치)
3. **일관성**: 90% 이상 (데이터 형식 통일)
4. **신선도**: 7일 이내 업데이트

### 검증 항목

- 사건번호 형식 검증
- 판결일 유효성 검증
- 법원명 표준화 검증
- 텍스트 인코딩 검증
- 메타데이터 완성도 검증

## 에러 처리

### 일반적인 에러

1. **API 키 오류**
   ```
   ValueError: LAW_OPEN_API_KEY 환경변수가 설정되지 않았습니다.
   ```
   - 해결: 환경변수 설정 확인

2. **API 요청 제한**
   ```
   Warning: 일일 API 요청 한도에 도달했습니다.
   ```
   - 해결: 다음 날 재시도 또는 API 키 추가 발급

3. **네트워크 오류**
   ```
   RequestException: API 요청 실패
   ```
   - 해결: 네트워크 연결 확인, 재시도 로직 동작

### 로그 확인

```bash
# 실시간 로그 확인
tail -f logs/integrated_data_collection.log

# 특정 데이터 유형 로그 확인
tail -f logs/law_collection.log
tail -f logs/precedent_collection.log
```

## 성능 최적화

### 메모리 사용량 최적화

- 배치 크기 조정: `DATA_COLLECTION_BATCH_SIZE=50`
- 스트리밍 처리로 대용량 파일 처리
- 불필요한 변수 즉시 삭제

### API 요청 최적화

- 요청 간격 조정: 1초 간격 유지
- 재시도 로직: 최대 3회, 지수 백오프
- 타임아웃 설정: 30초

## 모니터링

### 수집 진행률 확인

```bash
# 실시간 진행률 확인
python -c "
import json
with open('data/collection_metadata.json', 'r') as f:
    metadata = json.load(f)
    print(f'API 요청 수: {metadata[\"api_usage\"][\"total_requests\"]}')
    print(f'남은 요청 수: {metadata[\"api_usage\"][\"remaining_requests\"]}')
"
```

### 품질 지표 확인

```bash
# 품질 보고서 확인
cat docs/data_quality_report.md
```

## 문제 해결

### 자주 발생하는 문제

1. **메모리 부족**
   - 배치 크기 줄이기
   - 시스템 메모리 확인

2. **API 응답 지연**
   - 타임아웃 값 증가
   - 네트워크 상태 확인

3. **데이터 형식 오류**
   - API 응답 구조 확인
   - 파싱 로직 점검

### 지원

문제가 지속되면 다음을 확인하세요:

1. 로그 파일 검토
2. 환경변수 설정 확인
3. API 키 유효성 확인
4. 네트워크 연결 상태 확인

## 다음 단계

데이터 수집 완료 후:

1. 벡터 임베딩 생성
2. FAISS 인덱스 구축
3. 모델 파인튜닝
4. RAG 시스템 구현
