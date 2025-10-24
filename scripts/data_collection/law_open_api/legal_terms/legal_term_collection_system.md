# 법률 용어 수집 시스템

법령정보개방포털 API를 통해 법률 용어를 수집하고, 데이터베이스에 저장하며, 벡터스토어를 업데이트하는 통합 시스템입니다.

## 주요 기능

- **데이터 수집**: 법령정보개방포털 API를 통한 법률 용어 목록 및 상세 정보 수집
- **데이터베이스 관리**: SQLite를 사용한 체계적인 데이터 저장 및 관리
- **벡터스토어**: FAISS를 사용한 법률 용어 임베딩 및 유사도 검색
- **재시도 로직**: 타임아웃 발생 시 3분, 5분, 10분 대기 후 재시도
- **중단 후 재개**: 특정 페이지부터 수집 재개 가능
- **배치 처리**: 효율적인 대용량 데이터 처리
- **진행 상황 추적**: 실시간 수집 상태 모니터링

## 시스템 구조

```
LawFirmAI/
├── source/
│   ├── services/
│   │   ├── legal_term_collector.py          # 법률 용어 수집기
│   │   ├── legal_term_vector_store.py       # 벡터스토어 관리
│   │   └── legal_term_collection_manager.py # 통합 관리자
│   └── config/
│       └── legal_term_collection_config.py  # 설정 관리
├── scripts/
│   └── legal_term_collector.py             # 실행 스크립트
├── config/
│   └── legal_term_collection_config.yaml   # 설정 파일
└── data/
    ├── legal_terms.db                      # SQLite 데이터베이스
    ├── raw/legal_terms/                    # 원본 데이터
    ├── processed/legal_terms/              # 전처리된 데이터
    └── embeddings/legal_terms/             # 벡터 임베딩
```

## 설치 및 설정

### 1. 의존성 설치

```bash
pip install aiohttp sqlite3 numpy faiss-cpu transformers torch
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
LEGAL_API_OC_ID=your_email_id
SENTENCE_BERT_MODEL=jhgan/ko-sroberta-multitask
DATABASE_PATH=data/legal_terms.db
LOG_LEVEL=INFO
```

### 3. 설정 파일 수정

`config/legal_term_collection_config.yaml` 파일에서 필요한 설정을 수정합니다.

## 사용 방법

### 명령행 인터페이스

#### 전체 수집 실행
```bash
python scripts/legal_term_collector.py --full-collection
```

#### 특정 페이지 범위로 수집
```bash
python scripts/legal_term_collector.py --start-page 1 --end-page 10
```

#### 특정 용어 검색
```bash
python scripts/legal_term_collector.py --query "계약"
```

#### 가나다 검색
```bash
python scripts/legal_term_collector.py --gana "가"
```

#### 수집 상태 조회
```bash
python scripts/legal_term_collector.py --status
```

#### 수집 재개
```bash
python scripts/legal_term_collector.py --resume --page 5
```

#### 대화형 모드
```bash
python scripts/legal_term_collector.py --interactive
```

### 프로그래밍 인터페이스

```python
import asyncio
from source.services.legal_term_collection_manager import LegalTermCollectionManager, CollectionConfig
from source.config.legal_term_collection_config import get_config

async def main():
    config = get_config()
    manager = LegalTermCollectionManager(config)
    
    # 전체 수집 실행
    collection_config = CollectionConfig(
        start_page=1,
        end_page=10,
        query="계약",
        list_batch_size=10,
        detail_batch_size=50,
        vector_batch_size=100
    )
    
    success = await manager.collect_legal_terms(collection_config)
    
    if success:
        print("수집 완료")
    else:
        print("수집 실패")

asyncio.run(main())
```

## API 사용법

### 법령용어 목록 조회 API

**요청 URL**: `http://www.law.go.kr/DRF/lawSearch.do?target=lstrm`

**주요 파라미터**:
- `OC`: 사용자 이메일 ID (필수)
- `target`: lstrm (필수)
- `type`: JSON/XML/HTML (필수)
- `display`: 검색 결과 개수 (기본값: 20, 최대: 100)
- `page`: 페이지 번호 (기본값: 1)
- `query`: 검색 쿼리
- `gana`: 가나다 검색

**응답 필드**:
- `totalCnt`: 총 검색 건수
- `page`: 현재 페이지
- `법령용어ID`: 법령용어 ID
- `법령용어명`: 법령용어명
- `법령용어정의`: 법령용어 정의
- `출처`: 출처 정보

### 법령용어 상세 조회 API

**요청 URL**: `http://www.law.go.kr/DRF/lawService.do?target=lstrm`

**주요 파라미터**:
- `OC`: 사용자 이메일 ID (필수)
- `target`: lstrm (필수)
- `type`: JSON/XML/HTML (필수)
- `query`: 상세 조회할 법령용어명

## 데이터베이스 스키마

### legal_term_list 테이블
- `id`: 기본키
- `법령용어ID`: 법령용어 고유 ID
- `법령용어명`: 법령용어명
- `법령용어상세검색`: 상세 검색 정보
- `사전구분코드`: 사전 구분 코드
- `법령용어상세링크`: 상세 링크
- `법령종류코드`: 법령 종류 코드
- `lstrm_id`: 결과 번호
- `collected_at`: 수집 시간
- `processed`: 상세 정보 처리 여부
- `vectorized`: 벡터화 여부

### legal_term_details 테이블
- `id`: 기본키
- `법령용어일련번호`: 법령용어 일련번호
- `법령용어명_한글`: 한글 법령용어명
- `법령용어명_한자`: 한자 법령용어명
- `법령용어코드`: 법령용어 코드
- `법령용어코드명`: 법령용어 코드명
- `출처`: 출처 정보
- `법령용어정의`: 법령용어 정의
- `collected_at`: 수집 시간
- `processed`: 처리 여부

## 재시도 로직

시스템은 다음과 같은 재시도 로직을 구현합니다:

1. **타임아웃 발생 시**: 3분 대기 후 재시도
2. **두 번째 실패 시**: 5분 대기 후 재시도
3. **세 번째 실패 시**: 10분 대기 후 재시도
4. **최대 재시도 횟수 초과 시**: 해당 요청 건너뛰기

## 중단 후 재개

수집이 중단된 경우 다음과 같이 재개할 수 있습니다:

1. **자동 재개**: `auto_resume: true` 설정 시 자동으로 마지막 페이지부터 재개
2. **수동 재개**: `--resume --page N` 옵션으로 특정 페이지부터 재개
3. **상태 확인**: `--status` 옵션으로 현재 진행 상황 확인

## 배치 처리

시스템은 효율적인 배치 처리를 지원합니다:

- **목록 수집**: 기본 10개 페이지씩 배치 처리
- **상세 수집**: 기본 50개 용어씩 배치 처리
- **벡터화**: 기본 100개 용어씩 배치 처리

## 모니터링

실시간 모니터링 기능을 제공합니다:

- **진행 상황**: 현재 페이지, 총 페이지, 수집된 용어 수
- **성능 메트릭**: 처리 속도, 메모리 사용량, 오류율
- **로그**: 상세한 실행 로그 및 오류 추적

## 성능 최적화

다음과 같은 성능 최적화를 적용했습니다:

- **비동기 처리**: aiohttp를 사용한 비동기 API 요청
- **메모리 관리**: 배치 단위 처리로 메모리 사용량 최적화
- **캐싱**: 자주 사용되는 데이터 캐싱
- **압축**: 데이터 압축으로 저장 공간 절약

## 문제 해결

### 일반적인 문제

1. **API 요청 실패**
   - 네트워크 연결 확인
   - API 키 유효성 확인
   - 요청 제한 확인

2. **메모리 부족**
   - 배치 크기 줄이기
   - 메모리 제한 설정 조정

3. **데이터베이스 오류**
   - 디스크 공간 확인
   - 데이터베이스 파일 권한 확인

### 로그 확인

```bash
tail -f logs/legal_term_collection.log
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다.

## 연락처

프로젝트 관련 문의사항이 있으시면 이슈를 생성해 주세요.
