# lstrmAI API 데이터 수집 계획서

## 📋 목차

1. [API 개요](#api-개요)
2. [데이터베이스 스키마 설계](#데이터베이스-스키마-설계)
3. [파일 구조](#파일-구조)
4. [구현 계획](#구현-계획)
5. [수집 전략](#수집-전략)
6. [주요 특징](#주요-특징)
7. [실행 예시](#실행-예시)
8. [데이터 확인](#데이터-확인)
9. [주의사항](#주의사항)

---

## API 개요

### API 정보

**요청 URL**: `https://www.law.go.kr/DRF/lawSearch.do?target=lstrmAI`  
**Method**: GET  
**출력 형식**: JSON  
**용도**: 법령용어 검색 (법령정보지식베이스)

### 요청 파라미터

| 파라미터 | 타입 | 필수 | 설명 | 기본값/허용값 |
|---------|------|------|------|--------------|
| OC | string | 필수 | 사용자 이메일 ID (g4c@korea.kr일 경우 OC=g4c) | - |
| target | string | 필수 | 서비스 대상 | lstrmAI |
| type | char | 필수 | 출력 형태 | JSON |
| query | string | 선택 | 검색 질의 | - |
| display | int | 선택 | 검색된 결과 개수 | 20 (max=100) |
| page | int | 선택 | 검색 결과 페이지 | 1 |
| homonymYn | char | 선택 | 동음이의어 존재여부 | Y/N |

### 응답 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| target | string | 검색서비스 대상 |
| 키워드 | string | 검색 단어 |
| 검색결과개수 | int | 검색 건수 |
| section | string | 검색범위 |
| page | int | 현재 페이지번호 |
| numOfRows | int | 페이지 당 출력 결과 수 |
| 법령용어 id | string | 법령용어 순번 |
| 법령용어명 | string | 법령용어명 |
| 동음이의어존재여부 | string | 동음이의어 존재여부 |
| 비고 | string | 동음이의어 내용 |
| 용어간관계링크 | string | 법령용어-일상용어 연계 정보 상세링크 |
| 조문간관계링크 | string | 법령용어-조문 연계 정보 상세링크 |

### 샘플 URL

```bash
# JSON 형식으로 검색
https://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lstrmAI&type=JSON&query=계약

# 페이징 처리
https://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lstrmAI&type=JSON&query=계약&page=1&display=100
```

---

## 데이터베이스 스키마 설계

원본 JSON을 그대로 저장할 테이블 설계:

```sql
CREATE TABLE IF NOT EXISTS open_law_lstrm_ai_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- 검색 메타데이터
    search_keyword TEXT,
    search_page INTEGER,
    search_display INTEGER,
    homonym_yn TEXT,
    
    -- API 응답 원본 데이터 (JSON)
    raw_response_json TEXT NOT NULL,  -- 전체 응답 JSON 원본 저장
    
    -- 개별 결과 항목 (배열의 각 항목)
    term_id TEXT,                    -- 법령용어 id
    term_name TEXT,                  -- 법령용어명
    homonym_exists TEXT,             -- 동음이의어존재여부
    homonym_note TEXT,               -- 비고
    term_relation_link TEXT,         -- 용어간관계링크
    article_relation_link TEXT,      -- 조문간관계링크
    
    -- 수집 메타데이터
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    collection_method TEXT,          -- 'keyword', 'pagination', 'all'
    api_request_url TEXT,            -- 실제 요청 URL
    
    -- 통계 정보
    total_count INTEGER,              -- 검색결과개수
    page_number INTEGER,              -- page
    num_of_rows INTEGER,             -- numOfRows
    
    -- 인덱스
    UNIQUE(term_id, search_keyword, search_page)
);

CREATE INDEX IF NOT EXISTS idx_lstrm_ai_term_id ON open_law_lstrm_ai_data(term_id);
CREATE INDEX IF NOT EXISTS idx_lstrm_ai_keyword ON open_law_lstrm_ai_data(search_keyword);
CREATE INDEX IF NOT EXISTS idx_lstrm_ai_collected_at ON open_law_lstrm_ai_data(collected_at);
```

### 테이블 설계 특징

1. **원본 JSON 보존**: `raw_response_json` 필드에 전체 API 응답을 JSON 문자열로 저장
2. **개별 항목 저장**: 각 검색 결과 항목을 개별 레코드로 저장하여 검색 및 분석 용이
3. **중복 방지**: `term_id + search_keyword + search_page` 조합으로 UNIQUE 제약
4. **메타데이터 보존**: 수집 시점, 방법, 요청 URL 등 추적 가능

---

## 파일 구조

```
scripts/ingest/
├── ingest_lstrm_ai.py          # 메인 수집 스크립트
└── ...

lawfirm_langgraph/core/data/
└── connection_pool.py          # 연결 풀 (기존)
```

---

## 구현 계획

### 4.1 API 클라이언트 클래스

```python
class LstrmAIClient:
    """lstrmAI API 클라이언트"""
    
    def __init__(self, oc: str, base_url: str = "https://www.law.go.kr/DRF"):
        """
        Args:
            oc: 사용자 이메일 ID
            base_url: API 기본 URL
        """
        self.oc = oc
        self.base_url = base_url
        self.rate_limit_delay = 0.5  # 요청 간 지연 (초)
    
    def search_terms(
        self,
        query: str = "",
        page: int = 1,
        display: int = 100,
        homonym_yn: str = None
    ) -> Dict[str, Any]:
        """법령용어 검색"""
        params = {
            'OC': self.oc,
            'target': 'lstrmAI',
            'type': 'JSON',
            'query': query,
            'page': page,
            'display': display
        }
        if homonym_yn:
            params['homonymYn'] = homonym_yn
        
        return self._make_request(params)
    
    def _make_request(self, params: Dict) -> Dict[str, Any]:
        """API 요청 실행 (재시도 로직 포함)"""
        url = f"{self.base_url}/lawSearch.do"
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                time.sleep(self.rate_limit_delay)
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # 지수 백오프
                    continue
                raise
```

### 4.2 데이터 수집기

```python
class LstrmAICollector:
    """lstrmAI 데이터 수집기"""
    
    def __init__(self, client: LstrmAIClient, db_path: str):
        """
        Args:
            client: LstrmAIClient 인스턴스
            db_path: 데이터베이스 파일 경로
        """
        self.client = client
        self.db_path = db_path
        from lawfirm_langgraph.core.data.connection_pool import get_connection_pool
        self.connection_pool = get_connection_pool(db_path)
    
    def collect_by_keywords(
        self,
        keywords: List[str],
        max_pages_per_keyword: int = None
    ) -> int:
        """키워드 기반 수집"""
        total_saved = 0
        for keyword in keywords:
            logger.info(f"키워드 '{keyword}' 수집 시작")
            saved = self.collect_all_pages(
                query=keyword,
                max_pages=max_pages_per_keyword
            )
            total_saved += saved
            logger.info(f"키워드 '{keyword}' 수집 완료: {saved}건")
        return total_saved
    
    def collect_all_pages(
        self,
        query: str = "",
        max_pages: int = None
    ) -> int:
        """전체 페이지 수집"""
        page = 1
        total_saved = 0
        
        while True:
            if max_pages and page > max_pages:
                break
            
            try:
                response = self.client.search_terms(
                    query=query,
                    page=page,
                    display=100
                )
                
                # 응답 검증
                if not response or '검색결과개수' not in response:
                    break
                
                total_count = response.get('검색결과개수', 0)
                if total_count == 0:
                    break
                
                # 데이터 저장
                saved = self._save_response(
                    response=response,
                    search_keyword=query,
                    page=page,
                    display=100
                )
                total_saved += saved
                
                logger.info(f"페이지 {page} 수집 완료: {saved}건 저장")
                
                # 다음 페이지 확인
                num_of_rows = response.get('numOfRows', 0)
                if num_of_rows == 0 or saved == 0:
                    break
                
                page += 1
                
            except Exception as e:
                logger.error(f"페이지 {page} 수집 실패: {e}")
                break
        
        return total_saved
    
    def _save_response(
        self,
        response: Dict[str, Any],
        search_keyword: str,
        page: int,
        display: int,
        homonym_yn: str = None
    ) -> int:
        """응답 데이터를 DB에 저장 (원본 JSON 포함)"""
        conn = self.connection_pool.get_connection()
        try:
            cursor = conn.cursor()
            
            # 전체 응답을 JSON 문자열로 저장
            raw_json = json.dumps(response, ensure_ascii=False, indent=None)
            
            # 각 결과 항목을 개별 레코드로 저장
            items = response.get('items', []) or []
            if not items:
                # items가 없을 경우 다른 필드명 확인
                items = response.get('법령용어', []) or []
            
            saved_count = 0
            
            for item in items:
                # 요청 URL 생성
                request_url = self._build_request_url(
                    search_keyword, page, display, homonym_yn
                )
                
                cursor.execute("""
                    INSERT OR IGNORE INTO open_law_lstrm_ai_data (
                        search_keyword, search_page, search_display, homonym_yn,
                        raw_response_json,
                        term_id, term_name, homonym_exists, homonym_note,
                        term_relation_link, article_relation_link,
                        collection_method, api_request_url,
                        total_count, page_number, num_of_rows,
                        collected_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    search_keyword, page, display, homonym_yn,
                    raw_json,  # 원본 JSON 저장
                    item.get('법령용어 id') or item.get('법령용어id'),
                    item.get('법령용어명'),
                    item.get('동음이의어존재여부'),
                    item.get('비고'),
                    item.get('용어간관계링크'),
                    item.get('조문간관계링크'),
                    'keyword' if search_keyword else 'all',
                    request_url,
                    response.get('검색결과개수'),
                    response.get('page'),
                    response.get('numOfRows'),
                    datetime.now().isoformat()
                ))
                if cursor.rowcount > 0:
                    saved_count += 1
            
            conn.commit()
            return saved_count
        except Exception as e:
            conn.rollback()
            logger.error(f"데이터 저장 실패: {e}")
            raise
        finally:
            # 연결 풀 사용 시 close() 불필요
            pass
    
    def _build_request_url(
        self,
        query: str,
        page: int,
        display: int,
        homonym_yn: str = None
    ) -> str:
        """요청 URL 생성"""
        params = {
            'OC': self.client.oc,
            'target': 'lstrmAI',
            'type': 'JSON',
            'query': query,
            'page': page,
            'display': display
        }
        if homonym_yn:
            params['homonymYn'] = homonym_yn
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items() if v])
        return f"{self.client.base_url}/lawSearch.do?{query_string}"
```

### 4.3 메인 수집 스크립트

```python
# scripts/ingest/ingest_lstrm_ai.py

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lawfirm_langgraph.core.data.connection_pool import get_connection_pool

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _create_table(conn):
    """테이블 생성"""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS open_law_lstrm_ai_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            
            -- 검색 메타데이터
            search_keyword TEXT,
            search_page INTEGER,
            search_display INTEGER,
            homonym_yn TEXT,
            
            -- API 응답 원본 데이터 (JSON)
            raw_response_json TEXT NOT NULL,
            
            -- 개별 결과 항목
            term_id TEXT,
            term_name TEXT,
            homonym_exists TEXT,
            homonym_note TEXT,
            term_relation_link TEXT,
            article_relation_link TEXT,
            
            -- 수집 메타데이터
            collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            collection_method TEXT,
            api_request_url TEXT,
            
            -- 통계 정보
            total_count INTEGER,
            page_number INTEGER,
            num_of_rows INTEGER,
            
            UNIQUE(term_id, search_keyword, search_page)
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_lstrm_ai_term_id 
        ON open_law_lstrm_ai_data(term_id)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_lstrm_ai_keyword 
        ON open_law_lstrm_ai_data(search_keyword)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_lstrm_ai_collected_at 
        ON open_law_lstrm_ai_data(collected_at)
    """)
    
    conn.commit()
    logger.info("테이블 생성 완료")


def _load_keywords(keywords_str: str = None, keyword_file: str = None) -> List[str]:
    """키워드 로드"""
    keywords = []
    
    if keywords_str:
        keywords.extend([k.strip() for k in keywords_str.split(',') if k.strip()])
    
    if keyword_file:
        file_path = Path(keyword_file)
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                keywords.extend([line.strip() for line in f if line.strip()])
        else:
            logger.warning(f"키워드 파일을 찾을 수 없습니다: {keyword_file}")
    
    return list(set(keywords))  # 중복 제거


def main():
    parser = argparse.ArgumentParser(description='lstrmAI API 데이터 수집')
    parser.add_argument('--oc', required=True, help='사용자 이메일 ID')
    parser.add_argument('--keywords', help='검색 키워드 (쉼표 구분)')
    parser.add_argument('--keyword-file', help='키워드 파일 경로')
    parser.add_argument('--query', default='', help='검색 질의')
    parser.add_argument('--max-pages', type=int, help='최대 페이지 수')
    parser.add_argument('--display', type=int, default=100, help='페이지당 결과 수')
    parser.add_argument('--db-path', default='data/lawfirm.db', help='DB 경로')
    parser.add_argument('--rate-limit', type=float, default=0.5, help='요청 간 지연 (초)')
    
    args = parser.parse_args()
    
    # 연결 풀 사용
    connection_pool = get_connection_pool(args.db_path)
    
    # 테이블 생성
    with connection_pool.get_connection_context() as conn:
        _create_table(conn)
    
    # API 클라이언트 및 수집기 생성
    from scripts.ingest.lstrm_ai_client import LstrmAIClient
    from scripts.ingest.lstrm_ai_collector import LstrmAICollector
    
    client = LstrmAIClient(args.oc)
    client.rate_limit_delay = args.rate_limit
    
    collector = LstrmAICollector(client, args.db_path)
    
    # 수집 실행
    if args.keywords or args.keyword_file:
        keywords = _load_keywords(args.keywords, args.keyword_file)
        if not keywords:
            logger.error("키워드가 없습니다.")
            return
        
        logger.info(f"키워드 기반 수집 시작: {len(keywords)}개 키워드")
        total = collector.collect_by_keywords(keywords, args.max_pages)
    else:
        logger.info(f"전체 수집 시작: query='{args.query}'")
        total = collector.collect_all_pages(args.query, args.max_pages)
    
    logger.info(f"수집 완료: 총 {total}건의 데이터를 수집했습니다.")


if __name__ == '__main__':
    main()
```

---

## 수집 전략

### 5.1 키워드 기반 수집 (권장)

법률 도메인별 키워드를 사용하여 체계적으로 수집:

```python
# 민사법
keywords = ["계약", "해지", "손해배상", "위약금", "채무", "채권", "계약해제", "계약불이행"]

# 형사법
keywords = ["범죄", "형벌", "벌금", "징역", "구속", "기소", "공소", "재판"]

# 노동법
keywords = ["근로", "해고", "임금", "근로시간", "휴가", "부당해고", "근로계약", "임금체불"]

# 가족법
keywords = ["이혼", "양육권", "위자료", "재산분할", "친권", "면접교섭권", "부양"]

# 부동산법
keywords = ["아파트", "매매", "임대", "등기", "전세", "월세", "부동산", "소유권"]

# 상법
keywords = ["회사", "주식", "이사회", "주주", "합병", "분할", "상법", "법인"]
```

### 5.2 전체 수집

- `query` 없이 전체 페이지 순회
- 페이징으로 전체 데이터 수집
- 시간 소요가 크므로 주의 필요

---

## 주요 특징

### 1. 원본 JSON 보존
- `raw_response_json` 필드에 전체 API 응답을 JSON 문자열로 저장
- 나중에 원본 데이터 분석 및 재처리 가능

### 2. 연결 풀 사용
- `get_connection_pool()` 사용하여 스레드 안전성 보장
- 연결 재사용으로 성능 향상

### 3. 중복 방지
- `term_id + search_keyword + search_page` 조합으로 UNIQUE 제약
- 동일한 데이터 중복 저장 방지

### 4. Rate Limiting
- 요청 간 지연 시간 설정 (기본 0.5초)
- API 서버 부하 방지

### 5. 재시도 로직
- 네트워크 오류 시 자동 재시도 (최대 3회)
- 지수 백오프(exponential backoff) 적용

### 6. 로깅
- 수집 과정 상세 로깅
- 에러 발생 시 로그 기록

---

## 실행 예시

### 키워드 기반 수집

```bash
# 기본 키워드로 수집
python scripts/ingest/ingest_lstrm_ai.py \
    --oc test \
    --keywords "계약,손해배상,소송" \
    --max-pages 10 \
    --display 100

# 키워드 파일로 수집
python scripts/ingest/ingest_lstrm_ai.py \
    --oc test \
    --keyword-file data/keywords/legal_keywords.txt \
    --max-pages 5

# 요청 간 지연 시간 조정
python scripts/ingest/ingest_lstrm_ai.py \
    --oc test \
    --keywords "계약" \
    --rate-limit 1.0
```

### 전체 수집

```bash
# query 없이 전체 수집
python scripts/ingest/ingest_lstrm_ai.py \
    --oc test \
    --query "" \
    --max-pages 100

# 특정 질의로 수집
python scripts/ingest/ingest_lstrm_ai.py \
    --oc test \
    --query "법률" \
    --max-pages 50
```

### 환경 변수 사용

```bash
# .env 파일 또는 환경 변수
export LAW_OPEN_API_OC="your_email_id"

python scripts/ingest/ingest_lstrm_ai.py \
    --oc $LAW_OPEN_API_OC \
    --keywords "계약"
```

---

## 데이터 확인

### 원본 JSON 확인

```sql
-- 원본 JSON 확인
SELECT raw_response_json FROM open_law_lstrm_ai_data LIMIT 1;

-- JSON 파싱하여 확인
SELECT 
    json_extract(raw_response_json, '$.검색결과개수') as total_count,
    json_extract(raw_response_json, '$.page') as page
FROM open_law_lstrm_ai_data 
LIMIT 1;
```

### 통계 확인

```sql
-- 키워드별 통계
SELECT 
    search_keyword,
    COUNT(*) as count,
    COUNT(DISTINCT term_id) as unique_terms,
    MIN(collected_at) as first_collected,
    MAX(collected_at) as last_collected
FROM open_law_lstrm_ai_data
GROUP BY search_keyword
ORDER BY count DESC;

-- 수집 일자별 통계
SELECT 
    DATE(collected_at) as collection_date,
    COUNT(*) as count,
    COUNT(DISTINCT term_id) as unique_terms
FROM open_law_lstrm_ai_data
GROUP BY DATE(collected_at)
ORDER BY collection_date DESC;
```

### 특정 용어 검색

```sql
-- 용어명으로 검색
SELECT * FROM open_law_lstrm_ai_data 
WHERE term_name LIKE '%계약%'
ORDER BY collected_at DESC;

-- 용어 ID로 검색
SELECT * FROM open_law_lstrm_ai_data 
WHERE term_id = '12345';
```

### 원본 데이터 추출

```python
import sqlite3
import json

conn = sqlite3.connect('data/lawfirm.db')
cursor = conn.cursor()

# 원본 JSON 추출
cursor.execute("SELECT raw_response_json FROM open_law_lstrm_ai_data LIMIT 1")
row = cursor.fetchone()
if row:
    original_data = json.loads(row[0])
    print(json.dumps(original_data, ensure_ascii=False, indent=2))
```

---

## 주의사항

### 1. API 제한

- **요청 간 지연**: API 서버 부하 방지를 위해 요청 간 최소 0.5초 지연 권장
- **일일 요청 한도**: 국가법령정보센터의 일일 요청 한도 확인 필요
- **동시 요청 제한**: 동시 요청 수 제한 (권장: 1개씩 순차 처리)

### 2. 데이터 품질

- **중복 제거**: `term_id + search_keyword + search_page` 조합으로 자동 중복 방지
- **데이터 검증**: 필수 필드 존재 여부 확인
- **원본 보존**: `raw_response_json`에 원본 데이터 저장

### 3. 메모리 관리

- **배치 처리**: 대량 수집 시 배치 단위로 처리
- **주기적 저장**: 일정 간격으로 중간 저장
- **메모리 모니터링**: 메모리 사용량 추적

### 4. 에러 처리

- **재시도 로직**: 네트워크 오류 시 자동 재시도 (최대 3회)
- **에러 로깅**: 모든 에러를 로그 파일에 기록
- **부분 실패 처리**: 일부 용어 수집 실패 시에도 계속 진행

### 5. 데이터베이스

- **연결 풀 사용**: 반드시 `get_connection_pool()` 사용 (CRITICAL)
- **트랜잭션 관리**: 에러 발생 시 롤백 처리
- **인덱스 활용**: 검색 성능 향상을 위한 인덱스 활용

---

## 구현 단계

### Phase 1: 기본 구현
- [ ] API 클라이언트 구현 (`LstrmAIClient`)
- [ ] 데이터 수집기 구현 (`LstrmAICollector`)
- [ ] 데이터베이스 테이블 생성
- [ ] 메인 수집 스크립트 구현

### Phase 2: 고급 기능
- [ ] 재시도 로직 강화
- [ ] 로깅 시스템 개선
- [ ] 통계 및 모니터링
- [ ] 체크포인트 기능 (선택)

### Phase 3: 테스트
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 작성
- [ ] 실제 API 테스트

### Phase 4: 문서화
- [ ] 사용자 가이드 작성
- [ ] API 문서 업데이트

---

## 참고 자료

### 관련 문서

- [lstrmAIGuide 가이드](guides/lstrmAIGuide.md) - lstrmAI API 가이드
- [법령용어 수집 계획](legal_term_collection_plan.md) - 법령용어 수집 계획
- [Open Law API 가이드 맵](guide_id_map.md) - 전체 API 가이드 맵

### 외부 링크

- [국가법령정보센터 Open API](https://open.law.go.kr/LSO/openApi/guideList.do) - 공식 API 가이드
- [LAW OPEN DATA](http://www.law.go.kr/DRF/lawService.do) - API 엔드포인트

---

**작성일**: 2024-01-01  
**최종 수정일**: 2024-01-01  
**작성자**: LawFirmAI Development Team

