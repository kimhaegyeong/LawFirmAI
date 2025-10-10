# 국가법령정보센터 LAW OPEN API 개발자 가이드

## 개요
이 문서는 LawFirmAI 프로젝트에서 국가법령정보센터 LAW OPEN API를 활용한 개발을 위한 실무 가이드입니다.

## 빠른 시작

### 1. 환경 설정

```python
# 환경변수 설정
import os
os.environ["LAW_OPEN_API_OC"] = "your_email_id"  # 이메일 ID 부분만 입력

# API 클라이언트 초기화
from source.data.law_open_api_client import LawOpenAPIClient

client = LawOpenAPIClient()
```

### 2. 기본 사용법

```python
# 법령 목록 조회
laws = client.get_law_list_effective(
    query="민법",
    page=1,
    per_page=10,
    type="JSON"
)

# 법령 상세 조회
law_detail = client.get_law_detail_effective(
    law_id="법률 제12345호",
    type="JSON"
)
```

## API 카테고리별 개발 가이드

### 1. 법령 관련 API (18개)

#### 1.1 현행법령 조회
```python
# 시행일 기준 현행법령
laws_effective = client.get_law_list_effective(
    query="민법",
    page=1,
    per_page=20,
    type="JSON"
)

# 공포일 기준 현행법령
laws_promulgated = client.get_law_list_promulgated(
    query="상법",
    page=1,
    per_page=20,
    type="JSON"
)
```

#### 1.2 법령 연혁 조회
```python
# 법령 연혁 목록
law_history = client.get_law_history_list(
    law_id="법률 제12345호",
    type="JSON"
)

# 특정 연혁 상세
history_detail = client.get_law_history_detail(
    law_id="법률 제12345호",
    history_id="20230101",
    type="JSON"
)
```

#### 1.3 조항호목 조회
```python
# 시행일 기준 조항호목
jo_list_effective = client.get_law_jo_effective(
    law_id="법률 제12345호",
    type="JSON"
)

# 공포일 기준 조항호목
jo_list_promulgated = client.get_law_jo_promulgated(
    law_id="법률 제12345호",
    type="JSON"
)
```

#### 1.4 영문법령 조회
```python
# 영문 법령 목록
eng_laws = client.get_law_eng_list(
    query="Civil Act",
    type="JSON"
)

# 영문 법령 상세
eng_law_detail = client.get_law_eng_detail(
    law_id="법률 제12345호",
    type="JSON"
)
```

#### 1.5 이력 관련 조회
```python
# 법령 변경이력
change_history = client.get_law_change_list(
    law_id="법률 제12345호",
    type="JSON"
)

# 일자별 조문 개정 이력
day_jo_revise = client.get_law_day_jo_revise_list(
    law_id="법률 제12345호",
    start_date="20230101",
    end_date="20231231",
    type="JSON"
)

# 조문별 변경 이력
jo_change = client.get_law_jo_change_list(
    law_id="법률 제12345호",
    jo_number="제1조",
    type="JSON"
)
```

#### 1.6 연계 관련 조회
```python
# 법령-자치법규 연계 목록
ordinance_link = client.get_law_ordinance_link_list(
    law_id="법률 제12345호",
    type="JSON"
)

# 법령-자치법규 연계현황
link_status = client.get_law_ordinance_link_status(
    law_id="법률 제12345호",
    type="JSON"
)

# 위임법령 조회
delegated_laws = client.get_law_delegated(
    law_id="법률 제12345호",
    type="JSON"
)
```

#### 1.7 부가서비스 조회
```python
# 법령 체계도
system_diagram = client.get_law_system_list(
    law_id="법률 제12345호",
    type="JSON"
)

# 신구법 비교
old_new_comparison = client.get_law_old_new_list(
    law_id="법률 제12345호",
    compare_date="20230101",
    type="JSON"
)

# 3단 비교
three_way_comparison = client.get_law_three_compare_list(
    law_id="법률 제12345호",
    compare_dates=["20220101", "20230101", "20240101"],
    type="JSON"
)

# 법률명 약칭
abbreviations = client.get_law_abbreviation_list(
    query="민법",
    type="JSON"
)

# 삭제 데이터
deleted_data = client.get_law_deleted_data_list(
    start_date="20230101",
    end_date="20231231",
    type="JSON"
)

# 한눈보기
one_view = client.get_law_oneview_list(
    law_id="법률 제12345호",
    type="JSON"
)
```

### 2. 행정규칙 관련 API (4개)

```python
# 행정규칙 목록
admin_rules = client.get_admin_rule_list(
    query="시행규칙",
    page=1,
    per_page=20,
    type="JSON"
)

# 행정규칙 상세
admin_rule_detail = client.get_admin_rule_detail(
    rule_id="시행규칙 제123호",
    type="JSON"
)

# 행정규칙 신구법 비교
admin_rule_old_new = client.get_admin_rule_old_new_list(
    rule_id="시행규칙 제123호",
    compare_date="20230101",
    type="JSON"
)
```

### 3. 자치법규 관련 API (3개)

```python
# 자치법규 목록
local_ordinances = client.get_local_ordinance_list(
    query="조례",
    region="서울특별시",
    page=1,
    per_page=20,
    type="JSON"
)

# 자치법규 상세
ordinance_detail = client.get_local_ordinance_detail(
    ordinance_id="조례 제123호",
    type="JSON"
)

# 자치법규-법령 연계
ordinance_law_link = client.get_local_ordinance_law_link_list(
    ordinance_id="조례 제123호",
    type="JSON"
)
```

### 4. 판례 관련 API (2개)

```python
# 판례 목록
precedents = client.get_precedent_list(
    query="계약",
    court="대법원",
    start_date="20230101",
    end_date="20231231",
    page=1,
    per_page=20,
    type="JSON"
)

# 판례 상세
precedent_detail = client.get_precedent_detail(
    precedent_id="2023다12345",
    type="JSON"
)
```

### 5. 헌재결정례 관련 API (2개)

```python
# 헌재결정례 목록
constitutional_decisions = client.get_constitutional_list(
    query="기본권",
    decision_type="위헌",
    start_date="20230101",
    end_date="20231231",
    page=1,
    per_page=20,
    type="JSON"
)

# 헌재결정례 상세
constitutional_detail = client.get_constitutional_detail(
    decision_id="2023헌마123",
    type="JSON"
)
```

### 6. 법령해석례 관련 API (2개)

```python
# 법령해석례 목록
interpretations = client.get_interpretation_list(
    query="해석",
    ministry="법무부",
    start_date="20230101",
    end_date="20231231",
    page=1,
    per_page=20,
    type="JSON"
)

# 법령해석례 상세
interpretation_detail = client.get_interpretation_detail(
    interpretation_id="해석례 제123호",
    type="JSON"
)
```

### 7. 행정심판례 관련 API (2개)

```python
# 행정심판례 목록
appeals = client.get_appeal_list(
    query="처분",
    agency="국세청",
    start_date="20230101",
    end_date="20231231",
    page=1,
    per_page=20,
    type="JSON"
)

# 행정심판례 상세
appeal_detail = client.get_appeal_detail(
    appeal_id="심판례 제123호",
    type="JSON"
)
```

### 8. 위원회결정문 관련 API (24개)

```python
# 개인정보보호위원회
ppc_decisions = client.get_ppc_list(
    query="개인정보",
    start_date="20230101",
    end_date="20231231",
    page=1,
    per_page=20,
    type="JSON"
)

ppc_detail = client.get_ppc_detail(
    decision_id="결정 제123호",
    type="JSON"
)

# 공정거래위원회
ftc_decisions = client.get_ftc_list(
    query="공정거래",
    start_date="20230101",
    end_date="20231231",
    page=1,
    per_page=20,
    type="JSON"
)

# 금융위원회
fsc_decisions = client.get_fsc_list(
    query="금융",
    start_date="20230101",
    end_date="20231231",
    page=1,
    per_page=20,
    type="JSON"
)

# 노동위원회
nlrc_decisions = client.get_nlrc_list(
    query="노동",
    start_date="20230101",
    end_date="20231231",
    page=1,
    per_page=20,
    type="JSON"
)

# 기타 위원회들도 동일한 패턴으로 사용
```

### 9. 조약 관련 API (2개)

```python
# 조약 목록
treaties = client.get_treaty_list(
    query="협정",
    country="미국",
    start_date="20230101",
    end_date="20231231",
    page=1,
    per_page=20,
    type="JSON"
)

# 조약 상세
treaty_detail = client.get_treaty_detail(
    treaty_id="조약 제123호",
    type="JSON"
)
```

### 10. 별표ㆍ서식 관련 API (3개)

```python
# 법령 별표ㆍ서식
law_forms = client.get_law_form_list(
    law_id="법률 제12345호",
    type="JSON"
)

# 행정규칙 별표ㆍ서식
admin_rule_forms = client.get_admin_rule_form_list(
    rule_id="시행규칙 제123호",
    type="JSON"
)

# 자치법규 별표ㆍ서식
ordinance_forms = client.get_local_ordinance_form_list(
    ordinance_id="조례 제123호",
    type="JSON"
)
```

### 11. 학칙ㆍ공단ㆍ공공기관 관련 API (2개)

```python
# 학칙ㆍ공단ㆍ공공기관 목록
school_public_rules = client.get_school_public_rule_list(
    query="학칙",
    institution_type="대학교",
    page=1,
    per_page=20,
    type="JSON"
)

# 학칙ㆍ공단ㆍ공공기관 상세
rule_detail = client.get_school_public_rule_detail(
    rule_id="학칙 제123호",
    type="JSON"
)
```

### 12. 법령용어 관련 API (2개)

```python
# 법령 용어 목록
legal_terms = client.get_legal_term_list(
    query="계약",
    page=1,
    per_page=20,
    type="JSON"
)

# 법령 용어 상세
term_detail = client.get_legal_term_detail(
    term_id="계약",
    type="JSON"
)
```

### 13. 모바일 관련 API (12개)

```python
# 모바일 법령 목록
mobile_laws = client.get_mobile_law_list(
    query="민법",
    page=1,
    per_page=10,
    type="JSON"
)

# 모바일 법령 상세
mobile_law_detail = client.get_mobile_law_detail(
    law_id="법률 제12345호",
    type="JSON"
)

# 기타 모바일 API들도 동일한 패턴으로 사용
```

### 14. 맞춤형 관련 API (6개)

```python
# 맞춤형 법령 목록
custom_laws = client.get_custom_law_list(
    user_id="user123",
    preferences={
        "categories": ["민법", "상법"],
        "date_range": "2023-01-01,2023-12-31"
    },
    type="JSON"
)

# 맞춤형 법령 조문 목록
custom_law_jo = client.get_custom_law_jo_list(
    user_id="user123",
    law_id="법률 제12345호",
    type="JSON"
)
```

### 15. 법령정보지식베이스 관련 API (7개)

```python
# 법령용어 AI 조회
legal_term_ai = client.get_legal_term_ai(
    query="계약",
    type="JSON"
)

# 일상용어 조회
daily_term = client.get_daily_term(
    query="계약서",
    type="JSON"
)

# 법령용어-일상용어 연계
term_relation = client.get_legal_daily_term_relation(
    legal_term="계약",
    daily_term="계약서",
    type="JSON"
)

# 관련법령 조회
related_laws = client.get_related_law(
    law_id="법률 제12345호",
    type="JSON"
)
```

### 16. 중앙부처 1차 해석 관련 API (15개)

```python
# 고용노동부 법령해석
moel_interpretations = client.get_moel_interpretation_list(
    query="근로기준법",
    start_date="20230101",
    end_date="20231231",
    page=1,
    per_page=20,
    type="JSON"
)

moel_detail = client.get_moel_interpretation_detail(
    interpretation_id="해석 제123호",
    type="JSON"
)

# 기타 부처들도 동일한 패턴으로 사용
```

### 17. 특별행정심판 관련 API (4개)

```python
# 조세심판원 특별행정심판례
tt_appeals = client.get_tt_appeal_list(
    query="세법",
    start_date="20230101",
    end_date="20231231",
    page=1,
    per_page=20,
    type="JSON"
)

tt_appeal_detail = client.get_tt_appeal_detail(
    appeal_id="심판례 제123호",
    type="JSON"
)

# 해양안전심판원 특별행정심판례
kmst_appeals = client.get_kmst_appeal_list(
    query="해양",
    start_date="20230101",
    end_date="20231231",
    page=1,
    per_page=20,
    type="JSON"
)
```

## 에러 처리

```python
try:
    result = client.get_law_list_effective(
        query="민법",
        type="JSON"
    )
except LawOpenAPIError as e:
    print(f"API 오류: {e.message}")
    print(f"오류 코드: {e.code}")
except RateLimitError as e:
    print(f"요청 제한: {e.message}")
    print(f"재시도 가능 시간: {e.retry_after}")
except Exception as e:
    print(f"예상치 못한 오류: {e}")
```

## 캐싱 전략

```python
from functools import lru_cache
import time

class CachedLawOpenAPIClient(LawOpenAPIClient):
    def __init__(self, cache_ttl=3600):  # 1시간 캐시
        super().__init__()
        self.cache_ttl = cache_ttl
        self.cache = {}
    
    @lru_cache(maxsize=128)
    def get_law_list_effective(self, query, page=1, per_page=20, type="JSON"):
        cache_key = f"law_list_effective_{query}_{page}_{per_page}_{type}"
        
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
        
        result = super().get_law_list_effective(query, page, per_page, type)
        self.cache[cache_key] = (result, time.time())
        return result
```

## 배치 처리

```python
def batch_collect_laws(law_ids, batch_size=50):
    """법령 데이터 배치 수집"""
    results = []
    
    for i in range(0, len(law_ids), batch_size):
        batch = law_ids[i:i + batch_size]
        
        # 병렬 처리
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(client.get_law_detail_effective, law_id, "JSON")
                for law_id in batch
            ]
            
            batch_results = [future.result() for future in futures]
            results.extend(batch_results)
        
        # API 제한 고려하여 대기
        time.sleep(1)
    
    return results
```

## 모니터링 및 로깅

```python
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('law_api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('LawOpenAPI')

class MonitoredLawOpenAPIClient(LawOpenAPIClient):
    def get_law_list_effective(self, query, page=1, per_page=20, type="JSON"):
        start_time = time.time()
        
        try:
            result = super().get_law_list_effective(query, page, per_page, type)
            
            duration = time.time() - start_time
            logger.info(f"API 호출 성공: {query}, 소요시간: {duration:.2f}초")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"API 호출 실패: {query}, 오류: {e}, 소요시간: {duration:.2f}초")
            raise
```

## 성능 최적화 팁

1. **병렬 처리**: 여러 API 호출을 병렬로 처리
2. **캐싱**: 동일한 요청에 대한 결과 캐싱
3. **배치 처리**: 대량 데이터 수집 시 배치 단위로 처리
4. **요청 제한 준수**: API 제한을 고려한 요청 간격 조절
5. **에러 재시도**: 실패한 요청에 대한 재시도 로직

## 참고사항

- 모든 API는 HTML, XML, JSON 형식으로 응답 제공
- API 키(OC)는 보안을 위해 환경변수로 관리
- 요청 제한을 준수하여 서비스 안정성 확보
- 에러 처리 및 로깅을 통한 디버깅 용이성 확보

---

*이 가이드는 LawFirmAI 프로젝트의 개발을 위해 작성되었습니다.*
*최신 정보는 [공식 API 문서](https://open.law.go.kr/LSO/openApi/guideList.do)를 참조하시기 바랍니다.*
