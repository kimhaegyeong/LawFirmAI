# 국가법령정보센터 LAW OPEN API 매핑 테이블

## 개요
이 문서는 국가법령정보센터 LAW OPEN API의 파라미터와 실제 구현을 연결하는 매핑 테이블입니다. 개발 시 API 호출에 필요한 정보를 빠르게 참조할 수 있습니다.

## API 매핑 테이블

### 1. 법령 관련 API

| API 이름 | HTML Name | Target | Type | Description |
|---------|-----------|--------|------|-------------|
| 현행법령(시행일) 목록 조회 | `lsEfYdListGuide` | `lsEfYd` | List | 시행일 기준 현행법령 목록 |
| 현행법령(시행일) 본문 조회 | `lsEfYdInfoGuide` | `eflaw` | Detail | 시행일 기준 현행법령 본문 |
| 현행법령(공포일) 목록 조회 | `lsNwListGuide` | `lsNw` | List | 공포일 기준 현행법령 목록 |
| 현행법령(공포일) 본문 조회 | `lsNwInfoGuide` | `prlaw` | Detail | 공포일 기준 현행법령 본문 |
| 법령 연혁 목록 조회 | `lsHstListGuide` | `lsHst` | List | 법령 연혁 목록 |
| 법령 연혁 본문 조회 | `lsHstInfoGuide` | `hstlaw` | Detail | 법령 연혁 본문 |
| 현행법령(시행일) 조항호목 조회 | `lsEfYdJoListGuide` | `eflaw` | JoList | 시행일 기준 조항호목 |
| 현행법령(공포일) 조항호목 조회 | `lsNwJoListGuide` | `prlaw` | JoList | 공포일 기준 조항호목 |
| 영문 법령 목록 조회 | `lsEngListGuide` | `lsEng` | List | 영문 법령 목록 |
| 영문 법령 본문 조회 | `lsEngInfoGuide` | `englaw` | Detail | 영문 법령 본문 |
| 법령 변경이력 목록 조회 | `lsChgListGuide` | `lsChg` | List | 법령 변경이력 목록 |
| 일자별 조문 개정 이력 목록 조회 | `lsDayJoRvsListGuide` | `lsDayJoRvs` | List | 일자별 조문 개정 이력 |
| 조문별 변경 이력 목록 조회 | `lsJoChgListGuide` | `lsJoChg` | List | 조문별 변경 이력 |
| 법령 기준 자치법규 연계 목록 조회 | `lsOrdinConListGuide` | `lsOrdinCon` | List | 법령-자치법규 연계 목록 |
| 법령-자치법규 연계현황 조회 | `lsOrdinConGuide` | `lsOrdinCon` | Status | 법령-자치법규 연계현황 |
| 위임법령 조회 | `lsDelegated` | `lsDelegated` | Detail | 위임법령 정보 |
| 법령 체계도 목록 조회 | `lsStmdListGuide` | `lsStmd` | List | 법령 체계도 목록 |
| 법령 체계도 본문 조회 | `lsStmdInfoGuide` | `lsStmd` | Detail | 법령 체계도 본문 |
| 신구법 목록 조회 | `oldAndNewListGuide` | `lsNewOld` | List | 신구법 비교 목록 |
| 신구법 본문 조회 | `oldAndNewInfoGuide` | `lsNewOld` | Detail | 신구법 비교 본문 |
| 3단 비교 목록 조회 | `thdCmpListGuide` | `lsThdCmp` | List | 3단 비교 목록 |
| 3단 비교 본문 조회 | `thdCmpInfoGuide` | `lsThdCmp` | Detail | 3단 비교 본문 |
| 법률명 약칭 조회 | `lsAbrvListGuide` | `lsAbrv` | List | 법률명 약칭 목록 |
| 삭제 데이터 목록 조회 | `datDelHstGuide` | `datDelHst` | List | 삭제된 데이터 목록 |
| 한눈보기 목록 조회 | `oneViewListGuide` | `oneView` | List | 한눈보기 목록 |
| 한눈보기 본문 조회 | `oneViewInfoGuide` | `oneView` | Detail | 한눈보기 본문 |

### 2. 행정규칙 관련 API

| API 이름 | HTML Name | Target | Type | Description |
|---------|-----------|--------|------|-------------|
| 행정규칙 목록 조회 | `admrulListGuide` | `lsAdRul` | List | 행정규칙 목록 |
| 행정규칙 본문 조회 | `admrulInfoGuide` | `adRul` | Detail | 행정규칙 본문 |
| 행정규칙 신구법 비교 목록 조회 | `admrulOldAndNewListGuide` | `adRulNewOld` | List | 행정규칙 신구법 비교 목록 |
| 행정규칙 신구법 비교 본문 조회 | `admrulOldAndNewInfoGuide` | `adRulNewOld` | Detail | 행정규칙 신구법 비교 본문 |

### 3. 자치법규 관련 API

| API 이름 | HTML Name | Target | Type | Description |
|---------|-----------|--------|------|-------------|
| 자치법규 목록 조회 | `ordinListGuide` | `lsOrdinance` | List | 자치법규 목록 |
| 자치법규 본문 조회 | `ordinInfoGuide` | `ordinance` | Detail | 자치법규 본문 |
| 자치법규 기준 법령 연계 목록 조회 | `ordinLsConListGuide` | `ordinLsCon` | List | 자치법규-법령 연계 목록 |

### 4. 판례 관련 API

| API 이름 | HTML Name | Target | Type | Description |
|---------|-----------|--------|------|-------------|
| 판례 목록 조회 | `precListGuide` | `prec` | List | 판례 목록 |
| 판례 본문 조회 | `precInfoGuide` | `prec` | Detail | 판례 본문 |

### 5. 헌재결정례 관련 API

| API 이름 | HTML Name | Target | Type | Description |
|---------|-----------|--------|------|-------------|
| 헌재결정례 목록 조회 | `detcListGuide` | `detc` | List | 헌재결정례 목록 |
| 헌재결정례 본문 조회 | `detcInfoGuide` | `detc` | Detail | 헌재결정례 본문 |

### 6. 법령해석례 관련 API

| API 이름 | HTML Name | Target | Type | Description |
|---------|-----------|--------|------|-------------|
| 법령해석례 목록 조회 | `expcListGuide` | `expc` | List | 법령해석례 목록 |
| 법령해석례 본문 조회 | `expcInfoGuide` | `expc` | Detail | 법령해석례 본문 |

### 7. 행정심판례 관련 API

| API 이름 | HTML Name | Target | Type | Description |
|---------|-----------|--------|------|-------------|
| 행정심판례 목록 조회 | `deccListGuide` | `decc` | List | 행정심판례 목록 |
| 행정심판례 본문 조회 | `deccInfoGuide` | `decc` | Detail | 행정심판례 본문 |

### 8. 위원회결정문 관련 API

| 위원회명 | HTML Name (목록) | HTML Name (본문) | Target (목록) | Target (본문) | Type |
|---------|-----------------|-----------------|---------------|---------------|------|
| 개인정보보호위원회 | `ppcListGuide` | `ppcInfoGuide` | `lsPpc` | `ppc` | List/Detail |
| 고용보험심사위원회 | `eiacListGuide` | `eiacInfoGuide` | `lsEiac` | `eiac` | List/Detail |
| 공정거래위원회 | `ftcListGuide` | `ftcInfoGuide` | `lsFtc` | `ftc` | List/Detail |
| 국민권익위원회 | `acrListGuide` | `acrInfoGuide` | `lsAcr` | `acr` | List/Detail |
| 금융위원회 | `fscListGuide` | `fscInfoGuide` | `lsFsc` | `fsc` | List/Detail |
| 노동위원회 | `nlrcListGuide` | `nlrcInfoGuide` | `lsNlrc` | `nlrc` | List/Detail |
| 방송통신위원회 | `kccListGuide` | `kccInfoGuide` | `lsKcc` | `kcc` | List/Detail |
| 산업재해보상보험재심사위원회 | `iaciacListGuide` | `iaciacInfoGuide` | `lsIaciac` | `iaciac` | List/Detail |
| 중앙토지수용위원회 | `ocltListGuide` | `ocltInfoGuide` | `lsOclt` | `oclt` | List/Detail |
| 중앙환경분쟁조정위원회 | `eccListGuide` | `eccInfoGuide` | `lsEcc` | `ecc` | List/Detail |
| 증권선물위원회 | `sfcListGuide` | `sfcInfoGuide` | `lsSfc` | `sfc` | List/Detail |
| 국가인권위원회 | `nhrckListGuide` | `nhrckInfoGuide` | `lsNhrck` | `nhrck` | List/Detail |

### 9. 조약 관련 API

| API 이름 | HTML Name | Target | Type | Description |
|---------|-----------|--------|------|-------------|
| 조약 목록 조회 | `trtyListGuide` | `lsTreaty` | List | 조약 목록 |
| 조약 본문 조회 | `trtyInfoGuide` | `treaty` | Detail | 조약 본문 |

### 10. 별표ㆍ서식 관련 API

| API 이름 | HTML Name | Target | Type | Description |
|---------|-----------|--------|------|-------------|
| 법령 별표ㆍ서식 목록 조회 | `lsBylListGuide` | `lsByl` | List | 법령 별표ㆍ서식 목록 |
| 행정규칙 별표ㆍ서식 목록 조회 | `admrulBylListGuide` | `adRulByl` | List | 행정규칙 별표ㆍ서식 목록 |
| 자치법규 별표ㆍ서식 목록 조회 | `ordinBylListGuide` | `ordinByl` | List | 자치법규 별표ㆍ서식 목록 |

### 11. 학칙ㆍ공단ㆍ공공기관 관련 API

| API 이름 | HTML Name | Target | Type | Description |
|---------|-----------|--------|------|-------------|
| 학칙ㆍ공단ㆍ공공기관 목록 조회 | `schlPubRulListGuide` | `lsSchlPubRul` | List | 학칙ㆍ공단ㆍ공공기관 목록 |
| 학칙ㆍ공단ㆍ공공기관 본문 조회 | `schlPubRulInfoGuide` | `schlPubRul` | Detail | 학칙ㆍ공단ㆍ공공기관 본문 |

### 12. 법령용어 관련 API

| API 이름 | HTML Name | Target | Type | Description |
|---------|-----------|--------|------|-------------|
| 법령 용어 목록 조회 | `lsTrmListGuide` | `lsTrm` | List | 법령 용어 목록 |
| 법령 용어 본문 조회 | `lsTrmInfoGuide` | `lsTrm` | Detail | 법령 용어 본문 |

### 13. 모바일 관련 API

| API 이름 | HTML Name | Target | Type | Description |
|---------|-----------|--------|------|-------------|
| 법령 목록 조회 (모바일) | `mobLsListGuide` | `mobLs` | List | 모바일 법령 목록 |
| 법령 본문 조회 (모바일) | `mobLsInfoGuide` | `mobLs` | Detail | 모바일 법령 본문 |
| 행정규칙 목록 조회 (모바일) | `mobAdmrulListguide` | `mobAdRul` | List | 모바일 행정규칙 목록 |
| 행정규칙 본문 조회 (모바일) | `mobAdmrulInfoGuide` | `mobAdRul` | Detail | 모바일 행정규칙 본문 |
| 자치법규 목록 조회 (모바일) | `mobOrdinListGuide` | `mobOrdin` | List | 모바일 자치법규 목록 |
| 자치법규 본문 조회 (모바일) | `mobOrdinInfoGuide` | `mobOrdin` | Detail | 모바일 자치법규 본문 |
| 판례 목록 조회 (모바일) | `mobPrecListGuide` | `mobPrec` | List | 모바일 판례 목록 |
| 판례 본문 조회 (모바일) | `mobPrecInfoGuide` | `mobPrec` | Detail | 모바일 판례 본문 |
| 헌재결정례 목록 조회 (모바일) | `mobDetcListGuide` | `mobDetc` | List | 모바일 헌재결정례 목록 |
| 헌재결정례 본문 조회 (모바일) | `mobDetcInfoGuide` | `mobDetc` | Detail | 모바일 헌재결정례 본문 |
| 법령해석례 목록 조회 (모바일) | `mobExpcListGuide` | `mobExpc` | List | 모바일 법령해석례 목록 |
| 법령해석례 본문 조회 (모바일) | `mobExpcInfoGuide` | `mobExpc` | Detail | 모바일 법령해석례 본문 |
| 행정심판례 목록 조회 (모바일) | `mobDeccListGuide` | `mobDecc` | List | 모바일 행정심판례 목록 |
| 행정심판례 본문 조회 (모바일) | `mobDeccInfoGuide` | `mobDecc` | Detail | 모바일 행정심판례 본문 |
| 조약 목록 조회 (모바일) | `mobTrtyListGuide` | `mobTrty` | List | 모바일 조약 목록 |
| 조약 본문 조회 (모바일) | `mobTrtyInfoGuide` | `mobTrty` | Detail | 모바일 조약 본문 |
| 법령 별표ㆍ서식 목록 조회 (모바일) | `mobLsBylListGuide` | `mobLsByl` | List | 모바일 법령 별표ㆍ서식 목록 |
| 행정규칙 별표ㆍ서식 목록 조회 (모바일) | `mobAdmrulBylListGuide` | `mobAdRulByl` | List | 모바일 행정규칙 별표ㆍ서식 목록 |
| 자치법규 별표ㆍ서식 목록 조회 (모바일) | `mobOrdinBylListGuide` | `mobOrdinByl` | List | 모바일 자치법규 별표ㆍ서식 목록 |
| 법령 용어 목록 조회 (모바일) | `mobLsTrmListGuide` | `mobLsTrm` | List | 모바일 법령 용어 목록 |

### 14. 맞춤형 관련 API

| API 이름 | HTML Name | Target | Type | Description |
|---------|-----------|--------|------|-------------|
| 맞춤형 법령 목록 조회 | `custLsListGuide` | `custLs` | List | 맞춤형 법령 목록 |
| 맞춤형 법령 조문 목록 조회 | `custLsJoListGuide` | `custLsJo` | List | 맞춤형 법령 조문 목록 |
| 맞춤형 행정규칙 목록 조회 | `custAdmrulListGuide` | `custAdRul` | List | 맞춤형 행정규칙 목록 |
| 맞춤형 행정규칙 조문 목록 조회 | `custAdmrulJoListGuide` | `custAdRulJo` | List | 맞춤형 행정규칙 조문 목록 |
| 맞춤형 자치법규 목록 조회 | `custOrdinListGuide` | `custOrdin` | List | 맞춤형 자치법규 목록 |
| 맞춤형 자치법규 조문 목록 조회 | `custOrdinJoListGuide` | `custOrdinJo` | List | 맞춤형 자치법규 조문 목록 |

### 15. 법령정보지식베이스 관련 API

| API 이름 | HTML Name | Target | Type | Description |
|---------|-----------|--------|------|-------------|
| 법령용어 조회 | `lstrmAIGuide` | `lsTrmAI` | AI | 법령용어 AI 조회 |
| 일상용어 조회 | `dlytrmGuide` | `dlyTrm` | Detail | 일상용어 조회 |
| 법령용어-일상용어 연계 조회 | `lstrmRltGuide` | `lsTrmRlt` | Relation | 법령용어-일상용어 연계 |
| 일상용어-법령용어 연계 조회 | `dlytrmRltGuide` | `dlyTrmRlt` | Relation | 일상용어-법령용어 연계 |
| 법령용어-조문 연계 조회 | `lstrmRltJoGuide` | `lsTrmRltJo` | Relation | 법령용어-조문 연계 |
| 조문-법령용어 연계 조회 | `joRltLstrmGuide` | `joRltLsTrm` | Relation | 조문-법령용어 연계 |
| 관련법령 조회 | `lsRltGuide` | `lsRlt` | Relation | 관련법령 조회 |

### 16. 중앙부처 1차 해석 관련 API

| 부처명 | HTML Name (목록) | HTML Name (본문) | Target (목록) | Target (본문) | Type |
|-------|-----------------|-----------------|---------------|---------------|------|
| 고용노동부 | `cgmExpcMoelListGuide` | `cgmExpcMoelInfoGuide` | `cgmExpcMoel` | `cgmExpcMoel` | List/Detail |
| 국토교통부 | `cgmExpcMolitListGuide` | `cgmExpcMolitInfoGuide` | `cgmExpcMolit` | `cgmExpcMolit` | List/Detail |
| 기획재정부 | `cgmExpcMoefListGuide` | - | `cgmExpcMoef` | - | List |
| 해양수산부 | `cgmExpcMofListGuide` | `cgmExpcMofInfoGuide` | `cgmExpcMof` | `cgmExpcMof` | List/Detail |
| 행정안전부 | `cgmExpcMoisListGuide` | `cgmExpcMoisInfoGuide` | `cgmExpcMois` | `cgmExpcMois` | List/Detail |
| 환경부 | `cgmExpcMeListGuide` | `cgmExpcMeInfoGuide` | `cgmExpcMe` | `cgmExpcMe` | List/Detail |
| 관세청 | `cgmExpcKcsListGuide` | `cgmExpcKcsInfoGuide` | `cgmExpcKcs` | `cgmExpcKcs` | List/Detail |
| 국세청 | `cgmExpcNtsListGuide` | - | `cgmExpcNts` | - | List |

### 17. 특별행정심판 관련 API

| 기관명 | HTML Name (목록) | HTML Name (본문) | Target (목록) | Target (본문) | Type |
|-------|-----------------|-----------------|---------------|---------------|------|
| 조세심판원 | `specialDeccTtListGuide` | `specialDeccTtInfoGuide` | `specialDeccTt` | `specialDeccTt` | List/Detail |
| 해양안전심판원 | `specialDeccKmstListGuide` | `specialDeccKmstInfoGuide` | `specialDeccKmst` | `specialDeccKmst` | List/Detail |

---

## API 호출 패턴

### 기본 URL 구조
```
http://www.law.go.kr/DRF/lawSearch.do?OC={OC}&target={target}&type={type}&{additional_params}
```

### 공통 파라미터
- `OC`: 사용자 이메일 ID (필수)
- `target`: 서비스 대상 (필수)
- `type`: 출력 형태 - HTML, XML, JSON (필수)

### 추가 파라미터 예시
- `query`: 검색어
- `page`: 페이지 번호
- `perPage`: 페이지당 결과 수
- `sort`: 정렬 기준
- `date`: 날짜 범위

### 응답 형식
- **HTML**: 웹 페이지 형태
- **XML**: 구조화된 XML 데이터
- **JSON**: JSON 형태의 데이터

---

## 개발 시 주의사항

1. **API 키 관리**: OC 파라미터는 보안이 중요하므로 환경변수로 관리
2. **요청 제한**: API 호출 빈도 제한을 준수
3. **에러 처리**: API 응답 상태 코드 및 에러 메시지 처리
4. **데이터 검증**: 응답 데이터의 유효성 검증
5. **캐싱**: 동일한 요청에 대한 캐싱 전략 적용
6. **로깅**: API 호출 로그 기록 및 모니터링

---

*이 매핑 테이블은 국가법령정보센터 LAW OPEN API 가이드를 기반으로 작성되었습니다.*
*최신 정보는 [공식 사이트](https://open.law.go.kr/LSO/openApi/guideList.do)를 참조하시기 바랍니다.*
