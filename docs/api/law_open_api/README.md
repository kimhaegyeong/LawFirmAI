# 국가법령정보센터 LAW OPEN API 가이드

## 개요
국가법령정보센터에서 제공하는 LAW OPEN API의 상세 가이드입니다. 각 API별로 분리된 문서를 통해 효율적으로 API를 활용할 수 있습니다.

**최신 업데이트 (2025-09-26)**:
- ✅ **네트워크 안정성 향상**: DNS 해결 실패, 타임아웃 오류 처리 개선
- ✅ **재시도 로직 강화**: 지수 백오프 방식으로 재시도 간격 점진적 증가
- ✅ **타임아웃 설정 개선**: 연결 타임아웃(30초)과 읽기 타임아웃(120초) 분리
- ✅ **재시도 횟수 증가**: 5회 → 10회로 증가
- ✅ **에러 핸들링 개선**: 상세한 오류 메시지 및 해결 방법 제시

## API 카테고리별 가이드

### 1. 법령 (Laws) - 18개 API

#### 1.1 본문 관련 API (6개)
- [현행법령(시행일) 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsEfYdListGuide) - `lsEfYdListGuide`
- [현행법령(시행일) 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsEfYdInfoGuide) - `lsEfYdInfoGuide`
- [현행법령(공포일) 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsNwListGuide) - `lsNwListGuide`
- [현행법령(공포일) 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsNwInfoGuide) - `lsNwInfoGuide`
- [법령 연혁 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsHstListGuide) - `lsHstListGuide`
- [법령 연혁 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsHstInfoGuide) - `lsHstInfoGuide`

#### 1.2 조항호목 관련 API (2개)
- [현행법령(시행일) 본문 조항호목 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsEfYdJoListGuide) - `lsEfYdJoListGuide`
- [현행법령(공포일) 본문 조항호목 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsNwJoListGuide) - `lsNwJoListGuide`

#### 1.3 영문법령 관련 API (2개)
- [영문 법령 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsEngListGuide) - `lsEngListGuide`
- [영문 법령 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsEngInfoGuide) - `lsEngInfoGuide`

#### 1.4 이력 관련 API (3개)
- [법령 변경이력 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsChgListGuide) - `lsChgListGuide`
- [일자별 조문 개정 이력 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsDayJoRvsListGuide) - `lsDayJoRvsListGuide`
- [조문별 변경 이력 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsJoChgListGuide) - `lsJoChgListGuide`

#### 1.5 연계 관련 API (3개)
- [법령 기준 자치법규 연계 관련 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsOrdinConListGuide) - `lsOrdinConListGuide`
- [법령-자치법규 연계현황 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsOrdinConGuide) - `lsOrdinConGuide`
- [위임법령 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsDelegated) - `lsDelegated`

#### 1.6 부가서비스 관련 API (10개)
- [법령 체계도 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsStmdListGuide) - `lsStmdListGuide`
- [법령 체계도 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsStmdInfoGuide) - `lsStmdInfoGuide`
- [신구법 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=oldAndNewListGuide) - `oldAndNewListGuide`
- [신구법 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=oldAndNewInfoGuide) - `oldAndNewInfoGuide`
- [3단 비교 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=thdCmpListGuide) - `thdCmpListGuide`
- [3단 비교 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=thdCmpInfoGuide) - `thdCmpInfoGuide`
- [법률명 약칭 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsAbrvListGuide) - `lsAbrvListGuide`
- [삭제 데이터 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=datDelHstGuide) - `datDelHstGuide`
- [한눈보기 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=oneViewListGuide) - `oneViewListGuide`
- [한눈보기 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=oneViewInfoGuide) - `oneViewInfoGuide`

### 2. 행정규칙 (Administrative Rules) - 4개 API
- [행정규칙 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=admrulListGuide) - `admrulListGuide`
- [행정규칙 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=admrulInfoGuide) - `admrulInfoGuide`
- [행정규칙 신구법 비교 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=admrulOldAndNewListGuide) - `admrulOldAndNewListGuide`
- [행정규칙 신구법 비교 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=admrulOldAndNewInfoGuide) - `admrulOldAndNewInfoGuide`

### 3. 자치법규 (Local Ordinances) - 3개 API
- [자치법규 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=ordinListGuide) - `ordinListGuide`
- [자치법규 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=ordinInfoGuide) - `ordinInfoGuide`
- [자치법규 기준 법령 연계 관련 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=ordinLsConListGuide) - `ordinLsConListGuide`

### 4. 판례 (Precedents) - 2개 API
- [판례 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=precListGuide) - `precListGuide`
- [판례 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=precInfoGuide) - `precInfoGuide`

### 5. 헌재결정례 (Constitutional Court Decisions) - 2개 API
- [헌재결정례 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=detcListGuide) - `detcListGuide`
- [헌재결정례 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=detcInfoGuide) - `detcInfoGuide`

### 6. 법령해석례 (Legal Interpretations) - 2개 API
- [법령해석례 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=expcListGuide) - `expcListGuide`
- [법령해석례 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=expcInfoGuide) - `expcInfoGuide`

### 7. 행정심판례 (Administrative Appeals) - 2개 API
- [행정심판례 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=deccListGuide) - `deccListGuide`
- [행정심판례 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=deccInfoGuide) - `deccInfoGuide`

### 8. 위원회결정문 (Committee Decisions) - 24개 API

#### 8.1 개인정보보호위원회 (2개)
- [개인정보보호위원회 결정문 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=ppcListGuide) - `ppcListGuide`
- [개인정보보호위원회 결정문 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=ppcInfoGuide) - `ppcInfoGuide`

#### 8.2 고용보험심사위원회 (2개)
- [고용보험심사위원회 결정문 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=eiacListGuide) - `eiacListGuide`
- [고용보험심사위원회 결정문 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=eiacInfoGuide) - `eiacInfoGuide`

#### 8.3 공정거래위원회 (2개)
- [공정거래위원회 결정문 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=ftcListGuide) - `ftcListGuide`
- [공정거래위원회 결정문 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=ftcInfoGuide) - `ftcInfoGuide`

#### 8.4 국민권익위원회 (2개)
- [국민권익위원회 결정문 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=acrListGuide) - `acrListGuide`
- [국민권익위원회 결정문 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=acrInfoGuide) - `acrInfoGuide`

#### 8.5 금융위원회 (2개)
- [금융위원회 결정문 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=fscListGuide) - `fscListGuide`
- [금융위원회 결정문 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=fscInfoGuide) - `fscInfoGuide`

#### 8.6 노동위원회 (2개)
- [노동위원회 결정문 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=nlrcListGuide) - `nlrcListGuide`
- [노동위원회 결정문 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=nlrcInfoGuide) - `nlrcInfoGuide`

#### 8.7 방송통신위원회 (2개)
- [방송통신위원회 결정문 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=kccListGuide) - `kccListGuide`
- [방송통신위원회 결정문 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=kccInfoGuide) - `kccInfoGuide`

#### 8.8 산업재해보상보험재심사위원회 (2개)
- [산업재해보상보험재심사위원회 결정문 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=iaciacListGuide) - `iaciacListGuide`
- [산업재해보상보험재심사위원회 결정문 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=iaciacInfoGuide) - `iaciacInfoGuide`

#### 8.9 중앙토지수용위원회 (2개)
- [중앙토지수용위원회 결정문 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=ocltListGuide) - `ocltListGuide`
- [중앙토지수용위원회 결정문 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=ocltInfoGuide) - `ocltInfoGuide`

#### 8.10 중앙환경분쟁조정위원회 (2개)
- [중앙환경분쟁조정위원회 결정문 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=eccListGuide) - `eccListGuide`
- [중앙환경분쟁조정위원회 결정문 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=eccInfoGuide) - `eccInfoGuide`

#### 8.11 증권선물위원회 (2개)
- [증권선물위원회 결정문 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=sfcListGuide) - `sfcListGuide`
- [증권선물위원회 결정문 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=sfcInfoGuide) - `sfcInfoGuide`

#### 8.12 국가인권위원회 (2개)
- [국가인권위원회 결정문 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=nhrckListGuide) - `nhrckListGuide`
- [국가인권위원회 결정문 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=nhrckInfoGuide) - `nhrckInfoGuide`

### 9. 조약 (Treaties) - 2개 API
- [조약 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=trtyListGuide) - `trtyListGuide`
- [조약 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=trtyInfoGuide) - `trtyInfoGuide`

### 10. 별표ㆍ서식 (Forms and Tables) - 3개 API
- [법령 별표ㆍ서식 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsBylListGuide) - `lsBylListGuide`
- [행정규칙 별표ㆍ서식 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=admrulBylListGuide) - `admrulBylListGuide`
- [자치법규 별표ㆍ서식 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=ordinBylListGuide) - `ordinBylListGuide`

### 11. 학칙ㆍ공단ㆍ공공기관 (School Rules, Public Corporations, Public Institutions) - 2개 API
- [학칙ㆍ공단ㆍ공공기관 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=schlPubRulListGuide) - `schlPubRulListGuide`
- [학칙ㆍ공단ㆍ공공기관 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=schlPubRulInfoGuide) - `schlPubRulInfoGuide`

### 12. 법령용어 (Legal Terms) - 2개 API
- [법령 용어 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsTrmListGuide) - `lsTrmListGuide`
- [법령 용어 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsTrmInfoGuide) - `lsTrmInfoGuide`

### 13. 모바일 (Mobile) - 12개 API

#### 13.1 법령 (2개)
- [법령 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobLsListGuide) - `mobLsListGuide`
- [법령 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobLsInfoGuide) - `mobLsInfoGuide`

#### 13.2 행정규칙 (2개)
- [행정규칙 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobAdmrulListguide) - `mobAdmrulListguide`
- [행정규칙 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobAdmrulInfoGuide) - `mobAdmrulInfoGuide`

#### 13.3 자치법규 (2개)
- [자치법규 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobOrdinListGuide) - `mobOrdinListGuide`
- [자치법규 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobOrdinInfoGuide) - `mobOrdinInfoGuide`

#### 13.4 판례 (2개)
- [판례 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobPrecListGuide) - `mobPrecListGuide`
- [판례 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobPrecInfoGuide) - `mobPrecInfoGuide`

#### 13.5 헌재결정례 (2개)
- [헌재결정례 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobDetcListGuide) - `mobDetcListGuide`
- [헌재결정례 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobDetcInfoGuide) - `mobDetcInfoGuide`

#### 13.6 법령해석례 (2개)
- [법령해석례 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobExpcListGuide) - `mobExpcListGuide`
- [법령해석례 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobExpcInfoGuide) - `mobExpcInfoGuide`

#### 13.7 행정심판례 (2개)
- [행정심판례 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobDeccListGuide) - `mobDeccListGuide`
- [행정심판례 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobDeccInfoGuide) - `mobDeccInfoGuide`

#### 13.8 조약 (2개)
- [조약 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobTrtyListGuide) - `mobTrtyListGuide`
- [조약 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobTrtyInfoGuide) - `mobTrtyInfoGuide`

#### 13.9 별표ㆍ서식 (3개)
- [법령 별표ㆍ서식 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobLsBylListGuide) - `mobLsBylListGuide`
- [행정규칙 별표ㆍ서식 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobAdmrulBylListGuide) - `mobAdmrulBylListGuide`
- [자치법규 별표ㆍ서식 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobOrdinBylListGuide) - `mobOrdinBylListGuide`

#### 13.10 법령 용어 (1개)
- [법령 용어 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=mobLsTrmListGuide) - `mobLsTrmListGuide`

### 14. 맞춤형 (Customized) - 6개 API

#### 14.1 법령 (2개)
- [맞춤형 법령 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=custLsListGuide) - `custLsListGuide`
- [맞춤형 법령 조문 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=custLsJoListGuide) - `custLsJoListGuide`

#### 14.2 행정규칙 (2개)
- [맞춤형 행정규칙 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=custAdmrulListGuide) - `custAdmrulListGuide`
- [맞춤형 행정규칙 조문 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=custAdmrulJoListGuide) - `custAdmrulJoListGuide`

#### 14.3 자치법규 (2개)
- [맞춤형 자치법규 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=custOrdinListGuide) - `custOrdinListGuide`
- [맞춤형 자치법규 조문 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=custOrdinJoListGuide) - `custOrdinJoListGuide`

### 15. 법령정보지식베이스 (Legal Information Knowledge Base) - 7개 API

#### 15.1 용어 (2개)
- [법령용어 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lstrmAIGuide) - `lstrmAIGuide`
- [일상용어 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=dlytrmGuide) - `dlytrmGuide`

#### 15.2 용어 간 관계 (2개)
- [법령용어-일상용어 연계 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lstrmRltGuide) - `lstrmRltGuide`
- [일상용어-법령용어 연계 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=dlytrmRltGuide) - `dlytrmRltGuide`

#### 15.3 조문 간 관계 (2개)
- [법령용어-조문 연계 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lstrmRltJoGuide) - `lstrmRltJoGuide`
- [조문-법령용어 연계 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=joRltLstrmGuide) - `joRltLstrmGuide`

#### 15.4 법령 간 관계 (1개)
- [관련법령 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=lsRltGuide) - `lsRltGuide`

### 16. 중앙부처 1차 해석 (Central Government First Interpretation) - 15개 API

#### 16.1 고용노동부 (2개)
- [고용노동부 법령해석 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=cgmExpcMoelListGuide) - `cgmExpcMoelListGuide`
- [고용노동부 법령해석 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=cgmExpcMoelInfoGuide) - `cgmExpcMoelInfoGuide`

#### 16.2 국토교통부 (2개)
- [국토교통부 법령해석 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=cgmExpcMolitListGuide) - `cgmExpcMolitListGuide`
- [국토교통부 법령해석 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=cgmExpcMolitInfoGuide) - `cgmExpcMolitInfoGuide`

#### 16.3 기획재정부 (1개)
- [기획재정부 법령해석 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=cgmExpcMoefListGuide) - `cgmExpcMoefListGuide`

#### 16.4 해양수산부 (2개)
- [해양수산부 법령해석 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=cgmExpcMofListGuide) - `cgmExpcMofListGuide`
- [해양수산부 법령해석 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=cgmExpcMofInfoGuide) - `cgmExpcMofInfoGuide`

#### 16.5 행정안전부 (2개)
- [행정안전부 법령해석 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=cgmExpcMoisListGuide) - `cgmExpcMoisListGuide`
- [행정안전부 법령해석 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=cgmExpcMoisInfoGuide) - `cgmExpcMoisInfoGuide`

#### 16.6 환경부 (2개)
- [환경부 법령해석 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=cgmExpcMeListGuide) - `cgmExpcMeListGuide`
- [환경부 법령해석 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=cgmExpcMeInfoGuide) - `cgmExpcMeInfoGuide`

#### 16.7 관세청 (2개)
- [관세청 법령해석 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=cgmExpcKcsListGuide) - `cgmExpcKcsListGuide`
- [관세청 법령해석 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=cgmExpcKcsInfoGuide) - `cgmExpcKcsInfoGuide`

#### 16.8 국세청 (1개)
- [국세청 법령해석 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=cgmExpcNtsListGuide) - `cgmExpcNtsListGuide`

### 17. 특별행정심판 (Special Administrative Appeals) - 4개 API

#### 17.1 조세심판원 (2개)
- [조세심판원 특별행정심판례 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=specialDeccTtListGuide) - `specialDeccTtListGuide`
- [조세심판원 특별행정심판례 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=specialDeccTtInfoGuide) - `specialDeccTtInfoGuide`

#### 17.2 해양안전심판원 (2개)
- [해양안전심판원 특별행정심판례 목록 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=specialDeccKmstListGuide) - `specialDeccKmstListGuide`
- [해양안전심판원 특별행정심판례 본문 조회](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=specialDeccKmstInfoGuide) - `specialDeccKmstInfoGuide`

---

## API 통계 요약

| 카테고리 | API 개수 | 비고 |
|---------|---------|------|
| 법령 | 18개 | 본문, 조항호목, 영문법령, 이력, 연계, 부가서비스 |
| 행정규칙 | 4개 | 본문, 신구법 비교 |
| 자치법규 | 3개 | 본문, 연계 |
| 판례 | 2개 | 목록, 본문 |
| 헌재결정례 | 2개 | 목록, 본문 |
| 법령해석례 | 2개 | 목록, 본문 |
| 행정심판례 | 2개 | 목록, 본문 |
| 위원회결정문 | 24개 | 12개 위원회 × 2개 API |
| 조약 | 2개 | 목록, 본문 |
| 별표ㆍ서식 | 3개 | 법령, 행정규칙, 자치법규 |
| 학칙ㆍ공단ㆍ공공기관 | 2개 | 목록, 본문 |
| 법령용어 | 2개 | 목록, 본문 |
| 모바일 | 12개 | 각 카테고리별 모바일 최적화 |
| 맞춤형 | 6개 | 법령, 행정규칙, 자치법규 |
| 법령정보지식베이스 | 7개 | 용어, 관계, 연계 |
| 중앙부처 1차 해석 | 15개 | 8개 부처별 해석 |
| 특별행정심판 | 4개 | 조세심판원, 해양안전심판원 |
| **총계** | **120개** | |

---

## 현재 개발 상태

### ✅ 완료된 API 문서들 (43개)

#### 법령 관련 (8개)
- 현행법령(시행일) 목록/본문 조회
- 현행법령(공포일) 목록/본문 조회
- 법령 연혁 목록/본문 조회
- 영문법령 목록/본문 조회

#### 판례 관련 (2개)
- 판례 목록/본문 조회

#### 행정규칙 관련 (2개)
- 행정규칙 목록/본문 조회

#### 자치법규 관련 (2개)
- 자치법규 목록/본문 조회

#### 헌재결정례 관련 (2개)
- 헌재결정례 목록/본문 조회

#### 법령해석례 관련 (2개)
- 법령해석례 목록/본문 조회

#### 행정심판례 관련 (2개)
- 행정심판례 목록/본문 조회

#### 위원회결정문 관련 (22개)
- 11개 위원회 × 2개 API (목록/본문)

#### 기타 관련 (1개)
- 조약 목록/본문 조회

### 📝 추가 개발 예정 (77개)

#### 우선순위 1 (핵심 API) - 20개
- 법령 조항호목 관련 API (2개)
- 법령 이력 관련 API (3개)
- 법령 연계 관련 API (3개)
- 법령 부가서비스 API (10개)
- 행정규칙 신구법 비교 API (2개)

#### 우선순위 2 (확장 API) - 30개
- 별표ㆍ서식 관련 API (3개)
- 학칙ㆍ공단ㆍ공공기관 API (2개)
- 법령용어 API (2개)
- 모바일 API (12개)
- 맞춤형 API (6개)
- 법령정보지식베이스 API (7개)

#### 우선순위 3 (전문 API) - 27개
- 중앙부처 1차 해석 API (15개)
- 특별행정심판 API (4개)
- 추가 위원회결정문 API (8개)

---

## 공통 정보

### 기본 정보
- **서비스명**: 국가법령정보 공동활용 LAW OPEN DATA
- **제공기관**: 법제처
- **기본 URL**: `http://www.law.go.kr/DRF/`
- **인증**: OC (사용자 이메일 ID) 필수
- **응답 형식**: HTML, XML, JSON
- **문자 인코딩**: UTF-8

### 공통 파라미터
- `OC` (필수): 사용자 이메일의 ID 부분 (예: g4c@korea.kr → OC=g4c)
- `target` (필수): 서비스 대상
- `type` (필수): 출력 형태 (HTML/XML/JSON)

### 사용 제한사항
- 상업적 이용 제한
- 트래픽 제한
- API 키 보안 관리 필수

### 연락처
- **사용신청 문의**: 044-200-6786
- **이용 문의**: 02-2109-6446
- **이메일**: lawmanager@korea.kr
- **주소**: (30102) 세종특별자치시 도움5로 20, 정부세종청사 법제처

## 참고사항
- 체계도 등 부가서비스는 법령서비스 신청을 하면 추가신청 없이 이용 가능
- 모든 API는 HTML, XML, JSON 형식으로 응답 제공
- 법령서비스 신청 후 이용 가능한 서비스와 추가 신청이 필요한 서비스 구분 필요
- 모바일 API는 별도 엔드포인트 제공
- 맞춤형 서비스는 사용자별 맞춤 설정 가능

---

*이 문서는 국가법령정보센터 LAW OPEN API 가이드를 기반으로 작성되었습니다.*
*최신 정보는 [공식 사이트](https://open.law.go.kr/LSO/openApi/guideList.do)를 참조하시기 바랍니다.*
