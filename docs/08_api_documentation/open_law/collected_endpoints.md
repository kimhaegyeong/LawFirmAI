## Open Law API 수집 문서 (우선 8종)

출처: `openApiGuide('<GUIDE_ID>')` 상세페이지. 항목별 파라미터/샘플/응답필드는 가이드 상세에서 확인 후 아래 양식에 기입.

---

### 수집 우선순위(챗봇 성능 대비)

1) lsEfYdListGuide + lsEfYdJoListGuide: 메타 + 조문 텍스트(임베딩 단위)
2) lsEfYdInfoGuide: 법령 레벨 메타(전문은 선택)
3) precListGuide + precInfoGuide: 판례 전문/요지(문단 단위 임베딩 권장)
4) expcListGuide + expcInfoGuide: 법령해석례(질의요지/회신내용 중심, 문단 단위 임베딩 권장)
5) 용어/약칭: 질의 정규화·확장 품질 향상
6) 이력/관계: 시점 질의/탐색 고도화 시 추가

---

### 공통 템플릿(복붙용)

- GUIDE_ID: <GUIDE_ID>
- 분류/항목명: <분류 / 가이드 표시명>
- 상세 링크: openApiGuide('<GUIDE_ID>')

- 요청
  - Base URL: <요청 URL>
  - Method: GET
  - 필수 파라미터: | 이름 | 타입/형식 | 설명 | 허용값/비고 |
  - 선택 파라미터: | 이름 | 타입/형식 | 설명 | 기본값/허용값 |
  - 샘플 URL: (XML/JSON 각각 최소 1개, 대표 검색 케이스 포함)

- 응답
  - 포맷: XML/JSON
  - 구조: <루트/리스트/아이템 요약>
  - 주요 필드: | 필드 | 타입 | 설명 | 예시 |
  - 페이징/카운트: totalCnt, page, display 등 여부

- 제한/주의
  - 최대 display, 정렬 옵션, 코드값(필요 시 표로 추가)

- 관련 GUIDE_ID
  - 목록↔본문, 목록↔조항 등 상호 참조

---

## 1) 법령 본문/목록

### 1-1. lsEfYdListGuide — 현행법령(시행일) 목록 조회

- GUIDE_ID: lsEfYdListGuide
- 분류/항목명: 법령·본문 / 현행법령(시행일) 목록 조회
- 상세 링크: openApiGuide('lsEfYdListGuide')

- 요청
  - Base URL: http://www.law.go.kr/DRF/lawSearch.do?target=eflaw
  - Method: GET
  - 필수 파라미터
    | 이름 | 타입/형식 | 설명 | 허용값/비고 |
    | --- | --- | --- | --- |
    | OC | string | 사용자 이메일 ID | 예: g4c@korea.kr → OC=g4c |
    | target | string | 서비스 대상 | eflaw |
    | type | char | 출력 형태 | HTML/XML/JSON (기본 XML) |
  - 선택 파라미터
    | 이름 | 타입/형식 | 설명 | 기본값/허용값 |
    | --- | --- | --- | --- |
    | search | int | 검색범위 | 1: 법령명(기본), 2: 본문 |
    | query | string | 법령명 질의 | 예: "자동차" |
    | nw | int | 연혁 상태 | 1:연혁, 2:시행예정, 3:현행 (복수 조합 가능) |
    | LID | string | 법령ID | 예: 830 |
    | display | int | 페이지당 개수 | 기본 20, 최대 100 |
    | page | int | 페이지 번호 | 기본 1 |
    | sort | string | 정렬 옵션 | lasc(기본), ldes, dasc, ddes, nasc, ndes, efasc, efdes |
    | efYd | string | 시행일자 범위 | 예: 20090101~20090130 |
    | date | string | 공포일자 |  |
    | ancYd | string | 공포일자 범위 | 예: 20090101~20090130 |
    | ancNo | string | 공포번호 범위 | 예: 306~400 |
    | rrClsCd | string | 제·개정 종류 | 300201 제정, 300202 일부개정, ... |
    | nb | int | 공포번호 |  |
    | org | string | 소관부처 코드 | 예: 1613000 |
    | knd | string | 법령종류 코드 |  |
    | gana | string | 사전식 검색 | ga, na, da 등 |
    | popYn | string | 팝업 여부 | Y 시 팝업 |
  - 샘플 URL
    - XML: http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=eflaw&type=XML
    - JSON: http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=eflaw&type=JSON
    - 정렬: ...&sort=ddes
    - 부처: ...&org=1613000
    - 법령ID: ...&LID=830

- 응답
  - 포맷: XML/JSON
  - 주요 필드(발췌)
    | 필드 | 타입 | 설명 | 예시 |
    | --- | --- | --- | --- |
    | target | string | 검색서비스 대상 | eflaw |
    | 키워드 | string | 검색어 |  |
    | section | string | 검색범위 | 1/2 |
    | totalCnt | int | 검색건수 | 123 |
    | page | int | 결과페이지번호 | 1 |
    | 법령일련번호 | int | 법령 일련번호 |  |
    | 현행연혁코드 | string | 현행연혁코드 |  |
    | 법령명한글 | string | 법령명(한글) |  |
    | 법령약칭명 | string | 법령 약칭 |  |
    | 법령ID | int | 법령 ID | 830 |
    | 공포일자 | int | 공포일자 | 20200101 |
    | 공포번호 | int | 공포번호 | 12345 |
    | 제개정구분명 | string | 제·개정 구분 | 일부개정 |
    | 소관부처코드 | string | 소관부처 코드 | 1613000 |
    | 소관부처명 | string | 소관부처명 | 국토교통부 |
    | 법령구분명 | string | 법령 구분 | 법률 등 |
    | 시행일자 | int | 시행일자 | 20210101 |
    | 자법타법여부 | string | 자법/타법 여부 |  |
    | 법령상세링크 | string | 상세 링크 |  |

- 관련 GUIDE_ID
  - 본문 조회: lsEfYdInfoGuide
  - 조항 조회: lsEfYdJoListGuide

---

### 1-2. lsEfYdInfoGuide — 현행법령(시행일) 본문 조회

- GUIDE_ID: lsEfYdInfoGuide
- 분류/항목명: 법령·본문 / 현행법령(시행일) 본문 조회
- 상세 링크: openApiGuide('lsEfYdInfoGuide')

- 요청
  - Base URL: http://www.law.go.kr/DRF/lawService.do?target=eflaw
  - Method: GET
  - 필수 파라미터
    | 이름 | 타입/형식 | 설명 | 허용값/비고 |
    | --- | --- | --- | --- |
    | OC | string | 사용자 이메일 ID | 예: g4c@korea.kr → OC=g4c |
    | target | string | 서비스 대상 | eflaw |
    | type | char | 출력 형태 | HTML/XML/JSON (기본 XML) |
    | efYd | int | 법령의 시행일자 | ID 입력 시 미사용 |
  - 선택 파라미터
    | 이름 | 타입/형식 | 설명 | 기본값/허용값 |
    | --- | --- | --- | --- |
    | ID | char | 법령 ID | ID 또는 MST 중 하나 필수, ID로 조회 시 해당 법령의 현행 본문 |
    | MST | char | 법령 마스터번호(lsi_seq) | ID 또는 MST 중 하나 필수 |
    | JO | int/char(6) | 조번호(6자리: 조번호 4 + 조가지 2) | 예: 000200(제2조), 001002(제10조의2). 미입력 시 전체 조 |
    | chrClsCd | char | 원문/한글 여부 | 기본 한글(010202), 원문(010201) |
  - 샘플 URL
    - 자동차관리법 ID HTML 상세조회: http://www.law.go.kr/DRF/lawService.do?OC=test&target=eflaw&ID=1747&type=HTML
    - 자동차관리법 법령 Seq XML 조회: http://www.law.go.kr/DRF/lawService.do?OC=test&target=eflaw&MST=166520&efYd=20151007&type=XML
    - 자동차관리법 3조 XML 상세조회: http://www.law.go.kr/DRF/lawService.do?OC=test&target=eflaw&MST=166520&efYd=20151007&JO=000300&type=XML
    - 자동차관리법 ID JSON 상세조회: http://www.law.go.kr/DRF/lawService.do?OC=test&target=eflaw&ID=1747&type=JSON

- 응답
  - 포맷: XML/JSON
  - 주요 필드(발췌)
    | 필드 | 타입 | 설명 |
    | --- | --- | --- |
    | 법령ID | int | 법령ID |
    | 공포일자 | int | 공포일자 |
    | 공포번호 | int | 공포번호 |
    | 언어 | string | 언어종류 |
    | 법종구분 | string | 법종류의 구분 |
    | 법종구분코드 | string | 법종구분코드 |
    | 법령명_한글 | string | 한글법령명 |
    | 법령명_한자 | string | 법령명(한자) |
    | 법령명약칭 | string | 법령명 약칭 |
    | 편장절관 | int | 편장절관 일련번호 |
    | 소관부처코드 | int | 소관부처코드 |
    | 소관부처 | string | 소관부처명 |
    | 전화번호 | string | 전화번호 |
    | 시행일자 | int | 시행일자 |
    | 제개정구분 | string | 제개정 구분 |
    | 조문시행일자문자열 | string | 조문 시행일자 문자열 |
    | 별표시행일자문자열 | string | 별표 시행일자 문자열 |
    | 별표편집여부 | string | 별표 편집 여부 |
    | 공포법령여부 | string | 공포법령 여부 |
    | 소관부처명 | string | 소관부처명 |
    | 소관부처코드 | int | 소관부처코드 |
    | 부서명 | string | 연락부서명 |
    | 부서연락처 | string | 연락부서 전화번호 |
    | 공동부령구분 | string | 공동부령의 구분 |
    | 구분코드 | string | 공동부령 구분코드 |
    | 공포번호(공동부령) | string | 공동부령의 공포번호 |
    | 조문번호 | int | 조문번호 |
    | 조문가지번호 | int | 조문 가지번호 |
    | 조문여부 | string | 조문 여부 |
    | 조문제목 | string | 조문 제목 |
    | 조문시행일자 | int | 조문 시행일자 |
    | 조문제개정유형 | string | 조문 제개정 유형 |
    | 조문이동이전 | int | 조문 이동 이전 |
    | 조문이동이후 | int | 조문 이동 이후 |
    | 조문변경여부 | string | 해당 조문 내 변경 내용 존재 여부 |
    | 조문내용 | string | 조문 내용 |
    | 항번호 | int | 항 번호 |
    | 항제개정유형 | string | 항 제개정 유형 |
    | 항제개정일자문자열 | string | 항 제개정 일자 문자열 |
    | 항내용 | string | 항 내용 |
    | 호번호 | int | 호 번호 |
    | 호내용 | string | 호 내용 |
    | 조문참고자료 | string | 조문 참고자료 |
    | 부칙공포일자 | int | 부칙 공포일자 |
    | 부칙공포번호 | int | 부칙 공포번호 |
    | 부칙내용 | string | 부칙 내용 |
    | 별표번호 | int | 별표 번호 |
    | 별표가지번호 | int | 별표 가지번호 |
    | 별표구분 | string | 별표 구분 |
    | 별표제목 | string | 별표 제목 |
    | 별표제목문자열 | string | 별표 제목 문자열 |
    | 별표시행일자 | int | 별표 시행일자 |
    | 별표서식파일링크 | string | 별표 서식 파일 링크 |
    | 별표HWP파일명 | string | 별표 HWP 파일명 |
    | 별표서식PDF파일링크 | string | 별표 서식 PDF 파일 링크 |
    | 별표PDF파일명 | string | 별표 PDF 파일명 |
    | 별표이미지파일명 | string | 별표 이미지 파일명 |
    | 별표내용 | string | 별표 내용 |
    | 개정문내용 | string | 개정문 내용 |
    | 제개정이유내용 | string | 제개정 이유 내용 |

- 관련 GUIDE_ID
  - 목록: lsEfYdListGuide
  - 조항: lsEfYdJoListGuide

---

### 1-3. lsNwListGuide — 현행법령(공포일) 목록 조회

- GUIDE_ID: lsNwListGuide
- 분류/항목명: 법령·본문 / 현행법령(공포일) 목록 조회
- 상세 링크: openApiGuide('lsNwListGuide')

- 요청
  - Base URL: http://www.law.go.kr/DRF/lawSearch.do?target=nwlaw
  - Method: GET
  - 필수 파라미터
    | 이름 | 타입/형식 | 설명 | 허용값/비고 |
    | --- | --- | --- | --- |
    | OC | string | 사용자 이메일 ID | 예: g4c@korea.kr → OC=g4c |
    | target | string | 서비스 대상 | nwlaw |
    | type | char | 출력 형태 | HTML/XML/JSON (기본 XML) |
  - 선택 파라미터
    | 이름 | 타입/형식 | 설명 | 기본값/허용값 |
    | --- | --- | --- | --- |
    | search | int | 검색범위 | 1: 법령명(기본), 2: 본문 |
    | query | string | 법령명 질의 | 예: "자동차" |
    | LID | string | 법령ID | 예: 830 |
    | display | int | 페이지당 개수 | 기본 20, 최대 100 |
    | page | int | 페이지 번호 | 기본 1 |
    | sort | string | 정렬 옵션 | lasc, ldes, dasc, ddes, nasc, ndes |
    | date | string | 공포일자(YYYYMMDD) | |
    | ancYd | string | 공포일자 범위 | 예: 20090101~20090130 |
    | ancNo | string | 공포번호 범위 | 예: 306~400 |
    | rrClsCd | string | 제·개정 종류 | 코드값 참조 |
    | nb | int | 공포번호 | |
    | org | string | 소관부처 코드 | |
    | knd | string | 법령종류 코드 | |
    | gana | string | 사전식 검색 | ga, na, da 등 |
    | popYn | string | 팝업 여부 | Y 시 팝업 |
  - 샘플 URL
    - XML: http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=nwlaw&type=XML
    - JSON: http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=nwlaw&type=JSON
    - 공포일 범위: ...&ancYd=20200101~20201231
    - 정렬: ...&sort=ddes

- 관련 GUIDE_ID
  - 본문: lsNwInfoGuide
  - 조항: lsNwJoListGuide

---

### 1-4. lsNwInfoGuide — 현행법령(공포일) 본문 조회

- GUIDE_ID: lsNwInfoGuide
- 분류/항목명: 법령·본문 / 현행법령(공포일) 본문 조회
- 상세 링크: openApiGuide('lsNwInfoGuide')

- 요청
  - Base URL: http://www.law.go.kr/DRF/lawInfo.do?target=nwlaw
  - Method: GET
  - 필수 파라미터
    | 이름 | 타입/형식 | 설명 | 허용값/비고 |
    | --- | --- | --- | --- |
    | OC | string | 사용자 이메일 ID | 예: g4c@korea.kr → OC=g4c |
    | target | string | 서비스 대상 | nwlaw |
    | type | char | 출력 형태 | HTML/XML/JSON (기본 XML) |
    | LID | string | 법령ID | 목록 응답의 법령ID |
  - 선택 파라미터
    | 이름 | 타입/형식 | 설명 | 기본값/허용값 |
    | --- | --- | --- | --- |
    | date | string | 공포일자 | 예: 20200101 |
  - 샘플 URL
    - XML: http://www.law.go.kr/DRF/lawInfo.do?OC=test&target=nwlaw&type=XML&LID=830
    - JSON: http://www.law.go.kr/DRF/lawInfo.do?OC=test&target=nwlaw&type=JSON&LID=830
    - 공포일 지정: ...&date=20200101

- 응답
  - 포맷: XML/JSON
  - 구조/주요 필드: lsEfYdInfoGuide와 유사 (공포일 기준 메타 + 본문 전문)

- 관련 GUIDE_ID
  - 목록: lsNwListGuide
  - 조항: lsNwJoListGuide

---

## 2) 조항/호/목

### 2-1. lsEfYdJoListGuide — 현행법령(시행일) 본문 조항호목 상세 조회

- GUIDE_ID: lsEfYdJoListGuide
- 분류/항목명: 법령·조항호목 / 현행법령(시행일) 본문 조항호목 조회
- 상세 링크: openApiGuide('lsEfYdJoListGuide')

- 요청
  - Base URL: http://www.law.go.kr/DRF/lawService.do?target=eflawjosub
  - Method: GET
  - 필수 파라미터
    | 이름 | 타입/형식 | 설명 | 허용값/비고 |
    | --- | --- | --- | --- |
    | OC | string | 사용자 이메일 ID | 예: g4c@korea.kr → OC=g4c |
    | target | string | 서비스 대상 | eflawjosub |
    | type | char | 출력 형태 | HTML/XML/JSON (기본 XML) |
    | efYd | int | 법령 시행일자 | ID 입력 시 생략 |
    | JO | char | 조 번호(6자리) | 예: 제2조 → 000200 |
  - 선택 파라미터
    | 이름 | 타입/형식 | 설명 | 기본값/허용값 |
    | --- | --- | --- | --- |
    | ID | char | 법령ID | ID 또는 MST 중 1개 필수 |
    | MST | char | 법령 마스터번호(lsi_seq) | ID 또는 MST 중 1개 필수 |
    | HANG | char | 항 번호(6자리) | 예: 제1항 → 000100 |
    | HO | char | 호 번호(6자리) | 예: 제2호 → 000200 |
    | MOK | char | 목(한 글자) | 예: 가/나/다 등(UTF-8 인코딩) |
  - 샘플 URL
    - XML: http://www.law.go.kr/DRF/lawService.do?OC=test&target=eflawjosub&type=XML&MST=193412&efYd=20171019&JO=000300&HANG=000100&HO=000200&MOK=%EB%8B%A4
    - HTML: http://www.law.go.kr/DRF/lawService.do?OC=test&target=eflawjosub&type=HTML&MST=193412&efYd=20171019&JO=000300&HANG=000100&HO=000200&MOK=%EB%8B%A4
    - JSON: http://www.law.go.kr/DRF/lawService.do?OC=test&target=eflawjosub&type=JSON&MST=193412&efYd=20171019&JO=000300&HANG=000100&HO=000200&MOK=%EB%8B%A4

- 응답
  - 포맷: XML/JSON
  - 구조/주요 필드(발췌)
    | 필드 | 타입 | 설명 |
    | --- | --- | --- |
    | 법령키 | int | 법령키 |
    | 법령ID | int | 법령ID |
    | 시행일자 | int | 시행일자 |
    | 법종구분 | string | 법종구분/코드 |
    | 법령명_한글 | string | 법령명(한글) |
    | 조문번호 | int | 조문번호(6자리) |
    | 조문제목 | string | 조문제목 |
    | 조문내용 | string | 조문 본문 |
    | 항번호/항내용 | int/string | 항 상세 |
    | 호번호/호내용 | int/string | 호 상세 |
    | 목번호/목내용 | string/string | 목 상세 |

- 관련 GUIDE_ID
  - 목록: lsEfYdListGuide
  - 본문: lsEfYdInfoGuide

---

### 2-2. lsNwJoListGuide — 현행법령(공포일) 본문 조항호목 조회

- GUIDE_ID: lsNwJoListGuide
- 분류/항목명: 법령·조항호목 / 현행법령(공포일) 본문 조항호목 조회
- 상세 링크: openApiGuide('lsNwJoListGuide')

- 요청
  - Base URL: http://www.law.go.kr/DRF/joSearch.do?target=nwjo
  - Method: GET
  - 필수 파라미터
    | 이름 | 타입/형식 | 설명 | 허용값/비고 |
    | --- | --- | --- | --- |
    | OC | string | 사용자 이메일 ID | 예: g4c@korea.kr → OC=g4c |
    | target | string | 서비스 대상 | nwjo |
    | type | char | 출력 형태 | HTML/XML/JSON (기본 XML) |
    | LID | string | 법령ID | 목록/본문 응답의 법령ID |
  - 선택 파라미터
    | 이름 | 타입/형식 | 설명 | 기본값/허용값 |
    | --- | --- | --- | --- |
    | date | string | 공포일자 | 예: 20200101 |
    | joNo | string | 조문번호 필터 | 예: 제1조 |
    | include | string | 포함 범위 | article/paragraph/subparagraph/item 등 |
    | display | int | 페이지당 개수 | 기본 20, 최대 100 |
    | page | int | 페이지 번호 | 기본 1 |
  - 샘플 URL
    - XML: http://www.law.go.kr/DRF/joSearch.do?OC=test&target=nwjo&type=XML&LID=830
    - JSON: http://www.law.go.kr/DRF/joSearch.do?OC=test&target=nwjo&type=JSON&LID=830
    - 공포일 지정: ...&date=20200101

- 응답
  - 포맷: XML/JSON
  - 구조/주요 필드: lsEfYdJoListGuide와 동일 구조(공포일 기준)

- 관련 GUIDE_ID
  - 목록: lsNwListGuide
  - 본문: lsNwInfoGuide

---

## 3) 판례

### 3-1. precListGuide — 판례 목록 조회

- GUIDE_ID: precListGuide
- 분류/항목명: 판례·본문 / 판례 목록 조회
- 상세 링크: openApiGuide('precListGuide')

- 요청
  - Base URL: http://www.law.go.kr/DRF/precSearch.do?target=prec
  - Method: GET
  - 필수 파라미터
    | 이름 | 타입/형식 | 설명 | 허용값/비고 |
    | --- | --- | --- | --- |
    | OC | string | 사용자 이메일 ID | 예: g4c@korea.kr → OC=g4c |
    | target | string | 서비스 대상 | prec |
    | type | char | 출력 형태 | HTML/XML/JSON (기본 XML) |
  - 선택 파라미터(예시)
    | 이름 | 타입/형식 | 설명 | 기본값/허용값 |
    | --- | --- | --- | --- |
    | query | string | 검색어 | 사건명/판시사항 등 |
    | court | string | 법원 | 코드값(대법원/고등법원/지방법원 등) |
    | caseNo | string | 사건번호 | 예: 2020다12345 |
    | decisionYd | string | 선고일자 | YYYYMMDD |
    | caseType | string | 사건종류 | 코드값 |
    | sort | string | 정렬 | 최신/관련도 등(가이드 확인) |
    | display | int | 페이지당 개수 | 기본 20, 최대 100 |
    | page | int | 페이지 번호 | 기본 1 |
  - 샘플 URL
    - XML: http://www.law.go.kr/DRF/precSearch.do?OC=test&target=prec&type=XML&query=손해배상
    - JSON: http://www.law.go.kr/DRF/precSearch.do?OC=test&target=prec&type=JSON&query=손해배상
    - 사건번호: ...&caseNo=2020다12345
    - 선고일자: ...&decisionYd=20200101

- 응답
  - 포맷: XML/JSON
  - 구조/주요 필드(예시)
    | 필드 | 타입 | 설명 |
    | --- | --- | --- |
    | 사건명 | string | 판례 사건명 |
    | 사건번호 | string | 예: 2020다12345 |
    | 선고일자 | string | YYYYMMDD |
    | 법원명 | string | 대법원 등 |
    | 판시사항 | string | 요약 텍스트 |
    | 선고 | string | 판결 결과 요지 |
    | 본문상세링크 | string | 상세 조회 링크 |

- 관련 GUIDE_ID
  - 본문: precInfoGuide

---

## 4) 법령해석례

### 4-1. expcListGuide — 법령해석례 목록 조회

- GUIDE_ID: expcListGuide
- 분류/항목명: 법령해석례·본문 / 법령해석례 목록 조회
- 상세 링크: openApiGuide('expcListGuide')

- 요청
  - Base URL: http://www.law.go.kr/DRF/expcSearch.do?target=expc
  - Method: GET
  - 필수 파라미터
    | 이름 | 타입/형식 | 설명 | 허용값/비고 |
    | --- | --- | --- | --- |
    | OC | string | 사용자 이메일 ID | 예: g4c@korea.kr → OC=g4c |
    | target | string | 서비스 대상 | expc |
    | type | char | 출력 형태 | HTML/XML/JSON (기본 XML) |
  - 선택 파라미터(예시)
    | 이름 | 타입/형식 | 설명 | 기본값/허용값 |
    | --- | --- | --- | --- |
    | query | string | 검색어 | 안건명/질의요지 등 |
    | replyYd | string | 회신일자 | YYYYMMDD |
    | org | string | 소관부처/요청기관 | 코드값(가이드 확인) |
    | sort | string | 정렬 | 최신/관련도 등(가이드 확인) |
    | display | int | 페이지당 개수 | 기본 20, 최대 100 |
    | page | int | 페이지 번호 | 기본 1 |
  - 샘플 URL
    - XML: http://www.law.go.kr/DRF/expcSearch.do?OC=test&target=expc&type=XML&query=겸업금지
    - JSON: http://www.law.go.kr/DRF/expcSearch.do?OC=test&target=expc&type=JSON&query=겸업금지
    - 회신일자: ...&replyYd=20210101

- 응답
  - 포맷: XML/JSON
  - 구조/주요 필드(예시)
    | 필드 | 타입 | 설명 |
    | --- | --- | --- |
    | 안건명 | string | 해석 안건명 |
    | 해석ID | string | 목록 식별자(본문 조회에 사용) |
    | 회신일자 | string | YYYYMMDD |
    | 소관부처명 | string |  |
    | 질의요지 | string | 요약 |
    | 본문상세링크 | string | 상세 조회 링크 |

- 관련 GUIDE_ID
  - 본문: expcInfoGuide

---

### 4-2. expcInfoGuide — 법령해석례 본문 조회

- GUIDE_ID: expcInfoGuide
- 분류/항목명: 법령해석례·본문 / 법령해석례 본문 조회
- 상세 링크: openApiGuide('expcInfoGuide')

- 요청
  - Base URL: http://www.law.go.kr/DRF/expcInfo.do?target=expc
  - Method: GET
  - 필수 파라미터
    | 이름 | 타입/형식 | 설명 | 허용값/비고 |
    | --- | --- | --- | --- |
    | OC | string | 사용자 이메일 ID | 예: g4c@korea.kr → OC=g4c |
    | target | string | 서비스 대상 | expc |
    | type | char | 출력 형태 | HTML/XML/JSON (기본 XML) |
    | expcId | string | 해석례 식별자 | 목록 응답의 식별자(정확 명칭은 가이드 확인) |
  - 선택 파라미터
    | 이름 | 타입/형식 | 설명 | 기본값/허용값 |
    | --- | --- | --- | --- |
    | replyYd | string | 회신일자 | YYYYMMDD |
  - 샘플 URL
    - XML: http://www.law.go.kr/DRF/expcInfo.do?OC=test&target=expc&type=XML&expcId=XXXX
    - JSON: http://www.law.go.kr/DRF/expcInfo.do?OC=test&target=expc&type=JSON&expcId=XXXX

- 응답
  - 포맷: XML/JSON
  - 구조/주요 필드(예시)
    | 필드 | 타입 | 설명 |
    | --- | --- | --- |
    | 안건명 | string | 해석 안건명 |
    | 회신일자 | string | YYYYMMDD |
    | 소관부처명 | string |  |
    | 질의요지 | string | 요약 텍스트 |
    | 회신내용 | string | 본문(핵심 결론) |
    | 관련법령/조문 | string[] | 추출 또는 제공 시 매핑 |

- 관련 GUIDE_ID
  - 목록: expcListGuide

---
### 3-2. precInfoGuide — 판례 본문 조회

- GUIDE_ID: precInfoGuide
- 분류/항목명: 판례·본문 / 판례 본문 조회
- 상세 링크: openApiGuide('precInfoGuide')

- 요청
  - Base URL: http://www.law.go.kr/DRF/lawService.do?target=prec
  - Method: GET
  - 필수 파라미터
    | 이름 | 타입/형식 | 설명 | 허용값/비고 |
    | --- | --- | --- | --- |
    | OC | string | 사용자 이메일 ID | 예: g4c@korea.kr → OC=g4c |
    | target | string | 서비스 대상 | prec |
    | type | char | 출력 형태 | HTML/XML/JSON (기본 XML) |
    | ID | char | 판례 일련번호 | 필수 |
  - 선택 파라미터
    | 이름 | 타입/형식 | 설명 | 기본값/허용값 |
    | --- | --- | --- | --- |
    | LM | string | 판례명 | |
  - 참고
    - 국세청 판례 본문 조회는 HTML만 가능합니다.
  - 샘플 URL
    - HTML: http://www.law.go.kr/DRF/lawService.do?OC=test&target=prec&ID=228541&type=HTML
    - XML: http://www.law.go.kr/DRF/lawService.do?OC=test&target=prec&ID=228541&type=XML
    - JSON: http://www.law.go.kr/DRF/lawService.do?OC=test&target=prec&ID=228541&type=JSON

- 응답
  - 포맷: XML/JSON/HTML
  - 주요 필드(발췌)
    | 필드 | 타입 | 설명 |
    | --- | --- | --- |
    | 판례정보일련번호 | int | 판례정보일련번호 |
    | 사건명 | string | 사건명 |
    | 사건번호 | string | 사건번호 |
    | 선고일자 | int | 선고일자 |
    | 선고 | string | 선고 |
    | 법원명 | string | 법원명 |
    | 법원종류코드 | int | 법원종류코드(대법원:400201, 하위법원:400202) |
    | 사건종류명 | string | 사건종류명 |
    | 사건종류코드 | int | 사건종류코드 |
    | 판결유형 | string | 판결유형 |
    | 판시사항 | string | 판시사항 |
    | 판결요지 | string | 판결요지 |
    | 참조조문 | string | 참조조문 |
    | 참조판례 | string | 참조판례 |
    | 판례내용 | string | 판례내용 |

- 관련 GUIDE_ID
  - 목록: precListGuide

---

## 수집 가이드 (DevTools)

브라우저 콘솔에서 다음을 실행하여 항목별 상세 ID를 확인하고 페이지로 이동:

```
$$('a[onclick*="openApiGuide("]').map(a => a.getAttribute('onclick'))
```

표에 없는 파라미터 코드값/정렬옵션/제한사항은 상세 페이지 하단의 샘플/출력필드 표를 참고하여 보완.
