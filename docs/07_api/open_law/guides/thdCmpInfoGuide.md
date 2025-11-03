# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=thdCmp

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : thdCmp(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| knd | int(필수) | 3단비교 종류별 검색인용조문 : 1 / 위임조문 : 2 |
| ID | char | 법령 ID (ID 또는 MST 중 하나는 반드시 입력) |
| MST | char | 법령 마스터 번호 - 법령테이블의 lsi_seq 값을 의미함 |
| LM | string | 법령의 법령명(법령명 입력시 해당 법령 링크) |
| LD | int | 법령의 공포일자 |
| LN | int | 법령의 공포번호 |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=thdCmp&ID=001372&MST=162662&type=HTML
2. http://www.law.go.kr/DRF/lawService.do?OC=test&target=thdCmp&ID=001570&type=HTML
3. http://www.law.go.kr/DRF/lawService.do?OC=test&target=thdCmp&MST=236231&knd=1&type=XML
4. http://www.law.go.kr/DRF/lawService.do?OC=test&target=thdCmp&MST=222549&knd=2&type=XML
5. http://www.law.go.kr/DRF/lawService.do?OC=test&target=thdCmp&MST=222549&knd=2&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 기본정보 | string | 인용 삼단비교 기본정보 |
| 법령ID | int | 법령 ID |
| 시행령ID | int | 시행령 ID |
| 시행규칙ID | int | 시행규칙 ID |
| 법령명 | string | 법령 명 |
| 시행령명 | string | 법령시행령 명 |
| 시행규칙명 | string | 시행규칙 명 |
| 법령요약정보 | string | 법령 요약정보 |
| 시행령요약정보 | string | 시행령 요약정보 |
| 시행규칙요약정보 | string | 시행규칙 요약정보 |
| 삼단비교기준 | string | 삼단비교 기준 |
| 삼단비교존재여부 | int | 삼단비교 존재하지 않으면 N이 조회. |
| 시행일자 | int | 시행일자 |
| 관련삼단비교목록 | string | 관련 삼단비교 목록 |
| 목록명 | string | 목록명 |
| 삼단비교목록상세링크 | string | 인용조문 삼단비교 목록 상세링크 |
| 인용조문삼단비교 | string | 인용조문 삼단비교 |
| 법률조문 | string | 법률조문 |
| 조번호 | int | 조번호 |
| 조가지번호 | int | 조가지번호 |
| 조제목 | string | 조제목 |
| 조내용 | string | 조내용 |
| 시행령조문목록 | string | 시행령조문목록 |
| 시행령조문 | string | 하위 시행령조문 |
| 시행규칙조문목록 | string | 시행규칙조문목록 |
| 시행규칙조문 | string | 하위 시행규칙조문 |
| 위임행정규칙목록 | string | 위임행정규칙목록 |
| 위임행정규칙 | string | 위임행정규칙 |
| 위임행정규칙명 | string | 위임행정규칙명 |
| 위임행정규칙일련번호 | int | 위임행정규칙일련번호 |
| 위임행정규칙조번호 | int | 위임행정규칙조번호 |
| 위임행정규칙조가지번호 | int | 위임행정규칙조가지번호 |
