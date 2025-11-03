# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=oldAndNew

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : oldAndNew(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| ID | char | 법령 ID (ID 또는 MST 중 하나는 반드시 입력) |
| MST | char | 법령 마스터 번호 - 법령테이블의 lsi_seq 값을 의미함 |
| LM | string | 법령의 법령명(법령명 입력시 해당 법령 링크) |
| LD | int | 법령의 공포일자 |
| LN | int | 법령의 공포번호 |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=oldAndNew&ID=000170&MST=122682&type=HTML
2. http://www.law.go.kr/DRF/lawService.do?OC=test&target=oldAndNew&MST=136931&type=HTML
3. http://www.law.go.kr/DRF/lawService.do?OC=test&target=oldAndNew&MST=122682&type=XML
4. http://www.law.go.kr/DRF/lawService.do?OC=test&target=oldAndNew&MST=122682&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 구조문_기본정보 | string | 구조문_기본정보 |
| 법령일련번호 | int | 법령일련번호 |
| 법령ID | int | 법령ID |
| 시행일자 | int | 시행일자 |
| 공포일자 | int | 공포일자 |
| 공포번호 | int | 공포번호 |
| 현행여부 | string | 현행여부 |
| 제개정구분명 | string | 제개정구분명 |
| 법령명 | string | 법령 |
| 법종구분 | string | 법종구분 |
| 신조문_기본정보 | string | 구조문과 동일한 기본 정보 들어가 있음. |
| 구조문목록 | string | 구조문목록 |
| 조문 | string | 조문 |
| 신조문목록 | string | 신조문목록 |
| 조문 | string | 조문 |
| 신구법존재여부 | string | 신구법이 존재하지 않을 경우 N이 조회. |
