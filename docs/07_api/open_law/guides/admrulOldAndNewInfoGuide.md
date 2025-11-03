# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=admrulOldAndNew

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : admrulOldAndNew(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| ID | char | 행정규칙 일련번호 (ID 또는 LID 중 하나는 반드시 입력) |
| LID | char | 행정규칙 ID (ID 또는 LID 중 하나는 반드시 입력) |
| LM | string | 행정규칙명 조회하고자 하는 정확한 행정규칙명을 입력 |

샘플 URL

1. http://law.go.kr/DRF/lawService.do?OC=test&target=admrulOldAndNew&ID=2100000248758&type=HTML
2. http://law.go.kr/DRF/lawService.do?OC=test&target=admrulOldAndNew&ID=2100000248758&type=XML
3. http://law.go.kr/DRF/lawService.do?OC=test&target=admrulOldAndNew&ID=2100000248758&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 구조문_기본정보 | string | 구조문_기본정보 |
| 행정규칙일련번호 | int | 행정규칙일련번호 |
| 행정규칙ID | int | 행정규칙ID |
| 시행일자 | int | 시행일자 |
| 발령일자 | int | 발령일자 |
| 발령번호 | int | 발령번호 |
| 현행여부 | string | 현행여부 |
| 제개정구분명 | string | 제개정구분명 |
| 행정규칙명 | string | 행정규칙명 |
| 행정규칙종류 | string | 행정규칙종류 |
| 신조문_기본정보 | string | 구조문과 동일한 기본 정보 들어가 있음. |
| 구조문목록 | string | 구조문목록 |
| 조문 | string | 조문 |
| 신조문목록 | string | 신조문목록 |
| 조문 | string | 조문 |
| 신구법존재여부 | string | 신구법이 존재하지 않을 경우 N이 조회. |
