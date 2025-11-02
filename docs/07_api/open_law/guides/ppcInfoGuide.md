# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : https://www.law.go.kr/DRF/lawService.do?target=ppc

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상(개인정보보호위원회 : ppc) |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| ID | char(필수) | 결정문 일련번호 |

샘플 URL

1. https://www.law.go.kr/DRF/lawService.do?OC=test&target=ppc&ID=5&type=HTML
2. https://www.law.go.kr/DRF/lawService.do?OC=test&target=ppc&ID=3&type=XML
3. https://www.law.go.kr/DRF/lawService.do?OC=test&target=ppc&ID=9907&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 결정문일련번호 | int | 결정문 일련번호 |
| 기관명 | string | 기관명 |
| 결정 | string | 결정 |
| 회의종류 | string | 회의종류 |
| 안건번호 | string | 안건번호 |
| 안건명 | string | 안건명 |
| 신청인 | string | 신청인 |
| 의결연월일 | string | 의결연월일 |
| 주문 | string | 주문 |
| 이유 | string | 이유 |
| 배경 | string | 배경 |
| 이의제기방법및기간 | string | 이의제기방법및기간 |
| 주요내용 | string | 주요내용 |
| 의결일자 | string | 의결일자 |
| 위원서명 | string | 위원서명 |
| 별지 | string | 별지 |
| 결정요지 | string | 결정요지 |
