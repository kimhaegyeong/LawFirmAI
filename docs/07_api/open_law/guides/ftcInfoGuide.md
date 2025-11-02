# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : https://www.law.go.kr/DRF/lawService.do?target=ftc

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상(공정거래위원회 : ftc) |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| ID | char(필수) | 결정문 일련번호 |

샘플 URL

1. https://www.law.go.kr/DRF/lawService.do?OC=test&target=ftc&ID=331&type=HTML
2. https://www.law.go.kr/DRF/lawService.do?OC=test&target=ftc&ID=335&type=XML
3. http://www.law.go.kr/DRF/lawService.do?OC=test&target=ftc&ID=8111&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 결정문일련번호 | int | 결정문 일련번호 |
| 문서유형 | string | 출력 형태 : 의결서 / 시정권고서 |
| 사건번호 | string | 사건번호 |
| 사건명 | string | 사건명 |
| 피심정보명 | string | 피심정보명 |
| 피심정보내용 | string | 피심정보내용 |
| 회의종류 | string | 회의종류 |
| 결정번호 | string | 결정번호 |
| 결정일자 | string | 결정일자 |
| 원심결 | string | 원심결 |
| 재산정심결 | string | 재산정심결 |
| 후속심결 | string | 후속심결 |
| 심의정보명 | string | 심의정보명 |
| 심의정보내용 | string | 심의정보내용 |
| 의결문 | string | 의결문 |
| 주문 | string | 주문 |
| 신청취지 | string | 신청취지 |
| 이유 | string | 이유 |
| 의결일자 | string | 의결일자 |
| 위원정보 | string | 위원정보 |
| 각주번호 | int | 각주번호 |
| 각주내용 | string | 각주내용 |
| 별지 | string | 별지 |
| 결정요지 | string | 결정요지 |
