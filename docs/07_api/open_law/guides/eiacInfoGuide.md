# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : https://www.law.go.kr/DRF/lawService.do?target=eiac

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상(고용보험심사위원회 : eiac) |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| ID | char(필수) | 결정문 일련번호 |

샘플 URL

1. https://www.law.go.kr/DRF/lawService.do?OC=test&target=eiac&ID=11347&type=HTML
2. https://www.law.go.kr/DRF/lawService.do?OC=test&target=eiac&ID=11327&type=XML
3. https://www.law.go.kr/DRF/lawService.do?OC=test&target=eiac&ID=11165&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 결정문일련번호 | int | 결정문 일련번호 |
| 사건의분류 | string | 사건의 분류 |
| 의결서종류 | string | 의결서 종류 |
| 개요 | string | 개요 |
| 사건번호 | string | 사건번호 |
| 사건명 | string | 사건명 |
| 청구인 | string | 청구인 |
| 대리인 | string | 대리인 |
| 피청구인 | string | 피청구인 |
| 이해관계인 | string | 이해관계인 |
| 심사결정심사관 | string | 심사결정심사관 |
| 주문 | string | 주문 |
| 청구취지 | string | 청구취지 |
| 이유 | string | 이유 |
| 의결일자 | string | 의결일자 |
| 기관명 | string | 기관명 |
| 별지 | string | 별지 |
| 각주번호 | int | 각주번호 |
| 각주내용 | string | 각주내용 |
