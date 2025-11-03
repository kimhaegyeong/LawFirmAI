# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : https://www.law.go.kr/DRF/lawService.do?target=ecc

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상(중앙환경분쟁조정위원회 : ecc) |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| ID | char(필수) | 결정문 일련번호 |

샘플 URL

1. https://www.law.go.kr/DRF/lawService.do?OC=test&target=ecc&ID=5883&type=HTML
2. https://www.law.go.kr/DRF/lawService.do?OC=test&target=ecc&ID=5877&type=XML
3. https://www.law.go.kr/DRF/lawService.do?OC=test&target=ecc&ID=5729&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 결정문일련번호 | int | 결정문 일련번호 |
| 의결번호 | string | 의결번호 |
| 사건명 | string | 사건명 |
| 사건의개요 | string | 사건의 개요 |
| 신청인 | string | 신청인 |
| 피신청인 | string | 피신청인 |
| 분쟁의경과 | string | 분쟁의 경과 |
| 당사자주장 | string | 당사자 주장 |
| 사실조사결과 | string | 사실조사 결과 |
| 평가의견 | string | 평가의견 |
| 주문 | string | 주문 |
| 이유 | string | 이유 |
| 각주번호 | int | 각주번호 |
| 각주내용 | string | 각주내용 |
