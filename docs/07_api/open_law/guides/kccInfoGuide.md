# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : https://www.law.go.kr/DRF/lawService.do?target=kcc

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상(방송미디어통신위원회 : kcc) |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| ID | char(필수) | 결정문 일련번호 |

샘플 URL

1. https://www.law.go.kr/DRF/lawService.do?OC=test&target=kcc&ID=12549&type=HTML
2. https://www.law.go.kr/DRF/lawService.do?OC=test&target=kcc&ID=12547&type=XML
3. https://www.law.go.kr/DRF/lawService.do?OC=test&target=kcc&ID=11737&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 결정문일련번호 | int | 결정문 일련번호 |
| 기관명 | string | 기관명 |
| 의결서유형 | string | 의결서 유형 |
| 안건번호 | string | 안건번호 |
| 사건번호 | string | 사건번호 |
| 안건명 | string | 안건명 |
| 사건명 | string | 사건명 |
| 피심인 | string | 피심인 |
| 피심의인 | string | 피심의인 |
| 청구인 | string | 청구인 |
| 참고인 | string | 참고인 |
| 원심결정 | string | 원심결정 |
| 의결일자 | string | 의결일자 |
| 주문 | string | 주문 |
| 이유 | string | 이유 |
| 별지 | string | 별지 |
| 문서제공구분 | string | 문서제공구분(데이터 개방\|이유하단 이미지개방) |
| 각주번호 | int | 각주번호 |
| 각주내용 | string | 각주내용 |
