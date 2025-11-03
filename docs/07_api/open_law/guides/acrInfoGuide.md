# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : https://www.law.go.kr/DRF/lawService.do?target=acr

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상(국민권익위원회 : acr) |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| ID | char(필수) | 결정문 일련번호 |

샘플 URL

1. https://www.law.go.kr/DRF/lawService.do?OC=test&target=acr&ID=53&type=HTML
2. https://www.law.go.kr/DRF/lawService.do?OC=test&target=acr&ID=89&type=XML
3. https://www.law.go.kr/DRF/lawService.do?OC=test&target=acr&ID=1281&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 결정문일련번호 | int | 결정문 일련번호 |
| 기관명 | string | 기관명 |
| 회의종류 | string | 회의종류 |
| 결정구분 | string | 결정구분 |
| 의안번호 | string | 의안번호 |
| 민원표시 | string | 민원표시 |
| 제목 | string | 제목 |
| 신청인 | string | 신청인 |
| 대리인 | string | 대리인 |
| 피신청인 | string | 피신청인 |
| 관계기관 | string | 관계기관 |
| 의결일 | string | 의결일 |
| 주문 | string | 주문 |
| 이유 | string | 이유 |
| 별지 | string | 별지 |
| 의결문 | string | 의결문 |
| 의결일자 | string | 의결일자 |
| 위원정보 | string | 위원정보 |
| 결정요지 | string | 결정요지 |
