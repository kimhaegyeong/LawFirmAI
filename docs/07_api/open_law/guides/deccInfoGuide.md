# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=decc

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : decc(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| ID | char(필수) | 행정심판례 일련번호 |
| LM | string | 행정심판례명 |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=decc&ID=243263&type=HTML
2. http://www.law.go.kr/DRF/lawService.do?OC=test&target=decc&ID=245011&LM=과징금
3. http://www.law.go.kr/DRF/lawService.do?OC=test&target=decc&ID=223311&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 행정심판례일련번호 | int | 행정심판례일련번호 |
| 사건명 | string | 사건명 |
| 사건번호 | string | 사건번호 |
| 처분일자 | int | 처분일자 |
| 의결일자 | int | 의결일자 |
| 처분청 | string | 처분청 |
| 재결청 | string | 재결청 |
| 재결례유형명 | string | 재결례유형명 |
| 재결례유형코드 | int | 재결례유형코드 |
| 주문 | string | 주문 |
| 청구취지 | string | 청구취지 |
| 이유 | string | 이유 |
| 재결요지 | string | 재결요지 |
