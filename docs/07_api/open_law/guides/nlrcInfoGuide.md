# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : https://www.law.go.kr/DRF/lawService.do?target=nlrc

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상(노동위원회 : nlrc) |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| ID | char(필수) | 결정문 일련번호 |

샘플 URL

1. https://www.law.go.kr/DRF/lawService.do?OC=test&target=nlrc&ID=55&type=HTML
2. https://www.law.go.kr/DRF/lawService.do?OC=test&target=nlrc&ID=71&type=XML
3. https://www.law.go.kr/DRF/lawService.do?OC=test&target=nlrc&ID=129&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 결정문일련번호 | int | 결정문 일련번호 |
| 기관명 | string | 기관명 |
| 사건번호 | string | 사건번호 |
| 자료구분 | string | 자료구분 |
| 담당부서 | string | 담당부서 |
| 등록일 | string | 등록일 |
| 제목 | string | 제목 |
| 내용 | string | 내용 |
| 판정사항 | string | 판정사항 |
| 판정요지 | string | 판정요지 |
| 판정결과 | string | 판정결과 |
| 각주번호 | int | 각주번호 |
| 각주내용 | string | 각주내용 |
