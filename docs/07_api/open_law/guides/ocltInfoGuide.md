# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : https://www.law.go.kr/DRF/lawService.do?target=oclt

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상(중앙토지수용위원회 : oclt) |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| ID | char(필수) | 결정문 일련번호 |

샘플 URL

1. https://www.law.go.kr/DRF/lawService.do?OC=test&target=oclt&ID=4973&type=HTML
2. https://www.law.go.kr/DRF/lawService.do?OC=test&target=oclt&ID=4965&type=XML
3. https://www.law.go.kr/DRF/lawService.do?OC=test&target=oclt&ID=4971&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 결정문일련번호 | int | 결정문 일련번호 |
| 제목 | string | 제목 |
| 관련법리 | string | 관련 법리 |
| 관련규정 | string | 관련 규정 |
| 판단 | string | 판단 |
| 근거 | string | 근거 |
| 주해 | string | 주해 |
| 각주번호 | int | 각주번호 |
| 각주내용 | string | 각주내용 |
