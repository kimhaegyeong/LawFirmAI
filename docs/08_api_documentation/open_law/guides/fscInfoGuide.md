# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : https://www.law.go.kr/DRF/lawService.do?target=fsc

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상(금융위원회 : fsc) |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| ID | char(필수) | 결정문 일련번호 |

샘플 URL

1. https://www.law.go.kr/DRF/lawService.do?OC=test&target=fsc&ID=9211&type=HTML
2. https://www.law.go.kr/DRF/lawService.do?OC=test&target=fsc&ID=9169&type=XML
3. https://www.law.go.kr/DRF/lawService.do?OC=test&target=fsc&ID=13097&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 결정문일련번호 | int | 결정문 일련번호 |
| 기관명 | string | 기관명 |
| 의결번호 | string | 의결번호 |
| 안건명 | string | 안건명 |
| 조치대상자의인적사항 | string | 조치대상자의 인적사항 |
| 조치대상 | string | 조치대상 |
| 조치내용 | string | 조치내용 |
| 조치이유 | string | 조치이유 |
| 각주번호 | int | 각주번호 |
| 각주내용 | string | 각주내용 |
