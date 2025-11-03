# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=detc

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : detc(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| ID | char(필수) | 헌재결정례 일련번호 |
| LM | string | 헌재결정례명 |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=detc&ID=58386&type=HTML
2. http://www.law.go.kr/DRF/lawService.do?OC=test&target=detc&ID=127830&LM=자동차관리법제26조등위헌확인등&type=XML
3. http://www.law.go.kr/DRF/lawService.do?OC=test&target=detc&ID=58400&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 헌재결정례일련번호 | int | 헌재결정례일련번호 |
| 종국일자 | int | 종국일자 |
| 사건번호 | string | 사건번호 |
| 사건명 | string | 사건명 |
| 사건종류명 | string | 사건종류명 |
| 사건종류코드 | int | 사건종류코드 |
| 재판부구분코드 | int | 재판부구분코드(전원재판부:430201, 지정재판부:430202) |
| 판시사항 | string | 판시사항 |
| 결정요지 | string | 결정요지 |
| 전문 | string | 전문 |
| 참조조문 | string | 참조조문 |
| 참조판례 | string | 참조판례 |
| 심판대상조문 | string | 심판대상조문 |
