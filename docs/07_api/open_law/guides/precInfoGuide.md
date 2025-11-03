# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=prec

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : prec(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON*국세청 판례본문 조회는HTML만 가능합니다 |
| ID | char(필수) | 판례 일련번호 |
| LM | string | 판례명 |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=prec&ID=228541&type=HTML
2. http://www.law.go.kr/DRF/lawService.do?OC=test&target=prec&ID=228541&type=XML
3. http://www.law.go.kr/DRF/lawService.do?OC=test&target=prec&ID=228541&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 판례정보일련번호 | int | 판례정보일련번호 |
| 사건명 | string | 사건명 |
| 사건번호 | string | 사건번호 |
| 선고일자 | int | 선고일자 |
| 선고 | string | 선고 |
| 법원명 | string | 법원명 |
| 법원종류코드 | int | 법원종류코드(대법원:400201, 하위법원:400202) |
| 사건종류명 | string | 사건종류명 |
| 사건종류코드 | int | 사건종류코드 |
| 판결유형 | string | 판결유형 |
| 판시사항 | string | 판시사항 |
| 판결요지 | string | 판결요지 |
| 참조조문 | string | 참조조문 |
| 참조판례 | string | 참조판례 |
| 판례내용 | string | 판례내용 |
