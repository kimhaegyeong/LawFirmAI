# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=lstrm

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : lstrm(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| query | string | 상세조회하고자 하는 법령용어 명 |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=lstrm&query=선박&type=HTML
2. http://www.law.go.kr/DRF/lawService.do?OC=test&target=lstrm&query=선박&type=XML
3. http://www.law.go.kr/DRF/lawService.do?OC=test&target=lstrm&query=선박&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 법령용어 일련번호 | int | 법령용어 일련번호 |
| 법령용어명_한글 | string | 법령용어명 한글 |
| 법령용어명_한자 | string | 법령용어명한자 |
| 법령용어코드 | int | 법령용어코드 |
| 법령용어코드명 | string | 법령용어코드명 |
| 출처 | string | 출처 |
| 법령용어정의 | string | 법령용어정의 |
