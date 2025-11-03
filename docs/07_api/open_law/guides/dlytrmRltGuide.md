# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : https://www.law.go.kr/DRF/lawService.do?target=dlytrmRlt

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상(일상용어-법령용어 연계 : dlytrmRlt) |
| type | char(필수) | 출력 형태 : XML/JSON |
| query | string | 일상용어명에서 검색을 원하는 질의(query 또는 MST 중 하나는 반드시 입력) |
| MST | char | 일상용어명 일련번호 |
| trmRltCd | int | 용어관계동의어 : 140301반의어 : 140302상위어 : 140303하위어 : 140304연관어 : 140305 |

샘플 URL

1. https://www.law.go.kr/DRF/lawService.do?OC=test&target=dlytrmRlt&type=XML&query=민원
2. https://www.law.go.kr/DRF/lawService.do?OC=test&target=dlytrmRlt&type=JSON&query=민원

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색서비스 대상 |
| 키워드 | string | 검색 단어 |
| 검색결과개수 | int | 검색 건수 |
| 일상용어명 | string | 일상용어명 |
| 출처 | string | 일상용어 출처 |
| 연계용어 id | string | 연계용어 순번 |
| 법령용어명 | string | 법령용어명 |
| 비고 | string | 동음이의어 내용 |
| 용어관계코드 | string | 용어관계코드 |
| 용어관계 | string | 용어관계명 |
| 용어간관계링크 | string | 법령용어-일상용어 연계 정보 상세링크 |
| 조문간관계링크 | string | 법령용어-조문 연계 정보 상세링크 |
