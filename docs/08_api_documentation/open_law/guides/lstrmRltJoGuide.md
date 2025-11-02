# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : https://www.law.go.kr/DRF/lawService.do?target=lstrmRltJo

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상(법령용어-조문 연계 : lstrmRltJo) |
| type | char(필수) | 출력 형태 : XML/JSON |
| query | string(필수) | 법령용어명에서 검색을 원하는 질의 |

샘플 URL

1. https://www.law.go.kr/DRF/lawService.do?OC=test&target=lstrmRltJo&type=XML&query=민원
2. https://www.law.go.kr/DRF/lawService.do?OC=test&target=lstrmRltJo&type=JSON&query=민원

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색서비스 대상 |
| 키워드 | string | 검색 단어 |
| 검색결과개수 | int | 검색 건수 |
| 법령용어 id | string | 법령용어 순번 |
| 법령용어명 | string | 법령용어명 |
| 비고 | string | 동음이의어 내용 |
| 용어간관계링크 | string | 법령용어-일상용어 연계 정보 상세링크 |
| 연계법령 id | string | 연계법령 순번 |
| 법령명 | string | 법령명 |
| 조번호 | int | 조번호 |
| 조가지번호 | int | 조가지번호 |
| 조문내용 | string | 조문내용 |
| 용어구분코드 | string | 용어구분코드 |
| 용어구분 | string | 용어구분명 |
| 조문연계용어링크 | string | 조문-법령용어 연계 정보 상세링크 |
