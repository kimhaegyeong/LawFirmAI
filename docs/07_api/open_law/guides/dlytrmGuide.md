# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : https://www.law.go.kr/DRF/lawSearch.do?target=dlytrm

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상(일상용어 : dlytrm) |
| type | char(필수) | 출력 형태 : XML/JSON |
| query | string | 일상용어명에서 검색을 원하는 질의 |
| display | int | 검색된 결과 개수(default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |

샘플 URL

1. https://www.law.go.kr/DRF/lawSearch.do?OC=test&target=dlytrm&type=XML
2. https://www.law.go.kr/DRF/lawSearch.do?OC=test&target=dlytrm&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색서비스 대상 |
| 키워드 | string | 검색 단어 |
| 검색결과개수 | int | 검색 건수 |
| section | string | 검색범위 |
| page | int | 현재 페이지번호 |
| numOfRows | int | 페이지 당 출력 결과 수 |
| 일상용어 id | string | 일상용어 순번 |
| 일상용어명 | string | 일상용어명 |
| 출처 | string | 일상용어 출처 |
| 용어간관계링크 | string | 일상용어-법령용어 연계 정보 상세링크 |
