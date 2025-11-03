# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=oneview

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : oneview(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| query | string | 법령명에서 검색을 원하는 질의 |
| display | int | 검색된 결과 개수 (default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=oneview&type=XML
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=oneview&type=HTML
3. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=oneview&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색 대상 |
| 키워드 | string | 키워드 |
| section | string | 검색범위 |
| totalCnt | int | 검색결과갯수 |
| page | int | 출력페이지 |
| 법령 id | int | 검색결과번호 |
| 법령일련번호 | int | 법령일련번호 |
| 법령명 | string | 법령명 |
| 공포일자 | string | 공포일자 |
| 공포번호 | int | 공포번호 |
| 시행일자 | string | 시행일자 |
| 제공건수 | int | 제공건수 |
| 제공일자 | string | 제공일자 |
