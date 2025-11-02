# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=lnkOrd

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : lnkOrd(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| query | string | 법규명에서 검색을 원하는 질의 |
| display | int | 검색된 결과 개수 (default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |
| sort | string | 정렬옵션(기본 : lasc 자치법규오름차순) ldes 자치법규 내림차순dasc : 공포일자 오름차순ddes : 공포일자 내림차순nasc : 공포번호 오름차순ndes : 공포번호 내림차순efasc : 시행일자 오름차순efdes : 시행일자 내림차순 |
| popYn | string | 상세화면 팝업창 여부(팝업창으로 띄우고 싶을 때만 'popYn=Y') |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lnkOrd&type=XML
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lnkOrd&type=HTML
3. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lnkOrd&type=JSON&query=청소년
4. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lnkLsOrd&knd=002118&type=XML
5. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lnkLsOrd&knd=002118&type=HTML
6. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lnkLsOrd&knd=002118&type=JSON
7. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lnkOrg&org=3270000&type=XML
8. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lnkOrg&org=3270000&type=HTML
9. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lnkOrg&org=3270000&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색서비스 대상 |
| 키워드 | string | 검색 단어 |
| section | string | 검색범위 |
| totalCnt | int | 검색건수 |
| page | int | 결과페이지번호 |
| law id | int | 결과 번호 |
| 자치법규일련번호 | int | 자치법규 일련번호 |
| 자치법규명 | string | 자치법규명 |
| 자치법규ID | int | 자치법규ID |
| 공포일자 | int | 공포일자 |
| 공포번호 | int | 공포번호 |
| 제개정구분명 | string | 제개정구분명 |
| 자치법규종류 | string | 자치법규종류 |
| 시행일자 | int | 시행일자 |
