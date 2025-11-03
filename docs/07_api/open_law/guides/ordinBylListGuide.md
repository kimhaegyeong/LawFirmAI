# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=ordinbyl

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : ordinbyl(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 HTML/XML/JSON |
| search | int | 검색범위(기본 : 1 별표서식명)2 : 해당자치법규명검색3 : 별표본문검색 |
| query | string | 법령명에서 검색을 원하는 질의(default=*)(정확한 검색을 위한 문자열 검색 query="자동차") |
| display | int | 검색된 결과 개수 (default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |
| sort | string | 정렬옵션 (기본 : lasc 별표서식명 오름차순) ldes 별표서식명 내림차순 |
| org | string | 소관부처별 검색(소관부처코드 제공) |
| sborg | string | 지자체별 시·군·구 검색(지자체코드 제공)(필수값 : org, ex.서울특별시 구로구에 대한 검색-> org=6110000&sborg=3160000) |
| knd | string | 별표종류1 : 별표 2 : 서식 3 : 별도 4 : 별지 |
| gana | string | 사전식 검색(ga,na,da…,etc) |
| popYn | string | 상세화면 팝업창 여부(팝업창으로 띄우고 싶을 때만 'popYn=Y') |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=ordinbyl&type=XML
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=ordinbyl&type=HTML
3. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=ordinbyl&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색서비스 대상 |
| 키워드 | string | 검색 단어 |
| section | string | 검색범위 |
| totalCnt | int | 검색건수 |
| page | int | 현재 페이지번호 |
| ordinbyl id | int | 검색 결과 순번 |
| 별표일련번호 | string | 별표일련번호 |
| 관련자치법규일련번호 | string | 관련자치법규일련번호 |
| 별표명 | string | 별표명 |
| 관련자치법규명 | string | 관련자치법규명 |
| 별표번호 | string | 별표번호 |
| 별표종류 | string | 별표종류 |
| 지자체기관명 | string | 지자체기관명 |
| 전체기관명 | string | 전체기관명 |
| 자치법규시행일자 | string | 자치법규시행일자 |
| 공포일자 | string | 공포일자 |
| 공포번호 | string | 공포번호 |
| 제개정구분명 | string | 제개정구분명 |
| 별표서식파일링크 | string | 별표서식파일링크 |
| 별표자치법규상세링크 | string | 별표자치법규상세링크 |
