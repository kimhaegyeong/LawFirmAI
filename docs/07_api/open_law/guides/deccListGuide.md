# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=decc

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : decc(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| search | int | 검색범위(기본 : 1 행정심판례명)2 : 본문검색 |
| query | string | 검색범위에서 검색을 원하는 질의(정확한 검색을 위한 문자열 검색 query="자동차") |
| display | int | 검색된 결과 개수 (default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |
| cls | string | 재결례유형(출력 결과 필드에 있는 재결구분코드) |
| gana | string | 사전식 검색(ga,na,da…,etc) |
| date | int | 의결일자 |
| dpaYd | string | 처분일자 검색(20090101~20090130) |
| rslYd | string | 의결일자 검색(20090101~20090130) |
| sort | string | 정렬옵션(기본 : lasc 재결례명 오름차순)ldes 재결례명 내림차순dasc : 의결일자 오름차순ddes : 의결일자 내림차순nasc : 사건번호 오름차순ndes : 사건번호 내림차순 |
| popYn | string | 상세화면 팝업창 여부(팝업창으로 띄우고 싶을 때만 'popYn=Y') |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=decc&type=XML
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=decc&type=HTML
3. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=decc&type=JSON
4. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=decc&type=XML&gana=ga

출력 결과 필드(response field)

| 샘플 URL |
| --- |
| 1. 행정심판재결례 목록 XML 검색 |
| http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=decc&type=XML |
| 2. 행정심판재결례 목록 HTML 검색 |
| http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=decc&type=HTML |
| 3. 행정심판재결례 목록 JSON 검색 |
| http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=decc&type=JSON |
| 4. 행정심판재결례 목록 중 ‘ㄱ’으로 시작하는 재결례 목록 검색 |
| http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=decc&type=XML&gana=ga |
