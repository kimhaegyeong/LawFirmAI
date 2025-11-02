# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=expc

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : expc(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| search | int | 검색범위 (기본 : 1 법령해석례명) 2 : 본문검색 |
| query | string | 검색범위에서 검색을 원하는 질의(정확한 검색을 위한 문자열 검색 query="자동차") |
| display | int | 검색된 결과 개수 (default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |
| inq | inq | 질의기관 |
| rpl | int | 회신기관 |
| gana | string | 사전식 검색(ga,na,da…,etc) |
| itmno | int | 안건번호13-0217 검색을 원할시 itmno=130217 |
| regYd | string | 등록일자 검색(20090101~20090130) |
| explYd | string | 해석일자 검색(20090101~20090130) |
| sort | string | 정렬옵션 (기본 : lasc 법령해석례명 오름차순)ldes 법령해석례명 내림차순dasc : 해석일자 오름차순ddes : 해석일자 내림차순nasc : 안건번호 오름차순ndes : 안건번호 내림차순 |
| popYn | string | 상세화면 팝업창 여부(팝업창으로 띄우고 싶을 때만 'popYn=Y') |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=expc&type=XML&query=임차
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=expc&type=HTML&주차
3. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=expc&type=JSON&query=자동차

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색 대상 |
| 키워드 | string | 키워드 |
| section | string | 검색범위(lawNm:법령해석례명/bdyText:본문) |
| totalCnt | int | 검색결과갯수 |
| page | int | 출력페이지 |
| expc id | int | 검색결과번호 |
| 법령해석례일련번호 | int | 법령해석례일련번호 |
| 안건명 | string | 안건명 |
| 안건번호 | string | 안건번호 |
| 질의기관코드 | int | 질의기관코드 |
| 질의기관명 | string | 질의기관명 |
| 회신기관코드 | string | 회신기관코드 |
| 회신기관명 | string | 회신기관명 |
| 회신일자 | string | 회신일자 |
| 법령해석례상세링크 | string | 법령해석례상세링크 |
