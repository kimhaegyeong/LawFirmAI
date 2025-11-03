# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=ordin&mobileYn=Y

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : ordin(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 HTML/XML/JSON |
| nw | int | (1: 현행, 2: 연혁, 기본값: 현행) |
| search | int | 검색범위 (기본 : 1 자치법규명) 2 : 본문검색 |
| query | string | 검색범위에서 검색을 원하는 질의(default=*) |
| display | int | 검색된 결과 개수 (default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |
| sort | string | 정렬옵션(기본 : lasc 자치법규오름차순) ldes 자치법규 내림차순dasc : 공포일자 오름차순ddes : 공포일자 내림차순nasc : 공포번호 오름차순ndes : 공포번호 내림차순efasc : 시행일자 오름차순efdes : 시행일자 내림차순 |
| date | int | 자치법규 공포일자 검색 |
| efYd | string | 시행일자 범위 검색(20090101~20090130) |
| ancYd | string | 공포일자 범위 검색(20090101~20090130) |
| ancNo | string | 공포번호 범위 검색(306~400) |
| nb | int | 법령의 공포번호 검색 |
| org | string | 지자체별 도·특별시·광역시 검색(지자체코드 제공)(ex. 서울특별시에 대한 검색-> org=6110000) |
| sborg | string | 지자체별 시·군·구 검색(지자체코드 제공)(필수값 : org, ex.서울특별시 구로구에 대한 검색-> org=6110000&sborg=3160000) |
| knd | string | 법령종류(30001-조례 /30002-규칙 /30003-훈령/30004-예규/30006-기타/30010-고시/30011-의회규칙) |
| rrClsCd | string | 법령 제개정 종류(300201-제정 / 300202-일부개정 / 300203-전부개정300204-폐지 / 300205-폐지제정 / 300206-일괄개정300207-일괄폐지 / 300208-타법개정 / 300209-타법폐지300214-기타) |
| ordinFd | int | 분류코드별 검색. 분류코드는 지자체 분야코드 openAPI 참조 |
| lsChapNo | string | 법령분야별 검색(법령분야코드제공)(ex. 제1편 검색 lsChapNo=01000000 /제1편2장,제1편2장1절 lsChapNo=01020000,01020100) |
| gana | string(org 값 필수) | 사전식 검색 (ga,na,da…,etc) |
| mobileYn | char : Y (필수) | 모바일여부 |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=ordin&type=XML&mobileYn=Y
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=ordin&type=HTML&mobileYn=Y
3. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=ordin&type=JSON&mobileYn=Y
4. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=ordin&query=서울&type=HTML&mobileYn=Y

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색 대상 |
| 키워드 | string | 키워드 |
| section | string | 검색범위(ordinNm:자치법규명/bdyText:본문) |
| totalCnt | int | 검색결과갯수 |
| page | int | 출력페이지 |
| law id | int | 검색결과번호 |
| 자치법규일련번호 | int | 자치법규일련번호 |
| 자치법규명 | string | 자치법규명 |
| 자치법규ID | int | 자치법규ID |
| 공포일자 | string | 공포일자 |
| 공포번호 | string | 공포번호 |
| 제개정구분명 | string | 제개정구분명 |
| 지자체기관명 | string | 지자체기관명 |
| 자치법규종류 | string | 자치법규종류 |
| 시행일자 | string | 시행일자 |
| 자치법규상세링크 | string | 자치법규상세링크 |
| 자치법규분야명 | string | 자치법규분야명 |
| 참조데이터구분 | string | 참조데이터구분 |
