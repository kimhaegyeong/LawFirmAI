# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=lsHistory

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : lsHistory(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 HTML |
| query | string | 법령명에서 검색을 원하는 질의(정확한 검색을 위한 문자열 검색 query="자동차") |
| display | int | 검색된 결과 개수 (default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |
| sort | string | 정렬옵션(기본 : lasc 법령오름차순)ldes : 법령내림차순dasc : 공포일자 오름차순ddes : 공포일자 내림차순nasc : 공포번호 오름차순ndes : 공포번호 내림차순efasc : 시행일자 오름차순efdes : 시행일자 내림차순 |
| efYd | string | 시행일자 범위 검색(20090101~20090130) |
| date | string | 공포일자 검색 |
| ancYd | string | 공포일자 범위 검색(20090101~20090130) |
| ancNo | string | 공포번호 범위 검색(306~400) |
| rrClsCd | string | 법령 제개정 종류(300201-제정 / 300202-일부개정 / 300203-전부개정300204-폐지 / 300205-폐지제정 / 300206-일괄개정300207-일괄폐지 / 300209-타법개정 / 300210-타법폐지300208-기타) |
| org | string | 소관부처별 검색(소관부처코드 제공) |
| knd | string | 법령종류(코드제공) |
| lsChapNo | string | 법령분류(01-제1편 / 02-제2편 / 03-제3편... 44-제44편) |
| gana | string | 사전식 검색 (ga,na,da…,etc) |
| popYn | string | 상세화면 팝업창 여부(팝업창으로 띄우고 싶을 때만 'popYn=Y') |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lsHistory&type=HTML&query=자동차관리법
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lsHistory&type=HTML&org=1741000
