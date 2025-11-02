# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=lsStmd

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : lsStmd(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 HTML/XML/JSON |
| query | string | 법령명에서 검색을 원하는 질의 |
| display | int | 검색된 결과 개수 (default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |
| sort | string | 정렬옵션(기본 : lasc 법령오름차순)ldes : 법령내림차순dasc : 공포일자 오름차순ddes : 공포일자 내림차순nasc : 공포번호 오름차순ndes : 공포번호 내림차순efasc : 시행일자 오름차순efdes : 시행일자 내림차순 |
| efYd | string | 시행일자 범위 검색(20090101~20090130) |
| ancYd | string | 공포일자 범위 검색(20090101~20090130) |
| date | int | 공포일자 검색 |
| nb | int | 공포번호 검색 |
| ancNo | string | 공포번호 범위 검색 (10000~20000) |
| rrClsCd | string | 법령 제개정 종류(300201-제정 / 300202-일부개정 / 300203-전부개정300204-폐지 / 300205-폐지제정 / 300206-일괄개정300207-일괄폐지 / 300209-타법개정 / 300210-타법폐지300208-기타) |
| org | string | 소관부처별 검색(소관부처코드 제공) |
| knd | string | 법령종류(코드제공) |
| gana | string | 사전식 검색 (ga,na,da…,etc) |
| popYn | string | 상세화면 팝업창 여부(팝업창으로 띄우고 싶을 때만 'popYn=Y') |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lsStmd&type=HTML&query=자동차관리법
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lsStmd&type=HTML&gana=ga
3. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lsStmd&type=XML
4. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lsStmd&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색서비스 대상 |
| 키워드 | string | 검색 단어 |
| section | string | 검색범위 |
| totalCnt | int | 검색 건수 |
| page | int | 현재 페이지번호 |
| numOfRows | int | 페이지 당 출력 결과 수 |
| resultCode | int | 조회 여부(성공 : 00 / 실패 : 01) |
| resultMsg | int | 조회 여부(성공 : success / 실패 : fail) |
| law id | int | 검색 결과 순번 |
| 법령 일련번호 | int | 법령 일련번호 |
| 법령명 | string | 법령명 |
| 법령ID | int | 법령ID |
| 공포일자 | int | 공포일자 |
| 공포번호 | int | 공포번호 |
| 제개정구분명 | string | 제개정구분명 |
| 소관부처코드 | int | 소관부처코드 |
| 소관부처명 | string | 소관부처명 |
| 법령구분명 | string | 법령구분명 |
| 시행일자 | int | 시행일자 |
| 본문 상세링크 | string | 본문 상세링크 |
