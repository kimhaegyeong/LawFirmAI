# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=school(or

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상(대학 : school / 지방공사공단 : public / 공공기관 : pi) |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| nw | int | (1: 현행, 2: 연혁, 기본값: 현행) |
| search | int | 검색범위1 : 규정명(default)2 : 본문검색 |
| query | string | 검색범위에서 검색을 원하는 질의(정확한 검색을 위한 문자열 검색 query="자동차") |
| display | int | 검색된 결과 개수(default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |
| knd | string | 학칙공단 종류별 검색1 : 학칙 / 2 : 학교규정 / 3 : 학교지침 / 4 : 학교시행세칙/ 5 : 공단규정, 공공기관규정 |
| rrClsCd | string | 제정·개정 구분200401 : 제정 / 200402 : 전부개정 / 200403 : 일부개정 / 200404 : 폐지200405 : 일괄개정 / 200406 : 일괄폐지 / 200407 : 폐지제정200408 : 정정 / 200409 : 타법개정 / 200410 : 타법폐지 |
| date | int | 발령일자 검색 |
| prmlYd | string | 발령일자 범위 검색 |
| nb | int | 발령번호 검색 |
| gana | string | 사전식 검색 (ga,na,da…,etc) |
| sort | string | 정렬옵션lasc : 학칙공단명 오름차순(default)ldes : 학칙공단명 내림차순dasc : 발령일자 오름차순ddes : 발령일자 내림차순nasc : 발령번호 오름차순ndes : 발령번호 내림차순 |
| popYn | string | 상세화면 팝업창 여부(팝업창으로 띄우고 싶을 때만 'popYn=Y') |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=school&query=학교&type=HTML
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=school&query=학교&type=XML
3. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=school&query=학교&type=JSON

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
| admrul id | int | 검색 결과 순번 |
| 행정규칙일련번호 | int | 학칙공단 일련번호 |
| 행정규칙명 | string | 학칙공단명 |
| 행정규칙종류 | string | 학칙공단 종류 |
| 발령일자 | int | 발령일자 |
| 발령번호 | int | 발령번호 |
| 소관부처명 | string | 소관부처명 |
| 현행연혁구분 | string | 현행연혁구분 |
| 제개정구분코드 | string | 제개정구분코드 |
| 제개정구분명 | string | 제개정구분명 |
| 법령분류코드 | string | 법령분류코드 |
| 법령분류명 | string | 법령분류명 |
| 행정규칙ID | int | 학칙공단ID |
| 행정규칙상세링크 | string | 학칙공단 상세링크 |
| 시행일자 | int | 시행일자 |
| 생성일자 | int | 생성일자 |
