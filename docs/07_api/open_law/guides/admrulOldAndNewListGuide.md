# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=admrulOldAndNew

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : admrulOldAndNew(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 HTML/XML/JSON |
| query | string | 법령명에서 검색을 원하는 질의(정확한 검색을 위한 문자열 검색 query="자동차") |
| display | int | 검색된 결과 개수 (default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |
| org | string | 소관부처별 검색(소관부처코드 제공) |
| knd | string | 행정규칙 종류별 검색(1=훈령/2=예규/3=고시/4=공고/5=지침/6=기타) |
| gana | string | 사전식 검색 (ga,na,da…,etc) |
| sort | string | 정렬옵션(기본 : lasc 법령오름차순)ldes : 법령내림차순dasc : 발령일자 오름차순ddes : 발령일자 내림차순nasc : 발령번호 오름차순ndes : 발령번호 내림차순efasc : 시행일자 오름차순efdes : 시행일자 내림차순 |
| date | string | 행정규칙 발령일자 |
| prmlYd | string | 발령일자 기간검색(20090101~20090130) |
| nb | int | 행정규칙 발령번호ex)제2023-8호 검색을 원할시 nb=20238 |
| popYn | string | 상세화면 팝업창 여부(팝업창으로 띄우고 싶을 때만 'popYn=Y') |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=admrulOldAndNew&type=HTML
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=admrulOldAndNew&type=XML&query=119항공대
3. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=admrulOldAndNew&type=JSON&query=119항공대

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
| oldAndNew id | int | 검색 결과 순번 |
| 신구법일련번호 | int | 신구법 일련번호 |
| 현행연혁구분 | string | 현행연혁코드 |
| 신구법명 | string | 신구법명 |
| 신구법ID | int | 신구법ID |
| 발령일자 | int | 발령일자 |
| 발령번호 | int | 발령번호 |
| 제개정구분명 | string | 제개정구분명 |
| 소관부처코드 | int | 소관부처코드 |
| 소관부처명 | string | 소관부처명 |
| 법령구분명 | string | 법령구분명 |
| 시행일자 | int | 시행일자 |
| 신구법상세링크 | string | 신구법 상세링크 |
