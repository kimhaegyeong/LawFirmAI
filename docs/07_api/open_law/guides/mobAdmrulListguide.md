# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=admrul&mobileYn=Y

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : admrul(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 HTML/XML/JSON |
| nw | int | (1: 현행, 2: 연혁, 기본값: 현행) |
| search | int | 검색범위(기본 : 1 행정규칙명)2 : 본문검색 |
| query | string | 검색범위에서 검색을 원하는 질의(검색 결과 리스트) |
| display | int | 검색된 결과 개수 (default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |
| org | string | 소관부처별 검색(코드 별도 제공) |
| knd | string | 행정규칙 종류별 검색(1=훈령/2=예규/3=고시/4=공고/5=지침/6=기타) |
| gana | string | 사전식 검색(ga,na,da…,etc) |
| sort | string | 정렬옵션(기본 : lasc 행정규칙명 오른차순)ldes 행정규칙명 내림차순dasc : 발령일자 오름차순ddes : 발령일자 내림차순nasc : 발령번호 오름차순ndes : 발령번호 내림차순efasc : 시행일자 오름차순efdes : 시행일자 내림차순 |
| date | int | 행정규칙 발령일자 |
| prmlYd | string | 발령일자 기간검색(20090101~20090130) |
| nb | int | 행정규칙 발령번호 |
| mobileYn | char:Y(필수) | 모바일여부 |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=admrul&type=XML&mobileYn=Y
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=admrul&type=HTML&mobileYn=Y
3. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=admrul&type=JSON&mobileYn=Y
4. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=admrul&type=XML&mobileYn=Y&query=소방
5. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=admrul&type=XML&date=20150301&mobileYn=Y
6. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=admrul&type=XML&nb=331&mobileYn=Y

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색 대상 |
| 키워드 | string | 검색키워드 |
| section | string | 검색범위(AdmRulNm:행정규칙명/bdyText:본문) |
| totalCnt | int | 검색결과갯수 |
| page | int | 출력페이지 |
| admrul id | int | 검색결과번호 |
| 행정규칙일련번호 | int | 행정규칙일련번호 |
| 행정규칙명 | string | 행정규칙명 |
| 행정규칙종류 | string | 행정규칙종류 |
| 발령일자 | string | 발령일자 |
| 발령번호 | string | 발령번호 |
| 소관부처명 | string | 소관부처명 |
| 현행연혁구분 | string | 현행연혁구분 |
| 제개정구분코드 | string | 제개정구분코드 |
| 제개정구분명 | string | 제개정구분명 |
| 행정규칙ID | string | 행정규칙ID |
| 행정규칙상세링크 | string | 행정규칙상세링크 |
| 시행일자 | string | 시행일자 |
| 생성일자 | string | 생성일자 |
