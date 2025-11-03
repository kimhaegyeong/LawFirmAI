# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=trty

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : trty(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| search | int | 검색범위 (기본 : 1 조약명) 2 : 조약본문 |
| query | string | 검색범위에서 검색을 원하는 질의(검색 결과 리스트) |
| display | int | 검색된 결과 개수 (default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |
| gana | string | 사전식 검색 (ga,na,da…,etc) |
| eftYd | string | 발효일자 검색(20090101~20090130) |
| concYd | string | 체결일자 검색(20090101~20090130) |
| cls | int | 1 : 양자조약 2 : 다자조약 |
| natCd | int | 국가코드 |
| sort | string | 정렬옵션(기본 : lasc 조약명오름차순)ldes 조약명내림차순dasc : 발효일자 오름차순ddes : 발효일자 내림차순nasc : 조약번호 오름차순ndes : 조약번호 내림차순rasc : 관보게재일 오름차순rdes : 관보게재일 내림차순 |
| popYn | string | 상세화면 팝업창 여부(팝업창으로 띄우고 싶을 때만 'popYn=Y') |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=trty&type=XML
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=trty&ID=284&type=HTML
3. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=trty&ID=284&type=JSON
4. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=trty&type=XML&cls=2

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색 대상 |
| 키워드 | string | 키워드 |
| section | string | 검색범위(TrtyNm:조약명/bdyText:본문) |
| totalCnt | int | 검색결과갯수 |
| page | int | 출력페이지 |
| trty id | int | 검색결과번호 |
| 조약일련번호 | int | 조약일련번호 |
| 조약명 | string | 조약명 |
| 조약구분코드 | string | 조약구분코드 |
| 조약구분명 | string | 조약구분명 |
| 발효일자 | string | 발효일자 |
| 서명일자 | string | 서명일자 |
| 관보게제일자 | string | 관보게제일자 |
| 조약번호 | int | 조약번호 |
| 국가번호 | int | 국가번호 |
| 조약상세링크 | string | 조약상세링크 |
