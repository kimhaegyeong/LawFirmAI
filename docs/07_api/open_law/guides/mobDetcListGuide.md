# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=detc&mobileYn=Y

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : detc(필수) | 서비스 대상 |
| type | char | 출력 형태 : HTML/XML/JSON |
| search | int | 검색범위(기본 : 1 헌재결정례명)2 : 본문검색 |
| query | string | 검색범위에서 검색을 원하는 질의(검색 결과 리스트) |
| display | int | 검색된 결과 개수 (default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |
| gana | string | 사전식 검색(ga,na,da…,etc) |
| sort | string | 정렬옵션(기본 : lasc 사건명 오름차순) ldes 사건명 내림차순dasc : 선고일자 오름차순ddes : 선고일자 내림차순nasc : 사건번호 오름차순ndes : 사건번호 내림차순efasc : 종국일자 오름차순efdes : 종국일자 내림차순 |
| date | int | 종국일자 |
| nb | int | 사건번호 |
| mobileYn | char : Y (필수) | 모바일여부 |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=detc&type=XML&mobileYn=Y
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=detc&type=HTML&mobileYn=Y
3. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=detc&type=JSON&mobileYn=Y
4. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=detc&type=XML&mobileYn=Y&query=자동차
5. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=detc&type=XML&date=20150210&mobileYn=Y

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색 대상 |
| 키워드 | string | 키워드 |
| section | string | 검색범위(EvtNm:헌재결정례명/bdyText:본문) |
| totalCnt | int | 검색결과갯수 |
| page | int | 출력페이지 |
| detc id | int | 검색결과번호 |
| 헌재결정례일련번호 | int | 헌재결정례일련번호 |
| 종국일자 | string | 종국일자 |
| 사건번호 | string | 사건번호 |
| 사건명 | string | 사건명 |
| 헌재결정례상세링크 | string | 헌재결정례상세링크 |
