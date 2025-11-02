# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : https://www.law.go.kr/DRF/lawSearch.do?target=ecc

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상(중앙환경분쟁조정위원회 : ecc) |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| search | int | 검색범위1 : 사건명 (default)2 : 본문검색 |
| query | string | 검색범위에서 검색을 원하는 질의(IE 조회시 UTF-8 인코딩 필수) |
| display | int | 검색된 결과 개수(default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |
| gana | string | 사전식 검색 (ga,na,da…,etc) |
| sort | string | 정렬옵션lasc : 사건명 오름차순 (default)ldes : 사건명 내림차순nasc : 의결번호 오름차순ndes : 의결번호 내림차순 |
| popYn | string | 상세화면 팝업창 여부(팝업창으로 띄우고 싶을 때만 'popYn=Y') |

샘플 URL

1. https://www.law.go.kr/DRF/lawSearch.do?OC=test&target=ecc&type=HTML
2. https://www.law.go.kr/DRF/lawSearch.do?OC=test&target=ecc&type=XML
3. https://www.law.go.kr/DRF/lawSearch.do?OC=test&target=ecc&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색서비스 대상 |
| 키워드 | string | 검색 단어 |
| section | string | 검색범위 |
| totalCnt | int | 검색 건수 |
| page | int | 현재 페이지번호 |
| 기관명 | string | 위원회명 |
| ecc id | int | 검색 결과 순번 |
| 결정문일련번호 | int | 결정문 일련번호 |
| 사건명 | string | 사건명 |
| 의결번호 | string | 의결번호 |
| 결정문상세링크 | string | 결정문 상세링크 |
