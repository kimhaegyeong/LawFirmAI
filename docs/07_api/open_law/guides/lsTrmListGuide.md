# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=lstrm

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : lstrm(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| query | string | 법령용어명에서 검색을 원하는 질의 |
| display | int | 검색된 결과 개수 (default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |
| sort | string | 정렬옵션(기본 : lasc 법령용어명 오름차순)ldes : 법령용어명 내림차순rasc : 등록일자 오름차순rdes : 등록일자 내림차순 |
| regDt | string | 등록일자 범위 검색(20090101~20090130) |
| gana | string | 사전식 검색 (ga,na,da…,etc) |
| popYn | string | 상세화면 팝업창 여부(팝업창으로 띄우고 싶을 때만 'popYn=Y') |
| dicKndCd | int | 법령 종류 코드 (법령 : 010101, 행정규칙 : 010102) |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lstrm&type=XML
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lstrm&type=JSON
3. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lstrm&gana=ra&type=XML
4. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lstrm&query=자동차&type=HTML

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색서비스 대상 |
| 키워드 | string | 검색어 |
| section | string | 검색범위 |
| totalCnt | int | 검색건수 |
| page | int | 결과페이지번호 |
| lstrm id | int | 결과 번호 |
| 법령용어ID | string | 법령용어ID |
| 법령용어명 | string | 법령용어명 |
| 법령용어상세검색 | string | 법령용어상세검색 |
| 사전구분코드 | string | 사전구분코드(법령용어사전 : 011401, 법령정의사전 : 011402, 법령한영사전 : 011403) |
| 법령용어상세링크 | string | 법령용어상세링크 |
| 법령종류코드 | int | 법령 종류 코드(법령 : 010101, 행정규칙 : 010102) |
