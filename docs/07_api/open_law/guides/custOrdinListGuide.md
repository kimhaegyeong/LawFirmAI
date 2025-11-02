# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=couseOrdin

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : couseOrdin(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| vcode | string(필수) | 분류코드자치법규는 O로 시작하는 14자리 코드(O0000000000001) |
| display | int | 검색된 결과 개수 (default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |
| popYn | string | 상세화면 팝업창 여부(팝업창으로 띄우고 싶을 때만 'popYn=Y') |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=couseOrdin&type=XML&vcode=O0000000000602
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=couseOrdin&type=HTML&vcode=O0000000000602
3. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=couseOrdin&type=JSON&vcode=O0000000000602

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색서비스 대상 |
| vcode | string | 분류코드 |
| section | string | 검색범위 |
| totalCnt | int | 검색건수 |
| page | int | 결과페이지번호 |
| ordin id | int | 결과 번호 |
| 자치법규일련번호 | int | 자치법규일련번호 |
| 자치법규명 | string | 자치법규명 |
| 자치법규ID | int | 자치법규ID |
| 공포일자 | int | 공포일자 |
| 공포번호 | int | 공포번호 |
| 제개정구분명 | string | 제개정구분명 |
| 자치법규종류 | string | 자치법규종류 |
| 지자체기관명 | string | 지자체기관명 |
| 시행일자 | int | 시행일자 |
| 자치법규분야명 | string | 자치법규분야명 |
| 자치법규상세링크 | string | 자치법규상세링크 |
