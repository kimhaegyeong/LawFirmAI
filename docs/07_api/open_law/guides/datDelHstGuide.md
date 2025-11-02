# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=delHst

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상 : datDel |
| type | string | 출력 형태 XML/JSON |
| display | int | 검색된 결과 개수 (default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |
| knd | int | 데이터 종류법령 : 1행정규칙 : 2자치법규 : 3학칙공단 : 13 |
| delDt | int | 데이터 삭제 일자 검색 (YYYYMMDD) |
| frmDttoDt | int | 데이터 삭제 일자 범위 검색 (YYYYMMDD) |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=delHst&type=XML
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=delHst&knd=13&delDt=20231013&type=XML
3. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=delHst&knd=3&frmDt=20231013&toDt=20231016&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색서비스 대상 |
| totalCnt | int | 검색건수 |
| page | int | 결과페이지번호 |
| law id | int | 결과 번호 |
| 일련번호 | int | 데이터 일련번호 |
| 구분명 | string | 데이터 구분명 (법령 / 행정규칙 / 자치법규 등) |
| 삭제일자 | string | 데이터 삭제일자 |
