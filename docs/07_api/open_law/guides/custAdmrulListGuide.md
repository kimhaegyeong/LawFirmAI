# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=couseAdmrul

요청 변수 (request parameter)

| 샘플 URL |
| --- |
| 1. 분류코드가 A0000000000601인 행정규칙 맞춤형 분류 목록 검색 |
| http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=couseAdmrul&type=XML&vcode=A0000000000601 |
| 2. 분류코드가 A0000000000601인 행정규칙 맞춤형 분류 HTML 목록 검색 |
| http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=couseAdmrul&type=HTML&vcode=A0000000000601 |
| 3. 분류코드가 A0000000000601인 행정규칙 맞춤형 분류 JSON 목록 검색 |
| http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=couseAdmrul&type=JSON&vcode=A0000000000601 |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=couseAdmrul&type=XML&vcode=A0000000000601
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=couseAdmrul&type=HTML&vcode=A0000000000601
3. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=couseAdmrul&type=JSON&vcode=A0000000000601

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색서비스 대상 |
| vcode | string | 분류코드 |
| section | string | 검색범위 |
| totalCnt | int | 검색건수 |
| page | int | 결과페이지번호 |
| admrul id | int | 검색 결과 순번 |
| 행정규칙일련번호 | int | 행정규칙일련번호 |
| 행정규칙명 | string | 행정규칙명 |
| 행정규칙ID | int | 행정규칙ID |
| 발령일자 | int | 발령일자 |
| 발령번호 | int | 발령번호 |
| 행정규칙구분명 | string | 행정규칙구분명 |
| 소관부처코드 | int | 소관부처코드 |
| 소관부처명 | string | 소관부처명 |
| 제개정구분명 | string | 제개정구분명 |
| 행정규칙상세링크 | string | 행정규칙상세링크 |
