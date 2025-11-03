# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=couseLs

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : couseLs(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| vcode | string(필수) | 분류코드법령은 L로 시작하는 14자리 코드(L0000000000001) |
| display | int | 검색된 결과 개수 (default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |
| popYn | string | 상세화면 팝업창 여부(팝업창으로 띄우고 싶을 때만 'popYn=Y') |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=couseLs&type=XML&vcode=L0000000003384
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=couseLs&type=HTML&vcode=L0000000003384
3. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=couseLs&type=JSON&vcode=L0000000003384

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색서비스 대상 |
| vcode | string | 분류코드 |
| section | string | 검색범위 |
| totalCnt | int | 검색건수 |
| page | int | 결과페이지번호 |
| law id | int | 결과 번호 |
| 법령일련번호 | int | 법령일련번호 |
| 법령명한글 | string | 법령명한글 |
| 법령ID | int | 법령ID |
| 공포일자 | int | 공포일자 |
| 공포번호 | int | 공포번호 |
| 제개정구분명 | string | 제개정구분명 |
| 소관부처명 | string | 소관부처명 |
| 소관부처코드 | string | 소관부처코드 |
| 법령구분명 | string | 법령구분명 |
| 시행일자 | int | 시행일자 |
| 법령상세링크 | string | 법령상세링크 |
