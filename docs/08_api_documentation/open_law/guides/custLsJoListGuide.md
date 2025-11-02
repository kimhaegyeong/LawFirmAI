# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=couseLs

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : couseLs(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| vcode | string | 분류코드(필수)법령은 L로 시작하는 14자리 코드(L0000000000001) |
| lj=jo | string(필수) | 조문여부 |
| display | int | 검색된 결과 개수 (default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |
| popYn | string | 상세화면 팝업창 여부(팝업창으로 띄우고 싶을 때만 'popYn=Y') |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=couseLs&type=XML&lj=jo&vcode=L0000000003384
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=couseLs&type=HTML&lj=jo&vcode=L0000000003384
3. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=couseLs&type=JSON&lj=jo&vcode=L0000000003384

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색서비스 대상 |
| vcode | string | 분류코드 |
| section | string | 검색범위 |
| totalCnt | int | 페이지당 결과 수 |
| page | int | 페이지당 결과 수 |
| 법령 법령키 | int | 법령 법령키 |
| 법령ID | int | 법령ID |
| 법령명한글 | string | 법령명한글 |
| 공포일자 | int | 공포일자 |
| 공포번호 | int | 공포번호 |
| 제개정구분명 | string | 제개정구분명 |
| 법령구분명 | string | 법령구분명 |
| 시행일자 | int | 시행일자 |
| 조문번호 | int | 조문번호 |
| 조문가지번호 | int | 조문가지번호 |
| 조문제목 | string | 조문제목 |
| 조문시행일자 | int | 조문시행일자 |
| 조문제개정유형 | string | 조문제개정유형 |
| 조문제개정일자문자열 | string | 조문제개정일자문자열 |
| 조문상세링크 | string | 조문상세링크 |
