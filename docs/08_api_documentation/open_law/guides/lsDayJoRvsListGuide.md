# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=lsJoHstInf

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : lsJoHstInf(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 XML/JSON |
| regDt | int | 조문 개정일, 8자리 (20150101) |
| fromRegDt | int | 조회기간 시작일, 8자리 (20150101) |
| toRegDt | int | 조회기간 종료일, 8자리 (20150101) |
| ID | int | 법령ID |
| JO | int | 조문번호조문번호 4자리 + 조 가지번호 2자리(000202 : 제2조의2) |
| org | string | 소관부처별 검색(소관부처코드 제공) |
| page | int | 검색 결과 페이지 (default=1) |

샘플 URL

1. http://law.go.kr/DRF/lawSearch.do?target=lsJoHstInf&OC=test&regDt=20250401&type=XML
2. http://www.law.go.kr/DRF/lawSearch.do?target=lsJoHstInf&OC=test&fromRegDt=20250101&type=XML
3. http://www.law.go.kr/DRF/lawSearch.do?target=lsJoHstInf&OC=test&regDt=20250401&org=1352000&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색서비스 대상 |
| totalCnt | int | 검색한 기간에 개정 조문이 있는 법령의 건수 |
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
| jo num | string | 조 구분 번호 |
| 조문정보 | string | 조문정보 |
| 조문번호 | string | 조문번호 |
| 변경사유 | string | 변경사유 |
| 조문링크 | string | 조문링크 |
| 조문변경이력상세링크 | string | 조문변경이력상세링크 |
| 조문개정일 | int | 조문제개정일 |
| 조문시행일 | int | 조문시행일 |
