# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=lsJoHstInf

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : lsJoHstInf(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : XML/JSON |
| ID | char(필수) | 법령 ID |
| JO | int(필수) | 조번호6자리숫자 : 조번호(4자리)+조가지번호(2자리)(000200 : 2조, 001002 : 10조의 2) |
| display | int | 검색된 결과 개수 (default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=lsJoHstInf&ID=001971&JO=000500&type=XML
2. http://www.law.go.kr/DRF/lawService.do?OC=test&target=lsJoHstInf&ID=001971&JO=000500&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 법령ID | int | 법령ID |
| 법령명한글 | int | 법령명(한글) |
| 법령일련번호 | int | 법령일련번호 |
| 공포일자 | int | 공포일자 |
| 공포번호 | int | 공포번호 |
| 제개정구분명 | string | 제개정구분명 |
| 소관부처명 | string | 소관부처명 |
| 소관부처코드 | string | 소관부처코드 |
| 법령구분명 | string | 법령구분명 |
| 시행일자 | int | 시행일자 |
| 조문번호 | int | 조문번호 |
| 변경사유 | int | 변경사유 |
| 조문링크 | int | 변경사유 |
| 조문변경일 | int | 조문변경일 |
