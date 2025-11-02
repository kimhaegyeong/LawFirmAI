# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=lsAbrv

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : lsAbrv(필수) | 서비스 대상 |
| type | char | 출력 형태 : XML/JSON |
| stdDt | int | 등록일(검색시작날짜) |
| endDt | int | 등록일(검색종료날짜) |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lsAbrv&type=XML
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lsAbrv&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색서비스 대상 |
| totalCnt | int | 검색건수 |
| law id | int | 결과 번호 |
| 법령일련번호 | int | 법령일련번호 |
| 현행연혁코드 | string | 현행연혁코드 |
| 법령명한글 | string | 법령명한글 |
| 법령약칭명 | string | 법령약칭명 |
| 법령ID | int | 법령ID |
| 공포일자 | int | 공포일자 |
| 공포번호 | int | 공포번호 |
| 제개정구분명 | string | 제개정구분명 |
| 소관부처명 | string | 소관부처명 |
| 소관부처코드 | string | 소관부처코드 |
| 법령구분명 | string | 법령구분명 |
| 시행일자 | int | 시행일자 |
| 등록일 | int | 등록일 |
| 자법타법여부 | string | 자법타법여부 |
| 법령상세링크 | string | 법령상세링크 |
