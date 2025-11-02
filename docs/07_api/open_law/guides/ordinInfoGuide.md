# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=ordin

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| type | char(필수) | 출력 형태 HTML/XML/JSON |
| target | string : ordin(필수) | 서비스 대상 |
| ID | char | 자치법규ID |
| MST | string | 자치법규 일련번호 |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=ordin&MST=1316146&type=HTML&mobileYn=
2. http://www.law.go.kr/DRF/lawService.do?OC=test&target=ordin&ID=2026666&type=XML&mobileYn=
3. http://www.law.go.kr/DRF/lawService.do?OC=test&target=ordin&ID=2251458&type=JSON&mobileYn=

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 자치법규ID | int | 자치법규ID |
| 자치법규일련번호 | string | 자치법규일련번호 |
| 공포일자 | string | 공포일자 |
| 공포번호 | string | 공포번호 |
| 자치법규명 | string | 자치법규명 |
| 시행일자 | string | 시행일자 |
| 자치법규종류 | string | 자치법규종류(C0001-조례 /C0002-규칙 /C0003-훈령/C0004-예규/C0006-기타/C0010-고시/C0011-의회규칙) |
| 지자체기관명 | string | 지자체기관명 |
| 자치법규발의종류 | string | 자치법규발의종류 |
| 담당부서명 | string | 담당부서명 |
| 전화번호 | string | 전화번호 |
| 제개정정보 | string | 제개정정보 |
| 조문번호 | string | 조문번호 |
| 조문여부 | string | 해당 조문이 조일때 Y,그 외 편,장,절,관 일때는 N |
| 조제목 | string | 조제목 |
| 조내용 | string | 조내용 |
| 부칙공포일자 | int | 부칙공포일자 |
| 부칙공포번호 | int | 부칙공포번호 |
| 부칙내용 | string | 부칙내용 |
| 부칙내용 | string | 부칙내용 |
| 별표 | string | 별표(자치법규 별표는 서울시교육청과 본청만 제공합니다.) |
| 별표번호 | int | 별표번호 |
| 별표가지번호 | int | 별표가지번호 |
| 별표구분 | string | 별표구분 |
| 별표제목 | string | 별표제목 |
| 별표첨부파일명 | string | 별표첨부파일명 |
| 별표내용 | string | 별표내용 |
| 개정문내용 | string | 개정문내용 |
| 제개정이유내용 | string | 제개정이유내용 |
