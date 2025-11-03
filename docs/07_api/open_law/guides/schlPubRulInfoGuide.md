# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=school(or

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상(대학 : school / 지방공사공단 : public / 공공기관 : pi) |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| ID | char | 학칙공단 일련번호 |
| LID | char | 학칙공단 ID |
| LM | string | 학칙공단명조회하고자 하는 정확한 학칙공단명을 입력 |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=school&LID=2055825&type=HTML
2. http://www.law.go.kr/DRF/lawService.do?OC=test&target=school&ID=2200000075994&type=HTML
3. http://www.law.go.kr/DRF/lawService.do?OC=test&target=school&LID=2055825&type=XML
4. http://www.law.go.kr/DRF/lawService.do?OC=test&target=school&ID=2200000075994&type=XML
5. http://www.law.go.kr/DRF/lawService.do?OC=test&target=school&LID=2055825&type=JSON
6. http://www.law.go.kr/DRF/lawService.do?OC=test&target=school&ID=2200000075994&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 행정규칙일련번호 | int | 학칙공단 일련번호 |
| 행정규칙명 | string | 학칙공단명 |
| 행정규칙종류 | string | 학칙공단 종류 |
| 행정규칙종류코드 | string | 학칙공단 종류코드 |
| 발령일자 | int | 발령일자 |
| 발령번호 | string | 발령번호 |
| 제개정구분명 | string | 제개정구분명 |
| 제개정구분코드 | string | 제개정구분코드 |
| 조문형식여부 | string | 조문형식여부 |
| 행정규칙ID | int | 학칙공단ID |
| 소관부처명 | string | 소관부처명 |
| 소관부처코드 | string | 소관부처코드 |
| 담당부서기관코드 | string | 담당부서기관코드 |
| 담당부서기관명 | string | 담당부서기관명 |
| 담당자명 | string | 담당자명 |
| 전화번호 | string | 전화번호 |
| 현행여부 | string | 현행여부 |
| 생성일자 | string | 생성일자 |
| 조문내용 | string | 조문내용 |
| 부칙공포일자 | string | 부칙 공포일자 |
| 부칙공포번호 | string | 부칙 공포번호 |
| 부칙내용 | string | 부칙내용 |
| 별표단위 별표키 | string | 별표단위 별표키 |
| 별표번호 | string | 별표번호 |
| 별표가지번호 | string | 별표가지번호 |
| 별표구분 | string | 별표구분 |
| 별표제목 | string | 별표제목 |
| 별표서식파일링크 | string | 별표서식 파일링크 |
| 개정문내용 | string | 개정문내용 |
| 제개정이유내용 | string | 제개정이유내용 |
