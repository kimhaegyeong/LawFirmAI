# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=lsStmd

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : lsStmd(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| ID | char | 법령 ID (ID 또는 MST 중 하나는 반드시 입력) |
| MST | char | 법령 마스터 번호 - 법령테이블의 lsi_seq 값을 의미함 |
| LM | string | 법령의 법령명(법령명 입력시 해당 법령 링크) |
| LD | int | 법령의 공포일자 |
| LN | int | 법령의 공포번호 |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=lsStmd&MST=142362&type=HTML
2. http://www.law.go.kr/DRF/lawService.do?OC=test&target=lsStmd&MST=142591&type=HTML
3. http://www.law.go.kr/DRF/lawService.do?OC=test&target=lsStmd&MST=142362&type=XML
4. http://www.law.go.kr/DRF/lawService.do?OC=test&target=lsStmd&MST=142362&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 기본정보 | string | 기본정보 |
| 법령ID | int | 법령ID |
| 법령일련번호 | int | 법령일련번호 |
| 공포일자 | int | 공포일자 |
| 공포번호 | int | 공포번호 |
| 법종구분 | string | 법종구분 |
| 법령명 | string | 법령 |
| 시행일자 | int | 시행일자 |
| 제개정구분 | string | 제개정구분 |
| 상하위법 | string | 상하위법 |
| 법률 | string | 법률 |
| 시행령 | string | 시행령 |
| 시행규칙 | string | 시행규칙 |
| 본문 상세링크 | string | 본문 상세링크 |
