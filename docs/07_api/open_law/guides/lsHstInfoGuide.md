# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=lsHistory

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : lsHistory(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML |
| ID | char | 법령 ID (ID 또는 MST 중 하나는 반드시 입력) |
| MST | char | 법령 마스터 번호 - 법령테이블의 lsi_seq 값을 의미함 |
| LM | string | 법령의 법령명(법령명 입력시 해당 법령 링크) |
| LD | int | 법령의 공포일자 |
| LN | int | 법령의 공포번호 |
| chrClsCd | char | 원문/한글 여부생략(기본값) : 한글(010202 : 한글, 010201 : 원문) |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=lsHistory&MST=9094&type=HTML
2. http://www.law.go.kr/DRF/lawService.do?OC=test&target=lsHistory&MST=166500&type=HTML
