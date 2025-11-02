# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=law&mobileYn=Y

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : law(필수) | 서비스 대상 |
| ID | char | 법령 ID (ID 또는 MST 중 하나는 반드시 입력) |
| MST | char | 법령 마스터 번호 법령테이블의 lsi_seq 값을 의미함 |
| LM | string | 법령의 법령명(법령명 입력시 해당 법령 링크) |
| LD | int | 법령의 공포일자 |
| LN | int | 법령의 공포번호 |
| JO | int:6 | 조번호 생략(기본값) : 모든 조를 표시함 6자리숫자 : 조번호(4자리)+조가지번호(2자리) (000200 : 2조, 001002 : 10조의 2) |
| PD | char | 부칙표시 ON일 경우 부칙 목록만 출력 생략할 경우 법령 + 부칙 표시 |
| PN | int | 부칙번호 해당 부칙번호에 해당하는 부칙 보기 |
| BD | char | 별표표시 생략(기본값) : 법령+별표 ON : 모든 별표 표시 |
| BT | int | 별표구분 별표표시가 on일 경우 값을 읽어들임 (별표=1/서식=2/별지=3/별도=4/부록=5) |
| BN | int | 별표번호 별표표시가 on일 경우 값을 읽어들임 |
| BG | int | 별표가지번호 별표표시가 on일 경우 값을 읽어들임 |
| type | char | 출력 형태 : HTML |
| mobileYn | char : Y (필수) | 모바일여부 |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=law&ID=1747&type=HTML&mobileYn=Y
2. http://www.law.go.kr/DRF/lawService.do?OC=test&target=law&MST=91689&type=HTML&mobileYn=Y
