# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=oneview

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : oneview(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| MST | char | 법령 마스터 번호 - 법령테이블의 lsi_seq 값을 의미함 |
| LM | string | 법령의 법령명 |
| LD | int | 법령의 공포일자 |
| LN | int | 법령의 공포번호 |
| JO | int | 조번호생략(기본값) : 모든 조를 표시함6자리숫자 : 조번호(4자리)+조가지번호(2자리)(000200 : 2조, 001002 : 10조의 2) |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=oneview&type=HTML
2. http://www.law.go.kr/DRF/lawService.do?OC=test&target=oneview&type=XML
3. http://www.law.go.kr/DRF/lawService.do?OC=test&target=oneview&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 패턴일련번호 | int | 패턴일련번호 |
| 법령일련번호 | int | 법령일련번호 |
| 법령명 | string | 법령명 |
| 공포일자 | int | 공포일자 |
| 공포번호 | int | 공포번호 |
| 조문시행일자 | int | 조문시행일자 |
| 최초제공일자 | int | 최초제공일자 |
| 조번호 | int | 조번호 |
| 조제목 | string | 조제목 |
| 콘텐츠제목 | string | 콘텐츠제목 |
| 링크텍스트 | string | 링크텍스트 |
| 링크URL | string | 링크URL |
