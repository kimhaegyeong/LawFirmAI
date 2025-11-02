# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=drlaw

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : drlaw(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 HTML |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=drlaw&type=HTML
