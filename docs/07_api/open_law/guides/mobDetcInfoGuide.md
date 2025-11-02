# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=detc&mobileYn=Y

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : detc(필수) | 서비스 대상 |
| ID | char(필수) | 헌재결정례 일련번호 |
| LM | string | 헌재결정례명 |
| type | string | 출력 형태 : HTML |
| mobileYn | char : Y (필수) | 모바일여부 |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=detc&ID=58386&type=HTML&mobileYn=Y
2. http://www.law.go.kr/DRF/lawService.do?OC=test&target=detc&ID=127830&&LM=자동차관리법제26조등위헌확인등&type=HTML&mobileYn=Y
