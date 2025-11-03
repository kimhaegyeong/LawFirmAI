# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=decc&mobileYn=Y

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : decc(필수) | 서비스 대상 |
| ID | char | 행정심판례 일련번호 |
| LM | char | 행정심판례명 |
| type | char | 출력 형태 : HTML |
| mobileYn | char : Y (필수) | 모바일여부 |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=decc&ID=2782&type=HTML&mobileYn=Y
2. http://www.law.go.kr/DRF/lawService.do?OC=test&target=decc&ID=222883&LM=산림기술자
