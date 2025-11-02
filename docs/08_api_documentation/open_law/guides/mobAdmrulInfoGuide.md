# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=admrul&mobileYn=Y

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : admrul(필수) | 서비스 대상 |
| ID | char | 행정규칙 일련번호 |
| LM | Char | 행정규칙명 조회하고자 하는 정확한 행정규칙명을 입력 |
| type | Char | 출력 형태 : HTML |
| mobileYn | char : Y (필수) | 모바일여부 |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=admrul&ID=62505&type=HTML&mobileYn=Y
