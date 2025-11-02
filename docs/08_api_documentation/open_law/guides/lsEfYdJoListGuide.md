# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=eflawjosub

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : eflawjosub(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| ID | char | 법령 ID (ID 또는 MST 중 하나는 반드시 입력) |
| MST | char | 법령 마스터 번호 -법령테이블의 lsi_seq 값을 의미함 |
| efYd | int(필수) | 법령의 시행일자(ID 입력시에는 무시하는 값으로 입력하지 않음) |
| JO | char(필수) | 조 번호 6자리숫자예) 제2조 : 000200, 제10조의2 : 001002 |
| HANG | char | 항 번호 6자리숫자예) 제2항 : 000200 |
| HO | char | 호 번호 6자리숫자예) 제2호 : 000200, 제10호의2 : 001002 |
| MOK | char | 목 한자리 문자예) 가,나,다,라, … 카,타,파,하한글은 인코딩 하여 사용하여야 정상적으로 사용이가능URLDecoder.decode("다", "UTF-8") |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=eflawjosub&type=XML&MST=193412&efYd=20171019&JO=000300&HANG=000100&HO=000200&MOK=다
2. http://www.law.go.kr/DRF/lawService.do?OC=test&target=eflawjosub&type=HTML&MST=193412&efYd=20171019&JO=000300&HANG=000100&HO=000200&MOK=다
3. http://www.law.go.kr/DRF/lawService.do?OC=test&target=eflawjosub&type=JSON&MST=193412&efYd=20171019&JO=000300&HANG=000100&HO=000200&MOK=다

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 법령키 | int | 법령키 |
| 법령ID | int | 법령ID |
| 공포일자 | int | 공포일자 |
| 공포번호 | int | 공포번호 |
| 언어 | string | 언어 |
| 법종구분 | string | 법종구분 |
| 법종구분 코드 | string | 법종구분 코드 |
| 법령명_한글 | string | 법령명을 한글로 제공 |
| 법령명_한자 | string | 법령명을 한자로 제공 |
| 법령명_영어 | string | 법령명을 영어로 제공 |
| 편장절관 | int | 편장절관 |
| 소관부처코드 | int | 소관부처 코드 |
| 소관부처 | string | 소관부처명 |
| 전화번호 | string | 전화번호 |
| 시행일자 | int | 시행일자 |
| 제개정구분 | string | 제개정구분명 |
| 제안구분 | string | 제안구분 |
| 의결구분 | string | 의결구분 |
| 적용시작일자 | string | 적용시작일자 |
| 적용종료일자 | string | 적용종료일자 |
| 이전법령명 | string | 이전법령명 |
| 조문시행일자문자열 | string | 조문시행일자문자열 |
| 별표시행일자문자열 | string | 별표시행일자문자열 |
| 별표편집여부 | string | 별표편집여부 |
| 공포법령여부 | string | 공포법령여부(Y값이 있으면 해당 법령은 공포법령임) |
| 조문번호 | int | 조문번호 |
| 조문여부 | string | 조문여부 |
| 조문제목 | string | 조문제목 |
| 조문시행일자 | string | 조문시행일자 |
| 조문이동이전 | int | 조문이동이전번호 |
| 조문이동이후 | int | 조문이동이후번호 |
| 조문변경여부 | string | 조문변경여부(Y값이 있으면 해당 조문내에 변경 내용 있음 ) |
| 조문내용 | string | 조문내용 |
| 항번호 | int | 항번호 |
| 항내용 | string | 항내용 |
| 호번호 | int | 호번호 |
| 호내용 | string | 호내용 |
| 목번호 | string | 목번호 |
| 목내용 | string | 목내용 |
