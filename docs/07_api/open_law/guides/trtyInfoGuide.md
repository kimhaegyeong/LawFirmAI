# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=trty

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : trty(필수) | 서비스 대상 |
| type | char | 출력 형태 : HTML/XML/JSON |
| ID | char | 조약 ID |
| chrClsCd | char | 한글/영문 : 010202(한글)/ 010203(영문) (default = 010202) |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=trty&ID=983&type=HTML
2. http://www.law.go.kr/DRF/lawService.do?OC=test&target=trty&ID=2120&type=HTML
3. http://www.law.go.kr/DRF/lawService.do?OC=test&target=trty&ID=983&type=XML
4. http://www.law.go.kr/DRF/lawService.do?OC=test&target=trty&ID=2120&type=XML
5. http://www.law.go.kr/DRF/lawService.do?OC=test&target=trty&ID=983&type=JSON
6. http://www.law.go.kr/DRF/lawService.do?OC=test&target=trty&ID=2120&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 조약일련번호 | int | 조약일련번호 |
| 조약명_한글 | string | 조약명한글 |
| 조약명_영문 | string | 조약명영문 |
| 조약구분코드 | int | 조약구분코드(양자조약:440101, 다자조약:440102) |
| 대통령재가일자 | int | 대통령재가일자 |
| 발효일자 | int | 발효일자 |
| 조약번호 | int | 조약번호 |
| 관보게재일자 | int | 관보게재일자 |
| 국무회의심의일자 | int | 국무회의심의일자 |
| 국무회의심의회차 | int | 국무회의심의회차 |
| 국회비준동의여부 | string | 국회비준동의여부 |
| 국회비준동의일자 | string | 국회비준동의일자 |
| 서명일자 | int | 서명일자 |
| 서명장소 | string | 서명장소 |
| 비고 | string | 비고 |
| 추가정보 | string | 추가정보 |
| 체결대상국가 | string | 체결대상국가 |
| 체결대상국가한글 | string | 체결대상국가한글 |
| 우리측국내절차완료통보 | int | 우리측국내절차완료통보일 |
| 상대국측국내절차완료통보 | int | 상대국측국내절차완료통보일 |
| 양자조약분야코드 | int | 양자조약분야코드 |
| 양자조약분야명 | string | 양자조약분야명 |
| 제2외국어종류 | string | 제2외국어종류 |
| 국가코드 | string | 국가코드 |
| 조약내용 | string | 조약내용 |
| 체결일자 | string | 체결일자 |
| 체결장소 | string | 체결장소 |
| 기탁처 | string | 기탁처 |
| 다자조약분야코드 | string | 다자조약분야코드 |
| 다자조약분야명 | string | 다자조약분야명 |
| 수락서기탁일자 | string | 수락서기탁일자 |
| 국내발효일자 | string | 국내발효일자 |
