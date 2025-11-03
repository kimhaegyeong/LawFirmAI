# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=expc

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : expc(필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| ID | int(필수) | 법령해석례 일련번호 |
| LM | string | 법령해석례명 |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=expc&ID=334617&type=HTML
2. http://www.law.go.kr/DRF/lawService.do?OC=test&target=expc&ID=315191&LM=여성가족부
3. http://www.law.go.kr/DRF/lawService.do?OC=test&target=expc&ID=330471&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 법령해석례일련번호 | int | 법령해석례일련번호 |
| 안건명 | string | 안건명 |
| 안건번호 | string | 안건번호 |
| 해석일자 | int | 해석일자 |
| 해석기관코드 | int | 해석기관코드 |
| 해석기관명 | string | 해석기관명 |
| 질의기관코드 | int | 질의기관코드 |
| 질의기관명 | string | 질의기관명 |
| 관리기관코드 | int | 관리기관코드 |
| 등록일시 | int | 등록일시 |
| 질의요지 | string | 질의요지 |
| 회답 | string | 회답 |
| 이유 | string | 이유 |
