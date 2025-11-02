# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawService.do?target=lsDelegated

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : lsDelegated (필수) | 서비스 대상 |
| type | char(필수) | 출력 형태 XML/JSON |
| ID | char | 법령 ID (ID 또는 MST 중 하나는 반드시 입력,ID로 검색하면 그 법령의 현행 법령 본문 조회) |
| MST | char | 법령 마스터 번호 - 법령테이블의 lsi_seq 값을 의미함 |

샘플 URL

1. http://www.law.go.kr/DRF/lawService.do?OC=test&target=lsDelegated&type=XML&ID=000900
2. http://www.law.go.kr/DRF/lawService.do?OC=test&target=lsDelegated&type=JSON&ID=000900

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 법령일련번호 | int | 법령일련번호 |
| 법령명 | string | 법령명 |
| 법령ID | int | 법령ID |
| 공포일자 | int | 공포일자 |
| 공포번호 | int | 공포번호 |
| 소관부처코드 | int | 소관부처코드 |
| 전화번호 | string | 전화번호 |
| 시행일자 | int | 시행일자 |
| 조문번호 | string | 조문번호 |
| 조문제목 | string | 조문제목 |
| 위임구분 | string | 위임된 법령의 종류 |
| 위임법령일련번호 | string | 위임된 법령의 일련번호 |
| 위임법령제목 | string | 위임된 법령의 제목 |
| 위임법령조문번호 | string | 위임된 법령의 조문번호 |
| 위임법령조문가지번호 | string | 위임된 법령의 조문 가지번호 |
| 위임법령조문제목 | string | 위임된 법령의 조문 제목 |
| 링크텍스트 | string | 위임된 법령에 대한 링크를 걸어줘야 하는 텍스트 |
| 라인텍스트 | string | 링크텍스트가 포함된 텍스트(조문내용) |
| 조항호목 | string | 링크텍스트와 라인텍스트가 포함된 조항호목 |
| 위임행정규칙일련번호 | string | 위임된 행정규칙의 일련번호 |
| 위임행정규칙제목 | string | 위임된 행정규칙의 제목 |
| 링크텍스트 | string | 위임된 행정규칙에 대한 링크를 걸어줘야 하는 텍스트 |
| 라인텍스트 | string | 링크텍스트가 포함된 텍스트(조문내용) |
| 조항호목 | string | 링크텍스트와 라인텍스트가 포함된 조항호목 |
| 위임자치법규일련번호 | string | 위임된 자치법규의 일련번호 |
| 위임자치법규제목 | string | 위임된 자치법규의 제목 |
| 링크텍스트 | string | 위임된 자치법규에 대한 링크를 걸어줘야 하는 텍스트 |
| 라인텍스트 | string | 링크텍스트가 포함된 텍스트(조문내용) |
| 조항호목 | string | 링크텍스트와 라인텍스트가 포함된 조항호목 |
