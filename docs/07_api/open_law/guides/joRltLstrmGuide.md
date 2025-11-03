# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : https://www.law.go.kr/DRF/lawService.do?target=joRltLstrm

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상(조문-법령용어 연계 : joRltLstrm) |
| type | char(필수) | 출력 형태 : XML/JSON |
| query | string | 법령명에서 검색을 원하는 질의(query 또는 ID 중 하나는 반드시 입력) |
| ID | int | 법령 ID |
| JO | int(필수) | 조번호조번호 4자리 + 조가지번호 2자리(000200 : 2조, 000202 : 제2조의2) |

샘플 URL

1. https://www.law.go.kr/DRF/lawService.do?OC=test&target=joRltLstrm&type=XML&query=상법시행법&JO=000400
2. https://www.law.go.kr/DRF/lawService.do?OC=test&target=joRltLstrm&type=JSON&query=상법시행법&JO=000400

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색서비스 대상 |
| 키워드 | string | 검색 단어 |
| 검색결과개수 | int | 검색 건수 |
| 법령조문 id | string | 법령조문 순번 |
| 법령명 | string | 법령명 |
| 조번호 | int | 조번호 |
| 조가지번호 | int | 조가지번호 |
| 조문내용 | string | 조문내용 |
| 연계용어 id | string | 연계용어 순번 |
| 법령용어명 | string | 법령용어명 |
| 비고 | string | 동음이의어 내용 |
| 용어구분코드 | string | 용어구분코드 |
| 용어구분 | string | 용어구분명 |
| 용어간관계링크 | string | 법령용어-일상용어 연계 정보 상세링크 |
| 용어연계조문링크 | string | 법령용어-조문 연계 정보 상세링크 |
