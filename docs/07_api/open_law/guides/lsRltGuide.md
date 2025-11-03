# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : https://www.law.go.kr/DRF/lawSearch.do?target=lsRlt

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상(관련법령 : lsRlt) |
| type | char(필수) | 출력 형태 : XML/JSON |
| query | string | 기준법령명에서 검색을 원하는 질의 |
| ID | int | 법령 ID |
| lsRltCd | int | 법령 간 관계 코드 |

샘플 URL

1. https://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lsRlt&type=XML
2. https://www.law.go.kr/DRF/lawSearch.do?OC=test&target=lsRlt&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| target | string | 검색서비스 대상 |
| 키워드 | string | 검색 단어 |
| 검색결과개수 | int | 검색 건수 |
| 기준법령ID | int | 기준법령 ID |
| 기준법령명 | string | 기준법령명 |
| 기준법령상세링크 | string | 기준법령 본문 조회링크 |
| 관련법령 id | string | 관련법령 순번 |
| 관련법령ID | int | 관련법령 ID |
| 관련법령명 | string | 관련법령명 |
| 법령간관계코드 | string | 법령간관계코드 |
| 법령간관계 | string | 법령간관계 |
| 관련법령상세링크 | string | 관련법령 본문 조회링크 |
| 관련법령조회링크 | string | 해당 관련법령을 기준법령으로 한 관련법령 정보 조회링크 |
