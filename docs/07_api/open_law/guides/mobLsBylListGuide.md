# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : http://www.law.go.kr/DRF/lawSearch.do?target=licbyl&mobileYn=Y

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string : licbyl(필수) | 서비스 대상 |
| type | char | 출력 형태 HTML/XML/JSON |
| search | int | "검색범위 (기본 : 1 별표서식명) 2 : 해당법령검색 3 : 별표본문검색" |
| query | string | 법령명에서 검색을 원하는 질의(default=*) |
| display | int | 검색된 결과 개수 (default=20 max=100) |
| page | int | 검색 결과 페이지 (default=1) |
| sort | string | "정렬옵션 (기본 : lasc 별표서식명 오름차순) ldes 별표서식명 내림차순" |
| org | string | 소관부처별 검색(소관부처코드 제공) |
| knd | string | 별표종류1 : 별표 2 : 서식 3 : 별지 4 : 별도 5 : 부록 |
| gana | string | 사전식 검색(ga,na,da…,etc) |
| mobileYn | char : Y (필수) | 모바일여부 |

샘플 URL

1. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=licbyl&type=XML&mobileYn=Y
2. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=licbyl&type=HTML&mobileYn=Y
3. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=licbyl&type=JSON&mobileYn=Y
4. http://www.law.go.kr/DRF/lawSearch.do?OC=test&target=licbyl&type=XML&org=1320000&mobileYn=Y
