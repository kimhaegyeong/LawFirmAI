# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : https://www.law.go.kr/DRF/lawService.do?target=nhrck

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상(국가인권위원회 : nhrck) |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| ID | char(필수) | 결정문 일련번호 |
| LM | char | 결정문명 |
| fields | string | 응답항목 옵션(사건명, 사건번호, ...)*빈 값일 경우 전체 항목 표출*출력 형태 HTML일 경우 적용 불가능 |

샘플 URL

1. https://www.law.go.kr/DRF/lawService.do?OC=test&target=nhrck&ID=331&type=HTML
2. https://www.law.go.kr/DRF/lawService.do?OC=test&target=nhrck&ID=335&type=XML
3. https://www.law.go.kr/DRF/lawService.do?OC=test&target=nhrck&ID=3157&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 결정문일련번호 | int | 결정문일련번호 |
| 기관명 | string | 기관명 |
| 위원회명 | string | 위원회명 |
| 사건명 | string | 사건명 |
| 사건번호 | string | 사건번호 |
| 의결일자 | string | 의결일자 |
| 주문 | string | 주문 |
| 이유 | string | 이유 |
| 위원정보 | string | 위원정보 |
| 별지 | string | 별지 |
| 결정요지 | string | 결정요지 |
| 판단요지 | string | 판단요지 |
| 주문요지 | string | 주문요지 |
| 분류명 | string | 분류명 |
| 결정유형 | string | 결정유형 |
| 신청인 | string | 신청인 |
| 피신청인 | string | 피신청인 |
| 피해자 | string | 피해자 |
| 피조사자 | string | 피조사자 |
| 원본다운로드URL | string | 원본다운로드URL |
| 바로보기URL | string | 바로보기URL |
| 결정례전문 | string | 결정례전문 |
| 데이터기준일시 | string | 데이터기준일시 |
