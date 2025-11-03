# 국가법령정보 공동활용 LAW OPEN DATA

- 요청 URL : https://www.law.go.kr/DRF/lawService.do?target=iaciac

요청 변수 (request parameter)

| 요청변수 | 값 | 설명 |
| --- | --- | --- |
| OC | string(필수) | 사용자 이메일의 ID(g4c@korea.kr일경우 OC값=g4c) |
| target | string(필수) | 서비스 대상(산업재해보상보험재심사위원회 : iaciac) |
| type | char(필수) | 출력 형태 : HTML/XML/JSON |
| ID | char(필수) | 결정문 일련번호 |

샘플 URL

1. https://www.law.go.kr/DRF/lawService.do?OC=test&target=iaciac&ID=7515&type=HTML
2. https://www.law.go.kr/DRF/lawService.do?OC=test&target=iaciac&ID=7513&type=XML
3. https://www.law.go.kr/DRF/lawService.do?OC=test&target=iaciac&ID=12713&type=JSON

출력 결과 필드(response field)

| 필드 | 값 | 설명 |
| --- | --- | --- |
| 결정문일련번호 | int | 결정문 일련번호 |
| 사건대분류 | string | 사건 대분류 |
| 사건중분류 | string | 사건 중분류 |
| 사건소분류 | string | 사건 소분류 |
| 쟁점 | string | 쟁점 |
| 사건번호 | string | 사건번호 |
| 의결일자 | string | 의결일자 |
| 사건 | string | 사건 |
| 청구인 | string | 청구인 |
| 재해근로자 | string | 재해근로자 |
| 재해자 | string | 재해자 |
| 피재근로자 | string | 피재근로자/피재자성명/피재자/피재자(망인) |
| 진폐근로자 | string | 진폐근로자 |
| 수진자 | string | 수진자 |
| 원처분기관 | string | 원처분기관 |
| 주문 | string | 주문 |
| 청구취지 | string | 청구취지 |
| 이유 | string | 이유 |
| 별지 | string | 별지 |
| 문서제공구분 | string | 문서제공구분(데이터 개방\|이유하단 이미지개방) |
| 각주번호 | int | 각주번호 |
| 각주내용 | string | 각주내용 |
