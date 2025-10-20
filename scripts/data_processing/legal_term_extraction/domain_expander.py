# -*- coding: utf-8 -*-
"""
도메인별 법률 용어 확장기
법률 분야별로 체계적인 용어 확장을 수행
"""

import json
import logging
from typing import Dict, List, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class DomainTermExpander:
    """도메인별 법률 용어 확장기"""
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 도메인별 용어 사전 정의
        self.domain_terms = {
            "형사법": {
                "범죄유형": [
                    "살인", "강도", "절도", "사기", "횡령", "배임", "뇌물", 
                    "강간", "성폭력", "성추행", "강제추행", "강제성교",
                    "폭행", "상해", "협박", "감금", "약취유인", "유괴",
                    "방화", "절도", "강도", "강도상해", "강도살인",
                    "교통사고", "음주운전", "무면허운전", "사고후미조치"
                ],
                "처벌형태": [
                    "형벌", "벌금", "징역", "금고", "집행유예", "선고유예", 
                    "보호관찰", "사회봉사", "수강명령", "기소유예", "공소유예",
                    "형의감경", "형의면제", "형의정지", "형의실효"
                ],
                "소송절차": [
                    "기소", "공소", "공소제기", "공소장", "공판", "증거조사", 
                    "선고", "항소", "상고", "재심", "비상상고", "특별항고",
                    "구속", "구속영장", "체포", "체포영장", "수사", "수사기관"
                ],
                "관련인물": [
                    "피고인", "검사", "변호인", "국선변호인", "증인", "피해자",
                    "고발인", "고소인", "참고인", "감정인", "통역인", "번역인",
                    "재판장", "판사", "법정경위", "법정서기"
                ]
            },
            "민사법": {
                "계약관계": [
                    "계약", "계약서", "약정", "합의", "계약관계", "계약체결",
                    "계약해지", "계약위반", "계약이행", "계약금", "위약금",
                    "매매계약", "임대차계약", "도급계약", "위임계약", "고용계약"
                ],
                "손해배상": [
                    "손해배상", "배상", "보상", "위자료", "정신적손해", "재산적손해",
                    "휴업손해", "치료비", "간병비", "교통비", "의료비", "장례비",
                    "과실상계", "과실비율", "과실", "고의", "과실상계"
                ],
                "소유권": [
                    "소유권", "점유권", "지상권", "전세권", "저당권", "질권", "유치권",
                    "소유권이전", "소유권보존", "소유권확인", "점유권확인", "점유권보존",
                    "등기", "등기부등본", "등기신청", "등기이전", "등기말소"
                ],
                "상속": [
                    "상속", "상속인", "피상속인", "유언", "유류분", "상속포기", "상속승인",
                    "상속재산", "상속분", "상속분할", "상속세", "상속등기",
                    "유언서", "유언집행", "유언집행자", "유언무효", "유언취소"
                ]
            },
            "가족법": {
                "혼인관계": [
                    "혼인", "이혼", "혼인무효", "혼인취소", "별거", "위자료",
                    "혼인신고", "혼인신고서", "혼인신고서류", "혼인신고서류",
                    "재혼", "재혼금지기간", "혼인신고서류", "혼인신고서류"
                ],
                "친자관계": [
                    "친생자", "양자", "친권", "양육권", "양육비", "면접교섭권",
                    "친생자관계", "양자관계", "친생자관계", "양자관계",
                    "친생자관계", "양자관계", "친생자관계", "양자관계"
                ],
                "재산관계": [
                    "재산분할", "재산분배", "부부재산", "특별재산", "공동재산",
                    "재산분할청구", "재산분할소송", "재산분할협의", "재산분할조정"
                ]
            },
            "상사법": {
                "회사법": [
                    "주식회사", "유한회사", "합명회사", "합자회사", "주주", "이사", "감사",
                    "주식", "주식발행", "주식양도", "주식매수", "주식매도",
                    "자본금", "자본금증자", "자본금감자", "자본금감자"
                ],
                "상행위": [
                    "상행위", "상인", "상호", "상업장부", "상업등기", "상업사용인",
                    "상업사용인", "상업사용인", "상업사용인", "상업사용인"
                ],
                "어음수표": [
                    "어음", "수표", "어음할인", "어음교환", "어음보증", "어음배서",
                    "어음할인", "어음교환", "어음보증", "어음배서"
                ]
            }
        }
        
        # 용어 관계 정의
        self.term_relationships = {
            "동의어": {},
            "관련용어": {},
            "상위개념": {},
            "하위개념": {},
            "대립개념": {}
        }
    
    def expand_domain_terms(self, extracted_terms: Dict) -> Dict:
        """도메인별 용어 확장"""
        self.logger.info("Starting domain term expansion")
        
        expanded_terms = {}
        
        for domain, categories in self.domain_terms.items():
            expanded_terms[domain] = {}
            
            for category, terms in categories.items():
                expanded_terms[domain][category] = terms
        
        # 추출된 용어와 도메인 용어 통합
        integrated_terms = self._integrate_extracted_and_domain_terms(
            extracted_terms, expanded_terms
        )
        
        return integrated_terms
    
    def _integrate_extracted_and_domain_terms(self, extracted_terms: Dict, domain_terms: Dict) -> Dict:
        """추출된 용어와 도메인 용어 통합"""
        integrated = {}
        
        # 도메인별로 용어 통합
        for domain, categories in domain_terms.items():
            integrated[domain] = {}
            
            for category, terms in categories.items():
                # 기존 용어와 새 용어 통합
                existing_terms = extracted_terms.get(category, [])
                all_terms = list(set(existing_terms + terms))
                
                integrated[domain][category] = all_terms
        
        return integrated
    
    def build_term_relationships(self, terms: Dict) -> Dict:
        """용어 관계 구축"""
        relationships = {
            "동의어": {},
            "관련용어": {},
            "상위개념": {},
            "하위개념": {},
            "대립개념": {}
        }
        
        # 기본 관계 정의
        relationships["동의어"] = {
            "손해배상": ["배상", "보상", "피해보상"],
            "계약": ["계약서", "약정", "합의"],
            "이혼": ["혼인해소", "혼인무효", "별거"],
            "소송": ["소송절차", "소송제기", "소송진행"],
            "형벌": ["처벌", "제재", "처분"]
        }
        
        relationships["관련용어"] = {
            "손해배상": ["불법행위", "채무불이행", "과실", "고의"],
            "계약": ["계약해지", "계약위반", "계약이행", "계약금"],
            "이혼": ["위자료", "재산분할", "양육비", "면접교섭권"],
            "소송": ["원고", "피고", "소장", "답변서", "증거"],
            "형벌": ["벌금", "징역", "금고", "집행유예"]
        }
        
        relationships["상위개념"] = {
            "손해배상": ["민사책임", "불법행위책임"],
            "계약": ["법률행위", "채권관계"],
            "이혼": ["혼인관계", "가족관계"],
            "소송": ["소송절차", "재판절차"],
            "형벌": ["형사처벌", "형사제재"]
        }
        
        relationships["하위개념"] = {
            "손해배상": ["재산적손해", "정신적손해", "위자료"],
            "계약": ["매매계약", "임대차계약", "도급계약"],
            "이혼": ["협의이혼", "재판이혼"],
            "소송": ["민사소송", "형사소송", "행정소송"],
            "형벌": ["벌금", "징역", "금고"]
        }
        
        relationships["대립개념"] = {
            "손해배상": ["정당행위", "자기방어", "긴급피난"],
            "계약": ["무효", "취소", "해지"],
            "이혼": ["혼인", "재혼"],
            "소송": ["화해", "조정", "중재"],
            "형벌": ["무죄", "기소유예", "공소유예"]
        }
        
        return relationships
    
    def generate_enhanced_dictionary(self, extracted_terms: Dict, domain_terms: Dict) -> Dict:
        """향상된 사전 생성"""
        enhanced_dict = {}
        
        # 모든 용어 수집
        all_terms = set()
        for pattern_terms in extracted_terms.values():
            all_terms.update(pattern_terms)
        
        for domain_categories in domain_terms.values():
            for category_terms in domain_categories.values():
                all_terms.update(category_terms)
        
        # 각 용어에 대한 정보 생성
        for term in all_terms:
            enhanced_dict[term] = {
                "synonyms": self._get_synonyms(term),
                "related_terms": self._get_related_terms(term),
                "related_laws": self._get_related_laws(term),
                "precedent_keywords": self._get_precedent_keywords(term),
                "domain": self._get_domain(term, domain_terms),
                "category": self._get_category(term, domain_terms),
                "confidence": self._calculate_confidence(term, extracted_terms),
                "frequency": self._get_frequency(term, extracted_terms)
            }
        
        return enhanced_dict
    
    def _get_synonyms(self, term: str) -> List[str]:
        """동의어 반환"""
        synonym_map = {
            "손해배상": ["배상", "보상", "피해보상"],
            "계약": ["계약서", "약정", "합의"],
            "이혼": ["혼인해소", "혼인무효", "별거"],
            "소송": ["소송절차", "소송제기", "소송진행"],
            "형벌": ["처벌", "제재", "처분"],
            "법원": ["법정", "재판부", "법원판결"],
            "검사": ["검찰", "검찰관", "검사장"],
            "변호사": ["변호인", "법정변호인", "국선변호인"],
            "피고인": ["피고", "피의자", "피고인"],
            "원고": ["원고인", "청구인", "신청인"],
            "판결": ["선고", "판정", "재판"],
            "소장": ["소송장", "청구서", "신청서"],
            "답변서": ["답변", "반박서", "이의서"],
            "증거": ["증명자료", "증명서류", "증명물"],
            "징역": ["형벌", "형", "처벌"],
            "벌금": ["과태료", "과료", "벌칙"],
            "집행유예": ["형의집행유예", "집행정지", "집행면제"],
            "보호관찰": ["보호처분", "보호조치", "보호명령"],
            "기소": ["공소", "공소제기", "기소제기"],
            "공판": ["공판정", "재판정", "법정"],
            "항소": ["상소", "항소제기", "상소제기"],
            "상고": ["최종상소", "상고제기", "최종항소"],
            "재심": ["재심청구", "재심제기", "재심신청"],
            "구속": ["구금", "구치", "구속영장"],
            "체포": ["체포영장", "긴급체포", "현행범체포"],
            "수사": ["수사기관", "수사절차", "수사진행"],
            "증인": ["증인신문", "증인출석", "증인보호"],
            "감정인": ["감정", "감정서", "감정의견"],
            "통역인": ["통역", "번역", "번역인"],
            "재판장": ["주심판사", "재판관", "판사"],
            "판사": ["법관", "재판관", "법정판사"],
            "법정경위": ["법정보안", "법정질서", "법정경비"],
            "법정서기": ["법정기록", "법정기록원", "법정서기관"]
        }
        return synonym_map.get(term, [])
    
    def _get_related_terms(self, term: str) -> List[str]:
        """관련 용어 반환"""
        related_map = {
            "손해배상": ["불법행위", "채무불이행", "과실", "고의", "인과관계", "책임능력"],
            "계약": ["계약해지", "계약위반", "계약이행", "계약금", "위약금", "손해배상"],
            "이혼": ["위자료", "재산분할", "양육비", "면접교섭권", "이혼사유", "혼인관계"],
            "소송": ["원고", "피고", "소장", "답변서", "증거", "판결", "항소", "상고"],
            "형벌": ["벌금", "징역", "금고", "집행유예", "선고유예", "보호관찰"],
            "법원": ["재판부", "법정", "판사", "재판장", "법정서기", "법정경위"],
            "검사": ["검찰", "검찰관", "검사장", "수사", "기소", "공소"],
            "변호사": ["변호인", "법정변호인", "국선변호인", "변호", "법정변호"],
            "피고인": ["피고", "피의자", "피고인", "피고인권", "피고인보호"],
            "원고": ["원고인", "청구인", "신청인", "원고권", "원고보호"],
            "판결": ["선고", "판정", "재판", "판결서", "판결문", "판결요지"],
            "소장": ["소송장", "청구서", "신청서", "소장제출", "소장기재"],
            "답변서": ["답변", "반박서", "이의서", "답변서제출", "답변서기재"],
            "증거": ["증명자료", "증명서류", "증명물", "증거조사", "증거신청"],
            "징역": ["형벌", "형", "처벌", "형의집행", "형의실효"],
            "벌금": ["과태료", "과료", "벌칙", "벌금납부", "벌금집행"],
            "집행유예": ["형의집행유예", "집행정지", "집행면제", "집행유예기간"],
            "보호관찰": ["보호처분", "보호조치", "보호명령", "보호관찰기간"],
            "기소": ["공소", "공소제기", "기소제기", "기소유예", "공소유예"],
            "공판": ["공판정", "재판정", "법정", "공판절차", "공판진행"],
            "항소": ["상소", "항소제기", "상소제기", "항소기간", "항소이유"],
            "상고": ["최종상소", "상고제기", "최종항소", "상고기간", "상고이유"],
            "재심": ["재심청구", "재심제기", "재심신청", "재심사유", "재심절차"],
            "구속": ["구금", "구치", "구속영장", "구속기간", "구속해제"],
            "체포": ["체포영장", "긴급체포", "현행범체포", "체포기간", "체포해제"],
            "수사": ["수사기관", "수사절차", "수사진행", "수사기간", "수사종료"],
            "증인": ["증인신문", "증인출석", "증인보호", "증인선서", "증인거부"],
            "감정인": ["감정", "감정서", "감정의견", "감정신청", "감정비용"],
            "통역인": ["통역", "번역", "번역인", "통역신청", "통역비용"],
            "재판장": ["주심판사", "재판관", "판사", "재판장권한", "재판장지휘"],
            "판사": ["법관", "재판관", "법정판사", "판사임명", "판사보수"],
            "법정경위": ["법정보안", "법정질서", "법정경비", "법정경위업무"],
            "법정서기": ["법정기록", "법정기록원", "법정서기관", "법정서기업무"]
        }
        return related_map.get(term, [])
    
    def _get_related_laws(self, term: str) -> List[str]:
        """관련 법률 반환"""
        law_map = {
            "손해배상": ["민법 제750조", "민법 제751조", "민법 제393조", "민법 제763조"],
            "계약": ["민법 제105조", "민법 제543조", "민법 제544조", "민법 제545조"],
            "이혼": ["민법 제840조", "민법 제841조", "민법 제842조", "민법 제843조"],
            "소송": ["민사소송법", "형사소송법", "행정소송법", "민사집행법"],
            "형벌": ["형법", "형사소송법", "형의집행법", "보호관찰법"],
            "법원": ["법원조직법", "법원행정처법", "법원공무원법"],
            "검사": ["검찰청법", "검사법", "검찰청법시행령"],
            "변호사": ["변호사법", "법무사법", "법정변호인법"],
            "피고인": ["형사소송법", "피고인권리보장법", "국선변호인법"],
            "원고": ["민사소송법", "행정소송법", "특허법"],
            "판결": ["민사소송법", "형사소송법", "행정소송법"],
            "소장": ["민사소송법", "민사소송규칙", "소송촉진법"],
            "답변서": ["민사소송법", "민사소송규칙", "소송촉진법"],
            "증거": ["민사소송법", "형사소송법", "증거법"],
            "징역": ["형법", "형의집행법", "형의집행규칙"],
            "벌금": ["형법", "형의집행법", "형의집행규칙"],
            "집행유예": ["형법", "형의집행법", "형의집행규칙"],
            "보호관찰": ["보호관찰법", "소년법", "형의집행법"],
            "기소": ["형사소송법", "검찰청법", "기소편의주의"],
            "공판": ["형사소송법", "민사소송법", "공판절차규칙"],
            "항소": ["민사소송법", "형사소송법", "행정소송법"],
            "상고": ["민사소송법", "형사소송법", "행정소송법"],
            "재심": ["민사소송법", "형사소송법", "재심법"],
            "구속": ["형사소송법", "구속영장법", "구속기간법"],
            "체포": ["형사소송법", "체포영장법", "긴급체포법"],
            "수사": ["형사소송법", "수사법", "수사기관법"],
            "증인": ["민사소송법", "형사소송법", "증인보호법"],
            "감정인": ["민사소송법", "형사소송법", "감정법"],
            "통역인": ["민사소송법", "형사소송법", "통역법"],
            "재판장": ["법원조직법", "민사소송법", "형사소송법"],
            "판사": ["법원조직법", "판사법", "법관법"],
            "법정경위": ["법원조직법", "법정경위법", "법원공무원법"],
            "법정서기": ["법원조직법", "법정서기법", "법원공무원법"]
        }
        return law_map.get(term, [])
    
    def _get_precedent_keywords(self, term: str) -> List[str]:
        """판례 키워드 반환"""
        precedent_map = {
            "손해배상": ["손해배상청구권", "배상책임", "손해배상소송", "배상액산정"],
            "계약": ["계약해지권", "계약위반", "계약이행청구", "계약금반환"],
            "이혼": ["이혼사유", "위자료청구", "재산분할청구", "양육비지급"],
            "소송": ["소송제기", "소송비용", "소송대리", "소송진행"],
            "형벌": ["형벌선고", "형벌집행", "형벌면제", "형벌감경"],
            "법원": ["법원판결", "법원결정", "법원명령", "법원처분"],
            "검사": ["검사기소", "검사수사", "검사처분", "검사명령"],
            "변호사": ["변호사선임", "변호사보수", "변호사윤리", "변호사법"],
            "피고인": ["피고인권리", "피고인보호", "피고인진술", "피고인증언"],
            "원고": ["원고권리", "원고보호", "원고진술", "원고증언"],
            "판결": ["판결선고", "판결문", "판결요지", "판결이유"],
            "소장": ["소장제출", "소장기재", "소장내용", "소장서식"],
            "답변서": ["답변서제출", "답변서기재", "답변서내용", "답변서서식"],
            "증거": ["증거조사", "증거신청", "증거제출", "증거능력"],
            "징역": ["징역선고", "징역집행", "징역면제", "징역감경"],
            "벌금": ["벌금선고", "벌금납부", "벌금집행", "벌금감경"],
            "집행유예": ["집행유예선고", "집행유예기간", "집행유예취소"],
            "보호관찰": ["보호관찰처분", "보호관찰기간", "보호관찰취소"],
            "기소": ["기소제기", "기소유예", "기소처분", "기소명령"],
            "공판": ["공판절차", "공판진행", "공판정", "공판기일"],
            "항소": ["항소제기", "항소기간", "항소이유", "항소심리"],
            "상고": ["상고제기", "상고기간", "상고이유", "상고심리"],
            "재심": ["재심청구", "재심사유", "재심절차", "재심심리"],
            "구속": ["구속영장", "구속기간", "구속해제", "구속연장"],
            "체포": ["체포영장", "체포기간", "체포해제", "체포연장"],
            "수사": ["수사절차", "수사진행", "수사기간", "수사종료"],
            "증인": ["증인신문", "증인출석", "증인보호", "증인거부"],
            "감정인": ["감정신청", "감정의견", "감정비용", "감정서"],
            "통역인": ["통역신청", "통역비용", "통역서", "통역의견"],
            "재판장": ["재판장권한", "재판장지휘", "재판장결정", "재판장명령"],
            "판사": ["판사임명", "판사보수", "판사윤리", "판사법"],
            "법정경위": ["법정경위업무", "법정경위권한", "법정경위지휘"],
            "법정서기": ["법정서기업무", "법정서기권한", "법정서기지휘"]
        }
        return precedent_map.get(term, [])
    
    def _get_domain(self, term: str, domain_terms: Dict) -> str:
        """용어의 도메인 반환"""
        for domain, categories in domain_terms.items():
            for category, terms in categories.items():
                if term in terms:
                    return domain
        return "기타"
    
    def _get_category(self, term: str, domain_terms: Dict) -> str:
        """용어의 카테고리 반환"""
        for domain, categories in domain_terms.items():
            for category, terms in categories.items():
                if term in terms:
                    return category
        return "기타"
    
    def _calculate_confidence(self, term: str, extracted_terms: Dict) -> float:
        """용어 신뢰도 계산"""
        # 기본 신뢰도
        confidence = 0.5
        
        # 추출된 용어인 경우 신뢰도 증가
        for pattern_terms in extracted_terms.values():
            if term in pattern_terms:
                confidence += 0.3
                break
        
        # 관련 정보가 있는 경우 신뢰도 증가
        if self._get_synonyms(term):
            confidence += 0.1
        if self._get_related_laws(term):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _get_frequency(self, term: str, extracted_terms: Dict) -> int:
        """용어 빈도 반환"""
        frequency = 0
        for pattern_terms in extracted_terms.values():
            frequency += pattern_terms.count(term)
        return frequency
    
    def save_enhanced_dictionary(self, enhanced_dict: Dict, output_path: str):
        """향상된 사전 저장"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_dict, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Enhanced dictionary saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving enhanced dictionary: {e}")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='도메인별 법률 용어 확장기')
    parser.add_argument('--input_file', type=str, required=True,
                       help='추출된 용어 파일 경로')
    parser.add_argument('--output_file', type=str,
                       default='data/extracted_terms/domain_expanded_terms.json',
                       help='출력 파일 경로')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='로그 레벨')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 도메인 확장 실행
    expander = DomainTermExpander()
    
    # 입력 파일 로드
    with open(args.input_file, 'r', encoding='utf-8') as f:
        extracted_data = json.load(f)
    
    extracted_terms = extracted_data.get('filtered_terms', {})
    
    # 도메인별 용어 확장
    domain_terms = expander.expand_domain_terms(extracted_terms)
    
    # 향상된 사전 생성
    enhanced_dict = expander.generate_enhanced_dictionary(extracted_terms, domain_terms)
    
    # 결과 저장
    expander.save_enhanced_dictionary(enhanced_dict, args.output_file)
    
    # 결과 출력
    print("\n=== 도메인별 용어 확장 결과 ===")
    print(f"총 용어 수: {len(enhanced_dict)}")
    
    print("\n=== 도메인별 용어 수 ===")
    domain_counts = {}
    for term, info in enhanced_dict.items():
        domain = info.get('domain', '기타')
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    for domain, count in domain_counts.items():
        print(f"{domain}: {count}개")


if __name__ == "__main__":
    main()
