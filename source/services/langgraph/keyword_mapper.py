# -*- coding: utf-8 -*-
"""
법률 질문별 필수 키워드 매핑 시스템
답변 품질 향상을 위한 키워드 포함도 개선
"""

import re
from typing import List, Dict, Set


class LegalKeywordMapper:
    """법률 질문별 필수 키워드 매핑"""
    
    # 질문 유형별 키워드 매핑
    KEYWORD_MAPPING = {
        "contract_review": {
            "계약서": ["당사자", "목적", "조건", "기간", "해지", "손해배상", "계약금", "위약금", "체결", "효력", "무효", "취소", "해제"],
            "계약": ["체결", "효력", "무효", "취소", "해제", "해지", "이행", "채무", "채권"],
            "매매": ["대금", "인도", "등기", "하자", "담보책임", "매도인", "매수인", "소유권이전"],
            "임대": ["임대료", "보증금", "임대차", "임차인", "임대인", "계약갱신"],
            "도급": ["도급", "하도급", "완성", "인수", "대금", "하자담보책임"]
        },
        "family_law": {
            "이혼": ["협의이혼", "조정이혼", "재판이혼", "가정법원", "신청", "위자료", "재산분할", "양육비", "면접교섭권"],
            "상속": ["상속인", "상속분", "유류분", "상속포기", "한정승인", "상속재산", "유언", "유증"],
            "양육": ["양육권", "친권", "면접교섭권", "양육비", "양육비지급", "양육비이행"],
            "입양": ["입양", "입양신고", "입양허가", "입양취소", "입양무효"],
            "친생자": ["친생자", "인지", "인지청구", "친생자관계존부확인"]
        },
        "criminal_law": {
            "절도": ["타인의", "재물", "절취", "의사", "형법", "329조", "불법영득의사", "고의"],
            "사기": ["기망", "착오", "재산상", "손해", "형법", "347조", "사기죄", "편취"],
            "폭행": ["폭행", "상해", "협박", "형법", "260조", "폭행죄", "상해죄", "협박죄"],
            "강도": ["강도", "강도죄", "강도상해", "강도살인", "형법", "333조"],
            "강간": ["강간", "강간죄", "강제추행", "성폭력", "형법", "297조"],
            "살인": ["살인", "살인죄", "형법", "250조", "고의", "미수"],
            "횡령": ["횡령", "횡령죄", "형법", "355조", "타인소유", "불법영득의사"]
        },
        "civil_law": {
            "손해배상": ["손해", "가해", "인과관계", "청구", "소송", "합의", "과실", "불법행위", "민법", "750조"],
            "계약": ["계약", "계약체결", "계약이행", "계약위반", "계약해제", "계약해지", "민법", "537조"],
            "불법행위": ["불법행위", "과실", "손해", "인과관계", "민법", "750조", "손해배상"],
            "채권": ["채권", "채무", "이행", "이행지체", "이행불능", "민법", "387조"],
            "소유권": ["소유권", "점유", "취득시효", "소유권이전", "민법", "186조"],
            "담보": ["담보", "담보물권", "저당권", "질권", "유치권", "민법", "342조"]
        },
        "labor_law": {
            "해고": ["부당해고", "노동위원회", "구제신청", "원직복직", "임금상당액", "해고사유", "해고절차"],
            "임금": ["임금", "최저임금", "임금지급", "임금체불", "임금채권", "임금보전"],
            "근로시간": ["근로시간", "휴게시간", "휴일", "연장근로", "야간근로", "휴일근로"],
            "휴가": ["휴가", "연차휴가", "경조휴가", "출산휴가", "육아휴직"],
            "산업재해": ["산업재해", "산재", "산재보험", "요양급여", "휴업급여", "장해급여"],
            "노동조합": ["노동조합", "단체교섭", "단체협약", "파업", "쟁의행위"]
        },
        "property_law": {
            "부동산": ["부동산", "토지", "건물", "등기", "등기부등본", "소유권이전등기"],
            "매매": ["매매계약", "매매대금", "계약금", "중도금", "잔금", "인도", "등기이전"],
            "임대": ["임대차", "임대료", "보증금", "임대차계약", "전세", "월세"],
            "등기": ["등기", "등기부등본", "등기이전", "등기명의", "등기원인"],
            "공시": ["공시", "공시제도", "공시법", "공시등기", "공시송달"]
        },
        "intellectual_property": {
            "특허": ["특허", "특허권", "특허출원", "특허등록", "특허침해", "특허심판원"],
            "상표": ["상표", "상표권", "상표출원", "상표등록", "상표침해", "상표심판원"],
            "저작권": ["저작권", "저작물", "저작인격권", "저작재산권", "저작권침해"],
            "디자인": ["디자인", "디자인권", "디자인출원", "디자인등록", "디자인침해"],
            "지적재산권": ["지적재산권", "IP", "지식재산권", "무체재산권"]
        },
        "tax_law": {
            "소득세": ["소득세", "소득세신고", "종합소득세", "근로소득세", "사업소득세", "가산세"],
            "부가가치세": ["부가가치세", "VAT", "매출세액", "매입세액", "세액공제"],
            "법인세": ["법인세", "법인세신고", "법인세율", "법인세과세표준"],
            "상속세": ["상속세", "상속세신고", "상속세율", "상속세과세표준"],
            "증여세": ["증여세", "증여세신고", "증여세율", "증여세과세표준"],
            "가산세": ["무신고가산세", "과소신고가산세", "납부지연가산세", "가산세율"]
        },
        "civil_procedure": {
            "소송": ["소송", "민사소송", "소송제기", "소장", "답변서", "준비서면"],
            "관할": ["관할", "보통재판적", "특별재판적", "토지관할", "사물관할", "관할법원"],
            "증거": ["증거", "증거조사", "증인", "증거서류", "증거능력", "증명력"],
            "판결": ["판결", "판결선고", "판결문", "판결확정", "판결집행"],
            "집행": ["집행", "강제집행", "집행문", "집행신청", "집행정지"]
        }
    }
    
    # 법률 용어 사전 (법적정확성 향상용)
    LEGAL_TERMS = {
        "법조문": ["법률", "조항", "항", "호", "법령", "시행령", "시행규칙"],
        "판례": ["대법원", "판례", "판결", "선고", "확정", "상고", "항소"],
        "법원": ["법원", "지방법원", "고등법원", "대법원", "가정법원", "행정법원"],
        "절차": ["절차", "신청", "제기", "심리", "판결", "선고", "확정"],
        "권리": ["권리", "의무", "청구권", "형성권", "지배권", "항변권"],
        "계약": ["계약", "계약체결", "계약이행", "계약위반", "계약해제", "계약해지"],
        "손해": ["손해", "손해배상", "정신적손해", "재산적손해", "이익상실"],
        "과실": ["과실", "고의", "중과실", "경과실", "과실상계"],
        "효력": ["효력", "무효", "취소", "해제", "해지", "실효"],
        "기간": ["기간", "시효", "제척기간", "소멸시효", "취득시효"]
    }
    
    @classmethod
    def get_keywords_for_question(cls, question: str, query_type: str) -> List[str]:
        """질문에서 추출한 키워드와 질문 유형에 따른 필수 키워드 반환"""
        keywords = []
        
        # 질문 유형별 기본 키워드 추가
        if query_type in cls.KEYWORD_MAPPING:
            for category, keyword_list in cls.KEYWORD_MAPPING[query_type].items():
                if category in question:
                    keywords.extend(keyword_list)
        
        # 질문에서 직접 추출한 키워드 추가
        question_keywords = cls._extract_keywords_from_question(question)
        keywords.extend(question_keywords)
        
        # 법률 용어 추가 (법적정확성 향상)
        legal_terms = cls._get_relevant_legal_terms(query_type)
        keywords.extend(legal_terms)
        
        return list(set(keywords))  # 중복 제거
    
    @classmethod
    def _extract_keywords_from_question(cls, question: str) -> List[str]:
        """질문에서 직접 키워드 추출"""
        keywords = []
        
        # 질문에서 법률 관련 키워드 추출
        legal_patterns = [
            r'(\w+법)',  # 법률명 (예: 민법, 형법)
            r'(\w+조)',  # 법조문 (예: 750조, 329조)
            r'(\w+원)',  # 법원명 (예: 대법원, 지방법원)
            r'(\w+위원회)',  # 위원회 (예: 노동위원회, 특허심판원)
            r'(\w+신고)',  # 신고 관련 (예: 소득세신고, 상속세신고)
            r'(\w+절차)',  # 절차 관련 (예: 이혼절차, 소송절차)
            r'(\w+권)',  # 권리 관련 (예: 소유권, 저작권)
            r'(\w+금)',  # 금액 관련 (예: 계약금, 위자료)
        ]
        
        for pattern in legal_patterns:
            matches = re.findall(pattern, question)
            keywords.extend(matches)
        
        return keywords
    
    @classmethod
    def _get_relevant_legal_terms(cls, query_type: str) -> List[str]:
        """질문 유형에 따른 관련 법률 용어 반환"""
        legal_terms = []
        
        # 모든 질문 유형에 공통으로 적용되는 기본 법률 용어
        basic_terms = ["법률", "법조문", "판례", "법원", "절차", "권리", "의무"]
        legal_terms.extend(basic_terms)
        
        # 질문 유형별 특화된 법률 용어
        type_specific_terms = {
            "contract_review": ["계약", "효력", "무효", "취소", "해제", "손해배상"],
            "family_law": ["가정법원", "조정", "재판", "신청", "위자료"],
            "criminal_law": ["형법", "구성요건", "법정형", "고의", "과실"],
            "civil_law": ["민법", "불법행위", "손해", "인과관계", "과실"],
            "labor_law": ["근로기준법", "노동위원회", "구제신청", "원직복직"],
            "property_law": ["등기", "소유권", "매매", "임대", "공시"],
            "intellectual_property": ["특허권", "저작권", "상표권", "침해", "구제"],
            "tax_law": ["세법", "신고", "납부", "가산세", "세액"],
            "civil_procedure": ["민사소송법", "관할", "증거", "판결", "집행"]
        }
        
        if query_type in type_specific_terms:
            legal_terms.extend(type_specific_terms[query_type])
        
        return legal_terms
    
    @classmethod
    def get_required_keywords_for_type(cls, query_type: str) -> List[str]:
        """질문 유형별 필수 키워드 반환"""
        required_keywords = []
        
        if query_type in cls.KEYWORD_MAPPING:
            for category, keyword_list in cls.KEYWORD_MAPPING[query_type].items():
                required_keywords.extend(keyword_list)
        
        return list(set(required_keywords))
    
    @classmethod
    def calculate_keyword_coverage(cls, answer: str, required_keywords: List[str]) -> float:
        """답변에서 필수 키워드 포함 비율 계산"""
        if not required_keywords:
            return 1.0
        
        answer_lower = answer.lower()
        matched_keywords = 0
        
        for keyword in required_keywords:
            if keyword.lower() in answer_lower:
                matched_keywords += 1
        
        return matched_keywords / len(required_keywords)
    
    @classmethod
    def get_missing_keywords(cls, answer: str, required_keywords: List[str]) -> List[str]:
        """답변에서 누락된 키워드 반환"""
        answer_lower = answer.lower()
        missing_keywords = []
        
        for keyword in required_keywords:
            if keyword.lower() not in answer_lower:
                missing_keywords.append(keyword)
        
        return missing_keywords
