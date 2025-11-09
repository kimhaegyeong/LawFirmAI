# -*- coding: utf-8 -*-
"""
법률 질문별 필수 키워드 매핑 시스템
답변 품질 향상을 위한 키워드 포함도 개선
가중치 기반 키워드 시스템으로 정교화
"""

import re
import json
import os
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime


class LegalKeywordMapper:
    """법률 질문별 필수 키워드 매핑 (가중치 기반)"""
    
    # 가중치 기반 키워드 매핑 (core: 1.0, important: 0.8, supporting: 0.6)
    WEIGHTED_KEYWORD_MAPPING = {
        "contract_review": {
            "계약서": {
                "core": ["당사자", "목적", "조건", "기간"],  # 핵심 키워드
                "important": ["해지", "손해배상", "계약금", "위약금"],  # 중요 키워드
                "supporting": ["체결", "효력", "무효", "취소", "해제"]  # 보조 키워드
            },
            "계약": {
                "core": ["체결", "효력", "무효", "취소"],
                "important": ["해지", "이행", "채무", "채권"],
                "supporting": ["계약금", "위약금", "손해배상"]
            },
            "매매": {
                "core": ["대금", "인도", "등기", "소유권이전"],
                "important": ["하자", "담보책임", "매도인", "매수인"],
                "supporting": ["계약금", "중도금", "잔금"]
            },
            "임대": {
                "core": ["임대료", "보증금", "임대차"],
                "important": ["임차인", "임대인", "계약갱신"],
                "supporting": ["전세", "월세", "임대차보호법"]
            }
        },
        "family_law": {
            "이혼": {
                "core": ["협의이혼", "조정이혼", "재판이혼", "가정법원"],
                "important": ["위자료", "재산분할", "양육비", "면접교섭권"],
                "supporting": ["신청", "조정", "재판", "확정"]
            },
            "상속": {
                "core": ["상속인", "상속분", "유류분", "상속재산"],
                "important": ["상속포기", "한정승인", "유언", "유증"],
                "supporting": ["상속세", "상속등기", "상속재산분할"]
            },
            "양육": {
                "core": ["양육권", "친권", "면접교섭권"],
                "important": ["양육비", "양육비지급", "양육비이행"],
                "supporting": ["양육비조정", "양육비강제집행"]
            }
        },
        "criminal_law": {
            "절도": {
                "core": ["타인의", "재물", "절취", "불법영득의사"],
                "important": ["형법", "329조", "고의", "의사"],
                "supporting": ["절도죄", "상습절도", "야간절도"]
            },
            "사기": {
                "core": ["기망", "착오", "재산상", "손해"],
                "important": ["형법", "347조", "사기죄", "편취"],
                "supporting": ["사기미수", "상습사기", "컴퓨터사기"]
            },
            "폭행": {
                "core": ["폭행", "상해", "협박"],
                "important": ["형법", "260조", "폭행죄", "상해죄"],
                "supporting": ["협박죄", "상습폭행", "특수폭행"]
            }
        },
        "civil_law": {
            "손해배상": {
                "core": ["손해", "가해", "인과관계", "과실"],
                "important": ["청구", "소송", "합의", "불법행위"],
                "supporting": ["민법", "750조", "정신적손해", "재산적손해"]
            },
            "계약": {
                "core": ["계약", "계약체결", "계약이행"],
                "important": ["계약위반", "계약해제", "계약해지"],
                "supporting": ["민법", "537조", "계약금", "위약금"]
            },
            "불법행위": {
                "core": ["불법행위", "과실", "손해", "인과관계"],
                "important": ["민법", "750조", "손해배상"],
                "supporting": ["고의", "중과실", "경과실", "과실상계"]
            }
        },
        "labor_law": {
            "해고": {
                "core": ["부당해고", "노동위원회", "구제신청"],
                "important": ["원직복직", "임금상당액", "해고사유"],
                "supporting": ["해고절차", "해고예고", "해고제한"]
            },
            "임금": {
                "core": ["임금", "최저임금", "임금지급"],
                "important": ["임금체불", "임금채권", "임금보전"],
                "supporting": ["임금채권보장기금", "임금지급명령"]
            },
            "근로시간": {
                "core": ["근로시간", "휴게시간", "휴일"],
                "important": ["연장근로", "야간근로", "휴일근로"],
                "supporting": ["근로기준법", "근로시간단축", "선택적근로시간"]
            }
        },
        "property_law": {
            "부동산": {
                "core": ["부동산", "토지", "건물", "등기"],
                "important": ["등기부등본", "소유권이전등기"],
                "supporting": ["등기명의", "등기원인", "등기이전"]
            },
            "매매": {
                "core": ["매매계약", "매매대금", "계약금"],
                "important": ["중도금", "잔금", "인도", "등기이전"],
                "supporting": ["매매대금지급", "소유권이전", "매매계약서"]
            }
        },
        "intellectual_property": {
            "특허": {
                "core": ["특허", "특허권", "특허출원", "특허등록"],
                "important": ["특허침해", "특허심판원"],
                "supporting": ["특허법", "특허청", "특허심사"]
            },
            "상표": {
                "core": ["상표", "상표권", "상표출원", "상표등록"],
                "important": ["상표침해", "상표심판원"],
                "supporting": ["상표법", "상표청", "상표심사"]
            },
            "저작권": {
                "core": ["저작권", "저작물", "저작인격권"],
                "important": ["저작재산권", "저작권침해"],
                "supporting": ["저작권법", "저작권위원회", "저작권보호"]
            }
        },
        "tax_law": {
            "소득세": {
                "core": ["소득세", "소득세신고", "종합소득세"],
                "important": ["근로소득세", "사업소득세", "가산세"],
                "supporting": ["소득세법", "소득세율", "소득세과세표준"]
            },
            "부가가치세": {
                "core": ["부가가치세", "VAT", "매출세액"],
                "important": ["매입세액", "세액공제"],
                "supporting": ["부가가치세법", "부가가치세신고", "부가가치세율"]
            }
        },
        "civil_procedure": {
            "소송": {
                "core": ["소송", "민사소송", "소송제기", "소장"],
                "important": ["답변서", "준비서면"],
                "supporting": ["민사소송법", "소송비용", "소송보조"]
            },
            "관할": {
                "core": ["관할", "보통재판적", "특별재판적"],
                "important": ["토지관할", "사물관할", "관할법원"],
                "supporting": ["관할이전", "관할합의", "관할위반"]
            }
        }
    }
    
    # 기존 키워드 매핑 (하위 호환성을 위해 유지)
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
        """질문에서 추출한 키워드와 질문 유형에 따른 필수 키워드 반환 (가중치 기반)"""
        keywords = []
        
        # 가중치 기반 키워드 매핑 사용
        if query_type in cls.WEIGHTED_KEYWORD_MAPPING:
            for category, weight_levels in cls.WEIGHTED_KEYWORD_MAPPING[query_type].items():
                if category in question:
                    # 모든 가중치 레벨의 키워드 추가
                    for level, keyword_list in weight_levels.items():
                        keywords.extend(keyword_list)
        
        # 질문에서 직접 추출한 키워드 추가
        question_keywords = cls._extract_keywords_from_question(question)
        keywords.extend(question_keywords)
        
        # 법률 용어 추가 (법적정확성 향상)
        legal_terms = cls._get_relevant_legal_terms(query_type)
        keywords.extend(legal_terms)
        
        return list(set(keywords))  # 중복 제거
    
    @classmethod
    def get_weighted_keywords_for_question(cls, question: str, query_type: str) -> Dict[str, List[str]]:
        """가중치별로 분류된 키워드 반환"""
        weighted_keywords = {
            "core": [],
            "important": [],
            "supporting": []
        }
        
        if query_type in cls.WEIGHTED_KEYWORD_MAPPING:
            for category, weight_levels in cls.WEIGHTED_KEYWORD_MAPPING[query_type].items():
                if category in question:
                    for level, keyword_list in weight_levels.items():
                        weighted_keywords[level].extend(keyword_list)
        
        # 중복 제거
        for level in weighted_keywords:
            weighted_keywords[level] = list(set(weighted_keywords[level]))
        
        return weighted_keywords
    
    @classmethod
    def calculate_weighted_keyword_coverage(cls, answer: str, query_type: str, question: str = "") -> Dict[str, float]:
        """가중치를 고려한 키워드 포함도 계산"""
        weighted_keywords = cls.get_weighted_keywords_for_question(question or "", query_type)
        
        answer_lower = answer.lower()
        coverage_results = {}
        
        # 각 가중치 레벨별 포함도 계산
        weights = {"core": 1.0, "important": 0.8, "supporting": 0.6}
        
        total_score = 0.0
        max_score = 0.0
        
        for level, keywords in weighted_keywords.items():
            matched_count = sum(1 for kw in keywords if kw.lower() in answer_lower)
            level_coverage = matched_count / len(keywords) if keywords else 0.0
            
            coverage_results[f"{level}_coverage"] = level_coverage
            coverage_results[f"{level}_matched"] = matched_count
            coverage_results[f"{level}_total"] = len(keywords)
            
            # 가중치 적용
            weight = weights[level]
            total_score += matched_count * weight
            max_score += len(keywords) * weight
        
        # 전체 가중치 포함도
        coverage_results["weighted_coverage"] = total_score / max_score if max_score > 0 else 0.0
        coverage_results["overall_coverage"] = sum(coverage_results[f"{level}_matched"] for level in weights.keys()) / sum(coverage_results[f"{level}_total"] for level in weights.keys()) if sum(coverage_results[f"{level}_total"] for level in weights.keys()) > 0 else 0.0
        
        return coverage_results
    
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
        """답변에서 필수 키워드 포함 비율 계산 (기존 호환성 유지)"""
        if not required_keywords:
            return 1.0
        
        answer_lower = answer.lower()
        matched_keywords = 0
        
        for keyword in required_keywords:
            if keyword.lower() in answer_lower:
                matched_keywords += 1
        
        return matched_keywords / len(required_keywords)
    
    @classmethod
    def get_keyword_analysis_report(cls, answer: str, query_type: str, question: str = "") -> Dict[str, any]:
        """키워드 분석 상세 보고서 생성"""
        weighted_keywords = cls.get_weighted_keywords_for_question(question or "", query_type)
        coverage_results = cls.calculate_weighted_keyword_coverage(answer, query_type, question)
        
        # 누락된 키워드 분석
        answer_lower = answer.lower()
        missing_keywords = {}
        
        for level, keywords in weighted_keywords.items():
            missing = [kw for kw in keywords if kw.lower() not in answer_lower]
            missing_keywords[level] = missing
        
        # 키워드 중요도 점수 계산
        importance_scores = {}
        weights = {"core": 1.0, "important": 0.8, "supporting": 0.6}
        
        for level in weights.keys():
            matched = coverage_results[f"{level}_matched"]
            total = coverage_results[f"{level}_total"]
            importance_scores[level] = (matched / total * weights[level]) if total > 0 else 0.0
        
        return {
            "coverage_results": coverage_results,
            "missing_keywords": missing_keywords,
            "importance_scores": importance_scores,
            "recommendations": cls._generate_keyword_recommendations(coverage_results, missing_keywords),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    @classmethod
    def _generate_keyword_recommendations(cls, coverage_results: Dict[str, float], missing_keywords: Dict[str, List[str]]) -> List[str]:
        """키워드 포함도 기반 개선 권장사항 생성"""
        recommendations = []
        
        # 핵심 키워드 누락 시 우선 권장
        if missing_keywords.get("core"):
            recommendations.append(f"핵심 키워드 '{', '.join(missing_keywords['core'][:3])}' 포함 필요")
        
        # 중요 키워드 누락 시 권장
        if missing_keywords.get("important"):
            recommendations.append(f"중요 키워드 '{', '.join(missing_keywords['important'][:3])}' 포함 권장")
        
        # 전체 포함도가 낮은 경우
        if coverage_results.get("overall_coverage", 0) < 0.5:
            recommendations.append("전체적으로 더 많은 관련 키워드 포함 필요")
        
        # 핵심 키워드 포함도가 낮은 경우
        if coverage_results.get("core_coverage", 0) < 0.7:
            recommendations.append("핵심 키워드 포함도를 높여 답변의 정확성 향상 필요")
        
        return recommendations
    
    @classmethod
    def get_missing_keywords(cls, answer: str, required_keywords: List[str]) -> List[str]:
        """답변에서 누락된 키워드 반환"""
        answer_lower = answer.lower()
        missing_keywords = []
        
        for keyword in required_keywords:
            if keyword.lower() not in answer_lower:
                missing_keywords.append(keyword)
        
        return missing_keywords


class ContextAwareKeywordMapper:
    """컨텍스트 인식 키워드 매핑 시스템"""
    
    def __init__(self):
        self.context_patterns = {
            "질문형": ["어떻게", "무엇인가", "언제", "어디서", "왜", "무엇을", "어떤"],
            "절차형": ["절차", "방법", "과정", "단계", "순서", "절차는", "방법은"],
            "비교형": ["차이점", "비교", "구분", "다른점", "차이", "비교하면", "구분하면"],
            "문제해결형": ["문제", "해결", "방법", "대처", "대응", "해결방법", "대처방안"],
            "법적효력형": ["효력", "무효", "취소", "해제", "해지", "효력이", "무효인지"],
            "요건형": ["요건", "조건", "구성요건", "요구사항", "필요조건"],
            "효과형": ["효과", "결과", "영향", "효과는", "결과는", "영향은"],
            "기간형": ["기간", "시효", "제한", "기한", "기간은", "시효는"],
            "비용형": ["비용", "수수료", "금액", "가격", "비용은", "수수료는"],
            "권리형": ["권리", "의무", "권한", "권리는", "의무는", "권한은"]
        }
        
        self.context_keywords = {
            "질문형": ["정의", "개념", "의미", "내용", "설명", "개요"],
            "절차형": ["신청", "제기", "심리", "판결", "절차", "단계", "과정"],
            "비교형": ["구분", "차이", "비교", "대조", "상이점", "공통점"],
            "문제해결형": ["해결방법", "대처방안", "구제절차", "해결책", "방안"],
            "법적효력형": ["효력", "무효", "취소", "해제", "해지", "실효", "소멸"],
            "요건형": ["요건", "조건", "구성요건", "필수요건", "요구사항"],
            "효과형": ["효과", "결과", "영향", "수익", "손실", "변화"],
            "기간형": ["기간", "시효", "제한", "기한", "만료", "연장"],
            "비용형": ["비용", "수수료", "금액", "가격", "요금", "부담"],
            "권리형": ["권리", "의무", "권한", "자격", "지위", "지분"]
        }
    
    def identify_context(self, question: str) -> str:
        """질문의 컨텍스트 식별"""
        question_lower = question.lower()
        
        # 각 컨텍스트 패턴별로 점수 계산
        context_scores = {}
        
        for context, patterns in self.context_patterns.items():
            score = sum(1 for pattern in patterns if pattern in question_lower)
            if score > 0:
                context_scores[context] = score
        
        # 가장 높은 점수의 컨텍스트 반환
        if context_scores:
            return max(context_scores.items(), key=lambda x: x[1])[0]
        
        return "일반형"
    
    def get_contextual_keywords(self, question: str, query_type: str) -> Dict[str, List[str]]:
        """컨텍스트를 고려한 키워드 반환"""
        # 기본 키워드 가져오기
        base_keywords = LegalKeywordMapper.get_keywords_for_question(question, query_type)
        
        # 컨텍스트 식별
        context = self.identify_context(question)
        
        # 컨텍스트별 추가 키워드
        contextual_keywords = self.context_keywords.get(context, [])
        
        # 가중치별 키워드도 가져오기
        weighted_keywords = LegalKeywordMapper.get_weighted_keywords_for_question(question, query_type)
        
        return {
            "base_keywords": base_keywords,
            "contextual_keywords": contextual_keywords,
            "weighted_keywords": weighted_keywords,
            "identified_context": context,
            "all_keywords": list(set(base_keywords + contextual_keywords))
        }
    
    def analyze_question_intent(self, question: str) -> Dict[str, any]:
        """질문의 의도 분석"""
        context = self.identify_context(question)
        
        # 질문 유형별 의도 분석
        intent_analysis = {
            "primary_intent": context,
            "confidence": self._calculate_context_confidence(question, context),
            "secondary_intents": self._identify_secondary_intents(question),
            "question_type": self._classify_question_type(question),
            "complexity_level": self._assess_complexity(question)
        }
        
        return intent_analysis
    
    def _calculate_context_confidence(self, question: str, context: str) -> float:
        """컨텍스트 식별 신뢰도 계산"""
        question_lower = question.lower()
        patterns = self.context_patterns.get(context, [])
        
        if not patterns:
            return 0.0
        
        matches = sum(1 for pattern in patterns if pattern in question_lower)
        return matches / len(patterns)
    
    def _identify_secondary_intents(self, question: str) -> List[str]:
        """보조 의도 식별"""
        question_lower = question.lower()
        secondary_intents = []
        
        for context, patterns in self.context_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                secondary_intents.append(context)
        
        return secondary_intents
    
    def _classify_question_type(self, question: str) -> str:
        """질문 유형 분류"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["어떻게", "방법", "절차"]):
            return "방법론적"
        elif any(word in question_lower for word in ["무엇", "정의", "개념"]):
            return "개념적"
        elif any(word in question_lower for word in ["언제", "기간", "시점"]):
            return "시간적"
        elif any(word in question_lower for word in ["왜", "이유", "원인"]):
            return "원인적"
        elif any(word in question_lower for word in ["비용", "금액", "수수료"]):
            return "경제적"
        else:
            return "일반적"
    
    def _assess_complexity(self, question: str) -> str:
        """질문 복잡도 평가"""
        # 질문 길이와 키워드 수를 기반으로 복잡도 평가
        word_count = len(question.split())
        keyword_count = len(re.findall(r'\b\w+\b', question))
        
        if word_count > 20 or keyword_count > 15:
            return "고복잡도"
        elif word_count > 10 or keyword_count > 8:
            return "중복잡도"
        else:
            return "저복잡도"
    
    def get_enhanced_keyword_mapping(self, question: str, query_type: str) -> Dict[str, any]:
        """향상된 키워드 매핑 결과 반환"""
        contextual_data = self.get_contextual_keywords(question, query_type)
        intent_analysis = self.analyze_question_intent(question)
        
        # 키워드 우선순위 계산
        keyword_priority = self._calculate_keyword_priority(
            contextual_data["all_keywords"],
            intent_analysis["primary_intent"],
            intent_analysis["complexity_level"]
        )
        
        return {
            "keywords": contextual_data,
            "intent_analysis": intent_analysis,
            "keyword_priority": keyword_priority,
            "recommendations": self._generate_contextual_recommendations(
                contextual_data, intent_analysis
            )
        }
    
    def _calculate_keyword_priority(self, keywords: List[str], context: str, complexity: str) -> Dict[str, float]:
        """키워드 우선순위 계산"""
        priority_scores = {}
        
        # 컨텍스트별 키워드 가중치
        context_weights = {
            "질문형": {"정의": 1.0, "개념": 0.9, "의미": 0.8},
            "절차형": {"절차": 1.0, "단계": 0.9, "과정": 0.8},
            "비교형": {"비교": 1.0, "차이": 0.9, "구분": 0.8},
            "문제해결형": {"해결": 1.0, "방법": 0.9, "대처": 0.8},
            "법적효력형": {"효력": 1.0, "무효": 0.9, "취소": 0.8}
        }
        
        # 복잡도별 가중치
        complexity_multiplier = {
            "고복잡도": 1.2,
            "중복잡도": 1.0,
            "저복잡도": 0.8
        }
        
        for keyword in keywords:
            base_score = 0.5  # 기본 점수
            
            # 컨텍스트별 가중치 적용
            if context in context_weights:
                for ctx_keyword, weight in context_weights[context].items():
                    if ctx_keyword in keyword:
                        base_score = max(base_score, weight)
            
            # 복잡도 가중치 적용
            final_score = base_score * complexity_multiplier.get(complexity, 1.0)
            priority_scores[keyword] = min(final_score, 1.0)  # 최대 1.0으로 제한
        
        return priority_scores
    
    def _generate_contextual_recommendations(self, contextual_data: Dict, intent_analysis: Dict) -> List[str]:
        """컨텍스트 기반 권장사항 생성"""
        recommendations = []
        
        context = intent_analysis["primary_intent"]
        complexity = intent_analysis["complexity_level"]
        
        # 컨텍스트별 권장사항
        if context == "절차형":
            recommendations.append("단계별 절차 설명을 포함하여 답변 구조화")
        elif context == "비교형":
            recommendations.append("비교 대상별 차이점을 명확히 구분하여 설명")
        elif context == "문제해결형":
            recommendations.append("구체적인 해결방안과 대처절차를 제시")
        elif context == "법적효력형":
            recommendations.append("법적 효력과 그 결과를 명확히 설명")
        
        # 복잡도별 권장사항
        if complexity == "고복잡도":
            recommendations.append("복잡한 내용을 단계별로 나누어 설명")
        elif complexity == "저복잡도":
            recommendations.append("핵심 내용에 집중하여 간결하게 설명")
        
        return recommendations


class AdaptiveKeywordMapper:
    """동적 키워드 학습 시스템"""
    
    def __init__(self, data_file: str = "data/keyword_effectiveness.json"):
        self.data_file = data_file
        self.keyword_effectiveness = {}
        self.user_feedback_history = []
        self.question_patterns = {}
        self.load_data()
    
    def load_data(self):
        """저장된 데이터 로드"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.keyword_effectiveness = data.get('keyword_effectiveness', {})
                    self.user_feedback_history = data.get('user_feedback_history', [])
                    self.question_patterns = data.get('question_patterns', {})
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            self.keyword_effectiveness = {}
            self.user_feedback_history = []
            self.question_patterns = {}
    
    def save_data(self):
        """데이터 저장"""
        try:
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            data = {
                'keyword_effectiveness': self.keyword_effectiveness,
                'user_feedback_history': self.user_feedback_history[-1000:],  # 최근 1000개만 유지
                'question_patterns': self.question_patterns,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"데이터 저장 실패: {e}")
    
    def update_keyword_effectiveness(self, question: str, keywords: List[str], 
                                   user_rating: float, answer_quality: float, 
                                   query_type: str = ""):
        """사용자 피드백을 바탕으로 키워드 효과성 업데이트"""
        timestamp = datetime.now().isoformat()
        
        # 피드백 기록 저장
        feedback_record = {
            'timestamp': timestamp,
            'question': question,
            'keywords': keywords,
            'user_rating': user_rating,
            'answer_quality': answer_quality,
            'query_type': query_type
        }
        self.user_feedback_history.append(feedback_record)
        
        # 각 키워드별 효과성 업데이트
        for keyword in keywords:
            if keyword not in self.keyword_effectiveness:
                self.keyword_effectiveness[keyword] = {
                    'total_usage': 0,
                    'total_rating': 0.0,
                    'total_quality': 0.0,
                    'effectiveness_score': 0.5,  # 초기값
                    'usage_count': 0,
                    'last_updated': timestamp
                }
            
            data = self.keyword_effectiveness[keyword]
            data['total_usage'] += 1
            data['total_rating'] += user_rating
            data['total_quality'] += answer_quality
            data['usage_count'] += 1
            data['last_updated'] = timestamp
            
            # 효과성 점수 계산 (사용자 평점 40% + 답변 품질 60%)
            avg_rating = data['total_rating'] / data['total_usage']
            avg_quality = data['total_quality'] / data['total_usage']
            data['effectiveness_score'] = avg_rating * 0.4 + avg_quality * 0.6
        
        # 질문 패턴 분석
        self._analyze_question_pattern(question, keywords, user_rating, answer_quality, query_type)
        
        # 데이터 저장
        self.save_data()
    
    def _analyze_question_pattern(self, question: str, keywords: List[str], 
                                user_rating: float, answer_quality: float, 
                                query_type: str):
        """질문 패턴 분석"""
        pattern = self._extract_pattern(question)
        
        if pattern not in self.question_patterns:
            self.question_patterns[pattern] = {
                'keywords': {},
                'quality_scores': [],
                'usage_count': 0,
                'avg_quality': 0.0
            }
        
        pattern_data = self.question_patterns[pattern]
        pattern_data['usage_count'] += 1
        
        # 키워드별 품질 점수 기록
        for keyword in keywords:
            if keyword not in pattern_data['keywords']:
                pattern_data['keywords'][keyword] = []
            pattern_data['keywords'][keyword].append(answer_quality)
        
        pattern_data['quality_scores'].append(answer_quality)
        pattern_data['avg_quality'] = sum(pattern_data['quality_scores']) / len(pattern_data['quality_scores'])
    
    def _extract_pattern(self, question: str) -> str:
        """질문에서 패턴 추출"""
        question_lower = question.lower()
        
        if "어떻게" in question_lower:
            return "방법_질문"
        elif "무엇인가" in question_lower:
            return "정의_질문"
        elif "언제" in question_lower:
            return "시점_질문"
        elif "왜" in question_lower:
            return "원인_질문"
        elif "비용" in question_lower or "금액" in question_lower:
            return "비용_질문"
        elif "절차" in question_lower or "방법" in question_lower:
            return "절차_질문"
        else:
            return "일반_질문"
    
    def get_effective_keywords(self, query_type: str, limit: int = 10) -> List[str]:
        """효과성이 높은 키워드 우선 반환"""
        # 기본 키워드 가져오기
        base_keywords = LegalKeywordMapper.get_required_keywords_for_type(query_type)
        
        # 효과성 점수 기준으로 정렬
        scored_keywords = []
        for kw in base_keywords:
            effectiveness_data = self.keyword_effectiveness.get(kw, {})
            score = effectiveness_data.get('effectiveness_score', 0.5)
            scored_keywords.append((kw, score))
        
        scored_keywords.sort(key=lambda x: x[1], reverse=True)
        return [kw for kw, score in scored_keywords[:limit]]
    
    def get_pattern_based_keywords(self, question: str, query_type: str) -> List[str]:
        """패턴 기반 키워드 추천"""
        pattern = self._extract_pattern(question)
        
        if pattern in self.question_patterns:
            pattern_data = self.question_patterns[pattern]
            
            # 평균 품질 점수가 높은 키워드 우선 추천
            keyword_scores = {}
            for keyword, scores in pattern_data['keywords'].items():
                avg_score = sum(scores) / len(scores)
                keyword_scores[keyword] = avg_score
            
            # 점수 기준 정렬
            sorted_keywords = sorted(keyword_scores.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            return [kw for kw, score in sorted_keywords[:10]]
        
        # 패턴이 없으면 기본 키워드 반환
        return LegalKeywordMapper.get_keywords_for_question(question, query_type)
    
    def get_learning_insights(self) -> Dict[str, any]:
        """학습 인사이트 생성"""
        if not self.user_feedback_history:
            return {"message": "아직 충분한 피드백 데이터가 없습니다."}
        
        # 최근 피드백 분석
        recent_feedback = self.user_feedback_history[-100:] if len(self.user_feedback_history) > 100 else self.user_feedback_history
        
        # 평균 품질 점수
        avg_quality = sum(f['answer_quality'] for f in recent_feedback) / len(recent_feedback)
        avg_rating = sum(f['user_rating'] for f in recent_feedback) / len(recent_feedback)
        
        # 가장 효과적인 키워드
        top_keywords = sorted(
            self.keyword_effectiveness.items(),
            key=lambda x: x[1]['effectiveness_score'],
            reverse=True
        )[:5]
        
        # 가장 많이 사용되는 패턴
        pattern_usage = {}
        for feedback in recent_feedback:
            pattern = self._extract_pattern(feedback['question'])
            pattern_usage[pattern] = pattern_usage.get(pattern, 0) + 1
        
        top_patterns = sorted(pattern_usage.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "recent_performance": {
                "avg_quality": avg_quality,
                "avg_user_rating": avg_rating,
                "feedback_count": len(recent_feedback)
            },
            "top_effective_keywords": [
                {"keyword": kw, "score": data['effectiveness_score'], "usage": data['total_usage']}
                for kw, data in top_keywords
            ],
            "popular_patterns": [
                {"pattern": pattern, "usage_count": count}
                for pattern, count in top_patterns
            ],
            "learning_progress": {
                "total_keywords_tracked": len(self.keyword_effectiveness),
                "total_patterns_tracked": len(self.question_patterns),
                "total_feedback_count": len(self.user_feedback_history)
            }
        }
    
    def recommend_keyword_improvements(self, query_type: str) -> List[str]:
        """키워드 개선 권장사항 생성"""
        recommendations = []
        
        # 효과성이 낮은 키워드 식별
        low_effectiveness_keywords = [
            kw for kw, data in self.keyword_effectiveness.items()
            if data['effectiveness_score'] < 0.4 and data['total_usage'] > 5
        ]
        
        if low_effectiveness_keywords:
            recommendations.append(f"효과성이 낮은 키워드 개선 필요: {', '.join(low_effectiveness_keywords[:3])}")
        
        # 사용 빈도가 낮은 키워드 식별
        low_usage_keywords = [
            kw for kw, data in self.keyword_effectiveness.items()
            if data['total_usage'] < 3 and data['effectiveness_score'] > 0.7
        ]
        
        if low_usage_keywords:
            recommendations.append(f"잠재력이 높은 키워드 활용 권장: {', '.join(low_usage_keywords[:3])}")
        
        # 패턴별 개선사항
        for pattern, data in self.question_patterns.items():
            if data['avg_quality'] < 0.5 and data['usage_count'] > 10:
                recommendations.append(f"'{pattern}' 패턴의 답변 품질 개선 필요")
        
        return recommendations
    
    def get_adaptive_keyword_mapping(self, question: str, query_type: str) -> Dict[str, any]:
        """적응형 키워드 매핑 결과 반환"""
        # 기본 키워드
        base_keywords = LegalKeywordMapper.get_keywords_for_question(question, query_type)
        
        # 효과성 기반 키워드
        effective_keywords = self.get_effective_keywords(query_type, 10)
        
        # 패턴 기반 키워드
        pattern_keywords = self.get_pattern_based_keywords(question, query_type)
        
        # 모든 키워드 통합
        all_keywords = list(set(base_keywords + effective_keywords + pattern_keywords))
        
        # 키워드별 효과성 점수 추가
        keyword_scores = {}
        for kw in all_keywords:
            effectiveness_data = self.keyword_effectiveness.get(kw, {})
            keyword_scores[kw] = effectiveness_data.get('effectiveness_score', 0.5)
        
        return {
            "base_keywords": base_keywords,
            "effective_keywords": effective_keywords,
            "pattern_keywords": pattern_keywords,
            "all_keywords": all_keywords,
            "keyword_scores": keyword_scores,
            "learning_insights": self.get_learning_insights(),
            "improvement_recommendations": self.recommend_keyword_improvements(query_type)
        }


class EnhancedSemanticKeywordMapper:
    """향상된 의미적 키워드 매핑 시스템"""
    
    def __init__(self):
        # 확장된 법률 용어 간의 의미적 관계 정의
        self.semantic_relations = {
        "제1항": {
                "synonyms": [],
                "related": [
                        "대법원",
                        "제2항",
                        "지원",
                        "시행령",
                        "위원",
                        "일부",
                        "대통령령",
                        "여부"
                ],
                "context": [
                        "대법원",
                        "지원",
                        "시행령",
                        "위원",
                        "일부"
                ]
        },
        "대법원": {
                "synonyms": [],
                "related": [
                        "제1항",
                        "제2항",
                        "지원",
                        "시행령",
                        "위원",
                        "일부",
                        "대통령령",
                        "여부"
                ],
                "context": [
                        "지원",
                        "시행령",
                        "위원",
                        "일부",
                        "대통령령"
                ]
        },
        "제2항": {
                "synonyms": [],
                "related": [
                        "제1항",
                        "대법원",
                        "지원",
                        "시행령",
                        "위원",
                        "일부",
                        "대통령령",
                        "여부"
                ],
                "context": [
                        "대법원",
                        "지원",
                        "시행령",
                        "위원",
                        "일부"
                ]
        },
        "지원": {
                "synonyms": [],
                "related": [
                        "제1항",
                        "대법원",
                        "제2항",
                        "시행령",
                        "위원",
                        "일부",
                        "대통령령",
                        "여부"
                ],
                "context": [
                        "대법원",
                        "시행령",
                        "위원",
                        "일부",
                        "대통령령"
                ]
        },
        "시행령": {
                "synonyms": [],
                "related": [
                        "제1항",
                        "대법원",
                        "제2항",
                        "지원",
                        "위원",
                        "일부",
                        "대통령령",
                        "여부"
                ],
                "context": [
                        "대법원",
                        "지원",
                        "위원",
                        "일부",
                        "대통령령"
                ]
        },
        "위원": {
                "synonyms": [],
                "related": [
                        "제1항",
                        "대법원",
                        "제2항",
                        "지원",
                        "시행령",
                        "일부",
                        "대통령령",
                        "여부"
                ],
                "context": [
                        "대법원",
                        "지원",
                        "시행령",
                        "일부",
                        "대통령령"
                ]
        },
        "일부": {
                "synonyms": [],
                "related": [
                        "제1항",
                        "대법원",
                        "제2항",
                        "지원",
                        "시행령",
                        "위원",
                        "대통령령",
                        "여부"
                ],
                "context": [
                        "대법원",
                        "지원",
                        "시행령",
                        "위원",
                        "대통령령"
                ]
        },
        "대통령령": {
                "synonyms": [],
                "related": [
                        "제1항",
                        "대법원",
                        "제2항",
                        "지원",
                        "시행령",
                        "위원",
                        "일부",
                        "여부"
                ],
                "context": [
                        "대법원",
                        "지원",
                        "시행령",
                        "위원",
                        "일부"
                ]
        },
        "여부": {
                "synonyms": [],
                "related": [
                        "제1항",
                        "대법원",
                        "제2항",
                        "지원",
                        "시행령",
                        "위원",
                        "일부",
                        "대통령령"
                ],
                "context": [
                        "대법원",
                        "지원",
                        "시행령",
                        "위원",
                        "일부"
                ]
        },
        "대법": {
                "synonyms": [],
                "related": [
                        "제1항",
                        "대법원",
                        "제2항",
                        "지원",
                        "시행령",
                        "위원",
                        "일부",
                        "대통령령"
                ],
                "context": [
                        "대법원",
                        "지원",
                        "시행령",
                        "위원",
                        "일부"
                ]
        },
        "신청": {
                "synonyms": [],
                "related": [
                        "공무원",
                        "이의신청",
                        "신고",
                        "허가",
                        "승인을",
                        "변경신고",
                        "허가를",
                        "승인"
                ],
                "context": [
                        "공무원",
                        "공무원의",
                        "고위공무원",
                        "공무원으",
                        "공무원이"
                ]
        },
        "공무원": {
                "synonyms": [],
                "related": [
                        "신청",
                        "이의신청",
                        "신고",
                        "허가",
                        "승인을",
                        "변경신고",
                        "허가를",
                        "승인"
                ],
                "context": [
                        "공무원의",
                        "고위공무원",
                        "공무원으",
                        "공무원이",
                        "공무원은"
                ]
        },
        "이의신청": {
                "synonyms": [],
                "related": [
                        "신청",
                        "공무원",
                        "신고",
                        "허가",
                        "승인을",
                        "변경신고",
                        "허가를",
                        "승인"
                ],
                "context": [
                        "공무원",
                        "공무원의",
                        "고위공무원",
                        "공무원으",
                        "공무원이"
                ]
        },
        "신고": {
                "synonyms": [],
                "related": [
                        "신청",
                        "공무원",
                        "이의신청",
                        "허가",
                        "승인을",
                        "변경신고",
                        "허가를",
                        "승인"
                ],
                "context": [
                        "공무원",
                        "공무원의",
                        "고위공무원",
                        "공무원으",
                        "공무원이"
                ]
        },
        "허가": {
                "synonyms": [],
                "related": [
                        "신청",
                        "공무원",
                        "이의신청",
                        "신고",
                        "승인을",
                        "변경신고",
                        "허가를",
                        "승인"
                ],
                "context": [
                        "공무원",
                        "공무원의",
                        "고위공무원",
                        "공무원으",
                        "공무원이"
                ]
        },
        "승인을": {
                "synonyms": [],
                "related": [
                        "신청",
                        "공무원",
                        "이의신청",
                        "신고",
                        "허가",
                        "변경신고",
                        "허가를",
                        "승인"
                ],
                "context": [
                        "공무원",
                        "공무원의",
                        "고위공무원",
                        "공무원으",
                        "공무원이"
                ]
        },
        "변경신고": {
                "synonyms": [],
                "related": [
                        "신청",
                        "공무원",
                        "이의신청",
                        "신고",
                        "허가",
                        "승인을",
                        "허가를",
                        "승인"
                ],
                "context": [
                        "공무원",
                        "공무원의",
                        "고위공무원",
                        "공무원으",
                        "공무원이"
                ]
        },
        "허가를": {
                "synonyms": [],
                "related": [
                        "신청",
                        "공무원",
                        "이의신청",
                        "신고",
                        "허가",
                        "승인을",
                        "변경신고",
                        "승인"
                ],
                "context": [
                        "공무원",
                        "공무원의",
                        "고위공무원",
                        "공무원으",
                        "공무원이"
                ]
        },
        "승인": {
                "synonyms": [],
                "related": [
                        "신청",
                        "공무원",
                        "이의신청",
                        "신고",
                        "허가",
                        "승인을",
                        "변경신고",
                        "허가를"
                ],
                "context": [
                        "공무원",
                        "공무원의",
                        "고위공무원",
                        "공무원으",
                        "공무원이"
                ]
        },
        "신고를": {
                "synonyms": [],
                "related": [
                        "신청",
                        "공무원",
                        "이의신청",
                        "신고",
                        "허가",
                        "승인을",
                        "변경신고",
                        "허가를"
                ],
                "context": [
                        "공무원",
                        "공무원의",
                        "고위공무원",
                        "공무원으",
                        "공무원이"
                ]
        },
        "채권": {
                "synonyms": [],
                "related": [
                        "위반행위",
                        "소유권",
                        "소유권이",
                        "민사소송",
                        "채권의",
                        "회생채권",
                        "민사소송법",
                        "손해배상청"
                ],
                "context": [
                        "민사소송법",
                        "손해배상청",
                        "채무자회생법",
                        "법위반",
                        "법률위반"
                ]
        },
        "위반행위": {
                "synonyms": [],
                "related": [
                        "채권",
                        "소유권",
                        "소유권이",
                        "민사소송",
                        "채권의",
                        "회생채권",
                        "민사소송법",
                        "손해배상청"
                ],
                "context": [
                        "민사소송법",
                        "손해배상청",
                        "채무자회생법",
                        "법위반",
                        "법률위반"
                ]
        },
        "소유권": {
                "synonyms": [],
                "related": [
                        "채권",
                        "위반행위",
                        "소유권이",
                        "민사소송",
                        "채권의",
                        "회생채권",
                        "민사소송법",
                        "손해배상청"
                ],
                "context": [
                        "민사소송법",
                        "손해배상청",
                        "채무자회생법",
                        "법위반",
                        "법률위반"
                ]
        },
        "소유권이": {
                "synonyms": [],
                "related": [
                        "채권",
                        "위반행위",
                        "소유권",
                        "민사소송",
                        "채권의",
                        "회생채권",
                        "민사소송법",
                        "손해배상청"
                ],
                "context": [
                        "민사소송법",
                        "손해배상청",
                        "채무자회생법",
                        "법위반",
                        "법률위반"
                ]
        },
        "민사소송": {
                "synonyms": [],
                "related": [
                        "채권",
                        "위반행위",
                        "소유권",
                        "소유권이",
                        "채권의",
                        "회생채권",
                        "민사소송법",
                        "손해배상청"
                ],
                "context": [
                        "민사소송법",
                        "손해배상청",
                        "채무자회생법",
                        "법위반",
                        "법률위반"
                ]
        },
        "채권의": {
                "synonyms": [],
                "related": [
                        "채권",
                        "위반행위",
                        "소유권",
                        "소유권이",
                        "민사소송",
                        "회생채권",
                        "민사소송법",
                        "손해배상청"
                ],
                "context": [
                        "민사소송법",
                        "손해배상청",
                        "채무자회생법",
                        "법위반",
                        "법률위반"
                ]
        },
        "회생채권": {
                "synonyms": [],
                "related": [
                        "채권",
                        "위반행위",
                        "소유권",
                        "소유권이",
                        "민사소송",
                        "채권의",
                        "민사소송법",
                        "손해배상청"
                ],
                "context": [
                        "민사소송법",
                        "손해배상청",
                        "채무자회생법",
                        "법위반",
                        "법률위반"
                ]
        },
        "민사소송법": {
                "synonyms": [],
                "related": [
                        "채권",
                        "위반행위",
                        "소유권",
                        "소유권이",
                        "민사소송",
                        "채권의",
                        "회생채권",
                        "손해배상청"
                ],
                "context": [
                        "손해배상청",
                        "채무자회생법",
                        "법위반",
                        "법률위반",
                        "항소장각하명령"
                ]
        },
        "손해배상청": {
                "synonyms": [],
                "related": [
                        "채권",
                        "위반행위",
                        "소유권",
                        "소유권이",
                        "민사소송",
                        "채권의",
                        "회생채권",
                        "민사소송법"
                ],
                "context": [
                        "민사소송법",
                        "채무자회생법",
                        "법위반",
                        "법률위반",
                        "항소장각하명령"
                ]
        },
        "채권을": {
                "synonyms": [],
                "related": [
                        "채권",
                        "위반행위",
                        "소유권",
                        "소유권이",
                        "민사소송",
                        "채권의",
                        "회생채권",
                        "민사소송법"
                ],
                "context": [
                        "민사소송법",
                        "손해배상청",
                        "채무자회생법",
                        "법위반",
                        "법률위반"
                ]
        },
        "부동산": {
                "synonyms": [],
                "related": [
                        "부동산에",
                        "부동산의",
                        "부동산을",
                        "등기신청",
                        "등기부",
                        "전세권",
                        "부동산투",
                        "부동산등"
                ],
                "context": [
                        "부동산에",
                        "부동산의",
                        "부동산을",
                        "등기부",
                        "부동산투"
                ]
        },
        "부동산에": {
                "synonyms": [],
                "related": [
                        "부동산",
                        "부동산의",
                        "부동산을",
                        "등기신청",
                        "등기부",
                        "전세권",
                        "부동산투",
                        "부동산등"
                ],
                "context": [
                        "부동산",
                        "부동산의",
                        "부동산을",
                        "등기부",
                        "부동산투"
                ]
        },
        "부동산의": {
                "synonyms": [],
                "related": [
                        "부동산",
                        "부동산에",
                        "부동산을",
                        "등기신청",
                        "등기부",
                        "전세권",
                        "부동산투",
                        "부동산등"
                ],
                "context": [
                        "부동산",
                        "부동산에",
                        "부동산을",
                        "등기부",
                        "부동산투"
                ]
        },
        "부동산을": {
                "synonyms": [],
                "related": [
                        "부동산",
                        "부동산에",
                        "부동산의",
                        "등기신청",
                        "등기부",
                        "전세권",
                        "부동산투",
                        "부동산등"
                ],
                "context": [
                        "부동산",
                        "부동산에",
                        "부동산의",
                        "등기부",
                        "부동산투"
                ]
        },
        "등기신청": {
                "synonyms": [],
                "related": [
                        "부동산",
                        "부동산에",
                        "부동산의",
                        "부동산을",
                        "등기부",
                        "전세권",
                        "부동산투",
                        "부동산등"
                ],
                "context": [
                        "부동산",
                        "부동산에",
                        "부동산의",
                        "부동산을",
                        "등기부"
                ]
        },
        "등기부": {
                "synonyms": [],
                "related": [
                        "부동산",
                        "부동산에",
                        "부동산의",
                        "부동산을",
                        "등기신청",
                        "전세권",
                        "부동산투",
                        "부동산등"
                ],
                "context": [
                        "부동산",
                        "부동산에",
                        "부동산의",
                        "부동산을",
                        "부동산투"
                ]
        },
        "전세권": {
                "synonyms": [],
                "related": [
                        "부동산",
                        "부동산에",
                        "부동산의",
                        "부동산을",
                        "등기신청",
                        "등기부",
                        "부동산투",
                        "부동산등"
                ],
                "context": [
                        "부동산",
                        "부동산에",
                        "부동산의",
                        "부동산을",
                        "등기부"
                ]
        },
        "부동산투": {
                "synonyms": [],
                "related": [
                        "부동산",
                        "부동산에",
                        "부동산의",
                        "부동산을",
                        "등기신청",
                        "등기부",
                        "전세권",
                        "부동산등"
                ],
                "context": [
                        "부동산",
                        "부동산에",
                        "부동산의",
                        "부동산을",
                        "등기부"
                ]
        },
        "부동산등": {
                "synonyms": [],
                "related": [
                        "부동산",
                        "부동산에",
                        "부동산의",
                        "부동산을",
                        "등기신청",
                        "등기부",
                        "전세권",
                        "부동산투"
                ],
                "context": [
                        "부동산",
                        "부동산에",
                        "부동산의",
                        "부동산을",
                        "등기부"
                ]
        },
        "부동산실": {
                "synonyms": [],
                "related": [
                        "부동산",
                        "부동산에",
                        "부동산의",
                        "부동산을",
                        "등기신청",
                        "등기부",
                        "전세권",
                        "부동산투"
                ],
                "context": [
                        "부동산",
                        "부동산에",
                        "부동산의",
                        "부동산을",
                        "등기부"
                ]
        },
        "판결": {
                "synonyms": [],
                "related": [
                        "원심판결",
                        "원고의",
                        "형사소송",
                        "재판장",
                        "불법행위",
                        "판결에",
                        "판결한다",
                        "형사소송법"
                ],
                "context": [
                        "원심판결",
                        "원고의",
                        "불법행위",
                        "형사소송법",
                        "헌법재판"
                ]
        },
        "원심판결": {
                "synonyms": [],
                "related": [
                        "판결",
                        "원고의",
                        "형사소송",
                        "재판장",
                        "불법행위",
                        "판결에",
                        "판결한다",
                        "형사소송법"
                ],
                "context": [
                        "원고의",
                        "불법행위",
                        "형사소송법",
                        "헌법재판",
                        "처벌법"
                ]
        },
        "원고의": {
                "synonyms": [],
                "related": [
                        "판결",
                        "원심판결",
                        "형사소송",
                        "재판장",
                        "불법행위",
                        "판결에",
                        "판결한다",
                        "형사소송법"
                ],
                "context": [
                        "원심판결",
                        "불법행위",
                        "형사소송법",
                        "헌법재판",
                        "처벌법"
                ]
        },
        "형사소송": {
                "synonyms": [],
                "related": [
                        "판결",
                        "원심판결",
                        "원고의",
                        "재판장",
                        "불법행위",
                        "판결에",
                        "판결한다",
                        "형사소송법"
                ],
                "context": [
                        "원심판결",
                        "원고의",
                        "불법행위",
                        "형사소송법",
                        "헌법재판"
                ]
        },
        "재판장": {
                "synonyms": [],
                "related": [
                        "판결",
                        "원심판결",
                        "원고의",
                        "형사소송",
                        "불법행위",
                        "판결에",
                        "판결한다",
                        "형사소송법"
                ],
                "context": [
                        "원심판결",
                        "원고의",
                        "불법행위",
                        "형사소송법",
                        "헌법재판"
                ]
        },
        "불법행위": {
                "synonyms": [],
                "related": [
                        "판결",
                        "원심판결",
                        "원고의",
                        "형사소송",
                        "재판장",
                        "판결에",
                        "판결한다",
                        "형사소송법"
                ],
                "context": [
                        "원심판결",
                        "원고의",
                        "형사소송법",
                        "헌법재판",
                        "처벌법"
                ]
        },
        "판결에": {
                "synonyms": [],
                "related": [
                        "판결",
                        "원심판결",
                        "원고의",
                        "형사소송",
                        "재판장",
                        "불법행위",
                        "판결한다",
                        "형사소송법"
                ],
                "context": [
                        "원심판결",
                        "원고의",
                        "불법행위",
                        "형사소송법",
                        "헌법재판"
                ]
        },
        "판결한다": {
                "synonyms": [],
                "related": [
                        "판결",
                        "원심판결",
                        "원고의",
                        "형사소송",
                        "재판장",
                        "불법행위",
                        "판결에",
                        "형사소송법"
                ],
                "context": [
                        "원심판결",
                        "원고의",
                        "불법행위",
                        "형사소송법",
                        "헌법재판"
                ]
        },
        "형사소송법": {
                "synonyms": [],
                "related": [
                        "판결",
                        "원심판결",
                        "원고의",
                        "형사소송",
                        "재판장",
                        "불법행위",
                        "판결에",
                        "판결한다"
                ],
                "context": [
                        "원심판결",
                        "원고의",
                        "불법행위",
                        "헌법재판",
                        "처벌법"
                ]
        },
        "판결주": {
                "synonyms": [],
                "related": [
                        "판결",
                        "원심판결",
                        "원고의",
                        "형사소송",
                        "재판장",
                        "불법행위",
                        "판결에",
                        "판결한다"
                ],
                "context": [
                        "원심판결",
                        "원고의",
                        "불법행위",
                        "형사소송법",
                        "헌법재판"
                ]
        },
        "가정법원": {
                "synonyms": [],
                "related": [
                        "위자료청",
                        "혼인신고",
                        "서울가정법원",
                        "양육책임",
                        "법정상속",
                        "상속세부",
                        "위자료청구권",
                        "상속증여세법"
                ],
                "context": [
                        "위자료청",
                        "서울가정법원",
                        "법정상속",
                        "상속세부",
                        "상속증여세법"
                ]
        },
        "위자료청": {
                "synonyms": [],
                "related": [
                        "가정법원",
                        "혼인신고",
                        "서울가정법원",
                        "양육책임",
                        "법정상속",
                        "상속세부",
                        "위자료청구권",
                        "상속증여세법"
                ],
                "context": [
                        "가정법원",
                        "서울가정법원",
                        "법정상속",
                        "상속세부",
                        "상속증여세법"
                ]
        },
        "혼인신고": {
                "synonyms": [],
                "related": [
                        "가정법원",
                        "위자료청",
                        "서울가정법원",
                        "양육책임",
                        "법정상속",
                        "상속세부",
                        "위자료청구권",
                        "상속증여세법"
                ],
                "context": [
                        "가정법원",
                        "위자료청",
                        "서울가정법원",
                        "법정상속",
                        "상속세부"
                ]
        },
        "서울가정법원": {
                "synonyms": [],
                "related": [
                        "가정법원",
                        "위자료청",
                        "혼인신고",
                        "양육책임",
                        "법정상속",
                        "상속세부",
                        "위자료청구권",
                        "상속증여세법"
                ],
                "context": [
                        "가정법원",
                        "위자료청",
                        "법정상속",
                        "상속세부",
                        "상속증여세법"
                ]
        },
        "양육책임": {
                "synonyms": [],
                "related": [
                        "가정법원",
                        "위자료청",
                        "혼인신고",
                        "서울가정법원",
                        "법정상속",
                        "상속세부",
                        "위자료청구권",
                        "상속증여세법"
                ],
                "context": [
                        "가정법원",
                        "위자료청",
                        "서울가정법원",
                        "법정상속",
                        "상속세부"
                ]
        },
        "법정상속": {
                "synonyms": [],
                "related": [
                        "가정법원",
                        "위자료청",
                        "혼인신고",
                        "서울가정법원",
                        "양육책임",
                        "상속세부",
                        "위자료청구권",
                        "상속증여세법"
                ],
                "context": [
                        "가정법원",
                        "위자료청",
                        "서울가정법원",
                        "상속세부",
                        "상속증여세법"
                ]
        },
        "상속세부": {
                "synonyms": [],
                "related": [
                        "가정법원",
                        "위자료청",
                        "혼인신고",
                        "서울가정법원",
                        "양육책임",
                        "법정상속",
                        "위자료청구권",
                        "상속증여세법"
                ],
                "context": [
                        "가정법원",
                        "위자료청",
                        "서울가정법원",
                        "법정상속",
                        "상속증여세법"
                ]
        },
        "위자료청구권": {
                "synonyms": [],
                "related": [
                        "가정법원",
                        "위자료청",
                        "혼인신고",
                        "서울가정법원",
                        "양육책임",
                        "법정상속",
                        "상속세부",
                        "상속증여세법"
                ],
                "context": [
                        "가정법원",
                        "위자료청",
                        "서울가정법원",
                        "법정상속",
                        "상속세부"
                ]
        },
        "상속증여세법": {
                "synonyms": [],
                "related": [
                        "가정법원",
                        "위자료청",
                        "혼인신고",
                        "서울가정법원",
                        "양육책임",
                        "법정상속",
                        "상속세부",
                        "위자료청구권"
                ],
                "context": [
                        "가정법원",
                        "위자료청",
                        "서울가정법원",
                        "법정상속",
                        "상속세부"
                ]
        },
        "상속세법": {
                "synonyms": [],
                "related": [
                        "가정법원",
                        "위자료청",
                        "혼인신고",
                        "서울가정법원",
                        "양육책임",
                        "법정상속",
                        "상속세부",
                        "위자료청구권"
                ],
                "context": [
                        "가정법원",
                        "위자료청",
                        "서울가정법원",
                        "법정상속",
                        "상속세부"
                ]
        },
        "상법": {
                "synonyms": [],
                "related": [
                        "감사위원",
                        "감사원",
                        "회사로부",
                        "주주명부",
                        "부이사관",
                        "감사위원회",
                        "토지보상법",
                        "국가배상법"
                ],
                "context": [
                        "감사위원",
                        "감사원",
                        "회사로부",
                        "주주명부",
                        "부이사관"
                ]
        },
        "감사위원": {
                "synonyms": [],
                "related": [
                        "상법",
                        "감사원",
                        "회사로부",
                        "주주명부",
                        "부이사관",
                        "감사위원회",
                        "토지보상법",
                        "국가배상법"
                ],
                "context": [
                        "상법",
                        "감사원",
                        "회사로부",
                        "주주명부",
                        "부이사관"
                ]
        },
        "감사원": {
                "synonyms": [],
                "related": [
                        "상법",
                        "감사위원",
                        "회사로부",
                        "주주명부",
                        "부이사관",
                        "감사위원회",
                        "토지보상법",
                        "국가배상법"
                ],
                "context": [
                        "상법",
                        "감사위원",
                        "회사로부",
                        "주주명부",
                        "부이사관"
                ]
        },
        "회사로부": {
                "synonyms": [],
                "related": [
                        "상법",
                        "감사위원",
                        "감사원",
                        "주주명부",
                        "부이사관",
                        "감사위원회",
                        "토지보상법",
                        "국가배상법"
                ],
                "context": [
                        "상법",
                        "감사위원",
                        "감사원",
                        "주주명부",
                        "부이사관"
                ]
        },
        "주주명부": {
                "synonyms": [],
                "related": [
                        "상법",
                        "감사위원",
                        "감사원",
                        "회사로부",
                        "부이사관",
                        "감사위원회",
                        "토지보상법",
                        "국가배상법"
                ],
                "context": [
                        "상법",
                        "감사위원",
                        "감사원",
                        "회사로부",
                        "부이사관"
                ]
        },
        "부이사관": {
                "synonyms": [],
                "related": [
                        "상법",
                        "감사위원",
                        "감사원",
                        "회사로부",
                        "주주명부",
                        "감사위원회",
                        "토지보상법",
                        "국가배상법"
                ],
                "context": [
                        "상법",
                        "감사위원",
                        "감사원",
                        "회사로부",
                        "주주명부"
                ]
        },
        "감사위원회": {
                "synonyms": [],
                "related": [
                        "상법",
                        "감사위원",
                        "감사원",
                        "회사로부",
                        "주주명부",
                        "부이사관",
                        "토지보상법",
                        "국가배상법"
                ],
                "context": [
                        "상법",
                        "감사위원",
                        "감사원",
                        "회사로부",
                        "주주명부"
                ]
        },
        "토지보상법": {
                "synonyms": [],
                "related": [
                        "상법",
                        "감사위원",
                        "감사원",
                        "회사로부",
                        "주주명부",
                        "부이사관",
                        "감사위원회",
                        "국가배상법"
                ],
                "context": [
                        "상법",
                        "감사위원",
                        "감사원",
                        "회사로부",
                        "주주명부"
                ]
        },
        "국가배상법": {
                "synonyms": [],
                "related": [
                        "상법",
                        "감사위원",
                        "감사원",
                        "회사로부",
                        "주주명부",
                        "부이사관",
                        "감사위원회",
                        "토지보상법"
                ],
                "context": [
                        "상법",
                        "감사위원",
                        "감사원",
                        "회사로부",
                        "주주명부"
                ]
        },
        "외부감사": {
                "synonyms": [],
                "related": [
                        "상법",
                        "감사위원",
                        "감사원",
                        "회사로부",
                        "주주명부",
                        "부이사관",
                        "감사위원회",
                        "토지보상법"
                ],
                "context": [
                        "상법",
                        "감사위원",
                        "감사원",
                        "회사로부",
                        "주주명부"
                ]
        },
        "근로기준법": {
                "synonyms": [],
                "related": [
                        "최저임금법",
                        "노동조합법",
                        "노동위원회",
                        "근로의무",
                        "임금지급의무",
                        "선원근로",
                        "중앙노동위원회",
                        "해고처분"
                ],
                "context": [
                        "최저임금법",
                        "노동조합법",
                        "노동위원회",
                        "선원근로",
                        "중앙노동위원회"
                ]
        },
        "최저임금법": {
                "synonyms": [],
                "related": [
                        "근로기준법",
                        "노동조합법",
                        "노동위원회",
                        "근로의무",
                        "임금지급의무",
                        "선원근로",
                        "중앙노동위원회",
                        "해고처분"
                ],
                "context": [
                        "근로기준법",
                        "노동조합법",
                        "노동위원회",
                        "선원근로",
                        "중앙노동위원회"
                ]
        },
        "노동조합법": {
                "synonyms": [],
                "related": [
                        "근로기준법",
                        "최저임금법",
                        "노동위원회",
                        "근로의무",
                        "임금지급의무",
                        "선원근로",
                        "중앙노동위원회",
                        "해고처분"
                ],
                "context": [
                        "근로기준법",
                        "최저임금법",
                        "노동위원회",
                        "선원근로",
                        "중앙노동위원회"
                ]
        },
        "노동위원회": {
                "synonyms": [],
                "related": [
                        "근로기준법",
                        "최저임금법",
                        "노동조합법",
                        "근로의무",
                        "임금지급의무",
                        "선원근로",
                        "중앙노동위원회",
                        "해고처분"
                ],
                "context": [
                        "근로기준법",
                        "최저임금법",
                        "노동조합법",
                        "선원근로",
                        "중앙노동위원회"
                ]
        },
        "근로의무": {
                "synonyms": [],
                "related": [
                        "근로기준법",
                        "최저임금법",
                        "노동조합법",
                        "노동위원회",
                        "임금지급의무",
                        "선원근로",
                        "중앙노동위원회",
                        "해고처분"
                ],
                "context": [
                        "근로기준법",
                        "최저임금법",
                        "노동조합법",
                        "노동위원회",
                        "선원근로"
                ]
        },
        "임금지급의무": {
                "synonyms": [],
                "related": [
                        "근로기준법",
                        "최저임금법",
                        "노동조합법",
                        "노동위원회",
                        "근로의무",
                        "선원근로",
                        "중앙노동위원회",
                        "해고처분"
                ],
                "context": [
                        "근로기준법",
                        "최저임금법",
                        "노동조합법",
                        "노동위원회",
                        "선원근로"
                ]
        },
        "선원근로": {
                "synonyms": [],
                "related": [
                        "근로기준법",
                        "최저임금법",
                        "노동조합법",
                        "노동위원회",
                        "근로의무",
                        "임금지급의무",
                        "중앙노동위원회",
                        "해고처분"
                ],
                "context": [
                        "근로기준법",
                        "최저임금법",
                        "노동조합법",
                        "노동위원회",
                        "중앙노동위원회"
                ]
        },
        "중앙노동위원회": {
                "synonyms": [],
                "related": [
                        "근로기준법",
                        "최저임금법",
                        "노동조합법",
                        "노동위원회",
                        "근로의무",
                        "임금지급의무",
                        "선원근로",
                        "해고처분"
                ],
                "context": [
                        "근로기준법",
                        "최저임금법",
                        "노동조합법",
                        "노동위원회",
                        "선원근로"
                ]
        },
        "해고처분": {
                "synonyms": [],
                "related": [
                        "근로기준법",
                        "최저임금법",
                        "노동조합법",
                        "노동위원회",
                        "근로의무",
                        "임금지급의무",
                        "선원근로",
                        "중앙노동위원회"
                ],
                "context": [
                        "근로기준법",
                        "최저임금법",
                        "노동조합법",
                        "노동위원회",
                        "선원근로"
                ]
        },
        "근로지원": {
                "synonyms": [],
                "related": [
                        "근로기준법",
                        "최저임금법",
                        "노동조합법",
                        "노동위원회",
                        "근로의무",
                        "임금지급의무",
                        "선원근로",
                        "중앙노동위원회"
                ],
                "context": [
                        "근로기준법",
                        "최저임금법",
                        "노동조합법",
                        "노동위원회",
                        "선원근로"
                ]
        },
        "특허권": {
                "synonyms": [],
                "related": [
                        "특허청",
                        "특허청장",
                        "특허출원",
                        "특허법",
                        "특허법원",
                        "특허심판원",
                        "특허권의",
                        "특허권자"
                ],
                "context": [
                        "특허청",
                        "특허청장",
                        "특허출원",
                        "특허법",
                        "특허법원"
                ]
        },
        "특허청": {
                "synonyms": [],
                "related": [
                        "특허권",
                        "특허청장",
                        "특허출원",
                        "특허법",
                        "특허법원",
                        "특허심판원",
                        "특허권의",
                        "특허권자"
                ],
                "context": [
                        "특허청장",
                        "특허출원",
                        "특허법",
                        "특허법원",
                        "특허심판원"
                ]
        },
        "특허청장": {
                "synonyms": [],
                "related": [
                        "특허권",
                        "특허청",
                        "특허출원",
                        "특허법",
                        "특허법원",
                        "특허심판원",
                        "특허권의",
                        "특허권자"
                ],
                "context": [
                        "특허청",
                        "특허출원",
                        "특허법",
                        "특허법원",
                        "특허심판원"
                ]
        },
        "특허출원": {
                "synonyms": [],
                "related": [
                        "특허권",
                        "특허청",
                        "특허청장",
                        "특허법",
                        "특허법원",
                        "특허심판원",
                        "특허권의",
                        "특허권자"
                ],
                "context": [
                        "특허청",
                        "특허청장",
                        "특허법",
                        "특허법원",
                        "특허심판원"
                ]
        },
        "특허법": {
                "synonyms": [],
                "related": [
                        "특허권",
                        "특허청",
                        "특허청장",
                        "특허출원",
                        "특허법원",
                        "특허심판원",
                        "특허권의",
                        "특허권자"
                ],
                "context": [
                        "특허청",
                        "특허청장",
                        "특허출원",
                        "특허법원",
                        "특허심판원"
                ]
        },
        "특허법원": {
                "synonyms": [],
                "related": [
                        "특허권",
                        "특허청",
                        "특허청장",
                        "특허출원",
                        "특허법",
                        "특허심판원",
                        "특허권의",
                        "특허권자"
                ],
                "context": [
                        "특허청",
                        "특허청장",
                        "특허출원",
                        "특허법",
                        "특허심판원"
                ]
        },
        "특허심판원": {
                "synonyms": [],
                "related": [
                        "특허권",
                        "특허청",
                        "특허청장",
                        "특허출원",
                        "특허법",
                        "특허법원",
                        "특허권의",
                        "특허권자"
                ],
                "context": [
                        "특허청",
                        "특허청장",
                        "특허출원",
                        "특허법",
                        "특허법원"
                ]
        },
        "특허권의": {
                "synonyms": [
                        "특허권자"
                ],
                "related": [
                        "특허권",
                        "특허청",
                        "특허청장",
                        "특허출원",
                        "특허법",
                        "특허법원",
                        "특허심판원",
                        "특허권을"
                ],
                "context": [
                        "특허청",
                        "특허청장",
                        "특허출원",
                        "특허법",
                        "특허법원"
                ]
        },
        "특허권자": {
                "synonyms": [
                        "특허권의"
                ],
                "related": [
                        "특허권",
                        "특허청",
                        "특허청장",
                        "특허출원",
                        "특허법",
                        "특허법원",
                        "특허심판원",
                        "특허권을"
                ],
                "context": [
                        "특허청",
                        "특허청장",
                        "특허출원",
                        "특허법",
                        "특허법원"
                ]
        },
        "특허권을": {
                "synonyms": [],
                "related": [
                        "특허권",
                        "특허청",
                        "특허청장",
                        "특허출원",
                        "특허법",
                        "특허법원",
                        "특허심판원",
                        "특허권의"
                ],
                "context": [
                        "특허청",
                        "특허청장",
                        "특허출원",
                        "특허법",
                        "특허법원"
                ]
        }
}
        
        # 키워드 간 의미적 거리 매트릭스
        self.semantic_distance = self._build_semantic_distance_matrix()
        
        # 도메인별 용어 그룹
        self.domain_groups = self._build_domain_groups()
    
    def _build_semantic_distance_matrix(self) -> Dict[str, Dict[str, float]]:
        """의미적 거리 매트릭스 구축"""
        distance_matrix = {}
        
        for term, relations in self.semantic_relations.items():
            distance_matrix[term] = {}
            
            # 동의어는 거리 0.1
            for synonym in relations["synonyms"]:
                distance_matrix[term][synonym] = 0.1
            
            # 관련 용어는 거리 0.3
            for related in relations["related"]:
                distance_matrix[term][related] = 0.3
            
            # 컨텍스트 용어는 거리 0.5
            for context in relations["context"]:
                distance_matrix[term][context] = 0.5
            
            # 자기 자신은 거리 0
            distance_matrix[term][term] = 0.0
        
        return distance_matrix
    
    def _build_domain_groups(self) -> Dict[str, List[str]]:
        """도메인별 용어 그룹 구축"""
        domain_groups = defaultdict(list)
        
        for term, relations in self.semantic_relations.items():
            # 컨텍스트에서 도메인 추출
            for context in relations["context"]:
                if context in ["형사법", "민사법", "가족법", "상사법", "노동법", "부동산법", "특허법", "행정법"]:
                    domain_groups[context].append(term)
                    break
            else:
                domain_groups["기타"].append(term)
        
        return dict(domain_groups)
    
    def calculate_semantic_similarity(self, keyword1: str, keyword2: str) -> float:
        """두 키워드 간의 의미적 유사도 계산"""
        # 직접적인 의미적 관계 확인
        for term, relations in self.semantic_relations.items():
            if keyword1 == term:
                if keyword2 in relations["synonyms"]:
                    return 0.9
                elif keyword2 in relations["related"]:
                    return 0.7
                elif keyword2 in relations["context"]:
                    return 0.5
            
            if keyword2 == term:
                if keyword1 in relations["synonyms"]:
                    return 0.9
                elif keyword1 in relations["related"]:
                    return 0.7
                elif keyword1 in relations["context"]:
                    return 0.5
        
        # 부분 문자열 매칭
        if keyword1 in keyword2 or keyword2 in keyword1:
            return 0.6
        
        # 공통 문자 기반 유사도
        common_chars = set(keyword1) & set(keyword2)
        if common_chars:
            similarity = len(common_chars) / max(len(keyword1), len(keyword2))
            return similarity * 0.4
        
        return 0.0
    
    def find_semantic_related_keywords(self, target_keyword: str, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """의미적으로 관련된 키워드 찾기"""
        related_keywords = []
        
        for keyword, relations in self.semantic_relations.items():
            similarity = self.calculate_semantic_similarity(target_keyword, keyword)
            if similarity >= threshold:
                related_keywords.append((keyword, similarity))
        
        # 유사도 기준으로 정렬
        related_keywords.sort(key=lambda x: x[1], reverse=True)
        return related_keywords
    
    def expand_keywords_semantically(self, keywords: List[str], expansion_factor: float = 0.7) -> List[str]:
        """키워드를 의미적으로 확장"""
        expanded_keywords = set(keywords)
        
        for keyword in keywords:
            related_keywords = self.find_semantic_related_keywords(keyword, expansion_factor)
            for related_keyword, similarity in related_keywords[:5]:  # 상위 5개만 추가
                expanded_keywords.add(related_keyword)
        
        return list(expanded_keywords)
    
    def get_semantic_keyword_clusters(self, keywords: List[str]) -> Dict[str, List[str]]:
        """키워드의 의미적 클러스터 생성"""
        clusters = defaultdict(list)
        
        for keyword in keywords:
            # 가장 유사한 대표 키워드 찾기
            best_match = None
            best_similarity = 0.0
            
            for cluster_center, relations in self.semantic_relations.items():
                similarity = self.calculate_semantic_similarity(keyword, cluster_center)
                if similarity > best_similarity and similarity >= 0.5:
                    best_similarity = similarity
                    best_match = cluster_center
            
            if best_match:
                clusters[best_match].append(keyword)
            else:
                clusters[keyword].append(keyword)  # 독립 클러스터
        
        return dict(clusters)
    
    def analyze_keyword_semantic_coverage(self, answer: str, keywords: List[str]) -> Dict[str, any]:
        """키워드의 의미적 커버리지 분석"""
        answer_lower = answer.lower()
        
        # 직접 매칭
        direct_matches = [kw for kw in keywords if kw.lower() in answer_lower]
        
        # 의미적 매칭
        semantic_matches = []
        for keyword in keywords:
            related_keywords = self.find_semantic_related_keywords(keyword, 0.6)
            for related_kw, similarity in related_keywords:
                if related_kw.lower() in answer_lower:
                    semantic_matches.append((keyword, related_kw, similarity))
        
        # 클러스터 분석
        clusters = self.get_semantic_keyword_clusters(keywords)
        cluster_coverage = {}
        for cluster_center, cluster_keywords in clusters.items():
            cluster_matches = [kw for kw in cluster_keywords if kw.lower() in answer_lower]
            cluster_coverage[cluster_center] = {
                "total_keywords": len(cluster_keywords),
                "matched_keywords": len(cluster_matches),
                "coverage_ratio": len(cluster_matches) / len(cluster_keywords) if cluster_keywords else 0
            }
        
        return {
            "direct_matches": direct_matches,
            "semantic_matches": semantic_matches,
            "cluster_coverage": cluster_coverage,
            "overall_coverage": len(direct_matches) / len(keywords) if keywords else 0,
            "semantic_coverage": len(set([match[0] for match in semantic_matches])) / len(keywords) if keywords else 0
        }
    
    def get_semantic_keyword_recommendations(self, question: str, query_type: str, base_keywords: List[str]) -> Dict[str, any]:
        """의미적 키워드 추천"""
        # 질문에서 도메인 추출
        question_domains = []
        for domain, domain_keywords in self.domain_groups.items():
            for keyword in domain_keywords:
                if keyword in question:
                    question_domains.append(domain)
                    break
        
        # 도메인별 관련 키워드 추천
        domain_recommendations = {}
        for domain in question_domains:
            if domain in self.domain_groups:
                domain_keywords = self.domain_groups[domain]
                # 기존 키워드와 유사한 도메인 키워드 추천
                recommended = []
                for base_kw in base_keywords:
                    for domain_kw in domain_keywords:
                        similarity = self.calculate_semantic_similarity(base_kw, domain_kw)
                        if similarity >= 0.6 and domain_kw not in base_keywords:
                            recommended.append((domain_kw, similarity))
                
                recommended.sort(key=lambda x: x[1], reverse=True)
                domain_recommendations[domain] = [kw for kw, sim in recommended[:5]]
        
        # 의미적 확장 키워드
        expanded_keywords = self.expand_keywords_semantically(base_keywords, 0.7)
        new_keywords = [kw for kw in expanded_keywords if kw not in base_keywords]
        
        return {
            "domain_recommendations": domain_recommendations,
            "expanded_keywords": new_keywords,
            "semantic_clusters": self.get_semantic_keyword_clusters(base_keywords),
            "recommended_keywords": new_keywords[:10]  # 상위 10개 추천
        }
    
    def get_domain_statistics(self) -> Dict[str, any]:
        """도메인별 통계 정보"""
        stats = {}
        for domain, keywords in self.domain_groups.items():
            stats[domain] = {
                "total_keywords": len(keywords),
                "top_keywords": keywords[:5],  # 상위 5개 키워드
                "semantic_relations_count": len([kw for kw in keywords if kw in self.semantic_relations])
            }
        return stats
    
    def export_semantic_relations(self, output_file: str):
        """의미적 관계를 JSON 파일로 내보내기"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.semantic_relations, f, ensure_ascii=False, indent=2)
    
    def load_semantic_relations(self, input_file: str):
        """JSON 파일에서 의미적 관계 로드"""
        with open(input_file, 'r', encoding='utf-8') as f:
            self.semantic_relations = json.load(f)
        self.semantic_distance = self._build_semantic_distance_matrix()
        self.domain_groups = self._build_domain_groups()

class EnhancedKeywordMapper:
    """통합된 향상된 키워드 매핑 시스템"""
    
    def __init__(self):
        self.legal_mapper = LegalKeywordMapper()
        self.context_mapper = ContextAwareKeywordMapper()
        self.adaptive_mapper = AdaptiveKeywordMapper()
        self.semantic_mapper = EnhancedSemanticKeywordMapper()
    
    def get_comprehensive_keyword_mapping(self, question: str, query_type: str) -> Dict[str, any]:
        """종합적인 키워드 매핑 결과 반환"""
        # 1. 기본 키워드 매핑
        base_keywords = self.legal_mapper.get_keywords_for_question(question, query_type)
        
        # 2. 가중치 기반 키워드 매핑
        weighted_coverage = self.legal_mapper.calculate_weighted_keyword_coverage("", query_type, question)
        weighted_keywords = self.legal_mapper.get_weighted_keywords_for_question(question, query_type)
        
        # 3. 컨텍스트 인식 키워드 매핑
        contextual_data = self.context_mapper.get_contextual_keywords(question, query_type)
        
        # 4. 적응형 키워드 매핑
        adaptive_data = self.adaptive_mapper.get_adaptive_keyword_mapping(question, query_type)
        
        # 5. 의미적 키워드 매핑
        semantic_data = self.semantic_mapper.get_semantic_keyword_recommendations(question, query_type, base_keywords)
        
        # 6. 모든 키워드 통합 및 중복 제거
        all_keywords = list(set(
            base_keywords + 
            contextual_data["all_keywords"] + 
            adaptive_data["all_keywords"] + 
            semantic_data["recommended_keywords"]
        ))
        
        # 7. 키워드 우선순위 계산
        keyword_priority = self._calculate_comprehensive_priority(
            all_keywords, question, query_type, 
            weighted_keywords, contextual_data, adaptive_data, semantic_data
        )
        
        return {
            "base_keywords": base_keywords,
            "weighted_keywords": weighted_keywords,
            "contextual_data": contextual_data,
            "adaptive_data": adaptive_data,
            "semantic_data": semantic_data,
            "all_keywords": all_keywords,
            "keyword_priority": keyword_priority,
            "comprehensive_analysis": self._generate_comprehensive_analysis(
                question, query_type, all_keywords, keyword_priority
            )
        }
    
    def _calculate_comprehensive_priority(self, keywords: List[str], question: str, query_type: str,
                                        weighted_keywords: Dict, contextual_data: Dict, 
                                        adaptive_data: Dict, semantic_data: Dict) -> Dict[str, float]:
        """종합적인 키워드 우선순위 계산"""
        priority_scores = {}
        
        for keyword in keywords:
            score = 0.0
            
            # 1. 가중치 기반 점수 (30%)
            if keyword in weighted_keywords.get("core", []):
                score += 0.3 * 1.0
            elif keyword in weighted_keywords.get("important", []):
                score += 0.3 * 0.8
            elif keyword in weighted_keywords.get("supporting", []):
                score += 0.3 * 0.6
            
            # 2. 컨텍스트 기반 점수 (25%)
            if keyword in contextual_data.get("contextual_keywords", []):
                score += 0.25 * 0.9
            elif keyword in contextual_data.get("base_keywords", []):
                score += 0.25 * 0.7
            
            # 3. 적응형 점수 (25%)
            effectiveness_score = adaptive_data.get("keyword_scores", {}).get(keyword, 0.5)
            score += 0.25 * effectiveness_score
            
            # 4. 의미적 점수 (20%)
            if keyword in semantic_data.get("recommended_keywords", []):
                score += 0.2 * 0.8
            elif keyword in semantic_data.get("expanded_keywords", []):
                score += 0.2 * 0.6
            
            priority_scores[keyword] = min(score, 1.0)
        
        return priority_scores
    
    def _generate_comprehensive_analysis(self, question: str, query_type: str, 
                                       keywords: List[str], priority_scores: Dict[str, float]) -> Dict[str, any]:
        """종합적인 분석 결과 생성"""
        # 상위 우선순위 키워드
        top_keywords = sorted(priority_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 키워드 분포 분석
        high_priority = [kw for kw, score in priority_scores.items() if score >= 0.8]
        medium_priority = [kw for kw, score in priority_scores.items() if 0.5 <= score < 0.8]
        low_priority = [kw for kw, score in priority_scores.items() if score < 0.5]
        
        return {
            "top_keywords": top_keywords,
            "priority_distribution": {
                "high_priority": len(high_priority),
                "medium_priority": len(medium_priority),
                "low_priority": len(low_priority)
            },
            "keyword_diversity_score": len(set(keywords)) / len(keywords) if keywords else 0,
            "recommendations": self._generate_final_recommendations(question, query_type, priority_scores),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _generate_final_recommendations(self, question: str, query_type: str, 
                                      priority_scores: Dict[str, float]) -> List[str]:
        """최종 권장사항 생성"""
        recommendations = []
        
        # 고우선순위 키워드가 부족한 경우
        high_priority_count = sum(1 for score in priority_scores.values() if score >= 0.8)
        if high_priority_count < 3:
            recommendations.append("고우선순위 키워드 추가 필요")
        
        # 키워드 다양성 부족한 경우
        unique_keywords = len(set(priority_scores.keys()))
        if unique_keywords < 5:
            recommendations.append("키워드 다양성 확보 필요")
        
        # 컨텍스트별 권장사항
        context = self.context_mapper.identify_context(question)
        if context == "절차형":
            recommendations.append("절차 관련 키워드 강화 권장")
        elif context == "비교형":
            recommendations.append("비교 분석 키워드 추가 권장")
        
        return recommendations
    
    def update_feedback(self, question: str, keywords: List[str], user_rating: float, 
                       answer_quality: float, query_type: str = ""):
        """사용자 피드백 업데이트"""
        self.adaptive_mapper.update_keyword_effectiveness(
            question, keywords, user_rating, answer_quality, query_type
        )
    
    def get_keyword_effectiveness_report(self) -> Dict[str, any]:
        """키워드 효과성 보고서"""
        return {
            "adaptive_insights": self.adaptive_mapper.get_learning_insights(),
            "semantic_analysis": self.semantic_mapper.get_semantic_keyword_clusters([]),
            "context_patterns": self.context_mapper.context_patterns,
            "legal_keyword_coverage": len(self.legal_mapper.KEYWORD_MAPPING)
        }
