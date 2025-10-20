#!/usr/bin/env python3
"""
법률 용어 사전 확장을 위한 용어 추출 스크립트
기존 법령 및 판례 데이터에서 법률 용어를 자동으로 추출하고 분류합니다.
"""

import os
import json
import re
import logging
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/legal_term_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LegalTerm:
    """법률 용어 데이터 클래스"""
    term: str
    category: str
    domain: str
    frequency: int
    sources: List[str]
    synonyms: List[str]
    related_terms: List[str]
    context: List[str]
    confidence: float

class LegalTermExtractor:
    """법률 용어 추출기"""
    
    def __init__(self):
        self.extracted_terms: Dict[str, LegalTerm] = {}
        self.domain_patterns = self._initialize_domain_patterns()
        self.legal_patterns = self._initialize_legal_patterns()
        self.stop_words = self._initialize_stop_words()
        
    def _initialize_domain_patterns(self) -> Dict[str, List[str]]:
        """도메인별 패턴 초기화"""
        return {
            "형사법": [
                r"범죄", r"처벌", r"형벌", r"구속", r"기소", r"공소", r"피고", r"검사",
                r"변호사", r"재판", r"판결", r"형사소송", r"불법행위", r"과실", r"고의"
            ],
            "민사법": [
                r"계약", r"손해배상", r"소유권", r"채권", r"채무", r"이행", r"위반",
                r"해지", r"해제", r"무효", r"취소", r"민사소송", r"소장", r"답변서"
            ],
            "가족법": [
                r"혼인", r"이혼", r"상속", r"양육", r"위자료", r"재산분할", r"양육권",
                r"면접교섭권", r"양육비", r"가정법원", r"가족법", r"이혼법"
            ],
            "상사법": [
                r"회사", r"주식", r"어음", r"수표", r"상행위", r"회사법", r"상법",
                r"주주", r"이사", r"감사", r"자본금", r"주식회사"
            ],
            "노동법": [
                r"근로", r"근로자", r"근로계약", r"임금", r"근로시간", r"해고",
                r"부당해고", r"노동위원회", r"근로기준법", r"노동조합법"
            ],
            "부동산법": [
                r"부동산", r"토지", r"건물", r"등기", r"소유권이전", r"매매",
                r"임대차", r"전세", r"월세", r"등기부등본", r"부동산등기법"
            ],
            "특허법": [
                r"특허", r"특허권", r"특허출원", r"특허등록", r"특허침해", r"발명",
                r"특허청", r"특허심판원", r"특허법", r"특허심사"
            ],
            "행정법": [
                r"행정처분", r"행정소송", r"행정법", r"허가", r"인가", r"승인",
                r"신고", r"신청", r"행정기관", r"공무원", r"행정절차"
            ]
        }
    
    def _initialize_legal_patterns(self) -> List[str]:
        """법률 용어 패턴 초기화"""
        return [
            # 조문 패턴
            r"제\d+조",
            r"제\d+항",
            r"제\d+호",
            r"제\d+장",
            r"제\d+절",
            r"제\d+편",
            
            # 법률명 패턴
            r"[가-힣]+법",
            r"[가-힣]+규칙",
            r"[가-힣]+령",
            r"[가-힣]+시행령",
            r"[가-힣]+시행규칙",
            
            # 권리/의무 패턴
            r"[가-힣]+권",
            r"[가-힣]+의무",
            r"[가-힣]+책임",
            r"[가-힣]+의무",
            
            # 절차 패턴
            r"[가-힣]+절차",
            r"[가-힣]+신청",
            r"[가-힣]+신고",
            r"[가-힣]+허가",
            r"[가-힣]+인가",
            r"[가-힣]+승인",
            
            # 기관 패턴
            r"[가-힣]+원",
            r"[가-힣]+청",
            r"[가-힣]+부",
            r"[가-힣]+위원회",
            r"[가-힣]+법원",
            
            # 행위 패턴
            r"[가-힣]+행위",
            r"[가-힣]+처분",
            r"[가-힣]+결정",
            r"[가-힣]+명령",
            r"[가-힣]+지시"
        ]
    
    def _initialize_stop_words(self) -> Set[str]:
        """불용어 초기화"""
        return {
            "것", "수", "등", "및", "또는", "그", "이", "저", "의", "가", "을", "를",
            "에", "에서", "로", "으로", "와", "과", "는", "은", "도", "만", "부터",
            "까지", "까지의", "에의", "에대한", "에관한", "에따른", "에의한"
        }
    
    def extract_terms_from_laws(self, law_data_dir: str) -> Dict[str, LegalTerm]:
        """법령 데이터에서 용어 추출"""
        logger.info(f"법령 데이터에서 용어 추출 시작: {law_data_dir}")
        
        extracted_terms = defaultdict(lambda: {
            'frequency': 0,
            'sources': [],
            'domains': set(),
            'contexts': set()
        })
        
        # 법령 파일들 처리
        for root, dirs, files in os.walk(law_data_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # 법령 데이터에서 용어 추출
                        if 'laws' in data:
                            for law in data['laws']:
                                self._extract_from_law(law, extracted_terms, file_path)
                                
                    except Exception as e:
                        logger.error(f"파일 처리 오류 {file_path}: {e}")
                        continue
        
        # LegalTerm 객체로 변환
        legal_terms = {}
        for term, data in extracted_terms.items():
            if len(term) >= 2 and term not in self.stop_words:  # 최소 길이 및 불용어 필터링
                legal_terms[term] = LegalTerm(
                    term=term,
                    category=self._categorize_term(term),
                    domain=self._determine_domain(term, data['domains']),
                    frequency=data['frequency'],
                    sources=data['sources'],
                    synonyms=[],
                    related_terms=[],
                    context=list(data['contexts']),
                    confidence=self._calculate_confidence(term, data)
                )
        
        logger.info(f"법령에서 추출된 용어 수: {len(legal_terms)}")
        return legal_terms
    
    def extract_terms_from_precedents(self, precedent_data_dir: str) -> Dict[str, LegalTerm]:
        """판례 데이터에서 용어 추출"""
        logger.info(f"판례 데이터에서 용어 추출 시작: {precedent_data_dir}")
        
        extracted_terms = defaultdict(lambda: {
            'frequency': 0,
            'sources': [],
            'domains': set(),
            'contexts': set()
        })
        
        # 판례 파일들 처리
        for root, dirs, files in os.walk(precedent_data_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # 판례 데이터에서 용어 추출
                        if 'cases' in data:
                            for case in data['cases']:
                                self._extract_from_precedent(case, extracted_terms, file_path)
                                
                    except Exception as e:
                        logger.error(f"파일 처리 오류 {file_path}: {e}")
                        continue
        
        # LegalTerm 객체로 변환
        legal_terms = {}
        for term, data in extracted_terms.items():
            if len(term) >= 2 and term not in self.stop_words:
                legal_terms[term] = LegalTerm(
                    term=term,
                    category=self._categorize_term(term),
                    domain=self._determine_domain(term, data['domains']),
                    frequency=data['frequency'],
                    sources=data['sources'],
                    synonyms=[],
                    related_terms=[],
                    context=list(data['contexts']),
                    confidence=self._calculate_confidence(term, data)
                )
        
        logger.info(f"판례에서 추출된 용어 수: {len(legal_terms)}")
        return legal_terms
    
    def _extract_from_law(self, law: Dict[str, Any], extracted_terms: Dict, file_path: str):
        """개별 법령에서 용어 추출"""
        # 법령명에서 용어 추출
        if 'law_name' in law and law['law_name']:
            self._extract_terms_from_text(law['law_name'], extracted_terms, file_path, "법령명")
        
        # 조문에서 용어 추출
        if 'articles' in law:
            for article in law['articles']:
                if 'article_title' in article and article['article_title']:
                    self._extract_terms_from_text(article['article_title'], extracted_terms, file_path, "조문제목")
                
                if 'article_content' in article and article['article_content']:
                    self._extract_terms_from_text(article['article_content'], extracted_terms, file_path, "조문내용")
    
    def _extract_from_precedent(self, case: Dict[str, Any], extracted_terms: Dict, file_path: str):
        """개별 판례에서 용어 추출"""
        # 사건명에서 용어 추출
        if 'case_name' in case and case['case_name']:
            self._extract_terms_from_text(case['case_name'], extracted_terms, file_path, "사건명")
        
        # 판시사항에서 용어 추출
        if 'sections' in case:
            for section in case['sections']:
                if section.get('has_content', False) and section.get('section_content'):
                    section_type = section.get('section_type_korean', '기타')
                    self._extract_terms_from_text(section['section_content'], extracted_terms, file_path, section_type)
    
    def _extract_terms_from_text(self, text: str, extracted_terms: Dict, file_path: str, context: str):
        """텍스트에서 용어 추출"""
        if not text or not isinstance(text, str):
            return
        
        # 패턴 매칭으로 용어 추출
        for pattern in self.legal_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]  # 그룹이 있는 경우 첫 번째 그룹 사용
                
                if match and len(match) >= 2:
                    extracted_terms[match]['frequency'] += 1
                    extracted_terms[match]['sources'].append(file_path)
                    extracted_terms[match]['contexts'].add(context)
                    
                    # 도메인 분류
                    domain = self._classify_domain(match)
                    if domain:
                        extracted_terms[match]['domains'].add(domain)
        
        # 일반적인 법률 용어 추출 (2-4글자 한글)
        general_terms = re.findall(r'[가-힣]{2,4}', text)
        for term in general_terms:
            if term not in self.stop_words and self._is_legal_term(term):
                extracted_terms[term]['frequency'] += 1
                extracted_terms[term]['sources'].append(file_path)
                extracted_terms[term]['contexts'].add(context)
                
                domain = self._classify_domain(term)
                if domain:
                    extracted_terms[term]['domains'].add(domain)
    
    def _classify_domain(self, term: str) -> str:
        """용어의 도메인 분류"""
        for domain, patterns in self.domain_patterns.items():
            for pattern in patterns:
                if re.search(pattern, term):
                    return domain
        return "기타"
    
    def _is_legal_term(self, term: str) -> bool:
        """법률 용어 여부 판단"""
        legal_indicators = [
            '법', '규칙', '령', '권', '의무', '책임', '절차', '신청', '신고',
            '허가', '인가', '승인', '원', '청', '부', '위원회', '법원',
            '행위', '처분', '결정', '명령', '지시', '소송', '재판', '판결'
        ]
        
        return any(indicator in term for indicator in legal_indicators)
    
    def _categorize_term(self, term: str) -> str:
        """용어 카테고리 분류"""
        if '법' in term or '규칙' in term or '령' in term:
            return "법률명"
        elif '조' in term or '항' in term or '호' in term:
            return "조문"
        elif '권' in term:
            return "권리"
        elif '의무' in term or '책임' in term:
            return "의무"
        elif '절차' in term or '신청' in term or '신고' in term:
            return "절차"
        elif '원' in term or '청' in term or '부' in term:
            return "기관"
        elif '소송' in term or '재판' in term or '판결' in term:
            return "소송"
        else:
            return "일반"
    
    def _determine_domain(self, term: str, domains: Set[str]) -> str:
        """주요 도메인 결정"""
        if not domains:
            return "기타"
        
        # 가장 빈번한 도메인 반환
        domain_counts = Counter(domains)
        return domain_counts.most_common(1)[0][0]
    
    def _calculate_confidence(self, term: str, data: Dict) -> float:
        """용어 신뢰도 계산"""
        confidence = 0.0
        
        # 빈도수 기반 점수 (0-0.4)
        frequency_score = min(data['frequency'] / 10.0, 0.4)
        confidence += frequency_score
        
        # 소스 다양성 점수 (0-0.3)
        source_diversity = min(len(set(data['sources'])) / 5.0, 0.3)
        confidence += source_diversity
        
        # 컨텍스트 다양성 점수 (0-0.2)
        context_diversity = min(len(data['contexts']) / 3.0, 0.2)
        confidence += context_diversity
        
        # 도메인 명확성 점수 (0-0.1)
        if len(data['domains']) == 1:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def merge_and_deduplicate(self, law_terms: Dict[str, LegalTerm], precedent_terms: Dict[str, LegalTerm]) -> Dict[str, LegalTerm]:
        """용어 통합 및 중복 제거"""
        logger.info("용어 통합 및 중복 제거 시작")
        
        merged_terms = {}
        
        # 법령 용어 추가
        for term, legal_term in law_terms.items():
            merged_terms[term] = legal_term
        
        # 판례 용어 통합
        for term, legal_term in precedent_terms.items():
            if term in merged_terms:
                # 중복된 용어의 경우 빈도수와 소스 통합
                existing_term = merged_terms[term]
                existing_term.frequency += legal_term.frequency
                existing_term.sources.extend(legal_term.sources)
                existing_term.context.extend(legal_term.context)
                existing_term.confidence = max(existing_term.confidence, legal_term.confidence)
            else:
                merged_terms[term] = legal_term
        
        # 품질 필터링 (신뢰도 0.3 이상, 빈도수 2 이상)
        filtered_terms = {
            term: legal_term for term, legal_term in merged_terms.items()
            if legal_term.confidence >= 0.3 and legal_term.frequency >= 2
        }
        
        logger.info(f"통합 후 용어 수: {len(merged_terms)}")
        logger.info(f"품질 필터링 후 용어 수: {len(filtered_terms)}")
        
        return filtered_terms
    
    def generate_semantic_relations(self, terms: Dict[str, LegalTerm]) -> Dict[str, Dict[str, List[str]]]:
        """의미적 관계 생성"""
        logger.info("의미적 관계 생성 시작")
        
        semantic_relations = {}
        
        # 도메인별 그룹화
        domain_groups = defaultdict(list)
        for term, legal_term in terms.items():
            domain_groups[legal_term.domain].append(term)
        
        # 각 도메인별로 의미적 관계 생성
        for domain, domain_terms in domain_groups.items():
            if len(domain_terms) < 3:
                continue
            
            # 상위 빈도수 용어들을 대표 용어로 선택
            domain_term_freq = [(term, terms[term].frequency) for term in domain_terms]
            domain_term_freq.sort(key=lambda x: x[1], reverse=True)
            
            representative_terms = [term for term, freq in domain_term_freq[:5]]
            
            if representative_terms:
                main_term = representative_terms[0]
                synonyms = representative_terms[1:3] if len(representative_terms) > 1 else []
                related_terms = domain_terms[:10]  # 관련 용어
                
                semantic_relations[main_term] = {
                    "synonyms": synonyms,
                    "related": related_terms,
                    "context": [domain]
                }
        
        logger.info(f"생성된 의미적 관계 수: {len(semantic_relations)}")
        return semantic_relations
    
    def save_results(self, terms: Dict[str, LegalTerm], semantic_relations: Dict[str, Dict[str, List[str]]], output_dir: str):
        """결과 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 용어 사전 저장
        terms_dict = {}
        for term, legal_term in terms.items():
            terms_dict[term] = {
                "term": legal_term.term,
                "category": legal_term.category,
                "domain": legal_term.domain,
                "frequency": legal_term.frequency,
                "sources": legal_term.sources,
                "synonyms": legal_term.synonyms,
                "related_terms": legal_term.related_terms,
                "context": legal_term.context,
                "confidence": legal_term.confidence
            }
        
        terms_file = os.path.join(output_dir, "extracted_legal_terms.json")
        with open(terms_file, 'w', encoding='utf-8') as f:
            json.dump(terms_dict, f, ensure_ascii=False, indent=2)
        
        # 의미적 관계 저장
        relations_file = os.path.join(output_dir, "semantic_relations.json")
        with open(relations_file, 'w', encoding='utf-8') as f:
            json.dump(semantic_relations, f, ensure_ascii=False, indent=2)
        
        # 통계 보고서 생성
        self._generate_statistics_report(terms, semantic_relations, output_dir)
        
        logger.info(f"결과 저장 완료: {output_dir}")
    
    def _generate_statistics_report(self, terms: Dict[str, LegalTerm], semantic_relations: Dict[str, Dict[str, List[str]]], output_dir: str):
        """통계 보고서 생성"""
        stats = {
            "extraction_summary": {
                "total_terms": len(terms),
                "total_semantic_relations": len(semantic_relations),
                "extraction_date": datetime.now().isoformat()
            },
            "domain_distribution": {},
            "category_distribution": {},
            "confidence_distribution": {},
            "frequency_distribution": {}
        }
        
        # 도메인별 분포
        domain_counts = Counter(term.domain for term in terms.values())
        stats["domain_distribution"] = dict(domain_counts)
        
        # 카테고리별 분포
        category_counts = Counter(term.category for term in terms.values())
        stats["category_distribution"] = dict(category_counts)
        
        # 신뢰도 분포
        confidence_ranges = {"high": 0, "medium": 0, "low": 0}
        for term in terms.values():
            if term.confidence >= 0.7:
                confidence_ranges["high"] += 1
            elif term.confidence >= 0.4:
                confidence_ranges["medium"] += 1
            else:
                confidence_ranges["low"] += 1
        stats["confidence_distribution"] = confidence_ranges
        
        # 빈도수 분포
        frequency_ranges = {"high": 0, "medium": 0, "low": 0}
        for term in terms.values():
            if term.frequency >= 10:
                frequency_ranges["high"] += 1
            elif term.frequency >= 5:
                frequency_ranges["medium"] += 1
            else:
                frequency_ranges["low"] += 1
        stats["frequency_distribution"] = frequency_ranges
        
        # 보고서 저장
        report_file = os.path.join(output_dir, "extraction_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

def main():
    """메인 실행 함수"""
    logger.info("법률 용어 추출 시작")
    
    # 데이터 디렉토리 설정
    law_data_dir = "data/processed/assembly/law"
    precedent_data_dir = "data/processed/assembly/precedent"
    output_dir = "data/extracted_terms"
    
    # 추출기 초기화
    extractor = LegalTermExtractor()
    
    try:
        # 법령에서 용어 추출
        law_terms = extractor.extract_terms_from_laws(law_data_dir)
        
        # 판례에서 용어 추출
        precedent_terms = extractor.extract_terms_from_precedents(precedent_data_dir)
        
        # 용어 통합 및 중복 제거
        merged_terms = extractor.merge_and_deduplicate(law_terms, precedent_terms)
        
        # 의미적 관계 생성
        semantic_relations = extractor.generate_semantic_relations(merged_terms)
        
        # 결과 저장
        extractor.save_results(merged_terms, semantic_relations, output_dir)
        
        logger.info("법률 용어 추출 완료")
        
    except Exception as e:
        logger.error(f"용어 추출 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()
