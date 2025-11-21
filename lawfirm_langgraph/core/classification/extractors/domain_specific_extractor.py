import re
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from collections import defaultdict

from .hybrid_keyword_manager import HybridKeywordManager, ExpansionStrategy

logger = get_logger(__name__)

class LegalDomain(Enum):
    """법률 도메인 열거형"""
    CIVIL_LAW = "민사법"
    CRIMINAL_LAW = "형사법"
    FAMILY_LAW = "가족법"
    COMMERCIAL_LAW = "상사법"
    LABOR_LAW = "노동법"
    REAL_ESTATE_LAW = "부동산법"
    INTELLECTUAL_PROPERTY_LAW = "지적재산권법"
    TAX_LAW = "세법"
    CIVIL_PROCEDURE_LAW = "민사소송법"
    CRIMINAL_PROCEDURE_LAW = "형사소송법"
    GENERAL = "기타/일반"

class DomainSpecificExtractor:
    """도메인별 특화 용어 추출기 (하이브리드 키워드 관리 시스템 사용)"""
    
    def __init__(self, 
                 data_dir: str = "data",
                 cache_dir: str = "data/cache",
                 min_keyword_threshold: int = 20,
                 expansion_strategy: ExpansionStrategy = ExpansionStrategy.HYBRID):
        self.logger = get_logger(__name__)
        self.domain_patterns = self._initialize_domain_patterns()
        
        # 하이브리드 키워드 매니저 초기화
        self.keyword_manager = HybridKeywordManager(
            data_dir=data_dir,
            cache_dir=cache_dir,
            min_keyword_threshold=min_keyword_threshold,
            expansion_strategy=expansion_strategy
        )
        
        # 도메인 매핑
        self.domain_mapping = {
            LegalDomain.CIVIL_LAW: '민사법',
            LegalDomain.CRIMINAL_LAW: '형사법',
            LegalDomain.FAMILY_LAW: '가족법',
            LegalDomain.COMMERCIAL_LAW: '상사법',
            LegalDomain.LABOR_LAW: '노동법',
            LegalDomain.REAL_ESTATE_LAW: '부동산법',
            LegalDomain.INTELLECTUAL_PROPERTY_LAW: '지적재산권법',
            LegalDomain.TAX_LAW: '세법',
            LegalDomain.CIVIL_PROCEDURE_LAW: '민사소송법',
            LegalDomain.CRIMINAL_PROCEDURE_LAW: '형사소송법',
            LegalDomain.GENERAL: '기타/일반'
        }
        
        # 키워드 캐시 (초기화 시 로드)
        self._domain_keywords_cache = {}
        self._cache_initialized = False
    
    def _initialize_domain_patterns(self) -> Dict[LegalDomain, Dict[str, List[str]]]:
        """도메인별 특화 패턴 초기화"""
        return {
            LegalDomain.CIVIL_LAW: {
                "contract_terms": [
                    r'[가-힣]+계약',
                    r'계약[가-힣]+',
                    r'[가-힣]+계약서',
                    r'계약[가-힣]*서'
                ],
                "damage_terms": [
                    r'손해[가-힣]*',
                    r'[가-힣]*손해',
                    r'배상[가-힣]*',
                    r'[가-힣]*배상'
                ],
                "ownership_terms": [
                    r'소유권[가-힣]*',
                    r'[가-힣]*소유권',
                    r'소유[가-힣]*',
                    r'[가-힣]*소유'
                ],
                "obligation_terms": [
                    r'의무[가-힣]*',
                    r'[가-힣]*의무',
                    r'책임[가-힣]*',
                    r'[가-힣]*책임'
                ]
            },
            LegalDomain.CRIMINAL_LAW: {
                "crime_terms": [
                    r'[가-힣]+죄',
                    r'죄[가-힣]*',
                    r'범죄[가-힣]*',
                    r'[가-힣]*범죄'
                ],
                "penalty_terms": [
                    r'형[가-힣]*',
                    r'[가-힣]*형',
                    r'처벌[가-힣]*',
                    r'[가-힣]*처벌'
                ],
                "investigation_terms": [
                    r'수사[가-힣]*',
                    r'[가-힣]*수사',
                    r'기소[가-힣]*',
                    r'[가-힣]*기소'
                ]
            },
            LegalDomain.FAMILY_LAW: {
                "marriage_terms": [
                    r'혼인[가-힣]*',
                    r'[가-힣]*혼인',
                    r'결혼[가-힣]*',
                    r'[가-힣]*결혼'
                ],
                "divorce_terms": [
                    r'이혼[가-힣]*',
                    r'[가-힣]*이혼',
                    r'별거[가-힣]*',
                    r'[가-힣]*별거'
                ],
                "parental_terms": [
                    r'양육[가-힣]*',
                    r'[가-힣]*양육',
                    r'친권[가-힣]*',
                    r'[가-힣]*친권'
                ]
            },
            LegalDomain.COMMERCIAL_LAW: {
                "company_terms": [
                    r'회사[가-힣]*',
                    r'[가-힣]*회사',
                    r'법인[가-힣]*',
                    r'[가-힣]*법인'
                ],
                "stock_terms": [
                    r'주식[가-힣]*',
                    r'[가-힣]*주식',
                    r'주주[가-힣]*',
                    r'[가-힣]*주주'
                ],
                "commercial_terms": [
                    r'상행위[가-힣]*',
                    r'[가-힣]*상행위',
                    r'상사[가-힣]*',
                    r'[가-힣]*상사'
                ]
            },
            LegalDomain.LABOR_LAW: {
                "employment_terms": [
                    r'근로[가-힣]*',
                    r'[가-힣]*근로',
                    r'고용[가-힣]*',
                    r'[가-힣]*고용'
                ],
                "wage_terms": [
                    r'임금[가-힣]*',
                    r'[가-힣]*임금',
                    r'급여[가-힣]*',
                    r'[가-힣]*급여'
                ],
                "dismissal_terms": [
                    r'해고[가-힣]*',
                    r'[가-힣]*해고',
                    r'퇴직[가-힣]*',
                    r'[가-힣]*퇴직'
                ]
            },
            LegalDomain.REAL_ESTATE_LAW: {
                "property_terms": [
                    r'부동산[가-힣]*',
                    r'[가-힣]*부동산',
                    r'토지[가-힣]*',
                    r'[가-힣]*토지'
                ],
                "registration_terms": [
                    r'등기[가-힣]*',
                    r'[가-힣]*등기',
                    r'등록[가-힣]*',
                    r'[가-힣]*등록'
                ],
                "transaction_terms": [
                    r'매매[가-힣]*',
                    r'[가-힣]*매매',
                    r'임대[가-힣]*',
                    r'[가-힣]*임대'
                ]
            },
            LegalDomain.INTELLECTUAL_PROPERTY_LAW: {
                "patent_terms": [
                    r'특허[가-힣]*',
                    r'[가-힣]*특허',
                    r'발명[가-힣]*',
                    r'[가-힣]*발명'
                ],
                "trademark_terms": [
                    r'상표[가-힣]*',
                    r'[가-힣]*상표',
                    r'브랜드[가-힣]*',
                    r'[가-힣]*브랜드'
                ],
                "copyright_terms": [
                    r'저작권[가-힣]*',
                    r'[가-힣]*저작권',
                    r'저작[가-힣]*',
                    r'[가-힣]*저작'
                ]
            },
            LegalDomain.TAX_LAW: {
                "tax_terms": [
                    r'세금[가-힣]*',
                    r'[가-힣]*세금',
                    r'세[가-힣]*',
                    r'[가-힣]*세'
                ],
                "income_terms": [
                    r'소득[가-힣]*',
                    r'[가-힣]*소득',
                    r'수입[가-힣]*',
                    r'[가-힣]*수입'
                ],
                "corporate_terms": [
                    r'법인[가-힣]*',
                    r'[가-힣]*법인',
                    r'기업[가-힣]*',
                    r'[가-힣]*기업'
                ]
            },
            LegalDomain.CIVIL_PROCEDURE_LAW: {
                "lawsuit_terms": [
                    r'소송[가-힣]*',
                    r'[가-힣]*소송',
                    r'소[가-힣]*',
                    r'[가-힣]*소'
                ],
                "evidence_terms": [
                    r'증거[가-힣]*',
                    r'[가-힣]*증거',
                    r'입증[가-힣]*',
                    r'[가-힣]*입증'
                ],
                "procedure_terms": [
                    r'절차[가-힣]*',
                    r'[가-힣]*절차',
                    r'절차[가-힣]*',
                    r'[가-힣]*절차'
                ]
            },
            LegalDomain.CRIMINAL_PROCEDURE_LAW: {
                "investigation_terms": [
                    r'수사[가-힣]*',
                    r'[가-힣]*수사',
                    r'조사[가-힣]*',
                    r'[가-힣]*조사'
                ],
                "prosecution_terms": [
                    r'기소[가-힣]*',
                    r'[가-힣]*기소',
                    r'공소[가-힣]*',
                    r'[가-힣]*공소'
                ],
                "defense_terms": [
                    r'변호[가-힣]*',
                    r'[가-힣]*변호',
                    r'변호인[가-힣]*',
                    r'[가-힣]*변호인'
                ]
            }
        }
    
    async def _get_domain_keywords(self, domain: LegalDomain) -> List[str]:
        """도메인별 키워드 조회 (하이브리드 시스템 사용)"""
        if not self._cache_initialized:
            await self._initialize_keyword_cache()
        
        domain_name = self.domain_mapping.get(domain, '기타/일반')
        return self._domain_keywords_cache.get(domain_name, [])
    
    async def _initialize_keyword_cache(self):
        """키워드 캐시 초기화"""
        try:
            all_keywords = await self.keyword_manager.get_all_domain_keywords()
            self._domain_keywords_cache = all_keywords
            self._cache_initialized = True
            self.logger.info(f"키워드 캐시 초기화 완료: {len(all_keywords)}개 도메인")
        except Exception as e:
            self.logger.error(f"키워드 캐시 초기화 실패: {e}")
            self._cache_initialized = False
    
    def extract_domain_specific_terms(self, text: str, domain: LegalDomain) -> List[str]:
        """특정 도메인의 용어 추출"""
        if domain not in self.domain_patterns:
            return []
        
        extracted_terms = []
        patterns = self.domain_patterns[domain]
        
        for pattern_group, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text)
                extracted_terms.extend(matches)
        
        return list(set(extracted_terms))
    
    async def classify_domain_by_keywords(self, text: str) -> Dict[LegalDomain, float]:
        """키워드 기반 도메인 분류 (하이브리드 시스템 사용)"""
        domain_scores = defaultdict(float)
        text_lower = text.lower()
        
        # 모든 도메인의 키워드 조회
        for domain in LegalDomain:
            try:
                keywords = await self._get_domain_keywords(domain)
                score = 0.0
                matched_keywords = 0
                
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        # 키워드 길이에 따른 가중치 (긴 키워드일수록 높은 가중치)
                        length_weight = min(2.0, len(keyword) / 3.0)
                        
                        # 키워드 빈도 계산
                        frequency = text_lower.count(keyword.lower())
                        
                        # 최종 점수 = 기본점수 * 길이가중치 * 빈도가중치
                        keyword_score = 1.0 * length_weight * min(3.0, frequency)
                        score += keyword_score
                        matched_keywords += 1
                
                # 도메인별 특화 가중치 적용
                domain_weights = {
                    LegalDomain.CIVIL_LAW: 1.2,      # 민사법 가중치 증가
                    LegalDomain.CRIMINAL_LAW: 1.1,    # 형사법 가중치 증가
                    LegalDomain.FAMILY_LAW: 1.0,      # 가족법 기본 가중치
                    LegalDomain.COMMERCIAL_LAW: 1.0,  # 상사법 기본 가중치
                    LegalDomain.LABOR_LAW: 1.0,       # 노동법 기본 가중치
                    LegalDomain.REAL_ESTATE_LAW: 1.1, # 부동산법 가중치 증가
                    LegalDomain.INTELLECTUAL_PROPERTY_LAW: 1.0, # 지적재산권법 기본 가중치
                    LegalDomain.TAX_LAW: 1.1,        # 세법 가중치 증가
                    LegalDomain.CIVIL_PROCEDURE_LAW: 1.2, # 민사소송법 가중치 증가
                    LegalDomain.CRIMINAL_PROCEDURE_LAW: 1.3, # 형사소송법 가중치 대폭 증가
                    LegalDomain.GENERAL: 0.5          # 일반 도메인 가중치 감소
                }
                
                # 도메인별 가중치 적용
                domain_weight = domain_weights.get(domain, 1.0)
                score *= domain_weight
                
                # 매칭된 키워드 수에 따른 보너스
                if matched_keywords > 0:
                    bonus = min(2.0, matched_keywords * 0.1)
                    score += bonus
                
                domain_scores[domain] = score
                
            except Exception as e:
                self.logger.error(f"도메인 분류 중 오류 ({domain}): {e}")
                domain_scores[domain] = 0.0
        
        # 정규화
        total_score = sum(domain_scores.values())
        if total_score > 0:
            for domain in domain_scores:
                domain_scores[domain] = domain_scores[domain] / total_score
        
        return dict(domain_scores)
    
    def analyze_context_patterns(self, text: str) -> Dict[LegalDomain, float]:
        """문맥 패턴 분석을 통한 도메인 분류"""
        context_scores = defaultdict(float)
        
        # 문맥 패턴 정의
        context_patterns = {
            LegalDomain.CIVIL_LAW: [
                r'계약.*위반', r'손해.*배상', r'소유권.*이전', r'채권.*채무',
                r'불법행위.*과실', r'등기.*절차', r'시효.*소멸'
            ],
            LegalDomain.CRIMINAL_LAW: [
                r'범죄.*구성요건', r'형.*선고', r'수사.*기관', r'기소.*공소',
                r'자백.*배제', r'증거.*능력', r'처벌.*규정'
            ],
            LegalDomain.FAMILY_LAW: [
                r'혼인.*신고', r'이혼.*절차', r'양육.*권', r'친권.*행사',
                r'상속.*분', r'유언.*집행', r'가족.*관계'
            ],
            LegalDomain.COMMERCIAL_LAW: [
                r'주식.*회사', r'주주.*총회', r'이사.*선임', r'자본.*금',
                r'회사.*해산', r'청산.*절차', r'상행위.*법'
            ],
            LegalDomain.LABOR_LAW: [
                r'근로.*계약', r'임금.*지급', r'해고.*절차', r'근로.*시간',
                r'부당해고.*구제', r'노동.*조합', r'산업.*재해'
            ],
            LegalDomain.REAL_ESTATE_LAW: [
                r'부동산.*등기', r'소유권.*이전', r'매매.*계약', r'임대.*차',
                r'등기.*부', r'공시.*제도', r'보증금.*반환'
            ],
            LegalDomain.INTELLECTUAL_PROPERTY_LAW: [
                r'특허.*출원', r'상표.*등록', r'저작권.*침해', r'디자인.*권',
                r'영업비밀.*보호', r'침해.*소송', r'정지.*청구'
            ],
            LegalDomain.TAX_LAW: [
                r'소득세.*신고', r'법인세.*납부', r'부가가치세.*계산',
                r'세무.*조사', r'조세.*불복', r'가산세.*부과'
            ],
            LegalDomain.CIVIL_PROCEDURE_LAW: [
                r'소송.*제기', r'증거.*조사', r'변론.*절차', r'판결.*선고',
                r'항소.*신청', r'강제.*집행', r'소장.*제출'
            ],
            LegalDomain.CRIMINAL_PROCEDURE_LAW: [
                r'수사.*절차', r'기소.*결정', r'공판.*절차', r'변호인.*선임',
                r'증거.*능력', r'자백.*배제', r'재심.*신청'
            ]
        }
        
        for domain, patterns in context_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                score += len(matches) * 2.0  # 문맥 패턴 매칭 시 높은 점수
            
            context_scores[domain] = score
        
        # 정규화
        total_score = sum(context_scores.values())
        if total_score > 0:
            for domain in context_scores:
                context_scores[domain] = context_scores[domain] / total_score
        
        return dict(context_scores)
    
    async def get_enhanced_domain_classification(self, text: str) -> Dict[LegalDomain, float]:
        """키워드 + 문맥 패턴 통합 분류 (하이브리드 시스템 사용)"""
        keyword_scores = await self.classify_domain_by_keywords(text)
        context_scores = self.analyze_context_patterns(text)
        
        # 통합 점수 계산 (키워드 70% + 문맥 30%)
        integrated_scores = defaultdict(float)
        
        for domain in LegalDomain:
            keyword_score = keyword_scores.get(domain, 0.0)
            context_score = context_scores.get(domain, 0.0)
            
            integrated_score = keyword_score * 0.7 + context_score * 0.3
            integrated_scores[domain] = integrated_score
        
        return dict(integrated_scores)
    
    def extract_all_domain_terms(self, text: str) -> Dict[LegalDomain, List[str]]:
        """모든 도메인의 용어 추출"""
        all_domain_terms = {}
        
        for domain in LegalDomain:
            if domain == LegalDomain.GENERAL:
                continue
            
            terms = self.extract_domain_specific_terms(text, domain)
            if terms:
                all_domain_terms[domain] = terms
        
        return all_domain_terms
    
    async def get_domain_confidence(self, text: str, domain: LegalDomain) -> float:
        """특정 도메인에 대한 신뢰도 계산 (하이브리드 시스템 사용)"""
        domain_scores = await self.classify_domain_by_keywords(text)
        return domain_scores.get(domain, 0.0)
    
    async def get_primary_domain(self, text: str) -> Tuple[LegalDomain, float]:
        """주요 도메인 식별 (하이브리드 시스템 사용)"""
        domain_scores = await self.get_enhanced_domain_classification(text)
        
        if not domain_scores:
            return LegalDomain.GENERAL, 0.0
        
        primary_domain = max(domain_scores, key=domain_scores.get)
        confidence = domain_scores[primary_domain]
        
        return primary_domain, confidence
    
    async def enhance_term_extraction(self, text: str, base_terms: List[str]) -> Dict[str, Any]:
        """기본 용어 추출 결과를 도메인별로 강화 (하이브리드 시스템 사용)"""
        # 도메인 분류
        primary_domain, domain_confidence = await self.get_primary_domain(text)
        
        # 도메인별 용어 추출
        domain_terms = self.extract_all_domain_terms(text)
        
        # 기본 용어와 도메인별 용어 통합
        enhanced_terms = set(base_terms)
        for domain, terms in domain_terms.items():
            enhanced_terms.update(terms)
        
        # 도메인별 가중치 적용
        weighted_terms = {}
        primary_domain_keywords = await self._get_domain_keywords(primary_domain)
        
        for term in enhanced_terms:
            weight = 0.5  # 기본 가중치
            
            # 주 도메인에 속하는 용어는 가중치 증가
            if primary_domain in domain_terms and term in domain_terms[primary_domain]:
                weight += 0.3
            
            # 키워드에 포함된 용어는 가중치 증가
            if term in primary_domain_keywords:
                weight += 0.2
            
            weighted_terms[term] = min(weight, 1.0)
        
        return {
            "enhanced_terms": list(enhanced_terms),
            "primary_domain": primary_domain,
            "domain_confidence": domain_confidence,
            "domain_terms": domain_terms,
            "weighted_terms": weighted_terms
        }
    
    def get_domain_specific_weights(self, term: str, domain: LegalDomain) -> float:
        """도메인별 용어 가중치 계산"""
        try:
            # 기본 가중치
            base_weight = 0.5
            
            # 도메인별 특화 가중치
            domain_weights = {
                LegalDomain.CIVIL_LAW: 0.8,
                LegalDomain.CRIMINAL_LAW: 0.9,
                LegalDomain.FAMILY_LAW: 0.7,
                LegalDomain.COMMERCIAL_LAW: 0.8,
                LegalDomain.LABOR_LAW: 0.7,
                LegalDomain.REAL_ESTATE_LAW: 0.8,
                LegalDomain.INTELLECTUAL_PROPERTY_LAW: 0.9,
                LegalDomain.TAX_LAW: 0.8,
                LegalDomain.CIVIL_PROCEDURE_LAW: 0.9,
                LegalDomain.CRIMINAL_PROCEDURE_LAW: 0.9,
                LegalDomain.GENERAL: 0.5
            }
            
            # 도메인별 가중치 적용
            domain_weight = domain_weights.get(domain, 0.5)
            
            # 용어 길이에 따른 가중치 조정
            length_weight = min(1.0, len(term) / 10.0)
            
            # 최종 가중치 계산
            final_weight = base_weight * domain_weight * length_weight
            
            return min(1.0, final_weight)
            
        except Exception as e:
            self.logger.error(f"도메인별 가중치 계산 중 오류 발생: {e}")
            return 0.5
    
    # 하이브리드 시스템 관련 새로운 메서드들
    async def refresh_domain_keywords(self, domain: LegalDomain, force_refresh: bool = True) -> List[str]:
        """특정 도메인의 키워드 새로고침"""
        domain_name = self.domain_mapping.get(domain, '기타/일반')
        keywords, metadata = await self.keyword_manager.get_domain_keywords(domain_name, force_refresh)
        
        # 캐시 업데이트
        if self._cache_initialized:
            self._domain_keywords_cache[domain_name] = keywords
        
        self.logger.info(f"도메인 키워드 새로고침 완료: {domain_name} ({len(keywords)}개, 소스: {metadata.get('source', 'unknown')})")
        return keywords
    
    async def get_domain_statistics(self) -> Dict[str, Any]:
        """도메인별 키워드 통계 조회"""
        return await self.keyword_manager.get_domain_statistics()
    
    async def invalidate_domain_cache(self, domain: LegalDomain) -> bool:
        """특정 도메인의 캐시 무효화"""
        domain_name = self.domain_mapping.get(domain, '기타/일반')
        result = self.keyword_manager.invalidate_domain_cache(domain_name)
        
        # 메모리 캐시에서도 제거
        if domain_name in self._domain_keywords_cache:
            del self._domain_keywords_cache[domain_name]
        
        return result
    
    async def clear_all_cache(self) -> bool:
        """모든 캐시 삭제"""
        result = self.keyword_manager.clear_all_cache()
        self._domain_keywords_cache.clear()
        self._cache_initialized = False
        return result
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        return self.keyword_manager.get_cache_statistics()
    
    def set_expansion_strategy(self, strategy: ExpansionStrategy):
        """확장 전략 설정"""
        self.keyword_manager.set_expansion_strategy(strategy)
    
    def set_minimum_threshold(self, threshold: int):
        """최소 키워드 임계값 설정"""
        self.keyword_manager.set_minimum_threshold(threshold)
    
    async def expand_domain_keywords_if_needed(self, domain: LegalDomain) -> bool:
        """도메인 키워드 확장이 필요한지 확인하고 실행"""
        domain_name = self.domain_mapping.get(domain, '기타/일반')
        
        # 현재 키워드 수 확인
        current_keywords = await self._get_domain_keywords(domain)
        current_count = len(current_keywords)
        
        if current_count < self.keyword_manager.min_keyword_threshold:
            self.logger.info(f"도메인 키워드 확장 필요: {domain_name} ({current_count}/{self.keyword_manager.min_keyword_threshold})")
            
            # 키워드 확장 실행
            expanded_keywords = await self.refresh_domain_keywords(domain, force_refresh=True)
            expanded_count = len(expanded_keywords)
            
            self.logger.info(f"도메인 키워드 확장 완료: {domain_name} ({current_count} → {expanded_count})")
            return True
        
        return False
