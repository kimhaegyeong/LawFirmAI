import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from enum import Enum

from .keyword_database_loader import KeywordDatabaseLoader
from .ai_keyword_generator import AIKeywordGenerator, KeywordExpansionResult
from .keyword_cache import KeywordCache

logger = logging.getLogger(__name__)

class ExpansionStrategy(Enum):
    """키워드 확장 전략"""
    DATABASE_ONLY = "database_only"
    AI_ONLY = "ai_only"
    HYBRID = "hybrid"
    FALLBACK = "fallback"

class HybridKeywordManager:
    """하이브리드 키워드 관리 시스템"""
    
    def __init__(self, 
                 data_dir: str = "data",
                 cache_dir: str = "data/cache",
                 min_keyword_threshold: int = 20,
                 expansion_strategy: ExpansionStrategy = ExpansionStrategy.HYBRID):
        
        self.logger = logging.getLogger(__name__)
        self.min_keyword_threshold = min_keyword_threshold
        self.expansion_strategy = expansion_strategy
        
        # 컴포넌트 초기화
        self.database_loader = KeywordDatabaseLoader(data_dir)
        self.ai_generator = AIKeywordGenerator()
        self.cache = KeywordCache(cache_dir)
        
        # 도메인 매핑
        self.domain_mapping = {
            '민사법': 'CIVIL_LAW',
            '형사법': 'CRIMINAL_LAW',
            '가족법': 'FAMILY_LAW',
            '상사법': 'COMMERCIAL_LAW',
            '노동법': 'LABOR_LAW',
            '부동산법': 'REAL_ESTATE_LAW',
            '지적재산권법': 'INTELLECTUAL_PROPERTY_LAW',
            '세법': 'TAX_LAW',
            '민사소송법': 'CIVIL_PROCEDURE_LAW',
            '형사소송법': 'CRIMINAL_PROCEDURE_LAW',
            '기타/일반': 'GENERAL'
        }
        
        self.reverse_domain_mapping = {v: k for k, v in self.domain_mapping.items()}
    
    async def get_domain_keywords(self, domain: str, force_refresh: bool = False) -> Tuple[List[str], Dict[str, Any]]:
        """도메인별 키워드 조회 (하이브리드 방식)"""
        metadata = {
            'source': 'unknown',
            'expansion_applied': False,
            'cache_used': False,
            'ai_expansion_used': False,
            'database_used': False,
            'total_keywords': 0,
            'confidence': 0.0
        }
        
        try:
            # 1. 캐시에서 먼저 확인 (강제 새로고침이 아닌 경우)
            if not force_refresh:
                cached_keywords = self.cache.get(domain)
                if cached_keywords:
                    metadata.update({
                        'source': 'cache',
                        'cache_used': True,
                        'total_keywords': len(cached_keywords),
                        'confidence': 0.9
                    })
                    self.logger.info(f"캐시에서 키워드 로드: {domain} ({len(cached_keywords)}개)")
                    return cached_keywords, metadata
            
            # 2. 데이터베이스에서 키워드 로드
            db_keywords = self.database_loader.load_all_keywords()
            domain_keywords = db_keywords.get(domain, [])
            metadata['database_used'] = True
            
            self.logger.info(f"데이터베이스에서 키워드 로드: {domain} ({len(domain_keywords)}개)")
            
            # 3. 키워드 수 확인 및 확장 필요성 판단
            if len(domain_keywords) < self.min_keyword_threshold:
                self.logger.info(f"키워드 부족으로 확장 필요: {domain} ({len(domain_keywords)}/{self.min_keyword_threshold})")
                
                # 4. 확장 전략에 따른 키워드 확장
                expanded_keywords = await self._expand_keywords(domain, domain_keywords)
                
                if expanded_keywords:
                    domain_keywords = expanded_keywords
                    metadata.update({
                        'expansion_applied': True,
                        'ai_expansion_used': True,
                        'total_keywords': len(domain_keywords),
                        'confidence': 0.8
                    })
                else:
                    # AI 확장 실패 시 폴백 확장
                    domain_keywords = self.ai_generator.expand_keywords_with_fallback(domain, domain_keywords)
                    metadata.update({
                        'expansion_applied': True,
                        'ai_expansion_used': False,
                        'total_keywords': len(domain_keywords),
                        'confidence': 0.6
                    })
            else:
                metadata.update({
                    'total_keywords': len(domain_keywords),
                    'confidence': 0.9
                })
            
            # 5. 중복 제거 및 정렬
            domain_keywords = self._deduplicate_and_sort(domain_keywords)
            
            # 6. 캐시에 저장
            cache_metadata = {
                'expansion_strategy': self.expansion_strategy.value,
                'min_threshold': self.min_keyword_threshold,
                'expansion_applied': metadata['expansion_applied']
            }
            self.cache.set(domain, domain_keywords, cache_metadata)
            
            metadata['source'] = 'hybrid'
            metadata['total_keywords'] = len(domain_keywords)
            
            self.logger.info(f"하이브리드 키워드 로드 완료: {domain} ({len(domain_keywords)}개)")
            return domain_keywords, metadata
            
        except Exception as e:
            self.logger.error(f"키워드 로드 실패 ({domain}): {e}")
            # 폴백: 기본 키워드 반환
            fallback_keywords = self._get_fallback_keywords(domain)
            metadata.update({
                'source': 'fallback',
                'total_keywords': len(fallback_keywords),
                'confidence': 0.3
            })
            return fallback_keywords, metadata
    
    async def _expand_keywords(self, domain: str, base_keywords: List[str]) -> List[str]:
        """키워드 확장 실행"""
        if self.expansion_strategy == ExpansionStrategy.DATABASE_ONLY:
            return base_keywords
        
        if self.expansion_strategy == ExpansionStrategy.AI_ONLY:
            return await self._ai_expand_keywords(domain, base_keywords)
        
        if self.expansion_strategy == ExpansionStrategy.HYBRID:
            # AI 확장 시도
            expanded_keywords = await self._ai_expand_keywords(domain, base_keywords)
            if expanded_keywords and len(expanded_keywords) > len(base_keywords):
                return expanded_keywords
            else:
                # AI 확장 실패 시 폴백
                return self.ai_generator.expand_keywords_with_fallback(domain, base_keywords)
        
        return base_keywords
    
    async def _ai_expand_keywords(self, domain: str, base_keywords: List[str]) -> List[str]:
        """AI를 사용한 키워드 확장"""
        try:
            target_count = max(self.min_keyword_threshold, len(base_keywords) * 2)
            result = await self.ai_generator.expand_domain_keywords(
                domain, base_keywords, target_count
            )
            
            if result.api_call_success and result.expanded_keywords:
                self.logger.info(f"AI 키워드 확장 성공: {domain} ({len(result.expanded_keywords)}개)")
                return result.expanded_keywords
            else:
                self.logger.warning(f"AI 키워드 확장 실패: {domain}")
                return []
                
        except Exception as e:
            self.logger.error(f"AI 키워드 확장 중 오류 ({domain}): {e}")
            return []
    
    def _deduplicate_and_sort(self, keywords: List[str]) -> List[str]:
        """키워드 중복 제거 및 정렬"""
        # 중복 제거
        unique_keywords = list(set(keywords))
        
        # 길이순 정렬 (짧은 것부터)
        unique_keywords.sort(key=len)
        
        # 한글 키워드 우선 정렬
        korean_keywords = [kw for kw in unique_keywords if any('\uac00' <= char <= '\ud7af' for char in kw)]
        other_keywords = [kw for kw in unique_keywords if kw not in korean_keywords]
        
        return korean_keywords + other_keywords
    
    def _get_fallback_keywords(self, domain: str) -> List[str]:
        """폴백 키워드 반환"""
        fallback_keywords = {
            '민사법': ['계약', '손해배상', '소유권', '채권', '채무', '불법행위'],
            '형사법': ['살인', '절도', '사기', '강도', '범죄', '형', '처벌'],
            '가족법': ['혼인', '이혼', '양육권', '친권', '상속', '유언'],
            '상사법': ['회사', '주식', '주주', '이사', '상행위', '법인'],
            '노동법': ['근로', '고용', '임금', '해고', '부당해고', '근로계약'],
            '부동산법': ['부동산', '토지', '등기', '매매', '임대', '소유권이전'],
            '지적재산권법': [
                '특허', '특허권', '특허출원', '특허등록', '특허침해', '특허무효',
                '상표', '상표권', '상표등록', '상표출원', '상표침해', '상표무효',
                '저작권', '저작물', '저작자', '저작권침해', '저작권등록', '저작권보호',
                '디자인', '디자인권', '디자인등록', '디자인침해', '디자인무효',
                '영업비밀', '노하우', '기술정보', '영업정보', '비밀정보',
                '데이터베이스', 'DB', '데이터베이스권', '데이터베이스보호',
                '반도체배치설계', '반도체설계', '배치설계', '반도체배치설계권',
                '신지식재산권', '디지털콘텐츠', '소프트웨어', '알고리즘', 'AI발명',
                '발명', '발명자', '발명등록', '발명특허', '기술특허', '실용신안',
                '브랜드', '로고', '서비스표', '단체표장', '지리적표시',
                '저작', '창작', '원작', '번역', '편집', '각색', '공연',
                '외관디자인', '제품디자인', '패키지디자인',
                '특허청', '특허심사관', '특허심판관', '특허변호사', '특허대리인',
                '상표심사관', '상표심판관', '상표변호사', '상표대리인',
                '저작권심의조정위원회', '저작권보호원', '저작권변호사',
                '디자인변호사', '영업비밀변호사'
            ],
            '세법': ['세금', '소득세', '법인세', '부가가치세', '신고', '납부'],
            '민사소송법': ['소송', '소', '증거', '입증', '절차', '판결'],
            '형사소송법': [
                '수사', '수사기관', '수사절차', '수사개시', '수사종료', '수사중단',
                '기소', '기소권', '기소유예', '기소중지', '기소장', '기소결정',
                '공소', '공소제기', '공소유지', '공소취소', '공소장', '공소사실',
                '변호', '변호인', '변호권', '변호인선임', '변호인접견', '변호인지위',
                '증거', '증거능력', '증거력', '증거조사', '증거제출', '증거보전',
                '재판', '재판절차', '재판권', '재판관', '재판정', '재판기록',
                '공판', '공판절차', '공판정', '공판기일', '공판준비', '공판조서',
                '경찰', '검찰', '수사관', '수사팀', '수사보고서', '수사기록',
                '국선변호인', '선임변호인', '변호인보조인',
                '증거개시', '증거동의', '증거배제', '증거법칙', '자백배제법칙',
                '재판개시', '재판진행', '재판종료', '재판결과', '재판효력',
                '공판개시', '공판진행', '공판종료', '공판결과', '공판효력'
            ],
            '기타/일반': ['법률', '법령', '판례', '법원', '검찰', '변호사']
        }
        
        return fallback_keywords.get(domain, ['법률', '법령', '판례'])
    
    async def get_all_domain_keywords(self, force_refresh: bool = False) -> Dict[str, List[str]]:
        """모든 도메인의 키워드 조회"""
        all_keywords = {}
        
        for domain in self.domain_mapping.keys():
            try:
                keywords, _ = await self.get_domain_keywords(domain, force_refresh)
                all_keywords[domain] = keywords
            except Exception as e:
                self.logger.error(f"도메인 키워드 로드 실패 ({domain}): {e}")
                all_keywords[domain] = self._get_fallback_keywords(domain)
        
        return all_keywords
    
    async def get_domain_statistics(self) -> Dict[str, Any]:
        """도메인별 키워드 통계"""
        stats = {}
        
        for domain in self.domain_mapping.keys():
            try:
                keywords, metadata = await self.get_domain_keywords(domain)
                stats[domain] = {
                    'keyword_count': len(keywords),
                    'source': metadata.get('source', 'unknown'),
                    'expansion_applied': metadata.get('expansion_applied', False),
                    'confidence': metadata.get('confidence', 0.0),
                    'cache_info': self.cache.get_domain_cache_info(domain)
                }
            except Exception as e:
                self.logger.error(f"도메인 통계 조회 실패 ({domain}): {e}")
                stats[domain] = {
                    'keyword_count': 0,
                    'source': 'error',
                    'expansion_applied': False,
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        return stats
    
    def invalidate_domain_cache(self, domain: str) -> bool:
        """특정 도메인의 캐시 무효화"""
        return self.cache.invalidate(domain)
    
    def clear_all_cache(self) -> bool:
        """모든 캐시 삭제"""
        return self.cache.clear_all()
    
    def cleanup_expired_cache(self) -> int:
        """만료된 캐시 정리"""
        return self.cache.cleanup_expired()
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        return self.cache.get_cache_stats()
    
    def set_expansion_strategy(self, strategy: ExpansionStrategy):
        """확장 전략 설정"""
        self.expansion_strategy = strategy
        self.logger.info(f"확장 전략 변경: {strategy.value}")
    
    def set_minimum_threshold(self, threshold: int):
        """최소 키워드 임계값 설정"""
        self.min_keyword_threshold = threshold
        self.logger.info(f"최소 키워드 임계값 변경: {threshold}")
