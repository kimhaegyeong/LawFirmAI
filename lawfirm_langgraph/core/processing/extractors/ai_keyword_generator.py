import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)

@dataclass
class KeywordExpansionResult:
    """키워드 확장 결과"""
    domain: str
    base_keywords: List[str]
    expanded_keywords: List[str]
    confidence: float
    expansion_method: str
    api_call_success: bool

class AIKeywordGenerator:
    """AI 모델을 사용한 키워드 확장 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gemini_client = None
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Gemini API 클라이언트 초기화"""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage
            
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key:
                self.gemini_client = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-exp",
                    temperature=0.3,
                    max_output_tokens=1000,
                    timeout=30,
                    api_key=api_key
                )
                self.logger.info("Gemini API 클라이언트 초기화 완료")
            else:
                self.logger.warning("GOOGLE_API_KEY가 설정되지 않음. AI 키워드 확장 비활성화")
        except ImportError as e:
            self.logger.warning(f"Gemini API 라이브러리 없음: {e}")
        except Exception as e:
            self.logger.error(f"Gemini API 초기화 실패: {e}")
    
    async def expand_domain_keywords(self, domain: str, base_keywords: List[str], 
                                   target_count: int = 50, query: Optional[str] = None,
                                   query_type: Optional[str] = None) -> KeywordExpansionResult:
        """도메인별 키워드 확장 (개선: 쿼리 컨텍스트 및 질문 유형 추가)"""
        # 키워드 검증 및 정제
        validated_base_keywords = self._validate_keywords(base_keywords)
        
        if not validated_base_keywords:
            self.logger.warning(f"No valid keywords found after validation, using original keywords")
            validated_base_keywords = base_keywords[:5]  # 최대 5개만 사용
        
        if not self.gemini_client:
            return KeywordExpansionResult(
                domain=domain,
                base_keywords=validated_base_keywords,
                expanded_keywords=validated_base_keywords,
                confidence=0.0,
                expansion_method="fallback",
                api_call_success=False
            )
        
        try:
            prompt = self._create_enhanced_expansion_prompt(
                domain, validated_base_keywords, target_count, query=query, query_type=query_type
            )
            response = await self._call_gemini_api(prompt)
            expanded_keywords = self._parse_gemini_response(response)
            
            # 확장된 키워드도 검증
            validated_expanded_keywords = self._validate_keywords(expanded_keywords)
            
            return KeywordExpansionResult(
                domain=domain,
                base_keywords=base_keywords,
                expanded_keywords=validated_expanded_keywords,
                confidence=0.8,
                expansion_method="gemini_ai_enhanced",
                api_call_success=True
            )
            
        except Exception as e:
            self.logger.error(f"키워드 확장 실패 ({domain}): {e}")
            return KeywordExpansionResult(
                domain=domain,
                base_keywords=validated_base_keywords,
                expanded_keywords=validated_base_keywords,
                confidence=0.0,
                expansion_method="error",
                api_call_success=False
            )
    
    def _validate_keywords(self, keywords: List[str]) -> List[str]:
        """키워드 검증 및 정제 (깨진 문자, 잘못된 키워드 제거)"""
        valid_keywords = []
        
        for keyword in keywords:
            if not keyword or not isinstance(keyword, str):
                continue
                
            keyword = keyword.strip()
            
            # 최소 길이 검증
            if len(keyword) < 2:
                continue
            
            # 깨진 문자 패턴 감지 (한글 범위 밖의 문자가 많으면 제거)
            garbled_chars = sum(1 for c in keyword if ord(c) > 0xFF and (ord(c) < 0xAC00 or ord(c) > 0xD7A3))
            garbled_ratio = garbled_chars / max(len(keyword), 1)
            
            if garbled_ratio > 0.3:  # 30% 이상 깨진 문자면 제거
                self.logger.warning(f"Skipping garbled keyword: {keyword[:50]}...")
                continue
            
            # '?' 문자가 많으면 제거 (깨진 문자 표시)
            if keyword.count('?') > len(keyword) * 0.2:
                self.logger.warning(f"Skipping keyword with too many '?' characters: {keyword[:50]}...")
                continue
            
            valid_keywords.append(keyword)
        
        return valid_keywords
    
    def _create_expansion_prompt(self, domain: str, base_keywords: List[str], target_count: int) -> str:
        """키워드 확장을 위한 프롬프트 생성 (기존 메서드 유지 - 하위 호환성)"""
        return self._create_enhanced_expansion_prompt(domain, base_keywords, target_count, query=None, query_type=None)
    
    def _create_enhanced_expansion_prompt(
        self, 
        domain: str, 
        base_keywords: List[str], 
        target_count: int,
        query: Optional[str] = None,
        query_type: Optional[str] = None
    ) -> str:
        """개선된 키워드 확장 프롬프트 생성 (쿼리 컨텍스트 및 질문 유형 포함)"""
        # 키워드 검증
        validated_keywords = self._validate_keywords(base_keywords)
        
        if not validated_keywords:
            self.logger.warning(f"All keywords failed validation, using original keywords")
            validated_keywords = base_keywords[:5]
        
        # 쿼리 컨텍스트 추가
        query_context = f"\n**원본 질문**: {query}\n" if query else ""
        
        # 질문 유형별 특화 지시사항
        type_specific_guidance = ""
        if query_type:
            type_guidance_map = {
                "law_inquiry": """
**법령 조회 특화**:
- 관련 법령명과 조문 번호를 우선적으로 포함하세요
- 예: "민법 제750조", "형법 제250조"
- 법령의 핵심 개념과 용어를 확장하세요
""",
                "precedent_search": """
**판례 검색 특화**:
- 판례 관련 키워드를 우선적으로 확장하세요
- 예: "대법원 판례", "판결 요지", "사건번호"
- 유사 판례 검색을 위한 법리 용어를 포함하세요
""",
                "contract_review": """
**계약 검토 특화**:
- 계약 관련 실무 용어를 확장하세요
- 예: "계약서 검토", "계약 조건", "계약 위반"
- 관련 법령: "민법 제105조", "계약법"
"""
            }
            type_specific_guidance = type_guidance_map.get(query_type, "")
        
        return f"""
당신은 한국 법률 분야의 전문가입니다. 주어진 {domain} 분야의 기본 키워드들을 바탕으로 검색에 유용한 관련 키워드를 확장해주세요.
{query_context}
**기본 키워드**: {', '.join(validated_keywords[:10])}

**Phase 5: 고급 키워드 매칭 기법 적용**

**1. 다층 키워드 매칭 시스템** (가중치 우선순위):

**1.1 계층적 키워드 매칭**:
- Level 1: 직접 문자열 매칭 (100% 가중치)
  - 정확한 키워드 문자열 그대로 매칭
  - 예: "손해배상" → "손해배상" (정확 일치)
  
- Level 2: 형태소 분석 기반 매칭 (80% 가중치)
  - 어간/어미 분리하여 매칭
  - 예: "손해배상" → "손해", "배상", "손해배상하다", "손해배상한"
  
- Level 3: 의미 기반 매칭 (70% 가중치)
  - 동의어/유의어 매칭
  - 예: "손해배상" → "손해 배상", "손해보상", "배상책임", "손해전보", "손해보상청구"
  
- Level 4: 확장 키워드 매칭 (60% 가중치)
  - 관련 용어 및 상위/하위 개념
  - 예: "손해배상" → "불법행위", "과실", "인과관계", "손해액"

**1.2 키워드 그룹 매칭**:
- 관련 키워드를 그룹으로 묶어 매칭
- 그룹 내 일부 키워드 매칭 시 부분 점수 부여
- 예: "계약" 그룹 → ["계약서", "계약관계", "계약당사자", "계약조건", "계약체결", "계약해지"]
- 예: "손해배상" 그룹 → ["손해배상청구", "손해배상책임", "손해배상소송", "손해배상액"]

**1.3 동적 키워드 가중치** (문서 유형별):
- 법령 조문: 법령명/조문번호 가중치 증가 (예: "민법 제750조" → 가중치 1.0)
- 판례: 사건명/법원 가중치 증가 (예: "대법원 2020다12345" → 가중치 1.0)
- 계약서: 계약 관련 용어 가중치 증가 (예: "계약서", "계약조건" → 가중치 0.9)

**2. 컨텍스트 기반 키워드 매칭**:

**2.1 문서 컨텍스트 분석**:
- 문서의 주제와 키워드의 관련성 분석
- 컨텍스트 일치 시 가중치 증가
- 예: "임대차 계약서" 컨텍스트에서 "임대인", "임차인", "보증금" 키워드 가중치 증가

**2.2 쿼리-문서 의미적 유사도 통합**:
- 쿼리 전체와 문서의 의미적 유사도 계산
- 키워드 매칭과 의미적 유사도 결합
- 예: "임대차 계약서 작성" 쿼리 → "임대차", "계약서", "작성" 키워드 + 의미적 유사도

**2.3 문서 구조 기반 매칭**:
- 법령 조문: 조문 번호 우선 매칭 (예: "제750조" 우선)
- 판례: 사건명/법원 우선 매칭 (예: "대법원", "사건번호" 우선)
- 계약서: 계약 당사자/조건 우선 매칭

**3. 고급 키워드 확장 기법**:

**3.1 도메인 특화 키워드 확장**:
- 법률 분야별 전문 사전 활용
- 판례 키워드 데이터베이스 구축
- 예: "민사법" → "계약", "불법행위", "소유권", "채권", "채무"
- 예: "형사법" → "살인죄", "상해죄", "절도죄", "사기죄"

**3.2 계층적 키워드 확장**:
- 상위 개념 → 하위 개념 확장
- 예: "계약" → "매매계약", "임대차계약", "고용계약", "도급계약"
- 예: "불법행위" → "고의 불법행위", "과실 불법행위", "공동 불법행위"

**3.3 역방향 키워드 확장**:
- 검색 결과에서 자주 나타나는 키워드 역추적
- 관련 키워드 사전 구축
- 예: "손해배상" 검색 결과에서 자주 나타나는 "인과관계", "과실", "손해액" 역추적

**기본 확장 원칙** (우선순위 순):
1. **동의어 및 유사어 확장** (최우선)
   - 의미가 동일하거나 매우 유사한 다른 표현
   - 예: "손해배상" → "손해 배상", "손해보상", "배상책임", "손해전보", "손해보상청구"
   - 예: "불법행위" → "불법", "위법행위", "불법적 행위", "불법행위책임", "불법행위로 인한 손해"

2. **복합어 분리 및 확장**
   - 복합어를 구성하는 개별 단어들
   - 예: "불법행위" → "불법", "행위", "불법 행위"
   - 예: "손해배상청구권" → "손해배상", "청구권", "손해배상청구", "배상청구권"

3. **직접 관련 법률 용어**
   - 기본 키워드와 직접적으로 관련된 법률 용어
   - 예: "계약" → "계약서", "계약관계", "계약당사자", "계약조건", "계약체결", "계약해지"

4. **관련 법령 및 조문**
   - 관련 법령명과 조문 번호
   - 예: "손해배상" → "민법 제750조", "민법 제751조", "불법행위", "손해배상청구"
   - 예: "계약" → "민법 제105조", "계약법", "계약의 성립", "계약의 효력"

5. **판례 검색 강화 키워드**
   - 판례에서 자주 사용되는 표현
   - 예: "손해배상 판례", "대법원 손해배상 판결", "손해배상 사건", "손해배상 참고판례"

6. **실무 용어 및 구체적 표현**
   - 법률 실무에서 자주 사용되는 용어
   - 예: "계약서 검토", "계약 해지", "계약 위반", "계약 불이행", "계약 해제"

{type_specific_guidance}

**출력 형식** (JSON만):
{{
    "expanded_keywords": [
        "키워드1",
        "키워드2",
        "키워드3"
    ],
    "synonyms": ["동의어1", "동의어2", "동의어3"],
    "related_terms": ["관련용어1", "관련용어2"],
    "legal_references": ["법령1", "법령2"],
    "precedent_keywords": ["판례키워드1", "판례키워드2"],
    "keyword_groups": {{
        "그룹명1": ["키워드1", "키워드2", "키워드3"],
        "그룹명2": ["키워드4", "키워드5"]
    }},
    "hierarchical_keywords": {{
        "상위개념": ["하위개념1", "하위개념2", "하위개념3"]
    }},
    "weighted_keywords": {{
        "키워드1": 1.0,
        "키워드2": 0.8,
        "키워드3": 0.7,
        "키워드4": 0.6
    }},
    "contextual_keywords": {{
        "문서유형": ["해당유형의특화키워드1", "해당유형의특화키워드2"]
    }}
}}

**주의사항**:
- 최대 {target_count}개의 키워드를 생성하되, 품질을 우선시하세요.
- 다층 매칭 시스템을 활용하여 다양한 레벨의 키워드를 생성하세요.
- 키워드 그룹을 활용하여 관련 키워드를 묶어주세요.
- 문서 유형에 따라 동적 가중치를 적용하세요.
- 계층적 확장을 통해 상위/하위 개념을 포함하세요.
- 원본 질문의 맥락을 고려하여 관련성 높은 키워드만 생성하세요.
- 각 키워드는 2글자 이상이어야 합니다.
"""
    
    async def _call_gemini_api(self, prompt: str) -> str:
        """Gemini API 호출"""
        try:
            from langchain_core.messages import HumanMessage
            
            message = HumanMessage(content=prompt)
            response = self.gemini_client.invoke([message])
            return response.content
        except Exception as e:
            self.logger.error(f"Gemini API 호출 실패: {e}")
            raise
    
    def _parse_gemini_response(self, response: str) -> List[str]:
        """Gemini 응답 파싱 (개선: 새로운 JSON 형식 지원)"""
        try:
            # JSON 부분 추출
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                self.logger.warning("JSON 형식 응답을 찾을 수 없음")
                return []
            
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            # 모든 키워드 수집 (새로운 형식 우선, 기존 형식도 지원)
            all_keywords = set()
            
            # 새로운 형식: expanded_keywords, synonyms, related_terms, legal_references, precedent_keywords
            if 'expanded_keywords' in data:
                all_keywords.update(data['expanded_keywords'])
            
            if 'synonyms' in data:
                all_keywords.update(data['synonyms'])
            
            if 'related_terms' in data:
                all_keywords.update(data['related_terms'])
            
            if 'legal_references' in data:
                all_keywords.update(data['legal_references'])
            
            if 'precedent_keywords' in data:
                all_keywords.update(data['precedent_keywords'])
            
            # Phase 5: 고급 키워드 매칭 기법 - 새로운 필드 지원
            if 'keyword_groups' in data:
                for group_name, keywords in data['keyword_groups'].items():
                    if isinstance(keywords, list):
                        all_keywords.update(keywords)
            
            if 'hierarchical_keywords' in data:
                for parent_concept, child_concepts in data['hierarchical_keywords'].items():
                    if isinstance(child_concepts, list):
                        all_keywords.add(parent_concept)
                        all_keywords.update(child_concepts)
            
            if 'weighted_keywords' in data:
                if isinstance(data['weighted_keywords'], dict):
                    all_keywords.update(data['weighted_keywords'].keys())
            
            if 'contextual_keywords' in data:
                for doc_type, keywords in data['contextual_keywords'].items():
                    if isinstance(keywords, list):
                        all_keywords.update(keywords)
            
            # 기존 형식 지원 (하위 호환성)
            if 'categories' in data:
                for category, keywords in data['categories'].items():
                    if isinstance(keywords, list):
                        all_keywords.update(keywords)
            
            return list(all_keywords)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON 파싱 실패: {e}")
            return []
        except Exception as e:
            self.logger.error(f"응답 파싱 실패: {e}")
            return []
    
    def expand_keywords_with_fallback(self, domain: str, base_keywords: List[str]) -> List[str]:
        """폴백 메서드를 사용한 키워드 확장"""
        # AI 확장이 실패한 경우 기본 확장 규칙 적용
        expanded = set(base_keywords)
        
        # 도메인별 기본 확장 규칙
        expansion_rules = {
            '민사법': [
                '계약서', '계약관계', '계약당사자', '계약조건', '계약위반', '계약해지',
                '손해배상청구', '손해배상책임', '손해배상소송', '손해배상액',
                '소유권이전', '소유권보호', '소유권확인', '소유권침해',
                '채권자', '채권증서', '채권양도', '채권양수', '채권소멸',
                '채무자', '채무증서', '채무이행', '채무불이행', '채무소멸'
            ],
            '형사법': [
                '살인죄', '상해죄', '폭행죄', '협박죄', '강도죄', '절도죄', '사기죄',
                '횡령죄', '배임죄', '강간죄', '강제추행죄', '명예훼손죄', '모독죄',
                '공갈죄', '강요죄', '주거침입죄', '방화죄', '교통사고처리특별법위반죄',
                '정범', '공범', '방조범', '교사범', '미수범', '기수범', '예비범', '음모범'
            ],
            '가족법': [
                '혼인신고', '혼인관계', '혼인신고서', '혼인관계증명서',
                '이혼신고', '이혼관계', '이혼신고서', '이혼관계증명서',
                '양육비', '양육권', '양육관계', '양육비증액', '양육비감액',
                '면접교섭', '면접교섭권', '면접교섭관계', '면접교섭비',
                '친권자', '친권행사', '친권제한', '친권상실', '친권회복'
            ]
        }
        
        # 도메인별 확장 규칙 적용
        if domain in expansion_rules:
            expanded.update(expansion_rules[domain])
        
        return list(expanded)
    
    def get_expansion_confidence(self, domain: str, expanded_keywords: List[str]) -> float:
        """키워드 확장 신뢰도 계산"""
        base_score = 0.5
        
        # 키워드 수에 따른 보너스
        count_bonus = min(0.3, len(expanded_keywords) / 100.0)
        
        # 도메인별 가중치
        domain_weights = {
            '민사법': 1.0,
            '형사법': 1.0,
            '가족법': 0.9,
            '상사법': 0.9,
            '노동법': 0.9,
            '부동산법': 0.9,
            '지적재산권법': 0.8,
            '세법': 0.8,
            '민사소송법': 1.0,
            '형사소송법': 1.0,
            '기타/일반': 0.5
        }
        
        domain_weight = domain_weights.get(domain, 0.5)
        
        return min(1.0, (base_score + count_bonus) * domain_weight)
