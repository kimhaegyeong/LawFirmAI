# -*- coding: utf-8 -*-
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
                                   target_count: int = 50) -> KeywordExpansionResult:
        """도메인별 키워드 확장"""
        if not self.gemini_client:
            return KeywordExpansionResult(
                domain=domain,
                base_keywords=base_keywords,
                expanded_keywords=base_keywords,
                confidence=0.0,
                expansion_method="fallback",
                api_call_success=False
            )
        
        try:
            prompt = self._create_expansion_prompt(domain, base_keywords, target_count)
            response = await self._call_gemini_api(prompt)
            expanded_keywords = self._parse_gemini_response(response)
            
            return KeywordExpansionResult(
                domain=domain,
                base_keywords=base_keywords,
                expanded_keywords=expanded_keywords,
                confidence=0.8,
                expansion_method="gemini_ai",
                api_call_success=True
            )
            
        except Exception as e:
            self.logger.error(f"키워드 확장 실패 ({domain}): {e}")
            return KeywordExpansionResult(
                domain=domain,
                base_keywords=base_keywords,
                expanded_keywords=base_keywords,
                confidence=0.0,
                expansion_method="error",
                api_call_success=False
            )
    
    def _create_expansion_prompt(self, domain: str, base_keywords: List[str], target_count: int) -> str:
        """키워드 확장을 위한 프롬프트 생성"""
        return f"""
당신은 한국 법률 분야의 전문가입니다. 주어진 {domain} 분야의 기본 키워드들을 바탕으로 관련 키워드를 확장해주세요.

**기본 키워드**: {', '.join(base_keywords[:10])}

다음 조건을 만족하는 키워드를 생성해주세요:

1. **직접 관련 용어**: 기본 키워드와 직접적으로 관련된 법률 용어
2. **하위 개념**: 기본 키워드의 세부 분류나 하위 개념
3. **상위 개념**: 기본 키워드가 속하는 더 넓은 범위의 개념
4. **실무 용어**: 법률 실무에서 자주 사용되는 용어
5. **법령/판례 용어**: 관련 법령이나 판례에서 자주 나오는 용어
6. **동의어/유사어**: 의미가 비슷한 다른 표현들

**출력 형식**:
```json
{{
    "expanded_keywords": [
        "키워드1",
        "키워드2",
        "키워드3"
    ],
    "categories": {{
        "direct_related": ["직접 관련 용어들"],
        "sub_concepts": ["하위 개념들"],
        "super_concepts": ["상위 개념들"],
        "practical_terms": ["실무 용어들"],
        "legal_terms": ["법령/판례 용어들"],
        "synonyms": ["동의어들"]
    }}
}}
```

최대 {target_count}개의 키워드를 생성해주세요. 중복을 피하고, 한국 법률 분야에 특화된 용어를 우선적으로 포함해주세요.
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
        """Gemini 응답 파싱"""
        try:
            # JSON 부분 추출
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                self.logger.warning("JSON 형식 응답을 찾을 수 없음")
                return []
            
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            # 모든 키워드 수집
            all_keywords = set()
            
            if 'expanded_keywords' in data:
                all_keywords.update(data['expanded_keywords'])
            
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
