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
                # 성능 최적화: 더 빠른 모델 사용 및 타임아웃 단축
                self.gemini_client = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-exp",  # 빠른 모델 유지
                    temperature=0.2,  # 0.3 → 0.2 (더 빠른 응답)
                    max_output_tokens=500,  # 1000 → 500 (응답 시간 단축)
                    timeout=2.5,  # 30 → 2.5초 (타임아웃 단축)
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
        
        # 성능 최적화: 프롬프트 단순화 (응답 시간 단축)
        return f"""한국 법률 전문가로서 {domain} 분야 키워드를 확장하세요.
{query_context}
**기본 키워드**: {', '.join(validated_keywords[:10])}

**확장 원칙** (우선순위):
1. 동의어/유의어: "손해배상" → "손해 배상", "손해보상", "배상책임"
2. 관련 법률 용어: "계약" → "계약서", "계약관계", "계약조건"
3. 관련 법령: "손해배상" → "민법 제750조", "불법행위"
4. 판례 키워드: "손해배상 판례", "대법원 판결"

{type_specific_guidance}

**출력 형식** (JSON만):
{{
    "expanded_keywords": ["키워드1", "키워드2", ...]
}}

**주의**: 최대 {target_count}개, 2글자 이상, 관련성 높은 키워드만.
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
        """Gemini 응답 파싱 (개선: 새로운 JSON 형식 지원 및 강화된 파싱)"""
        try:
            import re
            
            # 방법 1: 코드 블록 내부 JSON 추출
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 방법 2: 코드 블록 없이 JSON만 있는 경우
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # 방법 3: 첫 번째 { 부터 마지막 } 까지 추출
                    start_idx = response.find('{')
                    end_idx = response.rfind('}') + 1
                    
                    if start_idx == -1 or end_idx == 0:
                        self.logger.warning("JSON 형식 응답을 찾을 수 없음. 응답 내용: " + response[:200])
                        # 방법 4: JSON이 아닌 경우에도 키워드 추출 시도 (줄바꿈으로 구분된 리스트)
                        lines = [line.strip() for line in response.split('\n') if line.strip() and not line.strip().startswith('#')]
                        if lines:
                            self.logger.info(f"JSON 형식이 아니지만 {len(lines)}개의 키워드를 추출 시도")
                            return lines[:20]  # 최대 20개
                        return []
                    
                    json_str = response[start_idx:end_idx]
            
            # JSON 파싱 시도 (강화된 복구 로직)
            data = None
            json_str_cleaned = json_str
            
            # 1차 시도: 원본 JSON 파싱
            try:
                data = json.loads(json_str_cleaned)
            except json.JSONDecodeError as e:
                error_pos = getattr(e, 'pos', None)
                error_msg = str(e)
                self.logger.debug(f"JSON 파싱 실패 (1차 시도): {error_msg}")
                if error_pos is not None and error_pos < len(json_str_cleaned):
                    context_start = max(0, error_pos - 50)
                    context_end = min(len(json_str_cleaned), error_pos + 50)
                    context = json_str_cleaned[context_start:context_end]
                    self.logger.debug(f"   오류 위치 주변 컨텍스트: ...{context}... (위치: {error_pos})")
                self.logger.warning(f"JSON 파싱 실패, 정리 후 재시도: {error_msg}")
                
                # 2차 시도: 기본 정리 (제어 문자 제거, trailing comma 제거)
                json_str_cleaned = re.sub(r'[^\x20-\x7E\n\r\t]', '', json_str)  # 제어 문자 제거
                json_str_cleaned = re.sub(r',\s*}', '}', json_str_cleaned)  # trailing comma 제거
                json_str_cleaned = re.sub(r',\s*]', ']', json_str_cleaned)  # trailing comma in array
                try:
                    data = json.loads(json_str_cleaned)
                except json.JSONDecodeError:
                    # 3차 시도: 더 강력한 정리 (주석 제거, 잘못된 이스케이프 수정)
                    json_str_cleaned = re.sub(r'//.*?$', '', json_str_cleaned, flags=re.MULTILINE)  # 주석 제거
                    json_str_cleaned = re.sub(r'/\*.*?\*/', '', json_str_cleaned, flags=re.DOTALL)  # 블록 주석 제거
                    json_str_cleaned = re.sub(r'\\"', '"', json_str_cleaned)  # 잘못된 이스케이프 수정
                    json_str_cleaned = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str_cleaned)  # 따옴표 없는 키 추가
                    try:
                        data = json.loads(json_str_cleaned)
                    except json.JSONDecodeError:
                        # 4차 시도: 부분 JSON 추출 (첫 번째 유효한 JSON 객체만)
                        try:
                            # 중괄호 매칭으로 첫 번째 완전한 JSON 객체 추출
                            brace_count = 0
                            start_pos = -1
                            for i, char in enumerate(json_str):
                                if char == '{':
                                    if start_pos == -1:
                                        start_pos = i
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0 and start_pos != -1:
                                        partial_json = json_str[start_pos:i+1]
                                        data = json.loads(partial_json)
                                        self.logger.info(f"부분 JSON 추출 성공 (위치: {start_pos}-{i+1})")
                                        break
                        except (json.JSONDecodeError, ValueError) as e4:
                            error_pos = getattr(e4, 'pos', None) if isinstance(e4, json.JSONDecodeError) else None
                            error_msg = str(e4)
                            self.logger.error(f"JSON 파싱 모든 시도 실패: {error_msg}")
                            if error_pos is not None:
                                self.logger.error(f"   최종 오류 위치: {error_pos}, JSON 길이: {len(json_str)}")
                            self.logger.debug(f"   실패한 JSON 문자열 (처음 500자): {json_str[:500]}")
                            # 최후의 수단: 키워드를 직접 추출
                            keywords = re.findall(r'"([^"]+)"', json_str)
                            # 배열 형식도 시도
                            if not keywords:
                                keywords = re.findall(r'["\']([^"\']+)["\']', json_str)
                            # 줄바꿈으로 구분된 리스트도 시도
                            if not keywords:
                                lines = [line.strip() for line in json_str.split('\n') 
                                        if line.strip() and not line.strip().startswith('#') 
                                        and not line.strip().startswith('//')]
                                keywords = [line for line in lines if len(line) > 1 and not line.startswith('{') and not line.startswith('}')]
                            
                            if keywords:
                                self.logger.info(f"JSON 파싱 실패했지만 {len(keywords)}개의 키워드를 텍스트에서 추출")
                                return keywords[:20]
                            return []
            
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
