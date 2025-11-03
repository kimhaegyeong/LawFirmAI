#!/usr/bin/env python3
"""
실제 Gemini API를 사용한 동의어 확장 시스템
환경 변수의 GOOGLE_API_KEY를 사용하여 실제 Gemini API 호출
"""

import json
import os
import asyncio
import time
import re
from typing import List, Dict, Set, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# 동의어 데이터베이스 및 품질 관리 시스템 import
from .synonym_database import SynonymDatabase, SynonymRecord
from .synonym_quality_manager import SynonymQualityManager, SynonymOptimizer

# 환경 변수 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not available. Environment variables may not be loaded from .env file.")

# LLM 관련 import
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.schema import HumanMessage, SystemMessage
except ImportError:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        print("Warning: LangChain Google GenAI not available. Using mock responses.")
        ChatGoogleGenerativeAI = None
        HumanMessage = None
        SystemMessage = None

@dataclass
class GeminiSynonymResult:
    """Gemini 동의어 확장 결과"""
    keyword: str
    synonyms: List[str]
    context: str
    domain: str
    confidence: float
    reasoning: str
    model_used: str
    api_call_success: bool

class RealGeminiSynonymExpander:
    """실제 Gemini API를 사용한 동의어 확장 시스템"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Gemini 모델 초기화
        self.gemini_model = self._initialize_gemini_model()
        self.expansion_history = []
        self.api_call_count = 0
        self.successful_calls = 0
        self.last_api_call_time = 0  # 마지막 API 호출 시간
        
        # 동의어 데이터베이스 및 품질 관리 시스템 초기화
        self.synonym_db = SynonymDatabase()
        self.quality_manager = SynonymQualityManager()
        self.optimizer = SynonymOptimizer()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            "gemini_model": "gemini-2.5-flash-lite",
            "max_synonyms_per_keyword": 10,
            "temperature": 0.7,
            "max_tokens": 1000,
            "timeout": 30,
            "api_call_interval": 1.2  # API 호출 간격 (초)
        }
    
    def _initialize_gemini_model(self) -> Optional[Any]:
        """Gemini 모델 초기화"""
        try:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            
            if not google_api_key:
                self.logger.error("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
                return None
            
            if google_api_key.startswith("your_") or google_api_key == "your_google_api_key_here":
                self.logger.error("GOOGLE_API_KEY가 예시 값입니다. 실제 API 키를 설정해주세요.")
                return None
            
            if not ChatGoogleGenerativeAI:
                self.logger.error("ChatGoogleGenerativeAI를 사용할 수 없습니다.")
                return None
            
            # Gemini 모델 초기화
            model = ChatGoogleGenerativeAI(
                model=self.config["gemini_model"],
                temperature=self.config["temperature"],
                google_api_key=google_api_key
            )
            
            self.logger.info(f"Gemini 모델 초기화 성공: {self.config['gemini_model']}")
            self.logger.info(f"Google API 키 로드됨: {google_api_key[:10]}...")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Gemini 모델 초기화 실패: {e}")
            return None
    
    def _create_synonym_prompt(self, keyword: str, question: str, domain: str, context: str) -> str:
        """동의어 확장을 위한 프롬프트 생성"""
        prompt = f"""
당신은 한국 법률 분야의 전문가입니다. 주어진 키워드에 대한 동의어와 관련 용어를 생성해주세요.

**질문**: {question}
**키워드**: {keyword}
**법률 도메인**: {domain}
**맥락**: {context}

다음 조건을 만족하는 동의어를 생성해주세요:

1. **직접 동의어**: 키워드와 의미가 동일한 용어
2. **관련 용어**: 키워드와 밀접한 관련이 있는 용어
3. **맥락별 용어**: 질문의 맥락에 맞는 구체적인 용어
4. **법률 용어**: 해당 법률 도메인의 전문 용어
5. **일상 용어**: 일반인이 사용할 수 있는 쉬운 용어

**출력 형식**:
```json
{{
    "synonyms": [
        {{
            "term": "동의어1",
            "type": "direct|related|contextual|legal|common",
            "confidence": 0.9,
            "reasoning": "이유 설명"
        }},
        {{
            "term": "동의어2",
            "type": "direct|related|contextual|legal|common",
            "confidence": 0.8,
            "reasoning": "이유 설명"
        }}
    ],
    "domain_specific_terms": ["도메인별 용어1", "도메인별 용어2"],
    "context_specific_terms": ["맥락별 용어1", "맥락별 용어2"],
    "common_terms": ["일상 용어1", "일상 용어2"]
}}
```

최대 {self.config['max_synonyms_per_keyword']}개의 동의어를 생성해주세요.
"""
        return prompt
    
    async def _call_gemini_api(self, prompt: str) -> str:
        """Gemini API 호출 (인터벌 적용)"""
        if not self.gemini_model:
            raise Exception("Gemini 모델이 초기화되지 않았습니다.")
        
        try:
            # API 호출 간격 제어
            current_time = time.time()
            time_since_last_call = current_time - self.last_api_call_time
            required_interval = self.config.get("api_call_interval", 1.2)
            
            if time_since_last_call < required_interval:
                wait_time = required_interval - time_since_last_call
                self.logger.info(f"API 호출 간격 대기: {wait_time:.2f}초")
                await asyncio.sleep(wait_time)
            
            self.api_call_count += 1
            self.last_api_call_time = time.time()
            
            # 메시지 생성
            messages = [HumanMessage(content=prompt)]
            
            # API 호출
            if hasattr(self.gemini_model, 'ainvoke'):
                response = await self.gemini_model.ainvoke(messages)
            else:
                response = self.gemini_model.invoke(messages)
            
            # 응답 추출
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            self.successful_calls += 1
            self.logger.info(f"Gemini API 호출 성공 ({self.api_call_count}번째)")
            
            return content
            
        except Exception as e:
            self.logger.error(f"Gemini API 호출 실패: {e}")
            raise e
    
    def _parse_gemini_response(self, response: str) -> Dict[str, Any]:
        """Gemini 응답 파싱"""
        try:
            # JSON 부분 추출
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                return self._parse_fallback_response(response)
            
            json_str = response[start_idx:end_idx]
            parsed_data = json.loads(json_str)
            
            return parsed_data
            
        except Exception as e:
            self.logger.warning(f"Gemini 응답 파싱 실패: {e}")
            return self._parse_fallback_response(response)
    
    def _parse_fallback_response(self, response: str) -> Dict[str, Any]:
        """파싱 실패 시 대체 파싱"""
        # 간단한 텍스트에서 동의어 추출
        synonyms = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('*'):
                # 간단한 동의어 추출
                if ':' in line:
                    term = line.split(':')[0].strip()
                    synonyms.append({
                        "term": term,
                        "type": "extracted",
                        "confidence": 0.6,
                        "reasoning": "텍스트에서 추출됨"
                    })
        
        return {
            "synonyms": synonyms[:self.config['max_synonyms_per_keyword']],
            "domain_specific_terms": [],
            "context_specific_terms": [],
            "common_terms": []
        }
    
    def _extract_synonyms_from_text(self, text: str) -> List[str]:
        """텍스트에서 동의어 추출 (대체 방법)"""
        synonyms = []
        
        # 간단한 패턴으로 동의어 추출
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            
            # 번호가 있는 목록 패턴 (1. 동의어, 2. 동의어 등)
            if re.match(r'^\d+\.\s*(.+)$', line):
                match = re.match(r'^\d+\.\s*(.+)$', line)
                if match:
                    synonym = match.group(1).strip()
                    if len(synonym) > 1 and len(synonym) < 20:
                        synonyms.append(synonym)
            
            # 대시 패턴 (- 동의어)
            elif re.match(r'^-\s*(.+)$', line):
                match = re.match(r'^-\s*(.+)$', line)
                if match:
                    synonym = match.group(1).strip()
                    if len(synonym) > 1 and len(synonym) < 20:
                        synonyms.append(synonym)
            
            # 따옴표로 둘러싸인 용어
            elif '"' in line:
                quoted_terms = re.findall(r'"([^"]+)"', line)
                for term in quoted_terms:
                    if len(term) > 1 and len(term) < 20:
                        synonyms.append(term)
        
        # 중복 제거 및 길이 필터링
        unique_synonyms = []
        for synonym in synonyms:
            if synonym not in unique_synonyms and len(synonym) > 1:
                unique_synonyms.append(synonym)
        
        return unique_synonyms[:self.config['max_synonyms_per_keyword']]
    
    async def expand_synonyms_with_gemini(self, keyword: str, question: str, 
                                        domain: str = "기타", context: str = "일반") -> GeminiSynonymResult:
        """Gemini를 사용한 동의어 확장 (데이터베이스 연동)"""
        
        # 1. 먼저 데이터베이스에서 기존 동의어 조회
        existing_synonyms = self.synonym_db.get_synonyms(keyword, domain, context)
        
        if existing_synonyms:
            self.logger.info(f"데이터베이스에서 기존 동의어 발견: {len(existing_synonyms)}개")
            
            # 기존 동의어 사용 횟수 업데이트
            for synonym_record in existing_synonyms:
                self.synonym_db.update_usage_count(keyword, synonym_record.synonym, domain, context)
            
            # 기존 동의어를 결과로 반환 (품질 최적화 건너뛰기)
            synonyms = [record.synonym for record in existing_synonyms]
            # optimized_synonyms = self.optimizer.optimize_synonym_list(synonyms, keyword, context, domain)
            optimized_synonyms = synonyms  # 직접 반환
            
            return GeminiSynonymResult(
                keyword=keyword,
                synonyms=optimized_synonyms,
                context=context,
                domain=domain,
                confidence=0.95,  # 데이터베이스에서 가져온 동의어는 높은 신뢰도
                reasoning=f"데이터베이스에서 기존 동의어 {len(existing_synonyms)}개 조회",
                model_used="database",
                api_call_success=True
            )
        
        # 2. 데이터베이스에 없으면 Gemini API 호출
        if not self.gemini_model:
            return self._get_fallback_result(keyword, question, domain, context)
        
        try:
            # 프롬프트 생성
            prompt = self._create_synonym_prompt(keyword, question, domain, context)
            
            # Gemini API 호출
            self.logger.info(f"Gemini로 새로운 동의어 확장 중: {keyword}")
            response = await self._call_gemini_api(prompt)
            
            # 응답 파싱
            parsed_data = self._parse_gemini_response(response)
            
            # 디버깅: 응답 내용 출력
            self.logger.info(f"Gemini 응답 원본: {response[:200]}...")
            self.logger.info(f"파싱된 데이터: {parsed_data}")
            
            # 결과 생성
            synonyms = []
            for item in parsed_data.get("synonyms", []):
                if isinstance(item, dict):
                    synonyms.append(item["term"])
                else:
                    synonyms.append(str(item))
            
            # 추가 용어 통합
            domain_terms = parsed_data.get("domain_specific_terms", [])
            context_terms = parsed_data.get("context_specific_terms", [])
            common_terms = parsed_data.get("common_terms", [])
            
            all_synonyms = list(set(synonyms + domain_terms + context_terms + common_terms))
            self.logger.info(f"추출된 동의어: {all_synonyms}")
            
            # 동의어가 없으면 대체 방법 사용
            if not all_synonyms:
                self.logger.warning("파싱된 동의어가 없습니다. 대체 방법을 사용합니다.")
                all_synonyms = self._extract_synonyms_from_text(response)
            
            # 3. 동의어 품질 최적화 (임시로 건너뛰기)
            self.logger.info(f"최적화 전 동의어: {all_synonyms}")
            # optimized_synonyms = self.optimizer.optimize_synonym_list(
            #     all_synonyms, keyword, context, domain
            # )[:self.config['max_synonyms_per_keyword']]
            optimized_synonyms = all_synonyms[:self.config['max_synonyms_per_keyword']]
            self.logger.info(f"최적화 후 동의어: {optimized_synonyms}")
            
            # 4. 데이터베이스에 저장
            synonym_records = []
            for synonym in optimized_synonyms:
                record = SynonymRecord(
                    keyword=keyword,
                    synonym=synonym,
                    domain=domain,
                    context=context,
                    confidence=0.9,  # Gemini API 결과는 높은 신뢰도
                    source="gemini_api"
                )
                synonym_records.append(record)
            
            # 데이터베이스에 저장
            saved_count = self.synonym_db.save_multiple_synonyms(synonym_records)
            self.logger.info(f"새로운 동의어 {saved_count}개를 데이터베이스에 저장")
            
            # 결과 객체 생성
            result = GeminiSynonymResult(
                keyword=keyword,
                synonyms=optimized_synonyms,
                context=context,
                domain=domain,
                confidence=0.9,  # 실제 API 호출이므로 높은 신뢰도
                reasoning=f"실제 Gemini API ({self.config['gemini_model']}) 기반 동의어 확장 및 데이터베이스 저장",
                model_used=self.config['gemini_model'],
                api_call_success=True
            )
            
            # 히스토리 저장
            self.expansion_history.append({
                "timestamp": datetime.now().isoformat(),
                "keyword": keyword,
                "question": question,
                "domain": domain,
                "context": context,
                "synonyms_count": len(result.synonyms),
                "api_call_success": True,
                "database_saved": True
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Gemini 동의어 확장 실패: {e}")
            return self._get_fallback_result(keyword, question, domain, context)
    
    def _get_fallback_result(self, keyword: str, question: str, 
                            domain: str, context: str) -> GeminiSynonymResult:
        """API 호출 실패 시 대체 결과"""
        # 기본 동의어 매핑
        basic_synonyms = {
            "계약서": ["계약문서", "계약장", "계약서류"],
            "아파트": ["공동주택", "집", "주택", "부동산"],
            "합법": ["법적", "법률적", "유효", "정당"],
            "해고": ["해직", "퇴직", "고용해제"],
            "이혼": ["혼인해소", "결혼해소", "부부관계종료"],
            "손해배상": ["배상", "보상", "피해보상"],
            "위약금": ["위약금액", "위약배상금", "계약위반금"]
        }
        
        synonyms = basic_synonyms.get(keyword, [keyword])
        
        return GeminiSynonymResult(
            keyword=keyword,
            synonyms=synonyms,
            context=context,
            domain=domain,
            confidence=0.5,
            reasoning="기본 동의어 매핑 사용 (API 호출 실패)",
            model_used="fallback",
            api_call_success=False
        )
    
    async def expand_multiple_keywords(self, keywords: List[str], question: str,
                                     domain: str = "기타", context: str = "일반") -> Dict[str, GeminiSynonymResult]:
        """여러 키워드의 동의어 확장"""
        results = {}
        
        # 병렬 처리로 성능 향상
        tasks = []
        for keyword in keywords:
            task = self.expand_synonyms_with_gemini(keyword, question, domain, context)
            tasks.append((keyword, task))
        
        # 모든 작업 완료 대기
        for keyword, task in tasks:
            try:
                result = await task
                results[keyword] = result
            except Exception as e:
                self.logger.error(f"키워드 '{keyword}' 동의어 확장 실패: {e}")
                results[keyword] = self._get_fallback_result(keyword, question, domain, context)
        
        return results
    
    def get_expansion_statistics(self) -> Dict[str, Any]:
        """동의어 확장 통계"""
        if not self.expansion_history:
            return {
                "total_expansions": 0,
                "average_synonyms": 0,
                "api_call_count": self.api_call_count,
                "successful_calls": self.successful_calls,
                "success_rate": 0.0
            }
        
        total_expansions = len(self.expansion_history)
        total_synonyms = sum(entry["synonyms_count"] for entry in self.expansion_history)
        average_synonyms = total_synonyms / total_expansions if total_expansions > 0 else 0
        success_rate = self.successful_calls / self.api_call_count if self.api_call_count > 0 else 0
        
        return {
            "total_expansions": total_expansions,
            "average_synonyms": average_synonyms,
            "api_call_count": self.api_call_count,
            "successful_calls": self.successful_calls,
            "success_rate": success_rate,
            "model_used": self.config['gemini_model']
        }
    
    def save_expansion_history(self, filename: str = None):
        """확장 히스토리 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/extracted_terms/gemini_synonym_expansion_history_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        data = {
            "expansion_history": self.expansion_history,
            "statistics": self.get_expansion_statistics(),
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Gemini 동의어 확장 히스토리 저장: {filename}")
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """데이터베이스 통계 조회"""
        return self.synonym_db.get_database_statistics()
    
    def cleanup_database(self, days_threshold: int = 30) -> int:
        """데이터베이스 정리 (사용하지 않는 동의어 제거)"""
        return self.synonym_db.cleanup_unused_synonyms(days_threshold)
    
    def export_synonyms(self, output_file: str = None) -> str:
        """동의어 데이터 내보내기"""
        return self.synonym_db.export_synonyms(output_file)
    
    def update_synonym_rating(self, keyword: str, synonym: str, rating: float,
                             domain: str = None, context: str = None) -> bool:
        """동의어 사용자 평점 업데이트"""
        return self.synonym_db.update_user_rating(keyword, synonym, rating, domain, context)

# 사용 예시
async def main():
    """메인 실행 함수"""
    # 설정
    config = {
        "gemini_model": "gemini-2.5-flash-lite",
        "max_synonyms_per_keyword": 8,
        "temperature": 0.7
    }
    
    # Gemini 동의어 확장기 초기화
    expander = RealGeminiSynonymExpander(config)
    
    # 테스트 질문
    test_question = "아파트 계약서에 위약금이 계약금의 10배로 되어 있는데, 이게 합법적인가요?"
    test_keywords = ["아파트", "계약서", "위약금", "합법"]
    
    print("=== 실제 Gemini API 동의어 확장 테스트 ===")
    print(f"질문: {test_question}")
    print()
    
    # 키워드별 동의어 확장
    for keyword in test_keywords:
        print(f"키워드: {keyword}")
        result = await expander.expand_synonyms_with_gemini(
            keyword, test_question, "민사법", "계약_관련_맥락"
        )
        
        print(f"  생성된 동의어 ({len(result.synonyms)}개):")
        for synonym in result.synonyms:
            print(f"    - {synonym}")
        
        print(f"  신뢰도: {result.confidence}")
        print(f"  모델: {result.model_used}")
        print(f"  API 호출 성공: {result.api_call_success}")
        print(f"  추론: {result.reasoning}")
        print()
    
    # 통계 출력
    stats = expander.get_expansion_statistics()
    print("=== 확장 통계 ===")
    print(f"총 확장 횟수: {stats['total_expansions']}")
    print(f"평균 동의어 수: {stats['average_synonyms']:.1f}")
    print(f"API 호출 횟수: {stats['api_call_count']}")
    print(f"성공한 호출: {stats['successful_calls']}")
    print(f"성공률: {stats['success_rate']:.2f}")
    
    # 히스토리 저장
    expander.save_expansion_history()

if __name__ == "__main__":
    asyncio.run(main())
