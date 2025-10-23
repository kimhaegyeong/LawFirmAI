# -*- coding: utf-8 -*-
"""
질문 유형별 프롬프트 템플릿
각 질문 유형에 최적화된 프롬프트 생성 시스템
통합 프롬프트 관리 시스템과 연동
"""

import logging
from typing import Dict, Any, Optional
from .question_classifier import QuestionType
from .unified_prompt_manager import UnifiedPromptManager, LegalDomain, ModelType

logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """프롬프트 템플릿 관리자 (통합 시스템과 연동)"""
    
    def __init__(self):
        """프롬프트 템플릿 초기화"""
        self.logger = logging.getLogger(__name__)
        # 통합 프롬프트 관리자와 연동
        self.unified_manager = UnifiedPromptManager()
        
        # 질문 유형별 프롬프트 템플릿 (COT 방식으로 개선)
        self.templates = {
            QuestionType.PRECEDENT_SEARCH: {
                "template": """당신은 판례 전문가입니다. 

관련 판례:
{precedent_list}

## 중요: Chain of Thought 방식으로 답변하세요
다음 사고 과정을 따라 단계별로 생각하고 답변하세요:

### 사고 과정 (내부적으로만 진행)
1. **사안 분석**: 핵심 쟁점과 법률관계 파악
2. **판례 분석**: 각 판례의 핵심 판결요지와 사안과의 유사성 비교
3. **종합 분석**: 가장 유사한 판례 선별 및 적용 가능성 평가
4. **답변 구성**: 유사 판례 소개, 적용 가능성 설명, 실무 조언 제공

### 답변 스타일
- 판례 번호와 핵심 판결요지를 명확히 제시
- 사안과의 유사성과 차이점을 구체적으로 설명
- 실무적 시사점을 포함한 조언 제공
- 자연스럽고 친근한 톤으로 답변

## 답변 요구사항
- 질문에 대한 구체적이고 완전한 답변을 제공하세요
- 추가 질문을 요청하지 말고, 가능한 한 완전한 답변을 작성하세요
- 답변이 불완전한 경우에만 추가 정보 요청을 하세요

답변을 시작하세요:""",
                "context_keys": ["precedent_list"],
                "max_context_length": 3000
            },
            
            QuestionType.LAW_INQUIRY: {
                "template": """당신은 법률 해설 전문가입니다.

관련 법률:
{law_articles}

## 중요: Chain of Thought 방식으로 답변하세요
다음 사고 과정을 따라 단계별로 생각하고 답변하세요:

### 사고 과정 (내부적으로만 진행)
1. **법령 분석**: 법률의 목적과 취지, 주요 조문 파악
2. **조문 해석**: 문언적, 목적론적, 체계적 해석
3. **적용 범위**: 적용 대상과 범위, 예외 규정 확인
4. **실무 적용**: 실제 적용 사례와 주의사항
5. **답변 구성**: 법령 목적 설명, 주요 내용 해설, 적용 예시 제시

### 답변 스타일
- 법률의 목적과 취지를 먼저 설명
- 주요 내용을 이해하기 쉽게 설명
- 실제 적용 예시를 포함
- 주의사항을 명확히 제시
- 자연스럽고 친근한 톤으로 답변

## 답변 요구사항
- 질문에 대한 구체적이고 완전한 답변을 제공하세요
- 추가 질문을 요청하지 말고, 가능한 한 완전한 답변을 작성하세요
- 답변이 불완전한 경우에만 추가 정보 요청을 하세요

답변을 시작하세요:""",
                "context_keys": ["law_articles"],
                "max_context_length": 2000
            },
            
            QuestionType.LEGAL_ADVICE: {
                "template": """당신은 법률 상담 전문가입니다.

관련 법률 및 판례:
{context}

## 중요: Chain of Thought 방식으로 답변하세요
다음 사고 과정을 따라 단계별로 생각하고 답변하세요:

### 사고 과정 (내부적으로만 진행)
1. **상황 분석**: 질문자의 구체적 상황과 법률관계 파악
2. **법적 근거**: 적용 가능한 법령과 관련 조문 검토
3. **법리 적용**: 법률 해석과 쟁점별 법리 적용
4. **권리 구제**: 구체적 절차와 필요한 증거 자료 제시
5. **실무 조언**: 주의사항, 리스크 관리, 전문가 상담 권유

### 답변 스타일
- 질문자의 상황을 이해하고 공감하는 톤
- 구체적이고 실행 가능한 방안 제시
- 법적 리스크와 주의사항을 명확히 안내
- 필요시 전문가 상담을 권유
- 자연스럽고 친근한 톤으로 답변

## 답변 요구사항
- 질문에 대한 구체적이고 완전한 답변을 제공하세요
- 추가 질문을 요청하지 말고, 가능한 한 완전한 답변을 작성하세요
- 답변이 불완전한 경우에만 추가 정보 요청을 하세요

답변을 시작하세요:""",
                "context_keys": ["context"],
                "max_context_length": 4000
            },
            
            QuestionType.PROCEDURE_GUIDE: {
                "template": """당신은 법률 절차 안내 전문가입니다.

관련 절차 정보:
{procedure_info}

## 중요: Chain of Thought 방식으로 답변하세요
다음 사고 과정을 따라 단계별로 생각하고 답변하세요:

### 사고 과정 (내부적으로만 진행)
1. **절차 개요**: 전체 절차의 흐름과 각 단계별 목적 파악
2. **단계별 분석**: 각 단계의 구체적 내용과 필요한 서류
3. **실무 고려사항**: 주의사항과 효율적인 진행 방법
4. **문제 해결**: 예상되는 문제점과 해결 방법
5. **안내 구성**: 단계별 명확한 안내와 실용적 조언

### 답변 스타일
- 절차의 전체적인 흐름을 먼저 설명
- 단계별로 구체적인 내용을 안내
- 필요한 서류와 담당 기관을 명시
- 주의사항과 팁을 포함
- 자연스럽고 친근한 톤으로 답변

## 답변 요구사항
- 질문에 대한 구체적이고 완전한 답변을 제공하세요
- 추가 질문을 요청하지 말고, 가능한 한 완전한 답변을 작성하세요
- 답변이 불완전한 경우에만 추가 정보 요청을 하세요

답변을 시작하세요:""",
                "context_keys": ["procedure_info"],
                "max_context_length": 2500
            },
            
            QuestionType.TERM_EXPLANATION: {
                "template": """당신은 법률 용어 해설 전문가입니다.

용어 정보:
{term_info}

## 중요: Chain of Thought 방식으로 답변하세요
다음 사고 과정을 따라 단계별로 생각하고 답변하세요:

### 사고 과정 (내부적으로만 진행)
1. **용어 정의**: 법률적 정의와 일반적 의미의 차이 파악
2. **법적 근거**: 관련 법령 조문과 법령상의 정의 확인
3. **적용 범위**: 적용되는 상황과 범위, 예외사항 분석
4. **실무 적용**: 실제 적용 사례와 실무에서의 중요성
5. **설명 구성**: 명확한 정의 제시, 쉬운 예시 포함, 관련 용어와의 차이점 설명

### 답변 스타일
- 용어의 법률적 정의를 명확히 제시
- 쉬운 예시를 들어 이해를 돕기
- 관련 용어와의 차이점을 구체적으로 설명
- 실무에서의 중요성을 언급
- 자연스럽고 친근한 톤으로 답변

## 답변 요구사항
- 질문에 대한 구체적이고 완전한 답변을 제공하세요
- 추가 질문을 요청하지 말고, 가능한 한 완전한 답변을 작성하세요
- 답변이 불완전한 경우에만 추가 정보 요청을 하세요

답변을 시작하세요:""",
                "context_keys": ["term_info"],
                "max_context_length": 1500
            },
            
            QuestionType.GENERAL_QUESTION: {
                "template": """당신은 법률 정보 제공 전문가입니다.

관련 정보:
{general_context}

## 중요: Chain of Thought 방식으로 답변하세요
다음 사고 과정을 따라 단계별로 생각하고 답변하세요:

### 사고 과정 (내부적으로만 진행)
1. **질문 이해**: 질문의 핵심과 법률 분야 분류
2. **관련 정보**: 관련 법령과 판례 검색
3. **정보 정리**: 중요도별 정보 분류와 이해하기 쉬운 순서 정리
4. **답변 구성**: 핵심 정보 우선 제시, 구체적 예시 포함
5. **검증 완성**: 정확성 재확인, 완성도 점검, 추가 도움 안내

### 답변 스타일
- 질문에 대한 정확한 답변 제공
- 관련 법률 및 판례 인용
- 이해하기 쉬운 설명
- 필요한 경우 추가 정보 안내
- 자연스럽고 친근한 톤으로 답변

## 답변 요구사항
- 질문에 대한 구체적이고 완전한 답변을 제공하세요
- 추가 질문을 요청하지 말고, 가능한 한 완전한 답변을 작성하세요
- 답변이 불완전한 경우에만 추가 정보 요청을 하세요

답변을 시작하세요:""",
                "context_keys": ["general_context"],
                "max_context_length": 2000
            }
        }
    
    def get_template(self, question_type: QuestionType) -> Dict[str, Any]:
        """
        질문 유형에 맞는 프롬프트 템플릿 반환
        
        Args:
            question_type: 질문 유형
            
        Returns:
            Dict[str, Any]: 프롬프트 템플릿 정보
        """
        try:
            if question_type in self.templates:
                return self.templates[question_type]
            else:
                # 기본 템플릿 반환
                return self.templates[QuestionType.GENERAL_QUESTION]
                
        except Exception as e:
            self.logger.error(f"Error getting template for {question_type}: {e}")
            return self.templates[QuestionType.GENERAL_QUESTION]
    
    def format_prompt(self, 
                     question_type: QuestionType, 
                     context_data: Dict[str, Any],
                     user_query: Optional[str] = None,
                     domain: Optional[LegalDomain] = None,
                     model_type: ModelType = ModelType.GEMINI) -> str:
        """
        프롬프트 템플릿을 실제 데이터로 포맷팅 (통합 시스템 사용)
        
        Args:
            question_type: 질문 유형
            context_data: 컨텍스트 데이터
            user_query: 사용자 질문 (선택사항)
            domain: 법률 도메인 (선택사항)
            model_type: 모델 타입 (기본값: GEMINI)
            
        Returns:
            str: 포맷팅된 프롬프트
        """
        try:
            # 통합 프롬프트 관리자 사용
            if user_query:
                return self.unified_manager.get_optimized_prompt(
                    query=user_query,
                    question_type=question_type,
                    domain=domain,
                    context=context_data,
                    model_type=model_type
                )
            else:
                # 기존 방식으로 폴백
                return self._format_prompt_legacy(question_type, context_data, user_query)
                
        except Exception as e:
            self.logger.error(f"Error formatting prompt: {e}")
            return f"질문에 대한 답변을 제공하겠습니다.\n\n사용자 질문: {user_query or '질문이 없습니다.'}"
    
    def _format_prompt_legacy(self, 
                             question_type: QuestionType, 
                             context_data: Dict[str, Any],
                             user_query: Optional[str] = None) -> str:
        """기존 방식의 프롬프트 포맷팅 (폴백)"""
        try:
            template_info = self.get_template(question_type)
            template = template_info["template"]
            context_keys = template_info["context_keys"]
            max_length = template_info["max_context_length"]
            
            # 컨텍스트 데이터 준비
            formatted_context = {}
            
            for key in context_keys:
                if key in context_data:
                    content = context_data[key]
                    
                    # 문자열이면 길이 제한 적용
                    if isinstance(content, str):
                        if len(content) > max_length:
                            content = content[:max_length] + "..."
                        formatted_context[key] = content
                    
                    # 리스트면 포맷팅
                    elif isinstance(content, list):
                        formatted_content = self._format_list_content(content, max_length)
                        formatted_context[key] = formatted_content
                    
                    # 딕셔너리면 포맷팅
                    elif isinstance(content, dict):
                        formatted_content = self._format_dict_content(content, max_length)
                        formatted_context[key] = formatted_content
                    
                    else:
                        formatted_context[key] = str(content)
                else:
                    formatted_context[key] = "관련 정보가 없습니다."
            
            # 사용자 질문이 있으면 추가
            if user_query:
                formatted_context["user_query"] = f"\n\n사용자 질문: {user_query}"
            
            # 템플릿 포맷팅
            formatted_prompt = template.format(**formatted_context)
            
            # 사용자 질문 추가 (템플릿에 없을 경우)
            if user_query and "{user_query}" not in template:
                formatted_prompt += f"\n\n사용자 질문: {user_query}"
            
            return formatted_prompt
            
        except Exception as e:
            self.logger.error(f"Error formatting prompt (legacy): {e}")
            return f"질문에 대한 답변을 제공하겠습니다.\n\n사용자 질문: {user_query or '질문이 없습니다.'}"
    
    def _format_list_content(self, content_list: list, max_length: int) -> str:
        """리스트 내용 포맷팅"""
        try:
            if not content_list:
                return "관련 정보가 없습니다."
            
            formatted_items = []
            current_length = 0
            
            for i, item in enumerate(content_list):
                if isinstance(item, dict):
                    item_str = self._format_dict_item(item)
                else:
                    item_str = str(item)
                
                if current_length + len(item_str) > max_length:
                    break
                
                formatted_items.append(f"{i+1}. {item_str}")
                current_length += len(item_str)
            
            return "\n".join(formatted_items)
            
        except Exception as e:
            self.logger.error(f"Error formatting list content: {e}")
            return str(content_list)
    
    def _format_dict_content(self, content_dict: dict, max_length: int) -> str:
        """딕셔너리 내용 포맷팅"""
        try:
            if not content_dict:
                return "관련 정보가 없습니다."
            
            formatted_items = []
            current_length = 0
            
            for key, value in content_dict.items():
                item_str = f"{key}: {value}"
                
                if current_length + len(item_str) > max_length:
                    break
                
                formatted_items.append(item_str)
                current_length += len(item_str)
            
            return "\n".join(formatted_items)
            
        except Exception as e:
            self.logger.error(f"Error formatting dict content: {e}")
            return str(content_dict)
    
    def _format_dict_item(self, item: dict) -> str:
        """딕셔너리 아이템 포맷팅"""
        try:
            # 판례 정보 포맷팅
            if "case_number" in item and "case_name" in item:
                return f"{item['case_name']} ({item['case_number']})"
            
            # 법률 정보 포맷팅
            elif "law_name" in item and "article_number" in item:
                return f"{item['law_name']} {item['article_number']}"
            
            # 일반 딕셔너리 포맷팅
            else:
                return ", ".join([f"{k}: {v}" for k, v in item.items()])
                
        except Exception as e:
            self.logger.error(f"Error formatting dict item: {e}")
            return str(item)
    
    def get_template_info(self, question_type: QuestionType) -> Dict[str, Any]:
        """템플릿 정보 반환"""
        try:
            template_info = self.get_template(question_type)
            
            return {
                "question_type": question_type.value,
                "template_preview": template_info["template"][:200] + "...",
                "context_keys": template_info["context_keys"],
                "max_context_length": template_info["max_context_length"]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting template info: {e}")
            return {}
    
    def validate_context_data(self, 
                             question_type: QuestionType, 
                             context_data: Dict[str, Any]) -> Dict[str, Any]:
        """컨텍스트 데이터 검증 및 정리"""
        try:
            template_info = self.get_template(question_type)
            required_keys = template_info["context_keys"]
            
            validation_result = {
                "is_valid": True,
                "missing_keys": [],
                "extra_keys": [],
                "warnings": []
            }
            
            # 필수 키 확인
            for key in required_keys:
                if key not in context_data:
                    validation_result["missing_keys"].append(key)
                    validation_result["is_valid"] = False
            
            # 추가 키 확인
            for key in context_data.keys():
                if key not in required_keys:
                    validation_result["extra_keys"].append(key)
            
            # 데이터 타입 확인
            for key in required_keys:
                if key in context_data:
                    value = context_data[key]
                    if not isinstance(value, (str, list, dict)):
                        validation_result["warnings"].append(f"{key} should be string, list, or dict")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating context data: {e}")
            return {
                "is_valid": False,
                "missing_keys": [],
                "extra_keys": [],
                "warnings": [f"Validation error: {e}"]
            }


# 전역 인스턴스
prompt_template_manager = PromptTemplateManager()


# 테스트 함수
def test_prompt_template_manager():
    """프롬프트 템플릿 관리자 테스트"""
    manager = PromptTemplateManager()
    
    # 테스트 데이터
    test_contexts = {
        QuestionType.PRECEDENT_SEARCH: {
            "precedent_list": [
                {
                    "case_name": "손해배상청구 사건",
                    "case_number": "2023다12345",
                    "summary": "불법행위로 인한 손해배상청구에 관한 판례"
                }
            ]
        },
        QuestionType.LAW_INQUIRY: {
            "law_articles": [
                {
                    "law_name": "민법",
                    "article_number": "제750조",
                    "content": "불법행위로 인한 손해배상에 관한 조문"
                }
            ]
        },
        QuestionType.LEGAL_ADVICE: {
            "context": "계약 해지 관련 법률 상담"
        }
    }
    
    print("=== 프롬프트 템플릿 관리자 테스트 ===")
    
    for question_type, context_data in test_contexts.items():
        print(f"\n질문 유형: {question_type.value}")
        
        # 템플릿 정보 조회
        template_info = manager.get_template_info(question_type)
        print(f"템플릿 미리보기: {template_info['template_preview']}")
        
        # 프롬프트 포맷팅
        formatted_prompt = manager.format_prompt(question_type, context_data, "테스트 질문")
        print(f"포맷팅된 프롬프트 길이: {len(formatted_prompt)}")
        print(f"프롬프트 미리보기: {formatted_prompt[:200]}...")
        
        # 컨텍스트 데이터 검증
        validation = manager.validate_context_data(question_type, context_data)
        print(f"검증 결과: {validation['is_valid']}")
        if validation['warnings']:
            print(f"경고: {validation['warnings']}")


if __name__ == "__main__":
    test_prompt_template_manager()
