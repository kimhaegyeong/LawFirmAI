# -*- coding: utf-8 -*-
"""
질문 유형별 프롬프트 템플릿
각 질문 유형에 최적화된 프롬프트 생성 시스템
"""

import logging
from typing import Dict, Any, Optional
from .question_classifier import QuestionType

logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """프롬프트 템플릿 관리자"""
    
    def __init__(self):
        """프롬프트 템플릿 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 질문 유형별 프롬프트 템플릿
        self.templates = {
            QuestionType.PRECEDENT_SEARCH: {
                "template": """당신은 판례 전문가입니다. 

관련 판례:
{precedent_list}

위 판례를 바탕으로 다음과 같이 답변하세요:
1. 가장 유사한 판례 3개 소개 (사건번호, 판결요지)
2. 해당 사안에의 적용 가능성
3. 실무적 시사점

답변은 전문적이면서도 이해하기 쉽게 작성해주세요.""",
                "context_keys": ["precedent_list"],
                "max_context_length": 3000
            },
            
            QuestionType.LAW_INQUIRY: {
                "template": """당신은 법률 해설 전문가입니다.

관련 법률:
{law_articles}

위 법률을 다음 순서로 설명하세요:
1. 법률의 목적 및 취지
2. 주요 내용을 쉬운 말로 풀이
3. 실제 적용 예시
4. 주의사항

답변은 정확하고 이해하기 쉽게 작성해주세요.""",
                "context_keys": ["law_articles"],
                "max_context_length": 2000
            },
            
            QuestionType.LEGAL_ADVICE: {
                "template": """당신은 법률 상담 전문가입니다.

관련 법률 및 판례:
{context}

다음 구조로 조언하세요:
1. 상황 정리
2. 적용 가능한 법률 (조문 명시)
3. 관련 판례 (핵심 판결요지)
4. 권리 구제 방법 (단계별)
5. 필요한 증거 자료

주의: 이는 일반적인 법률 정보 제공이며, 구체적인 사안은 변호사와 상담하시기 바랍니다.""",
                "context_keys": ["context"],
                "max_context_length": 4000
            },
            
            QuestionType.PROCEDURE_GUIDE: {
                "template": """당신은 법률 절차 안내 전문가입니다.

관련 절차 정보:
{procedure_info}

다음 순서로 안내하세요:
1. 절차 개요
2. 필요한 서류 및 준비사항
3. 신청 방법 및 절차
4. 처리 기간 및 비용
5. 주의사항 및 팁

실용적이고 구체적인 안내를 제공해주세요.""",
                "context_keys": ["procedure_info"],
                "max_context_length": 2500
            },
            
            QuestionType.TERM_EXPLANATION: {
                "template": """당신은 법률 용어 해설 전문가입니다.

용어 정보:
{term_info}

다음 순서로 설명하세요:
1. 용어의 정의
2. 법적 근거 (관련 조문)
3. 실제 적용 사례
4. 관련 용어와의 차이점
5. 실무에서의 중요성

정확하고 명확한 설명을 제공해주세요.""",
                "context_keys": ["term_info"],
                "max_context_length": 1500
            },
            
            QuestionType.GENERAL_QUESTION: {
                "template": """당신은 법률 정보 제공 전문가입니다.

관련 정보:
{general_context}

다음 원칙에 따라 답변하세요:
1. 질문에 대한 정확한 답변 제공
2. 관련 법률 및 판례 인용
3. 이해하기 쉬운 설명
4. 필요한 경우 추가 정보 안내

전문적이면서도 친근한 톤으로 답변해주세요.""",
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
                     user_query: Optional[str] = None) -> str:
        """
        프롬프트 템플릿을 실제 데이터로 포맷팅
        
        Args:
            question_type: 질문 유형
            context_data: 컨텍스트 데이터
            user_query: 사용자 질문 (선택사항)
            
        Returns:
            str: 포맷팅된 프롬프트
        """
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
            self.logger.error(f"Error formatting prompt: {e}")
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
