#!/usr/bin/env python3
"""
예시 데이터베이스
법률 질문별 구체적이고 실용적인 예시를 제공합니다.
"""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class LegalExample:
    """법률 예시 데이터 클래스"""
    situation: str
    description: str
    analysis: str
    legal_basis: str
    practical_tips: List[str]

class ExampleDatabase:
    """예시 데이터베이스"""
    
    def __init__(self, data_file: str = "data/legal_examples.json"):
        self.data_file = data_file
        self.examples = self._load_examples()
    
    def _load_examples(self) -> Dict[str, List[LegalExample]]:
        """예시 데이터 로드"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    examples = {}
                    for key, example_list in data.items():
                        examples[key] = [
                            LegalExample(**example) for example in example_list
                        ]
                    return examples
        except Exception as e:
            print(f"예시 데이터 로드 실패: {e}")
        
        return self._get_default_examples()
    
    def _get_default_examples(self) -> Dict[str, List[LegalExample]]:
        """기본 예시 데이터 반환"""
        return {
            "민법_750조": [
                LegalExample(
                    situation="교통사고",
                    description="A가 신호를 위반하여 B의 차량과 충돌한 경우",
                    analysis="A의 과실(신호위반) + 위법행위(교통법규 위반) + B의 손해(차량 파손) + 인과관계(직접적 충돌) = 손해배상 책임",
                    legal_basis="민법 제750조 불법행위",
                    practical_tips=["사고 현장 사진 촬영", "보험사 신고", "병원 진료 기록 보관"]
                ),
                LegalExample(
                    situation="상품 결함",
                    description="제조사의 과실로 인한 불량 제품으로 소비자가 피해를 입은 경우",
                    analysis="제조사의 과실(품질관리 소홀) + 위법행위(제품안전법 위반) + 소비자의 손해(신체상해) + 인과관계(제품 결함으로 인한 사고) = 손해배상 책임",
                    legal_basis="민법 제750조, 제품안전법",
                    practical_tips=["제품 보관", "의료비 영수증 수집", "제조사에 신고"]
                )
            ],
            "계약서_작성": [
                LegalExample(
                    situation="부동산 임대차계약",
                    description="월세 50만원, 보증금 1000만원의 원룸 임대차계약",
                    analysis="임대인/임차인 정보, 임대료 및 지급일, 임대기간, 보증금, 특약사항을 명확히 기재",
                    legal_basis="민법 제618조 임대차",
                    practical_tips=["등기부등본 확인", "실제 방문 확인", "특약사항 명시"]
                )
            ],
            "부동산_매매": [
                LegalExample(
                    situation="아파트 매매",
                    description="3억원 아파트 매매 (계약금 3000만원, 중도금 1억원, 잔금 1억7천만원)",
                    analysis="매물 확인 → 계약서 작성 → 계약금 지급 → 중도금 지급 → 잔금 지급 및 등기",
                    legal_basis="민법 제565조 해약금",
                    practical_tips=["등기부등본 확인", "실제 방문", "주변 환경 조사", "중개사 수수료 확인"]
                )
            ],
            "이혼_소송": [
                LegalExample(
                    situation="협의이혼 실패 후 소송",
                    description="배우자의 부정행위로 인한 이혼 소송",
                    analysis="소장 작성 → 법원 제출 → 답변서 제출 → 조정 → 변론 → 판결",
                    legal_basis="민법 제840조 재판상 이혼",
                    practical_tips=["증거 수집", "변호사 선임", "자녀 양육 계획 수립"]
                )
            ],
            "손해배상_청구": [
                LegalExample(
                    situation="교통사고 손해배상",
                    description="상대방 과실로 인한 교통사고 피해",
                    analysis="사고 현장 확인 → 보험사 신고 → 병원 치료 → 손해 산정 → 협의 또는 소송",
                    legal_basis="민법 제750조 불법행위",
                    practical_tips=["사고 현장 사진", "병원 진료 기록", "수입 증명서류"]
                )
            ]
        }
    
    def get_examples(self, topic: str, count: int = 2) -> List[LegalExample]:
        """주제별 예시 반환"""
        return self.examples.get(topic, [])[:count]
    
    def get_example_by_situation(self, topic: str, situation: str) -> Optional[LegalExample]:
        """상황별 특정 예시 반환"""
        examples = self.examples.get(topic, [])
        for example in examples:
            if situation in example.situation:
                return example
        return None
    
    def add_example(self, topic: str, example: LegalExample) -> bool:
        """새로운 예시 추가"""
        try:
            if topic not in self.examples:
                self.examples[topic] = []
            self.examples[topic].append(example)
            return self._save_examples()
        except Exception as e:
            print(f"예시 추가 실패: {e}")
            return False
    
    def _save_examples(self) -> bool:
        """예시 데이터 저장"""
        try:
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            with open(self.data_file, 'w', encoding='utf-8') as f:
                # LegalExample 객체를 딕셔너리로 변환
                data = {}
                for key, example_list in self.examples.items():
                    data[key] = [
                        {
                            'situation': ex.situation,
                            'description': ex.description,
                            'analysis': ex.analysis,
                            'legal_basis': ex.legal_basis,
                            'practical_tips': ex.practical_tips
                        } for ex in example_list
                    ]
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"예시 데이터 저장 실패: {e}")
            return False

class DynamicExampleGenerator:
    """동적 예시 생성기"""
    
    def __init__(self):
        self.example_templates = {
            "법률조문": "예를 들어, {situation}의 경우 {analysis}가 됩니다.",
            "절차안내": "구체적으로 {step1} → {step2} → {step3} 순서로 진행됩니다.",
            "계약서": "실제로는 '{example_clause}'와 같이 명확하게 작성합니다.",
            "손해배상": "이런 경우 {damages}에 대한 손해배상을 청구할 수 있습니다."
        }
    
    def generate_example(self, topic: str, context: str, question_type: str = "일반") -> str:
        """상황에 맞는 예시 동적 생성"""
        try:
            from .gemini_client import GeminiClient
            gemini_client = GeminiClient()
            
            example_prompt = f"""
다음 주제에 대한 구체적이고 실용적인 예시를 생성해주세요:

주제: {topic}
상황: {context}
질문 유형: {question_type}

요구사항:
1. 실제 발생할 수 있는 구체적인 상황
2. 구체적인 숫자나 명칭 포함 (예: "100만원", "3개월", "서울시 강남구")
3. 단계별 설명 (필요한 경우)
4. 이해하기 쉬운 언어 사용
5. 실용적인 조언 포함

예시 형식:
- 상황: [구체적인 상황 설명]
- 분석: [법적 분석 또는 절차 설명]
- 실무 팁: [실용적인 조언]

예시:"""
            
            response = gemini_client.generate(example_prompt)
            return response.response
            
        except Exception as e:
            print(f"동적 예시 생성 실패: {e}")
            return self._get_fallback_example(topic, context)
    
    def _get_fallback_example(self, topic: str, context: str) -> str:
        """폴백 예시 생성"""
        fallback_examples = {
            "민법": f"예를 들어, {context}와 같은 상황에서는 민법의 관련 조항이 적용됩니다.",
            "계약서": f"실제로는 '{context}'와 같이 명확하게 작성하는 것이 중요합니다.",
            "부동산": f"부동산 거래에서는 {context} 등의 절차를 거치게 됩니다.",
            "이혼": f"이혼 절차에서는 {context} 등의 단계를 거치게 됩니다.",
            "손해배상": f"손해배상 청구 시에는 {context} 등의 요건이 필요합니다."
        }
        
        for key, example in fallback_examples.items():
            if key in topic.lower():
                return example
        
        return f"예를 들어, {context}와 같은 경우를 고려해볼 수 있습니다."
    
    def enhance_existing_example(self, example: str, topic: str) -> str:
        """기존 예시를 더 구체적으로 개선"""
        try:
            from .gemini_client import GeminiClient
            gemini_client = GeminiClient()
            
            enhancement_prompt = f"""
다음 예시를 더 구체적이고 실용적으로 개선해주세요:

주제: {topic}
기존 예시: {example}

개선 요구사항:
1. 구체적인 숫자나 명칭 추가
2. 실제 상황에 적용 가능한 실무 팁 추가
3. 단계별 설명 강화
4. 이해하기 쉬운 언어로 개선

개선된 예시:"""
            
            response = gemini_client.generate(enhancement_prompt)
            return response.response
            
        except Exception as e:
            print(f"예시 개선 실패: {e}")
            return example

# 전역 인스턴스
example_database = ExampleDatabase()
dynamic_generator = DynamicExampleGenerator()