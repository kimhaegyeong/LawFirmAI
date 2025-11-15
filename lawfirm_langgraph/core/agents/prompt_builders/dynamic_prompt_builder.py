# -*- coding: utf-8 -*-
"""
동적 프롬프트 빌더
검색 결과에 맞게 프롬프트를 동적으로 생성
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class DynamicPromptBuilder:
    """검색 결과 기반 동적 프롬프트 빌더"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_document_types(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """문서 타입별 개수 분석"""
        doc_types = {}
        for doc in documents:
            if not isinstance(doc, dict):
                continue
            
            doc_type = (
                doc.get("type") or 
                doc.get("source_type") or 
                doc.get("metadata", {}).get("type") or
                doc.get("metadata", {}).get("source_type") or
                "unknown"
            )
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        return doc_types
    
    def build_citation_guidance(
        self, 
        doc_types: Dict[str, int],
        document_count: int
    ) -> str:
        """검색 결과 기반 Citation 지침 생성"""
        has_statutes = doc_types.get("statute_article", 0) > 0
        has_cases = doc_types.get("case_paragraph", 0) > 0
        has_decisions = doc_types.get("decision_paragraph", 0) > 0
        has_interpretations = doc_types.get("interpretation_paragraph", 0) > 0
        
        guidance_parts = []
        
        # 법령 조문 인용 지침
        if has_statutes:
            statute_count = doc_types.get("statute_article", 0)
            if statute_count >= 2:
                guidance_parts.append("법령 조문을 최소 2개 이상 반드시 인용하세요.")
            else:
                guidance_parts.append("법령 조문을 가능한 경우 인용하세요.")
        elif document_count > 0:
            guidance_parts.append("법령 조문이 없으므로 일반적인 법률 원칙을 중심으로 답변하세요.")
        
        # 판례 인용 지침
        if has_cases:
            case_count = doc_types.get("case_paragraph", 0)
            if case_count >= 2:
                guidance_parts.append("판례를 인용하되, 법적 원칙 중심으로 설명하세요.")
            else:
                guidance_parts.append("판례를 참고하여 일반적인 법적 원칙을 설명하세요.")
        
        # 결정례 인용 지침
        if has_decisions:
            guidance_parts.append("결정례를 참고하세요.")
        
        # 해석례 인용 지침
        if has_interpretations:
            guidance_parts.append("법령 해석례를 활용하세요.")
        
        # 문서 수 기반 인용 요구사항
        if document_count >= 3:
            guidance_parts.append("최소 2개 이상의 문서를 인용하세요.")
        elif document_count >= 1:
            guidance_parts.append("제공된 문서를 최대한 활용하세요.")
        else:
            guidance_parts.append("일반적인 법률 지식을 바탕으로 답변하세요.")
        
        return "\n".join(f"- {part}" for part in guidance_parts)
    
    def build_document_type_guidance(self, doc_types: Dict[str, int]) -> str:
        """문서 타입별 활용 지침 생성"""
        guidance = []
        
        if doc_types.get("statute_article", 0) > 0:
            guidance.append("법령 조문을 우선적으로 활용하세요.")
        
        if doc_types.get("case_paragraph", 0) > 0:
            guidance.append("판례는 법적 원칙을 추출하여 일반적으로 설명하세요.")
        
        if doc_types.get("decision_paragraph", 0) > 0:
            guidance.append("결정례를 참고하여 답변을 구성하세요.")
        
        if doc_types.get("interpretation_paragraph", 0) > 0:
            guidance.append("법령 해석례를 활용하세요.")
        
        if not guidance:
            guidance.append("제공된 문서를 최대한 활용하세요.")
        
        return "\n".join(f"- {part}" for part in guidance)
    
    def build_simplified_prompt_section(
        self,
        documents: List[Dict[str, Any]],
        document_count: int
    ) -> str:
        """간소화된 프롬프트 섹션 생성"""
        if document_count == 0:
            return """
## 검색 결과
현재 관련 법률 문서를 찾지 못했습니다. 일반적인 법률 정보를 제공하되, 한계를 명시하세요.
"""
        
        doc_types = self.analyze_document_types(documents)
        citation_guidance = self.build_citation_guidance(doc_types, document_count)
        type_guidance = self.build_document_type_guidance(doc_types)
        
        return f"""
## 검색 결과 활용 지침

### 문서 타입 분포
{self._format_doc_types_summary(doc_types)}

### Citation 요구사항
{citation_guidance}

### 문서 타입별 활용 방법
{type_guidance}
"""
    
    def _format_doc_types_summary(self, doc_types: Dict[str, int]) -> str:
        """문서 타입 분포 요약 포맷팅"""
        type_names = {
            "statute_article": "법령 조문",
            "case_paragraph": "판례",
            "decision_paragraph": "결정례",
            "interpretation_paragraph": "해석례"
        }
        
        parts = []
        for doc_type, count in doc_types.items():
            type_name = type_names.get(doc_type, doc_type)
            parts.append(f"- {type_name}: {count}개")
        
        if not parts:
            return "- 문서 타입 정보 없음"
        
        return "\n".join(parts)

