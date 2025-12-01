# -*- coding: utf-8 -*-
"""
Document Analysis Processor
문서 분석 로직을 처리하는 프로세서
"""

import logging
import time
from typing import Any, Dict, List, Optional

try:
    from lawfirm_langgraph.core.processing.extractors import DocumentExtractor
except ImportError:
    from core.processing.extractors import DocumentExtractor
try:
    from lawfirm_langgraph.core.processing.parsers.response_parsers import DocumentParser
except ImportError:
    from core.processing.parsers.response_parsers import DocumentParser
try:
    from lawfirm_langgraph.core.workflow.builders.prompt_chain_executor import PromptChainExecutor
except ImportError:
    from core.workflow.builders.prompt_chain_executor import PromptChainExecutor
try:
    from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
except ImportError:
    from core.workflow.state.state_definitions import LegalWorkflowState
try:
    from lawfirm_langgraph.core.workflow.state.state_utils import (
        MAX_DOCUMENT_CONTENT_LENGTH,
        MAX_RETRIEVED_DOCS,
        prune_retrieved_docs,
    )
except ImportError:
    from core.workflow.state.state_utils import (
        MAX_DOCUMENT_CONTENT_LENGTH,
        MAX_RETRIEVED_DOCS,
        prune_retrieved_docs,
    )


class DocumentAnalysisProcessor:
    """문서 분석 프로세서"""

    def __init__(self, llm, logger, document_processor=None, llm_fast=None):
        self.llm = llm
        self.llm_fast = llm_fast
        self.logger = logger
        self.document_processor = document_processor

    def analyze_document(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """업로드된 문서 분석"""
        try:
            start_time = time.time()

            doc_text = state.get("document_analysis", {}).get("raw_text") if isinstance(state.get("document_analysis"), dict) else None

            if not doc_text:
                if state.get("document_analysis") and isinstance(state["document_analysis"], str):
                    doc_text = state["document_analysis"]
                else:
                    self.logger.info("No document provided for analysis, skipping")
                    return state

            doc_type = self.detect_document_type(doc_text)
            state["document_type"] = doc_type

            analysis_result = self.analyze_legal_document_with_chain(doc_text, doc_type)

            if isinstance(state["document_analysis"], dict):
                state["document_analysis"].update({
                    "document_type": doc_type,
                    "summary": analysis_result.get("summary", ""),
                    "analysis_time": time.time() - start_time
                })
            else:
                state["document_analysis"] = {
                    "document_type": doc_type,
                    "summary": analysis_result.get("summary", ""),
                    "analysis_time": time.time() - start_time
                }

            state["key_clauses"] = analysis_result.get("key_clauses", [])
            state["potential_issues"] = analysis_result.get("issues", [])

            doc_summary = self.create_document_summary(analysis_result)
            summary_doc = {
                "content": doc_summary[:MAX_DOCUMENT_CONTENT_LENGTH],
                "source": "Uploaded Document Analysis",
                "type": "document_analysis",
                "relevance_score": 1.0,
                "is_summarized": True
            }

            if len(state["retrieved_docs"]) >= MAX_RETRIEVED_DOCS:
                state["retrieved_docs"] = prune_retrieved_docs(
                    state["retrieved_docs"],
                    max_items=MAX_RETRIEVED_DOCS - 1,
                    max_content_per_doc=MAX_DOCUMENT_CONTENT_LENGTH
                )

            state["retrieved_docs"].insert(0, summary_doc)

        except Exception as e:
            self.logger.error(f"Error in document analysis: {e}", exc_info=True)
            state["errors"] = state.get("errors", []) + [str(e)]

        return state

    def detect_document_type(self, text: str) -> str:
        """문서 유형 감지"""
        type_keywords = {
            "contract": ["계약서", "계약", "갑", "을", "본 계약"],
            "complaint": ["고소장", "피고소인", "고소인", "고소취지"],
            "agreement": ["합의서", "합의", "쌍방"],
            "power_of_attorney": ["위임장", "위임인", "수임인"]
        }

        text_lower = text.lower()
        for doc_type, keywords in type_keywords.items():
            if any(k in text_lower for k in keywords):
                return doc_type

        return "general_legal_document"

    def analyze_legal_document(self, text: str, doc_type: str) -> Dict[str, Any]:
        """법률 문서 분석"""
        analysis = {
            "document_type": doc_type,
            "key_clauses": [],
            "issues": [],
            "summary": "",
            "recommendations": []
        }

        if doc_type == "contract":
            analysis["key_clauses"] = DocumentExtractor.extract_contract_clauses(text)
            analysis["issues"] = self.identify_contract_issues(text, analysis["key_clauses"])
        elif doc_type == "complaint":
            analysis["key_clauses"] = DocumentExtractor.extract_complaint_elements(text)
            analysis["issues"] = self.identify_complaint_issues(text)

        analysis["summary"] = self.generate_document_summary(text, doc_type, analysis)

        return analysis

    def identify_contract_issues(self, text: str, clauses: List[Dict]) -> List[Dict[str, Any]]:
        """계약서 잠재 문제점 식별"""
        issues = []

        required_clauses = ["payment", "period", "termination"]
        found_types = set()
        for c in clauses:
            clause_type = c.get("type")
            if clause_type is not None and isinstance(clause_type, (str, int, float, tuple)):
                found_types.add(clause_type)
            elif clause_type is not None:
                try:
                    found_types.add(str(clause_type))
                except Exception:
                    pass

        for req_type in required_clauses:
            if req_type not in found_types:
                issues.append({
                    "severity": "high",
                    "type": "missing_clause",
                    "description": f"필수 조항 누락: {req_type}",
                    "recommendation": f"{req_type} 조항을 추가하십시오"
                })

        vague_terms = ["기타", "등등", "적절한", "합당한"]
        for term in vague_terms:
            if term in text:
                issues.append({
                    "severity": "medium",
                    "type": "vague_term",
                    "description": f"불명확한 용어 사용: {term}",
                    "recommendation": "구체적인 용어로 대체하십시오"
                })

        return issues[:5]

    def identify_complaint_issues(self, text: str) -> List[Dict[str, Any]]:
        """고소장 문제점 식별"""
        issues = []

        required_elements = ["피고소인", "사실관계", "청구"]
        for elem in required_elements:
            if elem not in text:
                issues.append({
                    "severity": "high",
                    "type": "missing_element",
                    "description": f"필수 요소 누락: {elem}",
                    "recommendation": f"{elem} 정보를 추가하십시오"
                })

        return issues

    def analyze_legal_document_with_chain(self, text: str, doc_type: str) -> Dict[str, Any]:
        """
        Prompt Chaining을 사용한 법률 문서 분석 (다단계 체인)

        Step 1: 문서 유형 확인 (키워드 기반 결과 검증)
        Step 2: 주요 조항 추출 (문서 유형 기반)
        Step 3: 문제점 식별 (조항 기반)
        Step 4: 요약 생성 (조항 + 문제점 기반)
        Step 5: 개선 권고 생성 (문제점 기반)

        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            chain_executor = PromptChainExecutor(self.llm, self.logger)

            chain_steps = []

            def build_document_type_verification_prompt(prev_output, initial_input):
                doc_text = initial_input.get("text") if isinstance(initial_input, dict) else text[:2000]
                detected_type = initial_input.get("doc_type") if isinstance(initial_input, dict) else doc_type

                return f"""다음 문서의 유형을 확인하고 검증해주세요.

문서 내용 (일부):
{doc_text}

키워드 기반 감지 결과: {detected_type}

다음 문서 유형 중 하나로 확인해주세요:
- contract (계약서): 계약서, 갑/을, 계약 조건 등
- complaint (고소장): 고소장, 피고소인, 고소인 등
- agreement (합의서): 합의서, 합의, 쌍방 합의 등
- power_of_attorney (위임장): 위임장, 위임인, 수임인 등
- general_legal_document (일반 법률 문서): 위에 해당하지 않는 경우

다음 형식으로 응답해주세요:
{{
    "document_type": "contract" | "complaint" | "agreement" | "power_of_attorney" | "general_legal_document",
    "confidence": 0.0-1.0,
    "reasoning": "판단 근거 (한국어)"
}}
"""

            chain_steps.append({
                "name": "document_type_verification",
                "prompt_builder": build_document_type_verification_prompt,
                "input_extractor": lambda prev: prev,
                "output_parser": lambda response, prev: DocumentParser.parse_document_type_response(response),
                "validator": lambda output: output and isinstance(output, dict) and "document_type" in output,
                "required": True
            })

            def build_clause_extraction_prompt(prev_output, initial_input):
                if not isinstance(prev_output, dict):
                    prev_output = {}

                verified_doc_type = prev_output.get("document_type", doc_type)
                doc_text = initial_input.get("text") if isinstance(initial_input, dict) else text[:3000]

                if verified_doc_type == "contract":
                    return f"""다음 계약서 문서에서 주요 조항을 추출해주세요.

문서 내용:
{doc_text[:3000]}

다음 유형의 조항을 찾아주세요:
- payment (대금/지급): 대금, 금액, 지급, 결제 관련 조항
- period (기간/기한): 기간, 기한, 만료 관련 조항
- termination (해지/해제): 해지, 해제, 종료 관련 조항
- liability (책임): 책임, 손해배상, 위약금 관련 조항
- confidentiality (비밀/기밀): 비밀, 기밀, 보안 관련 조항

다음 형식으로 응답해주세요:
{{
    "key_clauses": [
        {{
            "type": "payment",
            "text": "제1조 대금은...",
            "article_number": "제1조"
        }},
        ...
    ],
    "clause_count": 5
}}
"""
                elif verified_doc_type == "complaint":
                    return f"""다음 고소장 문서에서 주요 요소를 추출해주세요.

문서 내용:
{doc_text[:3000]}

다음 요소를 찾아주세요:
- parties (당사자): 피고소인, 고소인, 피해자, 가해자 등
- facts (사실관계): 사실관계, 경위, 내용 등
- claims (청구사항): 청구, 요구, 주장 등

다음 형식으로 응답해주세요:
{{
    "key_clauses": [
        {{
            "type": "parties",
            "text": "피고소인: ...",
            "found": true
        }},
        ...
    ],
    "clause_count": 3
}}
"""
                else:
                    return f"""다음 문서에서 주요 내용을 추출해주세요.

문서 내용:
{doc_text[:3000]}

다음 형식으로 응답해주세요:
{{
    "key_clauses": [
        {{
            "type": "general",
            "text": "...",
            "summary": "..."
        }},
        ...
    ],
    "clause_count": 0
}}
"""

            chain_steps.append({
                "name": "clause_extraction",
                "prompt_builder": build_clause_extraction_prompt,
                "input_extractor": lambda prev: prev,
                "output_parser": lambda response, prev: self.parse_clause_extraction_response(response),
                "validator": lambda output: output and isinstance(output, dict) and "key_clauses" in output,
                "required": True
            })

            def build_issue_identification_prompt(prev_output, initial_input):
                if not isinstance(prev_output, dict):
                    prev_output = {}

                verified_doc_type = prev_output.get("document_type", doc_type)
                key_clauses = prev_output.get("key_clauses", [])
                doc_text = initial_input.get("text") if isinstance(initial_input, dict) else text[:3000]

                if verified_doc_type == "contract":
                    clauses_text = "\n".join([f"- {c.get('type', 'unknown')}: {c.get('text', '')[:100]}" for c in key_clauses[:5]])
                    return f"""다음 계약서의 조항을 검토하여 잠재적인 문제점을 식별해주세요.

문서 내용:
{doc_text[:3000]}

추출된 주요 조항:
{clauses_text}

다음 형식으로 응답해주세요:
{{
    "issues": [
        {{
            "severity": "high" | "medium" | "low",
            "type": "missing_clause" | "vague_term" | "unfair_clause" | "legal_risk",
            "description": "문제점 설명",
            "recommendation": "개선 권고사항"
        }},
        ...
    ],
    "issue_count": 3
}}
"""
                elif verified_doc_type == "complaint":
                    return f"""다음 고소장을 검토하여 잠재적인 문제점을 식별해주세요.

문서 내용:
{doc_text[:3000]}

다음 형식으로 응답해주세요:
{{
    "issues": [
        {{
            "severity": "high" | "medium" | "low",
            "type": "missing_element" | "insufficient_evidence" | "legal_issue",
            "description": "문제점 설명",
            "recommendation": "개선 권고사항"
        }},
        ...
    ],
    "issue_count": 3
}}
"""
                else:
                    return f"""다음 문서를 검토하여 잠재적인 문제점을 식별해주세요.

문서 내용:
{doc_text[:3000]}

다음 형식으로 응답해주세요:
{{
    "issues": [
        {{
            "severity": "high" | "medium" | "low",
            "type": "general_issue",
            "description": "문제점 설명",
            "recommendation": "개선 권고사항"
        }},
        ...
    ],
    "issue_count": 0
}}
"""

            chain_steps.append({
                "name": "issue_identification",
                "prompt_builder": build_issue_identification_prompt,
                "input_extractor": lambda prev: prev,
                "output_parser": lambda response, prev: self.parse_issue_identification_response_with_context(response, prev),
                "validator": lambda output: output and isinstance(output, dict),
                "required": True
            })

            def build_summary_generation_prompt(prev_output, initial_input):
                if not isinstance(prev_output, dict):
                    prev_output = {}

                verified_doc_type = prev_output.get("document_type", doc_type)
                key_clauses = prev_output.get("key_clauses", [])
                issues = prev_output.get("issues", [])

                clauses_summary = "\n".join([f"- {c.get('type', 'unknown')}: {c.get('text', '')[:80]}" for c in key_clauses[:3]])
                issues_summary = "\n".join([f"- [{i.get('severity', 'unknown')}] {i.get('description', '')[:80]}" for i in issues[:3]])

                return f"""다음 문서 분석 결과를 바탕으로 요약을 생성해주세요.

문서 유형: {verified_doc_type}

주요 조항:
{clauses_summary if clauses_summary else "없음"}

잠재 문제점:
{issues_summary if issues_summary else "없음"}

다음 형식으로 응답해주세요:
{{
    "summary": "문서 요약 (200자 이내)",
    "key_points": ["핵심 포인트 1", "핵심 포인트 2", ...]
}}
"""

            def extract_summary_input(prev_output):
                if not isinstance(prev_output, dict):
                    prev_output = {}
                return prev_output

            chain_steps.append({
                "name": "summary_generation",
                "prompt_builder": build_summary_generation_prompt,
                "input_extractor": extract_summary_input,
                "output_parser": lambda response, prev: self.parse_summary_response(response),
                "validator": lambda output: output and isinstance(output, dict) and "summary" in output,
                "required": True
            })

            initial_input_dict = {"text": text, "doc_type": doc_type}
            chain_result = chain_executor.execute_chain(
                chain_steps=chain_steps,
                initial_input=initial_input_dict,
                max_iterations=5
            )

            if chain_result and isinstance(chain_result, dict):
                final_output = chain_result.get("final_output", {})
                if isinstance(final_output, dict):
                    return {
                        "document_type": final_output.get("document_type", doc_type),
                        "key_clauses": final_output.get("key_clauses", []),
                        "issues": final_output.get("issues", []),
                        "summary": final_output.get("summary", ""),
                        "recommendations": final_output.get("recommendations", [])
                    }

            return self.analyze_legal_document(text, doc_type)

        except Exception as e:
            self.logger.error(f"Error in analyze_legal_document_with_chain: {e}", exc_info=True)
            return self.analyze_legal_document(text, doc_type)

    def generate_document_summary(self, text: str, doc_type: str, analysis: Dict[str, Any]) -> str:
        """문서 요약 생성"""
        try:
            key_clauses = analysis.get("key_clauses", [])
            issues = analysis.get("issues", [])

            if key_clauses or issues:
                summary_prompt = f"""다음 {doc_type} 문서 분석 결과를 바탕으로 요약을 생성해주세요.

주요 조항:
{chr(10).join([f"- {c.get('type', 'unknown')}: {c.get('text', '')[:100]}" for c in key_clauses[:3]])}

잠재 문제점:
{chr(10).join([f"- [{i.get('severity', 'unknown')}] {i.get('description', '')[:100]}" for i in issues[:3]])}

200자 이내로 요약해주세요."""
                
                llm = getattr(self, 'llm_fast', None) or self.llm
                response = llm.invoke(summary_prompt)
                if hasattr(response, 'content'):
                    return response.content
                return str(response)
            else:
                return self.generate_document_summary_fallback(text, doc_type, key_clauses, issues)

        except Exception as e:
            self.logger.warning(f"Failed to generate document summary: {e}")
            return self.generate_document_summary_fallback(text, doc_type, analysis.get("key_clauses", []), analysis.get("issues", []))

    def generate_document_summary_fallback(self, text: str, doc_type: str, key_clauses: List[Dict], issues: List[Dict]) -> str:
        """문서 요약 생성 (폴백)"""
        summary_parts = [f"{doc_type} 문서 분석 결과"]

        if key_clauses:
            summary_parts.append(f"주요 조항 {len(key_clauses)}개 발견")
        if issues:
            summary_parts.append(f"잠재 문제점 {len(issues)}개 발견")

        return " ".join(summary_parts)

    def create_document_summary(self, analysis: Dict[str, Any]) -> str:
        """문서 분석 요약 생성"""
        summary_parts = [f"## 업로드 문서 분석 ({analysis['document_type']})"]

        if analysis.get("key_clauses"):
            summary_parts.append("\n### 주요 조항")
            for clause in analysis["key_clauses"][:3]:
                summary_parts.append(f"- {clause['type']}: {clause['text'][:100]}...")

        if analysis.get("issues"):
            summary_parts.append("\n### 잠재 문제점")
            for issue in analysis["issues"]:
                summary_parts.append(f"- [{issue['severity']}] {issue['description']}")

        return "\n".join(summary_parts)

    def parse_clause_extraction_response(self, response: str) -> Dict[str, Any]:
        """조항 추출 응답 파싱"""
        try:
            return DocumentParser.parse_clause_extraction_response(response)
        except Exception as e:
            self.logger.warning(f"Failed to parse clause extraction response: {e}")
            return {
                "key_clauses": [],
                "clause_count": 0
            }

    def parse_issue_identification_response(self, response: str) -> Optional[Dict[str, Any]]:
        """문제점 식별 응답 파싱"""
        try:
            import json
            import re

            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                if "issues" in parsed:
                    return parsed

            return None
        except Exception as e:
            self.logger.warning(f"Failed to parse issue identification response: {e}")
            return None

    def parse_improvement_recommendations_response(self, response: str) -> Optional[Dict[str, Any]]:
        """개선 권고 응답 파싱"""
        try:
            import json
            import re

            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                if "recommendations" in parsed:
                    return parsed

            return None
        except Exception as e:
            self.logger.warning(f"Failed to parse improvement recommendations response: {e}")
            return None

    def parse_issue_identification_response_with_context(self, response: str, prev_output: Any) -> Optional[Dict[str, Any]]:
        """문제점 식별 응답 파싱 (이전 단계 출력 통합)"""
        try:
            import json
            import re

            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                if "issues" in parsed:
                    if isinstance(prev_output, dict):
                        parsed["key_clauses"] = prev_output.get("key_clauses", [])
                        parsed["document_type"] = prev_output.get("document_type", "")
                    return parsed

            return None
        except Exception as e:
            self.logger.warning(f"Failed to parse issue identification response: {e}")
            return None

    def parse_summary_response(self, response: str) -> Dict[str, Any]:
        """요약 응답 파싱"""
        try:
            import json
            import re

            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return parsed

            return {"summary": response[:200], "key_points": []}
        except Exception as e:
            self.logger.warning(f"Failed to parse summary response: {e}")
            return {"summary": response[:200] if response else "", "key_points": []}

