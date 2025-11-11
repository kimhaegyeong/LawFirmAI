"""
Sources 추출기
LangGraph state와 메시지 metadata에서 sources를 추출합니다.
"""
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import logging
import json
import re

logger = logging.getLogger(__name__)


class SourcesExtractorConstants:
    """SourcesExtractor 상수"""
    INVALID_SOURCES = {"semantic", "keyword", "unknown", "fts", "vector", "search", "text2sql", ""}
    MIN_SOURCE_LENGTH = 2
    MIN_SOURCE_LENGTH_TYPED = 1
    MAX_SOURCES_DETAIL_FOR_PROMPT = 5
    MAX_RELATED_QUESTIONS = 5
    MIN_QUESTION_LENGTH = 5
    LLM_MAX_RETRIES = 2
    MIN_ANSWER_LENGTH_FOR_ENHANCEMENT = 20


@dataclass
class ExtractionResult:
    """추출 결과 데이터 클래스"""
    sources: List[str]
    legal_references: List[str]
    sources_detail: List[Dict[str, Any]]
    related_questions: List[str]
    
    @classmethod
    def empty(cls) -> 'ExtractionResult':
        """빈 결과 생성"""
        return cls(
            sources=[],
            legal_references=[],
            sources_detail=[],
            related_questions=[]
        )
    
    def to_dict(self) -> Dict[str, List[Any]]:
        """딕셔너리로 변환"""
        return {
            "sources": self.sources,
            "legal_references": self.legal_references,
            "sources_detail": self.sources_detail,
            "related_questions": self.related_questions
        }


class SourceTypeProcessor:
    """Source 타입별 처리 로직"""
    
    @staticmethod
    def process_statute_article(doc: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[str]:
        """법령 조문 처리"""
        statute_name = (
            doc.get("statute_name") or
            doc.get("law_name") or
            metadata.get("statute_name") or
            metadata.get("law_name")
        )
        
        if not statute_name:
            return None
        
        article_no = (
            doc.get("article_no") or
            doc.get("article_number") or
            metadata.get("article_no") or
            metadata.get("article_number")
        )
        clause_no = doc.get("clause_no") or metadata.get("clause_no")
        item_no = doc.get("item_no") or metadata.get("item_no")
        
        source_parts = [statute_name]
        if article_no:
            source_parts.append(article_no)
        if clause_no:
            source_parts.append(f"제{clause_no}항")
        if item_no:
            source_parts.append(f"제{item_no}호")
        
        return " ".join(source_parts)
    
    @staticmethod
    def process_case_paragraph(doc: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[str]:
        """판례 처리"""
        court = doc.get("court") or metadata.get("court")
        casenames = doc.get("casenames") or metadata.get("casenames")
        doc_id = doc.get("doc_id") or metadata.get("doc_id") or metadata.get("case_id") or doc.get("id") or metadata.get("id")
        
        if not court and not casenames:
            court = metadata.get("court_name") or metadata.get("court_type")
            casenames = metadata.get("case_name") or metadata.get("title")
        
        if court or casenames or doc_id:
            source_parts = []
            if court:
                source_parts.append(court)
            if casenames:
                source_parts.append(casenames)
            if doc_id:
                source_parts.append(f"({doc_id})")
            if not court and not casenames and doc_id:
                source_parts.insert(0, "판례")
            return " ".join(source_parts) if source_parts else None
        
        return None
    
    @staticmethod
    def process_decision_paragraph(doc: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[str]:
        """결정례 처리"""
        org = doc.get("org") or metadata.get("org")
        doc_id = doc.get("doc_id") or metadata.get("doc_id") or metadata.get("decision_id") or doc.get("id") or metadata.get("id")
        
        if not org:
            org = metadata.get("org_name") or metadata.get("organization")
        
        if org or doc_id:
            source_parts = []
            if org:
                source_parts.append(org)
            if doc_id:
                source_parts.append(f"({doc_id})")
            if not org and doc_id:
                source_parts.insert(0, "결정례")
            return " ".join(source_parts) if source_parts else None
        
        return None
    
    @staticmethod
    def process_interpretation_paragraph(doc: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[str]:
        """해석례 처리"""
        org = doc.get("org") or metadata.get("org")
        title = doc.get("title") or metadata.get("title")
        
        if org or title:
            source_parts = []
            if org:
                source_parts.append(org)
            if title:
                source_parts.append(title)
            return " ".join(source_parts)
        
        return None
    
    @staticmethod
    def process_fallback(doc: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[str]:
        """Fallback 처리 (source_type이 없는 경우)"""
        source_raw = (
            doc.get("statute_name") or
            doc.get("law_name") or
            doc.get("source_name") or
            doc.get("source") or
            doc.get("title") or
            doc.get("document_id") or
            doc.get("name")
        )
        
        if source_raw and isinstance(source_raw, str):
            source_lower = source_raw.lower().strip()
            if source_lower not in SourcesExtractorConstants.INVALID_SOURCES and len(source_lower) >= SourcesExtractorConstants.MIN_SOURCE_LENGTH:
                return source_raw.strip()
        
        source_from_metadata = (
            metadata.get("statute_name") or
            metadata.get("statute_abbrv") or
            metadata.get("law_name") or
            metadata.get("court") or
            metadata.get("court_name") or
            metadata.get("org") or
            metadata.get("org_name") or
            metadata.get("title") or
            metadata.get("case_name")
        )
        
        if source_from_metadata:
            return source_from_metadata
        
        content = doc.get("content", "") or doc.get("text", "")
        if isinstance(content, str) and content:
            law_pattern = re.search(r'([가-힣]+법)\s*(?:제\d+조)?', content[:200])
            if law_pattern:
                return law_pattern.group(1)
        
        return None
    
    @staticmethod
    def validate_source(source: str, source_type: Optional[str] = None) -> bool:
        """Source 유효성 검증"""
        if not source or source == "Unknown":
            return False
        
        source_lower = source.lower().strip()
        if source_lower in SourcesExtractorConstants.INVALID_SOURCES:
            return False
        
        if source_type and source_type in ["statute_article", "case_paragraph", "decision_paragraph", "interpretation_paragraph"]:
            return len(source_lower) >= SourcesExtractorConstants.MIN_SOURCE_LENGTH_TYPED
        else:
            return len(source_lower) >= SourcesExtractorConstants.MIN_SOURCE_LENGTH


class RelatedQuestionsLLMGenerator:
    """연관질문 LLM 생성기"""
    
    def __init__(self, workflow_service):
        self.workflow_service = workflow_service
        self.logger = logging.getLogger(__name__)
    
    def get_llm_handler(self):
        """LLM 핸들러 가져오기"""
        try:
            if self.workflow_service:
                if hasattr(self.workflow_service, 'legal_workflow') and self.workflow_service.legal_workflow:
                    if hasattr(self.workflow_service.legal_workflow, 'llm'):
                        return type('Handler', (), {'llm': self.workflow_service.legal_workflow.llm})()
                
                if hasattr(self.workflow_service, 'conversation_flow_tracker') and self.workflow_service.conversation_flow_tracker:
                    tracker = self.workflow_service.conversation_flow_tracker
                    if hasattr(tracker, '_get_classification_handler'):
                        handler = tracker._get_classification_handler()
                        if handler and hasattr(handler, 'llm'):
                            return handler
        except Exception as e:
            self.logger.debug(f"[RelatedQuestionsLLMGenerator] Failed to get LLM handler: {e}")
        return None
    
    def extract_llm_response(self, response) -> str:
        """LLM 응답에서 텍스트 추출"""
        if hasattr(response, 'content'):
            return str(response.content)
        elif isinstance(response, str):
            return response
        elif hasattr(response, 'text'):
            return str(response.text)
        else:
            return str(response)
    
    def build_prompt(
        self,
        query: str,
        answer: str,
        sources_detail: List[Dict[str, Any]],
        legal_references: List[str]
    ) -> str:
        """프롬프트 구성"""
        statute_summary = []
        case_summary = []
        interpretation_summary = []
        
        for detail in sources_detail[:SourcesExtractorConstants.MAX_SOURCES_DETAIL_FOR_PROMPT]:
            if not isinstance(detail, dict):
                continue
            
            detail_type = detail.get("type", "")
            meta = detail.get("metadata", {})
            if not isinstance(meta, dict):
                continue
            
            if detail_type == "statute_article":
                statute = meta.get("statute_name", "") or meta.get("law_name", "")
                article = meta.get("article_no", "") or meta.get("article_number", "")
                if statute:
                    if article:
                        statute_summary.append(f"- {statute} 제{article}조")
                    else:
                        statute_summary.append(f"- {statute}")
            elif detail_type == "case_paragraph":
                court = meta.get("court", "")
                case_name = meta.get("case_name", "") or meta.get("casenames", "")
                if court or case_name:
                    case_summary.append(f"- {court} {case_name}".strip())
            elif detail_type == "interpretation_paragraph":
                org = meta.get("org", "")
                title = meta.get("title", "")
                if org or title:
                    interpretation_summary.append(f"- {org} {title}".strip())
        
        prompt = f"""당신은 법률 상담 AI 어시스턴트입니다. 사용자의 질문과 제공된 답변을 분석하여, 사용자가 다음에 물어볼 수 있는 연관 질문 5개를 생성해주세요.

## 사용자 질문
{query}

## 제공된 답변
{answer if answer else '답변이 아직 생성되지 않았습니다.'}

## 참고 법령
{chr(10).join(legal_references) if legal_references else '없음'}

## 관련 법령 정보
{chr(10).join(statute_summary) if statute_summary else '없음'}

## 관련 판례 정보
{chr(10).join(case_summary) if case_summary else '없음'}

## 해석례 정보
{chr(10).join(interpretation_summary) if interpretation_summary else '없음'}

## 질문 생성 가이드라인

1. **답변 내용 기반 질문**
   - 답변에서 언급된 법령의 다른 조문에 대한 질문
   - 답변에서 설명한 개념의 심화 질문
   - 답변에서 다룬 사례와 유사한 상황에 대한 질문

2. **답변에서 다루지 않은 관련 주제**
   - 답변에서 간략히 언급만 된 주제의 상세 질문
   - 관련 법령이나 판례에 대한 추가 질문
   - 실무 적용 방법에 대한 질문

3. **구체성과 실용성**
   - 일반적인 질문("더 자세히 알려주세요")은 피하기
   - 구체적인 법률 개념이나 절차에 대한 질문 생성
   - 실제 법률 문제 해결에 도움이 되는 질문

4. **질문 형식**
   - 각 질문은 한 문장으로 작성
   - 자연스러운 한국어로 작성
   - 질문형 종결어미 사용 ("~인가요?", "~있나요?", "~되나요?")

## 응답 형식 (반드시 JSON 형식으로 응답)

{{
    "related_questions": [
        "구체적인 연관 질문 1",
        "구체적인 연관 질문 2",
        "구체적인 연관 질문 3",
        "구체적인 연관 질문 4",
        "구체적인 연관 질문 5"
    ],
    "reasoning": "질문 생성 근거 (한국어로 간단히 설명)"
}}"""
        
        return prompt
    
    def generate(
        self,
        query: str,
        answer: str,
        sources_detail: List[Dict[str, Any]],
        legal_references: List[str],
        max_retries: int = SourcesExtractorConstants.LLM_MAX_RETRIES
    ) -> List[str]:
        """연관질문 생성"""
        if not query:
            return []
        
        for attempt in range(max_retries + 1):
            try:
                llm_handler = self.get_llm_handler()
                if not llm_handler:
                    if attempt < max_retries:
                        self.logger.debug(f"[RelatedQuestionsLLMGenerator] LLM handler not available, retrying... (attempt {attempt + 1}/{max_retries})")
                        continue
                    return []
                
                prompt = self.build_prompt(query, answer, sources_detail, legal_references)
                
                self.logger.info(f"[RelatedQuestionsLLMGenerator] Calling LLM for related_questions generation (attempt {attempt + 1})")
                response = llm_handler.llm.invoke(prompt)
                response_content = self.extract_llm_response(response)
                
                try:
                    result = json.loads(response_content)
                except json.JSONDecodeError:
                    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"related_questions"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_content, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group())
                    else:
                        raise
                
                questions = result.get("related_questions", [])
                
                valid_questions = [
                    str(q).strip() 
                    for q in questions 
                    if q and str(q).strip() and len(str(q).strip()) >= SourcesExtractorConstants.MIN_QUESTION_LENGTH
                ]
                
                if valid_questions:
                    self.logger.info(f"[RelatedQuestionsLLMGenerator] Generated {len(valid_questions)} related_questions using LLM (attempt {attempt + 1})")
                    return valid_questions[:SourcesExtractorConstants.MAX_RELATED_QUESTIONS]
                else:
                    if attempt < max_retries:
                        self.logger.debug(f"[RelatedQuestionsLLMGenerator] No valid questions generated, retrying... (attempt {attempt + 1}/{max_retries})")
                        continue
                    
            except json.JSONDecodeError as e:
                self.logger.warning(f"[RelatedQuestionsLLMGenerator] JSON parsing failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    continue
            except Exception as e:
                self.logger.warning(f"[RelatedQuestionsLLMGenerator] LLM call failed (attempt {attempt + 1}): {e}", exc_info=True)
                if attempt < max_retries:
                    continue
        
        return []


class SourcesExtractor:
    """Sources 추출기"""
    
    def __init__(self, workflow_service, session_service):
        self.workflow_service = workflow_service
        self.session_service = session_service
        self.source_processor = SourceTypeProcessor()
        self.llm_generator = RelatedQuestionsLLMGenerator(workflow_service)
    
    async def extract_from_message_metadata(
        self, 
        session_id: str, 
        message_id: Optional[str] = None
    ) -> Dict[str, List[Any]]:
        """메시지 metadata에서 sources 추출"""
        try:
            messages = self.session_service.get_messages(session_id)
            
            if message_id:
                for msg in messages:
                    msg_id = msg.get("message_id")
                    metadata_msg_id = msg.get("metadata", {}).get("message_id") if isinstance(msg.get("metadata"), dict) else None
                    if (msg_id == message_id or metadata_msg_id == message_id) and msg.get("role") == "assistant":
                        result = self._extract_from_message(msg)
                        if any(result.values()):
                            matched_id = msg_id if msg_id == message_id else metadata_msg_id
                            logger.info(f"Found sources in message metadata for message_id={message_id} (matched via {matched_id}): {len(result.get('sources', []))} sources, {len(result.get('legal_references', []))} legal_references, {len(result.get('sources_detail', []))} sources_detail, {len(result.get('related_questions', []))} related_questions")
                            return result
                
                logger.warning(f"Message with message_id={message_id} not found in session {session_id}. Available assistant messages: {[{'msg_id': m.get('message_id'), 'metadata_msg_id': m.get('metadata', {}).get('message_id') if isinstance(m.get('metadata'), dict) else None} for m in messages if m.get('role') == 'assistant']}")
            else:
                for msg in reversed(messages):
                    if msg.get("role") == "assistant":
                        result = self._extract_from_message(msg)
                        if any(result.values()):
                            logger.info(f"Found sources in assistant message metadata (message_id={msg.get('message_id')}): {len(result.get('sources', []))} sources, {len(result.get('legal_references', []))} legal_references, {len(result.get('sources_detail', []))} sources_detail, {len(result.get('related_questions', []))} related_questions")
                            return result
        except Exception as e:
            logger.warning(f"Failed to get sources from message metadata: {e}")
        
        return ExtractionResult.empty().to_dict()
    
    def _extract_from_message(self, msg: Dict[str, Any]) -> Dict[str, List[Any]]:
        """단일 메시지에서 sources 추출"""
        metadata = msg.get("metadata", {})
        if not isinstance(metadata, dict):
            return ExtractionResult.empty().to_dict()
        
        return {
            "sources": metadata.get("sources", []) if isinstance(metadata.get("sources"), list) else [],
            "legal_references": metadata.get("legal_references", []) if isinstance(metadata.get("legal_references"), list) else [],
            "sources_detail": metadata.get("sources_detail", []) if isinstance(metadata.get("sources_detail"), list) else [],
            "related_questions": metadata.get("related_questions", []) if isinstance(metadata.get("related_questions"), list) else []
        }
    
    async def extract_from_state(self, session_id: str) -> Dict[str, List[Any]]:
        """LangGraph state에서 sources 추출"""
        if not self.workflow_service or not self.workflow_service.app:
            logger.warning("Workflow service is not available")
            return ExtractionResult.empty().to_dict()
        
        try:
            config = {"configurable": {"thread_id": session_id}}
            final_state = await self.workflow_service.app.aget_state(config)
            
            if not final_state or not final_state.values:
                logger.warning(f"No state found for session_id: {session_id}")
                return ExtractionResult.empty().to_dict()
            
            state_values = final_state.values
            
            sources = self._extract_sources(state_values)
            legal_references = self._extract_legal_references(state_values)
            sources_detail = self._extract_sources_detail(state_values)
            related_questions = self._extract_related_questions(state_values)
            
            logger.info(f"Sources extracted from session {session_id}: {len(sources)} sources, {len(legal_references)} legal_references, {len(sources_detail)} sources_detail, {len(related_questions)} related_questions")
            
            return {
                "sources": sources,
                "legal_references": legal_references,
                "sources_detail": sources_detail,
                "related_questions": related_questions
            }
        except Exception as e:
            logger.error(f"Error extracting sources from state: {e}", exc_info=True)
            return ExtractionResult.empty().to_dict()
    
    def _extract_from_state_with_fallback(
        self,
        state_values: Dict[str, Any],
        key: str,
        extractor_func: Callable[[List[Dict[str, Any]]], List[Any]]
    ) -> List[Any]:
        """state에서 값을 추출하는 공통 패턴"""
        if key in state_values:
            value = state_values.get(key, [])
            if isinstance(value, list) and value:
                logger.debug(f"Found {len(value)} {key} at top level")
                return [str(v) for v in value if v and str(v).strip()]
        
        if "metadata" in state_values:
            metadata = state_values.get("metadata", {})
            if isinstance(metadata, dict) and key in metadata:
                metadata_value = metadata.get(key, [])
                if isinstance(metadata_value, list) and metadata_value:
                    logger.debug(f"Found {len(metadata_value)} {key} in metadata")
                    return [str(v) for v in metadata_value if v and str(v).strip()]
        
        if "retrieved_docs" in state_values:
            retrieved_docs = state_values.get("retrieved_docs", [])
            if isinstance(retrieved_docs, list):
                result = extractor_func(retrieved_docs)
                if result:
                    logger.info(f"Extracted {len(result)} {key} from retrieved_docs")
                    return result
        
        return []
    
    def _extract_sources(self, state_values: Dict[str, Any]) -> List[str]:
        """state에서 sources 추출"""
        return self._extract_from_state_with_fallback(
            state_values,
            "sources",
            self._extract_sources_from_retrieved_docs
        )
    
    def _extract_sources_from_retrieved_docs(self, retrieved_docs: List[Dict[str, Any]]) -> List[str]:
        """retrieved_docs에서 sources 추출"""
        seen_sources = set()
        sources = []
        
        for doc in retrieved_docs:
            if not isinstance(doc, dict):
                continue
            
            source = self._extract_source_from_doc(doc)
            
            if source:
                source_str = source.strip() if isinstance(source, str) else str(source).strip()
                source_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("source_type", "")
                
                if self.source_processor.validate_source(source_str, source_type):
                    if source_str not in seen_sources:
                        sources.append(source_str)
                        seen_sources.add(source_str)
        
        return sources
    
    def _extract_source_from_doc(self, doc: Dict[str, Any]) -> Optional[str]:
        """단일 doc에서 source 추출"""
        source_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("source_type", "")
        metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
        
        if source_type == "statute_article":
            return self.source_processor.process_statute_article(doc, metadata)
        elif source_type == "case_paragraph":
            return self.source_processor.process_case_paragraph(doc, metadata)
        elif source_type == "decision_paragraph":
            return self.source_processor.process_decision_paragraph(doc, metadata)
        elif source_type == "interpretation_paragraph":
            return self.source_processor.process_interpretation_paragraph(doc, metadata)
        else:
            return self.source_processor.process_fallback(doc, metadata)
    
    def _extract_legal_references(self, state_values: Dict[str, Any]) -> List[str]:
        """state에서 legal_references 추출"""
        return self._extract_from_state_with_fallback(
            state_values,
            "legal_references",
            self._extract_legal_references_from_retrieved_docs
        )
    
    def _extract_legal_references_from_retrieved_docs(self, retrieved_docs: List[Dict[str, Any]]) -> List[str]:
        """retrieved_docs에서 legal_references 추출"""
        seen_legal_refs = set()
        legal_refs = []
        
        for doc in retrieved_docs:
            if not isinstance(doc, dict):
                continue
            
            source_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("source_type", "")
            metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
            
            if source_type == "statute_article":
                statute_name = (
                    doc.get("statute_name") or
                    doc.get("law_name") or
                    metadata.get("statute_name") or
                    metadata.get("law_name")
                )
                
                if statute_name:
                    article_no = (
                        doc.get("article_no") or
                        doc.get("article_number") or
                        metadata.get("article_no") or
                        metadata.get("article_number")
                    )
                    clause_no = doc.get("clause_no") or metadata.get("clause_no")
                    item_no = doc.get("item_no") or metadata.get("item_no")
                    
                    legal_ref_parts = [statute_name]
                    if article_no:
                        legal_ref_parts.append(article_no)
                    if clause_no:
                        legal_ref_parts.append(f"제{clause_no}항")
                    if item_no:
                        legal_ref_parts.append(f"제{item_no}호")
                    
                    legal_ref = " ".join(legal_ref_parts)
                    
                    if legal_ref and legal_ref not in seen_legal_refs:
                        legal_refs.append(legal_ref)
                        seen_legal_refs.add(legal_ref)
        
        return legal_refs
    
    def _extract_related_questions(self, state_values: Dict[str, Any]) -> List[str]:
        """state에서 related_questions 추출 및 생성 (LLM 우선)"""
        logger.debug(f"[sources_extractor] _extract_related_questions called, state_keys: {list(state_values.keys()) if isinstance(state_values, dict) else 'N/A'}")
        
        if "metadata" in state_values:
            metadata = state_values.get("metadata", {})
            if isinstance(metadata, dict) and "related_questions" in metadata:
                related_questions = metadata.get("related_questions", [])
                if isinstance(related_questions, list):
                    questions = [str(q).strip() for q in related_questions if q and str(q).strip()]
                    if questions:
                        logger.info(f"[sources_extractor] Found {len(questions)} related_questions in metadata")
                        return questions
        
        query = state_values.get("query", "")
        answer = state_values.get("answer", "")
        sources_detail = state_values.get("sources_detail", [])
        legal_references = state_values.get("legal_references", [])
        
        if not query:
            logger.debug("[sources_extractor] Query is empty, cannot generate related_questions")
            return []
        
        llm_questions = self.llm_generator.generate(
            query, answer or "", sources_detail, legal_references
        )
        if llm_questions:
            logger.info(f"[sources_extractor] Generated {len(llm_questions)} related_questions using direct LLM call")
            return llm_questions
        
        if self.workflow_service and hasattr(self.workflow_service, 'conversation_flow_tracker') and self.workflow_service.conversation_flow_tracker:
            try:
                from lawfirm_langgraph.core.services.conversation_manager import ConversationContext, ConversationTurn
                from datetime import datetime
                
                query_type = state_values.get("metadata", {}).get("query_type", "general_question")
                
                enhanced_answer = answer or ""
                if len(enhanced_answer.strip()) < SourcesExtractorConstants.MIN_ANSWER_LENGTH_FOR_ENHANCEMENT and sources_detail:
                    statute_info = []
                    for detail in sources_detail[:2]:
                        if isinstance(detail, dict) and detail.get("type") == "statute_article":
                            meta = detail.get("metadata", {})
                            if isinstance(meta, dict):
                                statute = meta.get("statute_name") or meta.get("law_name", "")
                                article = meta.get("article_no", "")
                                if statute:
                                    if article:
                                        statute_info.append(f"{statute} 제{article}조")
                                    else:
                                        statute_info.append(statute)
                    if statute_info:
                        enhanced_answer = f"{enhanced_answer} 관련 법령: {', '.join(statute_info)}".strip()
                
                turn = ConversationTurn(
                    user_query=query,
                    bot_response=enhanced_answer,
                    timestamp=datetime.now(),
                    question_type=query_type
                )
                
                context = ConversationContext(
                    session_id="default",
                    turns=[turn],
                    entities={},
                    topic_stack=[],
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )
                
                suggested_questions = self.workflow_service.conversation_flow_tracker.suggest_follow_up_questions(context)
                
                if suggested_questions and len(suggested_questions) > 0:
                    questions = [str(q).strip() for q in suggested_questions if q and str(q).strip()]
                    if questions:
                        logger.info(f"[sources_extractor] Generated {len(questions)} related_questions using ConversationFlowTracker")
                        return questions
            except Exception as e:
                logger.warning(f"[sources_extractor] Failed to use ConversationFlowTracker: {e}", exc_info=True)
        
        logger.warning("[sources_extractor] All LLM methods failed, using minimal template as last resort")
        if query:
            return [
                f"{query}에 대한 더 자세한 정보가 필요하신가요?",
                f"{query}와 관련된 다른 질문이 있으신가요?"
            ]
        
        return []
    
    def _extract_sources_detail(self, state_values: Dict[str, Any]) -> List[Dict[str, Any]]:
        """state에서 sources_detail 추출"""
        if "sources_detail" in state_values:
            sources_detail_list = state_values.get("sources_detail", [])
            if isinstance(sources_detail_list, list):
                return sources_detail_list
        
        if "metadata" in state_values:
            metadata = state_values.get("metadata", {})
            if isinstance(metadata, dict) and "sources_detail" in metadata:
                metadata_sources_detail = metadata.get("sources_detail", [])
                if isinstance(metadata_sources_detail, list):
                    return metadata_sources_detail
        
        if "retrieved_docs" in state_values:
            return self._generate_sources_detail_from_retrieved_docs(
                state_values.get("retrieved_docs", [])
            )
        
        return []
    
    def _generate_sources_detail_from_retrieved_docs(
        self, 
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """retrieved_docs에서 sources_detail 생성"""
        if not isinstance(retrieved_docs, list) or not retrieved_docs:
            return []
        
        try:
            from lawfirm_langgraph.core.services.unified_source_formatter import UnifiedSourceFormatter
            formatter = UnifiedSourceFormatter()
            
            sources_detail = []
            for doc in retrieved_docs:
                if not isinstance(doc, dict):
                    continue
                
                source_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("source_type", "")
                if not source_type:
                    metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                    if metadata.get("case_id") or metadata.get("court") or metadata.get("casenames"):
                        source_type = "case_paragraph"
                    elif metadata.get("decision_id") or metadata.get("org"):
                        source_type = "decision_paragraph"
                    elif metadata.get("interpretation_number") or (metadata.get("org") and metadata.get("title")):
                        source_type = "interpretation_paragraph"
                    elif metadata.get("statute_name") or metadata.get("law_name") or metadata.get("article_no"):
                        source_type = "statute_article"
                    else:
                        continue
                
                metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                merged_metadata = {**metadata}
                
                for key in ["statute_name", "law_name", "article_no", "article_number", "clause_no", "item_no",
                           "court", "doc_id", "casenames", "org", "title", "announce_date", "decision_date", "response_date"]:
                    if key in doc:
                        merged_metadata[key] = doc[key]
                
                source_info_detail = formatter.format_source(source_type, merged_metadata)
                
                detail_dict = {
                    "name": source_info_detail.name,
                    "type": source_info_detail.type,
                    "url": source_info_detail.url or "",
                    "metadata": source_info_detail.metadata or {}
                }
                
                if source_info_detail.metadata:
                    meta = source_info_detail.metadata
                    
                    if source_type == "statute_article":
                        if meta.get("statute_name"):
                            detail_dict["statute_name"] = meta["statute_name"]
                        if meta.get("article_no"):
                            detail_dict["article_no"] = meta["article_no"]
                        if meta.get("clause_no"):
                            detail_dict["clause_no"] = meta["clause_no"]
                        if meta.get("item_no"):
                            detail_dict["item_no"] = meta["item_no"]
                    elif source_type == "case_paragraph":
                        if meta.get("doc_id"):
                            detail_dict["case_number"] = meta["doc_id"]
                        if meta.get("court"):
                            detail_dict["court"] = meta["court"]
                        if meta.get("casenames"):
                            detail_dict["case_name"] = meta["casenames"]
                    elif source_type == "decision_paragraph":
                        if meta.get("doc_id"):
                            detail_dict["decision_number"] = meta["doc_id"]
                        if meta.get("org"):
                            detail_dict["org"] = meta["org"]
                        if meta.get("decision_date"):
                            detail_dict["decision_date"] = meta["decision_date"]
                        if meta.get("result"):
                            detail_dict["result"] = meta["result"]
                    elif source_type == "interpretation_paragraph":
                        if meta.get("doc_id"):
                            detail_dict["interpretation_number"] = meta["doc_id"]
                        if meta.get("org"):
                            detail_dict["org"] = meta["org"]
                        if meta.get("title"):
                            detail_dict["title"] = meta["title"]
                        if meta.get("response_date"):
                            detail_dict["response_date"] = meta["response_date"]
                
                content = doc.get("content") or doc.get("text") or ""
                if content:
                    detail_dict["content"] = content
                
                sources_detail.append(detail_dict)
            
            return sources_detail
        except Exception as e:
            logger.warning(f"Error generating sources_detail from retrieved_docs: {e}")
            return []
