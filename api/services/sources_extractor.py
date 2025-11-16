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
        
        sources_detail = metadata.get("sources_detail", []) if isinstance(metadata.get("sources_detail"), list) else []
        if sources_detail:
            sources_detail = self._normalize_sources_detail(sources_detail)
        
        # legal_references는 sources_detail에서 추출 (deprecated)
        extracted_legal_refs = self._extract_legal_references_from_sources_detail_only(sources_detail)
        existing_legal_refs = metadata.get("legal_references", []) if isinstance(metadata.get("legal_references"), list) else []
        all_legal_refs = list(set(extracted_legal_refs + existing_legal_refs))
        
        # 타입별 그룹화 (새로운 기능) - 판례의 참조 법령 포함
        sources_by_type = self._get_sources_by_type_with_reference_statutes(sources_detail) if sources_detail else {
            "statute_article": [],
            "case_paragraph": [],
            "decision_paragraph": [],
            "interpretation_paragraph": [],
            "regulation_paragraph": []
        }
        
        # sources_by_type만 반환 (sources_detail은 sources_by_type에서 재구성 가능)
        return {
            "sources_by_type": sources_by_type,  # 유일한 필요한 필드
            "related_questions": metadata.get("related_questions", []) if isinstance(metadata.get("related_questions"), list) else [],
            # 하위 호환성을 위해 deprecated 필드도 포함 (점진적 제거)
            "sources": metadata.get("sources", []) if isinstance(metadata.get("sources"), list) else [],  # deprecated: sources_by_type에서 재구성 가능
            "legal_references": all_legal_refs,  # deprecated: sources_by_type에서 재구성 가능
            "sources_detail": sources_detail,  # deprecated: sources_by_type에서 재구성 가능
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
            
            sources_detail = self._enhance_sources_detail_with_sources(sources, sources_detail)
            sources_detail = self._normalize_sources_detail(sources_detail)
            
            # legal_references는 sources_detail에서 추출 (deprecated)
            extracted_legal_refs = self._extract_legal_references_from_sources_detail_only(sources_detail)
            
            # 기존 legal_references와 병합 (하위 호환성)
            existing_legal_refs = self._extract_legal_references(state_values)
            legal_references = self._extract_legal_references_from_sources_detail(sources_detail, existing_legal_refs)
            
            # 모든 legal_references 병합 (중복 제거)
            all_legal_refs = list(set(extracted_legal_refs + legal_references))
            
            # 타입별 그룹화 (새로운 기능)
            sources_by_type = self._get_sources_by_type(sources_detail)
            
            # 판례/결정례/해석례에서 참조조문 추출하여 법령 추가
            extracted_statutes = self._extract_statutes_from_reference_clauses(sources_detail)
            
            if extracted_statutes:
                # 기존 statute_article과 병합 (중복 제거)
                existing_statutes = sources_by_type.get("statute_article", [])
                existing_keys = {
                    f"{s.get('statute_name', '')}_{s.get('article_no', '')}_{s.get('clause_no', '')}_{s.get('item_no', '')}"
                    for s in existing_statutes if isinstance(s, dict)
                }
                
                for statute in extracted_statutes:
                    statute_key = f"{statute.get('statute_name', '')}_{statute.get('article_no', '')}_{statute.get('clause_no', '')}_{statute.get('item_no', '')}"
                    if statute_key not in existing_keys:
                        existing_statutes.append(statute)
                        existing_keys.add(statute_key)
                
                sources_by_type["statute_article"] = existing_statutes
                logger.info(f"Extracted {len(extracted_statutes)} statutes from reference clauses")
            
            logger.info(f"Sources extracted from session {session_id}: {len(sources_by_type.get('statute_article', []))} statutes, {len(sources_by_type.get('case_paragraph', []))} cases, {len(sources_by_type.get('decision_paragraph', []))} decisions, {len(sources_by_type.get('interpretation_paragraph', []))} interpretations, {len(related_questions)} related_questions")
            
            # sources_by_type만 반환 (sources_detail은 sources_by_type에서 재구성 가능)
            return {
                "sources_by_type": sources_by_type,  # 유일한 필요한 필드
                "related_questions": related_questions,
                # 하위 호환성을 위해 deprecated 필드도 포함 (점진적 제거)
                "sources": sources,  # deprecated: sources_by_type에서 재구성 가능
                "legal_references": all_legal_refs,  # deprecated: sources_by_type에서 재구성 가능
                "sources_detail": sources_detail,  # deprecated: sources_by_type에서 재구성 가능
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
            
            # source_processor가 실패하면 fallback 사용
            if not source:
                source = self._extract_source_fallback(doc, retrieved_docs)
            
            if source:
                source_str = source.strip() if isinstance(source, str) else str(source).strip()
                source_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("source_type", "")
                
                # validate_source 검증
                is_valid = self.source_processor.validate_source(source_str, source_type)
                
                # fallback에서 추출한 경우 또는 유효한 source인 경우 추가
                # fallback에서 추출한 경우는 validate_source가 실패해도 추가 (조문 번호 등이 있을 수 있음)
                # 단, "_source"로 끝나는 기본값은 제외
                if source_str and not source_str.endswith("_source"):
                    if is_valid or (len(source_str) >= 2 and source_str not in SourcesExtractorConstants.INVALID_SOURCES):
                        # 너무 긴 문장 제외 (100자 이상)
                        if len(source_str) <= 100:
                            if source_str not in seen_sources:
                                sources.append(source_str)
                                seen_sources.add(source_str)
                                logger.debug(f"[sources_extractor] Added source: {source_str} (validated: {is_valid})")
                        else:
                            logger.debug(f"[sources_extractor] Skipped too long source: {source_str[:50]}...")
                    else:
                        logger.debug(f"[sources_extractor] Skipped invalid source: {source_str}")
        
        return sources
    
    def _extract_statute_name_from_context(self, doc: Dict[str, Any], all_docs: List[Dict[str, Any]], article_no: str = "") -> Optional[str]:
        """같은 세션의 다른 문서(판례 등)에서 법령명 추출"""
        if not article_no:
            # doc의 content에서 조문 번호 추출
            content = doc.get("content") or doc.get("text") or ""
            match = re.search(r'제\s*(\d+)\s*조', content)
            if match:
                article_no = match.group(1)
            else:
                # metadata에서 조문 번호 추출
                article_no = doc.get("article_no") or doc.get("article_number") or ""
                if article_no:
                    # "제XXX조" 형식에서 숫자만 추출
                    match = re.search(r'(\d+)', article_no)
                    if match:
                        article_no = match.group(1)
        
        if article_no:
            # 다른 문서들에서 같은 조문 번호를 가진 법령명 찾기
            for other_doc in all_docs:
                if other_doc.get("type") == "case_paragraph":
                    other_content = other_doc.get("content") or other_doc.get("text") or ""
                    # "민법 제XXX조" 패턴 찾기
                    pattern = r'([가-힣]{1,20}법)\s*제\s*' + re.escape(article_no) + r'\s*조'
                    match = re.search(pattern, other_content)
                    if match:
                        extracted = match.group(1).strip()
                        if len(extracted) <= 20 and extracted.count(' ') <= 2:
                            logger.debug(f"[sources_extractor] Extracted statute name from context (case reference): {extracted} for article {article_no}")
                            return extracted
        
        return None
    
    def _get_statute_name_from_db(self, law_id: Optional[str], statute_id: Optional[str]) -> Optional[str]:
        """데이터베이스에서 법령명 조회 (타임아웃 및 예외 처리 강화)"""
        if not law_id and not statute_id:
            return None
        
        try:
            # 데이터베이스 연결 시도 (타임아웃 설정)
            import sqlite3
            import os
            
            # 데이터베이스 경로 찾기
            db_paths = [
                os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'lawfirm_v2.db'),
                os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'lawfirm_v2.db'),
                os.getenv('DATABASE_PATH', ''),
            ]
            
            for db_path in db_paths:
                if db_path and os.path.exists(db_path):
                    try:
                        # 타임아웃 설정 (5초)
                        conn = sqlite3.connect(db_path, timeout=5.0)
                        conn.row_factory = sqlite3.Row
                        try:
                            cursor = conn.cursor()
                            
                            if law_id:
                                # assembly_laws 테이블에서 조회
                                cursor.execute("SELECT law_name FROM assembly_laws WHERE law_id = ? LIMIT 1", (law_id,))
                                result = cursor.fetchone()
                                if result:
                                    statute_name = result[0] if isinstance(result, sqlite3.Row) else result[0]
                                    logger.debug(f"[sources_extractor] Found statute name from DB (law_id): {statute_name}")
                                    return statute_name
                            
                            if statute_id:
                                # statutes 테이블에서 조회
                                cursor.execute("SELECT name FROM statutes WHERE id = ? LIMIT 1", (statute_id,))
                                result = cursor.fetchone()
                                if result:
                                    statute_name = result[0] if isinstance(result, sqlite3.Row) else result[0]
                                    logger.debug(f"[sources_extractor] Found statute name from DB (statute_id): {statute_name}")
                                    return statute_name
                                
                                # statute_articles를 통해 statutes 조회
                                cursor.execute("""
                                    SELECT s.name FROM statutes s
                                    INNER JOIN statute_articles sa ON s.id = sa.statute_id
                                    WHERE sa.id = ? LIMIT 1
                                """, (statute_id,))
                                result = cursor.fetchone()
                                if result:
                                    statute_name = result[0] if isinstance(result, sqlite3.Row) else result[0]
                                    logger.debug(f"[sources_extractor] Found statute name from DB (via statute_articles): {statute_name}")
                                    return statute_name
                        finally:
                            conn.close()
                        break
                    except sqlite3.OperationalError as e:
                        # 데이터베이스 잠금 오류 등은 무시하고 계속 진행
                        logger.debug(f"[sources_extractor] Database operational error (skipping): {e}")
                        continue
                    except Exception as db_error:
                        # 기타 데이터베이스 오류는 로그만 남기고 계속 진행
                        logger.debug(f"[sources_extractor] Database error (skipping): {db_error}")
                        continue
        except Exception as e:
            # 모든 예외를 잡아서 스트리밍이 중단되지 않도록 함
            logger.debug(f"[sources_extractor] Failed to get statute name from DB (non-blocking): {e}")
        
        return None
    
    def _extract_source_fallback(self, doc: Dict[str, Any], all_docs: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """fallback source 추출 (metadata가 비어있어도 content나 다른 필드에서 추출)"""
        source_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("source_type", "")
        metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
        
        logger.debug(f"[sources_extractor] Using fallback source extraction: type={source_type}, has_metadata={bool(metadata)}, metadata_keys={list(metadata.keys()) if metadata else []}")
        
        # doc의 최상위 레벨 필드도 확인 (강화)
        merged_metadata = {**metadata}
        for key in ["statute_name", "law_name", "article_no", "article_number", "clause_no", "item_no",
                   "court", "doc_id", "casenames", "org", "title", "case_id", "decision_id", "interpretation_number",
                   "law_id", "statute_id", "abbrv", "statute_abbrv"]:
            if key in doc and not merged_metadata.get(key):
                merged_metadata[key] = doc[key]
        
        if source_type == "case_paragraph":
            court = merged_metadata.get("court") or ""
            case_name = merged_metadata.get("casenames") or ""
            doc_id = merged_metadata.get("doc_id") or merged_metadata.get("case_id") or ""
            
            if court or case_name or doc_id:
                result = f"{court} {case_name} {doc_id}".strip()
                logger.debug(f"[sources_extractor] Extracted case source from metadata: {result}")
                return result
            
            # content에서 판례 정보 추출 시도
            content = doc.get("content") or doc.get("text") or ""
            if content:
                # 패턴 1: 날짜 포함 형식 "대법원 2007. 12. 27. 선고 2006다9408 판결"
                pattern1 = r'(대법원|지방법원|고등법원|특허법원|가정법원|행정법원|헌법재판소)\s+(\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.)\s*선고\s*(\d{4}[가나다라마바사아자차카타파하]\d+)'
                match = re.search(pattern1, content)
                if match:
                    result = f"{match.group(1)} {match.group(3)}"
                    logger.debug(f"[sources_extractor] Extracted case source from content (with date): {result}")
                    return result
                
                # 패턴 2: 날짜 없는 형식 "대법원 2021다275611"
                pattern2 = r'(대법원|지방법원|고등법원|특허법원|가정법원|행정법원|헌법재판소)\s+(\d{4}[가나다라마바사아자차카타파하]\d+)'
                match = re.search(pattern2, content)
                if match:
                    result = f"{match.group(1)} {match.group(2)}"
                    logger.debug(f"[sources_extractor] Extracted case source from content (no date): {result}")
                    return result
                
        elif source_type == "statute_article":
            statute_name = merged_metadata.get("statute_name") or merged_metadata.get("law_name") or ""
            article_no = merged_metadata.get("article_no") or merged_metadata.get("article_number") or ""
            clause_no = merged_metadata.get("clause_no") or ""
            item_no = merged_metadata.get("item_no") or ""
            
            # 1. law_id나 statute_id가 있으면 데이터베이스에서 조회
            if not statute_name:
                law_id = merged_metadata.get("law_id") or doc.get("law_id")
                statute_id = merged_metadata.get("statute_id") or doc.get("statute_id")
                if law_id or statute_id:
                    db_statute_name = self._get_statute_name_from_db(law_id, statute_id)
                    if db_statute_name:
                        statute_name = db_statute_name
                        logger.debug(f"[sources_extractor] Extracted statute name from DB: {statute_name}")
            
            # 2. abbrv나 statute_abbrv가 있으면 사용 (약어는 나중에 처리)
            if not statute_name:
                abbrv = merged_metadata.get("abbrv") or merged_metadata.get("statute_abbrv") or doc.get("abbrv") or doc.get("statute_abbrv")
                if abbrv:
                    # 약어를 법령명으로 변환 (일반적인 약어 매핑)
                    abbrv_to_name = {
                        "민법": "민법",
                        "형법": "형법",
                        "상법": "상법",
                        "민소법": "민사소송법",
                        "형소법": "형사소송법",
                    }
                    if abbrv in abbrv_to_name:
                        statute_name = abbrv_to_name[abbrv]
                        logger.debug(f"[sources_extractor] Extracted statute name from abbrv: {statute_name}")
            
            # 3. 같은 세션의 다른 문서(판례)에서 법령명 추출
            if not statute_name and all_docs:
                extracted_from_context = self._extract_statute_name_from_context(doc, all_docs, article_no.replace("제", "").replace("조", "").strip() if article_no else "")
                if extracted_from_context:
                    statute_name = extracted_from_context
                    logger.debug(f"[sources_extractor] Extracted statute name from context: {statute_name}")
            
            # 4. metadata에 법령명이 없으면 content에서 추출 시도
            content = doc.get("content") or doc.get("text") or ""
            if not statute_name and content:
                # 패턴 1: 「법령명」 형식
                pattern1 = r'「([^」]+)」'
                match = re.search(pattern1, content)
                if match:
                    statute_name = match.group(1).strip()
                    logger.debug(f"[sources_extractor] Extracted statute name from content (quotes): {statute_name}")
                else:
                    # 패턴 2: "법령명" + "법" 형식 (예: "민사소송 등 인지법", "건축법")
                    # 최대 20자로 제한하여 너무 긴 문장 제외
                    pattern2 = r'([가-힣]{1,20}법)(?:\s|$|\.|,|\)|「|」|제)'
                    match = re.search(pattern2, content)
                    if match:
                        extracted = match.group(1).strip()
                        # "규칙 또는 처분의 헌법" 같은 긴 문장 제외 (공백이 많으면 제외)
                        if len(extracted) <= 20 and extracted.count(' ') <= 2:
                            statute_name = extracted
                            logger.debug(f"[sources_extractor] Extracted statute name from content (pattern): {statute_name}")
                        else:
                            logger.debug(f"[sources_extractor] Skipped too long statute name: {extracted}")
            
            # 5. content에서 상대 참조 해석 ("전3조", "전조" 등)
            if not article_no and content:
                relative_ref_pattern = r'전\s*(\d+)\s*조|전\s*조'
                match = re.search(relative_ref_pattern, content)
                if match:
                    # 상대 참조가 있으면 일반적으로 "민법"으로 추정
                    if not statute_name:
                        statute_name = "민법"
                    logger.debug(f"[sources_extractor] Found relative reference in content, using default statute: {statute_name}")
            
            # 6. 조문 번호가 없으면 content에서 추출 시도
            if not article_no and content:
                # 패턴 1: "민법 제XXX조", "형법 제XXX조" 등 (주요 법령명)
                # 법령명 바로 앞에 공백이나 구두점이 있어야 함 (문맥 제거)
                pattern1 = r'(?:^|\.|,|\)|\(|「|」|에|의|을|를|이|가|은|는|으로|로|에서|부터|까지|와|과|및|또는)\s*(민법|형법|상법|공법|행정법|민사소송법|형사소송법|가족법|상속법|채권법|물권법|계약법|불법행위법|부동산법|임대차보호법|전세권법|근저당법|저당권법|담보법|보증법|연대보증법|보증채무법|채권담보법|소유권법|점유권법|지상권법|지역권법|질권법|유치권법|우선변제권법|담보물권법|민사집행법)\s*제\s*(\d+)\s*조'
                match = re.search(pattern1, content)
                if match:
                    if not statute_name:
                        statute_name = match.group(1).strip()
                    article_no = f"제{match.group(2)}조"
                    logger.debug(f"[sources_extractor] Extracted statute and article from content: {statute_name} {article_no}")
                else:
                    # 패턴 2: "법령명 제XXX조" (앞에 법령명이 있는 경우)
                    pattern2 = r'([가-힣]{1,20}법)\s*제\s*(\d+)\s*조'
                    match = re.search(pattern2, content)
                    if match:
                        if not statute_name:
                            extracted_statute = match.group(1).strip()
                            # 너무 긴 문장 제외
                            if len(extracted_statute) <= 20 and extracted_statute.count(' ') <= 2:
                                statute_name = extracted_statute
                        article_no = f"제{match.group(2)}조"
                        logger.debug(f"[sources_extractor] Extracted statute and article from content (pattern2): {statute_name} {article_no}")
                    else:
                        # 패턴 3: "제XXX조" (법령명 없이 조문만)
                        pattern3 = r'제\s*(\d+)\s*조'
                        match = re.search(pattern3, content)
                        if match:
                            article_no = f"제{match.group(1)}조"
                            # 조문 번호가 있으면 같은 세션의 다른 문서에서 법령명 찾기
                            if not statute_name and all_docs:
                                extracted_from_context = self._extract_statute_name_from_context(doc, all_docs, match.group(1))
                                if extracted_from_context:
                                    statute_name = extracted_from_context
                                else:
                                    # 일반적인 법령명 추정 (민법이 가장 일반적)
                                    statute_name = "민법"
                            elif not statute_name:
                                statute_name = "민법"
                            logger.debug(f"[sources_extractor] Extracted article from content: {article_no}, statute_name: {statute_name}")
            
            # 법령명이나 조문 번호가 있으면 반환
            if statute_name or article_no:
                parts = [statute_name] if statute_name else []
                if article_no:
                    parts.append(article_no)
                if clause_no:
                    parts.append(f"제{clause_no}항")
                if item_no:
                    parts.append(f"제{item_no}호")
                return " ".join(parts).strip()
            
            # 아무것도 없으면 None 반환 (기본값은 필터링됨)
            return None
            
        elif source_type == "decision_paragraph":
            org = merged_metadata.get("org") or ""
            doc_id = merged_metadata.get("doc_id") or merged_metadata.get("decision_id") or ""
            if org or doc_id:
                return f"{org} {doc_id}".strip()
            
        elif source_type == "interpretation_paragraph":
            org = merged_metadata.get("org") or ""
            title = merged_metadata.get("title") or ""
            doc_id = merged_metadata.get("doc_id") or merged_metadata.get("interpretation_number") or ""
            if org or title or doc_id:
                return f"{org} {title} {doc_id}".strip()
        
        return None
    
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
            
            # doc의 최상위 레벨 필드도 확인 (강화)
            merged_metadata = {**metadata}
            for key in ["statute_name", "law_name", "article_no", "article_number", "clause_no", "item_no",
                       "law_id", "statute_id", "abbrv", "statute_abbrv"]:
                if key in doc and not merged_metadata.get(key):
                    merged_metadata[key] = doc[key]
            
            if source_type == "statute_article":
                statute_name = (
                    merged_metadata.get("statute_name") or
                    merged_metadata.get("law_name")
                )
                
                # 1. law_id나 statute_id가 있으면 데이터베이스에서 조회
                if not statute_name:
                    law_id = merged_metadata.get("law_id") or doc.get("law_id")
                    statute_id = merged_metadata.get("statute_id") or doc.get("statute_id")
                    if law_id or statute_id:
                        db_statute_name = self._get_statute_name_from_db(law_id, statute_id)
                        if db_statute_name:
                            statute_name = db_statute_name
                            logger.debug(f"[sources_extractor] Extracted statute name from DB for legal_references: {statute_name}")
                
                # 2. abbrv나 statute_abbrv가 있으면 사용
                if not statute_name:
                    abbrv = merged_metadata.get("abbrv") or merged_metadata.get("statute_abbrv") or doc.get("abbrv") or doc.get("statute_abbrv")
                    if abbrv:
                        abbrv_to_name = {
                            "민법": "민법",
                            "형법": "형법",
                            "상법": "상법",
                            "민소법": "민사소송법",
                            "형소법": "형사소송법",
                        }
                        if abbrv in abbrv_to_name:
                            statute_name = abbrv_to_name[abbrv]
                            logger.debug(f"[sources_extractor] Extracted statute name from abbrv for legal_references: {statute_name}")
                
                # 3. 같은 세션의 다른 문서(판례)에서 법령명 추출
                if not statute_name:
                    extracted_from_context = self._extract_statute_name_from_context(doc, retrieved_docs)
                    if extracted_from_context:
                        statute_name = extracted_from_context
                        logger.debug(f"[sources_extractor] Extracted statute name from context for legal_references: {statute_name}")
                
                # 4. metadata에 법령명이 없으면 content에서 추출 시도
                content = doc.get("content") or doc.get("text") or ""
                if not statute_name and content:
                    # 패턴 1: 「법령명」 형식
                    pattern1 = r'「([^」]+)」'
                    match = re.search(pattern1, content)
                    if match:
                        statute_name = match.group(1).strip()
                        logger.debug(f"[sources_extractor] Extracted statute name for legal_references (quotes): {statute_name}")
                    else:
                        # 패턴 2: "법령명" + "법" 형식 (최대 20자로 제한)
                        pattern2 = r'([가-힣]{1,20}법)(?:\s|$|\.|,|\)|「|」|제)'
                        match = re.search(pattern2, content)
                        if match:
                            extracted = match.group(1).strip()
                            # "규칙 또는 처분의 헌법" 같은 긴 문장 제외
                            if len(extracted) <= 20 and extracted.count(' ') <= 2:
                                statute_name = extracted
                                logger.debug(f"[sources_extractor] Extracted statute name for legal_references (pattern): {statute_name}")
                            else:
                                logger.debug(f"[sources_extractor] Skipped too long statute name for legal_references: {extracted}")
                
                article_no = (
                    merged_metadata.get("article_no") or
                    merged_metadata.get("article_number")
                )
                
                # 5. content에서 상대 참조 해석 ("전3조", "전조" 등)
                if not article_no and content:
                    relative_ref_pattern = r'전\s*(\d+)\s*조|전\s*조'
                    match = re.search(relative_ref_pattern, content)
                    if match:
                        if not statute_name:
                            statute_name = "민법"
                        logger.debug(f"[sources_extractor] Found relative reference in content for legal_references, using default statute: {statute_name}")
                
                # 6. 조문 번호가 없으면 content에서 추출 시도
                if not article_no and content:
                    # 패턴 1: "민법 제XXX조", "형법 제XXX조" 등
                    pattern1 = r'(민법|형법|상법|공법|행정법|민사소송법|형사소송법|가족법|상속법|채권법|물권법|계약법|불법행위법|부동산법|임대차보호법|전세권법|근저당법|저당권법|담보법|보증법|연대보증법|보증채무법|채권담보법)\s*제\s*(\d+)\s*조'
                    match = re.search(pattern1, content)
                    if match:
                        if not statute_name:
                            statute_name = match.group(1).strip()
                        article_no = f"제{match.group(2)}조"
                        logger.debug(f"[sources_extractor] Extracted statute and article for legal_references: {statute_name} {article_no}")
                    else:
                        # 패턴 2: "법령명 제XXX조"
                        pattern2 = r'([가-힣]{1,20}법)\s*제\s*(\d+)\s*조'
                        match = re.search(pattern2, content)
                        if match:
                            if not statute_name:
                                extracted_statute = match.group(1).strip()
                                if len(extracted_statute) <= 20 and extracted_statute.count(' ') <= 2:
                                    statute_name = extracted_statute
                            article_no = f"제{match.group(2)}조"
                            logger.debug(f"[sources_extractor] Extracted statute and article for legal_references (pattern2): {statute_name} {article_no}")
                        else:
                            # 패턴 3: "제XXX조" (법령명 없이 조문만)
                            pattern3 = r'제\s*(\d+)\s*조'
                            match = re.search(pattern3, content)
                            if match:
                                article_no = f"제{match.group(1)}조"
                                # 조문 번호가 있으면 같은 세션의 다른 문서에서 법령명 찾기
                                if not statute_name:
                                    extracted_from_context = self._extract_statute_name_from_context(doc, retrieved_docs, match.group(1))
                                    if extracted_from_context:
                                        statute_name = extracted_from_context
                                    else:
                                        statute_name = "민법"
                                logger.debug(f"[sources_extractor] Extracted article for legal_references: {article_no}, statute_name: {statute_name}")
                
                # 법령명이나 조문 번호가 있으면 legal_references에 추가
                if statute_name or article_no:
                    clause_no = merged_metadata.get("clause_no")
                    item_no = merged_metadata.get("item_no")
                    
                    legal_ref_parts = [statute_name] if statute_name else []
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
                        logger.debug(f"[sources_extractor] Added legal_reference: {legal_ref}")
        
        return legal_refs
    
    def _extract_legal_references_from_sources_detail(
        self,
        sources_detail: List[Dict[str, Any]],
        existing_legal_refs: List[str]
    ) -> List[str]:
        """sources_detail의 content에서 법조문 참조 추출하여 legal_references에 추가"""
        if not sources_detail:
            return existing_legal_refs
        
        seen_legal_refs = set(existing_legal_refs)
        legal_refs = list(existing_legal_refs)
        
        for detail in sources_detail:
            if not isinstance(detail, dict):
                continue
            
            content = detail.get("content", "")
            if not isinstance(content, str):
                continue
            
            # 개선된 법령 패턴: "구", "신", "개정" 등의 접두사 처리, 긴 법률명 지원
            # 예: "구 지방세법 제131조 제1항 제2호"
            # 예: "대부업 등의 등록 및 금융이용자 보호에 관한 법률 제2조 제1호"
            statute_pattern = r'(?:구|신|개정|폐지)?\s*([가-힣\s]{1,50}법률?)\s*제\s*(\d+)\s*조(?:\s*제\s*(\d+)\s*항)?(?:\s*제\s*(\d+)\s*호)?'
            matches = re.finditer(statute_pattern, content)
            
            for match in matches:
                statute_name = match.group(1).strip()
                article_no = match.group(2)
                clause_no = match.group(3)
                item_no = match.group(4)
                
                # 법률명 정리: 연속 공백을 하나로
                statute_name = re.sub(r'\s+', ' ', statute_name).strip()
                
                # 법률명이 너무 짧거나 길면 제외 (최소 2자, 최대 50자)
                if len(statute_name) < 2 or len(statute_name) > 50:
                    continue
                
                # "법" 또는 "법률"로 끝나지 않으면 제외
                if not statute_name.endswith('법') and not statute_name.endswith('법률'):
                    continue
                
                # 조문 번호 구성
                article_str = f"제{article_no}조"
                if clause_no:
                    article_str += f" 제{clause_no}항"
                if item_no:
                    article_str += f" 제{item_no}호"
                
                legal_ref = f"{statute_name} {article_str}"
                
                if legal_ref not in seen_legal_refs:
                    legal_refs.append(legal_ref)
                    seen_legal_refs.add(legal_ref)
                    logger.debug(f"[sources_extractor] Extracted legal_reference from sources_detail: {legal_ref}")
        
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
            # state_values의 metadata에 저장
            if isinstance(state_values, dict):
                if "metadata" not in state_values:
                    state_values["metadata"] = {}
                if isinstance(state_values["metadata"], dict):
                    state_values["metadata"]["related_questions"] = llm_questions
                    logger.info(f"[sources_extractor] Saved {len(llm_questions)} related_questions to state metadata")
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
        """state에서 sources_detail 추출 및 정규화"""
        sources_detail = []
        
        if "sources_detail" in state_values:
            sources_detail_list = state_values.get("sources_detail", [])
            if isinstance(sources_detail_list, list):
                sources_detail = sources_detail_list
        
        if not sources_detail and "metadata" in state_values:
            metadata = state_values.get("metadata", {})
            if isinstance(metadata, dict) and "sources_detail" in metadata:
                metadata_sources_detail = metadata.get("sources_detail", [])
                if isinstance(metadata_sources_detail, list):
                    sources_detail = metadata_sources_detail
        
        if not sources_detail and "retrieved_docs" in state_values:
            sources_detail = self._generate_sources_detail_from_retrieved_docs(
                state_values.get("retrieved_docs", [])
            )
        
        if sources_detail:
            return self._normalize_sources_detail(sources_detail)
        
        return []
    
    def _generate_sources_detail_from_retrieved_docs(
        self, 
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """retrieved_docs에서 sources_detail 생성"""
        if not isinstance(retrieved_docs, list) or not retrieved_docs:
            return []
        
        try:
            UnifiedSourceFormatter = None
            try:
                from lawfirm_langgraph.core.generation.formatters.unified_source_formatter import UnifiedSourceFormatter
            except (ImportError, AttributeError):
                try:
                    from core.generation.formatters.unified_source_formatter import UnifiedSourceFormatter
                except (ImportError, AttributeError):
                    try:
                        from lawfirm_langgraph.core.services.unified_source_formatter import UnifiedSourceFormatter
                    except (ImportError, AttributeError):
                        pass
            
            if UnifiedSourceFormatter is None:
                logger.warning("UnifiedSourceFormatter not found, using fallback method for sources_detail")
                return self._generate_sources_detail_fallback(retrieved_docs)
            
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
                    elif metadata.get("decision_id"):
                        source_type = "decision_paragraph"
                    elif (metadata.get("interpretation_number") or 
                          metadata.get("interpretation_serial_number") or
                          metadata.get("expcId") or
                          (metadata.get("org") and metadata.get("title") and metadata.get("response_date")) or
                          (metadata.get("org") and metadata.get("title") and not metadata.get("decision_id"))):
                        source_type = "interpretation_paragraph"
                    elif metadata.get("org") and not metadata.get("title"):
                        source_type = "decision_paragraph"
                    elif metadata.get("statute_name") or metadata.get("law_name") or metadata.get("article_no"):
                        source_type = "statute_article"
                    else:
                        source_type = "regulation_paragraph"
                
                metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                merged_metadata = {**metadata}
                
                for key in ["statute_name", "law_name", "article_no", "article_number", "clause_no", "item_no",
                           "court", "doc_id", "casenames", "org", "title", "announce_date", "decision_date", "response_date"]:
                    if key in doc:
                        merged_metadata[key] = doc[key]
                
                source_info_detail = formatter.format_source(source_type, merged_metadata)
                
                # 원본 문서 URL 생성
                source_id = merged_metadata.get("source_id") or doc.get("source_id") or merged_metadata.get("id")
                original_url = None
                if source_id and source_type:
                    original_url = f"/api/documents/original/{source_type}/{source_id}"
                
                detail_dict = {
                    "name": source_info_detail.name,
                    "type": source_info_detail.type,
                    "url": source_info_detail.url or "",
                    "original_url": original_url,
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
                        doc_id = (
                            meta.get("doc_id") or 
                            merged_metadata.get("doc_id") or 
                            doc.get("doc_id") or 
                            merged_metadata.get("case_id") or
                            ""
                        )
                        detail_dict["case_number"] = doc_id
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
                    elif source_type == "regulation_paragraph":
                        if meta.get("title"):
                            detail_dict["title"] = meta["title"]
                        if meta.get("doc_id"):
                            detail_dict["doc_id"] = meta["doc_id"]
                        if meta.get("url") or meta.get("detail_url"):
                            detail_dict["url"] = meta.get("url") or meta.get("detail_url")
                
                content = doc.get("content") or doc.get("text") or ""
                if content:
                    detail_dict["content"] = content
                
                sources_detail.append(detail_dict)
            
            return sources_detail
        except Exception as e:
            logger.warning(f"Error generating sources_detail from retrieved_docs: {e}")
            return self._generate_sources_detail_fallback(retrieved_docs)
    
    def _generate_sources_detail_fallback(
        self, 
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """retrieved_docs에서 sources_detail 생성 (fallback 방법)"""
        if not isinstance(retrieved_docs, list) or not retrieved_docs:
            return []
        
        sources_detail = []
        for doc in retrieved_docs:
            logger.debug(f"[sources_extractor] Processing doc for sources_detail: type={doc.get('type')}, has_metadata={bool(doc.get('metadata'))}, metadata_keys={list(doc.get('metadata', {}).keys()) if isinstance(doc.get('metadata'), dict) else []}")
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
            
            # 원본 문서 URL 생성
            source_id = merged_metadata.get("source_id") or doc.get("source_id") or merged_metadata.get("id")
            original_url = None
            if source_id and source_type:
                original_url = f"/api/documents/original/{source_type}/{source_id}"
            
            # doc의 최상위 레벨 필드도 확인 (metadata에 없으면 doc에서 가져오기, 강화)
            for key in ["statute_name", "law_name", "article_no", "article_number", "clause_no", "item_no",
                       "court", "doc_id", "casenames", "org", "title", "announce_date", "decision_date", "response_date",
                       "case_id", "decision_id", "interpretation_number", "law_id", "statute_id", "abbrv", "statute_abbrv"]:
                if key in doc and not merged_metadata.get(key):
                    merged_metadata[key] = doc[key]
            
            # 원본 문서 URL 생성
            source_id = merged_metadata.get("source_id") or doc.get("source_id") or merged_metadata.get("id")
            original_url = None
            if source_id and source_type:
                original_url = f"/api/documents/original/{source_type}/{source_id}"
            
            detail_dict = {
                "name": "",
                "type": source_type,
                "url": "",
                "original_url": original_url,
                "metadata": merged_metadata
            }
            
            if source_type == "statute_article":
                statute_name = merged_metadata.get("statute_name") or merged_metadata.get("law_name") or ""
                article_no = merged_metadata.get("article_no") or merged_metadata.get("article_number") or ""
                clause_no = merged_metadata.get("clause_no") or ""
                item_no = merged_metadata.get("item_no") or ""
                
                # 1. law_id나 statute_id가 있으면 데이터베이스에서 조회
                if not statute_name:
                    law_id = merged_metadata.get("law_id") or doc.get("law_id")
                    statute_id = merged_metadata.get("statute_id") or doc.get("statute_id")
                    if law_id or statute_id:
                        db_statute_name = self._get_statute_name_from_db(law_id, statute_id)
                        if db_statute_name:
                            statute_name = db_statute_name
                            merged_metadata["statute_name"] = statute_name
                            detail_dict["metadata"]["statute_name"] = statute_name
                            logger.debug(f"[sources_extractor] Extracted statute name from DB for sources_detail: {statute_name}")
                
                # 2. abbrv나 statute_abbrv가 있으면 사용
                if not statute_name:
                    abbrv = merged_metadata.get("abbrv") or merged_metadata.get("statute_abbrv") or doc.get("abbrv") or doc.get("statute_abbrv")
                    if abbrv:
                        abbrv_to_name = {
                            "민법": "민법",
                            "형법": "형법",
                            "상법": "상법",
                            "민소법": "민사소송법",
                            "형소법": "형사소송법",
                        }
                        if abbrv in abbrv_to_name:
                            statute_name = abbrv_to_name[abbrv]
                            merged_metadata["statute_name"] = statute_name
                            detail_dict["metadata"]["statute_name"] = statute_name
                            logger.debug(f"[sources_extractor] Extracted statute name from abbrv for sources_detail: {statute_name}")
                
                # 3. 같은 세션의 다른 문서(판례)에서 법령명 추출
                if not statute_name:
                    extracted_from_context = self._extract_statute_name_from_context(doc, retrieved_docs, article_no.replace("제", "").replace("조", "").strip() if article_no else "")
                    if extracted_from_context:
                        statute_name = extracted_from_context
                        merged_metadata["statute_name"] = statute_name
                        detail_dict["metadata"]["statute_name"] = statute_name
                        logger.debug(f"[sources_extractor] Extracted statute name from context for sources_detail: {statute_name}")
                
                # 4. metadata에 법령명이 없으면 content에서 추출 시도
                content = doc.get("content") or doc.get("text") or ""
                if not statute_name and content:
                    # 패턴 1: 「법령명」 형식
                    pattern1 = r'「([^」]+)」'
                    match = re.search(pattern1, content)
                    if match:
                        statute_name = match.group(1).strip()
                        merged_metadata["statute_name"] = statute_name
                        detail_dict["metadata"]["statute_name"] = statute_name
                        logger.debug(f"[sources_extractor] Extracted statute name for sources_detail (quotes): {statute_name}")
                    else:
                        # 패턴 2: "법령명" + "법" 형식 (최대 20자로 제한)
                        pattern2 = r'([가-힣]{1,20}법)(?:\s|$|\.|,|\)|「|」|제)'
                        match = re.search(pattern2, content)
                        if match:
                            extracted = match.group(1).strip()
                            # "규칙 또는 처분의 헌법" 같은 긴 문장 제외
                            if len(extracted) <= 20 and extracted.count(' ') <= 2:
                                statute_name = extracted
                                merged_metadata["statute_name"] = statute_name
                                detail_dict["metadata"]["statute_name"] = statute_name
                                logger.debug(f"[sources_extractor] Extracted statute name for sources_detail (pattern): {statute_name}")
                            else:
                                logger.debug(f"[sources_extractor] Skipped too long statute name for sources_detail: {extracted}")
                
                # 5. content에서 상대 참조 해석 ("전3조", "전조" 등)
                if not article_no and content:
                    relative_ref_pattern = r'전\s*(\d+)\s*조|전\s*조'
                    match = re.search(relative_ref_pattern, content)
                    if match:
                        if not statute_name:
                            statute_name = "민법"
                            merged_metadata["statute_name"] = statute_name
                            detail_dict["metadata"]["statute_name"] = statute_name
                        logger.debug(f"[sources_extractor] Found relative reference in content for sources_detail, using default statute: {statute_name}")
                
                # 6. 조문 번호가 없으면 content에서 추출 시도
                if not article_no and content:
                    # 패턴 1: "민법 제XXX조", "형법 제XXX조" 등 (문맥 제거)
                    pattern1 = r'(?:^|\.|,|\)|\(|「|」|에|의|을|를|이|가|은|는|으로|로|에서|부터|까지|와|과|및|또는)\s*(민법|형법|상법|공법|행정법|민사소송법|형사소송법|가족법|상속법|채권법|물권법|계약법|불법행위법|부동산법|임대차보호법|전세권법|근저당법|저당권법|담보법|보증법|연대보증법|보증채무법|채권담보법|민사집행법)\s*제\s*(\d+)\s*조'
                    match = re.search(pattern1, content)
                    if match:
                        if not statute_name:
                            statute_name = match.group(1).strip()
                            merged_metadata["statute_name"] = statute_name
                            detail_dict["metadata"]["statute_name"] = statute_name
                        article_no = f"제{match.group(2)}조"
                        merged_metadata["article_no"] = article_no
                        detail_dict["metadata"]["article_no"] = article_no
                        logger.debug(f"[sources_extractor] Extracted statute and article for sources_detail: {statute_name} {article_no}")
                    else:
                        # 패턴 2: "법령명 제XXX조" (앞에 법령명이 있는 경우)
                        pattern2 = r'([가-힣]{1,20}법)\s*제\s*(\d+)\s*조'
                        match = re.search(pattern2, content)
                        if match:
                            extracted_statute = match.group(1).strip()
                            # 너무 긴 문장 제외
                            if len(extracted_statute) <= 20 and extracted_statute.count(' ') <= 2:
                                if not statute_name:
                                    statute_name = extracted_statute
                                    merged_metadata["statute_name"] = statute_name
                                    detail_dict["metadata"]["statute_name"] = statute_name
                                article_no = f"제{match.group(2)}조"
                                merged_metadata["article_no"] = article_no
                                detail_dict["metadata"]["article_no"] = article_no
                                logger.debug(f"[sources_extractor] Extracted statute and article for sources_detail (pattern2): {statute_name} {article_no}")
                        else:
                            # 패턴 3: "제XXX조" (법령명 없이 조문만)
                            pattern3 = r'제\s*(\d+)\s*조'
                            match = re.search(pattern3, content)
                            if match:
                                article_no = f"제{match.group(1)}조"
                                merged_metadata["article_no"] = article_no
                                detail_dict["metadata"]["article_no"] = article_no
                                # 조문 번호가 있으면 같은 세션의 다른 문서에서 법령명 찾기
                                if not statute_name:
                                    extracted_from_context = self._extract_statute_name_from_context(doc, retrieved_docs, match.group(1))
                                    if extracted_from_context:
                                        statute_name = extracted_from_context
                                    else:
                                        statute_name = "민법"
                                    merged_metadata["statute_name"] = statute_name
                                    detail_dict["metadata"]["statute_name"] = statute_name
                                logger.debug(f"[sources_extractor] Extracted article for sources_detail: {article_no}, statute_name: {statute_name}")
                
                name_parts = []
                if statute_name:
                    name_parts.append(statute_name)
                if article_no:
                    name_parts.append(article_no)
                if clause_no:
                    name_parts.append(f"제{clause_no}항")
                if item_no:
                    name_parts.append(f"제{item_no}호")
                
                detail_dict["name"] = " ".join(name_parts).strip() if name_parts else "법령"
                detail_dict["statute_name"] = statute_name
                detail_dict["article_no"] = article_no
                if clause_no:
                    detail_dict["clause_no"] = clause_no
                if item_no:
                    detail_dict["item_no"] = item_no
            
            elif source_type == "case_paragraph":
                # 판례 정보 추출
                court = merged_metadata.get("court") or ""
                doc_id = merged_metadata.get("doc_id") or merged_metadata.get("case_id") or ""
                casenames = merged_metadata.get("casenames") or ""
                announce_date = merged_metadata.get("announce_date") or ""
                
                content = doc.get("content") or doc.get("text") or ""
                
                # content에서 판례 정보 추출
                if not court and content:
                    # 법원명 패턴: "대법원", "지방법원", "고등법원" 등
                    court_pattern = r'(대법원|지방법원|고등법원|특허법원|가정법원|행정법원|헌법재판소)'
                    match = re.search(court_pattern, content)
                    if match:
                        court = match.group(1)
                        merged_metadata["court"] = court
                        detail_dict["metadata"]["court"] = court
                        logger.debug(f"[sources_extractor] Extracted court from content: {court}")
                
                if not doc_id and content:
                    # 사건번호 패턴: "2021다275611", "2000다72572" 등
                    case_number_pattern = r'(\d{4}[가나다라마바사아자차카타파하]\d+)'
                    match = re.search(case_number_pattern, content)
                    if match:
                        doc_id = match.group(1)
                        merged_metadata["doc_id"] = doc_id
                        detail_dict["metadata"]["doc_id"] = doc_id
                        logger.debug(f"[sources_extractor] Extracted case number from content: {doc_id}")
                
                if not announce_date and content:
                    # 날짜 패턴: "2021. 5. 24.", "2000. 3. 15." 등
                    date_pattern = r'(\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.)'
                    match = re.search(date_pattern, content)
                    if match:
                        announce_date = match.group(1).strip()
                        merged_metadata["announce_date"] = announce_date
                        detail_dict["metadata"]["announce_date"] = announce_date
                        logger.debug(f"[sources_extractor] Extracted announce_date from content: {announce_date}")
                
                # 판례명 생성 (case_number만 표시)
                detail_dict["name"] = doc_id if doc_id else "판례"
                detail_dict["court"] = court
                detail_dict["case_number"] = doc_id
                if casenames:
                    detail_dict["case_name"] = casenames
                
                if doc_id and not detail_dict.get("url"):
                    url = self._generate_case_url(doc_id)
                    if url:
                        detail_dict["url"] = url
            
            elif source_type == "decision_paragraph":
                # 결정례 정보 추출
                org = merged_metadata.get("org") or ""
                doc_id = merged_metadata.get("doc_id") or merged_metadata.get("decision_id") or ""
                decision_date = merged_metadata.get("decision_date") or ""
                result = merged_metadata.get("result") or ""
                
                content = doc.get("content") or doc.get("text") or ""
                
                # content에서 기관명 추출
                if not org and content:
                    # 기관명 패턴: "고용노동부", "법제처" 등
                    org_pattern = r'(고용노동부|법제처|행정안전부|기획재정부|교육부|과학기술정보통신부|외교부|통일부|법무부|국방부|문화체육관광부|농림축산식품부|산업통상자원부|보건복지부|환경부|국토교통부|해양수산부|중소벤처기업부|여성가족부|국세청|관세청|공정거래위원회|금융감독원)'
                    match = re.search(org_pattern, content)
                    if match:
                        org = match.group(1)
                        merged_metadata["org"] = org
                        detail_dict["metadata"]["org"] = org
                        logger.debug(f"[sources_extractor] Extracted org from content: {org}")
                
                # content에서 날짜 추출
                if not decision_date and content:
                    date_pattern = r'(\d{4}\.\s*\d{1,2}\.\s*\d{1,2})'
                    match = re.search(date_pattern, content)
                    if match:
                        decision_date = match.group(1).strip()
                        merged_metadata["decision_date"] = decision_date
                        detail_dict["metadata"]["decision_date"] = decision_date
                        logger.debug(f"[sources_extractor] Extracted decision_date from content: {decision_date}")
                
                # 결정례명 생성 (decision_number만 표시)
                detail_dict["name"] = doc_id if doc_id else "결정례"
                detail_dict["org"] = org
                detail_dict["decision_number"] = doc_id
                if decision_date:
                    detail_dict["decision_date"] = decision_date
                if result:
                    detail_dict["result"] = result
            
            elif source_type == "interpretation_paragraph":
                # 해석례 정보 추출
                org = merged_metadata.get("org") or ""
                title = merged_metadata.get("title") or ""
                doc_id = merged_metadata.get("doc_id") or merged_metadata.get("interpretation_number") or ""
                response_date = merged_metadata.get("response_date") or ""
                
                content = doc.get("content") or doc.get("text") or ""
                
                # content에서 기관명 추출
                if not org and content:
                    # 기관명 패턴
                    org_pattern = r'(고용노동부|법제처|행정안전부|기획재정부|교육부|과학기술정보통신부|외교부|통일부|법무부|국방부|문화체육관광부|농림축산식품부|산업통상자원부|보건복지부|환경부|국토교통부|해양수산부|중소벤처기업부|여성가족부|국세청|관세청|공정거래위원회|금융감독원)'
                    match = re.search(org_pattern, content)
                    if match:
                        org = match.group(1)
                        merged_metadata["org"] = org
                        detail_dict["metadata"]["org"] = org
                        logger.debug(f"[sources_extractor] Extracted org from content for interpretation: {org}")
                
                # content에서 날짜 추출
                if not response_date and content:
                    date_pattern = r'(\d{4}\.\s*\d{1,2}\.\s*\d{1,2})'
                    match = re.search(date_pattern, content)
                    if match:
                        response_date = match.group(1).strip()
                        merged_metadata["response_date"] = response_date
                        detail_dict["metadata"]["response_date"] = response_date
                        logger.debug(f"[sources_extractor] Extracted response_date from content: {response_date}")
                
                # 해석례명 생성 (interpretation_number만 표시)
                detail_dict["name"] = doc_id if doc_id else "해석례"
                detail_dict["org"] = org
                detail_dict["title"] = title
                detail_dict["interpretation_number"] = doc_id
                if response_date:
                    detail_dict["response_date"] = response_date
            
            content = doc.get("content") or doc.get("text") or ""
            if content:
                detail_dict["content"] = content
            
            sources_detail.append(detail_dict)
        
        return sources_detail
    
    def _normalize_sources_detail(self, sources_detail: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """sources_detail을 타입별로 정규화하여 metadata 중첩 구조를 최상위 레벨로 이동"""
        if not isinstance(sources_detail, list):
            return []
        
        normalized = []
        for detail in sources_detail:
            if not isinstance(detail, dict):
                continue
            
            source_type = detail.get("type", "")
            metadata = detail.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            
            normalized_detail = {
                "type": source_type,
                "name": detail.get("name", ""),
                "url": detail.get("url", ""),
                "content": detail.get("content", "")
            }
            
            if source_type == "statute_article":
                normalized_detail["statute_name"] = (
                    detail.get("statute_name") or 
                    metadata.get("statute_name") or 
                    metadata.get("law_name") or 
                    ""
                )
                normalized_detail["article_no"] = (
                    detail.get("article_no") or 
                    metadata.get("article_no") or 
                    metadata.get("article_number") or 
                    ""
                )
                if detail.get("clause_no") or metadata.get("clause_no"):
                    normalized_detail["clause_no"] = detail.get("clause_no") or metadata.get("clause_no")
                if detail.get("item_no") or metadata.get("item_no"):
                    normalized_detail["item_no"] = detail.get("item_no") or metadata.get("item_no")
            
            elif source_type == "case_paragraph":
                normalized_detail["case_number"] = (
                    detail.get("case_number") or 
                    metadata.get("doc_id") or 
                    metadata.get("case_id") or 
                    ""
                )
                if detail.get("case_name") or metadata.get("casenames") or metadata.get("case_name"):
                    normalized_detail["case_name"] = (
                        detail.get("case_name") or 
                        metadata.get("casenames") or 
                        metadata.get("case_name") or 
                        ""
                    )
                if detail.get("court") or metadata.get("court"):
                    normalized_detail["court"] = detail.get("court") or metadata.get("court")
                if detail.get("decision_date") or metadata.get("announce_date") or metadata.get("decision_date"):
                    normalized_detail["decision_date"] = (
                        detail.get("decision_date") or 
                        metadata.get("announce_date") or 
                        metadata.get("decision_date") or 
                        ""
                    )
            
            elif source_type == "decision_paragraph":
                normalized_detail["decision_number"] = (
                    detail.get("decision_number") or 
                    metadata.get("doc_id") or 
                    metadata.get("decision_id") or 
                    ""
                )
                if detail.get("org") or metadata.get("org"):
                    normalized_detail["org"] = detail.get("org") or metadata.get("org")
                if detail.get("decision_date") or metadata.get("decision_date"):
                    normalized_detail["decision_date"] = detail.get("decision_date") or metadata.get("decision_date")
                if detail.get("result") or metadata.get("result"):
                    normalized_detail["result"] = detail.get("result") or metadata.get("result")
            
            elif source_type == "interpretation_paragraph":
                normalized_detail["interpretation_number"] = (
                    detail.get("interpretation_number") or 
                    metadata.get("doc_id") or 
                    metadata.get("interpretation_id") or 
                    ""
                )
                if detail.get("title") or metadata.get("title"):
                    normalized_detail["title"] = detail.get("title") or metadata.get("title")
                if detail.get("org") or metadata.get("org"):
                    normalized_detail["org"] = detail.get("org") or metadata.get("org")
                if detail.get("response_date") or metadata.get("response_date"):
                    normalized_detail["response_date"] = detail.get("response_date") or metadata.get("response_date")
            
            normalized.append(normalized_detail)
        
        return normalized
    
    def _get_sources_by_type(
        self,
        sources_detail: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """sources_detail을 타입별로 그룹화"""
        grouped = {
            "statute_article": [],
            "case_paragraph": [],
            "decision_paragraph": [],
            "interpretation_paragraph": [],
            "regulation_paragraph": []
        }
        
        for detail in sources_detail:
            if not isinstance(detail, dict):
                continue
            
            source_type = detail.get("type", "")
            if source_type in grouped:
                grouped[source_type].append(detail)
        
        return grouped
    
    def _get_sources_by_type_with_reference_statutes(
        self,
        sources_detail: List[Dict[str, Any]]
    ) -> Dict[str, List[Any]]:
        """
        sources_by_type을 생성하고 판례의 참조 법령을 자동으로 추가하는 헬퍼 함수
        """
        try:
            sources_by_type = self._get_sources_by_type(sources_detail) if sources_detail else {
                "statute_article": [],
                "case_paragraph": [],
                "decision_paragraph": [],
                "interpretation_paragraph": [],
                "regulation_paragraph": []
            }
            
            if sources_detail:
                try:
                    extracted_statutes = self._extract_statutes_from_reference_clauses(sources_detail)
                    
                    if extracted_statutes:
                        existing_statutes = sources_by_type.get("statute_article", [])
                        existing_keys = {
                            f"{s.get('statute_name', '')}_{s.get('article_no', '')}_{s.get('clause_no', '')}_{s.get('item_no', '')}"
                            for s in existing_statutes if isinstance(s, dict)
                        }
                        
                        for statute in extracted_statutes:
                            statute_key = f"{statute.get('statute_name', '')}_{statute.get('article_no', '')}_{statute.get('clause_no', '')}_{statute.get('item_no', '')}"
                            if statute_key not in existing_keys:
                                existing_statutes.append(statute)
                                existing_keys.add(statute_key)
                        
                        sources_by_type["statute_article"] = existing_statutes
                except Exception as extract_error:
                    logger.warning(f"Failed to extract reference statutes: {extract_error}", exc_info=True)
                    # 참조 법령 추출 실패해도 기본 sources_by_type은 반환
            
            return sources_by_type
        except Exception as e:
            logger.error(f"Failed to get sources_by_type_with_reference_statutes: {e}", exc_info=True)
            # 최종 예외 발생 시 기본 구조 반환
            return {
                "statute_article": [],
                "case_paragraph": [],
                "decision_paragraph": [],
                "interpretation_paragraph": [],
                "regulation_paragraph": []
            }
    
    def _extract_statutes_from_reference_clauses(
        self, 
        sources_detail: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        판례/결정례/해석례에서 참조조문 추출하여 법령 정보 반환
        데이터베이스에 저장된 참조조문 메타데이터를 사용
        """
        extracted_statutes = []
        seen_statutes = set()
        
        for detail in sources_detail:
            if not isinstance(detail, dict):
                continue
                
            source_type = detail.get("type", "")
            metadata = detail.get("metadata", {})
            
            # 판례, 결정례, 해석례만 처리
            if source_type not in ["case_paragraph", "decision_paragraph", "interpretation_paragraph"]:
                continue
            
            # 데이터베이스에서 참조조문 가져오기
            reference_statutes = self._get_reference_statutes_from_db(detail, source_type)
            
            if not reference_statutes:
                # Fallback: metadata에서 직접 가져오기 (이미 추출되어 있는 경우)
                if isinstance(metadata, dict):
                    reference_statutes = metadata.get("reference_statutes", [])
            
            # 법령 정보 구성
            for statute in reference_statutes:
                if isinstance(statute, str):
                    # JSON 문자열인 경우 파싱
                    import json
                    try:
                        statute = json.loads(statute)
                    except (json.JSONDecodeError, TypeError):
                        continue
                
                if not isinstance(statute, dict):
                    continue
                
                statute_name = statute.get("statute_name")
                article_no = statute.get("article_no")
                
                if not statute_name or not article_no:
                    continue
                
                # 중복 제거
                key = f"{statute_name}_{article_no}_{statute.get('clause_no', '')}_{statute.get('item_no', '')}"
                if key in seen_statutes:
                    continue
                seen_statutes.add(key)
                
                # 법령 정보 구성
                statute_info = {
                    "type": "statute_article",
                    "statute_name": statute_name,
                    "article_no": article_no,
                    "name": f"{statute_name} {article_no}",
                    "source_from": source_type,
                    "source_doc_id": detail.get("doc_id") or detail.get("case_number") or detail.get("decision_number") or detail.get("interpretation_number"),
                }
                
                if statute.get("clause_no"):
                    statute_info["clause_no"] = statute.get("clause_no")
                    statute_info["name"] += f" 제{statute.get('clause_no')}항"
                
                if statute.get("item_no"):
                    statute_info["item_no"] = statute.get("item_no")
                    statute_info["name"] += f" 제{statute.get('item_no')}호"
                
                statute_info["url"] = self._generate_statute_url(statute_name, article_no)
                
                # 법령 본문 조회
                statute_content = self._get_statute_content_from_db(statute_name, article_no, statute.get("clause_no"), statute.get("item_no"))
                if statute_content:
                    statute_info["content"] = statute_content
                
                if "metadata" not in statute_info:
                    statute_info["metadata"] = {}
                statute_info["metadata"]["source_from"] = source_type
                statute_info["metadata"]["source_doc_id"] = statute_info["source_doc_id"]
                
                extracted_statutes.append(statute_info)
        
        return extracted_statutes
    
    def _get_reference_statutes_from_db(
        self, 
        detail: Dict[str, Any], 
        source_type: str
    ) -> List[Dict[str, Any]]:
        """데이터베이스에서 참조조문 가져오기"""
        try:
            import sqlite3
            import os
            import json
            from pathlib import Path
            
            # doc_id 추출
            doc_id = (
                detail.get("doc_id") or 
                detail.get("metadata", {}).get("doc_id") or
                detail.get("case_number") or
                detail.get("decision_number") or
                detail.get("interpretation_number")
            )
            
            if not doc_id:
                return []
            
            # 데이터베이스 경로 가져오기
            db_path = os.getenv("DATABASE_PATH", "./data/lawfirm_v2.db")
            if not os.path.isabs(db_path):
                # 상대 경로인 경우 프로젝트 루트 기준으로 변환
                project_root = Path(__file__).parent.parent.parent
                db_path = str(project_root / db_path)
            
            if not os.path.exists(db_path):
                logger.debug(f"Database not found at {db_path}, skipping reference statutes extraction")
                return []
            
            try:
                with sqlite3.connect(db_path, timeout=5.0) as conn:
                    conn.row_factory = sqlite3.Row
                    if source_type == "case_paragraph":
                        cursor = conn.execute(
                            "SELECT reference_statutes FROM cases WHERE doc_id = ?",
                            (doc_id,)
                        )
                    elif source_type == "decision_paragraph":
                        cursor = conn.execute(
                            "SELECT reference_statutes FROM decisions WHERE doc_id = ?",
                            (doc_id,)
                        )
                    elif source_type == "interpretation_paragraph":
                        cursor = conn.execute(
                            "SELECT reference_statutes FROM interpretations WHERE doc_id = ?",
                            (doc_id,)
                        )
                    else:
                        return []
                    
                    row = cursor.fetchone()
                    if row and row[0]:
                        try:
                            return json.loads(row[0])
                        except (json.JSONDecodeError, TypeError) as json_error:
                            logger.warning(f"Failed to parse reference_statutes JSON for {doc_id}: {json_error}")
                            return []
                
                return []
            except sqlite3.Error as db_error:
                logger.warning(f"Database error getting reference statutes for {doc_id}: {db_error}")
                return []
            
        except Exception as e:
            logger.warning(f"Failed to get reference statutes from DB for {source_type}: {e}", exc_info=True)
            return []
    
    def _get_statute_content_from_db(
        self,
        statute_name: str,
        article_no: str,
        clause_no: Optional[str] = None,
        item_no: Optional[str] = None
    ) -> Optional[str]:
        """데이터베이스에서 법령 본문 조회"""
        try:
            import sqlite3
            import os
            from pathlib import Path
            
            if not statute_name or not article_no:
                return None
            
            # 데이터베이스 경로 가져오기
            db_path = os.getenv("DATABASE_PATH", "./data/lawfirm_v2.db")
            if not os.path.isabs(db_path):
                project_root = Path(__file__).parent.parent.parent
                db_path = str(project_root / db_path)
            
            if not os.path.exists(db_path):
                return None
            
            try:
                with sqlite3.connect(db_path, timeout=5.0) as conn:
                    conn.row_factory = sqlite3.Row
                    
                    # statutes 테이블에서 statute_id 조회
                    cursor = conn.execute(
                        "SELECT id FROM statutes WHERE name = ? LIMIT 1",
                        (statute_name,)
                    )
                    statute_row = cursor.fetchone()
                    if not statute_row:
                        return None
                    
                    statute_id = statute_row['id'] if isinstance(statute_row, sqlite3.Row) else statute_row[0]
                    
                    # statute_articles 테이블에서 본문 조회
                    # 우선순위: 정확한 항/호 매칭 > 항만 매칭 > 조문 전체
                    content = None
                    
                    # 1. 항과 호가 모두 있는 경우: 정확한 항/호 매칭
                    if clause_no and item_no:
                        cursor = conn.execute(
                            """
                            SELECT text FROM statute_articles
                            WHERE statute_id = ? AND article_no = ? AND clause_no = ? AND item_no = ?
                            LIMIT 1
                            """,
                            (statute_id, article_no, clause_no, item_no)
                        )
                        row = cursor.fetchone()
                        if row:
                            content = row['text'] if isinstance(row, sqlite3.Row) else row[0]
                    
                    # 2. 항만 있는 경우: 항 매칭
                    if not content and clause_no:
                        cursor = conn.execute(
                            """
                            SELECT text FROM statute_articles
                            WHERE statute_id = ? AND article_no = ? AND clause_no = ? AND item_no IS NULL
                            LIMIT 1
                            """,
                            (statute_id, article_no, clause_no)
                        )
                        row = cursor.fetchone()
                        if row:
                            content = row['text'] if isinstance(row, sqlite3.Row) else row[0]
                    
                    # 3. 조문 전체 조회 (항/호가 없거나 매칭 실패한 경우)
                    if not content:
                        # 조문 전체를 조회 (여러 항/호가 있으면 합침)
                        cursor = conn.execute(
                            """
                            SELECT GROUP_CONCAT(text, '\n\n') as full_text
                            FROM statute_articles
                            WHERE statute_id = ? AND article_no = ?
                            ORDER BY 
                                CASE WHEN clause_no IS NULL THEN 1 ELSE 0 END,
                                clause_no,
                                CASE WHEN item_no IS NULL THEN 1 ELSE 0 END,
                                item_no
                            """,
                            (statute_id, article_no)
                        )
                        row = cursor.fetchone()
                        if row:
                            content = row['full_text'] if isinstance(row, sqlite3.Row) else row[0]
                    
                    if content and content.strip():
                        return content.strip()
                    
                    return None
            except sqlite3.Error as db_error:
                logger.debug(f"Database error getting statute content: {db_error}")
                return None
            
        except Exception as e:
            logger.debug(f"Failed to get statute content from DB: {e}")
            return None
    
    def _generate_statute_url(self, statute_name: str, article_no: str) -> str:
        """법령 조문 URL 생성"""
        if not statute_name or not article_no:
            return ""
        
        # 법제처 URL 형식
        encoded_name = statute_name.replace(" ", "")
        return f"https://www.law.go.kr/LSW/lsInfoP.do?efYd=&lsiSeq=&ancYnChk=0&ancYnChk=0&lawNm={encoded_name}&articleNo={article_no}"
    
    def _format_legal_reference_from_detail(
        self,
        detail: Dict[str, Any]
    ) -> Optional[str]:
        """sources_detail 항목에서 legal_reference 문자열 생성"""
        if detail.get("type") != "statute_article":
            return None
        
        statute_name = detail.get("statute_name") or ""
        article_no = detail.get("article_no") or ""
        clause_no = detail.get("clause_no")
        item_no = detail.get("item_no")
        
        if not statute_name and not article_no:
            return None
        
        parts = []
        if statute_name:
            parts.append(statute_name)
        if article_no:
            parts.append(article_no)
        if clause_no:
            parts.append(f"제{clause_no}항")
        if item_no:
            parts.append(f"제{item_no}호")
        
        return " ".join(parts) if parts else None
    
    def _extract_legal_references_from_sources_detail_only(
        self,
        sources_detail: List[Dict[str, Any]]
    ) -> List[str]:
        """sources_detail에서만 legal_references 추출 (deprecated용)"""
        legal_refs = []
        seen = set()
        
        for detail in sources_detail:
            if not isinstance(detail, dict):
                continue
            
            legal_ref = self._format_legal_reference_from_detail(detail)
            if legal_ref and legal_ref not in seen:
                legal_refs.append(legal_ref)
                seen.add(legal_ref)
        
        return legal_refs
    
    def _parse_source_string(self, source_str: str) -> Dict[str, Any]:
        """sources 배열의 문자열을 파싱하여 정보 추출"""
        if not isinstance(source_str, str):
            return {}
        
        parsed = {}
        
        case_pattern = r'case_([가-힣0-9]+)'
        case_match = re.search(case_pattern, source_str)
        if case_match:
            parsed["doc_id"] = f"case_{case_match.group(1)}"
            parsed["case_id"] = case_match.group(1)
            parsed["source_type"] = "case_paragraph"
        
        decision_pattern = r'(?:decision_|detc_)([가-힣0-9\-]+)'
        decision_match = re.search(decision_pattern, source_str)
        if decision_match:
            parsed["doc_id"] = f"decision_{decision_match.group(1)}"
            parsed["decision_id"] = decision_match.group(1)
            parsed["source_type"] = "decision_paragraph"
        
        interpretation_pattern = r'(?:interpretation_|interp_|expc_)([가-힣0-9\-]+)'
        interpretation_match = re.search(interpretation_pattern, source_str)
        if interpretation_match:
            matched_id = interpretation_match.group(1)
            if source_str.find("expc_") != -1:
                parsed["doc_id"] = f"expc_{matched_id}"
            elif source_str.find("interp_") != -1:
                parsed["doc_id"] = f"interp_{matched_id}"
            else:
                parsed["doc_id"] = f"interpretation_{matched_id}"
            parsed["interpretation_id"] = matched_id
            parsed["source_type"] = "interpretation_paragraph"
        
        case_name_pattern = r'^(.+?)\s*\(case_'
        case_name_match = re.search(case_name_pattern, source_str)
        if case_name_match:
            parsed["casenames"] = case_name_match.group(1).strip()
            parsed["case_name"] = case_name_match.group(1).strip()
        
        return parsed
    
    def _generate_case_url(self, doc_id: str) -> str:
        """판례 URL 생성"""
        if not doc_id:
            return ""
        
        if doc_id.startswith("case_"):
            return f"http://www.law.go.kr/DRF/lawService.do?target=prec&ID={doc_id}&type=HTML"
        
        return ""
    
    def _generate_decision_url(self, doc_id: str, metadata: Dict[str, Any] = None) -> str:
        """결정례 URL 생성"""
        if not doc_id:
            return ""
        
        if metadata is None:
            metadata = {}
        
        decision_serial_number = (
            metadata.get("decision_serial_number") or 
            metadata.get("헌재결정례일련번호") or 
            metadata.get("결정ID") or
            doc_id
        )
        
        if not decision_serial_number:
            return ""
        
        return f"http://www.law.go.kr/DRF/lawService.do?target=detc&ID={decision_serial_number}&type=HTML"
    
    def _generate_interpretation_url(self, doc_id: str, metadata: Dict[str, Any] = None) -> str:
        """해석례 URL 생성"""
        if not doc_id:
            return ""
        
        if metadata is None:
            metadata = {}
        
        interpretation_serial_number = (
            metadata.get("interpretation_serial_number") or 
            metadata.get("법령해석례일련번호") or 
            metadata.get("해석ID") or 
            metadata.get("expcId") or
            doc_id
        )
        
        if not interpretation_serial_number:
            return ""
        
        return f"http://www.law.go.kr/DRF/lawService.do?target=expc&ID={interpretation_serial_number}&type=HTML"
    
    def _enhance_sources_detail_with_sources(
        self,
        sources: List[str],
        sources_detail: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """sources 배열의 정보를 사용하여 sources_detail 보완"""
        if not sources:
            return sources_detail if sources_detail else []
        
        sources_parsed = {}
        for source_str in sources:
            parsed = self._parse_source_string(source_str)
            if parsed.get("doc_id"):
                sources_parsed[parsed["doc_id"]] = parsed
        
        if not sources_detail:
            sources_detail = []
        
        existing_doc_ids = set()
        for detail in sources_detail:
            if not isinstance(detail, dict):
                continue
            
            source_type = detail.get("type", "")
            metadata = detail.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
                detail["metadata"] = metadata
            
            if source_type == "case_paragraph":
                doc_id = detail.get("case_number") or metadata.get("doc_id") or ""
                
                if doc_id:
                    existing_doc_ids.add(doc_id)
                
                parsed_info = sources_parsed.get(doc_id) if doc_id else None
                if parsed_info:
                    if not metadata.get("doc_id") and parsed_info.get("doc_id"):
                        metadata["doc_id"] = parsed_info["doc_id"]
                        detail["case_number"] = parsed_info["doc_id"]
                    
                    if not metadata.get("casenames") and parsed_info.get("casenames"):
                        metadata["casenames"] = parsed_info["casenames"]
                        detail["case_name"] = parsed_info["casenames"]
                    
                    if not detail.get("name") or detail.get("name") == "판례":
                        detail["name"] = doc_id if doc_id else "판례"
                
                if doc_id and not detail.get("url"):
                    url = self._generate_case_url(doc_id)
                    if url:
                        detail["url"] = url
                
                if not detail.get("case_number") and doc_id:
                    detail["case_number"] = doc_id
                    metadata["doc_id"] = doc_id
                
                if not detail.get("case_name") and metadata.get("casenames"):
                    detail["case_name"] = metadata["casenames"]
            
            elif source_type == "decision_paragraph":
                doc_id = detail.get("decision_number") or metadata.get("doc_id") or ""
                
                if doc_id:
                    existing_doc_ids.add(doc_id)
                
                if not detail.get("name") or detail.get("name") == "결정례":
                    detail["name"] = doc_id if doc_id else "결정례"
                
                if doc_id and not detail.get("url"):
                    url = self._generate_decision_url(doc_id, metadata)
                    if url:
                        detail["url"] = url
                
                if not detail.get("decision_number") and doc_id:
                    detail["decision_number"] = doc_id
                    metadata["doc_id"] = doc_id
            
            elif source_type == "interpretation_paragraph":
                doc_id = detail.get("interpretation_number") or metadata.get("doc_id") or ""
                
                if doc_id:
                    existing_doc_ids.add(doc_id)
                
                if not detail.get("name") or detail.get("name") == "해석례":
                    detail["name"] = doc_id if doc_id else "해석례"
                
                if doc_id and not detail.get("url"):
                    url = self._generate_interpretation_url(doc_id, metadata)
                    if url:
                        detail["url"] = url
                
                if not detail.get("interpretation_number") and doc_id:
                    detail["interpretation_number"] = doc_id
                    metadata["doc_id"] = doc_id
        
        for doc_id, parsed_info in sources_parsed.items():
            if doc_id not in existing_doc_ids:
                source_type_from_sources = parsed_info.get("source_type", "case_paragraph")
                if not source_type_from_sources:
                    if "case_" in doc_id:
                        source_type_from_sources = "case_paragraph"
                    elif "decision_" in doc_id or "detc_" in doc_id:
                        source_type_from_sources = "decision_paragraph"
                    elif "interp_" in doc_id or "expc_" in doc_id or "interpretation_" in doc_id:
                        source_type_from_sources = "interpretation_paragraph"
                
                if source_type_from_sources == "case_paragraph":
                    new_detail = {
                        "name": doc_id if doc_id else "판례",
                        "type": "case_paragraph",
                        "url": self._generate_case_url(doc_id),
                        "metadata": {
                            "doc_id": doc_id,
                            "casenames": parsed_info.get("casenames", ""),
                            "court": "",
                            "announce_date": "",
                            "case_type": None
                        },
                        "case_number": doc_id
                    }
                    if parsed_info.get("casenames"):
                        new_detail["case_name"] = parsed_info["casenames"]
                    sources_detail.append(new_detail)
                    logger.debug(f"[sources_extractor] Added missing case sources_detail from sources: {doc_id}")
                
                elif source_type_from_sources == "decision_paragraph":
                    new_detail = {
                        "name": doc_id if doc_id else "결정례",
                        "type": "decision_paragraph",
                        "url": self._generate_decision_url(doc_id, {}),
                        "metadata": {
                            "doc_id": doc_id,
                            "org": "",
                            "decision_date": "",
                            "result": ""
                        },
                        "decision_number": doc_id
                    }
                    sources_detail.append(new_detail)
                    logger.debug(f"[sources_extractor] Added missing decision sources_detail from sources: {doc_id}")
                
                elif source_type_from_sources == "interpretation_paragraph":
                    new_detail = {
                        "name": doc_id if doc_id else "해석례",
                        "type": "interpretation_paragraph",
                        "url": self._generate_interpretation_url(doc_id, {}),
                        "metadata": {
                            "doc_id": doc_id,
                            "org": "",
                            "title": "",
                            "response_date": ""
                        },
                        "interpretation_number": doc_id
                    }
                    sources_detail.append(new_detail)
                    logger.debug(f"[sources_extractor] Added missing interpretation sources_detail from sources: {doc_id}")
        
        return sources_detail
