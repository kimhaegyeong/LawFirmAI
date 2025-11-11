"""
Sources 추출기
LangGraph state와 메시지 metadata에서 sources를 추출합니다.
"""
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class SourcesExtractor:
    """Sources 추출기"""
    
    def __init__(self, workflow_service, session_service):
        self.workflow_service = workflow_service
        self.session_service = session_service
    
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
        
        return {"sources": [], "legal_references": [], "sources_detail": [], "related_questions": []}
    
    def _extract_from_message(self, msg: Dict[str, Any]) -> Dict[str, List[Any]]:
        """단일 메시지에서 sources 추출"""
        metadata = msg.get("metadata", {})
        if not isinstance(metadata, dict):
            return {"sources": [], "legal_references": [], "sources_detail": [], "related_questions": []}
        
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
            return {"sources": [], "legal_references": [], "sources_detail": [], "related_questions": []}
        
        try:
            config = {"configurable": {"thread_id": session_id}}
            final_state = await self.workflow_service.app.aget_state(config)
            
            if not final_state or not final_state.values:
                logger.warning(f"No state found for session_id: {session_id}")
                return {"sources": [], "legal_references": [], "sources_detail": [], "related_questions": []}
            
            state_values = final_state.values
            
            # sources 추출
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
            return {"sources": [], "legal_references": [], "sources_detail": [], "related_questions": []}
    
    def _extract_sources(self, state_values: Dict[str, Any]) -> List[str]:
        """state에서 sources 추출"""
        # 1. 최상위 레벨
        if "sources" in state_values:
            sources_list = state_values.get("sources", [])
            if isinstance(sources_list, list):
                sources = [str(s) for s in sources_list if s and str(s).strip()]
                if sources:
                    logger.debug(f"Found {len(sources)} sources at top level")
                    return sources
        
        # 2. metadata 안에서
        if "metadata" in state_values:
            metadata = state_values.get("metadata", {})
            if isinstance(metadata, dict) and "sources" in metadata:
                metadata_sources = metadata.get("sources", [])
                if isinstance(metadata_sources, list):
                    sources = [str(s) for s in metadata_sources if s and str(s).strip()]
                    if sources:
                        logger.debug(f"Found {len(sources)} sources in metadata")
                        return sources
        
        # 3. retrieved_docs에서 (prepare_final_response_part와 동일한 로직 사용)
        if "retrieved_docs" in state_values:
            retrieved_docs = state_values.get("retrieved_docs", [])
            if isinstance(retrieved_docs, list):
                seen_sources = set()
                sources = []
                
                for doc in retrieved_docs:
                    if not isinstance(doc, dict):
                        continue
                    
                    source = None
                    source_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("source_type", "")
                    metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                    
                    # prepare_final_response_part와 동일한 로직으로 source 생성
                    # 1. statute_article (법령 조문) 처리
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
                            
                            source_parts = [statute_name]
                            if article_no:
                                source_parts.append(article_no)
                            if clause_no:
                                source_parts.append(f"제{clause_no}항")
                            if item_no:
                                source_parts.append(f"제{item_no}호")
                            
                            source = " ".join(source_parts)
                    
                    # 2. case_paragraph (판례) 처리
                    elif source_type == "case_paragraph":
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
                            source = " ".join(source_parts) if source_parts else None
                    
                    # 3. decision_paragraph (결정례) 처리
                    elif source_type == "decision_paragraph":
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
                            source = " ".join(source_parts) if source_parts else None
                    
                    # 4. interpretation_paragraph (해석례) 처리
                    elif source_type == "interpretation_paragraph":
                        org = doc.get("org") or metadata.get("org")
                        title = doc.get("title") or metadata.get("title")
                        
                        if org or title:
                            source_parts = []
                            if org:
                                source_parts.append(org)
                            if title:
                                source_parts.append(title)
                            source = " ".join(source_parts)
                    
                    # 5. 기존 로직 (source_type이 없는 경우 또는 위에서 source를 찾지 못한 경우)
                    if not source:
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
                            invalid_sources = ["semantic", "keyword", "unknown", "fts", "vector", "search", "text2sql", ""]
                            if source_lower not in invalid_sources and len(source_lower) >= 2:
                                source = source_raw.strip()
                        
                        if not source:
                            source = (
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
                        
                        if not source:
                            content = doc.get("content", "") or doc.get("text", "")
                            if isinstance(content, str) and content:
                                import re
                                law_pattern = re.search(r'([가-힣]+법)\s*(?:제\d+조)?', content[:200])
                                if law_pattern:
                                    source = law_pattern.group(1)
                    
                    # source가 있으면 추가 (유효성 검증)
                    if source:
                        if isinstance(source, str):
                            source_str = source.strip()
                        else:
                            try:
                                source_str = str(source).strip()
                            except Exception:
                                source_str = None
                        
                        if source_str:
                            source_lower = source_str.lower().strip()
                            invalid_sources = ["semantic", "keyword", "unknown", "fts", "vector", "search", "text2sql", ""]
                            is_valid_source = False
                            if source_type and source_type in ["statute_article", "case_paragraph", "decision_paragraph", "interpretation_paragraph"]:
                                if source_lower not in invalid_sources and len(source_lower) >= 1:
                                    is_valid_source = True
                            else:
                                if source_lower not in invalid_sources and len(source_lower) >= 2:
                                    is_valid_source = True
                            
                            if is_valid_source:
                                if source_str not in seen_sources and source_str != "Unknown":
                                    sources.append(source_str)
                                    seen_sources.add(source_str)
                
                if sources:
                    logger.info(f"Extracted {len(sources)} sources from retrieved_docs")
                    return sources
        
        return []
    
    def _extract_legal_references(self, state_values: Dict[str, Any]) -> List[str]:
        """state에서 legal_references 추출"""
        # 1. 최상위 레벨
        if "legal_references" in state_values:
            legal_refs = state_values.get("legal_references", [])
            if isinstance(legal_refs, list):
                legal_refs_list = [str(r) for r in legal_refs if r and str(r).strip()]
                if legal_refs_list:
                    logger.debug(f"Found {len(legal_refs_list)} legal_references at top level")
                    return legal_refs_list
        
        # 2. metadata 안에서
        if "metadata" in state_values:
            metadata = state_values.get("metadata", {})
            if isinstance(metadata, dict) and "legal_references" in metadata:
                metadata_legal_refs = metadata.get("legal_references", [])
                if isinstance(metadata_legal_refs, list):
                    legal_refs_list = [str(r) for r in metadata_legal_refs if r and str(r).strip()]
                    if legal_refs_list:
                        logger.debug(f"Found {len(legal_refs_list)} legal_references in metadata")
                        return legal_refs_list
        
        # 3. retrieved_docs에서 statute_article 타입 문서 추출
        if "retrieved_docs" in state_values:
            retrieved_docs = state_values.get("retrieved_docs", [])
            if isinstance(retrieved_docs, list):
                seen_legal_refs = set()
                legal_refs = []
                
                for doc in retrieved_docs:
                    if not isinstance(doc, dict):
                        continue
                    
                    source_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("source_type", "")
                    metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                    
                    # statute_article 타입 문서만 legal_references에 추가
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
                            
                            # 법령 참조 형식으로 변환
                            legal_ref_parts = [statute_name]
                            if article_no:
                                legal_ref_parts.append(article_no)
                            if clause_no:
                                legal_ref_parts.append(f"제{clause_no}항")
                            if item_no:
                                legal_ref_parts.append(f"제{item_no}호")
                            
                            legal_ref = " ".join(legal_ref_parts)
                            
                            # 중복 제거
                            if legal_ref and legal_ref not in seen_legal_refs:
                                legal_refs.append(legal_ref)
                                seen_legal_refs.add(legal_ref)
                
                if legal_refs:
                    logger.info(f"Extracted {len(legal_refs)} legal_references from retrieved_docs")
                    return legal_refs
        
        return []
    
    def _extract_related_questions(self, state_values: Dict[str, Any]) -> List[str]:
        """state에서 related_questions 추출"""
        # 1. metadata 안에서
        if "metadata" in state_values:
            metadata = state_values.get("metadata", {})
            if isinstance(metadata, dict) and "related_questions" in metadata:
                related_questions = metadata.get("related_questions", [])
                if isinstance(related_questions, list):
                    questions = [str(q).strip() for q in related_questions if q and str(q).strip()]
                    if questions:
                        logger.debug(f"Found {len(questions)} related_questions in metadata")
                        return questions
        
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
        
        # retrieved_docs에서 직접 생성
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
                # source_type이 없으면 metadata에서 추론
                if not source_type:
                    metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                    # case_id, court, casenames가 있으면 case_paragraph
                    if metadata.get("case_id") or metadata.get("court") or metadata.get("casenames"):
                        source_type = "case_paragraph"
                    # decision_id, org가 있으면 decision_paragraph
                    elif metadata.get("decision_id") or metadata.get("org"):
                        source_type = "decision_paragraph"
                    # interpretation_number, org가 있으면 interpretation_paragraph
                    elif metadata.get("interpretation_number") or (metadata.get("org") and metadata.get("title")):
                        source_type = "interpretation_paragraph"
                    # statute_name, article_no가 있으면 statute_article
                    elif metadata.get("statute_name") or metadata.get("law_name") or metadata.get("article_no"):
                        source_type = "statute_article"
                    else:
                        continue
                
                metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                merged_metadata = {**metadata}
                
                # doc의 필드를 metadata에 병합
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
                
                # metadata의 정보를 최상위 레벨로 추출
                if source_info_detail.metadata:
                    meta = source_info_detail.metadata
                    
                    # 법령 조문인 경우
                    if source_type == "statute_article":
                        if meta.get("statute_name"):
                            detail_dict["statute_name"] = meta["statute_name"]
                        if meta.get("article_no"):
                            detail_dict["article_no"] = meta["article_no"]
                        if meta.get("clause_no"):
                            detail_dict["clause_no"] = meta["clause_no"]
                        if meta.get("item_no"):
                            detail_dict["item_no"] = meta["item_no"]
                    
                    # 판례인 경우
                    elif source_type == "case_paragraph":
                        if meta.get("doc_id"):
                            detail_dict["case_number"] = meta["doc_id"]
                        if meta.get("court"):
                            detail_dict["court"] = meta["court"]
                        if meta.get("casenames"):
                            detail_dict["case_name"] = meta["casenames"]
                    
                    # 결정례인 경우
                    elif source_type == "decision_paragraph":
                        if meta.get("doc_id"):
                            detail_dict["decision_number"] = meta["doc_id"]
                        if meta.get("org"):
                            detail_dict["org"] = meta["org"]
                        if meta.get("decision_date"):
                            detail_dict["decision_date"] = meta["decision_date"]
                        if meta.get("result"):
                            detail_dict["result"] = meta["result"]
                    
                    # 해석례인 경우
                    elif source_type == "interpretation_paragraph":
                        if meta.get("doc_id"):
                            detail_dict["interpretation_number"] = meta["doc_id"]
                        if meta.get("org"):
                            detail_dict["org"] = meta["org"]
                        if meta.get("title"):
                            detail_dict["title"] = meta["title"]
                        if meta.get("response_date"):
                            detail_dict["response_date"] = meta["response_date"]
                
                # 상세본문 추가
                content = doc.get("content") or doc.get("text") or ""
                if content:
                    detail_dict["content"] = content
                
                sources_detail.append(detail_dict)
            
            return sources_detail
        except Exception as e:
            logger.warning(f"Error generating sources_detail from retrieved_docs: {e}")
            return []

