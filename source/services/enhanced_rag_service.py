# -*- coding: utf-8 -*-
"""
Enhanced RAG Service with AKLS Integration
기존 RAG 시스템에 AKLS 표준판례 자료를 통합한 향상된 검색 증강 생성 서비스
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import sys
import os

# source 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.langchain_rag_service import LangChainRAGService, RAGResult
from services.akls_search_engine import AKLSSearchEngine, AKLSSearchResult
from services.semantic_search_engine import SemanticSearchEngine

logger = logging.getLogger(__name__)


@dataclass
class EnhancedRAGResult:
    """향상된 RAG 결과 데이터 클래스"""
    response: str
    confidence: float
    sources: List[Dict[str, Any]]
    akls_sources: List[Dict[str, Any]]
    search_type: str
    law_area: Optional[str]
    metadata: Dict[str, Any]


class EnhancedRAGService:
    """AKLS 자료가 통합된 향상된 RAG 서비스"""
    
    def __init__(self, config=None):
        """향상된 RAG 서비스 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 기존 RAG 서비스 초기화
        self.base_rag_service = LangChainRAGService(config)
        
        # AKLS 검색 엔진 초기화
        self.akls_search_engine = AKLSSearchEngine()
        
        # 의미적 검색 엔진 초기화
        self.semantic_search_engine = SemanticSearchEngine()
        
        # 법률 영역 매핑
        self.law_area_mapping = {
            "형법": "criminal_law",
            "상법": "commercial_law",
            "민사소송법": "civil_procedure",
            "행정법": "administrative_law",
            "헌법": "constitutional_law",
            "형사소송법": "criminal_procedure",
            "민법": "civil_law",
            "표준판례": "standard_precedent"
        }
        
        # 쿼리 라우팅 키워드
        self.routing_keywords = {
            "akls": ["표준판례", "대법원", "판례", "사건번호", "법원"],
            "law": ["법령", "법률", "조문", "제조", "법조문"],
            "precedent": ["판례", "사건", "재판", "판결", "소송"]
        }
        
        self.logger.info("Enhanced RAG Service 초기화 완료")
    
    def route_query_to_source(self, query: str) -> Tuple[str, Optional[str]]:
        """쿼리 분석하여 최적의 데이터 소스 선택"""
        query_lower = query.lower()
        
        # AKLS 우선 키워드 확인
        akls_score = sum(1 for keyword in self.routing_keywords["akls"] if keyword in query_lower)
        
        # 법령 키워드 확인
        law_score = sum(1 for keyword in self.routing_keywords["law"] if keyword in query_lower)
        
        # 판례 키워드 확인
        precedent_score = sum(1 for keyword in self.routing_keywords["precedent"] if keyword in query_lower)
        
        # 법률 영역 추출
        law_area = None
        for korean_name, english_code in self.law_area_mapping.items():
            if korean_name in query:
                law_area = english_code
                break
        
        # 라우팅 결정
        if akls_score > 0:
            return "akls_precedents", law_area
        elif law_score > precedent_score:
            return "assembly_laws", law_area
        elif precedent_score > 0:
            return "assembly_precedents", law_area
        else:
            return "hybrid_search", law_area
    
    def search_with_akls(self, query: str, top_k: int = 5) -> EnhancedRAGResult:
        """AKLS 자료를 포함한 통합 검색"""
        try:
            # 쿼리 라우팅
            source_type, law_area = self.route_query_to_source(query)
            
            # 기본 검색 결과
            base_results = []
            akls_results = []
            
            if source_type == "akls_precedents":
                # AKLS 우선 검색
                akls_results = self.akls_search_engine.search(query, top_k=top_k)
                if law_area:
                    akls_results = self.akls_search_engine.search_by_law_area(query, law_area, top_k=top_k)
                
                # 보완적 검색
                try:
                    base_results = self.base_rag_service.process_query(query)
                    if hasattr(base_results, 'sources'):
                        base_results = base_results.sources
                    else:
                        base_results = []
                except:
                    base_results = []
                
            elif source_type == "assembly_laws":
                # 법령 우선 검색
                try:
                    base_results = self.base_rag_service.process_query(query)
                    if hasattr(base_results, 'sources'):
                        base_results = base_results.sources
                    else:
                        base_results = []
                except:
                    base_results = []
                
                # AKLS 보완 검색
                akls_results = self.akls_search_engine.search(query, top_k=top_k//2)
                
            elif source_type == "assembly_precedents":
                # 판례 우선 검색
                try:
                    base_results = self.base_rag_service.process_query(query)
                    if hasattr(base_results, 'sources'):
                        base_results = base_results.sources
                    else:
                        base_results = []
                except:
                    base_results = []
                
                # AKLS 표준판례 보완
                akls_results = self.akls_search_engine.search_standard_precedents(query, top_k=top_k//2)
                
            else:
                # 하이브리드 검색
                try:
                    base_results = self.base_rag_service.process_query(query)
                    if hasattr(base_results, 'sources'):
                        base_results = base_results.sources
                    else:
                        base_results = []
                except:
                    base_results = []
                akls_results = self.akls_search_engine.search(query, top_k=top_k//2)
            
            # 결과 통합 및 랭킹
            combined_sources = self._merge_and_rank_results(base_results, akls_results)
            
            # 답변 생성
            response, confidence = self._generate_enhanced_response(query, combined_sources)
            
            # 소스 분리
            sources = [src for src in combined_sources if src.get("source") != "akls"]
            akls_sources = [src for src in combined_sources if src.get("source") == "akls"]
            
            return EnhancedRAGResult(
                response=response,
                confidence=confidence,
                sources=sources,
                akls_sources=akls_sources,
                search_type=source_type,
                law_area=law_area,
                metadata={
                    "total_sources": len(combined_sources),
                    "base_sources": len(sources),
                    "akls_sources": len(akls_sources),
                    "query_routing": source_type
                }
            )
            
        except Exception as e:
            self.logger.error(f"Enhanced RAG 검색 실패: {e}")
            # 폴백: 기본 RAG 서비스 사용
            try:
                fallback_result = self.base_rag_service.process_query(query)
                return EnhancedRAGResult(
                    response=getattr(fallback_result, 'response', '죄송합니다. 답변을 생성할 수 없습니다.'),
                    confidence=getattr(fallback_result, 'confidence', 0.3),
                    sources=getattr(fallback_result, 'sources', []),
                    akls_sources=[],
                    search_type="fallback",
                    law_area=None,
                    metadata={"error": str(e), "fallback": True}
                )
            except Exception as fallback_error:
                self.logger.error(f"폴백 처리도 실패: {fallback_error}")
                return EnhancedRAGResult(
                    response="죄송합니다. 시스템 오류가 발생했습니다.",
                    confidence=0.1,
                    sources=[],
                    akls_sources=[],
                    search_type="error",
                    law_area=None,
                    metadata={"error": str(e), "fallback_error": str(fallback_error)}
                )
    
    def _merge_and_rank_results(self, base_results: List[Any], akls_results: List[AKLSSearchResult]) -> List[Dict[str, Any]]:
        """검색 결과 통합 및 랭킹"""
        combined_sources = []
        
        # 기본 검색 결과 추가
        for result in base_results:
            if hasattr(result, 'content'):
                # RAGResult 객체인 경우
                source_info = {
                    "content": result.content,
                    "metadata": getattr(result, 'metadata', {}),
                    "score": getattr(result, 'score', 0.5),
                    "source": "assembly",
                    "type": "law_or_precedent"
                }
            elif isinstance(result, dict):
                # 딕셔너리인 경우
                source_info = {
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "score": result.get("score", 0.5),
                    "source": "assembly",
                    "type": "law_or_precedent"
                }
            else:
                # 기타 경우
                source_info = {
                    "content": str(result),
                    "metadata": {},
                    "score": 0.5,
                    "source": "assembly",
                    "type": "law_or_precedent"
                }
            combined_sources.append(source_info)
        
        # AKLS 검색 결과 추가
        for result in akls_results:
            source_info = {
                "content": result.content,
                "metadata": {
                    **result.metadata,
                    "law_area": result.law_area,
                    "case_number": result.case_number,
                    "court": result.court,
                    "date": result.date,
                    "extracted_sections": result.extracted_sections
                },
                "score": result.score,
                "source": "akls",
                "type": "standard_precedent"
            }
            combined_sources.append(source_info)
        
        # 점수순 정렬
        combined_sources.sort(key=lambda x: x["score"], reverse=True)
        
        return combined_sources
    
    def _generate_enhanced_response(self, query: str, sources: List[Dict[str, Any]]) -> Tuple[str, float]:
        """향상된 답변 생성"""
        try:
            # 컨텍스트 구성
            context_parts = []
            akls_context = []
            
            for source in sources[:5]:  # 상위 5개 소스만 사용
                if source["source"] == "akls":
                    # AKLS 표준판례 컨텍스트
                    akls_info = f"[표준판례] {source['metadata'].get('filename', 'N/A')}"
                    if source["metadata"].get("case_number"):
                        akls_info += f" (사건번호: {source['metadata']['case_number']})"
                    if source["metadata"].get("court"):
                        akls_info += f" ({source['metadata']['court']})"
                    
                    akls_context.append(f"{akls_info}\n{source['content'][:500]}...")
                else:
                    # 일반 법률/판례 컨텍스트
                    context_parts.append(f"{source['content'][:500]}...")
            
            # 프롬프트 구성
            if akls_context:
                enhanced_prompt = f"""
다음은 법률 관련 질문에 대한 답변을 위한 자료입니다.

[일반 법률/판례 자료]
{chr(10).join(context_parts)}

[표준판례 자료]
{chr(10).join(akls_context)}

질문: {query}

위 자료를 바탕으로 정확하고 상세한 답변을 제공해주세요. 표준판례가 있는 경우 해당 판례의 내용을 우선적으로 참조하여 답변하세요.
"""
            else:
                enhanced_prompt = f"""
다음은 법률 관련 질문에 대한 답변을 위한 자료입니다.

{chr(10).join(context_parts)}

질문: {query}

위 자료를 바탕으로 정확하고 상세한 답변을 제공해주세요.
"""
            
            # 답변 생성 (기존 RAG 서비스의 생성기 사용)
            response = self.base_rag_service.answer_generator.generate_response(enhanced_prompt)
            
            # 신뢰도 계산 (AKLS 소스가 있으면 높은 신뢰도)
            akls_count = sum(1 for source in sources if source["source"] == "akls")
            confidence = min(0.9, 0.6 + (akls_count * 0.1))
            
            return response, confidence
            
        except Exception as e:
            self.logger.error(f"향상된 답변 생성 실패: {e}")
            # 폴백 답변
            return "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다.", 0.3
    
    def search_by_law_area(self, query: str, law_area: str, top_k: int = 5) -> EnhancedRAGResult:
        """특정 법률 영역에서 검색"""
        # AKLS에서 해당 영역 검색
        akls_results = self.akls_search_engine.search_by_law_area(query, law_area, top_k=top_k)
        
        # 일반 검색도 수행
        base_results = self.base_rag_service.semantic_search(query, top_k=top_k)
        
        # 결과 통합
        combined_sources = self._merge_and_rank_results(base_results, akls_results)
        
        # 답변 생성
        response, confidence = self._generate_enhanced_response(query, combined_sources)
        
        # 소스 분리
        sources = [src for src in combined_sources if src.get("source") != "akls"]
        akls_sources = [src for src in combined_sources if src.get("source") == "akls"]
        
        return EnhancedRAGResult(
            response=response,
            confidence=confidence,
            sources=sources,
            akls_sources=akls_sources,
            search_type="law_area_specific",
            law_area=law_area,
            metadata={
                "total_sources": len(combined_sources),
                "base_sources": len(sources),
                "akls_sources": len(akls_sources),
                "target_law_area": law_area
            }
        )
    
    def get_akls_statistics(self) -> Dict[str, Any]:
        """AKLS 자료 통계 정보"""
        try:
            stats = self.akls_search_engine.get_law_area_statistics()
            return {
                "total_documents": sum(stats.values()),
                "law_area_distribution": stats,
                "index_available": self.akls_search_engine.index is not None
            }
        except Exception as e:
            self.logger.error(f"AKLS 통계 조회 실패: {e}")
            return {"error": str(e)}
    
    def process_query(self, query: str) -> EnhancedRAGResult:
        """메인 쿼리 처리 메서드 (기존 인터페이스 호환)"""
        return self.search_with_akls(query)


def main():
    """Enhanced RAG Service 테스트"""
    import sys
    import os
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Enhanced RAG Service 초기화
        enhanced_rag = EnhancedRAGService()
        
        # AKLS 통계 확인
        stats = enhanced_rag.get_akls_statistics()
        print(f"=== AKLS 통계 ===")
        print(f"총 문서 수: {stats.get('total_documents', 0)}")
        print(f"인덱스 사용 가능: {stats.get('index_available', False)}")
        
        if "law_area_distribution" in stats:
            print(f"\n법률 영역별 문서 수:")
            for area, count in stats["law_area_distribution"].items():
                korean_name = enhanced_rag.law_area_mapping.get(area, area)
                print(f"  {korean_name}: {count}개")
        
        # 테스트 쿼리
        test_queries = [
            "계약 해지에 대한 판례",
            "형법 제250조",
            "손해배상 책임",
            "대법원 표준판례"
        ]
        
        print(f"\n=== Enhanced RAG 테스트 ===")
        
        for query in test_queries:
            print(f"\n--- 쿼리: '{query}' ---")
            
            try:
                result = enhanced_rag.search_with_akls(query, top_k=3)
                
                print(f"검색 유형: {result.search_type}")
                print(f"법률 영역: {result.law_area}")
                print(f"신뢰도: {result.confidence:.3f}")
                print(f"총 소스 수: {result.metadata['total_sources']}")
                print(f"일반 소스: {result.metadata['base_sources']}")
                print(f"AKLS 소스: {result.metadata['akls_sources']}")
                print(f"답변: {result.response[:200]}...")
                
            except Exception as e:
                print(f"쿼리 처리 실패: {e}")
        
    except Exception as e:
        print(f"Enhanced RAG Service 테스트 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
