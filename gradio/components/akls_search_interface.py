# -*- coding: utf-8 -*-
"""
AKLS Search Interface Component for LawFirmAI
법률전문대학원협의회 표준판례 검색 전용 인터페이스 컴포넌트
"""

import gradio as gr
import logging
import sys
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.services.enhanced_rag_service import EnhancedRAGService
from source.services.akls_search_engine import AKLSSearchEngine

logger = logging.getLogger(__name__)


class AKLSSearchInterface:
    """AKLS 검색 인터페이스 클래스"""
    
    def __init__(self):
        """AKLS 검색 인터페이스 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 서비스 초기화
        self.enhanced_rag = None
        self.akls_search_engine = None
        
        # 법률 영역 옵션
        self.law_area_options = [
            ("전체", "all"),
            ("형법", "criminal_law"),
            ("상법", "commercial_law"),
            ("민법", "civil_law"),
            ("민사소송법", "civil_procedure"),
            ("형사소송법", "criminal_procedure"),
            ("행정법", "administrative_law"),
            ("헌법", "constitutional_law")
        ]
        
        # 사건 유형 옵션
        self.case_type_options = [
            ("전체", "all"),
            ("민사 (다)", "다"),
            ("형사 (고)", "고"),
            ("가사 (드)", "드"),
            ("행정 (구)", "구"),
            ("특허 (허)", "허")
        ]
        
        self._initialize_services()
    
    def _initialize_services(self):
        """서비스 초기화"""
        try:
            self.enhanced_rag = EnhancedRAGService()
            self.akls_search_engine = AKLSSearchEngine()
            self.logger.info("AKLS 서비스 초기화 완료")
        except Exception as e:
            self.logger.error(f"AKLS 서비스 초기화 실패: {e}")
    
    def search_akls_precedents(self, query: str, law_area: str, case_type: str, top_k: int) -> Tuple[str, List[List[str]]]:
        """AKLS 표준판례 검색"""
        try:
            if not query.strip():
                return "검색어를 입력해주세요.", []
            
            # 검색 실행
            if law_area != "all":
                result = self.enhanced_rag.search_by_law_area(query, law_area, top_k)
            else:
                result = self.enhanced_rag.search_with_akls(query, top_k)
            
            # 사건 유형 필터링
            if case_type != "all":
                filtered_akls_sources = []
                for source in result.akls_sources:
                    if source.get("metadata", {}).get("case_number"):
                        case_number = source["metadata"]["case_number"]
                        if case_type in case_number:
                            filtered_akls_sources.append(source)
                result.akls_sources = filtered_akls_sources
            
            # 결과 포맷팅
            response_text = self._format_search_response(result)
            
            # 테이블 데이터 생성
            table_data = self._create_result_table(result.akls_sources)
            
            return response_text, table_data
            
        except Exception as e:
            self.logger.error(f"AKLS 검색 실패: {e}")
            return f"검색 중 오류가 발생했습니다: {str(e)}", []
    
    def _format_search_response(self, result) -> str:
        """검색 결과 응답 포맷팅"""
        response_parts = []
        
        # 기본 정보
        response_parts.append(f"**검색 결과** (총 {result.metadata['total_sources']}개 소스)")
        response_parts.append(f"- 일반 법률/판례: {result.metadata['base_sources']}개")
        response_parts.append(f"- AKLS 표준판례: {result.metadata['akls_sources']}개")
        response_parts.append(f"- 검색 유형: {result.search_type}")
        response_parts.append(f"- 신뢰도: {result.confidence:.2f}")
        
        if result.law_area:
            law_area_korean = dict(self.law_area_options).get(result.law_area, result.law_area)
            response_parts.append(f"- 법률 영역: {law_area_korean}")
        
        response_parts.append("")
        response_parts.append("**답변:**")
        response_parts.append(result.response)
        
        return "\n".join(response_parts)
    
    def _create_result_table(self, akls_sources: List[Dict[str, Any]]) -> List[List[str]]:
        """검색 결과 테이블 생성"""
        if not akls_sources:
            return []
        
        table_data = []
        for i, source in enumerate(akls_sources, 1):
            metadata = source.get("metadata", {})
            
            # 기본 정보
            filename = metadata.get("filename", "N/A")
            case_number = metadata.get("case_number", "N/A")
            court = metadata.get("court", "N/A")
            date = metadata.get("date", "N/A")
            law_area = metadata.get("law_area", "N/A")
            score = f"{source.get('score', 0):.3f}"
            
            # 법률 영역 한국어 변환
            law_area_korean = dict(self.law_area_options).get(law_area, law_area)
            
            # 내용 미리보기
            content_preview = source.get("content", "")[:100] + "..." if len(source.get("content", "")) > 100 else source.get("content", "")
            
            table_data.append([
                str(i),
                filename,
                case_number,
                court,
                date,
                law_area_korean,
                score,
                content_preview
            ])
        
        return table_data
    
    def get_akls_statistics(self) -> str:
        """AKLS 통계 정보 조회"""
        try:
            stats = self.enhanced_rag.get_akls_statistics()
            
            if "error" in stats:
                return f"통계 조회 실패: {stats['error']}"
            
            stats_text = []
            stats_text.append("**AKLS 표준판례 통계**")
            stats_text.append(f"- 총 문서 수: {stats.get('total_documents', 0)}개")
            stats_text.append(f"- 인덱스 사용 가능: {'예' if stats.get('index_available', False) else '아니오'}")
            
            if "law_area_distribution" in stats:
                stats_text.append("\n**법률 영역별 문서 수:**")
                for area, count in stats["law_area_distribution"].items():
                    korean_name = dict(self.law_area_options).get(area, area)
                    stats_text.append(f"- {korean_name}: {count}개")
            
            return "\n".join(stats_text)
            
        except Exception as e:
            self.logger.error(f"AKLS 통계 조회 실패: {e}")
            return f"통계 조회 중 오류가 발생했습니다: {str(e)}"
    
    def create_interface(self) -> gr.Tab:
        """AKLS 검색 인터페이스 생성"""
        with gr.Tab("📚 AKLS 표준판례 검색") as akls_tab:
            gr.Markdown("""
            # 법률전문대학원협의회 표준판례 검색
            
            법률전문대학원협의회에서 제공하는 표준판례 자료를 검색할 수 있습니다.
            """)
            
            with gr.Row():
                with gr.Column(scale=3):
                    # 검색 입력
                    search_query = gr.Textbox(
                        label="검색어",
                        placeholder="예: 계약 해지, 손해배상, 형법 제250조",
                        lines=2
                    )
                    
                    with gr.Row():
                        law_area_dropdown = gr.Dropdown(
                            choices=[(korean, english) for korean, english in self.law_area_options],
                            value="all",
                            label="법률 영역",
                            info="특정 법률 영역에서만 검색"
                        )
                        
                        case_type_dropdown = gr.Dropdown(
                            choices=[(korean, english) for korean, english in self.case_type_options],
                            value="all",
                            label="사건 유형",
                            info="특정 사건 유형에서만 검색"
                        )
                        
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="검색 결과 수",
                            info="최대 검색 결과 개수"
                        )
                    
                    # 검색 버튼
                    search_button = gr.Button("🔍 검색", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    # 통계 정보
                    stats_button = gr.Button("📊 통계 조회", variant="secondary")
                    stats_output = gr.Markdown()
            
            # 검색 결과
            with gr.Row():
                with gr.Column(scale=2):
                    # 답변 결과
                    response_output = gr.Markdown(
                        label="검색 결과",
                        value="검색어를 입력하고 검색 버튼을 클릭하세요."
                    )
                
                with gr.Column(scale=3):
                    # 결과 테이블
                    results_table = gr.Dataframe(
                        headers=["순번", "파일명", "사건번호", "법원", "날짜", "법률영역", "점수", "내용 미리보기"],
                        datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
                        label="상세 검색 결과",
                        interactive=False,
                        wrap=True
                    )
            
            # 이벤트 핸들러
            search_button.click(
                fn=self.search_akls_precedents,
                inputs=[search_query, law_area_dropdown, case_type_dropdown, top_k_slider],
                outputs=[response_output, results_table]
            )
            
            stats_button.click(
                fn=self.get_akls_statistics,
                inputs=[],
                outputs=[stats_output]
            )
            
            # 예시 쿼리
            gr.Examples(
                examples=[
                    ["계약 해지에 대한 판례", "all", "all", 5],
                    ["손해배상 책임", "civil_law", "다", 3],
                    ["형법 제250조", "criminal_law", "고", 5],
                    ["대법원 표준판례", "all", "all", 10],
                    ["민사소송법", "civil_procedure", "다", 5]
                ],
                inputs=[search_query, law_area_dropdown, case_type_dropdown, top_k_slider],
                label="예시 검색어"
            )
        
        return akls_tab


def create_akls_interface() -> gr.Tab:
    """AKLS 인터페이스 생성 함수"""
    interface = AKLSSearchInterface()
    return interface.create_interface()


if __name__ == "__main__":
    # 테스트용 실행
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 인터페이스 생성 및 실행
    interface = create_akls_interface()
    
    # Gradio 앱 실행
    app = gr.Blocks(title="AKLS 표준판례 검색")
    with app:
        interface
    
    app.launch(server_name="0.0.0.0", server_port=7861)
