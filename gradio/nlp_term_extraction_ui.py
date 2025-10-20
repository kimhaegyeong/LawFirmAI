import gradio as gr
import sys
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.services.nlp_term_extraction_pipeline import NLPTermExtractionPipeline
from source.services.domain_specific_extractor import DomainSpecificExtractor, LegalDomain

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPTermExtractionUI:
    """NLP 기반 법률 용어 추출 UI"""
    
    def __init__(self):
        self.pipeline = None
        self.domain_extractor = DomainSpecificExtractor()
        self.logger = logging.getLogger(__name__)
        
        # 결과 저장 디렉토리 생성
        Path("results").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
    
    def initialize_pipeline(self, gemini_api_key: str = None, use_gemini: bool = False):
        """파이프라인 초기화"""
        try:
            if use_gemini and gemini_api_key:
                self.pipeline = NLPTermExtractionPipeline(gemini_api_key=gemini_api_key)
                self.logger.info("Gemini API를 사용한 파이프라인 초기화 완료")
            else:
                self.pipeline = NLPTermExtractionPipeline()
                self.logger.info("기본 파이프라인 초기화 완료")
            
            return "✅ 파이프라인 초기화 완료"
        except Exception as e:
            self.logger.error(f"파이프라인 초기화 실패: {e}")
            return f"❌ 파이프라인 초기화 실패: {str(e)}"
    
    def extract_terms_from_text(self, 
                               text: str, 
                               use_gemini: bool = False,
                               gemini_api_key: str = None) -> Dict[str, Any]:
        """텍스트에서 용어 추출"""
        if not self.pipeline:
            return {"error": "파이프라인이 초기화되지 않았습니다."}
        
        if not text.strip():
            return {"error": "텍스트를 입력해주세요."}
        
        try:
            # 파이프라인 실행
            results = self.pipeline.run_full_pipeline(
                texts=[text],
                use_gemini=use_gemini
            )
            
            # 도메인별 강화
            enhanced_results = self.domain_extractor.enhance_term_extraction(
                text, 
                results.get("extraction", {}).get("all_extracted_terms", [])
            )
            
            # 결과 정리
            extracted_terms = results.get("extraction", {}).get("all_extracted_terms", [])
            validated_terms = results.get("validation", {}).get("validated_terms", [])
            
            return {
                "extracted_terms": extracted_terms,
                "validated_terms": validated_terms,
                "enhanced_terms": enhanced_results["enhanced_terms"],
                "primary_domain": enhanced_results["primary_domain"].value,
                "domain_confidence": enhanced_results["domain_confidence"],
                "domain_terms": {k.value: v for k, v in enhanced_results["domain_terms"].items()},
                "weighted_terms": enhanced_results["weighted_terms"],
                "performance": results.get("performance", {}),
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"용어 추출 중 오류: {e}")
            return {"error": f"용어 추출 중 오류가 발생했습니다: {str(e)}"}
    
    def batch_extract_terms(self, 
                           texts: List[str], 
                           use_gemini: bool = False,
                           gemini_api_key: str = None) -> Dict[str, Any]:
        """배치 용어 추출"""
        if not self.pipeline:
            return {"error": "파이프라인이 초기화되지 않았습니다."}
        
        if not texts or not any(text.strip() for text in texts):
            return {"error": "유효한 텍스트를 입력해주세요."}
        
        try:
            # 빈 텍스트 제거
            valid_texts = [text.strip() for text in texts if text.strip()]
            
            # 파이프라인 실행
            results = self.pipeline.run_full_pipeline(
                texts=valid_texts,
                use_gemini=use_gemini
            )
            
            # 결과 저장
            self.pipeline.save_results(results, "results/batch_extraction_results.json")
            
            return {
                "total_texts": len(valid_texts),
                "extracted_terms": results.get("extraction", {}).get("all_extracted_terms", []),
                "validated_terms": results.get("validation", {}).get("validated_terms", []),
                "performance": results.get("performance", {}),
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"배치 용어 추출 중 오류: {e}")
            return {"error": f"배치 용어 추출 중 오류가 발생했습니다: {str(e)}"}
    
    def analyze_domain_distribution(self, text: str) -> Dict[str, Any]:
        """도메인 분포 분석"""
        try:
            domain_scores = self.domain_extractor.classify_domain_by_keywords(text)
            primary_domain, confidence = self.domain_extractor.get_primary_domain(text)
            
            # 도메인별 용어 추출
            domain_terms = self.domain_extractor.extract_all_domain_terms(text)
            
            return {
                "domain_scores": {k.value: v for k, v in domain_scores.items()},
                "primary_domain": primary_domain.value,
                "confidence": confidence,
                "domain_terms": {k.value: v for k, v in domain_terms.items()},
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"도메인 분석 중 오류: {e}")
            return {"error": f"도메인 분석 중 오류가 발생했습니다: {str(e)}"}
    
    def create_interface(self):
        """Gradio 인터페이스 생성"""
        with gr.Blocks(title="NLP 기반 법률 용어 추출 시스템", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🏛️ NLP 기반 법률 용어 추출 시스템")
            gr.Markdown("자연어 처리 기술을 활용하여 법률 문서에서 용어를 자동으로 추출하고 분류합니다.")
            
            with gr.Tabs():
                # 단일 텍스트 처리 탭
                with gr.Tab("📝 단일 텍스트 처리"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            text_input = gr.Textbox(
                                label="법률 텍스트 입력",
                                placeholder="법률 관련 텍스트를 입력하세요...",
                                lines=10,
                                max_lines=20
                            )
                            
                            with gr.Row():
                                use_gemini = gr.Checkbox(
                                    label="Gemini API 사용",
                                    value=False,
                                    info="더 정확한 검증을 위해 Gemini API를 사용합니다"
                                )
                                gemini_api_key = gr.Textbox(
                                    label="Gemini API 키",
                                    placeholder="GEMINI_API_KEY 또는 GOOGLE_API_KEY",
                                    type="password",
                                    visible=False
                                )
                            
                            extract_btn = gr.Button("🔍 용어 추출", variant="primary")
                        
                        with gr.Column(scale=1):
                            init_btn = gr.Button("⚙️ 파이프라인 초기화", variant="secondary")
                            init_status = gr.Textbox(label="초기화 상태", interactive=False)
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📊 추출 결과")
                            extracted_terms_output = gr.Textbox(
                                label="추출된 용어",
                                lines=5,
                                interactive=False
                            )
                            
                            validated_terms_output = gr.Textbox(
                                label="검증된 용어",
                                lines=5,
                                interactive=False
                            )
                        
                        with gr.Column():
                            gr.Markdown("### 🎯 도메인 분석")
                            primary_domain_output = gr.Textbox(
                                label="주요 도메인",
                                interactive=False
                            )
                            
                            domain_confidence_output = gr.Textbox(
                                label="도메인 신뢰도",
                                interactive=False
                            )
                    
                    with gr.Row():
                        gr.Markdown("### 📈 성능 지표")
                        performance_output = gr.Textbox(
                            label="성능 보고서",
                            lines=10,
                            interactive=False
                        )
                
                # 배치 처리 탭
                with gr.Tab("📚 배치 처리"):
                    with gr.Row():
                        with gr.Column():
                            batch_texts_input = gr.Textbox(
                                label="배치 텍스트 입력 (한 줄에 하나씩)",
                                placeholder="첫 번째 법률 텍스트\n두 번째 법률 텍스트\n...",
                                lines=15,
                                max_lines=30
                            )
                            
                            with gr.Row():
                                batch_use_gemini = gr.Checkbox(
                                    label="Gemini API 사용",
                                    value=False
                                )
                                batch_gemini_key = gr.Textbox(
                                    label="Gemini API 키",
                                    placeholder="GEMINI_API_KEY 또는 GOOGLE_API_KEY",
                                    type="password",
                                    visible=False
                                )
                            
                            batch_extract_btn = gr.Button("🚀 배치 처리 시작", variant="primary")
                        
                        with gr.Column():
                            batch_results_output = gr.Textbox(
                                label="배치 처리 결과",
                                lines=20,
                                interactive=False
                            )
                
                # 도메인 분석 탭
                with gr.Tab("🔍 도메인 분석"):
                    with gr.Row():
                        with gr.Column():
                            domain_text_input = gr.Textbox(
                                label="분석할 텍스트",
                                placeholder="도메인 분류를 원하는 텍스트를 입력하세요...",
                                lines=8
                            )
                            
                            analyze_domain_btn = gr.Button("🎯 도메인 분석", variant="primary")
                        
                        with gr.Column():
                            domain_scores_output = gr.Textbox(
                                label="도메인별 점수",
                                lines=10,
                                interactive=False
                            )
                            
                            domain_terms_output = gr.Textbox(
                                label="도메인별 용어",
                                lines=10,
                                interactive=False
                            )
                
                # 설정 탭
                with gr.Tab("⚙️ 설정"):
                    gr.Markdown("### 시스템 설정")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Gemini API 설정")
                            gemini_key_setting = gr.Textbox(
                                label="API 키",
                                placeholder="GEMINI_API_KEY 또는 GOOGLE_API_KEY",
                                type="password"
                            )
                            
                            test_gemini_btn = gr.Button("🧪 API 연결 테스트", variant="secondary")
                            gemini_test_result = gr.Textbox(
                                label="테스트 결과",
                                interactive=False
                            )
                        
                        with gr.Column():
                            gr.Markdown("#### 시스템 정보")
                            system_info = gr.Textbox(
                                label="시스템 상태",
                                value="시스템 정보를 로드하는 중...",
                                lines=10,
                                interactive=False
                            )
            
            # 이벤트 핸들러
            def toggle_gemini_key(use_gemini):
                return gr.update(visible=use_gemini)
            
            def toggle_batch_gemini_key(use_gemini):
                return gr.update(visible=use_gemini)
            
            def initialize_pipeline_handler(gemini_key, use_gemini):
                ui = NLPTermExtractionUI()
                return ui.initialize_pipeline(gemini_key, use_gemini)
            
            def extract_terms_handler(text, use_gemini, gemini_key):
                ui = NLPTermExtractionUI()
                ui.initialize_pipeline(gemini_key, use_gemini)
                return ui.extract_terms_from_text(text, use_gemini, gemini_key)
            
            def batch_extract_handler(texts, use_gemini, gemini_key):
                ui = NLPTermExtractionUI()
                ui.initialize_pipeline(gemini_key, use_gemini)
                texts_list = [t.strip() for t in texts.split('\n') if t.strip()]
                return ui.batch_extract_terms(texts_list, use_gemini, gemini_key)
            
            def analyze_domain_handler(text):
                ui = NLPTermExtractionUI()
                return ui.analyze_domain_distribution(text)
            
            # 이벤트 연결
            use_gemini.change(fn=toggle_gemini_key, inputs=[use_gemini], outputs=[gemini_api_key])
            batch_use_gemini.change(fn=toggle_batch_gemini_key, inputs=[batch_use_gemini], outputs=[batch_gemini_key])
            
            init_btn.click(
                fn=initialize_pipeline_handler,
                inputs=[gemini_api_key, use_gemini],
                outputs=[init_status]
            )
            
            extract_btn.click(
                fn=extract_terms_handler,
                inputs=[text_input, use_gemini, gemini_api_key],
                outputs=[extracted_terms_output, validated_terms_output, primary_domain_output, domain_confidence_output, performance_output]
            )
            
            batch_extract_btn.click(
                fn=batch_extract_handler,
                inputs=[batch_texts_input, batch_use_gemini, batch_gemini_key],
                outputs=[batch_results_output]
            )
            
            analyze_domain_btn.click(
                fn=analyze_domain_handler,
                inputs=[domain_text_input],
                outputs=[domain_scores_output, domain_terms_output]
            )
            
            # 시스템 정보 로드
            def load_system_info():
                return f"""
시스템 정보:
- Python 버전: {sys.version}
- 작업 디렉토리: {os.getcwd()}
- 로그 디렉토리: {Path('logs').absolute()}
- 결과 디렉토리: {Path('results').absolute()}

지원 기능:
✅ 법률 텍스트 전처리
✅ 다중 방법론 용어 추출
✅ 도메인별 특화 추출
✅ Gemini API 검증 (선택사항)
✅ 성능 모니터링
✅ 배치 처리
                """
            
            interface.load(fn=load_system_info, outputs=[system_info])
        
        return interface

def main():
    """메인 함수"""
    # UI 생성 및 실행
    ui = NLPTermExtractionUI()
    interface = ui.create_interface()
    
    # 서버 실행
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    main()
