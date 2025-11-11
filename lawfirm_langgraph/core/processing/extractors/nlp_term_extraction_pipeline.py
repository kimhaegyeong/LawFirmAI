import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

from .legal_text_preprocessor import LegalTextPreprocessor
from .multi_method_term_extractor import MultiMethodTermExtractor
from .gemini_validation_pipeline import GeminiValidationPipeline
from .term_integration_system import TermIntegrationSystem
from .performance_evaluator import PerformanceEvaluator
from .domain_specific_extractor import DomainSpecificExtractor, LegalDomain
from .gemini_client import GeminiClient

logger = logging.getLogger(__name__)

class NLPTermExtractionPipeline:
    """NLP 기반 법률 용어 추출 통합 파이프라인"""
    
    def __init__(self, 
                 gemini_api_key: Optional[str] = None,
                 db_path: str = "data/legal_terms_database.json",
                 log_file: str = "logs/classification_performance.json"):
        
        # 컴포넌트 초기화
        self.preprocessor = LegalTextPreprocessor()
        self.term_extractor = MultiMethodTermExtractor()
        self.domain_extractor = DomainSpecificExtractor()  # 도메인별 특화 추출기 추가
        self.gemini_client = GeminiClient(api_key=gemini_api_key) if gemini_api_key else None
        self.validation_pipeline = GeminiValidationPipeline(self.gemini_client) if self.gemini_client else None
        self.integration_system = TermIntegrationSystem(db_path)
        self.performance_evaluator = PerformanceEvaluator(log_file)
        
        self.logger = logging.getLogger(__name__)
    
    def extract_terms_from_texts(self, texts: List[str]) -> Dict[str, Any]:
        """텍스트에서 용어 추출"""
        self.logger.info(f"용어 추출 시작: {len(texts)}개 텍스트")
        
        # 1. 텍스트 전처리
        preprocessed_data = self.preprocessor.batch_preprocess(texts)
        
        # 2. 용어 추출
        extraction_results = []
        all_extracted_terms = []
        
        for i, data in enumerate(preprocessed_data):
            if data.get("error"):
                self.logger.warning(f"텍스트 {i+1} 전처리 실패: {data['error']}")
                continue
            
            # 기본 용어 추출
            extracted_terms = self.term_extractor.extract_and_merge(data["cleaned_text"])
            
            # 도메인별 특화 용어 추출
            domain_enhancement = self.domain_extractor.enhance_term_extraction(
                data["cleaned_text"], extracted_terms
            )
            
            # 기본 용어와 도메인별 강화된 용어 통합
            enhanced_terms = domain_enhancement["enhanced_terms"]
            all_extracted_terms.extend(enhanced_terms)
            
            extraction_results.append({
                "text_index": i,
                "original_text": data["original_text"],
                "cleaned_text": data["cleaned_text"],
                "extracted_terms": extracted_terms,
                "enhanced_terms": enhanced_terms,
                "domain_info": {
                    "primary_domain": domain_enhancement["primary_domain"].value,
                    "domain_confidence": domain_enhancement["domain_confidence"],
                    "weighted_terms": domain_enhancement["weighted_terms"]
                },
                "term_count": len(enhanced_terms)
            })
        
        # 중복 제거
        unique_terms = list(set(all_extracted_terms))
        
        self.logger.info(f"용어 추출 완료: {len(unique_terms)}개 고유 용어")
        
        return {
            "extraction_results": extraction_results,
            "all_extracted_terms": unique_terms,
            "total_terms": len(unique_terms)
        }
    
    def validate_terms_with_gemini(self, terms: List[str], contexts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Gemini를 사용한 용어 검증"""
        if not self.validation_pipeline:
            self.logger.warning("Gemini 클라이언트가 설정되지 않았습니다. 검증을 건너뜁니다.")
            return []
        
        self.logger.info(f"용어 검증 시작: {len(terms)}개 용어")
        
        # 컨텍스트 준비
        if contexts is None:
            contexts = [None] * len(terms)
        
        # 배치 검증
        term_context_pairs = list(zip(terms, contexts))
        validated_terms = self.validation_pipeline.batch_validate(term_context_pairs)
        
        # 고품질 용어만 필터링
        high_quality_terms = [
            term for term in validated_terms
            if term.get("is_valid", False) and term.get("is_high_quality", False)
        ]
        
        self.logger.info(f"용어 검증 완료: {len(high_quality_terms)}개 고품질 용어")
        
        return high_quality_terms
    
    def process_terms_without_gemini(self, terms: List[str]) -> List[Dict[str, Any]]:
        """Gemini 없이 용어 처리 (기본 검증)"""
        self.logger.info(f"기본 용어 처리 시작: {len(terms)}개 용어")
        
        processed_terms = []
        
        for term in terms:
            # 기본 검증 (길이, 문자 등)
            if self._basic_term_validation(term):
                processed_terms.append({
                    "term": term,
                    "is_valid": True,
                    "is_high_quality": True,
                    "final_confidence": 0.8,  # 기본 신뢰도
                    "domain": "기타/일반",
                    "domain_confidence": 0.5,
                    "quality_score": 70,
                    "quality_details": {"기본검증": 70},
                    "definition": f"{term}에 대한 정의가 필요합니다.",
                    "synonyms": [],
                    "related_terms": [],
                    "context_keywords": [term],
                    "weight": 0.5,
                    "suggestions": "Gemini 검증을 통해 더 정확한 정보를 얻을 수 있습니다.",
                    "validation_timestamp": datetime.now().isoformat()
                })
        
        self.logger.info(f"기본 용어 처리 완료: {len(processed_terms)}개 용어")
        
        return processed_terms
    
    def _basic_term_validation(self, term: str) -> bool:
        """기본 용어 검증"""
        if not term or len(term.strip()) < 2:
            return False
        
        # 한글 포함 확인
        if not any('\uac00' <= char <= '\ud7af' for char in term):
            return False
        
        # 특수문자 제한
        if any(char in term for char in ['<', '>', '&', '"', "'", '\\', '/']):
            return False
        
        return True
    
    def run_full_pipeline(self, 
                         texts: List[str],
                         ground_truth: Optional[List[str]] = None,
                         use_gemini: bool = True) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        self.logger.info("NLP 용어 추출 파이프라인 시작")
        
        pipeline_results = {
            "start_time": datetime.now().isoformat(),
            "input_text_count": len(texts),
            "use_gemini": use_gemini
        }
        
        try:
            # 1. 용어 추출
            extraction_results = self.extract_terms_from_texts(texts)
            pipeline_results["extraction"] = extraction_results
            
            # 2. 용어 검증
            if use_gemini and self.validation_pipeline:
                validated_terms = self.validate_terms_with_gemini(
                    extraction_results["all_extracted_terms"]
                )
            else:
                validated_terms = self.process_terms_without_gemini(
                    extraction_results["all_extracted_terms"]
                )
            
            pipeline_results["validation"] = {
                "validated_terms": validated_terms,
                "validated_count": len(validated_terms)
            }
            
            # 3. 용어 통합 및 데이터베이스 업데이트
            integration_results = self.integration_system.full_pipeline(
                extraction_results["all_extracted_terms"],
                validated_terms
            )
            pipeline_results["integration"] = integration_results
            
            # 4. 성능 평가
            performance_results = self.performance_evaluator.evaluate_full_pipeline(
                extraction_results["all_extracted_terms"],
                validated_terms,
                ground_truth
            )
            pipeline_results["performance"] = performance_results
            
            # 5. 성능 보고서 생성
            performance_report = self.performance_evaluator.generate_performance_report(performance_results)
            pipeline_results["performance_report"] = performance_report
            
            pipeline_results["end_time"] = datetime.now().isoformat()
            pipeline_results["status"] = "success"
            
            self.logger.info("파이프라인 실행 완료")
            
        except Exception as e:
            self.logger.error(f"파이프라인 실행 중 오류: {e}")
            pipeline_results["error"] = str(e)
            pipeline_results["status"] = "error"
            pipeline_results["end_time"] = datetime.now().isoformat()
        
        return pipeline_results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """결과 저장"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"결과 저장 완료: {output_file}")
    
    def load_sample_texts(self, sample_file: str) -> List[str]:
        """샘플 텍스트 로드"""
        sample_path = Path(sample_file)
        
        if not sample_path.exists():
            self.logger.warning(f"샘플 파일이 존재하지 않습니다: {sample_path}")
            return []
        
        with open(sample_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "texts" in data:
            return data["texts"]
        else:
            self.logger.warning("샘플 파일 형식이 올바르지 않습니다.")
            return []
    
    def create_sample_texts(self, output_path: str):
        """샘플 텍스트 생성"""
        sample_texts = [
            "계약서에서 손해배상 조항을 검토해주세요. 계약 위반 시 배상 책임이 어떻게 규정되어 있는지 확인이 필요합니다.",
            "이혼 소송에서 양육권과 면접교섭권에 대해 문의드립니다. 자녀의 최선의 이익을 고려한 판단 기준은 무엇인가요?",
            "특허 침해 소송에서 증거 수집 방법과 증거능력에 대해 알고 싶습니다. 디자인권 침해와의 차이점도 궁금합니다.",
            "회사에서 직원을 해고할 때 노동법상 절차를 어떻게 따라야 하나요? 부당해고에 해당하는 경우는 어떤 것인가요?",
            "부동산 매매계약에서 등기 이전 절차와 관련하여 주의사항을 알려주세요. 계약금과 중도금의 법적 효력은 어떻게 되나요?",
            "세무조사에서 조세불복 신청 절차에 대해 문의드립니다. 법인세 신고 시 주의해야 할 사항은 무엇인가요?",
            "형사소송에서 변호인 선임권과 자백배제법칙에 대해 알고 싶습니다. 수사 단계에서의 권리와 의무는 무엇인가요?",
            "민사소송에서 소장 작성과 증거제출 절차를 설명해주세요. 준비서면 제출 시기와 방법은 어떻게 되나요?"
        ]
        
        sample_data = {
            "description": "법률 용어 추출을 위한 샘플 텍스트",
            "created_at": datetime.now().isoformat(),
            "texts": sample_texts
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"샘플 텍스트 생성 완료: {output_file}")
        return sample_texts
