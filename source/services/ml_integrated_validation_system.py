# -*- coding: utf-8 -*-
"""
ML 통합 개선된 다단계 검증 시스템
머신러닝 기반 패턴 학습과 자동 튜닝을 통합한 최종 검증 시스템
"""

import sys
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from source.services.multi_stage_validation_system import MultiStageValidationSystem
from source.services.ml_pattern_learning_system import MLPatternLearningSystem
try:
    from source.services.simple_text_classifier import SimpleTextClassifier
except Exception:
    SimpleTextClassifier = None
try:
    from source.services.bert_classifier import BERTClassifier
except Exception:
    BERTClassifier = None
try:
    from source.services.boundary_referee import BoundaryReferee
except Exception:
    BoundaryReferee = None
try:
    from source.services.llm_referee import LLMReferee
except Exception:
    LLMReferee = None

logger = logging.getLogger(__name__)

class MLIntegratedValidationSystem:
    """ML 통합 개선된 다단계 검증 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 기존 개선된 시스템 (백업용)
        self.improved_system = MultiStageValidationSystem()
        
        # ML 패턴 학습 시스템 (옵션)
        try:
            self.ml_system = MLPatternLearningSystem()
        except Exception as e:
            self.logger.warning(f"ML 패턴 학습 시스템 초기화 실패: {e}")
            self.ml_system = None
        
        # ML 예측 가중치 (기존 시스템과 ML 예측의 조합 비율)
        self.ml_weight = 0.3  # ML 예측이 30% 가중치
        self.traditional_weight = 0.7  # 기존 시스템이 70% 가중치
        self.use_simple_classifier = os.getenv("USE_SIMPLE_CLASSIFIER", "0") == "1"
        self.use_bert_classifier = os.getenv("USE_BERT_CLASSIFIER", "1") == "1"
        self.use_boundary_referee = os.getenv("USE_BOUNDARY_REFEREE", "1") == "1"
        self.use_llm_referee = os.getenv("USE_LLM_REFEREE", "0") == "1"
        
        # Simple classifier 초기화
        if self.use_simple_classifier and SimpleTextClassifier is not None:
            self.simple_classifier = SimpleTextClassifier()
        else:
            self.simple_classifier = None
            
        # BERT classifier 초기화
        if self.use_bert_classifier and BERTClassifier is not None:
            self.bert_classifier = BERTClassifier()
        else:
            self.bert_classifier = None
            
        # Boundary referee 초기화
        if self.use_boundary_referee and BoundaryReferee is not None:
            self.boundary_referee = BoundaryReferee()
        else:
            self.boundary_referee = None
            
        # LLM referee 초기화
        if self.use_llm_referee and LLMReferee is not None:
            self.llm_referee = LLMReferee()
        else:
            self.llm_referee = None
        
        # ML 예측 활성화 여부
        self.ml_enabled = True
        
        self.logger.info("ML 통합 검증 시스템 초기화 완료")
    
    def validate(self, query: str, category: Optional[str] = None, subcategory: Optional[str] = None, 
                user_id: Optional[str] = None, session_id: Optional[str] = None, collect_feedback: bool = True) -> Dict[str, Any]:
        """ML 통합 검증 수행"""
        try:
            self.logger.info(f"ML 통합 검증 시작: {query[:50]}...")
            
            # 1. 기존 개선된 시스템으로 검증
            traditional_result = self.improved_system.validate(query)
            
            # MultiStageValidationResult를 딕셔너리로 변환
            traditional_dict = {
                "final_decision": traditional_result.final_decision.value,
                "confidence": traditional_result.confidence,
                "reasoning": traditional_result.reasoning,
                "edge_case_info": {
                    "is_edge_case": False,  # 기본값
                    "edge_case_type": None
                }
            }
            
            # 2. ML 시스템으로 예측 (활성화된 경우)
            ml_prediction = None
            if self.ml_enabled and self.ml_system:
                try:
                    ml_prediction = self.ml_system.get_ml_prediction(
                        query=query,
                        confidence=traditional_dict.get("confidence", 0.0),
                        edge_case_type=traditional_dict.get("edge_case_info", {}).get("edge_case_type", None)
                    )
                except Exception as e:
                    self.logger.warning(f"ML 예측 오류: {e}")
                    ml_prediction = None

            # 2-1. Simple classifier 보조 예측 (옵션)
            simple_pred = None
            if self.simple_classifier and self.simple_classifier.is_available():
                try:
                    simple_pred = self.simple_classifier.predict(query, category, subcategory)
                except Exception as e:
                    self.logger.warning(f"단순 분류기 예측 오류: {e}")
            
            # 2-2. BERT classifier 보조 예측 (옵션)
            bert_pred = None
            if self.bert_classifier and self.bert_classifier.is_available():
                try:
                    bert_pred = self.bert_classifier.predict(query, category, subcategory)
                except Exception as e:
                    self.logger.warning(f"BERT 분류기 예측 오류: {e}")
            
            # 3. 결과 통합
            final_result = self._integrate_results(traditional_dict, ml_prediction, simple_pred, bert_pred)
            
            # 4. 피드백 수집 준비 (사용자 피드백 대기)
            if collect_feedback:
                self._prepare_feedback_collection(
                    query=query,
                    result=final_result,
                    user_id=user_id,
                    session_id=session_id
                )
            
            self.logger.info(f"ML 통합 검증 완료: {final_result['final_decision']}")
            # 경계 심판기 적용
            # 5. Boundary referee 적용 (옵션)
            if self.boundary_referee is not None:
                final_result = self.boundary_referee.re_evaluate(final_result, category)
            
            # 6. LLM referee 적용 (옵션)
            if self.llm_referee is not None:
                final_result = self.llm_referee.re_evaluate(final_result, category)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"ML 통합 검증 오류: {e}")
            # 오류 시 기존 시스템 결과 반환 (딕셔너리 형태로)
            traditional_result = self.improved_system.validate(query)
            return {
                "final_decision": traditional_result.final_decision.value,
                "confidence": traditional_result.confidence,
                "reasoning": traditional_result.reasoning,
                "edge_case_info": {
                    "is_edge_case": False,
                    "edge_case_type": None
                }
            }
    
    def _integrate_results(self, traditional_result: Dict[str, Any], 
                          ml_prediction: Optional[Dict[str, float]],
                          simple_pred: Optional[Dict[str, float]] = None,
                          bert_pred: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """기존 시스템과 ML 예측 결과 통합"""
        try:
            # 기본 결과는 기존 시스템 결과 사용
            final_result = traditional_result.copy()
            
            # ML 예측이 있는 경우 통합
            combined_restricted = []
            combined_allowed = []

            if ml_prediction and self.ml_enabled:
                # 기존 시스템의 신뢰도
                traditional_confidence = traditional_result["confidence"]
                
                # ML 예측의 신뢰도
                ml_confidence = max(ml_prediction.values())
                
                # 가중 평균으로 최종 신뢰도 계산
                final_confidence = (
                    traditional_confidence * self.traditional_weight +
                    ml_confidence * self.ml_weight
                )
                combined_restricted.append(ml_prediction.get("restricted", 0.0))
                combined_allowed.append(ml_prediction.get("allowed", 0.0))
                
                # ML 예측이 강하게 제한을 권하는 경우 재검토
                if ml_prediction["restricted"] > 0.8 and traditional_result["final_decision"] == "allowed":
                    # Edge Case가 아닌 경우에만 ML 예측 반영
                    if not traditional_result["edge_case_info"]["is_edge_case"]:
                        final_result["final_decision"] = "restricted"
                        final_result["confidence"] = final_confidence
                        final_result["reasoning"].append(f"ML 예측으로 인한 제한 결정 (ML 신뢰도: {ml_confidence:.2f})")
                
                # ML 예측이 강하게 허용을 권하는 경우 재검토
                elif ml_prediction["allowed"] > 0.8 and traditional_result["final_decision"] == "restricted":
                    # Edge Case인 경우에만 ML 예측 반영
                    if traditional_result["edge_case_info"]["is_edge_case"]:
                        final_result["final_decision"] = "allowed"
                        final_result["confidence"] = final_confidence
                        final_result["reasoning"].append(f"ML 예측으로 인한 허용 결정 (ML 신뢰도: {ml_confidence:.2f})")
                
                # ML 예측 정보 추가
                final_result["ml_prediction"] = ml_prediction
                final_result["ml_weight"] = self.ml_weight
                final_result["traditional_weight"] = self.traditional_weight

            # 단순 분류기 보조 통합
            if simple_pred:
                combined_restricted.append(simple_pred.get("restricted", 0.0))
                combined_allowed.append(simple_pred.get("allowed", 0.0))
                final_result["simple_pred"] = simple_pred

            # BERT 분류기 보조 통합
            if bert_pred:
                combined_restricted.append(bert_pred.get("restricted", 0.0))
                combined_allowed.append(bert_pred.get("allowed", 0.0))
                final_result["bert_pred"] = bert_pred
                
                # BERT 분류기가 강한 신뢰도를 보이는 경우 우선 적용
                bert_confidence = max(bert_pred.values())
                if bert_confidence > 0.9:
                    if bert_pred["restricted"] > bert_pred["allowed"] and traditional_result["final_decision"] == "allowed":
                        final_result["final_decision"] = "restricted"
                        final_result["reasoning"].append(f"BERT 분류기 강한 제한 신호 (신뢰도: {bert_confidence:.2f})")
                    elif bert_pred["allowed"] > bert_pred["restricted"] and traditional_result["final_decision"] == "restricted":
                        final_result["final_decision"] = "allowed"
                        final_result["reasoning"].append(f"BERT 분류기 강한 허용 신호 (신뢰도: {bert_confidence:.2f})")

            # 보조 예측들의 평균을 참조해 경계 케이스만 소극적으로 보정
            if combined_restricted or combined_allowed:
                avg_restricted = sum(combined_restricted) / max(len(combined_restricted), 1)
                avg_allowed = sum(combined_allowed) / max(len(combined_allowed), 1)
                # 기존 결과가 허용이고 보조가 강한 제한(>0.75)일 때만 제한으로 보정
                if traditional_result["final_decision"] == "allowed" and avg_restricted >= 0.75:
                    final_result["final_decision"] = "restricted"
                    final_result["reasoning"].append("보조 ML/단순 분류기 일치로 제한 보정")
                # 기존 결과가 제한이고 보조가 강한 허용(>0.85)이며 edge-case이면 허용으로 보정
                if traditional_result["final_decision"] == "restricted" and avg_allowed >= 0.85 and traditional_result["edge_case_info"].get("is_edge_case"):
                    final_result["final_decision"] = "allowed"
                    final_result["reasoning"].append("보조 ML/단순 분류기 일치로 허용 보정(엣지 케이스)")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"결과 통합 오류: {e}")
            return traditional_result
    
    def _prepare_feedback_collection(self, query: str, result: Dict[str, Any], 
                                   user_id: Optional[str], session_id: Optional[str]):
        """피드백 수집 준비"""
        try:
            # 피드백 수집을 위한 정보 저장 (실제 구현에서는 세션에 저장)
            feedback_info = {
                "query": query,
                "predicted_result": result["final_decision"],
                "confidence": result["confidence"],
                "edge_case_type": result["edge_case_info"]["edge_case_type"],
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            
            # 실제 구현에서는 세션 스토리지나 데이터베이스에 저장
            self.logger.info(f"피드백 수집 준비 완료: {query[:30]}...")
            
        except Exception as e:
            self.logger.error(f"피드백 수집 준비 오류: {e}")
    
    def collect_user_feedback(self, query: str, user_feedback: str, 
                            user_id: Optional[str] = None, session_id: Optional[str] = None) -> bool:
        """사용자 피드백 수집"""
        try:
            # 실제 구현에서는 세션에서 예측 결과를 가져와야 함
            # 여기서는 간단한 예시로 구현
            
            return self.ml_system.collect_user_feedback(
                query=query,
                predicted_result="restricted",  # 실제로는 세션에서 가져와야 함
                user_feedback=user_feedback,
                confidence=0.8,  # 실제로는 세션에서 가져와야 함
                edge_case_type=None,  # 실제로는 세션에서 가져와야 함
                user_id=user_id,
                session_id=session_id
            )
            
        except Exception as e:
            self.logger.error(f"사용자 피드백 수집 오류: {e}")
            return False
    
    def learn_from_feedback(self, days: int = 7) -> Dict[str, Any]:
        """피드백으로부터 학습"""
        try:
            learning_results = self.ml_system.learn_from_feedback(days)
            
            # 학습 결과에 따라 ML 가중치 조정
            if learning_results:
                best_model = max(learning_results.values(), key=lambda x: x.accuracy)
                if best_model.accuracy > 0.9:
                    # 높은 정확도면 ML 가중치 증가
                    self.ml_weight = min(0.5, self.ml_weight + 0.05)
                    self.traditional_weight = 1.0 - self.ml_weight
                    self.logger.info(f"ML 가중치 증가: {self.ml_weight:.2f}")
                elif best_model.accuracy < 0.7:
                    # 낮은 정확도면 ML 가중치 감소
                    self.ml_weight = max(0.1, self.ml_weight - 0.05)
                    self.traditional_weight = 1.0 - self.ml_weight
                    self.logger.info(f"ML 가중치 감소: {self.ml_weight:.2f}")
            
            return learning_results
            
        except Exception as e:
            self.logger.error(f"피드백 학습 오류: {e}")
            return {}
    
    def auto_tune_system(self, days: int = 7) -> List[Any]:
        """시스템 자동 튜닝"""
        try:
            tuning_results = self.ml_system.auto_tune_system(days)
            
            # 튜닝 결과에 따라 시스템 파라미터 조정
            for tuning in tuning_results:
                if tuning.parameter_name == "edge_case_pattern_threshold":
                    # Edge Case 패턴 임계값 조정
                    self.logger.info(f"Edge Case 패턴 임계값 조정: {tuning.old_value} → {tuning.new_value}")
                
                elif tuning.parameter_name == "personal_advice_keyword_weight":
                    # 개인적 조언 키워드 가중치 조정
                    self.logger.info(f"개인적 조언 키워드 가중치 조정: {tuning.old_value} → {tuning.new_value}")
                
                elif tuning.parameter_name == "medical_advice_pattern_weight":
                    # 의료법 조언 패턴 가중치 조정
                    self.logger.info(f"의료법 조언 패턴 가중치 조정: {tuning.old_value} → {tuning.new_value}")
            
            return tuning_results
            
        except Exception as e:
            self.logger.error(f"자동 튜닝 오류: {e}")
            return []
    
    def get_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """성능 보고서 생성"""
        try:
            report = self.ml_system.get_performance_report(days)
            
            # ML 통합 시스템 정보 추가
            report["ml_integration"] = {
                "ml_enabled": self.ml_enabled,
                "ml_weight": self.ml_weight,
                "traditional_weight": self.traditional_weight,
                "system_version": "ML_Integrated_v1.0"
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"성능 보고서 생성 오류: {e}")
            return {}
    
    def enable_ml(self, enabled: bool = True):
        """ML 예측 활성화/비활성화"""
        self.ml_enabled = enabled
        self.logger.info(f"ML 예측 {'활성화' if enabled else '비활성화'}")
    
    def adjust_ml_weight(self, ml_weight: float):
        """ML 가중치 조정"""
        if 0.0 <= ml_weight <= 1.0:
            self.ml_weight = ml_weight
            self.traditional_weight = 1.0 - ml_weight
            self.logger.info(f"ML 가중치 조정: {ml_weight:.2f}")
        else:
            self.logger.warning(f"잘못된 ML 가중치: {ml_weight}")
    
    def run_continuous_learning(self, interval_hours: int = 24):
        """지속적 학습 실행"""
        try:
            self.ml_system.run_continuous_learning(interval_hours)
        except Exception as e:
            self.logger.error(f"지속적 학습 실행 오류: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        try:
            return {
                "ml_enabled": self.ml_enabled,
                "ml_weight": self.ml_weight,
                "traditional_weight": self.traditional_weight,
                "ml_models_loaded": len(self.ml_system.pattern_learner.models) > 0,
                "feedback_samples": len(self.ml_system.feedback_collector.load_feedback_data(30)),
                "last_learning": datetime.now().isoformat(),
                "system_version": "ML_Integrated_v1.0"
            }
        except Exception as e:
            self.logger.error(f"시스템 상태 조회 오류: {e}")
            return {}
