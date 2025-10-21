# -*- coding: utf-8 -*-
"""
머신러닝 기반 패턴 학습 시스템
사용자 피드백을 학습하여 법률 제한 시스템의 성능을 자동으로 개선합니다.
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

@dataclass
class FeedbackData:
    """피드백 데이터 구조"""
    query: str
    predicted_result: str  # "allowed" or "restricted"
    actual_result: str     # "allowed" or "restricted"
    user_feedback: str     # "correct", "incorrect", "unsure"
    confidence: float
    timestamp: datetime
    edge_case_type: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class PatternLearningResult:
    """패턴 학습 결과"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    feature_importance: Dict[str, float]
    training_samples: int
    validation_samples: int
    timestamp: datetime

@dataclass
class AutoTuningResult:
    """자동 튜닝 결과"""
    parameter_name: str
    old_value: Any
    new_value: Any
    performance_improvement: float
    confidence: float
    timestamp: datetime

class FeedbackCollector:
    """피드백 수집기"""
    
    def __init__(self, feedback_file: str = "data/ml_feedback/feedback_data.jsonl"):
        self.feedback_file = feedback_file
        self.logger = logging.getLogger(__name__)
        self._ensure_directory()
    
    def _ensure_directory(self):
        """디렉토리 생성"""
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
    
    def collect_feedback(self, feedback: FeedbackData) -> bool:
        """피드백 수집"""
        try:
            feedback_dict = asdict(feedback)
            feedback_dict['timestamp'] = feedback.timestamp.isoformat()
            
            with open(self.feedback_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(feedback_dict, ensure_ascii=False) + '\n')
            
            self.logger.info(f"피드백 수집 완료: {feedback.query[:50]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"피드백 수집 오류: {e}")
            return False
    
    def load_feedback_data(self, days: int = 30) -> List[FeedbackData]:
        """피드백 데이터 로드"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            feedback_data = []
            
            if not os.path.exists(self.feedback_file):
                return feedback_data
            
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        feedback = FeedbackData(
                            query=data['query'],
                            predicted_result=data['predicted_result'],
                            actual_result=data['actual_result'],
                            user_feedback=data['user_feedback'],
                            confidence=data['confidence'],
                            timestamp=datetime.fromisoformat(data['timestamp']),
                            edge_case_type=data.get('edge_case_type'),
                            user_id=data.get('user_id'),
                            session_id=data.get('session_id')
                        )
                        
                        if feedback.timestamp >= cutoff_date:
                            feedback_data.append(feedback)
                            
                    except Exception as e:
                        self.logger.warning(f"피드백 데이터 파싱 오류: {e}")
                        continue
            
            self.logger.info(f"피드백 데이터 로드 완료: {len(feedback_data)}개")
            return feedback_data
            
        except Exception as e:
            self.logger.error(f"피드백 데이터 로드 오류: {e}")
            return []

class PatternLearner:
    """패턴 학습기"""
    
    def __init__(self, model_dir: str = "data/ml_models"):
        self.model_dir = model_dir
        self.logger = logging.getLogger(__name__)
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words=None,
            min_df=2,
            max_df=0.95
        )
        self.scaler = StandardScaler()
        self.models = {}
        self._ensure_directory()
    
    def _ensure_directory(self):
        """모델 디렉토리 생성"""
        os.makedirs(self.model_dir, exist_ok=True)
    
    def prepare_features(self, feedback_data: List[FeedbackData]) -> Tuple[np.ndarray, np.ndarray]:
        """특성 준비"""
        try:
            # 텍스트 특성 추출
            queries = [feedback.query for feedback in feedback_data]
            text_features = self.vectorizer.fit_transform(queries).toarray()
            
            # 추가 특성 생성
            additional_features = []
            for feedback in feedback_data:
                features = [
                    feedback.confidence,
                    len(feedback.query),
                    feedback.query.count('?'),
                    feedback.query.count('!'),
                    1 if feedback.edge_case_type else 0,
                    feedback.timestamp.hour,
                    feedback.timestamp.weekday()
                ]
                additional_features.append(features)
            
            additional_features = np.array(additional_features)
            
            # 특성 결합
            combined_features = np.hstack([text_features, additional_features])
            
            # 라벨 생성
            labels = []
            for feedback in feedback_data:
                if feedback.user_feedback == "correct":
                    labels.append(1 if feedback.predicted_result == "restricted" else 0)
                elif feedback.user_feedback == "incorrect":
                    labels.append(0 if feedback.predicted_result == "restricted" else 1)
                else:  # unsure
                    labels.append(1 if feedback.predicted_result == "restricted" else 0)
            
            labels = np.array(labels)
            
            self.logger.info(f"특성 준비 완료: {combined_features.shape}")
            return combined_features, labels
            
        except Exception as e:
            self.logger.error(f"특성 준비 오류: {e}")
            return np.array([]), np.array([])
    
    def train_models(self, feedback_data: List[FeedbackData]) -> Dict[str, PatternLearningResult]:
        """모델 학습"""
        try:
            if len(feedback_data) < 10:
                self.logger.warning("학습 데이터가 부족합니다.")
                return {}
            
            # 특성 준비
            X, y = self.prepare_features(feedback_data)
            if len(X) == 0:
                return {}
            
            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 특성 스케일링
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            results = {}
            
            # Random Forest 모델
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            rf_model.fit(X_train_scaled, y_train)
            rf_pred = rf_model.predict(X_test_scaled)
            
            results['random_forest'] = PatternLearningResult(
                model_name='random_forest',
                accuracy=accuracy_score(y_test, rf_pred),
                precision=precision_score(y_test, rf_pred, average='weighted'),
                recall=recall_score(y_test, rf_pred, average='weighted'),
                f1_score=f1_score(y_test, rf_pred, average='weighted'),
                feature_importance=self._get_feature_importance(rf_model),
                training_samples=len(X_train),
                validation_samples=len(X_test),
                timestamp=datetime.now()
            )
            
            # Logistic Regression 모델
            lr_model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
            lr_model.fit(X_train_scaled, y_train)
            lr_pred = lr_model.predict(X_test_scaled)
            
            results['logistic_regression'] = PatternLearningResult(
                model_name='logistic_regression',
                accuracy=accuracy_score(y_test, lr_pred),
                precision=precision_score(y_test, lr_pred, average='weighted'),
                recall=recall_score(y_test, lr_pred, average='weighted'),
                f1_score=f1_score(y_test, lr_pred, average='weighted'),
                feature_importance=self._get_feature_importance(lr_model),
                training_samples=len(X_train),
                validation_samples=len(X_test),
                timestamp=datetime.now()
            )
            
            # 모델 저장
            self.models['random_forest'] = rf_model
            self.models['logistic_regression'] = lr_model
            
            self._save_models()
            
            self.logger.info(f"모델 학습 완료: {len(results)}개 모델")
            return results
            
        except Exception as e:
            self.logger.error(f"모델 학습 오류: {e}")
            return {}
    
    def _get_feature_importance(self, model) -> Dict[str, float]:
        """특성 중요도 추출"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                return {}
            
            # 특성 이름 생성
            feature_names = []
            
            # TF-IDF 특성 이름
            if hasattr(self.vectorizer, 'get_feature_names_out'):
                tfidf_names = self.vectorizer.get_feature_names_out()
                feature_names.extend(tfidf_names)
            
            # 추가 특성 이름
            additional_names = [
                'confidence', 'query_length', 'question_marks', 'exclamation_marks',
                'is_edge_case', 'hour', 'weekday'
            ]
            feature_names.extend(additional_names)
            
            # 중요도 딕셔너리 생성
            importance_dict = {}
            for i, importance in enumerate(importances):
                if i < len(feature_names):
                    importance_dict[feature_names[i]] = float(importance)
            
            return importance_dict
            
        except Exception as e:
            self.logger.error(f"특성 중요도 추출 오류: {e}")
            return {}
    
    def _save_models(self):
        """모델 저장"""
        try:
            for name, model in self.models.items():
                model_path = os.path.join(self.model_dir, f"{name}_model.pkl")
                joblib.dump(model, model_path)
            
            # 벡터라이저와 스케일러 저장
            joblib.dump(self.vectorizer, os.path.join(self.model_dir, "vectorizer.pkl"))
            joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.pkl"))
            
            self.logger.info("모델 저장 완료")
            
        except Exception as e:
            self.logger.error(f"모델 저장 오류: {e}")
    
    def load_models(self):
        """모델 로드"""
        try:
            for name in ['random_forest', 'logistic_regression']:
                model_path = os.path.join(self.model_dir, f"{name}_model.pkl")
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
            
            # 벡터라이저와 스케일러 로드
            vectorizer_path = os.path.join(self.model_dir, "vectorizer.pkl")
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            
            if os.path.exists(vectorizer_path):
                self.vectorizer = joblib.load(vectorizer_path)
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            self.logger.info(f"모델 로드 완료: {len(self.models)}개 모델")
            
        except Exception as e:
            self.logger.error(f"모델 로드 오류: {e}")
    
    def predict(self, query: str, confidence: float, edge_case_type: Optional[str] = None) -> Dict[str, float]:
        """예측 수행"""
        try:
            if not self.models:
                self.load_models()
            
            if not self.models:
                return {'restricted': 0.5, 'allowed': 0.5}
            
            # 특성 준비
            text_features = self.vectorizer.transform([query]).toarray()
            
            additional_features = np.array([[
                confidence,
                len(query),
                query.count('?'),
                query.count('!'),
                1 if edge_case_type else 0,
                datetime.now().hour,
                datetime.now().weekday()
            ]])
            
            combined_features = np.hstack([text_features, additional_features])
            scaled_features = self.scaler.transform(combined_features)
            
            # 예측 수행
            predictions = {}
            for name, model in self.models.items():
                pred_proba = model.predict_proba(scaled_features)[0]
                predictions[name] = {
                    'restricted': float(pred_proba[1]),
                    'allowed': float(pred_proba[0])
                }
            
            # 앙상블 예측
            ensemble_pred = {
                'restricted': np.mean([pred['restricted'] for pred in predictions.values()]),
                'allowed': np.mean([pred['allowed'] for pred in predictions.values()])
            }
            
            return ensemble_pred
            
        except Exception as e:
            self.logger.error(f"예측 오류: {e}")
            return {'restricted': 0.5, 'allowed': 0.5}

class AutoTuner:
    """자동 튜닝 시스템"""
    
    def __init__(self, config_file: str = "data/ml_config/auto_tuning_config.json"):
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)
        self.tuning_history = []
        self._ensure_directory()
        self._load_config()
    
    def _ensure_directory(self):
        """디렉토리 생성"""
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
    
    def _load_config(self):
        """설정 로드"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                self.config = {
                    "tuning_enabled": True,
                    "min_feedback_samples": 50,
                    "performance_threshold": 0.8,
                    "tuning_parameters": {
                        "keyword_threshold": {"min": 0.01, "max": 0.1, "step": 0.01},
                        "pattern_threshold": {"min": 0.01, "max": 0.1, "step": 0.01},
                        "context_threshold": {"min": 0.01, "max": 0.1, "step": 0.01},
                        "intent_threshold": {"min": 0.01, "max": 0.1, "step": 0.01}
                    }
                }
                self._save_config()
                
        except Exception as e:
            self.logger.error(f"설정 로드 오류: {e}")
            self.config = {}
    
    def _save_config(self):
        """설정 저장"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"설정 저장 오류: {e}")
    
    def analyze_performance(self, feedback_data: List[FeedbackData]) -> Dict[str, float]:
        """성능 분석"""
        try:
            if not feedback_data:
                return {}
            
            total_samples = len(feedback_data)
            correct_predictions = sum(1 for f in feedback_data if f.user_feedback == "correct")
            incorrect_predictions = sum(1 for f in feedback_data if f.user_feedback == "incorrect")
            
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0
            
            # 카테고리별 성능 분석
            edge_cases = [f for f in feedback_data if f.edge_case_type]
            personal_advice = [f for f in feedback_data if "제 경우" in f.query or "저는" in f.query]
            medical_advice = [f for f in feedback_data if any(keyword in f.query for keyword in ["의료", "의학", "장애"])]
            
            performance = {
                "overall_accuracy": accuracy,
                "total_samples": total_samples,
                "correct_predictions": correct_predictions,
                "incorrect_predictions": incorrect_predictions,
                "edge_cases_accuracy": self._calculate_category_accuracy(edge_cases),
                "personal_advice_accuracy": self._calculate_category_accuracy(personal_advice),
                "medical_advice_accuracy": self._calculate_category_accuracy(medical_advice)
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"성능 분석 오류: {e}")
            return {}
    
    def _calculate_category_accuracy(self, category_data: List[FeedbackData]) -> float:
        """카테고리별 정확도 계산"""
        if not category_data:
            return 0.0
        
        correct = sum(1 for f in category_data if f.user_feedback == "correct")
        return correct / len(category_data)
    
    def suggest_tuning(self, performance: Dict[str, float]) -> List[AutoTuningResult]:
        """튜닝 제안"""
        try:
            if not self.config.get("tuning_enabled", False):
                return []
            
            suggestions = []
            
            # 전체 정확도가 임계값 이하인 경우
            if performance.get("overall_accuracy", 0) < self.config.get("performance_threshold", 0.8):
                suggestions.extend(self._suggest_threshold_tuning(performance))
            
            # 카테고리별 성능이 낮은 경우
            if performance.get("edge_cases_accuracy", 0) < 0.9:
                suggestions.append(AutoTuningResult(
                    parameter_name="edge_case_pattern_threshold",
                    old_value=0.05,
                    new_value=0.03,
                    performance_improvement=0.1,
                    confidence=0.8,
                    timestamp=datetime.now()
                ))
            
            if performance.get("personal_advice_accuracy", 0) < 0.9:
                suggestions.append(AutoTuningResult(
                    parameter_name="personal_advice_keyword_weight",
                    old_value=1.0,
                    new_value=1.2,
                    performance_improvement=0.15,
                    confidence=0.85,
                    timestamp=datetime.now()
                ))
            
            if performance.get("medical_advice_accuracy", 0) < 0.9:
                suggestions.append(AutoTuningResult(
                    parameter_name="medical_advice_pattern_weight",
                    old_value=1.0,
                    new_value=1.3,
                    performance_improvement=0.12,
                    confidence=0.82,
                    timestamp=datetime.now()
                ))
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"튜닝 제안 오류: {e}")
            return []
    
    def _suggest_threshold_tuning(self, performance: Dict[str, float]) -> List[AutoTuningResult]:
        """임계값 튜닝 제안"""
        suggestions = []
        
        tuning_params = self.config.get("tuning_parameters", {})
        
        for param_name, param_config in tuning_params.items():
            current_value = param_config.get("current", param_config["min"])
            
            if performance.get("overall_accuracy", 0) < 0.8:
                # 정확도가 낮으면 임계값을 낮춤
                new_value = max(param_config["min"], current_value - param_config["step"])
                suggestions.append(AutoTuningResult(
                    parameter_name=param_name,
                    old_value=current_value,
                    new_value=new_value,
                    performance_improvement=0.05,
                    confidence=0.7,
                    timestamp=datetime.now()
                ))
        
        return suggestions
    
    def apply_tuning(self, tuning_result: AutoTuningResult) -> bool:
        """튜닝 적용"""
        try:
            # 튜닝 이력 저장
            self.tuning_history.append(asdict(tuning_result))
            
            # 설정 업데이트
            if tuning_result.parameter_name in self.config.get("tuning_parameters", {}):
                self.config["tuning_parameters"][tuning_result.parameter_name]["current"] = tuning_result.new_value
            
            self._save_config()
            
            self.logger.info(f"튜닝 적용 완료: {tuning_result.parameter_name} = {tuning_result.new_value}")
            return True
            
        except Exception as e:
            self.logger.error(f"튜닝 적용 오류: {e}")
            return False

class PerformanceMonitor:
    """성능 모니터링 시스템"""
    
    def __init__(self, metrics_file: str = "data/ml_metrics/performance_metrics.json"):
        self.metrics_file = metrics_file
        self.logger = logging.getLogger(__name__)
        self._ensure_directory()
    
    def _ensure_directory(self):
        """디렉토리 생성"""
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
    
    def record_metrics(self, metrics: Dict[str, Any]):
        """메트릭 기록"""
        try:
            metrics['timestamp'] = datetime.now().isoformat()
            
            # 기존 메트릭 로드
            existing_metrics = []
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    existing_metrics = json.load(f)
            
            # 새 메트릭 추가
            existing_metrics.append(metrics)
            
            # 최근 100개만 유지
            if len(existing_metrics) > 100:
                existing_metrics = existing_metrics[-100:]
            
            # 저장
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(existing_metrics, f, ensure_ascii=False, indent=2)
            
            self.logger.info("메트릭 기록 완료")
            
        except Exception as e:
            self.logger.error(f"메트릭 기록 오류: {e}")
    
    def get_performance_trend(self, days: int = 7) -> Dict[str, List[float]]:
        """성능 트렌드 분석"""
        try:
            if not os.path.exists(self.metrics_file):
                return {}
            
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                metrics_history = json.load(f)
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            trend_data = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': [],
                'timestamps': []
            }
            
            for metric in metrics_history:
                metric_date = datetime.fromisoformat(metric['timestamp'])
                if metric_date >= cutoff_date:
                    trend_data['accuracy'].append(metric.get('accuracy', 0))
                    trend_data['precision'].append(metric.get('precision', 0))
                    trend_data['recall'].append(metric.get('recall', 0))
                    trend_data['f1_score'].append(metric.get('f1_score', 0))
                    trend_data['timestamps'].append(metric['timestamp'])
            
            return trend_data
            
        except Exception as e:
            self.logger.error(f"성능 트렌드 분석 오류: {e}")
            return {}

class MLPatternLearningSystem:
    """머신러닝 기반 패턴 학습 시스템 (통합 관리자)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feedback_collector = FeedbackCollector()
        self.pattern_learner = PatternLearner()
        self.auto_tuner = AutoTuner()
        self.performance_monitor = PerformanceMonitor()
        
        # 모델 로드
        self.pattern_learner.load_models()
    
    def collect_user_feedback(self, query: str, predicted_result: str, 
                            user_feedback: str, confidence: float,
                            edge_case_type: Optional[str] = None,
                            user_id: Optional[str] = None,
                            session_id: Optional[str] = None) -> bool:
        """사용자 피드백 수집"""
        try:
            # 실제 결과 결정
            actual_result = predicted_result if user_feedback == "correct" else (
                "restricted" if predicted_result == "allowed" else "allowed"
            )
            
            feedback = FeedbackData(
                query=query,
                predicted_result=predicted_result,
                actual_result=actual_result,
                user_feedback=user_feedback,
                confidence=confidence,
                timestamp=datetime.now(),
                edge_case_type=edge_case_type,
                user_id=user_id,
                session_id=session_id
            )
            
            return self.feedback_collector.collect_feedback(feedback)
            
        except Exception as e:
            self.logger.error(f"피드백 수집 오류: {e}")
            return False
    
    def learn_from_feedback(self, days: int = 7) -> Dict[str, PatternLearningResult]:
        """피드백으로부터 학습"""
        try:
            # 피드백 데이터 로드
            feedback_data = self.feedback_collector.load_feedback_data(days)
            
            if len(feedback_data) < 10:
                self.logger.warning("학습할 피드백 데이터가 부족합니다.")
                return {}
            
            # 모델 학습
            learning_results = self.pattern_learner.train_models(feedback_data)
            
            # 성능 메트릭 기록
            for result in learning_results.values():
                metrics = {
                    'model_name': result.model_name,
                    'accuracy': result.accuracy,
                    'precision': result.precision,
                    'recall': result.recall,
                    'f1_score': result.f1_score,
                    'training_samples': result.training_samples,
                    'validation_samples': result.validation_samples
                }
                self.performance_monitor.record_metrics(metrics)
            
            self.logger.info(f"피드백 학습 완료: {len(learning_results)}개 모델")
            return learning_results
            
        except Exception as e:
            self.logger.error(f"피드백 학습 오류: {e}")
            return {}
    
    def auto_tune_system(self, days: int = 7) -> List[AutoTuningResult]:
        """시스템 자동 튜닝"""
        try:
            # 피드백 데이터 로드
            feedback_data = self.feedback_collector.load_feedback_data(days)
            
            if len(feedback_data) < self.auto_tuner.config.get("min_feedback_samples", 50):
                self.logger.warning("튜닝할 피드백 데이터가 부족합니다.")
                return []
            
            # 성능 분석
            performance = self.auto_tuner.analyze_performance(feedback_data)
            
            # 튜닝 제안
            tuning_suggestions = self.auto_tuner.suggest_tuning(performance)
            
            # 튜닝 적용
            applied_tunings = []
            for suggestion in tuning_suggestions:
                if self.auto_tuner.apply_tuning(suggestion):
                    applied_tunings.append(suggestion)
            
            self.logger.info(f"자동 튜닝 완료: {len(applied_tunings)}개 적용")
            return applied_tunings
            
        except Exception as e:
            self.logger.error(f"자동 튜닝 오류: {e}")
            return []
    
    def get_ml_prediction(self, query: str, confidence: float, 
                         edge_case_type: Optional[str] = None) -> Dict[str, float]:
        """ML 모델 예측"""
        try:
            return self.pattern_learner.predict(query, confidence, edge_case_type)
            
        except Exception as e:
            self.logger.error(f"ML 예측 오류: {e}")
            return {'restricted': 0.5, 'allowed': 0.5}
    
    def get_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """성능 보고서 생성"""
        try:
            # 피드백 데이터 로드
            feedback_data = self.feedback_collector.load_feedback_data(days)
            
            # 성능 분석
            performance = self.auto_tuner.analyze_performance(feedback_data)
            
            # 성능 트렌드
            trend_data = self.performance_monitor.get_performance_trend(days)
            
            # 최근 학습 결과
            recent_learning = self.learn_from_feedback(days)
            
            report = {
                'period_days': days,
                'total_feedback_samples': len(feedback_data),
                'performance_metrics': performance,
                'performance_trend': trend_data,
                'recent_learning_results': {name: asdict(result) for name, result in recent_learning.items()},
                'tuning_history': self.auto_tuner.tuning_history[-10:],  # 최근 10개
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"성능 보고서 생성 오류: {e}")
            return {}
    
    def run_continuous_learning(self, interval_hours: int = 24):
        """지속적 학습 실행"""
        try:
            while True:
                self.logger.info("지속적 학습 시작...")
                
                # 피드백 학습
                learning_results = self.learn_from_feedback()
                
                # 자동 튜닝
                tuning_results = self.auto_tune_system()
                
                # 성능 보고서 생성
                report = self.get_performance_report()
                
                self.logger.info(f"지속적 학습 완료: {len(learning_results)}개 모델 학습, {len(tuning_results)}개 튜닝 적용")
                
                # 다음 실행까지 대기
                import time
                time.sleep(interval_hours * 3600)
                
        except KeyboardInterrupt:
            self.logger.info("지속적 학습 중단")
        except Exception as e:
            self.logger.error(f"지속적 학습 오류: {e}")

