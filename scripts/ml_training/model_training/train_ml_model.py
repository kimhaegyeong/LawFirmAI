#!/usr/bin/env python3
"""
머신러닝 모델 훈련 스크립트
조문 분류를 위한 RandomForest 모델 훈련 및 검증
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModelTrainer:
    """머신러닝 모델 훈련 클래스"""
    
    def __init__(self, training_data_path: str):
        """
        초기화
        
        Args:
            training_data_path: 훈련 데이터 파일 경로
        """
        self.training_data_path = Path(training_data_path)
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        
    def load_training_data(self) -> Tuple[List[Dict[str, Any]], List[int]]:
        """훈련 데이터 로드 및 전처리"""
        logger.info(f"Loading training data from {self.training_data_path}")
        
        with open(self.training_data_path, 'r', encoding='utf-8') as f:
            training_samples = json.load(f)
        
        logger.info(f"Loaded {len(training_samples)} training samples")
        
        # 특성과 레이블 분리
        X_features = []
        X_text = []
        y = []
        
        for sample in training_samples:
            features = sample['features']
            
            # 수치형 특성 추출
            numerical_features = [
                features['position_ratio'],
                features['is_at_start'],
                features['is_at_end'],
                features['has_sentence_end'],
                features['has_reference_pattern'],
                features['article_number'],
                features['is_supplementary'],
                features['context_before_length'],
                features['context_after_length'],
                features['has_title'],
                features['has_parentheses'],
                features['has_quotes'],
                features['legal_term_count'],
                features['number_count'],
                features['article_length'],
                features['reference_density']
            ]
            
            X_features.append(numerical_features)
            
            # 텍스트 특성 (문맥 정보)
            context_text = f"{sample.get('article_number', '')} {sample.get('article_title', '')}"
            X_text.append(context_text)
            
            # 레이블 (real_article: 1, reference: 0)
            label = 1 if sample['label'] == 'real_article' else 0
            y.append(label)
        
        logger.info(f"Extracted {len(X_features)} feature vectors")
        logger.info(f"Real articles: {sum(y)}, References: {len(y) - sum(y)}")
        
        return X_features, X_text, y
    
    def prepare_features(self, X_features: List[List[float]], X_text: List[str]) -> np.ndarray:
        """특성 벡터 준비"""
        logger.info("Preparing feature vectors...")
        
        # 수치형 특성을 numpy 배열로 변환
        numerical_features = np.array(X_features)
        
        # 텍스트 특성을 TF-IDF로 변환
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words=None,  # 한국어는 불용어 제거하지 않음
            min_df=2,
            max_df=0.95
        )
        
        text_features = self.vectorizer.fit_transform(X_text)
        
        # 수치형 특성과 텍스트 특성 결합
        combined_features = np.hstack([
            numerical_features,
            text_features.toarray()
        ])
        
        logger.info(f"Combined feature shape: {combined_features.shape}")
        
        # 특성 이름 저장
        numerical_feature_names = [
            'position_ratio', 'is_at_start', 'is_at_end', 'has_sentence_end',
            'has_reference_pattern', 'article_number', 'is_supplementary',
            'context_before_length', 'context_after_length', 'has_title',
            'has_parentheses', 'has_quotes', 'legal_term_count', 'number_count',
            'article_length', 'reference_density'
        ]
        
        text_feature_names = [f"tfidf_{i}" for i in range(text_features.shape[1])]
        self.feature_names = numerical_feature_names + text_feature_names
        
        return combined_features
    
    def train_model(self, X: np.ndarray, y: List[int]) -> None:
        """모델 훈련"""
        logger.info("Training RandomForest model...")
        
        # 훈련/검증 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # 하이퍼파라미터 튜닝
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        logger.info("Performing hyperparameter tuning...")
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # 그리드 서치 (시간 단축을 위해 제한적)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # 모델 평가
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        # 상세 평가 리포트
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred, target_names=['Reference', 'Real Article']))
        
        # 혼동 행렬
        cm = confusion_matrix(y_test, y_pred)
        print("\n=== Confusion Matrix ===")
        print(cm)
        
        # 특성 중요도 분석
        self._analyze_feature_importance(X_test, y_test)
        
        return X_test, y_test, y_pred
    
    def _analyze_feature_importance(self, X_test: np.ndarray, y_test: List[int]) -> None:
        """특성 중요도 분석"""
        logger.info("Analyzing feature importance...")
        
        feature_importance = self.model.feature_importances_
        
        # 상위 20개 특성 중요도 출력
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\n=== Top 20 Feature Importance ===")
        print(importance_df.head(20))
        
        # 특성 중요도 시각화
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title('Top 15 Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # 그래프 저장
        output_dir = Path("models")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Feature importance plot saved to models/feature_importance.png")
    
    def save_model(self, model_path: str) -> None:
        """모델 저장"""
        output_path = Path(model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 모델과 벡터라이저를 함께 저장
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, output_path)
        logger.info(f"Model saved to {output_path}")
    
    def cross_validate(self, X: np.ndarray, y: List[int]) -> None:
        """교차 검증 수행"""
        logger.info("Performing cross-validation...")
        
        cv_scores = cross_val_score(
            self.model, X, y, cv=5, scoring='accuracy', n_jobs=-1
        )
        
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


def main():
    """메인 함수"""
    # 훈련 데이터 경로
    training_data_path = "data/training/article_classification_training_data.json"
    
    # 훈련기 생성
    trainer = MLModelTrainer(training_data_path)
    
    # 훈련 데이터 로드
    X_features, X_text, y = trainer.load_training_data()
    
    # 특성 준비
    X = trainer.prepare_features(X_features, X_text)
    
    # 모델 훈련
    X_test, y_test, y_pred = trainer.train_model(X, y)
    
    # 교차 검증
    trainer.cross_validate(X, y)
    
    # 모델 저장
    model_path = "models/article_classifier.pkl"
    trainer.save_model(model_path)
    
    print("\n=== Training Complete ===")
    print(f"Model saved to: {model_path}")
    print("Ready for integration with the article parser!")


if __name__ == "__main__":
    main()