#!/usr/bin/env python3
"""
ë¨¸ì‹ ?¬ë‹ ëª¨ë¸ ?ˆë ¨ ?¤í¬ë¦½íŠ¸
ì¡°ë¬¸ ë¶„ë¥˜ë¥??„í•œ RandomForest ëª¨ë¸ ?ˆë ¨ ë°?ê²€ì¦?
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

# ë¡œê¹… ?¤ì •
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModelTrainer:
    """ë¨¸ì‹ ?¬ë‹ ëª¨ë¸ ?ˆë ¨ ?´ë˜??""
    
    def __init__(self, training_data_path: str):
        """
        ì´ˆê¸°??
        
        Args:
            training_data_path: ?ˆë ¨ ?°ì´???Œì¼ ê²½ë¡œ
        """
        self.training_data_path = Path(training_data_path)
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        
    def load_training_data(self) -> Tuple[List[Dict[str, Any]], List[int]]:
        """?ˆë ¨ ?°ì´??ë¡œë“œ ë°??„ì²˜ë¦?""
        logger.info(f"Loading training data from {self.training_data_path}")
        
        with open(self.training_data_path, 'r', encoding='utf-8') as f:
            training_samples = json.load(f)
        
        logger.info(f"Loaded {len(training_samples)} training samples")
        
        # ?¹ì„±ê³??ˆì´ë¸?ë¶„ë¦¬
        X_features = []
        X_text = []
        y = []
        
        for sample in training_samples:
            features = sample['features']
            
            # ?˜ì¹˜???¹ì„± ì¶”ì¶œ
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
            
            # ?ìŠ¤???¹ì„± (ë¬¸ë§¥ ?•ë³´)
            context_text = f"{sample.get('article_number', '')} {sample.get('article_title', '')}"
            X_text.append(context_text)
            
            # ?ˆì´ë¸?(real_article: 1, reference: 0)
            label = 1 if sample['label'] == 'real_article' else 0
            y.append(label)
        
        logger.info(f"Extracted {len(X_features)} feature vectors")
        logger.info(f"Real articles: {sum(y)}, References: {len(y) - sum(y)}")
        
        return X_features, X_text, y
    
    def prepare_features(self, X_features: List[List[float]], X_text: List[str]) -> np.ndarray:
        """?¹ì„± ë²¡í„° ì¤€ë¹?""
        logger.info("Preparing feature vectors...")
        
        # ?˜ì¹˜???¹ì„±??numpy ë°°ì—´ë¡?ë³€??
        numerical_features = np.array(X_features)
        
        # ?ìŠ¤???¹ì„±??TF-IDFë¡?ë³€??
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words=None,  # ?œêµ­?´ëŠ” ë¶ˆìš©???œê±°?˜ì? ?ŠìŒ
            min_df=2,
            max_df=0.95
        )
        
        text_features = self.vectorizer.fit_transform(X_text)
        
        # ?˜ì¹˜???¹ì„±ê³??ìŠ¤???¹ì„± ê²°í•©
        combined_features = np.hstack([
            numerical_features,
            text_features.toarray()
        ])
        
        logger.info(f"Combined feature shape: {combined_features.shape}")
        
        # ?¹ì„± ?´ë¦„ ?€??
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
        """ëª¨ë¸ ?ˆë ¨"""
        logger.info("Training RandomForest model...")
        
        # ?ˆë ¨/ê²€ì¦??°ì´??ë¶„í• 
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # ?˜ì´?¼íŒŒ?¼ë????œë‹
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        logger.info("Performing hyperparameter tuning...")
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # ê·¸ë¦¬???œì¹˜ (?œê°„ ?¨ì¶•???„í•´ ?œí•œ??
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # ëª¨ë¸ ?‰ê?
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        # ?ì„¸ ?‰ê? ë¦¬í¬??
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred, target_names=['Reference', 'Real Article']))
        
        # ?¼ë™ ?‰ë ¬
        cm = confusion_matrix(y_test, y_pred)
        print("\n=== Confusion Matrix ===")
        print(cm)
        
        # ?¹ì„± ì¤‘ìš”??ë¶„ì„
        self._analyze_feature_importance(X_test, y_test)
        
        return X_test, y_test, y_pred
    
    def _analyze_feature_importance(self, X_test: np.ndarray, y_test: List[int]) -> None:
        """?¹ì„± ì¤‘ìš”??ë¶„ì„"""
        logger.info("Analyzing feature importance...")
        
        feature_importance = self.model.feature_importances_
        
        # ?ìœ„ 20ê°??¹ì„± ì¤‘ìš”??ì¶œë ¥
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\n=== Top 20 Feature Importance ===")
        print(importance_df.head(20))
        
        # ?¹ì„± ì¤‘ìš”???œê°??
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title('Top 15 Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # ê·¸ë˜???€??
        output_dir = Path("models")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Feature importance plot saved to models/feature_importance.png")
    
    def save_model(self, model_path: str) -> None:
        """ëª¨ë¸ ?€??""
        output_path = Path(model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ëª¨ë¸ê³?ë²¡í„°?¼ì´?€ë¥??¨ê»˜ ?€??
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, output_path)
        logger.info(f"Model saved to {output_path}")
    
    def cross_validate(self, X: np.ndarray, y: List[int]) -> None:
        """êµì°¨ ê²€ì¦??˜í–‰"""
        logger.info("Performing cross-validation...")
        
        cv_scores = cross_val_score(
            self.model, X, y, cv=5, scoring='accuracy', n_jobs=-1
        )
        
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    # ?ˆë ¨ ?°ì´??ê²½ë¡œ
    training_data_path = "data/training/article_classification_training_data.json"
    
    # ?ˆë ¨ê¸??ì„±
    trainer = MLModelTrainer(training_data_path)
    
    # ?ˆë ¨ ?°ì´??ë¡œë“œ
    X_features, X_text, y = trainer.load_training_data()
    
    # ?¹ì„± ì¤€ë¹?
    X = trainer.prepare_features(X_features, X_text)
    
    # ëª¨ë¸ ?ˆë ¨
    X_test, y_test, y_pred = trainer.train_model(X, y)
    
    # êµì°¨ ê²€ì¦?
    trainer.cross_validate(X, y)
    
    # ëª¨ë¸ ?€??
    model_path = "models/article_classifier.pkl"
    trainer.save_model(model_path)
    
    print("\n=== Training Complete ===")
    print(f"Model saved to: {model_path}")
    print("Ready for integration with the article parser!")


if __name__ == "__main__":
    main()