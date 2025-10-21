#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 TF-IDF + LogisticRegression 분류기
 - 입력: 질의 텍스트
 - 출력: restricted/allowed 확률
 - 학습 데이터: test_results/massive_test_results_*.json 의 detailed_results (label=expected_restricted)
"""

import os
import glob
import json
from typing import Dict, Any, List, Tuple

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import numpy as np


DEFAULT_MODEL_PATH = os.path.join("models", "simple_text_classifier.joblib")


def load_latest_dataset() -> Tuple[List[str], List[int], List[str], List[str]]:
    files = glob.glob(os.path.join("test_results", "massive_test_results_*.json"))
    files = [f for f in files if not f.endswith("_analysis.json")]
    if not files:
        raise FileNotFoundError("학습을 위한 결과 파일이 없습니다. 먼저 통합 테스트를 실행하세요.")
    latest = max(files, key=os.path.getctime)
    with open(latest, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    X: List[str] = []
    y: List[int] = []
    categories: List[str] = []
    subcategories: List[str] = []
    for r in data.get("detailed_results", []):
        q = r.get("query", "")
        label = 1 if r.get("expected_restricted", False) else 0
        category = r.get("category", "unknown")
        subcategory = r.get("subcategory", "unknown")
        if q:
            X.append(q)
            y.append(label)
            categories.append(category)
            subcategories.append(subcategory)
    if not X:
        raise ValueError("데이터셋이 비어있습니다")
    return X, y, categories, subcategories


def _char_ngrams(X: List[str]) -> List[str]:
    return X


def build_pipeline() -> Tuple[TfidfVectorizer, OneHotEncoder, LogisticRegression]:
    # 문자 n-gram 포함, 규제 강화(C↓)
    tfidf = TfidfVectorizer(ngram_range=(1, 5), analyzer="char_wb", min_df=2, max_features=200000)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
    return tfidf, ohe, clf


def train_and_save(model_path: str = DEFAULT_MODEL_PATH) -> str:
    X_text, y, categories, subcategories = load_latest_dataset()
    
    # 텍스트 특성 추출
    tfidf, ohe, clf = build_pipeline()
    X_tfidf = tfidf.fit_transform(X_text)
    
    # 카테고리/서브카테고리 one-hot 인코딩
    df_cat = pd.DataFrame({'category': categories, 'subcategory': subcategories})
    X_cat = ohe.fit_transform(df_cat)
    
    # 특성 결합
    X_combined = hstack([X_tfidf, X_cat])
    
    # 모델 학습
    clf.fit(X_combined, y)
    
    # 모델 저장
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_data = {
        'tfidf': tfidf,
        'ohe': ohe,
        'clf': clf
    }
    joblib.dump(model_data, model_path)
    return model_path


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    return joblib.load(model_path)


def predict_proba(query: str, category: str = "unknown", subcategory: str = "unknown", model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, float]:
    model_data = load_model(model_path)
    tfidf = model_data['tfidf']
    ohe = model_data['ohe']
    clf = model_data['clf']
    
    # 텍스트 특성 추출
    X_tfidf = tfidf.transform([query])
    
    # 카테고리/서브카테고리 one-hot 인코딩
    df_cat = pd.DataFrame({'category': [category], 'subcategory': [subcategory]})
    X_cat = ohe.transform(df_cat)
    
    # 특성 결합
    X_combined = hstack([X_tfidf, X_cat])
    
    # 예측
    proba = clf.predict_proba(X_combined)[0]
    
    # 클래스 순서는 model.classes_ (0: allowed, 1: restricted)로 가정
    class_to_idx = {int(c): i for i, c in enumerate(clf.classes_)}
    allowed_p = float(proba[class_to_idx.get(0, 0)])
    restricted_p = float(proba[class_to_idx.get(1, 1)])
    
    # 민감군 강제 제한 기준 상향 (카테고리별 차별화)
    sensitive_categories = ["personal_legal_advice", "medical_legal_advice", "criminal_case_advice", "illegal_activity_assistance"]
    if category in sensitive_categories:
        # 의료법 카테고리는 더 관대한 기준 적용
        if category == "medical_legal_advice":
            restricted_p = max(restricted_p, 0.5)  # 의료법은 0.5로 완화
        # 형사법 카테고리는 적당한 엄격함 적용 (완화)
        elif category == "criminal_case_advice":
            restricted_p = max(restricted_p, 0.65)  # 형사법은 0.65로 완화 (0.8 → 0.65)
        # 불법행위 카테고리도 더 엄격한 기준 적용
        elif category == "illegal_activity_assistance":
            restricted_p = max(restricted_p, 0.75)  # 불법행위는 0.75로 강화
        # 개인 법률 자문은 기본 기준 적용
        else:
            restricted_p = max(restricted_p, 0.6)  # 기본 0.6 유지
        allowed_p = 1.0 - restricted_p
    
    return {"allowed": allowed_p, "restricted": restricted_p}


class SimpleTextClassifier:
    """간단한 텍스트 분류기 클래스"""
    
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model_data = None
        self._load_model()
    
    def _load_model(self):
        """모델 로딩"""
        try:
            self.model_data = load_model(self.model_path)
        except FileNotFoundError:
            # 모델이 없으면 학습
            self.model_data = None
            print(f"모델 파일이 없습니다. 먼저 학습을 실행하세요: {self.model_path}")
    
    def train(self) -> str:
        """모델 학습"""
        return train_and_save(self.model_path)
    
    def predict(self, query: str, category: str = "unknown", subcategory: str = "unknown") -> Dict[str, float]:
        """예측 수행"""
        if self.model_data is None:
            raise RuntimeError("모델이 로드되지 않았습니다. 먼저 train()을 실행하세요.")
        return predict_proba(query, category, subcategory, self.model_path)
    
    def is_available(self) -> bool:
        """모델 사용 가능 여부"""
        return self.model_data is not None
