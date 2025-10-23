#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT 기반 텍스트 분류기
- 입력: 질의 텍스트, 카테고리, 서브카테고리
- 출력: restricted/allowed 확률
- 학습 데이터: test_results/massive_test_results_*.json 의 detailed_results
"""

import os
import glob
import json
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
import pandas as pd

# 한국어 BERT 모델 사용
MODEL_NAME = "klue/bert-base"
DEFAULT_MODEL_PATH = os.path.join("models", "bert_classifier")


class LegalDataset(Dataset):
    """법률 질의 데이터셋"""
    
    def __init__(self, texts, labels, categories, subcategories, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.categories = categories
        self.subcategories = subcategories
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        category = str(self.categories[idx])
        subcategory = str(self.subcategories[idx])
        
        # 텍스트에 카테고리 정보 추가
        full_text = f"[{category}] [{subcategory}] {text}"
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_latest_dataset() -> Tuple[List[str], List[int], List[str], List[str]]:
    """최신 테스트 결과에서 데이터셋 로드"""
    files = glob.glob(os.path.join("test_results", "massive_test_results_*.json"))
    files = [f for f in files if not f.endswith("_analysis.json")]
    if not files:
        raise FileNotFoundError("학습을 위한 결과 파일이 없습니다. 먼저 통합 테스트를 실행하세요.")
    latest = max(files, key=os.path.getctime)
    
    with open(latest, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    
    texts = []
    labels = []
    categories = []
    subcategories = []
    
    for r in data.get("detailed_results", []):
        query = r.get("query", "")
        label = 1 if r.get("expected_restricted", False) else 0
        category = r.get("category", "unknown")
        subcategory = r.get("subcategory", "unknown")
        
        if query:
            texts.append(query)
            labels.append(label)
            categories.append(category)
            subcategories.append(subcategory)
    
    if not texts:
        raise ValueError("데이터셋이 비어있습니다")
    
    return texts, labels, categories, subcategories


def compute_metrics(eval_pred):
    """평가 메트릭 계산"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_bert_classifier(model_path: str = DEFAULT_MODEL_PATH) -> str:
    """BERT 분류기 학습"""
    print("BERT 분류기 학습 시작...")
    
    # 데이터 로드
    texts, labels, categories, subcategories = load_latest_dataset()
    
    # 토크나이저와 모델 초기화
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # 데이터셋 생성
    dataset = LegalDataset(texts, labels, categories, subcategories, tokenizer)
    
    # 학습/검증 분할
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 학습 설정
    training_args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{model_path}/logs',
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )
    
    # 트레이너 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 학습 실행
    trainer.train()
    
    # 모델 저장
    trainer.save_model()
    tokenizer.save_pretrained(model_path)
    
    print(f"BERT 분류기 학습 완료: {model_path}")
    return model_path


def load_bert_model(model_path: str = DEFAULT_MODEL_PATH):
    """BERT 모델 로드"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    return tokenizer, model


def predict_proba(query: str, category: str = "unknown", subcategory: str = "unknown", 
                 model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, float]:
    """BERT 모델로 예측"""
    tokenizer, model = load_bert_model(model_path)
    
    # 텍스트에 카테고리 정보 추가
    full_text = f"[{category}] [{subcategory}] {query}"
    
    # 토크나이징
    inputs = tokenizer(
        full_text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )
    
    # 예측
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
    
    # 결과 반환
    allowed_p = float(probabilities[0][0])  # 0: allowed
    restricted_p = float(probabilities[0][1])  # 1: restricted
    
    # 카테고리별 차별화 적용 (균형잡힌 기준)
    sensitive_categories = ["personal_legal_advice", "medical_legal_advice", "criminal_case_advice", "illegal_activity_assistance"]
    if category in sensitive_categories:
        if category == "medical_legal_advice":
            restricted_p = max(restricted_p, 0.35)  # 의료법은 0.35로 조정 (0.25 → 0.35)
        elif category == "criminal_case_advice":
            restricted_p = max(restricted_p, 0.5)   # 형사법은 0.5로 조정 (0.4 → 0.5)
        elif category == "illegal_activity_assistance":
            restricted_p = max(restricted_p, 0.6)   # 불법행위는 0.6으로 조정 (0.5 → 0.6)
        else:
            restricted_p = max(restricted_p, 0.45)  # 개인 자문은 0.45로 조정 (0.35 → 0.45)
        allowed_p = 1.0 - restricted_p
    
    return {"allowed": allowed_p, "restricted": restricted_p}


class BERTClassifier:
    """BERT 기반 분류기 클래스"""
    
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """모델 로딩"""
        try:
            self.tokenizer, self.model = load_bert_model(self.model_path)
        except FileNotFoundError:
            self.tokenizer = None
            self.model = None
            print(f"BERT 모델 파일이 없습니다. 먼저 학습을 실행하세요: {self.model_path}")
    
    def train(self) -> str:
        """모델 학습"""
        return train_bert_classifier(self.model_path)
    
    def predict(self, query: str, category: str = "unknown", subcategory: str = "unknown") -> Dict[str, float]:
        """예측 수행"""
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. 먼저 train()을 실행하세요.")
        return predict_proba(query, category, subcategory, self.model_path)
    
    def is_available(self) -> bool:
        """모델 사용 가능 여부"""
        return self.model is not None


if __name__ == "__main__":
    # 학습 실행
    model_path = train_bert_classifier()
    print(f"✅ BERT 분류기 학습 완료: {model_path}")
