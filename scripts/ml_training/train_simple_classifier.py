#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 TF-IDF + LogisticRegression 분류기 학습 스크립트
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from source.services.simple_text_classifier import train_and_save, DEFAULT_MODEL_PATH


def main():
    path = train_and_save(DEFAULT_MODEL_PATH)
    print(f"✅ 분류기 학습 완료: {path}")


if __name__ == "__main__":
    main()


