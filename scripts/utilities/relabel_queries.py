#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
생성된 massive_test_queries_*.json 파일의 expected_restricted 라벨을
카테고리/질의 규칙에 따라 재정렬합니다.
"""

import os
import sys
import json
import glob
from typing import Dict, Any

from scripts.massive_test_query_generator import MassiveTestQueryGenerator


def relabel_file(path: str) -> str:
    gen = MassiveTestQueryGenerator()
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    queries = data.get("queries", [])
    count_changed = 0
    for q in queries:
        cat = q.get("category", "")
        txt = q.get("query", "")
        new_label = gen._determine_expected_result(cat, txt)
        if q.get("expected_restricted") != new_label:
            q["expected_restricted"] = new_label
            count_changed += 1
    out_path = path.replace(".json", "_relabeled.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    # Windows cp949 출력 이슈 회피: ASCII만 출력
    print(f"Relabeled: {out_path} (changed {count_changed})")
    return out_path


def main():
    files = glob.glob(os.path.join("test_results", "massive_test_queries_*.json"))
    if not files:
        print("질의 파일이 없습니다. 먼저 질의 생성기를 실행하세요.")
        return
    latest = max(files, key=os.path.getctime)
    relabel_file(latest)


if __name__ == "__main__":
    main()


