#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
판례명 추출 문제 분석 및 수정
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from core.data.assembly_playwright_client import AssemblyPlaywrightClient

def analyze_case_name_extraction():
    """판례명 추출 문제 분석"""

    with AssemblyPlaywrightClient(headless=False) as client:
        print("🔍 Analyzing case name extraction issue...")

        # 민사 판례 첫 번째 항목으로 이동
        precedents = client.get_precedent_list_page_by_category("PREC00_001", 1, 10)

        if precedents:
            first_precedent = precedents[0]
            print(f"📋 List page case name: {first_precedent['case_name']}")

            # 상세 페이지로 이동
            url = f"{client.BASE_URL}/law/lawsPrecInqyDetl1010.do"
            url_params = []
            for key, value in first_precedent['params'].items():
                if value:
                    url_params.append(f"{key}={value}")

            params_str = "&".join(url_params)
            full_url = f"{url}?{params_str}"

            print(f"🌐 Navigating to: {full_url}")
            client.page.goto(full_url, wait_until='domcontentloaded')
            client.page.wait_for_timeout(5000)

            # 페이지에서 판례명 후보들 찾기
            print(f"\n📊 Case Name Candidates Analysis:")

            # 1. 모든 h1 태그 확인
            print(f"\n1️⃣ H1 Tags:")
            h1_elements = client.page.locator("h1").all()
            for i, h1 in enumerate(h1_elements):
                text = h1.inner_text().strip()
                print(f"   H1-{i+1}: {text}")

            # 2. 모든 h2 태그 확인
            print(f"\n2️⃣ H2 Tags:")
            h2_elements = client.page.locator("h2").all()
            for i, h2 in enumerate(h2_elements):
                text = h2.inner_text().strip()
                print(f"   H2-{i+1}: {text}")

            # 3. 판례명이 포함될 수 있는 특정 패턴 찾기
            print(f"\n3️⃣ Case Name Patterns:")

            # 대괄호로 둘러싸인 텍스트 (예: [배당이의의소])
            bracket_patterns = client.page.locator("text=/\\[.*?\\]/").all()
            for i, pattern in enumerate(bracket_patterns):
                text = pattern.inner_text().strip()
                print(f"   Bracket-{i+1}: {text}")

            # 판례 관련 키워드가 있는 텍스트
            case_keywords = ["소", "사건", "청구", "소송", "이의", "항고", "항소", "상고"]
            for keyword in case_keywords:
                elements = client.page.locator(f"text={keyword}").all()
                if elements:
                    print(f"   Keyword '{keyword}': {len(elements)} occurrences")
                    for i, elem in enumerate(elements[:3]):  # 처음 3개만
                        text = elem.inner_text().strip()
                        if len(text) < 100:  # 너무 긴 텍스트 제외
                            print(f"     {i+1}: {text}")

            # 4. 페이지 제목 확인
            print(f"\n4️⃣ Page Title:")
            title = client.page.title()
            print(f"   Title: {title}")

            # 5. 특정 클래스나 ID가 있는 요소들 확인
            print(f"\n5️⃣ Specific Elements:")

            # contents 클래스 확인
            contents_elements = client.page.locator(".contents").all()
            for i, elem in enumerate(contents_elements):
                text = elem.inner_text().strip()
                print(f"   Contents-{i+1}: {text[:200]}...")

            # 6. 전체 텍스트에서 판례명 패턴 찾기
            print(f"\n6️⃣ Text Pattern Analysis:")
            full_text = client.page.locator("body").inner_text()
            lines = full_text.split('\n')

            # 대괄호로 둘러싸인 텍스트 찾기
            import re
            bracket_matches = re.findall(r'\[([^\]]+)\]', full_text)
            if bracket_matches:
                print(f"   Bracket matches: {bracket_matches}")

            # 판례명으로 보이는 짧은 텍스트들 찾기
            short_lines = [line.strip() for line in lines if 5 <= len(line.strip()) <= 50 and line.strip()]
            print(f"   Short lines (5-50 chars): {len(short_lines)}")
            for i, line in enumerate(short_lines[:10]):  # 처음 10개만
                print(f"     {i+1}: {line}")

if __name__ == "__main__":
    analyze_case_name_extraction()
