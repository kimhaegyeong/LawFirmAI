#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
판례 상세 페이지 목차 구조 분석
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from core.data.assembly_playwright_client import AssemblyPlaywrightClient

def analyze_precedent_structure():
    """판례 상세 페이지의 목차 구조 분석"""

    with AssemblyPlaywrightClient(headless=False) as client:
        # 민사 판례 첫 번째 항목으로 이동
        print("🔍 Analyzing precedent detail page structure...")

        # 판례 목록 페이지로 이동
        precedents = client.get_precedent_list_page_by_category("PREC00_001", 1, 10)

        if precedents:
            first_precedent = precedents[0]
            print(f"📋 Analyzing: {first_precedent['case_name']}")

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

            # 페이지 구조 분석
            print("\n📊 Page Structure Analysis:")

            # 1. 헤더 정보
            print("\n1️⃣ Header Information:")
            try:
                h1_elements = client.page.locator("h1").all()
                for i, h1 in enumerate(h1_elements):
                    print(f"   H1-{i+1}: {h1.inner_text().strip()}")
            except Exception as e:
                print(f"   H1 Error: {e}")

            # 2. 목차/섹션 구조
            print("\n2️⃣ Table of Contents / Sections:")
            try:
                # 다양한 목차 패턴 시도
                toc_selectors = [
                    "div.toc", "div.table-of-contents", "div.contents",
                    "ul.toc", "ol.toc", "div.section",
                    "div.content-section", "div.precedent-section"
                ]

                for selector in toc_selectors:
                    elements = client.page.locator(selector).all()
                    if elements:
                        print(f"   Found {selector}: {len(elements)} elements")
                        for i, elem in enumerate(elements[:3]):  # 처음 3개만
                            text = elem.inner_text().strip()
                            if text:
                                print(f"     {i+1}: {text[:100]}...")

                # 일반적인 헤딩 태그들
                heading_tags = ["h2", "h3", "h4", "h5", "h6"]
                for tag in heading_tags:
                    elements = client.page.locator(tag).all()
                    if elements:
                        print(f"   {tag.upper()} headings: {len(elements)}")
                        for i, elem in enumerate(elements[:5]):  # 처음 5개만
                            text = elem.inner_text().strip()
                            if text:
                                print(f"     {i+1}: {text}")

            except Exception as e:
                print(f"   TOC Error: {e}")

            # 3. 판례 관련 특정 섹션들
            print("\n3️⃣ Legal Sections:")
            legal_sections = [
                "판시사항", "판결요지", "사건", "원고", "피고", "원심", "상고",
                "항소", "소송", "계약", "손해", "배상", "위약", "해지", "무효", "취소",
                "주문", "이유", "참조조문", "참조판례", "판례집", "법원", "선고일"
            ]

            for section in legal_sections:
                try:
                    elements = client.page.locator(f"text={section}").all()
                    if elements:
                        print(f"   '{section}': {len(elements)} occurrences")
                        for i, elem in enumerate(elements[:2]):  # 처음 2개만
                            parent_text = elem.locator("..").inner_text().strip()
                            if parent_text:
                                print(f"     {i+1}: {parent_text[:100]}...")
                except Exception as e:
                    print(f"   '{section}' Error: {e}")

            # 4. 테이블 구조
            print("\n4️⃣ Table Structure:")
            try:
                tables = client.page.locator("table").all()
                print(f"   Tables found: {len(tables)}")
                for i, table in enumerate(tables):
                    rows = table.locator("tr").all()
                    print(f"   Table {i+1}: {len(rows)} rows")
                    if rows:
                        # 첫 번째 행 (헤더) 확인
                        first_row = rows[0]
                        cells = first_row.locator("td, th").all()
                        headers = [cell.inner_text().strip() for cell in cells]
                        print(f"     Headers: {headers}")
            except Exception as e:
                print(f"   Table Error: {e}")

            # 5. 전체 텍스트 구조 분석
            print("\n5️⃣ Text Structure Analysis:")
            try:
                full_text = client.page.locator("body").inner_text()
                lines = full_text.split('\n')

                print(f"   Total lines: {len(lines)}")
                print(f"   Total characters: {len(full_text)}")

                # 빈 줄이 아닌 라인들
                non_empty_lines = [line.strip() for line in lines if line.strip()]
                print(f"   Non-empty lines: {len(non_empty_lines)}")

                # 긴 라인들 (50자 이상)
                long_lines = [line for line in non_empty_lines if len(line) >= 50]
                print(f"   Long lines (50+ chars): {len(long_lines)}")

                # 첫 10개 긴 라인 출력
                print("   Sample long lines:")
                for i, line in enumerate(long_lines[:10]):
                    print(f"     {i+1}: {line[:80]}...")

            except Exception as e:
                print(f"   Text Analysis Error: {e}")

            # 6. HTML 구조 저장
            print("\n6️⃣ Saving HTML structure for analysis...")
            try:
                html_content = client.page.content()
                with open("precedent_structure_analysis.html", "w", encoding="utf-8") as f:
                    f.write(html_content)
                print("   HTML saved to: precedent_structure_analysis.html")
            except Exception as e:
                print(f"   HTML Save Error: {e}")

if __name__ == "__main__":
    analyze_precedent_structure()
