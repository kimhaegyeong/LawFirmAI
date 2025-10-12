#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
국회 법률정보시스템 판례 페이지 구조 분석
"""

import sys
from pathlib import Path
import json
import time
from playwright.sync_api import sync_playwright

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.assembly_playwright_client import AssemblyPlaywrightClient

def debug_precedent_page():
    print(f"🔍 국회 법률정보시스템 판례 페이지 구조 분석 시작")
    
    analysis_results = {
        "page_url": "https://likms.assembly.go.kr/law/lawsPrecInqyList2010.do",
        "html_length": 0,
        "table_structure": {},
        "pagination_info": {},
        "form_elements": {},
        "javascript_functions": [],
        "network_requests": [],
        "console_logs": []
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.set_default_timeout(60000)
        
        # 네트워크 요청 로깅
        page.on("request", lambda request: analysis_results["network_requests"].append({
            "url": request.url,
            "method": request.method,
            "headers": request.headers
        }))
        page.on("response", lambda response: print(f"📥 응답: {response.status} {response.url}"))
        page.on("console", lambda msg: analysis_results["console_logs"].append({"type": msg.type, "text": msg.text}))

        # 판례 목록 페이지 접근
        url = "https://likms.assembly.go.kr/law/lawsPrecInqyList2010.do?genActiontypeCd=2ACT1010&genDoctreattypeCd=&genMenuId=menu_serv_nlaw_lawt_4020&procWorkId="
        print(f"\n🌐 페이지 접근: {url}")
        page.goto(url, wait_until='domcontentloaded')
        print(f"⏳ 페이지 로딩 대기...")
        page.wait_for_timeout(5000)

        analysis_results["html_length"] = len(page.content())
        print(f"\n📊 페이지 구조 분석:")
        print(f"📄 전체 HTML 길이: {analysis_results['html_length']} chars")

        # 테이블 구조 분석
        print(f"\n📋 테이블 구조 분석:")
        tables = page.locator("table").all()
        print(f"테이블 개수: {len(tables)}")
        
        for i, table in enumerate(tables):
            try:
                rows = table.locator("tr").all()
                cols = table.locator("td, th").all()
                table_info = {
                    "table_index": i,
                    "row_count": len(rows),
                    "col_count": len(cols),
                    "headers": [],
                    "sample_data": []
                }
                
                # 헤더 분석
                if rows:
                    header_row = rows[0]
                    headers = header_row.locator("th, td").all()
                    for header in headers:
                        header_text = header.inner_text().strip()
                        if header_text:
                            table_info["headers"].append(header_text)
                
                # 샘플 데이터 분석 (첫 번째 데이터 행)
                if len(rows) > 1:
                    data_row = rows[1]
                    cells = data_row.locator("td").all()
                    for cell in cells:
                        cell_text = cell.inner_text().strip()
                        if cell_text:
                            table_info["sample_data"].append(cell_text)
                
                analysis_results["table_structure"][f"table_{i}"] = table_info
                print(f"   테이블 {i+1}: {len(rows)}행, {len(cols)}열")
                print(f"   헤더: {table_info['headers']}")
                print(f"   샘플: {table_info['sample_data'][:3]}...")
                
            except Exception as e:
                print(f"   테이블 {i+1} 분석 오류: {e}")

        # 페이지네이션 분석
        print(f"\n📄 페이지네이션 분석:")
        pagination_selectors = [
            "span.page_no", "div.pagination", "div.page", "ul.pagination",
            "span[class*='page']", "div[class*='page']", "ul[class*='page']"
        ]
        
        pagination_info = {}
        for selector in pagination_selectors:
            elements = page.locator(selector).all()
            if elements:
                pagination_info[selector] = {
                    "count": len(elements),
                    "text": [elem.inner_text().strip() for elem in elements],
                    "html": [elem.inner_html() for elem in elements]
                }
                print(f"   {selector}: {len(elements)}개 발견")
                for elem in elements:
                    text = elem.inner_text().strip()
                    if text:
                        print(f"     텍스트: {text[:100]}...")
        
        analysis_results["pagination_info"] = pagination_info

        # 폼 요소 분석
        print(f"\n📝 폼 요소 분석:")
        forms = page.locator("form").all()
        print(f"폼 개수: {len(forms)}")
        
        for i, form in enumerate(forms):
            form_info = {
                "form_index": i,
                "action": form.get_attribute("action"),
                "method": form.get_attribute("method"),
                "inputs": [],
                "selects": [],
                "buttons": []
            }
            
            # 입력 필드
            inputs = form.locator("input").all()
            for input_elem in inputs:
                input_info = {
                    "type": input_elem.get_attribute("type"),
                    "name": input_elem.get_attribute("name"),
                    "id": input_elem.get_attribute("id"),
                    "value": input_elem.get_attribute("value")
                }
                form_info["inputs"].append(input_info)
            
            # 선택 박스
            selects = form.locator("select").all()
            for select_elem in selects:
                select_info = {
                    "name": select_elem.get_attribute("name"),
                    "id": select_elem.get_attribute("id"),
                    "options": []
                }
                options = select_elem.locator("option").all()
                for option in options:
                    option_info = {
                        "value": option.get_attribute("value"),
                        "text": option.inner_text().strip()
                    }
                    select_info["options"].append(option_info)
                form_info["selects"].append(select_info)
            
            # 버튼
            buttons = form.locator("button, input[type='submit'], input[type='button']").all()
            for button in buttons:
                button_info = {
                    "type": button.get_attribute("type"),
                    "value": button.get_attribute("value"),
                    "text": button.inner_text().strip(),
                    "onclick": button.get_attribute("onclick")
                }
                form_info["buttons"].append(button_info)
            
            analysis_results["form_elements"][f"form_{i}"] = form_info
            print(f"   폼 {i+1}: action='{form_info['action']}', method='{form_info['method']}'")
            print(f"     입력: {len(form_info['inputs'])}개")
            print(f"     선택: {len(form_info['selects'])}개")
            print(f"     버튼: {len(form_info['buttons'])}개")

        # JavaScript 함수 분석
        print(f"\n🔧 JavaScript 함수 분석:")
        script_tags = page.locator("script").all()
        js_functions = []
        
        for script in script_tags:
            script_content = script.inner_html()
            if script_content:
                # 함수 정의 찾기
                if "function" in script_content:
                    lines = script_content.split('\n')
                    for line in lines:
                        if "function" in line and "(" in line:
                            func_name = line.split("function")[1].split("(")[0].strip()
                            if func_name:
                                js_functions.append(func_name)
        
        analysis_results["javascript_functions"] = list(set(js_functions))
        print(f"발견된 함수: {analysis_results['javascript_functions']}")

        # 네트워크 요청 분석
        print(f"\n🌐 네트워크 요청 분석:")
        print(f"📤 요청 개수: {len(analysis_results['network_requests'])}")
        
        # 판례 관련 요청 필터링
        precedent_requests = [req for req in analysis_results["network_requests"] 
                            if 'prec' in req['url'].lower() or 'precedent' in req['url'].lower()]
        print(f"📋 판례 관련 요청: {len(precedent_requests)}개")
        
        for req in precedent_requests[:5]:  # 처음 5개만 출력
            print(f"   {req['method']} {req['url']}")

        # 스크린샷 저장
        screenshot_path = "precedent_page_analysis.png"
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"\n📸 스크린샷 저장: {screenshot_path}")

        # 결과 JSON 저장
        result_file = "precedent_page_analysis_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 분석 결과 저장: {result_file}")

        print(f"\n🔍 브라우저가 열려있습니다. 수동으로 확인해보세요.")
        print(f"⏳ 60초 후 자동으로 닫힙니다...")
        time.sleep(60)
        browser.close()

if __name__ == "__main__":
    debug_precedent_page()
