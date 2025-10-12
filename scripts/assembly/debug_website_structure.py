#!/usr/bin/env python3
"""
국회 법률정보시스템 웹사이트 구조 분석 도구
실제 법률 내용이 어떻게 로드되는지 분석
"""

import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright
import json

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def analyze_website_structure():
    """웹사이트 구조 분석"""
    
    # 테스트할 법률 정보
    test_law = {
        'cont_id': '1981022300000003',
        'cont_sid': '0030',
        'law_name': '집행관수수료규칙'
    }
    
    print("🔍 국회 법률정보시스템 구조 분석 시작")
    print(f"📋 테스트 법률: {test_law['law_name']}")
    print(f"📋 cont_id: {test_law['cont_id']}")
    print(f"📋 cont_sid: {test_law['cont_sid']}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # 브라우저를 보이게 실행
        page = browser.new_page()
        
        # 네트워크 요청 모니터링
        requests = []
        responses = []
        
        def handle_request(request):
            requests.append({
                'url': request.url,
                'method': request.method,
                'headers': dict(request.headers)
            })
        
        def handle_response(response):
            responses.append({
                'url': response.url,
                'status': response.status,
                'headers': dict(response.headers)
            })
        
        page.on("request", handle_request)
        page.on("response", handle_response)
        
        try:
            # 1. 법률 상세 페이지 접근
            url = f"https://likms.assembly.go.kr/law/lawsLawtInqyDetl1010.do"
            params = f"?contId={test_law['cont_id']}&contSid={test_law['cont_sid']}&viewGb=PROM&genMenuId=menu_serv_nlaw_lawt_1020"
            
            print(f"\n🌐 페이지 접근: {url}{params}")
            page.goto(url + params, wait_until='domcontentloaded')
            
            # 2. 페이지 로딩 대기
            print("⏳ 페이지 로딩 대기...")
            time.sleep(5)
            
            # 3. 페이지 구조 분석
            print("\n📊 페이지 구조 분석:")
            
            # HTML 구조 분석
            html_content = page.content()
            print(f"📄 전체 HTML 길이: {len(html_content)} chars")
            
            # iframe 분석
            iframes = page.locator("iframe").all()
            print(f"🖼️ iframe 개수: {len(iframes)}")
            
            for i, iframe in enumerate(iframes):
                try:
                    src = iframe.get_attribute("src")
                    name = iframe.get_attribute("name")
                    id_attr = iframe.get_attribute("id")
                    print(f"   iframe {i+1}: src='{src}', name='{name}', id='{id_attr}'")
                    
                    # iframe 내용 확인
                    iframe_content = iframe.content_frame()
                    if iframe_content:
                        iframe_text = iframe_content.locator("body").inner_text()
                        print(f"   iframe {i+1} 내용 길이: {len(iframe_text)} chars")
                        
                        # 법률 내용 확인
                        if any(keyword in iframe_text for keyword in ['제1조', '제2조', '조문']):
                            print(f"   ✅ iframe {i+1}에 법률 내용 발견!")
                            print(f"   📝 샘플 내용: {iframe_text[:200]}...")
                        else:
                            print(f"   ❌ iframe {i+1}에 법률 내용 없음")
                except Exception as e:
                    print(f"   ⚠️ iframe {i+1} 접근 오류: {e}")
            
            # JavaScript 실행 후 동적 콘텐츠 확인
            print("\n🔧 JavaScript 실행 후 분석:")
            
            # 모든 div 요소 확인
            divs = page.locator("div").all()
            print(f"📦 div 요소 개수: {len(divs)}")
            
            # 법률 관련 클래스나 ID가 있는 요소 찾기
            law_elements = []
            for selector in [
                "div[class*='law']",
                "div[id*='law']", 
                "div[class*='content']",
                "div[id*='content']",
                "div[class*='text']",
                "div[id*='text']",
                "div[class*='view']",
                "div[id*='view']"
            ]:
                elements = page.locator(selector).all()
                for elem in elements:
                    try:
                        text = elem.inner_text().strip()
                        if len(text) > 100 and any(keyword in text for keyword in ['제1조', '제2조', '조문', '법률']):
                            law_elements.append({
                                'selector': selector,
                                'text_length': len(text),
                                'sample': text[:200]
                            })
                    except:
                        continue
            
            print(f"📋 법률 관련 요소 발견: {len(law_elements)}개")
            for elem in law_elements:
                print(f"   {elem['selector']}: {elem['text_length']} chars")
                print(f"   샘플: {elem['sample']}...")
            
            # 네트워크 요청 분석
            print(f"\n🌐 네트워크 요청 분석:")
            print(f"📤 요청 개수: {len(requests)}")
            print(f"📥 응답 개수: {len(responses)}")
            
            # 법률 관련 요청 찾기
            law_requests = []
            for req in requests:
                if any(keyword in req['url'].lower() for keyword in ['law', 'cont', 'detl', 'ajax']):
                    law_requests.append(req)
            
            print(f"📋 법률 관련 요청: {len(law_requests)}개")
            for req in law_requests[:5]:  # 처음 5개만 출력
                print(f"   {req['method']} {req['url']}")
            
            # 페이지 스크린샷 저장
            screenshot_path = "website_structure_analysis.png"
            page.screenshot(path=screenshot_path)
            print(f"\n📸 스크린샷 저장: {screenshot_path}")
            
            # 분석 결과 저장
            analysis_result = {
                'test_law': test_law,
                'html_length': len(html_content),
                'iframe_count': len(iframes),
                'law_elements': law_elements,
                'requests': requests,
                'responses': responses,
                'timestamp': time.time()
            }
            
            with open("website_analysis_result.json", "w", encoding="utf-8") as f:
                json.dump(analysis_result, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 분석 결과 저장: website_analysis_result.json")
            
            # 브라우저를 잠시 열어두어 수동 확인 가능
            print(f"\n🔍 브라우저가 열려있습니다. 수동으로 확인해보세요.")
            print(f"⏳ 30초 후 자동으로 닫힙니다...")
            time.sleep(30)
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    analyze_website_structure()
