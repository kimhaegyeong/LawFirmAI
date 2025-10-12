#!/usr/bin/env python3
"""
국회 법률정보시스템 JavaScript 동적 로딩 분석 도구
iframe이 JavaScript로 어떻게 로드되는지 분석
"""

import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright
import json

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def analyze_javascript_loading():
    """JavaScript 동적 로딩 분석"""
    
    # 테스트할 법률 정보
    test_law = {
        'cont_id': '1981022300000003',
        'cont_sid': '0030',
        'law_name': '집행관수수료규칙'
    }
    
    print("🔍 JavaScript 동적 로딩 분석 시작")
    print(f"📋 테스트 법률: {test_law['law_name']}")
    
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
                'headers': dict(request.headers),
                'timestamp': time.time()
            })
            print(f"📤 요청: {request.method} {request.url}")
        
        def handle_response(response):
            responses.append({
                'url': response.url,
                'status': response.status,
                'headers': dict(response.headers),
                'timestamp': time.time()
            })
            print(f"📥 응답: {response.status} {response.url}")
        
        page.on("request", handle_request)
        page.on("response", handle_response)
        
        try:
            # 1. 법률 상세 페이지 접근
            url = f"https://likms.assembly.go.kr/law/lawsLawtInqyDetl1010.do"
            params = f"?contId={test_law['cont_id']}&contSid={test_law['cont_sid']}&viewGb=PROM&genMenuId=menu_serv_nlaw_lawt_1020"
            
            print(f"\n🌐 페이지 접근: {url}{params}")
            page.goto(url + params, wait_until='domcontentloaded')
            
            # 2. JavaScript 실행 대기
            print("⏳ JavaScript 실행 대기...")
            
            # iframe이 로드될 때까지 대기
            try:
                page.wait_for_selector("iframe[name='I_TARGET']", timeout=10000)
                print("✅ iframe 발견!")
            except:
                print("⚠️ iframe 대기 시간 초과")
            
            # 더 긴 시간 대기
            time.sleep(10)
            
            # 3. iframe 내용 확인
            print("\n🖼️ iframe 내용 분석:")
            iframes = page.locator("iframe").all()
            
            for i, iframe in enumerate(iframes):
                try:
                    src = iframe.get_attribute("src")
                    name = iframe.get_attribute("name")
                    print(f"   iframe {i+1}: src='{src}', name='{name}'")
                    
                    # iframe 내용 확인
                    iframe_content = iframe.content_frame()
                    if iframe_content:
                        iframe_text = iframe_content.locator("body").inner_text()
                        print(f"   iframe {i+1} 내용 길이: {len(iframe_text)} chars")
                        
                        if len(iframe_text) > 100:
                            print(f"   📝 샘플 내용: {iframe_text[:300]}...")
                            
                            # 법률 내용 확인
                            if any(keyword in iframe_text for keyword in ['제1조', '제2조', '조문']):
                                print(f"   ✅ iframe {i+1}에 법률 내용 발견!")
                            else:
                                print(f"   ❌ iframe {i+1}에 법률 내용 없음")
                    else:
                        print(f"   ⚠️ iframe {i+1} 내용 접근 불가")
                        
                except Exception as e:
                    print(f"   ❌ iframe {i+1} 오류: {e}")
            
            # 4. 페이지의 모든 텍스트에서 법률 내용 찾기
            print("\n📄 전체 페이지 텍스트 분석:")
            full_text = page.locator("body").inner_text()
            print(f"전체 텍스트 길이: {len(full_text)} chars")
            
            # 법률 관련 키워드 검색
            keywords = ['제1조', '제2조', '제3조', '조문', '총칙', '부칙', '시행령', '법률', '규칙']
            found_keywords = []
            
            for keyword in keywords:
                if keyword in full_text:
                    found_keywords.append(keyword)
                    # 키워드 주변 텍스트 추출
                    start = full_text.find(keyword)
                    context = full_text[max(0, start-50):start+200]
                    print(f"   '{keyword}' 발견: {context}...")
            
            print(f"발견된 키워드: {found_keywords}")
            
            # 5. JavaScript 콘솔 로그 확인
            print("\n🔧 JavaScript 콘솔 로그:")
            console_logs = []
            
            def handle_console(msg):
                console_logs.append({
                    'type': msg.type,
                    'text': msg.text,
                    'timestamp': time.time()
                })
                print(f"   {msg.type}: {msg.text}")
            
            page.on("console", handle_console)
            
            # 6. 특정 URL 직접 접근 시도
            print("\n🔄 특정 URL 직접 접근 시도:")
            
            # lawsNormInqyMain1010.do URL 시도
            norm_url = f"https://likms.assembly.go.kr/law/lawsNormInqyMain1010.do?mappingId=%2FlawsNormInqyMain1010.do&genActiontypeCd=2ACT1010"
            print(f"🌐 접근 시도: {norm_url}")
            
            try:
                page.goto(norm_url, wait_until='domcontentloaded')
                time.sleep(5)
                
                norm_text = page.locator("body").inner_text()
                print(f"📄 Norm 페이지 텍스트 길이: {len(norm_text)} chars")
                
                if any(keyword in norm_text for keyword in ['제1조', '제2조', '조문']):
                    print("✅ Norm 페이지에 법률 내용 발견!")
                    print(f"📝 샘플: {norm_text[:300]}...")
                else:
                    print("❌ Norm 페이지에 법률 내용 없음")
                    
            except Exception as e:
                print(f"❌ Norm 페이지 접근 오류: {e}")
            
            # 7. 분석 결과 저장
            analysis_result = {
                'test_law': test_law,
                'iframe_count': len(iframes),
                'found_keywords': found_keywords,
                'requests': requests,
                'responses': responses,
                'console_logs': console_logs,
                'timestamp': time.time()
            }
            
            with open("javascript_analysis_result.json", "w", encoding="utf-8") as f:
                json.dump(analysis_result, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 분석 결과 저장: javascript_analysis_result.json")
            
            # 브라우저를 잠시 열어두어 수동 확인 가능
            print(f"\n🔍 브라우저가 열려있습니다. 수동으로 확인해보세요.")
            print(f"⏳ 60초 후 자동으로 닫힙니다...")
            time.sleep(60)
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    analyze_javascript_loading()
