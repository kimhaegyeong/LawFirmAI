# -*- coding: utf-8 -*-
"""
Assembly Playwright Client
국회 법률정보시스템 Playwright 클라이언트

Playwright를 사용하여 국회 법률정보시스템에서 법률과 판례 데이터를 수집합니다.
- JavaScript 렌더링 지원
- 메모리 관리 및 Rate limiting
- 체크포인트 지원
"""

from playwright.sync_api import sync_playwright, Page, Browser
import time
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import psutil
import gc
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs

logger = get_logger(__name__)


class AssemblyPlaywrightClient:
    """국회 법률정보시스템 Playwright 클라이언트"""
    
    BASE_URL = "https://likms.assembly.go.kr"
    LAW_LIST_URL = "/law/lawsLawtInqyList2010.do"
    PRECEDENT_LIST_URL = "/law/lawsPrecInqyList2010.do"
    
    def __init__(self, 
                 rate_limit=3.0,
                 timeout=30000,
                 headless=True,
                 memory_limit_mb=800):
        """
        Playwright 클라이언트 초기화
        
        Args:
            rate_limit: 요청 간 대기 시간 (초)
            timeout: 페이지 로딩 타임아웃 (밀리초)
            headless: 헤드리스 모드 여부
            memory_limit_mb: 메모리 사용량 제한 (MB)
        """
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.headless = headless
        self.memory_limit_mb = memory_limit_mb
        
        self.playwright = None
        self.browser = None
        self.page = None
        
        self.last_request_time = 0
        self.request_count = 0
        
        self.logger = get_logger(__name__)
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def start(self):
        """Playwright 브라우저 시작"""
        try:
            print(f"🌐 Starting Playwright browser...")
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(
                headless=self.headless,
                args=['--disable-dev-shm-usage', '--no-sandbox', '--disable-gpu']
            )
            
            # 새 페이지 생성
            self.page = self.browser.new_page()
            self.page.set_default_timeout(self.timeout)
            
            # User-Agent 설정
            self.page.set_extra_http_headers({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 LawFirmAI/1.0 Research'
            })
            
            print(f"✅ Playwright browser started successfully")
            
        except Exception as e:
            print(f"❌ Failed to start Playwright: {e}")
            raise
    
    def close(self):
        """브라우저 종료"""
        try:
            print(f"🔒 Closing Playwright browser...")
            if self.page:
                self.page.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            
            print(f"✅ Playwright browser closed successfully")
            
        except Exception as e:
            print(f"⚠️ Error closing browser: {e}")
    
    def check_memory_usage(self) -> float:
        """메모리 사용량 체크 및 정리"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.memory_limit_mb * 0.8:
                self.logger.warning(f"⚠️ High memory: {memory_mb:.2f}MB, cleaning up")
                gc.collect()
                
                memory_after = process.memory_info().rss / 1024 / 1024
                self.logger.info(f"✅ After cleanup: {memory_after:.2f}MB")
                
                if memory_after > self.memory_limit_mb:
                    raise MemoryError(f"Memory limit exceeded: {memory_after:.2f}MB")
            
            return memory_mb
            
        except Exception as e:
            self.logger.error(f"❌ Memory check failed: {e}")
            raise
    
    def _enforce_rate_limit(self):
        """Rate limiting 적용"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _extract_params(self, href: str) -> Dict[str, str]:
        """URL에서 파라미터 추출"""
        try:
            parsed_url = urlparse(href)
            params = parse_qs(parsed_url.query)
            return {k: v[0] if v else '' for k, v in params.items()}
        except Exception as e:
            print(f"         ⚠️ Failed to extract params from {href}: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """클라이언트 통계 정보 반환"""
        return {
            'request_count': self.request_count,
            'rate_limit': self.rate_limit,
            'timeout': self.timeout,
            'memory_limit_mb': self.memory_limit_mb
        }
    
    # ===== 법률 수집 메서드 =====
    
    def get_law_list_page(self, page_num=1, page_size=10) -> List[Dict]:
        """
        법률 목록 페이지 조회 (페이지당 10개씩 표시됨)
        
        Args:
            page_num: 페이지 번호 (1부터 시작)
            page_size: 페이지당 항목 수 (실제로는 10개 고정)
        
        Returns:
            List[Dict]: 법률 목록
        """
        self._enforce_rate_limit()
        self.check_memory_usage()
        
        # URL 구성
        url = f"{self.BASE_URL}{self.LAW_LIST_URL}"
        params = "?genActiontypeCd=2ACT1010&genDoctreattypeCd=DOCT2004&genMenuId=menu_serv_nlaw_lawt_1020"
        
        try:
            # 첫 페이지로 이동
            print(f"      🌐 Navigating to law list page {page_num}...")
            self.page.goto(url + params, wait_until='domcontentloaded')
            print(f"      ✅ Page loaded successfully")
            
            # 첫 페이지가 아니면 해당 페이지로 이동
            if page_num > 1:
                try:
                    print(f"      🔄 Navigating to page {page_num}...")
                    
                    # 방법 1: 페이지 번호 링크 직접 클릭
                    try:
                        # 페이지 번호 링크 찾기
                        page_selector = f"span.page_no a[onclick*='pageCall({page_num})']"
                        page_link = self.page.locator(page_selector).first
                        
                        if page_link.is_visible():
                            print(f"      🖱️ Clicking page {page_num} link...")
                            page_link.click()
                            
                            # AJAX 요청 완료 대기
                            print(f"      ⏳ Waiting for AJAX request to complete...")
                            self.page.wait_for_timeout(5000)  # 5초 대기
                            
                            # 테이블이 업데이트될 때까지 추가 대기
                            self.page.wait_for_selector("table tr", timeout=10000)
                            print(f"      ✅ Navigated to page {page_num} via click")
                        else:
                            print(f"      ⚠️ Page {page_num} link not visible, trying JavaScript...")
                            
                            # 방법 2: JavaScript로 직접 호출
                            js_code = f"pageCall({page_num});"
                            self.page.evaluate(js_code)
                            self.page.wait_for_timeout(5000)
                            print(f"      ✅ Navigated to page {page_num} via JavaScript")
                            
                    except Exception as click_e:
                        print(f"      ⚠️ Click failed: {click_e}, trying JavaScript...")
                        
                        # 방법 3: JavaScript로 직접 호출
                        try:
                            js_code = f"pageCall({page_num});"
                            self.page.evaluate(js_code)
                            self.page.wait_for_timeout(5000)
                            print(f"      ✅ Navigated to page {page_num} via JavaScript fallback")
                        except Exception as js_e:
                            print(f"      ❌ All navigation methods failed: {js_e}")
                            
                except Exception as e:
                    print(f"      ❌ Failed to navigate to page {page_num}: {e}")
            
            # 테이블 파싱
            print(f"      🔍 Parsing law table...")
            laws = self._parse_law_table()
            
            self.request_count += 1
            print(f"      ✅ Parsing completed: {len(laws)} laws found on page {page_num}")
            return laws
            
        except Exception as e:
            print(f"      ❌ Failed to get law list page {page_num}: {e}")
            raise
    
    def _parse_law_table(self) -> List[Dict]:
        """법률 목록 테이블 파싱"""
        laws = []
        
        try:
            # 테이블 행 찾기
            rows = self.page.locator("table tr").all()
            print(f"         📋 Found {len(rows)} table rows")
            
            for i, row in enumerate(rows[1:], 1):  # 헤더 제외
                try:
                    cells = row.locator("td").all()
                    if len(cells) < 8:
                        print(f"         Row {i}: Only {len(cells)} cells, skipping")
                        continue
                    
                    # 법령명 링크 (첫 번째 링크만 선택)
                    link = cells[1].locator("a").first
                    href = link.get_attribute("href")
                    
                    if not href:
                        print(f"         Row {i}: No href found")
                        continue
                    
                    # 법령명 텍스트 추출
                    law_name = link.inner_text().strip()
                    if not law_name or law_name == "국회법률정보시스템":
                        print(f"         Row {i}: Invalid law name: '{law_name}'")
                        continue
                    
                    # URL 파라미터 추출
                    params = self._extract_params(href)
                    
                    law = {
                        'row_number': cells[0].inner_text().strip(),
                        'law_name': law_name,
                        'category': cells[2].inner_text().strip(),
                        'law_type': cells[3].inner_text().strip(),
                        'promulgation_number': cells[4].inner_text().strip(),
                        'promulgation_date': cells[5].inner_text().strip(),
                        'enforcement_date': cells[6].inner_text().strip(),
                        'amendment_type': cells[7].inner_text().strip(),
                        'cont_id': params.get('contId', ''),
                        'cont_sid': params.get('contSid', ''),
                        'detail_url': self.BASE_URL + href
                    }
                    laws.append(law)
                    print(f"         Row {i}: ✅ {law_name[:30]}...")
                    
                except Exception as e:
                    print(f"         Row {i}: ❌ Failed to parse: {str(e)[:50]}...")
                    continue
            
        except Exception as e:
            print(f"         ❌ Failed to parse law table: {e}")
            raise
        
        return laws
    
    def get_law_detail(self, cont_id: str, cont_sid: str) -> Dict:
        """법률 상세 정보 조회 - 올바른 URL 패턴 사용"""
        self._enforce_rate_limit()
        
        # 올바른 URL 패턴 사용 (lawsLawtInqyDetl1030.do)
        url = f"{self.BASE_URL}/law/lawsLawtInqyDetl1030.do"
        params = f"?genActiontypeCd=2ACT1010&menuId=menu_serv_nlaw_lawt_1020&contId={cont_id}&scContSid=&contSid={cont_sid}&basicDt=20251010&revNo=&cachePreid=ALL&hanTranceYn=&viewGb=&selTabClass=&uid=WC10QH10F95K8178924YF57"
        
        try:
            print(f"         🌐 Loading law detail: {cont_id}...")
            print(f"         🔗 Using correct URL pattern: lawsLawtInqyDetl1030.do")
            self.page.goto(url + params, wait_until='domcontentloaded')
            
            # JavaScript 로딩 대기
            print(f"         ⏳ Waiting for JavaScript content to load...")
            self.page.wait_for_timeout(5000)  # 5초 대기
            
            # 법령명 추출
            law_name = ""
            try:
                # 법령명은 h1 태그에 있음
                law_name_element = self.page.locator("h1").first
                if law_name_element.is_visible():
                    law_name = law_name_element.inner_text().strip()
                    print(f"         📝 Found law name: {law_name}")
                else:
                    print(f"         ⚠️ Could not find law name in h1")
                    law_name = f"법률_{cont_id}"
            except Exception as e:
                print(f"         ⚠️ Error extracting law name: {e}")
                law_name = f"법률_{cont_id}"
            
            # 실제 법률 내용 추출
            print(f"         📄 Extracting law content...")
            
            # 페이지의 모든 텍스트 가져오기
            full_text = self.page.locator("body").inner_text()
            print(f"         📄 Full page text length: {len(full_text)} chars")
            
            # 법률 관련 키워드가 있는 부분 찾기
            lines = full_text.split('\n')
            law_lines = []
            
            for line in lines:
                line = line.strip()
                if (len(line) > 10 and 
                    any(keyword in line for keyword in ['제1조', '제2조', '제3조', '제4조', '제5조', '조문', '시행령', '법률', '규칙', '시행', '공포', '조항', '제목', '총칙', '부칙'])):
                    law_lines.append(line)
            
            if law_lines:
                law_content = '\n'.join(law_lines)
                print(f"         ✅ Extracted law content ({len(law_content)} chars)")
                
                # 법률 내용 검증
                if any(keyword in law_content for keyword in ['제1조', '제2조', '조문']):
                    print(f"         ✅ Valid law content found - contains legal articles!")
                else:
                    print(f"         ⚠️ Law content found but no legal articles detected")
            else:
                print(f"         ❌ No law content found")
                law_content = ""
            
            # HTML 전체 내용
            content_html = self.page.content()
            
            print(f"         ✅ Content extraction completed")
            
            return {
                'cont_id': cont_id,
                'cont_sid': cont_sid,
                'law_name': law_name,
                'law_content': law_content,
                'content_html': content_html,
                'collected_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"         ❌ Failed to get law detail {cont_id}: {e}")
            raise
    
    # ===== 판례 수집 메서드 =====
    
    def get_precedent_list_page(self, page_num=1, page_size=10) -> List[Dict]:
        """
        판례 목록 페이지 조회 (페이지당 10개씩 표시됨)
        
        Args:
            page_num: 페이지 번호 (1부터 시작)
            page_size: 페이지당 항목 수 (실제로는 10개 고정)
        
        Returns:
            List[Dict]: 판례 목록
        """
        self._enforce_rate_limit()
        self.check_memory_usage()
        
        # URL 구성
        url = f"{self.BASE_URL}{self.PRECEDENT_LIST_URL}"
        params = "?genActiontypeCd=2ACT1010&genDoctreattypeCd=&genMenuId=menu_serv_nlaw_lawt_4020&procWorkId="
        
        try:
            # 첫 페이지로 이동
            print(f"      🌐 Navigating to precedent list page {page_num}...")
            self.page.goto(url + params, wait_until='domcontentloaded')
            print(f"      ✅ Page loaded successfully")
            
            # 첫 페이지가 아니면 해당 페이지로 이동
            if page_num > 1:
                try:
                    print(f"      🔄 Navigating to page {page_num}...")
                    
                    # 방법 1: 페이지 번호 링크 직접 클릭
                    try:
                        # 페이지 번호 링크 찾기
                        page_selector = f"span.page_no a[href*='pageNum={page_num}']"
                        page_link = self.page.locator(page_selector).first
                        
                        if page_link.is_visible():
                            print(f"      🖱️ Clicking page {page_num} link...")
                            page_link.click()
                            
                            # AJAX 요청 완료 대기
                            print(f"      ⏳ Waiting for AJAX request to complete...")
                            self.page.wait_for_timeout(5000)  # 5초 대기
                            
                            # 테이블이 업데이트될 때까지 추가 대기
                            self.page.wait_for_selector("table tr", timeout=10000)
                            print(f"      ✅ Navigated to page {page_num} via click")
                        else:
                            print(f"      ⚠️ Page {page_num} link not visible, trying JavaScript...")
                            
                            # 방법 2: JavaScript로 직접 호출
                            js_code = f"pageCall({page_num});"
                            self.page.evaluate(js_code)
                            self.page.wait_for_timeout(5000)
                            print(f"      ✅ Navigated to page {page_num} via JavaScript")
                            
                    except Exception as click_e:
                        print(f"      ⚠️ Click failed: {click_e}, trying JavaScript...")
                        
                        # 방법 3: JavaScript로 직접 호출
                        try:
                            js_code = f"pageCall({page_num});"
                            self.page.evaluate(js_code)
                            self.page.wait_for_timeout(5000)
                            print(f"      ✅ Navigated to page {page_num} via JavaScript fallback")
                        except Exception as js_e:
                            print(f"      ❌ All navigation methods failed: {js_e}")
                            
                except Exception as e:
                    print(f"      ❌ Failed to navigate to page {page_num}: {e}")
            
            # 테이블 파싱
            print(f"      🔍 Parsing precedent table...")
            precedents = self._parse_precedent_table()
            
            self.request_count += 1
            print(f"      ✅ Parsing completed: {len(precedents)} precedents found on page {page_num}")
            return precedents
            
        except Exception as e:
            print(f"      ❌ Failed to get precedent list page {page_num}: {e}")
            raise

    def get_precedent_list_page_by_category(self, category_code: str, page_num=1, page_size=10) -> List[Dict]:
        """
        분야별 판례 목록 페이지 조회 (페이지당 10개씩 표시됨)
        
        Args:
            category_code: 분야 코드 (PREC00_001, PREC00_002 등)
            page_num: 페이지 번호 (1부터 시작)
            page_size: 페이지당 항목 수 (실제로는 10개 고정)
        
        Returns:
            List[Dict]: 판례 목록
        """
        self._enforce_rate_limit()
        self.check_memory_usage()
        
        # URL 구성 (분야별 필터링)
        url = f"{self.BASE_URL}{self.PRECEDENT_LIST_URL}"
        params = f"?genActiontypeCd=2ACT1010&genDoctreattypeCd=&genMenuId=menu_serv_nlaw_lawt_4020&procWorkId=&topicCd={category_code}"
        
        try:
            # 첫 페이지로 이동
            print(f"      🌐 Navigating to precedent list page {page_num} (category: {category_code})...")
            self.page.goto(url + params, wait_until='domcontentloaded')
            print(f"      ✅ Page loaded successfully")
            
            # 첫 페이지가 아니면 해당 페이지로 이동
            if page_num > 1:
                try:
                    print(f"      🔄 Navigating to page {page_num}...")
                    
                    # 방법 1: 페이지 번호 링크 직접 클릭
                    try:
                        # 페이지 번호 링크 찾기
                        page_selector = f"span.page_no a[href*='pageNum={page_num}']"
                        page_link = self.page.locator(page_selector).first
                        
                        if page_link.is_visible():
                            print(f"      🖱️ Clicking page {page_num} link...")
                            page_link.click()
                            
                            # AJAX 요청 완료 대기
                            print(f"      ⏳ Waiting for AJAX request to complete...")
                            self.page.wait_for_timeout(5000)  # 5초 대기
                            
                            # 테이블이 업데이트될 때까지 추가 대기
                            self.page.wait_for_selector("table tr", timeout=10000)
                            print(f"      ✅ Navigated to page {page_num} via click")
                        else:
                            print(f"      ⚠️ Page {page_num} link not visible, trying JavaScript...")
                            
                            # 방법 2: JavaScript로 직접 호출
                            js_code = f"pageCall({page_num});"
                            self.page.evaluate(js_code)
                            self.page.wait_for_timeout(5000)
                            print(f"      ✅ Navigated to page {page_num} via JavaScript")
                            
                    except Exception as click_e:
                        print(f"      ⚠️ Click failed: {click_e}, trying JavaScript...")
                        
                        # 방법 3: JavaScript로 직접 호출
                        try:
                            js_code = f"pageCall({page_num});"
                            self.page.evaluate(js_code)
                            self.page.wait_for_timeout(5000)
                            print(f"      ✅ Navigated to page {page_num} via JavaScript fallback")
                        except Exception as js_e:
                            print(f"      ❌ All navigation methods failed: {js_e}")
                            
                except Exception as e:
                    print(f"      ❌ Failed to navigate to page {page_num}: {e}")
            
            # 테이블 파싱
            print(f"      🔍 Parsing precedent table...")
            precedents = self._parse_precedent_table()
            
            self.request_count += 1
            print(f"      ✅ Parsing completed: {len(precedents)} precedents found on page {page_num}")
            return precedents
            
        except Exception as e:
            print(f"      ❌ Failed to get precedent list page {page_num}: {e}")
            raise

    def _parse_precedent_table(self) -> List[Dict]:
        """판례 목록 테이블 파싱"""
        precedents = []
        try:
            rows = self.page.locator("table tr").all()
            print(f"         📋 Found {len(rows)} table rows")
            for i, row in enumerate(rows[1:], 1):  # 헤더 제외
                try:
                    cells = row.locator("td").all()
                    if len(cells) < 6:
                        print(f"         Row {i}: Only {len(cells)} cells, skipping")
                        continue
                    
                    # 판례 상세 링크 추출
                    link = cells[1].locator("a").first
                    href = link.get_attribute("href")
                    if not href:
                        print(f"         Row {i}: No href found")
                        continue
                    
                    case_name = link.inner_text().strip()
                    if not case_name:
                        print(f"         Row {i}: Invalid case name")
                        continue
                    
                    # URL 파라미터 추출
                    params = self._extract_params(href)
                    
                    precedent = {
                        'row_number': cells[0].inner_text().strip(),
                        'case_name': case_name,
                        'case_number': cells[2].inner_text().strip(),
                        'decision_date': cells[3].inner_text().strip(),
                        'field': cells[4].inner_text().strip(),
                        'court': cells[5].inner_text().strip(),
                        'detail_url': self.BASE_URL + href,
                        'params': params
                    }
                    precedents.append(precedent)
                    print(f"         Row {i}: ✅ {case_name[:30]}...")
                    
                except Exception as e:
                    print(f"         Row {i}: ❌ Failed to parse: {str(e)[:50]}...")
                    continue
        except Exception as e:
            print(f"         ❌ Failed to parse precedent table: {e}")
            raise
        return precedents

    def get_precedent_detail(self, precedent_item: Dict) -> Dict:
        """판례 상세 정보 조회 (구조화된 데이터 포함, 메모리 최적화)"""
        self._enforce_rate_limit()
        
        # 판례 상세 URL 구성
        url = f"{self.BASE_URL}/law/lawsPrecInqyDetl1010.do"
        
        # URL 파라미터 구성
        url_params = []
        for key, value in precedent_item['params'].items():
            if value:
                url_params.append(f"{key}={value}")
        
        params_str = "&".join(url_params)
        full_url = f"{url}?{params_str}"
        
        try:
            print(f"         🌐 Loading precedent detail...")
            print(f"         🔗 URL: {full_url}")
            self.page.goto(full_url, wait_until='domcontentloaded')
            
            # JavaScript 로딩 대기
            print(f"         ⏳ Waiting for JavaScript content to load...")
            self.page.wait_for_timeout(5000)  # 5초 대기
            
            # 메모리 정리: 페이지 캐시 클리어
            self.page.evaluate("() => { if (window.gc) window.gc(); }")
            
            # 판례명 추출
            case_name = ""
            try:
                # 방법 1: 목록 페이지에서 가져온 판례명 사용 (가장 정확)
                list_case_name = precedent_item.get('case_name', '')
                if list_case_name and list_case_name != "국회법률정보시스템":
                    case_name = list_case_name
                    print(f"         📝 Using list page case name: {case_name}")
                else:
                    # 방법 2: 상세 페이지에서 대괄호로 둘러싸인 판례명 찾기
                    print(f"         🔍 Searching for case name in brackets...")
                    bracket_patterns = self.page.locator("text=/\\[.*?\\]/").all()
                    
                    for pattern in bracket_patterns:
                        text = pattern.inner_text().strip()
                        # 대괄호 안의 텍스트가 판례명인지 확인
                        if (text.startswith('[') and text.endswith(']') and 
                            len(text) > 3 and len(text) < 50 and
                            any(keyword in text for keyword in ['소', '사건', '청구', '소송', '이의', '항고', '항소', '상고'])):
                            case_name = text.strip('[]').strip()
                            print(f"         📝 Found case name in brackets: {case_name}")
                            break
                    
                    # 방법 3: contents 클래스에서 판례명 찾기
                    if not case_name:
                        print(f"         🔍 Searching in contents class...")
                        contents_elements = self.page.locator(".contents").all()
                        for elem in contents_elements:
                            text = elem.inner_text().strip()
                            # 대괄호로 둘러싸인 판례명 찾기
                            import re
                            bracket_matches = re.findall(r'\[([^\]]+)\]', text)
                            for match in bracket_matches:
                                if (len(match.strip()) > 2 and len(match.strip()) < 50 and
                                    any(keyword in match for keyword in ['소', '사건', '청구', '소송', '이의', '항고', '항소', '상고'])):
                                    case_name = match.strip()
                                    print(f"         📝 Found case name in contents: {case_name}")
                                    break
                            if case_name:
                                break
                    
                    # 방법 4: 마지막 수단으로 목록 페이지 판례명 사용
                    if not case_name:
                        case_name = precedent_item.get('case_name', f"판례_{precedent_item.get('case_number', 'unknown')}")
                        print(f"         ⚠️ Using fallback case name: {case_name}")
                        
            except Exception as e:
                print(f"         ⚠️ Error extracting case name: {e}")
                case_name = precedent_item.get('case_name', f"판례_{precedent_item.get('case_number', 'unknown')}")
            
            # 구조화된 판례 내용 추출
            print(f"         📄 Extracting structured precedent content...")
            
            # 페이지의 모든 텍스트 가져오기
            full_text = self.page.locator("body").inner_text()
            print(f"         📄 Full page text length: {len(full_text)} chars")
            
            # 구조화된 섹션 추출
            structured_content = self._extract_structured_precedent_content(full_text)
            
            # HTML 전체 내용 (메모리 최적화: 크기 제한)
            content_html = self.page.content()
            if len(content_html) > 2000000:  # 2MB 제한
                content_html = content_html[:2000000] + "... [TRUNCATED]"
                print(f"         ⚠️ HTML content truncated to 2MB")
            
            print(f"         ✅ Structured content extraction completed")
            
            # 메모리 최적화: 구조화된 내용 크기 제한
            if 'full_text' in structured_content and len(structured_content['full_text']) > 1000000:  # 1MB 제한
                structured_content['full_text'] = structured_content['full_text'][:1000000] + "... [TRUNCATED]"
                print(f"         ⚠️ Structured content truncated to 1MB")
            
            # 목록 정보와 구조화된 상세 정보를 통합하여 반환
            result = {
                # 목록 정보 (기존)
                'row_number': precedent_item.get('row_number', ''),
                'case_name': case_name,
                'case_number': precedent_item.get('case_number', ''),
                'decision_date': precedent_item.get('decision_date', ''),
                'field': precedent_item.get('field', ''),
                'court': precedent_item.get('court', ''),
                'detail_url': precedent_item.get('detail_url', full_url),
                
                # 구조화된 상세 정보
                'structured_content': structured_content,
                'precedent_content': structured_content.get('full_text', ''),  # 하위 호환성
                'content_html': content_html,
                'full_text_length': len(full_text),
                'extracted_content_length': len(structured_content.get('full_text', '')),
                
                # 메타데이터
                'params': precedent_item.get('params', {}),
                'collected_at': datetime.now().isoformat(),
                'source_url': full_url
            }
            
            # 메모리 정리: 중간 변수들 삭제
            del full_text, structured_content, content_html
            import gc
            gc.collect()
            
            return result
            
        except Exception as e:
            print(f"         ❌ Failed to get precedent detail: {e}")
            raise

    def _extract_structured_precedent_content(self, full_text: str) -> Dict:
        """판례 내용을 구조화된 섹션으로 추출"""
        lines = full_text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        structured = {
            'case_info': {},  # 사건 정보
            'legal_sections': {},  # 법적 섹션들
            'parties': {},  # 당사자 정보
            'procedural_info': {},  # 절차 정보
            'full_text': full_text,
            'extraction_metadata': {}
        }
        
        try:
            # 1. 사건 정보 추출
            print(f"         🔍 Extracting case information...")
            case_info = self._extract_case_info(non_empty_lines)
            structured['case_info'] = case_info
            
            # 2. 법적 섹션 추출
            print(f"         📋 Extracting legal sections...")
            legal_sections = self._extract_legal_sections(non_empty_lines)
            structured['legal_sections'] = legal_sections
            
            # 3. 당사자 정보 추출
            print(f"         👥 Extracting parties information...")
            parties = self._extract_parties_info(non_empty_lines)
            structured['parties'] = parties
            
            # 4. 절차 정보 추출
            print(f"         ⚖️ Extracting procedural information...")
            procedural_info = self._extract_procedural_info(non_empty_lines)
            structured['procedural_info'] = procedural_info
            
            # 5. 추출 메타데이터
            structured['extraction_metadata'] = {
                'total_lines': len(lines),
                'non_empty_lines': len(non_empty_lines),
                'extracted_sections': len([k for k, v in legal_sections.items() if v]),
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            print(f"         ✅ Extracted {len([k for k, v in legal_sections.items() if v])} legal sections")
            
        except Exception as e:
            print(f"         ⚠️ Error in structured extraction: {e}")
            structured['extraction_error'] = str(e)
        
        return structured

    def _extract_case_info(self, lines: List[str]) -> Dict:
        """사건 정보 추출"""
        case_info = {}
        
        # 판례 번호, 법원, 선고일 추출
        for line in lines:
            if '선고' in line and '판결' in line:
                case_info['case_title'] = line
                # 법원 추출
                if '대법원' in line:
                    case_info['court'] = '대법원'
                elif '고등법원' in line:
                    case_info['court'] = '고등법원'
                elif '지방법원' in line:
                    case_info['court'] = '지방법원'
                
                # 선고일 추출 (YYYY. M. D. 형식)
                import re
                date_match = re.search(r'(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.', line)
                if date_match:
                    case_info['decision_date'] = f"{date_match.group(1)}-{date_match.group(2).zfill(2)}-{date_match.group(3).zfill(2)}"
                
                # 사건번호 추출
                case_num_match = re.search(r'(\d{4}[가-힣]\d+)', line)
                if case_num_match:
                    case_info['case_number'] = case_num_match.group(1)
                break
        
        return case_info

    def _extract_legal_sections(self, lines: List[str]) -> Dict:
        """법적 섹션들 추출"""
        sections = {
            '판시사항': '',
            '판결요지': '',
            '참조조문': '',
            '참조판례': '',
            '주문': '',
            '이유': ''
        }
        
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 섹션 헤더 감지
            if line in sections:
                # 이전 섹션 저장
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # 새 섹션 시작
                current_section = line
                current_content = []
                continue
            
            # 현재 섹션에 내용 추가
            if current_section:
                current_content.append(line)
            else:
                # 섹션 헤더가 없으면 일반 내용으로 처리
                if not any(section in line for section in sections.keys()):
                    if not sections['이유']:  # 이유 섹션이 비어있으면 일반 내용을 이유로 분류
                        sections['이유'] = line
        
        # 마지막 섹션 저장
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections

    def _extract_parties_info(self, lines: List[str]) -> Dict:
        """당사자 정보 추출"""
        parties = {
            'plaintiff': '',  # 원고
            'defendant': '',  # 피고
            'appellant': '',  # 상고인
            'appellee': ''    # 피상고인
        }
        
        for line in lines:
            line = line.strip()
            if '원고' in line and '상고인' in line:
                parties['plaintiff'] = line
                parties['appellant'] = line
            elif '피고' in line and '피상고인' in line:
                parties['defendant'] = line
                parties['appellee'] = line
            elif '원고' in line:
                parties['plaintiff'] = line
            elif '피고' in line:
                parties['defendant'] = line
        
        return parties

    def _extract_procedural_info(self, lines: List[str]) -> Dict:
        """절차 정보 추출"""
        procedural = {
            'lower_court': '',  # 원심법원
            'lower_court_decision': '',  # 원심판결
            'appeal_type': '',  # 상고/항소 등
            'final_decision': ''  # 최종 판결
        }
        
        for line in lines:
            line = line.strip()
            if '원심판결' in line:
                procedural['lower_court_decision'] = line
            elif '지방법원' in line or '고등법원' in line:
                procedural['lower_court'] = line
            elif '상고' in line and '기각' in line:
                procedural['final_decision'] = line
            elif '파기' in line and '환송' in line:
                procedural['final_decision'] = line
        
        return procedural