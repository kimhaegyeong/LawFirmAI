"""
참조조문 추출기
데이터 수집 시점에 판례/결정례/해석례의 content에서 참조조문을 추출합니다.
"""
import re
import json
from typing import List, Dict, Any, Optional

class ReferenceStatuteExtractor:
    """참조조문 추출기 (데이터 수집 시점에 사용)"""
    
    def extract_from_content(self, content: str) -> List[Dict[str, Any]]:
        """
        content에서 참조조문 추출
        
        Args:
            content: 판례/결정례/해석례의 전체 텍스트
            
        Returns:
            추출된 법령 정보 리스트
        """
        if not content or not isinstance(content, str):
            return []
        
        # "참조조문" 위치 찾기 (콜론 포함/미포함 모두 처리)
        ref_pos = -1
        ref_marker_len = 0
        
        # 다양한 형식 시도
        markers = [
            ("참조조문:", 4),  # "참조조문:" (한글 콜론)
            ("참조조문：", 4),  # "참조조문：" (전각 콜론)
            ("참조조문 ", 4),  # "참조조문 " (공백)
            ("참조조문\n", 4),  # "참조조문\n" (줄바꿈)
            ("참조조문", 4),    # "참조조문" (바로 다음 단어)
        ]
        
        for marker, marker_len in markers:
            pos = content.find(marker)
            if pos != -1:
                ref_pos = pos
                ref_marker_len = marker_len
                break
        
        if ref_pos == -1:
            return []
        
        # 시작 위치 계산
        start_pos = ref_pos + ref_marker_len
        
        # "참조판례" 위치 찾기
        ref_precedent_pos = content.find("참조판례:", start_pos)
        if ref_precedent_pos == -1:
            ref_precedent_pos = content.find("참조판례：", start_pos)
        if ref_precedent_pos == -1:
            ref_precedent_pos = content.find("참조판례", start_pos)
        
        # "전문" 위치 찾기
        full_text_pos = content.find("\n전문", start_pos)
        if full_text_pos == -1:
            full_text_pos = content.find("전문", start_pos)
        
        # "이유" 위치 찾기
        reason_pos = content.find("\n이유", start_pos)
        
        # 종료 위치 결정 (가장 가까운 마커 사용)
        ref_end = len(content)
        for pos in [ref_precedent_pos, full_text_pos, reason_pos]:
            if pos != -1 and pos < ref_end:
                ref_end = pos
        
        # 참조조문 텍스트 추출
        ref_text = content[start_pos:ref_end].strip()
        
        # "참조판례" 키워드가 포함된 경우 제거
        if ref_text.endswith("참조판례"):
            ref_text = ref_text[:-4].strip()
        
        # 개선: "참조조문 참조판례" 사이에 법령 정보가 없는 경우
        # "참조판례" 이후 "전문" 이전까지도 검색
        if not ref_text or len(ref_text) < 10:
            # "참조판례"와 "전문" 사이에서 법령 정보 찾기
            if ref_precedent_pos != -1 and full_text_pos != -1:
                between_text = content[ref_precedent_pos:full_text_pos]
                # "참조판례" 키워드 제거
                between_text = between_text.replace("참조판례:", "").replace("참조판례：", "").replace("참조판례", "").strip()
                if between_text and len(between_text) > 10:
                    # 법령 패턴이 있는지 확인
                    import re
                    if re.search(r'제\s*\d+\s*조', between_text[:200]):
                        ref_text = between_text[:200]
        
        # 개선: "참조조문" 바로 뒤에 법령 정보가 있는 경우
        # (예: "참조조문 국세기본법 제14조")
        if not ref_text or len(ref_text) < 10:
            # "참조조문" 이후 300자 내에서 법령 패턴 직접 검색
            search_range = content[start_pos:start_pos+300]
            import re
            # 법령 패턴 직접 검색 (더 넓은 범위)
            direct_pattern = r'([가-힣\s]+법률?)\s*제\s*(\d+)(?:조의\d+)?\s*조'
            direct_match = re.search(direct_pattern, search_range)
            if direct_match:
                # 직접 추출 시도
                statutes = self._parse_reference_text(search_range[:400])
                if statutes:
                    return statutes
        
        if not ref_text:
            return []
        
        # 법령 추출
        statutes = self._parse_reference_text(ref_text)
        
        # 개선: 추출 실패 시 "전문" 섹션에서도 검색
        if not statutes and full_text_pos != -1:
            # "전문" 이후 800자 내에서 법령 정보 검색
            after_full_text = content[full_text_pos:full_text_pos+800]
            # "전문" 키워드 제거
            after_full_text = after_full_text.replace("전문", "").strip()
            if after_full_text:
                import re
                # 법령 패턴이 있는지 확인
                if re.search(r'제\s*\d+\s*조', after_full_text[:500]):
                    # "판시사항" 섹션에서 법령 정보 추출
                    statutes = self._parse_reference_text(after_full_text[:500])
        
        return statutes
    
    def _parse_reference_text(self, text: str) -> List[Dict[str, Any]]:
        """참조조문 텍스트 파싱"""
        statutes = []
        seen = set()
        last_statute_name = None  # 이전 법령명 추적 (생략된 경우 사용)
        
        # 구분자로 분리 (다양한 구분자 처리)
        items = []
        # 먼저 큰 구분자로 분리 (" / " 또는 " /")
        if " / " in text:
            # 큰 구분자로 분리한 후 각 항목을 다시 세부 구분자로 분리
            big_parts = [item.strip() for item in text.split(" / ")]
            for part in big_parts:
                # 각 큰 부분을 다시 세부 구분자로 분리
                if " , " in part:
                    items.extend([item.strip() for item in part.split(" , ")])
                elif "," in part:
                    items.extend([item.strip() for item in part.split(",")])
                else:
                    items.append(part.strip())
        elif " /" in text:
            big_parts = [item.strip() for item in text.split(" /")]
            for part in big_parts:
                if " , " in part:
                    items.extend([item.strip() for item in part.split(" , ")])
                elif "," in part:
                    items.extend([item.strip() for item in part.split(",")])
                else:
                    items.append(part.strip())
        elif " , " in text:
            items = [item.strip() for item in text.split(" , ")]
        elif "," in text:
            items = [item.strip() for item in text.split(",")]
        else:
            items = [text.strip()]
        
        # 각 항목 파싱
        for item in items:
            if not item:
                continue
            
            # "[1]" 같은 번호 제거
            item = re.sub(r'^\[\d+\]\s*', '', item).strip()
            
            if not item:
                continue
            
            # 패턴 1: 완전한 법령명 + 조문
            # 예: "민법 제536조", "민법 제493조 제2항", "공인중개사법 제2조 제1호"
            # "제N조의M" 패턴도 처리
            # 법령명은 조사(가, 이, 을, 를)로 시작하지 않도록 개선
            pattern1 = r'([가-힣][가-힣\s]{0,50}?(?:법|법률))\s*제\s*(\d+)(?:조의\d+)?\s*조(?:\s*제\s*(\d+)\s*항)?(?:\s*제\s*(\d+)\s*호)?'
            match1 = re.search(pattern1, item)
            
            if match1:
                statute_name = match1.group(1).strip()
                article_no = match1.group(2)
                clause_no = match1.group(3)
                item_no = match1.group(4)
                
                # 법령명 정리
                statute_name = re.sub(r'\s+', ' ', statute_name).strip()
                
                # 조사로 시작하는 경우 제거 (가, 이, 을, 를, 에, 의 등)
                unwanted_starters = ['가 ', '이 ', '을 ', '를 ', '에 ', '의 ', '에서 ', '으로 ', '로 ', '와 ', '과 ']
                for starter in unwanted_starters:
                    if statute_name.startswith(starter):
                        statute_name = statute_name[len(starter):].strip()
                
                # 불필요한 키워드 제거 (참조조문, 판시사항 등)
                unwanted_prefixes = ['참조조문', '참조판례', '판시사항', '실질과세에', '지연손해금', '산정에', '대하여']
                for prefix in unwanted_prefixes:
                    if statute_name.startswith(prefix):
                        statute_name = statute_name[len(prefix):].strip()
                
                # 불필요한 단어 제거 (중간에 있는 경우, "관한"은 제외)
                unwanted_words = ['에서', '정한', '규정', '취지']
                for word in unwanted_words:
                    # 단어 경계를 고려하여 제거
                    statute_name = re.sub(rf'\b{re.escape(word)}\b\s*', '', statute_name).strip()
                
                # 연속된 공백 정리
                statute_name = re.sub(r'\s+', ' ', statute_name).strip()
                
                # "법" 또는 "법률"로 끝나는 부분만 추출
                # "~에 관한 ~법" 패턴 보존
                # 법령명은 보통 짧고, 조사로 시작하지 않음
                # 뒤에서부터 "법" 또는 "법률"을 찾아서 추출
                statute_match = None
                # 먼저 "법" 또는 "법률"로 끝나는 가장 긴 부분 찾기
                for i in range(len(statute_name), 0, -1):
                    substr = statute_name[:i]
                    if substr.endswith('법') or substr.endswith('법률'):
                        # 조사로 시작하지 않는지 확인
                        if not any(substr.startswith(s) for s in ['가 ', '이 ', '을 ', '를 ', '에 ', '의 ', '에서 ', '으로 ', '로 ', '와 ', '과 ']):
                            statute_match = re.search(r'([가-힣]{2,50}(?:법|법률))', substr)
                            if statute_match:
                                statute_name = statute_match.group(1).strip()
                                break
                
                if not statute_match:
                    # Fallback: 기존 방식
                    statute_match = re.search(r'([가-힣\s]+(?:법|법률))', statute_name)
                    if statute_match:
                        statute_name = statute_match.group(1).strip()
                
                if statute_match:
                    # "에 특별법" -> "에 관한 특별법" 복원 시도
                    if '에 특별법' in statute_name or '에 특례법' in statute_name:
                        statute_name = statute_name.replace('에 특별법', '에 관한 특별법')
                        statute_name = statute_name.replace('에 특례법', '에 관한 특례법')
                    # "관한"으로 시작하는 경우 제거
                    if statute_name.startswith('관한'):
                        statute_name = statute_name[2:].strip()
                    # "관한"으로 시작하는 경우 (공백 포함)
                    if statute_name.startswith('관한 '):
                        statute_name = statute_name[3:].strip()
                
                # 비정상적으로 긴 법령명 제거 (100자 이상)
                if len(statute_name) > 100:
                    # 다시 짧은 법령명 추출 시도
                    short_match = re.search(r'([가-힣]{2,30}(?:법|법률))', statute_name)
                    if short_match:
                        statute_name = short_match.group(1).strip()
                    else:
                        continue
                
                # "참조조문", "판시사항" 등이 포함된 경우 제거
                if '참조조문' in statute_name or '판시사항' in statute_name or '참조판례' in statute_name:
                    # 법령명만 추출
                    clean_match = re.search(r'([가-힣]{2,30}(?:법|법률))', statute_name)
                    if clean_match:
                        statute_name = clean_match.group(1).strip()
                    else:
                        continue
                
                # 법령명이 "법" 또는 "법률"로 끝나지 않으면 제외
                if not statute_name.endswith("법") and not statute_name.endswith("법률"):
                    continue
                
                # 법령명이 너무 짧거나 비정상적이면 제외
                if len(statute_name) < 2:
                    continue
                
                last_statute_name = statute_name
            else:
                # 패턴 2: "법 제1조" 형식 (법령명이 "법"으로만 표시)
                pattern2 = r'법\s*제\s*(\d+)\s*조(?:\s*제\s*(\d+)\s*항)?(?:\s*제\s*(\d+)\s*호)?'
                match2 = re.search(pattern2, item)
                
                if match2:
                    # "법"만 있는 경우 이전 법령명 사용
                    if last_statute_name:
                        statute_name = last_statute_name
                    else:
                        continue  # 이전 법령명이 없으면 스킵
                    article_no = match2.group(1)
                    clause_no = match2.group(2)
                    item_no = match2.group(3)
                else:
                    # 패턴 3: "제1조" 형식 (법령명 완전 생략)
                    pattern3 = r'제\s*(\d+)\s*조(?:\s*제\s*(\d+)\s*항)?(?:\s*제\s*(\d+)\s*호)?'
                    match3 = re.search(pattern3, item)
                    
                    if match3:
                        # 법령명이 생략된 경우 이전 법령명 사용
                        if last_statute_name:
                            statute_name = last_statute_name
                        else:
                            continue  # 이전 법령명이 없으면 스킵
                        article_no = match3.group(1)
                        clause_no = match3.group(2)
                        item_no = match3.group(3)
                    else:
                        continue
            
            # 중복 제거
            key = f"{statute_name}_{article_no}_{clause_no or ''}_{item_no or ''}"
            if key in seen:
                continue
            seen.add(key)
            
            statute_info = {
                "statute_name": statute_name,
                "article_no": article_no,
            }
            
            if clause_no:
                statute_info["clause_no"] = clause_no
            
            if item_no:
                statute_info["item_no"] = item_no
            
            statutes.append(statute_info)
        
        return statutes
    
    def to_json(self, statutes: List[Dict[str, Any]]) -> Optional[str]:
        """JSON 문자열로 변환"""
        if not statutes:
            return None
        return json.dumps(statutes, ensure_ascii=False)
    
    def from_json(self, json_str: str) -> List[Dict[str, Any]]:
        """JSON 문자열에서 파싱"""
        if not json_str:
            return []
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return []

