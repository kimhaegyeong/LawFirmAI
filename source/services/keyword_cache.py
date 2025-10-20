import json
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

class KeywordCache:
    """키워드 캐싱 시스템"""
    
    def __init__(self, cache_dir: str = "data/cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours
        self.logger = logging.getLogger(__name__)
        
        # 메모리 캐시
        self.memory_cache = {}
        self.cache_metadata = {}
    
    def get(self, domain: str) -> Optional[List[str]]:
        """캐시에서 키워드 조회"""
        cache_key = self._generate_cache_key(domain)
        
        # 메모리 캐시에서 먼저 확인
        if cache_key in self.memory_cache:
            if self._is_cache_valid(cache_key):
                self.logger.debug(f"메모리 캐시에서 키워드 로드: {domain}")
                return self.memory_cache[cache_key]['keywords']
            else:
                # 만료된 캐시 제거
                del self.memory_cache[cache_key]
        
        # 파일 캐시에서 확인
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                if self._is_file_cache_valid(cache_data):
                    keywords = cache_data.get('keywords', [])
                    # 메모리 캐시에 저장
                    self.memory_cache[cache_key] = {
                        'keywords': keywords,
                        'timestamp': cache_data['timestamp']
                    }
                    self.logger.debug(f"파일 캐시에서 키워드 로드: {domain}")
                    return keywords
                else:
                    # 만료된 파일 캐시 삭제
                    cache_file.unlink()
                    self.logger.debug(f"만료된 파일 캐시 삭제: {domain}")
            except Exception as e:
                self.logger.error(f"캐시 파일 읽기 실패 ({domain}): {e}")
        
        return None
    
    def set(self, domain: str, keywords: List[str], metadata: Optional[Dict] = None) -> bool:
        """캐시에 키워드 저장"""
        try:
            cache_key = self._generate_cache_key(domain)
            timestamp = time.time()
            
            # 메모리 캐시에 저장
            self.memory_cache[cache_key] = {
                'keywords': keywords,
                'timestamp': timestamp
            }
            
            # 파일 캐시에 저장
            cache_data = {
                'domain': domain,
                'keywords': keywords,
                'timestamp': timestamp,
                'metadata': metadata or {},
                'created_at': datetime.now().isoformat()
            }
            
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"키워드 캐시 저장 완료: {domain} ({len(keywords)}개)")
            return True
            
        except Exception as e:
            self.logger.error(f"캐시 저장 실패 ({domain}): {e}")
            return False
    
    def invalidate(self, domain: str) -> bool:
        """특정 도메인의 캐시 무효화"""
        try:
            cache_key = self._generate_cache_key(domain)
            
            # 메모리 캐시에서 제거
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
            
            # 파일 캐시에서 제거
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                cache_file.unlink()
            
            self.logger.debug(f"캐시 무효화 완료: {domain}")
            return True
            
        except Exception as e:
            self.logger.error(f"캐시 무효화 실패 ({domain}): {e}")
            return False
    
    def clear_all(self) -> bool:
        """모든 캐시 삭제"""
        try:
            # 메모리 캐시 클리어
            self.memory_cache.clear()
            
            # 파일 캐시 클리어
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            
            self.logger.info("모든 캐시 삭제 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"캐시 전체 삭제 실패: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        memory_count = len(self.memory_cache)
        file_count = len(list(self.cache_dir.glob("*.json")))
        
        # 파일 캐시 크기 계산
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))
        
        return {
            'memory_cache_count': memory_count,
            'file_cache_count': file_count,
            'total_cache_size_bytes': total_size,
            'cache_dir': str(self.cache_dir),
            'ttl_hours': self.ttl_hours
        }
    
    def cleanup_expired(self) -> int:
        """만료된 캐시 정리"""
        cleaned_count = 0
        
        # 메모리 캐시 정리
        expired_keys = []
        for cache_key, cache_data in self.memory_cache.items():
            if not self._is_cache_valid(cache_key):
                expired_keys.append(cache_key)
        
        for cache_key in expired_keys:
            del self.memory_cache[cache_key]
            cleaned_count += 1
        
        # 파일 캐시 정리
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                if not self._is_file_cache_valid(cache_data):
                    cache_file.unlink()
                    cleaned_count += 1
                    
            except Exception as e:
                self.logger.warning(f"캐시 파일 정리 중 오류 ({cache_file}): {e}")
                # 읽을 수 없는 파일은 삭제
                cache_file.unlink()
                cleaned_count += 1
        
        if cleaned_count > 0:
            self.logger.info(f"만료된 캐시 {cleaned_count}개 정리 완료")
        
        return cleaned_count
    
    def _generate_cache_key(self, domain: str) -> str:
        """캐시 키 생성"""
        return hashlib.md5(domain.encode('utf-8')).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """메모리 캐시 유효성 검사"""
        if cache_key not in self.memory_cache:
            return False
        
        cache_data = self.memory_cache[cache_key]
        timestamp = cache_data.get('timestamp', 0)
        
        return time.time() - timestamp < (self.ttl_hours * 3600)
    
    def _is_file_cache_valid(self, cache_data: Dict) -> bool:
        """파일 캐시 유효성 검사"""
        timestamp = cache_data.get('timestamp', 0)
        return time.time() - timestamp < (self.ttl_hours * 3600)
    
    def get_domain_cache_info(self, domain: str) -> Optional[Dict[str, Any]]:
        """특정 도메인의 캐시 정보 반환"""
        cache_key = self._generate_cache_key(domain)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            return {
                'domain': cache_data.get('domain'),
                'keyword_count': len(cache_data.get('keywords', [])),
                'created_at': cache_data.get('created_at'),
                'timestamp': cache_data.get('timestamp'),
                'is_valid': self._is_file_cache_valid(cache_data),
                'metadata': cache_data.get('metadata', {})
            }
        except Exception as e:
            self.logger.error(f"캐시 정보 조회 실패 ({domain}): {e}")
            return None
