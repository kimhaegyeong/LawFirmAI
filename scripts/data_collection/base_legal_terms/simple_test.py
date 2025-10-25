"""
법령정보지식베이스 법령용어 수집 시스템 간단 테스트

이 스크립트는 수집 시스템의 기본 기능을 간단히 테스트합니다.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# 설정 파일 import
from base_legal_term_collection_config import BaseLegalTermCollectionConfig as Config

def test_config():
    """설정 파일 테스트"""
    print("=== 설정 파일 테스트 ===")
    
    try:
        config = Config()
        
        # 설정 유효성 검증
        if not config.validate_config():
            print("❌ 설정 파일 유효성 검증 실패")
            return False
        
        # 주요 설정 확인
        api_config = config.get_api_config()
        collection_config = config.get_collection_config()
        file_storage_config = config.get_file_storage_config()
        
        print(f"✅ API 설정: {api_config.get('base_url')}")
        print(f"✅ 수집 설정: 배치 크기 {collection_config.get('list_batch_size')}")
        print(f"✅ 파일 저장 설정: {file_storage_config.get('base_dir')}")
        
        print("✅ 설정 파일 테스트 성공")
        return True
        
    except Exception as e:
        print(f"❌ 설정 파일 테스트 실패: {e}")
        return False

def test_file_structure():
    """파일 구조 테스트"""
    print("\n=== 파일 구조 테스트 ===")
    
    try:
        config = Config()
        file_storage_config = config.get_file_storage_config()
        base_dir = Path(file_storage_config.get("base_dir", "data/base_legal_terms"))
        
        # 필요한 디렉토리 확인
        required_dirs = [
            base_dir / "raw" / "term_lists",
            base_dir / "raw" / "term_details",
            base_dir / "raw" / "term_relations",
            base_dir / "processed" / "cleaned_terms",
            base_dir / "processed" / "normalized_terms",
            base_dir / "processed" / "validated_terms",
            base_dir / "processed" / "integrated_terms",
            base_dir / "embeddings",
            base_dir / "database",
            base_dir / "logs",
            base_dir / "progress",
            base_dir / "reports",
            base_dir / "config"
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                print(f"❌ 필수 디렉토리 누락: {directory}")
                return False
            print(f"✅ 디렉토리 확인: {directory}")
        
        print("✅ 파일 구조 테스트 성공")
        return True
        
    except Exception as e:
        print(f"❌ 파일 구조 테스트 실패: {e}")
        return False

def test_data_processing():
    """데이터 처리 테스트"""
    print("\n=== 데이터 처리 테스트 ===")
    
    try:
        # 샘플 용어 데이터
        sample_term = {
            "법령용어ID": "test_001",
            "법령용어명": "계약",
            "법령용어정의": "당사자 일방이 상대방에 대하여 일정한 행위를 약속하고, 상대방이 그 약속에 대하여 대가를 지급할 것을 약속하는 법률행위",
            "동음이의어내용": "",
            "용어관계정보": [],
            "조문관계정보": []
        }
        
        # 간단한 데이터 정제 테스트
        def clean_term_data(term_data):
            cleaned_data = {}
            for key, value in term_data.items():
                if isinstance(value, str):
                    cleaned_value = value.strip()
                    cleaned_data[key] = cleaned_value
                else:
                    cleaned_data[key] = value
            return cleaned_data
        
        cleaned_term = clean_term_data(sample_term)
        
        # 용어명 정규화 테스트
        def normalize_term_name(term_name):
            import re
            normalized = term_name.strip()
            normalized = re.sub(r'\([^)]*\)', '', normalized)
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            return normalized
        
        normalized_name = normalize_term_name(cleaned_term.get('법령용어명', ''))
        
        # 키워드 추출 테스트
        def extract_keywords(term_name, definition):
            import re
            keywords = []
            words = re.findall(r'[가-힣]+', term_name)
            keywords.extend(words)
            
            definition_words = re.findall(r'[가-힣]{2,}', definition)
            word_freq = {}
            for word in definition_words:
                if len(word) >= 2:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            for word, freq in sorted_words[:10]:
                if word not in keywords:
                    keywords.append(word)
            
            return keywords[:20]
        
        keywords = extract_keywords(
            cleaned_term.get('법령용어명', ''),
            cleaned_term.get('법령용어정의', '')
        )
        
        # 카테고리 분류 테스트
        def categorize_term(term_name, definition):
            categories = {
                "민사법": ["계약", "손해", "배상", "소유", "물권", "채권", "가족", "상속"],
                "형사법": ["범죄", "형벌", "처벌", "구금", "수사", "기소", "재판"],
                "행정법": ["행정", "허가", "인가", "신고", "신청", "처분", "행정행위"]
            }
            
            text = f"{term_name} {definition}"
            category_scores = {}
            
            for category, keywords in categories.items():
                score = 0
                for keyword in keywords:
                    if keyword in text:
                        score += 1
                category_scores[category] = score
            
            if category_scores:
                best_category = max(category_scores.items(), key=lambda x: x[1])
                if best_category[1] > 0:
                    return best_category[0]
            
            return "기타"
        
        category = categorize_term(
            cleaned_term.get('법령용어명', ''),
            cleaned_term.get('법령용어정의', '')
        )
        
        # 품질 점수 계산 테스트
        def calculate_quality_score(term_data):
            score = 0.0
            
            term_name = term_data.get('법령용어명', '')
            if len(term_name) >= 2:
                score += 30
            
            definition = term_data.get('법령용어정의', '')
            if len(definition) >= 20:
                score += 40
            elif len(definition) >= 10:
                score += 20
            
            return min(score, 100.0)
        
        quality_score = calculate_quality_score(cleaned_term)
        
        print(f"✅ 원본 용어명: {sample_term['법령용어명']}")
        print(f"✅ 정규화된 용어명: {normalized_name}")
        print(f"✅ 카테고리: {category}")
        print(f"✅ 품질 점수: {quality_score}")
        print(f"✅ 키워드: {keywords}")
        
        print("✅ 데이터 처리 테스트 성공")
        return True
        
    except Exception as e:
        print(f"❌ 데이터 처리 테스트 실패: {e}")
        return False

def create_sample_data():
    """샘플 데이터 생성"""
    print("\n=== 샘플 데이터 생성 ===")
    
    try:
        config = Config()
        file_storage_config = config.get_file_storage_config()
        term_lists_dir = Path(file_storage_config.get("term_lists_dir", "data/base_legal_terms/raw/term_lists"))
        
        # 샘플 용어 데이터
        sample_terms = [
            {
                "법령용어ID": "test_001",
                "법령용어명": "계약",
                "동음이의어존재여부": "N",
                "비고": "",
                "용어간관계링크": "/test/link1",
                "조문간관계링크": "/test/link2",
                "수집일시": datetime.now().isoformat()
            },
            {
                "법령용어ID": "test_002",
                "법령용어명": "손해배상",
                "동음이의어존재여부": "Y",
                "비고": "손해배상(損害賠償)",
                "용어간관계링크": "/test/link3",
                "조문간관계링크": "/test/link4",
                "수집일시": datetime.now().isoformat()
            },
            {
                "법령용어ID": "test_003",
                "법령용어명": "소유권",
                "동음이의어존재여부": "N",
                "비고": "",
                "용어간관계링크": "/test/link5",
                "조문간관계링크": "/test/link6",
                "수집일시": datetime.now().isoformat()
            }
        ]
        
        # 샘플 파일 저장
        sample_file = term_lists_dir / f"sample_terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_terms, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 샘플 데이터 생성: {sample_file}")
        return True
        
    except Exception as e:
        print(f"❌ 샘플 데이터 생성 실패: {e}")
        return False

def save_test_report(test_results):
    """테스트 보고서 저장"""
    try:
        report_data = {
            "테스트보고서": {
                "테스트일시": datetime.now().isoformat(),
                "테스트결과": test_results,
                "전체성공여부": "성공" if all(test_results.values()) else "실패",
                "성공한테스트": [k for k, v in test_results.items() if v],
                "실패한테스트": [k for k, v in test_results.items() if not v]
            }
        }
        
        reports_dir = Path("data/base_legal_terms/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = reports_dir / f"simple_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 테스트 보고서 저장: {report_file}")
        
    except Exception as e:
        print(f"❌ 테스트 보고서 저장 실패: {e}")

def main():
    """메인 실행 함수"""
    print("=== 법령정보지식베이스 법령용어 수집 시스템 간단 테스트 ===")
    
    test_results = {
        "config_test": False,
        "file_structure_test": False,
        "data_processing_test": False,
        "sample_data_test": False
    }
    
    try:
        # 1. 설정 파일 테스트
        test_results["config_test"] = test_config()
        
        # 2. 파일 구조 테스트
        test_results["file_structure_test"] = test_file_structure()
        
        # 3. 데이터 처리 테스트
        test_results["data_processing_test"] = test_data_processing()
        
        # 4. 샘플 데이터 생성 테스트
        test_results["sample_data_test"] = create_sample_data()
        
        # 5. 테스트 보고서 저장
        save_test_report(test_results)
        
        # 결과 출력
        print("\n=== 테스트 결과 ===")
        for test_name, result in test_results.items():
            status = "성공" if result else "실패"
            print(f"{test_name}: {status}")
        
        all_passed = all(test_results.values())
        print(f"\n전체 테스트 결과: {'성공' if all_passed else '실패'}")
        
        if all_passed:
            print("\n🎉 모든 테스트 통과! 시스템이 정상적으로 설정되었습니다.")
        else:
            print("\n⚠️ 일부 테스트 실패. 로그를 확인해주세요.")
            
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
