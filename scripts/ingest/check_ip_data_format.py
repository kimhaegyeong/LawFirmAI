"""
지식재산권법 데이터 형식 확인 스크립트
"""
import json
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def check_data_format():
    """지식재산권법 데이터 형식 확인"""
    data_path = Path("data/aihub/02.지식재산권법 LLM 사전학습 및 Instruction Tuning 데이터/3.개방데이터/1.데이터/Training/01.원천데이터")
    
    if not data_path.exists():
        print(f"❌ 경로를 찾을 수 없습니다: {data_path}")
        print(f"   절대 경로: {data_path.resolve()}")
        return
    
    print(f"📁 데이터 경로: {data_path}")
    print(f"   절대 경로: {data_path.resolve()}\n")
    
    # JSON 파일 찾기
    json_files = list(data_path.rglob("*.json"))
    
    if not json_files:
        print("❌ JSON 파일을 찾을 수 없습니다.")
        return
    
    print(f"✅ 총 {len(json_files)}개의 JSON 파일 발견\n")
    print("=" * 80)
    
    # 샘플 파일 3개 확인
    sample_files = json_files[:3]
    
    for idx, json_file in enumerate(sample_files, 1):
        print(f"\n📄 샘플 파일 {idx}: {json_file.name}")
        try:
            rel_path = json_file.relative_to(_PROJECT_ROOT)
            print(f"   경로: {rel_path}")
        except ValueError:
            print(f"   경로: {json_file}")
        print("-" * 80)
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ JSON 파싱 성공")
            print(f"\n📋 키 목록:")
            for key in data.keys():
                value = data[key]
                if isinstance(value, list):
                    print(f"   - {key}: List[{len(value)} items]")
                    if value and isinstance(value[0], str):
                        print(f"     예시: {value[0][:100]}...")
                elif isinstance(value, dict):
                    print(f"   - {key}: Dict[{len(value)} keys]")
                elif isinstance(value, str):
                    print(f"   - {key}: String (길이: {len(value)})")
                    print(f"     내용: {value[:100]}...")
                else:
                    print(f"   - {key}: {type(value).__name__} = {value}")
            
            print(f"\n📝 전체 데이터 구조 (처음 1000자):")
            print(json.dumps(data, ensure_ascii=False, indent=2)[:1000])
            if len(json.dumps(data, ensure_ascii=False)) > 1000:
                print("...")
            
            # 판례 형식인지 확인
            print(f"\n🔍 판례 형식 호환성 체크:")
            required_fields = ['doc_id', 'sentences', 'casenames', 'court', 'announce_date']
            optional_fields = ['case_type', 'casetype', 'normalized_court', 'full_text', 'content']
            
            found_required = []
            found_optional = []
            
            for field in required_fields:
                if field in data:
                    found_required.append(field)
                    print(f"   ✅ {field}: {type(data[field]).__name__}")
                else:
                    print(f"   ❌ {field}: 없음")
            
            for field in optional_fields:
                if field in data:
                    found_optional.append(field)
                    print(f"   ⚠️  {field}: {type(data[field]).__name__} (선택적)")
            
            compatibility = len(found_required) / len(required_fields) * 100
            print(f"\n   호환성: {compatibility:.1f}% ({len(found_required)}/{len(required_fields)} 필수 필드 발견)")
            
            if compatibility >= 60:
                print(f"   💡 ingest_cases.py 사용 가능할 것으로 예상")
            else:
                print(f"   ⚠️  ingest_cases.py 수정 필요할 수 있음")
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 오류: {e}")
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        
        print("=" * 80)
    
    # 전체 파일 통계
    print(f"\n📊 전체 파일 통계:")
    print(f"   총 JSON 파일 수: {len(json_files)}")
    
    # 하위 디렉토리 구조 확인
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    if subdirs:
        print(f"\n📁 하위 디렉토리:")
        for subdir in subdirs[:10]:  # 최대 10개만 표시
            files_in_dir = list(subdir.rglob("*.json"))
            print(f"   - {subdir.name}: {len(files_in_dir)}개 파일")
        if len(subdirs) > 10:
            print(f"   ... 외 {len(subdirs) - 10}개 디렉토리")

if __name__ == "__main__":
    check_data_format()

