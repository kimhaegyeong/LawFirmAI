import json
import os

# 제2조의 하위 조문 파싱 결과 확인
processed_dir = 'data/processed/assembly/law/2025101201_final/20251012'
files = [f for f in os.listdir(processed_dir) if f.endswith('.json') and not f.startswith('metadata_')]

print('=== 제2조 하위 조문 파싱 결과 확인 ===')

ho_count = 0
mok_count = 0
total_article_2 = 0
sample_results = []

for file in files:
    file_path = os.path.join(processed_dir, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    articles = data.get('articles', [])
    for article in articles:
        if article.get('article_number') == '2':
            total_article_2 += 1
            sub_articles = article.get('sub_articles', [])
            ho_items = []
            mok_items = []
            
            for sub_article in sub_articles:
                if sub_article.get('type') == 'ho':
                    ho_count += 1
                    ho_items.append(sub_article.get('content', '')[:50])
                elif sub_article.get('type') == 'mok':
                    mok_count += 1
                    mok_items.append(sub_article.get('content', '')[:50])
            
            if ho_items or mok_items:
                sample_results.append({
                    'law_name': data.get('law_name', 'Unknown'),
                    'ho_count': len(ho_items),
                    'mok_count': len(mok_items),
                    'ho_samples': ho_items[:3],
                    'mok_samples': mok_items[:3]
                })

print(f'제2조 총 개수: {total_article_2}')
print(f'호(號) 단위 파싱된 개수: {ho_count}')
print(f'목(目) 단위 파싱된 개수: {mok_count}')
print(f'호(號) 파싱 성공률: {ho_count/total_article_2*100:.1f}%' if total_article_2 > 0 else 'N/A')

print('\n=== 샘플 파싱 결과 ===')
for i, result in enumerate(sample_results[:10]):
    print(f'\n{i+1}. {result["law_name"]}')
    print(f'   호(號): {result["ho_count"]}개')
    print(f'   목(目): {result["mok_count"]}개')
    if result["ho_samples"]:
        print(f'   호(號) 샘플: {result["ho_samples"]}')
    if result["mok_samples"]:
        print(f'   목(目) 샘플: {result["mok_samples"]}')

# 전체 하위 조문 타입 분포 확인
print('\n=== 전체 하위 조문 타입 분포 ===')
type_counts = {}
for file in files:
    file_path = os.path.join(processed_dir, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    articles = data.get('articles', [])
    for article in articles:
        sub_articles = article.get('sub_articles', [])
        for sub_article in sub_articles:
            sub_type = sub_article.get('type', 'unknown')
            type_counts[sub_type] = type_counts.get(sub_type, 0) + 1

for sub_type, count in sorted(type_counts.items()):
    print(f'{sub_type}: {count}개')
