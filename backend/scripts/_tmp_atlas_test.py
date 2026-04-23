"""Test atlas domain download success rate more thoroughly"""
import pandas as pd
import requests
import time
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

csv_path = r'C:\Users\hp\Desktop\AI assistant\data\raw\hub\datasets\Imaging\New folder\fitzpatrick17k (1).csv'
df = pd.read_csv(csv_path)
df['domain'] = df['url'].dropna().apply(lambda u: urlparse(u).netloc)

atlas_df = df[df['domain'] == 'atlasdermatologico.com.br'].copy()
print(f"Atlas rows: {len(atlas_df)}")

# Test 50 random atlas URLs
sample = atlas_df.sample(50, random_state=42)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
}

ok = 0
fail_404 = 0
fail_other = 0
fail_exc = 0

def test_url(url):
    try:
        r = requests.get(url, timeout=15, headers=headers)
        return r.status_code, len(r.content)
    except Exception as e:
        return -1, 0

with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(test_url, row['url']): row['url'] for _, row in sample.iterrows()}
    for future in as_completed(futures):
        status, size = future.result()
        if status == 200 and size > 500:
            ok += 1
        elif status == 404:
            fail_404 += 1
        elif status == -1:
            fail_exc += 1
        else:
            fail_other += 1
            print(f"  Other status {status}, size={size}")

print(f"\nResults from 50 random atlas URLs:")
print(f"  OK (200, >500 bytes): {ok}")
print(f"  404: {fail_404}")
print(f"  Other HTTP error: {fail_other}")
print(f"  Exception: {fail_exc}")
print(f"  Success rate: {ok/50*100:.1f}%")
print(f"\nEstimated downloadable atlas images: ~{int(ok/50 * len(atlas_df))}")
