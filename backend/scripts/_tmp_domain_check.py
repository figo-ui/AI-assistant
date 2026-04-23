import pandas as pd
from urllib.parse import urlparse

csv_path = r'C:\Users\hp\Desktop\AI assistant\data\raw\hub\datasets\Imaging\New folder\fitzpatrick17k (1).csv'
df = pd.read_csv(csv_path)
df['domain'] = df['url'].dropna().apply(lambda u: urlparse(u).netloc)

print('Domain breakdown:')
print(df['domain'].value_counts())
print()
print(f'Total rows: {len(df)}')
print(f'dermaamin rows: {(df["domain"]=="www.dermaamin.com").sum()}')
print(f'atlasdermatologico rows: {(df["domain"]=="atlasdermatologico.com.br").sum()}')
print()

# Check label distribution for atlas-only rows
atlas_df = df[df['domain'] == 'atlasdermatologico.com.br']
print(f'Atlas label count: {atlas_df["label"].nunique()} unique labels')
print(atlas_df['label'].value_counts().head(20))
print()
print('Atlas three_partition_label:')
print(atlas_df['three_partition_label'].value_counts())
print()
print('Atlas nine_partition_label:')
print(atlas_df['nine_partition_label'].value_counts())
