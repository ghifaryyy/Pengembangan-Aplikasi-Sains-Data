import csv, json, os

csv_path = r'UMKM-data/synthetic_umkm_data.csv'
out_path = r'UMKM-data/umkm_preview.json'

stats = {'total': 0, 'Elite': 0, 'Growth': 0, 'Struggling': 0, 'Critical': 0}
rows = []
MAX = 1000

with open(csv_path, encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        stats['total'] += 1
        cls = row.get('Class', '')
        if cls in stats:
            stats[cls] += 1
        if i < MAX:
            rows.append({
                'ID': int(row.get('ID') or 0),
                'Monthly_Revenue': float(row.get('Monthly_Revenue') or 0),
                'Net_Profit_Margin': float(row.get('Net_Profit_Margin (%)') or 0),
                'Burn_Rate_Ratio': float(row.get('Burn_Rate_Ratio') or 0),
                'Transaction_Count': int(row.get('Transaction_Count') or 0),
                'Avg_Historical_Rating': float(row.get('Avg_Historical_Rating') or 0),
                'Sentiment_Score': float(row.get('Sentiment_Score') or 0),
                'Review_Text': row.get('Review_Text', ''),
                'Class': cls
            })

output = {'stats': stats, 'rows': rows}
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, separators=(',', ':'))

size_kb = os.path.getsize(out_path) // 1024
print(f'Done! {stats["total"]} total rows, {len(rows)} preview rows, JSON: {size_kb} KB')
