# Synthetic UMKM Dataset Documentation

## Ringkasan
Dataset ini adalah **data sintetis** yang meniru karakteristik operasional UMKM secara realistis pada level bulanan. Dataset dirancang untuk eksplorasi data, pemodelan machine learning, simulasi bisnis, dan analisis sentimen pada ulasan pelanggan.

## Tujuan Dataset
Dataset memodelkan keterkaitan metrik inti UMKM dalam satu kerangka yang konsisten:

- Skala bisnis: pendapatan dan volume transaksi
- Efisiensi operasional: burn rate dan margin laba
- Kualitas layanan: rating, volatility review, dan latency
- Maturitas usaha: lama operasional
- Retensi pelanggan: repeat order rate
- Kematangan digital: digital adoption score
- Tekanan pasar: competitiveness lokasi
- Persepsi pelanggan: sentiment score dari teks ulasan

## Struktur Kolom

| Fitur | Tipe Data | Skala/Satuan | Deskripsi Teknis |
|---|---|---|---|
| `ID` | Integer | Bilangan bulat | Identitas unik setiap baris data. |
| `Monthly_Revenue` | Integer | IDR | Total nilai penjualan bulanan. |
| `Net_Profit_Margin (%)` | Float | Persentase (%) | Rasio laba bersih terhadap pendapatan. |
| `Burn_Rate_Ratio` | Float | Rasio | Pengeluaran operasional dibanding pendapatan (`&gt; 1.0` cenderung defisit). |
| `Transaction_Count` | Integer | Frekuensi | Jumlah transaksi unik dalam periode observasi. |
| `Avg_Historical_Rating` | Float | Skala 1-5 | Rata-rata skor penilaian pelanggan. |
| `Review_Text` | String | Teks | Ulasan pelanggan sintetis untuk kebutuhan NLP. |
| `Review_Volatility` | Float | Indeks | Fluktuasi kualitas layanan/ulasan. |
| `Business_Tenure_Months` | Integer | Bulan | Lama usaha beroperasi. |
| `Repeat_Order_Rate (%)` | Float | Persentase (%) | Proporsi repeat order pelanggan. |
| `Digital_Adoption_Score` | Float | Skala 1-10 | Tingkat adopsi kanal dan proses digital. |
| `Peak_Hour_Latency` | Categorical | `Low`/`Med`/`High` | Kualitas layanan saat jam sibuk. |
| `Location_Competitiveness` | Integer | Jumlah | Kerapatan kompetitor di area yang sama. |
| `Sentiment_Score` | Float | -1.0 s/d 1.0 | Skor sentimen hasil konversi `Review_Text`. |
| `Class` | Categorical | `Elite`/`Growth`/`Struggling`/`Critical` | Variabel target kelas bisnis. |

## Logika Sintesis Data
Generator tidak mengacak nilai secara independen, tetapi mengikuti logika operasional berikut:

1. Bangkitkan faktor dasar: `Business_Tenure_Months` dan `Location_Competitiveness`.
2. Turunkan `Digital_Adoption_Score` dari maturitas bisnis dengan noise.
3. Bentuk `Transaction_Count` dari maturitas, adopsi digital, dan kompetisi.
4. Hitung `Monthly_Revenue` dari `Transaction_Count` x AOV lognormal + noise musiman.
5. Turunkan `Peak_Hour_Latency` dari tekanan transaksi, adopsi digital, dan kompetisi.
6. Bentuk `Burn_Rate_Ratio` dari kompetisi, latency, adopsi digital, dan noise.
7. Hitung `Net_Profit_Margin (%)` dengan hubungan terbalik terhadap burn rate.
8. Bentuk `Repeat_Order_Rate (%)` dari digital adoption, tenure, kompetisi, dan noise.
9. Turunkan `Review_Volatility` dari latency dan kondisi burn rate.
10. Bentuk `Avg_Historical_Rating` dari sinyal kualitas operasional.
11. Generate `Review_Text` yang konsisten dengan rating, volatility, dan latency.
12. Konversi `Review_Text` menjadi `Sentiment_Score` berbasis keyword.
13. Klasifikasikan `Class` menggunakan threshold persentil agar distribusi kelas lebih seimbang.

## Logika Variabel Target `Class`
Label target dibentuk secara rule-based dengan prioritas kondisi dan threshold persentil:

- `Elite`: kombinasi margin tinggi, burn rate rendah, repeat order tinggi, dan rating tinggi.
- `Struggling`: sinyal profitabilitas melemah dan/atau kualitas layanan menurun.
- `Critical`: kondisi risiko tinggi (burn rate ekstrem, usaha sangat baru di pasar kompetitif, atau kombinasi rugi berat + rating rendah).
- `Growth`: kondisi default di luar tiga kondisi ekstrem di atas.

## Karakteristik Realisme

- Distribusi finansial menggunakan lognormal agar mengikuti pola skew data ekonomi nyata.
- Antarvariabel dibangun saling terkait (bukan random independent).
- Terdapat noise terkontrol untuk menghindari data terlalu sempurna.
- Ada post-adjustment untuk bisnis dengan burn rate sangat tinggi agar rating dan repeat order tetap masuk akal.

## Contoh Use Case

- EDA: analisis distribusi pendapatan, margin, burn rate, dan segmentasi kelas bisnis.
- ML klasifikasi: prediksi `Class`.
- ML regresi: prediksi `Net_Profit_Margin (%)`.
- NLP: analisis sentimen pada `Review_Text` dan validasi terhadap `Sentiment_Score`.

## Batasan Dataset

- Dataset ini **bukan data riil** dan tidak merepresentasikan entitas bisnis spesifik.
- Hubungan antarvariabel adalah asumsi desain generator.
- Tidak ditujukan untuk inferensi kausal kebijakan nyata tanpa kalibrasi ke data empiris.
- `Review_Text` bersifat template-based, bukan percakapan pengguna asli.

## Panduan Pemakaian Cepat

```python
import pandas as pd

df = pd.read_csv("synthetic_umkm_data.csv")
print(df.shape)
print(df.head())
print(df["Class"].value_counts())
```

## Reproducibility
Generator menggunakan seed tetap (`SEED = 42`) sehingga data dapat direproduksi selama parameter generator tidak diubah.