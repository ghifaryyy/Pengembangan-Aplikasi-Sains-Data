# 📊 UMKM Quality & Sustainability Predictor

## 📌 Ringkasan Proyek
Dataset ini adalah data sintetis yang meniru karakteristik operasional UMKM secara realistis pada level bulanan. Proyek ini bertujuan untuk mengatasi bias informasi dalam menilai kesehatan UMKM, di mana penilaian sering kali hanya berpatokan pada omzet kotor.

## 💡 Solusi yang Ditawarkan
Kami membangun **Sistem Prediksi Kesehatan dan Keberlanjutan UMKM** berbasis *Machine Learning*. Sistem ini mengintegrasikan:
1. **Kesehatan Finansial** (*Net Profit Margin* & *Burn Rate*)
2. **Efisiensi Operasional** (*Peak Hour Latency*)
3. **Persepsi Pelanggan** (*Sentiment Score* dari ulasan)

Outputnya adalah klasifikasi UMKM ke dalam 4 kelas: **Elite, Growth, Struggling, dan Critical**. Ini bertindak sebagai sistem *Credit Scoring* alternatif dan alat diagnosa (Early Warning System) untuk mencegah kebangkrutan.

## 🗂️ Spesifikasi Data (Data Dictionary)

| Nama Fitur | Tipe Data | Deskripsi Teknis |
| :--- | :--- | :--- |
| **ID** | Integer | Identitas unik entitas bisnis. |
| **Monthly_Revenue** | Integer | Total omzet bulanan (IDR). Distribusi *Lognormal* (skewed). |
| **Net_Profit_Margin (%)** | Float | Rasio laba bersih setelah semua beban operasional. |
| **Burn_Rate_Ratio** | Float | Rasio pengeluaran vs pendapatan (> 1.0 berarti defisit). |
| **Transaction_Count** | Integer | Frekuensi transaksi unik dalam sebulan. |
| **Avg_Historical_Rating** | Float | Skor rata-rata kumulatif (1-5). |
| **Review_Text** | String | Ulasan pelanggan (Raw Text) untuk kebutuhan NLP. |
| **Review_Volatility** | Float | Indeks fluktuasi kualitas layanan/rating. |
| **Business_Tenure_Months** | Integer | Usia operasional bisnis dalam bulan. |
| **Repeat_Order_Rate (%)** | Float | Persentase pelanggan lama yang bertransaksi kembali. |
| **Digital_Adoption_Score** | Float | Skor kesiapan teknologi (Skala 1-10). |
| **Peak_Hour_Latency** | Categorical | Kecepatan layanan saat jam sibuk (Low/Med/High). |
| **Location_Competitiveness** | Integer | Jumlah kompetitor sejenis dalam radius 1 km. |
| **Sentiment_Score** | Float | Skor numerik hasil ekstraksi NLP (-1.0 s/d 1.0). |
| **Class** | Categorical | **Target Variabel:** Elite, Growth, Struggling, Critical. |
