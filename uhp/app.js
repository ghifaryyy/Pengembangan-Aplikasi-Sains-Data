/* ============================================================
   UMKM Health Predictor — app.js
   Prediction Engine + NLP + CSV Loader + Chart Renderer
   ============================================================ */

'use strict';

// ─── Global State ────────────────────────────────────────────
let allData = [];
let filteredData = [];
let currentPage = 1;
const PAGE_SIZE = 15;
let dataStats = { total: 0, Elite: 0, Growth: 0, Struggling: 0, Critical: 0 };

// ─── Indonesian Sentiment Lexicon ────────────────────────────
const POSITIVE_WORDS = [
  'lancar','bagus','baik','ramah','cepat','tepat','senang','puas','luar biasa',
  'konsisten','terjaga','mudah','responsif','komunikatif','repeat order','selalu',
  'terbaik','mantap','oke','cocok','suka','mau lagi','puas','recommended','rekomendasi',
  'kualitas','sesuai','memuaskan','excellent','great','good','fast','nice',
  'pengiriman cepat','admin komunikatif','proses checkout tidak ribet','mudah dipakai',
  'pemesanan mudah','pelayanan cepat','pesanan selalu tepat','seimbang'
];

const NEGATIVE_WORDS = [
  'lambat','buruk','jelek','kecewa','kurang','telat','terlambat','bermasalah',
  'tidak puas','tidak baik','naik harga','mahal','sering kosong','stok kosong',
  'lama','mengecewakan','menurun','rugi','susah','tidak responsif','tidak ada',
  'ribet','gagal','batal','tidak sesuai','protes','banyak masalah','kualitas buruk',
  'tidak konsisten','respons lambat','keterlambatan','komplain','tidak ramah',
  'harga naik','layanan tidak membaik','kosong','terlambat','proses bermasalah',
  'tidak oke','kurang memuaskan','tidak tepat','sering terlambat'
];

const STRONG_POSITIVE = ['sangat bagus','sangat puas','luar biasa','perfect','excellent','sempurna','terbaik'];
const STRONG_NEGATIVE = ['sangat buruk','sangat kecewa','sangat lambat','sangat bermasalah','terburuk','parah'];

// ─── Sentiment Engine ────────────────────────────────────────
function computeSentiment(text) {
  if (!text || text.trim() === '') return 0;
  const lower = text.toLowerCase();
  let score = 0;

  STRONG_POSITIVE.forEach(w => { if (lower.includes(w)) score += 0.65; });
  STRONG_NEGATIVE.forEach(w => { if (lower.includes(w)) score -= 0.65; });
  POSITIVE_WORDS.forEach(w => { if (lower.includes(w)) score += 0.25; });
  NEGATIVE_WORDS.forEach(w => { if (lower.includes(w)) score -= 0.25; });

  // Clamp to [-1, 1]
  return Math.max(-1, Math.min(1, score));
}

function sentimentLabel(score) {
  if (score >= 0.5) return 'Sangat Positif';
  if (score >= 0.2) return 'Positif';
  if (score >= -0.15) return 'Netral';
  if (score >= -0.4) return 'Negatif';
  return 'Sangat Negatif';
}

// ─── Prediction Engine (Rule-Based Ensemble Simulation) ──────
function predictClass(revenue, expenses, transactions, tenure, sentimentScore) {
  const netProfit = revenue - expenses;
  const npm = revenue > 0 ? ((netProfit / revenue) * 100) : 0;
  const burnRate = revenue > 0 ? (expenses / revenue) : 1.5;

  // Normalize features for confidence scoring
  const npmNorm = Math.max(-35, Math.min(45, npm));
  const burnNorm = Math.max(0.5, Math.min(1.5, burnRate));
  const sentNorm = sentimentScore;

  // --- Classification Rules (derived from dataset analysis) ---
  let predictedClass = '';
  let confidence = 0;

  // Elite: High profit, low burn rate, positive sentiment
  if (burnRate < 0.8 && npm >= 15 && netProfit > 0) {
    predictedClass = 'Elite';
    const burnScore = (0.8 - burnRate) / 0.3;   // 0-1
    const npmScore  = Math.min(1, (npm - 15) / 25);
    const sentScore = Math.max(0, (sentNorm + 1) / 2);
    confidence = 0.45 + (burnScore * 0.25) + (npmScore * 0.2) + (sentScore * 0.1);
  }
  // Critical: Very negative burn rate and large losses
  else if ((burnRate >= 1.15 && npm <= -18) || (burnRate >= 1.2 && sentNorm < -0.3)) {
    predictedClass = 'Critical';
    const burnScore = Math.min(1, (burnRate - 1.15) / 0.3);
    const lossScore = Math.min(1, Math.abs(npm + 18) / 17);
    const sentScore = Math.max(0, (-sentNorm + 0.3) / 1.3);
    confidence = 0.45 + (burnScore * 0.25) + (lossScore * 0.2) + (sentScore * 0.1);
  }
  // Struggling: Moderate to high burn, net loss
  else if (burnRate >= 1.0 || npm < 0) {
    predictedClass = 'Struggling';
    const burnScore = Math.min(1, (burnRate - 1.0) / 0.15);
    const lossScore = Math.min(1, Math.abs(Math.min(0, npm)) / 20);
    const sentScore = Math.max(0, (-sentNorm + 1) / 2);
    confidence = 0.40 + (burnScore * 0.2) + (lossScore * 0.2) + (sentScore * 0.1);
  }
  // Growth: Positive profit, stable burn rate
  else {
    predictedClass = 'Growth';
    const npmScore  = Math.min(1, npm / 15);
    const burnScore = Math.min(1, (1.0 - burnRate) / 0.2);
    const sentScore = Math.max(0, (sentNorm + 1) / 2);
    confidence = 0.38 + (npmScore * 0.25) + (burnScore * 0.2) + (sentScore * 0.1);
  }

  // Tenure boost (experienced businesses are more stable)
  if (tenure > 60) confidence = Math.min(0.98, confidence + 0.03);
  if (tenure < 6) confidence = Math.max(0.30, confidence - 0.05);

  // Clamp
  confidence = Math.max(0.30, Math.min(0.98, confidence));

  return { predictedClass, confidence, npm, burnRate, netProfit };
}

// ─── Feature Importance Weights ──────────────────────────────
function getFeatureImportance(predictedClass, sentimentScore, burnRate) {
  // Base weights from RF/XGBoost domain knowledge
  const base = [
    { name: 'Burn Rate Ratio',        value: 0.28 },
    { name: 'Net Profit Margin',      value: 0.24 },
    { name: 'Sentiment Score',        value: 0.18 },
    { name: 'Monthly Revenue',        value: 0.13 },
    { name: 'Transaction Count',      value: 0.09 },
    { name: 'Business Tenure',        value: 0.08 },
  ];

  // Dynamic adjustment based on prediction outcome
  if (predictedClass === 'Critical' || predictedClass === 'Struggling') {
    base[0].value = 0.31; // Burn rate more important
    base[2].value = 0.20; // Sentiment
  } else if (predictedClass === 'Elite') {
    base[3].value = 0.16; // Revenue higher
    base[2].value = 0.15;
  }
  if (Math.abs(sentimentScore) > 0.4) {
    base[2].value += 0.03; // Sentiment boost if strong
  }

  // Normalize to 100%
  const total = base.reduce((s, f) => s + f.value, 0);
  return base
    .map(f => ({ ...f, pct: Math.round((f.value / total) * 100) }))
    .sort((a, b) => b.value - a.value);
}

// ─── Insight Recommendations ─────────────────────────────────
const INSIGHTS = {
  Elite: {
    icon: '🚀',
    text: 'Bisnis sangat sehat! Arus kas positif dan sentimen pelanggan mendukung pertumbuhan berkelanjutan.',
    actions: [
      { tag: 'Ekspansi', color: 'var(--elite)', text: 'Fokus pada ekspansi pasar dan akuisisi segmen baru' },
      { tag: 'Loyalitas', color: 'var(--elite)', text: 'Tingkatkan program retensi dan loyalitas pelanggan' },
      { tag: 'Scaling', color: 'var(--elite)', text: 'Pertimbangkan diversifikasi produk atau buka gerai baru' },
    ],
  },
  Growth: {
    icon: '📈',
    text: 'Bisnis berkembang dengan baik. Profitabilitas positif namun efisiensi biaya masih bisa dioptimalkan.',
    actions: [
      { tag: 'Efisiensi', color: 'var(--growth)', text: 'Review biaya operasional dan negosiasi supplier untuk margin lebih baik' },
      { tag: 'Digital', color: 'var(--growth)', text: 'Tingkatkan adopsi digital untuk mendorong pertumbuhan transaksi' },
      { tag: 'Monitor', color: 'var(--growth)', text: 'Pantau burn rate agar tidak mendekati 1.0 secara konsisten' },
    ],
  },
  Struggling: {
    icon: '⚠️',
    text: 'Kondisi kritis pada arus kas. Pengeluaran melebihi atau mendekati pendapatan — perlu tindakan segera.',
    actions: [
      { tag: 'Harga', color: 'var(--struggling)', text: 'Tinjau ulang strategi penetapan harga untuk meningkatkan margin' },
      { tag: 'Burn Rate', color: 'var(--struggling)', text: 'Identifikasi dan kurangi pengeluaran yang tidak memberikan ROI' },
      { tag: 'Sentimen', color: 'var(--struggling)', text: 'Fokus perbaikan layanan pelanggan untuk memperbaiki ulasan' },
    ],
  },
  Critical: {
    icon: '🚨',
    text: 'Risiko tinggi! Bisnis mengalami kerugian signifikan. Diperlukan restrukturisasi segera untuk mencegah kolaps.',
    actions: [
      { tag: 'Darurat', color: 'var(--critical)', text: 'Audit menyeluruh semua pos pengeluaran dan potong yang tidak esensial' },
      { tag: 'Revenue', color: 'var(--critical)', text: 'Tingkatkan pendapatan: promo agresif, produk bundling, atau pivot model bisnis' },
      { tag: 'Konsultasi', color: 'var(--critical)', text: 'Pertimbangkan konsultasi finansial profesional atau akses program UMKM' },
    ],
  },
};

// ─── UI Updaters ─────────────────────────────────────────────
function updateSentimentPreview() {
  const text = document.getElementById('inputReview').value;
  const score = computeSentiment(text);
  const meter = document.getElementById('sentimentMeter');
  const preview = document.getElementById('sentimentPreview');

  // Map score [-1, 1] to position [5%, 95%]
  const pos = ((score + 1) / 2) * 90 + 5;
  meter.style.width = pos + '%';
  preview.textContent = score >= 0 ? '+' + score.toFixed(2) : score.toFixed(2);

  if (score >= 0.2) { preview.style.color = 'var(--elite)'; }
  else if (score <= -0.2) { preview.style.color = 'var(--critical)'; }
  else { preview.style.color = 'var(--text-muted)'; }
}

function updateSlider(type) {
  if (type === 'transaction') {
    document.getElementById('transactionVal').textContent = document.getElementById('inputTransactions').value;
  } else {
    document.getElementById('tenureVal').textContent = document.getElementById('inputTenure').value;
  }
}

function formatIDR(num) {
  if (Math.abs(num) >= 1_000_000) return 'Rp ' + (num / 1_000_000).toFixed(1) + ' Jt';
  if (Math.abs(num) >= 1_000) return 'Rp ' + (num / 1_000).toFixed(0) + ' Rb';
  return 'Rp ' + Math.round(num);
}

const CLASS_COLORS = {
  Elite: 'var(--elite)',
  Growth: 'var(--growth)',
  Struggling: 'var(--struggling)',
  Critical: 'var(--critical)',
};

const CLASS_GLOW = {
  Elite: 'var(--elite-glow)',
  Growth: 'var(--growth-glow)',
  Struggling: 'var(--struggling-glow)',
  Critical: 'var(--critical-glow)',
};

const FI_COLORS = [
  '#60a5fa', '#a78bfa', '#34d399', '#fb923c', '#f472b6', '#fbbf24'
];

function renderFeatureImportance(features) {
  const container = document.getElementById('featureChart');
  container.innerHTML = '';
  const maxPct = features[0].pct;

  features.forEach((f, i) => {
    const row = document.createElement('div');
    row.className = 'fi-row';
    row.innerHTML = `
      <span class="fi-label">${f.name}</span>
      <div class="fi-bar-track">
        <div class="fi-bar-fill" style="width:0%; background:${FI_COLORS[i]};" data-width="${(f.pct / maxPct * 100).toFixed(1)}"></div>
      </div>
      <span class="fi-pct">${f.pct}%</span>`;
    container.appendChild(row);
  });

  // Animate bars after a tick
  requestAnimationFrame(() => {
    container.querySelectorAll('.fi-bar-fill').forEach(bar => {
      setTimeout(() => { bar.style.width = bar.dataset.width + '%'; }, 100);
    });
  });
}

function renderInsights(cls) {
  const data = INSIGHTS[cls];
  document.getElementById('insightIconBig').textContent = data.icon;
  document.getElementById('insightText').textContent = data.text;

  const actEl = document.getElementById('insightActions');
  actEl.innerHTML = data.actions.map(a => `
    <div class="insight-action-item">
      <span class="tag" style="background:${a.color}20;color:${a.color};">${a.tag}</span>
      <span>${a.text}</span>
    </div>`).join('');

  // Color insights card
  const card = document.getElementById('insightsCard');
  card.style.borderColor = CLASS_COLORS[cls].replace('var(', '').replace(')', '');
  card.style.borderColor = `rgba(${cls === 'Elite' ? '0,230,118' : cls === 'Growth' ? '96,165,250' : cls === 'Struggling' ? '251,191,36' : '248,113,113'},0.3)`;
}

function setConfidenceRing(pct, cls) {
  const ring = document.getElementById('confidenceRing');
  const pctEl = document.getElementById('confidencePct');
  const circumference = 2 * Math.PI * 50; // r=50

  ring.style.stroke = CLASS_COLORS[cls];
  pctEl.style.color = CLASS_COLORS[cls];

  const offset = circumference * (1 - pct);
  ring.style.strokeDashoffset = offset;
  pctEl.textContent = Math.round(pct * 100) + '%';
}

// ─── Main Analysis Function ───────────────────────────────────
function analyzeUMKM() {
  const btn = document.getElementById('analyzeBtn');
  btn.classList.add('loading');

  setTimeout(() => {
    const revenue     = parseFloat(document.getElementById('inputRevenue').value) || 0;
    const expenses    = parseFloat(document.getElementById('inputExpenses').value) || 0;
    const transactions = parseInt(document.getElementById('inputTransactions').value) || 100;
    const tenure      = parseInt(document.getElementById('inputTenure').value) || 12;
    const review      = document.getElementById('inputReview').value;

    const sentimentScore = computeSentiment(review);
    const { predictedClass, confidence, npm, burnRate, netProfit } = predictClass(
      revenue, expenses, transactions, tenure, sentimentScore
    );

    const features = getFeatureImportance(predictedClass, sentimentScore, burnRate);
    const color = CLASS_COLORS[predictedClass];
    const glow  = CLASS_GLOW[predictedClass];

    // ── Show results ──
    document.getElementById('welcomeState').style.display = 'none';
    const resultsEl = document.getElementById('resultsPanel');
    resultsEl.style.display = 'flex';
    resultsEl.style.animation = 'none';
    void resultsEl.offsetWidth; // reflow
    resultsEl.style.animation = '';

    // ── Metric Cards ──
    // Net Profit
    const npEl = document.getElementById('metricNetProfit');
    npEl.textContent = formatIDR(netProfit);
    npEl.style.color = netProfit >= 0 ? 'var(--elite)' : 'var(--critical)';
    document.getElementById('metricNPM').textContent = `Margin: ${npm.toFixed(2)}%`;
    const trendNP = document.getElementById('trendNetProfit');
    trendNP.textContent = netProfit >= 0 ? '+' : '−';
    trendNP.style.background = netProfit >= 0 ? 'var(--elite-dim)' : 'var(--critical-dim)';
    trendNP.style.color = netProfit >= 0 ? 'var(--elite)' : 'var(--critical)';

    // Sentiment
    const sentEl = document.getElementById('metricSentiment');
    sentEl.textContent = (sentimentScore >= 0 ? '+' : '') + sentimentScore.toFixed(3);
    sentEl.style.color = sentimentScore >= 0.2 ? 'var(--elite)' : sentimentScore <= -0.2 ? 'var(--critical)' : 'var(--text-primary)';
    document.getElementById('metricSentimentLabel').textContent = sentimentLabel(sentimentScore);
    // Gauge
    const gauge = document.getElementById('sentimentGauge');
    gauge.innerHTML = '<div style="height:100%;border-radius:2px;transition:width 0.6s ease,background 0.3s"></div>';
    const gfill = gauge.firstChild;
    const gpos = ((sentimentScore + 1) / 2) * 100;
    const gcol = sentimentScore >= 0.2 ? 'var(--elite)' : sentimentScore <= -0.2 ? 'var(--critical)' : 'var(--struggling)';
    setTimeout(() => { gfill.style.width = gpos + '%'; gfill.style.background = gcol; }, 50);

    // Burn Rate
    const brEl = document.getElementById('metricBurnRate');
    brEl.textContent = burnRate.toFixed(3);
    const brColor = burnRate < 0.8 ? 'var(--elite)' : burnRate < 1.0 ? 'var(--growth)' : burnRate < 1.2 ? 'var(--struggling)' : 'var(--critical)';
    brEl.style.color = brColor;
    document.getElementById('metricBurnLabel').textContent =
      burnRate < 0.8 ? 'Sangat Efisien' : burnRate < 1.0 ? 'Efisien' : burnRate < 1.2 ? 'Risiko Sedang' : 'Risiko Tinggi';
    const brBarFill = document.getElementById('burnRateBar');
    brBarFill.style.background = brColor;
    const brPct = Math.min(100, (burnRate / 1.5) * 100);
    setTimeout(() => { brBarFill.style.width = brPct + '%'; }, 50);

    // ── Health Badge ──
    const badgeText = document.getElementById('badgeText');
    badgeText.textContent = predictedClass.toUpperCase();
    badgeText.style.color = color;

    const badgeCard = document.getElementById('healthBadge');
    badgeCard.style.setProperty('--badge-color', color);

    const hCard = document.querySelector('.health-badge-card');
    hCard.style.setProperty('--badge-color', color);
    hCard.style.setProperty('--badge-glow', glow);
    hCard.className = 'health-badge-card animated';

    document.getElementById('badgeDesc').textContent = INSIGHTS[predictedClass].icon + ' ' + INSIGHTS[predictedClass].text;

    // ── Confidence Ring ──
    setConfidenceRing(confidence, predictedClass);

    // ── Feature Importance ──
    renderFeatureImportance(features);

    // ── Insights ──
    renderInsights(predictedClass);

    // ── Metric card top-border animations ──
    document.querySelectorAll('.metric-card').forEach((c, i) => {
      c.style.setProperty('--metric-accent', [color, color, brColor][i] || color);
      c.classList.remove('updated');
      setTimeout(() => c.classList.add('updated'), 50);
    });

    btn.classList.remove('loading');
  }, 600);
}

// ─── Quick Fill Presets ───────────────────────────────────────
const PRESETS = {
  elite:     { rev: 20000000, exp: 13000000, trx: 160, tenure: 90,  review: 'Pelayanan cepat dan ramah, pesanan selalu tepat. Aplikasi pemesanan mudah dipakai dan responsif. Kualitas produk konsisten.' },
  growth:    { rev: 8000000,  exp: 6500000,  trx: 120, tenure: 36,  review: 'Harga dan kualitas seimbang, pengalaman biasa saja. Pelayanan standar, masih bisa ditingkatkan.' },
  struggling:{ rev: 5000000,  exp: 5400000,  trx: 80,  tenure: 18,  review: 'Kadang stok kosong saat jam ramai. Secara umum oke, hanya respon chat kadang lambat.' },
  critical:  { rev: 2000000,  exp: 3000000,  trx: 25,  tenure: 4,   review: 'Respons admin lambat dan informasi kurang jelas. Harga naik tapi layanan tidak membaik. Pesanan sering terlambat.' },
};

function quickFill(type) {
  const p = PRESETS[type];
  document.getElementById('inputRevenue').value     = p.rev;
  document.getElementById('inputExpenses').value    = p.exp;
  document.getElementById('inputTransactions').value = p.trx;
  document.getElementById('inputTenure').value      = p.tenure;
  document.getElementById('inputReview').value      = p.review;
  updateSlider('transaction');
  updateSlider('tenure');
  updateSentimentPreview();
}

// ─── Tab Switching ────────────────────────────────────────────
const facEl = () => document.getElementById('floatingAction');

function switchTab(tab) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));

  if (tab === 'predict') {
    document.getElementById('tabPredict').classList.add('active');
    document.getElementById('navPredict').classList.add('active');
    const fac = facEl();
    if (fac) fac.classList.remove('hidden');
  } else {
    document.getElementById('tabData').classList.add('active');
    document.getElementById('navData').classList.add('active');
    const fac = facEl();
    if (fac) fac.classList.add('hidden');
    // Only render if data is already loaded
    if (allData.length > 0) renderTable();
  }
}

// ─── CSV Loading & Data Explorer ─────────────────────────────

/**
 * Split one CSV line into fields, respecting quoted commas.
 * e.g. 'a,"b,c",d' → ['a', 'b,c', 'd']
 */
function splitCSVLine(line) {
  const fields = [];
  let cur = '';
  let inQ = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      if (inQ && line[i + 1] === '"') { cur += '"'; i++; } // escaped ""
      else inQ = !inQ;
    } else if (c === ',' && !inQ) {
      fields.push(cur); cur = '';
    } else {
      cur += c;
    }
  }
  fields.push(cur);
  return fields;
}

async function loadCSV() {
  const statusEl = document.getElementById('dataStatus');
  const dotEl    = document.querySelector('.status-dot');
  const tbodyEl  = document.getElementById('tableBody');

  const setMsg = (html, isError = false) => {
    if (tbodyEl) tbodyEl.innerHTML =
      `<tr><td colspan="9" class="table-loading" style="${isError ? 'color:var(--critical)' : ''}">${html}</td></tr>`;
  };

  statusEl.textContent = 'Memuat data...';
  setMsg('⏳ Memuat data...');

  try {
    // Load pre-generated JSON preview (251 KB) — much faster than 20 MB CSV
    const res = await fetch('UMKM-data/umkm_preview.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);

    const data = await res.json();
    if (!data || !data.rows || data.rows.length === 0) throw new Error('File JSON kosong atau format tidak valid.');

    // Populate stats from full-dataset stats embedded in JSON
    const s         = data.stats || {};
    dataStats.total      = s.total      || data.rows.length;
    dataStats.Elite      = s.Elite      || 0;
    dataStats.Growth     = s.Growth     || 0;
    dataStats.Struggling = s.Struggling || 0;
    dataStats.Critical   = s.Critical   || 0;

    // Map JSON fields to the shape renderTable() expects
    allData = data.rows.map(r => ({
      ID:                      r.ID,
      Monthly_Revenue:         r.Monthly_Revenue,
      'Net_Profit_Margin (%)': r.Net_Profit_Margin,
      Burn_Rate_Ratio:         r.Burn_Rate_Ratio,
      Transaction_Count:       r.Transaction_Count,
      Avg_Historical_Rating:   r.Avg_Historical_Rating,
      Sentiment_Score:         r.Sentiment_Score,
      Review_Text:             r.Review_Text,
      Class:                   r.Class,
    }));
    filteredData = [...allData];

    updateDataStats();
    statusEl.textContent = `${dataStats.total.toLocaleString()} data loaded`;
    dotEl.classList.add('ready');
    renderTable();

  } catch (err) {
    statusEl.textContent = 'Gagal memuat data';
    setMsg(
      `❌ <strong>Gagal memuat data</strong><br>
       <span style="font-size:11px;color:var(--text-muted)">${err.message}</span><br><br>
       <span style="font-size:11px;color:var(--text-muted)">
         Pastikan server berjalan:<br>
         <code style="background:var(--bg-elevated);padding:2px 6px;border-radius:4px;">python -m http.server 8080</code><br>
         lalu buka <code>http://localhost:8080</code>
       </span>`,
      true
    );
    console.error('[UHP] Data Error:', err);
  }
}

function updateDataStats() {
  const t = dataStats.total || 1;

  // Welcome state
  document.getElementById('wsTotal').textContent     = dataStats.total.toLocaleString();
  document.getElementById('wsElite').textContent     = dataStats.Elite.toLocaleString();
  document.getElementById('wsGrowth').textContent    = dataStats.Growth.toLocaleString();
  document.getElementById('wsStruggling').textContent= dataStats.Struggling.toLocaleString();
  document.getElementById('wsCritical').textContent  = dataStats.Critical.toLocaleString();

  // Explorer stats
  document.getElementById('expElite').textContent     = dataStats.Elite.toLocaleString();
  document.getElementById('expGrowth').textContent    = dataStats.Growth.toLocaleString();
  document.getElementById('expStruggling').textContent= dataStats.Struggling.toLocaleString();
  document.getElementById('expCritical').textContent  = dataStats.Critical.toLocaleString();

  // Bars
  const setBar = (id, cls) => {
    const el = document.getElementById(id);
    if (el) el.style.width = ((dataStats[cls] / t) * 100).toFixed(1) + '%';
  };
  setBar('expEliteBar', 'Elite');
  setBar('expGrowthBar', 'Growth');
  setBar('expStrugglingBar', 'Struggling');
  setBar('expCriticalBar', 'Critical');
}

function filterData() {
  const query = (document.getElementById('searchInput').value || '').toLowerCase();
  const classFilter = document.getElementById('classFilter').value;

  filteredData = allData.filter(row => {
    const matchClass = !classFilter || row.Class === classFilter;
    const matchText  = !query || (
      (row.Review_Text || '').toLowerCase().includes(query) ||
      (row.Class || '').toLowerCase().includes(query) ||
      String(row.ID || '').includes(query)
    );
    return matchClass && matchText;
  });

  currentPage = 1;
  document.getElementById('filterCount').textContent =
    `${filteredData.length.toLocaleString()} hasil`;
  renderTable();
}

function renderTable() {
  const tbody = document.getElementById('tableBody');
  const start = (currentPage - 1) * PAGE_SIZE;
  const pageData = filteredData.slice(start, start + PAGE_SIZE);
  const total = filteredData.length;
  const totalPages = Math.ceil(total / PAGE_SIZE);

  if (pageData.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="table-loading">Tidak ada data ditemukan.</td></tr>';
  } else {
    tbody.innerHTML = pageData.map(row => `
      <tr>
        <td style="color:var(--text-muted)">${row.ID}</td>
        <td>${formatIDR(row.Monthly_Revenue)}</td>
        <td style="color:${(row['Net_Profit_Margin (%)'] || 0) >= 0 ? 'var(--elite)' : 'var(--critical)'}">
          ${Number(row['Net_Profit_Margin (%)']).toFixed(1)}%
        </td>
        <td style="color:${(row.Burn_Rate_Ratio || 1) < 1 ? 'var(--growth)' : 'var(--critical)'}">
          ${Number(row.Burn_Rate_Ratio).toFixed(3)}
        </td>
        <td>${row.Transaction_Count}</td>
        <td>${Number(row.Avg_Historical_Rating).toFixed(2)}</td>
        <td style="color:${(row.Sentiment_Score || 0) >= 0 ? 'var(--elite)' : 'var(--critical)'}">
          ${(row.Sentiment_Score >= 0 ? '+' : '') + Number(row.Sentiment_Score).toFixed(2)}
        </td>
        <td>
          <span class="class-badge class-${row.Class.toLowerCase()}">${row.Class}</span>
        </td>
        <td class="review-cell" title="${(row.Review_Text || '').replace(/"/g,'&quot;')}">
          ${row.Review_Text || '—'}
        </td>
      </tr>`).join('');
  }

  // Pagination
  document.getElementById('pageInfo').textContent = `Hal. ${currentPage} / ${totalPages || 1} (${total.toLocaleString()} data)`;
  document.getElementById('prevBtn').disabled = currentPage <= 1;
  document.getElementById('nextBtn').disabled = currentPage >= totalPages;
}

function changePage(dir) {
  const totalPages = Math.ceil(filteredData.length / PAGE_SIZE);
  currentPage = Math.max(1, Math.min(totalPages, currentPage + dir));
  renderTable();
  document.querySelector('.table-wrapper').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── Real-time sentiment preview ─────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const reviewEl = document.getElementById('inputReview');
  if (reviewEl) {
    reviewEl.addEventListener('input', updateSentimentPreview);
    updateSentimentPreview(); // Initial
  }
  updateSlider('transaction');
  updateSlider('tenure');

  // Load CSV data
  loadCSV();
});
