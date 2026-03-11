"""
SCTR Symbol list with DG / Loss% from last peak (Close > 12D EMA).
Pop-up chart per symbol.

Data source: Google Sheet (pre-populated by fetch_to_sheet.py).
  • Loaded once at startup → zero API calls during normal operation
  • Re-run fetch_to_sheet.py each day to refresh prices
"""

import io
import threading
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from flask import Flask, jsonify, render_template_string, Response, request, send_from_directory

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SHEET_ID  = "1kTA8Xy5vCRxMaLhlXLxrKQ_j8VYtDf_7vMGI-R-yUJs"
GCP_KEY   = Path(__file__).parent / "gcp_key.json"

_GCP_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def _gcp_creds():
    """Load GCP credentials from file (local) or GCP_KEY_JSON env var (Railway)."""
    import os, json as _json
    env_json = os.environ.get("GCP_KEY_JSON")
    if env_json:
        info = _json.loads(env_json)
        return Credentials.from_service_account_info(info, scopes=_GCP_SCOPES)
    return Credentials.from_service_account_file(str(GCP_KEY), scopes=_GCP_SCOPES)
CSV_PATH  = Path(__file__).parent / "stockcharts_SCTR.csv"

# ── Load symbol list from sheet header (source of truth) ─────────────────────
def load_symbols() -> list:
    try:
        import warnings; warnings.filterwarnings('ignore')
        gc = gspread.authorize(_gcp_creds())
        ws = gc.open_by_key(SHEET_ID).sheet1
        header = ws.row_values(1)
        syms = [h for h in header if h and h != "Date"]
        print(f"[symbols] loaded {len(syms)} symbols from sheet header")
        return syms
    except Exception as e:
        print(f"[symbols] sheet read failed ({e}), falling back to CSV top-100")
        df = pd.read_csv(CSV_PATH)
        return df.iloc[:100, 0].tolist()

SYMBOLS: list = load_symbols()

# ── In-memory cache ───────────────────────────────────────────────────────────
_ohlcv_cache: dict = {}   # symbol -> pd.DataFrame (60 rows, Close column)
_stats_cache: dict = {}
_cache_lock = threading.Lock()
_prefetch_done = threading.Event()

# ── Google Sheets loader ──────────────────────────────────────────────────────
def _load_from_sheet() -> pd.DataFrame:
    """Read the sheet and return a wide DataFrame: index=Date, cols=symbols."""
    gc = gspread.authorize(_gcp_creds())
    ws = gc.open_by_key(SHEET_ID).sheet1
    rows = ws.get_all_values()          # list of lists
    if not rows:
        raise ValueError("Sheet is empty")
    header = rows[0]                    # ["Date", "SNDK", "LITE", …]
    data   = rows[1:]
    df = pd.DataFrame(data, columns=header)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df                           # wide: index=dates, cols=symbols

def _build_ohlcv(close_series: pd.Series) -> pd.DataFrame:
    """Wrap a Close series into a minimal OHLCV DataFrame."""
    return pd.DataFrame({"Close": close_series}).dropna()

def _prefetch_all():
    """Load all close prices from Google Sheet and populate caches."""
    print("[prefetch] loading prices from Google Sheet …")
    try:
        wide = _load_from_sheet()
        ok = 0
        for sym in SYMBOLS:
            if sym not in wide.columns:
                with _cache_lock:
                    _stats_cache[sym] = {"dg": None, "loss_pct": None,
                                         "peak_date": None, "peak_price": None,
                                         "latest_price": None,
                                         "error": "not in sheet"}
                continue
            df = _build_ohlcv(wide[sym])
            stats = _compute_stats(sym, df)
            with _cache_lock:
                _ohlcv_cache[sym] = df
                _stats_cache[sym] = stats
            ok += 1
        print(f"[prefetch] complete — {ok}/{len(SYMBOLS)} symbols loaded")
    except Exception as e:
        print(f"[prefetch] ERROR: {e}")
    _prefetch_done.set()

# ── Analytics ─────────────────────────────────────────────────────────────────
def _compute_stats(symbol: str, df: pd.DataFrame) -> dict:
    try:
        close = df["Close"]
        ema12 = close.ewm(span=12, adjust=False).mean()
        above = (close > ema12).values

        i = len(above) - 1
        while i >= 0 and not above[i]:
            i -= 1
        if i < 0:
            return {"dg": None, "loss_pct": None, "peak_date": None,
                    "peak_price": None, "latest_price": round(float(close.iloc[-1]), 2),
                    "g1d": None, "g3d": None, "g5d": None, "g20d": None}
        j = i
        while j >= 0 and above[j]:
            j -= 1

        seg = close.iloc[j + 1: i + 1]
        peak_idx = seg.idxmax()
        peak_price = float(seg[peak_idx])
        latest = float(close.iloc[-1])
        dg = int(len(close) - 1 - close.index.get_loc(peak_idx))
        loss = round((latest - peak_price) / peak_price * 100, 1)

        def _gain(n):
            if len(close) < n + 1:
                return None
            prev = float(close.iloc[-(n + 1)])
            if prev == 0:
                return None
            return round((latest - prev) / prev * 100, 2)

        return {
            "dg": dg,
            "loss_pct": loss,
            "peak_date": peak_idx.strftime("%Y-%m-%d"),
            "peak_price": round(peak_price, 2),
            "latest_price": round(latest, 2),
            "g1d": _gain(1),
            "g3d": _gain(3),
            "g5d": _gain(5),
            "g20d": _gain(20),
        }
    except Exception as e:
        return {"dg": None, "loss_pct": None, "peak_date": None,
                "peak_price": None, "latest_price": None,
                "g1d": None, "g3d": None, "g5d": None, "g20d": None,
                "error": str(e)}

# Kick off prefetch on import (non-blocking)
threading.Thread(target=_prefetch_all, daemon=True).start()

# ── Chart builder (uses cache) ────────────────────────────────────────────────
def build_chart_png(symbol: str) -> bytes:
    with _cache_lock:
        df = _ohlcv_cache.get(symbol)
    if df is None:
        raise ValueError(f"{symbol} not in cache — re-run fetch_to_sheet.py")

    close = df["Close"]
    ema5  = close.ewm(span=5,  adjust=False).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    def xover(c, e):
        cross = (c.shift(1) >= e.shift(1)) & (c < e)
        xd, yp = [], []
        for i in np.where(cross)[0]:
            if i == 0: continue
            c0,c1 = c.iloc[i-1], c.iloc[i]
            e0,e1 = e.iloc[i-1], e.iloc[i]
            denom = (c1-c0)-(e1-e0)
            t = 0.5 if abs(denom)<1e-12 else max(0,min(1,(e0-c0)/denom))
            d0 = mdates.date2num(df.index[i-1])
            d1_ = mdates.date2num(df.index[i])
            xd.append(d0+t*(d1_-d0)); yp.append(c0+t*(c1-c0))
        return xd, yp

    x12, y12 = xover(close, ema12)

    above = (close > ema12).values
    pk_d, pk_p = [], []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]: j += 1
            seg = close.iloc[i:j]; pid = seg.idxmax()
            pk_d.append(pid); pk_p.append(seg[pid]); i = j
        else:
            i += 1

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.fill_between(df.index, close, ema12, where=(close > ema12),
                    interpolate=True, color="#00c896", alpha=0.2)
    if pk_d:
        ax.scatter(pk_d, pk_p, color="#ff4b4b", marker="^", s=80, zorder=6,
                   edgecolors="white", linewidths=0.8)
    ax.plot(df.index, close, color="#00bfff", linewidth=1.8, label="Close")
    ax.plot(df.index, ema5,  color="#f5a623", linewidth=1.2, label="EMA5",  linestyle="--")
    ax.plot(df.index, ema12, color="#ff7f0e", linewidth=1.2, label="EMA12")
    ax.plot(df.index, ema26, color="#2ca02c", linewidth=1.2, label="EMA26")
    if x12:
        ax.scatter(x12, y12, color="#ff7f0e", marker="v", s=60, zorder=5,
                   edgecolors="white", linewidths=0.8)
    ax.set_title(f"{symbol}  –  Last 60 Trading Days", color="white", fontsize=11, pad=8)
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=35)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.grid(True, color="#222", linewidth=0.5)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.4,
              labelcolor="white", facecolor="#111")
    plt.tight_layout(pad=1.2)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(); plt.style.use("default")
    buf.seek(0)
    return buf.getvalue()

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="icon" type="image/png" sizes="192x192" href="/favicon-192.png"/>
<link rel="icon" type="image/png" sizes="32x32" href="/favicon.png"/>
<link rel="apple-touch-icon" href="/apple-touch-icon.png"/>
<link rel="shortcut icon" href="/favicon.ico"/>
<title>Priceslides</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',sans-serif;font-size:14px}
  h1{padding:1rem 1.2rem .4rem;font-size:1.1rem;color:#58a6ff;letter-spacing:.5px}
  .subtitle{padding:0 1.2rem .8rem;color:#8b949e;font-size:.78rem}
  #status{padding:.3rem 1.2rem .6rem;font-size:.75rem;color:#e3b341}
  .topbar{display:flex;align-items:center;justify-content:space-between;padding:.6rem 1.2rem .2rem}
  .topbar h1{padding:0}
  .btn-orange{
    display:inline-flex;align-items:center;gap:.4rem;
    background:#ff9500;color:#000;font-weight:700;font-size:.8rem;
    border:none;border-radius:6px;padding:.4rem .9rem;cursor:pointer;
    text-decoration:none;letter-spacing:.3px;transition:background .15s;
  }
  .btn-orange:hover{background:#ffb340}
  .btn-action{
    display:inline-flex;align-items:center;gap:.4rem;
    background:#21262d;color:#e6edf3;font-weight:600;font-size:.8rem;
    border:1px solid #30363d;border-radius:6px;padding:.4rem .9rem;cursor:pointer;
    text-decoration:none;transition:background .15s;margin-left:.4rem;
  }
  .btn-action:hover{background:#30363d}
  .btn-add{background:#238636;border-color:#238636;color:#fff}
  .btn-add:hover{background:#2ea043}
  .btn-del{background:none;border:none;color:#6e3030;font-size:.85rem;cursor:pointer;padding:.15rem .4rem;border-radius:4px}
  .btn-del:hover{background:#3d1a1a;color:#f85149}
  td.del-col{text-align:center;padding:.3rem .4rem}
  .add-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:200;justify-content:center;align-items:center}
  .add-overlay.show{display:flex}
  .add-box{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:1.4rem 1.6rem;width:min(90vw,340px);box-shadow:0 8px 40px rgba(0,0,0,.7)}
  .add-box h3{color:#58a6ff;margin-bottom:.8rem;font-size:.95rem}
  .add-box input{width:100%;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#e6edf3;padding:.5rem .7rem;font-size:.95rem;margin-bottom:.8rem;outline:none}
  .add-box input:focus{border-color:#58a6ff}
  .add-box-btns{display:flex;gap:.6rem;justify-content:flex-end}
  .add-msg{font-size:.78rem;margin-bottom:.6rem;min-height:1rem}

  .wrap{overflow-x:auto;height:calc(100vh - 90px);overflow-y:auto}
  table{width:100%;border-collapse:collapse}
  thead th{
    background:#161b22;color:#8b949e;font-size:.72rem;text-transform:uppercase;
    letter-spacing:.6px;padding:.55rem .8rem;border-bottom:1px solid #30363d;
    position:sticky;top:0;z-index:2;white-space:nowrap;
    user-select:none;cursor:pointer;
  }
  thead th:hover{background:#1c2128;color:#e6edf3}
  .sort-icon{
    display:inline-flex;flex-direction:column;margin-left:4px;
    vertical-align:middle;line-height:1;gap:1px;
  }
  .sort-icon span{font-size:.55rem;color:#444;line-height:1}
  thead th:hover .sort-icon span{color:#666}
  thead th.sort-asc  .sort-icon .asc  {color:#58a6ff}
  thead th.sort-desc .sort-icon .desc {color:#58a6ff}
  tbody tr{border-bottom:1px solid #21262d;cursor:pointer;transition:background .15s}
  tbody tr:hover{background:#1c2128}
  td{padding:.55rem .8rem}
  td.sym{font-weight:600;color:#58a6ff;font-size:.9rem}
  td.sym.orange{color:#ff9500;font-weight:700}
  tr.orange-row td.sym{color:#ff9500}
  tr.orange-row{border-left:3px solid #ff9500}
  td.num{text-align:right;font-variant-numeric:tabular-nums}
  td.loss-pos{color:#3fb950} td.loss-neg{color:#f85149}
  td.dg{color:#e3b341}
  .loading{color:#555;font-size:.8rem}
  .err{color:#6e3030;font-size:.8rem}

  .overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);
            z-index:100;justify-content:center;align-items:center}
  .overlay.show{display:flex}
  .modal{background:#161b22;border:1px solid #30363d;border-radius:10px;
         width:min(92vw,680px);max-height:92vh;overflow:hidden;
         box-shadow:0 8px 40px rgba(0,0,0,.7)}
  .modal-header{display:flex;justify-content:space-between;align-items:center;
    padding:.7rem 1rem;border-bottom:1px solid #30363d;background:#0d1117}
  .modal-ticker{font-size:1.1rem;font-weight:700;color:#58a6ff}
  .modal-close{background:none;border:none;color:#8b949e;font-size:1.3rem;
               cursor:pointer;padding:.2rem .5rem;border-radius:4px}
  .modal-close:hover{background:#21262d;color:#fff}
  .modal-stats{display:flex;gap:1.2rem;padding:.6rem 1rem;background:#0d1117;
    border-bottom:1px solid #21262d;flex-wrap:wrap}
  .stat-box{display:flex;flex-direction:column;gap:.1rem}
  .stat-label{font-size:.65rem;color:#8b949e;text-transform:uppercase;letter-spacing:.5px}
  .stat-val{font-size:.95rem;font-weight:600}
  .stat-val.green{color:#3fb950} .stat-val.red{color:#f85149} .stat-val.yellow{color:#e3b341}
  .modal-chart{padding:.8rem;text-align:center;min-height:220px;
               display:flex;align-items:center;justify-content:center}
  .modal-chart img{max-width:100%;border-radius:6px}
  .modal-footer{padding:.5rem 1rem;text-align:right;
                border-top:1px solid #21262d;background:#0d1117}
  .btn-ext{display:inline-block;padding:.35rem .8rem;margin-left:.5rem;
    border:1px solid #30363d;border-radius:5px;color:#58a6ff;font-size:.8rem;
    text-decoration:none;transition:background .15s}
  .btn-ext:hover{background:#1c2128}
  .spinner{border:3px solid #30363d;border-top-color:#58a6ff;border-radius:50%;
           width:36px;height:36px;animation:spin .8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="topbar">
  <h1>📈 Priceslides</h1>
  <div style="display:flex;align-items:center;gap:.4rem">
    <button class="btn-action" id="btn-refresh" onclick="doRefresh()">🔄 Refresh</button>
    <button class="btn-action btn-add" onclick="openAdd()">＋ Add</button>
    <a class="btn-orange" href="/orange">🟠 ORANGE</a>
  </div>
</div>
<p class="subtitle">Tap a symbol to view chart &nbsp;|&nbsp; DG = trading days since last peak (Close&nbsp;&gt;&nbsp;12D EMA) &nbsp;|&nbsp; Click any column header to sort</p>
<div id="status">⏳ Loading data…</div>

<div class="wrap">
<table id="main-table">
  <thead><tr>
    <th data-col="idx"   data-type="num"  >#<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="sym"   data-type="str"  >Symbol<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="dg"    data-type="num"  class="num">DG<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="loss"  data-type="num"  class="num">Loss %<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="pdate" data-type="str"  class="num">Last Peak<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="pprice"data-type="num"  class="num">Peak $<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="now"   data-type="num"  class="num">Now $<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="g1d"   data-type="num"  class="num">1D %<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="g3d"   data-type="num"  class="num">3D %<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="g5d"   data-type="num"  class="num">5D %<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="g20d"  data-type="num"  class="num">20D %<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th style="width:2rem"></th>
  </tr></thead>
  <tbody id="tbody"></tbody>
</table>
</div>

<!-- Modal -->
<div class="overlay" id="overlay" onclick="closeOnBg(event)">
  <div class="modal">
    <div class="modal-header">
      <span class="modal-ticker" id="m-ticker">—</span>
      <button class="modal-close" onclick="closeModal()">✕</button>
    </div>
    <div class="modal-stats">
      <div class="stat-box"><span class="stat-label">Days Gone (DG)</span>
        <span class="stat-val yellow" id="m-dg">—</span></div>
      <div class="stat-box"><span class="stat-label">Loss %</span>
        <span class="stat-val" id="m-loss">—</span></div>
      <div class="stat-box"><span class="stat-label">Last Peak</span>
        <span class="stat-val" id="m-pdate">—</span></div>
      <div class="stat-box"><span class="stat-label">Peak $</span>
        <span class="stat-val" id="m-pprice">—</span></div>
      <div class="stat-box"><span class="stat-label">Now $</span>
        <span class="stat-val green" id="m-now">—</span></div>
    </div>
    <div class="modal-chart" id="m-chart"><div class="spinner"></div></div>
    <div class="modal-footer">
      <a class="btn-ext" id="btn-cnbc" href="#" target="_blank">📺 CNBC</a>
      <a class="btn-ext" id="btn-sc"   href="#" target="_blank">📈 StockCharts</a>
    </div>
  </div>
</div>

<script>
const symbols = {{ symbols|tojson }};
const orangeSyms = new Set({{ orange|tojson }});
let cachedStats = {};
// rowData: array of objects, one per symbol, with sortable values
let rowData = symbols.map((sym, i) => ({
  idx: i + 1, sym,
  dg: null, loss: null, pdate: null, pprice: null, now: null,
  g1d: null, g3d: null, g5d: null, g20d: null,
  raw: null   // full stats dict
}));

let sortCol = null, sortAsc = true;

// ── Render tbody from rowData ─────────────────────────────────────────────────
function renderTable() {
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = '';
  rowData.forEach((r, i) => {
    const d = r.raw;
    const hasData = d && d.dg != null;
    const lv = hasData ? d.loss_pct : null;
    const isOrange = orangeSyms.has(r.sym);
    const row = document.createElement('tr');
    if (isOrange) row.classList.add('orange-row');
    row.innerHTML = `
      <td style="color:#8b949e;font-size:.75rem">${r.idx}</td>
      <td class="sym${isOrange?' orange':''}" style="cursor:pointer" onclick="openModal('${r.sym}')">${r.sym}</td>
      <td class="num dg${hasData?'':' loading'}" onclick="openModal('${r.sym}')" style="cursor:pointer">${hasData ? d.dg : (d&&d.error?'—':'…')}</td>
      <td class="num ${hasData?(lv>=0?'loss-pos':'loss-neg'):(d&&d.error?'err':'loading')}" onclick="openModal('${r.sym}')" style="cursor:pointer">
        ${hasData ? (lv>0?'+':'')+ lv +'%' : (d&&d.error?'—':'…')}</td>
      <td class="num" style="color:#8b949e;font-size:.8rem;cursor:pointer" onclick="openModal('${r.sym}')">${hasData&&d.peak_date?d.peak_date:'—'}</td>
      <td class="num" style="color:#8b949e;font-size:.8rem;cursor:pointer" onclick="openModal('${r.sym}')">${hasData&&d.peak_price?'$'+d.peak_price:'—'}</td>
      <td class="num" style="color:#8b949e;font-size:.8rem;cursor:pointer" onclick="openModal('${r.sym}')">${hasData&&d.latest_price?'$'+d.latest_price:'—'}</td>
      <td class="num ${hasData&&d.g1d!=null?(d.g1d>=0?'loss-pos':'loss-neg'):'loading'}" onclick="openModal('${r.sym}')" style="cursor:pointer">${hasData&&d.g1d!=null?(d.g1d>0?'+':'')+d.g1d+'%':'—'}</td>
      <td class="num ${hasData&&d.g3d!=null?(d.g3d>=0?'loss-pos':'loss-neg'):'loading'}" onclick="openModal('${r.sym}')" style="cursor:pointer">${hasData&&d.g3d!=null?(d.g3d>0?'+':'')+d.g3d+'%':'—'}</td>
      <td class="num ${hasData&&d.g5d!=null?(d.g5d>=0?'loss-pos':'loss-neg'):'loading'}" onclick="openModal('${r.sym}')" style="cursor:pointer">${hasData&&d.g5d!=null?(d.g5d>0?'+':'')+d.g5d+'%':'—'}</td>
      <td class="num ${hasData&&d.g20d!=null?(d.g20d>=0?'loss-pos':'loss-neg'):'loading'}" onclick="openModal('${r.sym}')" style="cursor:pointer">${hasData&&d.g20d!=null?(d.g20d>0?'+':'')+d.g20d+'%':'—'}</td>
      <td class="del-col"><button class="btn-del" onclick="doDelete('${r.sym}')" title="Delete">✕</button></td>`;
    tbody.appendChild(row);
  });
}

// ── Sorting ───────────────────────────────────────────────────────────────────
const COL_KEY = { idx:'idx', sym:'sym', dg:'dg', loss:'loss',
                  pdate:'pdate', pprice:'pprice', now:'now',
                  g1d:'g1d', g3d:'g3d', g5d:'g5d', g20d:'g20d' };

function sortBy(col, type) {
  if (sortCol === col) { sortAsc = !sortAsc; }
  else { sortCol = col; sortAsc = true; }
  document.querySelectorAll('thead th').forEach(th => {
    th.classList.remove('sort-asc','sort-desc');
    if (th.dataset.col === col)
      th.classList.add(sortAsc ? 'sort-asc' : 'sort-desc');
  });
  const key = COL_KEY[col];
  rowData.sort((a, b) => {
    let av = a[key], bv = b[key];
    if (av === null && bv === null) return 0;
    if (av === null) return 1;
    if (bv === null) return -1;
    let cmp = (type === 'str') ? av.localeCompare(bv) : av - bv;
    return sortAsc ? cmp : -cmp;
  });
  renderTable();
}
document.querySelectorAll('thead th').forEach(th => {
  if (th.dataset.col) th.addEventListener('click', () => sortBy(th.dataset.col, th.dataset.type));
});

// ── Load data ─────────────────────────────────────────────────────────────────
async function loadAll() {
  const status = document.getElementById('status');
  try {
    const r = await fetch('/api/all');
    const data = await r.json();
    cachedStats = data;
    let loaded = 0, errors = 0;
    rowData.forEach(r => {
      const d = data[r.sym] || { error: 'no data' };
      r.raw=d; r.dg=d.dg??null; r.loss=d.loss_pct??null;
      r.pdate=d.peak_date??null; r.pprice=d.peak_price??null; r.now=d.latest_price??null;
      r.g1d=d.g1d??null; r.g3d=d.g3d??null; r.g5d=d.g5d??null; r.g20d=d.g20d??null;
      if (d.dg != null) loaded++; else errors++;
    });
    renderTable();
    status.textContent = `✅ ${loaded} loaded, ${errors} unavailable`;
    setTimeout(() => { status.style.display = 'none'; }, 4000);
  } catch(e) { status.textContent = `⚠️ Load failed: ${e}`; }
}
renderTable();
loadAll();

// ── Refresh (latest close only — 1 credit per symbol) ─────────────────────────
async function doRefresh() {
  const btn = document.getElementById('btn-refresh');
  const status = document.getElementById('status');
  btn.disabled = true; btn.textContent = '⏳ Refreshing…';
  status.style.display = 'block';
  status.textContent = `⏳ Refreshing ${rowData.length} symbols (1 credit each)…`;
  try {
    const r = await fetch('/api/refresh', {method:'POST'});
    const data = await r.json();
    // reload stats
    const r2 = await fetch('/api/all');
    const stats = await r2.json();
    cachedStats = stats;
    rowData.forEach(r => {
      const d = stats[r.sym]||{}; r.raw=d;
      r.dg=d.dg??null; r.loss=d.loss_pct??null;
      r.pdate=d.peak_date??null; r.pprice=d.peak_price??null; r.now=d.latest_price??null;
      r.g1d=d.g1d??null; r.g3d=d.g3d??null; r.g5d=d.g5d??null; r.g20d=d.g20d??null;
    });
    renderTable();
    status.textContent = `✅ Refreshed ${data.updated.length} symbols, ${data.errors.length} errors`;
    setTimeout(()=>{ status.style.display='none'; }, 5000);
  } catch(e) { status.textContent = `⚠️ Refresh failed: ${e}`; }
  btn.disabled = false; btn.textContent = '🔄 Refresh';
}

// ── Delete ────────────────────────────────────────────────────────────────────
async function doDelete(sym) {
  if (!confirm(`Delete ${sym}? This removes it from the sheet and web app.`)) return;
  const status = document.getElementById('status');
  status.style.display='block'; status.textContent=`⏳ Deleting ${sym}…`;
  try {
    const r = await fetch(`/api/delete/${encodeURIComponent(sym)}`, {method:'POST'});
    const data = await r.json();
    if (!data.ok) throw new Error(data.error);
    rowData = rowData.filter(r => r.sym !== sym);
    rowData.forEach((r,i) => r.idx = i+1);
    renderTable();
    status.textContent = `✅ ${sym} deleted`;
    setTimeout(()=>{ status.style.display='none'; }, 3000);
  } catch(e) { status.textContent = `⚠️ Delete failed: ${e}`; }
}

// ── Add symbol ────────────────────────────────────────────────────────────────
function openAdd() {
  document.getElementById('add-input').value='';
  document.getElementById('add-msg').textContent='';
  document.getElementById('add-overlay').classList.add('show');
  document.getElementById('add-input').focus();
}
function closeAdd() { document.getElementById('add-overlay').classList.remove('show'); }

async function submitAdd() {
  const sym = document.getElementById('add-input').value.trim().toUpperCase();
  const msg = document.getElementById('add-msg');
  if (!sym) { msg.style.color='#f85149'; msg.textContent='Enter a symbol.'; return; }
  msg.style.color='#e3b341'; msg.textContent=`⏳ Fetching ${sym} (~60 credits)…`;
  document.getElementById('add-submit').disabled = true;
  try {
    const r = await fetch('/api/add', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({symbol:sym, mark_orange:false})
    });
    const data = await r.json();
    if (!data.ok) throw new Error(data.error);
    if (!rowData.find(r=>r.sym===sym)) {
      rowData.push({idx:rowData.length+1, sym, dg:null,loss:null,pdate:null,pprice:null,now:null,g1d:null,g3d:null,g5d:null,g20d:null,raw:null});
    }
    // reload stats for new symbol
    const r2 = await fetch(`/api/stats/${encodeURIComponent(sym)}`);
    const d = await r2.json();
    cachedStats[sym]=d;
    const rd = rowData.find(r=>r.sym===sym);
    if (rd) { rd.raw=d; rd.dg=d.dg??null; rd.loss=d.loss_pct??null; rd.pdate=d.peak_date??null; rd.pprice=d.peak_price??null; rd.now=d.latest_price??null; rd.g1d=d.g1d??null; rd.g3d=d.g3d??null; rd.g5d=d.g5d??null; rd.g20d=d.g20d??null; }
    renderTable();
    msg.style.color='#3fb950'; msg.textContent=`✅ ${sym} added (${data.rows||60} rows)`;
    setTimeout(closeAdd, 1500);
  } catch(e) { msg.style.color='#f85149'; msg.textContent=`⚠️ ${e}`; }
  document.getElementById('add-submit').disabled = false;
}

// ── Modal ─────────────────────────────────────────────────────────────────────
async function openModal(sym) {
  document.getElementById('m-ticker').textContent = sym;
  document.getElementById('btn-cnbc').href = `https://www.cnbc.com/quotes/${sym}`;
  document.getElementById('btn-sc').href   = `https://stockcharts.com/h-sc/ui?s=${sym}`;

  const d = cachedStats[sym];
  if (d) fillModalStats(d);
  else {
    clearModalStats();
    fetch(`/api/stats/${encodeURIComponent(sym)}`)
      .then(r=>r.json()).then(d=>{ cachedStats[sym]=d; fillModalStats(d); }).catch(()=>{});
  }

  document.getElementById('m-chart').innerHTML = '<div class="spinner"></div>';
  document.getElementById('overlay').classList.add('show');

  try {
    const r2 = await fetch(`/api/chart/${encodeURIComponent(sym)}`);
    const blob = await r2.blob();
    document.getElementById('m-chart').innerHTML =
      `<img src="${URL.createObjectURL(blob)}" alt="${sym}"/>`;
  } catch(e) {
    document.getElementById('m-chart').innerHTML =
      `<p style="color:#f85149">Chart unavailable</p>`;
  }
}

function fillModalStats(d) {
  document.getElementById('m-dg').textContent = d.dg ?? '—';
  const lv = d.loss_pct;
  const lEl = document.getElementById('m-loss');
  lEl.textContent = lv!=null ? `${lv>0?'+':''}${lv}%` : '—';
  lEl.className = `stat-val ${lv!=null?(lv>=0?'green':'red'):''}`;
  document.getElementById('m-pdate').textContent  = d.peak_date  || '—';
  document.getElementById('m-pprice').textContent = d.peak_price ? `$${d.peak_price}` : '—';
  document.getElementById('m-now').textContent    = d.latest_price ? `$${d.latest_price}` : '—';
}
function clearModalStats() {
  ['m-dg','m-loss','m-pdate','m-pprice','m-now'].forEach(id=>{
    document.getElementById(id).textContent='…';
  });
}
function closeModal() { document.getElementById('overlay').classList.remove('show'); }
function closeOnBg(e) { if(e.target===document.getElementById('overlay')) closeModal(); }
document.addEventListener('keydown', e=>{ if(e.key==='Escape'){ closeModal(); closeAdd(); } });
</script>

<!-- Add Symbol Modal -->
<div class="add-overlay" id="add-overlay" onclick="if(event.target===this)closeAdd()">
  <div class="add-box">
    <h3>＋ Add Symbol</h3>
    <input id="add-input" type="text" placeholder="e.g. AAPL" autocomplete="off"
           onkeydown="if(event.key==='Enter')submitAdd()"/>
    <div class="add-msg" id="add-msg"></div>
    <div class="add-box-btns">
      <button class="btn-action" onclick="closeAdd()">Cancel</button>
      <button class="btn-action btn-add" id="add-submit" onclick="submitAdd()">Add</button>
    </div>
  </div>
</div>
</body>
</html>
"""

# ── Shared orange set ─────────────────────────────────────────────────────────
ORANGE_SYMS = {
    'TMO','ADSK','GOOG','LRCX','IBM','ARKX','ARKG','EWJ','GLD','PLTR',
    'AMAT','TSLA','AAPL','AMZN','CRCL','CRM','MU','NVDA','ISRG','SNDK','RKLB'
}

# ── Orange sub-page HTML ───────────────────────────────────────────────────────
ORANGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="icon" type="image/png" sizes="192x192" href="/favicon-192.png"/>
<link rel="icon" type="image/png" sizes="32x32" href="/favicon.png"/>
<link rel="apple-touch-icon" href="/apple-touch-icon.png"/>
<link rel="shortcut icon" href="/favicon.ico"/>
<title>Priceslides – Orange</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',sans-serif;font-size:14px}
  #status{padding:.3rem 1.2rem .6rem;font-size:.75rem;color:#e3b341}
  .topbar{display:flex;align-items:center;justify-content:space-between;padding:.6rem 1.2rem .2rem}
  .topbar h1{font-size:1.1rem;letter-spacing:.5px}
  .topbar h1 span{color:#ff9500}
  .btn-back{
    display:inline-flex;align-items:center;gap:.4rem;
    background:#21262d;color:#e6edf3;font-weight:600;font-size:.8rem;
    border:1px solid #30363d;border-radius:6px;padding:.4rem .9rem;cursor:pointer;
    text-decoration:none;transition:background .15s;
  }
  .btn-back:hover{background:#30363d}
  .subtitle{padding:0 1.2rem .8rem;color:#8b949e;font-size:.78rem}
  .wrap{overflow-x:auto;height:calc(100vh - 90px);overflow-y:auto}
  table{width:100%;border-collapse:collapse}
  thead th{
    background:#161b22;color:#8b949e;font-size:.72rem;text-transform:uppercase;
    letter-spacing:.6px;padding:.55rem .8rem;border-bottom:1px solid #30363d;
    position:sticky;top:0;z-index:2;white-space:nowrap;
    user-select:none;cursor:pointer;
  }
  thead th:hover{background:#1c2128;color:#e6edf3}
  .sort-icon{display:inline-flex;flex-direction:column;margin-left:4px;vertical-align:middle;line-height:1;gap:1px}
  .sort-icon span{font-size:.55rem;color:#444;line-height:1}
  thead th:hover .sort-icon span{color:#666}
  thead th.sort-asc  .sort-icon .asc {color:#ff9500}
  thead th.sort-desc .sort-icon .desc{color:#ff9500}
  tbody tr{border-bottom:1px solid #21262d;cursor:pointer;transition:background .15s;border-left:3px solid #ff9500}
  tbody tr:hover{background:#1c2128}
  td{padding:.55rem .8rem}
  td.sym{font-weight:700;color:#ff9500;font-size:.9rem}
  td.num{text-align:right;font-variant-numeric:tabular-nums}
  td.loss-pos{color:#3fb950} td.loss-neg{color:#f85149}
  td.dg{color:#e3b341}
  .loading{color:#555;font-size:.8rem}
  .err{color:#6e3030;font-size:.8rem}
  .btn-action{display:inline-flex;align-items:center;gap:.4rem;background:#21262d;color:#e6edf3;font-weight:600;font-size:.8rem;border:1px solid #30363d;border-radius:6px;padding:.4rem .9rem;cursor:pointer;text-decoration:none;transition:background .15s;margin-left:.4rem}
  .btn-action:hover{background:#30363d}
  .btn-add{background:#238636;border-color:#238636;color:#fff}
  .btn-add:hover{background:#2ea043}
  .btn-del{background:none;border:none;color:#6e3030;font-size:.85rem;cursor:pointer;padding:.15rem .4rem;border-radius:4px}
  .btn-del:hover{background:#3d1a1a;color:#f85149}
  td.del-col{text-align:center;padding:.3rem .4rem}
  .add-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:200;justify-content:center;align-items:center}
  .add-overlay.show{display:flex}
  .add-box{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:1.4rem 1.6rem;width:min(90vw,340px);box-shadow:0 8px 40px rgba(0,0,0,.7)}
  .add-box h3{color:#ff9500;margin-bottom:.8rem;font-size:.95rem}
  .add-box input{width:100%;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#e6edf3;padding:.5rem .7rem;font-size:.95rem;margin-bottom:.8rem;outline:none}
  .add-box input:focus{border-color:#ff9500}
  .add-box-btns{display:flex;gap:.6rem;justify-content:flex-end}
  .add-msg{font-size:.78rem;margin-bottom:.6rem;min-height:1rem}
  .overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:100;justify-content:center;align-items:center}
  .overlay.show{display:flex}
  .modal{background:#161b22;border:1px solid #30363d;border-radius:10px;width:min(92vw,680px);max-height:92vh;overflow:hidden;box-shadow:0 8px 40px rgba(0,0,0,.7)}
  .modal-header{display:flex;justify-content:space-between;align-items:center;padding:.7rem 1rem;border-bottom:1px solid #30363d;background:#0d1117}
  .modal-ticker{font-size:1.1rem;font-weight:700;color:#ff9500}
  .modal-close{background:none;border:none;color:#8b949e;font-size:1.3rem;cursor:pointer;padding:.2rem .5rem;border-radius:4px}
  .modal-close:hover{background:#21262d;color:#fff}
  .modal-stats{display:flex;gap:1.2rem;padding:.6rem 1rem;background:#0d1117;border-bottom:1px solid #21262d;flex-wrap:wrap}
  .stat-box{display:flex;flex-direction:column;gap:.1rem}
  .stat-label{font-size:.65rem;color:#8b949e;text-transform:uppercase;letter-spacing:.5px}
  .stat-val{font-size:.95rem;font-weight:600}
  .stat-val.green{color:#3fb950} .stat-val.red{color:#f85149} .stat-val.yellow{color:#e3b341}
  .modal-chart{padding:.8rem;text-align:center;min-height:220px;display:flex;align-items:center;justify-content:center}
  .modal-chart img{max-width:100%;border-radius:6px}
  .modal-footer{padding:.5rem 1rem;text-align:right;border-top:1px solid #21262d;background:#0d1117}
  .btn-ext{display:inline-block;padding:.35rem .8rem;margin-left:.5rem;border:1px solid #30363d;border-radius:5px;color:#58a6ff;font-size:.8rem;text-decoration:none;transition:background .15s}
  .btn-ext:hover{background:#1c2128}
  .spinner{border:3px solid #30363d;border-top-color:#ff9500;border-radius:50%;width:36px;height:36px;animation:spin .8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="topbar">
  <h1>🟠 <span>Orange</span> Watch</h1>
  <div style="display:flex;align-items:center;gap:.4rem">
    <button class="btn-action" id="btn-refresh" onclick="doRefresh()">🔄 Refresh</button>
    <button class="btn-action btn-add" onclick="openAdd()">＋ Add</button>
    <a class="btn-back" href="/">← All Symbols</a>
  </div>
</div>
<p class="subtitle">Tap a symbol to view chart &nbsp;|&nbsp; DG = trading days since last peak (Close&nbsp;&gt;&nbsp;12D EMA) &nbsp;|&nbsp; Click any column header to sort</p>
<div id="status">⏳ Loading data…</div>

<div class="wrap">
<table id="main-table">
  <thead><tr>
    <th data-col="idx"   data-type="num">#<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="sym"   data-type="str">Symbol<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="dg"    data-type="num" class="num">DG<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="loss"  data-type="num" class="num">Loss %<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="pdate" data-type="str" class="num">Last Peak<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="pprice"data-type="num" class="num">Peak $<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="now"   data-type="num" class="num">Now $<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="g1d"   data-type="num" class="num">1D %<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="g3d"   data-type="num" class="num">3D %<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="g5d"   data-type="num" class="num">5D %<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th data-col="g20d"  data-type="num" class="num">20D %<span class="sort-icon"><span class="asc">▲</span><span class="desc">▼</span></span></th>
    <th style="width:2rem"></th>
  </tr></thead>
  <tbody id="tbody"></tbody>
</table>
</div>

<!-- Modal -->
<div class="overlay" id="overlay" onclick="closeOnBg(event)">
  <div class="modal">
    <div class="modal-header">
      <span class="modal-ticker" id="m-ticker">—</span>
      <button class="modal-close" onclick="closeModal()">✕</button>
    </div>
    <div class="modal-stats">
      <div class="stat-box"><span class="stat-label">Days Gone (DG)</span><span class="stat-val yellow" id="m-dg">—</span></div>
      <div class="stat-box"><span class="stat-label">Loss %</span><span class="stat-val" id="m-loss">—</span></div>
      <div class="stat-box"><span class="stat-label">Last Peak</span><span class="stat-val" id="m-pdate">—</span></div>
      <div class="stat-box"><span class="stat-label">Peak $</span><span class="stat-val" id="m-pprice">—</span></div>
      <div class="stat-box"><span class="stat-label">Now $</span><span class="stat-val green" id="m-now">—</span></div>
    </div>
    <div class="modal-chart" id="m-chart"><div class="spinner"></div></div>
    <div class="modal-footer">
      <a class="btn-ext" id="btn-cnbc" href="#" target="_blank">📺 CNBC</a>
      <a class="btn-ext" id="btn-sc"   href="#" target="_blank">📈 StockCharts</a>
    </div>
  </div>
</div>

<script>
const symbols = {{ symbols|tojson }};
let cachedStats = {};
let rowData = symbols.map((sym, i) => ({
  idx: i+1, sym, dg:null, loss:null, pdate:null, pprice:null, now:null,
  g1d:null, g3d:null, g5d:null, g20d:null, raw:null
}));
let sortCol=null, sortAsc=true;

function renderTable() {
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = '';
  rowData.forEach(r => {
    const d = r.raw;
    const hasData = d && d.dg != null;
    const lv = hasData ? d.loss_pct : null;
    const row = document.createElement('tr');
    row.innerHTML = `
      <td style="color:#8b949e;font-size:.75rem">${r.idx}</td>
      <td class="sym" style="cursor:pointer" onclick="openModal('${r.sym}')">${r.sym}</td>
      <td class="num dg${hasData?'':' loading'}" style="cursor:pointer" onclick="openModal('${r.sym}')">${hasData?d.dg:(d&&d.error?'—':'…')}</td>
      <td class="num ${hasData?(lv>=0?'loss-pos':'loss-neg'):(d&&d.error?'err':'loading')}" style="cursor:pointer" onclick="openModal('${r.sym}')">
        ${hasData?(lv>0?'+':'')+lv+'%':(d&&d.error?'—':'…')}</td>
      <td class="num" style="color:#8b949e;font-size:.8rem;cursor:pointer" onclick="openModal('${r.sym}')">${hasData&&d.peak_date?d.peak_date:'—'}</td>
      <td class="num" style="color:#8b949e;font-size:.8rem;cursor:pointer" onclick="openModal('${r.sym}')">${hasData&&d.peak_price?'$'+d.peak_price:'—'}</td>
      <td class="num" style="color:#8b949e;font-size:.8rem;cursor:pointer" onclick="openModal('${r.sym}')">${hasData&&d.latest_price?'$'+d.latest_price:'—'}</td>
      <td class="num ${hasData&&d.g1d!=null?(d.g1d>=0?'loss-pos':'loss-neg'):'loading'}" onclick="openModal('${r.sym}')" style="cursor:pointer">${hasData&&d.g1d!=null?(d.g1d>0?'+':'')+d.g1d+'%':'—'}</td>
      <td class="num ${hasData&&d.g3d!=null?(d.g3d>=0?'loss-pos':'loss-neg'):'loading'}" onclick="openModal('${r.sym}')" style="cursor:pointer">${hasData&&d.g3d!=null?(d.g3d>0?'+':'')+d.g3d+'%':'—'}</td>
      <td class="num ${hasData&&d.g5d!=null?(d.g5d>=0?'loss-pos':'loss-neg'):'loading'}" onclick="openModal('${r.sym}')" style="cursor:pointer">${hasData&&d.g5d!=null?(d.g5d>0?'+':'')+d.g5d+'%':'—'}</td>
      <td class="num ${hasData&&d.g20d!=null?(d.g20d>=0?'loss-pos':'loss-neg'):'loading'}" onclick="openModal('${r.sym}')" style="cursor:pointer">${hasData&&d.g20d!=null?(d.g20d>0?'+':'')+d.g20d+'%':'—'}</td>
      <td class="del-col"><button class="btn-del" onclick="doDelete('${r.sym}')" title="Delete">✕</button></td>`;
    tbody.appendChild(row);
  });
}

function sortBy(col, type) {
  if (sortCol===col) sortAsc=!sortAsc; else { sortCol=col; sortAsc=true; }
  document.querySelectorAll('thead th').forEach(th => {
    th.classList.remove('sort-asc','sort-desc');
    if (th.dataset.col===col) th.classList.add(sortAsc?'sort-asc':'sort-desc');
  });
  rowData.sort((a,b) => {
    let av=a[col], bv=b[col];
    if (av===null&&bv===null) return 0;
    if (av===null) return 1; if (bv===null) return -1;
    let cmp = type==='str' ? av.localeCompare(bv) : av-bv;
    return sortAsc?cmp:-cmp;
  });
  renderTable();
}
document.querySelectorAll('thead th').forEach(th => {
  if (th.dataset.col) th.addEventListener('click', () => sortBy(th.dataset.col, th.dataset.type));
});

async function loadAll() {
  const status = document.getElementById('status');
  try {
    const r = await fetch('/api/all');
    const data = await r.json();
    cachedStats = data;
    let loaded=0, errors=0;
    rowData.forEach(r => {
      const d = data[r.sym]||{error:'no data'};
      r.raw=d; r.dg=d.dg??null; r.loss=d.loss_pct??null;
      r.pdate=d.peak_date??null; r.pprice=d.peak_price??null; r.now=d.latest_price??null;
      r.g1d=d.g1d??null; r.g3d=d.g3d??null; r.g5d=d.g5d??null; r.g20d=d.g20d??null;
      if (d.dg!=null) loaded++; else errors++;
    });
    renderTable();
    status.textContent=`✅ ${loaded} loaded, ${errors} unavailable`;
    setTimeout(()=>{ status.style.display='none'; }, 4000);
  } catch(e) { status.textContent=`⚠️ Load failed: ${e}`; }
}
renderTable();
loadAll();

async function doRefresh() {
  const btn = document.getElementById('btn-refresh');
  const status = document.getElementById('status');
  btn.disabled=true; btn.textContent='⏳ Refreshing…';
  status.style.display='block';
  status.textContent=`⏳ Refreshing ${rowData.length} symbols (1 credit each)…`;
  try {
    const r = await fetch('/api/refresh', {method:'POST'});
    const data = await r.json();
    const r2 = await fetch('/api/all');
    const stats = await r2.json();
    cachedStats = stats;
    rowData.forEach(r => {
      const d = stats[r.sym]||{}; r.raw=d;
      r.dg=d.dg??null; r.loss=d.loss_pct??null;
      r.pdate=d.peak_date??null; r.pprice=d.peak_price??null; r.now=d.latest_price??null;
      r.g1d=d.g1d??null; r.g3d=d.g3d??null; r.g5d=d.g5d??null; r.g20d=d.g20d??null;
    });
    renderTable();
    status.textContent=`✅ Refreshed ${data.updated.length} symbols, ${data.errors.length} errors`;
    setTimeout(()=>{ status.style.display='none'; }, 5000);
  } catch(e) { status.textContent=`⚠️ Refresh failed: ${e}`; }
  btn.disabled=false; btn.textContent='🔄 Refresh';
}

async function doDelete(sym) {
  if (!confirm(`Delete ${sym}? Removes from Orange list, main list, and sheet.`)) return;
  const status = document.getElementById('status');
  status.style.display='block'; status.textContent=`⏳ Deleting ${sym}…`;
  try {
    const r = await fetch(`/api/delete/${encodeURIComponent(sym)}`, {method:'POST'});
    const data = await r.json();
    if (!data.ok) throw new Error(data.error);
    rowData = rowData.filter(r => r.sym !== sym);
    rowData.forEach((r,i) => r.idx = i+1);
    renderTable();
    status.textContent=`✅ ${sym} deleted`;
    setTimeout(()=>{ status.style.display='none'; }, 3000);
  } catch(e) { status.textContent=`⚠️ Delete failed: ${e}`; }
}

function openAdd() {
  document.getElementById('add-input').value='';
  document.getElementById('add-msg').textContent='';
  document.getElementById('add-overlay').classList.add('show');
  document.getElementById('add-input').focus();
}
function closeAdd() { document.getElementById('add-overlay').classList.remove('show'); }

async function submitAdd() {
  const sym = document.getElementById('add-input').value.trim().toUpperCase();
  const msg = document.getElementById('add-msg');
  if (!sym) { msg.style.color='#f85149'; msg.textContent='Enter a symbol.'; return; }
  msg.style.color='#e3b341'; msg.textContent=`⏳ Fetching ${sym} & marking orange (~60 credits)…`;
  document.getElementById('add-submit').disabled=true;
  try {
    const r = await fetch('/api/add', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({symbol:sym, mark_orange:true})
    });
    const data = await r.json();
    if (!data.ok) throw new Error(data.error);
    if (!rowData.find(r=>r.sym===sym)) {
      rowData.push({idx:rowData.length+1, sym, dg:null,loss:null,pdate:null,pprice:null,now:null,g1d:null,g3d:null,g5d:null,g20d:null,raw:null});
    }
    const r2 = await fetch(`/api/stats/${encodeURIComponent(sym)}`);
    const d = await r2.json();
    cachedStats[sym]=d;
    const rd = rowData.find(r=>r.sym===sym);
    if (rd) { rd.raw=d; rd.dg=d.dg??null; rd.loss=d.loss_pct??null; rd.pdate=d.peak_date??null; rd.pprice=d.peak_price??null; rd.now=d.latest_price??null; rd.g1d=d.g1d??null; rd.g3d=d.g3d??null; rd.g5d=d.g5d??null; rd.g20d=d.g20d??null; }
    renderTable();
    msg.style.color='#3fb950'; msg.textContent=`✅ ${sym} added & marked orange`;
    setTimeout(closeAdd, 1500);
  } catch(e) { msg.style.color='#f85149'; msg.textContent=`⚠️ ${e}`; }
  document.getElementById('add-submit').disabled=false;
}

async function openModal(sym) {
  document.getElementById('m-ticker').textContent=sym;
  document.getElementById('btn-cnbc').href=`https://www.cnbc.com/quotes/${sym}`;
  document.getElementById('btn-sc').href=`https://stockcharts.com/h-sc/ui?s=${sym}`;
  const d=cachedStats[sym];
  if (d) fillModalStats(d);
  else { clearModalStats(); fetch(`/api/stats/${encodeURIComponent(sym)}`).then(r=>r.json()).then(d=>{cachedStats[sym]=d;fillModalStats(d);}).catch(()=>{}); }
  document.getElementById('m-chart').innerHTML='<div class="spinner"></div>';
  document.getElementById('overlay').classList.add('show');
  try {
    const r2=await fetch(`/api/chart/${encodeURIComponent(sym)}`);
    const blob=await r2.blob();
    document.getElementById('m-chart').innerHTML=`<img src="${URL.createObjectURL(blob)}" alt="${sym}"/>`;
  } catch(e) { document.getElementById('m-chart').innerHTML=`<p style="color:#f85149">Chart unavailable</p>`; }
}
function fillModalStats(d) {
  document.getElementById('m-dg').textContent=d.dg??'—';
  const lv=d.loss_pct; const lEl=document.getElementById('m-loss');
  lEl.textContent=lv!=null?`${lv>0?'+':''}${lv}%`:'—';
  lEl.className=`stat-val ${lv!=null?(lv>=0?'green':'red'):''}`;
  document.getElementById('m-pdate').textContent=d.peak_date||'—';
  document.getElementById('m-pprice').textContent=d.peak_price?`$${d.peak_price}`:'—';
  document.getElementById('m-now').textContent=d.latest_price?`$${d.latest_price}`:'—';
}
function clearModalStats() { ['m-dg','m-loss','m-pdate','m-pprice','m-now'].forEach(id=>{ document.getElementById(id).textContent='…'; }); }
function closeModal() { document.getElementById('overlay').classList.remove('show'); }
function closeOnBg(e) { if(e.target===document.getElementById('overlay')) closeModal(); }
document.addEventListener('keydown', e=>{ if(e.key==='Escape'){ closeModal(); closeAdd(); } });
</script>

<!-- Add Symbol Modal -->
<div class="add-overlay" id="add-overlay" onclick="if(event.target===this)closeAdd()">
  <div class="add-box">
    <h3>🟠 Add to Orange Watch</h3>
    <input id="add-input" type="text" placeholder="e.g. NVDA" autocomplete="off"
           onkeydown="if(event.key==='Enter')submitAdd()"/>
    <div class="add-msg" id="add-msg"></div>
    <div class="add-box-btns">
      <button class="btn-action" onclick="closeAdd()">Cancel</button>
      <button class="btn-action btn-add" id="add-submit" onclick="submitAdd()">Add & Mark Orange</button>
    </div>
  </div>
</div>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML, symbols=SYMBOLS, orange=list(ORANGE_SYMS))


@app.route("/orange")
def orange_page():
    orange_symbols = [s for s in SYMBOLS if s in ORANGE_SYMS]
    return render_template_string(ORANGE_HTML, symbols=orange_symbols)


@app.route("/api/all")
def api_all():
    """Return all cached stats in one shot."""
    with _cache_lock:
        return jsonify(dict(_stats_cache))


@app.route("/api/stats/<symbol>")
def api_stats(symbol):
    with _cache_lock:
        if symbol in _stats_cache:
            return jsonify(_stats_cache[symbol])
    # not cached yet – compute live
    try:
        df = _ohlcv_cache.get(symbol)
        if df is None:
            raise ValueError(f"{symbol} not in cache — re-run fetch_to_sheet.py")
        result = _compute_stats(symbol, df)
    except Exception as e:
        result = {"error": str(e), "dg": None, "loss_pct": None,
                  "peak_date": None, "peak_price": None, "latest_price": None}
    return jsonify(result)


@app.route("/api/chart/<symbol>")
def api_chart(symbol):
    try:
        return Response(build_chart_png(symbol), mimetype="image/png")
    except Exception as e:
        return str(e), 503


@app.route("/api/status")
def api_status():
    with _cache_lock:
        n = len(_stats_cache)
    return jsonify({"cached": n, "total": len(SYMBOLS), "done": _prefetch_done.is_set()})


@app.route("/health")
def health():
    return "ok", 200


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        Path(__file__).parent / "static",
        "favicon.ico", mimetype="image/vnd.microsoft.icon"
    )

@app.route("/favicon.png")
def favicon_png():
    return send_from_directory(
        Path(__file__).parent / "static",
        "favicon-32.png", mimetype="image/png"
    )

@app.route("/favicon-192.png")
def favicon_192():
    return send_from_directory(
        Path(__file__).parent / "static",
        "favicon-192.png", mimetype="image/png"
    )

@app.route("/apple-touch-icon.png")
def apple_touch_icon():
    return send_from_directory(
        Path(__file__).parent / "static",
        "apple-touch-icon.png", mimetype="image/png"
    )


# ── Helper: fetch one symbol from MarketData (60 days) ────────────────────────
MD_TOKEN  = "ekcwREpFQ3FXalJUWVZCX3BQMEZPeURfN2RscHA1cnliN3A4MFB3QnhXVT0"

def _md_fetch_60(symbol: str) -> pd.DataFrame:
    """Fetch 60-day OHLCV from MarketData.app (uses ~60 credits)."""
    from datetime import datetime, timedelta
    from_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    import requests as _req
    r = _req.get(
        f"https://api.marketdata.app/v1/stocks/candles/D/{symbol}/",
        params={"from": from_date},
        headers={"Authorization": f"Token {MD_TOKEN}"},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("s") != "ok":
        raise ValueError(data.get("errmsg", "no data"))
    dates  = pd.to_datetime(data["t"], unit="s").normalize()
    closes = [float(c) for c in data["c"]]
    df = pd.DataFrame({"Close": closes}, index=dates).tail(60)
    df.index.name = "Date"
    return df


def _md_fetch_latest(symbol: str) -> tuple:
    """Fetch only the most recent close from MarketData.app (1 credit)."""
    import requests as _req
    r = _req.get(
        f"https://api.marketdata.app/v1/stocks/candles/D/{symbol}/",
        params={"countback": 1},
        headers={"Authorization": f"Token {MD_TOKEN}"},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("s") != "ok":
        raise ValueError(data.get("errmsg", "no data"))
    date  = pd.to_datetime(data["t"][-1], unit="s").normalize()
    close = float(data["c"][-1])
    return date, close


def _sheet_ws():
    """Return the Google Sheet worksheet."""
    import warnings; warnings.filterwarnings("ignore")
    gc = gspread.authorize(_gcp_creds())
    return gc.open_by_key(SHEET_ID).sheet1


def _sheet_color_orange(ws, symbol: str):
    """Color a symbol's header cell orange in the sheet."""
    header = ws.row_values(1)
    if symbol not in header:
        return
    col_idx = header.index(symbol) + 1
    ws.spreadsheet.batch_update({"requests": [{"repeatCell": {
        "range": {"sheetId": ws.id,
                  "startRowIndex": 0, "endRowIndex": 1,
                  "startColumnIndex": col_idx - 1, "endColumnIndex": col_idx},
        "cell": {"userEnteredFormat": {
            "backgroundColor": {"red": 1.0, "green": 0.6, "blue": 0.0},
            "textFormat": {"bold": True,
                           "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
        }},
        "fields": "userEnteredFormat(backgroundColor,textFormat)",
    }}]})


# ── /api/refresh  — update only the latest row (1 credit per symbol) ──────────
@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """Refresh last-day close for every cached symbol. 1 credit each."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    updated, errors = [], []

    try:
        ws = _sheet_ws()
        header = ws.row_values(1)
        col_a  = ws.col_values(1)
        date_strs = col_a[1:]

        with _cache_lock:
            syms = list(_ohlcv_cache.keys())

        def _refresh_one(sym):
            try:
                date, close = _md_fetch_latest(sym)
                date_str = date.strftime("%Y-%m-%d")

                with _cache_lock:
                    df = _ohlcv_cache.get(sym)
                    if df is not None:
                        if date not in df.index:
                            new_row = pd.DataFrame({"Close": [close]}, index=[date])
                            new_row.index.name = "Date"
                            df = pd.concat([df, new_row]).tail(60)
                        else:
                            df.at[date, "Close"] = close
                        _ohlcv_cache[sym] = df
                        _stats_cache[sym] = _compute_stats(sym, df)

                if sym in header:
                    col_idx = header.index(sym) + 1
                    if date_str in date_strs:
                        row_idx = date_strs.index(date_str) + 2
                        ws.update_cell(row_idx, col_idx, round(close, 2))
                    else:
                        new_row_idx = len(date_strs) + 2
                        ws.update_cell(new_row_idx, 1, date_str)
                        ws.update_cell(new_row_idx, col_idx, round(close, 2))

                return ("ok", sym)
            except Exception as e:
                return ("err", sym, str(e))

        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(_refresh_one, sym): sym for sym in syms}
            for fut in as_completed(futures):
                result = fut.result()
                if result[0] == "ok":
                    updated.append(result[1])
                else:
                    errors.append({"sym": result[1], "error": result[2]})

    except Exception as e:
        return jsonify({"updated": updated, "errors": errors, "fatal": str(e)})

    return jsonify({"updated": updated, "errors": errors})


# ── /api/delete/<symbol> ───────────────────────────────────────────────────────
@app.route("/api/delete/<symbol>", methods=["POST"])
def api_delete(symbol):
    """Remove symbol from cache, SYMBOLS list, ORANGE_SYMS, and sheet column."""
    global SYMBOLS
    try:
        # remove from memory
        with _cache_lock:
            _ohlcv_cache.pop(symbol, None)
            _stats_cache.pop(symbol, None)
        if symbol in SYMBOLS:
            SYMBOLS = [s for s in SYMBOLS if s != symbol]
        ORANGE_SYMS.discard(symbol)

        # remove column from sheet
        ws = _sheet_ws()
        header = ws.row_values(1)
        if symbol in header:
            col_idx = header.index(symbol) + 1
            ws.spreadsheet.batch_update({"requests": [{"deleteDimension": {
                "range": {
                    "sheetId": ws.id,
                    "dimension": "COLUMNS",
                    "startIndex": col_idx - 1,
                    "endIndex": col_idx,
                }
            }}]})
        return jsonify({"ok": True, "symbol": symbol})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── /api/add  — add a new symbol ──────────────────────────────────────────────
@app.route("/api/add", methods=["POST"])
def api_add():
    """
    Add a new symbol: fetch 60-day history, append column to sheet,
    update cache. If mark_orange=true, also color orange in sheet + ORANGE_SYMS.
    """
    global SYMBOLS
    body = request.get_json(force=True)
    symbol = body.get("symbol", "").upper().strip()
    mark_orange = bool(body.get("mark_orange", False))

    if not symbol:
        return jsonify({"ok": False, "error": "no symbol"}), 400
    if symbol in SYMBOLS:
        # already exists — just mark orange if requested
        if mark_orange and symbol not in ORANGE_SYMS:
            ORANGE_SYMS.add(symbol)
            try:
                ws = _sheet_ws()
                _sheet_color_orange(ws, symbol)
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "symbol": symbol, "already_existed": True,
                        "orange": symbol in ORANGE_SYMS})

    try:
        # fetch 60-day data
        df = _md_fetch_60(symbol)

        # update in-memory cache
        with _cache_lock:
            _ohlcv_cache[symbol] = df
            _stats_cache[symbol] = _compute_stats(symbol, df)
        SYMBOLS.append(symbol)
        if mark_orange:
            ORANGE_SYMS.add(symbol)

        # append column to sheet
        ws = _sheet_ws()
        col_a = ws.col_values(1)
        date_strs = col_a[1:]           # existing date rows
        header = ws.row_values(1)
        next_col = len(header) + 1
        import gspread.utils as gu
        col_letter = gu.rowcol_to_a1(1, next_col)[:-1]

        # build column values aligned to existing dates
        df_dict = {d.strftime("%Y-%m-%d"): round(v, 2)
                   for d, v in df["Close"].items()}
        col_values = [[symbol]] + [[df_dict.get(d, "")] for d in date_strs]
        ws.update(f"{col_letter}1", col_values)

        if mark_orange:
            _sheet_color_orange(ws, symbol)

        return jsonify({"ok": True, "symbol": symbol, "rows": len(df),
                        "orange": symbol in ORANGE_SYMS})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
