"""
fetch_to_sheet.py
─────────────────
Fetches 60-day daily Close prices for the first 100 symbols from
MarketData.app, then writes them into Google Sheets.

Sheet layout (Sheet1):
  Row 1  : headers  → Date | SYM1 | SYM2 | … | SYM100
  Row 2+ : data     → 2025-01-02 | 185.20 | … 

Usage:
  python3 fetch_to_sheet.py
  python3 fetch_to_sheet.py --dry-run   (fetch only, don't write to sheet)
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import gspread
from google.oauth2.service_account import Credentials

# ── Config ────────────────────────────────────────────────────────────────────
SHEET_ID       = "1kTA8Xy5vCRxMaLhlXLxrKQ_j8VYtDf_7vMGI-R-yUJs"
GCP_KEY        = Path(__file__).parent / "gcp_key.json"
CSV_PATH       = Path(__file__).parent / "stockcharts_SCTR.csv"
import os as _os
MD_TOKEN       = _os.environ.get("MD_TOKEN", "ekcwREpFQ3FXalJUWVZCX3BQMEZPeURfN2RscHA1cnliN3A4MFB3QnhXVT0")
N_SYMBOLS      = 100
TRADING_DAYS   = 60
DRY_RUN        = "--dry-run" in sys.argv

# ── Load symbols from Google Sheet header (source of truth) ───────────────────
_GCP_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def _gcp_creds():
    import json as _json
    env_json = _os.environ.get("GCP_KEY_JSON")
    if env_json:
        info = _json.loads(env_json)
        return Credentials.from_service_account_info(info, scopes=_GCP_SCOPES)
    return Credentials.from_service_account_file(str(GCP_KEY), scopes=_GCP_SCOPES)

try:
    _gc = gspread.authorize(_gcp_creds())
    _ws = _gc.open_by_key(SHEET_ID).sheet1
    _header = _ws.row_values(1)
    symbols = [h for h in _header if h and h != "Date"]
    print(f"Symbols loaded from sheet header: {len(symbols)} symbols")
except Exception as _e:
    print(f"Could not read sheet header ({_e}), falling back to CSV top-{N_SYMBOLS}")
    symbols = pd.read_csv(CSV_PATH).iloc[:N_SYMBOLS, 0].tolist()

print(f"Symbols (first 5): {symbols[:5]} … ({len(symbols)} total)")

# ── MarketData.app fetch ──────────────────────────────────────────────────────
# Individual candles endpoint: 1 credit per symbol
# GET /v1/stocks/candles/D/{symbol}/?from=DATE&token=TOKEN
from_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")  # 90 cal days ≈ 60 trading days

session = requests.Session()
session.headers.update({"Authorization": f"Token {MD_TOKEN}"})

def fetch_closes(symbol: str) -> pd.Series:
    url = f"https://api.marketdata.app/v1/stocks/candles/D/{symbol}/"
    r = session.get(url, params={"from": from_date}, timeout=15)
    if r.status_code == 429:
        print(f"  [{symbol}] rate-limited, waiting 10s …")
        time.sleep(10)
        r = session.get(url, params={"from": from_date}, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("s") != "ok":
        raise ValueError(f"{symbol}: {data.get('errmsg','no data')}")
    dates  = pd.to_datetime(data["t"], unit="s").normalize()
    closes = data["c"]
    s = pd.Series(closes, index=dates, name=symbol)
    return s.tail(TRADING_DAYS)

print(f"\nFetching {N_SYMBOLS} symbols from MarketData.app …")
all_series = {}
ok = 0
for i, sym in enumerate(symbols):
    try:
        s = fetch_closes(sym)
        all_series[sym] = s
        ok += 1
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N_SYMBOLS} done ({ok} ok)")
    except Exception as e:
        print(f"  [{sym}] ERROR: {e}")

print(f"\nFetch complete: {ok}/{N_SYMBOLS} symbols loaded")

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame(all_series)          # index=dates, columns=symbols
df.index.name = "Date"
df = df.sort_index()                   # oldest → newest
df.index = df.index.strftime("%Y-%m-%d")
print(f"Shape: {df.shape}  (rows=trading days, cols=symbols)")
print(df.tail(3))

if DRY_RUN:
    print("\n--dry-run: skipping Google Sheets write")
    sys.exit(0)

# ── Write to Google Sheet ─────────────────────────────────────────────────────
print("\nConnecting to Google Sheets …")
creds = Credentials.from_service_account_file(
    str(GCP_KEY),
    scopes=[
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ],
)
gc = gspread.authorize(creds)
sh = gc.open_by_key(SHEET_ID)
ws = sh.sheet1
ws.clear()

# Build rows: header + data
header = ["Date"] + df.columns.tolist()
rows   = [header]
for date_str, row in df.iterrows():
    rows.append([date_str] + [
        round(float(v), 2) if pd.notna(v) else ""
        for v in row.values
    ])

ws.update("A1", rows)
ws.freeze(rows=1)   # freeze header row

# Bold the header
ws.format("A1:CZ1", {"textFormat": {"bold": True}})

print(f"✅ Written {len(rows)-1} rows × {len(header)} columns to Google Sheet")
print(f"   https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit")
