# Priceslides

Stock watch list with EMA-based peak detection, loss tracking, and pop-up charts.

## Live
[priceslides.up.railway.app](https://priceslides.up.railway.app)

## Run locally
```bash
python3 -m pip install -r requirements.txt
PORT=5002 python3 sctr_app.py
```
Then open http://localhost:5002

## Daily data refresh
```bash
python3 fetch_to_sheet.py
```

## Environment variables (Railway)
| Variable | Description |
|---|---|
| `GCP_KEY_JSON` | Google service account key as single-line JSON |
| `MD_TOKEN` | MarketData.app API token |
