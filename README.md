# AI Labs Case Study Assessment

Senior AI Engineer candidate assessment for Stratpoint AI Labs.

---

## Overview

Two end-to-end AI solutions built in Python, served through a unified Gradio web application:

- **Challenge 1 — Solar Yield Prediction:** XGBoost model trained on Australian weather data to predict daily solar energy yield, exposed through a LangChain ReAct agent that answers natural language questions
- **Challenge 2 — Receipt Q&A:** Five-tier OCR pipeline using EasyOCR and Gemini multimodal LLM to extract structured data from receipt images, with a LangGraph-orchestrated Q&A interface

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Gradio Application                 │
│                                                     │
│  Tab 1: Solar Yield          Tab 2: Receipt Q&A     │
│  ─────────────────           ────────────────────   │
│  LangChain ReAct Agent       LangGraph Pipeline     │
│  Gemini 2.5 Flash            EasyOCR + Gemini       │
│  XGBoost + Rasterio          5-Tier OCR             │
└─────────────────────────────────────────────────────┘
```

### Challenge 1 — Solar Yield Prediction

```
Global Solar Atlas TIFF + Kaggle Weather Data
                    ↓
             Rasterio extraction
            XGBoost (Optuna tuned)
                    ↓
            LangChain ReAct Agent
            ↓                   ↓
    predict_solar_yield  get_city_solar_stats
                    ↓
              Gradio Chat UI
         (tool call observability)
```

### Challenge 2 — Receipt Q&A

```
Receipt Image
      ↓
Five-Tier Pipeline (selectable)
  Tier 1: Raw EasyOCR
  Tier 2: Gemini multimodal OCR
  Tier 3: EasyOCR + Entity LLM
  Tier 4: Gemini OCR + Entity LLM
  Tier 5: Unified Multimodal LLM
      ↓
Structured entities (company, date, address, total)
      ↓
Gradio ChatInterface Q&A
```

---

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- Google API key (Gemini 2.5 Flash)
- Docker + Docker Compose (optional)

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd ai_labs_assessment
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and add your Google API key:

```
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. (Optional) Download raw datasets

Only required if you want to reproduce the training and evaluation notebooks.

| Dataset               | Used in                      | Link                                                                                                   |
| --------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------ |
| `weatherAUS.csv`      | `01_solar_exploration.ipynb` | [Kaggle Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/data) |
| `Australia_GISdata_*` | `01_solar_exploration.ipynb` | [Global Solar Atlas](https://globalsolaratlas.info/download/australia)                                 |
| `SROIE2019/`          | `02_ocr_exploration.ipynb`   | [Kaggle SROIE datasetv2](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2)                       |

Place each dataset under `data/raw/` as shown in the project structure.

---

## Running the App

### Option A — Local Python

```bash
pip install uv
uv sync
uv run python src/app.py
```

Open `http://localhost:7860` in your browser.

### Option B — Docker

```bash
docker compose up
```

Open `http://localhost:7860` in your browser.

> **Note:** EasyOCR runs on CPU inside Docker (no MPS/GPU). Tier 1 and Tier 3 will be slower than running locally on Apple Silicon.

---

## Project Structure

```
ai_labs_assessment/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
├── README.md
│
├── notebooks/
│   ├── 01_solar_exploration.ipynb
│   └── 02_ocr_exploration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   │
│   ├── challenge_1_solar/
│   │   ├── __init__.py
│   │   ├── schemas.py          # WeatherInput Pydantic model
│   │   ├── model.py            # XGBoost inference + yield formula
│   │   ├── tools.py            # LangChain tools
│   │   └── agent.py            # ReAct agent + streaming
│   │
│   └── challenge_2_receipts/
│       ├── __init__.py
│       ├── schemas.py          # ReceiptEntities Pydantic model
│       ├── ocr.py              # EasyOCR wrapper
│       ├── pipelines.py        # Five-tier pipeline functions
│       └── graph.py            # LangGraph pipeline nodes
│
└── data/
    ├── processed/
    │   ├── holdout_weather.csv         # 2010 holdout weather data
    │   ├── results_checkpoint.json     # OCR evaluation results
    │   └── xgb_model.json              # Trained XGBoost model
    │
    └── raw/
        ├── weatherAUS.csv              # Kaggle Australia Daily Weather Data
        ├── Australia_GISdata_*/        # Global Solar Atlas GeoTIFFs
        └── SROIE2019/                  # SROIE V2 receipt dataset
            ├── train/                  # 626 receipts (box/, entities/, img/)
            └── test/                   # 347 receipts (box/, entities/, img/)
```

---

## Challenge 1 — Solar Yield Prediction

### Model

- **Target:** PVOUT (kWh/kWp/day) extracted from Global Solar Atlas TIFF via Rasterio
- **Features:** 22 weather features from Kaggle Australia Daily Weather Data
- **Training year:** 2009 (16,424 rows, 45 cities)
- **Holdout year:** 2010 (16,417 rows, 45 cities)
- **Tuning:** Optuna (100 trials, TPESampler)

| Metric | Baseline | Tuned  |
| ------ | -------- | ------ |
| RMSE   | 0.2613   | 0.2546 |
| MAE    | 0.1962   | 0.1872 |
| R²     | 0.5617   | 0.5840 |

### Yield Formula

```
Panel area (m²)    = farm_area_ha × 10,000 × GCR
Installed kWp      = Panel area × panel_efficiency
Daily yield (kWh)  = Installed kWp × PVOUT
```

Defaults: GCR=0.35, panel efficiency=18%. Both can be overridden via natural language.

### Agent Capabilities

- Date resolution (`"tomorrow"`, `"next Monday"` → YYYY-MM-DD)
- Qualitative weather inference (`"hot and clear"` → full feature vector)
- Historical fallback when no date or weather provided
- Fuzzy city matching (handles typos and casing)
- Multi-city comparison
- Yield formula explanation

### Example Questions

```
What is the expected daily yield for a 50ha solar farm in Sydney tomorrow?
I have a 100ha farm in Darwin. It's a scorching hot and perfectly clear day, what's my yield?
Heavy storms are expected in Melbourne next Monday for my 75ha farm, how bad will my yield be?
Which city gives the best return for a 100ha farm — Alice Springs or Melbourne? And why?
Can you explain how my daily solar yield is actually calculated?
```

---

## Challenge 2 — Receipt Q&A

### Pipeline Tiers

| Tier   | Description                             |
| ------ | --------------------------------------- |
| Tier 1 | Raw EasyOCR — baseline, no LLM          |
| Tier 2 | Gemini multimodal OCR                   |
| Tier 3 | EasyOCR + Entity analysis LLM           |
| Tier 4 | Gemini OCR + Entity analysis LLM        |
| Tier 5 | Unified Multimodal — single Gemini call |

### LLM Corrections (Tiers 3, 4, 5)

- Malformed word correction (`0↔O`, `1↔l`, `rn↔m`)
- Compound entity completion (`MYDIN MO` → `MYDIN MOHAMED HOLDINGS SDN BHD`)
- General normalisation (spacing, punctuation, currency formatting)

### Dataset

[SROIE V2](https://www.kaggle.com/datasets/urbikn/sroie2019-datasetv2) — 973 receipts with ground truth annotations for company, date, address, and total.

---

## AI Tool Usage

Claude (Anthropic) with the Context7 MCP server (for latest documentation lookup) was used as the primary development assistant — generating code, advising on architecture, and writing documentation. All code was reviewed, integrated, and debugged manually.

---

## Notes

- GSA PVOUT represents theoretical yield under idealised conditions. Apply a performance ratio of 0.75–0.80 for real-world generation estimates.
- R² is capped at ~0.58 due to the discrete nature of the PVOUT target (45 unique values across 16,000+ rows). Including city as a feature inflates R² to 0.99 but constitutes data leakage.
- EasyOCR confidence threshold is set to 0.2 — empirically validated on SROIE receipts. Higher thresholds discard critical entity text.
