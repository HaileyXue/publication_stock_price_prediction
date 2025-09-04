### README
## Publication–Stock Price Prediction

This project explores whether trends in scientific publication activity (via OpenAlex) are correlated with, or predictive of, stock sector performance.
It provides a complete pipeline for:
1. Fetching and aggregating stock prices from Stooq
2. Fetching publication counts and top topics from OpenAlex
3. Building a clean feature dataset
4. Visualizing correlations and trends
5. Training ML models (Logit, Random Forest, XGBoost) to predict next-week stock movement
6. Serving results in an interactive Streamlit app

## Project Structure
```
publication_stock_price_prediction/  
├── app.py                 # Main Streamlit app  
├── docker-compose.yml     # Compose config for running in Docker  
├── Dockerfile             # Container build recipe  
├── requirements.txt       # Python dependencies  
├── README.md              # This file  
├── .gitignore             # This file  

├── app/  
│   ├── config/  
│   │   └── sector_map.yaml   # Sector ↔ tickers ↔ topic mappings  
│   └── scripts/  
│       ├── 01_fetch_prices_stooq.py     # Fetch sector-level stock price data  
│       ├── 02_fetch_openalex_topics.py  # Fetch publications and topics from OpenAlex  
│       ├── 03_build_features.py         # Merge prices + pubs → feature table  
│       ├── 04_visualize.py              # Create plots and correlation heatmaps  
│       └── 05_train_eval.py             # Train & evaluate ML models  

├── data/  
│   ├── raw/                 # Raw fetched data (per-ticker prices, etc.)  
│   ├── processed/           # Cleaned + merged datasets  
│   │   ├── features/        # Final feature tables per sector  
│   │   └── topics/          # Publication topic tables  
│   └── reports/  
│       ├── plots/           # Visualization outputs (PNGs)  
│       └── models/          # Model metrics & feature importance  
```

All data folders are created automatically on first run.  

## Running the Streamlit App
**Option 1 - Deployed Website**  

Visit [Lit&Stock App](https://publicationstockpriceprediction-haileyxue.streamlit.app/) hosted on Streamlit.  

**Option 2 - Docker (no repo clone)**
1. Make sure your have Docker running on your local machine  
2. Run this line in terminal to start the app:  
```
docker run -p 8501:8501 haileyxue391/pub-stock-app:latest
```
3. Visit http://localhost:8501

**Option 3 — Docker**

1. Clone this repo:  
```
git clone https://github.com/HaileyXue/publication_stock_price_prediction.git
cd publication_stock_price_prediction
```
2. Run this line in terminal to build the Docker image and start the app:  
```
docker compose up --build
```
3. Visit http://localhost:8501  

**Option 4 — Local (no Docker)**
1. Clone this repo:  
```
git clone https://github.com/HaileyXue/publication_stock_price_prediction.git
cd publication_stock_price_prediction
```
2. Create a virtual environment and install deps:  
```
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
3. Run the app:  
```
streamlit run app.py
```
4. Open http://localhost:8501 in your browser.

## Usage Workflow
The Streamlit app automates the pipeline, but you can also run scripts manually.  
1. Fetch stock prices  
provide the time window and sector (in `sector_map.yaml`).  
```
python app/scripts/01_fetch_prices_stooq.py \
  --sector Semiconductors \
  --start 2023-09-01 \
  --end   2025-09-01
```
Prices data is written to `data/raw/prices/`.  

2. Fetch publications (OpenAlex)  
provide the time window and sector (in `sector_map.yaml`).  
```
python app/scripts/02_fetch_openalex_topics.py \
  --sector Semiconductors \
  --start 2024-09-01 \
  --end   2025-09-01
```
Publication data is written to `data/processed/topics/`.  

3. Build features  
provide sector.  
```
python app/scripts/03_build_features.py --sector Semiconductors
```
Full feature dataset is written to `data/processed/features/`.  

4. Visualize  
provide sector.  
```
python app/scripts/04_visualize.py --sector Semiconductors
```
Plots are saved under `data/reports/plots/`.  

5. Train models  
provide and sector and whether to use categorical features or not.  
``` 
python app/scripts/05_train_eval.py --sector Semiconductors
python app/scripts/05_train_eval.py --sector Semiconductors --use-categories
```
Model outputs are written under `data/reports/models/`.
Feature importance plots are saved under `data/reports/plots/`.  

## Features & Models

-  **Features used:**
  - Price & returns (ret_1d, close_mean), volume (vol_4w, vol_growth)
  - Publication counts (pub_4w, pub_growth)
  - Optional categorical: top 1 & top 5 publication topics

- **Models:**
  - Logistic Regression (balanced)
  - Random Forest (balanced subsample)
  - XGBoost (with scale_pos_weight)

- **Metrics:** ROC-AUC, PR-AUC, classification report  
- **Outputs:** ROC/PR curves, feature importance plots, JSON metrics  

## Tech Stack
- Python 3.11
- Streamlit
- pandas / numpy
- scikit-learn
- XGBoost
- seaborn / matplotlib
- pyalex (OpenAlex API client)
- pandas_datareader (Stooq price reader)

# Create a 3.11 venv (Homebrew’s python@3.11)
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
source .venv/bin/activate
python --version  # should show 3.11.x
pip install --upgrade pip
pip install -r requirements.txt

# commands to run the scripts with user-defined time window and sector:
python app/scripts/01_fetch_prices_stooq.py \  
  --sector Biotech \       
  --start  2024-09-01 \  
  --end    2025-09-01
python app/scripts/01_fetch_prices_stooq.py \
  --sector Semiconductors \
  --start  2023-09-01 \
  --end    2025-09-01  
python app/scripts/02_fetch_openalex_topics.py \
  --sector Semiconductors \
  --start  2024-09-01 \
  --end    2025-09-01
python app/scripts/03_build_features.py \
  --sector Semiconductors
python app/scripts/04_visualize.py \
  --sector Semiconductors
python app/scripts/05_train_eval.py --sector Semiconductors
python app/scripts/05_train_eval.py --sector Semiconductors --use-categories

# run app
streamlit run app.py

# Docker - clone repo and build image and run Docker
docker compose up --build

# Docker - just run this line with Docker running on local
docker run -p 8501:8501 haileyxue391/pub-stock-app:latest
