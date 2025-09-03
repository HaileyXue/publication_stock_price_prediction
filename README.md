# Create a 3.11 venv (Homebrew’s python@3.11)
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
source .venv/bin/activate
python --version  # should show 3.11.x
pip install --upgrade pip
pip install -r app/requirements.txt

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