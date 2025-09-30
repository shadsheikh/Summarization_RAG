python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn src.api:app --reload

## Create .env

PYTHONPATH=src/
