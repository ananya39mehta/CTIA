#!/usr/bin/env bash
set -e  # Exit immediately on error

echo "🔧 Setting up CTIA v2 environment..."
echo "====================================="

# -------- Check Python version --------
if ! command -v python3 &> /dev/null
then
    echo "❌ Python3 not found. Please install Python 3.10 or higher."
    exit 1
fi

PY_VERSION=$(python3 -V | awk '{print $2}')
echo "🐍 Using Python $PY_VERSION"

# -------- Create virtual environment --------
if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv venv
else
  echo "✅ Virtual environment already exists."
fi

# Activate venv
source venv/bin/activate
echo "✨ Virtual environment activated."

# -------- Install dependencies --------
if [ -f "requirements.txt" ]; then
  echo "📥 Installing dependencies..."
  pip install --upgrade pip
  pip install -r requirements.txt
else
  echo "⚠️ No requirements.txt found, skipping dependency installation."
fi

# -------- Initialize database --------
echo "🧱 Initializing local SQLite database..."
PYTHONPATH=. python - <<'PYCODE'
from app.db import init_db
print("✅ Database initialized successfully.")
PYCODE

# -------- Ensure model is trained --------
MODEL_PATH="models/probability_model.pkl"
if [ ! -f "$MODEL_PATH" ]; then
  echo "🧠 Training model (this may take a minute)..."
  PYTHONPATH=. python notebooks/eda_and_train.py
else
  echo "✅ Model file found ($MODEL_PATH). Skipping training."
fi

echo
echo "🚀 Setup Complete!"
echo "-------------------------------------"
echo "👉 To start the FastAPI app:"
echo "   source venv/bin/activate"
echo "   uvicorn app.main:app --reload --port 8000"
echo
echo "📊 Access API docs at: http://127.0.0.1:8000/docs"
echo "🧠 View Explainability at: /explain_ticket_visual/{ticket_id}"
echo
echo "💡 Tip: Run tests anytime using:"
echo "   PYTHONPATH=. pytest -v tests/"
