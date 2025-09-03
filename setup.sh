#!/usr/bin/env bash
set -e  # stop if error

echo "🔧 Setting up CTIA environment..."

# Check Python
if ! command -v python3 &> /dev/null
then
    echo "❌ Python3 not found. Please install Python 3.10+."
    exit 1
fi

# Create venv if missing
if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
  echo "📥 Installing dependencies..."
  pip install -r requirements.txt
else
  echo "⚠️ No requirements.txt found, skipping"
fi

echo "✅ Python environment ready (venv activated)"
echo
echo "👉 To run locally: uvicorn app.main:app --reload --port 8000"
echo "👉 To run with Docker: docker-compose up --build"
