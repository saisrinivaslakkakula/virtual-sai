#!/bin/bash

# ──────────────── 🔁 Git Pull or Clone ────────────────
REPO_URL="https://github.com/saisrinivaslakkakula/virtual-sai.git"
TARGET_DIR="/workspace/virtual-sai"
cd /workspace

if [ -d "$TARGET_DIR/.git" ]; then
    echo "📥 Pulling latest code..."
    cd $TARGET_DIR && git pull
else
    echo "🚀 Cloning fresh repo..."
    git clone $REPO_URL $TARGET_DIR
    cd $TARGET_DIR
fi

# ──────────────── 🧪 Check for 'data/' ────────────────
mkdir -p data
touch data/.keep

# ──────────────── 🐍 Python Setup ────────────────
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --root-user-action=ignore -r requirements.txt

# ──────────────── 🧠 Start Streamlit ────────────────
echo "🧠 Launching Streamlit app..."
nohup streamlit run admin.py --server.port=7860 --server.address=0.0.0.0 > streamlit.log 2>&1 &

# ──────────────── 🌐 Start Cloudflare Tunnel ────────────────
echo "🌐 Launching Cloudflare tunnel..."
nohup cloudflared tunnel --url http://localhost:7860 > cloudflare.log 2>&1 &

# ──────────────── 📍 Print Access Info ────────────────
echo "✅ App setup complete."
echo "Check logs:"
echo "   tail -f streamlit.log"
echo "   tail -f cloudflare.log"
