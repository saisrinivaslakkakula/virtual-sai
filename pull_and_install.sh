#!/bin/bash

# ──────────────── 🔍 Check Environment ────────────────
check_runpod() {
    if [ -d "/workspace" ] && [ -f "/.dockerenv" ]; then
        return 0  # We are in RunPod
    fi
    return 1  # We are not in RunPod
}

# ──────────────── 📂 Clone Repository ────────────────
REPO_URL="https://github.com/saisrinivaslakkakula/virtual-sai.git"
WORKSPACE_DIR="/workspace/virtual-sai"

echo "🔄 Setting up workspace..."
if [ -d "$WORKSPACE_DIR" ]; then
    echo "📂 Updating existing repository..."
    cd $WORKSPACE_DIR
    git pull
else
    echo "📥 Cloning repository..."
    git clone $REPO_URL $WORKSPACE_DIR
    cd $WORKSPACE_DIR
fi

# Skip Docker setup if we're in RunPod
if check_runpod; then
    echo "🚀 Running in RunPod environment - skipping Docker setup"
else
    echo "⚠️ Not running in RunPod - Docker setup would go here"
fi

# ──────────────── 🔍 Local Ollama Setup ────────────────
wait_for_ollama() {
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "✅ Ollama is running"
            return 0
        fi
        echo "⏳ Waiting for Ollama to start (attempt $attempt/$max_attempts)..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ Failed to start Ollama after $max_attempts attempts"
    return 1
}

echo "🔍 Checking if Ollama is running..."
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "🚀 Starting Ollama service..."
    nohup ollama serve > ollama.log 2>&1 &
    
    # Wait for Ollama to start
    if ! wait_for_ollama; then
        echo "❌ Failed to start Ollama. Check ollama.log for details."
        exit 1
    fi
else
    echo "✅ Ollama is already running"
fi

# ──────────────── 🤖 Check Model Status ────────────────
echo "🤖 Checking if Mistral model is available..."
if ! ollama list | grep -q "mistral"; then
    echo "📥 Pulling Mistral model..."
    ollama pull mistral
    
    echo "⚙️ Creating optimized model configuration..."
    cat > /tmp/Modelfile << EOL
FROM mistral
parameter temperature 0.7
parameter num_ctx 2048
parameter num_gpu 1
parameter num_thread 4
parameter stop "</s>"
parameter stop "user:"
EOL
    
    echo "🔧 Creating optimized model..."
    cd /tmp && ollama create optimized-mistral -f Modelfile
    cd $WORKSPACE_DIR
else
    echo "✅ Mistral model is already available"
fi

# ──────────────── 📂 Ensure data folder exists ────────────────
mkdir -p data
touch data/.keep

# ──────────────── 🐍 Python Setup ────────────────
echo "🐍 Setting up Python environment..."
# Use system Python in RunPod
if [ -f "/usr/local/lib/python3.10/dist-packages/pip" ]; then
    echo "Using system Python in RunPod environment"
    # Install requirements directly
    if [ -f "requirements.txt" ]; then
        echo "📦 Installing Python packages..."
        pip install -r requirements.txt
    else
        echo "⚠️ requirements.txt not found, installing core packages..."
        pip install streamlit>=1.44.1 sentence-transformers>=2.6.1 llama-index-core>=0.12.4 llama-index-llms-ollama>=0.5.4
    fi
else
    # Create virtual environment for non-RunPod environments
    python3 -m venv venv
    . venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
fi

# ──────────────── 🔍 Find Available Port ────────────────
find_available_port() {
    local port=$1
    while netstat -tln | grep -q ":$port"; do
        echo "Port $port is in use, trying next port..."
        port=$((port + 1))
    done
    echo $port
}

# ──────────────── 🧹 Cleanup Processes ────────────────
cleanup_processes() {
    echo "🧹 Cleaning up existing processes..."
    pkill -f "streamlit run" || true
    pkill -f cloudflared || true
    sleep 2
}

# ──────────────── 🚦 Wait for Service ────────────────
wait_for_service() {
    local port=$1
    local service=$2
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:$port > /dev/null; then
            echo "✅ $service is running on port $port"
            return 0
        fi
        echo "⏳ Waiting for $service (attempt $attempt/$max_attempts)..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service failed to start on port $port"
    return 1
}

# ──────────────── 🧠 Run Services ────────────────
echo "🧠 Launching services..."

if check_runpod; then
    # Clean up existing processes
    cleanup_processes
    
    # Find available port starting from 7860
    PORT=$(find_available_port 7860)
    echo "🔍 Using port: $PORT"
    
    # ──────────────── 🧠 Start Streamlit ────────────────
    echo "🧠 Launching Streamlit app..."
    nohup streamlit run app.py \
        --server.port=$PORT \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false \
        > streamlit.log 2>&1 &
    
    # Wait for Streamlit to start
    if ! wait_for_service $PORT "Streamlit"; then
        echo "❌ Failed to start Streamlit. Checking logs:"
        tail -n 50 streamlit.log
        exit 1
    fi
    
    # ──────────────── 🌐 Start Cloudflare Tunnel ────────────────
    echo "🌐 Launching Cloudflare tunnel..."
    if ! command -v cloudflared &> /dev/null; then
        echo "📥 Installing Cloudflared..."
        curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && \
        dpkg -i cloudflared.deb && \
        rm cloudflared.deb
    fi
    
    nohup cloudflared tunnel --url http://localhost:$PORT > cloudflare.log 2>&1 &
    
    # ──────────────── 📍 Print Access Info ────────────────
    echo "✅ App setup complete."
    echo "🔗 Local access: http://localhost:$PORT"
    echo "📝 Check logs:"
    echo "   tail -f streamlit.log"
    echo "   tail -f cloudflare.log"
    
    # Monitor logs in real-time
    echo "📊 Monitoring logs (Ctrl+C to stop)..."
    tail -f streamlit.log cloudflare.log &
    
    # Keep the script running and monitor services
    while true; do
        if ! pgrep -f "streamlit run" > /dev/null; then
            echo "⚠️ Streamlit process died. Check streamlit.log for errors."
            tail -n 50 streamlit.log
            exit 1
        fi
        if ! pgrep -f cloudflared > /dev/null; then
            echo "⚠️ Cloudflare tunnel died. Check cloudflare.log for errors."
            tail -n 50 cloudflare.log
            exit 1
        fi
        sleep 10
    done
else
    streamlit run app.py
fi
