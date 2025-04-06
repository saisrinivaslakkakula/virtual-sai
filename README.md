# 🧠 Virtual Sai — Your AI-Powered Career Assistant

Virtual Sai is a private, local-first AI assistant that answers questions about Sai Srinivas' work experience, based on resumes and journals. It's built using **LlamaIndex**, **local LLMs** (via Ollama), and **Streamlit** for the UI.

---

## 🚀 Features

-   🤖 Ask career-related questions to Virtual Sai
-   🧠 Powered by a custom Retrieval-Augmented Generation (RAG) engine
-   💬 Answers in **first-person** or **third-person** tone
-   📁 Upload documents to update Virtual Sai’s knowledge
-   🛡️ Private, offline-friendly — runs entirely on your machine

---

## 🧰 Tech Stack

-   [Streamlit](https://streamlit.io/) — for the UI
-   [LlamaIndex](https://www.llamaindex.ai/) — for document indexing + RAG
-   [Ollama](https://ollama.com) — for local language models (Mistral, Phi, etc.)
-   [Sentence Transformers](https://www.sbert.net/) — for document embeddings

---

## 📦 Installation (Local Setup)

> ⚠️ Recommended: MacBook with at least 16GB RAM

### 1. Clone the repo

```bash
git clone https://github.com/your-username/virtual-sai.git
cd virtual-sai
```

### 🔹 Step 2: Install Ollama (for Local LLM)

Go to: https://ollama.com/download
Download and install Ollama for your operating system (macOS recommended).

Then, in a terminal:

```
ollama run phi
```

### 🔹 Step 3: Set Up Python Virtual Environment

```
python3 -m venv venv
source venv/bin/activate
```

Then install all required packages:

```
pip install --upgrade pip
pip install -r requirements.txt
```

### 🔹 Step 4: Run the App

```
streamlit run app.py
```
