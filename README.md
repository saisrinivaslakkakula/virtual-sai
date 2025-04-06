# Virtual Sai - AI Assistant

An AI-powered assistant that can answer questions about Sai's work experience and background using RAG (Retrieval Augmented Generation).

## Requirements

-   Python 3.10+
-   [Ollama](https://ollama.ai) installed
-   8GB+ RAM recommended

## Quick Start

1. Install Ollama from [ollama.ai](https://ollama.ai)

2. Clone the repository:

```bash
git clone https://github.com/saisrinivaslakkakula/virtual-sai.git
cd virtual-sai
```

3. Run the setup script:

```bash
chmod +x pull_and_install.sh
./pull_and_install.sh
```

The script will:

-   Install required Python packages
-   Pull the Mistral model via Ollama
-   Start the Streamlit interface
-   Create a Cloudflare tunnel for remote access

## Usage

1. Access the web interface through:

    - Local: http://localhost:7860
    - Remote: Check the Cloudflare tunnel URL in the terminal

2. Upload documents (PDF, DOCX, TXT) using the file uploader
3. Ask questions about Sai's experience
4. Choose between first-person or third-person responses

## Features

-   RAG-powered responses using uploaded documents
-   Automatic fallback to base model for general queries
-   Support for PDF, DOCX, and TXT files
-   First-person and third-person response modes
-   Real-time document processing
-   Secure remote access via Cloudflare tunnel

## Troubleshooting

1. If Ollama is not running:

```bash
ollama serve
```

2. To verify Ollama and model:

```bash
ollama list
```

3. To check logs:

```bash
tail -f streamlit.log
tail -f cloudflare.log
```

## License

MIT License
