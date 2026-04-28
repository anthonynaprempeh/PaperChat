# PaperChat — AI Research Paper Assistant

An AI-powered chatbot that lets you upload research papers and ask questions about them. Get cited, accurate answers grounded exclusively in your papers.

## Features

- **Upload PDFs** directly via drag-and-drop or file picker
- **Paste URLs** from arXiv, bioRxiv, or any direct PDF link
- **Persistent library** — papers are remembered across sessions
- **Multi-paper chat** — ask questions across all indexed papers at once
- **Citations with page numbers** for every answer
- **Equation rendering** via MathJax
- **Figure description** — Claude can see and describe figures in your papers
- **Delete papers** from your library anytime

## Supported sources

| Source | Support |
|---|---|
| arXiv | ✅ Auto-fetch |
| bioRxiv / medRxiv | ✅ Auto-fetch |
| Direct PDF URLs | ✅ Auto-fetch |
| Upload PDF | ✅ Always works |
| Paywalled journals | ❌ Download PDF first |

## Setup

### Step 1 — Install Python 3.10+
```bash
python3 --version
```

### Step 2 — Install poppler (for figure extraction)
```bash
brew install poppler   # Mac
# or: sudo apt install poppler-utils  (Linux)
```

### Step 3 — Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — Set your Anthropic API key
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Step 6 — Run the app
```bash
python app.py
```

Open your browser at **http://localhost:8000**

## Usage

1. **Add a paper** — drag a PDF onto the sidebar, click to browse, or paste an arXiv/bioRxiv link
2. **Ask questions** — type in the chat box and press Enter
3. **Get cited answers** — every response includes the source paper and page number

## Author

Anthony Prempeh
