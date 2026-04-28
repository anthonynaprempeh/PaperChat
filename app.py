"""
app.py — PaperChat: General AI Research Paper Chatbot
- Upload PDFs directly
- Paste arXiv/PubMed/direct PDF URLs
- Persistent index across sessions
- Multi-paper chat with citations
"""

import os
import io
import json
import base64
import hashlib
import shutil
import subprocess
import tempfile
import urllib.request
import urllib.parse
from pathlib import Path

import anthropic
import pdfplumber
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
INDEX_DIR         = "./storage"
IMAGES_DIR        = "./page_images"
PAPERS_META_FILE  = "./papers_meta.json"
UI_FILE           = "./paperchat_ui.html"
TOP_K             = 5
MAX_IMAGE_PAGES   = 2
COST_PER_QUERY    = 0.008
TOTAL_BUDGET      = 10.00
COUNTER_FILE      = "./query_counter.json"
MODEL             = "claude-haiku-4-5-20251001"
CHUNK_SIZE        = 512
CHUNK_OVERLAP     = 64
IMAGE_DPI         = 150

SYSTEM_PROMPT = """You are PaperChat, an AI research assistant that answers questions based exclusively on uploaded research papers.

You have access to text and visual content from the indexed papers. Always:
- Answer using only information from the provided context
- Cite the source paper and page number for every claim
- Be precise and academically rigorous
- If the answer is not in the papers, say so clearly — never fabricate

## Formatting:
- Use LaTeX for equations: inline $eq$ or display $$eq$$
- Use **bold** for key terms
- Use markdown tables for comparisons
- Use headings for structured answers
- Always include page and paper citations like: *(Smith et al., 2023, p.4)*

If the question cannot be answered from the papers, respond:
"I couldn't find information about that in the uploaded papers. Please check the papers directly or try rephrasing."
"""

# ── Setup ─────────────────────────────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def setup_embed():
    Settings.llm = None
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

setup_embed()

# ── Papers metadata ───────────────────────────────────────────────────────────
def load_papers_meta():
    if Path(PAPERS_META_FILE).exists():
        try:
            with open(PAPERS_META_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_papers_meta(meta):
    with open(PAPERS_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

# ── Index management ──────────────────────────────────────────────────────────
_index = None

def get_index():
    global _index
    if _index is not None:
        return _index
    if Path(INDEX_DIR).exists() and any(Path(INDEX_DIR).iterdir()):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
            _index = load_index_from_storage(storage_context)
            return _index
        except Exception:
            pass
    return None

def rebuild_index(all_nodes):
    global _index
    Path(INDEX_DIR).mkdir(exist_ok=True)
    _index = VectorStoreIndex(all_nodes, show_progress=False)
    _index.storage_context.persist(persist_dir=INDEX_DIR)
    return _index

# ── PDF ingestion ─────────────────────────────────────────────────────────────
def ingest_pdf_bytes(pdf_bytes: bytes, pdf_name: str) -> list:
    """Ingest a PDF from bytes, return list of TextNodes."""
    Path(IMAGES_DIR).mkdir(exist_ok=True)
    pdf_hash = hashlib.md5(pdf_bytes).hexdigest()[:8]
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in pdf_name)
    images_subdir = Path(IMAGES_DIR) / f"{pdf_hash}_{safe_name}"
    images_subdir.mkdir(exist_ok=True)

    nodes = []
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Write to temp file for pdftoppm
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        # Rasterize pages
        has_poppler = shutil.which("pdftoppm") is not None
        image_files = []
        if has_poppler:
            prefix = str(images_subdir / "page")
            result = subprocess.run(
                ["pdftoppm", "-jpeg", "-r", str(IMAGE_DPI), tmp_path, prefix],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                image_files = sorted(images_subdir.glob("page*.jpg"))

        # Extract text
        pages_text = []
        try:
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    pages_text.append((page.extract_text() or "").strip())
        except Exception as e:
            pages_text = [""] * max(len(image_files), 1)

        # Build nodes
        for page_idx, page_text in enumerate(pages_text):
            page_num = page_idx + 1
            img_path = ""
            if image_files:
                pad = len(str(len(pages_text)))
                matches = [f for f in image_files
                           if f.stem.endswith(str(page_num).zfill(pad))]
                if matches:
                    img_path = str(matches[0])

            is_abstract = (page_num <= 3 and page_text and
                           "abstract" in page_text.lower())
            if is_abstract:
                retrieval_text = f"[ABSTRACT — {pdf_name}, page {page_num}]\n{page_text}"
            else:
                retrieval_text = page_text or f"[Page {page_num} of {pdf_name} — visual only]"

            node = TextNode(
                text=retrieval_text,
                metadata={
                    "file_name":   pdf_name,
                    "page":        page_num,
                    "page_label":  str(page_num),
                    "image_path":  img_path,
                    "is_abstract": is_abstract,
                    "pdf_hash":    pdf_hash,
                },
                excluded_embed_metadata_keys=["image_path", "is_abstract", "pdf_hash"],
                excluded_llm_metadata_keys=["image_path", "is_abstract", "pdf_hash"],
            )
            nodes.append(node)

    finally:
        os.unlink(tmp_path)

    return nodes


def add_paper_to_index(nodes: list, pdf_name: str, pdf_hash: str,
                       source_url: str = ""):
    """Add new nodes to the existing index or create a new one."""
    global _index
    meta = load_papers_meta()

    if pdf_hash in meta:
        return False, "already_indexed"

    existing_nodes = []
    idx = get_index()
    if idx is not None:
        # Collect existing nodes from storage
        try:
            from llama_index.core import SimpleDirectoryReader
            storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
            existing_nodes = list(storage_context.docstore.docs.values())
        except Exception:
            existing_nodes = []

    all_nodes = existing_nodes + nodes
    rebuild_index(all_nodes)

    meta[pdf_hash] = {
        "name":       pdf_name,
        "pages":      len(nodes),
        "source_url": source_url,
        "hash":       pdf_hash,
    }
    save_papers_meta(meta)
    return True, "indexed"


# ── URL fetching ──────────────────────────────────────────────────────────────
def resolve_paper_url(url: str):
    """Try to resolve a URL to a direct PDF download."""
    url = url.strip()

    # arXiv
    if "arxiv.org" in url:
        # Convert abs to pdf
        url = url.replace("/abs/", "/pdf/")
        if not url.endswith(".pdf"):
            url = url + ".pdf"
        return url, None

    # PubMed Central - try to get PDF link
    if "ncbi.nlm.nih.gov/pmc" in url or "pubmedcentral" in url:
        return None, "PubMed Central articles require manual PDF download. Please download the PDF and upload it directly."

    # bioRxiv / medRxiv
    if "biorxiv.org" in url or "medrxiv.org" in url:
        if "/content/" in url and not url.endswith(".pdf"):
            url = url + ".full.pdf"
        return url, None

    # Direct PDF link
    if url.endswith(".pdf") or "pdf" in url.lower():
        return url, None

    # DOI
    if url.startswith("10.") or "doi.org" in url:
        return None, "DOI links often point to paywalled journals. Please download the PDF and upload it directly, or use an open-access version (e.g. from arXiv)."

    # Generic URL — try as-is
    return url, None


def fetch_pdf_from_url(url: str):
    """Fetch PDF bytes from a URL. Returns (bytes, filename, error)."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (research chatbot)"
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower() and not url.endswith(".pdf"):
                return None, None, "URL does not appear to be a PDF. Try a direct PDF link."
            data = resp.read()
            # Guess filename from URL
            filename = url.split("/")[-1].split("?")[0]
            if not filename.endswith(".pdf"):
                filename = filename + ".pdf"
            if not filename or filename == ".pdf":
                filename = "paper.pdf"
            return data, filename, None
    except Exception as e:
        return None, None, f"Could not fetch URL: {str(e)}"


# ── Counter helpers ───────────────────────────────────────────────────────────
def load_counter():
    if Path(COUNTER_FILE).exists():
        try:
            with open(COUNTER_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"queries": 0, "spent": 0.0}

def increment_counter():
    data = load_counter()
    data["queries"] += 1
    data["spent"] = round(data["queries"] * COST_PER_QUERY, 4)
    with open(COUNTER_FILE, "w") as f:
        json.dump(data, f)
    return data

# ── Vision message builder ────────────────────────────────────────────────────
def build_messages(question, nodes):
    content = [{"type": "text",
                "text": f"Question: {question}\n\nContext from research papers:\n"}]
    images_added = 0
    for node in nodes:
        meta     = node.metadata or {}
        filename = meta.get("file_name", "Unknown")
        page     = meta.get("page_label", meta.get("page", "?"))
        img_path = meta.get("image_path", "")
        content.append({"type": "text",
                         "text": f"\n--- {filename}, page {page} ---\n{node.text}\n"})
        if img_path and images_added < MAX_IMAGE_PAGES and Path(img_path).exists():
            try:
                with open(img_path, "rb") as f:
                    img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
                content.append({"type": "text",
                                 "text": f"[Page image: {filename}, p.{page}]"})
                content.append({"type": "image",
                                 "source": {"type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": img_b64}})
                images_added += 1
            except Exception:
                pass
    content.append({"type": "text",
                     "text": "\nAnswer using only the context and images above."})
    return [{"role": "user", "content": content}]

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if not Path(UI_FILE).exists():
        return HTMLResponse("<h1>paperchat_ui.html not found.</h1>", status_code=404)
    with open(UI_FILE) as f:
        return HTMLResponse(f.read())

@app.get("/api/papers")
async def get_papers():
    meta = load_papers_meta()
    papers = [{"name": v["name"], "pages": v["pages"],
                "source_url": v.get("source_url", ""),
                "hash": k}
               for k, v in meta.items()]
    return JSONResponse({"papers": papers})

@app.get("/api/status")
async def status():
    counter = load_counter()
    meta    = load_papers_meta()
    return {
        "queries":       counter["queries"],
        "spent":         counter["spent"],
        "remaining":     round(TOTAL_BUDGET - counter["spent"], 4),
        "budget":        TOTAL_BUDGET,
        "papers_count":  len(meta),
    }

@app.post("/api/upload")
async def upload_paper(file: UploadFile = File(...)):
    """Upload a PDF file and add it to the index."""
    if not file.filename.endswith(".pdf"):
        return JSONResponse({"error": "Only PDF files are supported."}, status_code=400)
    try:
        pdf_bytes = await file.read()
        pdf_hash  = hashlib.md5(pdf_bytes).hexdigest()[:8]
        meta      = load_papers_meta()
        if pdf_hash in meta:
            return JSONResponse({"status": "already_indexed",
                                  "name": meta[pdf_hash]["name"],
                                  "message": "This paper is already in your library."})
        nodes = ingest_pdf_bytes(pdf_bytes, file.filename)
        if not nodes:
            return JSONResponse({"error": "Could not extract content from PDF."},
                                  status_code=400)
        add_paper_to_index(nodes, file.filename, pdf_hash)
        return JSONResponse({"status": "indexed",
                              "name": file.filename,
                              "pages": len(nodes),
                              "message": f"Successfully indexed {len(nodes)} pages."})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/fetch_url")
async def fetch_url(request: Request):
    """Fetch a paper from a URL and add it to the index."""
    body = await request.json()
    url  = body.get("url", "").strip()
    if not url:
        return JSONResponse({"error": "No URL provided."}, status_code=400)

    resolved_url, error = resolve_paper_url(url)
    if error:
        return JSONResponse({"error": error}, status_code=400)

    pdf_bytes, filename, fetch_error = fetch_pdf_from_url(resolved_url)
    if fetch_error:
        return JSONResponse({"error": fetch_error}, status_code=400)

    try:
        pdf_hash = hashlib.md5(pdf_bytes).hexdigest()[:8]
        meta     = load_papers_meta()
        if pdf_hash in meta:
            return JSONResponse({"status": "already_indexed",
                                  "name": meta[pdf_hash]["name"],
                                  "message": "This paper is already in your library."})
        nodes = ingest_pdf_bytes(pdf_bytes, filename)
        if not nodes:
            return JSONResponse({"error": "Could not extract content from PDF."},
                                  status_code=400)
        add_paper_to_index(nodes, filename, pdf_hash, source_url=url)
        return JSONResponse({"status": "indexed",
                              "name": filename,
                              "pages": len(nodes),
                              "message": f"Successfully indexed {len(nodes)} pages."})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.delete("/api/papers/{pdf_hash}")
async def delete_paper(pdf_hash: str):
    """Remove a paper from the index."""
    meta = load_papers_meta()
    if pdf_hash not in meta:
        return JSONResponse({"error": "Paper not found."}, status_code=404)

    paper_name = meta[pdf_hash]["name"]
    del meta[pdf_hash]
    save_papers_meta(meta)

    # Rebuild index without this paper's nodes
    try:
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        existing_nodes  = list(storage_context.docstore.docs.values())
        kept_nodes = [n for n in existing_nodes
                      if n.metadata.get("pdf_hash") != pdf_hash]
        if kept_nodes:
            rebuild_index(kept_nodes)
        else:
            # No papers left — clear index
            global _index
            _index = None
            if Path(INDEX_DIR).exists():
                shutil.rmtree(INDEX_DIR)
    except Exception:
        pass

    return JSONResponse({"status": "deleted", "name": paper_name})

@app.post("/api/query")
async def query(request: Request):
    body     = await request.json()
    question = body.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "No question provided."}, status_code=400)

    counter = load_counter()
    if counter["spent"] >= TOTAL_BUDGET:
        return JSONResponse({
            "answer": "⛔ Budget exhausted. Please top up your Anthropic API credits.",
            "citations": [], "budget_exhausted": True
        })

    idx = get_index()
    if idx is None:
        return JSONResponse({
            "answer": "No papers have been indexed yet. Please upload a PDF or paste a paper URL to get started.",
            "citations": []
        })

    retriever = idx.as_retriever(similarity_top_k=TOP_K)
    nodes = retriever.retrieve(question)
    if not nodes:
        return JSONResponse({
            "answer": "I couldn't find relevant content in the indexed papers for that question.",
            "citations": []
        })

    citations = []
    seen = set()
    for node in nodes:
        meta     = node.metadata or {}
        filename = meta.get("file_name", "Unknown")
        page     = meta.get("page_label", meta.get("page", "?"))
        has_img  = bool(meta.get("image_path"))
        score    = round(node.score or 0, 2)
        key      = f"{filename}|{page}"
        if key not in seen:
            seen.add(key)
            citations.append({"file": filename, "page": page,
                               "score": score, "hasImg": has_img})

    try:
        messages = build_messages(question, nodes)
        response = anthropic_client.messages.create(
            model=MODEL, max_tokens=2048,
            system=SYSTEM_PROMPT, messages=messages,
        )
        answer = response.content[0].text
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    counter_data = increment_counter()
    return JSONResponse({
        "answer":    answer,
        "citations": citations,
        "budget": {
            "queries": counter_data["queries"],
            "spent":   counter_data["spent"],
            "left":    round(TOTAL_BUDGET - counter_data["spent"], 4),
        }
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
