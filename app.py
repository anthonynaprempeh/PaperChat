"""
app.py — PaperChat: General AI Research Paper Chatbot
Uses BM25 keyword search (no heavy ML models) — fits in 512MB RAM
"""

import os
import json
import base64
import hashlib
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import anthropic
import pdfplumber
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
IMAGES_DIR        = "./page_images"
PAPERS_META_FILE  = "./papers_meta.json"
PAPERS_TEXT_FILE  = "./papers_text.json"
UI_FILE           = "./paperchat_ui.html"
TOP_K             = 6
MAX_IMAGE_PAGES   = 2
COST_PER_QUERY    = 0.008
TOTAL_BUDGET      = 10.00
COUNTER_FILE      = "./query_counter.json"
MODEL             = "claude-haiku-4-5-20251001"
CHUNK_SIZE        = 600
CHUNK_OVERLAP     = 80
IMAGE_DPI         = 120
PORT              = int(os.environ.get("PORT", 8000))

SYSTEM_PROMPT = """You are PaperChat, an AI research assistant that answers questions based exclusively on uploaded research papers.

Always:
- Answer using only information from the provided context
- Cite the source paper and page number for every claim like: *(Smith et al., p.4)*
- Be precise and academically rigorous
- If the answer is not in the papers, say so clearly — never fabricate

## Formatting:
- Use LaTeX for equations: inline $eq$ or display $$eq$$
- Use **bold** for key terms
- Use markdown tables for comparisons
- Always include paper and page citations

If the question cannot be answered from the papers, respond:
"I couldn't find information about that in the uploaded papers. Please try rephrasing or upload a relevant paper."
"""

# ── Anthropic client ──────────────────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── In-memory paper store (BM25-style keyword search) ─────────────────────────
# Format: { pdf_hash: { "name": str, "pages": [ {"page": int, "text": str, "img": str} ] } }
_paper_store = {}

def load_paper_store():
    global _paper_store
    if Path(PAPERS_TEXT_FILE).exists():
        try:
            with open(PAPERS_TEXT_FILE) as f:
                _paper_store = json.load(f)
        except Exception:
            _paper_store = {}

def save_paper_store():
    with open(PAPERS_TEXT_FILE, "w") as f:
        json.dump(_paper_store, f)

load_paper_store()

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

# ── BM25-style keyword search ─────────────────────────────────────────────────
import math
import re
from collections import Counter

def tokenize(text):
    return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())

def bm25_score(query_tokens, doc_tokens, avg_dl, k1=1.5, b=0.75):
    doc_len  = len(doc_tokens)
    freq     = Counter(doc_tokens)
    score    = 0.0
    for term in set(query_tokens):
        tf = freq.get(term, 0)
        if tf == 0:
            continue
        idf = math.log(1 + 1)  # simplified IDF
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / max(avg_dl, 1)))
        score += idf * tf_norm
    return score

def search_papers(question, top_k=TOP_K):
    """Search all indexed pages using BM25 keyword matching."""
    if not _paper_store:
        return []

    query_tokens = tokenize(question)
    if not query_tokens:
        return []

    # Compute average doc length
    all_pages = []
    for pdf_hash, paper in _paper_store.items():
        for page in paper.get("pages", []):
            tokens = tokenize(page.get("text", ""))
            all_pages.append({
                "hash":     pdf_hash,
                "name":     paper["name"],
                "page":     page["page"],
                "text":     page["text"],
                "img":      page.get("img", ""),
                "tokens":   tokens,
            })

    if not all_pages:
        return []

    avg_dl = sum(len(p["tokens"]) for p in all_pages) / len(all_pages)

    # Score all pages
    scored = []
    for p in all_pages:
        score = bm25_score(query_tokens, p["tokens"], avg_dl)
        if score > 0:
            scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]

# ── PDF ingestion ─────────────────────────────────────────────────────────────
def ingest_pdf_bytes(pdf_bytes: bytes, pdf_name: str) -> tuple:
    """Ingest PDF bytes. Returns (pdf_hash, page_count)."""
    pdf_hash = hashlib.md5(pdf_bytes).hexdigest()[:10]

    if pdf_hash in _paper_store:
        return pdf_hash, len(_paper_store[pdf_hash].get("pages", []))

    Path(IMAGES_DIR).mkdir(exist_ok=True)
    safe_name   = "".join(c if c.isalnum() or c in "._-" else "_" for c in pdf_name)
    images_subdir = Path(IMAGES_DIR) / f"{pdf_hash}_{safe_name}"
    images_subdir.mkdir(exist_ok=True)

    pages = []

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        # Rasterize pages (optional — skip if poppler not available)
        image_files = []
        if shutil.which("pdftoppm"):
            prefix = str(images_subdir / "page")
            result = subprocess.run(
                ["pdftoppm", "-jpeg", "-r", str(IMAGE_DPI), tmp_path, prefix],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                image_files = sorted(images_subdir.glob("page*.jpg"))

        # Extract text
        pages_text = []
        try:
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    pages_text.append((page.extract_text() or "").strip())
        except Exception:
            pages_text = [""] * max(len(image_files), 1)

        n_pages = len(pages_text)
        for page_idx, page_text in enumerate(pages_text):
            page_num = page_idx + 1
            img_path = ""
            if image_files:
                pad = len(str(n_pages))
                matches = [f for f in image_files
                           if f.stem.endswith(str(page_num).zfill(pad))]
                if matches:
                    img_path = str(matches[0])

            pages.append({
                "page": page_num,
                "text": page_text,
                "img":  img_path,
            })

    finally:
        os.unlink(tmp_path)

    _paper_store[pdf_hash] = {"name": pdf_name, "pages": pages}
    save_paper_store()
    return pdf_hash, len(pages)

# ── URL resolver ──────────────────────────────────────────────────────────────
def resolve_paper_url(url: str):
    url = url.strip()
    if "arxiv.org" in url:
        url = url.replace("/abs/", "/pdf/")
        if not url.endswith(".pdf"):
            url += ".pdf"
        return url, None
    if "biorxiv.org" in url or "medrxiv.org" in url:
        if "/content/" in url and not url.endswith(".pdf"):
            url += ".full.pdf"
        return url, None
    if "ncbi.nlm.nih.gov/pmc" in url:
        return None, "PubMed Central requires manual PDF download. Please upload the PDF directly."
    if url.startswith("10.") or "doi.org" in url:
        return None, "DOI links often point to paywalled journals. Please upload the PDF directly or use an arXiv link."
    return url, None

def fetch_pdf_from_url(url: str):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data     = resp.read()
            filename = url.split("/")[-1].split("?")[0]
            if not filename.endswith(".pdf"):
                filename += ".pdf"
            if filename == ".pdf":
                filename = "paper.pdf"
            return data, filename, None
    except Exception as e:
        return None, None, f"Could not fetch URL: {str(e)}"

# ── Counter ───────────────────────────────────────────────────────────────────
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
    data["spent"]    = round(data["queries"] * COST_PER_QUERY, 4)
    with open(COUNTER_FILE, "w") as f:
        json.dump(data, f)
    return data

# ── Message builder ───────────────────────────────────────────────────────────
def build_messages(question, results):
    content = [{"type": "text",
                "text": f"Question: {question}\n\nContext from research papers:\n"}]
    images_added = 0
    for r in results:
        content.append({"type": "text",
                         "text": f"\n--- {r['name']}, page {r['page']} ---\n{r['text']}\n"})
        if r.get("img") and images_added < MAX_IMAGE_PAGES and Path(r["img"]).exists():
            try:
                with open(r["img"], "rb") as f:
                    img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
                content.append({"type": "text",
                                 "text": f"[Page image: {r['name']}, p.{r['page']}]"})
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
                   allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if not Path(UI_FILE).exists():
        return HTMLResponse("<h1>paperchat_ui.html not found.</h1>", status_code=404)
    with open(UI_FILE) as f:
        return HTMLResponse(f.read())

@app.get("/api/papers")
async def get_papers():
    meta   = load_papers_meta()
    papers = [{"name": v["name"], "pages": v["pages"],
                "source_url": v.get("source_url", ""), "hash": k}
               for k, v in meta.items()]
    return JSONResponse({"papers": papers})

@app.get("/api/status")
async def status():
    counter = load_counter()
    meta    = load_papers_meta()
    return {"queries": counter["queries"], "spent": counter["spent"],
            "remaining": round(TOTAL_BUDGET - counter["spent"], 4),
            "budget": TOTAL_BUDGET, "papers_count": len(meta)}

@app.post("/api/upload")
async def upload_paper(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"error": "Only PDF files are supported."}, status_code=400)
    try:
        pdf_bytes = await file.read()
        pdf_hash  = hashlib.md5(pdf_bytes).hexdigest()[:10]
        meta      = load_papers_meta()
        if pdf_hash in meta:
            return JSONResponse({"status": "already_indexed",
                                  "name": meta[pdf_hash]["name"],
                                  "message": "This paper is already in your library."})
        pdf_hash, page_count = ingest_pdf_bytes(pdf_bytes, file.filename)
        meta[pdf_hash] = {"name": file.filename, "pages": page_count, "source_url": ""}
        save_papers_meta(meta)
        return JSONResponse({"status": "indexed", "name": file.filename,
                              "pages": page_count,
                              "message": f"Indexed {page_count} pages."})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/fetch_url")
async def fetch_url(request: Request):
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
        pdf_hash  = hashlib.md5(pdf_bytes).hexdigest()[:10]
        meta      = load_papers_meta()
        if pdf_hash in meta:
            return JSONResponse({"status": "already_indexed",
                                  "name": meta[pdf_hash]["name"],
                                  "message": "Already in library."})
        pdf_hash, page_count = ingest_pdf_bytes(pdf_bytes, filename)
        meta[pdf_hash] = {"name": filename, "pages": page_count, "source_url": url}
        save_papers_meta(meta)
        return JSONResponse({"status": "indexed", "name": filename,
                              "pages": page_count,
                              "message": f"Indexed {page_count} pages."})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.delete("/api/papers/{pdf_hash}")
async def delete_paper(pdf_hash: str):
    meta = load_papers_meta()
    if pdf_hash not in meta:
        return JSONResponse({"error": "Paper not found."}, status_code=404)
    name = meta[pdf_hash]["name"]
    del meta[pdf_hash]
    save_papers_meta(meta)
    if pdf_hash in _paper_store:
        del _paper_store[pdf_hash]
        save_paper_store()
    return JSONResponse({"status": "deleted", "name": name})

@app.post("/api/query")
async def query(request: Request):
    body     = await request.json()
    question = body.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "No question provided."}, status_code=400)
    counter = load_counter()
    if counter["spent"] >= TOTAL_BUDGET:
        return JSONResponse({"answer": "⛔ Budget exhausted.",
                              "citations": [], "budget_exhausted": True})
    if not _paper_store:
        return JSONResponse({
            "answer": "No papers have been indexed yet. Upload a PDF or paste a URL to get started.",
            "citations": []})
    results = search_papers(question, top_k=TOP_K)
    if not results:
        return JSONResponse({
            "answer": "I couldn't find relevant content in the indexed papers for that question.",
            "citations": []})
    citations = []
    seen = set()
    for r in results:
        key = f"{r['name']}|{r['page']}"
        if key not in seen:
            seen.add(key)
            citations.append({"file": r["name"], "page": r["page"],
                               "score": 0.9, "hasImg": bool(r.get("img"))})
    try:
        messages = build_messages(question, results)
        response = anthropic_client.messages.create(
            model=MODEL, max_tokens=2048,
            system=SYSTEM_PROMPT, messages=messages)
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
    uvicorn.run(app, host="0.0.0.0", port=PORT)
