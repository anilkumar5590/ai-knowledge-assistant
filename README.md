# AI Knowledge Assistant

[![GitHub](https://img.shields.io/badge/github-anilkumar5590%2Fai--knowledge--assistant-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/anilkumar5590/ai-knowledge-assistant)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

A Retrieval-Augmented Generation (RAG) web app. Upload a PDF, DOCX or TXT file,
ask a question in plain language, and get an answer that is grounded in your own
document — with the exact source passages shown underneath.

Everything runs **locally on your machine**. No document text ever leaves your
computer, and no API key is required.

**Repository:** <https://github.com/anilkumar5590/ai-knowledge-assistant>

---

## Table of Contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Step-by-step setup](#step-by-step-setup)
- [Running the app](#running-the-app)
- [Using the app](#using-the-app)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Performance notes](#performance-notes)
- [Troubleshooting](#troubleshooting)
- [Security note](#security-note)
- [Feedback](#feedback)
- [Follow us](#follow-us)
- [License](#license)

---

## What it does

| Feature | Detail |
|---|---|
| Upload | PDF, DOCX, TXT (drag-and-drop or file picker) |
| Q&A | Natural language questions, answered from your documents only |
| Grounded answers | Cites the source passage and section for every answer |
| Streaming | Answers stream token-by-token over Server-Sent Events |
| Conversation | Follow-up questions understand prior turns |
| Document scope | Search all documents, or filter to a selected subset |
| Manage | List, select and delete indexed documents |
| Privacy | Fully local. Qdrant runs in-memory, so nothing is persisted |

---

## How it works

```
  Upload PDF/DOCX/TXT
          |
          v
  [1] Text extraction      document_loader.py
      PyMuPDF / python-docx / plain read
          |
          v
  [2] Chunking             rag_service.py
      Heading-aware split, 110 words + 20 overlap
          |
          v
  [3] Embedding            all-MiniLM-L6-v2 (384 dims)
          |
          v
  [4] Vector store         Qdrant, in-memory, cosine distance
          |
          |
   ------ User asks a question ------
          |
          v
  [5] Retrieval            question embedded, top-6 chunks found
      + filtering           (optionally restricted to selected docs)
          |
          v
  [6] Context building     every sentence scored against the
                           question, best ones kept within a
                           240-word budget
          |
          v
  [7] Generation            Qwen2.5-1.5B-Instruct (CPU)
      + streaming           answers strictly from the passages
          |
          v
  Answer + cited sources
```

Two details worth knowing, because they are why the answers are trustworthy:

- **The model is not allowed to guess.** The system prompt instructs it to use
  only the retrieved passages and to say so plainly when the answer is absent.
  Small models drift otherwise, so the rule is stated explicitly.
- **The context is selected, not truncated.** Instead of cutting each chunk to
  its first N words (which silently deletes anything late in a chunk), every
  sentence from every retrieved passage is scored against the question and the
  best ones are kept in original order.

---

## Requirements

**Required**

- Python 3.11 or newer
- ~6 GB free disk space (PyTorch plus downloaded models)
- 4 GB RAM minimum, 8 GB+ recommended

**Optional but strongly recommended**

- An NVIDIA GPU with 6 GB+ VRAM. Without one the app still works, just slower.
- [Ollama](https://ollama.com) — makes CPU inference dramatically faster.

> **Note on first run:** the embedding model (~90 MB) and the LLM (~3 GB) are
> downloaded automatically from Hugging Face on first start. This step needs a
> working internet connection and happens only once.

---

## Step-by-step setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/anilkumar5590/ai-knowledge-assistant.git
cd ai-knowledge-assistant
```

> Already have the code? Just `cd` into the project folder and continue to Step 2.

### Step 2 — Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(<environment name>)` at the start of your prompt.

> If PowerShell blocks activation with an "execution policy" error, either run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activation
> and use the `.\venv\Scripts\python.exe` path directly in every command below.

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This is the longest step (~2 GB of downloads). Let it finish.

### Step 4 — Verify the installation

```bash
python -c "import torch, transformers, sentence_transformers, qdrant_client, fitz, docx, fastapi; print('All dependencies OK')"
```

Expected output: `All dependencies OK`

### Step 5 — Optional: install Ollama for much faster CPU answers

Only relevant if you do **not** have a GPU. Ollama runs quantised llama.cpp
kernels and is typically 5–10x faster than PyTorch on a CPU.

1. Download and install Ollama from <https://ollama.com>
2. Pull the model:

   ```bash
   ollama pull qwen2.5:1.5b
   ```

3. Leave Ollama running. The app detects it automatically at startup and
   switches to it without any configuration.

   To skip detection and force one backend or the other, see
   [Configuration](#configuration).

### Step 6 — Start the app

```bash
uvicorn main:app --reload
```

Wait for this line:

```
Loading embedding model...
Loading LLM (Qwen/Qwen2.5-1.5B-Instruct)...
LLM loaded successfully.
device         : cpu (threads: 8, cores: 4)
llm backend    : transformers
...
```

### Step 7 — Open the app

Go to:

```
http://127.0.0.1:8000
```

You should see the chat interface. Confirm the backend is connected by checking
the status indicator in the corner.

### Step 8 — Upload a document and ask a question

1. Drag a PDF, DOCX or TXT file onto the dropzone (or click to browse).
2. Wait for "Indexed successfully" and the document count to increase.
3. Type a question, e.g. *"What certifications do I have?"*
4. Read the answer and expand the sources below it to see which passages were used.

---

## Running the app

### Start (normal use)

```bash
uvicorn main:app --reload
```

### Start (production-style, no auto-reload)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Start on a different port

```bash
uvicorn main:app --port 8080 --reload
```

### Stop the server

Press `Ctrl + C` in the terminal running uvicorn.

> `--reload` restarts the server when you edit a `.py` file, which also means
> the models reload. Use it while developing, drop it for normal use.

---

## Using the app

**Uploading**

- Drag and drop a file, or click the dropzone to open a file picker.
- Multiple files can be selected at once.
- Re-uploading a file with the same name **replaces** the old version rather than
  creating a duplicate, so you can edit a document and re-upload it safely.
- Accepted formats: `.pdf`, `.docx`, `.txt`. Anything else is rejected.

**Asking questions**

- Click **+ New chat** to start a fresh conversation.
- Past conversations are listed in the sidebar and can be reopened or cleared.
- The last few turns are replayed to the model, so "What about the dates?" works
  as a follow-up.

**Scoping to specific documents**

- Tick the checkboxes next to documents in the sidebar to search only those.
- **Select all** / **Clear** toggle the whole set.

**Deleting**

- Hover a document and click its delete control. Its uploaded file and all
  indexed chunks are removed.

---

## Configuration

Every setting is an environment variable. On Windows PowerShell:

```powershell
$env:LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
```

On macOS / Linux:

```bash
export LLM_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
```

### Core settings

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BACKEND` | `auto` | `auto`, `transformers`, or `ollama`. `auto` prefers Ollama if it is running. |
| `LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Hugging Face model when using `transformers`. |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server address. |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Model tag used with Ollama. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model. Keep 384-dim output if changing the collection size. |
| `LLM_THREADS` | physical core count | CPU threads for inference. |

### Retrieval settings

| Variable | Default | Purpose |
|---|---|---|
| `TOP_K` | `6` | Number of chunks retrieved per question. |
| `CHUNK_SIZE` | `110` | Words per chunk when indexing. |
| `CHUNK_OVERLAP` | `20` | Overlapping words between chunks. |
| `CONTEXT_WORD_BUDGET` | `240` | Word budget for the assembled context. |
| `HISTORY_TURNS` | `2` | Previous turns replayed for follow-ups. |

### Generation settings

| Variable | Default | Purpose |
|---|---|---|
| `MAX_NEW_TOKENS` | `384` | Maximum answer length. Raise for long list-style answers. |

### Useful combinations

**Fast on CPU, acceptable accuracy:**

```bash
$env:LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
$env:MAX_NEW_TOKENS = "256"
```

**Faster and still accurate on CPU (recommended):**

```bash
ollama pull qwen2.5:1.5b
$env:LLM_BACKEND = "ollama"
```

**Best quality on a GPU:**

```bash
$env:LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
```

### Check the active configuration

```
GET /health
```

```json
{
  "status": "ok",
  "documents": 3,
  "backend": "ollama",
  "model": "qwen2.5:1.5b",
  "device": "cpu",
  "streaming": true
}
```

The same summary is printed to the terminal at startup.

---

## API reference

Interactive docs: <http://127.0.0.1:8000/docs>

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web interface |
| `GET` | `/health` | Status, active backend, model, document count |
| `POST` | `/ask` | Ask a question, returns the full answer |
| `POST` | `/ask/stream` | Ask a question, streams tokens as Server-Sent Events |
| `POST` | `/upload` | Upload and index a document |
| `GET` | `/documents` | List indexed documents with chunk counts |
| `DELETE` | `/documents/{id}` | Delete a document and all its chunks |

### Example: ask a question

```bash
curl -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" `
  -d '{"question": "What certifications do I have?"}'
```

```json
{
  "answer": "- AWS Certified Solutions Architect, issued March 2023, credential ID AWS-88412\n- Google Cloud Professional, issued June 2022",
  "sources": [
    {
      "source": "resume.pdf",
      "section": "CERTIFICATIONS",
      "document_id": "33b991f3-...",
      "chunk_id": 7,
      "score": 0.81,
      "text": "CERTIFICATIONS AWS Certified Solutions Architect, issued March 2023..."
    }
  ],
  "timing": {
    "search_seconds": 0.31,
    "first_token_seconds": 1.84,
    "generation_seconds": 11.2,
    "total_seconds": 13.05,
    "output_tokens": 58,
    "tokens_per_second": 5.18
  }
}
```

### Example: restrict to specific documents

```json
{
  "question": "What are the payment terms?",
  "document_ids": ["33b991f3-5934-4d9f-9ace-2a2ff513516d"]
}
```

### Streaming event types

`/ask/stream` emits Server-Sent Events, one JSON object per `data:` line:

| Type | Meaning |
|---|---|
| `stage` | Progress marker, `stage` is `search` or `generate` |
| `delta` | A fragment of answer text, append as it arrives |
| `done` | Final answer, sources, timings and token counts |
| `error` | Something failed; `message` explains it |

The stream always terminates with the literal line `data: [DONE]`.

---

## Performance notes

Measured on a 4-core Tiger Lake CPU with no GPU, streaming a grounded answer:

| Model | Speed | Notes |
|---|---|---|
| Qwen2.5-3B | ~0.14–0.33 tok/s | 5–12 minutes per answer. Not usable. |
| **Qwen2.5-1.5B** | ~1.4 tok/s | First text in 2–8s, ~12–30s total. **Default.** Grounded and reliable. |
| Qwen2.5-0.5B | ~4 tok/s | 3–10s total, but invents facts. Only for testing. |

The 1.5B model is the default deliberately: an assistant that fabricates an
answer is worse than a slow one. If answers feel too slow, install Ollama rather
than downgrading the model.

The first request after startup is the slowest, because the embedding model and
LLM are loaded into memory. Subsequent requests are faster.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'fitz'`**
PyMuPDF is not installed. Run `pip install -r requirements.txt` inside the
activated environment. Note the import name is `fitz` but the package is
`pymupdf`.

**`ModuleNotFoundError: No module named 'No module named sentence_transformers'`**
You are not running the activated environment. Re-activate the venv (Step 2) or
call `.\venv\Scripts\python.exe` explicitly.

**`UploadFile` errors or uploads silently fail**
`python-multipart` is missing. It is required for any FastAPI file upload:
`pip install python-multipart`

**`Address already in use`**
Another process holds port 8000. Either stop it, or use another port:
`uvicorn main:app --port 8080 --reload`

**Answers are very slow**
Expected on CPU with the `transformers` backend. Install Ollama and pull
`qwen2.5:1.5b`, or use the 0.5B model if accuracy matters less than speed.

**The model download fails or times out**
The first run pulls several GB from Hugging Face. Check your internet
connection and retry; downloads resume from cache.

**Answer says it has no information even though the document contains it**
The relevant section may not have been retrieved. Try rephrasing with words that
appear literally in the document, raise `TOP_K`, or raise
`CONTEXT_WORD_BUDGET`.

**The model repeats the same sentence over and over**
Small models can loop. The app has a loop breaker, but lowering `MAX_NEW_TOKENS`
also helps.

**Uploaded PDF yields no text**
It is probably a scanned image. This app does not OCR; the text layer must
already exist in the PDF.

**Documents disappear after a restart**
This is expected. Qdrant runs in-memory, so the index is rebuilt as you upload.
Re-upload after restarting the server.

---

## Security note

- **Do not expose this app to the internet as-is.** It binds locally, has no
  authentication, and allows anyone who can reach it to read your indexed
  documents.
- **Never commit `.env`.** It is already listed in `.gitignore`.
- If you ever add an API key to `.env`, treat it as compromised once it has been
  shared or committed — rotate it rather than deleting the line.
- Uploads are limited to `.pdf`, `.docx`, `.txt` by extension. If you accept
  files from untrusted users, add a content check and a file size limit.

---

## Feedback

If you have any feedback, please reach out to us at
[konathalaanilkumar143@gmail.com](mailto:konathalaanilkumar143@gmail.com).

Found a bug or have a feature idea? Open an issue on GitHub:
<https://github.com/anilkumar5590/ai-knowledge-assistant/issues>

---

## Follow us

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anilkumarkonathala/)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE)
file for details.
