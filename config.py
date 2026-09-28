"""
Runtime configuration for the AI Knowledge Assistant.

The LLM is the bottleneck on a CPU-only machine, so every
knob that affects speed lives here and can be overridden
with environment variables.
"""

import os
import sys

import torch


# ==================================================
# 1. ENVIRONMENT HELPERS
# ==================================================

def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ==================================================
# 2. HARDWARE DETECTION
# ==================================================

CUDA_AVAILABLE = torch.cuda.is_available()

DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"


def physical_cores():
    """
    Physical cores, not logical ones. Hyper-threading
    does not help dense CPU inference.
    """

    try:
        import psutil

        count = psutil.cpu_count(logical=False)

        if count:
            return int(count)
    except Exception:
        pass

    return max(1, (os.cpu_count() or 4) // 2)


CPU_CORES = physical_cores()


# ==================================================
# 3. MODEL SELECTION
# ==================================================

# Measured on this 4-core Tiger Lake CPU (float32, 4 threads),
# streaming a grounded answer from `documents/exchange_rates.txt`:
#
#   Qwen2.5-3B   ~0.14-0.33 tok/s -> 5-12 minutes. Unusable.
#   Qwen2.5-1.5B ~1.4 tok/s, first text in 2-8s, ~12-30s total.
#                  Answers were correctly grounded, including
#                  "the passage does not mention a schedule".
#   Qwen2.5-0.5B ~4 tok/s, ~3-10s total, but it invents facts
#                  (called JMD "Japanese Yen", claimed the docs
#                  state a publishing schedule).
#
# 1.5B is the default because a knowledge assistant that
# fabricates is worse than a slow one. Set
# LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct to trade accuracy for speed.
CPU_DEFAULT_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
GPU_DEFAULT_LLM = "Qwen/Qwen2.5-3B-Instruct"

LLM_MODEL = os.environ.get("LLM_MODEL") or (
    GPU_DEFAULT_LLM if CUDA_AVAILABLE else CPU_DEFAULT_LLM
)

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

LLM_BACKEND = os.environ.get("LLM_BACKEND", "auto").strip().lower()

OLLAMA_URL = os.environ.get(
    "OLLAMA_URL",
    "http://localhost:11434",
)

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL") or (
    "qwen2.5:3b"
    if CUDA_AVAILABLE
    else "qwen2.5:1.5b"
)

# Only useful on GPU; half precision is emulated and
# therefore SLOWER on most CPUs.
DTYPE = (
    torch.float16
    if CUDA_AVAILABLE
    else torch.float32
)


# ==================================================
# 4. GENERATION SETTINGS
# ==================================================

# Answers are allowed to be complete. A resume question
# ("what certifications do I have?") needs every name, date
# and credential id listed, so this is a list of facts rather
# than a single sentence.
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 384)

# Top passages handed to the LLM. Chunks are small and
# heading aware, so a few more of them costs little and
# stops relevant sections being missed.
TOP_K = env_int("TOP_K", 6)

# Word budget for the context built from the retrieved
# passages. This is NOT a per-chunk truncation: the most
# relevant sentences are chosen across every passage, so
# nothing is dropped for being late in a chunk.
CONTEXT_WORD_BUDGET = env_int("CONTEXT_WORD_BUDGET", 240)

# Chunk size used when indexing a document. Smaller chunks
# mean a matched passage is about one section of the
# document rather than a page of it.
CHUNK_SIZE = env_int("CHUNK_SIZE", 110)
CHUNK_OVERLAP = env_int("CHUNK_OVERLAP", 20)

# Number of previous turns replayed to the LLM.
HISTORY_TURNS = env_int("HISTORY_TURNS", 2)

# Set once at import so every request uses the tuned value.
if not CUDA_AVAILABLE:
    torch.set_num_threads(
        env_int("LLM_THREADS", max(1, CPU_CORES))
    )


# ==================================================
# 5. BACKEND RESOLUTION
# ==================================================

def ollama_available():
    """
    True when an Ollama server is reachable.
    """

    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"{OLLAMA_URL}/api/tags",
            timeout=1.5,
        ) as response:
            return response.status == 200
    except Exception:
        return False


def resolve_backend():
    """
    Decide which inference runtime to use.

    Ollama runs quantised llama.cpp kernels, which are far
    faster than transformers on a CPU. When it is installed
    we use it automatically, otherwise we fall back to
    transformers.
    """

    global BACKEND_NAME, ACTIVE_MODEL

    if LLM_BACKEND == "ollama":
        BACKEND_NAME = "ollama"
    elif LLM_BACKEND == "transformers":
        BACKEND_NAME = "transformers"
    elif ollama_available():
        BACKEND_NAME = "ollama"
    else:
        BACKEND_NAME = "transformers"

    ACTIVE_MODEL = (
        OLLAMA_MODEL if BACKEND_NAME == "ollama" else LLM_MODEL
    )

    return BACKEND_NAME


BACKEND_NAME = "transformers"
ACTIVE_MODEL = LLM_MODEL


def describe():
    """
    Human readable startup summary.
    """

    backend = BACKEND_NAME
    model = ACTIVE_MODEL

    lines = [
        f"device         : {DEVICE} "
        f"(threads: {torch.get_num_threads()}, cores: {CPU_CORES})",
        f"llm backend    : {backend}",
        f"llm model      : {model}",
        f"dtype          : {DTYPE}",
        f"retrieval      : top {TOP_K} chunks, "
        f"{CONTEXT_WORD_BUDGET}-word context budget",
        f"max new tokens : {MAX_NEW_TOKENS}",
    ]

    if not CUDA_AVAILABLE and backend == "transformers":
        lines.append(
            "speed note     : CPU-only, expect first text in "
            "a few seconds and a full answer in 10-30s. Install "
            "Ollama (ollama.com) + `ollama pull qwen2.5:1.5b` "
            "for quantised kernels, or set "
            "LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct for faster "
            "but less reliable answers"
        )

    return "\n".join(lines)
