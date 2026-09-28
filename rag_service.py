
import json
import re
import time
import uuid
import urllib.request

import torch

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
    FilterSelector,
)

import config


# ==================================================
# 1. MODEL CONFIGURATION
# ==================================================

config.resolve_backend()

EMBEDDING_MODEL = config.EMBEDDING_MODEL

LLM_MODEL = config.LLM_MODEL


# ==================================================
# 2. LOAD EMBEDDING MODEL
# ==================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

embedding_model.eval()


# ==================================================
# 3. LOAD LLM
# ==================================================

# A generation lock keeps concurrent requests from
# interleaving tokens from different questions.
generation_lock = __import__("threading").Lock()

tokenizer = None
llm = None

if config.BACKEND_NAME == "transformers":

    print(f"Loading LLM ({LLM_MODEL})...")

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL
    )

    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        dtype=config.DTYPE,
        low_cpu_mem_usage=True,
    )

    llm.eval()

    print("LLM loaded successfully.")

else:

    print(
        f"Using Ollama at {config.OLLAMA_URL} "
        f"with model {config.OLLAMA_MODEL}"
    )


print(config.describe())


# ==================================================
# 4. INITIALIZE QDRANT
# ==================================================

qdrant = QdrantClient(":memory:")

COLLECTION_NAME = "documents"


# ==================================================
# 5. CREATE QDRANT COLLECTION
# ==================================================

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE,
    ),
)


# ==================================================
# 7. CHUNKING FUNCTION
# ==================================================

HEADING_PATTERN = re.compile(
    r"^[A-Z][A-Z0-9 &/,'()\-\.]{2,60}$"
)


def is_heading(line: str) -> bool:
    """
    A short all-caps line with no sentence punctuation is
    treated as a section heading. This is what lets a chunk
    know it came from CERTIFICATIONS rather than from WORK
    EXPERIENCE, so the model can be told which part of the
    document it is reading.
    """

    candidate = line.strip()

    if not candidate or len(candidate.split()) > 8:
        return False

    if not HEADING_PATTERN.match(candidate):
        return False

    if candidate.endswith((".", ",", ";")):
        return False

    return sum(
        1
        for character in candidate
        if character.isupper()
    ) >= 2


def split_sections(text: str):
    """
    Split a document into (heading, body) sections.

    Falls back to a single untitled section when no
    headings are found, so plain prose still works.
    """

    sections = []

    heading = ""
    buffer = []

    for line in text.splitlines():

        stripped = line.strip()

        if is_heading(stripped):

            body = " ".join(buffer).strip()

            if body:
                sections.append(
                    (heading, body)
                )

            heading = stripped
            buffer = []
            continue

        buffer.append(stripped)

    body = " ".join(buffer).strip()

    if body:
        sections.append(
            (heading, body)
        )

    if not sections:

        stripped = " ".join(
            text.split()
        )

        if stripped:
            sections.append(
                ("", stripped)
            )

    return sections


def create_chunks(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
):

    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = (
        overlap
        if overlap is not None
        else config.CHUNK_OVERLAP
    )

    chunks = []

    for heading, body in split_sections(text):

        words = body.split()

        if not words:
            continue

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk = " ".join(
                words[start:end]
            ).strip()

            if chunk:

                chunks.append(
                    {
                        "text": chunk,
                        "section": heading,
                    }
                )

            if end >= len(words):
                break

            start = max(
                start + 1,
                end - overlap,
            )

    return chunks


# ==================================================
# 7b. DOCUMENT HELPERS
# ==================================================

def list_documents():
    """
    Return every indexed document together
    with its chunk count.
    """

    documents = {}

    offset = None

    while True:

        records, offset = qdrant.scroll(

            collection_name=COLLECTION_NAME,

            limit=256,

            offset=offset,

            with_payload=[
                "document_id",
                "source",
            ],

            with_vectors=False,
        )

        for record in records:

            payload = record.payload or {}

            document_id = payload.get("document_id")

            if not document_id:

                continue

            if document_id not in documents:

                documents[document_id] = {

                    "document_id": document_id,

                    "filename": payload.get(
                        "source",
                        "Unknown document",
                    ),

                    "chunks": 0,
                }

            documents[document_id]["chunks"] += 1

        if offset is None:

            break

    result = list(
        documents.values()
    )

    result.sort(
        key=lambda item: item[
            "filename"
        ].lower()
    )

    return result


def delete_document(document_id):
    """
    Remove every chunk belonging to a document.
    """

    qdrant.delete(

        collection_name=COLLECTION_NAME,

        points_selector=FilterSelector(

            filter=Filter(

                must=[

                    FieldCondition(

                        key="document_id",

                        match=MatchAny(
                            any=[document_id],
                        ),
                    )
                ]
            )
        ),
    )

    remaining = [
        document
        for document in list_documents()
        if document["document_id"] != document_id
    ]

    return {
        "message": "Document removed successfully.",
        "documents": remaining,
    }


# ==================================================
# 8. INGEST DOCUMENT
# ==================================================

def ingest_document(
    text: str,
    filename: str,
    document_id: str,
):

    print(
        f"\nIndexing document: {filename}"
    )


    # --------------------------------------------------
    # Create chunks
    # --------------------------------------------------

    chunks = create_chunks(
        text
    )


    if not chunks:

        return {
            "chunks": 0,
            "message": (
                "No chunks were created."
            ),
        }


    print(
        f"Created {len(chunks)} chunks."
    )


    # --------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------

    print(
        "Generating embeddings..."
    )

    embeddings = embedding_model.encode(
        [
            chunk["text"]
            for chunk in chunks
        ]
    )


    # --------------------------------------------------
    # Create Qdrant points
    # --------------------------------------------------

    points = []


    for index, (
        chunk,
        embedding,
    ) in enumerate(
        zip(chunks, embeddings)
    ):

        points.append(

            PointStruct(

                # Qdrant requires a valid UUID
                id=str(
                    uuid.uuid4()
                ),

                vector=embedding.tolist(),

                payload={

                    "text": chunk["text"],

                    "source": filename,

                    "section": chunk["section"],

                    "document_id": document_id,

                    "chunk_id": index,
                },
            )
        )


    # --------------------------------------------------
    # Store vectors
    # --------------------------------------------------

    qdrant.upsert(

        collection_name=COLLECTION_NAME,

        points=points,
    )


    print(
        f"Indexed {filename}: "
        f"{len(chunks)} chunks"
    )


    return {

        "chunks": len(chunks),

        "message": (
            "Document indexed successfully."
        ),
    }


# ==================================================
# 9. RETRIEVAL
# ==================================================

def retrieve(
    question: str,
    document_ids=None,
    top_k: int = None,
):
    """
    Find the passages most relevant to a question.

    Returns a list of dicts with the chunk text, its
    source and the similarity score.
    """

    top_k = top_k or config.TOP_K

    start = time.time()

    question_embedding = embedding_model.encode(
        question
    ).tolist()

    embedding_time = (
        time.time() - start
    )


    # --------------------------------------------------
    # Restrict the search to selected documents
    # --------------------------------------------------

    document_ids = [
        str(item)
        for item in (document_ids or [])
        if str(item).strip()
    ]


    query_filter = None

    if document_ids:

        query_filter = Filter(

            must=[

                FieldCondition(

                    key="document_id",

                    match=MatchAny(
                        any=document_ids,
                    ),
                )
            ]
        )


    retrieval_start = time.time()

    results = qdrant.query_points(

        collection_name=COLLECTION_NAME,

        query=question_embedding,

        limit=top_k,

        query_filter=query_filter,
    )

    retrieval_time = (
        time.time() - retrieval_start
    )


    passages = []

    for result in results.points:

        payload = result.payload

        passages.append(

            {
                "source": payload["source"],
                "section": (
                    payload.get("section")
                    or ""
                ),
                "document_id": payload["document_id"],
                "chunk_id": payload["chunk_id"],
                "score": float(result.score),
                "text": payload["text"],
            }
        )


    return {

        "passages": passages,

        "timing": {
            "embedding_seconds": round(embedding_time, 3),
            "retrieval_seconds": round(retrieval_time, 3),
        },
    }


# ==================================================
# 10. PROMPT BUILDING
# ==================================================

def strip_repetition(text: str, window: int = 60, limit: int = 3):
    """
    Cut an answer at the point where the model starts
    looping.

    Small models on CPU occasionally fall into a cycle
    even with penalties set. Rather than show the user the
    same sentence forty times, the answer is truncated at
    the second occurrence and the loop removed.
    """

    cleaned = (text or "").strip()

    if len(cleaned) < window * (limit + 1):
        return cleaned

    seen = {}

    for index in range(
        0,
        len(cleaned) - window,
    ):

        block = cleaned[index:index + window]

        count = seen.get(block, 0) + 1
        seen[block] = count

        if count >= limit:

            return cleaned[:index].rstrip()

    return cleaned


class LoopBreaker(StoppingCriteria):
    """
    Halt generation when the model falls into a loop.

    The usual defences are not usable for a RAG answer:

    - repetition_penalty down-weights tokens that must
      repeat when copying a fact, so "March 2023" became
      "March 2022" and "CGPA 8.7" became "8,7".
    - no_repeat_ngram_size blocks any six word repeat, but
      the right answer to a document question usually *is*
      a verbatim repeat, so it produced "Wroted" and
      "inested" mid sentence.

    Both change which tokens are allowed, so both corrupt
    the answer. This watches the text instead and stops
    the loop once it is clearly established, which leaves
    copying untouched.
    """

    def __init__(
        self,
        tokenizer,
        prompt_length: int,
        window: int = 40,
        limit: int = 4,
    ):

        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.window = window
        self.limit = limit
        self.tail = ""
        self.counts = {}

    def __call__(
        self,
        input_ids,
        scores,
        **kwargs,
    ):

        if input_ids.shape[-1] <= self.prompt_length:
            return False

        try:

            newest = input_ids[
                0,
                self.prompt_length:,
            ]

            piece = self.tokenizer.decode(
                newest[-4:],
                skip_special_tokens=True,
            )

        except Exception:

            return False


        self.tail = (
            self.tail + piece
        )[-self.window * 8:]


        if len(self.tail) < self.window:
            return False


        block = self.tail[-self.window:]

        count = self.counts.get(block, 0) + 1
        self.counts[block] = count

        return count >= self.limit


SYSTEM_PROMPT = (
    "You are a precise knowledge assistant that answers "
    "questions about the user's own documents."
    "\n\nRules:\n"
    "- Use ONLY the numbered context passages. Your own "
    "knowledge is not a source; if the passages do not "
    "contain something, say so plainly.\n"
    "- Copy names, titles, codes, dates, numbers and "
    "identifiers exactly as written. Never expand or "
    "correct an abbreviation from memory.\n"
    "- Scan every passage before answering, including the "
    "ones later in the list.\n"
    "- When the passages list several items, enumerate "
    "every one of them, including the last one in the "
    "list. Never stop at the first few and never collapse "
    "a list into a category name.\n"
    "- Carry over the details given for each item, such as "
    "grades, dates, durations, identifiers and scores, and "
    "add nothing of your own.\n"
    "- Use a markdown bullet list when there are several "
    "items, one item per line.\n"
    "- Reply with the answer only. No preamble, no "
    "restating the question."
)


def split_sentences(text: str):
    """
    Split a passage into sentences.

    Newlines from bullet lists and PDFs are treated as
    boundaries too, because a resume is mostly bullets and
    each bullet is its own fact.
    """

    flattened = re.sub(
        r"\s+",
        " ",
        text.replace("\n", " \n "),
    )

    parts = re.split(
        r"(?<=[.!?])\s+|\n",
        flattened,
    )

    sentences = []

    for part in parts:

        cleaned = part.strip(" \n\t-*•")

        if cleaned:
            sentences.append(cleaned)

    return sentences


def select_relevant_text(
    question: str,
    passages,
    word_budget: int = None,
):
    """
    Choose the sentences from the retrieved passages that
    best answer the question.

    This replaces truncating every passage to its first N
    words, which silently deleted anything late in a chunk
    (a CERTIFICATIONS section sitting after WORK
    EXPERIENCE never reached the model at all).

    Every sentence is scored against the question in one
    batch, then the best ones are kept in document order so
    the context still reads naturally.
    """

    word_budget = (
        word_budget
        if word_budget is not None
        else config.CONTEXT_WORD_BUDGET
    )

    candidates = []

    for passage_index, passage in enumerate(
        passages,
        start=1,
    ):

        sentences = split_sentences(
            passage["text"]
        )

        for position, sentence in enumerate(
            sentences
        ):

            candidates.append(
                {
                    "passage_index": passage_index,
                    "position": position,
                    "sentence": sentence,
                    "words": len(sentence.split()),
                }
            )


    if not candidates:

        return [], []


    try:

        vectors = embedding_model.encode(
            [
                item["sentence"]
                for item in candidates
            ],
            convert_to_tensor=True,
        )

        query_vector = embedding_model.encode(
            question,
            convert_to_tensor=True,
        )

        scores = torch.nn.functional.cosine_similarity(
            vectors,
            query_vector,
            dim=1,
        )

        for item, score in zip(
            candidates,
            scores.tolist(),
        ):

            item["score"] = float(score)

    except Exception:

        for item in candidates:

            item["score"] = 0.0


    # Always keep the opening sentence of each passage: it
    # is usually the topic sentence and gives the model
    # something to orient on.

    for item in candidates:

        if item["position"] == 0:

            item["score"] += 0.35


    selected = []

    used = 0

    for item in sorted(
        candidates,
        key=lambda entry: entry["score"],
        reverse=True,
    ):

        if used + item["words"] > word_budget:
            continue

        selected.append(item)
        used += item["words"]

        if used >= word_budget * 0.9:
            break


    if not selected:

        best = max(
            candidates,
            key=lambda entry: entry["score"],
        )

        selected = [best]
        used = best["words"]


    # --------------------------------------------------
    # Rebuild the context in document order
    # --------------------------------------------------

    by_passage = {}

    for item in selected:

        by_passage.setdefault(
            item["passage_index"],
            [],
        ).append(item)


    blocks = []

    kept_sources = []

    for passage_index in sorted(by_passage):

        passage = passages[
            passage_index - 1
        ]

        items = sorted(
            by_passage[passage_index],
            key=lambda entry: entry["position"],
        )

        label = passage["source"]

        section = (
            passage.get("section")
            or ""
        ).strip()

        if section:

            label = f"{label} - {section}"

        kept_sources.append(
            {
                "source": passage["source"],
                "section": section,
                "document_id": passage["document_id"],
                "chunk_id": passage["chunk_id"],
                "score": round(
                    passage["score"],
                    4,
                ),
                "text": " ".join(
                    " ".join(
                        item["sentence"]
                        for item in items
                    ).split()[:120]
                ),
            }
        )

        blocks.append(
            f"[Passage {len(kept_sources)} - {label}]\n"
            + " ".join(
                item["sentence"]
                for item in items
            )
        )


    return blocks, kept_sources


def build_messages(
    question: str,
    passages,
    history=None,
    return_sources=False,
):
    """
    Build the chat messages, including the last few
    turns so follow-up questions make sense.
    """

    context_parts, kept_sources = select_relevant_text(
        question,
        passages,
    )

    context = "\n\n".join(
        context_parts
    )


    user_turn = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the passages above. Quote names, "
        "titles, dates and numbers exactly as written, and "
        "include the extra details the passage gives about "
        "the answer, not only the headline. If the passages "
        "do not contain the answer, say that they do not."
    )


    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]


    # --------------------------------------------------
    # Replay recent turns for conversational context
    # --------------------------------------------------

    for turn in (history or [])[
        -config.HISTORY_TURNS * 2:
    ]:

        content = (turn or {}).get("content")

        role = (turn or {}).get("role")

        if not content:
            continue

        if role == "user":
            messages.append({
                "role": "user",
                "content": content,
            })
        elif role == "assistant":
            messages.append({
                "role": "assistant",
                "content": content,
            })


    messages.append({
        "role": "user",
        "content": user_turn,
    })


    if return_sources:

        return messages, kept_sources

    return messages


def build_sources(passages):
    """
    Trim passages down to what the UI displays.
    """

    return [
        {
            "source": passage["source"],
            "document_id": passage["document_id"],
            "chunk_id": passage["chunk_id"],
            "score": passage["score"],
            "text": " ".join(
                passage["text"].split()[:90]
            ),
        }
        for passage in passages
    ]


# ==================================================
# 11. GENERATION
# ==================================================

def stream_tokens(messages, stats=None):
    """
    Yield answer text as the model produces it.

    Works with both backends so the API does not care
    which one is active. When `stats` is a dict it is
    filled with real prompt/output token counts.
    """

    if config.BACKEND_NAME == "ollama":

        yield from _stream_ollama(messages)

        return


    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )


    arguments = {
        **inputs,
        "max_new_tokens": config.MAX_NEW_TOKENS,
        "do_sample": False,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "stopping_criteria": StoppingCriteriaList(
            [
                LoopBreaker(
                    tokenizer,
                    prompt_length=int(
                        inputs["input_ids"].shape[-1]
                    ),
                )
            ]
        ),
        "streamer": streamer,
    }


    prompt_tokens = int(
        inputs["input_ids"].shape[-1]
    )
    output_tokens = 0


    def run_generation():

        nonlocal output_tokens

        try:
            output = llm.generate(
                **arguments
            )

            output_tokens = int(
                output.shape[-1]
            ) - prompt_tokens
        except Exception:
            output_tokens = 0


    with generation_lock:

        thread = __import__("threading").Thread(
            target=run_generation
        )

        thread.start()

        for piece in streamer:

            if not piece:
                continue

            yield piece

        thread.join()

    if stats is not None:
        stats["prompt_tokens"] = prompt_tokens
        stats["output_tokens"] = max(
            0,
            output_tokens,
        )


def _stream_ollama(messages):
    """
    Stream from an Ollama server, which runs quantised
    llama.cpp kernels and is much faster on a CPU.
    """

    payload = json.dumps(
        {
            "model": config.OLLAMA_MODEL,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0,
                "num_predict": config.MAX_NEW_TOKENS,
            },
        }
    ).encode("utf-8")


    request = urllib.request.Request(
        f"{config.OLLAMA_URL}/api/chat",
        data=payload,
        headers={
            "Content-Type": "application/json",
        },
    )


    with urllib.request.urlopen(
        request,
        timeout=600,
    ) as response:

        for raw_line in response:

            line = raw_line.decode(
                "utf-8"
            ).strip()

            if not line:
                continue

            event = json.loads(line)

            piece = (
                event.get("message", {})
                .get("content", "")
            )

            if piece:
                yield piece

            if event.get("done"):
                break


# ==================================================
# 12. ASK QUESTION
# ==================================================

def ask_question_stream(
    question: str,
    document_ids=None,
    history=None,
):
    """
    Generator used by the streaming endpoint.

    Yields dicts: stage, delta, sources, timing, error.
    """

    total_start = time.time()

    yield {
        "type": "stage",
        "stage": "search",
    }


    retrieved = retrieve(
        question,
        document_ids=document_ids,
    )

    passages = retrieved["passages"]

    search_time = time.time() - total_start


    # --------------------------------------------------
    # No documents at all
    # --------------------------------------------------

    if not passages:

        if document_ids:

            answer = (
                "I could not find anything relevant "
                "to that question in the selected "
                "documents."
            )
        else:
            answer = (
                "No documents have been uploaded yet. "
                "Upload a PDF, DOCX or TXT file and ask "
                "your question again."
            )

        yield {
            "type": "done",
            "answer": answer,
            "sources": [],
            "timing": {
                "search_seconds": round(search_time, 2),
                "total_seconds": round(search_time, 2),
            },
        }

        return


    yield {
        "type": "stage",
        "stage": "generate",
    }


    messages, used_sources = build_messages(
        question,
        passages,
        history=history,
        return_sources=True,
    )


    first_token_time = None
    text_parts = []
    stats = {}

    generation_start = time.time()

    for piece in stream_tokens(messages, stats=stats):

        if first_token_time is None:
            first_token_time = time.time()

        text_parts.append(piece)

        yield {
            "type": "delta",
            "text": piece,
        }


    generation_time = time.time() - generation_start

    answer = strip_repetition(
        "".join(text_parts).strip()
    )


    output_tokens = stats.get("output_tokens") or 0
    prompt_tokens = stats.get("prompt_tokens") or 0

    if not output_tokens and answer:
        output_tokens = max(
            1,
            len(
                tokenizer.encode(
                    answer,
                    add_special_tokens=False,
                )
            ),
        )


    total_time = time.time() - total_start


    yield {
        "type": "done",
        "answer": answer,
        "sources": used_sources or build_sources(passages),
        "timing": {
            "search_seconds": round(search_time, 2),
            "first_token_seconds": (
                round(
                    first_token_time - total_start,
                    2,
                )
                if first_token_time
                else None
            ),
            "generation_seconds": round(generation_time, 2),
            "total_seconds": round(total_time, 2),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "tokens_per_second": (
                round(
                    output_tokens / generation_time,
                    2,
                )
                if generation_time > 0 and output_tokens
                else None
            ),
        },
        "retrieval": retrieved["timing"],
    }


def ask_question(
    question: str,
    document_ids=None,
    history=None,
):
    """
    Non-streaming variant kept for API compatibility.
    """

    answer = ""
    sources = []
    timing = {}

    for event in ask_question_stream(
        question,
        document_ids=document_ids,
        history=history,
    ):

        if event["type"] == "done":
            answer = event["answer"]
            sources = event["sources"]
            timing = event["timing"]

        elif event["type"] == "delta":
            answer += event["text"]


    return {
        "answer": answer,
        "sources": sources,
        "timing": timing,
    }
