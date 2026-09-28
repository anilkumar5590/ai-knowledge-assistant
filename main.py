# from pathlib import Path
# import shutil
# import uuid

# from fastapi import (
#     FastAPI,
#     File,
#     UploadFile,
#     HTTPException,
# )

# from pydantic import BaseModel

# from document_loader import extract_text

# from rag_service import (
#     ask_question,
#     ingest_document,
# )


# # --------------------------------------------------
# # FastAPI application
# # --------------------------------------------------

# app = FastAPI(
#     title="AI Knowledge Assistant",
#     description="RAG-based AI Knowledge Assistant",
#     version="1.0.0",
# )


# # --------------------------------------------------
# # Directories
# # --------------------------------------------------

# UPLOAD_DIR = Path("uploaded_documents")

# UPLOAD_DIR.mkdir(
#     exist_ok=True
# )


# # --------------------------------------------------
# # Question model
# # --------------------------------------------------

# class QuestionRequest(BaseModel):

#     question: str


# # --------------------------------------------------
# # Root endpoint
# # --------------------------------------------------

# @app.get("/")
# def root():

#     return {
#         "message": "AI Knowledge Assistant API is running"
#     }


# # --------------------------------------------------
# # Ask question
# # --------------------------------------------------

# @app.post("/ask")
# def ask(
#     request: QuestionRequest
# ):

#     return ask_question(
#         request.question
#     )


# # --------------------------------------------------
# # Upload document
# # --------------------------------------------------

# @app.post("/upload")
# async def upload_document(
#     file: UploadFile = File(...)
# ):

#     # --------------------------------------------------
#     # 1. Validate filename
#     # --------------------------------------------------

#     if not file.filename:

#         raise HTTPException(
#             status_code=400,
#             detail="No file selected.",
#         )


#     # --------------------------------------------------
#     # 2. Get file extension
#     # --------------------------------------------------

#     extension = Path(
#         file.filename
#     ).suffix.lower()


#     # --------------------------------------------------
#     # 3. Validate file type
#     # --------------------------------------------------

#     allowed_extensions = {
#         ".pdf",
#         ".docx",
#         ".txt",
#     }


#     if extension not in allowed_extensions:

#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 "Unsupported file type. "
#                 "Only PDF, DOCX and TXT "
#                 "are supported."
#             ),
#         )


#     # --------------------------------------------------
#     # 4. Create document ID
#     # --------------------------------------------------

#     document_id = str(
#         uuid.uuid4()
#     )


#     # --------------------------------------------------
#     # 5. Create saved filename
#     # --------------------------------------------------

#     saved_filename = (
#         f"{document_id}{extension}"
#     )


#     file_path = (
#         UPLOAD_DIR / saved_filename
#     )


#     # --------------------------------------------------
#     # 6. Save uploaded file
#     # --------------------------------------------------

#     with file_path.open("wb") as buffer:

#         shutil.copyfileobj(
#             file.file,
#             buffer
#         )


#     # --------------------------------------------------
#     # 7. Extract text
#     # --------------------------------------------------

#     try:

#         text = extract_text(
#             str(file_path),
#             extension,
#         )

#     except Exception as error:

#         file_path.unlink(
#             missing_ok=True
#         )

#         raise HTTPException(
#             status_code=500,
#             detail=(
#                 f"Failed to extract document: {error}"
#             ),
#         )


#     # --------------------------------------------------
#     # 8. Validate extracted text
#     # --------------------------------------------------

#     if not text.strip():

#         file_path.unlink(
#             missing_ok=True
#         )

#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 "The uploaded document "
#                 "does not contain readable text."
#             ),
#         )


#     # --------------------------------------------------
#     # 9. Index document into Qdrant
#     # --------------------------------------------------

#     try:

#         indexing_result = ingest_document(
#             text=text,
#             filename=file.filename,
#             document_id=document_id,
#         )

#     except Exception as error:

#         file_path.unlink(
#             missing_ok=True
#         )

#         raise HTTPException(
#             status_code=500,
#             detail=(
#                 f"Failed to index document: {error}"
#             ),
#         )


#     # --------------------------------------------------
#     # 10. Return result
#     # --------------------------------------------------

#     return {
#         "message": (
#             "Document uploaded and indexed successfully."
#         ),
#         "document_id": document_id,
#         "filename": file.filename,
#         "file_type": extension,
#         "characters": len(text),
#         "chunks": indexing_result["chunks"],
#     }



from pathlib import Path
import json
import shutil
import uuid

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import (
    FileResponse,
    StreamingResponse,
)

from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, field_validator

from document_loader import extract_text

import config

from rag_service import (
    ask_question,
    ask_question_stream,
    ingest_document,
    list_documents,
    delete_document,
)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="AI Knowledge Assistant",
    description="RAG-based AI Knowledge Assistant",
    version="1.0.0",
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=False,
    allow_methods=[
        "*",
    ],
    allow_headers=[
        "*",
    ],
)


# --------------------------------------------------
# Directories
# --------------------------------------------------

UPLOAD_DIR = Path("uploaded_documents")

UPLOAD_DIR.mkdir(
    exist_ok=True
)


FRONTEND_DIR = Path(
    __file__
).resolve().parent / "frontend"


# --------------------------------------------------
# Question model
# --------------------------------------------------

class Turn(BaseModel):

    role: str

    content: str


class QuestionRequest(BaseModel):

    question: str

    document_ids: list[str] | None = None

    history: list[Turn] | None = None

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value):

        if not value or not value.strip():

            raise ValueError(
                "question must not be empty"
            )

        return value.strip()


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():

    index_file = (
        FRONTEND_DIR / "index.html"
    )

    if index_file.exists():

        return FileResponse(
            str(index_file)
        )

    return {
        "message": "AI Knowledge Assistant API is running"
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "documents": len(
            list_documents()
        ),
        "backend": config.BACKEND_NAME,
        "model": config.ACTIVE_MODEL,
        "device": config.DEVICE,
        "streaming": True,
    }


# --------------------------------------------------
# Ask question (non-streaming)
# --------------------------------------------------

@app.post("/ask")
def ask(
    request: QuestionRequest
):

    return ask_question(
        request.question,
        document_ids=request.document_ids,
        history=[
            turn.model_dump()
            for turn in (request.history or [])
        ],
    )


# --------------------------------------------------
# Ask question (streaming)
# --------------------------------------------------

@app.post("/ask/stream")
def ask_stream(
    request: QuestionRequest
):
    """
    Server-sent events. The first token reaches the
    browser in a few seconds instead of after the whole
    answer has been generated.
    """

    def generate():

        try:

            for event in ask_question_stream(
                request.question,
                document_ids=request.document_ids,
                history=[
                    turn.model_dump()
                    for turn in (request.history or [])
                ],
            ):

                yield (
                    "data: "
                    + json.dumps(event)
                    + "\n\n"
                )

        except Exception as error:

            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "error",
                        "message": str(error),
                    }
                )
                + "\n\n"
            )

        finally:

            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------
# List documents
# --------------------------------------------------

@app.get("/documents")
def get_documents():

    documents = list_documents()

    return {
        "documents": documents,
        "count": len(documents),
    }


# --------------------------------------------------
# Delete document
# --------------------------------------------------

@app.delete(
    "/documents/{document_id}"
)
def remove_document(
    document_id: str,
):

    result = delete_document(
        document_id
    )

    for path in UPLOAD_DIR.glob(
        f"{document_id}.*"
    ):

        path.unlink(
            missing_ok=True
        )

    return result


# --------------------------------------------------
# Upload document
# --------------------------------------------------

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    # --------------------------------------------------
    # 1. Validate filename
    # --------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected.",
        )


    # --------------------------------------------------
    # 2. Get file extension
    # --------------------------------------------------

    extension = Path(
        file.filename
    ).suffix.lower()


    # --------------------------------------------------
    # 3. Validate file type
    # --------------------------------------------------

    allowed_extensions = {
        ".pdf",
        ".docx",
        ".txt",
    }


    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Only PDF, DOCX and TXT "
                "are supported."
            ),
        )


    # --------------------------------------------------
    # 4. Replace any earlier copy of the same filename
    # --------------------------------------------------

    # Re-uploading a file the user just edited should
    # update it, not leave a stale duplicate behind that
    # still gets searched.

    replaced = None

    for existing in list_documents():

        if existing["filename"] != file.filename:
            continue

        delete_document(
            existing["document_id"]
        )

        for stale in UPLOAD_DIR.glob(
            f"{existing['document_id']}.*"
        ):

            stale.unlink(
                missing_ok=True
            )

        replaced = existing["document_id"]
        break


    # --------------------------------------------------
    # 5. Create document ID
    # --------------------------------------------------

    document_id = str(
        uuid.uuid4()
    )


    # --------------------------------------------------
    # 6. Create saved filename
    # --------------------------------------------------

    saved_filename = (
        f"{document_id}{extension}"
    )


    file_path = (
        UPLOAD_DIR / saved_filename
    )


    # --------------------------------------------------
    # 7. Save uploaded file
    # --------------------------------------------------

    with file_path.open("wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )


    # --------------------------------------------------
    # 7. Extract text
    # --------------------------------------------------

    try:

        text = extract_text(
            str(file_path),
            extension,
        )

    except Exception as error:

        file_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to extract document: {error}"
            ),
        )


    # --------------------------------------------------
    # 8. Validate extracted text
    # --------------------------------------------------

    if not text.strip():

        file_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded document "
                "does not contain readable text."
            ),
        )


    # --------------------------------------------------
    # 9. Index document into Qdrant
    # --------------------------------------------------

    try:

        indexing_result = ingest_document(
            text=text,
            filename=file.filename,
            document_id=document_id,
        )

    except Exception as error:

        file_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to index document: {error}"
            ),
        )


    # --------------------------------------------------
    # 12. Return result
    # --------------------------------------------------

    return {
        "message": (
            "Document replaced and re-indexed successfully."
            if replaced
            else
            "Document uploaded and indexed successfully."
        ),
        "document_id": document_id,
        "filename": file.filename,
        "file_type": extension,
        "characters": len(text),
        "chunks": indexing_result["chunks"],
        "replaced": bool(replaced),
    }


# --------------------------------------------------
# Frontend
# --------------------------------------------------

if FRONTEND_DIR.exists():

    app.mount(
        "/",
        StaticFiles(
            directory=str(FRONTEND_DIR),
            html=True,
        ),
        name="frontend",
    )