from pathlib import Path

import fitz
from docx import Document


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    """

    document = fitz.open(file_path)

    pages = []

    for page in document:
        pages.append(page.get_text())

    document.close()

    return "\n".join(pages)


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from a DOCX file.
    """

    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(paragraph.text)

    return "\n".join(paragraphs)


def extract_text_from_txt(file_path: str) -> str:
    """
    Read text from a TXT file.
    """

    path = Path(file_path)

    return path.read_text(
        encoding="utf-8"
    )


def extract_text(
    file_path: str,
    file_extension: str,
) -> str:

    extension = file_extension.lower()

    if extension == ".pdf":

        return extract_text_from_pdf(
            file_path
        )

    elif extension == ".docx":

        return extract_text_from_docx(
            file_path
        )

    elif extension == ".txt":

        return extract_text_from_txt(
            file_path
        )

    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )