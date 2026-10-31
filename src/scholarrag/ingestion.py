
import re
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Clean extracted PDF text while preserving paragraph structure."""

    # Replace non-breaking spaces
    text = text.replace("\u00a0", " ")

    # Remove unwanted control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Remove spaces before newlines
    text = re.sub(r"[ \t]+\n", "\n", text)

    # Normalize excessive spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_pdf(file_path):
    """Extract and clean text from every PDF page."""

    file_path = Path(file_path)
    reader = PdfReader(file_path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):

        extracted_text = page.extract_text()

        if extracted_text and extracted_text.strip():

            cleaned_text = clean_text(extracted_text)

            if not cleaned_text:
                continue

            document = Document(
                page_content=cleaned_text,
                metadata={
                    "source": file_path.name,
                    "page": page_number
                }
            )

            documents.append(document)

    return documents


if __name__ == "__main__":

    pdf_path = Path("data/raw/sample.pdf")

    if not pdf_path.exists():
        print("PDF not found!")

    else:
        documents = load_pdf(pdf_path)

        print("PDF successfully loaded!")
        print("Total documents:", len(documents))

        if documents:
            print("\nFirst document metadata:")
            print(documents[0].metadata)

            print("\nFirst document content:")
            print(documents[0].page_content[:1000])
