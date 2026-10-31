
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter
from scholarrag.ingestion import load_pdf


def create_chunks(documents):
    """Split documents into smaller chunks while preserving metadata."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


def analyze_chunks(chunks):
    """Analyze chunk sizes and their distribution across PDF pages."""

    if not chunks:
        print("No chunks available for analysis.")
        return

    chunk_lengths = [len(chunk.page_content) for chunk in chunks]

    print("\n========== CHUNK ANALYSIS ==========")

    print("Total chunks:", len(chunks))
    print("Average chunk size:", round(sum(chunk_lengths) / len(chunk_lengths), 2))
    print("Smallest chunk:", min(chunk_lengths))
    print("Largest chunk:", max(chunk_lengths))

    page_distribution = Counter(
        chunk.metadata["page"] for chunk in chunks
    )

    print("\n========== CHUNKS PER PAGE ==========")

    for page, count in sorted(page_distribution.items()):
        print(f"Page {page}: {count} chunks")

    print("\n========== FIRST CHUNK DETAILS ==========")

    print("Metadata:", chunks[0].metadata)
    print("Character count:", len(chunks[0].page_content))

    print("\nContent preview:")
    print(chunks[0].page_content[:500])


if __name__ == "__main__":

    pdf_path = "data/raw/sample.pdf"

    documents = load_pdf(pdf_path)

    chunks = create_chunks(documents)

    print("Total pages:", len(documents))

    analyze_chunks(chunks)
