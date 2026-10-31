from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from scholarrag.qdrant_db import get_qdrant_client, COLLECTION_NAME
from scholarrag.ingestion import load_pdf
from scholarrag.chunking import create_chunks
from scholarrag.embeddings import embed_chunks


def delete_existing_source(source: str):
    """Delete previously stored chunks belonging to one PDF."""
    client = get_qdrant_client()

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source),
                )
            ]
        ),
        wait=True,
    )

    print(f"Deleted old chunks for: {source}")


def store_embeddings(embedded_chunks):
    """Store embedded document chunks in Qdrant."""
    client = get_qdrant_client()
    points = []

    for chunk in embedded_chunks:
        unique_string = (
            f"{chunk['metadata']['source']}:"
            f"{chunk['metadata']['page']}:"
            f"{chunk['text']}"
        )

        point = PointStruct(
            id=str(uuid5(NAMESPACE_URL, unique_string)),
            vector=chunk["vector"],
            payload={
                "text": chunk["text"],
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
            },
        )
        points.append(point)

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

    print(f"Successfully stored {len(points)} chunks in Qdrant!")


def process_pdf(pdf_path: str | Path) -> int:
    """Extract, chunk, embed and store one PDF."""
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path.name}")

    print(f"\n========== PROCESSING: {pdf_path.name} ==========")

    # 1. Extract text
    documents = load_pdf(pdf_path)

    if not documents:
        print("No extractable text found. Skipping PDF.")
        return 0

    # 2. Create chunks
    chunks = create_chunks(documents)

    if not chunks:
        print("No chunks created. Skipping PDF.")
        return 0

    print("Pages extracted:", len(documents))
    print("Chunks created:", len(chunks))

    # 3. Generate embeddings before deleting the existing version.
    # This preserves the old indexed copy if embedding fails.
    embedded_chunks = embed_chunks(chunks)

    # 4. Replace only this PDF's old chunks
    delete_existing_source(pdf_path.name)

    # 5. Store the new embeddings
    store_embeddings(embedded_chunks)

    return len(embedded_chunks)


def process_all_pdfs(data_directory: str | Path = "data/raw") -> int:
    """Process every PDF in a directory."""
    data_path = Path(data_directory)

    if not data_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {data_path}")

    pdf_files = sorted(data_path.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/raw.")
        return 0

    print("Total PDFs found:", len(pdf_files))

    total_chunks = 0

    for pdf_path in pdf_files:
        try:
            total_chunks += process_pdf(pdf_path)
        except Exception as error:
            print(f"Error processing {pdf_path.name}: {error}")

    print("\n========== INGESTION SUMMARY ==========")
    print("PDFs discovered:", len(pdf_files))
    print("Total chunks processed:", total_chunks)

    client = get_qdrant_client()
    info = client.get_collection(COLLECTION_NAME)
    print("Total points in Qdrant:", info.points_count)

    return total_chunks


if __name__ == "__main__":
    process_all_pdfs()