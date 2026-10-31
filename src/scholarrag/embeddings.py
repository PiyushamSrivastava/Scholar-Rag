
import os
from scholarrag.chunking import create_chunks
from scholarrag.ingestion import load_pdf

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_NAME = "gemini-embedding-2"
EMBEDDING_DIMENSION = 768

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_embedding(text: str) -> list[float]:
    """Generate a vector embedding for a document chunk."""

    formatted_text = f"title: none | text: {text}"

    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=formatted_text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION
        )
    )

    return response.embeddings[0].values

def embed_chunks(chunks):
    """Generate one embedding for each document chunk."""

    embedded_chunks = []

    for index, chunk in enumerate(chunks, start=1):

        vector = generate_embedding(chunk.page_content)

        embedded_chunks.append({
            "id": index,
            "text": chunk.page_content,
            "metadata": chunk.metadata.copy(),
            "vector": vector
        })

        print(f"Embedded chunk {index}/{len(chunks)}")

    return embedded_chunks


def generate_query_embedding(query: str) -> list[float]:
    """Generate an embedding for a user's search question."""

    formatted_query = f"task: question answering | query: {query}"

    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=formatted_query,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION
        )
    )

    return response.embeddings[0].values





if __name__ == "__main__":

    pdf_path = "data/raw/sample.pdf"

    # 1. Extract PDF pages
    documents = load_pdf(pdf_path)

    # 2. Create chunks
    chunks = create_chunks(documents)

    print("Total chunks:", len(chunks))

    # 3. Generate embeddings
    embedded_chunks = embed_chunks(chunks)

    # 4. Inspect results
    print("\n========== EMBEDDING SUMMARY ==========")

    print("Total embedded chunks:", len(embedded_chunks))
    print("Vector dimensions:", len(embedded_chunks[0]["vector"]))

    print("\nFirst chunk metadata:")
    print(embedded_chunks[0]["metadata"])

    print("\nFirst chunk text:")
    print(embedded_chunks[0]["text"][:300])

    print("\nFirst chunk vector preview:")
    print(embedded_chunks[0]["vector"][:5])

