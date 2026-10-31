import math

from qdrant_client.models import Filter, FieldCondition, MatchValue

from scholarrag.embeddings import generate_query_embedding
from scholarrag.qdrant_db import get_qdrant_client, COLLECTION_NAME


SIMILARITY_THRESHOLD = 0.65


def cosine_similarity(vector_a, vector_b):
    """Calculate cosine similarity between two vectors."""

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def search_documents(
    query: str,
    top_k: int = 5,
    fetch_k: int = 15,
    diversity: float = 0.3,
    source: str | None = None
):
    """
    Retrieve relevant and diverse chunks using MMR.

    If source is provided, search only that PDF.
    Otherwise, search all indexed PDFs.

    diversity = 0.0 -> prioritize relevance.
    diversity = 1.0 -> prioritize diversity.
    """

    # Validate input parameters
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if fetch_k < top_k:
        raise ValueError("fetch_k must be greater than or equal to top_k.")

    if not 0.0 <= diversity <= 1.0:
        raise ValueError("diversity must be between 0 and 1.")

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    # Connect to Qdrant and embed the query
    client = get_qdrant_client()
    query_vector = generate_query_embedding(query)

    # Optional filter: restrict search to one PDF
    query_filter = None

    if source:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source)
                )
            ]
        )

    # Fetch candidate chunks
    candidates = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=fetch_k,
        with_payload=True,
        with_vectors=True
    ).points

    # Apply relevance threshold
    candidates = [
        point for point in candidates
        if point.score >= SIMILARITY_THRESHOLD
    ]

    if not candidates:
        return []

    # Select the most relevant initial chunk
    selected = [candidates.pop(0)]

    # Apply MMR to select diverse chunks
    while candidates and len(selected) < top_k:
        best_candidate = None
        best_mmr_score = float("-inf")

        for candidate in candidates:
            similarities = [
                cosine_similarity(candidate.vector, selected_point.vector)
                for selected_point in selected
            ]

            redundancy = max(similarities)
            relevance = candidate.score

            mmr_score = (
                (1 - diversity) * relevance
                - diversity * redundancy
            )

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_candidate = candidate

        selected.append(best_candidate)
        candidates.remove(best_candidate)

    return selected


if __name__ == "__main__":
    print("\n========== SCHOLARRAG MMR RETRIEVAL ==========")

    user_query = input("\nAsk a question about your PDFs: ")

    # Set to a filename to filter, or None to search all PDFs.
    selected_source = "sample4.pdf"

    try:
        results = search_documents(
            query=user_query,
            source=selected_source
        )

        print("\n========== MMR SEARCH RESULTS ==========")
        print("Search scope:", selected_source or "All PDFs")
        print("Similarity threshold:", SIMILARITY_THRESHOLD)
        print("Accepted chunks:", len(results))

        if not results:
            print("\nNo relevant documents found.")

        for index, result in enumerate(results, start=1):
            print(f"\nResult {index}")
            print("Similarity score:", round(result.score, 4))
            print("Source:", result.payload["source"])
            print("Page:", result.payload["page"])
            print("Content:")
            print(result.payload["text"])
            print("-" * 60)

    except Exception as error:
        print("\nRetrieval error:", error)