import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

from scholarrag.retrieval import search_documents


load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from your .env file.")

client = genai.Client(api_key=api_key)


def validate_citations(answer: str, valid_labels: set[str]) -> dict:
    """Check whether citation labels exist in the retrieved context."""
    used_labels = set(re.findall(r"\[S\d+\]", answer))
    invalid_labels = used_labels - valid_labels

    return {
        "valid": len(invalid_labels) == 0,
        "used_labels": sorted(used_labels),
        "invalid_labels": sorted(invalid_labels),
    }


def generate_answer(
    query: str,
    top_k: int = 5,
    source: str | None = None,
) -> dict:
    """Retrieve context from one PDF or all PDFs, then generate an answer."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    print("\n[1/3] Searching Qdrant...")
    print(f"PDF filter: {source}" if source else "PDF filter: all indexed PDFs")

    retrieved_chunks = search_documents(
        query=query,
        top_k=top_k,
        source=source,
    )

    print(f"Retrieved {len(retrieved_chunks)} relevant chunks.")

    if not retrieved_chunks:
        return {
            "answer": (
                "I couldn't find relevant information in the "
                "uploaded documents to answer this question."
            ),
            "sources": [],
            "citation_validation": {
                "valid": True,
                "used_labels": [],
                "invalid_labels": [],
            },
        }

    print("\n[2/3] Preparing context...")

    context_parts = []
    sources = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        payload = chunk.payload
        label = f"[S{index}]"

        context_parts.append(
            f"""
{label}
Source: {payload["source"]}
Page: {payload["page"]}

Content:
{payload["text"]}
"""
        )

        sources.append({
            "label": label,
            "source": payload["source"],
            "page": payload["page"],
            "score": round(chunk.score, 4),
            "text": payload["text"],
        })

    context = "\n".join(context_parts)
    valid_labels = {item["label"] for item in sources}

    print("Context prepared successfully.")
    print("\n[3/3] Gemini is generating your answer...")

    prompt = f"""
You are ScholarRAG, a research document assistant.

Answer using ONLY the provided context.

RULES:
1. Never invent facts or use outside knowledge.
2. Support factual claims with citations.
3. Use only the exact citation labels provided, such as [S1].
4. Never create or modify citation labels.
5. If the answer is unavailable, clearly say so.
6. Be clear and concise.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1000,
            ),
        )

        answer = (response.text or "").strip()

        if not answer:
            answer = "Gemini returned an empty response."

        citation_validation = validate_citations(answer, valid_labels)

        return {
            "answer": answer,
            "sources": sources,
            "citation_validation": citation_validation,
        }

    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"
        print("\nGemini API error:", error_message)

        return {
            "answer": f"Gemini API error: {error_message}",
            "sources": sources,
            "citation_validation": None,
        }


if __name__ == "__main__":
    print("\n========== SCHOLARRAG ==========")

    question = input("\nAsk your research question: ")

    # None searches all indexed PDFs.
    selected_source = None

    # To search only one PDF, use:
    # selected_source = "sample4.pdf"

    try:
        result = generate_answer(
            question,
            source=selected_source,
        )

        print("\n========== GENERATED ANSWER ==========")
        print(result["answer"])

        print("\n========== CITATION VALIDATION ==========")
        validation = result["citation_validation"]

        if validation is not None:
            print("Valid citation labels:", validation["valid"])
            print("Used labels:", validation["used_labels"])
            print("Invalid labels:", validation["invalid_labels"])
        else:
            print("Citation validation unavailable.")

        print("\n========== SOURCE REFERENCES ==========")

        if not result["sources"]:
            print("No sources available.")

        for item in result["sources"]:
            print(
                f'{item["label"]} | '
                f'{item["source"]} | '
                f'Page {item["page"]} | '
                f'Similarity: {item["score"]}'
            )

    except Exception as error:
        print("\nScholarRAG error:", repr(error))