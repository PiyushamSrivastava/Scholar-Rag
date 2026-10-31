from pathlib import Path

import streamlit as st

from scholarrag.rag_pipeline import generate_answer
from scholarrag.vector_store import process_pdf


st.set_page_config(
    page_title="ScholarRAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 ScholarRAG")
st.caption("Ask questions about your indexed research documents.")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_FOLDER = PROJECT_ROOT / "data" / "raw"
PDF_FOLDER.mkdir(parents=True, exist_ok=True)


def save_and_index_pdf(uploaded_file) -> str:
    """Save and index one uploaded PDF without overwriting an existing file."""

    filename = Path(uploaded_file.name).name

    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise ValueError("Please upload a file with a .pdf extension.")

    pdf_bytes = uploaded_file.getvalue()

    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("The uploaded file does not appear to be a valid PDF.")

    destination = PDF_FOLDER / filename

    if destination.exists():
        raise FileExistsError(
            f"'{filename}' already exists. Rename the new file before uploading."
        )

    destination.write_bytes(pdf_bytes)

    try:
        chunks_created = process_pdf(destination)

        if chunks_created == 0:
            raise ValueError(
                "No text chunks were indexed. The PDF may be scanned or image-only."
            )

    except Exception:
        destination.unlink(missing_ok=True)
        raise

    return filename


with st.expander("Add a PDF", expanded=False):
    uploaded_pdf = st.file_uploader(
        "Choose a research PDF",
        type=["pdf"],
        key="pdf_uploader",
    )

    if uploaded_pdf is not None:
        if st.button("Save and index PDF", type="primary"):
            try:
                with st.spinner("Saving and indexing the PDF..."):
                    saved_filename = save_and_index_pdf(uploaded_pdf)

                st.session_state["upload_message"] = (
                    f"Successfully saved and indexed: {saved_filename}"
                )
                st.rerun()

            except Exception as error:
                st.error(f"Could not save or index the PDF: {error}")


upload_message = st.session_state.pop("upload_message", None)
if upload_message:
    st.success(upload_message)


pdf_files = sorted(path.name for path in PDF_FOLDER.glob("*.pdf"))

if not pdf_files:
    st.warning("No PDFs found. Upload a PDF to get started.")
    st.stop()


source_options = ["All indexed PDFs", *pdf_files]
selected_option = st.selectbox("Search in", source_options)

selected_source = (
    None if selected_option == "All indexed PDFs"
    else selected_option
)

with st.form("question_form"):
    question = st.text_area(
        "Your question",
        placeholder="Ask something about your documents...",
    )
    submitted = st.form_submit_button("Ask ScholarRAG", type="primary")


if submitted:
    st.session_state.pop("scholarrag_result", None)
    st.session_state.pop("scholarrag_question", None)
    st.session_state.pop("scholarrag_source", None)

    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner("Searching documents and generating an answer..."):
                result = generate_answer(
                    query=question.strip(),
                    source=selected_source,
                )

            st.session_state["scholarrag_result"] = result
            st.session_state["scholarrag_question"] = question.strip()
            st.session_state["scholarrag_source"] = selected_option

        except Exception as error:
            st.error(f"ScholarRAG encountered an error: {error}")


result = st.session_state.get("scholarrag_result")

if result:
    st.divider()
    st.subheader("Answer")
    st.write(result.get("answer", "No answer was returned."))

    sources = result.get("sources") or []
    validation = result.get("citation_validation")

    if sources and validation is not None:
        if validation.get("valid"):
            st.caption("Citation labels match retrieved source chunks.")
        else:
            invalid_labels = ", ".join(
                validation.get("invalid_labels", [])
            )
            st.warning(f"Unrecognized citation labels: {invalid_labels}")

    st.subheader("Sources")

    if sources:
        for item in sources:
            title = (
                f"{item['label']} — {item['source']}, "
                f"page {item['page']} "
                f"(similarity: {item['score']})"
            )

            with st.expander(title):
                passage = item.get("text", "").strip()

                if passage:
                    st.markdown("**Retrieved passage**")
                    st.write(passage)
                else:
                    st.caption(
                        "The pipeline did not include the retrieved passage."
                    )
    else:
        st.info("No source references were returned.")