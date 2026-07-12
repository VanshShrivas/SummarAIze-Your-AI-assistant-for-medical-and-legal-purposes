import streamlit as st
from utils import (
    set_background,
    call_api,
    extract_text,
    build_rag_index,
    check_grounding,
    retrieve_relevant_chunks
)

set_background()

def summarize_text(text):
    prompt = (
        "Summarize this legal document (this may be a legal notice too), "
        "highlighting key laws, sections involved, and judgements (and further "
        "actions to be taken by the receiver in case this is a legal notice; "
        "if this is not a legal notice, do not involve anything related to it). "
        "Do not show any internal reasoning, chain-of-thought, or explanation "
        f"of your process:\n\n{text}\n\n"
        "Keep it simple and easy to understand. Identify the laws and sections "
        "involved and what the receiver should do based on the judgements."
    )
    messages = [
        {"role": "system", "content": "You are an AI model that summarizes legal documents in simple, clear language."},
        {"role": "user",   "content": prompt},
    ]
    return call_api(messages)


def ask_followup(question, chunks, embeddings):
    relevant_chunks = retrieve_relevant_chunks(question, chunks, embeddings)
    context = "\n\n---\n\n".join(relevant_chunks)

    prompt = (
        "Using ONLY the following excerpts from the legal document:\n\n"
        f"{context}\n\n"
        f"Answer this question: {question}\n\n"
        "If the answer cannot be found in the excerpts, respond with exactly: "
        "'This information is not available in the uploaded document.'\n"
        "Answer in simple, clear language without excessive legal jargon."
    )
    messages = [
        {
            "role": "system",
            "content": "You are a legal document assistant. Answer questions strictly based on the provided document excerpts only."
        },
        {"role": "user", "content": prompt},
    ]
    answer = call_api(messages)
    is_grounded = check_grounding(answer, relevant_chunks) if answer else True
    return answer, is_grounded


st.title("⚖️ Legal Document Summarization")

# Session state initialisation
for key in ["legal_summary", "legal_last_file", "legal_chunks", "legal_embeddings"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.markdown("Upload a legal document to summarize and ask questions about it.")
uploaded_file = st.file_uploader("Upload any PDF/DOCX", type=["pdf", "docx"], key="legal")

if uploaded_file:
    # Reset state when a new file is uploaded
    if st.session_state.legal_last_file != uploaded_file.name:
        st.session_state.legal_summary    = None
        st.session_state.legal_last_file  = uploaded_file.name
        st.session_state.legal_chunks     = None
        st.session_state.legal_embeddings = None

    text = extract_text(uploaded_file)
    if text:
        # Generate summary once
        if st.session_state.legal_summary is None:
            with st.spinner("Hol'up! Let us Cook... 🧑‍🍳"):
                st.session_state.legal_summary = summarize_text(text)

        # Build RAG index once
        if st.session_state.legal_chunks is None:
            chunks, embeddings = build_rag_index(text)
            st.session_state.legal_chunks     = chunks
            st.session_state.legal_embeddings = embeddings

        if st.session_state.legal_summary:
            st.subheader("Legal Summary/Explanation:")
            st.write(st.session_state.legal_summary)

            st.subheader("Follow-up Questions")
            question = st.text_input("Ask a question about your legal document:")

            if question:
                    with st.spinner("Searching document and generating answer..."):
                        answer, is_grounded = ask_followup(
                            question,
                            st.session_state.legal_chunks,
                            st.session_state.legal_embeddings,
                        )

                    if answer:
                        st.subheader("Answer")
                        st.write(answer)

                        # Hallucination badge
                        if is_grounded:
                            st.success("✅ Answer grounded in your document")
                        else:
                            st.warning(
                                "⚠️ This answer may go beyond what is in your document — "
                                "please verify with a qualified lawyer."
                            )
    else:
        st.error("No text could be extracted from this file. If it's a scanned image, OCR is required.")
else:
    st.session_state.legal_summary    = None
    st.session_state.legal_last_file  = None
    st.session_state.legal_chunks     = None
    st.session_state.legal_embeddings = None
