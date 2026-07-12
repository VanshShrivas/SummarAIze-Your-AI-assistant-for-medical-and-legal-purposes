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
        "Summarize this medical document/report by focusing on diagnosis, "
        "test results, prescribed medications, and doctor's advice. "
        "Do not show any internal reasoning, chain-of-thought, or explanation "
        f"of your process:\n\n{text}\n\n"
        "Keep it simple and easy to understand for a patient without a medical background."
    )
    messages = [
        {"role": "system", "content": "You are an AI model that summarizes medical documents in simple, clear language."},
        {"role": "user",   "content": prompt},
    ]
    return call_api(messages)


def ask_followup(question, chunks, embeddings):
    relevant_chunks = retrieve_relevant_chunks(question, chunks, embeddings)
    context = "\n\n---\n\n".join(relevant_chunks)

    prompt = (
        "Using ONLY the following excerpts from the medical document:\n\n"
        f"{context}\n\n"
        f"Answer this question: {question}\n\n"
        "If the answer cannot be found in the excerpts, respond with exactly: "
        "'This information is not available in the uploaded document.'\n"
        "Answer in simple, clear language without excessive medical jargon."
    )
    messages = [
        {
            "role": "system",
            "content": "You are a medical assistant. Answer questions strictly based on the provided document excerpts only."
        },
        {"role": "user", "content": prompt},
    ]
    answer = call_api(messages)
    is_grounded = check_grounding(answer, relevant_chunks) if answer else True
    return answer, is_grounded


st.title("🩺 Medical Report Summarization")

# Session state initialisation
for key in ["medical_summary", "medical_last_file", "medical_chunks", "medical_embeddings"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.markdown("Upload a medical report to summarize and ask questions about it.")
uploaded_file = st.file_uploader("Upload any PDF/DOCX", type=["pdf", "docx"], key="medical")

if uploaded_file:
    # Reset state when a new file is uploaded
    if st.session_state.medical_last_file != uploaded_file.name:
        st.session_state.medical_summary    = None
        st.session_state.medical_last_file  = uploaded_file.name
        st.session_state.medical_chunks     = None
        st.session_state.medical_embeddings = None

    text = extract_text(uploaded_file)
    if text:
        # Generate summary once
        if st.session_state.medical_summary is None:
            with st.spinner("Hol'up! Let us Cook... 🧑‍🍳"):
                st.session_state.medical_summary = summarize_text(text)

        # Build RAG index once
        if st.session_state.medical_chunks is None:
            chunks, embeddings = build_rag_index(text)
            st.session_state.medical_chunks     = chunks
            st.session_state.medical_embeddings = embeddings

        if st.session_state.medical_summary:
            st.subheader("Medical Summary/Explanation:")
            st.write(st.session_state.medical_summary)

            st.subheader("Follow-up Questions")
            question = st.text_input("Ask a question about your medical document:")

            if question:
                    with st.spinner("Searching document and generating answer..."):
                        answer, is_grounded = ask_followup(
                            question,
                            st.session_state.medical_chunks,
                            st.session_state.medical_embeddings,
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
                                "please verify with a qualified medical professional."
                            )
    else:
        st.error("No text could be extracted from this file. If it's a scanned image, OCR is required.")
else:
    st.session_state.medical_summary    = None
    st.session_state.medical_last_file  = None
    st.session_state.medical_chunks     = None
    st.session_state.medical_embeddings = None
