import streamlit as st
import pdfplumber
import docx
import io
import time
import numpy as np
from groq import Groq, RateLimitError
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Shared UI Background ---
def set_background():
    background = """
    <style>
    [data-testid="stApp"]{
        background-color: #000000;
        opacity: 0.8;
        background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #000000 6px ), repeating-linear-gradient( #43434355, #434343 );
    }
    </style>
    """
    st.markdown(background, unsafe_allow_html=True)


# --- Groq API Config ---
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=GROQ_API_KEY)
else:
    client = None

PRIMARY_MODEL  = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "mixtral-8x7b-32768"
MAX_RETRIES    = 3


def call_api(messages, temperature=0.2, max_tokens=4096):
    if not client:
        st.error("GROQ_API_KEY is missing from secrets.toml")
        return None
        
    models_to_try = [PRIMARY_MODEL, FALLBACK_MODEL]

    for model_index, model in enumerate(models_to_try):
        if model_index > 0:
            st.warning(
                f"⚠️ `{PRIMARY_MODEL}` exhausted after {MAX_RETRIES} retries. "
                f"Switching to fallback model `{FALLBACK_MODEL}`..."
            )

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content

            except RateLimitError as e:
                retry_after = 30
                try:
                    retry_after = int(e.response.headers.get("retry-after", 30))
                except Exception:
                    pass

                is_last_attempt = (attempt == MAX_RETRIES - 1)
                if is_last_attempt:
                    break

                label = st.empty()
                for remaining in range(retry_after, 0, -1):
                    label.warning(
                        f"⏳ Rate limited — retrying in **{remaining}s** "
                        f"(attempt {attempt + 1} of {MAX_RETRIES} on `{model}`)..."
                    )
                    time.sleep(1)
                label.empty()

            except Exception as e:
                st.error(f"❌ Unexpected API error: `{e}`")
                return None

    total = MAX_RETRIES * len(models_to_try)
    st.error(
        f"❌ Both `{PRIMARY_MODEL}` and `{FALLBACK_MODEL}` are rate-limited "
        f"after {MAX_RETRIES} attempts each ({total} total tries). "
        "Please wait a minute and try again."
    )
    return None


# --- Text Extraction ---
def extract_text(uploaded_file):
    filetype = uploaded_file.name.split(".")[-1].lower()
    if filetype == "pdf":
        return extract_text_from_pdf(uploaded_file)
    elif filetype == "docx":
        return extract_text_from_docx(uploaded_file)
    else:
        st.error("Incompatible file type. Please upload a PDF or DOCX file.")
        return ""

def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        return "\n".join(
            [page.extract_text() for page in pdf.pages if page.extract_text()]
        )

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])


# --- Embedding & RAG Utilities ---
@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        if end < len(text) and ' ' in text[start:end]:
            last_space = text.rfind(' ', start, end)
            if last_space != -1:
                end = last_space + 1
                
        chunks.append(text[start:end])
        start = max(start + 1, end - overlap)
        
    return chunks

def build_rag_index(text):
    model = load_embedding_model()
    chunks = chunk_text(text)
    with st.spinner("🔍 Indexing document for smart retrieval..."):
        embeddings = model.encode(chunks, show_progress_bar=False)
    return chunks, embeddings

def retrieve_relevant_chunks(question, chunks, embeddings, top_k=5):
    model = load_embedding_model()
    q_embedding = model.encode([question])
    scores = cosine_similarity(q_embedding, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


def check_grounding(answer, context_chunks):
    context = "\n---\n".join(context_chunks)
    messages = [
        {
            "role": "system",
            "content": "You are a fact-checker. Reply ONLY with GROUNDED or NOT_GROUNDED."
        },
        {
            "role": "user",
            "content": (
                f"Document excerpts:\n{context}\n\n"
                f"Answer to verify:\n{answer}\n\n"
                "Is this answer derivable solely from the document excerpts above? "
                "Reply only: GROUNDED or NOT_GROUNDED"
            ),
        },
    ]
    try:
        response = client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=10,
        )
        result = response.choices[0].message.content.strip().upper()
        return "NOT_GROUNDED" not in result
    except Exception:
        return False
