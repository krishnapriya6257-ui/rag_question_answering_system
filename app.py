
import streamlit as st
import numpy as np
import torch

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="RAG Question Answering System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG-Based Question Answering System")

st.markdown("""
### Intelligent Document Assistant

Upload a PDF or text document and ask questions.
The system retrieves relevant information and generates
answers using Artificial Intelligence.
""")

st.divider()


# ==========================================
# 2. LOAD AI MODELS
# ==========================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_language_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1
    )


with st.spinner("Loading AI models..."):
    embedding_model = load_embedding_model()
    language_model = load_language_model()


# ==========================================
# 3. EXTRACT TEXT FROM DOCUMENT
# ==========================================

def extract_text(file):

    if file.name.lower().endswith(".pdf"):

        pdf = PdfReader(file)

        text = ""

        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    elif file.name.lower().endswith(".txt"):

        return file.getvalue().decode("utf-8")

    return ""


# ==========================================
# 4. TEXT CHUNKING
# ==========================================

def split_text(text, chunk_size=100, overlap=20):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ==========================================
# 5. GENERATE EMBEDDINGS
# ==========================================

def generate_embeddings(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


# ==========================================
# 6. RETRIEVE RELEVANT INFORMATION
# ==========================================

def retrieve_information(question, chunks, embeddings):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarity_scores = np.dot(
        embeddings,
        question_embedding[0]
    )

    top_indices = np.argsort(
        similarity_scores
    )[-3:][::-1]

    retrieved_chunks = [
        chunks[index] for index in top_indices
    ]

    scores = [
        float(similarity_scores[index])
        for index in top_indices
    ]

    return retrieved_chunks, scores


# ==========================================
# 7. GENERATE ANSWER USING FLAN-T5
# ==========================================

def generate_answer(question, context):

    prompt = f"""
    Answer the question using only the information
    provided in the context.

    If the answer is not present in the context,
    say that the information is not available
    in the uploaded document.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    result = language_model(
        prompt,
        max_new_tokens=150,
        do_sample=False
    )

    return result[0]["generated_text"]


# ==========================================
# 8. DOCUMENT UPLOAD SECTION
# ==========================================

st.header("📄 Upload Your Document")

uploaded_file = st.file_uploader(
    "Choose a PDF or TXT file",
    type=["pdf", "txt"]
)


if uploaded_file:

    if st.button("Process Document", type="primary"):

        with st.spinner("Processing your document..."):

            try:

                text = extract_text(uploaded_file)

                if not text.strip():
                    st.error("No readable text found.")
                    st.stop()

                chunks = split_text(text)

                if not chunks:
                    st.error("Unable to create text chunks.")
                    st.stop()

                embeddings = generate_embeddings(chunks)

                st.session_state["chunks"] = chunks
                st.session_state["embeddings"] = embeddings
                st.session_state["filename"] = uploaded_file.name

                st.success("Document processed successfully!")

                st.info(
                    f"Document: {uploaded_file.name} | "
                    f"Total chunks: {len(chunks)}"
                )

            except Exception as error:

                st.error(f"Processing error: {error}")


# ==========================================
# 9. QUESTION AND ANSWER SECTION
# ==========================================

if "chunks" in st.session_state:

    st.divider()

    st.header("💬 Ask Questions")

    st.write(
        f"Currently using: {st.session_state['filename']}"
    )

    question = st.text_input(
        "Enter your question:",
        placeholder="Example: What is Artificial Intelligence?"
    )

    if st.button("Generate Answer", type="primary"):

        if question.strip():

            with st.spinner(
                "Retrieving information and generating answer..."
            ):

                try:

                    retrieved_chunks, scores = retrieve_information(
                        question,
                        st.session_state["chunks"],
                        st.session_state["embeddings"]
                    )

                    context = "\n\n".join(retrieved_chunks)

                    answer = generate_answer(
                        question,
                        context
                    )

                    st.subheader("🤖 Generated Answer")

                    st.success(answer)

                    with st.expander(
                        "View Retrieved Information"
                    ):

                        for i, chunk in enumerate(
                            retrieved_chunks, start=1
                        ):

                            st.markdown(
                                f"**Relevant Section {i}**"
                            )

                            st.write(chunk)

                            st.caption(
                                f"Similarity Score: {scores[i-1]:.4f}"
                            )

                            st.divider()

                except Exception as error:

                    st.error(f"Question answering error: {error}")

        else:

            st.warning("Please enter a question.")


# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

st.markdown("""
**Project:** RAG-Based Question Answering System

**Technologies:** Python | Streamlit | Sentence Transformers |
FLAN-T5 | NumPy | PyPDF

**Purpose:** To retrieve relevant information from uploaded
documents and generate context-based answers.
""")

