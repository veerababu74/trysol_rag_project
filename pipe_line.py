

from doc_loader import load_document
from chunking import chunk_documents
from embeddings import get_embeddings
import vector_db as vs
from llm import get_llm
from logger import get_logger

log = get_logger("pipeline")

# We build the embeddings object ONCE and reuse it (saves time).
log.info("Initialising embeddings model...")
embeddings = get_embeddings()
log.info("Embeddings model ready")


def ingest_document(file_path: str) -> int:
    log.info("[INGEST] Start: %s", file_path)

    log.debug("[INGEST] Loading document...")
    documents = load_document(file_path)
    log.info("[INGEST] Loaded %d document(s)", len(documents))

    log.debug("[INGEST] Chunking...")
    chunks = chunk_documents(documents)
    log.info("[INGEST] Created %d chunk(s)", len(chunks))

    log.debug("[INGEST] Embedding & storing in FAISS...")
    vs.faiss_add(chunks, embeddings)
    log.info("[INGEST] Done. %d chunks indexed.", len(chunks))

    return len(chunks)


def _get_retriever(k: int = 4):
    log.debug("[RETRIEVE] Loading FAISS index, k=%d", k)
    db = vs.load_faiss(embeddings)
    log.debug("[RETRIEVE] FAISS index loaded")
    return db.as_retriever(search_kwargs={"k": k})


# The prompt is how we tell the LLM to ONLY use our documents and not make
# things up (this reduces hallucination).
PROMPT_TEMPLATE = """You are a helpful assistant.
Answer the QUESTION using ONLY the CONTEXT below.
If the answer is not present in the context, say "I don't know based on the provided documents."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def chat(question: str, k: int = 4):
    log.info("[CHAT] Question: %r", question)

    retriever = _get_retriever(k)
    docs = retriever.invoke(question)
    log.info("[CHAT] Retrieved %d chunk(s): %s",
             len(docs), [d.metadata.get('source') for d in docs])

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    log.debug("[CHAT] Prompt length: %d chars", len(prompt))

    log.debug("[CHAT] Calling LLM...")
    llm = get_llm()
    response = llm.invoke(prompt)
    log.info("[CHAT] LLM responded (%d chars)", len(response.content))

    sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})
    return response.content, sources
