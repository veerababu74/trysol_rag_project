
import os

from langchain_community.vectorstores import FAISS

from config import FAISS_DIR
from logger import get_logger

log = get_logger("vector_db")


def load_faiss(embeddings):
    log.debug("Loading FAISS index from %s", FAISS_DIR)
    db = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    log.debug("FAISS index loaded")
    return db


def faiss_add(chunks, embeddings):
    os.makedirs(os.path.dirname(FAISS_DIR), exist_ok=True)

    index_file = os.path.join(FAISS_DIR, "index.faiss")
    if os.path.exists(index_file):
        log.info("Replacing existing FAISS index with %d new chunk(s)", len(chunks))
        import shutil
        shutil.rmtree(FAISS_DIR, ignore_errors=True)
        os.makedirs(FAISS_DIR, exist_ok=True)
    else:
        log.debug("No existing FAISS index — creating new with %d chunk(s)", len(chunks))

    db = FAISS.from_documents(chunks, embeddings)

    db.save_local(FAISS_DIR)
    log.info("FAISS index saved to %s", FAISS_DIR)
    return db
