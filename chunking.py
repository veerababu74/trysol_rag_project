
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from logger import get_logger

log = get_logger("chunking")


def chunk_documents(
    documents, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
):
    log.debug("Chunking %d doc(s) — size=%d overlap=%d",
              len(documents), chunk_size, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    log.info("Chunking produced %d chunk(s)", len(chunks))
    return chunks
