
import os

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
)
from logger import get_logger

log = get_logger("doc_loader")

# Map a file extension -> the loader class that knows how to read it.
LOADER_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
    ".csv": CSVLoader,
}


def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    log.info("Loading document: %s (ext=%s)", file_path, ext)

    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        log.warning("Unsupported file type: %s", ext)
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported types: {', '.join(LOADER_MAP.keys())}"
        )

    loader = loader_cls(file_path)
    documents = loader.load()
    log.info("Loaded %d page(s) from %s", len(documents), os.path.basename(file_path))

    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)

    return documents
