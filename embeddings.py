

from langchain_mistralai import MistralAIEmbeddings
from config import EMBEDDING_MODEL, MISTRAL_API_KEY
from logger import get_logger

log = get_logger("embeddings")


def get_embeddings():
    log.info("Creating MistralAI embeddings (model=%s)", EMBEDDING_MODEL)
    return MistralAIEmbeddings(
        model=EMBEDDING_MODEL,
        mistral_api_key=MISTRAL_API_KEY,
    )
