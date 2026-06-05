

from config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    GROQ_API_KEY,
    OPENAI_CHAT_MODEL,
    GROQ_CHAT_MODEL,
)
from logger import get_logger

log = get_logger("llm")


def get_llm():
    log.info("Creating LLM: provider=%s", LLM_PROVIDER)
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        log.debug("Using ChatGroq model=%s", GROQ_CHAT_MODEL)
        return ChatGroq(model=GROQ_CHAT_MODEL, api_key=GROQ_API_KEY)

    from langchain_openai import ChatOpenAI
    log.debug("Using ChatOpenAI model=%s", OPENAI_CHAT_MODEL)
    return ChatOpenAI(model=OPENAI_CHAT_MODEL, openai_api_key=OPENAI_API_KEY)
