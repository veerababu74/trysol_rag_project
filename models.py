from pydantic import BaseModel
class ChatRequest(BaseModel):
    question: str
    k: int = 4  # how many chunks to retrieve


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]