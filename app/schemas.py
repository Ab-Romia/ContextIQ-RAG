from pydantic import BaseModel, Field


class DocumentRequest(BaseModel):
    context: str = Field(..., min_length=10, description="The full document or text to be indexed.")


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=2,
                        description="The user's question to be answered based on the indexed context.")


class ChatResponse(BaseModel):
    response: str


class IndexResponse(BaseModel):
    message: str
    documents_added: int