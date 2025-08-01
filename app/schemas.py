from pydantic import BaseModel, Field

class DocumentRequest(BaseModel):
    """
    Schema for the request to index a new document.
    """
    context: str = Field(
        ...,
        min_length=10,
        description="The full document or text to be indexed."
    )

class ChatRequest(BaseModel):
    """
    Schema for the request to generate a response.
    """
    prompt: str = Field(
        ...,
        min_length=2,
        description="The user's question to be answered based on the indexed context."
    )

class ChatResponse(BaseModel):
    """
    Schema for the AI's response.
    """
    response: str

class IndexResponse(BaseModel):
    """
    Schema for the response after indexing a document.
    """
    message: str
    documents_added: int

class GeneralResponse(BaseModel):
    """
    A generic response model for simple status messages.
    """
    message: str
