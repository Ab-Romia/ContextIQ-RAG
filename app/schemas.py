from pydantic import BaseModel, Field
from typing import Optional


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


class TaskRequest(BaseModel):
    """
    Schema for executing a specific task like summarization.
    """
    task_name: str = Field(
        ...,
        description="The name of the task to perform, e.g., 'summarize'."
    )


class IndexResponse(BaseModel):
    """
    Schema for the response after indexing a document.
    """
    message: str = Field(
        ...,
        description="A confirmation message."
    )
    chunks_added: int = Field(
        ...,
        description="The number of document chunks added to the index."
    )


class ChatResponse(BaseModel):
    """
    Schema for the response from a chat query.
    """
    response: str = Field(
        ...,
        description="The generated response from the AI."
    )


class TaskResponse(BaseModel):
    """
    Schema for the response from a task execution.
    """
    task_name: str = Field(
        ...,
        description="The name of the task that was performed."
    )
    result: str = Field(
        ...,
        description="The result of the task execution."
    )


class ApiKeyRequest(BaseModel):
    """
    Schema for the request to test an API key.
    """
    api_key: str = Field(
        ...,
        min_length=1,
        description="The OpenRouter API key to test."
    )


class ApiKeyTestResponse(BaseModel):
    """
    Schema for the response from the API key test endpoint.
    """
    valid: bool
    message: str
    model_info: Optional[dict]