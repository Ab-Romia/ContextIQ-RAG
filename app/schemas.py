from pydantic import BaseModel, Field
from typing import Optional, List


class ConversationMessage(BaseModel):
    """Schema for a single message in the conversation history"""
    role: str = Field(..., description="Role of the message sender ('user' or 'assistant')")
    content: str = Field(..., description="Content of the message")


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
    conversation_history: Optional[List[ConversationMessage]] = Field(
        default=None,
        description="Optional conversation history for context-aware responses"
    )

class TaskRequest(BaseModel):
    """
    Schema for executing a specific task like summarization or planning.
    """
    context: str = Field(..., description="The full context for the task.")
    task_type: str = Field(..., description="The type of task to perform (e.g., 'summarize', 'plan').")
    prompt: Optional[str] = Field(None, description="An optional prompt to guide the task.")

class ApiKeyRequest(BaseModel):
    """
    Schema for API key testing.
    """
    api_key: str = Field(
        ...,
        min_length=10,
        description="The API key to test (OpenRouter or OpenAI)."
    )
    provider: Optional[str] = Field(
        None,
        description="The provider for the API key ('openrouter' or 'openai'). Auto-detected if not provided."
    )

class ChatResponse(BaseModel):
    """
    Schema for the AI's response.
    """
    response: str

class TaskResponse(BaseModel):
    """
    Schema for the result of a task.
    """
    result: str

class IndexResponse(BaseModel):
    """
    Schema for the response after indexing a document.
    """
    message: str
    documents_added: int
    # ✨ ADDED: Field to return extracted text to the UI
    extracted_text: Optional[str] = None

class GeneralResponse(BaseModel):
    """
    A generic response model for simple status messages.
    """
    message: str

class ApiKeyTestResponse(BaseModel):
    """
    Schema for API key test response.
    """
    valid: bool
    message: str
    model_info: Optional[dict] = None