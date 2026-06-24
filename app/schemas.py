from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from typing import Optional, List
from datetime import date, time


class ChunkResponse(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    content: str
    vector_id: str

    model_config = ConfigDict(from_attributes=True)


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    chunking_strategy: str
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    chunk_count: int
    chunks: List[ChunkResponse] = []

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    chunking_strategy: str
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    chunk_count: int

    model_config = ConfigDict(from_attributes=True)


class BookingResponse(BaseModel):
    id: str
    conversation_id: str
    name: str
    email: str
    date: date
    time: time
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatMessageResponse(BaseModel):
    id: int
    role: str
    content: str
    retrieved_chunk_ids: List[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = Field(
        None, description="Optional conversation UUID"
    )
    message: str = Field(..., description="The user's query or message")


class ChatResponse(BaseModel):
    conversation_id: str
    assistant_message: str
    booking: Optional[BookingResponse] = None
    context_used: List[str] = Field(
        default_factory=list,
        description="List of chunk IDs referenced during vector store retrieval.",
    )

    model_config = ConfigDict(from_attributes=True)
