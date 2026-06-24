from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional, List


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
    chunks: List[ChunkResponse] = []

    model_config = ConfigDict(from_attributes=True)
