from datetime import datetime
from typing import Generator, List
from sqlalchemy import (
    create_engine,
    ForeignKey,
    String,
    Text,
    DateTime,
    Integer,
    JSON,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
    Session,
)
from app.config import settings

## setting up engine and the session factory
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={
        "check_same_thread": False
    },  # so that the same thread that created it uses it
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class ConversationModel(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    messages: Mapped[List["ChatMessageModel"]] = relationship(
        "ChatMessageModel", back_populates="conversation", cascade="all, delete-orphan"
    )
    bookings: Mapped[List["BookingModel"]] = relationship(
        "BookingModel", back_populates="conversation", cascade="all, delete-orphan"
    )


class ChatMessageModel(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String, nullable=False)  # "user" or "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunk_ids: Mapped[List[str]] = mapped_column(
        JSON, default=list, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    conversation: Mapped["ConversationModel"] = relationship(
        "ConversationModel", back_populates="messages"
    )


class BookingModel(Base):
    __tablename__ = "bookings"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    conversation_id: Mapped[str] = mapped_column(
        String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, nullable=False)
    date: Mapped[str] = mapped_column(String, nullable=False)
    time: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(
        String, default="pending"
    )  # e.g., pending, confirmed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    conversation: Mapped["ConversationModel"] = relationship(
        "ConversationModel", back_populates="bookings"
    )


class DocumentModel(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_type: Mapped[str] = mapped_column(String, nullable=False)  # e.g., "pdf", "txt"
    chunking_strategy: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(
        String, default="pending"
    )  # e.g., pending, processing, completed, failed
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # now since we ahve one to many relationship to chunks
    chunks: Mapped[List["ChunkModel"]] = relationship(
        "ChunkModel", back_populates="document", cascade="all, delete-orphan"
    )


class ChunkModel(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    vector_id: Mapped[str] = mapped_column(
        String, nullable=False
    )  # Points to Qdrant Point ID
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now()
    )

    # Relationship back to Document
    document: Mapped["DocumentModel"] = relationship(
        "DocumentModel", back_populates="chunks"
    )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Startup tables initialization
def init_db():
    Base.metadata.create_all(bind=engine)
