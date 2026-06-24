from datetime import datetime
from typing import Generator, List
from sqlalchemy import (
    create_engine,
    ForeignKey,
    String,
    Text,
    DateTime,
    Integer,
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
