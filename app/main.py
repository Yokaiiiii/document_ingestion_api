from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status
from app.config import settings
from app.database import init_db, get_db, DocumentModel, ChunkModel
from sqlalchemy.orm import Session
import os
import uuid
import shutil
from app.services import (
    extract_text,
    generate_embeddings,
    VectorStoreService,
    chunk_text,
    EmbeddingModelLoader,
)
from app.schemas import DocumentResponse, DocumentListResponse
from typing import List


@asynccontextmanager
async def lifespan(app: FastAPI):
    # this will initialize the db before the server starts
    init_db()

    EmbeddingModelLoader()  # doing this so that we load the model even before accepting traffic

    v_store = VectorStoreService()
    v_store.ensure_collection_exists()

    yield


app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan,
)  # we are importing the app name from the env file itself


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "database_configured": bool(settings.DATABASE_URL),
        "qdrant_path_configured": bool(settings.QDRANT_PATH),
    }


@app.post(
    "/api/v1/documents/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    chunking_strategy: str = Form(...),
    db: Session = Depends(get_db),
):
    # validating the inputs
    if chunking_strategy not in ["fixed_size", "semantic"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported chunking strategy '{chunking_strategy}'. Only 'fixed_size' and 'semantic' is supported.",
        )

    filename = file.filename or "unknown"
    extension = os.path.splitext(filename)[1].lower()
    if extension not in [".pdf", ".txt"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{extension}'. Only  '.pdf', '.txt' is supported.",
        )

    # now making a uuid
    document_id = str(uuid.uuid4())
    saved_filename = f"{document_id}{extension}"
    file_path = os.path.join(settings.UPLOAD_DIR, saved_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Filed to write file to local dist engine storage: {str(e)}",
        )

    # now on to sql part
    db_document = DocumentModel(
        id=document_id,
        filename=filename,
        file_type=extension.replace(".", ""),
        chunking_strategy=chunking_strategy,
        status="processing",
        chunk_count=0,
    )

    db.add(db_document)
    db.commit()
    db.refresh(db_document)

    vector_ids_allocated = []

    try:
        extracted_text = extract_text(file_path)
        if not extracted_text.strip():
            raise ValueError(f"The upload document has no text in it")

        chunks = chunk_text(extracted_text, strategy=chunking_strategy)
        if not chunks:
            raise ValueError("Document chunking resulted in zero chunks")

        embeddings = generate_embeddings(chunks)

        v_store = VectorStoreService()
        vector_ids_allocated = [str(uuid.uuid4()) for _ in chunks]
        payloads = [
            {
                "document_id": document_id,
                "chunk_index": idx,
                "filename": filename,
            }
            for idx in range(len(chunks))
        ]

        v_store.upsert_vectors(
            vector_ids=vector_ids_allocated, embeddings=embeddings, payloads=payloads
        )

        for idx, text_content in enumerate(chunks):
            db_chunks = ChunkModel(
                id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=idx,
                content=text_content,
                vector_id=vector_ids_allocated[idx],
            )
            db.add(db_chunks)

        # now again shifting the workflow to done
        db_document.status = "completed"
        db_document.chunk_count = len(chunks)
        db.commit()
        db.refresh(db_document)

        return db_document
    except Exception as pipeline_error:
        db.rollback()

        db_document.status = "failed"
        db_document.error_message = str(pipeline_error)
        db.commit()
        db.refresh(db_document)

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

        if vector_ids_allocated:
            try:
                v_store = VectorStoreService()
                v_store.delete_vectors(vector_ids=vector_ids_allocated)
            except Exception:
                pass

        if isinstance(pipeline_error, ValueError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(pipeline_error)
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal pipeline error: {str(pipeline_error)}",
        )


@app.get(
    "/api/v1/documents",
    response_model=List[DocumentListResponse],
    status_code=status.HTTP_200_OK,
)
def list_documents(db: Session = Depends(get_db)):
    """return a summary of list of all tracked document metadata records"""
    documents = db.query(DocumentModel).order_by(DocumentModel.created_at.desc()).all()

    return documents


@app.get(
    "/api/v1/documents/{document_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
)
def get_document_details(document_id: str, db: Session = Depends(get_db)):
    """Fetching the document indicated by the document_id"""
    document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with document_id {document_id} not found in the database.",
        )

    return document
