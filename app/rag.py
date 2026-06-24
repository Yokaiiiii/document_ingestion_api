from typing import List, Dict, Any
from app.services import EmbeddingModelLoader, VectorStoreService


def retrieve_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not query or not query.strip():  # emtpy that is
        return []
    try:
        # first gonna make the vector embedding of the query too
        loader = EmbeddingModelLoader()

        query_vector = loader.model.encode([query], convert_to_numpy=True)[0].tolist()

        vector_store = VectorStoreService()

        search_results = (
            vector_store.search(query_vector=query_vector, top_k=top_k) or []
        )

        results = []
        for point in search_results:
            payload = point.payload or {}
            results.append(
                {
                    "text": payload.get("content", ""),
                    "vector_id": str(point.id),
                    "similarity_score": point.score,
                    "chunk_index": payload.get("chunk_index", 0),
                    "document_id": payload.get("document_id", ""),
                }
            )

        return results
    except Exception as e:
        print(f"RAG retrieval error encountered: {str(e)}")
        return []


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Cleaning the chunks into a clean single string"""
    if not chunks:
        return "No relevant context found in database indices."

    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        block = (
            f"[{i}] From document_id: {chunk['document_id']}, chunk_index: {chunk['chunk_index']} "
            f"(Score: {chunk.get('similarity_score', 0.0):.4f})\n"
            f"{chunk['text'].strip()}"
        )
        context_blocks.append(block)

    return "\n\n".join(context_blocks)
