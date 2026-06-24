import uuid
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.database import BookingModel, ChatMessageModel, ConversationModel

from app.llm import call_ollama, extract_booking
from app.memory import ConversationMemory
from app.rag import retrieve_chunks, format_context


class ChatManager:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.memory = ConversationMemory()

    def process_message(
        self, user_message: str, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

            conversation = ConversationModel(id=conversation_id)

            self.db.add(conversation)
            self.db.flush()
        else:
            conversation = (
                self.db.query(ConversationModel)
                .filter(ConversationModel.id == conversation_id)
                .first()
            )

            if conversation is None:
                raise ValueError(f"Conversation '{conversation_id}' not found")

        history = self.memory.get_history(conversation_id)

        retrieved_chunks = retrieve_chunks(query=user_message, top_k=3)

        # Filter chunks by similarity score threshold
        CONTEXT_THRESHOLD = 0.3  # Only use chunks with score > 0.3
        high_quality_chunks = [
            c for c in retrieved_chunks if c["similarity_score"] > CONTEXT_THRESHOLD
        ]

        if high_quality_chunks:
            context = format_context(high_quality_chunks)
            context_used = [c["vector_id"] for c in high_quality_chunks]
        else:
            context = ""
            context_used = []

        enriched_message = f"""
            The following information was retrieved from uploaded documents.

            Use it to answer the user's question.

            Context:
            {context}

            User Question:
            {user_message}
            """

        # Label history messages clearly
        labeled_history = []
        for msg in history:
            labeled_msg = msg.copy()
            labeled_msg["content"] = f"[HISTORY] {msg['content']}"
            labeled_history.append(labeled_msg)

        # Label current user message
        labeled_current = f"""[CURRENT USER MESSAGE]
        {enriched_message}"""

        messages = labeled_history + [{"role": "user", "content": labeled_current}]
        print(f"\n-------\nThis is the message thats going to llm {messages}\n-----\n")

        # now on to generating response
        assistant_response = call_ollama(messages)

        booking_data = extract_booking(assistant_response)

        booking = None

        try:
            if booking_data:
                booking = BookingModel(
                    id=str(uuid.uuid4()),
                    conversation_id=conversation_id,
                    name=booking_data["name"],
                    email=booking_data["email"],
                    date=str(booking_data["date"]),
                    time=str(booking_data["time"]),
                    status="pending",
                )

                self.db.add(booking)

            # Save user message
            self.db.add(
                ChatMessageModel(
                    conversation_id=conversation_id,
                    role="user",
                    content=user_message,
                    retrieved_chunk_ids=context_used,
                )
            )

            # Save assistant response
            self.db.add(
                ChatMessageModel(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=assistant_response,
                    retrieved_chunk_ids=[],
                )
            )
            # update redis memory finally
            self.memory.add_message(
                conversation_id=conversation_id,
                role="user",
                content=user_message,
            )

            self.memory.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=assistant_response,
            )
            self.db.commit()

        except Exception as e:
            self.db.rollback()
            print(f"Got exception during getting booking info: {str(e)}")
            raise

        return {
            "conversation_id": conversation_id,
            "assistant_message": assistant_response,
            "context_used": context_used,
            "booking": booking,
        }
