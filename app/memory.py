import json
from typing import List, Dict
from redis import Redis
from app.config import settings


class ConversationMemory:
    def __init__(self) -> None:
        self.client = Redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.ttl = 86400  # 24 hours

    def _key(self, conversation_id: str) -> str:
        return f"chat_memory:{conversation_id}"

    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Returns full conversation history"""
        key = self._key(conversation_id)

        try:
            messages = self.client.lrange(key, 0, -1)
            return [json.loads(msg) for msg in messages]
        except Exception as e:
            print(f"Redis memory tracking failed: {str(e)}")
            return []

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """Appends a message to Redis list"""
        key = self._key(conversation_id)

        message = json.dumps({"role": role, "content": content})

        try:
            self.client.rpush(key, message)
            self.client.expire(key, self.ttl)
        except Exception:
            pass

    def clear_history(self, conversation_id: str) -> None:
        """Optional: clears conversation memory"""
        key = self._key(conversation_id)
        self.client.delete(key)
