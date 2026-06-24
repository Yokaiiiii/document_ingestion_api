import json
import ollama
from typing import List, Dict, Optional, Any
from app.config import settings
from app.schemas import BookingExtraction

# system prompt
SYSTEM_PROMPT = """You are a professional interview scheduling assistant. 
Your sole objective is to help users seamlessly schedule their interviews.

Guidelines:
1. Be polite, concise, and helpful.
2. You must collect exactly four pieces of information before a booking can be finalized:
   - Full Name
   - Email Address
   - Date (Format: YYYY-MM-DD)
   - Time (Format: HH:MM or HH:MM AM/PM)
3. Prompt for missing information naturally. Do not ask for everything all at once if it overwhelms the user.
4. Crucial: Once the user provides all 4 required fields, you must explicitly confirm them back to the user in your final response using this exact string format so the backend parser can see it:
   Booking Confirmed - Name: [Name], Email: [Email], Date: [Date], Time: [Time]"""


def call_ollama(
    message: List[Dict[str, str]], system_prompt: str = SYSTEM_PROMPT
) -> str:
    try:
        # Ensure host is a string (settings may have a specialized type)
        client = ollama.Client(host=settings.OLLAMA_API_URL)

        full_message = [{"role": "system", "content": system_prompt}] + message

        response = client.chat(
            model=settings.OLLAMA_MODEL_NAME,
            messages=full_message,
            options={"temperature": 0.3},
        )

        return response.get("message", {}).get("content", "").strip()

    except Exception as e:
        print(f"Ollama client invocation failure: {str(e)}")
        return "I am currently experiencing technical difficulties processing your request. Please try again later."


def extract_booking(response: str) -> Optional[Dict[str, Any]]:
    """
    Guarantees structured extraction by forcing Mistral to respond
    strictly according to a Pydantic schema using Ollama's format parameter.
    """
    if not response or not response.strip():
        return None

    extraction_prompt = f'Extract booking details from this message: "{response}"'

    try:
        client = ollama.Client(host=settings.OLLAMA_API_URL)

        # We pass the Pydantic schema into the 'format' parameter
        extraction_response = client.chat(
            model=settings.OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": extraction_prompt}],
            options={"temperature": 0.0},
            format=BookingExtraction.model_json_schema(),  # <--- Forces JSON Schema
        )

        raw_output = extraction_response.get("message", {}).get("content", "").strip()

        # Safe translation directly into your Pydantic object
        data = BookingExtraction.model_validate_json(raw_output)

        if data.extracted:
            return {
                "name": data.name,
                "email": data.email,
                "date": data.date,
                "time": data.time,
            }

    except Exception as e:
        print(f"Structured extraction error: {str(e)}")

    return None
