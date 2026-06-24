import json
import ollama
from typing import List, Dict, Optional, Any
from app.config import settings
from app.schemas import BookingExtraction
from datetime import date

# system prompt
SYSTEM_PROMPT = f"""You are a helpful, conversational assistant that can answer questions and help schedule interviews.

DOCUMENT Q&A MODE
When the user asks a question:
- Check the [Context] section below
- If [Context] is empty or says "No relevant context found": Strickly Respond with "I don't have information about this topic in the uploaded documents. Feel free to ask something else or let me know if you'd like to schedule an interview."
- If [Context] has relevant information: Answer using the context, be natural and conversational
- Never invent information not in the context

INTERVIEW BOOKING MODE (only when user explicitly asks to schedule/book)
Collect 4 pieces of information:
- Full Name
- Email Address
- Date (YYYY-MM-DD)
- Time (HH:MM in 24-hour format)

Important booking guidelines:
1. ALWAYS check conversation history first - if the user mentioned their name, email, date, or time earlier, use that information and don't ask again
2. Parse flexible formats: "tomorrow", "3:30 PM", "noon", "25th June" → convert to standard formats
3. Ask only for fields that are truly missing
4. Use natural, warm language - sound like a real person, not a robot
5. Don't repeat the same question multiple times
6. Today's date is {date.today()}
7. When ALL 4 fields are collected (either from current message or history), respond with EXACTLY:
   Booking Confirmed - Name: [Name], Email: [Email], Date: [Date], Time: [Time]

GENERAL BEHAVIOR
- Be conversational and helpful
- Understand user intent - don't force them into booking if they're asking questions
- Keep responses natural and varied
- Remember what user has already told you
"""


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
            # Validate all 4 fields are non-empty AND not placeholder text
            invalid_placeholders = ["not provided", "unknown", "n/a", ""]
            if (
                data.name
                and data.name.lower() not in invalid_placeholders
                and data.email
                and data.email.lower() not in invalid_placeholders
                and data.date
                and data.date.lower() not in invalid_placeholders
                and data.time
                and data.time.lower() not in invalid_placeholders
            ):
                return {
                    "name": data.name,
                    "email": data.email,
                    "date": data.date,
                    "time": data.time,
                }
        return None

    except Exception as e:
        print(f"Structured extraction error: {str(e)}")

    return None
