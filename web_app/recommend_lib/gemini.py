import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


def generate_book_recommendations(prompt: str) -> dict:
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-pro"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config={
            "include_thoughts": True,
        },
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_NONE",  # Block none
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_NONE",  # Block none
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_NONE",  # Block none
            ),
        ],
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            properties = {
                "items": genai.types.Schema(
                    type = genai.types.Type.ARRAY,
                    items = genai.types.Schema(
                        type = genai.types.Type.OBJECT,
                        required = ["id", "title", "author", "reason"],
                        properties = {
                            "id": genai.types.Schema(
                                type = genai.types.Type.INTEGER,
                                description = "Die ID (Zahl) vom Anfang der Zeile",
                            ),
                            "title": genai.types.Schema(
                                type = genai.types.Type.STRING,
                            ),
                            "author": genai.types.Schema(
                                type = genai.types.Type.STRING,
                            ),
                            "reason": genai.types.Schema(
                                type = genai.types.Type.STRING,
                            ),
                        },
                    ),
                ),
            },
        ),
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None
