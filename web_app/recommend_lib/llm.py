import os
import requests
import json
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

def generate_book_recommendations(prompt: str) -> dict:
    """
    Generates book recommendations using a local llama-server.

    Args:
        prompt (str): The prompt to send to the LLM.

    Returns:
        dict: A dictionary mimicking the standard LLM response object (has a .text attribute),
              or None if an error occurs. 
              The .text attribute contains the JSON string response.
    """
    
    server_url = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080/v1/chat/completions")
    
    # JSON schema for the output to ensure structured data
    # Note: explicit grammar/json_schema support depends on the server implementation.
    # standard llama-server supports json_schema in the body or grammar.
    # We will try to rely on the system prompt for now, or see if we can pass response_format.
    
    # We'll use a response class to mimic the standard response object for minimal refactoring
    class Response:
        def __init__(self, text):
            self.text = text

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful librarian. You must output ONLY JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "response_format": {
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "title": {"type": "string"},
                                "author": {"type": "string"},
                                "reason": {"type": "string"}
                            },
                            "required": ["id", "title", "author", "reason"]
                        }
                    }
                },
                "required": ["items"]
            }
        }
    }

    try:
        logger.info(f"Sending request to {server_url}")
        response = requests.post(server_url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        
        data = response.json()
        
        # Openai format: choices[0].message.content
        content = data['choices'][0]['message']['content']
        logger.debug(f"LLM Response: {content}")
        
        return Response(content)

    except Exception as e:
        logger.error(f"Error communicating with local LLM: {e}")
        return None
