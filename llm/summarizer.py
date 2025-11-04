# llm/summarizer.py

import os
from openai import OpenAI

client = None

def initialize_client():
    """Initializes the OpenAI client, reusing it for efficiency."""
    global client
    api_key = os.environ.get('LLM_API_KEY')
    base_url = os.environ.get('LLM_API_BASE_URL') # Optional: for custom endpoints
    if api_key and client is None:
        client = OpenAI(api_key=api_key, base_url=base_url)

def get_summary(content: str, model: str, prompt_template: str) -> str:
    """
    Calls an OpenAI-compatible LLM to get a summary of the given content.
    """
    initialize_client()
    
    if not client:
        return "Error: LLM client not initialized. Check LLM_API_KEY."
    if not content:
        return "(Content was empty, no summary generated)"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": prompt_template
                },
                {
                    "role": "user",
                    "content": content[:15000] # Truncate content to avoid exceeding token limits
                }
            ],
            temperature=0.5,
            timeout=180
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        return f"Error calling LLM API: {e}"
