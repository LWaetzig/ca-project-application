import os
from typing import Any

from ollama import Client

SYSTEM_PROMPT = (
    "You are a helpful data assistant. Answer using ONLY the provided context.\n"
    "Cite short snippets with bracketed dates when relevant. If the answer "
    "is not in context, say you don't know instead of guessing."
)

DEFAULT_LLM_BASE = os.environ.get("LLM_BASE_URL", "http://ollama:11434")
DEFAULT_LLM_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")


def request_ollama(
    messages: list[dict[str, str]],
    base_url: str = DEFAULT_LLM_BASE,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.2,
) -> str:

    client = Client(host=base_url, headers={"x-some-header": "some-value"})
    response = client.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
        stream=False,
    )

    return response.message.content or ""


def retrieve_context(query: str) -> tuple[str, list[dict[str, Any]]]:
    return "", []


def answer_with_retrieval(user_query: str) -> tuple[str, list[dict[str, Any]]]:
    context, sources = retrieve_context(user_query)

    found_context = bool(context and context.strip())
    if found_context:
        user_content = f"Context:\n{context}\n\nQuestion: {user_query}"
        system_prompt = SYSTEM_PROMPT
    else:
        # No context -> allow a general answer and be explicit about limitations
        system_prompt = (
            "You are a helpful assistant. No domain context is available for this query. "
            "Answer helpfully and concisely. If the user asked about the Bundestag data, "
            "say that you couldn't find relevant context and offer to refine the query."
        )
        user_content = user_query

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        answer = request_ollama(messages)

    except Exception as e:
        answer = f"LLM-Fehler: {e}"

    return answer, sources
