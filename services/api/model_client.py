"""HTTP client for communicating with the CLSA model service."""

import os
import logging

import httpx

logger = logging.getLogger(__name__)

MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8001")


async def call_model_service(
    prompt: str,
    precision_logic: float = 1.0,
    precision_eq: float = 1.0,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    return_deliberation: bool = False,
) -> dict:
    """Send a generation request to the model service.

    Returns the parsed JSON response from the model service.
    Raises httpx.HTTPStatusError on non-2xx responses.
    """
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "precision": {"logic": precision_logic, "eq": precision_eq},
        "temperature": temperature,
        "return_deliberation": return_deliberation,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{MODEL_SERVICE_URL}/generate", json=payload
        )
        response.raise_for_status()
        return response.json()
