"""Redis cache for rate limiting and response caching."""

import hashlib
import json
import os

import redis.asyncio as redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_redis: redis.Redis | None = None


async def get_redis() -> redis.Redis:
    """Get the shared Redis connection."""
    global _redis
    if _redis is None:
        _redis = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis


async def close_redis():
    """Close the Redis connection on shutdown."""
    global _redis
    if _redis is not None:
        await _redis.close()
        _redis = None


def _cache_key(prompt: str, precision_logic: float, precision_eq: float) -> str:
    """Build a deterministic cache key from request parameters."""
    raw = f"{prompt}:{precision_logic:.2f}:{precision_eq:.2f}"
    return f"clsa:cache:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


async def get_cached_response(
    prompt: str, precision_logic: float, precision_eq: float
) -> dict | None:
    """Look up a cached model response. Returns None on miss."""
    r = await get_redis()
    key = _cache_key(prompt, precision_logic, precision_eq)
    data = await r.get(key)
    if data is not None:
        return json.loads(data)
    return None


async def set_cached_response(
    prompt: str,
    precision_logic: float,
    precision_eq: float,
    response: dict,
    ttl_seconds: int = 300,
) -> None:
    """Cache a model response with a TTL."""
    r = await get_redis()
    key = _cache_key(prompt, precision_logic, precision_eq)
    await r.setex(key, ttl_seconds, json.dumps(response))
