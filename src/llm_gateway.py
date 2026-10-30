import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import APIConnectionError, Groq, InternalServerError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/llm_gateway.log"), logging.StreamHandler()],
)
logger = logging.getLogger("llm_gateway")

MAIN_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"

_client = Groq()

_MIN_INTERVAL = 1.0
_last_call_time = 0.0


def _wait_for_rate_limit():
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_call_time = time.time()


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, InternalServerError)),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _call_model(model, messages, temperature, max_tokens, reasoning_effort=None):
    _wait_for_rate_limit()
    logger.info(f"Calling model={model} (messages={len(messages)}, temperature={temperature}, reasoning_effort={reasoning_effort})")
    kwargs = {}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    return _client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, **kwargs,
    )


def call_llm(messages, temperature=0.0, max_tokens=1024, reasoning_effort=None):
    """
    Calls the main model with retry/backoff on transient errors.
    If the main model still fails after retries, falls back to the fallback model.
    reasoning_effort: 'low' / 'medium' / 'high' — controls reasoning-token budget for gpt-oss models.
    Returns (response_text, model_actually_used).
    """
    try:
        response = _call_model(MAIN_MODEL, messages, temperature, max_tokens, reasoning_effort)
        logger.info(f"Success with main model ({MAIN_MODEL})")
        return response.choices[0].message.content, MAIN_MODEL
    except Exception as e:
        logger.warning(f"Main model ({MAIN_MODEL}) failed after retries: {e}. Falling back to {FALLBACK_MODEL}")
        try:
            response = _call_model(FALLBACK_MODEL, messages, temperature, max_tokens, reasoning_effort)
            logger.info(f"Success with fallback model ({FALLBACK_MODEL})")
            return response.choices[0].message.content, FALLBACK_MODEL
        except Exception as e2:
            logger.error(f"Fallback model ({FALLBACK_MODEL}) also failed: {e2}")
            raise


if __name__ == "__main__":
    text, model_used = call_llm([{"role": "user", "content": "Explain RAG in one sentence."}])
    print(f"[model used: {model_used}]")
    print(text)