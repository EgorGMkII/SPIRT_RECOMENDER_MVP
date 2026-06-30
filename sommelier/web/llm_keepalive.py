"""Optional LLM warm-up and keep-alive for deployed web processes."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

WARMUP_PROMPT = "Reply with exactly: OK"


def warmup_llm(model_factory: Callable[[], Any] | None = None) -> None:
    """Make one minimal model call through the application's configured client."""

    if model_factory is None:
        from llm_module import get_langchain_openai_chat_model

        model_factory = get_langchain_openai_chat_model
    started = perf_counter()
    model_factory().invoke(WARMUP_PROMPT)
    logger.info("LLM warm-up completed in %.2f seconds", perf_counter() - started)


async def warmup_llm_safely(
    model_factory: Callable[[], Any] | None = None,
) -> bool:
    """Warm the model without blocking the event loop or failing application startup."""

    try:
        await asyncio.to_thread(warmup_llm, model_factory)
        return True
    except Exception:
        logger.exception("LLM warm-up failed")
        return False


async def keepalive_loop(
    interval_seconds: float,
    model_factory: Callable[[], Any] | None = None,
) -> None:
    """Keep the configured upstream model warm until the task is cancelled."""

    while True:
        await asyncio.sleep(interval_seconds)
        await warmup_llm_safely(model_factory)
