"""OpenRouter catalogue prices used by Inspect's per-sample cost limits."""

from __future__ import annotations

import httpx
from inspect_ai.model import ModelCost, ModelInfo, get_model_info, set_model_info


def model_cost(pricing: dict) -> ModelCost:
    """Convert OpenRouter per-token prices to Inspect per-million prices."""
    return ModelCost(
        input=1_000_000 * float(pricing["prompt"]),
        output=1_000_000 * float(pricing["completion"]),
        input_cache_write=1_000_000
        * float(pricing.get("input_cache_write", pricing["prompt"])),
        input_cache_read=1_000_000
        * float(pricing.get("input_cache_read", pricing["prompt"])),
    )


def openrouter_costs(models: list[str]) -> dict[str, ModelCost]:
    """Fetch and register current prices for the requested OpenRouter models."""
    response = httpx.get("https://openrouter.ai/api/v1/models", timeout=30)
    response.raise_for_status()
    catalogue = {row["id"]: row["pricing"] for row in response.json()["data"]}
    costs = {}
    for model in models:
        if not model.startswith("openrouter/"):
            raise SystemExit(f"Models with cost limits must use OpenRouter: {model}")
        model_id = model.removeprefix("openrouter/")
        if model_id not in catalogue:
            raise SystemExit(f"OpenRouter model is unavailable: {model_id}")
        costs[model] = model_cost(catalogue[model_id])
        if get_model_info(model) is None:
            set_model_info(model, ModelInfo(model=model_id))
    return costs
