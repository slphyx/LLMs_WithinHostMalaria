"""extracts billing from the inspect events to be used in cost analysis"""

from __future__ import annotations

from typing import Any


def model_billing(sample: Any) -> tuple[float, list[dict]]:
    """Return deduplicated OpenRouter charges for one sample."""
    calls, seen = [], set()
    for event in sample.events:
        call = getattr(event, "call", None)
        response = call.response if call else None
        usage = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(usage, dict) or usage.get("cost") is None:
            continue
        generation_id = response.get("id")
        identity = generation_id or event.uuid
        if identity in seen:
            continue
        seen.add(identity)
        calls.append(
            {
                "model": str(event.model),
                "served_model": response.get("model"),
                "provider": response.get("provider"),
                "generation_id": generation_id,
                "service_tier": response.get("service_tier"),
                "cost": usage["cost"],
                "cost_details": usage.get("cost_details"),
                "is_byok": usage.get("is_byok"),
            }
        )
    return sum(call["cost"] for call in calls), calls


def billing_for_model(calls: list[dict], model: str | None) -> dict | None:
    """Aggregate all calls for a requested model without dropping agent turns."""
    selected = [call for call in calls if call["model"] == model]
    if not selected:
        return None
    return {
        "source": "openrouter_response",
        "total_cost": sum(call["cost"] for call in selected),
        "calls": selected,
    }
