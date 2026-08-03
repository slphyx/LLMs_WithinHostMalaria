"""Audited web search through OpenRouter's server tool."""

from __future__ import annotations

import json
import os

import httpx
from inspect_ai.tool import Tool, ToolError, tool
from inspect_ai.util import store


@tool(parallel=False)
def openrouter_web_search(model: str, max_searches: int = 6) -> Tool:
    async def execute(query: str) -> str:
        """Search the web and return a concise synthesis with source links.

        Args:
            query: A focused factual or literature-search query.
        """
        calls = store().get("web_search_calls", [])
        if len(calls) >= max_searches:
            raise ToolError(f"The {max_searches}-search limit has been reached.")

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ToolError("OPENROUTER_API_KEY is not set.")

        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Search for the query below. Prefer primary research "
                                "and official sources. Return concise findings with "
                                f"links.\n\n{query}"
                            ),
                        }
                    ],
                    "tools": [
                        {
                            "type": "openrouter:web_search",
                            "parameters": {
                                "engine": "exa",
                                "max_results": 5,
                                "max_total_results": 5,
                                "search_context_size": "low",
                            },
                        }
                    ],
                    "reasoning": {"effort": "low"},
                    "temperature": 0,
                    "max_tool_calls": 1,
                    "max_tokens": 1200,
                },
            )

        if not response.is_success:
            raise ToolError(f"Web search failed with HTTP {response.status_code}.")

        data = response.json()
        if data.get("error"):
            raise ToolError(f"Web search failed: {data['error'].get('message', 'error')}")

        message = data["choices"][0]["message"]
        sources = []
        for annotation in message.get("annotations") or []:
            citation = annotation.get("url_citation") or {}
            if citation.get("url"):
                sources.append(
                    {"title": citation.get("title"), "url": citation["url"]}
                )

        usage = data.get("usage") or {}
        calls.append(
            {
                "query": query,
                "model": data.get("model", model),
                "generation_id": data.get("id"),
                "usage": usage,
            }
        )
        store().set("web_search_calls", calls)
        store().set(
            "web_search_cost",
            sum((call.get("usage") or {}).get("cost") or 0 for call in calls),
        )
        return json.dumps(
            {"answer": message.get("content", ""), "sources": sources},
            ensure_ascii=False,
        )

    return execute
