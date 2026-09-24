"""Price the replay: what would each strategy have cost at list API prices?"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .replay import Hit, Request


@dataclass(frozen=True)
class Prices:
    """USD per 1M tokens. Defaults are published list prices; dated in the
    results so readers can re-price."""

    name: str
    input_per_m: float
    output_per_m: float
    embed_per_m: float = 0.02  # text-embedding-3-small list price
    as_of: str = "2025"


PRICE_SHEETS = {
    "gpt-4o-mini": Prices("gpt-4o-mini", 0.15, 0.60),
    "gpt-4o": Prices("gpt-4o", 2.50, 10.00),
}


def spend(reqs: Sequence[Request], hits: Sequence[Hit], prices: Prices, semantic: bool) -> dict[str, float]:
    """Total spend with the cache vs without. Semantic caches embed every
    request (hit or miss); exact caches don't pay for embeddings."""
    served = {h.i for h in hits}
    base = sum(r.in_tokens * prices.input_per_m + r.out_tokens * prices.output_per_m for r in reqs) / 1e6
    llm = (
        sum(
            r.in_tokens * prices.input_per_m + r.out_tokens * prices.output_per_m
            for i, r in enumerate(reqs)
            if i not in served
        )
        / 1e6
    )
    emb = sum(r.in_tokens for r in reqs) * prices.embed_per_m / 1e6 if semantic else 0.0
    total = llm + emb
    return {
        "baseline_usd": round(base, 6),
        "with_cache_usd": round(total, 6),
        "embedding_usd": round(emb, 6),
        "saved_usd": round(base - total, 6),
        "saved_fraction": round((base - total) / base, 5) if base else 0.0,
    }
