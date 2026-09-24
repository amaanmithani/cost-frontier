from datetime import datetime, timedelta

import numpy as np

from costfrontier.cost import Prices, spend
from costfrontier.judge import wilson
from costfrontier.replay import Request, exact_hits, guard_key, normalize, semantic_hits

T0 = datetime(2023, 4, 9)


def req(i: int, prompt: str, user: str = "u1", hours: float = 0) -> Request:
    return Request(
        ts=T0 + timedelta(hours=hours or i),
        user=user,
        prompt=prompt,
        response=f"answer to {prompt}",
        in_tokens=10,
        out_tokens=90,
    )


def test_normalize_and_guard():
    assert normalize("  Hello,   World!! ") == "hello world"
    assert guard_key("Best hotels in Udaipur?") != guard_key("Best hotels in Munnar?")
    assert guard_key("What's the capital of France") == guard_key("what the capital of France is")
    assert guard_key("") == ""


def test_exact_hits_scope_and_ttl():
    rs = [req(0, "hi", "a"), req(1, "Hi!", "b"), req(2, "hi", "b"), req(3, "hi", "a")]
    assert [(h.i, h.j) for h in exact_hits(rs, "raw", "global", None)] == [(2, 0), (3, 0)]
    assert [(h.i, h.j) for h in exact_hits(rs, "normalized", "global", None)] == [(1, 0), (2, 0), (3, 0)]
    assert [(h.i, h.j) for h in exact_hits(rs, "raw", "user", None)] == [(3, 0)]
    # TTL: request 3 is 3h after request 0.
    assert [(h.i, h.j) for h in exact_hits(rs, "raw", "global", 2.5)] == [(2, 0)]


def test_semantic_hits_threshold_guard_and_only_misses_cached():
    rs = [
        req(0, "What is the capital of France"),
        req(1, "Capital of France?"),
        req(2, "What is the capital of France"),
        req(3, "Best hotels in Udaipur"),
        req(4, "Best hotels in Munnar"),
    ]
    e = np.array([[1, 0], [0.99, 0.141], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    hits = semantic_hits(rs, e, 0.95, "global", None, guard=False, block=2)
    assert [(h.i, h.j) for h in hits] == [(1, 0), (2, 0), (4, 3)]
    guarded = semantic_hits(rs, e, 0.95, "global", None, guard=True, block=2)
    assert (4, 3) not in [(h.i, h.j) for h in guarded]  # entity swap blocked
    assert semantic_hits(rs, e, 0.999, "user", 0.5, guard=False) == []


def test_spend():
    rs = [req(0, "a"), req(1, "a")]
    hits = exact_hits(rs, "raw", "global", None)
    p = Prices("x", input_per_m=1.0, output_per_m=10.0, embed_per_m=0.5)
    s = spend(rs, hits, p, semantic=False)
    assert s["baseline_usd"] == round(2 * (10 + 900) / 1e6, 6) and s["saved_fraction"] == 0.5
    sem = spend(rs, hits, p, semantic=True)
    assert sem["embedding_usd"] > 0 and sem["saved_usd"] < s["saved_usd"]
    assert spend([], [], p, False)["saved_fraction"] == 0.0


def test_wilson():
    p, lo, hi = wilson(0, 100)
    assert p == 0 and lo == 0 and 0.03 < hi < 0.04
    assert wilson(0, 0) == (0.0, 0.0, 0.0)
