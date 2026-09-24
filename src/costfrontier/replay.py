"""Replay a request trace through cache strategies.

A request i is a hit if an earlier request j (within the TTL, and in the same
scope: the whole service, or the same user) has an equal key (exact
strategies) or a similar enough embedding (semantic). On a hit, j's actual
response would have been served for i.
"""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

_WS = re.compile(r"\s+")
_PUNCT = str.maketrans("", "", string.punctuation + "“”‘’")


def normalize(text: str) -> str:
    """Case-, whitespace- and punctuation-insensitive key."""
    return _WS.sub(" ", text.translate(_PUNCT).lower()).strip()


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-.]*")


def guard_key(q: str) -> str:
    """Port of ModelMux's cache.GuardKey: the leading word (contractions
    folded) plus the set of capitalised-after-first and numeric tokens.
    Two queries must share it for a semantic hit."""
    toks = _TOKEN.findall(q)
    if not toks:
        return ""
    lead = str(toks[0]).lower().split("'")[0]
    ents = set()
    for i, t in enumerate(toks):
        t = t.strip("'-.")
        if not t:
            continue
        if any(c.isdigit() for c in t) or (i > 0 and t[0].isupper() and t != "I"):
            ents.add(t.lower())
    return lead + "|" + ",".join(sorted(ents))


@dataclass
class Request:
    ts: datetime
    user: str
    prompt: str
    response: str
    in_tokens: int
    out_tokens: int


@dataclass
class Hit:
    i: int  # the request served from cache
    j: int  # the earlier request whose response is served
    similarity: float


def exact_hits(reqs: Sequence[Request], key: str, scope: str, ttl_hours: float | None) -> list[Hit]:
    """key: 'raw' or 'normalized'; scope: 'global' or 'user'."""
    first: dict[tuple[str, str], int] = {}
    hits = []
    for i, r in enumerate(reqs):
        k = (r.user if scope == "user" else "", r.prompt if key == "raw" else normalize(r.prompt))
        j = first.get(k)
        if j is not None and (ttl_hours is None or (r.ts - reqs[j].ts).total_seconds() <= ttl_hours * 3600):
            hits.append(Hit(i, j, 1.0))
            continue  # a hit doesn't refresh the entry (the cached response stays j's)
        first[k] = i
    return hits


def semantic_hits(
    reqs: Sequence[Request],
    emb: np.ndarray,
    threshold: float,
    scope: str,
    ttl_hours: float | None,
    guard: bool,
    block: int = 512,
) -> list[Hit]:
    """Nearest earlier cached request by cosine similarity (rows of emb are
    unit vectors). Only misses are inserted into the cache, as a real cache
    would store only responses it generated."""
    n = len(reqs)
    ts = np.array([r.ts.timestamp() for r in reqs])
    users = np.array([r.user for r in reqs])
    gkeys = np.array([guard_key(r.prompt) for r in reqs]) if guard else None
    cached = np.zeros(n, dtype=bool)
    hits = []
    for start in range(0, n, block):
        stop = min(n, start + block)
        # Similarities of this block against everything before its end.
        sims = emb[start:stop] @ emb[:stop].T
        for bi, i in enumerate(range(start, stop)):
            cand = np.flatnonzero(cached[:i])
            if cand.size:
                ok = np.ones(cand.size, dtype=bool)
                if ttl_hours is not None:
                    ok &= ts[i] - ts[cand] <= ttl_hours * 3600
                if scope == "user":
                    ok &= users[cand] == users[i]
                if gkeys is not None:
                    ok &= gkeys[cand] == gkeys[i]
                cand = cand[ok]
            if cand.size:
                s = sims[bi, cand]
                k = int(np.argmax(s))
                if s[k] >= threshold:
                    hits.append(Hit(i, int(cand[k]), float(s[k])))
                    continue
            cached[i] = True
    return hits
