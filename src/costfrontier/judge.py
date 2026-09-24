"""Would a cached answer be acceptable for a new prompt?"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import httpx

PROMPT = """A user sent the NEW message below. Instead of writing a fresh reply, a cache
returned the REPLY, which was originally written for the OLD message.

Is the REPLY an acceptable answer to the NEW message? Acceptable means it
addresses what the NEW message actually asks, with no wrong or missing specifics
(names, numbers, places, languages, formats, lengths). A reply to a different
question, or one that ignores a detail of the NEW message, is not acceptable.
Answer with exactly one word: yes or no.

OLD message:
{old}

NEW message:
{new}

REPLY:
{reply}"""


class AcceptJudge:
    """LLM judge over an OpenAI-compatible endpoint, cached on disk so a
    replay can be resumed and re-reported without re-asking."""

    def __init__(self, model: str, base_url: str, cache_path: Path, max_chars: int = 3000) -> None:
        self.model = model
        self.http = httpx.Client(base_url=base_url, timeout=600)
        self.path = cache_path
        self.cache: dict[str, bool] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self.max = max_chars

    def _key(self, old: str, new: str, reply: str) -> str:
        return hashlib.sha256(f"{self.model}\x00{old}\x00{new}\x00{reply}".encode()).hexdigest()

    def acceptable(self, old: str, new: str, reply: str) -> bool:
        k = self._key(old, new, reply)
        if k in self.cache:
            return self.cache[k]
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 3,
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT.format(old=old[: self.max], new=new[: self.max], reply=reply[: self.max]),
                }
            ],
        }
        for attempt in range(3):
            try:
                r = self.http.post("/chat/completions", json=body)
                r.raise_for_status()
                break
            except httpx.HTTPError:
                if attempt == 2:
                    raise
        text = (r.json()["choices"][0]["message"]["content"] or "").strip().lower()
        ok = text.startswith("yes")
        self.cache[k] = ok
        if len(self.cache) % 20 == 0:
            self.save()
        return ok

    def save(self) -> None:
        self.path.write_text(json.dumps(self.cache))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)
