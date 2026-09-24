"""Replay the WildChat trace through every cache strategy and write
results/frontier.json.

Protocol: the trace is split in time order. The first half ("calibration")
is used to pick the semantic threshold; every reported number comes from the
second half ("evaluation"). Each strategy's wrong-answer rate is estimated by
judging a random sample of its hits (up to --judge-sample) with an LLM judge;
the judge itself is audited against hand labels (results/audit.json).

    uv run python scripts/run.py
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import tiktoken

from costfrontier.cost import PRICE_SHEETS, spend
from costfrontier.judge import AcceptJudge, wilson
from costfrontier.replay import Hit, Request, exact_hits, semantic_hits

ap = argparse.ArgumentParser()
ap.add_argument("--trace", default="data/trace.jsonl")
ap.add_argument("--embed-model", default="mxbai-embed-large")
ap.add_argument("--judge-model", default="llama3.1:8b")
ap.add_argument("--base-url", default="http://localhost:11434/v1")
ap.add_argument("--judge-sample", type=int, default=150)
ap.add_argument("--max-wrong", type=float, default=0.02, help="wrong-answer budget for choosing the threshold")
a = ap.parse_args()

enc = tiktoken.get_encoding("cl100k_base")
rows = [json.loads(line) for line in Path(a.trace).read_text().splitlines()]
rows = [r for r in rows if r["prompt"].strip()]  # empty prompts aren't requests
rows.sort(key=lambda r: (r["ts"], r["row"]))
reqs = [
    Request(
        ts=datetime.fromisoformat(r["ts"].replace("Z", "+00:00")),
        user=r["user"],
        prompt=r["prompt"],
        response=r["response"],
        in_tokens=len(enc.encode(r["prompt"], disallowed_special=())),
        out_tokens=len(enc.encode(r["response"], disallowed_special=())),
    )
    for r in rows
]
half = len(reqs) // 2
calib, evalset = reqs[:half], reqs[half:]
print(f"{len(reqs)} requests: {len(calib)} calibration, {len(evalset)} evaluation", flush=True)

# Embeddings (cached on disk; the prompt is truncated to the embedder's window).
emb_path = Path(f"data/emb-{a.embed_model}.npy")
if emb_path.exists():
    emb = np.load(emb_path)
else:
    http = httpx.Client(base_url=a.base_url, timeout=600)
    vecs = []
    for i in range(0, len(reqs), 32):
        batch = [r.prompt[:2000] for r in reqs[i : i + 32]]
        for attempt in range(5):
            try:
                resp = http.post("/embeddings", json={"model": a.embed_model, "input": batch})
                resp.raise_for_status()
                break
            except httpx.HTTPError:
                time.sleep(5 * (attempt + 1))
        vecs.extend(d["embedding"] for d in sorted(resp.json()["data"], key=lambda d: d["index"]))
        if i % 1024 == 0:
            print(f"embedded {i}/{len(reqs)}", flush=True)
    emb = np.asarray(vecs, dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    np.save(emb_path, emb)
emb_c, emb_e = emb[:half], emb[half:]

judge = AcceptJudge(a.judge_model, a.base_url, Path("data/judge-cache.json"))
rng = random.Random(20260924)


def assess(rs: list[Request], hits: list[Hit]) -> dict[str, object]:
    sample = rng.sample(hits, min(a.judge_sample, len(hits)))
    wrong = sum(not judge.acceptable(rs[h.j].prompt, rs[h.i].prompt, rs[h.j].response) for h in sample)
    p, lo, hi = wilson(wrong, len(sample))
    return {
        "hits": len(hits),
        "hit_rate": round(len(hits) / len(rs), 5),
        "judged": len(sample),
        "wrong_rate": round(p, 4),
        "wrong_rate_ci95": [round(lo, 4), round(hi, 4)],
        "expected_wrong_answers_per_10k_requests": round(10_000 * len(hits) / len(rs) * p, 1),
    }


def strategies(rs: list[Request], e: np.ndarray, thresholds: list[float]) -> dict[str, list[Hit]]:
    out: dict[str, list[Hit]] = {}
    for scope in ("global", "user"):
        out[f"exact/{scope}"] = exact_hits(rs, "raw", scope, None)
        out[f"normalized/{scope}"] = exact_hits(rs, "normalized", scope, None)
        for t in thresholds:
            for g in (False, True):
                out[f"semantic{'+guard' if g else ''}@{t}/{scope}"] = semantic_hits(rs, e, t, scope, None, g)
    return out


THRESHOLDS = [0.90, 0.93, 0.95, 0.97, 0.99]
t0 = time.time()
# 1) Calibration half: judge semantic strategies to choose each family's threshold.
cal = strategies(calib, emb_c, THRESHOLDS)
chosen: dict[str, float | None] = {}
cal_report = {}
for fam in ("semantic", "semantic+guard"):
    for scope in ("global", "user"):
        best = None
        for t in sorted(THRESHOLDS):
            r = assess(calib, cal[f"{fam}@{t}/{scope}"])
            cal_report[f"{fam}@{t}/{scope}"] = r
            # the lowest threshold whose judged wrong-rate is within budget
            if r["judged"] and r["wrong_rate"] <= a.max_wrong and best is None:
                best = t
        chosen[f"{fam}/{scope}"] = best
        print(f"calibration {fam}/{scope}: chosen threshold {best}", flush=True)
judge.save()

# 2) Evaluation half: exact strategies + semantic at the chosen thresholds (plus
# the whole sweep, for the frontier chart).
ev = strategies(evalset, emb_e, THRESHOLDS)
results = {}
for name, hits in ev.items():
    semantic = name.startswith("semantic")
    r = assess(evalset, hits)
    r["spend"] = {k: spend(evalset, hits, p, semantic) for k, p in PRICE_SHEETS.items()}
    fam, rest = name.split("@") if "@" in name else (name.split("/")[0], None)
    if rest:
        t, scope = rest.split("/")
        r["chosen_on_calibration"] = chosen.get(f"{fam}/{scope}") == float(t)
    results[name] = r
    print(name, r["hit_rate"], r["wrong_rate"], flush=True)
judge.save()

users = len({r.user for r in evalset})
Path("results").mkdir(exist_ok=True)
Path("results/frontier.json").write_text(
    json.dumps(
        {
            "what": "LLM spend saved by response caching on real traffic, and the wrong answers it costs",
            "trace": f"WildChat-1M (allenai, ODC-BY) rows 0-29,999 in time order, first-turn English non-toxic "
            f"non-redacted conversations: {len(reqs)} requests; calibration = first {len(calib)}, "
            f"evaluation = last "
            f"{len(evalset)} ({users} distinct hashed IPs)",
            "method": "exact = identical prompt text; normalized = case/whitespace/punctuation-insensitive; "
            f"semantic = "
            f"cosine similarity of {a.embed_model} embeddings to the nearest earlier cached prompt; "
            f"+guard = ModelMux's "
            f"lexical guard (same leading word and entity/number tokens); "
            f"scope global = one cache for all users, user "
            f"= per hashed IP. A hit serves the earlier prompt's actual WildChat reply. Wrong-answer rate: "
            f"{a.judge_model} judges a random sample of up to {a.judge_sample} hits per strategy (Wilson 95% CI). "
            f"Semantic thresholds chosen on the calibration half as the lowest with judged wrong-rate "
            f"<= {a.max_wrong}. "
            f"Token counts with tiktoken cl100k_base. No TTL.",
            "prices": {k: vars(v) for k, v in PRICE_SHEETS.items()},
            "chosen_thresholds": chosen,
            "calibration": cal_report,
            "evaluation": results,
            "seconds": round(time.time() - t0),
        },
        indent=2,
    )
    + "\n"
)
print("done")
