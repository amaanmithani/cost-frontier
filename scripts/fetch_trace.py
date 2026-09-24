"""Fetch a contiguous, time-ordered window of WildChat-1M (ODC-BY) through the
Hugging Face dataset-server API and keep first-turn English, non-toxic,
non-redacted conversations: the prompt, the assistant's actual reply, the
timestamp and the hashed user IP. Writes data/trace.jsonl (gitignored);
this script plus its offsets make the trace reproducible."""

import json
import sys
import time
from pathlib import Path

import httpx

START = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 30_000
API = "https://datasets-server.huggingface.co/rows"
out = Path("data/trace.jsonl")
out.parent.mkdir(exist_ok=True)
# Resume: skip rows already fetched (the file is append-only, in order).
done_rows = [json.loads(line)["row"] for line in out.read_text().splitlines()] if out.exists() else []
resume = max(done_rows) + 1 if done_rows else START
kept = len(done_rows)
with httpx.Client(timeout=60) as c, out.open("a") as f:
    for off in range(START, START + ROWS, 100):
        if off + 100 <= resume:
            continue
        time.sleep(0.3)  # be polite to the public API
        for attempt in range(10):
            try:
                r = c.get(
                    API,
                    params={
                        "dataset": "allenai/WildChat-1M",
                        "config": "default",
                        "split": "train",
                        "offset": off,
                        "length": 100,
                    },
                )
                r.raise_for_status()
                break
            except httpx.HTTPError:
                time.sleep(min(120, 5 * 2**attempt))
        else:
            raise SystemExit(f"failed at offset {off}")
        for item in r.json()["rows"]:
            if item["row_idx"] < resume:
                continue
            row = item["row"]
            conv = row["conversation"]
            if (
                row["language"] != "English"
                or row["toxic"]
                or row["redacted"]
                or len(conv) < 2
                or conv[0]["role"] != "user"
                or conv[1]["role"] != "assistant"
            ):
                continue
            f.write(
                json.dumps(
                    {
                        "row": item["row_idx"],
                        "ts": row["timestamp"],
                        "user": row["hashed_ip"],
                        "model": row["model"],
                        "prompt": conv[0]["content"],
                        "response": conv[1]["content"],
                    }
                )
                + "\n"
            )
            kept += 1
        if off % 2000 == 0:
            print(f"offset {off}: kept {kept}", flush=True)
print(f"kept {kept} of {ROWS} rows from offset {START}")
