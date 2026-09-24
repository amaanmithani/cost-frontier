import json

import httpx

from costfrontier.judge import AcceptJudge


def test_judge_caches_and_parses(tmp_path):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Yes." if len(calls) == 1 else "no"}}]})

    j = AcceptJudge("m", "http://x", tmp_path / "c.json")
    j.http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    assert j.acceptable("old", "new", "reply") is True
    assert j.acceptable("old", "new", "reply") is True  # cached
    assert j.acceptable("old", "new2", "reply") is False
    assert len(calls) == 2
    j.save()
    assert len(json.loads((tmp_path / "c.json").read_text())) == 2
    j2 = AcceptJudge("m", "http://x", tmp_path / "c.json")
    assert j2.acceptable("old", "new", "reply") is True  # loaded from disk, no HTTP
