# cost-frontier — spec

How much LLM spend can caching really save on **real** traffic, and what does
each saving cost in wrong answers? Vendor claims for semantic caching are
typically "30–70% of calls". This project measures it on a public log of real
user prompts and reports both axes: money saved and answers that were wrong
because a cached answer to a different question was served.

## Goals (v1)

| # | Capability | Done when |
|---|---|---|
| G1 | A replayable trace: a seeded sample of first-turn English prompts from WildChat-1M (ODC-BY), in their original time order, with token counts | trace file + script that rebuilds it |
| G2 | Strategies replayed over the trace: no cache; exact cache (raw text); normalized exact cache (case/whitespace/punctuation); semantic cache over a sweep of thresholds, with and without ModelMux's lexical guard; TTL variants | one results row per strategy |
| G3 | Cost model: token counts × a configurable price sheet (defaults: two published API price points, stated with date); cache hits cost only the embedding call | prices in config, not code |
| G4 | **Wrong-answer rate** for every strategy: each cache hit is judged "would the cached answer be an acceptable answer to this new prompt?" by an LLM judge, with the judge itself audited on a hand-labelled sample and its agreement reported | judge-vs-human agreement published next to the numbers |
| G5 | The frontier: savings vs wrong-answer rate, one point per strategy, and the recommendation it implies | interactive chart (static site) |

## Non-goals
Provider prompt-caching discounts (they depend on provider-specific prefix rules),
latency (ModelMux measures that), multi-turn context.

## Honesty rules
- Thresholds are chosen on the first half of the trace and evaluated on the second.
- Every number comes from a committed script and results file.
- The hand-labelled audit states who labelled it and how many items.
