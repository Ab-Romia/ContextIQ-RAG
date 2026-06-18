# Hybrid RAG retrieval on a free CPU, and the evaluation where the fancy pipeline did not win

> Also published at https://romia.dev/blog/contextiq-hybrid-rag-retrieval
> Portable Markdown for reposting. The diagrams load from romia.dev.

I keep running into RAG tutorials that end at the same place. Split a document into chunks, embed them, retrieve the handful that look most similar to the question, paste them into a prompt. The demo works on the slide. Then you point it at a real corpus where several documents look almost identical, ask a question that hinges on one specific term, and the whole thing falls apart. The retrieval was never the hard part of the slide; it is the entire part the slide skipped.

So I built [ContextIQ](https://github.com/Ab-Romia/ContextIQ-RAG), a retrieval pipeline that picks up exactly where those tutorials stop. It searches two different ways and merges the results, double-checks the top candidates with a more careful model, writes answers that cite their sources, and, the part I care about most, grades itself against a simple baseline so I can say which numbers I actually trust. The whole thing runs on a free CPU, no GPU and no paid embedding API. This post walks through every stage in plain language, then shows the evaluation, including the place where the full pipeline did not win.

## Retrieval is where RAG lives or dies

Retrieval-augmented generation is a simple idea. Instead of hoping a language model memorized a fact, you fetch the relevant text at question time and hand it to the model to read. The quality of the answer is bounded by the quality of what you fetched. If the right passage never makes it into the prompt, no amount of clever prompting saves you. The model will either guess or, worse, answer confidently from its training data and sound exactly as fluent as when it is right.

To make this concrete, and to make the evaluation honest, I built a corpus designed to be hard in a specific way: seven fictional company handbooks that share the same section structure. They all have an on-call policy, an expense policy, a parental leave policy. The wording differs, the policies differ, but the shape is identical. One handbook is the target; the other six are distractors. Ask something like, what is the single-transaction limit for expenses without prior approval, and a naive retriever has to separate the one passage that answers it from six near-duplicates that look just as relevant. That is the trap, and it is the trap most real document sets set for you too.

The constraint I gave myself was a free CPU environment. No GPU, no PyTorch in the image, no paid API for embeddings. Every model choice below is shaped by that: small models that run through ONNX Runtime, a lexical search library with no C extension to compile. The constraint is real, and I will name where it costs me.

Here is the whole pipeline in one picture. Refer back to it as each stage comes up.

![ContextIQ retrieval pipeline](https://romia.dev/blog/contextiq/hybrid-retrieval-pipeline.svg)

*Chunk, then embed and index two ways, then fuse, then rerank, then generate.*

## Chunking and contextual headers

I split each document on its Markdown headings first, then split each section into token-bounded pieces of 400 tokens with 60 tokens of overlap, using a tiktoken-based splitter from `langchain-text-splitters`. A chunk has to be small enough to be specific and large enough to stand on its own. Splitting on structure before size keeps related sentences together instead of cutting a policy in half. Every chunk also carries its heading path, so a citation can later point at a real section, not an anonymous byte offset.

A retrieved chunk often reads like an orphan. A sentence like, it raised the limit to 40, means nothing without knowing which document and section it came from. So before embedding, I prepend a small deterministic header built from the source title, the heading path, and the chunk's position. Only the embedded text carries this header; what gets shown to you stays clean.

```python
header = " > ".join(bit for bit in header_bits if bit)
position = f"part {chunk.ordinal + 1} of {total}"
chunk.augmented_text = f"[{header} | {position}]\n{chunk.text}"
# The reader still sees the original, unprefixed chunk.text.
```

I want to be precise about what this is and is not. A more elaborate version of this idea asks a language model to write a custom sentence of context for every single chunk. That works better, but it costs a model call per chunk. Mine is the cheap version, a fixed template with no model calls, so I do not claim the larger gains the model-generated approach reports.

## Two ways to search, and why you need both

There are two classic ways to find a passage, and they fail in opposite directions.

The first is dense retrieval. An embedding turns text into a list of numbers that captures its meaning, so two passages about the same idea land close together, like points near each other on a map, even when they share no words. You embed every chunk, embed the query the same way, and take the chunks whose vectors sit closest. Its strength is meaning: it can find a passage about supported devices when the question asks about devices per site in different words. Its weakness is that it smooths over exact tokens. A product name, a precise figure like `4,000 Pebbles`, an exact amount: a small embedding model places those near a dozen similar-looking strings, because to it they mean roughly the same thing.

The second is lexical retrieval, the classic being BM25, which scores passages by how many of the query's exact terms they contain, weighted by how rare those terms are. BM25 locks onto the literal token and the literal number that dense search blurs. Its weakness is the mirror image: if the answer passage never uses the query's words, BM25 scores it near zero.

So neither retriever is enough on its own, and they fail on different questions.

![Why hybrid beats either retriever alone](https://romia.dev/blog/contextiq/why-hybrid-beats-either-alone.svg)

*Each retriever fails on a different kind of question, so running both covers what either one misses.*

In ContextIQ the dense side is `bge-small-en-v1.5`, a 384-dimensional model that runs through ONNX Runtime via `fastembed`, about 67 MB and no PyTorch. The lexical side is `bm25s`, which runs BM25 over scipy sparse matrices with no C extension to compile. The dense vectors live in an in-memory Chroma store. Chroma can generate its own embeddings, but I turn that off and feed it the vectors I already computed, so the whole system speaks one model's language. Indexing is additive: re-indexing one source replaces only that source, which fixed the original design's defining bug, an implicit wipe of the entire store on every upload.

## Fusion: combine by rank, not by score

Now the question both searches leave open. Dense search gives me cosine similarities, BM25 gives me its own scores, and these live on completely incompatible scales. Averaging them is meaningless. The trick is to throw away the raw scores and rank everything by position instead. Position is comparable across the two searches even when the raw scores are not.

That is reciprocal rank fusion. Each retriever contributes one over the quantity k plus rank to a passage's combined score, where the constant `k` softens how much the very top spots dominate. Sixty is the standard value. I pull a deep pool, 50 candidates per retriever, before fusing.

```python
for rank, (chunk, score) in enumerate(dense_ranked, start=1):
    cand = candidate_for(chunk)
    cand.rrf_score += 1.0 / (k + rank)

for rank, (chunk, score) in enumerate(sparse_ranked, start=1):
    cand = candidate_for(chunk)
    cand.rrf_score += 1.0 / (k + rank)

return sorted(candidates.values(), key=lambda c: c.rrf_score, reverse=True)
```

A passage that ranks well in either retriever survives into the fused pool. A passage that ranks well in both gets contributions from both and rises to the top. There is nothing to tune beyond `k`.

## Reranking: getting the right passage to the top

Fusion puts the right passage into the pool. It does not reliably put it at position one. The right answer might land at rank twenty, surrounded by near-duplicates from the other six handbooks. This is where a cross-encoder earns its place.

A cross-encoder reads the query and a candidate passage together, in one pass, and scores how relevant the passage actually is. This is far more accurate than comparing two vectors that were embedded separately, because the model attends to both texts at once. Running it over all 99 chunks for every query would be slow; running it over just the roughly 50 the cheap retrievers already shortlisted is fast. That is the whole two-stage design: a fast rough filter, then a slow careful judge. I use `ms-marco-MiniLM-L-6-v2` through `fastembed`, about 80 MB.

```python
def rerank(query: str, candidates: list[Candidate]) -> list[Candidate]:
    if not candidates:
        return []
    scores = list(_model().rerank(query, [c.chunk.text for c in candidates]))
    for candidate, score in zip(candidates, scores):
        candidate.rerank_score = float(score)
    return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
```

![What reranking does](https://romia.dev/blog/contextiq/what-reranking-does.svg)

*Reranking's value is rescuing the correct passage from rank twenty and lifting it to the top, where it will actually make it into the prompt.*

This is the single biggest quality lever in the pipeline, and it is why I keep the pool deep at 50 rather than shallow. Reranking the top three results changes almost nothing; they were already the top three. The value shows up when reranking rescues the correct passage from rank twenty. A shallow pool would have thrown that passage away before the reranker ever saw it.

## Grounded generation: cite, or say I don't know

The reranked top passages, up to five of them, become numbered sources. The model is told to answer only from them, cite every claim by its marker, and, when the sources do not cover the question, say so and stop rather than fall back on training knowledge. Here is the actual system prompt:

```python
SYSTEM_PROMPT = (
    "You are a careful assistant that answers strictly from the provided sources. "
    "Use only the information in the numbered sources below. Cite every claim with its "
    "source marker in square brackets, for example [1] or [2][3]. If the sources do not "
    "contain the answer, say that the provided context does not cover it and stop. Do "
    "not use outside knowledge and do not guess."
)
```

This matters more than it looks. A RAG system that quietly answers from memory when retrieval fails is indistinguishable from a plain chatbot, and that is the harder failure to catch, because the answer still sounds confident. Forcing citations and forcing abstention is how you make retrieval failures visible.

One debugging note worth recording: I first defaulted to a reasoning model, and the streamed answer kept showing up empty. Reasoning models on OpenRouter put their chain of thought in a separate field and do not stream the final answer reliably. Switching to an instruction-tuned model, on a free tier so the demo runs without a credit card, gave me one that streams cleanly and follows the grounding and citation rules.

## The evaluation, and which numbers I trust

Here is the part most write-ups skip. I built four versions of the pipeline, each one an arm, the usual word for a variant in an experiment, and ran all four against the same questions, so I could see what each stage actually buys.

The set is small and I want that stated up front: 21 hand-written questions, 18 answerable and 3 deliberately unanswerable to test whether the system abstains, over 99 chunks from one target handbook plus six distractors. The numbers describe direction on one small set. They are not a benchmark, and I would not let anyone cite them as one. A quick gloss on the metrics:

- `hit@k`: did a right passage land in the top k at all.
- `recall@5`: here the same as hit@5, since each question has exactly one gold passage.
- `MRR`, mean reciprocal rank: how high up the first right passage sits.
- `nDCG@5`: a fuller ranking-quality score that rewards putting the right passage high.

| Arm | hit@3 | hit@5 | recall@5 | MRR | nDCG@5 |
| --- | --- | --- | --- | --- | --- |
| A: TF-IDF baseline | 0.78 | 0.89 | 0.89 | **0.81** | **0.82** |
| B: Dense only | 0.67 | 0.72 | 0.72 | 0.57 | 0.58 |
| C: Hybrid (dense + BM25, RRF) | 0.78 | **0.94** | **0.94** | 0.60 | 0.68 |
| D: Hybrid + rerank | **0.83** | 0.89 | 0.89 | 0.78 | 0.80 |

Two things I will defend, and they are the whole honest headline: the naive dense-only approach is the worst across the board, and the full pipeline gives the best precision at the very top. Everything else depends on the corpus.

No single arm wins every metric. Hybrid has the best hit@5 and recall. The reranked arm has the best hit@3 and a strong MRR. And the plain TF-IDF baseline posts the single highest MRR and nDCG of anything I ran, because my golden questions are keyword-rich: they name specific policies, products, and numbers, which is exactly what lexical matching keys on. That is not a sign the pipeline failed. It is a reminder that lexical search is a real baseline, not a straw man, and that the shape of your questions decides who wins.

The narrow result I do trust, stated narrowly: dense-only retrieval is the weakest arm, because a small embedding model cannot separate near-duplicate policy passages that share a section structure. That is the naive pipeline most tutorials produce. Adding lexical search recovers recall, putting the right passage in the top five 94 percent of the time. And reranking fixes the ordering hybrid leaves rough, lifting hit@3 to its best value and nearly doubling MRR over hybrid alone.

## What it does not do

The limits are as worth stating as the wins.

- The index is not durable. It lives in memory for the life of the process, so a restart clears it. This is a single-session demo by design, not a database, and not built for many people at once.
- The evaluation is one small golden set on one corpus, 21 questions over 99 chunks. Illustrative of direction, not a benchmark.
- The chunk headers are a cheap template, not the model-generated per-chunk context that some published results measure. No such gain is claimed.
- Everything is constrained to a free CPU: no GPU, no PyTorch, no paid embedding API. A larger embedding model would likely close some of dense retrieval's gap.

Retrieval and the full trace run with no API key at all; only the final answer generation needs a key, which stays in the browser and is never stored. If you want to see it work, or pick apart the places it does not, the code is on [GitHub](https://github.com/Ab-Romia/ContextIQ-RAG) and there is a [live demo](https://huggingface.co/spaces/Ab-Romia/Context-Aware-AI) you can paste a document into and watch the trace, retriever by retriever, all the way to a cited answer. Bring a corpus with confusable documents; that is where the difference between a demo and a working system shows up.
