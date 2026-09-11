# RRK Paper Notes

> **Paper:** *Efficient Listwise Reranking with Compressed Document Representations*  
> **Authors:** Hervé Déjean, Stéphane Clinchant (NAVER LABS Europe)  
> **arXiv:** 2604.26483 (2026)  
> **Purpose of this note:** This is not only a summary of RRK. It records the parts of RRK that are directly relevant to our QuRO direction: **query-independent precomputed context representations + query-conditioned readout/conversion + frozen or lightly adapted downstream LLM**.

---

## 1. Why RRK matters to QuRO

RRK starts from a very similar systems observation to ours: repeatedly feeding full documents into a large Transformer is wasteful when the document-side representation can be computed once and reused.

Its basic pipeline is

\[
D \xrightarrow{f_{\theta_c}} C_D=(c_1,\ldots,c_l)
\xrightarrow[Q]{g_{\theta_r}} s(Q,D),
\]

where the document is first converted into a short sequence of **soft memory tokens**, and the expensive query-time model operates on those compressed tokens rather than on the full document text.

This gives RRK two important properties:

1. **document-side computation can be moved offline / cached**;
2. **query-time cost depends on the compressed representation length rather than the original document length**.

This is extremely close to the system motivation of QuRO. However, the downstream task differs:

- RRK: compressed representation → query-conditioned **reranking score**;
- QuRO: compressed/precomputed representation → query-conditioned **generator-compatible soft tokens** → generation.

So RRK should be treated as an important methodological precursor and experimental reference, but not as the final architecture we should copy.

---

## 2. RRK method

### 2.1 Query-independent document compression

RRK uses a PISCO-style compressor to map a document to a fixed number of soft tokens:

\[
f_{\theta_c}: d_i \rightarrow c_i,
\qquad
c_i\in\mathbb{R}^{l\times d}.
\]

The main experiments use **8 memory tokens per document**.

The key systems property is that compression is **query-independent**. Once training is complete, document representations can therefore be precomputed and stored offline.

This is exactly the property we want from the first stage of QuRO:

\[
D\rightarrow Z_D,
\]

where \(Z_D\) should be reusable across many future queries.

The important distinction is that **query-independent does not mean task-independent, nor does it mean that the compressor should necessarily remain frozen during training**. RRK's experiments make this distinction especially important; see Section 5 below.

---

### 2.2 Listwise reranking over compressed representations

Instead of concatenating the full text of all candidate documents, RRK constructs an LLM input approximately as

\[
X=(q;c_1;[SEP];c_2;[SEP];\ldots;c_k;[SEP];q).
\]

The decoder produces hidden states

\[
H=\mathrm{Decoder}_{\theta_r}(X).
\]

RRK uses the final query token hidden state as a query representation and the separator hidden states as document representations. Relevance is then computed by cosine similarity:

\[
s_i=\cos(\mathbf q,\mathbf h_i).
\]

Thus the large decoder is effectively acting as a **query-conditioned consumer/readout of compressed document representations**.

This observation is more important to QuRO than RRK's exact scoring head. RRK demonstrates that a powerful Transformer can operate directly over multi-token soft document representations and extract task-relevant information conditioned on a query.

---

### 2.3 Joint optimization

RRK does not simply take an arbitrary frozen compressed representation and assume that it is sufficient. Its ranking objective can backpropagate through both the reranker and compressor:

\[
s_i=g_{\theta_r}\bigl(q,f_{\theta_c}(d_i)\bigr).
\]

Therefore the representation learned by the compressor can adapt toward the information needed by the downstream ranking task.

This becomes one of the most important lessons for our own experimental design: **architectural separation between offline compression and online readout does not imply statistical or optimization independence between them.**

---

## 3. Efficiency argument

For conventional listwise reranking over full documents, the Transformer input length scales roughly as

\[
|q|+k|d|,
\]

leading to attention complexity

\[
O\left((|q|+k|d|)^2\right).
\]

RRK instead processes approximately

\[
2|q|+k(l+1),
\]

so query-time attention becomes

\[
O\left((2|q|+k(l+1))^2\right),
\]

with \(l=8\) in the main setting.

This explains an important empirical result: **a larger 8B model can be faster than much smaller rerankers when its input representation is sufficiently short.**

The broader lesson for QuRO is that model parameter count alone is not the correct efficiency variable. We should explicitly measure:

- original context length \(L\);
- cached latent length \(N\);
- query-conditioned output length \(M\);
- query-time FLOPs / latency;
- cache/storage cost;
- downstream generation quality.

The real trade-off is therefore closer to

\[
\text{quality}=F(L/N,\,N,\,M,\,\text{readout capacity},\,\text{generator}).
\]

---

## 4. RRK experimental results worth remembering

### 4.1 Main effectiveness/efficiency result

RRK uses a Qwen2.5-8B-Instruct backbone and compressed multi-token document representations. On BeIR, the reported RRK configuration with 512-token documents reaches approximately **58.4 average nDCG@10**, while operating at the paper's 1× latency reference (about **0.06 s/query** in the reported setup).

A full-text Qwen2.5-8B configuration reaches somewhat higher effectiveness (about **59.7**) but at roughly **20×** the processing time in the corresponding comparison.

The key message is not that compression is lossless. It is that RRK moves to a substantially better **effectiveness–latency Pareto point**.

This is an important framing for QuRO as well. We do not need to prove that a compressed latent preserves every bit of the original document. We need to show that, under a meaningful compute/cache budget, it preserves enough information for strong query-conditioned downstream behavior.

---

### 4.2 Long-document result

The advantage becomes larger for long documents because the compressed representation length remains fixed while full-text inference grows with document length.

RRK reports strong results on MS MARCO Document with inputs up to 2048 tokens while retaining an 8-token compressed representation. The reported nDCG@10 values include approximately:

- DL19: **72.0**;
- DL20: **68.6**.

This corresponds to compression factors that can reach roughly **256×** in token-count terms.

For QuRO this suggests that the strongest application regime may not be short contexts. The method becomes more compelling when:

\[
L\gg N,M,
\]

and the same context is reused across multiple queries or agent steps.

---

## 5. The most important RRK ablation: is query-independent compression reliable?

RRK's compressor ablation is especially important for our project.

Reported BeIR average results are approximately:

| Compressor setting | nDCG@10 |
|---|---:|
| Frozen PISCO compressor | 55.5 |
| Compressor trained from scratch jointly with reranker | 57.7 |
| Fine-tuned PISCO compressor | **58.4** |

This result should prevent us from making an overly strong claim such as:

> "Once a query-independent compressed representation is sufficiently strong, compression and readout can be studied independently."

RRK gives evidence against assuming that in advance.

A better interpretation is:

> **Query-independent at inference time does not imply independently optimized at training time.**

The document representation can remain query-independent and cacheable after training while still being adapted jointly with the downstream query-conditioned module during training.

Therefore, for QuRO, the right question is not simply:

> Is the compressed representation good enough?

but rather:

> **How does compression quality interact with the capacity and training of the query-conditioned readout?**

This motivates explicitly studying **Compression × Readout coupling**.

---

## 6. Implication for QuRO: architectural separation, optimization coupling

Our general abstraction should be

\[
D\xrightarrow{C_\phi}Z_D,
\]

followed at query time by

\[
P_{Q,D}=R_\theta(Q,Z_D),
\]

and finally

\[
Y=G(P_{Q,D},Q),
\]

where:

- \(C_\phi\): query-independent context encoder/compressor;
- \(Z_D\): precomputed reusable context representation;
- \(R_\theta\): query-conditioned readout / representation converter;
- \(P_{Q,D}\): a short sequence of query-specific soft tokens;
- \(G\): downstream generator, ideally frozen or minimally adapted.

The stages are **architecturally separable** because \(Z_D\) can be cached before the query arrives. But RRK suggests they may remain **statistically and optimization-wise coupled**:

\[
\boxed{
\text{offline/online separation}\;\neq\;\text{training independence}
}
\]

This is a useful conceptual distinction for the paper.

---

## 7. Query-conditioned readout should NOT be defined as Perceiver IO

A central design decision for QuRO is that the research contribution should not be phrased as simply "use Perceiver IO after compression."

The more general object is

\[
\boxed{P_{Q,D}=R_\theta(Q,Z_D)}.
\]

Perceiver IO is one possible implementation of \(R_\theta\), but the hypothesis is broader:

> Given a reusable query-independent representation \(Z_D\), a query-conditioned readout should selectively extract the information relevant to the current query and convert it into a small set of generator-consumable soft tokens.

Candidate implementations include:

1. attention pooling;
2. gated pooling/readout;
3. standard cross-attention Transformer;
4. Perceiver IO-style latent querying;
5. LLM-based readout analogous to RRK;
6. tree/hierarchical cross-attention for large \(N\).

This makes the paper about **Query-conditioned Readout over Precomputed Context Embeddings**, rather than about one specific architecture.

---

## 8. Perceiver IO can simultaneously perform readout and projection

An important simplification relative to our earlier pipeline is that a separate projector is not necessarily required.

The naive formulation would be

\[
Z_D\xrightarrow{R(Q,\cdot)}H_{Q,D}
\xrightarrow{\text{projector}}P_{Q,D}
\xrightarrow{G}Y.
\]

However, a Perceiver IO-style decoder uses an output-query array to determine the structure of the output. We can therefore design the query-conditioned readout so that its output already has the generator's required dimensionality:

\[
P_{Q,D}\in\mathbb{R}^{M\times d_g},
\]

where

- \(M\) = number of soft context tokens presented to the generator;
- \(d_g\) = generator embedding dimension.

The resulting pipeline becomes simply

\[
Q,Z_D
\xrightarrow{R_\theta}
P_{Q,D}\in\mathbb{R}^{M\times d_g}
\xrightarrow{\texttt{inputs\_embeds}}
G.
\]

Thus the readout module performs two functions simultaneously:

1. **query-conditioned information extraction**;
2. **representation-space conversion into the generator embedding space**.

This is conceptually cleaner than treating projection as an unrelated final adapter.

Importantly, this property should be presented as a capability of the general readout interface, not as a reason the whole method must use Perceiver IO. For fair ablations, alternative readouts should also be configured to produce the same output shape \(M\times d_g\).

---

## 9. What RRK does NOT establish

RRK is strong evidence that multi-token soft compressed representations can support an expensive downstream Transformer efficiently, but it does not establish several things that QuRO needs to investigate.

### 9.1 It does not prove that generic frozen compression is universally sufficient

The 55.5 → 58.4 compressor ablation actually suggests the opposite: task adaptation matters.

### 9.2 It does not isolate query-conditioned readout as a general generation problem

RRK ultimately predicts ranking scores. QuRO asks for a much richer object:

\[
(Q,Z_D)\rightarrow P_{Q,D}\rightarrow \text{free-form generation}.
\]

Generation may require preserving more fine-grained evidence than ranking.

### 9.3 It does not determine the optimal allocation between compressed-latent budget and readout budget

RRK largely fixes the compressed representation length. For QuRO, two independent sequence budgets matter:

\[
N=|Z_D|,\qquad M=|P_{Q,D}|.
\]

Understanding the \(N\times M\) trade-off can become one of our central experiments.

---

## 10. Experiments RRK motivates for QuRO

### RQ1 — At what compression budget is the precomputed representation sufficient?

Sweep

\[
N\in\{128,64,32,16,8\}
\]

or equivalently compression ratios such as

\[
4\times,8\times,16\times,32\times,64\times.
\]

Use the same strong readout and compare downstream generation quality.

The goal is **not** to claim compression is lossless or independent of readout. The goal is to identify the feasible region in which \(Z_D\) retains enough task-relevant information.

---

### RQ2 — Compression × Readout coupling

Use compressor variants such as:

- **C1:** generic / frozen compressor;
- **C2:** task-adapted compressor;
- **C3:** compressor jointly optimized with readout.

Cross them with readouts such as:

- **R1:** simple pooling / attention pooling;
- **R2:** standard cross-attention;
- **R3:** Perceiver IO-style readout;
- optionally **R4:** LLM-based readout.

This produces a matrix

| | R1 weak | R2 cross-attn | R3 Perceiver |
|---|---:|---:|---:|
| C1 Frozen |  |  |  |
| C2 Adapted |  |  |  |
| C3 Joint |  |  |  |

The key quantity is not only absolute performance but the interaction:

\[
\Delta_C(R_{weak})\quad\text{vs.}\quad\Delta_C(R_{strong}).
\]

If stronger readout benefits disproportionately from better compression, that is evidence of **compression–readout synergy**, not a failure of the framework.

---

### RQ3 — Compression budget × query-conditioned output budget

Sweep

\[
N\in\{64,32,16,8\},
\qquad
M\in\{4,8,16,32\}.
\]

Measure

\[
\mathrm{Performance}(N,M).
\]

A heatmap can diagnose two distinct bottlenecks:

- performance mainly controlled by \(N\) → **compression bottleneck**;
- performance mainly controlled by \(M\) → **readout/output bottleneck**;
- strong interaction → the two budgets must be co-designed.

This experiment is potentially more informative than a single "compression ratio" curve.

---

### RQ4 — Is the gain really from query-conditioned readout?

With the same compressor and exactly the same output shape \(M\times d_g\), compare:

- query-independent mean pooling;
- learned query-independent soft pooling;
- query-conditioned attention pooling;
- cross-attention;
- Perceiver IO;
- optionally LLM readout.

This directly tests whether query conditioning provides value beyond simply learning a better static projection of \(Z_D\).

---

### RQ5 — Query-swap / query sensitivity

Fix the same cached document representation \(Z_D\), but use two different queries:

\[
P_1=R(Q_1,Z_D),
\qquad
P_2=R(Q_2,Z_D).
\]

We should demonstrate that

\[
P_1\neq P_2
\]

in a task-meaningful way: the generator should recover different relevant evidence from the same cached context depending on the query.

Useful analysis includes:

- answer accuracy under query swaps;
- attention/attribution from output queries to cached latents;
- similarity/diversity of generated soft tokens;
- examples where different questions target different parts of a long context.

This would make the phrase **query-conditioned readout** empirically concrete.

---

## 11. Recommended QuRO framing after reading RRK

### Avoid this claim

> We first build a sufficiently good query-independent compression, then independently study the readout module.

This is too strong because RRK shows that downstream task adaptation of the compressor can matter substantially.

### Prefer this framing

> We architecturally separate reusable context encoding from query-time computation, while explicitly studying the optimization coupling between context compression and query-conditioned readout.

A compact formulation is

\[
\boxed{
D\xrightarrow{C_\phi}Z_D
\xrightarrow{R_\theta(Q,\cdot)}P_{Q,D}
\xrightarrow{G}Y
}
\]

with the following hypotheses:

1. **Reusable substrate:** a query-independent \(Z_D\) can preserve a reusable information substrate and be cached across queries;
2. **Selective readout:** \(R_\theta\) determines which parts of that substrate are useful for the current query;
3. **Direct conversion:** \(R_\theta\) can directly output \(M\times d_g\) generator-space embeddings, avoiding a separate projector;
4. **Coupled learning:** \(C_\phi\) and \(R_\theta\) may need joint/task-aware optimization even though only \(R_\theta\) is query-dependent at inference time.

This is stronger and more defensible than claiming compression and readout are independent.

---

## 12. RRK → QuRO: direct comparison

| Dimension | RRK | QuRO direction |
|---|---|---|
| Offline representation | PISCO soft document tokens | precomputed context embeddings / compressed latent \(Z_D\) |
| Query-independent cache | Yes | Yes, core systems requirement |
| Query-time module | 8B listwise reranker | lightweight/efficient query-conditioned readout |
| Output | relevance scores | \(M\) generator-compatible soft tokens |
| Downstream task | reranking | generation / QA / agent context use |
| Compressor adaptation | jointly/task adapted is beneficial | explicitly study frozen vs adapted vs joint |
| Query conditioning | performed by reranker over compressed docs | explicit object of study \(R_\theta(Q,Z_D)\) |
| Output-space conversion | not central | central: directly output \(d_g\)-dimensional soft tokens |
| Main efficiency gain | shorter reranker input | cached document compute + short query-time readout + short generator prefix |

---

## 13. The main lesson from RRK

The most useful lesson from RRK is **not merely "soft compression works."**

It is:

\[
\boxed{
\text{A reusable query-independent representation can dramatically reduce query-time cost,}
}
\]

while simultaneously

\[
\boxed{
\text{the quality of that representation and the downstream query-conditioned consumer should be co-designed.}
}
\]

For QuRO, this naturally leads to the research question:

> **Given a reusable precomputed representation of a long context, how should an LLM efficiently extract query-relevant information and directly convert it into a small number of generator-compatible soft embeddings?**

RRK provides strong motivation for the first half of this question. Our novelty should primarily live in the second half: **general query-conditioned readout for generation, its interaction with precomputed compression, and the allocation of compression/readout budgets.**

---

## 14. Immediate implementation / experiment checklist

- [ ] Implement/cache query-independent \(Z_D\).
- [ ] Make readout API uniformly output `[B, M, d_g]`.
- [ ] Start with cross-attention and Perceiver IO as two strong readout implementations.
- [ ] Do **not** add a separate projector when the readout can directly emit \(d_g\).
- [ ] Add a simple attention-pooling baseline.
- [ ] Compare frozen vs task-adapted vs jointly trained compressor.
- [ ] Run \(N\times M\) budget grid.
- [ ] Add raw/full-context upper bound.
- [ ] Measure quality, latency, FLOPs, memory/cache size and query-time tokens together.
- [ ] Add query-swap analysis to verify query-dependent extraction.
- [ ] Later evaluate LLM-based and tree/hierarchical readouts if simple cross-attention becomes the scaling bottleneck.

---

## 15. One-sentence takeaway for future discussions

**RRK shows that query-independent soft document representations can make a large query-time Transformer highly efficient, but its compressor ablation also shows that cacheability does not imply training independence; QuRO should therefore study a general query-conditioned readout \(R(Q,Z_D)\)—not just Perceiver IO—that jointly selects query-relevant information and converts it directly into generator-space soft tokens, while explicitly measuring its coupling with the precomputed compressor.**
