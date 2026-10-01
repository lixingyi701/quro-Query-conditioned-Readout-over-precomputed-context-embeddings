# P0 pre-registration (written 2026-10-01T14:48:26+08:00, before any full run; git 70b55c1)
- Data: HotpotQA dev.jsonl, all 2000 rows, K=10 retrieved docs, docs clipped at 128 tokens (= compressor), greedy, max_new_tokens=32.
- Decoders: released PISCO adapter; P1 (/data02/quro/runs/oscale_P1/checkpoint_last.pt); reference: bare Mistral (RG/AG only).
- Primary metric: substring (released PISCO is verbose, EM is degenerate for it). Secondary: F1, EM, answer NLL.
- Primary contrast: within each decoder, RG - D0, paired bootstrap 95% CI over rows.
- Minimum practical gap: 2.0 substring points. "Stable gap" = delta >= 0.02 AND CI lower bound > 0, under P1.
- If no stable gap under P1: stop the "repair compression damage" line (plan §2). Otherwise -> P1 (K=2 gold / per-hop).
- Sanity: P1 D0 substring should reproduce Stage C (0.599).
# P1 part 1 pre-registration (written 2026-10-01T15:48:06+08:00, before full runs; git 70b55c1 + uncommitted scripts/eval_gold_mixed.py)
- Same dev 2000 rows, harness and decoders as P0 (P1 primary, released PISCO secondary). K=2 gold paragraphs, 8 conditions {MM,RR,RM,MR}x{orig,swap}.
- Primary metric substring; secondary EM, NLL. Paired bootstrap 95% CI.
- Reading rules (plan §3 table), applied to bridge rows under P1:
  * K=2 gap = RR-MM. If K=2 gap >= 2 pts (CI lo>0) -> the gap survives without distractors: reading/composition/compression, not just interference.
    If K=2 gap CI includes 0 while K=10 gap (+9.3) holds -> interference/selection is the primary candidate.
  * Role localisation (bridge rows with exactly one answer-containing gold doc): compare raw(bridge only)-MM and raw(answer only)-MM.
    A doc whose raw-ification recovers >= 50% of the K=2 gap is the localisation candidate; both partial -> composition candidate.
  * Order: MM swap-orig and RR swap-orig reported as checks; |delta| >= 2 pts flags order sensitivity.
# Scoped P3 pre-registration (written 2026-10-01T15:59:02+08:00, before full runs; git 70b55c1 + uncommitted scripts/patch_causal_order.py)
Question: in D2, does the question->memory state change (P2: ~1e-4 cos / 1.3% relL2) contribute to answering, in a fixed decoder?
- Decoders: released PISCO (primary, = P2 checkpoint); P1 checkpoint (secondary). Neither was trained on D2.
- Receivers: 200 exact-position pairs (dco.select_pairs, seed 42), both directions -> 400. Dev half = pairs 0-99, holdout = 100-199.
- Primary tests (no layer selection, all 400 receivers): block_QM vs none; xq_all vs none.
- Metrics: answer NLL (nats/token, teacher-forced gold), substring (primary QA; released PISCO EM is degenerate), partner_substring (wrong-answer transfer).
- Validity (else RERUN_INVALID): id_all bit-exact (same_pred = 1.0, |dNLL| < 1e-4); xdoc_all positive control dNLL CI lo > 0; block leak check passed in-run.
- Verdict per primary test:
  * ANSWER_FUNCTIONAL: dNLL >= +0.02 with CI lo > 0 AND d-substring CI hi < 0.
  * NLL_ONLY: dNLL criterion met, substring not.
  * HARMFUL_ZERO_SHOT: dNLL CI hi < 0 (removing/replacing Q->M helps) -- the untrained path hurts.
  * NO_DETECTABLE_FUNCTION: otherwise (bounded by the CI; not a claim that a trained D2 cannot use it).
- Magnitude context: report each primary effect as a fraction of xdoc_all's dNLL.
- Secondary: xq@{4,12,20}: layer with the largest dNLL on dev half, reported on holdout only.
- Training decision follows plan §6: only ANSWER_FUNCTIONAL (and positive direction) licenses T-D2 matched adaptation.
