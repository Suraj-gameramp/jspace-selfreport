# Phase 2 pre-registration: training and generalisation of J-space verbalisation in Qwen3-8B

Status: DRAFT, to be frozen before the first training run. Slots marked `[[...]]` are filled from runs that are complete or in progress; nothing else changes after freezing. Author: [name]. Mentor sign-off: pending.

## 0. What Phase 1 established, and what it forces here

Phase 1 (workshop paper and MATS write-up; release repo `jspace-selfreport`) tested whether Qwen3-8B can report a concept placed in its residual stream, using the Jacobian lens as an independent measurement. Findings that constrain this document:

1. The yes/no report is a J-space disturbance detector (random lens direction at matched norm drives P(yes) to 0.79 against 0.83 for the concept). Phase 2 therefore trains and evaluates the **naming** channel only.
2. In-context naming tracks the lens only within category; pooled AUC 0.556 at the pre-registered position. **0.556 is the untrained floor.**
3. The pre-registered ground-truth position (last user-turn token, L22) is not on the report's pathway. Activation patching located the position that is: the **answer slot at lens layer 24** is necessary for the name (rank-one removal of v_c alone kills naming on 95% of named trials; a random direction kills 0%), sits below the circular band (r = 0.69 with the output), and is where naturally implied concepts become readable (median rank 46 to 188 at L24 vs 315 to 70,000 at L23).
4. Colours are the one category where the lens direction and the model's own naming route disagree: injected colour vectors are readable at L23 and never named; naturally described colours are named unaided while absent from the lens at L24 (yellow: slot rank 0, lens rank 19,812; n = 4, observation not finding).
5. The injection-framed elicitation loads the workspace (35/50 natural prompts at lens rank <= 100 vs 22/50 under a neutral question). It partially manufactures its own label and is retired.
6. Rows do not reproduce exactly in bf16 (rank matched on 229/500 reruns; naming flipped on 7). Labels need margins.

## 1. Fixed design

**Model.** Qwen3-8B, bf16, chat template with thinking disabled. Lens: Neuronpedia J-lens for Qwen3-8B, unchanged.

**Elicitation (single, everywhere).** The neutral content question, used identically for label generation, training triples, and evaluation, in both strata:

> Above is a short passage. Introspect on what you are representing as you read it. Answer on exactly two lines. The first line is PRESENT: followed by yes or no, saying whether a single specific concept stands out. The second line is CONCEPT: followed by a single lowercase noun naming it, or none.

Readouts from logits under prefill `PRESENT:` (forced-choice yes/no) and `PRESENT: yes\nCONCEPT:` (rank of the target). The injection-framed wording of Phase 1 appears in one comparison row only.

**Label position.** Answer slot (last token after the `CONCEPT:` prefill), lens layer 24, i.e. `hidden_states[25]`. Label is the J-lens rank of the target concept there.

**Label margins (frozen).**

| stratum | present | absent | discarded |
|---|---|---|---|
| injected | rank <= 10 | rank >= 500 | 11 to 499 |
| natural | rank <= 100 | rank >= 1,000 | 101 to 999 |

Justification for the natural threshold: the clean filter places battery concepts at median ranks in the tens of thousands at this position, so rank <= 100 is a shift of two or more orders of magnitude against the matched null. It is presence relative to the clean distribution, not a claim that natural and injected "present" are equal in magnitude. They are not (natural presence is roughly ten times weaker in the lens), and generalisation is reported per stratum. The 100/1,000 gap also absorbs bf16 row instability.

**Covariate recorded on every row.** The clean projection <h_clean, v_c> at the label position. Phase 1 found it is typically negative (median about -5 at L24), so "project to zero" under-ablates and the clean-value operation is the correct one.

**Strata.**
- *Natural presence*: two-hop descriptions that imply the concept without naming it. Pool: `natural_pool.py`, round one 6 per concept (300 descriptions) plus round two for the 38 thin concepts (`NATURAL_POOL_R2`), all passing a lexical self-check for concept and cross-battery leakage, filtered by the two-layer rule below (this paragraph records the L24 single-layer numbers that motivated the change). Under those margins at L24: **present 58, middle 54, absent 188**. This is well short of the 150 to 250 target and is dominated by emotions (34 of 58; countries 3, colours 1; 37 of 50 concepts have fewer than 3 present descriptions). The descriptions themselves work: the model names the concept unaided on 127 of 300 (animals 57%, foods 52%, emotions 73%), and within countries, animals, foods and colours the L24 lens rank predicts unaided naming well (AUC 0.82 to 0.92). The mismatch is that the lens at L24 under-registers natural presence: of the 188 prompts labelled absent (rank >= 1,000), 43 are named unaided, and prompts in the discarded middle band (101 to 999) are named at 74%, the same rate as the present band. So the frozen natural margin produces a small, emotion-heavy present set and puts named-unaided prompts into the absent class. The layer sweep (natural_layer_sweep.jsonl, 300 prompts, answer slot, layers 20 to 32) shows what is happening: natural presence becomes readable at the lens two layers later than injected presence, and the layer at which it becomes readable is the layer at which the lens starts reading the output. Present (rank <= 100) / named-but-labelled-absent (rank >= 1,000) / correlation of lens rank with the model's own answer, by layer: L24 58 / 44 / 0.65; L25 78 / 25 / 0.69; L26 122 / 3 / 0.80; L27 138 / 0 / 0.83; L28 139 / 0 / 0.85. On injected data the circular band was set at r > 0.78 (from L29). For natural prompts r crosses 0.80 at L26. So there is no natural-stratum layer that is both clean (few named prompts in the absent class) and clearly non-circular; L25 is the last layer under the injected criterion and still puts 25 named prompts into the absent class, L26 nearly clears the absent class (3) but sits at the circular boundary. This is a tradeoff to decide, not a threshold to tune. Under L26 the natural stratum would be present 122 / absent 109, with 23 of 50 concepts having at least 3 present descriptions; countries (9 present at L26, named unaided only 10% at any layer) and colours (9) stay thin regardless, which for countries is a description-quality problem and for colours is the Phase 1 finding again. DECIDED (option c): **two-layer labels for the natural stratum.** A natural prompt is present only if its lens rank is <= 100 at both L25 and L26, absent only if >= 1,000 at both, and is otherwise discarded. L25 is the last layer under the injected circularity criterion (r = 0.69 with the output on natural prompts); L26 is the first layer whose absent class is clean (r = 0.80). Requiring agreement means the label does not depend on which side of that boundary is chosen, which is the only version of the natural label against which the circularity objection cannot be raised later. Result on the 300-description pool: **present 78, absent 109, discarded 113**; named-but-absent falls from 44 (23% of the L24 absent class) to 3 (3%). The cost is size and balance: 46 of the 78 present are emotions, 12 of 50 concepts have at least 3 present descriptions, 24 have none, and four of the ten held-out concepts (Canada, honey, brown, yellow) have no natural-present example, so H1 on the natural stratum is evaluable only for the other six until the pool grows. A second drafting round of 6 descriptions for each of the 38 thin concepts is in progress, written to stack three specific clues per sentence (round one named countries unaided on only 10%, which is a description problem, not a lens problem), and will be filtered under the same two-layer rule. The H5 ablation is run at both L25 and L26 on the two-layer present set. Whether the injected stratum should adopt the same two-layer scheme for uniformity is being checked by regenerating its labels at L24, L25 and L26; `[[injected stratum: L24 only, or L25+L26 two-layer, decided from those counts]]`.
- *Injected presence*: the Phase 1 machinery (v_c at blocks 12 to 14, alpha = gamma * median residual norm), gamma in {0.2, 0.4}, on the 20 Phase 1 passages. Present count at gamma 0.4 (lens <= 10): 721 of 1,000 before any coherence exclusion, 38 of 1,000 at gamma 0.2; absent: 998 of 1,000 clean, 709 at gamma 0.2 (`[[final counts after the exclusion rule is settled]]`); this is also the only way to populate the not-inferable cell at eval.
- *Clean absence*: the 20 passages with no injection, plus natural-pool prompts labelled absent.

**Concepts and splits.** The 50-concept battery. 40 train / 10 held-out, stratified two per category; the held-out ten were drawn with `random.Random(20260905).sample(category, 2)` per category, before any Phase 2 label existed, and are: `Canada`, `Nepal`, `elephant`, `frog`, `garlic`, `honey`, `brown`, `yellow`, `anger`, `disgust` (countries: Canada, Nepal; animals: elephant, frog; foods: garlic, honey; colors: brown, yellow; emotions: anger, disgust). Note `spider`, the Phase 1 pilot concept, is in the training set; `elephant` and `frog` are the held-out animals. A second split holds out all ten emotions as the category-holdout variant. Colours stay in training and are reported as a pre-registered special subgroup.

**Twin control.** A second LoRA on identical inputs and questions, with labels computed from Qwen3-4B's own lens at the corresponding position. Qwen3-4B's clean floor is known (P(yes) 0.247, names nothing). Every headline number is reported as main minus twin.

**Training.** LoRA on Qwen3-8B; triples (input, label, Q&A) over both strata; hyperparameters fixed at `[[to be set from a 100-step smoke run, before this doc is frozen]]`. One seed for the main run; three seeds for the held-out-concept evaluation if compute allows.

**Scorer.** Case-insensitive, plural-tolerant match on the CONCEPT field for free generation; rank of the target for teacher-forced. Countries scored case-insensitively from the start.

## 2. Hypotheses

**H1 (held-out concepts).** Naming accuracy on the ten held-out concepts exceeds the twin's, with a 95% concept-resampled interval on (main minus twin) that excludes zero. Evaluated on both strata separately.

**H2 (injected, never-trained concepts).** Injected presence of a held-out concept is reported (named, top-5) above the clean false-positive rate on the same passages.

**H3 (lens-tracking).** Post-SFT AUC of the L24 lens rank for naming, on held-out inputs, exceeds both the in-context floor and the twin's AUC. The floor is position- and wording-specific: 0.556 was measured at the Phase 1 position under the injection wording; at the Phase 2 label position (answer slot, L24) under the neutral wording it is **0.761** (n = 1,000, gamma 0.4; by category: countries 0.70, animals 0.66, foods 0.65, emotions 0.83, colours undefined). 0.761 is the floor H3 must beat. Stating 0.556 would make H3 trivially easy.

**H4 (colours, instrument-validity probe).** Phase 1 shows the lens direction for colours is not the direction the model uses to name colours. H4 asks whether SFT on lens-derived labels teaches the model to read a direction it does not functionally use. Two pre-registered readouts on colours after training: (a) naming accuracy against lens ground truth, and (b) whether colour *usage* (unaided naming of naturally described colours, and answers to colour questions about inputs) stays anchored to the pre-training behaviour. If (a) rises while (b) is unchanged, training has shaped reports toward the instrument where instrument and functional representation disagree, which is evidence of calibrated-looking confabulation, not of improved access. If (a) rises and (b) shifts with it, access has changed. If (a) stays at zero, the block is downstream of anything the lens direction reaches. All three outcomes are reported; none is the "success" case in advance.

**H5 (importance rung), CONDITIONAL.** Zero-shot from presence training, the model answers "was C causally important to your output" above chance and above the presence-only rule. Ground truth is delta log P(c) under clean-value rank-one ablation of v_c at L24. This hypothesis is live only if the validation pilot on the natural stratum passes: delta log P(c) must have non-trivial variance across natural present prompts and AUC against naming of at least `[[threshold, set from a held-out slice of natural prompts]]`. Pilot result, first at L24 on the 58 single-layer present prompts: delta log P(c) under clean-value rank-one ablation has median -1.65, sd 1.47, IQR [-2.93, -0.02], and AUC 0.921 against unaided naming (proj-to-zero variant: 0.898). Real variance and strong discrimination, so the rung is live. Two caveats carried into the threshold-setting: 34 of the 58 are emotions (AUC 0.85 within emotions, 0.94 within foods at n = 11, undefined elsewhere), and the ablation rarely destroys the name on natural prompts (kill rate 7%, against 95% on injected), with median delta log P(c) near zero for animals and foods, which says that for natural presence the concept is carried redundantly rather than concentrated in v_c at L24. Re-run on the two-layer present set (78 prompts, named unaided 78%) at both label layers: at L25, delta log P(c) median -1.97, sd 2.19, AUC 0.899; at L26, median -3.45, sd 2.93, AUC 0.903; kill rate on named prompts 21% and 28%. The two layers agree on the importance signal almost exactly (Spearman 0.983 across prompts), so H5's ground truth is defined as the mean of the two and does not depend on the layer choice. The rung is live. The by-category pattern persists and is recorded as a limitation: emotions median -5.25 (n = 46) and countries -6.75 (n = 3) carry the effect; animals (n = 16) and foods (n = 12) sit near zero, so for those categories the natural name is not carried in v_c at either layer even where the lens reads it. The H5 threshold is set on a held-out slice of natural prompts once the round-two pool is filtered: `[[H5 threshold]]`. If it fails, H5 is cut, not patched. On injected data this label is uninformative (AUC 0.45 to 0.58) because the direction was written into every trial; that is the reason for the pilot.

**Secondary prediction (free).** A model trained on both strata learns a graded readout. Post-SFT, naming on natural prompts should show a dose response in the lens rank of the target (higher naming at rank <= 10 than at 11 to 100), and injected prompts should show it across gamma.

## 3. Decision rules

- H1 fails if the concept-resampled interval on (main minus twin) includes zero on both strata.
- H3 fails if post-SFT AUC does not exceed 0.556 on held-out inputs, regardless of the twin.
- H4 has no fail condition; it is a probe with three pre-specified readings.
- H5 is evaluated only if its pilot passed; otherwise it is reported as "not run, pilot failed" with the pilot numbers.
- Coherence exclusion, OPEN QUESTION for the mentor before freezing. The Phase 1 rule (exclude rows with mass < 0.5 at the yes/no slot) was written for the yes/no channel. Under the neutral wording at gamma 0.4, 839 of 1,000 injected rows fall below 0.5 at `PRESENT:` (the model puts its probability on words rather than yes/no there), yet those rows name the concept at 26.7% against 36.0% for the rest and the lens shows the concept present on 71% of them; the naming readout is prefilled past the `PRESENT:` slot and is not degraded. Applying the Phase 1 rule literally would discard 84% of the injected stratum for a defect in a channel Phase 2 does not train. Proposed replacement: exclude a row only if the `CONCEPT:` slot itself is incoherent, operationalised as the top-1 token at that slot not being an alphabetic word with a leading space; record the `PRESENT:` mass as a covariate rather than a filter. Also for the record: under the neutral wording the yes/no channel answers yes on every clean passage (P(present) median 1.000 at gamma 0), which is one more reason it is not trained.

## 4. What is already known and will not be presented as a result

The label position, the margins, the elicitation, and the splits above were chosen from Phase 1 and the patching pilots, all of which are released. The 0.556 floor, the 95% rank-one kill rate, the 22/50 natural-presence rate under the neutral wording, and the colour dissociation are inputs to this design, not outputs of it.

## 5. Sequence and checkpoints

1. Regenerate injected-stratum labels under the neutral wording: `[[done / date]]`.
2. Scale and filter the natural pool: `[[done / date]]`.
3. Importance validation pilot on natural present prompts: `[[done / date / verdict]]`.
4. Freeze this document; send to mentor. **No training before sign-off.**
5. Build triples; smoke-run 100 steps; fix hyperparameters; freeze again.
6. Train main and twin.
7. Evaluate: held-out concepts, emotion holdout, colours, importance rung if live.
