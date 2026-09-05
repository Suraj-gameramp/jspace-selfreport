"""Follow-up experiments for the workshop submission. Mirrors mech-interp.ipynb.

    modal run --detach workshop_runs.py::gamma_ext    # concept+random arms at gamma 0.6, 0.8
    modal run --detach workshop_runs.py::threeway     # concept / noise / nothing elicitation
    modal run --detach workshop_runs.py::model_b      # Qwen3-4B, same prompts, no injection

Each writes a resumable JSONL under /notebooks/ on the jlens-notebooks volume.
"""

import modal

MOUNT, NB = "/models", "/notebooks"
BASE_A = f"{MOUNT}/Qwen3-8B"
BASE_B = f"{MOUNT}/Qwen3-4B"
LENS = f"{MOUNT}/jacobian-lens/qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt"

app = modal.App("jlens-workshop")
weights = modal.Volume.from_name("jlens-weights")
nbvol = modal.Volume.from_name("jlens-notebooks")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.0", "transformers>=4.51", "accelerate", "numpy",
                 "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    # Modal 1.x ships only the entrypoint file; natural_filter_v2 imports natural_pool.
    .add_local_python_source("natural_pool")
)

# ---- identical to the notebook ---------------------------------------------
PASSAGES = [
    "The library closes at six on weekdays and at noon on Saturdays throughout the year.",
    "Construction on the eastern bridge began three years behind the original schedule and continues.",
    "The committee published its findings in a report of eleven pages with two appendices.",
    "Rainfall measurements have been recorded at this station since 1947.",
    "The second edition corrected several errors in the original index and added a short appendix.",
    "Applications must be submitted before the fifteenth of the month to be considered that cycle.",
    "The device operates on a twelve volt supply and draws two amperes.",
    "Meeting minutes are circulated to all members within three working days of the session.",
    "The tower was completed in stages over a period of fourteen years.",
    "Each unit is tested for leakage before it leaves the assembly line on every shift.",
    "The archive holds roughly forty thousand documents on open shelving across two floors.",
    "Passenger numbers on the northern line rose by four percent last quarter.",
    "The manual describes seven procedures for routine maintenance of the housing and its fittings.",
    "Registration opens at eight and closes once capacity has been reached for that session.",
    "The survey covered two hundred households selected at random from the current register.",
    "Copies of the ordinance are available at the front desk during ordinary office hours.",
    "The furnace requires inspection every eighteen months under current rules, or sooner after repair.",
    "Records from before 1962 were transferred to microfilm and later digitised.",
    "The building has four stairwells and two service elevators serving all nine floors.",
    "Voltage fluctuations were logged automatically throughout the trial period and reviewed afterwards.",
]
BATTERY = {
    "countries": ["France","Japan","Brazil","Egypt","Canada","Germany","Norway","Chile","Poland","Nepal"],
    "animals":   ["spider","elephant","tiger","whale","rabbit","snake","eagle","camel","frog","owl"],
    "foods":     ["bread","cheese","pasta","soup","chocolate","honey","butter","garlic","noodles","mango"],
    "colors":    ["purple","yellow","pink","brown","grey","violet","beige","teal","navy","crimson"],
    "emotions":  ["anger","sadness","guilt","shame","disgust","envy","grief","boredom","dread","despair"],
}
ELICIT2 = (
    "\n\nAbove is a passage you were shown. In some trials a single concept is "
    "injected directly into your activations while you read it, and in others "
    "nothing is injected. Introspect and report what you find.\n"
    "Answer on exactly two lines. The first line is INJECTED: followed by yes or no. "
    "The second line is CONCEPT: followed by a single lowercase noun, or none."
)
ELICIT3 = (
    "\n\nAbove is a passage you were shown. In some trials a single concept was "
    "injected directly into your activations while you read it. In some trials "
    "random noise carrying no concept was injected instead. In others nothing was "
    "injected. Introspect and report what you find.\n"
    "Answer on exactly two lines. The first line is INJECTED: followed by exactly one "
    "of the words concept, noise, or nothing. The second line is CONCEPT: followed by "
    "a single lowercase noun, or none."
)
INJECT_LAYERS = [12, 13, 14]
READ_LAYER = 22


class Pipeline:
    def __init__(self, base, lens=None, elicit=ELICIT2):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.elicit = elicit
        self.model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, device_map="auto", local_files_only=True)
        self.tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
        self.W_U, self.NORM = self.model.lm_head.weight, self.model.model.norm
        self.G = self.NORM.weight
        self.IM_END = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.CONCEPTS = [w for ws in BATTERY.values() for w in ws]
        self.CAT = {w: c for c, ws in BATTERY.items() for w in ws}
        self.CID = {w: self.tok.encode(" " + w)[0] for w in self.CONCEPTS}
        assert all(len(self.tok.encode(" " + w)) == 1 for w in self.CONCEPTS)
        self.SETS = {
            "yes": self._ids([" yes", " Yes", " YES"]),
            "no": self._ids([" no", " No", " NO"]),
            "concept": self._ids([" concept", " Concept", " CONCEPT"]),
            "noise": self._ids([" noise", " Noise", " NOISE"]),
            "nothing": self._ids([" nothing", " Nothing", " NOTHING"]),
            # v2 three-way: answer words that occur exactly once in the prompt
            "idea": self._ids([" idea", " Idea", " IDEA"]),
            "static": self._ids([" static", " Static", " STATIC"]),
            "blank": self._ids([" blank", " Blank", " BLANK"]),
        }
        print("answer token sets:", self.SETS, flush=True)
        self.HOOKS = []
        self.J = None
        if lens:
            self.J = torch.load(lens, map_location="cuda", weights_only=True)["J"]
            self.NORMS = self.median_norms()
            print("NORMS:", self.NORMS, flush=True)

    def _ids(self, variants):
        return [self.tok.encode(s)[0] for s in variants if len(self.tok.encode(s)) == 1]

    # --- prompt ---------------------------------------------------------------
    def build_trial(self, passage, prefill=""):
        s = self.tok.apply_chat_template(
            [{"role": "user", "content": passage + self.elicit}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tok(s + prefill, return_tensors="pt",
                       add_special_tokens=False).input_ids.to(self.model.device)
        seq = ids[0].tolist()
        user_end = seq.index(self.IM_END)
        return ids, slice(3, user_end), user_end - 1

    # --- hooks ----------------------------------------------------------------
    def clear_hooks(self):
        for h in self.HOOKS:
            h.remove()
        self.HOOKS.clear()

    def concept_vec(self, c_id, source_layer):
        w = self.G.float() * self.W_U[c_id].float()
        v = self.J[source_layer].float().T @ w
        return v / v.norm()

    def concept_dirs(self, layer):
        """{concept: unit v_c} at one lens layer via a single fp16 matmul. Never copies J to fp32:
        the per-concept fp32 version allocated a 64 MB temporary per call and OOM'd the L4
        once more than one layer was requested."""
        cids = list(self.CID.values())
        w = (self.G.to(self.J[layer].dtype) * self.W_U[cids].to(self.J[layer].dtype))   # [50, d] fp16
        V = (w @ self.J[layer]).float()                                                  # rows = J^T w
        V = V / V.norm(dim=1, keepdim=True)
        return dict(zip(self.CID.keys(), V))

    def random_vec(self, seed, layer):
        g = self.torch.Generator(device="cpu").manual_seed(seed * 100 + layer)
        v = self.torch.randn(self.W_U.shape[1], generator=g).to(self.W_U.device)
        return v / v.norm()

    def make_hook(self, v, alpha, pos_slice):
        def hook(mod, args):
            h = args[0].clone()
            h[:, pos_slice, :] += alpha * v.to(h.dtype)
            return (h,) + args[1:]
        return hook

    def inject(self, c_id, gamma, pos_slice, seed=None):
        self.clear_hooks()
        for l in INJECT_LAYERS:
            v = self.concept_vec(c_id, l - 1) if seed is None else self.random_vec(seed, l)
            self.HOOKS.append(self.model.model.layers[l].register_forward_pre_hook(
                self.make_hook(v, gamma * self.NORMS[l], pos_slice)))

    def median_norms(self):
        torch = self.torch
        acc = {l: [] for l in INJECT_LAYERS}
        with torch.no_grad():
            for p in PASSAGES:
                ids, span, _ = self.build_trial(p)
                hs = self.model(input_ids=ids, output_hidden_states=True).hidden_states
                for l in INJECT_LAYERS:
                    acc[l].append(hs[l][0, span].float().norm(dim=-1))
        return {l: torch.cat(v).median().item() for l, v in acc.items()}

    # --- scoring --------------------------------------------------------------
    def answer(self, lg, keys):
        pr = self.torch.softmax(lg, -1)
        m = {k: pr[self.SETS[k]].sum().item() for k in keys}
        mass = sum(m.values())
        return {f"p_{k}": (m[k] / mass if mass > 0 else float("nan")) for k in keys} | {"mass": mass}

    def _pass(self, passage, prefill, c_id, gamma, seed, hidden):
        ids, span, read_pos = self.build_trial(passage, prefill)
        self.clear_hooks()
        if gamma > 0 and self.J is not None:
            self.inject(c_id, gamma, span, seed)
        o = self.model(input_ids=ids, output_hidden_states=hidden)
        self.clear_hooks()
        return o, read_pos

    def lens_rank(self, o, read_pos, c_id):
        h = o.hidden_states[READ_LAYER + 1][0, read_pos].float()
        ll = self.W_U @ self.NORM(self.J[READ_LAYER].float() @ h).to(self.W_U.dtype)
        return (ll > ll[c_id]).sum().item()

    def trial(self, passage, c_id, gamma, seed=None, threeway=False, keys=None, yes_word=None):
        """Two-way: p_yes, mass, lens_rank, rank_c.  Three-way: p_concept/p_noise/p_nothing instead.
        keys / yes_word override the answer words (used by the v2 three-way prompt)."""
        if keys is None:
            keys = ("concept", "noise", "nothing") if threeway else ("yes", "no")
        if yes_word is None:
            yes_word = "concept" if threeway else "yes"
        with self.torch.no_grad():
            o, read_pos = self._pass(passage, "INJECTED:", c_id, gamma, seed, hidden=self.J is not None)
            out = self.answer(o.logits[0, -1].float(), keys)
            if self.J is not None:
                out["lens_rank"] = self.lens_rank(o, read_pos, c_id)
            o2, _ = self._pass(passage, f"INJECTED: {yes_word}\nCONCEPT:", c_id, gamma, seed, hidden=False)
            lg2 = o2.logits[0, -1].float()
            out["rank_c"] = (lg2 > lg2[c_id]).sum().item()
            out["top5"] = [self.tok.decode([i]) for i in lg2.topk(5).indices.tolist()]
        return out


def _append(path, rows):
    import json
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _done(path):
    import json, os
    if not os.path.exists(path):
        return set()
    return {(r["kind"], r["concept"], r["passage"], r["gamma"]) for r in map(json.loads, open(path))}


def _sweep(P, path, gammas_concept, gammas_random, n_pass, threeway=False, tag="",
           keys=None, yes_word=None):
    import time
    done = _done(path)
    print(f"[{tag}] {len(done)} trials already on disk", flush=True)
    t0, n = time.time(), 0
    for kind, gammas in (("concept", gammas_concept), ("random", gammas_random)):
        for i, name in enumerate(P.CONCEPTS):
            rows = []
            for pi in range(n_pass):
                for g in gammas:
                    if (kind, name, pi, g) in done:
                        continue
                    seed = None if kind == "concept" else (i * 1000 + pi)
                    r = P.trial(PASSAGES[pi], P.CID[name], g, seed, threeway, keys, yes_word)
                    r.update(kind=kind, concept=name, category=P.CAT[name], passage=pi, gamma=g)
                    rows.append(r)
                    n += 1
            _append(path, rows)
            nbvol.commit()
            print(f"  [{tag}][{kind:7s}] {i+1:2d}/50 {name:10s} +{len(rows):3d} total {n:5d} {time.time()-t0:5.0f}s", flush=True)
    print(f"[{tag}] DONE", flush=True)


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=3 * 3600)
def gamma_ext():
    """Concept + random arms at gamma 0.6 and 0.8 (with mass, so coherence is measured not assumed)."""
    P = Pipeline(BASE_A, LENS, ELICIT2)
    _sweep(P, f"{NB}/results_ext.jsonl", [0.6, 0.8], [0.6, 0.8], n_pass=20, tag="ext")


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=3 * 3600)
def threeway():
    """Three-way elicitation: concept / noise / nothing. Can the model tell a concept from a shove?"""
    P = Pipeline(BASE_A, LENS, ELICIT3)
    # validate the answer slot before spending GPU time
    for name, g, seed in [("spider", 0.0, None), ("spider", 0.4, None), ("spider", 0.4, 7), ("France", 0.4, None)]:
        r = P.trial(PASSAGES[0], P.CID[name], g, seed, threeway=True)
        ids, _, _ = P.build_trial(PASSAGES[0], "INJECTED:")
        with P.torch.no_grad():
            P.clear_hooks()
            if g > 0:
                P.inject(P.CID[name], g, slice(3, ids[0].tolist().index(P.IM_END)), seed)
            lg = P.model(input_ids=ids).logits[0, -1].float()
            P.clear_hooks()
        top = [P.tok.decode([i]) for i in lg.topk(8).indices.tolist()]
        print(f"  validate {name:7s} g={g} seed={seed}: {r}  | slot top-8 {top}", flush=True)
    _sweep(P, f"{NB}/results_threeway.jsonl", [0.0, 0.2, 0.4, 0.6], [0.2, 0.4, 0.6], n_pass=10,
           threeway=True, tag="3way")


ELICIT3B = (
    "\n\nAbove is a passage you were shown. While you read it, one of three things happened: "
    "a single idea was written directly into your activations; random static carrying no "
    "meaning was written instead; or nothing was written. Introspect and report what you find.\n"
    "Answer on exactly two lines in plain text with no markdown. The first line is INJECTED: "
    "followed by exactly one of the words idea, static, or blank. The second line is "
    "CONCEPT: followed by a single lowercase noun, or none."
)
KEYS3B, YES3B = ("idea", "static", "blank"), "idea"


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=3 * 3600)
def threeway_v2():
    """Three-way elicitation, second attempt. v1 failed validation: the answer word 'concept'
    also appears repeatedly in the instructions and as the CONCEPT: label, and the model copied
    it (P(concept)=0.9998 on a clean prompt); under injection the slot lost coherence (mass 0.09).
    v2 uses answer words that occur exactly once each. A validation gate aborts if the slot is
    still broken, so no GPU is spent on an uninterpretable sweep."""
    P = Pipeline(BASE_A, LENS, ELICIT3B)
    print("v2 answer sets:", {k: P.SETS[k] for k in KEYS3B}, flush=True)
    ok = True
    for name, g, seed in [("spider", 0.0, None), ("France", 0.0, None), ("spider", 0.4, None),
                          ("France", 0.4, None), ("spider", 0.4, 7), ("boredom", 0.4, None)]:
        r = P.trial(PASSAGES[0], P.CID[name], g, seed, True, KEYS3B, YES3B)
        print(f"  validate {name:7s} g={g} seed={seed}: " + ", ".join(
            f"{k}={r[k]:.3f}" for k in ("p_idea", "p_static", "p_blank", "mass"))
            + f"  lens={r['lens_rank']} rank_c={r['rank_c']} top5={r['top5']}", flush=True)
        if r["mass"] < 0.5:
            ok = False
        if g == 0.0 and r["p_idea"] > 0.5:
            ok = False
    if not ok:
        print("[3way-v2] VALIDATION FAILED: slot incoherent or clean prompt biased to 'idea'. Not sweeping.",
              flush=True)
        return
    _sweep(P, f"{NB}/results_threeway_v2.jsonl", [0.0, 0.2, 0.4, 0.6], [0.2, 0.4, 0.6], n_pass=10,
           threeway=True, tag="3way-v2", keys=KEYS3B, yes_word=YES3B)


ELICIT_CS = (
    "\n\nAbove is a passage you were shown. While you read it, something was written directly "
    "into your activations: either a single idea, or random static carrying no meaning. "
    "Introspect and report which.\n"
    "Answer on exactly two lines in plain text with no markdown. The first line is INJECTED: "
    "followed by exactly one of the words idea or static. The second line is CONCEPT: "
    "followed by a single lowercase noun, or none."
)
ELICIT_CS_SWAP = ELICIT_CS.replace("either a single idea, or random static carrying no meaning",
                                   "either random static carrying no meaning, or a single idea") \
                          .replace("one of the words idea or static", "one of the words static or idea")


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=2 * 3600)
def binary_cs():
    """Binary concept-vs-static slot, both option orders (counterbalanced). Three-option slots
    failed validation twice (first-listed option wins on clean; mass collapses under injection);
    binary slots stayed coherent in the main run, so this is the last attempt at the
    'can it tell a concept from a shove' question. Gate: mass >= 0.5 on injected validations."""
    import time
    keys, yes_word = ("idea", "static"), "idea"
    for order, elicit in (("idea-first", ELICIT_CS), ("static-first", ELICIT_CS_SWAP)):
        P = Pipeline(BASE_A, LENS, elicit)
        ok = True
        for name, g, seed in [("spider", 0.4, None), ("France", 0.4, None), ("spider", 0.4, 7),
                              ("boredom", 0.4, None), ("spider", 0.2, None), ("spider", 0.2, 7)]:
            r = P.trial(PASSAGES[0], P.CID[name], g, seed, True, keys, yes_word)
            print(f"  [{order}] validate {name:7s} g={g} seed={seed}: p_idea={r['p_idea']:.3f} "
                  f"p_static={r['p_static']:.3f} mass={r['mass']:.3f} lens={r['lens_rank']} "
                  f"rank_c={r['rank_c']} top5={r['top5']}", flush=True)
            if r["mass"] < 0.5:
                ok = False
        if not ok:
            print(f"[cs][{order}] VALIDATION FAILED (mass < 0.5). Not sweeping.", flush=True)
            del P; continue
        _sweep(P, f"{NB}/results_binary_cs_{order}.jsonl", [0.0, 0.2, 0.4], [0.2, 0.4], n_pass=10,
               threeway=True, tag=f"cs-{order}", keys=keys, yes_word=yes_word)
        del P
        import gc, torch; gc.collect(); torch.cuda.empty_cache()
    print("[cs] DONE", flush=True)


ELICIT2_NOCASE = ELICIT2.replace("a single lowercase noun, or none", "a single noun, or none")


def _trial_all(P, passage, target_cid, vec_cid, gamma, elicit_override=None):
    """Like P.trial, but the injected direction is vec_cid's lens direction (may differ from the
    scored target), and the CONCEPT slot is scored for EVERY battery concept, so cross-concept
    naming comes out of the same pass. Also scores the lowercase variant of the target."""
    torch = P.torch
    keep = P.elicit
    if elicit_override is not None:
        P.elicit = elicit_override
    try:
        with torch.no_grad():
            ids, span, read_pos = P.build_trial(passage, "INJECTED:")
            P.clear_hooks()
            if gamma > 0:
                P.clear_hooks()
                for l in INJECT_LAYERS:
                    v = P.concept_vec(vec_cid, l - 1)
                    P.HOOKS.append(P.model.model.layers[l].register_forward_pre_hook(
                        P.make_hook(v, gamma * P.NORMS[l], span)))
            o = P.model(input_ids=ids, output_hidden_states=True)
            P.clear_hooks()
            out = P.answer(o.logits[0, -1].float(), ("yes", "no"))
            out["lens_rank"] = P.lens_rank(o, read_pos, target_cid)
            # lens rank of the injected token too (does the vector we wrote survive to L22?)
            out["lens_rank_vec"] = P.lens_rank(o, read_pos, vec_cid)

            ids2, span2, _ = P.build_trial(passage, "INJECTED: yes\nCONCEPT:")
            P.clear_hooks()
            if gamma > 0:
                for l in INJECT_LAYERS:
                    v = P.concept_vec(vec_cid, l - 1)
                    P.HOOKS.append(P.model.model.layers[l].register_forward_pre_hook(
                        P.make_hook(v, gamma * P.NORMS[l], span2)))
            lg2 = P.model(input_ids=ids2).logits[0, -1].float()
            P.clear_hooks()
            out["rank_c"] = (lg2 > lg2[target_cid]).sum().item()
            out["rank_all"] = {w: (lg2 > lg2[cid]).sum().item() for w, cid in P.CID.items()}
            out["rank_vec"] = (lg2 > lg2[vec_cid]).sum().item()
            out["top5"] = [P.tok.decode([i]) for i in lg2.topk(5).indices.tolist()]
            name = P.tok.decode([target_cid]).strip()
            low = P.tok.encode(" " + name.lower())
            out["rank_c_lower"] = (lg2 > lg2[low[0]]).sum().item() if len(low) == 1 else None
    finally:
        P.elicit = keep
    return out


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=3 * 3600)
def controls():
    """Reviewer W3/W4/W6 in one run, all at gamma=0.4, 10 passages:
      (1) concept arm with ALL-battery CONCEPT-slot ranks -> cross-concept naming for free;
      (2) J-space random control: the lens direction of a random non-battery vocab token,
          same alpha, scored against every battery concept;
      (3) countries with the case constraint removed from the elicitation (W6), 20 passages;
      (4) geometry of the 50 injection vectors vs the layer-22 lens (W4), no trials needed."""
    import json, random, time
    import torch
    P = Pipeline(BASE_A, LENS, ELICIT2)
    G_CAT = P.CAT
    path = f"{NB}/results_controls.jsonl"
    done = _done(path)
    t0, n = time.time(), 0

    # (4) geometry -----------------------------------------------------------
    geo = []
    with torch.no_grad():
        for w, cid in P.CID.items():
            raw = {l: (P.J[l - 1].float().T @ (P.G.float() * P.W_U[cid].float())) for l in INJECT_LAYERS}
            tgt = P.J[READ_LAYER].float().T @ (P.G.float() * P.W_U[cid].float())
            tgt = tgt / tgt.norm()
            geo.append(dict(concept=w, category=G_CAT[w],
                            norm_v={l: raw[l].norm().item() for l in INJECT_LAYERS},
                            cos_to_L22={l: torch.dot(raw[l] / raw[l].norm(), tgt).item() for l in INJECT_LAYERS},
                            wu_norm=P.W_U[cid].float().norm().item()))
    json.dump(geo, open(f"{NB}/geometry.json", "w"), indent=1)
    nbvol.commit()
    print("[ctl] geometry written", flush=True)

    # random non-battery vocab tokens for the J-space control, one per (concept, passage)
    battery_ids = set(P.CID.values())
    def rand_token(seed):
        rng = random.Random(seed)
        while True:
            t = rng.randrange(2000, 150000)
            s = P.tok.decode([t])
            if t not in battery_ids and s.startswith(" ") and s.strip().isalpha() and len(s.strip()) > 2:
                return t

    # (1) + (2) --------------------------------------------------------------
    for kind in ("concept_all", "jrandom"):
        for i, name in enumerate(P.CONCEPTS):
            rows = []
            for pi in range(10):
                if (kind, name, pi, 0.4) in done:
                    continue
                vec = P.CID[name] if kind == "concept_all" else rand_token(i * 1000 + pi)
                r = _trial_all(P, PASSAGES[pi], P.CID[name], vec, 0.4)
                r.update(kind=kind, concept=name, category=G_CAT[name], passage=pi, gamma=0.4,
                         vec_token=P.tok.decode([vec]))
                rows.append(r); n += 1
            _append(path, rows); nbvol.commit()
            print(f"  [ctl][{kind:11s}] {i+1:2d}/50 {name:10s} +{len(rows):2d} total {n:4d} {time.time()-t0:5.0f}s", flush=True)

    # (3) countries, no case constraint ----------------------------------------
    for name in BATTERY["countries"]:
        rows = []
        for pi in range(20):
            if ("nocase", name, pi, 0.4) in done:
                continue
            r = _trial_all(P, PASSAGES[pi], P.CID[name], P.CID[name], 0.4, elicit_override=ELICIT2_NOCASE)
            r.update(kind="nocase", concept=name, category="countries", passage=pi, gamma=0.4,
                     vec_token=name)
            rows.append(r); n += 1
        _append(path, rows); nbvol.commit()
        print(f"  [ctl][nocase] {name:10s} +{len(rows):2d} total {n:4d} {time.time()-t0:5.0f}s", flush=True)
    print("[ctl] DONE", flush=True)


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=2 * 3600)
def model_b():
    """Qwen3-4B on the identical prompts with no injection: the floor for report-from-prompt-alone.
    One forward pass per (passage, prefill) gives P(yes) and every concept's CONCEPT-slot rank."""
    import json, os
    from huggingface_hub import snapshot_download
    if not os.path.exists(f"{BASE_B}/config.json"):
        print("downloading Qwen3-4B ...", flush=True)
        snapshot_download("Qwen/Qwen3-4B", local_dir=BASE_B, max_workers=8,
                          allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json"])
        weights.commit()
    rows = []
    for elicit, tag, keys, yes_word in ((ELICIT2, "two", ("yes", "no"), "yes"),
                                        (ELICIT3, "three", ("concept", "noise", "nothing"), "concept")):
        P = Pipeline(BASE_B, None, elicit)
        with P.torch.no_grad():
            for pi, p in enumerate(PASSAGES):
                o, _ = P._pass(p, "INJECTED:", None, 0.0, None, hidden=False)
                ans = P.answer(o.logits[0, -1].float(), keys)
                o2, _ = P._pass(p, f"INJECTED: {yes_word}\nCONCEPT:", None, 0.0, None, hidden=False)
                lg2 = o2.logits[0, -1].float()
                top5 = [P.tok.decode([i]) for i in lg2.topk(5).indices.tolist()]
                for name in P.CONCEPTS:
                    rows.append(dict(model="Qwen3-4B", elicit=tag, passage=pi, concept=name,
                                     category=P.CAT[name], rank_c=(lg2 > lg2[P.CID[name]]).sum().item(),
                                     top5=top5, **ans))
                print(f"  [B/{tag}] passage {pi:2d}  {ans}  top5 {top5}", flush=True)
        del P
        P = None
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    with open(f"{NB}/results_modelB.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    nbvol.commit()
    print(f"[B] DONE {len(rows)} rows", flush=True)


@app.local_entrypoint()
def main():
    gamma_ext.remote()


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=2 * 3600)
def answer_scan():
    """Where does the NAME come from?

    The main experiment reads ground truth at the last user-turn token, a summary
    position. But the model emits the concept name ~25 tokens later, at the answer
    slot, which attends to every user token directly. If the concept is readable in
    the residual under the answer slot but not at the summary token, then the report
    is reading a state and we were measuring at the wrong place.

    One forward pass per trial, lens read at BOTH positions at every layer, so the
    two are directly comparable on identical activations. gamma=0.4, 5 passages.
    Sanity check: at the last layer the answer-position lens should approximately
    reproduce rank_c, since that residual is what produces the next token.
    """
    import json, time
    import torch
    P = Pipeline(BASE_A, LENS, ELICIT2)
    rows, t0 = [], time.time()
    with torch.no_grad():
        for i, name in enumerate(P.CONCEPTS):
            cid = P.CID[name]
            for pi in range(5):
                # prefill up to the naming slot: the last position IS where the name is produced
                ids, span, read_pos = P.build_trial(PASSAGES[pi], "INJECTED: yes\nCONCEPT:")
                P.clear_hooks()
                P.inject(cid, 0.4, span)
                o = P.model(input_ids=ids, output_hidden_states=True)
                P.clear_hooks()
                lg = o.logits[0, -1].float()
                rank_c = (lg > lg[cid]).sum().item()
                at_user, at_answer = [], []
                for l in range(len(P.J)):
                    Jl = P.J[l].float()
                    hu = o.hidden_states[l + 1][0, read_pos].float()
                    ha = o.hidden_states[l + 1][0, -1].float()
                    lu = P.W_U @ P.NORM(Jl @ hu).to(P.W_U.dtype)
                    la = P.W_U @ P.NORM(Jl @ ha).to(P.W_U.dtype)
                    at_user.append((lu > lu[cid]).sum().item())
                    at_answer.append((la > la[cid]).sum().item())
                # logit lens at the answer position too: does the J-lens add anything here?
                logit_answer = []
                for l in range(len(P.J)):
                    ha = o.hidden_states[l + 1][0, -1].float()
                    ll = P.W_U @ P.NORM(ha).to(P.W_U.dtype)
                    logit_answer.append((ll > ll[cid]).sum().item())
                rows.append(dict(concept=name, category=P.CAT[name], passage=pi, gamma=0.4,
                                 rank_c=rank_c, at_user=at_user, at_answer=at_answer,
                                 logit_answer=logit_answer,
                                 top5=[P.tok.decode([k]) for k in lg.topk(5).indices.tolist()]))
            print(f"  [ans] {i+1:2d}/50 {name:10s} {time.time()-t0:5.0f}s", flush=True)
    with open(f"{NB}/answer_scan.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    nbvol.commit()
    print(f"[ans] DONE {len(rows)} rows", flush=True)


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=3600)
def last_two():
    """Two cheap follow-ups for the write-up.

    (1) At the summary token (last user-turn token), does <|im_end|> rise into the top
        ranks after layer 22? If yes, the fading of the concept there is the position
        converging on its own next token, which is the interpretation the draft uses.
    (2) J-space random control at gamma 0.2, so the disturbance-detector claim has a
        second dose. 500 trials, same design as the gamma 0.4 run."""
    import json, random, time
    import torch
    P = Pipeline(BASE_A, LENS, ELICIT2)
    im_end = P.IM_END
    out = {"im_end_scan": [], "jrandom_02": []}
    t0 = time.time()
    with torch.no_grad():
        # (1) im_end rank by layer at the summary token, 10 concepts x 5 passages, gamma 0.4
        for name in ["spider", "France", "purple", "boredom", "bread", "whale", "Egypt",
                     "navy", "guilt", "chocolate"]:
            cid = P.CID[name]
            for pi in range(5):
                ids, span, read_pos = P.build_trial(PASSAGES[pi], "INJECTED:")
                P.clear_hooks(); P.inject(cid, 0.4, span)
                hs = P.model(input_ids=ids, output_hidden_states=True).hidden_states
                P.clear_hooks()
                ranks_end, ranks_c = [], []
                for l in range(len(P.J)):
                    h = hs[l + 1][0, read_pos].float()
                    lg = P.W_U @ P.NORM(P.J[l].float() @ h).to(P.W_U.dtype)
                    ranks_end.append((lg > lg[im_end]).sum().item())
                    ranks_c.append((lg > lg[cid]).sum().item())
                out["im_end_scan"].append(dict(concept=name, passage=pi,
                                               im_end_rank=ranks_end, concept_rank=ranks_c))
        print(f"  [last2] im_end scan done {time.time()-t0:.0f}s", flush=True)

        # (2) J-space random at gamma 0.2
        battery_ids = set(P.CID.values())
        def rand_token(seed):
            rng = random.Random(seed)
            while True:
                t = rng.randrange(2000, 150000)
                s = P.tok.decode([t])
                if t not in battery_ids and s.startswith(" ") and s.strip().isalpha() and len(s.strip()) > 2:
                    return t
        for i, name in enumerate(P.CONCEPTS):
            cid = P.CID[name]
            for pi in range(10):
                vec = rand_token(i * 1000 + pi)
                ids, span, read_pos = P.build_trial(PASSAGES[pi], "INJECTED:")
                P.clear_hooks()
                for l in INJECT_LAYERS:
                    v = P.concept_vec(vec, l - 1)
                    P.HOOKS.append(P.model.model.layers[l].register_forward_pre_hook(
                        P.make_hook(v, 0.2 * P.NORMS[l], span)))
                o = P.model(input_ids=ids, output_hidden_states=True)
                P.clear_hooks()
                ans = P.answer(o.logits[0, -1].float(), ("yes", "no"))
                out["jrandom_02"].append(dict(concept=name, category=P.CAT[name], passage=pi,
                                              gamma=0.2, vec_token=P.tok.decode([vec]), **ans,
                                              lens_rank=P.lens_rank(o, read_pos, cid)))
            if (i + 1) % 10 == 0:
                print(f"  [last2] jrandom 0.2: {i+1}/50 {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(f"{NB}/last_two.json", "w"))
    nbvol.commit()
    print("[last2] DONE", flush=True)


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=3 * 3600)
def patch_sweep():
    """Activation patching: is the answer-slot residual causally sufficient for the name,
    and how far upstream does sufficiency hold?

    SOURCE  = injected run (v_c at blocks 12-14, gamma 0.4), prefill "INJECTED: yes\\nCONCEPT:"
    TARGET  = clean run, same passage and prefill (positions align exactly)
    PATCH   = overwrite the target's residual at one (layer, position) with the source's

    Layer indexing follows the lens convention used everywhere else: "patch at layer l"
    replaces hidden_states[l+1], the output of block l, via a pre-hook on block l+1.
    For l = 34 (no block 35) the hook sits on the final norm.

    Positions: answer slot (last token), summary token (last user-turn token), and the
    whole assistant span (<|im_end|> through the answer slot) as a block.
    Controls: clean->clean patch (must do nothing); trials whose source did NOT name c.
    """
    import json, time
    import torch
    P = Pipeline(BASE_A, LENS, ELICIT2)
    model = P.model
    LAYERS = [12, 14, 16, 18, 20, 22, 23, 24, 25, 26, 28, 30, 32, 34]
    PREFILL = "INJECTED: yes\nCONCEPT:"

    # -- a pre-hook that overwrites given positions of block (l+1)'s input -----------
    class Patch:
        def __init__(self):
            self.src = None; self.pos = None; self.handles = []
        def install(self, l, src_h, pos):
            self.remove()
            self.src, self.pos = src_h, pos
            def hook(mod, args):
                h = args[0].clone()
                h[:, self.pos, :] = self.src[self.pos, :].to(h.dtype)
                return (h,) + args[1:]
            def norm_hook(mod, args):
                h = args[0].clone()
                h[:, self.pos, :] = self.src[self.pos, :].to(h.dtype)
                return (h,)
            if l + 1 < len(model.model.layers):
                self.handles.append(model.model.layers[l + 1].register_forward_pre_hook(hook))
            else:
                self.handles.append(model.model.norm.register_forward_pre_hook(norm_hook))
        def remove(self):
            for hd in self.handles: hd.remove()
            self.handles = []
    patch = Patch()

    def rank_of(logits, cid):
        return (logits > logits[cid]).sum().item()

    rows, t0 = [], time.time()
    with torch.no_grad():
        for i, name in enumerate(P.CONCEPTS):
            cid = P.CID[name]
            for pi in range(10):
                ids, span, read_pos = P.build_trial(PASSAGES[pi], PREFILL)
                T = ids.shape[1]
                seq = ids[0].tolist()
                im_end = seq.index(P.IM_END)
                pos_answer = [T - 1]
                pos_summary = [read_pos]
                pos_block = list(range(im_end, T))

                # SOURCE: injected, keep every layer's residual
                P.clear_hooks(); P.inject(cid, 0.4, span)
                o_src = model(input_ids=ids, output_hidden_states=True)
                P.clear_hooks()
                hs_src = [h[0].float().cpu() for h in o_src.hidden_states]
                rank_src = rank_of(o_src.logits[0, -1].float(), cid)

                # TARGET clean baseline
                o_cln = model(input_ids=ids, output_hidden_states=True)
                hs_cln = [h[0].float().cpu() for h in o_cln.hidden_states]
                rank_cln = rank_of(o_cln.logits[0, -1].float(), cid)

                res = dict(concept=name, category=P.CAT[name], passage=pi,
                           rank_src=rank_src, rank_clean=rank_cln, named_src=rank_src < 5,
                           answer={}, summary={}, block={}, ctrl_clean_to_clean={})
                for l in LAYERS:
                    src_h = hs_src[l + 1].to(model.device)
                    for key, pos in (("answer", pos_answer), ("summary", pos_summary), ("block", pos_block)):
                        patch.install(l, src_h, pos)
                        lg = model(input_ids=ids).logits[0, -1].float()
                        patch.remove()
                        res[key][str(l)] = rank_of(lg, cid)
                    if l in (24,):   # clean->clean control at the key layer, answer slot
                        patch.install(l, hs_cln[l + 1].to(model.device), pos_answer)
                        lg = model(input_ids=ids).logits[0, -1].float()
                        patch.remove()
                        res["ctrl_clean_to_clean"][str(l)] = rank_of(lg, cid)
                rows.append(res)
            print(f"  [patch] {i+1:2d}/50 {name:10s}  named_src so far "
                  f"{sum(r['named_src'] for r in rows)}/{len(rows)}  {time.time()-t0:5.0f}s", flush=True)
            if (i + 1) % 5 == 0:
                with open(f"{NB}/patch_sweep.jsonl", "w") as f:
                    for r in rows: f.write(json.dumps(r) + "\n")
                nbvol.commit()
    with open(f"{NB}/patch_sweep.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    nbvol.commit()
    print(f"[patch] DONE {len(rows)} trials", flush=True)


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=3 * 3600)
def necessity_sweep():
    """The complement of patch_sweep: NECESSITY rather than sufficiency.

    Sufficiency asked: does the source's residual at (l, answer slot), dropped into a
    clean run, produce the name?  It did not until layer ~32.
    Necessity asks: in the SOURCE run itself, if we overwrite the answer-slot residual
    at layer l with the CLEAN run's value (the reverse patch), does the name go away?
    A position can be necessary without being sufficient: the concept may be assembled
    from that residual plus continued attention to the injected user tokens.

    Also tests the user-turn span as a block (overwrite ALL injected user positions at
    layer l with clean values): if naming dies, late layers are still reading the
    injection from the user turn, which is the 'reinforcement' hypothesis.
    """
    import json, time
    import torch
    P = Pipeline(BASE_A, LENS, ELICIT2)
    model = P.model
    LAYERS = [16, 20, 22, 23, 24, 25, 26, 28, 30, 32]
    PREFILL = "INJECTED: yes\nCONCEPT:"

    class Patch:
        def __init__(self): self.handles = []
        def install(self, l, src_h, pos):
            self.remove()
            def hook(mod, args):
                h = args[0].clone(); h[:, pos, :] = src_h[pos, :].to(h.dtype); return (h,) + args[1:]
            def norm_hook(mod, args):
                h = args[0].clone(); h[:, pos, :] = src_h[pos, :].to(h.dtype); return (h,)
            if l + 1 < len(model.model.layers):
                self.handles.append(model.model.layers[l + 1].register_forward_pre_hook(hook))
            else:
                self.handles.append(model.model.norm.register_forward_pre_hook(norm_hook))
        def remove(self):
            for hd in self.handles: hd.remove()
            self.handles = []
    patch = Patch()
    rank_of = lambda lg, cid: (lg > lg[cid]).sum().item()

    rows, t0 = [], time.time()
    with torch.no_grad():
        for i, name in enumerate(P.CONCEPTS):
            cid = P.CID[name]
            for pi in range(10):
                ids, span, read_pos = P.build_trial(PASSAGES[pi], PREFILL)
                T = ids.shape[1]
                pos_answer = [T - 1]
                pos_user = list(range(span.start, span.stop))

                # clean residuals (what we will overwrite WITH)
                o_cln = model(input_ids=ids, output_hidden_states=True)
                hs_cln = [h[0].float() for h in o_cln.hidden_states]

                # source rank, no ablation
                P.clear_hooks(); P.inject(cid, 0.4, span)
                rank_src = rank_of(model(input_ids=ids).logits[0, -1].float(), cid)
                P.clear_hooks()

                res = dict(concept=name, category=P.CAT[name], passage=pi,
                           rank_src=rank_src, named_src=rank_src < 5,
                           ablate_answer={}, ablate_user_span={})
                for l in LAYERS:
                    for key, pos in (("ablate_answer", pos_answer), ("ablate_user_span", pos_user)):
                        # injection hooks on blocks 12-14 AND the reverse patch on block l+1
                        P.clear_hooks(); P.inject(cid, 0.4, span)
                        patch.install(l, hs_cln[l + 1], pos)
                        lg = model(input_ids=ids).logits[0, -1].float()
                        patch.remove(); P.clear_hooks()
                        res[key][str(l)] = rank_of(lg, cid)
                rows.append(res)
            print(f"  [nec] {i+1:2d}/50 {name:10s} named_src {sum(r['named_src'] for r in rows)}/{len(rows)} {time.time()-t0:5.0f}s", flush=True)
            if (i + 1) % 5 == 0:
                with open(f"{NB}/necessity_sweep.jsonl", "w") as f:
                    for r in rows: f.write(json.dumps(r) + "\n")
                nbvol.commit()
    with open(f"{NB}/necessity_sweep.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    nbvol.commit()
    print(f"[nec] DONE {len(rows)}", flush=True)


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=2 * 3600)
def rank_one_pilot():
    """Content-necessity, not position-necessity.

    Full-residual replacement at the answer slot kills naming 94-100% from L23, but that
    deletes the whole position: concept and answer-formation state together. Here we
    remove ONLY the concept's lens direction, rank-one, and leave the rest intact.

    At lens layer l (hidden_states[l+1], pre-hook on block l+1), answer slot, in the
    injected run:
      full      h <- h_clean                                   (reference, as before)
      proj0     h <- h - <h, v> v                              (project v_c out to zero)
      projclean h <- h - (<h, v> - <h_clean, v>) v             (set projection to clean value)
      projrand  h <- h - <h, u> u,  u = random J-space dir     (null: a different direction)
    where v = unit lens direction J_l^T (g * W_U[c]) at that layer.
    Records rank of c and log P(c) at the CONCEPT slot after each, so the importance
    label (delta log-prob) can be read off directly.
    """
    import json, random, time
    import torch
    P = Pipeline(BASE_A, LENS, ELICIT2)
    model = P.model
    LAYERS = [22, 23, 24, 26]
    PREFILL = "INJECTED: yes\nCONCEPT:"
    battery = set(P.CID.values())

    def lens_dir(cid, l):
        v = P.J[l].float().T @ (P.G.float() * P.W_U[cid].float()); return v / v.norm()

    def rand_dir(seed, l):
        rng = random.Random(seed)
        while True:
            t = rng.randrange(2000, 150000); s = P.tok.decode([t])
            if t not in battery and s.startswith(" ") and s.strip().isalpha() and len(s.strip()) > 2:
                return lens_dir(t, l)

    class Edit:
        def __init__(self): self.h = []
        def install(self, l, fn):
            self.remove()
            def hook(mod, args):
                h = args[0].clone(); h[:, -1, :] = fn(h[:, -1, :].float()).to(h.dtype); return (h,) + args[1:]
            self.h.append(model.model.layers[l + 1].register_forward_pre_hook(hook))
        def remove(self):
            for x in self.h: x.remove()
            self.h = []
    ed = Edit()

    def score(lg, cid):
        lp = torch.log_softmax(lg, -1)
        return (lg > lg[cid]).sum().item(), lp[cid].item()

    rows, t0 = [], time.time()
    with torch.no_grad():
        for i, name in enumerate(P.CONCEPTS):
            cid = P.CID[name]
            for pi in range(10):
                ids, span, _ = P.build_trial(PASSAGES[pi], PREFILL)
                o_c = model(input_ids=ids, output_hidden_states=True)
                hs_c = {l: o_c.hidden_states[l + 1][0, -1].float() for l in LAYERS}
                P.clear_hooks(); P.inject(cid, 0.4, span)
                o_s = model(input_ids=ids, output_hidden_states=True)
                hs_s = {l: o_s.hidden_states[l + 1][0, -1].float() for l in LAYERS}
                P.clear_hooks()
                r_src, lp_src = score(o_s.logits[0, -1].float(), cid)
                res = dict(concept=name, category=P.CAT[name], passage=pi,
                           rank_src=r_src, logp_src=lp_src, named_src=r_src < 5, layers={})
                for l in LAYERS:
                    v = lens_dir(cid, l); u = rand_dir(i * 1000 + pi, l)
                    hc = hs_c[l]; proj_c = torch.dot(hc, v).item(); proj_s = torch.dot(hs_s[l], v).item()
                    edits = {
                        "full":      lambda h: hc.unsqueeze(0).expand_as(h),
                        "proj0":     lambda h: h - (h @ v).unsqueeze(-1) * v,
                        "projclean": lambda h: h - ((h @ v) - proj_c).unsqueeze(-1) * v,
                        "projrand":  lambda h: h - (h @ u).unsqueeze(-1) * u,
                    }
                    out = {"proj_src": proj_s, "proj_clean": proj_c}
                    for k, fn in edits.items():
                        P.clear_hooks(); P.inject(cid, 0.4, span); ed.install(l, fn)
                        r, lp = score(model(input_ids=ids).logits[0, -1].float(), cid)
                        ed.remove(); P.clear_hooks()
                        out[k] = {"rank": r, "logp": lp}
                    res["layers"][str(l)] = out
                rows.append(res)
            print(f"  [r1] {i+1:2d}/50 {name:10s} named {sum(r['named_src'] for r in rows)}/{len(rows)} {time.time()-t0:5.0f}s", flush=True)
            if (i + 1) % 10 == 0:
                with open(f"{NB}/rank_one_pilot.jsonl", "w") as f:
                    for r in rows: f.write(json.dumps(r) + "\n")
                nbvol.commit()
    with open(f"{NB}/rank_one_pilot.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    nbvol.commit()
    print(f"[r1] DONE {len(rows)}", flush=True)


# ---- Natural-presence prompt pool ------------------------------------------------------
# One two-hop description per concept: implies the concept, never names it, and avoids
# naming any OTHER battery concept. Drafted with Claude; the filter below decides which
# survive (concept readable at the answer slot, L23, with no injection).
NATURAL = {
    "France": "The republic whose capital lies on the Seine and whose revolution began in 1789.",
    "Japan": "The island nation whose capital is Tokyo and whose currency is the yen.",
    "Brazil": "The largest country in South America, whose official language is Portuguese.",
    "Egypt": "The country of the Nile, the pyramids at Giza, and the city of Cairo.",
    "Canada": "The country directly north of the United States whose capital is Ottawa.",
    "Germany": "The central European country whose capital is Berlin and whose currency was the mark.",
    "Norway": "The Scandinavian kingdom of fjords whose capital is Oslo.",
    "Chile": "The long narrow country along the Pacific coast of South America, capital Santiago.",
    "Poland": "The central European country whose capital is Warsaw, on the Vistula.",
    "Nepal": "The Himalayan country whose capital is Kathmandu, on the south side of Everest.",
    "spider": "The eight-legged creature that spins silk webs to trap flying insects.",
    "elephant": "The largest land animal, with a trunk, tusks, and large flapping ears.",
    "tiger": "The largest of the big cats, striped, native to the forests of Asia.",
    "whale": "The largest animal that has ever lived, a marine mammal that breathes through a blowhole.",
    "rabbit": "The long-eared burrowing mammal that hops and is often kept as a pet.",
    "snake": "The legless reptile that slithers and sheds its skin, some species venomous.",
    "eagle": "The large bird of prey with a hooked beak that appears on the United States seal.",
    "camel": "The humped desert animal used for transport across the Sahara.",
    "frog": "The amphibian that begins life as a tadpole and catches insects with its tongue.",
    "owl": "The nocturnal bird of prey that hoots and can turn its head almost all the way round.",
    "bread": "The staple baked from flour, water, and yeast, sliced for sandwiches.",
    "cheese": "The dairy product made by curdling milk, aged in wheels and served on a board with crackers.",
    "pasta": "The Italian staple made from durum wheat, boiled and served with sauce.",
    "soup": "The liquid dish of stock and vegetables, served hot in a bowl with a spoon.",
    "chocolate": "The sweet made from roasted cacao beans, sold in bars and melted for cakes.",
    "honey": "The sweet substance that bees make from nectar and store in wax combs.",
    "butter": "The dairy fat churned from cream, spread on toast and used in baking.",
    "garlic": "The pungent bulb of cloves crushed into sauces and said to repel vampires.",
    "noodles": "The long thin strands of dough boiled and served in Asian broths and stir-fries.",
    "mango": "The tropical stone fruit with sweet fibrous flesh, eaten ripe or pickled green in South Asia.",
    "purple": "The colour of amethyst and of royal robes, between red and blue on the spectrum.",
    "yellow": "The colour of lemons, ripe bananas, and the centre of a daisy.",
    "pink": "The pale colour of flamingos, cherry blossom, and cooked salmon.",
    "brown": "The colour of soil, tree bark, and roasted coffee beans.",
    "grey": "The colour of ash, overcast skies, and unpainted concrete.",
    "violet": "The colour at the shortest visible wavelength, just past indigo, named for a small flower.",
    "beige": "The pale sandy colour of unbleached wool and neutral office walls.",
    "teal": "The dark blue-green colour named after the stripe on a small duck.",
    "navy": "The very dark blue of naval uniforms and formal blazers.",
    "crimson": "The deep red of arterial blood and Harvard's official colour.",
    "anger": "The hot emotion of being wronged, with clenched fists and a raised voice.",
    "sadness": "The low emotion after a loss, with tears and a heavy quiet.",
    "guilt": "The gnawing feeling of having done something wrong to another person.",
    "shame": "The feeling of wanting to hide after being exposed as inadequate.",
    "disgust": "The recoiling feeling at something rotten, with a wrinkled nose.",
    "envy": "The bitter wish to have what another person has.",
    "grief": "The long sorrow that follows the death of someone loved.",
    "boredom": "The restless dullness of having nothing engaging to do for hours.",
    "dread": "The sinking anticipation of something bad that is certain to come.",
    "despair": "The complete loss of hope that anything will get better.",
}
ELICIT_CONTENT = (
    "\n\nAbove is a short passage. Introspect on what you are representing as you read it.\n"
    "Answer on exactly two lines. The first line is PRESENT: followed by yes or no, saying "
    "whether a single specific concept stands out. The second line is CONCEPT: followed by "
    "a single lowercase noun naming it, or none."
)


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=3600)
def natural_filter():
    """Which natural-presence prompts put their concept at the answer slot, L23, unaided?
    Two wordings: Phase 1's injection elicitation (for comparability) and a content
    question (what Phase 2 will actually ask). No injection anywhere. Also records every
    other battery concept's rank so leaks are visible."""
    import json
    import torch
    P = Pipeline(BASE_A, LENS, ELICIT2)
    rows = []
    with torch.no_grad():
        for name, passage in NATURAL.items():
            cid = P.CID[name]
            for wording, elicit, prefill in (("injection", ELICIT2, "INJECTED: yes\nCONCEPT:"),
                                             ("content", ELICIT_CONTENT, "PRESENT: yes\nCONCEPT:")):
                P.elicit = elicit
                ids, span, read_pos = P.build_trial(passage, prefill)
                o = P.model(input_ids=ids, output_hidden_states=True)
                lg = o.logits[0, -1].float()
                def lens_at(l, pos):
                    h = o.hidden_states[l + 1][0, pos].float()
                    return (P.W_U @ P.NORM(P.J[l].float() @ h).to(P.W_U.dtype)).float()
                l23 = lens_at(23, -1); l24 = lens_at(24, -1); u22 = lens_at(22, read_pos)
                rk = lambda v, c: (v > v[c]).sum().item()
                rows.append(dict(concept=name, category=P.CAT[name], wording=wording, passage=passage,
                                 n_tokens=len(P.tok.encode(passage)),
                                 lens_answer_L23=rk(l23, cid), lens_answer_L24=rk(l24, cid),
                                 lens_user_L22=rk(u22, cid), rank_c=rk(lg, cid),
                                 top5_slot=[P.tok.decode([k]) for k in lg.topk(5).indices.tolist()],
                                 top5_lens23=[P.tok.decode([k]) for k in l23.topk(5).indices.tolist()],
                                 other_battery_L23={w: rk(l23, c2) for w, c2 in P.CID.items() if w != name and rk(l23, c2) <= 25}))
            print(f"  [nat] {name:10s} inj L23={rows[-2]['lens_answer_L23']:6d} slot={rows[-2]['rank_c']:5d} | "
                  f"content L23={rows[-1]['lens_answer_L23']:6d} slot={rows[-1]['rank_c']:5d}", flush=True)
    P.elicit = ELICIT2
    with open(f"{NB}/natural_filter.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    nbvol.commit()
    print(f"[nat] DONE {len(rows)}", flush=True)


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=2 * 3600)
def labels_neutral():
    """Phase 2 label generation for the INJECTED stratum under the single neutral
    elicitation (ELICIT_CONTENT), answer slot, lens layer 24. Also gamma 0 (absent
    labels) and gamma 0.2 (the free dose-response prediction).

    Per (concept, passage, gamma):
      lens_L24         lens rank of c at answer slot, hidden_states[25], last position   <- the label
      rank_c           rank of c in the CONCEPT slot                                     <- naming
      p_present        forced choice yes/no after 'PRESENT:'
      proj_src         <h_src, v_c>  at L24 answer slot
      proj_clean       <h_clean, v_c> at L24 answer slot  (the covariate the mentor asked for)
    Clean residuals are computed once per passage and reused across concepts.
    """
    import json, time
    import torch
    P = Pipeline(BASE_A, LENS, ELICIT_CONTENT)
    model = P.model
    L = 24
    PRE_YN, PRE_C = "PRESENT:", "PRESENT: yes\nCONCEPT:"
    vdir = {w: (lambda v: v / v.norm())(P.J[L].float().T @ (P.G.float() * P.W_U[c].float())) for w, c in P.CID.items()}

    def lens_rank_at(h, cid):
        lg = P.W_U @ P.NORM(P.J[L].float() @ h).to(P.W_U.dtype)
        return (lg > lg[cid]).sum().item()

    rows, t0 = [], time.time()
    with torch.no_grad():
        for pi, passage in enumerate(PASSAGES):
            ids_c, span, _ = P.build_trial(passage, PRE_C)
            ids_y, _, _ = P.build_trial(passage, PRE_YN)
            o_cln = model(input_ids=ids_c, output_hidden_states=True)
            h_cln = o_cln.hidden_states[L + 1][0, -1].float()
            lg_cln_c = o_cln.logits[0, -1].float()
            pr_cln_y = P.answer(model(input_ids=ids_y).logits[0, -1].float(), ("yes", "no"))
            for name, cid in P.CID.items():
                v = vdir[name]
                # gamma 0: clean, shared forward passes
                rows.append(dict(concept=name, category=P.CAT[name], passage=pi, gamma=0.0,
                                 lens_L24=lens_rank_at(h_cln, cid), rank_c=(lg_cln_c > lg_cln_c[cid]).sum().item(),
                                 p_present=pr_cln_y["p_yes"], mass=pr_cln_y["mass"],
                                 proj_src=torch.dot(h_cln, v).item(), proj_clean=torch.dot(h_cln, v).item()))
                for g in (0.2, 0.4):
                    P.clear_hooks(); P.inject(cid, g, span)
                    o = model(input_ids=ids_c, output_hidden_states=True)
                    h = o.hidden_states[L + 1][0, -1].float(); lg = o.logits[0, -1].float()
                    pr = P.answer(model(input_ids=ids_y).logits[0, -1].float(), ("yes", "no"))
                    P.clear_hooks()
                    rows.append(dict(concept=name, category=P.CAT[name], passage=pi, gamma=g,
                                     lens_L24=lens_rank_at(h, cid), rank_c=(lg > lg[cid]).sum().item(),
                                     p_present=pr["p_yes"], mass=pr["mass"],
                                     proj_src=torch.dot(h, v).item(), proj_clean=torch.dot(h_cln, v).item()))
            print(f"  [lab] passage {pi+1:2d}/20  {time.time()-t0:5.0f}s", flush=True)
            with open(f"{NB}/labels_neutral.jsonl", "w") as f:
                for r in rows: f.write(json.dumps(r) + "\n")
            nbvol.commit()
    print(f"[lab] DONE {len(rows)}", flush=True)


# ===== Phase 2 natural stratum: scaled pool, neutral wording, L24, with covariate ============
NAT_PRESENT, NAT_ABSENT = 100, 1000     # frozen natural-stratum margins
INJ_PRESENT, INJ_ABSENT = 10, 500       # frozen injected-stratum margins


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=2 * 3600)
def natural_filter_v2():
    """Filter the scaled natural pool (natural_pool.py, ~6 descriptions per concept) under the
    single neutral elicitation, reading the label at the answer slot, L24. Records the clean
    projection covariate and every other battery concept within rank 25 (leak check).
    Output rows carry the frozen natural-stratum label: present / middle / absent."""
    import json, time
    import torch
    from natural_pool import NATURAL_POOL
    P = Pipeline(BASE_A, LENS, ELICIT_CONTENT)
    model = P.model
    L = 24
    vdir = {w: (lambda v: v / v.norm())(P.J[L].float().T @ (P.G.float() * P.W_U[c].float())) for w, c in P.CID.items()}
    rows, t0 = [], time.time()
    with torch.no_grad():
        for name, descs in NATURAL_POOL.items():
            cid = P.CID[name]
            for di, passage in enumerate(descs):
                ids, span, _ = P.build_trial(passage, "PRESENT: yes\nCONCEPT:")
                o = model(input_ids=ids, output_hidden_states=True)
                h = o.hidden_states[L + 1][0, -1].float(); lg = o.logits[0, -1].float()
                lens = (P.W_U @ P.NORM(P.J[L].float() @ h).to(P.W_U.dtype)).float()
                rk = lambda v, c: (v > v[c]).sum().item()
                r_lens = rk(lens, cid)
                ids_y, _, _ = P.build_trial(passage, "PRESENT:")
                pr = P.answer(model(input_ids=ids_y).logits[0, -1].float(), ("yes", "no"))
                rows.append(dict(concept=name, category=P.CAT[name], desc_idx=di, passage=passage,
                                 n_tokens=len(P.tok.encode(passage)),
                                 lens_L24=r_lens, rank_c=rk(lg, cid), p_present=pr["p_yes"], mass=pr["mass"],
                                 proj_clean=torch.dot(h, vdir[name]).item(),
                                 label=("present" if r_lens <= NAT_PRESENT else "absent" if r_lens >= NAT_ABSENT else "middle"),
                                 top5_slot=[P.tok.decode([k]) for k in lg.topk(5).indices.tolist()],
                                 leaks={w: rk(lens, c2) for w, c2 in P.CID.items() if w != name and rk(lens, c2) <= 25}))
            n_pres = sum(r["label"] == "present" for r in rows if r["concept"] == name)
            print(f"  [nat2] {name:10s} present {n_pres}/{len(descs)}  {time.time()-t0:4.0f}s", flush=True)
    with open(f"{NB}/natural_pool_labels.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    nbvol.commit()
    print(f"[nat2] DONE {len(rows)}  present {sum(r['label']=='present' for r in rows)}  "
          f"absent {sum(r['label']=='absent' for r in rows)}  middle {sum(r['label']=='middle' for r in rows)}", flush=True)


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=2 * 3600)
def importance_pilot_natural():
    """Validation pilot for the importance rung (H5), on the NATURAL stratum only.
    For every natural prompt labelled present (lens <= 100 at L24), apply the clean-value
    rank-one ablation of v_c at the answer slot, L24, and record delta log P(c) and delta rank.
    'Clean value' here is the projection of a matched neutral passage's residual, since a
    natural prompt has no injected/clean pair: we use the median clean projection for that
    concept over the 20 Phase 1 passages (from labels_neutral.jsonl), and also report the
    project-to-zero variant. The rung survives only if delta log P(c) has real variance across
    prompts and AUC against naming is well above 0.5."""
    import json, time
    import numpy as np
    import torch
    P = Pipeline(BASE_A, LENS, ELICIT_CONTENT)
    model = P.model
    L = 24
    pool = [json.loads(l) for l in open(f"{NB}/natural_pool_labels.jsonl")]
    present = [r for r in pool if r["label"] == "present"]
    lab = [json.loads(l) for l in open(f"{NB}/labels_neutral.jsonl")]
    clean_proj = {}
    for w in P.CID:
        xs = [r["proj_clean"] for r in lab if r["concept"] == w and r["gamma"] == 0.0]
        clean_proj[w] = float(np.median(xs)) if xs else 0.0
    vdir = {w: (lambda v: v / v.norm())(P.J[L].float().T @ (P.G.float() * P.W_U[c].float())) for w, c in P.CID.items()}

    class Edit:
        def __init__(self): self.h = []
        def install(self, fn):
            self.remove()
            def hook(mod, args):
                x = args[0].clone(); x[:, -1, :] = fn(x[:, -1, :].float()).to(x.dtype); return (x,) + args[1:]
            self.h.append(model.model.layers[L + 1].register_forward_pre_hook(hook))
        def remove(self):
            for k in self.h: k.remove()
            self.h = []
    ed = Edit()
    def score(lg, cid):
        return (lg > lg[cid]).sum().item(), torch.log_softmax(lg, -1)[cid].item()

    rows, t0 = [], time.time()
    with torch.no_grad():
        for i, r in enumerate(present):
            name, cid, v = r["concept"], P.CID[r["concept"]], vdir[r["concept"]]
            pc = clean_proj[name]
            ids, _, _ = P.build_trial(r["passage"], "PRESENT: yes\nCONCEPT:")
            r0, lp0 = score(model(input_ids=ids).logits[0, -1].float(), cid)
            out = dict(concept=name, category=r["category"], desc_idx=r["desc_idx"], lens_L24=r["lens_L24"],
                       rank_base=r0, logp_base=lp0, named_base=r0 < 5, clean_proj_used=pc, proj_natural=r["proj_clean"])
            for key, fn in (("projclean", lambda h: h - ((h @ v) - pc).unsqueeze(-1) * v),
                            ("proj0", lambda h: h - (h @ v).unsqueeze(-1) * v)):
                ed.install(fn); rk, lp = score(model(input_ids=ids).logits[0, -1].float(), cid); ed.remove()
                out[key] = {"rank": rk, "logp": lp, "dlogp": lp - lp0}
            rows.append(out)
            if (i + 1) % 25 == 0:
                print(f"  [imp] {i+1}/{len(present)}  {time.time()-t0:4.0f}s", flush=True)
    with open(f"{NB}/importance_natural.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    nbvol.commit()
    d = np.array([r["projclean"]["dlogp"] for r in rows]); nm = np.array([r["named_base"] for r in rows])
    print(f"[imp] DONE {len(rows)} present prompts; dlogp median {np.median(d):+.2f} sd {d.std():.2f}; named {nm.mean():.2f}", flush=True)


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=3600)
def natural_layer_sweep():
    """Diagnostic for the natural-stratum margin. The scaled pool put only 58/300 prompts at
    lens rank <= 100 at L24 while the model names the concept unaided on 127/300, and 43 of
    the prompts labelled ABSENT (>= 1000) are named unaided. Is there a layer at the answer
    slot, below the circular band, where natural presence is readable? Records the lens rank
    of the target at layers 20..32, answer slot, neutral wording, no injection, plus the
    logit lens at the same layers for comparison."""
    import json, time
    import torch
    from natural_pool import NATURAL_POOL
    P = Pipeline(BASE_A, LENS, ELICIT_CONTENT)
    model = P.model
    LAYERS = list(range(20, 33))
    rows, t0 = [], time.time()
    with torch.no_grad():
        for name, descs in NATURAL_POOL.items():
            cid = P.CID[name]
            for di, passage in enumerate(descs):
                ids, _, _ = P.build_trial(passage, "PRESENT: yes\nCONCEPT:")
                o = model(input_ids=ids, output_hidden_states=True)
                lg = o.logits[0, -1].float()
                jl, ll = {}, {}
                for l in LAYERS:
                    h = o.hidden_states[l + 1][0, -1].float()
                    v = P.W_U @ P.NORM(P.J[l].float() @ h).to(P.W_U.dtype)
                    w = P.W_U @ P.NORM(h).to(P.W_U.dtype)
                    jl[str(l)] = (v > v[cid]).sum().item(); ll[str(l)] = (w > w[cid]).sum().item()
                rows.append(dict(concept=name, category=P.CAT[name], desc_idx=di,
                                 rank_c=(lg > lg[cid]).sum().item(), jlens=jl, logitlens=ll))
    with open(f"{NB}/natural_layer_sweep.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    nbvol.commit()
    print(f"[nls] DONE {len(rows)} {time.time()-t0:.0f}s", flush=True)


NAT_LAYERS = (25, 26)   # two-layer natural labels: agreement required


def _two_layer_label(a, b, pres=NAT_PRESENT, absn=NAT_ABSENT):
    if a <= pres and b <= pres: return "present"
    if a >= absn and b >= absn: return "absent"
    return "discard"


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=2 * 3600)
def labels_neutral_multi():
    """Injected-stratum labels at L24, L25, L26 (same design as labels_neutral) so the
    mentor can choose between L24-only and a uniform two-layer scheme across strata."""
    import json, time
    import torch
    P = Pipeline(BASE_A, LENS, ELICIT_CONTENT)
    model = P.model
    LAYERS = (24, 25, 26)
    vdir = {l: P.concept_dirs(l) for l in LAYERS}
    torch.cuda.empty_cache()
    def lens_rank(h, l, cid):
        lg = P.W_U @ P.NORM(P.J[l].float() @ h).to(P.W_U.dtype); return (lg > lg[cid]).sum().item()
    rows, t0 = [], time.time()
    with torch.no_grad():
        for pi, passage in enumerate(PASSAGES):
            ids, span, _ = P.build_trial(passage, "PRESENT: yes\nCONCEPT:")
            o_c = model(input_ids=ids, output_hidden_states=True)
            hc = {l: o_c.hidden_states[l + 1][0, -1].float() for l in LAYERS}
            lgc = o_c.logits[0, -1].float()
            for name, cid in P.CID.items():
                rows.append(dict(concept=name, category=P.CAT[name], passage=pi, gamma=0.0,
                                 rank_c=(lgc > lgc[cid]).sum().item(),
                                 lens={str(l): lens_rank(hc[l], l, cid) for l in LAYERS},
                                 proj_clean={str(l): torch.dot(hc[l], vdir[l][name]).item() for l in LAYERS}))
                for g in (0.2, 0.4):
                    P.clear_hooks(); P.inject(cid, g, span)
                    o = model(input_ids=ids, output_hidden_states=True); P.clear_hooks()
                    lg = o.logits[0, -1].float()
                    rows.append(dict(concept=name, category=P.CAT[name], passage=pi, gamma=g,
                                     rank_c=(lg > lg[cid]).sum().item(),
                                     lens={str(l): lens_rank(o.hidden_states[l + 1][0, -1].float(), l, cid) for l in LAYERS},
                                     proj_clean={str(l): torch.dot(hc[l], vdir[l][name]).item() for l in LAYERS}))
            print(f"  [labm] passage {pi+1:2d}/20 {time.time()-t0:4.0f}s", flush=True)
    with open(f"{NB}/labels_neutral_multi.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    nbvol.commit(); print(f"[labm] DONE {len(rows)}", flush=True)


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=2 * 3600)
def importance_pilot_2layer():
    """H5 validation on the two-layer natural present set (L25 and L26 agree, <=100).
    Clean-value rank-one ablation of v_c at the answer slot at L25 and at L26 separately.
    Clean projection per (concept, layer) = median over the 20 Phase 1 passages, computed here."""
    import json, time
    import numpy as np
    import torch
    P = Pipeline(BASE_A, LENS, ELICIT_CONTENT)
    model = P.model
    sweep = [json.loads(l) for l in open(f"{NB}/natural_layer_sweep.jsonl")]
    from natural_pool import NATURAL_POOL
    present = [r for r in sweep if _two_layer_label(r["jlens"]["25"], r["jlens"]["26"]) == "present"]
    vdir = {l: P.concept_dirs(l) for l in NAT_LAYERS}
    torch.cuda.empty_cache()
    # clean projections at L25/L26 from the 20 neutral passages
    cp = {l: {w: [] for w in P.CID} for l in NAT_LAYERS}
    with torch.no_grad():
        for passage in PASSAGES:
            ids, _, _ = P.build_trial(passage, "PRESENT: yes\nCONCEPT:")
            o = model(input_ids=ids, output_hidden_states=True)
            for l in NAT_LAYERS:
                h = o.hidden_states[l + 1][0, -1].float()
                for w in P.CID: cp[l][w].append(torch.dot(h, vdir[l][w]).item())
    cp = {l: {w: float(np.median(v)) for w, v in d.items()} for l, d in cp.items()}
    class Edit:
        def __init__(self): self.h = []
        def install(self, l, fn):
            self.remove()
            def hook(mod, args):
                x = args[0].clone(); x[:, -1, :] = fn(x[:, -1, :].float()).to(x.dtype); return (x,) + args[1:]
            self.h.append(model.model.layers[l + 1].register_forward_pre_hook(hook))
        def remove(self):
            for k in self.h: k.remove()
            self.h = []
    ed = Edit()
    score = lambda lg, cid: ((lg > lg[cid]).sum().item(), torch.log_softmax(lg, -1)[cid].item())
    rows, t0 = [], time.time()
    with torch.no_grad():
        for i, r in enumerate(present):
            name, cid = r["concept"], P.CID[r["concept"]]
            passage = NATURAL_POOL[name][r["desc_idx"]]
            ids, _, _ = P.build_trial(passage, "PRESENT: yes\nCONCEPT:")
            r0, lp0 = score(model(input_ids=ids).logits[0, -1].float(), cid)
            out = dict(concept=name, category=r["category"], desc_idx=r["desc_idx"], lens_L25=r["jlens"]["25"], lens_L26=r["jlens"]["26"],
                       rank_base=r0, logp_base=lp0, named_base=r0 < 5)
            for l in NAT_LAYERS:
                v, pc = vdir[l][name], cp[l][name]
                ed.install(l, lambda h, v=v, pc=pc: h - ((h @ v) - pc).unsqueeze(-1) * v)
                rk, lp = score(model(input_ids=ids).logits[0, -1].float(), cid); ed.remove()
                out[f"L{l}"] = {"rank": rk, "logp": lp, "dlogp": lp - lp0, "clean_proj": pc}
            rows.append(out)
    with open(f"{NB}/importance_2layer.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    nbvol.commit()
    for l in NAT_LAYERS:
        d = np.array([r[f"L{l}"]["dlogp"] for r in rows]); print(f"[imp2] L{l}: n={len(rows)} dlogp median {np.median(d):+.2f} sd {d.std():.2f}", flush=True)
    print(f"[imp2] DONE {len(rows)}", flush=True)


@app.function(image=image, gpu="L4", volumes={MOUNT: weights, NB: nbvol}, timeout=3600)
def natural_filter_r2():
    """Filter the round-two descriptions (NATURAL_POOL_R2, thin concepts only) under the
    two-layer rule: lens rank at L25 and L26, answer slot, neutral wording, no injection.
    Same fields as natural_filter_v2 plus both layers and the two-layer label."""
    import json, time
    import torch
    from natural_pool import NATURAL_POOL_R2
    P = Pipeline(BASE_A, LENS, ELICIT_CONTENT)
    model = P.model
    vdir = {l: P.concept_dirs(l) for l in NAT_LAYERS}
    torch.cuda.empty_cache()
    rows, t0 = [], time.time()
    with torch.no_grad():
        for name, descs in NATURAL_POOL_R2.items():
            cid = P.CID[name]
            for di, passage in enumerate(descs):
                ids, _, _ = P.build_trial(passage, "PRESENT: yes\nCONCEPT:")
                o = model(input_ids=ids, output_hidden_states=True)
                lg = o.logits[0, -1].float()
                rk = lambda v, c: (v > v[c]).sum().item()
                lens, proj = {}, {}
                for l in NAT_LAYERS:
                    h = o.hidden_states[l + 1][0, -1].float()
                    v = (P.W_U @ P.NORM(P.J[l].float() @ h).to(P.W_U.dtype)).float()
                    lens[str(l)] = rk(v, cid); proj[str(l)] = torch.dot(h, vdir[l][name]).item()
                    if l == 26:
                        leaks = {w: rk(v, c2) for w, c2 in P.CID.items() if w != name and rk(v, c2) <= 25}
                rows.append(dict(concept=name, category=P.CAT[name], round=2, desc_idx=di, passage=passage,
                                 n_tokens=len(P.tok.encode(passage)), lens=lens, proj_clean=proj,
                                 rank_c=rk(lg, cid), label=_two_layer_label(lens["25"], lens["26"]),
                                 top5_slot=[P.tok.decode([k]) for k in lg.topk(5).indices.tolist()], leaks=leaks))
            n_pres = sum(r["label"] == "present" for r in rows if r["concept"] == name)
            print(f"  [r2] {name:10s} present {n_pres}/{len(descs)}  named {sum(r['rank_c']<5 for r in rows if r['concept']==name)}/{len(descs)}  {time.time()-t0:4.0f}s", flush=True)
    with open(f"{NB}/natural_pool_r2_labels.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    nbvol.commit()
    print(f"[r2] DONE {len(rows)} present {sum(r['label']=='present' for r in rows)} absent {sum(r['label']=='absent' for r in rows)} discard {sum(r['label']=='discard' for r in rows)}", flush=True)
