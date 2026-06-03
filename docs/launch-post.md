# Rules, not weights: running Tsetlin Machines on your Mac's GPU — and feeding them molecules the smart way

Most of ML right now is "throw a big differentiable model at it." This post is about the
opposite instinct: a model that learns **plain if-then rules you can read**, why we got it
running fast on Apple Silicon, and — the fun part — how the way you *feed* it turns out to
matter far more than the model itself.

Everything's open source and `pip`-installable (macOS only): **[mlxTM](https://github.com/guillaume-osmo/mlxTM)**.

---

## The 2-minute Tsetlin Machine

Imagine a **jury of tiny detectives**. Each one knows one simple rule — *"if fragment A is
present AND fragment B is absent, I vote 'active'."* A **Tsetlin Machine (TM)** is exactly that:
a pile of these AND-rules ("clauses"), each casting a weighted vote. The class with the most
votes wins.

Three things an ML person should file away:

- **It only speaks in switches.** Every input must be a 0/1 bit. (Hold that thought — it's the
  whole second half of this post.)
- **Inference is basically a tiny binary neural net** — two matrix multiplies and a threshold.
  GPUs love it.
- **It's interpretable for free.** A trained TM *is* a readable rule set — closer to RuleFit
  than to a black box.

Each "should I use this feature in this rule?" decision is made by a little **learning automaton**
— picture a thermostat that nudges toward *include* or *exclude* as it sees more examples. No
gradients, no backprop. Just rules earning their place.

## Why it wouldn't run on a Mac (and how we fixed it)

The good Tsetlin libraries ([cair/tmu](https://github.com/cair/tmu),
[PyTsetlinMachineCUDA](https://github.com/cair/PyTsetlinMachineCUDA)) **only speak CUDA** — they
won't even compile on Apple Silicon. So we rewrote the whole thing in Apple's
[MLX](https://github.com/ml-explore/mlx) / Metal, from scratch.

`mlxTM` ships five flavors — a dense one, a bit-packed one (32 switches per machine word), a
weighted "coalesced" one, a **fully bit-packed trainer** (we taught a Metal kernel to *count in
binary with carries* — the GPU trick that makes training fast), and a scikit-learn-style wrapper.

```python
from mlx_tm import DenseTsetlinMachine
tm = DenseTsetlinMachine(n_clauses=20, T=15, s=3.9).fit(X_bits, y, epochs=30)
tm.predict(X_test)
```

How'd it do? It nails the classic **Noisy-XOR** sanity test (100%), and runs **~14× faster to
train, ~46× faster to predict** than a CPU baseline — all on the laptop's GPU. The bit-packed
trick gives another 6–13×. (We checked it bit-for-bit against a plain NumPy reference, so this
isn't hand-waving.)

## The real game: a rule-learner only eats switches

Here's the catch nobody warns you about. The TM is great, but it needs **bits**. Molecules come
as continuous descriptors or sparse fingerprints. So the actual problem becomes:

> **How do you turn a molecule into the *right* switches?**

Two sub-questions: **how** do you switch-ify a number, and **which** features do you keep? We
benchmarked both, honestly, under proper cross-validation (no hyperparameter-tuning magic).

### Switch-ifying a number (binarization)

The simplest trick is a **thermometer**: picture a rising volume bar. A value of 0.7 lights up the
"> 0.2", "> 0.4", "> 0.6" notches but not "> 0.8". How many notches? **4–5 is plenty; 8 is a waste**
(the data literally got *slightly worse* at 8). Good rule of thumb to keep.

We also tried fancier encoders — and stole one straight from the **LLM-quantization** playbook.
Methods like [QuaRot](https://arxiv.org/abs/2404.00456) and QuIP get LLMs down to 2–4 bits by
**rotating** the data first — spin the space so the outliers flatten out, *then* snap to bits.
(Think: rotating a messy room so everything lines up before you photograph it in low resolution.)
It's a fantastic **compression** lever (same accuracy at half the bits) — but, honestly, **not an
accuracy lever**. Thermometer-at-4-bits is a strong, boring default.

### Picking features without peeking (RPCholesky)

When you have thousands of features, you select. We use **RPCholesky** — think of it as assembling
a **diverse committee** that doesn't keep repeating itself. The key property: **it never looks at
the labels.** No peeking at the answers means no leakage, and you can compute the selection *once*
and reuse it for every task. (ML translation: a label-free, Nyström-style diverse-subset picker.)

## molFTP: features that already think like a rule-learner

This is where it gets nice. A normal fingerprint (ECFP) asks a yes/no question: *"is fragment X
present?"* [molFTP](https://github.com/osmoai/molftp) asks a richer one: *"how strongly does
fragment X lean toward active vs inactive?"* — a little **significance score** per fragment.

Notice that's **exactly what the TM is already voting on.** molFTP scores fragments by
significance; the TM builds rules out of significant fragments and votes. They're the same idea at
two levels — so molFTP is a *natural* front-end for a TM, not a bolt-on. (Getting molFTP's C++ to
load on a Mac was its own yak-shave — re-signing the library and redirecting it to the right
graphics/RDKit dylibs — but it runs.)

And it's efficient: molFTP's compact **27-number** summary already rivals a 2048-bit ECFP
fingerprint.

### The 20,000-fragment problem (and the oldest trick in the book)

Go past the summary and molFTP hands you the *full* signal: **28,000–66,000** fragment scores. Way
too many switches for the rule-learner. So how do you shrink it?

- **RPCholesky? No.** It's a net for a dense lake; these are sparse needles, and the informative
  fragments are *rare* — exactly what it throws away. It didn't just underperform, it **collapsed**
  (one target dropped to near-coin-flip).
- **The hashing trick? Yes.** This is just **ECFP folding** in disguise: stuff 66k fragments into
  ~8k buckets and let collisions wash out. **No measurable signal lost**, and it *beats* plain ECFP.

That gives a clean, memorable rule:

> **Net for dense lakes, magnet for sparse needles** — RPCholesky for continuous descriptors,
> hashing/significance for sparse fragment keys. Use the wrong one and you *lose*.

## A second front-end: bond-centered fingerprints, Sort&Slice, and OOV

The baseline everyone reaches for is a **folded** 2048-bit ECFP. But folding glues unrelated
fragments onto the same switch (hash collisions) — and a collided, ambiguous switch is *noise* to a
rule-learner. [bcfp](https://github.com/osmoai/bcfp) offers two fixes the TM cares about:

- **BCFP** — the same Morgan idea, but **bond-centered** (*"is this bond-environment present?"*)
  instead of atom-centered. A complementary view; concatenating ECFP+BCFP is the paper's
  combined representation.
- **Sort&Slice + OOV** — instead of folding, **keep the top-K most frequent training fragments as
  their own clean switches**, and add a single **out-of-vocabulary** bucket that catches test-set
  fragments never seen in training. It's a label-free, frequency-ranked selection (a cousin of the
  RPCholesky idea, but for sparse bits) with a built-in distribution-shift safety net.

Why a TM should *prefer* this: a folded fingerprint hands the rule-learner thousands of collided,
ambiguous literals; Sort&Slice hands it a few hundred clean, meaningful ones. Fewer, better switches
→ tighter rules. (We found a small RDKit-2026 bug in bcfp's presence path while wiring this up, and
fixed it upstream — see the bcfp repo.)

## "Wait — is it cheating?"

molFTP uses the labels to score fragments, so the natural worry is leakage. The clean way to
settle it (every ML person knows this one): **shuffle the labels and rerun the whole pipeline.**
If the model still "works," it was cheating. Ours dropped to **0.48 AUC — pure coin-flip.** Honest.
(We fit molFTP *per fold* on training labels only; the held-out fold never sees its own answers.)

## So what do you actually get?

A **molFTP → Tsetlin Machine** pipeline that:

- **matches ECFP** on the molecular benchmarks (sometimes beats it),
- is **tiny and interpretable** — every rule reads like *"has significant-active fragment A AND not
  significant-inactive fragment B → active,"*
- is **leakage-proven**, and
- runs **on your Mac's GPU.**

And the honest meta-lesson: on these tasks, **simple baselines are hard to beat**. The fancy
machinery (rotation binarization, RPCholesky, huge descriptor sets) pays off in **speed,
interpretability, leakage-safety, and Apple-GPU support** — not in a magic accuracy jump. That's a
more useful truth than another "SOTA" claim.

## 📊 Show me the numbers

We kept the prose light, so here are the receipts — cross-validated **ROC-AUC** on two
opioid-target datasets (MDR1, MOR) from the **TM-QSAR-Benchmark**
([code](https://github.com/PaulC61/TM-QSAR-Benchmark),
[paper](https://pubs.acs.org/doi/10.1021/acs.jcim.5c03109)) — the study that first benchmarked
Tsetlin Machines against Random Forest and XGBoost for molecular property prediction. We reuse its
datasets (and its ECFP-2048 → TM as the baseline to beat), then ask a different question: with the
TM running on Apple's GPU, how far does *feature engineering* move the needle?

**How we turned molecules into bits** (quick logistic-regression probe, identical splits):

| representation → bits | MDR1 | MOR |
|---|---|---|
| ECFP-2048 (presence) | 0.974 | 0.933 |
| ECFP-2048 **count** → thermometer ×3 | 0.976 | 0.933 |
| RDKit2D-217 → RPCholesky-128 → thermometer | 0.973 | 0.887 |
| Osmordred-3585 → RPCholesky-1024 → thermometer | **0.980** | 0.913 |
| molFTP 27-d aggregate → thermometer | 0.960 | 0.916 |
| molFTP per-key, full 28k–66k ×2-bit | 0.969 | **0.943** |
| molFTP per-key → **feature-hash 8192** ×2-bit | 0.970 | 0.940 |
| molFTP per-key → RPCholesky-1024 ×2-bit | 0.956 | 0.828 |

*Read it as:* feature hashing keeps the full molFTP signal (0.940 ≈ full 0.943) and beats ECFP on
MOR; RPCholesky on those sparse keys throws signal away (0.828). Net for lakes, magnet for needles.

**On the actual Tsetlin Machine** — extending the
[TM-QSAR-Benchmark](https://pubs.acs.org/doi/10.1021/acs.jcim.5c03109)'s **ECFP-2048 → TM** baseline
to **curated [molFTP](https://github.com/osmoai/molftp)** and **[bcfp](https://github.com/osmoai/bcfp)**
(ECFP/BCFP Sort&Slice + OOV). One self-consistent run — every row through the *same* Coalesced TM
(400 clauses / 20 epochs, 3-fold CV, presence-binarized unless noted):

| features → mlxTM | MDR1 | MOR |
|---|---|---|
| ECFP-2048 (presence) — *the paper's descriptor* | 0.965 | 0.901 |
| ECFP-2048 **count** → thermo ×3 | **0.978** | 0.889 |
| curated **molFTP** 27-d → thermo ×4 | 0.962 | **0.917** |
| **bcfp** ECFP Sort&Slice-512 + OOV | 0.964 | 0.907 |
| **bcfp** BCFP Sort&Slice-512 + OOV | 0.969 | 0.887 |
| **bcfp** ECFP+BCFP Sort&Slice-512 + OOV | 0.972 | 0.909 |

*Read it as:* there's **no single winner** — **count-ECFP takes MDR1** (0.978, *"this ring occurs
≥ 2 times"* is a rule a TM loves), **curated molFTP-27 takes MOR** (0.917, on just 27 numbers), and
**bcfp's ECFP+BCFP Sort&Slice+OOV beats the plain-ECFP baseline on *both*** (+0.007 MDR1, +0.008
MOR) — bond-centered view plus clean, collision-free switches earn their keep. BCFP *alone* is
weaker; it needs its ECFP partner. The meta-lesson holds: with a rule-learner, **how you switch-ify
the molecule matters more than the model**, and there's no free lunch across targets. A permutation
test confirmed no leakage.

> **vs. the paper.** The TM-QSAR-Benchmark's headline (MOR ECFP **0.93**) uses a heavier,
> Optuna-tuned **1600-clause / 50-epoch** TM under grouped CV; our table above is a lighter, fixed
> **400-clause** GPU run that trades a little raw accuracy for speed and interpretability. The point
> here is the *feature* axis they didn't explore — molFTP and ECFP+BCFP Sort&Slice both clear their
> ECFP→TM baseline. A clause-for-clause rematch at their config is the natural next experiment.

## A cheat-sheet for the ML crowd

| we said | you already know it as |
|---|---|
| Tsetlin clause | a learned if-then rule (RuleFit-ish) |
| binarization | quantization, but to 1-bit switches |
| rotation binarizer | QuaRot / QuIP incoherence; ITQ / SimHash |
| RPCholesky select | label-free Nyström / diverse-subset selection |
| molFTP scores | per-fragment target encoding (leakage-safe) |
| the hashing trick | the feature-hashing / ECFP-folding trick |
| the leakage check | a permutation test |

## References & further reading

- **TM-QSAR-Benchmark** — the study this post builds on: it benchmarked Tsetlin Machines against
  Random Forest and XGBoost for QSAR, and supplied the MDR1/MOR opioid datasets and the
  ECFP-2048 → TM baseline we extend.
  [code](https://github.com/PaulC61/TM-QSAR-Benchmark) ·
  [paper, *J. Chem. Inf. Model.* 2025](https://pubs.acs.org/doi/10.1021/acs.jcim.5c03109)
- **molFTP** — fragment-target prevalence features.
  [code](https://github.com/osmoai/molftp) · [paper, arXiv:2510.06029](https://arxiv.org/abs/2510.06029)
- **bcfp** — bond-centered fingerprints (ECFP/BCFP) with Sort&Slice + OOV.
  [code](https://github.com/osmoai/bcfp) · [paper, arXiv:2510.04837](https://arxiv.org/abs/2510.04837)
- **mlxTM** — this library: Tsetlin Machines on Apple's GPU. [code](https://github.com/guillaume-osmo/mlxTM)
- **MLX** — Apple's array framework. [code](https://github.com/ml-explore/mlx)
- Binarization lineage: [QuaRot](https://arxiv.org/abs/2404.00456) / QuIP (rotation), ITQ / SimHash;
  RPCholesky (Chen, Epperly & Tropp, 2022 — label-free Nyström selection).

## Try it

```bash
git clone https://github.com/guillaume-osmo/mlxTM && cd mlxTM
pip install -e .            # macOS / Apple Silicon only (MLX has no Linux/Windows wheels)
python examples/noisy_xor.py
```

Five GPU backends, the feature utilities (binarizers, RPCholesky), 27 passing tests, MIT-licensed.
Kick the tires, and tell me where it breaks.
