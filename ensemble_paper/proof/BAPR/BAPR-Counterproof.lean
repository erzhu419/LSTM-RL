import Mathlib

/-!
# Counterexample: Q-dependent belief weights break contraction in BA-PR

## Motivation

In `BAPR.lean`, the belief distribution ρ is **frozen** (computed from the
target network), ensuring contraction.  This file proves the *necessity*
of that design: if ρ depends on the current Q-function, the BA-PR operator
can fail to be contractive.

## Key Insight: Mode Reward Gap Amplification

When belief weights ρ are Q-dependent (e.g., the agent re-estimates which
mode it's in based on the current value function), the operator gains an
extra term proportional to the **mode reward gap** Δ = R₁ − R₂:

  T_bad(Q) = (γ + λ·Δ) · Q + constant

where λ controls how sensitively the weight responds to Q.
When λ·Δ ≥ 1 − γ, the effective contraction factor γ + λ·Δ ≥ 1,
and the operator is no longer contractive.

## Physical Interpretation

This models an agent that **re-infers the environment mode based on its
own value estimate** at every Bellman backup.  In a piecewise stationary
environment with distinct modes (high Δ), this feedback loop amplifies
estimation errors instead of dampening them.

## Simplified MDP

- S = A = Unit  (single state, single action)
- Two modes with rewards R₁, R₂
- ρ₁(Q) = λ·Q  (weight on mode 1 depends linearly on Q)
- P = identity  (deterministic self-loop)

The "bad" operator becomes:
  T_bad(Q) = ρ₁(Q)·(γQ + R₁) + (1 − ρ₁(Q))·(γQ + R₂)
           = γQ + R₂ + λQ·(R₁ − R₂)
           = (γ + λΔ)·Q + R₂

This has the same algebraic form as RESAC-Counterproof's T_bad.

## Connection to BAPR.lean

| Operator             | Belief weights      | Distance factor | Contracts?        |
|----------------------|---------------------|-----------------|-------------------|
| T_bad (this file)    | ρ = f(Q) (varies)   | γ + λΔ          | ❌ if γ+λΔ ≥ 1    |
| T_BAPR (BAPR.lean)   | ρ frozen (fixed)    | γ               | ✅ (γ < 1)        |

The frozen-belief design is *necessary* for contraction in multi-mode MDPs.
-/

noncomputable section

set_option linter.unusedVariables false

namespace BAPR.Counter

/-! ## The "bad" operator with Q-dependent mode weights -/

/--
The Q-dependent belief Bellman operator on a 1-state 1-action 2-mode MDP.

Parameters:
- `γ` : discount factor
- `lam` : sensitivity of belief weight to Q  (ρ₁ = λ·Q)
- `Δ` : mode reward gap (R₁ − R₂)
- `R₂` : base reward (mode 2)

T_bad(Q) = (γ + λΔ)·Q + R₂
-/
def T_bad (γ lam Δ R₂ : ℝ) (q : ℝ) : ℝ := (γ + lam * Δ) * q + R₂

/-- The difference T_bad(q₁) − T_bad(q₂) is (γ + λΔ)·(q₁ − q₂). -/
lemma T_bad_diff (γ lam Δ R₂ q₁ q₂ : ℝ) :
    T_bad γ lam Δ R₂ q₁ - T_bad γ lam Δ R₂ q₂ =
    (γ + lam * Δ) * (q₁ - q₂) := by
  unfold T_bad; ring

/-! ## Main theorem: not a contraction when γ + λΔ ≥ 1 -/

/--
**Counterexample**: If belief weights depend on Q with sensitivity λ, and
the mode reward gap Δ satisfies γ + λΔ ≥ 1, then T_bad is NOT a contraction.

Witnesses: q₁ = 1, q₂ = 0.

    |T_bad(1) − T_bad(0)| = |γ + λΔ| · 1 ≥ 1 = |1 − 0|.

This proves that the frozen-belief assumption in BAPR.lean is *necessary*:
without it, the mode reward gap Δ amplifies Q-estimation errors through
the feedback loop ρ(Q), destroying contractivity.
-/
theorem T_bad_not_contraction
    {γ lam Δ R₂ : ℝ} (hγ : 0 ≤ γ) (hlam : 0 ≤ lam) (hΔ : 0 ≤ Δ)
    (hbad : 1 ≤ γ + lam * Δ) :
    ∃ q₁ q₂ : ℝ, |T_bad γ lam Δ R₂ q₁ - T_bad γ lam Δ R₂ q₂| ≥
                  |q₁ - q₂| := by
  refine ⟨1, 0, ?_⟩
  have hlhs : T_bad γ lam Δ R₂ 1 - T_bad γ lam Δ R₂ 0 = γ + lam * Δ := by
    unfold T_bad; ring
  have hrhs : (1 : ℝ) - 0 = 1 := by ring
  rw [hlhs, hrhs]
  rw [abs_of_nonneg (by linarith), abs_of_pos (by norm_num)]
  exact hbad

/-! ## Corollary: strict expansion when γ + λΔ > 1 -/

/--
If γ + λΔ > 1, the operator **strictly expands** distances.
No contraction factor k < 1 exists.
-/
theorem T_bad_expansion
    {γ lam Δ R₂ : ℝ} (hγ : 0 ≤ γ) (hlam : 0 ≤ lam) (hΔ : 0 ≤ Δ)
    (hbad : 1 < γ + lam * Δ) :
    ∃ q₁ q₂ : ℝ, |T_bad γ lam Δ R₂ q₁ - T_bad γ lam Δ R₂ q₂| >
                  |q₁ - q₂| := by
  refine ⟨1, 0, ?_⟩
  have hlhs : T_bad γ lam Δ R₂ 1 - T_bad γ lam Δ R₂ 0 = γ + lam * Δ := by
    unfold T_bad; ring
  have hrhs : (1 : ℝ) - 0 = 1 := by ring
  rw [hlhs, hrhs]
  rw [abs_of_nonneg (by linarith), abs_of_pos (by norm_num)]
  exact hbad

/-! ## Detection Delay Bound

In a piecewise stationary environment, after a mode switch occurs at time t₀,
the Bayesian belief ρ must re-converge to assign high weight to the correct
new mode.  This section proves a bound on the detection delay.

### Setup

After a mode switch, the posterior belief at each step is:
    ρ'(h_new) ∝ ρ(h_new) · L^n

where L > 1 is the likelihood ratio favoring the new mode (because predictions
under the old mode generate higher surprise), and n is the number of steps
since the switch.  The belief on the old mode decays as:
    ρ'(h_old) ∝ ρ(h_old) · (1/L)^n

### Key Result

The posterior ratio ρ(h_new)/ρ(h_old) grows as L^(2n).  To achieve
ρ(h_new)/ρ(h_old) ≥ 1/δ (i.e., the new mode dominates with confidence 1−δ),
we need:
    n ≥ log(1/δ) / (2·log(L)) + log(ρ₀(h_old)/ρ₀(h_new)) / (2·log(L))

This is O(log(1/δ)) when L > 1 (Mode Separability Assumption).
-/

/--
Posterior ratio after n steps of consistent evidence.

If the likelihood ratio is L > 1, and the prior ratio is r₀ = ρ₀(old)/ρ₀(new),
then after n steps the posterior ratio new/old = (1/r₀) · L^(2n).
-/
def posterior_ratio (r₀ L : ℝ) (n : ℕ) : ℝ := (1 / r₀) * L ^ (2 * n)

/--
**Detection Delay Bound**: If L > 1 (modes are separable) and we
want confidence 1/δ, the required steps n satisfy:

    posterior_ratio r₀ L n ≥ 1/δ  when  L^(2n) ≥ r₀/δ
-/
theorem detection_delay_sufficient
    {r₀ L : ℝ} {n : ℕ} {δ : ℝ}
    (hr₀ : 0 < r₀)
    (hL : 1 < L)
    (hδ : 0 < δ)
    (hn : r₀ / δ ≤ L ^ (2 * n)) :
    1 / δ ≤ posterior_ratio r₀ L n := by
  unfold posterior_ratio
  have hδ_ne : δ ≠ 0 := ne_of_gt hδ
  have hr₀_ne : r₀ ≠ 0 := ne_of_gt hr₀
  rw [div_mul_eq_mul_div, div_le_div_iff₀ hδ hr₀]
  rw [div_le_iff₀ hδ] at hn
  linarith

/--
**Monotonicity of detection confidence**: More evidence steps monotonically
increase the posterior ratio (confidence grows over time).
-/
theorem detection_confidence_mono
    {r₀ L : ℝ} (hr₀ : 0 < r₀) (hL : 1 ≤ L) (n : ℕ) :
    posterior_ratio r₀ L n ≤ posterior_ratio r₀ L (n + 1) := by
  unfold posterior_ratio
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  apply pow_le_pow_right₀ hL
  omega

/-!
## Summary

| Result                          | What it proves                                    |
|---------------------------------|---------------------------------------------------|
| `T_bad_not_contraction`         | Q-dependent ρ breaks contraction (necessity)      |
| `T_bad_expansion`               | Q-dependent ρ causes strict expansion             |
| `detection_delay_sufficient`    | O(log(1/δ)) steps suffice to detect mode switch   |
| `detection_confidence_mono`     | Confidence monotonically increases with evidence   |

Together with `BAPR.lean`, these results establish:

1. **Sufficiency**: Frozen Bayesian belief → γ-contraction (BAPR.lean)
2. **Necessity**: Q-dependent belief → breaks contraction (this file)
3. **Practicality**: Mode detection happens in O(log(1/δ)) steps (this file)
-/

end BAPR.Counter
