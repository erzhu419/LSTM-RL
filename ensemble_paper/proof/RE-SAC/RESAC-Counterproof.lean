import Mathlib

/-!
# Counterexample: Q-dependent aleatoric penalty breaks contraction

This file formally proves that if the aleatoric penalty is a *function of Q*
(rather than a fixed constant), the Bellman operator may fail to be a
γ-contraction — even when γ < 1.

## Connection to RE-SAC (Proof.lean)

In Proof.lean, κ_ale is a **fixed** scalar, cancelling in T(Q₁) − T(Q₂),
which gives a genuine γ-contraction (γ < 1).

Here we show the *opposite*: if the aleatoric term scales with Q (coefficient
`lam`), the effective factor becomes γ + lam.  When γ + lam ≥ 1 the operator
is no longer contractive.

## Simplified MDP

- S = A = Unit  (single state, single action)
- R = 0,  P = identity  (deterministic self-loop)

This collapses max_a Q(s,a) = Q(s,a), giving the "bad" operator:

  T_bad Q = γ · Q + lam · Q = (γ + lam) · Q
-/

noncomputable section

namespace RESAC.Counter

/-! ## The "bad" operator -/

/--
The aleatoric-penalty-dependent Bellman operator on a 1-state 1-action MDP.
-/
def T_bad (γ lam : ℝ) (q : ℝ) : ℝ := (γ + lam) * q

lemma T_bad_diff (γ lam q₁ q₂ : ℝ) :
    T_bad γ lam q₁ - T_bad γ lam q₂ = (γ + lam) * (q₁ - q₂) := by
  unfold T_bad; ring

/-! ## Main theorem: not a contraction when γ + lam ≥ 1 -/

/--
**Counterexample**: if γ ≥ 0, lam ≥ 0, and γ + lam ≥ 1, then T_bad is NOT
a contraction.  Witnesses: q₁ = 1, q₂ = 0.

  |T_bad q₁ − T_bad q₂| = (γ + lam) · 1 ≥ 1 = |q₁ − q₂|.
-/
theorem T_bad_not_contraction
    {γ lam : ℝ} (hγ : 0 ≤ γ) (hlam : 0 ≤ lam) (hbad : 1 ≤ γ + lam) :
    ∃ q₁ q₂ : ℝ, |T_bad γ lam q₁ - T_bad γ lam q₂| ≥ |q₁ - q₂| := by
  refine ⟨1, 0, ?_⟩
  have hlhs : T_bad γ lam 1 - T_bad γ lam 0 = γ + lam := by unfold T_bad; ring
  have hrhs : (1 : ℝ) - 0 = 1 := by ring
  rw [hlhs, hrhs]
  rw [abs_of_nonneg (by linarith), abs_of_pos (by norm_num)]
  exact hbad

/-! ## Corollary: strict expansion when γ + lam > 1 -/

/--
If γ + lam > 1, T_bad **strictly expands** distances.
It cannot be a γ'-contraction for any γ' < 1.
-/
theorem T_bad_expansion
    {γ lam : ℝ} (hγ : 0 ≤ γ) (hlam : 0 ≤ lam) (hbad : 1 < γ + lam) :
    ∃ q₁ q₂ : ℝ, |T_bad γ lam q₁ - T_bad γ lam q₂| > |q₁ - q₂| := by
  refine ⟨1, 0, ?_⟩
  have hlhs : T_bad γ lam 1 - T_bad γ lam 0 = γ + lam := by unfold T_bad; ring
  have hrhs : (1 : ℝ) - 0 = 1 := by ring
  rw [hlhs, hrhs]
  rw [abs_of_nonneg (by linarith), abs_of_pos (by norm_num)]
  exact hbad

/-!
## Summary table

| Operator    | Aleatoric term    | Distance factor | Contracts?      |
|-------------|-------------------|-----------------|-----------------|
| T_bad       | lam·Q  (varies)   | γ + lam         | ❌ if γ+lam ≥ 1 |
| T (RE-SAC)  | κ      (fixed)    | γ               | ✅ (γ < 1)      |

The fixed-κ design in Proof.lean is *necessary* for contraction.
-/

end RESAC.Counter
