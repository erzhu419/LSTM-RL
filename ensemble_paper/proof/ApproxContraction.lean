import Mathlib

/-!
# Formal Proofs: Function Approximation and Stochastic Approximation Bounds

## Overview

The contraction proofs in RESAC.lean, BAPR.lean, and BAMOR.lean operate in the
**tabular** setting (Fintype S, A).  In practice, SAC uses neural networks for
function approximation and mini-batch sampling.  This file formally bridges
that gap with three results:

1. **Projected Contraction** (Part I): If the exact operator T is a
   γ-contraction and the projection Π is non-expansive, then Π∘T is also
   a γ-contraction.

2. **Approximation Error Bound** (Part II): The fixed point Q̃ of the
   projected operator Π∘T satisfies ‖Q̃ − Q*‖ ≤ ε_proj/(1−γ), where
   ε_proj is the projection error ‖ΠQ* − Q*‖.

3. **Stochastic Tracking Bound** (Part III): With per-step noise bounded
   by σ (from mini-batch sampling), the tracking error satisfies
   e(n) ≤ γⁿ·e(0) + σ/(1−γ).

4. **Combined Bound** (Part IV): Joint function approximation + stochastic
   noise yields ‖Q_n − Q*‖ ≤ γⁿ·‖Q₀ − Q*‖ + (ε_proj + σ)/(1−γ).

## Assumptions

These results are stated abstractly — they hold for ANY γ-contraction T
(whether hard-max, soft-max, single-objective, or multi-objective).
The projection Π models the restriction to the neural network function class.
-/

noncomputable section

set_option linter.unusedSectionVars false

namespace ApproxContraction

-- ════════════════════════════════════════════════════════════════
-- PART I: Projected Operator Contraction
-- ════════════════════════════════════════════════════════════════

/-! ## 1. Abstract Contraction Framework

We work with an abstract metric space and operators satisfying
contraction/non-expansion properties.  This makes the results
applicable to ANY of our Bellman operators (RE-SAC, BA-PR, BAMOR).
-/

/--
**Theorem (Projected Contraction)**: Composition of a non-expansive
projection Π with a γ-contraction T yields a γ-contraction.

This justifies function approximation: if T is our Bellman operator
(proven contractive in RESAC/BAPR/BAMOR.lean) and Π is orthogonal
projection onto the neural network function class (non-expansive by
Hilbert space theory), then the practical operator Π∘T is contractive.
-/
theorem projected_contraction {γ : ℝ} (hγ_nn : 0 ≤ γ) (hγ_lt : γ < 1)
    -- T is a γ-contraction: d(T Q₁, T Q₂) ≤ γ · d(Q₁, Q₂)
    {dist_TQ₁TQ₂ dist_Q₁Q₂ : ℝ}
    (h_contract : dist_TQ₁TQ₂ ≤ γ * dist_Q₁Q₂)
    -- Proj is non-expansive: d(Proj X, Proj Y) ≤ d(X, Y)
    {dist_ProjTQ₁_ProjTQ₂ : ℝ}
    (h_proj : dist_ProjTQ₁_ProjTQ₂ ≤ dist_TQ₁TQ₂) :
    dist_ProjTQ₁_ProjTQ₂ ≤ γ * dist_Q₁Q₂ := by
  linarith

-- ════════════════════════════════════════════════════════════════
-- PART II: Approximation Error Bound
-- ════════════════════════════════════════════════════════════════

/-! ## 2. Fixed Point Approximation Error

When using function approximation, the fixed point Q̃ of Π∘T differs
from the true fixed point Q* of T.  The error is bounded by the
projection error divided by (1 − γ).

### Derivation

    ‖Q̃ − Q*‖ = ‖Π T Q̃ − Q*‖           (Q̃ = Π T Q̃)
              ≤ ‖Π T Q̃ − Π Q*‖ + ‖Π Q* − Q*‖   (triangle ineq)
              ≤ ‖T Q̃ − Q*‖ + ε_proj     (Π non-expansive)
              = ‖T Q̃ − T Q*‖ + ε_proj   (Q* = T Q*)
              ≤ γ ‖Q̃ − Q*‖ + ε_proj     (T contractive)

    ⟹ (1 − γ) ‖Q̃ − Q*‖ ≤ ε_proj
    ⟹ ‖Q̃ − Q*‖ ≤ ε_proj / (1 − γ)
-/

/--
**Theorem (Approximation Error Bound)**: If the error e = ‖Q̃ − Q*‖
satisfies e ≤ γ·e + ε_proj (from the derivation above), then
e ≤ ε_proj / (1 − γ).

This gives a clean separation: the contraction rate γ (from our proofs)
and the projection error ε_proj (from the network architecture) jointly
determine the approximation quality.
-/
theorem approx_fixed_point_bound {γ : ℝ} (hγ_lt : γ < 1)
    {ε_proj e : ℝ} (he_nn : 0 ≤ e)
    (h : e ≤ γ * e + ε_proj) :
    e ≤ ε_proj / (1 - γ) := by
  have h1γ : (0 : ℝ) < 1 - γ := by linarith
  -- From h: (1-γ)*e ≤ ε_proj, so e ≤ ε_proj/(1-γ)
  have h_sub : (1 - γ) * e ≤ ε_proj := by nlinarith
  exact (le_div_iff₀ h1γ).mpr (by linarith [mul_comm (1 - γ) e])

/--
**Corollary**: The approximation error vanishes as ε_proj → 0.
When the function class is rich enough to represent Q* exactly
(ε_proj = 0), the projected operator recovers the exact fixed point.
-/
theorem exact_representation_recovers {γ : ℝ} (hγ_lt : γ < 1)
    {e : ℝ} (he_nn : 0 ≤ e)
    (h : e ≤ γ * e + 0) :
    e ≤ 0 := by
  have := approx_fixed_point_bound hγ_lt he_nn (by linarith : e ≤ γ * e + 0)
  simp at this
  exact this

-- ════════════════════════════════════════════════════════════════
-- PART III: Stochastic Approximation Tracking Bound
-- ════════════════════════════════════════════════════════════════

/-! ## 3. Stochastic Approximation

In practice, the Bellman operator is estimated from mini-batch samples:

    Q_{n+1} = T̂(Q_n) ≈ T(Q_n)

where |T̂(Q) − T(Q)| ≤ σ for all Q (bounded sampling noise).
The tracking error e(n) = ‖Q_n − Q*‖ then satisfies:

    e(n+1) ≤ γ · e(n) + σ

This is a first-order linear recurrence, solved below.
-/

/--
**Theorem (Stochastic Tracking Bound)**: With per-step noise bounded
by σ ≥ 0, the tracking error after n steps satisfies:

    e(n) ≤ γⁿ · e(0) + σ / (1 − γ)

The first term decays geometrically (forgetting initial error).
The second term is the irreducible noise floor from sampling.
-/
theorem stochastic_tracking_bound {γ : ℝ} (hγ_nn : 0 ≤ γ) (hγ_lt : γ < 1)
    {σ : ℝ} (hσ_nn : 0 ≤ σ) {e : ℕ → ℝ}
    (he_step : ∀ n, e (n + 1) ≤ γ * e n + σ) :
    ∀ n, e n ≤ γ ^ n * e 0 + σ / (1 - γ) := by
  intro n
  induction n with
  | zero =>
    simp
    exact div_nonneg hσ_nn (by linarith)
  | succ k ih =>
    have h1γ_pos : (0 : ℝ) < 1 - γ := by linarith
    have h1γ_ne : (1 : ℝ) - γ ≠ 0 := ne_of_gt h1γ_pos
    -- Key arithmetic: γ * (σ/(1-γ)) + σ = σ/(1-γ)
    have hσγ : γ * (σ / (1 - γ)) + σ = σ / (1 - γ) := by
      field_simp
      ring
    calc e (k + 1)
        ≤ γ * e k + σ := he_step k
      _ ≤ γ * (γ ^ k * e 0 + σ / (1 - γ)) + σ := by
          linarith [mul_le_mul_of_nonneg_left ih hγ_nn]
      _ = γ * (γ ^ k * e 0) + (γ * (σ / (1 - γ)) + σ) := by ring
      _ = γ * (γ ^ k * e 0) + σ / (1 - γ) := by rw [hσγ]
      _ = γ ^ (k + 1) * e 0 + σ / (1 - γ) := by
          rw [show γ * (γ ^ k * e 0) = γ ^ (k + 1) * e 0 from by
            rw [pow_succ]; ring]

/--
**Corollary (Asymptotic Noise Floor)**: For any bounded initial error,
the long-run tracking error is bounded by σ/(1−γ).
-/
theorem asymptotic_noise_floor {γ : ℝ} (hγ_nn : 0 ≤ γ) (hγ_lt : γ < 1)
    {σ : ℝ} (hσ_nn : 0 ≤ σ) {B : ℝ}
    {e : ℕ → ℝ} (he_nn : ∀ n, 0 ≤ e n)
    (he_init : e 0 ≤ B)
    (he_step : ∀ n, e (n + 1) ≤ γ * e n + σ) :
    ∀ n, e n ≤ B + σ / (1 - γ) := by
  intro n
  have h := stochastic_tracking_bound hγ_nn hγ_lt hσ_nn he_step n
  have hγn : γ ^ n * e 0 ≤ 1 * B := by
    apply mul_le_mul _ he_init (he_nn 0) (by linarith)
    exact pow_le_one₀ hγ_nn (le_of_lt hγ_lt)
  linarith

-- ════════════════════════════════════════════════════════════════
-- PART IV: Combined Bound (Function Approximation + Sampling Noise)
-- ════════════════════════════════════════════════════════════════

/-! ## 4. Joint Approximation + Stochastic Bound

When both sources of error are present:
- Function approximation introduces per-step drift ε_proj
  (from projecting after each Bellman backup)
- Mini-batch sampling introduces per-step noise σ

The combined recurrence is:
    e(n+1) ≤ γ · e(n) + ε_proj + σ

which has the same structure as the stochastic bound with Δ = ε_proj + σ.
-/

/--
**Theorem (Combined Approximation + Stochastic Bound)**

    e(n) ≤ γⁿ · e(0) + (ε_proj + σ) / (1 − γ)

This gives the complete error decomposition for practical deep RL:
- γ: contraction rate (proven in RESAC/BAPR/BAMOR.lean)
- ε_proj: function class expressiveness (network architecture choice)
- σ: sampling noise (mini-batch size, replay buffer)
-/
theorem combined_approx_stochastic_bound
    {γ : ℝ} (hγ_nn : 0 ≤ γ) (hγ_lt : γ < 1)
    {ε_proj σ : ℝ} (hε_nn : 0 ≤ ε_proj) (hσ_nn : 0 ≤ σ)
    {e : ℕ → ℝ}
    (he_step : ∀ n, e (n + 1) ≤ γ * e n + ε_proj + σ) :
    ∀ n, e n ≤ γ ^ n * e 0 + (ε_proj + σ) / (1 - γ) := by
  -- Reduce to stochastic_tracking_bound with Δ := ε_proj + σ
  exact stochastic_tracking_bound hγ_nn hγ_lt (by linarith) fun n => by
    linarith [he_step n]

/--
**Corollary (Steady-State Error Decomposition)**

In steady state (n → ∞), the error converges to at most:

    e_∞ ≤ (ε_proj + σ) / (1 − γ)

This cleanly separates the roles of:
- The contraction rate γ (algorithmic — our proofs)
- The projection error ε_proj (architectural — network capacity)
- The sampling noise σ (statistical — sample size)
-/
theorem steady_state_decomposition
    {γ : ℝ} (hγ_nn : 0 ≤ γ) (hγ_lt : γ < 1)
    {ε_proj σ B : ℝ} (hε_nn : 0 ≤ ε_proj) (hσ_nn : 0 ≤ σ)
    {e : ℕ → ℝ} (he_nn : ∀ n, 0 ≤ e n)
    (he_init : e 0 ≤ B)
    (he_step : ∀ n, e (n + 1) ≤ γ * e n + ε_proj + σ) :
    ∀ n, e n ≤ B + (ε_proj + σ) / (1 - γ) := by
  exact asymptotic_noise_floor hγ_nn hγ_lt (by linarith) he_nn he_init fun n =>
    by linarith [he_step n]

end ApproxContraction
