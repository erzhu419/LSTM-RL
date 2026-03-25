import Mathlib

/-!
# Formal Proof: BA-PR SAC Bellman Operator is a Contraction Mapping

## Overview

BA-PR SAC (Bayesian Amnesic Piecewise-Robust SAC) extends RE-SAC with Bayesian
change-point detection for piecewise stationary environments.  It maintains a
belief distribution `ρ` over possible "run-lengths" (time since the last
environment change).  The operator is a **convex combination** of
mode-conditional Bellman operators weighted by this belief:

    T_BAPR(Q)(s,a) = Σ_h ρ(h) · T_h(Q)(s,a)

## Proof Strategy

The proof exploits a fundamental property:
> **A convex combination of γ-contractions is itself a γ-contraction.**

Each mode-conditional operator T_h is a standard Bellman operator with frozen
penalties (same structure as RE-SAC).  The weighted sum T_BAPR inherits
γ-contractivity because ρ ≥ 0 and Σ ρ = 1.

## Assumptions

1. **Frozen Belief**: ρ is computed from the target network — frozen during
   one Bellman backup.  (Encoded structurally: ρ is a parameter of T_BAPR.)
2. **Mode Separability / Metastable Period**: Used for outer-loop piecewise
   convergence analysis, NOT needed for single-step contraction.
3. **Compact Support**: S, A, H are Fintype.

## Proof Structure

1. `max_over_a_mono`           — max is monotone in Q
2. `max_over_a_add_const`      — max distributes over constant shift
3. `max_over_a_nonexpansive`   — max is 1-Lipschitz in L∞
4. `t_mode_pointwise_bound`    — per-mode pointwise contraction bound
5. `bapr_contraction`          — Main theorem: T_BAPR is a γ-contraction
-/

noncomputable section

set_option linter.unusedSectionVars false

namespace BAPR

/-! ## 1. Space Definitions

- `S` : physical state space (Fintype)
- `A` : action space (Fintype)
- `H` : mode/run-length space — each h ∈ H determines environment parameters
-/

variable {S A H : Type} [Fintype S] [Fintype A] [Fintype H]
                         [Nonempty S] [Nonempty A] [Nonempty H]

/-- Hyperparameters for the BA-PR framework -/
structure Params where
  γ        : ℝ
  lam_epi  : ℝ
  hγ       : 0 ≤ γ ∧ γ < 1
  hlam_epi : 0 ≤ lam_epi

/-! ## 2. The Max Operator -/

/-- V(s) = max_a Q(s, a) -/
def max_over_a (Q : S × A → ℝ) (s : S) : ℝ :=
  (Finset.univ : Finset A).sup' Finset.univ_nonempty (fun a => Q (s, a))

/-! ## 3. Intermediate Lemmas -/

/-- max_over_a is monotone in Q. -/
lemma max_over_a_mono {Q₁ Q₂ : S × A → ℝ} (h : Q₁ ≤ Q₂) (s : S) :
    max_over_a Q₁ s ≤ max_over_a Q₂ s := by
  unfold max_over_a
  apply Finset.sup'_le
  intro a _
  have hq : Q₁ (s, a) ≤ Q₂ (s, a) := h (s, a)
  have hle : Q₂ (s, a) ≤ Finset.univ.sup' Finset.univ_nonempty
      (fun b => Q₂ (s, b)) :=
    Finset.le_sup' (fun b => Q₂ (s, b)) (Finset.mem_univ a)
  linarith

/-- max_over_a distributes over constant addition. -/
lemma max_over_a_add_const (Q : S × A → ℝ) (c : ℝ) (s : S) :
    max_over_a (Q + fun _ => c) s = max_over_a Q s + c := by
  unfold max_over_a
  simp only [Pi.add_apply]
  apply le_antisymm
  · apply Finset.sup'_le
    intro a _
    have : Q (s, a) ≤ Finset.univ.sup' Finset.univ_nonempty
        (fun b => Q (s, b)) :=
      Finset.le_sup' (fun b => Q (s, b)) (Finset.mem_univ a)
    linarith
  · have key : Finset.univ.sup' Finset.univ_nonempty (fun a => Q (s, a)) ≤
        Finset.univ.sup' Finset.univ_nonempty (fun a => Q (s, a) + c) - c := by
      apply Finset.sup'_le
      intro a _
      have : Q (s, a) + c ≤ Finset.univ.sup' Finset.univ_nonempty
          (fun b => Q (s, b) + c) :=
        Finset.le_sup' (fun b => Q (s, b) + c) (Finset.mem_univ a)
      linarith
    linarith

/-- max_over_a is 1-Lipschitz in L∞. -/
lemma max_over_a_nonexpansive {Q₁ Q₂ : S × A → ℝ} (s : S) {ε : ℝ}
    (hpw : ∀ sa, |Q₁ sa - Q₂ sa| ≤ ε) :
    |max_over_a Q₁ s - max_over_a Q₂ s| ≤ ε := by
  rw [abs_le]
  constructor
  · rw [neg_le_sub_iff_le_add]
    unfold max_over_a
    apply Finset.sup'_le
    intro a _
    have habs : |Q₁ (s, a) - Q₂ (s, a)| ≤ ε := hpw (s, a)
    have hle : Q₂ (s, a) ≤ Q₁ (s, a) + ε := by linarith [(abs_le.mp habs).1]
    have hmax₁ : Q₁ (s, a) ≤ Finset.univ.sup' Finset.univ_nonempty
        (fun b => Q₁ (s, b)) :=
      Finset.le_sup' (fun b => Q₁ (s, b)) (Finset.mem_univ a)
    linarith
  · unfold max_over_a
    rw [sub_le_iff_le_add]
    apply Finset.sup'_le
    intro a _
    have habs : |Q₁ (s, a) - Q₂ (s, a)| ≤ ε := hpw (s, a)
    have hle : Q₁ (s, a) ≤ Q₂ (s, a) + ε := by linarith [(abs_le.mp habs).2]
    have hmax₂ : Q₂ (s, a) ≤ Finset.univ.sup' Finset.univ_nonempty
        (fun b => Q₂ (s, b)) :=
      Finset.le_sup' (fun b => Q₂ (s, b)) (Finset.mem_univ a)
    linarith

/-! ## 4. Mode-Conditional Bellman Operator

For each mode h ∈ H (run-length since last change point), we have a
mode-specific reward R_h, transition P_h, and epistemic penalty Γ_epi_h.
The aleatoric penalty κ_ale is a frozen scalar (from target network).
-/

/-- Mode-conditional Bellman operator for mode h. -/
def T_mode (p : Params) (R : H → S → A → ℝ) (P : H → S → A → S → ℝ)
    (Γ_epi : H → S → A → ℝ) (κ_ale : ℝ) (h : H)
    (Q : S × A → ℝ) : S × A → ℝ := fun ⟨s, a⟩ =>
  R h s a + p.γ * (
    (∑ s' : S, P h s a s' * max_over_a Q s') -
    p.lam_epi * Γ_epi h s a -
    κ_ale
  )

/-! ## 5. BA-PR Operator: Convex Combination over Modes

T_BAPR(Q)(s,a) = Σ_h ρ(h) · T_h(Q)(s,a)

where ρ is the frozen Bayesian belief over run-lengths.
-/

/-- The BA-PR Bellman operator: weighted mixture of mode-conditional operators. -/
def T_BAPR (p : Params) (R : H → S → A → ℝ) (P : H → S → A → S → ℝ)
    (Γ_epi : H → S → A → ℝ) (κ_ale : ℝ) (ρ : H → ℝ)
    (Q : S × A → ℝ) : S × A → ℝ := fun sa =>
  ∑ h : H, ρ h * T_mode p R P Γ_epi κ_ale h Q sa

-- ============================================================
-- BLACKWELL CONDITIONS FOR BA-PR
-- ============================================================

/-! ## 5a. Monotonicity (Blackwell Condition i)

Each T_h is monotone (same structure as RE-SAC), and a convex combination
of monotone operators is monotone.
-/

/-- Per-mode monotonicity: T_h(Q₁) ≤ T_h(Q₂) when Q₁ ≤ Q₂. -/
lemma t_mode_monotonicity (p : Params) (R : H → S → A → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → ℝ) (κ_ale : ℝ)
    (hP_nn : ∀ h s a s', 0 ≤ P h s a s')
    (h : H) (Q₁ Q₂ : S × A → ℝ) (hle : Q₁ ≤ Q₂) :
    T_mode p R P Γ_epi κ_ale h Q₁ ≤ T_mode p R P Γ_epi κ_ale h Q₂ := by
  intro ⟨s, a⟩
  dsimp only [T_mode]
  have hsum : ∑ s' : S, P h s a s' * max_over_a Q₁ s' ≤
              ∑ s' : S, P h s a s' * max_over_a Q₂ s' := by
    apply Finset.sum_le_sum; intro s' _
    exact mul_le_mul_of_nonneg_left (max_over_a_mono hle s') (hP_nn h s a s')
  nlinarith [p.hγ.1]

/-- **Lemma (BA-PR Monotonicity)**: T_BAPR(Q₁) ≤ T_BAPR(Q₂) when Q₁ ≤ Q₂.
    Convex combination of monotone operators is monotone. -/
lemma bapr_monotonicity (p : Params) (R : H → S → A → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → ℝ) (κ_ale : ℝ)
    (ρ : H → ℝ) (hP_nn : ∀ h s a s', 0 ≤ P h s a s')
    (hρ_nn : ∀ h, 0 ≤ ρ h)
    (Q₁ Q₂ : S × A → ℝ) (hle : Q₁ ≤ Q₂) :
    T_BAPR p R P Γ_epi κ_ale ρ Q₁ ≤ T_BAPR p R P Γ_epi κ_ale ρ Q₂ := by
  intro sa
  dsimp only [T_BAPR]
  apply Finset.sum_le_sum; intro h _
  exact mul_le_mul_of_nonneg_left
    (t_mode_monotonicity p R P Γ_epi κ_ale hP_nn h Q₁ Q₂ hle sa) (hρ_nn h)

/-! ## 5b. Discounting (Blackwell Condition ii)

Each T_h satisfies T_h(Q + c) = T_h(Q) + γc, and weighted sums preserve
this property: T_BAPR(Q + c) = T_BAPR(Q) + γc.
-/

/-- Per-mode discounting: T_h(Q + c) = T_h(Q) + γ·c. -/
lemma t_mode_discounting (p : Params) (R : H → S → A → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → ℝ) (κ_ale : ℝ)
    (hP_prob : ∀ h s a, ∑ s' : S, P h s a s' = 1)
    (h : H) (Q : S × A → ℝ) (c : ℝ) :
    T_mode p R P Γ_epi κ_ale h (Q + fun _ => c) =
    T_mode p R P Γ_epi κ_ale h Q + fun _ => p.γ * c := by
  funext ⟨s, a⟩
  simp only [T_mode, Pi.add_apply]
  have hmax : ∀ s', max_over_a (Q + fun _ => c) s' = max_over_a Q s' + c :=
    fun s' => max_over_a_add_const Q c s'
  have hsum : ∑ s', P h s a s' * max_over_a (Q + fun _ => c) s' =
      (∑ s', P h s a s' * max_over_a Q s') + c := by
    simp_rw [hmax, mul_add]
    rw [Finset.sum_add_distrib, ← Finset.sum_mul, hP_prob h s a, one_mul]
  rw [hsum]; ring

/-- **Lemma (BA-PR Discounting)**: T_BAPR(Q + c) = T_BAPR(Q) + γ·c.
    Weighted sum preserves discounting when Σ ρ = 1. -/
lemma bapr_discounting (p : Params) (R : H → S → A → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → ℝ) (κ_ale : ℝ)
    (ρ : H → ℝ) (hP_prob : ∀ h s a, ∑ s' : S, P h s a s' = 1)
    (hρ_sum : ∑ h : H, ρ h = 1)
    (Q : S × A → ℝ) (c : ℝ) :
    T_BAPR p R P Γ_epi κ_ale ρ (Q + fun _ => c) =
    T_BAPR p R P Γ_epi κ_ale ρ Q + fun _ => p.γ * c := by
  funext sa
  simp only [T_BAPR, Pi.add_apply]
  have h_mode : ∀ h, T_mode p R P Γ_epi κ_ale h (Q + fun _ => c) sa =
      T_mode p R P Γ_epi κ_ale h Q sa + p.γ * c := by
    intro h
    have := congr_fun (t_mode_discounting p R P Γ_epi κ_ale hP_prob h Q c) sa
    simp [Pi.add_apply] at this; exact this
  simp_rw [h_mode, mul_add, Finset.sum_add_distrib,
           ← Finset.sum_mul, hρ_sum, one_mul]

/-! ## 6. Per-Mode Pointwise Contraction Bound


This is the inner building block: each T_h satisfies the same pointwise bound
as the standard RE-SAC operator (Proof.lean).  Frozen penalties cancel exactly.
-/

/-- Per-mode pointwise bound: |T_h(Q₁)(s,a) − T_h(Q₂)(s,a)| ≤ γ·ε -/
lemma t_mode_pointwise_bound (p : Params) (R : H → S → A → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → ℝ) (κ_ale : ℝ)
    (hP_nn : ∀ h s a s', 0 ≤ P h s a s')
    (hP_prob : ∀ h s a, ∑ s' : S, P h s a s' = 1)
    {Q₁ Q₂ : S × A → ℝ} {ε : ℝ} (hpw : ∀ sa, |Q₁ sa - Q₂ sa| ≤ ε)
    (h : H) (s : S) (a : A) :
    |T_mode p R P Γ_epi κ_ale h Q₁ (s, a) -
     T_mode p R P Γ_epi κ_ale h Q₂ (s, a)| ≤ p.γ * ε := by
  dsimp only [T_mode]
  -- R, Γ_epi, κ_ale cancel (frozen penalties — Assumption 1)
  have hS : ∑ s' : S, P h s a s' * max_over_a Q₁ s' -
            ∑ s' : S, P h s a s' * max_over_a Q₂ s' =
            ∑ s' : S, P h s a s' *
              (max_over_a Q₁ s' - max_over_a Q₂ s') := by
    rw [← Finset.sum_sub_distrib]
    congr 1; ext s'; ring
  have hsimp : R h s a + p.γ * (∑ s' : S, P h s a s' * max_over_a Q₁ s' -
      p.lam_epi * Γ_epi h s a - κ_ale) -
    (R h s a + p.γ * (∑ s' : S, P h s a s' * max_over_a Q₂ s' -
      p.lam_epi * Γ_epi h s a - κ_ale)) =
    p.γ * ∑ s' : S, P h s a s' *
      (max_over_a Q₁ s' - max_over_a Q₂ s') := by
    linear_combination p.γ * hS
  rw [hsimp, abs_mul, abs_of_nonneg p.hγ.1]
  apply mul_le_mul_of_nonneg_left _ p.hγ.1
  -- Bound |∑ P*(max Q₁ - max Q₂)| ≤ ε  (same as RE-SAC)
  calc |∑ s' : S, P h s a s' * (max_over_a Q₁ s' - max_over_a Q₂ s')|
      ≤ ∑ s' : S, |P h s a s' * (max_over_a Q₁ s' - max_over_a Q₂ s')| :=
          Finset.abs_sum_le_sum_abs _ _
    _ = ∑ s' : S, P h s a s' * |max_over_a Q₁ s' - max_over_a Q₂ s'| := by
          congr 1; ext s'
          rw [abs_mul, abs_of_nonneg (hP_nn h s a s')]
    _ ≤ ∑ s' : S, P h s a s' * ε := by
          apply Finset.sum_le_sum; intro s' _
          exact mul_le_mul_of_nonneg_left
            (max_over_a_nonexpansive s' hpw) (hP_nn h s a s')
    _ = ε := by rw [← Finset.sum_mul, hP_prob h s a, one_mul]

/-! ## 7. Main Theorem: BA-PR Operator is a γ-Contraction

The key new ingredient compared to RE-SAC: the **convex combination argument**.

    |Σ ρ_h · (T_h Q₁ − T_h Q₂)|
    ≤ Σ ρ_h · |T_h Q₁ − T_h Q₂|      (triangle inequality, ρ ≥ 0)
    ≤ Σ ρ_h · γε                       (per-mode contraction)
    = γε · Σ ρ_h                        (factor out)
    = γε                                (Σ ρ = 1)
-/

instance : PseudoMetricSpace (S × A → ℝ) :=
  show PseudoMetricSpace (∀ _ : S × A, ℝ) from inferInstance

/--
**Theorem (BA-PR Operator is a γ-Contraction)**

Under:
  * Frozen belief ρ ≥ 0 with Σ ρ = 1  (Bayesian posterior, frozen)
  * Per-mode transitions P_h ≥ 0 with Σ P_h = 1
  * Frozen penalties κ_ale, Γ_epi (from target network)

The weighted-mixture operator T_BAPR is a γ-contraction in L∞.
-/
theorem bapr_contraction (p : Params) (R : H → S → A → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → ℝ) (κ_ale : ℝ)
    (ρ : H → ℝ)
    (hP_nn   : ∀ h s a s', 0 ≤ P h s a s')
    (hP_prob : ∀ h s a, ∑ s' : S, P h s a s' = 1)
    (hρ_nn   : ∀ h, 0 ≤ ρ h)
    (hρ_sum  : ∑ h : H, ρ h = 1) :
    ∃ k < 1, ∀ (Q₁ Q₂ : S × A → ℝ),
    dist (T_BAPR p R P Γ_epi κ_ale ρ Q₁)
         (T_BAPR p R P Γ_epi κ_ale ρ Q₂) ≤ k * dist Q₁ Q₂ := by
  use p.γ
  refine ⟨p.hγ.2, fun Q₁ Q₂ => ?_⟩
  set ε := dist Q₁ Q₂
  have hε_nn : 0 ≤ ε := dist_nonneg
  rw [dist_pi_le_iff (mul_nonneg p.hγ.1 hε_nn)]
  intro ⟨s, a⟩
  simp only [Real.dist_eq]
  -- Pointwise L∞ bound
  have hpw : ∀ sa, |Q₁ sa - Q₂ sa| ≤ ε := by
    intro sa
    have h2 := (dist_pi_le_iff hε_nn).mp (le_refl ε) sa
    simpa [Real.dist_eq] using h2
  -- Unfold T_BAPR
  dsimp only [T_BAPR]
  -- Factor: Σ ρ·T_h(Q₁) − Σ ρ·T_h(Q₂) = Σ ρ·(T_h(Q₁) − T_h(Q₂))
  have h_diff :
    (∑ h : H, ρ h * T_mode p R P Γ_epi κ_ale h Q₁ (s, a)) -
    (∑ h : H, ρ h * T_mode p R P Γ_epi κ_ale h Q₂ (s, a)) =
    ∑ h : H, ρ h * (T_mode p R P Γ_epi κ_ale h Q₁ (s, a) -
                     T_mode p R P Γ_epi κ_ale h Q₂ (s, a)) := by
    rw [← Finset.sum_sub_distrib]
    congr 1; ext h; ring
  rw [h_diff]
  -- Per-mode bound
  have h_mode : ∀ h,
      |T_mode p R P Γ_epi κ_ale h Q₁ (s, a) -
       T_mode p R P Γ_epi κ_ale h Q₂ (s, a)| ≤ p.γ * ε :=
    fun h => t_mode_pointwise_bound p R P Γ_epi κ_ale hP_nn hP_prob hpw h s a
  -- Convex combination argument
  calc |∑ h : H, ρ h * (T_mode p R P Γ_epi κ_ale h Q₁ (s, a) -
                         T_mode p R P Γ_epi κ_ale h Q₂ (s, a))|
      ≤ ∑ h : H, |ρ h * (T_mode p R P Γ_epi κ_ale h Q₁ (s, a) -
                          T_mode p R P Γ_epi κ_ale h Q₂ (s, a))| :=
          Finset.abs_sum_le_sum_abs _ _
    _ = ∑ h : H, ρ h * |T_mode p R P Γ_epi κ_ale h Q₁ (s, a) -
                         T_mode p R P Γ_epi κ_ale h Q₂ (s, a)| := by
          congr 1; ext h
          rw [abs_mul, abs_of_nonneg (hρ_nn h)]
    _ ≤ ∑ h : H, ρ h * (p.γ * ε) := by
          apply Finset.sum_le_sum; intro h _
          exact mul_le_mul_of_nonneg_left (h_mode h) (hρ_nn h)
    _ = p.γ * ε := by
          rw [← Finset.sum_mul, hρ_sum, one_mul]

/-! ## 8. Bridging RE-SAC and BA-PR: κ-Driven Belief Update

This section formally proves that the Bayesian belief update — driven by
RE-SAC's aleatoric risk estimate κ — preserves the probability distribution
properties required by `bapr_contraction`.

### Connection to RE-SAC

RE-SAC produces `κ(s,a)` (aleatoric uncertainty estimate).  In BA-PR, the
"surprise" signal for Bayesian change-point detection is derived from κ:

    surprise = |κ_current − κ_target|

A large surprise indicates the environment has likely changed (κ estimation
under the old mode produces high error), triggering belief redistribution
toward shorter run-lengths.

### Formal Structure

The Bayesian update rule is:

    ρ'(h) = ρ(h) · L(h, σ) / Z     where Z = Σ_h ρ(h) · L(h, σ)

where L(h, σ) is a likelihood function mapping (run-length, surprise) to a
non-negative score.  We prove:

1. `update_belief_nonneg` : ρ' ≥ 0
2. `update_belief_sum_one` : Σ ρ' = 1
3. `bapr_contraction_after_update` : contraction holds for ρ'
-/

/-- Bayesian belief update: reweight prior ρ by likelihood L, then normalize. -/
def update_belief (ρ : H → ℝ) (L : H → ℝ) (Z : ℝ) : H → ℝ :=
  fun h => ρ h * L h / Z

/-- The normalization constant Z = Σ_h ρ(h) · L(h). -/
def normalization_const (ρ : H → ℝ) (L : H → ℝ) : ℝ :=
  ∑ h : H, ρ h * L h

/-- Updated belief is non-negative when ρ ≥ 0, L ≥ 0, Z > 0. -/
lemma update_belief_nonneg (ρ : H → ℝ) (L : H → ℝ) (Z : ℝ)
    (hρ_nn : ∀ h, 0 ≤ ρ h) (hL_nn : ∀ h, 0 ≤ L h) (hZ_pos : 0 < Z) :
    ∀ h, 0 ≤ update_belief ρ L Z h := by
  intro h
  unfold update_belief
  apply div_nonneg
  · exact mul_nonneg (hρ_nn h) (hL_nn h)
  · exact le_of_lt hZ_pos

/-- Updated belief sums to 1 when Z = Σ ρ·L and Z > 0. -/
lemma update_belief_sum_one (ρ : H → ℝ) (L : H → ℝ) (Z : ℝ)
    (hZ_pos : 0 < Z) (hZ_eq : Z = normalization_const ρ L) :
    ∑ h : H, update_belief ρ L Z h = 1 := by
  subst hZ_eq
  unfold update_belief normalization_const
  have hne : (∑ h : H, ρ h * L h) ≠ 0 := ne_of_gt hZ_pos
  have step : ∀ h, ρ h * L h / (∑ h : H, ρ h * L h) =
      ρ h * L h * (∑ h : H, ρ h * L h)⁻¹ := by
    intro h; rw [div_eq_mul_inv]
  simp_rw [step]
  rw [← Finset.sum_mul]
  exact mul_inv_cancel₀ hne

/--
**Corollary: BA-PR contraction holds after κ-driven belief update.**

Since `bapr_contraction` holds for ANY probability distribution ρ, and
the Bayesian update driven by RE-SAC's κ (via surprise → likelihood)
preserves the probability distribution properties, the contraction
theorem automatically applies to the updated belief ρ'.

This formally bridges RE-SAC (κ estimation) and BA-PR (belief-weighted
contraction): RE-SAC's output feeds BA-PR's belief update, and the
mathematical guarantees are preserved through the update.
-/
theorem bapr_contraction_after_update (p : Params) (R : H → S → A → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → ℝ) (κ_ale : ℝ)
    (ρ : H → ℝ) (L : H → ℝ)
    (hP_nn   : ∀ h s a s', 0 ≤ P h s a s')
    (hP_prob : ∀ h s a, ∑ s' : S, P h s a s' = 1)
    (hρ_nn   : ∀ h, 0 ≤ ρ h)
    (hL_nn   : ∀ h, 0 ≤ L h)
    (hZ_pos  : 0 < normalization_const ρ L) :
    let ρ' := update_belief ρ L (normalization_const ρ L)
    ∃ k < 1, ∀ (Q₁ Q₂ : S × A → ℝ),
    dist (T_BAPR p R P Γ_epi κ_ale ρ' Q₁)
         (T_BAPR p R P Γ_epi κ_ale ρ' Q₂) ≤ k * dist Q₁ Q₂ := by
  intro ρ'
  exact bapr_contraction p R P Γ_epi κ_ale ρ' hP_nn hP_prob
    (update_belief_nonneg ρ L _ hρ_nn hL_nn hZ_pos)
    (update_belief_sum_one ρ L _ hZ_pos rfl)

end BAPR
