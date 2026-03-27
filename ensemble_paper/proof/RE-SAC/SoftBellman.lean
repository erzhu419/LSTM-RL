import Mathlib

/-!
# Formal Proof: Entropy-Regularized (Soft) Bellman Operator is a Contraction

## Motivation

RESAC.lean proves contraction for the hard-max operator V(s) = max_a Q(s,a).
However, SAC-family algorithms use the **entropy-regularized** (soft) operator:

    V_soft(s) = τ · log Σ_a exp(Q(s,a) / τ)

where τ > 0 is the temperature parameter.  This file proves that LogSumExp
is also non-expansive (1-Lipschitz in L∞), and therefore the soft Bellman
operator with RE-SAC's frozen penalty terms inherits γ-contraction.

## Key Insight: Exponential Bounding (No Jensen Needed)

    Q₁(s,a) ≤ Q₂(s,a) + ε
    ⟹ exp(Q₁(s,a)/τ) ≤ exp(ε/τ) · exp(Q₂(s,a)/τ)
    ⟹ Σ exp(Q₁/τ) ≤ exp(ε/τ) · Σ exp(Q₂/τ)
    ⟹ τ·log(Σ exp(Q₁/τ)) ≤ ε + τ·log(Σ exp(Q₂/τ))

## Proof Structure

1. `sum_exp_pos`               — Σ exp(·) > 0
2. `soft_max_le_add`           — one-sided LogSumExp bound
3. `soft_max_nonexpansive`     — LogSumExp is 1-Lipschitz in L∞
4. `soft_monotonicity`         — Blackwell condition (i)
5. `soft_discounting`          — Blackwell condition (ii)
6. `soft_bellman_contraction`  — Main theorem: T_soft is a γ-contraction
-/

noncomputable section

set_option linter.unusedSectionVars false

namespace SoftBellman

variable {S A : Type} [Fintype S] [Fintype A] [Nonempty S] [Nonempty A]

/-- Hyperparameters for the entropy-regularized RE-SAC framework. -/
structure Params where
  γ        : ℝ
  lam_epi  : ℝ
  τ        : ℝ
  hγ       : 0 ≤ γ ∧ γ < 1
  hlam_epi : 0 ≤ lam_epi
  hτ       : 0 < τ

-- ============================================================
-- LOGSUMEXP VALUE FUNCTION
-- ============================================================

/-- The soft (LogSumExp) value function:
    V_soft(s) = τ · log(Σ_a exp(Q(s,a) / τ))

    In SAC, this equals E_{a∼π*}[Q(s,a) − τ log π*(a|s)] where
    π*(a|s) ∝ exp(Q(s,a)/τ) is the optimal entropy-regularized policy. -/
def soft_max_over_a (Q : S × A → ℝ) (s : S) (τ : ℝ) : ℝ :=
  τ * Real.log (∑ a : A, Real.exp (Q (s, a) / τ))

/-- The sum of exponentials is strictly positive. -/
lemma sum_exp_pos (Q : S × A → ℝ) (s : S) (τ : ℝ) :
    0 < ∑ a : A, Real.exp (Q (s, a) / τ) := by
  apply Finset.sum_pos
  · intro a _; exact Real.exp_pos _
  · exact Finset.univ_nonempty

-- ============================================================
-- LOGSUMEXP NON-EXPANSIVENESS
-- ============================================================

/-- One-sided LogSumExp bound: if Q₁(s,a) ≤ Q₂(s,a) + ε for all a,
    then soft_max(Q₁, s) ≤ soft_max(Q₂, s) + ε. -/
lemma soft_max_le_add {Q₁ Q₂ : S × A → ℝ} (s : S) {τ : ℝ} (hτ : 0 < τ)
    {ε : ℝ} (hpw : ∀ a, Q₁ (s, a) ≤ Q₂ (s, a) + ε) :
    soft_max_over_a Q₁ s τ ≤ soft_max_over_a Q₂ s τ + ε := by
  unfold soft_max_over_a
  -- Step 1: exp(Q₁(s,a)/τ) ≤ exp(Q₂(s,a)/τ) · exp(ε/τ) for each a
  have h_exp : ∀ a, Real.exp (Q₁ (s, a) / τ) ≤
      Real.exp (Q₂ (s, a) / τ) * Real.exp (ε / τ) := by
    intro a
    rw [← Real.exp_add]
    apply Real.exp_le_exp.mpr
    rw [← add_div]
    exact div_le_div_of_nonneg_right (hpw a) (le_of_lt hτ)
  -- Step 2: Σ exp(Q₁/τ) ≤ exp(ε/τ) · Σ exp(Q₂/τ)
  have h_sum : ∑ a : A, Real.exp (Q₁ (s, a) / τ) ≤
      Real.exp (ε / τ) * ∑ a : A, Real.exp (Q₂ (s, a) / τ) := by
    calc ∑ a, Real.exp (Q₁ (s, a) / τ)
        ≤ ∑ a, Real.exp (Q₂ (s, a) / τ) * Real.exp (ε / τ) :=
          Finset.sum_le_sum fun a _ => h_exp a
      _ = (∑ a, Real.exp (Q₂ (s, a) / τ)) * Real.exp (ε / τ) :=
          (Finset.sum_mul ..).symm
      _ = Real.exp (ε / τ) * ∑ a, Real.exp (Q₂ (s, a) / τ) := mul_comm ..
  -- Step 3: log(Σ exp(Q₁/τ)) ≤ ε/τ + log(Σ exp(Q₂/τ))
  have h_sum1_pos := sum_exp_pos Q₁ s τ
  have h_sum2_pos := sum_exp_pos Q₂ s τ
  have h_log : Real.log (∑ a, Real.exp (Q₁ (s, a) / τ)) ≤
      ε / τ + Real.log (∑ a, Real.exp (Q₂ (s, a) / τ)) := by
    calc Real.log (∑ a, Real.exp (Q₁ (s, a) / τ))
        ≤ Real.log (Real.exp (ε / τ) * ∑ a, Real.exp (Q₂ (s, a) / τ)) :=
          Real.log_le_log h_sum1_pos h_sum
      _ = Real.log (Real.exp (ε / τ)) +
          Real.log (∑ a, Real.exp (Q₂ (s, a) / τ)) :=
          Real.log_mul (ne_of_gt (Real.exp_pos _)) (ne_of_gt h_sum2_pos)
      _ = ε / τ + Real.log (∑ a, Real.exp (Q₂ (s, a) / τ)) := by
          rw [Real.log_exp]
  -- Step 4: τ * log(Σ exp(Q₁/τ)) ≤ τ * (ε/τ + log(Σ exp(Q₂/τ))) = ε + τ * log(Σ exp(Q₂/τ))
  have h4 := mul_le_mul_of_nonneg_left h_log (le_of_lt hτ)
  have hτ_ne : τ ≠ 0 := ne_of_gt hτ
  have hετ : τ * (ε / τ) = ε := mul_div_cancel₀ ε hτ_ne
  nlinarith

/-- **LogSumExp is 1-Lipschitz (non-expansive) in L∞.**
    |soft_max(Q₁, s) − soft_max(Q₂, s)| ≤ ε
    whenever ∀ (s,a), |Q₁(s,a) − Q₂(s,a)| ≤ ε. -/
lemma soft_max_nonexpansive {Q₁ Q₂ : S × A → ℝ} (s : S)
    {τ : ℝ} (hτ : 0 < τ)
    {ε : ℝ} (hpw : ∀ sa, |Q₁ sa - Q₂ sa| ≤ ε) :
    |soft_max_over_a Q₁ s τ - soft_max_over_a Q₂ s τ| ≤ ε := by
  rw [abs_le]
  have hdir1 : soft_max_over_a Q₁ s τ ≤ soft_max_over_a Q₂ s τ + ε :=
    soft_max_le_add s hτ fun a => by
      have := (abs_le.mp (hpw (s, a))).2; linarith
  have hdir2 : soft_max_over_a Q₂ s τ ≤ soft_max_over_a Q₁ s τ + ε :=
    soft_max_le_add (Q₁ := Q₂) (Q₂ := Q₁) s hτ fun a => by
      have := (abs_le.mp (hpw (s, a))).1; linarith
  constructor <;> linarith

-- ============================================================
-- THE SOFT RE-SAC OPERATOR
-- ============================================================

/-- The entropy-regularized RE-SAC Bellman operator.
    Replaces hard max with LogSumExp; penalty terms unchanged. -/
def T_soft (p : Params) (R : S → A → ℝ) (P : S → A → S → ℝ)
    (Γ_epi : S → A → ℝ) (κ_ale : ℝ)
    (Q : S × A → ℝ) : S × A → ℝ := fun ⟨s, a⟩ =>
  R s a + p.γ * (
    (∑ s' : S, P s a s' * soft_max_over_a Q s' p.τ) -
    p.lam_epi * Γ_epi s a -
    κ_ale
  )

-- ============================================================
-- LEMMA 1: Monotonicity (Blackwell Condition i)
-- ============================================================

/-- soft_max_over_a is monotone in Q. -/
lemma soft_max_mono {Q₁ Q₂ : S × A → ℝ} (h : Q₁ ≤ Q₂) (s : S)
    {τ : ℝ} (hτ : 0 < τ) :
    soft_max_over_a Q₁ s τ ≤ soft_max_over_a Q₂ s τ := by
  unfold soft_max_over_a
  apply mul_le_mul_of_nonneg_left _ (le_of_lt hτ)
  apply Real.log_le_log (sum_exp_pos Q₁ s τ)
  apply Finset.sum_le_sum
  intro a _
  exact Real.exp_le_exp.mpr (div_le_div_of_nonneg_right (h (s, a)) (le_of_lt hτ))

/-- **Lemma 1 (Monotonicity)**: If Q₁ ≤ Q₂, then T_soft(Q₁) ≤ T_soft(Q₂). -/
lemma soft_monotonicity (p : Params) (R : S → A → ℝ) (P : S → A → S → ℝ)
    (Γ_epi : S → A → ℝ) (κ_ale : ℝ)
    (hP_nn : ∀ s a s', 0 ≤ P s a s')
    (Q₁ Q₂ : S × A → ℝ) (h : Q₁ ≤ Q₂) :
    T_soft p R P Γ_epi κ_ale Q₁ ≤ T_soft p R P Γ_epi κ_ale Q₂ := by
  intro ⟨s, a⟩
  dsimp only [T_soft]
  have hsum : ∑ s' : S, P s a s' * soft_max_over_a Q₁ s' p.τ ≤
              ∑ s' : S, P s a s' * soft_max_over_a Q₂ s' p.τ := by
    apply Finset.sum_le_sum
    intro s' _
    exact mul_le_mul_of_nonneg_left (soft_max_mono h s' p.hτ) (hP_nn s a s')
  nlinarith [p.hγ.1]

-- ============================================================
-- LEMMA 2: Discounting (Blackwell Condition ii)
-- ============================================================

/-- soft_max distributes over constant addition:
    soft_max(Q + c, s) = soft_max(Q, s) + c. -/
lemma soft_max_add_const (Q : S × A → ℝ) (c : ℝ) (s : S)
    {τ : ℝ} (hτ : 0 < τ) :
    soft_max_over_a (Q + fun _ => c) s τ = soft_max_over_a Q s τ + c := by
  unfold soft_max_over_a
  simp only [Pi.add_apply]
  -- (Q(s,a) + c) / τ = Q(s,a)/τ + c/τ
  have hτ_ne : τ ≠ 0 := ne_of_gt hτ
  have h_shift : ∀ a, (Q (s, a) + c) / τ = Q (s, a) / τ + c / τ := by
    intro a; rw [add_div]
  simp_rw [h_shift, Real.exp_add]
  -- Factor out exp(c/τ): Σ exp(Q/τ) * exp(c/τ) = exp(c/τ) * Σ exp(Q/τ)
  rw [← Finset.sum_mul]
  rw [Real.log_mul (ne_of_gt (sum_exp_pos Q s τ)) (ne_of_gt (Real.exp_pos _))]
  rw [Real.log_exp]
  have hτ_ne : τ ≠ 0 := ne_of_gt hτ
  have : τ * (Real.log (∑ x, Real.exp (Q (s, x) / τ)) + c / τ) =
    τ * Real.log (∑ x, Real.exp (Q (s, x) / τ)) + c := by
    rw [mul_add, mul_div_cancel₀ c hτ_ne]
  linarith

/-- **Lemma 2 (Discounting)**: T_soft(Q + c) = T_soft(Q) + γ·c. -/
lemma soft_discounting (p : Params) (R : S → A → ℝ) (P : S → A → S → ℝ)
    (Γ_epi : S → A → ℝ) (κ_ale : ℝ)
    (hP_prob : ∀ s a, ∑ s' : S, P s a s' = 1)
    (Q : S × A → ℝ) (c : ℝ) :
    T_soft p R P Γ_epi κ_ale (Q + fun _ => c) =
    T_soft p R P Γ_epi κ_ale Q + fun _ => p.γ * c := by
  funext ⟨s, a⟩
  simp only [T_soft, Pi.add_apply]
  have hmax : ∀ s', soft_max_over_a (Q + fun _ => c) s' p.τ =
      soft_max_over_a Q s' p.τ + c :=
    fun s' => soft_max_add_const Q c s' p.hτ
  have hsum : ∑ s', P s a s' * soft_max_over_a (Q + fun _ => c) s' p.τ =
      (∑ s', P s a s' * soft_max_over_a Q s' p.τ) + c := by
    simp_rw [hmax, mul_add]
    rw [Finset.sum_add_distrib]
    congr 1
    rw [← Finset.sum_mul, hP_prob s a, one_mul]
  rw [hsum]; ring

-- ============================================================
-- THEOREM: Contraction Mapping
-- ============================================================

instance : PseudoMetricSpace (S × A → ℝ) :=
  show PseudoMetricSpace (∀ _ : S × A, ℝ) from inferInstance

/--
**Theorem (Soft RE-SAC Operator is a γ-Contraction)**

Under:
  * (A1) P(s'|s,a) ≥ 0
  * (A2) Σ_{s'} P(s'|s,a) = 1
  * κ_ale fixed (independent of Q)
  * τ > 0

The entropy-regularized operator T_soft is a γ-contraction in L∞.
The proof is structurally identical to the hard-max case (RESAC.lean),
replacing max_over_a_nonexpansive with soft_max_nonexpansive.
-/
theorem soft_bellman_contraction (p : Params) (R : S → A → ℝ)
    (P : S → A → S → ℝ) (Γ_epi : S → A → ℝ) (κ_ale : ℝ)
    (hP_nn   : ∀ s a s', 0 ≤ P s a s')
    (hP_prob : ∀ s a, ∑ s' : S, P s a s' = 1) :
    ∃ k < 1, ∀ (Q₁ Q₂ : S × A → ℝ),
    dist (T_soft p R P Γ_epi κ_ale Q₁)
         (T_soft p R P Γ_epi κ_ale Q₂) ≤ k * dist Q₁ Q₂ := by
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
  -- T_soft(Q₁)(s,a) − T_soft(Q₂)(s,a) = γ · Σ P·(softmax Q₁ − softmax Q₂)
  -- (κ_ale and Γ_epi cancel exactly)
  dsimp only [T_soft]
  have hS : ∑ s' : S, P s a s' * soft_max_over_a Q₁ s' p.τ -
            ∑ s' : S, P s a s' * soft_max_over_a Q₂ s' p.τ =
            ∑ s' : S, P s a s' *
              (soft_max_over_a Q₁ s' p.τ - soft_max_over_a Q₂ s' p.τ) := by
    rw [← Finset.sum_sub_distrib]
    congr 1; ext s'; ring
  have hsimp : R s a + p.γ * (∑ s', P s a s' * soft_max_over_a Q₁ s' p.τ -
      p.lam_epi * Γ_epi s a - κ_ale) -
    (R s a + p.γ * (∑ s', P s a s' * soft_max_over_a Q₂ s' p.τ -
      p.lam_epi * Γ_epi s a - κ_ale)) =
    p.γ * ∑ s', P s a s' *
      (soft_max_over_a Q₁ s' p.τ - soft_max_over_a Q₂ s' p.τ) := by
    linear_combination p.γ * hS
  rw [hsimp, abs_mul, abs_of_nonneg p.hγ.1]
  apply mul_le_mul_of_nonneg_left _ p.hγ.1
  -- Bound |Σ P·(softmax Q₁ − softmax Q₂)| ≤ ε
  calc |∑ s' : S, P s a s' *
        (soft_max_over_a Q₁ s' p.τ - soft_max_over_a Q₂ s' p.τ)|
      ≤ ∑ s' : S, |P s a s' *
          (soft_max_over_a Q₁ s' p.τ - soft_max_over_a Q₂ s' p.τ)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ = ∑ s' : S, P s a s' *
          |soft_max_over_a Q₁ s' p.τ - soft_max_over_a Q₂ s' p.τ| := by
        congr 1; ext s'
        rw [abs_mul, abs_of_nonneg (hP_nn s a s')]
    _ ≤ ∑ s' : S, P s a s' * ε := by
        apply Finset.sum_le_sum; intro s' _
        exact mul_le_mul_of_nonneg_left
          (soft_max_nonexpansive s' p.hτ hpw) (hP_nn s a s')
    _ = ε := by rw [← Finset.sum_mul, hP_prob s a, one_mul]

end SoftBellman
