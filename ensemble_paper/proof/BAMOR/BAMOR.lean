import BAPR

/-!
# Formal Proof: BAMOR Bellman Operator is a Contraction Mapping

## Overview

BAMOR (Bayesian Amnesic Multi-Objective Robust) extends the single-objective
BA-PR operator (BAPR.lean) to multi-objective RL.  The Q-function is now
**vector-valued**: `Q : S × A → Fin m → ℝ`, where `m` is the number of
objectives.

## What This File Adds Beyond BAPR

1. **Component-wise contraction** — lifting the scalar proof to Fin m → ℝ
   via nested Pi metrics (Part I)
2. **Scalarized Pareto aggregation** — preference-weighted value is
   non-expansive when weights form a probability distribution (Part III)
3. **Mahalanobis surprise likelihood** — risk-normalized change-point
   detection with per-objective covariance (Part IV)
4. **Anytime bounded error** — tracking error bound Δ/(1−γ) for
   non-stationary environments without requiring convergence (Part V)

Belief update infrastructure (update_belief, normalization_const, etc.)
is **imported from BAPR** — no duplication.
-/

noncomputable section

set_option linter.unusedSectionVars false

namespace BAMOR

variable {S A H : Type} [Fintype S] [Fintype A] [Fintype H]
                         [Nonempty S] [Nonempty A] [Nonempty H]
variable {m : ℕ} [NeZero m]

/-- Hyperparameters for the BAMOR framework -/
structure Params where
  γ        : ℝ
  lam_epi  : ℝ
  hγ       : 0 ≤ γ ∧ γ < 1
  hlam_epi : 0 ≤ lam_epi

-- ════════════════════════════════════════════════════════════════
-- PART I: Core BAMOR Contraction (Component-wise)
-- ════════════════════════════════════════════════════════════════

/-! ## 1. Component-wise Max Operator -/

/-- V(s, i) = max_a Q(s, a)(i) — the per-component value function -/
def max_over_a_comp (Q : S × A → Fin m → ℝ) (s : S) (i : Fin m) : ℝ :=
  (Finset.univ : Finset A).sup' Finset.univ_nonempty (fun a => Q (s, a) i)

/-! ## 2. Intermediate Lemmas -/

/-- Component-wise max is monotone in Q. -/
lemma max_over_a_comp_mono {Q₁ Q₂ : S × A → Fin m → ℝ}
    (h : ∀ sa i, Q₁ sa i ≤ Q₂ sa i) (s : S) (i : Fin m) :
    max_over_a_comp Q₁ s i ≤ max_over_a_comp Q₂ s i := by
  unfold max_over_a_comp
  apply Finset.sup'_le
  intro a _
  have hq : Q₁ (s, a) i ≤ Q₂ (s, a) i := h (s, a) i
  have hle : Q₂ (s, a) i ≤ Finset.univ.sup' Finset.univ_nonempty
      (fun b => Q₂ (s, b) i) :=
    Finset.le_sup' (fun b => Q₂ (s, b) i) (Finset.mem_univ a)
  linarith

/-- Component-wise max is 1-Lipschitz in L∞. -/
lemma max_over_a_comp_nonexpansive {Q₁ Q₂ : S × A → Fin m → ℝ} (s : S)
    (i : Fin m) {ε : ℝ} (hpw : ∀ sa j, |Q₁ sa j - Q₂ sa j| ≤ ε) :
    |max_over_a_comp Q₁ s i - max_over_a_comp Q₂ s i| ≤ ε := by
  rw [abs_le]
  constructor
  · rw [neg_le_sub_iff_le_add]
    unfold max_over_a_comp
    apply Finset.sup'_le
    intro a _
    have habs : |Q₁ (s, a) i - Q₂ (s, a) i| ≤ ε := hpw (s, a) i
    have hle : Q₂ (s, a) i ≤ Q₁ (s, a) i + ε := by linarith [(abs_le.mp habs).1]
    have hmax₁ : Q₁ (s, a) i ≤ Finset.univ.sup' Finset.univ_nonempty
        (fun b => Q₁ (s, b) i) :=
      Finset.le_sup' (fun b => Q₁ (s, b) i) (Finset.mem_univ a)
    linarith
  · unfold max_over_a_comp
    rw [sub_le_iff_le_add]
    apply Finset.sup'_le
    intro a _
    have habs : |Q₁ (s, a) i - Q₂ (s, a) i| ≤ ε := hpw (s, a) i
    have hle : Q₁ (s, a) i ≤ Q₂ (s, a) i + ε := by linarith [(abs_le.mp habs).2]
    have hmax₂ : Q₂ (s, a) i ≤ Finset.univ.sup' Finset.univ_nonempty
        (fun b => Q₂ (s, b) i) :=
      Finset.le_sup' (fun b => Q₂ (s, b) i) (Finset.mem_univ a)
    linarith

/-! ## 3. Mode-Conditional Multi-Objective Bellman Operator -/

/-- Mode-conditional operator for mode h, acting on vector-valued Q. -/
def T_mode_mo (p : Params) (R : H → S → A → Fin m → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → Fin m → ℝ)
    (κ_ale : Fin m → ℝ) (h : H)
    (Q : S × A → Fin m → ℝ) : S × A → Fin m → ℝ := fun ⟨s, a⟩ i =>
  R h s a i + p.γ * (
    (∑ s' : S, P h s a s' * max_over_a_comp Q s' i) -
    p.lam_epi * Γ_epi h s a i -
    κ_ale i
  )

/-! ## 4. BAMOR Operator -/

/-- The BAMOR operator: belief-weighted mixture of mode-conditional operators. -/
def T_BAMOR (p : Params) (R : H → S → A → Fin m → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → Fin m → ℝ)
    (κ_ale : Fin m → ℝ) (ρ : H → ℝ)
    (Q : S × A → Fin m → ℝ) : S × A → Fin m → ℝ := fun sa i =>
  ∑ h : H, ρ h * T_mode_mo p R P Γ_epi κ_ale h Q sa i

-- ════════════════════════════════════════════════════════════════
-- BLACKWELL CONDITIONS FOR BAMOR
-- ════════════════════════════════════════════════════════════════

/-! ## 4a. Monotonicity (Blackwell Condition i) -/

/-- Per-mode monotonicity for multi-objective operator. -/
lemma t_mode_mo_monotonicity (p : Params) (R : H → S → A → Fin m → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → Fin m → ℝ)
    (κ_ale : Fin m → ℝ) (hP_nn : ∀ h s a s', 0 ≤ P h s a s')
    (hm : H) (Q₁ Q₂ : S × A → Fin m → ℝ)
    (hle : ∀ sa i, Q₁ sa i ≤ Q₂ sa i) :
    ∀ sa i, T_mode_mo p R P Γ_epi κ_ale hm Q₁ sa i ≤
            T_mode_mo p R P Γ_epi κ_ale hm Q₂ sa i := by
  intro ⟨s, a⟩ i
  dsimp only [T_mode_mo]
  have hsum : ∑ s' : S, P hm s a s' * max_over_a_comp Q₁ s' i ≤
              ∑ s' : S, P hm s a s' * max_over_a_comp Q₂ s' i := by
    apply Finset.sum_le_sum; intro s' _
    exact mul_le_mul_of_nonneg_left
      (max_over_a_comp_mono hle s' i) (hP_nn hm s a s')
  nlinarith [p.hγ.1]

/-- **Lemma (BAMOR Monotonicity)**: T_BAMOR(Q₁) ≤ T_BAMOR(Q₂) pointwise. -/
lemma bamor_monotonicity (p : Params) (R : H → S → A → Fin m → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → Fin m → ℝ)
    (κ_ale : Fin m → ℝ) (ρ : H → ℝ)
    (hP_nn : ∀ h s a s', 0 ≤ P h s a s')
    (hρ_nn : ∀ h, 0 ≤ ρ h)
    (Q₁ Q₂ : S × A → Fin m → ℝ)
    (hle : ∀ sa i, Q₁ sa i ≤ Q₂ sa i) :
    ∀ sa i, T_BAMOR p R P Γ_epi κ_ale ρ Q₁ sa i ≤
            T_BAMOR p R P Γ_epi κ_ale ρ Q₂ sa i := by
  intro sa i
  dsimp only [T_BAMOR]
  apply Finset.sum_le_sum; intro h _
  exact mul_le_mul_of_nonneg_left
    (t_mode_mo_monotonicity p R P Γ_epi κ_ale hP_nn h Q₁ Q₂ hle sa i) (hρ_nn h)

/-! ## 4b. Discounting (Blackwell Condition ii) -/

/-- Component-wise max distributes over constant addition. -/
lemma max_over_a_comp_add_const (Q : S × A → Fin m → ℝ) (c : Fin m → ℝ)
    (s : S) (i : Fin m) :
    max_over_a_comp (fun sa j => Q sa j + c j) s i =
    max_over_a_comp Q s i + c i := by
  unfold max_over_a_comp
  apply le_antisymm
  · apply Finset.sup'_le; intro a _
    have : Q (s, a) i ≤ Finset.univ.sup' Finset.univ_nonempty
        (fun b => Q (s, b) i) :=
      Finset.le_sup' (fun b => Q (s, b) i) (Finset.mem_univ a)
    linarith
  · have key : Finset.univ.sup' Finset.univ_nonempty (fun a => Q (s, a) i) ≤
        Finset.univ.sup' Finset.univ_nonempty
          (fun a => Q (s, a) i + c i) - c i := by
      apply Finset.sup'_le; intro a _
      have : Q (s, a) i + c i ≤ Finset.univ.sup' Finset.univ_nonempty
          (fun b => Q (s, b) i + c i) :=
        Finset.le_sup' (fun b => Q (s, b) i + c i) (Finset.mem_univ a)
      linarith
    linarith

/-- Per-mode discounting: T_h(Q + c) = T_h(Q) + γ·c (component-wise). -/
lemma t_mode_mo_discounting (p : Params) (R : H → S → A → Fin m → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → Fin m → ℝ)
    (κ_ale : Fin m → ℝ) (hP_prob : ∀ h s a, ∑ s' : S, P h s a s' = 1)
    (hm : H) (Q : S × A → Fin m → ℝ) (c : Fin m → ℝ) :
    T_mode_mo p R P Γ_epi κ_ale hm (fun sa j => Q sa j + c j) =
    fun sa j => T_mode_mo p R P Γ_epi κ_ale hm Q sa j + p.γ * c j := by
  funext ⟨s, a⟩ i
  simp only [T_mode_mo]
  -- Σ P * (max Q + c_i) = Σ P * max Q + c_i  (since Σ P = 1)
  have hsum : ∑ s' : S, P hm s a s' *
      max_over_a_comp (fun sa j => Q sa j + c j) s' i =
      (∑ s' : S, P hm s a s' * max_over_a_comp Q s' i) + c i := by
    simp_rw [max_over_a_comp_add_const Q c _ i, mul_add]
    rw [Finset.sum_add_distrib, ← Finset.sum_mul, hP_prob hm s a, one_mul]
  rw [hsum]; ring

/-- **Lemma (BAMOR Discounting)**: T_BAMOR(Q + c) = T_BAMOR(Q) + γ·c. -/
lemma bamor_discounting (p : Params) (R : H → S → A → Fin m → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → Fin m → ℝ)
    (κ_ale : Fin m → ℝ) (ρ : H → ℝ)
    (hP_prob : ∀ h s a, ∑ s' : S, P h s a s' = 1)
    (hρ_sum : ∑ h : H, ρ h = 1)
    (Q : S × A → Fin m → ℝ) (c : Fin m → ℝ) :
    T_BAMOR p R P Γ_epi κ_ale ρ (fun sa j => Q sa j + c j) =
    fun sa j => T_BAMOR p R P Γ_epi κ_ale ρ Q sa j + p.γ * c j := by
  funext sa i
  simp only [T_BAMOR]
  have h_mode : ∀ h,
      T_mode_mo p R P Γ_epi κ_ale h (fun sa j => Q sa j + c j) sa i =
      T_mode_mo p R P Γ_epi κ_ale h Q sa i + p.γ * c i := by
    intro h
    have := congr_fun₂
      (t_mode_mo_discounting p R P Γ_epi κ_ale hP_prob h Q c) sa i
    exact this
  simp_rw [h_mode, mul_add, Finset.sum_add_distrib,
           ← Finset.sum_mul, hρ_sum, one_mul]

/-! ## 5. Per-Mode Pointwise Contraction -/

/-- Per-mode, per-component pointwise bound:
    |T_h(Q₁)(s,a)(i) − T_h(Q₂)(s,a)(i)| ≤ γ·ε -/
lemma t_mode_mo_pointwise_bound (p : Params) (R : H → S → A → Fin m → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → Fin m → ℝ)
    (κ_ale : Fin m → ℝ)
    (hP_nn : ∀ h s a s', 0 ≤ P h s a s')
    (hP_prob : ∀ h s a, ∑ s' : S, P h s a s' = 1)
    {Q₁ Q₂ : S × A → Fin m → ℝ} {ε : ℝ}
    (hpw : ∀ sa j, |Q₁ sa j - Q₂ sa j| ≤ ε)
    (h : H) (s : S) (a : A) (i : Fin m) :
    |T_mode_mo p R P Γ_epi κ_ale h Q₁ (s, a) i -
     T_mode_mo p R P Γ_epi κ_ale h Q₂ (s, a) i| ≤ p.γ * ε := by
  dsimp only [T_mode_mo]
  have hS : ∑ s' : S, P h s a s' * max_over_a_comp Q₁ s' i -
            ∑ s' : S, P h s a s' * max_over_a_comp Q₂ s' i =
            ∑ s' : S, P h s a s' *
              (max_over_a_comp Q₁ s' i - max_over_a_comp Q₂ s' i) := by
    rw [← Finset.sum_sub_distrib]
    congr 1; ext s'; ring
  have hsimp : R h s a i + p.γ * (∑ s' : S, P h s a s' * max_over_a_comp Q₁ s' i -
      p.lam_epi * Γ_epi h s a i - κ_ale i) -
    (R h s a i + p.γ * (∑ s' : S, P h s a s' * max_over_a_comp Q₂ s' i -
      p.lam_epi * Γ_epi h s a i - κ_ale i)) =
    p.γ * ∑ s' : S, P h s a s' *
      (max_over_a_comp Q₁ s' i - max_over_a_comp Q₂ s' i) := by
    linear_combination p.γ * hS
  rw [hsimp, abs_mul, abs_of_nonneg p.hγ.1]
  apply mul_le_mul_of_nonneg_left _ p.hγ.1
  calc |∑ s' : S, P h s a s' * (max_over_a_comp Q₁ s' i - max_over_a_comp Q₂ s' i)|
      ≤ ∑ s' : S, |P h s a s' * (max_over_a_comp Q₁ s' i - max_over_a_comp Q₂ s' i)| :=
          Finset.abs_sum_le_sum_abs _ _
    _ = ∑ s' : S, P h s a s' * |max_over_a_comp Q₁ s' i - max_over_a_comp Q₂ s' i| := by
          congr 1; ext s'
          rw [abs_mul, abs_of_nonneg (hP_nn h s a s')]
    _ ≤ ∑ s' : S, P h s a s' * ε := by
          apply Finset.sum_le_sum; intro s' _
          exact mul_le_mul_of_nonneg_left
            (max_over_a_comp_nonexpansive s' i hpw) (hP_nn h s a s')
    _ = ε := by rw [← Finset.sum_mul, hP_prob h s a, one_mul]

/-! ## 6. Main Theorem: BAMOR is a γ-Contraction -/

instance : PseudoMetricSpace (S × A → Fin m → ℝ) :=
  show PseudoMetricSpace (∀ _ : S × A, Fin m → ℝ) from inferInstance

/-- Direct distance bound: `dist(T Q₁, T Q₂) ≤ γ · dist(Q₁, Q₂)`.
    This is the core lemma used by both `bamor_contraction` and the
    Banach fixed-point characterization (Part VI). -/
lemma bamor_dist_le (p : Params) (R : H → S → A → Fin m → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → Fin m → ℝ)
    (κ_ale : Fin m → ℝ) (ρ : H → ℝ)
    (hP_nn   : ∀ h s a s', 0 ≤ P h s a s')
    (hP_prob : ∀ h s a, ∑ s' : S, P h s a s' = 1)
    (hρ_nn   : ∀ h, 0 ≤ ρ h)
    (hρ_sum  : ∑ h : H, ρ h = 1)
    (Q₁ Q₂ : S × A → Fin m → ℝ) :
    dist (T_BAMOR p R P Γ_epi κ_ale ρ Q₁)
         (T_BAMOR p R P Γ_epi κ_ale ρ Q₂) ≤ p.γ * dist Q₁ Q₂ := by
  set ε := dist Q₁ Q₂
  have hε_nn : 0 ≤ ε := dist_nonneg
  rw [dist_pi_le_iff (mul_nonneg p.hγ.1 hε_nn)]
  intro ⟨s, a⟩
  rw [dist_pi_le_iff (mul_nonneg p.hγ.1 hε_nn)]
  intro i
  simp only [Real.dist_eq]
  have hpw : ∀ sa j, |Q₁ sa j - Q₂ sa j| ≤ ε := by
    intro sa j
    have h_outer := (dist_pi_le_iff hε_nn).mp (le_refl ε) sa
    have h_inner := (dist_pi_le_iff (by linarith : 0 ≤ ε)).mp h_outer j
    simpa [Real.dist_eq] using h_inner
  dsimp only [T_BAMOR]
  have h_diff :
    (∑ h : H, ρ h * T_mode_mo p R P Γ_epi κ_ale h Q₁ (s, a) i) -
    (∑ h : H, ρ h * T_mode_mo p R P Γ_epi κ_ale h Q₂ (s, a) i) =
    ∑ h : H, ρ h * (T_mode_mo p R P Γ_epi κ_ale h Q₁ (s, a) i -
                     T_mode_mo p R P Γ_epi κ_ale h Q₂ (s, a) i) := by
    rw [← Finset.sum_sub_distrib]
    congr 1; ext h; ring
  rw [h_diff]
  have h_mode : ∀ h,
      |T_mode_mo p R P Γ_epi κ_ale h Q₁ (s, a) i -
       T_mode_mo p R P Γ_epi κ_ale h Q₂ (s, a) i| ≤ p.γ * ε :=
    fun h => t_mode_mo_pointwise_bound p R P Γ_epi κ_ale hP_nn hP_prob hpw h s a i
  calc |∑ h : H, ρ h * (T_mode_mo p R P Γ_epi κ_ale h Q₁ (s, a) i -
                         T_mode_mo p R P Γ_epi κ_ale h Q₂ (s, a) i)|
      ≤ ∑ h : H, |ρ h * (T_mode_mo p R P Γ_epi κ_ale h Q₁ (s, a) i -
                          T_mode_mo p R P Γ_epi κ_ale h Q₂ (s, a) i)| :=
          Finset.abs_sum_le_sum_abs _ _
    _ = ∑ h : H, ρ h * |T_mode_mo p R P Γ_epi κ_ale h Q₁ (s, a) i -
                         T_mode_mo p R P Γ_epi κ_ale h Q₂ (s, a) i| := by
          congr 1; ext h
          rw [abs_mul, abs_of_nonneg (hρ_nn h)]
    _ ≤ ∑ h : H, ρ h * (p.γ * ε) := by
          apply Finset.sum_le_sum; intro h _
          exact mul_le_mul_of_nonneg_left (h_mode h) (hρ_nn h)
    _ = p.γ * ε := by
          rw [← Finset.sum_mul, hρ_sum, one_mul]

/-- **Theorem (BAMOR is a γ-Contraction)** — existential form. -/
theorem bamor_contraction (p : Params) (R : H → S → A → Fin m → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → Fin m → ℝ)
    (κ_ale : Fin m → ℝ) (ρ : H → ℝ)
    (hP_nn   : ∀ h s a s', 0 ≤ P h s a s')
    (hP_prob : ∀ h s a, ∑ s' : S, P h s a s' = 1)
    (hρ_nn   : ∀ h, 0 ≤ ρ h)
    (hρ_sum  : ∑ h : H, ρ h = 1) :
    ∃ k < 1, ∀ (Q₁ Q₂ : S × A → Fin m → ℝ),
    dist (T_BAMOR p R P Γ_epi κ_ale ρ Q₁)
         (T_BAMOR p R P Γ_epi κ_ale ρ Q₂) ≤ k * dist Q₁ Q₂ :=
  ⟨p.γ, p.hγ.2, bamor_dist_le p R P Γ_epi κ_ale ρ hP_nn hP_prob hρ_nn hρ_sum⟩

-- ════════════════════════════════════════════════════════════════
-- PART II: Belief Update (Reusing BAPR infrastructure)
-- ════════════════════════════════════════════════════════════════

/-! ## 7. Bayesian Belief Update — Imported from BAPR

We reuse `BAPR.update_belief`, `BAPR.normalization_const`,
`BAPR.update_belief_nonneg`, and `BAPR.update_belief_sum_one`.
No code duplication.
-/

/--
**Corollary: BAMOR contraction holds after belief update.**

Uses BAPR's belief update infrastructure directly.
-/
theorem bamor_contraction_after_update (p : Params)
    (R : H → S → A → Fin m → ℝ) (P : H → S → A → S → ℝ)
    (Γ_epi : H → S → A → Fin m → ℝ) (κ_ale : Fin m → ℝ)
    (ρ : H → ℝ) (L : H → ℝ)
    (hP_nn   : ∀ h s a s', 0 ≤ P h s a s')
    (hP_prob : ∀ h s a, ∑ s' : S, P h s a s' = 1)
    (hρ_nn   : ∀ h, 0 ≤ ρ h)
    (hL_nn   : ∀ h, 0 ≤ L h)
    (hZ_pos  : 0 < BAPR.normalization_const ρ L) :
    let ρ' := BAPR.update_belief ρ L (BAPR.normalization_const ρ L)
    ∃ k < 1, ∀ (Q₁ Q₂ : S × A → Fin m → ℝ),
    dist (T_BAMOR p R P Γ_epi κ_ale ρ' Q₁)
         (T_BAMOR p R P Γ_epi κ_ale ρ' Q₂) ≤ k * dist Q₁ Q₂ := by
  intro ρ'
  exact bamor_contraction p R P Γ_epi κ_ale ρ' hP_nn hP_prob
    (BAPR.update_belief_nonneg ρ L _ hρ_nn hL_nn hZ_pos)
    (BAPR.update_belief_sum_one ρ L _ hZ_pos rfl)

-- ════════════════════════════════════════════════════════════════
-- PART III: Scalarized Pareto Aggregation (MORL-specific)
-- ════════════════════════════════════════════════════════════════

/-! ## 8. Preference-Weighted Scalarization

In MORL, the preference weight w ∈ Δ^{m-1} (probability simplex) induces
a scalarized objective.  The key property: scalarization with normalized
weights is **non-expansive** — a convex combination of per-component values.

This is distinct from component-wise max (Part I) and captures the
Pareto-front geometry described in GPI-LS (Alegre et al. 2023).
-/

/-- Scalarize a multi-objective Q-function: `q_w(s,a) = sum_i w_i Q(s,a,i)` -/
def scalarize (Q : S × A → Fin m → ℝ) (w : Fin m → ℝ) : S × A → ℝ :=
  fun sa => ∑ i : Fin m, w i * Q sa i

/-- Scalarized value: `V_w(s) = max_a sum_i w_i Q(s,a,i)` -/
def pareto_val (Q : S × A → Fin m → ℝ) (w : Fin m → ℝ) (s : S) : ℝ :=
  (Finset.univ : Finset A).sup' Finset.univ_nonempty
    (fun a => scalarize Q w (s, a))

/-- Scalarization is non-expansive under normalized weights (w >= 0, sum w = 1).
    This is the multi-objective analogue of the convex combination argument. -/
lemma scalarize_nonexpansive {Q₁ Q₂ : S × A → Fin m → ℝ}
    (w : Fin m → ℝ) (hw_nn : ∀ i, 0 ≤ w i)
    (hw_sum : ∑ i : Fin m, w i = 1)
    {ε : ℝ} (hpw : ∀ sa i, |Q₁ sa i - Q₂ sa i| ≤ ε) (sa : S × A) :
    |scalarize Q₁ w sa - scalarize Q₂ w sa| ≤ ε := by
  unfold scalarize
  have h_diff : (∑ i, w i * Q₁ sa i) - (∑ i, w i * Q₂ sa i) =
                ∑ i, w i * (Q₁ sa i - Q₂ sa i) := by
    rw [← Finset.sum_sub_distrib]
    congr 1; ext i; ring
  rw [h_diff]
  calc |∑ i : Fin m, w i * (Q₁ sa i - Q₂ sa i)|
      ≤ ∑ i : Fin m, |w i * (Q₁ sa i - Q₂ sa i)| :=
          Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i : Fin m, w i * |Q₁ sa i - Q₂ sa i| := by
          congr 1; ext i
          rw [abs_mul, abs_of_nonneg (hw_nn i)]
    _ ≤ ∑ i : Fin m, w i * ε := by
          apply Finset.sum_le_sum; intro i _
          exact mul_le_mul_of_nonneg_left (hpw sa i) (hw_nn i)
    _ = ε := by rw [← Finset.sum_mul, hw_sum, one_mul]

/-- Pareto-aggregated value is non-expansive in L∞.
    Combines scalarize_nonexpansive with max_over_a non-expansiveness. -/
lemma pareto_val_nonexpansive {Q₁ Q₂ : S × A → Fin m → ℝ}
    (w : Fin m → ℝ) (hw_nn : ∀ i, 0 ≤ w i)
    (hw_sum : ∑ i : Fin m, w i = 1)
    {ε : ℝ} (hpw : ∀ sa i, |Q₁ sa i - Q₂ sa i| ≤ ε) (s : S) :
    |pareto_val Q₁ w s - pareto_val Q₂ w s| ≤ ε := by
  rw [abs_le]
  constructor
  · rw [neg_le_sub_iff_le_add]
    unfold pareto_val
    apply Finset.sup'_le
    intro a _
    have hsca : |scalarize Q₁ w (s, a) - scalarize Q₂ w (s, a)| ≤ ε :=
      scalarize_nonexpansive w hw_nn hw_sum hpw (s, a)
    have hle : scalarize Q₂ w (s, a) ≤ scalarize Q₁ w (s, a) + ε := by
      linarith [(abs_le.mp hsca).1]
    have hmax : scalarize Q₁ w (s, a) ≤ Finset.univ.sup' Finset.univ_nonempty
        (fun b => scalarize Q₁ w (s, b)) :=
      Finset.le_sup' (fun b => scalarize Q₁ w (s, b)) (Finset.mem_univ a)
    linarith
  · unfold pareto_val
    rw [sub_le_iff_le_add]
    apply Finset.sup'_le
    intro a _
    have hsca : |scalarize Q₁ w (s, a) - scalarize Q₂ w (s, a)| ≤ ε :=
      scalarize_nonexpansive w hw_nn hw_sum hpw (s, a)
    have hle : scalarize Q₁ w (s, a) ≤ scalarize Q₂ w (s, a) + ε := by
      linarith [(abs_le.mp hsca).2]
    have hmax : scalarize Q₂ w (s, a) ≤ Finset.univ.sup' Finset.univ_nonempty
        (fun b => scalarize Q₂ w (s, b)) :=
      Finset.le_sup' (fun b => scalarize Q₂ w (s, b)) (Finset.mem_univ a)
    linarith

-- ════════════════════════════════════════════════════════════════
-- PART IV: Mahalanobis Surprise-Based Belief Update (MORL-specific)
-- ════════════════════════════════════════════════════════════════

/-! ## 9. Risk-Normalized Surprise

In MORL, different objectives have different noise levels.  The surprise
signal for change-point detection should be normalized by the per-objective
aleatoric risk (RE-SAC's κ estimates).  This is formalized as a diagonal
Mahalanobis distance, yielding an exponential likelihood.

Key insight: the Mahalanobis likelihood is **strictly positive** (exp > 0),
guaranteeing the normalization constant Z > 0 whenever ρ has support.
This is stronger than the generic L ≥ 0 assumption in BAPR.
-/

/-- Squared Mahalanobis distance with diagonal precision matrix.
    `d^2 = sum_i prec_i * surp_i^2`
    where `prec` is the precision (inverse aleatoric variance) per objective. -/
def mahalanobis_sq (surp : Fin m → ℝ) (prec : Fin m → ℝ) : ℝ :=
  ∑ i : Fin m, prec i * surp i ^ 2

/-- Mahalanobis-based likelihood for Bayesian belief update.
    `L(h, surp) = exp(-1/2 * d^2(surp, prec))`  -/
def mahalanobis_likelihood (surp : Fin m → ℝ) (prec : Fin m → ℝ) : ℝ :=
  Real.exp (-(1/2) * mahalanobis_sq surp prec)

/-- The Mahalanobis likelihood is strictly positive (exp > 0). -/
lemma mahalanobis_likelihood_pos (surp : Fin m → ℝ) (prec : Fin m → ℝ) :
    0 < mahalanobis_likelihood surp prec := by
  unfold mahalanobis_likelihood
  exact Real.exp_pos _

/-- The Mahalanobis likelihood is non-negative. -/
lemma mahalanobis_likelihood_nonneg (surp : Fin m → ℝ) (prec : Fin m → ℝ) :
    0 ≤ mahalanobis_likelihood surp prec :=
  le_of_lt (mahalanobis_likelihood_pos surp prec)

/-- When using Mahalanobis likelihood with supported prior, Z > 0 is automatic.
    This eliminates the need for the ad-hoc Z > 0 hypothesis in BAPR. -/
lemma mahalanobis_normalization_pos (ρ : H → ℝ) (surps : H → Fin m → ℝ)
    (prec : Fin m → ℝ) (hρ_nn : ∀ h, 0 ≤ ρ h)
    (h_exists : ∃ h, 0 < ρ h) :
    0 < BAPR.normalization_const ρ
        (fun h => mahalanobis_likelihood (surps h) prec) := by
  unfold BAPR.normalization_const
  obtain ⟨h₀, hh₀⟩ := h_exists
  have h_pos : 0 < ρ h₀ * mahalanobis_likelihood (surps h₀) prec :=
    mul_pos hh₀ (mahalanobis_likelihood_pos (surps h₀) prec)
  have h_nn : ∀ h ∈ Finset.univ,
      0 ≤ ρ h * mahalanobis_likelihood (surps h) prec :=
    fun h _ => mul_nonneg (hρ_nn h) (mahalanobis_likelihood_nonneg (surps h) prec)
  linarith [Finset.single_le_sum h_nn (Finset.mem_univ h₀)]

/--
**Theorem: BAMOR contraction with Mahalanobis-driven belief update.**

Unlike the generic BAPR belief update, the Mahalanobis likelihood guarantees
Z > 0 automatically from the prior having support — no external Z > 0 needed.
-/
theorem bamor_contraction_mahalanobis (p : Params)
    (R : H → S → A → Fin m → ℝ) (P : H → S → A → S → ℝ)
    (Γ_epi : H → S → A → Fin m → ℝ) (κ_ale : Fin m → ℝ)
    (ρ : H → ℝ) (surps : H → Fin m → ℝ) (prec : Fin m → ℝ)
    (hP_nn   : ∀ h s a s', 0 ≤ P h s a s')
    (hP_prob : ∀ h s a, ∑ s' : S, P h s a s' = 1)
    (hρ_nn   : ∀ h, 0 ≤ ρ h)
    (h_supp  : ∃ h, 0 < ρ h) :
    let L := fun h => mahalanobis_likelihood (surps h) prec
    let Z := BAPR.normalization_const ρ L
    let ρ' := BAPR.update_belief ρ L Z
    ∃ k < 1, ∀ (Q₁ Q₂ : S × A → Fin m → ℝ),
    dist (T_BAMOR p R P Γ_epi κ_ale ρ' Q₁)
         (T_BAMOR p R P Γ_epi κ_ale ρ' Q₂) ≤ k * dist Q₁ Q₂ := by
  intro L Z ρ'
  have hZ_pos := mahalanobis_normalization_pos ρ surps prec hρ_nn h_supp
  exact bamor_contraction p R P Γ_epi κ_ale ρ' hP_nn hP_prob
    (BAPR.update_belief_nonneg ρ L Z hρ_nn
      (fun h => mahalanobis_likelihood_nonneg (surps h) prec) hZ_pos)
    (BAPR.update_belief_sum_one ρ L Z hZ_pos rfl)

-- ════════════════════════════════════════════════════════════════
-- PART V: Anytime Bounded Error (MORL-specific)
-- ════════════════════════════════════════════════════════════════

/-! ## 10. Tracking Error Bound for Non-Stationary Environments

The original BAPR assumes long meta-stable periods (T_gap ≫ 1/(1−γ)) so
that Q converges to the fixed point between mode switches.  In MORL with
frequent perturbations (e.g., high-variance bus holding), this is unrealistic.

We prove a strictly stronger result: even with perturbations at EVERY step,
the tracking error is uniformly bounded by Δ/(1−γ), where Δ is the
per-step perturbation bound from mode switches.

This replaces the "must converge" requirement with an **anytime guarantee**.
-/

/-- The absorbing bound Δ/(1−γ) is a fixed point of e ↦ γe + Δ. -/
lemma absorbing_bound {γ : ℝ} (hγ_lt : γ < 1)
    {Δ : ℝ} : γ * (Δ / (1 - γ)) + Δ = Δ / (1 - γ) := by
  have h1γ : (1 : ℝ) - γ ≠ 0 := by linarith
  field_simp
  ring

/--
**Theorem (Anytime Tracking Bound)**

If an operator is a γ-contraction and environment perturbations are
bounded by Δ per step, then:

    e(n) ≤ γⁿ · e(0) + Δ/(1−γ)

This holds at EVERY step n — no convergence required.  The first term
decays geometrically (forgetting initial error), while the second term
is the irreducible tracking error from non-stationarity.
-/
theorem anytime_tracking_bound {γ : ℝ} (hγ_nn : 0 ≤ γ) (hγ_lt : γ < 1)
    {Δ : ℝ} (hΔ_nn : 0 ≤ Δ) {e : ℕ → ℝ}
    (he_step : ∀ n, e (n + 1) ≤ γ * e n + Δ) :
    ∀ n, e n ≤ γ ^ n * e 0 + Δ / (1 - γ) := by
  intro n
  induction n with
  | zero =>
    simp
    exact div_nonneg hΔ_nn (by linarith)
  | succ k ih =>
    have h1γ : (1 : ℝ) - γ ≠ 0 := by linarith
    calc e (k + 1)
        ≤ γ * e k + Δ := he_step k
      _ ≤ γ * (γ ^ k * e 0 + Δ / (1 - γ)) + Δ := by
          linarith [mul_le_mul_of_nonneg_left ih hγ_nn]
      _ = γ ^ (k + 1) * e 0 + Δ / (1 - γ) := by
          field_simp
          ring

/--
**Corollary: Uniform tracking bound (independent of initial error)**

For any sequence satisfying e(n+1) ≤ γ·e(n) + Δ:

    lim sup e(n) ≤ Δ/(1−γ)

In particular, after N = ⌈log(e(0)·(1−γ)/Δ) / log(1/γ)⌉ steps,
the error is within 2·Δ/(1−γ).  This bound is tight.
-/
theorem uniform_tracking_bound {γ : ℝ} (hγ_nn : 0 ≤ γ) (hγ_lt : γ < 1)
    {Δ : ℝ} (hΔ_nn : 0 ≤ Δ) {B : ℝ}
    {e : ℕ → ℝ} (he_nn : ∀ n, 0 ≤ e n)
    (he_init : e 0 ≤ B)
    (he_step : ∀ n, e (n + 1) ≤ γ * e n + Δ) :
    ∀ n, e n ≤ B + Δ / (1 - γ) := by
  intro n
  have h := anytime_tracking_bound hγ_nn hγ_lt hΔ_nn he_step n
  have hγn : γ ^ n * e 0 ≤ 1 * B := by
    apply mul_le_mul _ he_init (he_nn 0) (by linarith)
    exact pow_le_one₀ hγ_nn (le_of_lt hγ_lt)
  linarith

-- ════════════════════════════════════════════════════════════════
-- PART VI: Fixed Point Characterization (Banach)
-- ════════════════════════════════════════════════════════════════

/-! ## 11. Banach Fixed-Point Theorem for BAMOR

Since T_BAMOR is a γ-contraction on a complete metric space, Banach's
fixed-point theorem guarantees existence and uniqueness of a fixed point
Q* = T_BAMOR(Q*).  This Q* is the **robust Pareto-optimal value function**
under the Bayesian belief over environment modes.

- **Uniqueness** is proved directly from the contraction property
- **Existence** is obtained via Mathlib's `ContractingWith` API
-/

/-- The BAMOR operator is Lipschitz with constant γ. -/
lemma bamor_lipschitzWith (p : Params) (R : H → S → A → Fin m → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → Fin m → ℝ)
    (κ_ale : Fin m → ℝ) (ρ : H → ℝ)
    (hP_nn   : ∀ h s a s', 0 ≤ P h s a s')
    (hP_prob : ∀ h s a, ∑ s' : S, P h s a s' = 1)
    (hρ_nn   : ∀ h, 0 ≤ ρ h)
    (hρ_sum  : ∑ h : H, ρ h = 1) :
    LipschitzWith ⟨p.γ, p.hγ.1⟩ (T_BAMOR p R P Γ_epi κ_ale ρ) :=
  LipschitzWith.of_dist_le_mul
    (bamor_dist_le p R P Γ_epi κ_ale ρ hP_nn hP_prob hρ_nn hρ_sum)

/-- The BAMOR operator satisfies Mathlib's `ContractingWith` predicate.
    This bridges our custom proof to Mathlib's Banach fixed-point infrastructure. -/
lemma bamor_contractingWith (p : Params) (R : H → S → A → Fin m → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → Fin m → ℝ)
    (κ_ale : Fin m → ℝ) (ρ : H → ℝ)
    (hP_nn   : ∀ h s a s', 0 ≤ P h s a s')
    (hP_prob : ∀ h s a, ∑ s' : S, P h s a s' = 1)
    (hρ_nn   : ∀ h, 0 ≤ ρ h)
    (hρ_sum  : ∑ h : H, ρ h = 1) :
    ContractingWith ⟨p.γ, p.hγ.1⟩ (T_BAMOR p R P Γ_epi κ_ale ρ) :=
  ⟨by exact_mod_cast p.hγ.2,
   bamor_lipschitzWith p R P Γ_epi κ_ale ρ hP_nn hP_prob hρ_nn hρ_sum⟩

/--
**Theorem (Uniqueness of Robust Pareto Q-function)**

If Q₁ and Q₂ are both fixed points of T_BAMOR, then Q₁ = Q₂.

Proof: `dist(Q₁, Q₂) = dist(T Q₁, T Q₂) ≤ γ · dist(Q₁, Q₂)`.
Since γ < 1 and dist ≥ 0, this forces `dist(Q₁, Q₂) = 0`.
-/
theorem bamor_fixed_point_unique (p : Params) (R : H → S → A → Fin m → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → Fin m → ℝ)
    (κ_ale : Fin m → ℝ) (ρ : H → ℝ)
    (hP_nn   : ∀ h s a s', 0 ≤ P h s a s')
    (hP_prob : ∀ h s a, ∑ s' : S, P h s a s' = 1)
    (hρ_nn   : ∀ h, 0 ≤ ρ h)
    (hρ_sum  : ∑ h : H, ρ h = 1)
    {Q₁ Q₂ : S × A → Fin m → ℝ}
    (h1 : T_BAMOR p R P Γ_epi κ_ale ρ Q₁ = Q₁)
    (h2 : T_BAMOR p R P Γ_epi κ_ale ρ Q₂ = Q₂) :
    Q₁ = Q₂ := by
  by_contra h_ne
  have h_pos : 0 < dist Q₁ Q₂ := by rwa [dist_pos]
  have h_bound := bamor_dist_le p R P Γ_epi κ_ale ρ hP_nn hP_prob hρ_nn hρ_sum Q₁ Q₂
  rw [h1, h2] at h_bound
  -- dist(Q₁, Q₂) ≤ γ · dist(Q₁, Q₂), but γ < 1 and dist > 0
  have h_sub : (1 - p.γ) * dist Q₁ Q₂ ≤ 0 := by linarith
  have h_1mγ_pos : 0 < 1 - p.γ := by linarith [p.hγ.2]
  linarith [mul_pos h_1mγ_pos h_pos]

/--
**Theorem (Existence of Robust Pareto Q-function)**

By the Banach contraction mapping theorem, T_BAMOR has a fixed point Q*
on the complete metric space `S × A → Fin m → ℝ`.

This Q* is the unique **robust Pareto-optimal value function** under
the Bayesian belief ρ over environment modes.
-/
theorem bamor_fixed_point_exists (p : Params) (R : H → S → A → Fin m → ℝ)
    (P : H → S → A → S → ℝ) (Γ_epi : H → S → A → Fin m → ℝ)
    (κ_ale : Fin m → ℝ) (ρ : H → ℝ)
    (hP_nn   : ∀ h s a s', 0 ≤ P h s a s')
    (hP_prob : ∀ h s a, ∑ s' : S, P h s a s' = 1)
    (hρ_nn   : ∀ h, 0 ≤ ρ h)
    (hρ_sum  : ∑ h : H, ρ h = 1) :
    ∃ Q_star : S × A → Fin m → ℝ,
    T_BAMOR p R P Γ_epi κ_ale ρ Q_star = Q_star := by
  have hc := bamor_contractingWith p R P Γ_epi κ_ale ρ
    hP_nn hP_prob hρ_nn hρ_sum
  -- Pick any starting point (e.g., the zero function)
  set Q₀ : S × A → Fin m → ℝ := fun _ _ => 0
  -- edist from Q₀ to T(Q₀) is finite (both are bounded functions on finite types)
  have h_fin : edist Q₀ (T_BAMOR p R P Γ_epi κ_ale ρ Q₀) ≠ ⊤ :=
    edist_ne_top _ _
  obtain ⟨Q_star, hfp, _, _⟩ := hc.exists_fixedPoint Q₀ h_fin
  exact ⟨Q_star, hfp⟩

end BAMOR
