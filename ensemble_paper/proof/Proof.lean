import Mathlib

/-!
# Formal Proof: RE-SAC Bellman Operator is a Contraction Mapping

## Key insight: κ_ale is a fixed constant per training iteration

In RE-SAC, the aleatoric penalty `λ_ale * Σ_l ‖W_l‖₁` is computed from the
**current critic network weights** at each training step. The weights W_l are
fixed during a policy evaluation sweep (they are updated only after the Bellman
backup). Therefore the aleatoric penalty is a **constant scalar** κ_ale that
does NOT depend on which Q-function we apply the operator to.

This observation eliminates the need for any monotonicity assumption on
AleatoricPenalty, and the contraction proof becomes fully self-contained.
-/

noncomputable section

namespace RESAC

variable {S A : Type} [Fintype S] [Fintype A] [Nonempty S] [Nonempty A]

/-- Hyperparameters for the RE-SAC framework -/
structure Params where
  γ        : ℝ
  lam_epi  : ℝ
  hγ        : 0 ≤ γ ∧ γ < 1

/-- The Max operator: V(s) = maxₐ Q(s, a) -/
def max_over_a (Q : S × A → ℝ) (s : S) : ℝ :=
  (Finset.univ : Finset A).sup' Finset.univ_nonempty (fun a => Q (s, a))

/-!
## The RE-SAC Operator

`κ_ale` is the aleatoric penalty — a **fixed scalar** for the current
training step, equal to `λ_ale * Σ_l ‖W_l^(θ)‖₁`.  It does not depend on Q.
`Γ_epi` is the epistemic penalty — it is precomputed from the ensemble and
also does not change between T(Q₁) and T(Q₂) in the Bellman backup.
-/
def T (p : Params) (R : S → A → ℝ) (P : S → A → S → ℝ)
    (Γ_epi : S → A → ℝ) (κ_ale : ℝ)
    (Q : S × A → ℝ) : S × A → ℝ := fun ⟨s, a⟩ =>
  R s a + p.γ * (
    (∑ s' : S, P s a s' * max_over_a Q s') -
    p.lam_epi * Γ_epi s a -
    κ_ale
  )

-- ============================================================
-- INTERMEDIATE LEMMAS
-- ============================================================

/-- max_over_a is monotone in Q. -/
lemma max_over_a_mono {Q₁ Q₂ : S × A → ℝ} (h : Q₁ ≤ Q₂) (s : S) :
    max_over_a Q₁ s ≤ max_over_a Q₂ s := by
  unfold max_over_a
  apply Finset.sup'_le
  intro a _
  have hq : Q₁ (s, a) ≤ Q₂ (s, a) := h (s, a)
  have hle : Q₂ (s, a) ≤ Finset.univ.sup' Finset.univ_nonempty (fun b => Q₂ (s, b)) :=
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
    have : Q (s, a) ≤ Finset.univ.sup' Finset.univ_nonempty (fun b => Q (s, b)) :=
      Finset.le_sup' (fun b => Q (s, b)) (Finset.mem_univ a)
    linarith
  · have key : Finset.univ.sup' Finset.univ_nonempty (fun a => Q (s, a)) ≤
        Finset.univ.sup' Finset.univ_nonempty (fun a => Q (s, a) + c) - c := by
      apply Finset.sup'_le
      intro a _
      have : Q (s, a) + c ≤ Finset.univ.sup' Finset.univ_nonempty (fun b => Q (s, b) + c) :=
        Finset.le_sup' (fun b => Q (s, b) + c) (Finset.mem_univ a)
      linarith
    linarith

/-- max_over_a is 1-Lipschitz in L∞: |max Q₁ s − max Q₂ s| ≤ ‖Q₁ − Q₂‖∞. -/
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
    have hmax₁ : Q₁ (s, a) ≤ Finset.univ.sup' Finset.univ_nonempty (fun b => Q₁ (s, b)) :=
      Finset.le_sup' (fun b => Q₁ (s, b)) (Finset.mem_univ a)
    linarith
  · unfold max_over_a
    rw [sub_le_iff_le_add]
    apply Finset.sup'_le
    intro a _
    have habs : |Q₁ (s, a) - Q₂ (s, a)| ≤ ε := hpw (s, a)
    have hle : Q₁ (s, a) ≤ Q₂ (s, a) + ε := by linarith [(abs_le.mp habs).2]
    have hmax₂ : Q₂ (s, a) ≤ Finset.univ.sup' Finset.univ_nonempty (fun b => Q₂ (s, b)) :=
      Finset.le_sup' (fun b => Q₂ (s, b)) (Finset.mem_univ a)
    linarith

-- ============================================================
-- LEMMA 1: Monotonicity
-- ============================================================

/--
**Lemma 1 (Monotonicity)**: If Q₁ ≤ Q₂ pointwise, then T(Q₁) ≤ T(Q₂).

Proof: Since κ_ale does NOT depend on Q, it cancels in the comparison.
The only Q-dependent term is ∑ P * max_over_a Q, which is monotone
because max is monotone and P ≥ 0.
-/
lemma monotonicity (p : Params) (R : S → A → ℝ) (P : S → A → S → ℝ)
    (Γ_epi : S → A → ℝ) (κ_ale : ℝ)
    (hP_nn : ∀ s a s', 0 ≤ P s a s')
    (Q₁ Q₂ : S × A → ℝ) (h : Q₁ ≤ Q₂) :
    T p R P Γ_epi κ_ale Q₁ ≤ T p R P Γ_epi κ_ale Q₂ := by
  intro ⟨s, a⟩
  dsimp only [T]
  -- The κ_ale term is identical on both sides: it cancels.
  -- It suffices to show ∑ P * max Q₁ ≤ ∑ P * max Q₂.
  have hsum : ∑ s' : S, P s a s' * max_over_a Q₁ s' ≤
              ∑ s' : S, P s a s' * max_over_a Q₂ s' := by
    apply Finset.sum_le_sum
    intro s' _
    exact mul_le_mul_of_nonneg_left (max_over_a_mono h s') (hP_nn s a s')
  have hγ_mul := mul_le_mul_of_nonneg_left
    (show ∑ s' : S, P s a s' * max_over_a Q₁ s' - p.lam_epi * Γ_epi s a - κ_ale ≤
         ∑ s' : S, P s a s' * max_over_a Q₂ s' - p.lam_epi * Γ_epi s a - κ_ale by linarith)
    p.hγ.1
  linarith

-- ============================================================
-- LEMMA 2: Discounting
-- ============================================================

/--
**Lemma 2 (Discounting)**: T(Q + c) = T(Q) + γ·c for any constant c.

Proof:
  max(Q + c) s = max(Q) s + c            (max_over_a_add_const)
  The κ_ale term is constant (unaffected).
  ∑ P * (max Q + c) = ∑ P * max Q + c    (uses ∑ P = 1).
-/
lemma discounting (p : Params) (R : S → A → ℝ) (P : S → A → S → ℝ)
    (Γ_epi : S → A → ℝ) (κ_ale : ℝ)
    (hP_prob : ∀ s a, ∑ s' : S, P s a s' = 1)
    (Q : S × A → ℝ) (c : ℝ) :
    T p R P Γ_epi κ_ale (Q + fun _ => c) = T p R P Γ_epi κ_ale Q + fun _ => p.γ * c := by
  funext ⟨s, a⟩
  simp only [T, Pi.add_apply]
  have hmax : ∀ s', max_over_a (Q + fun _ => c) s' = max_over_a Q s' + c :=
    fun s' => max_over_a_add_const Q c s'
  -- ∑ P * (max Q s' + c) = ∑ P * max Q s' + c
  have hsum : ∑ s' : S, P s a s' * max_over_a (Q + fun _ => c) s' =
              (∑ s' : S, P s a s' * max_over_a Q s') + c := by
    simp_rw [hmax, mul_add]
    rw [Finset.sum_add_distrib]
    congr 1
    rw [← Finset.sum_mul, hP_prob s a, one_mul]
  rw [hsum]
  ring

-- ============================================================
-- THEOREM: Contraction Mapping
-- ============================================================

/-- The function space S × A → ℝ inherits the L∞ PseudoMetricSpace from Pi. -/
instance : PseudoMetricSpace (S × A → ℝ) :=
  show PseudoMetricSpace (∀ _ : S × A, ℝ) from inferInstance

/--
**Theorem (RE-SAC Operator is a γ-Contraction)**

Under:
  * (A1) P(s'|s,a) ≥ 0
  * (A2) ∑_s' P(s'|s,a) = 1
  * κ_ale is a fixed scalar (independent of Q)

The operator T is a γ-contraction in L∞.

Proof:
  Since κ_ale does not depend on Q, T(Q₁)(s,a) − T(Q₂)(s,a)
  = γ * ∑_s' P(s'|s,a) * (max Q₁ s' − max Q₂ s').
  
  By triangle inequality + A1 + A2 + 1-Lipschitz of max:
    |T(Q₁)(s,a) − T(Q₂)(s,a)| ≤ γ * ε   where ε = dist Q₁ Q₂.

  No assumption on the behaviour of AleatoricPenalty under Q is needed.
-/
theorem RE_SAC_contraction (p : Params) (R : S → A → ℝ) (P : S → A → S → ℝ)
    (Γ_epi : S → A → ℝ) (κ_ale : ℝ)
    (hP_nn   : ∀ s a s', 0 ≤ P s a s')
    (hP_prob : ∀ s a, ∑ s' : S, P s a s' = 1) :
    ∃ k < 1, ∀ (Q₁ Q₂ : S × A → ℝ),
    dist (T p R P Γ_epi κ_ale Q₁) (T p R P Γ_epi κ_ale Q₂) ≤ k * dist Q₁ Q₂ := by
  use p.γ
  refine ⟨p.hγ.2, fun Q₁ Q₂ => ?_⟩
  set ε := dist Q₁ Q₂
  have hε_nn : 0 ≤ ε := dist_nonneg
  rw [dist_pi_le_iff (mul_nonneg p.hγ.1 hε_nn)]
  intro ⟨s, a⟩
  simp only [Real.dist_eq]
  -- Pointwise L∞ bound from Pi metric
  have hpw : ∀ sa, |Q₁ sa - Q₂ sa| ≤ ε := by
    intro sa
    have h2 := (dist_pi_le_iff hε_nn).mp (le_refl ε) sa
    simpa [Real.dist_eq] using h2
  -- The key: T(Q₁)(s,a) - T(Q₂)(s,a) = γ * ∑ P*(maxQ₁ - maxQ₂)
  -- κ_ale cancels exactly.
  dsimp only [T]
  have hS : ∑ s' : S, P s a s' * max_over_a Q₁ s' -
            ∑ s' : S, P s a s' * max_over_a Q₂ s' =
            ∑ s' : S, P s a s' * (max_over_a Q₁ s' - max_over_a Q₂ s') := by
    rw [← Finset.sum_sub_distrib]
    congr 1; ext s'; ring
  have hsimp : R s a + p.γ * (∑ s' : S, P s a s' * max_over_a Q₁ s' -
      p.lam_epi * Γ_epi s a - κ_ale) -
    (R s a + p.γ * (∑ s' : S, P s a s' * max_over_a Q₂ s' -
      p.lam_epi * Γ_epi s a - κ_ale)) =
    p.γ * ∑ s' : S, P s a s' * (max_over_a Q₁ s' - max_over_a Q₂ s') := by
    linear_combination p.γ * hS
  rw [hsimp, abs_mul, abs_of_nonneg p.hγ.1]
  apply mul_le_mul_of_nonneg_left _ p.hγ.1
  -- Bound |∑ P*(maxQ₁ - maxQ₂)| ≤ ε
  calc |∑ s' : S, P s a s' * (max_over_a Q₁ s' - max_over_a Q₂ s')|
      ≤ ∑ s' : S, |P s a s' * (max_over_a Q₁ s' - max_over_a Q₂ s')| :=
          Finset.abs_sum_le_sum_abs _ _
    _ = ∑ s' : S, P s a s' * |max_over_a Q₁ s' - max_over_a Q₂ s'| := by
          congr 1; ext s'
          rw [abs_mul, abs_of_nonneg (hP_nn s a s')]
    _ ≤ ∑ s' : S, P s a s' * ε := by
          apply Finset.sum_le_sum; intro s' _
          exact mul_le_mul_of_nonneg_left (max_over_a_nonexpansive s' hpw) (hP_nn s a s')
    _ = ε := by rw [← Finset.sum_mul, hP_prob s a, one_mul]

end RESAC
