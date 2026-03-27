import Mathlib

/-!
# Formal Proof: RE-AMHT Bellman Operator is a Contraction Mapping

## Overview

This file proves that the RE-AMHT (Robust Ensemble Aleatoric-Modulated History
Truncation) Bellman operator is a γ-contraction in L∞ norm, establishing the
existence and uniqueness of its fixed point.

## Algorithm Context

RE-AMHT extends RE-SAC with an *Aleatoric-Modulated History Truncation* (AM-HT)
mechanism.  The key idea: the aleatoric risk `κ` estimated by RE-SAC is used as
a **meta-control signal** to adjust the observation window weight for environment
context inference:

  * **High noise (large κ)**: increase window → average over longer history
                               → stable but slow context estimation
  * **Low noise (small κ)**: decrease window → forget old observations quickly
                               → fast detection of environment changes

## Three Assumptions

1. **Two-Timescale Update (Assumption 1)**: The window weight `w` and aleatoric
   penalty `κ_ale` are computed from the *target* network and held frozen during
   one Bellman backup step.  This decouples `z` from the current Q-function.

2. **Contextual Lipschitz Continuity (Assumption 2)**: The encoder mapping from
   history to context `z` is Lipschitz continuous.  Under Assumption 1, this is
   not needed for the single-step contraction proof (since `z` is frozen), but
   is important for the outer-loop convergence analysis.

3. **Compact Support (Assumption 3)**: All state, context, and noise spaces are
   bounded — modeled here as `Fintype` (finite spaces are compact and bounded).

## Proof Strategy

Under Assumption 1, the context `z` is *frozen* during one backup step.  This
means the AM-HT operator on the augmented state space `(S × Z) × A` is
structurally identical to a standard Bellman operator with frozen penalty terms.
The proof follows Blackwell's sufficient conditions (monotonicity + discounting),
directly reusing the technique from `Proof.lean`.

## Proof Structure

1. `max_over_a_mono`              — max is monotone in Q
2. `max_over_a_add_const`         — max distributes over constant shift
3. `max_over_a_nonexpansive`      — max is 1-Lipschitz in L∞
4. `monotonicity`                 — Lemma 1 (Blackwell condition i)
5. `discounting`                  — Lemma 2 (Blackwell condition ii)
6. `re_amht_contraction`          — Main theorem: T_AMHT is a γ-contraction
-/

noncomputable section

set_option linter.unusedSectionVars false

namespace AMHT

/-!
## 1. Space Definitions (Assumption 3: Compact Support)

We model states `S`, actions `A`, and context `Z` as finite, nonempty types.
Finiteness is a strong form of compactness that makes sums well-defined and
avoids measure-theoretic machinery.

In MuJoCo/Gym environments:
  - `S` corresponds to the (discretized) joint-angle / velocity state space
  - `A` corresponds to the (discretized) torque actions
  - `Z` corresponds to the (discretized) environment parameter context
    (e.g., mass, friction coefficients)
-/

variable {S A Z : Type} [Fintype S] [Fintype A] [Fintype Z]
                         [Nonempty S] [Nonempty A] [Nonempty Z]

/-- The augmented state type: physical state × inferred context -/
abbrev AugState (S Z : Type) := S × Z

/-- Hyperparameters for the RE-AMHT framework -/
structure Params where
  γ        : ℝ
  lam_epi  : ℝ
  hγ       : 0 ≤ γ ∧ γ < 1
  hlam_epi : 0 ≤ lam_epi

/-!
## 2. The Max Operator

V(s, z) = max_a Q((s, z), a)

This is defined on the augmented state space.
-/

/-- The Max operator on augmented state: V(s,z) = max_a Q((s,z), a) -/
def max_over_a (Q : AugState S Z × A → ℝ) (sz : AugState S Z) : ℝ :=
  (Finset.univ : Finset A).sup' Finset.univ_nonempty (fun a => Q (sz, a))

/-!
## 3. The RE-AMHT Operator (Assumption 1: Frozen Parameters)

Under the Two-Timescale assumption, the following quantities are **frozen**
during one Bellman backup step (computed from the target network):

  - `κ_ale : ℝ`         — aleatoric penalty (scalar, independent of Q)
  - `Γ_epi : S → Z → A → ℝ` — epistemic penalty (from target ensemble)
  - The context mapping `s ↦ z` is already incorporated into the augmented
    transition kernel `P`, since `z` is fixed for the current backup step.

Key insight: because `z` is frozen, the transition `P` over `AugState S Z`
already encodes the context dynamics, and the operator has the same algebraic
form as the standard RE-SAC operator.
-/

/-- The RE-AMHT Bellman operator on augmented state space.

  T_AMHT(Q)(s,z,a) = R(s,z,a) + γ * (∑_{s',z'} P(s',z'|s,z,a) * max_a' Q(s',z',a')
                                       - λ_epi * Γ_epi(s,z,a) - κ_ale)

  Under Assumption 1, κ_ale is a fixed scalar and Γ_epi is precomputed from
  the target network — neither depends on the current Q.
-/
def T_AMHT (p : Params) (R : S → Z → A → ℝ) (P : S → Z → A → AugState S Z → ℝ)
    (Γ_epi : S → Z → A → ℝ) (κ_ale : ℝ)
    (Q : AugState S Z × A → ℝ) : AugState S Z × A → ℝ := fun ⟨⟨s, z⟩, a⟩ =>
  R s z a + p.γ * (
    (∑ sz' : AugState S Z, P s z a sz' * max_over_a Q sz') -
    p.lam_epi * Γ_epi s z a -
    κ_ale
  )

/-!
## 4. Intermediate Lemmas

These lemmas establish properties of `max_over_a` that are needed for the
Blackwell conditions.  They follow the same structure as `Proof.lean`.
-/

/-- max_over_a is monotone in Q. -/
lemma max_over_a_mono {Q₁ Q₂ : AugState S Z × A → ℝ} (h : Q₁ ≤ Q₂)
    (sz : AugState S Z) :
    max_over_a Q₁ sz ≤ max_over_a Q₂ sz := by
  unfold max_over_a
  apply Finset.sup'_le
  intro a _
  have hq : Q₁ (sz, a) ≤ Q₂ (sz, a) := h (sz, a)
  have hle : Q₂ (sz, a) ≤ Finset.univ.sup' Finset.univ_nonempty
      (fun b => Q₂ (sz, b)) :=
    Finset.le_sup' (fun b => Q₂ (sz, b)) (Finset.mem_univ a)
  linarith

/-- max_over_a distributes over constant addition. -/
lemma max_over_a_add_const (Q : AugState S Z × A → ℝ) (c : ℝ)
    (sz : AugState S Z) :
    max_over_a (Q + fun _ => c) sz = max_over_a Q sz + c := by
  unfold max_over_a
  simp only [Pi.add_apply]
  apply le_antisymm
  · apply Finset.sup'_le
    intro a _
    have : Q (sz, a) ≤ Finset.univ.sup' Finset.univ_nonempty
        (fun b => Q (sz, b)) :=
      Finset.le_sup' (fun b => Q (sz, b)) (Finset.mem_univ a)
    linarith
  · have key : Finset.univ.sup' Finset.univ_nonempty (fun a => Q (sz, a)) ≤
        Finset.univ.sup' Finset.univ_nonempty (fun a => Q (sz, a) + c) - c := by
      apply Finset.sup'_le
      intro a _
      have : Q (sz, a) + c ≤ Finset.univ.sup' Finset.univ_nonempty
          (fun b => Q (sz, b) + c) :=
        Finset.le_sup' (fun b => Q (sz, b) + c) (Finset.mem_univ a)
      linarith
    linarith

/-- max_over_a is 1-Lipschitz in L∞: |max Q₁ sz − max Q₂ sz| ≤ ε. -/
lemma max_over_a_nonexpansive {Q₁ Q₂ : AugState S Z × A → ℝ}
    (sz : AugState S Z) {ε : ℝ}
    (hpw : ∀ sa, |Q₁ sa - Q₂ sa| ≤ ε) :
    |max_over_a Q₁ sz - max_over_a Q₂ sz| ≤ ε := by
  rw [abs_le]
  constructor
  · rw [neg_le_sub_iff_le_add]
    unfold max_over_a
    apply Finset.sup'_le
    intro a _
    have habs : |Q₁ (sz, a) - Q₂ (sz, a)| ≤ ε := hpw (sz, a)
    have hle : Q₂ (sz, a) ≤ Q₁ (sz, a) + ε := by linarith [(abs_le.mp habs).1]
    have hmax₁ : Q₁ (sz, a) ≤ Finset.univ.sup' Finset.univ_nonempty
        (fun b => Q₁ (sz, b)) :=
      Finset.le_sup' (fun b => Q₁ (sz, b)) (Finset.mem_univ a)
    linarith
  · unfold max_over_a
    rw [sub_le_iff_le_add]
    apply Finset.sup'_le
    intro a _
    have habs : |Q₁ (sz, a) - Q₂ (sz, a)| ≤ ε := hpw (sz, a)
    have hle : Q₁ (sz, a) ≤ Q₂ (sz, a) + ε := by linarith [(abs_le.mp habs).2]
    have hmax₂ : Q₂ (sz, a) ≤ Finset.univ.sup' Finset.univ_nonempty
        (fun b => Q₂ (sz, b)) :=
      Finset.le_sup' (fun b => Q₂ (sz, b)) (Finset.mem_univ a)
    linarith

/-!
## 5. Lemma 1: Monotonicity (Blackwell Condition i)

If Q₁ ≤ Q₂ pointwise, then T_AMHT(Q₁) ≤ T_AMHT(Q₂).

Under Assumption 1, κ_ale and Γ_epi do NOT depend on Q, so they cancel.
The only Q-dependent term is ∑ P * max_over_a Q, which is monotone because
max is monotone and P ≥ 0.
-/

lemma monotonicity (p : Params) (R : S → Z → A → ℝ)
    (P : S → Z → A → AugState S Z → ℝ)
    (Γ_epi : S → Z → A → ℝ) (κ_ale : ℝ)
    (hP_nn : ∀ s z a sz', 0 ≤ P s z a sz')
    (Q₁ Q₂ : AugState S Z × A → ℝ) (h : Q₁ ≤ Q₂) :
    T_AMHT p R P Γ_epi κ_ale Q₁ ≤ T_AMHT p R P Γ_epi κ_ale Q₂ := by
  intro ⟨⟨s, z⟩, a⟩
  dsimp only [T_AMHT]
  have hsum : ∑ sz' : AugState S Z, P s z a sz' * max_over_a Q₁ sz' ≤
              ∑ sz' : AugState S Z, P s z a sz' * max_over_a Q₂ sz' := by
    apply Finset.sum_le_sum
    intro sz' _
    exact mul_le_mul_of_nonneg_left (max_over_a_mono h sz') (hP_nn s z a sz')
  have hγ_mul := mul_le_mul_of_nonneg_left
    (show ∑ sz' : AugState S Z, P s z a sz' * max_over_a Q₁ sz' -
          p.lam_epi * Γ_epi s z a - κ_ale ≤
          ∑ sz' : AugState S Z, P s z a sz' * max_over_a Q₂ sz' -
          p.lam_epi * Γ_epi s z a - κ_ale by linarith)
    p.hγ.1
  linarith

/-!
## 6. Lemma 2: Discounting (Blackwell Condition ii)

T_AMHT(Q + c) = T_AMHT(Q) + γ·c for any constant c.

The proof uses:
  - max(Q + c) sz = max(Q) sz + c       (max_over_a_add_const)
  - ∑ P(s'z'|s,z,a) = 1                 (transition is a distribution)
  - κ_ale is constant (unaffected by Q)
-/

lemma discounting (p : Params) (R : S → Z → A → ℝ)
    (P : S → Z → A → AugState S Z → ℝ)
    (Γ_epi : S → Z → A → ℝ) (κ_ale : ℝ)
    (hP_prob : ∀ s z a, ∑ sz' : AugState S Z, P s z a sz' = 1)
    (Q : AugState S Z × A → ℝ) (c : ℝ) :
    T_AMHT p R P Γ_epi κ_ale (Q + fun _ => c) =
    T_AMHT p R P Γ_epi κ_ale Q + fun _ => p.γ * c := by
  funext ⟨⟨s, z⟩, a⟩
  simp only [T_AMHT, Pi.add_apply]
  have hmax : ∀ sz', max_over_a (Q + fun _ => c) sz' = max_over_a Q sz' + c :=
    fun sz' => max_over_a_add_const Q c sz'
  have hsum : ∑ sz' : AugState S Z, P s z a sz' *
      max_over_a (Q + fun _ => c) sz' =
      (∑ sz' : AugState S Z, P s z a sz' * max_over_a Q sz') + c := by
    simp_rw [hmax, mul_add]
    rw [Finset.sum_add_distrib]
    congr 1
    rw [← Finset.sum_mul, hP_prob s z a, one_mul]
  rw [hsum]
  ring

/-!
## 7. Main Theorem: RE-AMHT Operator is a γ-Contraction

Under:
  * (A1) — κ_ale and Γ_epi are frozen (independent of Q) ← built into T_AMHT
  * (A3) — S, Z, A are finite (Fintype) ← in variable declarations
  * P(sz'|s,z,a) ≥ 0
  * ∑_{sz'} P(sz'|s,z,a) = 1

The operator T_AMHT is a γ-contraction in L∞.

Note on Assumption 2 (Lipschitz): Under Assumption 1, the context `z` is
completely frozen during one backup step.  Therefore, for any Q₁ and Q₂,
the `z` used in `T_AMHT(Q₁)` and `T_AMHT(Q₂)` is *identical*.  The
Lipschitz assumption is NOT needed for the single-step contraction proof.
It becomes relevant only in the outer-loop analysis when `target_Q` is
updated, which is a standard two-timescale convergence argument.
-/

instance : PseudoMetricSpace (AugState S Z × A → ℝ) :=
  show PseudoMetricSpace (∀ _ : AugState S Z × A, ℝ) from inferInstance

/--
**Theorem (RE-AMHT Operator is a γ-Contraction)**

The AM-HT Bellman operator on the augmented state space (S × Z) × A is a
γ-contraction in L∞ norm.  Combined with the Banach fixed-point theorem,
this establishes the existence and uniqueness of the optimal Q-function
Q* = T_AMHT(Q*) in the augmented MDP.
-/
theorem re_amht_contraction (p : Params) (R : S → Z → A → ℝ)
    (P : S → Z → A → AugState S Z → ℝ)
    (Γ_epi : S → Z → A → ℝ) (κ_ale : ℝ)
    (hP_nn   : ∀ s z a sz', 0 ≤ P s z a sz')
    (hP_prob : ∀ s z a, ∑ sz' : AugState S Z, P s z a sz' = 1) :
    ∃ k < 1, ∀ (Q₁ Q₂ : AugState S Z × A → ℝ),
    dist (T_AMHT p R P Γ_epi κ_ale Q₁) (T_AMHT p R P Γ_epi κ_ale Q₂) ≤
    k * dist Q₁ Q₂ := by
  use p.γ
  refine ⟨p.hγ.2, fun Q₁ Q₂ => ?_⟩
  set ε := dist Q₁ Q₂
  have hε_nn : 0 ≤ ε := dist_nonneg
  rw [dist_pi_le_iff (mul_nonneg p.hγ.1 hε_nn)]
  intro ⟨⟨s, z⟩, a⟩
  simp only [Real.dist_eq]
  -- Pointwise L∞ bound from Pi metric
  have hpw : ∀ sa, |Q₁ sa - Q₂ sa| ≤ ε := by
    intro sa
    have h2 := (dist_pi_le_iff hε_nn).mp (le_refl ε) sa
    simpa [Real.dist_eq] using h2
  -- The key: T_AMHT(Q₁)(s,z,a) - T_AMHT(Q₂)(s,z,a) =
  --   γ * ∑_{sz'} P(sz'|s,z,a) * (max Q₁ sz' - max Q₂ sz')
  --
  -- κ_ale and Γ_epi cancel exactly because they are frozen (Assumption 1).
  dsimp only [T_AMHT]
  have hS : ∑ sz' : AugState S Z, P s z a sz' * max_over_a Q₁ sz' -
            ∑ sz' : AugState S Z, P s z a sz' * max_over_a Q₂ sz' =
            ∑ sz' : AugState S Z, P s z a sz' *
              (max_over_a Q₁ sz' - max_over_a Q₂ sz') := by
    rw [← Finset.sum_sub_distrib]
    congr 1; ext sz'; ring
  have hsimp : R s z a + p.γ * (∑ sz' : AugState S Z,
      P s z a sz' * max_over_a Q₁ sz' -
      p.lam_epi * Γ_epi s z a - κ_ale) -
    (R s z a + p.γ * (∑ sz' : AugState S Z,
      P s z a sz' * max_over_a Q₂ sz' -
      p.lam_epi * Γ_epi s z a - κ_ale)) =
    p.γ * ∑ sz' : AugState S Z, P s z a sz' *
      (max_over_a Q₁ sz' - max_over_a Q₂ sz') := by
    linear_combination p.γ * hS
  rw [hsimp, abs_mul, abs_of_nonneg p.hγ.1]
  apply mul_le_mul_of_nonneg_left _ p.hγ.1
  -- Bound |∑ P*(max Q₁ - max Q₂)| ≤ ε
  calc |∑ sz' : AugState S Z, P s z a sz' *
          (max_over_a Q₁ sz' - max_over_a Q₂ sz')|
      ≤ ∑ sz' : AugState S Z,
          |P s z a sz' * (max_over_a Q₁ sz' - max_over_a Q₂ sz')| :=
          Finset.abs_sum_le_sum_abs _ _
    _ = ∑ sz' : AugState S Z,
          P s z a sz' * |max_over_a Q₁ sz' - max_over_a Q₂ sz'| := by
          congr 1; ext sz'
          rw [abs_mul, abs_of_nonneg (hP_nn s z a sz')]
    _ ≤ ∑ sz' : AugState S Z, P s z a sz' * ε := by
          apply Finset.sum_le_sum; intro sz' _
          exact mul_le_mul_of_nonneg_left
            (max_over_a_nonexpansive sz' hpw) (hP_nn s z a sz')
    _ = ε := by rw [← Finset.sum_mul, hP_prob s z a, one_mul]

end AMHT
