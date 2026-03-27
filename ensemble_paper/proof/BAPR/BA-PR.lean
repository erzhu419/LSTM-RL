
/-
  BA-PR SAC: 证明在分段平稳环境下，
  结合贝叶斯遗忘机制的增广算子满足收缩性
-/

-- 定义环境模式（离散模式切换）
inductive EnvMode where
  | stable : (z : ℝ) → EnvMode
  | jumping : (z_old z_new : ℝ) → EnvMode

-- 定义增广状态：(原始状态, 贝叶斯置信度)
structure AugmentedState where
  s : PhysicalState
  rho : BeliefDistribution -- 对“当前模式持续时间”的估计

-- 1. 证明贝叶斯更新算子的稳定性
theorem belief_update_stable (surprise : ℝ) :
  Lipschitz (update_rho rho surprise) := by
  -- 利用假设 2 证明惊奇度不会导致 rho 爆炸性跳变
  sorry

-- 2. 定义 BA-PR 算子
def T_BAPR (Q : QFunc) (rho : BeliefDistribution) : QFunc :=
  -- 关键：对所有可能的“历史长度”进行期望求和
  Sum_over_history (fun h =>
    (rho.prob h) * (Standard_Bellman Q (infer_z_from_h h))
  )

-- 3. 核心定理：分段收缩
theorem bapr_contraction (gamma : ℝ) (h_gamma : gamma < 1) :
  ∀ (Q1 Q2 : QFunc), dist (T_BAPR Q1 rho) (T_BAPR Q2 rho) ≤ gamma * dist Q1 Q2 := by
  -- 证明逻辑：
  -- 1. 展开求和项。
  -- 2. 利用 rho.prob 的归一化性质（Σ prob = 1）。
  -- 3. 展示对于每一个确定的历史长度 h，内部的 Standard_Bellman 都是 gamma-收缩的。
  -- 4. 凸组合性质：收缩映射的凸组合（加权平均）依然是收缩映射。
  intros Q1 Q2
  simp [T_BAPR]
  -- 利用线性性质提取 gamma
  have h_linear : ∀ h, dist (Bellman Q1 z_h) (Bellman Q2 z_h) ≤ gamma * dist Q1 Q2 := by
    intro h; apply standard_contraction -- 引用你 RE-SAC 的基础证明
  -- 关键步：凸组合证明
  calc
    dist (Σ p_h * B Q1) (Σ p_h * B Q2)
    ≤ Σ p_h * dist (B Q1) (B Q2)    := by apply dist_sum_le_sum_dist
    ≤ Σ p_h * (gamma * dist Q1 Q2)  := by apply sum_le_sum; exact h_linear
    = gamma * dist Q1 Q2 * (Σ p_h)  := by rw [mul_sum_constant]
    = gamma * dist Q1 Q2            := by rw [rho.sum_one]; ring
  done

