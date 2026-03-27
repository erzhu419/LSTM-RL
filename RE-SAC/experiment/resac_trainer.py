"""
RE-SAC Trainer: Regularized Ensemble Soft Actor-Critic.

Implements the RE-SAC algorithm as a TorchTrainer subclass compatible with the
rlkit framework from the DSAC codebase. This allows direct comparison with
SAC, TD3, and DSAC using identical training infrastructure.

Key features vs. vanilla SAC:
  - N ensemble Q-networks (vectorized) instead of twin-Q
  - Epistemic uncertainty penalty: policy optimizes q_mean + beta * q_std
  - OOD regularization: penalizes cross-ensemble Q-value disagreement
  - L1 weight regularization on critic parameters
  - Higher critic-to-actor update ratio

Optimizations:
  - AMP (automatic mixed precision) for GPU-bound ensemble forward/backward
  - expand() instead of repeat() for zero-copy broadcasting
  - Cached reg_norm (recomputed only on target network update)
"""
import gtimer as gt
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class RESACTrainer(TorchTrainer):
    """Regularized Ensemble Soft Actor-Critic trainer.

    Args:
        env: gym environment (for action space info)
        policy: TanhGaussianPolicy
        qf: VectorizedQNetwork (ensemble)
        target_qf: target VectorizedQNetwork
        ensemble_size: number of ensemble members
        beta: weight of epistemic uncertainty in policy loss (negative = pessimistic/LCB)
        beta_ood: weight of OOD regularization (Q-std across ensembles)
        weight_reg: weight of L1 regularization on critic weights
        beta_bc: weight of behavior cloning loss
        critic_actor_ratio: update critic N times per actor update
        discount: discount factor
        reward_scale: reward scaling factor
        policy_lr: policy learning rate
        qf_lr: Q-function learning rate
        soft_target_tau: Polyak averaging coefficient
        use_automatic_entropy_tuning: whether to auto-tune alpha
        target_entropy: target entropy for auto-tuning
        clip_norm: gradient clipping norm (0 = disabled)
    """

    def __init__(
            self,
            env,
            policy,
            qf,
            target_qf,
            ensemble_size=10,
            beta=-2.0,
            beta_ood=0.01,
            weight_reg=0.01,
            beta_bc=0.001,
            critic_actor_ratio=2,
            discount=0.99,
            reward_scale=1.0,
            alpha=0.2,
            policy_lr=3e-4,
            qf_lr=3e-4,
            optimizer_class=optim.Adam,
            soft_target_tau=5e-3,
            target_update_period=1,
            clip_norm=0.,
            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf = qf
        self.target_qf = target_qf

        self.ensemble_size = ensemble_size
        self.beta = beta
        self.beta_ood = beta_ood
        self.weight_reg = weight_reg
        self.beta_bc = beta_bc
        self.critic_actor_ratio = critic_actor_ratio

        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is not None:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )
        else:
            self.alpha = alpha

        self.qf_criterion = F.mse_loss

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.clip_norm = clip_norm

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        # AMP scaler for mixed precision training
        self.scaler = GradScaler()

        # Cached reg_norm — recomputed only after target network updates
        self._cached_reg_norm = None

    def _compute_reg_norm(self, model):
        """Compute L1 regularization norm across ensemble critic weights."""
        weight_norms = []
        bias_norms = []
        for name, param in model.named_parameters():
            if param.dim() == 3:  # VectorizedLinear weights: (E, in, out)
                if 'weight' in name:
                    weight_norms.append(torch.norm(param, p=1, dim=[1, 2]))
                elif 'bias' in name:
                    bias_norms.append(torch.norm(param, p=1, dim=[1, 2]))
        if not weight_norms:
            return ptu.zeros(self.ensemble_size)
        reg = torch.sum(torch.stack(weight_norms), dim=0)
        if bias_norms:
            reg = reg + torch.sum(torch.stack(bias_norms[:-1]), dim=0)
        return reg  # (ensemble_size,)

    def _get_reg_norm(self):
        """Return cached reg_norm, recomputing only when invalidated."""
        if self._cached_reg_norm is None:
            self._cached_reg_norm = self._compute_reg_norm(self.target_qf)
        return self._cached_reg_norm

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        batch_size = obs.shape[0]

        gt.stamp('preback_start', unique=False)

        # ---- Update Alpha ----
        with autocast():
            new_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
                obs,
                reparameterize=True,
                return_log_prob=True,
            )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha.exp() * (log_pi.float() + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.alpha

        gt.stamp('preback_alpha', unique=False)

        # ---- Cached regularization norm on target network ----
        reg_norm = self._get_reg_norm()  # (ensemble_size,)

        # ---- Update Q-Functions (ensemble) ----
        self.qf_optimizer.zero_grad()
        with autocast():
            with torch.no_grad():
                new_next_actions, _, _, new_log_pi, *_ = self.policy(
                    next_obs,
                    reparameterize=True,
                    return_log_prob=True,
                )
                # Target Q-values from ensemble: (ensemble_size, batch_size)
                target_q_values = self.target_qf(next_obs, new_next_actions)

                # Subtract entropy bonus per ensemble member
                # Use expand (zero-copy) instead of repeat
                new_log_pi_exp = new_log_pi.squeeze(-1).unsqueeze(0).expand(self.ensemble_size, -1)
                reg_norm_exp = (self.weight_reg * reg_norm).unsqueeze(-1).expand(-1, batch_size)

                target_q_values = target_q_values - alpha * new_log_pi_exp + reg_norm_exp

                # Q target: r + gamma * (1 - done) * target_Q
                q_target = self.reward_scale * rewards.T + \
                           (1. - terminals.T) * self.discount * target_q_values

            # Predicted Q-values: (ensemble_size, batch_size)
            predicted_q = self.qf(obs, actions)

            # MSE loss averaged across ensemble
            qf_loss = self.qf_criterion(predicted_q, q_target.detach())

            # OOD regularization: penalize Q-value disagreement across ensemble
            ood_loss = predicted_q.std(dim=0).mean()

            total_qf_loss = qf_loss + self.beta_ood * ood_loss

        self.scaler.scale(total_qf_loss).backward()
        if self.clip_norm > 0:
            self.scaler.unscale_(self.qf_optimizer)
            torch.nn.utils.clip_grad_norm_(self.qf.parameters(), self.clip_norm)
        self.scaler.step(self.qf_optimizer)

        gt.stamp('backward_qf', unique=False)

        # ---- Update Policy (less frequently) ----
        policy_loss = ptu.zeros(1)
        q_std_mean = ptu.zeros(1)

        if self._n_train_steps_total % self.critic_actor_ratio == 0:
            self.policy_optimizer.zero_grad()
            with autocast():
                # Re-evaluate new actions through ensemble
                q_values_dist = self.qf(obs, new_actions.float())  # (ensemble_size, batch_size)

                # Add regularization and subtract entropy (zero-copy expand)
                reg_norm_exp2 = (self.weight_reg * reg_norm).unsqueeze(-1).expand(-1, batch_size)
                log_pi_exp2 = log_pi.float().squeeze(-1).unsqueeze(0).expand(self.ensemble_size, -1)
                q_values_dist = q_values_dist + reg_norm_exp2 - alpha * log_pi_exp2

                # Mean-Std policy: q_mean + beta * q_std
                q_mean = q_values_dist.mean(dim=0)  # (batch_size,)
                q_std = q_values_dist.std(dim=0)     # (batch_size,)
                q_combined = q_mean + self.beta * q_std

                policy_loss_q = -q_combined.mean()

                # Behavior cloning regularization
                bc_loss = F.mse_loss(new_actions.float(), actions.detach())

                policy_loss = self.beta_bc * bc_loss + policy_loss_q
            q_std_mean = q_std.mean()

            self.scaler.scale(policy_loss).backward()
            if self.clip_norm > 0:
                self.scaler.unscale_(self.policy_optimizer)
                ptu.fast_clip_grad_norm(self.policy.parameters(), self.clip_norm)
            self.scaler.step(self.policy_optimizer)

        # Update scaler for next iteration
        self.scaler.update()

        gt.stamp('backward_policy', unique=False)

        # ---- Soft Updates ----
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf, self.target_qf, self.soft_target_tau)
            # Invalidate cached reg_norm since target network changed
            self._cached_reg_norm = None

        gt.stamp('soft_update', unique=False)

        # ---- Save statistics ----
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False

            self.eval_statistics['QF Loss'] = qf_loss.item()
            self.eval_statistics['OOD Loss'] = ood_loss.item()
            self.eval_statistics['Policy Loss'] = policy_loss.item()
            self.eval_statistics['Q Std Mean'] = q_std_mean.item()
            self.eval_statistics['Reg Norm Mean'] = (self.weight_reg * reg_norm.mean()).item()

            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(predicted_q.float().mean(dim=0)),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target.float().mean(dim=0)),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi.float()),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean.float()),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std.float()),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
            self.target_qf,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy.state_dict(),
            qf=self.qf.state_dict(),
            target_qf=self.target_qf.state_dict(),
        )
