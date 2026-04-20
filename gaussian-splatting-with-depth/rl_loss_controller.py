"""
rl_loss_controller.py — PPO-based adaptive loss weight controller for Z-Splat.

Problem
-------
Gamma-NLL (ZL ≈ -3.7) is ~100× larger in magnitude than L1 camera loss (≈ 0.03).
Fixed weights cause one loss to dominate the optimiser → poor PSNR. The right
balance is dataset-dependent: a sonar-only scene has no camera loss; a synthetic
RGB+sonar scene may have |ZL|/|L_cam| ≈ 4 instead of 100.

Dataset-agnostic design
-----------------------
Three properties ensure the policy generalises across datasets with different
loss scales without needing re-tuning:

1.  Auto-calibrated initial weights
    The controller observes |ZL| and |L_cam| for the first `auto_init_steps`
    iterations and sets weights so |ZL × z_w| = |L_cam × cam_w| from the start.
    Same balanced initial state on every dataset, regardless of absolute loss
    magnitudes.  No user-supplied initial weights needed.

2.  State expressed in relative space
    log(z_w) and log(cam_w) are replaced by log(z_w / z_w_init) and
    log(cam_w / cam_w_init).  Both equal 0 at the auto-calibrated starting
    point — the policy always sees 0 = balanced, regardless of scale.

3.  Fractional sparse reward
    ΔPSNR / max(baseline_psnr, 1.0) is a fractional improvement from the first
    eval checkpoint, comparable across scenes with different PSNR ranges
    (sonar-only ~15 dB vs RGB ~36 dB).

RL formulation
--------------
  State  (8-dim):
    s = [log(|ZL_ema × z_w| / |L_cam_ema × cam_w|),  # balance (0 = target)
         ZL_rel_trend,                                 # ZL relative rate of change
         L_cam_rel_trend,                              # L_cam relative rate of change
         r_tilde_mean - 0.5,                           # reflectivity drift
         r_tilde_trend,                                # r_tilde rate of change
         log(z_w   / z_w_init),                        # weight deviation from calibration
         log(cam_w / cam_w_init),                      # weight deviation from calibration
         step / max_steps]                             # training progress

  Action (2-dim, continuous):
    a = [Δlog(z_weight), Δlog(cam_weight)]   clipped to [-A_MAX, A_MAX]
    New weight = old_weight × exp(action)

  Reward (two timescales):
    Dense  (every adapt_every steps):
      r = -|balance_after| + 0.5 * (|balance_before| - |balance_after|)
    Sparse (at eval checkpoints):
      r = w_psnr * (ΔPSNR / baseline_psnr) + w_snll * (-Δsonar_nll / |baseline_nll|)

Cross-run learning
------------------
Policy weights are saved after each run.  On the next run (same or different
dataset) the policy is loaded and fine-tuned.  Auto-calibration re-runs every
time, so the policy only needs to learn the *shape* of the weight schedule, not
the absolute scale.

Usage
-----
    from rl_loss_controller import RLLossController

    ctrl = RLLossController(
        policy_path = "rl_policy.pt",   # load prior policy if exists
        target_ratio = 1.0,             # aim for |ZL_contrib| == |L_cam_contrib|
        max_steps    = 30_000,
    )

    # Inside training loop (every iteration):
    z_w, cam_w = ctrl.step(ZL.item(), L_cam.item(),
                           r_tilde_mean=..., iteration=iteration)

    # At eval checkpoint (after computing test PSNR and sonar NLL):
    ctrl.checkpoint_reward(psnr=psnr_test, sonar_nll=avg_snll, iteration=iteration)

    # After training — save policy so next run starts from here:
    ctrl.save("rl_policy.pt")
"""

import math
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ── Constants ────────────────────────────────────────────────────────────────

STATE_DIM  = 8
ACTION_DIM = 2
A_MAX      = 0.20          # max |Δlog(weight)| per step → ×exp(0.20) ≈ 1.22 per adapt step
HIDDEN     = 64
GAMMA      = 0.99          # discount factor
GAE_LAMBDA = 0.95          # GAE lambda
PPO_CLIP   = 0.2           # PPO clip ratio
PPO_EPOCHS = 4             # gradient steps per PPO update
MINI_BATCH = 32
LR_POLICY  = 3e-4
ENTROPY_COEF = 0.01        # entropy bonus to encourage exploration

# Maximum factor by which weights may deviate from auto-calibrated starting
# point in either direction.  20× ≈ 3 orders of magnitude of log headroom.
SCALE_LIMIT = 20.0


# ── Neural network ────────────────────────────────────────────────────────────

class _ActorCritic(nn.Module):
    """Shared-trunk actor-critic MLP."""

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden: int = HIDDEN):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),   nn.Tanh(),
        )
        self.actor_mean    = nn.Linear(hidden, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic        = nn.Linear(hidden, 1)

        # Initialise actor output near zero so early actions are small
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)

    def forward(self, state: torch.Tensor):
        h = self.trunk(state)
        mean    = self.actor_mean(h)
        std     = self.actor_log_std.exp().expand_as(mean)
        value   = self.critic(h).squeeze(-1)
        return mean, std, value

    def get_action(self, state: torch.Tensor):
        mean, std, value = self(state)
        dist = Normal(mean, std)
        action   = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.clamp(-A_MAX, A_MAX), log_prob, value

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        mean, std, value = self(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, value, entropy


# ── PPO rollout buffer ────────────────────────────────────────────────────────

class _PPOBuffer:
    """Stores (s, a, log_π, r, done) tuples for one PPO update cycle."""

    def __init__(self):
        self.states    = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.values    = []
        self.dones     = []

    def push(self, state, action, log_prob, reward, value, done=False):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def compute_returns(self, last_value: float, gamma: float = GAMMA,
                        lam: float = GAE_LAMBDA):
        """Generalised Advantage Estimation."""
        returns    = []
        advantages = []
        gae = 0.0
        next_val = last_value
        for r, v, d in zip(reversed(self.rewards), reversed(self.values),
                           reversed(self.dones)):
            delta = r + gamma * next_val * (1 - d) - v
            gae   = delta + gamma * lam * (1 - d) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + v)
            next_val = v
        return returns, advantages

    def as_tensors(self, device):
        s  = torch.stack(self.states).to(device)
        a  = torch.stack(self.actions).to(device)
        lp = torch.stack(self.log_probs).to(device)
        return s, a, lp

    def clear(self):
        self.__init__()


# ── Observation buffer (sliding window statistics) ───────────────────────────

class _ObsBuffer:
    """Keeps a rolling history of per-step scalars for computing trends."""

    def __init__(self, window: int = 200):
        self.zl      = deque(maxlen=window)
        self.l_cam   = deque(maxlen=window)
        self.r_tilde = deque(maxlen=window)

    def push(self, zl: float, l_cam: float, r_tilde: float):
        self.zl.append(zl)
        self.l_cam.append(l_cam)
        self.r_tilde.append(r_tilde)

    def _rel_trend(self, buf: deque) -> float:
        """
        Relative linear trend: (mean(last half) - mean(first half)) / |mean|.
        Scale-invariant: same value whether |mean| is 0.03 or 3.7.
        """
        if len(buf) < 10:
            return 0.0
        arr  = np.array(buf)
        mid  = len(arr) // 2
        diff = arr[mid:].mean() - arr[:mid].mean()
        base = abs(arr.mean()) + 1e-8
        return float(np.clip(diff / base, -3.0, 3.0))

    def ema(self, buf: deque, alpha: float = 0.05) -> float:
        if not buf:
            return 0.0
        v = buf[0]
        for x in list(buf)[1:]:
            v = alpha * x + (1 - alpha) * v
        return float(v)

    def state(self, z_weight: float, cam_weight: float,
              z_w_init: float, cam_w_init: float,
              step: int, max_steps: int, target_ratio: float) -> torch.Tensor:
        """
        Build the 8-dim state vector.

        Features 5 and 6 are expressed relative to the auto-calibrated starting
        weights (both = 0 at the balanced starting point on any dataset).
        """
        zl_ema    = self.ema(self.zl)
        lcam_ema  = self.ema(self.l_cam)
        r_ema     = self.ema(self.r_tilde)

        log_ratio = math.log(max(abs(zl_ema) * z_weight, 1e-8) /
                             max(abs(lcam_ema) * cam_weight, 1e-8))
        log_target = math.log(max(target_ratio, 1e-8))
        balance   = float(np.clip(log_ratio - log_target, -5.0, 5.0))

        # Relative weight features: 0 at calibrated start, scale-invariant
        log_z_rel   = float(np.clip(
            math.log(max(z_weight, 1e-8) / max(z_w_init, 1e-8)), -5.0, 5.0))
        log_cam_rel = float(np.clip(
            math.log(max(cam_weight, 1e-8) / max(cam_w_init, 1e-8)), -5.0, 5.0))

        s = torch.tensor([
            balance,                                          # 0: loss balance (0=good)
            self._rel_trend(self.zl),                         # 1: ZL relative trend
            self._rel_trend(self.l_cam),                      # 2: L_cam relative trend
            float(np.clip(r_ema - 0.5, -0.5, 0.5)),          # 3: r_tilde drift
            self._rel_trend(self.r_tilde),                    # 4: r_tilde relative trend
            log_z_rel,                                        # 5: log(z_w / z_w_init)
            log_cam_rel,                                      # 6: log(cam_w / cam_w_init)
            float(step / max(max_steps, 1)),                  # 7: step fraction
        ], dtype=torch.float32)
        return s


# ── Main controller ───────────────────────────────────────────────────────────

class RLLossController:
    """
    PPO-based adaptive loss weight controller.

    Adjusts z_weight and cam_weight every `adapt_every` training iterations to
    keep |ZL × z_w| ≈ |L_cam × cam_w| × target_ratio.

    Dataset-agnostic: auto-calibrates initial weights from the first
    `auto_init_steps` iterations so no dataset-specific tuning is needed.
    The policy is saved after training and loaded on subsequent runs for
    cross-run learning — it generalises because the state is expressed in
    relative (not absolute) terms.

    Parameters
    ----------
    policy_path      : path to load/save PPO policy (.pt)
    target_ratio     : desired |ZL_contrib| / |L_cam_contrib| ratio (1.0 = equal)
    max_steps        : total training iterations (for step_frac feature)
    adapt_every      : iters between weight updates
    auto_init_steps  : warmup iters to observe losses before calibrating weights
    w_psnr           : sparse reward weight for fractional ΔPSNR
    w_snll           : sparse reward weight for fractional Δsonar_NLL
    verbose          : print weight updates and rewards
    """

    def __init__(
        self,
        policy_path:      str   = "",
        target_ratio:     float = 1.0,
        max_steps:        int   = 30_000,
        adapt_every:      int   = 200,
        auto_init_steps:  int   = 100,
        w_psnr:           float = 10.0,
        w_snll:           float = 2.0,
        verbose:          bool  = True,
    ):
        self.policy_path    = policy_path
        self.target_ratio   = target_ratio
        self.max_steps      = max_steps
        self.adapt_every    = adapt_every
        self.auto_init_steps = auto_init_steps
        self.w_psnr         = w_psnr
        self.w_snll         = w_snll
        self.verbose        = verbose

        self.device = torch.device("cpu")   # policy is small — stays on CPU

        # Policy + single optimiser
        self.policy = _ActorCritic(STATE_DIM, ACTION_DIM, HIDDEN).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=LR_POLICY)

        # ── Auto-calibration state ──────────────────────────────────────────
        # Weights start as placeholders; set by _calibrate_weights() after
        # observing the first auto_init_steps iterations.
        self.z_w        = 1.0
        self.cam_w      = 1.0
        self._z_w_init  = 1.0   # set by calibration; used as state reference
        self._cam_w_init = 1.0
        self._calibrated = False
        self._calib_zl  : list = []    # |ZL| samples during warmup
        self._calib_lcam: list = []    # |L_cam| samples during warmup

        # Weight bounds are set after calibration (±SCALE_LIMIT from init)
        self._z_bounds   = (1e-5, 1e5)
        self._cam_bounds = (1e-5, 1e5)

        # ── Policy loading ──────────────────────────────────────────────────
        self._policy_trained = False
        if policy_path and os.path.exists(policy_path):
            self._load_policy(policy_path)
            self._policy_trained = True
            if verbose:
                print(f"[RLCtrl] Loaded policy from {policy_path}")
        else:
            if verbose:
                print("[RLCtrl] No policy found — using adaptive GradNorm "
                      "fallback until calibration + first PPO update")

        # ── Buffers ─────────────────────────────────────────────────────────
        self._obs = _ObsBuffer(window=200)
        self._buf = _PPOBuffer()

        # Current transition
        self._cur_state  = None
        self._cur_action = None
        self._cur_logp   = None
        self._cur_value  = None

        # Checkpoint tracking for sparse reward
        self._last_psnr       = None
        self._last_snll       = None
        self._baseline_psnr   = None   # first eval checkpoint PSNR (for normalisation)
        self._baseline_snll   = None
        self._dense_rewards_since_checkpoint: list = []

        # Step counters
        self._iter        = 0
        self._adapt_step  = 0
        self._last_printed_adapt = -1

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self, zl: float, l_cam: float, r_tilde_mean: float,
             iteration: int) -> tuple:
        """
        Called every training iteration.
        Returns current (z_weight, cam_weight).

        During the first `auto_init_steps` iterations the weights are held fixed
        while the controller observes the natural loss scales.  After calibration
        the RL/GradNorm control loop begins.
        """
        self._iter = iteration
        self._obs.push(zl, l_cam, r_tilde_mean)

        # ── Calibration phase ───────────────────────────────────────────────
        if not self._calibrated:
            self._calib_zl.append(abs(zl))
            self._calib_lcam.append(abs(l_cam))
            if len(self._calib_zl) >= self.auto_init_steps:
                self._calibrate_weights()
            return self.z_w, self.cam_w

        # ── Control phase ───────────────────────────────────────────────────
        if iteration % self.adapt_every == 0:
            self._adapt(iteration)

        return self.z_w, self.cam_w

    def checkpoint_reward(self, psnr: float, sonar_nll: float, iteration: int):
        """
        Called at eval checkpoints with test PSNR and sonar Gamma-NLL.
        Issues a normalised sparse reward and (if enough data) runs PPO update.
        """
        sparse = 0.0
        if self._last_psnr is not None:
            # Fractional improvement — comparable across different PSNR ranges
            baseline_p = max(abs(self._baseline_psnr), 1.0)
            baseline_n = max(abs(self._baseline_snll), 1e-4)
            delta_psnr = (psnr - self._last_psnr) / baseline_p
            delta_snll = (sonar_nll - self._last_snll) / baseline_n
            sparse = self.w_psnr * delta_psnr + self.w_snll * (-delta_snll)
            if self.verbose:
                print(f"[RLCtrl iter {iteration}] "
                      f"ΔPSNR={psnr - self._last_psnr:+.3f} "
                      f"({delta_psnr:+.4f} frac)  "
                      f"Δsnll={sonar_nll - self._last_snll:+.4f} "
                      f"({delta_snll:+.4f} frac)  "
                      f"sparse_r={sparse:+.3f}  "
                      f"z_w={self.z_w:.4f}  cam_w={self.cam_w:.4f}", flush=True)

        # Record baselines on first checkpoint
        if self._baseline_psnr is None:
            self._baseline_psnr = psnr
            self._baseline_snll = sonar_nll

        self._last_psnr = psnr
        self._last_snll = sonar_nll

        # Back-fill sparse reward into recent buffer entries
        if self._buf and sparse != 0.0:
            n       = len(self._buf)
            n_since = max(1, len(self._dense_rewards_since_checkpoint))
            start   = max(0, n - n_since)
            for i in range(start, n):
                self._buf.rewards[i] += sparse / max(n_since, 1)

        self._dense_rewards_since_checkpoint = []

        # PPO update if we have enough transitions
        if len(self._buf) >= MINI_BATCH * 2:
            self._ppo_update()

    def save(self, path: str = ""):
        """Save policy weights. Called after training completes."""
        target = path or self.policy_path
        if not target:
            return
        # Save only the policy network weights — no dataset-specific z_w/cam_w.
        # Auto-calibration will determine appropriate starting weights on the next run.
        torch.save({"policy_state": self.policy.state_dict()}, target)
        if self.verbose:
            print(f"[RLCtrl] Policy saved to {target}")

    # ── Internal: auto-calibration ───────────────────────────────────────────

    def _calibrate_weights(self):
        """
        Set z_w and cam_w so |ZL × z_w| = |L_cam × cam_w| based on observed
        loss magnitudes.  Also sets the per-run weight bounds (±SCALE_LIMIT from
        the calibrated starting point).

        This runs once per training session and is dataset-agnostic: the same
        balance equation applies regardless of absolute loss scale.
        """
        zl_mean  = max(float(np.mean(self._calib_zl)),  1e-8)
        lc_mean  = max(float(np.mean(self._calib_lcam)), 1e-8)

        # Geometric mean: z_w = sqrt(L_cam / ZL), cam_w = sqrt(ZL / L_cam)
        # → |ZL × z_w| = sqrt(ZL × L_cam) = |L_cam × cam_w|  (target_ratio = 1)
        z_w_cal   = math.sqrt(lc_mean / zl_mean)
        cam_w_cal = math.sqrt(zl_mean / lc_mean)

        # Scale for target_ratio ≠ 1:  want ZL_contrib = target * L_cam_contrib
        # → z_w = sqrt(target × L_cam / ZL), cam_w = sqrt(ZL / (target × L_cam))
        if self.target_ratio != 1.0:
            z_w_cal   *= math.sqrt(self.target_ratio)
            cam_w_cal /= math.sqrt(self.target_ratio)

        self.z_w         = z_w_cal
        self.cam_w       = cam_w_cal
        self._z_w_init   = z_w_cal
        self._cam_w_init = cam_w_cal

        # Bounds relative to calibrated start — same semantic on any dataset
        self._z_bounds   = (z_w_cal / SCALE_LIMIT, z_w_cal * SCALE_LIMIT)
        self._cam_bounds  = (cam_w_cal / SCALE_LIMIT, cam_w_cal * SCALE_LIMIT)

        self._calibrated = True

        if self.verbose:
            contrib = zl_mean * self.z_w
            print(f"[RLCtrl] Auto-calibrated from {len(self._calib_zl)} steps: "
                  f"|ZL|={zl_mean:.4f}  |L_cam|={lc_mean:.4f}  "
                  f"z_w={self.z_w:.4f}  cam_w={self.cam_w:.4f}  "
                  f"(each contrib≈{contrib:.4f})", flush=True)

    # ── Internal: adapt step ─────────────────────────────────────────────────

    def _adapt(self, iteration: int):
        """Compute new weights using policy (or GradNorm fallback), store transition."""
        state = self._obs.state(
            self.z_w, self.cam_w,
            self._z_w_init, self._cam_w_init,
            iteration, self.max_steps, self.target_ratio)
        state_t = state.unsqueeze(0).to(self.device)

        if self._policy_trained:
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(state_t)
            action   = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            value    = value.squeeze(0).item()
        else:
            # GradNorm-style fallback: push weights toward balance
            balance   = state[0].item()           # log(actual_ratio) - log(target)
            delta_z   = float(np.clip(-0.5 * balance, -A_MAX, A_MAX))
            delta_cam = float(np.clip( 0.1 * balance, -A_MAX, A_MAX))
            delta_z   += float(np.random.normal(0, 0.02))
            delta_cam += float(np.random.normal(0, 0.02))
            action    = torch.tensor([delta_z, delta_cam], dtype=torch.float32)
            with torch.no_grad():
                _, log_prob, value = self.policy.get_action(state_t)
            log_prob = log_prob.squeeze(0)
            value    = value.squeeze(0).item()

        # Apply action
        old_z   = self.z_w
        old_cam = self.cam_w
        self.z_w   = float(np.clip(
            self.z_w   * math.exp(action[0].item()), *self._z_bounds))
        self.cam_w = float(np.clip(
            self.cam_w * math.exp(action[1].item()), *self._cam_bounds))

        # Dense balance reward (improvement-shaped)
        balance_after = self._obs.state(
            self.z_w, self.cam_w,
            self._z_w_init, self._cam_w_init,
            iteration, self.max_steps, self.target_ratio)[0].item()
        balance_before = state[0].item()
        r_dense = -abs(balance_after) + 0.5 * (abs(balance_before) - abs(balance_after))
        r_dense = float(np.clip(r_dense, -2.0, 2.0))

        self._buf.push(state, action, log_prob, r_dense, value)
        self._dense_rewards_since_checkpoint.append(r_dense)
        self._adapt_step += 1

        if self.verbose and self._adapt_step % 5 == 1:
            zl_ema  = self._obs.ema(self._obs.zl)
            cam_ema = self._obs.ema(self._obs.l_cam)
            print(f"[RLCtrl iter {iteration}] "
                  f"z_w={old_z:.4f}→{self.z_w:.4f}  "
                  f"cam_w={old_cam:.4f}→{self.cam_w:.4f}  "
                  f"|ZL|×z={abs(zl_ema)*self.z_w:.4f}  "
                  f"|Lc|×c={abs(cam_ema)*self.cam_w:.4f}  "
                  f"balance={balance_after:+.3f}  "
                  f"r_dense={r_dense:+.3f}", flush=True)

    # ── Internal: PPO update ─────────────────────────────────────────────────

    def _ppo_update(self):
        """Run PPO on the accumulated buffer, then clear it."""
        if len(self._buf) < MINI_BATCH:
            return

        # Bootstrap value for last state
        with torch.no_grad():
            last_state = self._obs.state(
                self.z_w, self.cam_w,
                self._z_w_init, self._cam_w_init,
                self._iter, self.max_steps, self.target_ratio
            ).unsqueeze(0).to(self.device)
            _, _, last_val = self.policy(last_state)
            last_val = last_val.item()

        returns, advantages = self._buf.compute_returns(last_val)
        states, actions, old_log_probs = self._buf.as_tensors(self.device)

        ret_t = torch.tensor(returns,    dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        N = len(self._buf)
        total_actor_loss  = 0.0
        total_critic_loss = 0.0

        for _ in range(PPO_EPOCHS):
            idx = torch.randperm(N)
            for start in range(0, N, MINI_BATCH):
                mb = idx[start:start + MINI_BATCH]
                if len(mb) < 2:
                    continue

                new_log_probs, values, entropy = self.policy.evaluate(
                    states[mb], actions[mb])

                ratio       = (new_log_probs - old_log_probs[mb]).exp()
                surr1       = ratio * adv_t[mb]
                surr2       = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv_t[mb]
                actor_loss  = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy.mean()
                critic_loss = F.mse_loss(values, ret_t[mb])
                combined    = actor_loss + 0.5 * critic_loss

                self.opt.zero_grad()
                combined.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.opt.step()

                total_actor_loss  += actor_loss.item()
                total_critic_loss += critic_loss.item()

        if self.verbose:
            n_updates = PPO_EPOCHS * max(1, N // MINI_BATCH)
            print(f"[RLCtrl PPO update] transitions={N}  "
                  f"actor_loss={total_actor_loss/n_updates:.4f}  "
                  f"critic_loss={total_critic_loss/n_updates:.4f}", flush=True)

        self._policy_trained = True
        self._buf.clear()

    def _load_policy(self, path: str):
        """Load only the policy network weights — not dataset-specific z_w/cam_w."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(ckpt["policy_state"])
        # Do NOT restore z_w/cam_w: auto-calibration sets them fresh per dataset.


# ── Convenience: build controller from train_v2 args ─────────────────────────

def make_controller(args, max_steps: int) -> "RLLossController":
    """
    Build an RLLossController from train_v2.py argparse args.

    Looks for a shared policy at the path given by --rl_policy_path, or falls
    back to <model_path>/rl_policy.pt (run-local).  Weights are auto-calibrated
    from the first `auto_init_steps` iterations — no need to specify
    z_loss_weight / camera_loss_weight when using the RL controller.
    """
    # Prefer a user-supplied shared policy path so cross-dataset learning
    # accumulates in one file; fall back to run-local path.
    policy_path = getattr(args, "rl_policy_path", None) or os.path.join(
        getattr(args, "model_path", "."), "rl_policy.pt")

    return RLLossController(
        policy_path     = policy_path,
        target_ratio    = getattr(args, "rl_target_ratio",    1.0),
        max_steps       = max_steps,
        adapt_every     = getattr(args, "rl_adapt_every",     200),
        auto_init_steps = getattr(args, "rl_auto_init_steps", 100),
        verbose         = True,
    )


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Quick smoke test: simulate 30K steps with two synthetic datasets."""
    import random

    def run_sim(name, zl_base, lcam_base, psnr_start):
        print(f"\n{'='*60}")
        print(f"  Simulating dataset: {name}  "
              f"|ZL|≈{abs(zl_base):.3f}  |L_cam|≈{abs(lcam_base):.4f}")
        print(f"{'='*60}")

        ctrl = RLLossController(
            policy_path="rl_policy_test.pt" if os.path.exists("rl_policy_test.pt") else "",
            max_steps=30_000, adapt_every=200, auto_init_steps=100, verbose=True,
        )

        psnr = psnr_start
        snll = zl_base

        for it in range(1, 30_001):
            zl    = zl_base + random.gauss(0, abs(zl_base) * 0.05)
            l_cam = lcam_base * math.exp(-it / 20_000) + random.gauss(0, abs(lcam_base) * 0.05)
            r     = 0.5 + (it / 60_000)
            z_w, cam_w = ctrl.step(zl, l_cam, r, it)

            if it % 3000 == 0:
                balance = abs(math.log(max(abs(zl) * z_w, 1e-8) /
                                       max(abs(l_cam) * cam_w, 1e-8)))
                psnr += random.gauss(0.5 - 0.1 * balance, 0.3)
                snll  = zl_base - it / 100_000
                ctrl.checkpoint_reward(psnr, snll, it)
                print(f"  [sim {it}] z_w={z_w:.4f} cam_w={cam_w:.4f} PSNR={psnr:.2f}")

        ctrl.save("rl_policy_test.pt")

    # Dataset 1: AONeuS-like  (|ZL|≈3.7, |L_cam|≈0.03, PSNR≈25)
    run_sim("AONeuS-like",   zl_base=-3.7,  lcam_base=0.03, psnr_start=25.0)

    # Dataset 2: different scale  (|ZL|≈0.8, |L_cam|≈0.5, PSNR≈18)
    # Policy loaded from run 1 — should still auto-calibrate correctly
    run_sim("different-scale", zl_base=-0.8, lcam_base=0.5,  psnr_start=18.0)

    print("\nSmoke test complete.")
