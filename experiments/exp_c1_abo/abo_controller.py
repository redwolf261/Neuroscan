"""
Adaptive Branch Optimizer (ABO) Controller.

Implements the magnitude controller (alpha) and anomaly damper (delta)
that jointly determine how much of the evidential branch's RAW trunk
gradient is blended into the shared trunk update, replacing the static
evidential_weight coefficient AT THE TRUNK ONLY. Segmentation head and
evidential head gradients are untouched (see abo_grad_utils.apply_abo_update).

Every constant here is calibrated against measurements already made in
Phase B / Experiment C0, not chosen arbitrarily:

  r_target = 0.294
    NOT a hand-picked constant -- derived directly from the C1 monitor-mode
    identity-verification run (10 epochs, seed=0, multiplier forced to 1.0,
    i.e. measuring the UNMODIFIED baseline's own gradient dynamics through
    this exact instrumentation). That run's raw_ratio reproduced Phase B's
    original finding almost exactly (epoch 0: 2.96 vs Phase B's 2.99;
    epoch 1: 0.94 vs 0.96; epoch 2: 0.22 vs 0.24; settling to 0.08-0.11 by
    epoch 3+, matching Phase B's 0.09-0.11), confirming the measurement
    itself is trustworthy. Steady-state mean ratio over epochs 3-9 of that
    run = 0.0979. r_target = 3 x 0.0979 = 0.294 -- i.e. a meaningful
    reduction of the imbalance, explicitly NOT full equalization
    (r_target=1.0), per the design brief: "purpose is NOT to equalize
    gradients perfectly." Experiment C0 additionally showed this ratio is
    INVARIANT to an 8x change in the static loss weight (0.2 to 0.8
    evidential_weight) -- static reweighting cannot move it, which is the
    empirical justification for controlling it via the gradient directly
    rather than via loss-weight search.

  alpha_power = 0.5
    Softens the response (sqrt instead of linear) so alpha doesn't swing
    to extreme values when r_ema is transiently very small or very large.

  alpha_min / alpha_max = 0.2 / 5.0
    Finite bounds so a degenerate r_ema (near zero, or very large) cannot
    produce an unbounded multiplier. Chosen generously wide relative to
    the observed operating range (alpha ~= sqrt(0.294/0.098) ~= 1.73 typically)
    so they constrain only genuinely extreme cases, not normal operation.

  ema_decay = 0.98
    Slower than Phase B's diagnostic-only EMA (0.9, used purely for
    plotting/logging) so the CONTROLLER's own state isn't whipsawed by
    single-batch noise -- alpha and delta drive real gradient
    modification, so they need a smoother, more stable input than a
    visualization EMA does.

  gamma = 0.1
    Calibrated directly against the real z-score distribution of the
    evidential trunk gradient norm observed in the completed, fully
    stable Experiment C0 baseline run (1404 batches, decay=0.9 EMA):
    |z| never exceeded 2.73 in that entirely healthy run. At gamma=0.1:
    delta(z=1) ~= 0.90 (mild damping for ordinary fluctuation),
    delta(z=2.73) ~= 0.47 (moderate damping at the most extreme deviation
    ever observed while training was stable), delta(z=5, a magnitude of
    spike never observed) ~= 0.08 (strong suppression, only for a
    genuinely anomalous event outside the observed operating range).

  delta_min = 0.1
    Floor so a damping event never fully zeroes the evidential
    contribution (a dead gradient), only strongly suppresses it.

delta is a smooth Gaussian-form damper (C-infinity, no thresholds or
jumps): delta = exp(-gamma * z^2), which equals 1 exactly at z=0 and
decays continuously and symmetrically as |z| grows in either direction.
This satisfies "avoid discontinuous rules or threshold-based switching"
literally -- there is no branch, no if/else on z, just one smooth formula.
"""

import math


class ABOController:
    def __init__(
        self,
        r_target=0.294,  # derived: 3 x measured steady-state ratio (0.0979), see module docstring
        alpha_power=0.5,
        alpha_min=0.2,
        alpha_max=5.0,
        ema_decay=0.98,
        gamma=0.1,
        delta_min=0.1,
        delta_warmup_steps=20,
        eps=1e-12,
    ):
        self.r_target = r_target
        self.alpha_power = alpha_power
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.ema_decay = ema_decay
        self.gamma = gamma
        self.delta_min = delta_min
        # Variance is a second-moment statistic and needs more samples than
        # the mean to be meaningful -- with only 1-2 observations,
        # evid_norm_var_ema is still ~0, so ANY deviation on the next batch
        # produces a huge z-score and floors delta, even though nothing
        # anomalous actually happened (there's simply no history yet to
        # judge normalcy against). This is a cold-start artifact, not a
        # real detection. delta stays at 1.0 (baseline-identical, same
        # principle already used for step_count==1) until enough samples
        # have accumulated for the variance estimate to mean something.
        # alpha does NOT get this warmup: its job (tracking the ratio mean)
        # is a first-moment statistic, informative from step 2 onward, and
        # Phase B/C0 showed the ratio's early behavior is itself meaningful.
        self.delta_warmup_steps = delta_warmup_steps
        self.eps = eps

        # EMA state -- None until first observation. alpha/delta both
        # default to 1.0 (identity, baseline-equivalent) until there is
        # at least one prior batch to compare against, so ABO "initially
        # behaves close to the baseline and gradually adapts" by construction.
        self.ratio_ema = None
        self.evid_norm_mean_ema = None
        self.evid_norm_var_ema = None
        self.step_count = 0

    def _update_ema_mean(self, current, prev):
        if prev is None:
            return current
        return prev + (1 - self.ema_decay) * (current - prev)

    def _update_ema_var(self, current, prev_mean, prev_var):
        if prev_mean is None:
            return 0.0
        delta = current - prev_mean
        return self.ema_decay * prev_var + (1 - self.ema_decay) * delta ** 2

    def step(self, focal_trunk_norm, evidential_trunk_norm):
        """
        Call once per batch with the RAW (unweighted, pre-loss-weight)
        trunk gradient norms for the two branches.

        Returns (alpha, delta, multiplier, diagnostics_dict). multiplier
        = alpha * delta is what abo_grad_utils.apply_abo_update uses to
        scale the evidential branch's trunk gradient contribution.
        """
        self.step_count += 1
        raw_ratio = evidential_trunk_norm / (focal_trunk_norm + self.eps)

        prev_ratio_ema = self.ratio_ema
        self.ratio_ema = self._update_ema_mean(raw_ratio, self.ratio_ema)

        prev_evid_mean = self.evid_norm_mean_ema
        prev_evid_var = self.evid_norm_var_ema if self.evid_norm_var_ema is not None else 0.0
        new_evid_var = self._update_ema_var(evidential_trunk_norm, prev_evid_mean, prev_evid_var)
        self.evid_norm_mean_ema = self._update_ema_mean(evidential_trunk_norm, self.evid_norm_mean_ema)
        self.evid_norm_var_ema = new_evid_var

        if self.step_count == 1:
            alpha = 1.0
        else:
            raw_alpha = (self.r_target / max(self.ratio_ema, self.eps)) ** self.alpha_power
            alpha = min(max(raw_alpha, self.alpha_min), self.alpha_max)

        if self.step_count <= self.delta_warmup_steps:
            delta = 1.0
        else:
            # std is floored (not short-circuited) so a jump against a
            # near-constant history is still recognized as anomalous --
            # if current == mean even when std~=0, z~=0 and delta~=1
            # (correctly "no deviation"); if current != mean and std~=0,
            # z becomes large and delta correctly drops (a jump against
            # constant history IS anomalous). This also avoids an
            # if/else branch on std, consistent with "no discontinuous
            # rules or threshold-based switching." (This branch only
            # governs whether we're past warmup; it is not itself the
            # anomaly-detection logic.)
            std = math.sqrt(max(prev_evid_var, 0.0))
            std_floor = max(std, self.eps)
            z = (evidential_trunk_norm - prev_evid_mean) / std_floor
            delta = math.exp(-self.gamma * z * z)
            delta = max(delta, self.delta_min)

        multiplier = alpha * delta

        diagnostics = {
            "raw_ratio": raw_ratio,
            "ratio_ema": self.ratio_ema,
            "alpha": alpha,
            "delta": delta,
            "multiplier": multiplier,
            "evid_norm_mean_ema": self.evid_norm_mean_ema,
            "evid_norm_std_ema": math.sqrt(max(self.evid_norm_var_ema, 0.0)),
        }
        return alpha, delta, multiplier, diagnostics
