"""Encoder / Predictor / Policy head + EMA. Small ConvNet, CPU/MPS friendly."""
import copy

import torch
import torch.nn as nn

D_LATENT = 128


class Encoder(nn.Module):
    """56x56x6 (two stacked frames) -> d=128 latent.

    Output LayerNorm fixes the *gauge* (units) of the latent space: without a
    scale anchor, the free-energy entropy term inflates the norm (energy is
    quadratic in scale, entropy only logarithmic, EMA target lags). With the
    gauge fixed, the entropy term fights the real collapse mode: rank/dim
    collapse (logdet -> -inf), which remains fully detectable.
    """

    def __init__(self, in_ch=6, d=D_LATENT):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 4, stride=2, padding=1),
                nn.GroupNorm(8, cout), nn.SiLU())
        self.net = nn.Sequential(
            block(in_ch, 32),    # 28
            block(32, 64),       # 14
            block(64, 128),      # 7
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GroupNorm(8, 256), nn.SiLU(),  # 4
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.GroupNorm(8, 256), nn.SiLU(),  # 2
            nn.Flatten(),        # 256*2*2 = 1024
            nn.Linear(1024, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, d), nn.LayerNorm(d))
        self.d = d

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    """Latent dynamics: (z_t, a_t) -> z_{t+1}."""

    def __init__(self, d=D_LATENT, a_dim=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + a_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, d))

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))


class PolicyHead(nn.Module):
    """z_t (+ normalized agent pos) -> action *delta* (action - agent) / 512.

    Predicting the displacement rather than the absolute target prevents the
    trivial BC solution (output current agent pos -> agent never moves).
    """

    def __init__(self, d=D_LATENT, extra=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + extra, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2)

        )

    def forward(self, z, ag):
        return self.net(torch.cat([z, ag], dim=-1))


class FlowHead(nn.Module):
    """Flow-matching action head: velocity field v(x_t, t | z, ag) on straight
    CondOT paths x_t = (1-t)*x0 + t*a, x0 ~ N(0,I).

    Why: endgame PushT states admit several valid pushes; MSE regression
    averages those modes into an off-manifold action. A learned velocity
    field transports the Gaussian prior to *one mode* of the intent
    distribution instead of the mean of all modes. Actions are pre-scaled
    to ~unit std (act_std) so the prior matches the data scale.
    """

    def __init__(self, d=D_LATENT, extra=2, hidden=256, n_layers=3):
        super().__init__()
        cin = d + extra + 3  # z + ag + (x, y, t)
        mods = []
        for i in range(n_layers):
            mods += [nn.Linear(cin if i == 0 else hidden, hidden),
                     nn.LayerNorm(hidden), nn.SiLU()]
        mods += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*mods)

    def forward(self, x, t, z, ag):
        """x: (B,2) noisy action; t: (B,1) in [0,1]; -> velocity (B,2)."""
        return self.net(torch.cat([z, ag, x, t], dim=-1))


@torch.no_grad()
def flow_sample(head, z, ag, act_std, n_steps=20):
    """Euler-integrate the ODE from N(0,I) to an action sample; rescale."""
    x = torch.randn(z.shape[0], 2, device=z.device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((z.shape[0], 1), i * dt, device=z.device)
        x = x + head(x, t, z, ag) * dt
    return x * act_std  # (B,2) in delta units


@torch.no_grad()
def emma_update(online, target, momentum=0.999):
    for p, pt in zip(online.parameters(), target.parameters()):
        pt.mul_(momentum).add_(p.detach(), alpha=1 - momentum)


def make_target(online):
    target = copy.deepcopy(online)
    for p in target.parameters():
        p.requires_grad_(False)
    return target


def rollout_latents(predictor, z, actions, k):
    """Iteratively predict k steps. z: (B,d), actions: (B,K,2). Returns list of k predictions."""
    preds = []
    zi = z
    for i in range(k):
        zi = predictor(zi, actions[:, i])
        preds.append(zi)
    return preds
