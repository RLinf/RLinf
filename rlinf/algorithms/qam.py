# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch


def adjoint_targets(model, observations, initial_noise, step_noises):
    """Generate detached stochastic paths and backward adjoints.

    The last forward step uses the slow ODE. The backward drift is evaluated at
    t+h, including the last step, matching the published reference implementation.
    """
    h = 1 / model.flow_steps
    x = initial_noise
    xs, times = [], []
    with torch.no_grad():
        for i in range(model.flow_steps):
            t = torch.full_like(x[..., :1], i * h)
            xs.append(x)
            times.append(t)
            sigma = (2 * (1 - t + h) / (t + h)).sqrt()
            if i == model.flow_steps - 1:
                x = x + h * model.target_actor_slow(observations, x, t)
            else:
                v = model.actor_fast(observations, x, t)
                x = x + h * (2 * v - x / (t + h)) + h**0.5 * sigma * step_noises[i]
    with torch.enable_grad():
        end = x.detach().requires_grad_(True)
        q = model.target_critic(observations, end.clamp(-1, 1)).mean(0).sum()
        adj = -model.cfg.inv_temp * torch.autograd.grad(q, end)[0]
        terminal_adj = adj.detach()
        adjs = []
        for i in reversed(range(model.flow_steps)):
            xi = xs[i].detach().requires_grad_(True)
            t = times[i] + h
            drift = 2 * model.target_actor_slow(observations, xi, t) - xi / t
            vjp = torch.autograd.grad(drift, xi, grad_outputs=adj)[0]
            adj = (adj + h * vjp).detach()
            adjs.append(adj)
    return (
        torch.stack(xs).detach(),
        torch.stack(adjs[::-1]),
        torch.stack(times),
        terminal_adj,
    )


def qam_loss(model, batch, generator=None, noise=None):
    """Compute all losses against the same pre-update parameters.

    Explicit noise injection is for cross-framework parity tests. Targets never
    backpropagate through trajectory construction or the backward adjoint solver.
    """
    obs = batch["observations"]
    actions = batch["actions"].flatten(1)
    b, a = actions.shape
    if noise is None:

        def normal(shape):
            return torch.randn(shape, device=obs.device, generator=generator)

        noise = {
            "next": normal((b, model.cfg.best_of_n, a)),
            "flow": normal((b, a)),
            "time": torch.rand((b, 1), device=obs.device, generator=generator),
            "initial": normal((b, a)),
            "steps": normal((model.flow_steps, b, a)),
        }
    with torch.no_grad():
        nxt = batch["next_observations"][:, -1]
        next_actions = model.sample_actions(nxt, noises=noise["next"])
        qs = model.target_critic(nxt, next_actions)
        pessimistic_q = qs.mean(0) - model.cfg.rho * qs.std(0, correction=0)
        target = (
            batch["rewards"][:, -1]
            + model.cfg.discount**model.horizon * batch["masks"][:, -1] * pessimistic_q
        )
    q = model.critic(obs, actions)
    valid = batch["valid"][:, -1]
    critic_loss = ((q - target).square() * valid).mean()
    t, x0 = noise["time"], noise["flow"]
    velocity = model.actor_slow(obs, (1 - t) * x0 + t * actions, t)
    flow_loss = ((velocity - (actions - x0)).square().mean(-1) * valid).mean()
    xs, adjs, ts, terminal_adj = adjoint_targets(
        model, obs, noise["initial"], noise["steps"]
    )
    repeated_obs = obs.unsqueeze(0).expand(model.flow_steps, -1, -1)
    fine = model.actor_fast(repeated_obs, xs, ts)
    with torch.no_grad():
        base = model.target_actor_slow(repeated_obs, xs, ts)
    h = 1 / model.flow_steps
    sigma = (2 * (1 - ts + h) / (ts + h)).sqrt()
    adj_loss = ((fine - base) * 2 / sigma + sigma * adjs).square().sum(-1).sum(0).mean()
    total = critic_loss + flow_loss + adj_loss
    return total, {
        "critic_loss": critic_loss.detach(),
        "flow_loss": flow_loss.detach(),
        "adj_loss": adj_loss.detach(),
        "q_mean": q.detach().mean(),
        "adj_max": terminal_adj.abs().max(),
        "total_loss": total.detach(),
    }


def qam_update(model, optimizer, batch, generator=None, noise=None):
    """One Adam update with global clipping and pre-update-parameter EMA."""
    optimizer.zero_grad(set_to_none=True)
    loss, info = qam_loss(model, batch, generator, noise)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    # Official QAM mixes OLD online weights into targets, before the Adam step.
    with torch.no_grad():
        for source, target in (
            (model.critic, model.target_critic),
            (model.actor_slow, model.target_actor_slow),
        ):
            for p, tp in zip(source.parameters(), target.parameters()):
                tp.mul_(1 - model.cfg.tau).add_(p, alpha=model.cfg.tau)
    optimizer.step()
    return {**info, "grad_norm": grad_norm.detach()}
