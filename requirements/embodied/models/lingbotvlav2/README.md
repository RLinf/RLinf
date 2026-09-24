# LingBot-VLA 2.0 dependency

RLinf installs the upstream model package separately and provides the PPO
`BasePolicy` adapter. V1 and V2 both import as `lingbotvla`; use separate source
checkouts and virtual environments.

The installer pins the revision in `source.json` and applies `ppo-compat.patch`:

- Lazy dataset imports avoid loading offline data dependencies for inference.
- Checkpoint metadata reconstructs the released architecture.
- MoE backend and activation-checkpointing hooks align rollout and training.

Before/after SHA-256 hashes make patching idempotent and reject conflicting edits,
partial patches or the wrong revision without overwriting user changes. Replace
the patch with a pinned upstream/fork revision only after checkpoint loading,
likelihood replay, backward and PPO resume pass with the same behavior.

Source: [Robbyant/lingbot-vla-v2](https://github.com/robbyant/lingbot-vla-v2/tree/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8).
The source and RLinf compatibility additions use Apache-2.0; see [LICENSE](LICENSE).
The patch retains original notices and marks modified files. Model weights are
downloaded separately under their model-card licenses.
