# GitHub Actions

The upstream SGLang workflow set is intentionally not included in this FOCUS
fork.

Those workflows depend on upstream-only infrastructure such as self-hosted GPU
runners, release credentials, CI permission bots, and nightly hardware. In a
public FOCUS artifact they produce noisy failures that do not validate the
FOCUS changes.

FOCUS throughput and accuracy validation artifacts are kept under
`results/focus_sglang/`.
