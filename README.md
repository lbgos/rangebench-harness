# rangebench

A self-hosted benchmark harness for evaluating autonomous coding agents against containerized challenge tasks. Runs entirely on your own hardware: every attempt gets a fresh Docker Compose environment, a randomized flag, and a scoring loop that cannot be tricked by a model that read the writeup.

**The scored task suite is private and separate.** This repository ships the harness plus one example task (`tasks/jwt-none`, a classic JWT `alg=none` confusion) so you can run the full loop end to end out of the box.

## What it does

- **Isolated per-attempt environments.** Each attempt builds a scoped Compose project with internal-only networks. The Compose config is verified before anything starts: networks must be internal and project-scoped, services must declare their networks, nothing may be external. The agent container is joined only to declared networks.
- **Randomized per-attempt flags.** Stage flags are generated per attempt (`flag{32 hex}`) and read back only at scoring time, so memorized answers transfer nothing between attempts.
- **pass@k scoring.** Per-task pass@1..3 over N trials with the standard unbiased estimator, plus pooled average solve rates and Wilson intervals. `pass@3` capability sits next to the average so flaky-but-capable and stable agents are distinguishable.
- **Failure taxonomy.** Every ending is classified (solved / provider error / env error / protocol error / budget exhausted) with budget exhaustion kept separate from agent-gave-up, so calibration signals never pollute capability numbers.
- **Context compaction.** When an attempt's transcript grows past the model window, a summarization engine condenses the middle while confirmed stage facts and submission records are carried outside the lossy summary, with a deterministic fallback path that never calls the LLM.
- **Observation pager.** Oversized command output is saved in full to the attempt environment and shown as a bounded head/tail preview with a pointer, instead of being truncated away.
- **Wall-clock tiering.** Per-tier whole-attempt time caps act as a hang guard, with a per-run scale factor for models that generate slower. Turn caps are deliberately absent: the harness records steps and tokens without limiting reasoning.
- **Identity hashing.** Every task has a content identity hash; the run manifest records the harness source hash, task-set hash, and attacker image digest, so any drift mid-run invalidates the affected attempt.

## Quickstart

```bash
make check            # lint + typecheck + tests (no Docker needed)
python3 -m rangebench preflight   # build attacker image, pull task images
python3 -m rangebench list
python3 -m rangebench check jwt-none          # run the oracle against a live env
python3 -m rangebench run --model <model> --base-url <openai-compatible endpoint> --trials 3
```

An OpenAI-compatible endpoint is all that is required (the harness speaks plain `/v1/chat/completions`; an Anthropic wire-format client is also included). Runs land in `results/` with full JSONL transcripts, a manifest, and an HTML report.

## Layout

```
rangebench/   harness package: runner, env orchestration, agent clients, CLI
attacker/     the generic agent container image (tooling only, no task content)
tasks/        example task; scored suites live elsewhere
scripts/      gates, identity, remote helpers
tests/        138 tests, no Docker required
```

## Writing tasks

A task is a directory with `task.json` (tier, stages, compose file, readiness command, budgets), a Compose environment, a statement, and `solution/solve.sh` — a deterministic oracle that must solve it from a fresh boot. `scripts/check-gates.py` enforces the shape; `rangebench check` proves the oracle.

## License

MIT.
