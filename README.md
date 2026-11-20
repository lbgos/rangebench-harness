# RangeBench

Two benchmarks for LLM agents that run on your own machine. Each tree has its harness and one example task. The scored tasks are private.

- `cyber/`: RangeBench cyber. The agent attacks containerized targets and submits flags; scored with pass@k.
- `homelab/`: RangeBench homelab. The agent works through a long ops session (20 chained steps) in a simulated multi-host world; scored by step pass rate by position, false-done and rule violations.

## Results

[View the benchmark results](https://lbgos.dev/bench/).

[![RangeBench benchmark results](assets/results.png)](https://lbgos.dev/bench/)

The Qwen 3.8 Flash Next and GLM 5.3 Flash runs are local runs submitted by a participant. They ran on two RTX Pro 6000, with Qwen at NVFP4 and GLM at EXL3, stock and uncensored.

## Discussion

[Discuss RangeBench on r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1wvs8va/i_made_my_own_cybersecurity_benchmark_and_ran/).

## Private tasks

To run the scored tasks on your own setup, email contact@lbgos.dev or message [@lbgosna](https://x.com/lbgosna) on X.

## Cyber

Needs Python 3.11+, Docker and Compose. Run from `cyber/`.

```bash
cd cyber
make check
python3 -m rangebench preflight
python3 -m rangebench check jwt-none
python3 -m rangebench run --model <model> --base-url <url> --trials 3 jwt-none
```

`make check` runs ruff, mypy and the unit tests without Docker. A run needs an OpenAI-compatible `/v1/chat/completions` endpoint (`--provider anthropic` for `/v1/messages`) and writes transcripts, `manifest.json` and `report.html` under `cyber/results/`. Private tasks live in `cyber/suite/` (gitignored). [cyber/README.md](cyber/README.md) covers profiles, launching several models, scoring and the task format.

## Homelab

Needs Python 3.12+ and bwrap. Run from `homelab/`.

```bash
cd homelab
make check
python3 -m rangebench_homelab gate example --seeds 0
python3 -m rangebench_homelab run --profile example-api --session example --seed 0
python3 -m rangebench_homelab report <run-id>
```

`make check` runs the unit tests; tests that need bwrap skip without it. Model profiles live in `homelab/configs/` (see `example-local.toml`, `example-api.toml`). Private templates, profiles and rules go in `homelab/suite/` (gitignored) or the directory `$RANGEBENCH_HOMELAB_SUITE` names. For long runs use a capped, delegated scope so each sandboxed command gets its own memory limit:

```bash
systemd-run --scope -p MemoryMax=2G -p MemorySwapMax=0 -p Delegate=yes \
  python3 -m rangebench_homelab run --profile <profile> --suite medium
```

[homelab/DESIGN.md](homelab/DESIGN.md) explains what the bench measures; [homelab/AUTHORING.md](homelab/AUTHORING.md) is the template format and simulator reference.

## License

MIT.
