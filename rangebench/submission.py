"""Portable, allowlisted benchmark results. The file is self-reported, not attested."""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Any, cast
from urllib.parse import urlsplit

VERSION = "rangebench.submission.v1"
USAGE = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "api_calls",
    "api_requests",
    "usage_reported_calls",
    "input_reported_calls",
    "output_reported_calls",
    "cache_read_reported_calls",
    "cache_write_reported_calls",
)
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+@ -]{0,159}\Z")
SAFE_HASH = re.compile(r"[a-fA-F0-9]{16,64}\Z")
FAILS = {
    "refusal",
    "solved",
    "provider_error",
    "env_error",
    "protocol_error",
    "budget_exhausted",
    "normal",
    "skipped",
}


def _name(value: object, field: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value) or "//" in value:
        raise ValueError(f"{field}: unsafe or missing public name")
    return value


def _nonneg(value: object, field: str, *, integer: bool = True) -> int | float:
    if integer and type(value) is int and value >= 0:
        return value
    if (
        not integer
        and isinstance(value, int | float)
        and not isinstance(value, bool)
        and value >= 0
        and math.isfinite(value)
    ):
        return value
    else:
        raise ValueError(f"{field}: expected a finite nonnegative number")


def _usage(source: dict[str, Any]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    # Usage.as_dict() names these differently from the final task record.
    aliases = {
        "api_calls": "calls",
        "api_requests": "requests",
        "usage_reported_calls": "reported_calls",
    }
    for key in USAGE:
        value = source.get(key, source.get(aliases.get(key, key)))
        result[key] = None if value is None else int(_nonneg(value, key))
    return result


def _attempt(raw: dict[str, Any], kind: str) -> dict[str, Any]:
    usage = _usage(raw.get("usage", raw))
    if kind == "probe" and "usage" not in raw:
        usage = dict.fromkeys(USAGE)
    fail_class = raw.get("fail_class", "probe" if kind == "probe" else "unknown")
    if fail_class not in FAILS | {"probe"}:
        fail_class = "unknown"
    task = _name(raw["task"], "task")
    images = raw.get("service_image_ids") or {}
    if not isinstance(images, dict):
        raise ValueError("invalid image provenance")
    image_hashes = sorted(
        value
        for value in images.values()
        if isinstance(value, str) and re.fullmatch(r"sha256:[a-fA-F0-9]{64}", value)
    )
    scored = raw.get("scored", False) if type(raw.get("scored", False)) is bool else False
    solved = raw.get("solved", False) if type(raw.get("solved", False)) is bool else False
    if (
        kind == "task"
        and fail_class in {"provider_error", "env_error", "skipped", "refusal"}
        and scored
    ):
        raise ValueError("unscored failure marked scored")
    if kind == "task" and solved != (fail_class == "solved"):
        raise ValueError("solved outcome conflicts with failure class")
    return {
        "kind": kind,
        "task": task,
        "trial": int(_nonneg(raw.get("trial", 1), "trial")),
        "retry": int(_nonneg(raw.get("retry", 0), "retry")),
        "scored": scored,
        "solved": solved,
        "fail_class": fail_class,
        "wall_s": _nonneg(raw.get("wall_s", 0), "wall_s", integer=False),
        "turns_used": int(_nonneg(raw.get("turns_used", 0), "turns_used")),
        "effective_ctx_window": raw.get("effective_ctx_window"),
        "wall_clock_seconds": raw.get("wall_clock_seconds"),
        "wall_clock_scale": raw.get("wall_clock_scale"),
        "wall_clock_exceeded": raw.get("wall_clock_exceeded"),
        "service_image_digests": image_hashes,
        "usage": usage,
    }


def _summarize(
    attempts: list[dict[str, Any]],
    selected: list[str],
    catalog: dict[str, dict[str, Any]],
    trials: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in attempts:
        if a["kind"] in {"task", "skipped"}:
            by_task[a["task"]].append(a)
    tasks = []
    for tid in selected:
        records = by_task[tid]
        scored = [a for a in records if a["scored"]]
        n, c = len(scored), sum(a["solved"] for a in scored)
        pass3 = None if n < 3 else 1 - math.comb(n - c, 3) / math.comb(n, 3)
        item = catalog.get(tid, {})
        tasks.append(
            {
                "task": tid,
                "identity": item.get("identity"),
                "category": item.get("category"),
                "tier": item.get("tier"),
                "recorded": len(records),
                "scored": n,
                "solved": c,
                "invalid": sum(not a["scored"] and a["fail_class"] != "skipped" for a in records),
                "skipped": sum(a["fail_class"] == "skipped" for a in records),
                "pass_at_1": c / n if n else None,
                "pass_at_3": pass3,
            }
        )
    total_usage: dict[str, int | None] = {}
    billable = [a for a in attempts if a["kind"] != "skipped"]
    for key in USAGE:
        values = [a["usage"][key] for a in billable]
        total_usage[key] = sum(values) if all(v is not None for v in values) else None
    p1 = [t["pass_at_1"] for t in tasks if t["pass_at_1"] is not None]
    p3 = [t["pass_at_3"] for t in tasks if t["pass_at_3"] is not None]
    summary = {
        "selected_tasks": len(selected),
        "planned_slots": len(selected) * trials,
        "completed_slots": sum(t["recorded"] for t in tasks),
        "missing_slots": len(selected) * trials - sum(t["recorded"] for t in tasks),
        "recorded": sum(t["recorded"] for t in tasks),
        "scored": sum(t["scored"] for t in tasks),
        "solved": sum(t["solved"] for t in tasks),
        "invalid": sum(t["invalid"] for t in tasks),
        "skipped": sum(t["skipped"] for t in tasks),
        "pass_at_1": sum(p1) / len(p1) if p1 else None,
        "pass_at_1_tasks": len(p1),
        "pass_at_3": sum(p3) / len(p3) if p3 else None,
        "pass_at_3_tasks": len(p3),
        "usage": total_usage,
        "wall_s_all_attempts": sum(a["wall_s"] for a in billable),
        "turns_all_attempts": sum(a["turns_used"] for a in billable),
        "attempts_all": len(billable),
    }
    return tasks, summary


def _public_route(base_url: object) -> str | None:
    if not isinstance(base_url, str):
        return None
    host = urlsplit(base_url).hostname
    return "OpenRouter" if host == "openrouter.ai" else None


def _unknown_pricing() -> dict[str, Any]:
    return {
        "status": "unknown",
        "source": None,
        "as_of": None,
        "catalog_model_id": None,
        "usd_per_million": {"input": None, "output": None, "cache_read": None, "cache_write": None},
        "estimated_usd": None,
    }


def _cost(usage: dict[str, int | None], rates: dict[str, float | None]) -> float | None:
    calls = usage["api_calls"]
    input_coverage = usage["input_reported_calls"]
    output_coverage = usage["output_reported_calls"]
    if (
        calls is None
        or input_coverage is None
        or output_coverage is None
        or input_coverage < calls
        or output_coverage < calls
    ):
        return None
    inp, out = usage["input_tokens"], usage["output_tokens"]
    read, write = usage["cache_read_tokens"], usage["cache_write_tokens"]
    if inp is None or out is None or read is None or write is None:
        return None
    if any(
        (coverage := usage[key]) is None or coverage < calls
        for key in ("cache_read_reported_calls", "cache_write_reported_calls")
    ):
        return None
    if read + write > inp:
        raise ValueError("cache tokens exceed input tokens")
    portions = {
        "input": inp - read - write,
        "output": out,
        "cache_read": read,
        "cache_write": write,
    }
    if any(rates[key] is None and amount for key, amount in portions.items()):
        return None
    return sum(amount * (rates[key] or 0) / 1e6 for key, amount in portions.items())


def build_submission(
    run: dict[str, Any],
    *,
    pricing: dict[str, Any] | None = None,
    display_name: str | None = None,
    route_name: str | None = None,
    upstream_provider: str | None = None,
    sensitive_values: tuple[str, ...] = (),
) -> dict[str, Any]:
    if not run.get("finished") or not run.get("started"):
        raise ValueError("run is incomplete")
    if not isinstance(run.get("selected_tasks"), list) or not run["selected_tasks"]:
        raise ValueError("selected_tasks missing")
    selected = [_name(t, "selected_tasks") for t in run["selected_tasks"]]
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate selected task")
    raw_tasks = run.get("tasks", [])
    if not isinstance(raw_tasks, list):
        raise ValueError("tasks must be a list")
    attempts = [
        _attempt(t, "skipped" if t.get("fail_class") == "skipped" else "task") for t in raw_tasks
    ]
    final_slots = {(t["task"], t["trial"]) for t in attempts}
    infra_by_slot: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for raw in run.get("infra_attempts", []):
        infra_by_slot[(raw["task"], raw["trial"])].append(raw)
    for raw in run.get("infra_attempts", []):
        slot = (raw["task"], raw["trial"])
        final_record = next(
            (t for t in raw_tasks if (t["task"], t["trial"]) == slot),
            cast(dict[str, Any], {}),
        )
        # Old run files lack terminal. Only the last infra record in a slot can
        # also be the final task record; retain earlier retries, with unknown usage.
        terminal = raw.get("terminal") is True or (
            "terminal" not in raw
            and slot in final_slots
            and raw is infra_by_slot[slot][-1]
            and final_record.get("fail_class") == raw.get("fail_class")
        )
        if terminal:
            continue
        attempts.append(_attempt(raw, "infra_retry"))
    if isinstance(run.get("probe_attempt"), dict):
        probe_raw = {**run["probe_attempt"]}
        if "probe_usage" in run:
            probe_raw["usage"] = run["probe_usage"]
        if "probe_turns_used" in run:
            probe_raw["turns_used"] = run["probe_turns_used"]
        probe = _attempt(probe_raw, "probe")
        probe["scored"] = False
        probe["solved"] = False
        attempts.append(probe)
    if any(a["task"] not in selected for a in attempts if a["kind"] != "probe"):
        raise ValueError("attempt outside selected tasks")
    catalog = {
        t["task"]: {
            "category": _name(t["category"], "category"),
            "tier": int(_nonneg(t["tier"], "tier")),
        }
        for t in run.get("task_catalog", [])
    }
    for t in raw_tasks:
        catalog.setdefault(
            t["task"],
            {"category": _name(t["category"], "category"), "tier": int(_nonneg(t["tier"], "tier"))},
        )
    for tid, ident in run.get("task_identities", {}).items():
        if ident is not None and (not isinstance(ident, str) or not SAFE_HASH.fullmatch(ident)):
            raise ValueError("unsafe task identity")
        catalog.setdefault(tid, {})["identity"] = ident
    tasks, summary = _summarize(attempts, selected, catalog, run["trials"])
    pricing = pricing or _unknown_pricing()
    if pricing["status"] != "unknown":
        pricing["estimated_usd"] = _cost(summary["usage"], pricing["usd_per_million"])
    doc = {
        "schema_version": VERSION,
        "self_reported": True,
        "model": {
            "id": _name(run["model"], "model"),
            "display_name": _name(
                display_name or run.get("display_name") or run["model"], "display_name"
            ),
            "api_protocol": run.get("provider", "openai"),
            "route_name": _name(
                route_name or run.get("route_name") or _public_route(run.get("base_url")),
                "route_name",
            )
            if route_name or run.get("route_name") or _public_route(run.get("base_url"))
            else None,
            "upstream_provider": _name(
                upstream_provider or run.get("upstream_provider"), "upstream_provider"
            )
            if upstream_provider or run.get("upstream_provider")
            else None,
        },
        "benchmark": {
            "run_id": _name(run["id"], "run_id"),
            "started": run["started"],
            "finished": run["finished"],
            "harness_commit": run.get("harness_commit"),
            "harness_source_hash": run.get("harness_source_hash"),
            "task_set_hash": run.get("task_set_hash"),
            "attacker_digest": run.get("attacker_digest"),
            "selected_tasks": selected,
            "trials": run["trials"],
        },
        "policy": {
            "reasoning_effort": run.get("reasoning_effort"),
            "ctx_window": run["ctx_window"],
            "reserve": run["reserve"],
            "keep_tail": run["keep_tail"],
            "threshold": run["threshold"],
            "compact": run["compact"],
            "max_attempts": run.get("max_attempts"),
            "infra_retries": run.get("infra_retries"),
            "repeat_caps": run.get("repeat_caps"),
            "sampling": "fixed_trials"
            if run.get("repeat_caps") is True
            else "adaptive_repeat_skip"
            if run.get("repeat_caps") is False
            else "unknown",
            "wall_clock_scale_mode": run.get("wall_clock_scale_mode"),
            "wall_clock_scale": run.get("wall_clock_scale"),
            "wall_clock_reference_hash": run.get("wall_clock_reference_hash"),
            "model_tps": run.get("model_tps"),
        },
        "tasks": tasks,
        "attempts": attempts,
        "summary": summary,
        "pricing": pricing,
    }
    validate_submission(doc, sensitive_values=sensitive_values)
    return doc


def validate_submission(doc: dict[str, Any], *, sensitive_values: tuple[str, ...] = ()) -> None:
    if (
        not isinstance(doc, dict)
        or set(doc)
        != {
            "schema_version",
            "self_reported",
            "model",
            "benchmark",
            "policy",
            "tasks",
            "attempts",
            "summary",
            "pricing",
        }
        or doc["schema_version"] != VERSION
        or doc["self_reported"] is not True
    ):
        raise ValueError("invalid submission schema")

    def scan(value: Any) -> None:
        if isinstance(value, dict):
            for v in value.values():
                scan(v)
        elif isinstance(value, list):
            for v in value:
                scan(v)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite value")
        elif isinstance(value, str):
            if (
                "\n" in value
                or "\r" in value
                or "?" in value
                or "#" in value
                or "\\" in value
                or "//" in value
            ):
                raise ValueError("unsafe text")
            if any(secret and secret in value for secret in sensitive_values):
                raise ValueError("known credential in submission")
            for key, secret in os.environ.items():
                if (
                    any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
                    and len(secret) >= 8
                    and secret in value
                ):
                    raise ValueError("known credential in submission")

    scan(doc)
    m = doc["model"]
    if set(m) != {"id", "display_name", "api_protocol", "route_name", "upstream_provider"} or m[
        "api_protocol"
    ] not in {"openai", "anthropic"}:
        raise ValueError("invalid model metadata")
    for key in ("id", "display_name", "route_name", "upstream_provider"):
        if key in {"id", "display_name"} or m[key] is not None:
            _name(m[key], key)
    b = doc["benchmark"]
    if set(b) != {
        "run_id",
        "started",
        "finished",
        "harness_commit",
        "harness_source_hash",
        "task_set_hash",
        "attacker_digest",
        "selected_tasks",
        "trials",
    }:
        raise ValueError("invalid benchmark metadata")
    selected = b["selected_tasks"]
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
        raise ValueError("invalid selected tasks")
    for tid in selected:
        _name(tid, "task")
    for key in ("started", "finished"):
        if not isinstance(b[key], str) or not re.fullmatch(r"\d{4}-\d\d-\d\dT[0-9:.+Z-]+", b[key]):
            raise ValueError("invalid timestamp")
    _nonneg(b["trials"], "trials")
    if b["trials"] < 1:
        raise ValueError("trials must be positive")
    _name(b["run_id"], "run_id")
    for key in ("harness_commit", "harness_source_hash", "task_set_hash"):
        value = b[key]
        if value is not None and (not isinstance(value, str) or not SAFE_HASH.fullmatch(value)):
            raise ValueError(f"invalid {key}")
    if b["attacker_digest"] is not None and (
        not isinstance(b["attacker_digest"], str)
        or not re.fullmatch(r"sha256:[a-fA-F0-9]{64}", b["attacker_digest"])
    ):
        raise ValueError("invalid attacker digest")
    policy = doc["policy"]
    if set(policy) != {
        "reasoning_effort",
        "ctx_window",
        "reserve",
        "keep_tail",
        "threshold",
        "compact",
        "max_attempts",
        "infra_retries",
        "repeat_caps",
        "sampling",
        "wall_clock_scale_mode",
        "wall_clock_scale",
        "wall_clock_reference_hash",
        "model_tps",
    }:
        raise ValueError("invalid policy")
    if policy["reasoning_effort"] is not None:
        _name(policy["reasoning_effort"], "reasoning_effort")
    for key in ("ctx_window", "reserve", "keep_tail"):
        _nonneg(policy[key], key)
    for key in ("max_attempts", "infra_retries"):
        if policy[key] is not None:
            _nonneg(policy[key], key)
    if policy["repeat_caps"] is not None and type(policy["repeat_caps"]) is not bool:
        raise ValueError("invalid repeat_caps")
    expected_sampling = (
        "fixed_trials"
        if policy["repeat_caps"] is True
        else "adaptive_repeat_skip"
        if policy["repeat_caps"] is False
        else "unknown"
    )
    if policy["sampling"] != expected_sampling:
        raise ValueError("invalid sampling policy")
    threshold = _nonneg(policy["threshold"], "threshold", integer=False)
    if not 0 < threshold < 1 or policy["compact"] not in {"llm", "deterministic"}:
        raise ValueError("invalid compaction policy")
    if policy["wall_clock_scale_mode"] not in {"manual", "reference"}:
        raise ValueError("invalid wall clock mode")
    if policy["wall_clock_scale"] is not None:
        _nonneg(policy["wall_clock_scale"], "wall_clock_scale", integer=False)
    if (
        policy["model_tps"] is not None
        and _nonneg(policy["model_tps"], "model_tps", integer=False) <= 0
    ):
        raise ValueError("invalid model_tps")
    ref_hash = policy["wall_clock_reference_hash"]
    if ref_hash is not None and (
        not isinstance(ref_hash, str) or not SAFE_HASH.fullmatch(ref_hash)
    ):
        raise ValueError("invalid reference hash")
    attempts = doc["attempts"]
    if not isinstance(attempts, list):
        raise ValueError("invalid attempts")
    slots: set[tuple[str, int]] = set()
    for a in attempts:
        if set(a) != {
            "kind",
            "task",
            "trial",
            "retry",
            "scored",
            "solved",
            "fail_class",
            "wall_s",
            "turns_used",
            "usage",
            "effective_ctx_window",
            "wall_clock_seconds",
            "wall_clock_scale",
            "wall_clock_exceeded",
            "service_image_digests",
        } or a["kind"] not in {"task", "skipped", "infra_retry", "probe"}:
            raise ValueError("invalid attempt")
        _name(a["task"], "task")
        if a["kind"] != "probe" and a["task"] not in selected:
            raise ValueError("attempt outside selected tasks")
        if type(a["scored"]) is not bool or type(a["solved"]) is not bool:
            raise ValueError("invalid scored or solved")
        if a["kind"] != "task" and (a["scored"] or a["solved"]):
            raise ValueError("non-task attempt scored")
        if a["kind"] in {"task", "skipped"}:
            slot = (a["task"], a["trial"])
            if slot in slots:
                raise ValueError("duplicate trial")
            slots.add(slot)
        if a["trial"] < 1 or a["trial"] > b["trials"] and a["kind"] != "probe":
            raise ValueError("trial out of range")
        if a["fail_class"] not in FAILS | {"probe", "unknown"}:
            raise ValueError("invalid failure class")
        if a["kind"] == "skipped" and a["fail_class"] != "skipped":
            raise ValueError("invalid skipped attempt")
        if a["kind"] == "task" and (
            a["solved"] != (a["fail_class"] == "solved")
            or (
                a["fail_class"] in {"provider_error", "env_error", "skipped", "refusal"}
                and a["scored"]
            )
        ):
            raise ValueError("inconsistent task outcome")
        for key in ("trial", "retry", "turns_used"):
            _nonneg(a[key], key)
        for key in ("effective_ctx_window", "wall_clock_seconds"):
            if a[key] is not None:
                _nonneg(a[key], key)
        if a["wall_clock_scale"] is not None:
            _nonneg(a["wall_clock_scale"], "wall_clock_scale", integer=False)
        if a["wall_clock_exceeded"] is not None and type(a["wall_clock_exceeded"]) is not bool:
            raise ValueError("invalid wall_clock_exceeded")
        if not isinstance(a["service_image_digests"], list) or any(
            not isinstance(d, str) or not re.fullmatch(r"sha256:[a-fA-F0-9]{64}", d)
            for d in a["service_image_digests"]
        ):
            raise ValueError("invalid image digests")
        _nonneg(a["wall_s"], "wall_s", integer=False)
        if set(a["usage"]) != set(USAGE):
            raise ValueError("invalid usage")
        for key, value in a["usage"].items():
            if value is not None:
                _nonneg(value, key)
    for t in doc["tasks"]:
        if set(t) != {
            "task",
            "identity",
            "category",
            "tier",
            "recorded",
            "scored",
            "solved",
            "invalid",
            "skipped",
            "pass_at_1",
            "pass_at_3",
        }:
            raise ValueError("invalid task summary")
        _name(t["task"], "task")
        if t["category"] is not None:
            _name(t["category"], "category")
        if t["tier"] is not None:
            _nonneg(t["tier"], "tier")
        if t["identity"] is not None and (
            not isinstance(t["identity"], str) or not SAFE_HASH.fullmatch(t["identity"])
        ):
            raise ValueError("invalid task identity")
    catalog = {
        t["task"]: {"identity": t["identity"], "category": t["category"], "tier": t["tier"]}
        for t in doc["tasks"]
    }
    if set(catalog) != set(selected) or len(doc["tasks"]) != len(selected):
        raise ValueError("task coverage mismatch")
    tasks, summary = _summarize(attempts, selected, catalog, b["trials"])
    if doc["tasks"] != tasks or doc["summary"] != summary:
        raise ValueError("summary does not match attempts")
    p = doc["pricing"]
    if set(p) != {
        "status",
        "source",
        "as_of",
        "catalog_model_id",
        "usd_per_million",
        "estimated_usd",
    } or p["status"] not in {"unknown", "estimated"}:
        raise ValueError("invalid pricing")
    if set(p["usd_per_million"]) != {"input", "output", "cache_read", "cache_write"}:
        raise ValueError("invalid pricing fields")
    for value in p["usd_per_million"].values():
        if value is not None:
            _nonneg(value, "price", integer=False)
    if p["estimated_usd"] is not None:
        _nonneg(p["estimated_usd"], "estimated_usd", integer=False)
    if p["status"] == "unknown" and p != _unknown_pricing():
        raise ValueError("unknown pricing must not contain unchecked metadata or rates")
    if p["status"] != "unknown":
        if not isinstance(p["source"], str) or not SAFE_ID.fullmatch(p["source"]):
            raise ValueError("invalid price source")
        try:
            date.fromisoformat(p["as_of"])
        except (TypeError, ValueError):
            raise ValueError("invalid price date") from None
        if p["catalog_model_id"] is not None:
            _name(p["catalog_model_id"], "catalog_model_id")
        if p["estimated_usd"] != _cost(summary["usage"], p["usd_per_million"]):
            raise ValueError("estimated cost mismatch")


def public_price_lookup(
    model_id: str, route_name: str | None, upstream_provider: str | None = None
) -> dict[str, Any]:
    """Unauthenticated exact-ID public catalog lookup; failures leave price unknown."""
    pricing = _unknown_pricing()
    urls = [
        ("OpenRouter", "https://openrouter.ai/api/v1/models"),
        ("models.dev", "https://models.dev/api.json"),
    ]
    for source, url in urls:
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "rangebench/0.1"}), timeout=5
            ) as response:
                catalog = json.load(response)
            if source == "OpenRouter":
                matches = [m for m in catalog["data"] if m.get("id") == model_id]
                if len(matches) != 1 or matches[0].get("pricing", {}).get("overrides"):
                    continue
                rates = matches[0]["pricing"]
                mapped = {
                    "input": "prompt",
                    "output": "completion",
                    "cache_read": "input_cache_read",
                    "cache_write": "input_cache_write",
                }
                prices = {
                    k: float(Decimal(str(rates[v])) * Decimal(1_000_000))
                    if rates.get(v) is not None
                    else None
                    for k, v in mapped.items()
                }
            else:
                matches = [
                    m
                    for provider_id, provider in catalog.items()
                    if isinstance(provider, dict)
                    if upstream_provider is None or provider_id == upstream_provider
                    for mid, m in provider.get("models", {}).items()
                    if mid == model_id
                ]
                if len(matches) != 1:
                    continue
                rates = matches[0].get("cost", {})
                prices = {
                    k: float(rates[v]) if v in rates else None
                    for k, v in {
                        "input": "input",
                        "output": "output",
                        "cache_read": "cache_read",
                        "cache_write": "cache_write",
                    }.items()
                }
            if (
                prices["input"] is None
                or prices["output"] is None
                or any(v is not None and (v < 0 or not math.isfinite(v)) for v in prices.values())
            ):
                continue
            pricing.update(
                status="estimated",
                source=source,
                as_of=date.today().isoformat(),
                catalog_model_id=model_id,
                usd_per_million=prices,
            )
            return pricing
        except (OSError, ValueError, KeyError, TypeError, AttributeError, ArithmeticError):
            continue
    return pricing
