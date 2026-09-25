"""Docker orchestration: per-attempt compose projects, attacker container, flag reads."""

from __future__ import annotations

import errno
import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks"
ATTACKER_IMAGE = "rb-attacker:latest"
# Linux limits each argv string to 128 KiB including its terminator. Leave
# room for platform differences and report model-generated excess as a command failure.
MAX_COMMAND_BYTES = 120_000
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"
# Docker's inspect API has added and removed non-image Config fields across versions.
# Keep the fields that belong to the image configuration.
IMAGE_CONFIG_FIELDS = (
    "User",
    "ExposedPorts",
    "Env",
    "Entrypoint",
    "Cmd",
    "Volumes",
    "WorkingDir",
    "Labels",
    "StopSignal",
    "ArgsEscaped",
    "Healthcheck",
    "Shell",
    "OnBuild",
)


class EnvError(RuntimeError):
    pass


def wall_clock_default(tier: int) -> int:
    """Whole-attempt wall-clock cap in seconds when task.json sets no wall_clock."""
    if tier <= 2:
        return 600
    if tier == 3:
        return 1200
    return 1800


def image_content_fingerprint(image: dict) -> str:
    """Hash layers and image config across Docker inspect API versions."""
    config = image.get("Config")
    rootfs = image.get("RootFS")
    if not isinstance(config, dict) or not isinstance(rootfs, dict):
        raise EnvError("image inspect is missing Config or RootFS")
    layers = rootfs.get("Layers")
    if not isinstance(layers, list) or any(not isinstance(layer, str) for layer in layers):
        raise EnvError("image inspect has invalid RootFS layers")
    os_name = image.get("Os")
    architecture = image.get("Architecture")
    if (
        not isinstance(os_name, str)
        or not os_name
        or not isinstance(architecture, str)
        or not architecture
    ):
        raise EnvError("image inspect is missing OS or architecture")
    normalized_config = {}
    for name in IMAGE_CONFIG_FIELDS:
        value = config.get(name)
        if name == "Labels" and value is not None:
            if not isinstance(value, dict):
                raise EnvError("image inspect has invalid Config labels")
            value = {key: label for key, label in value.items() if key != COMPOSE_PROJECT_LABEL}
        if value is None or value is False or isinstance(value, (str, list, dict)) and not value:
            continue
        normalized_config[name] = value
    payload = {
        "Config": normalized_config,
        "RootFS.Layers": layers,
        "Os": os_name,
        "Architecture": architecture,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass
class Stage:
    name: str
    flag_file: str
    service: str


@dataclass
class Task:
    id: str
    dir: Path
    category: str
    tier: int
    statement: str
    wall_clock: int = 600  # seconds, whole-attempt cap; unset means tier default
    compose: str = "docker-compose.yml"
    attacker_networks: list[str] = field(default_factory=lambda: ["default"])
    deploy: list[str] = field(default_factory=list)  # copied into attacker /work
    shared_volumes: list[str] = field(
        default_factory=list
    )  # compose volume mounted into attacker, e.g. "incident:/work/incident:ro"
    stages: list[Stage] = field(default_factory=list)
    turns: int | None = None  # Optional diagnostic cap; release tasks have no turn limit.
    cmd_timeout: int = 120
    infra_timeout: int = 180  # minutes, soft infra guard, not scoring
    max_tokens: int = 32768  # per-turn generation limit
    max_output_tokens: int = 800000  # total completion+reasoning budget, soft kill only
    ready_service: str = ""
    ready_cmd: str = ""
    canary: str = ""

    @property
    def is_range(self) -> bool:
        return len(self.stages) > 1


def load_task(task_id: str) -> Task:
    d = TASKS_DIR / task_id
    if not d.is_dir():
        known = ", ".join(sorted(p.name for p in TASKS_DIR.iterdir() if (p / "task.json").exists()))
        raise EnvError(f"unknown task {task_id}; known: {known}")
    raw = json.loads((d / "task.json").read_text())
    stages = [Stage(**s) for s in raw.get("stages", [])]
    if not stages:
        raise EnvError(f"{task_id}: at least one stage required")
    # wall_min is deprecated, map to infra_timeout for backward compat
    infra = raw.get("infra_timeout")
    if infra is None:
        infra = raw.get("wall_min", 180)
    tier = int(raw["tier"])
    wall_clock = raw.get("wall_clock")
    if wall_clock is None:
        wall_clock = wall_clock_default(tier)
    wall_clock = int(wall_clock)
    if wall_clock <= 0:
        raise EnvError(f"{task_id}: wall_clock must be a positive number of seconds")
    return Task(
        id=task_id,
        dir=d,
        category=raw["category"],
        tier=tier,
        statement=raw["statement"],
        compose=raw.get("compose", "docker-compose.yml"),
        attacker_networks=raw.get("attacker_networks", ["default"]),
        deploy=raw.get("deploy", []),
        shared_volumes=raw.get("shared_volumes", []),
        stages=stages,
        turns=int(raw["turns"]) if raw.get("turns") is not None else None,
        cmd_timeout=int(raw.get("cmd_timeout", 120)),
        infra_timeout=int(infra),
        max_tokens=int(raw.get("max_tokens", 32768)),
        max_output_tokens=int(raw.get("max_output_tokens", 800000)),
        ready_service=raw.get("ready_service", ""),
        ready_cmd=raw.get("ready_cmd", ""),
        canary=raw.get("canary", ""),
        wall_clock=wall_clock,
    )


def load_all() -> list[Task]:
    return [load_task(p.name) for p in sorted(TASKS_DIR.iterdir()) if (p / "task.json").exists()]


def _run(cmd: list[str], timeout: int = 300, check: bool = True) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise EnvError(f"{' '.join(cmd[:6])}... timed out after {timeout}s") from exc
    if check and proc.returncode != 0:
        raise EnvError(f"{' '.join(cmd[:6])}... failed: {(proc.stderr or proc.stdout)[-800:]}")
    return proc


class TaskEnv:
    """One attempt of one task: fresh compose project + fresh attacker container."""

    def __init__(self, task: Task, project: str, attacker_image: str = ATTACKER_IMAGE):
        self.task = task
        self.project = project
        self.attacker = f"{project}-atk"
        self.attacker_image = attacker_image
        self.service_image_ids: dict[str, str] = {}
        self.service_image_fingerprints: dict[str, str] = {}

    def up(self, build: bool = True) -> None:
        compose = [
            "docker",
            "compose",
            "-p",
            self.project,
            "-f",
            str(self.task.dir / self.task.compose),
        ]
        services = self._verify_compose_config(compose)
        # Reused project names must not carry old flags or readiness markers.
        _run(["docker", "rm", "-f", self.attacker], check=False, timeout=60)
        _run(compose + ["down", "-v", "--remove-orphans"], timeout=300)
        _run(compose + (["up", "-d", "--build"] if build else ["up", "-d"]), timeout=1800)
        if self.task.ready_service:
            deadline = time.time() + 180
            while time.time() < deadline:
                probe = _run(
                    compose
                    + ["exec", "-T", self.task.ready_service, "sh", "-c", self.task.ready_cmd],
                    timeout=60,
                    check=False,
                )
                if probe.returncode == 0:
                    break
                time.sleep(3)
            else:
                self.down()
                raise EnvError(f"{self.task.id}: readiness probe never passed")
        nets = []
        for net in self.task.attacker_networks:
            full = net if "_" in net else f"{self.project}_{net}"
            nets.append(full)
        self.service_image_ids = self.inspect_service_images(compose, services)
        self.service_image_fingerprints = self.inspect_service_image_fingerprints(
            self.service_image_ids
        )
        self.verify_isolation(nets)
        vol_args: list[str] = []
        for v in self.task.shared_volumes:
            name, _, dest = v.partition(":")
            vol_args += ["-v", f"{self.project}_{name}:{dest}"]
        _run(["docker", "rm", "-f", self.attacker], check=False, timeout=60)
        # per-attempt pip cache isolation, no shared rb-pip-cache
        pip_vol = f"{self.project}-pip-cache:/root/.cache/pip"
        _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                self.attacker,
                "--network",
                nets[0],
                "--network-alias",
                "attacker",
                "--tmpfs",
                "/tmp:size=256m,mode=1777",
                "--ulimit",
                "fsize=2147483648",
                "-v",
                pip_vol,
                *vol_args,
                self.attacker_image,
                "sleep",
                "infinity",
            ],
            timeout=300,
        )
        for extra in nets[1:]:
            _run(["docker", "network", "connect", extra, self.attacker], check=False, timeout=60)
        if self.task.deploy:
            for path in self.task.deploy:
                src = self.task.dir / path
                if src.is_dir():
                    _run(
                        ["docker", "cp", str(src) + "/.", f"{self.attacker}:/work/{src.name}"],
                        timeout=120,
                    )
                else:
                    _run(["docker", "cp", str(src), f"{self.attacker}:/work/"], timeout=120)
        _run(
            ["docker", "exec", self.attacker, "chown", "-R", "agent:agent", "/work"],
            check=False,
            timeout=120,
        )

    def _verify_compose_config(self, compose: list[str]) -> list[str]:
        """Reject network escapes before Compose creates any container or network."""
        rendered = _run(compose + ["config", "--format", "json"], timeout=60).stdout
        try:
            config = json.loads(rendered)
        except json.JSONDecodeError as exc:
            raise EnvError(f"{self.task.id}: invalid Compose config JSON") from exc
        if not isinstance(config, dict):
            raise EnvError(f"{self.task.id}: invalid Compose config")
        networks = config.get("networks")
        services = config.get("services")
        if not isinstance(networks, dict) or not networks:
            raise EnvError(f"{self.task.id}: Compose config has no declared networks")
        if not isinstance(services, dict) or not services:
            raise EnvError(f"{self.task.id}: Compose config has no services")
        for name, network in networks.items():
            if not isinstance(network, dict) or network.get("internal") is not True:
                raise EnvError(f"{self.task.id}: network {name} allows external access")
            if network.get("external"):
                raise EnvError(f"{self.task.id}: network {name} is external")
            if network.get("name", f"{self.project}_{name}") != f"{self.project}_{name}":
                raise EnvError(f"{self.task.id}: network {name} is not project-scoped")
        for name, service in services.items():
            if not isinstance(service, dict):
                raise EnvError(f"{self.task.id}: invalid service {name}")
            if service.get("network_mode"):
                raise EnvError(f"{self.task.id}: service {name} bypasses Compose networks")
            attached = service.get("networks")
            if not isinstance(attached, dict) or not attached:
                raise EnvError(f"{self.task.id}: service {name} has no isolated network")
            if not set(attached).issubset(networks):
                raise EnvError(f"{self.task.id}: service {name} uses an undeclared network")
        if not set(self.task.attacker_networks).issubset(networks):
            raise EnvError(f"{self.task.id}: attacker network is not declared by Compose")
        return sorted(services)

    def inspect_service_images(self, compose: list[str], services: list[str]) -> dict[str, str]:
        """Record the exact images used by this attempt's Compose containers."""
        images: dict[str, str] = {}
        for service in services:
            # Include completed setup services; they still supplied the task image.
            container_ids = _run(
                compose + ["ps", "--all", "--quiet", service], timeout=30
            ).stdout.split()
            if not container_ids:
                raise EnvError(f"{self.task.id}: no container for service {service}")
            image_ids = {
                _run(
                    ["docker", "inspect", "--format", "{{.Image}}", container_id], timeout=30
                ).stdout.strip()
                for container_id in container_ids
            }
            if len(image_ids) != 1 or any(
                re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None for image_id in image_ids
            ):
                raise EnvError(f"{self.task.id}: invalid image ID for service {service}")
            images[service] = image_ids.pop()
        return images

    def inspect_service_image_fingerprints(self, images: dict[str, str]) -> dict[str, str]:
        """Inspect the exact container image IDs and hash their stable contents."""
        fingerprints: dict[str, str] = {}
        by_image_id: dict[str, str] = {}
        for service, image_id in images.items():
            if image_id not in by_image_id:
                output = _run(["docker", "image", "inspect", image_id], timeout=30).stdout
                try:
                    inspected = json.loads(output)
                except json.JSONDecodeError as exc:
                    raise EnvError(
                        f"{self.task.id}: invalid image inspect for service {service}"
                    ) from exc
                if (
                    not isinstance(inspected, list)
                    or len(inspected) != 1
                    or not isinstance(inspected[0], dict)
                ):
                    raise EnvError(f"{self.task.id}: invalid image inspect for service {service}")
                try:
                    by_image_id[image_id] = image_content_fingerprint(inspected[0])
                except EnvError as exc:
                    raise EnvError(
                        f"{self.task.id}: invalid image inspect for service {service}: {exc}"
                    ) from exc
            fingerprints[service] = by_image_id[image_id]
        return fingerprints

    def verify_isolation(self, attacker_networks: list[str]) -> None:
        """Reject Compose networks that would let a model reach outside the lab."""
        project_networks = _run(
            [
                "docker",
                "network",
                "ls",
                "--filter",
                f"label=com.docker.compose.project={self.project}",
                "--format",
                "{{.Name}}",
            ],
            timeout=30,
        ).stdout.splitlines()
        for network in sorted(set(project_networks) | set(attacker_networks)):
            internal = _run(
                ["docker", "network", "inspect", "--format", "{{.Internal}}", network],
                timeout=30,
            ).stdout.strip()
            if internal != "true":
                raise EnvError(f"{self.task.id}: network {network} allows external access")

    def exec(
        self, cmd: str, timeout: int = 120, user: str = "agent", workdir: str = "/work"
    ) -> tuple[int, str]:
        """Run a bash command in the attacker container, return (rc, output).

        GNU timeout stops the command process group. Commands that deliberately
        detach into a new session can outlive it; an outer timeout invalidates
        the attempt if Docker still has not returned.
        """

        try:
            command_bytes = len(cmd.encode("utf-8"))
        except UnicodeEncodeError:
            return 1, "[command contains invalid UTF-8]"
        if command_bytes > MAX_COMMAND_BYTES:
            return 1, f"[command exceeds {MAX_COMMAND_BYTES} UTF-8 bytes]"

        def attacker_available() -> bool:
            try:
                probe = subprocess.run(
                    ["docker", "exec", self.attacker, "true"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=10,
                )
            except subprocess.TimeoutExpired as exc:
                raise EnvError("Docker exec probe timed out") from exc
            except OSError as exc:
                raise EnvError(f"Docker exec probe could not start (errno {exc.errno})") from exc
            return probe.returncode == 0

        try:
            proc = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-u",
                    user,
                    "-w",
                    workdir,
                    self.attacker,
                    "timeout",
                    "--verbose",
                    "--kill-after=5s",
                    str(timeout),
                    "bash",
                    "-lc",
                    'exec 2>&1; exec bash -lc "$1"',
                    "_",
                    cmd,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout + 15,
            )
            stderr = (proc.stderr or "").lstrip()
            # docker exec forwards command stderr, so confirm a matching
            # message is from Docker before invalidating the attempt.
            if (
                proc.returncode != 0
                and stderr.startswith(
                    (
                        "Cannot connect to the Docker daemon",
                        "error during connect:",
                        "Error response from daemon:",
                    )
                )
                and not attacker_available()
            ):
                raise EnvError(f"Docker exec failed before attacker command: {stderr[-300:]}")
            # The child sends its stderr to stdout. Docker stderr contains only
            # timeout diagnostics or Docker errors, so a child exit 124 is not
            # mistaken for an expired deadline.
            if proc.returncode in (124, 137) and "timeout: sending signal" in stderr:
                return 124, f"[command timed out after {timeout}s]"
            out = (proc.stdout or "") + (
                ("\n[stderr]\n" + proc.stderr) if proc.stderr.strip() else ""
            )
            return proc.returncode, out
        except subprocess.TimeoutExpired as exc:
            if not attacker_available():
                raise EnvError("Docker exec unavailable after attacker command timeout") from exc
            raise EnvError("Docker exec did not finish after attacker command timeout") from exc
        except OSError as exc:
            if exc.errno == errno.E2BIG:
                return 1, "[command could not start: argument list too long]"
            raise EnvError(f"Docker exec could not start (errno {exc.errno})") from exc

    def read_flag(self, stage: Stage) -> str:
        compose = [
            "docker",
            "compose",
            "-p",
            self.project,
            "-f",
            str(self.task.dir / self.task.compose),
        ]
        proc = _run(
            compose + ["exec", "-T", "-u", "0", stage.service, "cat", stage.flag_file],
            timeout=60,
            check=False,
        )
        flag = (proc.stdout or "").strip()
        if proc.returncode != 0 or not flag:
            raise EnvError(
                f"cannot read flag {stage.service}:{stage.flag_file}: {(proc.stderr or '')[-300:]}"
            )
        return flag

    def logs(self, service: str, tail: int = 50) -> str:
        compose = [
            "docker",
            "compose",
            "-p",
            self.project,
            "-f",
            str(self.task.dir / self.task.compose),
        ]
        proc = _run(compose + ["logs", "--tail", str(tail), service], timeout=60, check=False)
        return proc.stdout or ""

    def down(self) -> tuple[bool, str]:
        """Best-effort teardown. Returns (ok, warning). Warning is non-empty if something failed but never kills caller."""
        if shutil.which("docker") is None:
            return True, ""
        warnings: list[str] = []
        for attempt in range(2):
            try:
                _run(["docker", "rm", "-f", self.attacker], check=False, timeout=45)
                break
            except Exception as exc:
                warnings.append(f"attacker rm attempt {attempt}: {exc}")
                continue
        try:
            _run(
                [
                    "docker",
                    "compose",
                    "-p",
                    self.project,
                    "-f",
                    str(self.task.dir / self.task.compose),
                    "down",
                    "-v",
                    "--remove-orphans",
                    "--timeout",
                    "10",
                ],
                check=False,
                timeout=300,
            )
        except Exception as exc:
            warnings.append(f"compose down: {exc}")
        # pip cache volume is per-attempt, remove it as well
        try:
            _run(["docker", "volume", "rm", f"{self.project}-pip-cache"], check=False, timeout=30)
        except Exception:
            pass
        ok = not warnings
        return ok, "; ".join(warnings)


def truncate_output(text: str, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
    head, tail = text[: limit // 2], text[-limit // 3 :]
    return f"{head}\n...[truncated {len(text) - len(head) - len(tail)} chars]...\n{tail}"
