"""GitHub mock behind the `gh` stub and `git@<host>:owner/repo.git` remotes.

World API (world.py):

    gh = sim.github("dev-bot", "private@example.com")  # account login and email; web merges sign with it
    gh.repo("acme/api", source=("ws", "/root/api"))     # bare repo at github.com:/root/acme/api.git
    pr = gh.pr("acme/api", head="fix-login", title="Fix login",
               checks=[Check("test", "fail", log="FAILED test_login")])
    gh.review(pr, "alice", "APPROVED")

Repos are real bare repos on the GitHub host, so `git push/fetch/clone git@github.com:acme/api.git`
works through the ssh stub. PRs live here and are dumped to <world>/gh.json after every change.
Merges are logged as `merge` events (action = squash|merge|rebase, text "<repo>#<n> <method>
author=<email>"); creates and comments as `gh` events.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .sim import Gap, Result, getopt, host_git

if TYPE_CHECKING:
    from .sim import Simulator

CHECK_STATES = ("pass", "fail", "pending")
REVIEW_STATES = ("APPROVED", "CHANGES_REQUESTED", "COMMENTED")
WEB_COMMITTER = ("GitHub", "noreply@github.com")
FIRST_RUN_ID = 4_812_000_001
JsonValue = str | int | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JQ = shutil.which("jq")
PRLIMIT = shutil.which("prlimit")
# `gh <cmd> --help`: the flags this mock takes (others are sim gaps).
GH_HELP = {
    "": "Work seamlessly with GitHub from the command line.\n\nUSAGE\n  gh <command> <subcommand> [flags]\n\n"
    "CORE COMMANDS\n  auth:        Authenticate gh and git with GitHub\n  pr:          Manage pull requests\n"
    "  repo:        Manage repositories\n  run:         View details about workflow runs\n\n"
    "ADDITIONAL COMMANDS\n  api:         Make an authenticated GitHub API request\n",
    "pr": "Work with GitHub pull requests.\n\nUSAGE\n  gh pr <command> [flags]\n\nGENERAL COMMANDS\n"
    "  create:      Create a pull request\n  list:        List pull requests in a repository\n\n"
    "TARGETED COMMANDS\n  checks:      Show CI status for a single pull request\n  comment:     Add a comment to a pull request\n"
    "  diff:        View changes in a pull request\n  edit:        Edit a pull request\n  merge:       Merge a pull request\n"
    "  view:        View a pull request\n",
    "run": "List, view, and watch recent workflow runs from GitHub Actions.\n\nUSAGE\n  gh run <command> [flags]\n\n"
    "AVAILABLE COMMANDS\n  list:        List recent workflow runs\n  rerun:       Rerun a run\n  view:        View a summary of a workflow run\n",
    "pr merge": "Merge a pull request on GitHub.\n\nUSAGE\n  gh pr merge [<number> | <url> | <branch>] [flags]\n\nFLAGS\n"
    "      --admin                   Use administrator privileges to merge a pull request that does not meet requirements\n"
    "  -A, --author-email text       Email text for merge commit author\n"
    "      --auto                    Automatically merge only after necessary requirements are met\n"
    "  -b, --body text               Body text for the merge commit\n"
    "  -d, --delete-branch           Delete the local and remote branch after merge\n"
    "      --disable-auto            Disable auto-merge for this pull request\n"
    "      --match-head-commit SHA   Commit SHA that the pull request head must match to allow merge\n"
    "  -m, --merge                   Merge the commits with the base branch\n"
    "  -r, --rebase                  Rebase the commits onto the base branch\n"
    "  -s, --squash                  Squash the commits into one commit and merge it into the base branch\n"
    "  -t, --subject text            Subject text for the merge commit\n",
    "pr edit": "Edit a pull request.\n\nUSAGE\n  gh pr edit [<number> | <url> | <branch>] [flags]\n\nFLAGS\n"
    "  -B, --base branch        Change the base branch for this pull request\n"
    '  -b, --body string        Set the new body.\n  -F, --body-file file     Read body text from file (use "-" to read from standard input)\n'
    "  -t, --title string       Set the new title.\n",
    "pr create": "Create a pull request on GitHub.\n\nUSAGE\n  gh pr create [flags]\n\nFLAGS\n"
    "  -B, --base branch          The branch into which you want your code merged\n"
    "  -b, --body string          Body for the pull request\n  -F, --body-file file       Read body text from file\n"
    "  -d, --draft                Mark pull request as a draft\n  -f, --fill                 Use commit info for title and body\n"
    "  -H, --head branch          The branch that contains commits for your pull request\n"
    "  -t, --title string         Title for the pull request\n",
    "pr view": "Display the title, body, and other information about a pull request.\n\nUSAGE\n  gh pr view [<number> | <url> | <branch>] [flags]\n\n"
    "FLAGS\n  -c, --comments          View pull request comments\n  -q, --jq expression     Filter JSON output using a jq expression\n"
    "      --json fields       Output JSON with the specified fields\n",
    "pr checks": "Show CI status for a single pull request.\n\nUSAGE\n  gh pr checks [<number> | <url> | <branch>] [flags]\n\n"
    "FLAGS\n      --json fields       Output JSON with the specified fields\n      --required          Only show checks that are required\n",
    "pr comment": "Add a comment to a GitHub pull request.\n\nUSAGE\n  gh pr comment [<number> | <url> | <branch>] [flags]\n\n"
    "FLAGS\n  -b, --body text        The comment body text\n  -F, --body-file file   Read body text from file\n",
    "pr list": "List pull requests in a GitHub repository.\n\nUSAGE\n  gh pr list [flags]\n\nFLAGS\n"
    "  -A, --author string     Filter by author\n  -B, --base string       Filter by base branch\n  -d, --draft             Filter by draft state\n"
    "  -H, --head string       Filter by head branch\n  -L, --limit int         Maximum number of items to fetch (default 30)\n"
    '  -s, --state string      Filter by state: {open|closed|merged|all} (default "open")\n',
    "run view": "View a summary of a workflow run.\n\nUSAGE\n  gh run view [<run-id>] [flags]\n\nFLAGS\n"
    "      --exit-status       Exit with non-zero status if run failed\n  -j, --job string        View a specific job ID from a run\n"
    "      --json fields       Output JSON with the specified fields\n      --log               View full log for either a run or specific job\n"
    "      --log-failed        View the log for any failed steps in a run or specific job\n",
    "run rerun": "Rerun an entire run, only failed jobs, or a specific job from a run.\n\nUSAGE\n  gh run rerun [<run-id>] [flags]\n\n"
    "FLAGS\n      --failed        Rerun only failed jobs, including dependencies\n",
    "api": "Makes an authenticated HTTP request to the GitHub API and prints the response.\n\nUSAGE\n  gh api <endpoint> [flags]\n\n"
    "FLAGS\n  -F, --field key=value       Add a typed parameter in key=value format\n"
    '  -X, --method string         The HTTP method for the request (default "GET")\n'
    "  -q, --jq string             Query to select values from the response using jq syntax\n"
    "  -f, --raw-field key=value   Add a string parameter in key=value format\n",
}


class GhError(Exception):
    """A gh error message for stderr (exit 1)."""


@dataclass
class Check:
    """A required (or optional) CI check on a PR, shown by `gh pr checks` and `gh run`.

    rerun: called after a push updates the PR head and on `gh run rerun`; returns the new
    (status, log). Read the pushed tree with gh.show(pr.repo, pr.head_sha, path)."""

    name: str
    status: str = "pass"  # pass | fail | pending
    log: str = ""
    required: bool = True
    workflow: str = "CI"
    rerun: Callable[[Gh, PullRequest], tuple[str, str]] | None = None
    run_id: int = 0
    attempt: int = 1


@dataclass
class Review:
    author: str
    state: str  # APPROVED | CHANGES_REQUESTED | COMMENTED
    body: str = ""


@dataclass
class Comment:
    author: str
    body: str
    id: int = 0


@dataclass
class PullRequest:
    repo: str
    number: int
    title: str
    head: str
    base: str
    body: str = ""
    author: str = ""
    draft: bool = False
    state: str = "OPEN"  # OPEN | CLOSED | MERGED
    checks: list[Check] = field(default_factory=list)
    reviews: list[Review] = field(default_factory=list)
    comments: list[Comment] = field(default_factory=list)
    head_sha: str = ""
    created: str = ""
    merge_commit: str = ""
    merge_method: str = ""
    merged_by: str = ""
    merged_at: str = ""
    auto_merge: dict[str, str] = field(
        default_factory=dict
    )  # gh pr merge --auto: the merge's options


@dataclass
class Repo:
    name: str  # owner/repo
    path: str  # bare repo on the GitHub host
    default_branch: str = "main"
    checks: list[Check] = field(
        default_factory=list
    )  # templates for PRs created with `gh pr create`


class Gh:
    """GitHub state for one world. Create through `sim.github(...)`."""

    def __init__(self, sim: Simulator, host: str, login: str, email: str, name: str) -> None:
        self.sim = sim
        self.host = host
        self.login = login
        self.email = email
        self.name = name
        self.repos: dict[str, Repo] = {}
        self.prs: list[PullRequest] = []
        self._next_id = FIRST_RUN_ID

    # world building ----------------------------------------------------------

    def repo(
        self,
        name: str,
        source: tuple[str, str] | None = None,
        default_branch: str = "main",
        checks: list[Check] | None = None,
    ) -> Repo:
        """A bare repo; source=(host, path) pushes every branch of a world repo into it.
        checks are copied onto PRs that `gh pr create` opens."""
        path = f"/root/{name}.git"
        real = self.sim.real_path(self.host, path)
        real.parent.mkdir(parents=True, exist_ok=True)
        self._git(real.parent, "init", "-q", "--bare", "-b", default_branch, real.name)
        if source is not None:
            self._git(
                self.sim.real_path(*source), "push", "-q", str(real), "refs/heads/*:refs/heads/*"
            )
        repo = Repo(name, path, default_branch, list(checks or []))
        self.repos[name] = repo
        self.save()
        return repo

    def pr(
        self,
        repo: str,
        head: str,
        title: str,
        base: str = "",
        body: str = "",
        author: str = "",
        draft: bool = False,
        checks: list[Check] | None = None,
        reviews: list[Review] | None = None,
    ) -> PullRequest:
        """An open PR from branch head (must exist in the repo) into base (default branch)."""
        r = self.repos[repo]
        sha = self.ref(repo, head)
        if not sha:
            raise ValueError(f"{repo} has no branch {head!r}")
        pr = PullRequest(
            repo,
            sum(1 for p in self.prs if p.repo == repo) + 1,
            title,
            head,
            base or r.default_branch,
            body,
            author or self.login,
            draft,
            head_sha=sha,
            created=self._now(),
            reviews=list(reviews or []),
        )
        for c in checks or []:
            self._add_check(pr, Check(c.name, c.status, c.log, c.required, c.workflow, c.rerun))
        self.prs.append(pr)
        self.save()
        return pr

    def review(self, pr: PullRequest, author: str, state: str, body: str = "") -> None:
        if state not in REVIEW_STATES:
            raise ValueError(f"review state must be one of {REVIEW_STATES}")
        pr.reviews.append(Review(author, state, body))
        self.save()

    def show(self, repo: str, rev: str, path: str) -> str | None:
        """File content at rev in the repo, None if absent."""
        return self._git(
            self._real(repo), "show", f"{rev}:{path.lstrip('/')}", check=False, strip=False
        )

    def ref(self, repo: str, branch: str) -> str:
        """Commit sha of a branch, "" if it does not exist."""
        return (
            self._git(
                self._real(repo),
                "rev-parse",
                "-q",
                "--verify",
                f"refs/heads/{branch}^{{commit}}",
                check=False,
            )
            or ""
        )

    def save(self) -> None:
        doc = {
            "account": {"login": self.login, "email": self.email},
            "repos": [
                {k: v for k, v in asdict(r).items() if k != "checks"} for r in self.repos.values()
            ],
            "prs": [_pr_doc(p) for p in self.prs],
        }
        (self.sim.root / "gh.json").write_text(json.dumps(doc, indent=1))

    # internals -----------------------------------------------------------------

    def _real(self, repo: str) -> Path:
        return self.sim.real_path(self.host, self.repos[repo].path)

    def _git(
        self,
        cwd: Path,
        *args: str,
        check: bool = True,
        env: dict[str, str] | None = None,
        strip: bool = True,
    ) -> str | None:
        """git on a world repo from outside the sandbox; None on failure unless check."""
        res = host_git(cwd, *args, env=env)
        if res.returncode != 0:
            if check:
                raise RuntimeError(f"git {' '.join(args)}: {res.stderr.strip()}")
            return None
        return res.stdout.strip() if strip else res.stdout

    def _now(self) -> str:
        return self.sim.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    def _add_check(self, pr: PullRequest, check: Check) -> None:
        if check.status not in CHECK_STATES:
            raise ValueError(f"check status must be one of {CHECK_STATES}")
        check.run_id = self._next_id
        self._next_id += 1
        pr.checks.append(check)

    def _url(self, pr: PullRequest) -> str:
        return f"https://{self.host}/{pr.repo}/pull/{pr.number}"

    def _run_url(self, pr: PullRequest, c: Check) -> str:
        return f"https://{self.host}/{pr.repo}/actions/runs/{c.run_id}/job/{c.run_id}"

    def on_push(self, path: str, before: dict[str, str], after: dict[str, str]) -> None:
        """A push to a bare repo on the GitHub host: move PR heads, re-run checks with rerun."""
        repo = next((r.name for r in self.repos.values() if r.path == path), None)
        if repo is None:
            return
        for ref, sha in after.items():
            if before.get(ref) == sha or not ref.startswith("refs/heads/"):
                continue
            for pr in self.prs:
                if (
                    pr.repo == repo
                    and pr.state == "OPEN"
                    and pr.head == ref.removeprefix("refs/heads/")
                ):
                    pr.head_sha = sha
                    for c in list(pr.checks):
                        if c.rerun is not None:
                            status, log = c.rerun(self, pr)
                            pr.checks.remove(c)
                            self._add_check(
                                pr, Check(c.name, status, log, c.required, c.workflow, c.rerun)
                            )
                    if pr.auto_merge and not self._blocking(pr):
                        with contextlib.suppress(
                            GhError
                        ):  # conflicts: auto-merge stays enabled, as on GitHub
                            self._merge(pr, pr.auto_merge)
        self.save()

    # gh command ------------------------------------------------------------------

    def handle(self, host: str, args: list[str], stdin: str | None) -> Result:
        try:
            return self._dispatch(host, args, stdin)
        except GhError as exc:
            return Result(stderr=f"{exc}\n", exit=1)

    def _dispatch(self, host: str, args: list[str], stdin: str | None) -> Result:
        if args in (["--version"], ["version"]):
            return Result(
                "gh version 2.83.0 (2026-10-01)\nhttps://github.com/cli/cli/releases/tag/v2.83.0\n",
                probe=True,
            )
        if args[:2] == ["auth", "status"]:
            return Result(
                f"{self.host}\n  ✓ Logged in to {self.host} account {self.login} (keyring)\n"
                "  - Active account: true\n  - Git operations protocol: ssh\n",
                probe=True,
            )
        if not args or args[0] == "help" or "--help" in args or "-h" in args:
            words = [a for a in args if not a.startswith("-") and a != "help"][:2]
            key = " ".join(words)
            if key not in GH_HELP:
                raise Gap(f"gh {key} --help".replace("  ", " "))
            return Result(GH_HELP[key], probe=True)
        if args[0] == "api":
            return self._api(host, args[1:], stdin)
        group, sub, rest = (args + ["", ""])[0], (args + ["", ""])[1], args[2:]
        handler = {
            ("pr", "list"): self._pr_list,
            ("pr", "view"): self._pr_view,
            ("pr", "checks"): self._pr_checks,
            ("pr", "diff"): self._pr_diff,
            ("pr", "merge"): self._pr_merge,
            ("pr", "create"): self._pr_create,
            ("pr", "comment"): self._pr_comment,
            ("pr", "edit"): self._pr_edit,
            ("run", "rerun"): self._run_rerun,
            ("run", "list"): self._run_list,
            ("run", "view"): self._run_view,
            ("repo", "view"): self._repo_view,
        }.get((group, sub))
        if handler is None:
            raise Gap(f"gh {group} {sub}".strip())
        return handler(host, rest, stdin)

    def _opts(
        self,
        cmd: str,
        args: list[str],
        short: str = "",
        short_valued: str = "",
        long: set[str] | None = None,
        long_valued: set[str] | None = None,
    ) -> tuple[dict[str, str], list[str]]:
        opts, pos = getopt(
            f"gh {cmd}",
            args,
            short=short,
            short_valued=short_valued + "Rq",
            long=frozenset(long or set()),
            long_valued=frozenset((long_valued or set()) | {"repo", "jq", "json"}),
        )
        return opts, pos

    def _cwd(self) -> str:
        return str(getattr(self.sim.tls, "cwd", "/root"))

    def _local_git(self, host: str, *args: str) -> str | None:
        """git in the caller's cwd on the world host, without searching above the host root."""
        ceiling = str(self.sim.hosts_dir / host)
        return self._git(
            self.sim.real_path(host, self._cwd()),
            *args,
            check=False,
            env={"GIT_CEILING_DIRECTORIES": ceiling},
        )

    def _repo(self, host: str, opts: dict[str, str]) -> str:
        name = opts.get("R", opts.get("repo"))
        if name is None:
            remotes = self._local_git(host, "remote", "-v")
            if remotes is None:
                raise GhError(
                    "failed to run git: fatal: not a git repository (or any of the parent directories): .git"
                )
            pattern = re.compile(
                rf"(?:ssh://)?(?:[\w.-]+@)?{re.escape(self.host)}[:/]/?([\w.-]+/[\w.-]+?)(?:\.git)?/?\s"
            )
            found = [
                m.group(1)
                for line in sorted(remotes.splitlines(), key=lambda r: not r.startswith("origin"))
                if (m := pattern.search(line + " "))
            ]
            if not found:
                raise GhError(
                    "none of the git remotes configured for this repository point to a known GitHub host. "
                    "To tell gh about a new GitHub host, please use `gh auth login`"
                )
            name = found[0]
        name = re.sub(rf"^(https://)?{re.escape(self.host)}/", "", name).removesuffix(".git")
        if name not in self.repos:
            raise GhError(
                f"GraphQL: Could not resolve to a Repository with the name '{name}'. (repository)"
            )
        return name

    def _select(self, host: str, repo: str, pos: list[str]) -> PullRequest:
        """PR by number, #number, URL or branch; none means the current branch's PR."""
        if len(pos) > 1:
            raise GhError(f"accepts at most 1 arg(s), received {len(pos)}")
        prs = [p for p in self.prs if p.repo == repo]
        if pos:
            sel = pos[0].lstrip("#")
            m = re.search(r"/pull/(\d+)", sel)
            if m:
                sel = m.group(1)
            if sel.isdigit():
                pr = next((p for p in prs if p.number == int(sel)), None)
                if pr is None:
                    raise GhError(
                        f"GraphQL: Could not resolve to a PullRequest with the number of {sel}. (repository.pullRequest)"
                    )
                return pr
            branch = sel
        else:
            branch = self._local_git(host, "rev-parse", "--abbrev-ref", "HEAD") or ""
        matches = [p for p in prs if p.head == branch]
        pr = next(
            (p for p in reversed(matches) if p.state == "OPEN"), matches[-1] if matches else None
        )
        if pr is None:
            raise GhError(f'no pull requests found for branch "{branch}"')
        return pr

    def _emit(self, opts: dict[str, str], docs: JsonValue, indent: int | None = 2) -> Result:
        jq = opts.get("q", opts.get("jq"))
        if jq is not None:
            return run_jq(jq, docs)
        return Result(
            json.dumps(docs, indent=indent, separators=None if indent else (",", ":")) + "\n",
            probe=True,
        )

    def _fields(
        self, opts: dict[str, str], doc_of: Callable[[str], JsonValue]
    ) -> dict[str, JsonValue]:
        return {f: doc_of(f) for f in opts["json"].split(",") if f}

    # pr ------------------------------------------------------------------------

    def _decision(self, pr: PullRequest) -> str:
        latest = {r.author: r.state for r in pr.reviews if r.state != "COMMENTED"}
        if "CHANGES_REQUESTED" in latest.values():
            return "CHANGES_REQUESTED"
        return "APPROVED" if "APPROVED" in latest.values() else "REVIEW_REQUIRED"

    def _diff_base(self, pr: PullRequest) -> str:
        return (
            pr.merge_commit + "^"
            if pr.merge_commit and pr.merge_method != "rebase"
            else f"refs/heads/{pr.base}"
        )

    def _files(self, pr: PullRequest) -> list[dict[str, JsonValue]]:
        """Changed files with line counts and status (added|modified|removed|renamed)."""
        real, rng = self._real(pr.repo), f"{self._diff_base(pr)}...{pr.head_sha}"
        nums = (
            self._git(real, "diff", "--numstat", "--no-renames", rng, check=False) or ""
        ).splitlines()
        kinds = {
            path: kind
            for kind, path in (
                line.split("\t", 1)
                for line in (
                    self._git(real, "diff", "--name-status", "--no-renames", rng, check=False) or ""
                ).splitlines()
                if "\t" in line
            )
        }
        out: list[dict[str, JsonValue]] = []
        for line in nums:
            add, delete, path = line.split("\t", 2)
            status = {"A": "added", "D": "removed"}.get(kinds.get(path, "M"), "modified")
            out.append(
                {
                    "path": path,
                    "additions": int(add) if add.isdigit() else 0,
                    "deletions": int(delete) if delete.isdigit() else 0,
                    "status": status,
                }
            )
        return out

    def _commits(self, pr: PullRequest) -> list[dict[str, JsonValue]]:
        fmt = "--format=%H%x00%s%x00%b%x00%an%x00%ae%x00%aI%x00%cI%x1e"
        log = (
            self._git(
                self._real(pr.repo),
                "log",
                "--reverse",
                fmt,
                f"{self._diff_base(pr)}..{pr.head_sha}",
                check=False,
                strip=False,
            )
            or ""
        )
        out: list[dict[str, JsonValue]] = []
        for rec in filter(None, (r.strip("\n") for r in log.split("\x1e"))):
            oid, subject, body, name, email, authored, committed = (rec.split("\x00") + [""] * 7)[
                :7
            ]
            login = self.login if email == self.email else ""
            out.append(
                {
                    "oid": oid,
                    "messageHeadline": subject,
                    "messageBody": body.strip(),
                    "authoredDate": authored,
                    "committedDate": committed,
                    "authors": [{"email": email, "id": "", "login": login, "name": name}],
                }
            )
        return out

    def _merge_state(self, pr: PullRequest) -> str:
        if pr.state != "OPEN":
            return "UNKNOWN"
        if pr.draft:
            return "DRAFT"
        if not self._merge_tree(pr):
            return "DIRTY"
        if self._blocking(pr) or self._decision(pr) == "CHANGES_REQUESTED":
            return "BLOCKED"
        if any(c.status == "fail" for c in pr.checks):
            return "UNSTABLE"
        return "CLEAN"

    def _pr_field(self, pr: PullRequest, name: str) -> JsonValue:
        if name == "mergeable":
            return (
                ("MERGEABLE" if self._merge_tree(pr) else "CONFLICTING")
                if pr.state == "OPEN"
                else "UNKNOWN"
            )
        if name == "mergeStateStatus":
            return self._merge_state(pr)
        if name in ("files", "additions", "deletions", "changedFiles"):
            files = self._files(pr)
            if name == "files":
                return [{k: f[k] for k in ("path", "additions", "deletions")} for f in files]
            if name == "changedFiles":
                return len(files)
            return sum(int(str(f[name])) for f in files)
        if name == "commits":
            return list(self._commits(pr))
        checks: list[JsonValue] = [
            {
                "__typename": "CheckRun",
                "name": c.name,
                "workflowName": c.workflow,
                "status": "IN_PROGRESS" if c.status == "pending" else "COMPLETED",
                "conclusion": {"pass": "SUCCESS", "fail": "FAILURE", "pending": ""}[c.status],
                "detailsUrl": self._run_url(pr, c),
            }
            for c in pr.checks
        ]
        values: dict[str, JsonValue] = {
            "number": pr.number,
            "title": pr.title,
            "state": pr.state,
            "body": pr.body,
            "url": self._url(pr),
            "headRefName": pr.head,
            "baseRefName": pr.base,
            "headRefOid": pr.head_sha,
            "isDraft": pr.draft,
            "author": {"login": pr.author},
            "createdAt": pr.created,
            "mergedAt": pr.merged_at or None,
            "mergedBy": {"login": pr.merged_by} if pr.merged_by else None,
            "mergeCommit": {"oid": pr.merge_commit} if pr.merge_commit else None,
            "reviewDecision": self._decision(pr),
            "reviews": [
                {"author": {"login": r.author}, "state": r.state, "body": r.body}
                for r in pr.reviews
            ],
            "comments": [{"author": {"login": c.author}, "body": c.body} for c in pr.comments],
            "statusCheckRollup": checks,
            "autoMergeRequest": {
                "mergeMethod": pr.auto_merge["method"].upper(),
                "authorEmail": pr.auto_merge["email"],
            }
            if pr.auto_merge
            else None,
            "closed": pr.state != "OPEN",
            "closedAt": pr.merged_at or None,
            "updatedAt": pr.merged_at or pr.created,
            "id": "PR_kwDO" + hashlib.sha256(f"{pr.repo}#{pr.number}".encode()).hexdigest()[:16],
            "labels": [],
            "assignees": [],
            "reviewRequests": [],
            "milestone": None,
            "projectCards": [],
            "isCrossRepository": False,
            "maintainerCanModify": True,
            "headRepository": {"id": "", "name": pr.repo.split("/")[1]},
            "headRepositoryOwner": {"id": "", "login": pr.repo.split("/")[0]},
            "baseRefOid": self.ref(pr.repo, pr.base),
            "latestReviews": [
                {"author": {"login": a}, "state": st, "body": b}
                for a, (st, b) in {r.author: (r.state, r.body) for r in pr.reviews}.items()
            ],
        }
        if name not in values:
            raise Gap(f"gh pr --json {name}")
        return values[name]

    def _repo_view(self, host: str, args: list[str], stdin: str | None) -> Result:
        opts, pos = self._opts("repo view", args, short="w", long={"web"})
        if "w" in opts or "web" in opts:
            raise Gap("gh repo view --web")
        if len(pos) > 1:
            raise GhError(f"accepts at most 1 arg(s), received {len(pos)}")
        r = self.repos[self._repo(host, {"R": pos[0]} if pos else opts)]
        values: dict[str, JsonValue] = {
            "name": r.name.split("/")[1],
            "nameWithOwner": r.name,
            "owner": {"login": r.name.split("/")[0]},
            "url": f"https://{self.host}/{r.name}",
            "sshUrl": f"git@{self.host}:{r.name}.git",
            "defaultBranchRef": {"name": r.default_branch},
            "description": "",
            "visibility": "PRIVATE",
            "isPrivate": True,
        }

        def field_of(name: str) -> JsonValue:
            if name not in values:
                raise Gap(f"gh repo view --json {name}")
            return values[name]

        if "json" in opts:
            return self._emit(opts, self._fields(opts, field_of))
        return Result(f"name:\t{r.name}\ndescription:\t\n--\nNo description provided\n", probe=True)

    def _pr_list(self, host: str, args: list[str], stdin: str | None) -> Result:
        opts, pos = self._opts(
            "pr list",
            args,
            short_valued="LAHBsl",
            long_valued={"limit", "author", "head", "base", "state", "label"},
            long={"draft"},
        )
        if pos:
            raise GhError(f"unknown argument {pos[0]!r}")
        repo = self._repo(host, opts)
        state = opts.get("s", opts.get("state", "open")).upper()
        if state not in ("OPEN", "CLOSED", "MERGED", "ALL"):
            raise GhError(
                f'invalid argument "{state.lower()}" for "-s, --state" flag: valid values are {{open|closed|merged|all}}'
            )
        if "l" in opts or "label" in opts:
            raise Gap("gh pr list --label")
        want = {
            "author": opts.get("A", opts.get("author")),
            "head": opts.get("H", opts.get("head")),
            "base": opts.get("B", opts.get("base")),
        }
        rows = [
            p
            for p in reversed(self.prs)
            if p.repo == repo
            and (state == "ALL" or p.state == state or (state == "CLOSED" and p.state == "MERGED"))
            and all(v is None or getattr(p, k) == v for k, v in want.items())
            and ("draft" not in opts or p.draft)
        ][: int(opts.get("L", opts.get("limit", "30")))]
        if "json" in opts:
            return self._emit(
                opts, [self._fields(opts, functools.partial(self._pr_field, p)) for p in rows]
            )
        return Result(
            "".join(
                f"{p.number}\t{p.title}\t{p.head}\t{'DRAFT' if p.draft and p.state == 'OPEN' else p.state}\t{p.created}\n"
                for p in rows
            ),
            probe=True,
        )

    def _pr_view(self, host: str, args: list[str], stdin: str | None) -> Result:
        opts, pos = self._opts("pr view", args, short="cw", long={"comments", "web"})
        if "w" in opts or "web" in opts:
            raise Gap("gh pr view --web")
        repo = self._repo(host, opts)
        pr = self._select(host, repo, pos)
        if "json" in opts:
            return self._emit(opts, self._fields(opts, lambda f: self._pr_field(pr, f)))
        reviewers = ", ".join(
            f"{r.author} ({r.state.replace('_', ' ').title()})" for r in pr.reviews
        )
        lines = [
            f"title:\t{pr.title}",
            f"state:\t{'DRAFT' if pr.draft and pr.state == 'OPEN' else pr.state}",
            f"author:\t{pr.author}",
            "labels:\t",
            "assignees:\t",
            f"reviewers:\t{reviewers}",
            "projects:\t",
            "milestone:\t",
            f"number:\t{pr.number}",
            f"url:\t{self._url(pr)}",
            f"auto-merge:\t{'enabled' if pr.auto_merge else 'disabled'}",
            "--",
            pr.body,
        ]
        if "c" in opts or "comments" in opts:
            for c in pr.comments:
                lines += [
                    "--",
                    f"author:\t{c.author}",
                    "association:\tmember",
                    "edited:\tfalse",
                    "status:\tnone",
                    "--",
                    c.body,
                ]
        return Result("\n".join(lines) + "\n", probe=True)

    def _pr_checks(self, host: str, args: list[str], stdin: str | None) -> Result:
        opts, pos = self._opts(
            "pr checks", args, long={"required", "watch", "fail-fast"}, long_valued={"interval"}
        )
        if "watch" in opts:
            raise Gap("gh pr checks --watch")
        repo = self._repo(host, opts)
        pr = self._select(host, repo, pos)
        checks = [c for c in pr.checks if c.required or "required" not in opts]
        if not checks:
            raise GhError(
                f"no {'required ' if 'required' in opts else ''}checks reported on the '{pr.head}' branch"
            )
        if "json" in opts:

            def check_field(c: Check, f: str) -> JsonValue:
                values: dict[str, JsonValue] = {
                    "name": c.name,
                    "state": {"pass": "SUCCESS", "fail": "FAILURE", "pending": "PENDING"}[c.status],
                    "bucket": c.status,
                    "link": self._run_url(pr, c),
                    "workflow": c.workflow,
                    "description": "",
                }
                if f not in values:
                    raise Gap(f"gh pr checks --json {f}")
                return values[f]

            return self._emit(
                opts, [self._fields(opts, functools.partial(check_field, c)) for c in checks]
            )
        out = "".join(
            f"{c.name}\t{c.status}\t{'0' if c.status == 'pending' else '1m12s'}\t{self._run_url(pr, c)}\t\n"
            for c in checks
        )
        code = (
            1
            if any(c.status == "fail" for c in checks)
            else 8
            if any(c.status == "pending" for c in checks)
            else 0
        )
        return Result(out, exit=code, probe=True)

    def _pr_diff(self, host: str, args: list[str], stdin: str | None) -> Result:
        opts, pos = self._opts("pr diff", args, long={"name-only", "patch"}, long_valued={"color"})
        repo = self._repo(host, opts)
        pr = self._select(host, repo, pos)
        base = (
            pr.merge_commit + "^"
            if pr.merge_commit and pr.merge_method != "rebase"
            else f"refs/heads/{pr.base}"
        )
        diff = self._git(
            self._real(repo),
            "diff",
            "--name-only" if "name-only" in opts else "--patch",
            f"{base}...{pr.head_sha}",
            check=False,
        )
        return Result((diff + "\n") if diff else "", probe=True)

    def _merge_tree(self, pr: PullRequest, base_sha: str = "") -> str | None:
        """Tree of base merged with the PR head, None on conflict."""
        base_sha = base_sha or self.ref(pr.repo, pr.base)
        return self._git(
            self._real(pr.repo),
            "merge-tree",
            "--write-tree",
            "--no-messages",
            base_sha,
            pr.head_sha,
            check=False,
        )

    def _commit(
        self,
        repo: str,
        tree: str,
        parents: list[str],
        message: str,
        author: tuple[str, str],
        committer: tuple[str, str],
        date: str = "",
    ) -> str:
        env = {
            "GIT_AUTHOR_NAME": author[0],
            "GIT_AUTHOR_EMAIL": author[1],
            "GIT_COMMITTER_NAME": committer[0],
            "GIT_COMMITTER_EMAIL": committer[1],
        }
        if date:
            env["GIT_AUTHOR_DATE"] = date
        args = ["commit-tree", tree, "-m", message]
        for p in parents:
            args += ["-p", p]
        sha = self._git(self._real(repo), *args, env=env)
        assert sha
        return sha

    def _pr_merge(self, host: str, args: list[str], stdin: str | None) -> Result:
        opts, pos = self._opts(
            "pr merge",
            args,
            short="smrd",
            short_valued="tb",
            long={"squash", "merge", "rebase", "delete-branch", "admin", "auto", "disable-auto"},
            long_valued={"author-email", "subject", "body", "match-head-commit"},
        )
        repo = self._repo(host, opts)
        pr = self._select(host, repo, pos)
        ref = f"{pr.repo}#{pr.number}"
        if "disable-auto" in opts:
            pr.auto_merge = {}
            self.save()
            return Result(f"✓ Auto-merge disabled for pull request {ref}\n")
        methods = [
            m
            for m, s in (("squash", "s"), ("merge", "m"), ("rebase", "r"))
            if m in opts or s in opts
        ]
        if not methods:
            raise GhError("--merge, --rebase, or --squash required when not running interactively")
        if len(methods) > 1:
            raise GhError("only one of --merge, --rebase, or --squash can be enabled")
        auto, admin = "auto" in opts, "admin" in opts
        if auto and admin:
            raise GhError("specify only one of `--auto`, `--disable-auto`, or `--admin`")
        if pr.state != "OPEN":
            raise GhError(
                f"X Pull request {ref} ({pr.title}) {'was already merged' if pr.state == 'MERGED' else 'is closed'}"
            )
        if pr.draft:
            raise GhError(f"X Pull request {ref} is still a draft")
        merge = {
            "method": methods[0],
            "email": opts.get("author-email", self.email),
            "subject": opts.get("t", opts.get("subject", "")),
            "body": opts.get("b", opts.get("body", "")),
            "delete": "yes" if "d" in opts or "delete-branch" in opts else "",
            "admin": "yes" if admin else "",
        }
        if self._blocking(pr) and not admin:
            if auto:
                pr.auto_merge = merge
                self.save()
                return Result(
                    f"✓ Pull request {ref} will be automatically merged via {methods[0]} when all requirements are met\n"
                )
            raise GhError(
                f"X Pull request {ref} is not mergeable: the base branch policy prohibits the merge.\n"
                "To have the pull request merged after all the requirements have been met, add the `--auto` flag.\n"
                "To use administrator privileges to immediately merge the pull request, add the `--admin` flag."
            )
        return Result(self._merge(pr, merge))

    def _blocking(self, pr: PullRequest) -> list[Check]:
        return [c for c in pr.checks if c.required and c.status != "pass"]

    def _merge(self, pr: PullRequest, merge: dict[str, str]) -> str:
        """Merge now with the options _pr_merge collected; returns gh's output."""
        repo, ref = pr.repo, f"{pr.repo}#{pr.number}"
        base_sha = self.ref(repo, pr.base)
        tree = self._merge_tree(pr, base_sha)
        if not tree:
            raise GhError(
                f"X Pull request {ref} is not mergeable: the merge commit cannot be cleanly created."
            )
        method, email = merge["method"], merge["email"]
        me = (self.name, email)
        if method == "squash":
            log = (
                self._git(
                    self._real(repo),
                    "log",
                    "--reverse",
                    "--format=* %s",
                    f"{base_sha}..{pr.head_sha}",
                )
                or ""
            )
            subject = merge["subject"] or f"{pr.title} (#{pr.number})"
            body = merge["body"] or log
            new = self._commit(
                repo, tree, [base_sha], f"{subject}\n\n{body}".rstrip(), me, WEB_COMMITTER
            )
        elif method == "merge":
            subject = (
                merge["subject"]
                or f"Merge pull request #{pr.number} from {repo.split('/')[0]}/{pr.head}"
            )
            body = merge["body"] or pr.title
            new = self._commit(
                repo,
                tree,
                [base_sha, pr.head_sha],
                f"{subject}\n\n{body}".rstrip(),
                me,
                WEB_COMMITTER,
            )
        else:
            new = base_sha
            commits = (
                self._git(
                    self._real(repo),
                    "rev-list",
                    "--reverse",
                    "--no-merges",
                    f"{base_sha}..{pr.head_sha}",
                )
                or ""
            ).split()
            for c in commits:
                fmt = (
                    self._git(self._real(repo), "log", "-1", "--format=%an%x00%ae%x00%aD%x00%B", c)
                    or ""
                )
                an, ae, ad, msg = (fmt.split("\x00") + ["", "", "", ""])[:4]
                step_tree = self._git(
                    self._real(repo),
                    "merge-tree",
                    "--write-tree",
                    "--no-messages",
                    f"--merge-base={c}^",
                    new,
                    c,
                    check=False,
                )
                if not step_tree:
                    raise GhError(
                        f"X Pull request {ref} is not mergeable: the merge commit cannot be cleanly created."
                    )
                new = self._commit(repo, step_tree, [new], msg.strip(), (an, ae), me, ad)
        self._git(self._real(repo), "update-ref", f"refs/heads/{pr.base}", new, base_sha)
        pr.state, pr.merge_commit, pr.merge_method = "MERGED", new, method
        pr.merged_by, pr.merged_at, pr.auto_merge = self.login, self._now(), {}
        verb = {"squash": "Squashed and merged", "merge": "Merged", "rebase": "Rebased and merged"}[
            method
        ]
        out = f"✓ {verb} pull request {ref} ({pr.title})\n"
        if merge["delete"]:
            self._git(self._real(repo), "update-ref", "-d", f"refs/heads/{pr.head}", check=False)
            out += f"✓ Deleted remote branch {pr.head}\n"
        self.sim.event(
            "merge",
            self.host,
            text=f"{ref} {method} author={email}" + (" admin" if merge["admin"] else ""),
            action=method,
            mutation=True,
            paths=[f"{self.host}:{self.repos[repo].path}"],
        )
        self.save()
        return out

    def _body(self, host: str, opts: dict[str, str], stdin: str | None) -> str | None:
        path = opts.get("F", opts.get("body-file"))
        if path is None:
            return opts.get("b", opts.get("body"))
        if path == "-":
            return stdin or ""
        text = self.sim.read(host, path if path.startswith("/") else f"{self._cwd()}/{path}")
        if text is None:
            raise GhError(f"open {path}: no such file or directory")
        return text

    def _pr_create(self, host: str, args: list[str], stdin: str | None) -> Result:
        opts, pos = self._opts(
            "pr create",
            args,
            short="dfw",
            short_valued="tbFBHralm",
            long={"draft", "fill", "fill-first", "web", "no-maintainer-edit"},
            long_valued={
                "title",
                "body",
                "body-file",
                "base",
                "head",
                "reviewer",
                "assignee",
                "label",
                "milestone",
            },
        )
        if pos:
            raise GhError(f"unknown argument {pos[0]!r}")
        if "w" in opts or "web" in opts:
            raise Gap("gh pr create --web")
        repo = self._repo(host, opts)
        r = self.repos[repo]
        head = (
            opts.get("H", opts.get("head"))
            or self._local_git(host, "rev-parse", "--abbrev-ref", "HEAD")
            or ""
        )
        head = head.split(":")[-1]
        base = opts.get("B", opts.get("base", r.default_branch))
        if not self.ref(repo, head):
            raise GhError(
                "aborted: you must first push the current branch to a remote, or use the --head flag"
            )
        if not self.ref(repo, base):
            raise GhError(
                "pull request create failed: GraphQL: Base ref must be a branch (createPullRequest)"
            )
        existing = next(
            (
                p
                for p in self.prs
                if p.repo == repo and p.head == head and p.base == base and p.state == "OPEN"
            ),
            None,
        )
        if existing:
            raise GhError(
                f'a pull request for branch "{head}" into branch "{base}" already exists:\n{self._url(existing)}'
            )
        if not self._git(self._real(repo), "rev-list", f"refs/heads/{base}..refs/heads/{head}"):
            raise GhError(
                f"pull request create failed: GraphQL: No commits between {base} and {head} (createPullRequest)"
            )
        title = opts.get("t", opts.get("title"))
        body = self._body(host, opts, stdin)
        if "f" in opts or "fill" in opts or "fill-first" in opts:
            subject = (
                self._git(self._real(repo), "log", "-1", "--format=%s", f"refs/heads/{head}")
                or head
            )
            title, body = title or subject, body if body is not None else ""
        if not title:
            raise GhError(
                "must provide `--title` and `--body` (or `--fill` or `--fill-first`) when not running interactively"
            )
        pr = self.pr(
            repo,
            head,
            title,
            base,
            body or "",
            self.login,
            "d" in opts or "draft" in opts,
            checks=[],
        )
        for c in r.checks:
            status, log = c.rerun(self, pr) if c.rerun else (c.status, c.log)
            self._add_check(pr, Check(c.name, status, log, c.required, c.workflow, c.rerun))
        self.save()
        self.sim.event(
            "gh",
            self.host,
            text=f"{repo}#{pr.number} create {head}->{base}",
            action="create",
            mutation=True,
        )
        return Result(self._url(pr) + "\n")

    def _pr_comment(self, host: str, args: list[str], stdin: str | None) -> Result:
        opts, pos = self._opts(
            "pr comment",
            args,
            short_valued="bF",
            long_valued={"body", "body-file"},
            long={"edit-last"},
        )
        if "edit-last" in opts:
            raise Gap("gh pr comment --edit-last")
        repo = self._repo(host, opts)
        pr = self._select(host, repo, pos)
        body = self._body(host, opts, stdin)
        if not body:
            raise GhError("--body or --body-file required when not running interactively")
        cid = self._next_id
        self._next_id += 1
        pr.comments.append(Comment(self.login, body, cid))
        self.save()
        self.sim.event(
            "gh", self.host, text=f"{repo}#{pr.number} comment", action="comment", mutation=True
        )
        return Result(f"{self._url(pr)}#issuecomment-{cid}\n")

    def _pr_edit(self, host: str, args: list[str], stdin: str | None) -> Result:
        opts, pos = self._opts(
            "pr edit",
            args,
            short_valued="tbFB",
            long_valued={
                "title",
                "body",
                "body-file",
                "base",
                "add-label",
                "remove-label",
                "add-reviewer",
                "remove-reviewer",
                "add-assignee",
                "remove-assignee",
                "milestone",
                "add-project",
                "remove-project",
            },
            long={"remove-milestone"},
        )
        for key in (
            "add-label",
            "remove-label",
            "add-reviewer",
            "remove-reviewer",
            "add-assignee",
            "remove-assignee",
            "milestone",
            "add-project",
            "remove-project",
            "remove-milestone",
        ):
            if key in opts:
                raise Gap(f"gh pr edit --{key}")
        repo = self._repo(host, opts)
        pr = self._select(host, repo, pos)
        return Result(
            self._edit(
                pr,
                opts.get("t", opts.get("title")),
                self._body(host, opts, stdin),
                opts.get("B", opts.get("base")),
            )
            + "\n"
        )

    def _edit(self, pr: PullRequest, title: str | None, body: str | None, base: str | None) -> str:
        """Change title, body or base of a PR; returns its URL. Logged as a `gh` edit event."""
        if title is None and body is None and base is None:
            raise GhError(
                "--title, --body, --body-file or --base required when not running interactively"
            )
        if title is not None and not title.strip():
            raise GhError("GraphQL: Title can't be blank (updatePullRequest)")
        if base is not None and not self.ref(pr.repo, base):
            raise GhError(
                f"GraphQL: Proposed base branch '{base}' was not found (updatePullRequest)"
            )
        changed = []
        if title is not None:
            pr.title = title
            changed.append(f"title={title}")
        if body is not None:
            pr.body = body
            changed.append("body")
        if base is not None:
            pr.base = base
            changed.append(f"base={base}")
        self.save()
        self.sim.event(
            "gh",
            self.host,
            text=f"{pr.repo}#{pr.number} edit " + " ".join(changed),
            action="edit",
            mutation=True,
        )
        return self._url(pr)

    # run -----------------------------------------------------------------------

    def _runs(self, repo: str) -> list[tuple[PullRequest, Check]]:
        return sorted(
            ((p, c) for p in self.prs if p.repo == repo for c in p.checks),
            key=lambda pc: -pc[1].run_id,
        )

    def _run_field(self, pr: PullRequest, c: Check, name: str) -> JsonValue:
        values: dict[str, JsonValue] = {
            "databaseId": c.run_id,
            "status": "in_progress" if c.status == "pending" else "completed",
            "conclusion": {"pass": "success", "fail": "failure", "pending": ""}[c.status],
            "name": c.workflow,
            "workflowName": c.workflow,
            "displayTitle": pr.title,
            "headBranch": pr.head,
            "headSha": pr.head_sha,
            "event": "pull_request",
            "url": f"https://{self.host}/{pr.repo}/actions/runs/{c.run_id}",
            "createdAt": pr.created,
            "startedAt": pr.created,
            "updatedAt": pr.created,
            "attempt": c.attempt,
            "number": c.run_id - FIRST_RUN_ID + 1,
            "workflowDatabaseId": 1000 + sum(map(ord, c.workflow)),
            "jobs": [self._job(pr, c)],
        }
        if name not in values:
            raise Gap(f"gh run --json {name}")
        return values[name]

    def _job(self, pr: PullRequest, c: Check) -> dict[str, JsonValue]:
        status = "in_progress" if c.status == "pending" else "completed"
        conclusion = {"pass": "success", "fail": "failure", "pending": ""}[c.status]
        steps: list[JsonValue] = [
            {"name": "Set up job", "status": "completed", "conclusion": "success", "number": 1},
            {"name": f"Run {c.name}", "status": status, "conclusion": conclusion, "number": 2},
            {
                "name": "Complete job",
                "status": status,
                "conclusion": "success" if status == "completed" else "",
                "number": 3,
            },
        ]
        return {
            "databaseId": c.run_id,
            "name": c.name,
            "status": status,
            "conclusion": conclusion,
            "startedAt": pr.created,
            "completedAt": pr.created if status == "completed" else "0001-01-01T00:00:00Z",
            "url": self._run_url(pr, c),
            "steps": steps,
        }

    def _run_rerun(self, host: str, args: list[str], stdin: str | None) -> Result:
        """Re-run a check on the same head: rerun(gh, pr) decides the result, else it repeats."""
        opts, pos = self._opts(
            "run rerun", args, short_valued="j", long={"failed", "debug"}, long_valued={"job"}
        )
        repo = self._repo(host, opts)
        run_id = pos[0] if pos else opts.get("j", opts.get("job"))
        if run_id is None:
            raise GhError("run or job ID required when not running interactively")
        found = next(((p, c) for p, c in self._runs(repo) if str(c.run_id) == run_id), None)
        if found is None:
            raise GhError(
                f"HTTP 404: Not Found (https://api.{self.host}/repos/{repo}/actions/runs/{run_id})"
            )
        pr, c = found
        if c.status == "pending":
            raise GhError(f"run {run_id} cannot be rerun; it is still in progress")
        if "failed" in opts and c.status != "fail":
            raise GhError(f"no failed jobs to rerun in run {run_id}")
        c.status, c.log = c.rerun(self, pr) if c.rerun else (c.status, c.log)
        c.attempt += 1
        if pr.auto_merge and not self._blocking(pr):
            with contextlib.suppress(GhError):
                self._merge(pr, pr.auto_merge)
        self.save()
        self.sim.event("gh", self.host, text=f"{repo}#{pr.number} rerun {c.name}", action="rerun")
        return Result(f"✓ Requested rerun of run {run_id}\n")

    def _run_list(self, host: str, args: list[str], stdin: str | None) -> Result:
        opts, pos = self._opts(
            "run list",
            args,
            short_valued="bLws",
            long_valued={"branch", "limit", "workflow", "status"},
        )
        repo = self._repo(host, opts)
        branch = opts.get("b", opts.get("branch"))
        workflow = opts.get("w", opts.get("workflow"))
        status = opts.get("s", opts.get("status"))
        rows = [
            (p, c)
            for p, c in self._runs(repo)
            if (branch is None or p.head == branch)
            and (workflow is None or c.workflow == workflow)
            and (
                status is None
                or status in (self._run_field(p, c, "status"), self._run_field(p, c, "conclusion"))
            )
        ][: int(opts.get("L", opts.get("limit", "20")))]
        if "json" in opts:
            return self._emit(
                opts,
                [self._fields(opts, functools.partial(self._run_field, p, c)) for p, c in rows],
            )
        return Result(
            "".join(
                f"{self._run_field(p, c, 'status')}\t{self._run_field(p, c, 'conclusion')}\t{p.title}\t{c.workflow}\t"
                f"{p.head}\tpull_request\t{c.run_id}\t1m12s\t5m\n"
                for p, c in rows
            ),
            probe=True,
        )

    def _run_view(self, host: str, args: list[str], stdin: str | None) -> Result:
        opts, pos = self._opts(
            "run view",
            args,
            short="vw",
            short_valued="j",
            long={"log", "log-failed", "verbose", "web", "exit-status"},
            long_valued={"job"},
        )
        if "w" in opts or "web" in opts:
            raise Gap("gh run view --web")
        repo = self._repo(host, opts)
        job = opts.get("j", opts.get("job"))
        run_id = pos[0] if pos else job
        if run_id is None:
            raise GhError("run or job ID required when not running interactively")
        found = next(((p, c) for p, c in self._runs(repo) if str(c.run_id) == run_id), None)
        if found is None:
            raise GhError(
                f"HTTP 404: Not Found (https://api.{self.host}/repos/{repo}/actions/runs/{run_id})"
            )
        pr, c = found
        if "json" in opts:
            return self._emit(opts, self._fields(opts, lambda f: self._run_field(pr, c, f)))
        if "log" in opts or "log-failed" in opts:
            if c.status == "pending":
                raise GhError(
                    f"run {c.run_id} is still in progress; logs will be available when it is complete"
                )
            if "log-failed" in opts and c.status != "fail":
                return Result(probe=True)
            return Result(
                "".join(f"{c.name}\tRun {c.name}\t{line}\n" for line in c.log.splitlines()),
                probe=True,
            )
        mark = {"pass": "✓", "fail": "X", "pending": "*"}[c.status]
        text = (
            f"{mark} {pr.head} {c.workflow} · {c.run_id}\nTriggered via pull_request about 5 minutes ago\n\n"
            f"JOBS\n{mark} {c.name} in 1m12s (ID {c.run_id})\n"
        )
        if c.status == "fail":
            text += f"\nTo see what failed, try: gh run view {c.run_id} --log-failed\n"
        code = 1 if "exit-status" in opts and c.status == "fail" else 0
        return Result(text, exit=code, probe=True)

    # api -----------------------------------------------------------------------

    def _api(self, host: str, args: list[str], stdin: str | None) -> Result:
        """`gh api` for the REST endpoints people reach for around PRs and runs:
        repos/O/R, .../pulls[/N[/files|/commits]] (GET; PATCH title/body/base), .../actions/runs/ID[/jobs]."""
        opts, pos = _api_opts(args)
        if len(pos) != 1:
            raise GhError(f"accepts 1 arg(s), received {len(pos)}")
        if "t" in opts or "template" in opts or "input" in opts:
            raise Gap("gh api --template/--input")
        fields = {
            k: v
            for key in ("f", "raw-field", "F", "field")
            for item in opts.get(key, "").split("\n")
            if item
            for k, _, v in [item.partition("=")]
        }
        method = opts.get("X", opts.get("method", "POST" if fields else "GET")).upper()
        path = pos[0].lstrip("/").split("?", 1)[0]
        if "{owner}" in path or "{repo}" in path:
            owner, name = self._repo(host, {}).split("/")
            path = path.replace("{owner}", owner).replace("{repo}", name)
        m = re.fullmatch(
            r"repos/([\w.-]+/[\w.-]+)(?:/(pulls|actions/runs)(?:/(\d+)(?:/(files|commits|jobs))?)?)?",
            path,
        )
        shape = re.sub(r"\d+", "N", path)
        if m is None or m.group(1) not in self.repos:
            if m is not None:
                return self._api_error(404)
            raise Gap(f"gh api {method} {shape}")
        repo, kind, num, tail = m.group(1), m.group(2), m.group(3), m.group(4)
        doc: JsonValue
        if method == "PATCH" and kind == "pulls" and num and not tail:
            unknown = set(fields) - {"title", "body", "base"}
            if unknown:
                raise Gap(f"gh api PATCH pulls fields {sorted(unknown)}")
            pr = next((p for p in self.prs if p.repo == repo and p.number == int(num)), None)
            if pr is None:
                return self._api_error(404)
            self._edit(pr, fields.get("title"), fields.get("body"), fields.get("base"))
            doc = self._rest_pr(pr)
        elif method != "GET":
            raise Gap(f"gh api {method} {shape}")
        elif kind is None:
            r = self.repos[repo]
            doc = {
                "name": repo.split("/")[1],
                "full_name": repo,
                "private": True,
                "default_branch": r.default_branch,
                "html_url": f"https://{self.host}/{repo}",
                "ssh_url": f"git@{self.host}:{repo}.git",
                "owner": {"login": repo.split("/")[0]},
            }
        elif kind == "pulls":
            prs = [p for p in self.prs if p.repo == repo]
            if num is None:
                doc = [self._rest_pr(p) for p in reversed(prs) if p.state == "OPEN"]
            else:
                pr = next((p for p in prs if p.number == int(num)), None)
                if pr is None:
                    return self._api_error(404)
                if tail == "files":
                    doc = [
                        {
                            "filename": f["path"],
                            "status": f["status"],
                            "additions": f["additions"],
                            "deletions": f["deletions"],
                            "changes": int(str(f["additions"])) + int(str(f["deletions"])),
                        }
                        for f in self._files(pr)
                    ]
                elif tail == "commits":
                    doc = [self._rest_commit(c) for c in self._commits(pr)]
                elif tail is None:
                    doc = self._rest_pr(pr)
                else:
                    raise Gap(f"gh api {method} {shape}")
        else:
            if num is None:
                runs = self._runs(repo)
                doc = {
                    "total_count": len(runs),
                    "workflow_runs": [self._rest_run(p, c) for p, c in runs],
                }
            else:
                found = next(((p, c) for p, c in self._runs(repo) if str(c.run_id) == num), None)
                if found is None:
                    return self._api_error(404)
                if tail == "jobs":
                    job = self._job(*found)
                    doc = {
                        "total_count": 1,
                        "jobs": [
                            {
                                "id": job["databaseId"],
                                "run_id": job["databaseId"],
                                "name": job["name"],
                                "status": job["status"],
                                "conclusion": job["conclusion"] or None,
                                "html_url": job["url"],
                                "steps": job["steps"],
                            }
                        ],
                    }
                elif tail is None:
                    doc = self._rest_run(*found)
                else:
                    raise Gap(f"gh api {method} {shape}")
        return self._emit(opts, doc, indent=None)

    def _api_error(self, status: int) -> Result:
        body = {
            "message": "Not Found",
            "documentation_url": "https://docs.github.com/rest",
            "status": str(status),
        }
        return Result(
            json.dumps(body, separators=(",", ":")) + "\n",
            f"gh: Not Found (HTTP {status})\n",
            exit=1,
        )

    @staticmethod
    def _rest_commit(c: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """One entry of REST `pulls/N/commits` from a `_commits` record."""
        authors = c["authors"]
        a = authors[0] if isinstance(authors, list) and isinstance(authors[0], dict) else {}
        return {
            "sha": c["oid"],
            "commit": {
                "message": (
                    str(c["messageHeadline"])
                    + ("\n\n" + str(c["messageBody"]) if c["messageBody"] else "")
                ),
                "author": {"name": a["name"], "email": a["email"], "date": c["authoredDate"]},
            },
        }

    def _rest_pr(self, pr: PullRequest) -> dict[str, JsonValue]:
        files = self._files(pr)
        state = self._merge_state(pr)
        return {
            "url": f"https://api.{self.host}/repos/{pr.repo}/pulls/{pr.number}",
            "html_url": self._url(pr),
            "number": pr.number,
            "state": "open" if pr.state == "OPEN" else "closed",
            "title": pr.title,
            "body": pr.body,
            "user": {"login": pr.author},
            "draft": pr.draft,
            "merged": pr.state == "MERGED",
            "merged_at": pr.merged_at or None,
            "closed_at": pr.merged_at or None,
            "merge_commit_sha": pr.merge_commit or None,
            "created_at": pr.created,
            "updated_at": pr.merged_at or pr.created,
            "mergeable": (state != "DIRTY") if pr.state == "OPEN" else None,
            "mergeable_state": state.lower(),
            "head": {"ref": pr.head, "sha": pr.head_sha, "repo": {"full_name": pr.repo}},
            "base": {
                "ref": pr.base,
                "sha": self.ref(pr.repo, pr.base),
                "repo": {"full_name": pr.repo},
            },
            "requested_reviewers": [],
            "labels": [],
            "commits": len(self._commits(pr)),
            "changed_files": len(files),
            "additions": sum(int(str(f["additions"])) for f in files),
            "deletions": sum(int(str(f["deletions"])) for f in files),
        }

    def _rest_run(self, pr: PullRequest, c: Check) -> dict[str, JsonValue]:
        return {
            "id": c.run_id,
            "name": c.workflow,
            "head_branch": pr.head,
            "head_sha": pr.head_sha,
            "event": "pull_request",
            "status": "in_progress" if c.status == "pending" else "completed",
            "conclusion": {"pass": "success", "fail": "failure", "pending": None}[c.status],
            "run_attempt": c.attempt,
            "html_url": f"https://{self.host}/{pr.repo}/actions/runs/{c.run_id}",
            "created_at": pr.created,
        }


def _api_opts(args: list[str]) -> tuple[dict[str, str], list[str]]:
    return getopt(
        "gh api",
        args,
        short="i",
        short_valued="XfFHqpt",
        long=frozenset({"include", "paginate", "silent", "slurp"}),
        long_valued=frozenset(
            {
                "method",
                "raw-field",
                "field",
                "header",
                "jq",
                "preview",
                "template",
                "input",
                "hostname",
            }
        ),
        multi=frozenset({"f", "F", "raw-field", "field", "H", "header"}),
    )


def run_jq(expr: str, doc: JsonValue) -> Result:
    """gh --jq: the real jq binary (raw strings, compact JSON) with an empty environment, no
    modules and capped memory/CPU; mini_jq where jq is missing."""
    if re.match(r"\s*(#[^\n]*\n\s*)*(import|include|module)\b", expr):
        raise Gap("gh --jq modules")
    if JQ is None:
        return Result("".join(_jq_text(v) + "\n" for v in mini_jq(expr, doc)), probe=True)
    limits = [PRLIMIT, "--as=536870912", "--cpu=5"] if PRLIMIT else []
    try:
        res = subprocess.run(
            [*limits, JQ, "-r", "-c", expr],
            input=json.dumps(doc),
            capture_output=True,
            text=True,
            env={},
            cwd="/",
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return Result(stderr="jq: timed out\n", exit=1)
    if res.returncode != 0:
        return Result(stderr=res.stderr or "jq: error\n", exit=1)
    return Result(res.stdout, probe=True)


def _pr_doc(p: PullRequest) -> dict[str, object]:
    doc = asdict(p)
    for c in doc["checks"]:
        c.pop("rerun", None)
    return doc


_JQ_STEP = re.compile(r"\.([A-Za-z_]\w*)|\[(\d*)\]|\.")


def mini_jq(expr: str, doc: JsonValue) -> list[JsonValue]:
    """The jq subset gh users reach for: paths (.a.b, .[], .[0].a) joined by pipes, plus `length`."""
    values = [doc]
    for stage in (s.strip() for s in expr.split("|")):
        if stage == "length":
            values = [len(v) if isinstance(v, (list, dict, str)) else 0 for v in values]
            continue
        pos = 0
        steps: list[tuple[str | None, str | None]] = []
        while pos < len(stage):
            m = _JQ_STEP.match(stage, pos)
            if m is None or m.end() == pos:
                raise Gap(f"gh --jq {expr}")
            steps.append((m.group(1), m.group(2)))
            pos = m.end()
        for key, idx in steps:
            nxt: list[JsonValue] = []
            for v in values:
                if key is not None:
                    nxt.append(v.get(key) if isinstance(v, dict) else None)
                elif idx == "":
                    nxt.extend(
                        v
                        if isinstance(v, list)
                        else list(v.values())
                        if isinstance(v, dict)
                        else []
                    )
                elif idx is not None:
                    nxt.append(v[int(idx)] if isinstance(v, list) and int(idx) < len(v) else None)
                else:
                    nxt.append(v)
            values = nxt
    return values


def _jq_text(v: JsonValue) -> str:
    return v if isinstance(v, str) else json.dumps(v, separators=(",", ":"))
