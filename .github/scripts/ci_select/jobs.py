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

"""Map every CI job to the files and lines it depends on.

The map is derived from what each job runs: the workflow steps, the
``install.sh`` arguments, the ``run*.sh`` script and its config, the Hydra
defaults that config composes, the Python import closure of the entry point
pruned by that config, and the Dockerfile stages a docker job builds.
"""

import posixpath
import re
import shlex
from dataclasses import dataclass, field

import bashscan
import hydra_cfg
import yaml
from pyscan import PyIndex
from snapshot import Snapshot

WORKFLOW_DIR = ".github/workflows"
CALLER = f"{WORKFLOW_DIR}/ci-tests.yml"
INSTALL = "requirements/install.sh"

# Scripts that walk a directory tree instead of importing from it.
TREE_SCANNERS = {
    "tests/unit_tests/check_missing_init.py": "rlinf/",
    "tests/unit_tests/check_import_rlinf_package.py": "rlinf/",
}

# Values a runner's hardware makes true without any config saying so.
RUNNER_VALUES = {
    "reason": {"nvidia", "cuda", "gpu"},
    "embodied": {"nvidia", "cuda", "gpu"},
    "rocm": {"amd", "rocm", "hip", "gpu"},
    "ascend": {"ascend", "npu"},
    "musa": {"musa", "mthreads", "gpu"},
}


@dataclass
class Job:
    name: str
    workflow: str
    runs_on: str
    files: set[str] = field(default_factory=set)
    lines: dict[str, set[int]] = field(default_factory=dict)
    # Python reachable with no config pruning, for judging what pruning drops.
    files_unpruned: set[str] = field(default_factory=set)
    # Dirs whose file list (not contents) the job reads, e.g. through ``ls``,
    # with the name pattern the listing is filtered by (None: any name).
    listings: dict[str, re.Pattern | None] = field(default_factory=dict)
    # Values each bash script was called with (install.sh flags, run.sh args).
    bash_values: dict[str, set[str]] = field(default_factory=dict)
    # Config values of each Python run; None for a run nothing prunes.
    py_values: list[frozenset[str] | None] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def gpu(self) -> bool:
        return not self.runs_on.startswith("ubuntu")

    def add_lines(self, path: str, lines) -> None:
        self.lines.setdefault(path, set()).update(lines)


_VAR = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}|\$(\w+|\d)")


def expand(text: str, env: dict[str, set[str]], limit: int = 32) -> list[str]:
    """All substitutions of known variables; unknown ones are left as is."""
    text = text.replace("$(pwd)", "").replace("${PWD}", "").replace("$PWD", "")
    # Each variable is substituted once per path, so ``X=$X:...`` terminates.
    out, todo = [], [(text, frozenset())]
    while todo and len(out) < limit:
        t, done = todo.pop()
        for m in _VAR.finditer(t):
            name = m.group(1) or m.group(3)
            if name in done:
                vals = {""}
            elif name in env:
                vals = env[name]
            elif m.group(2) is not None:
                vals = {m.group(2)}
            else:
                continue
            todo += [
                (t[: m.start()] + v + t[m.end() :], done | {name})
                for v in sorted(vals)[:limit]
            ]
            break
        else:
            out.append(t)
    return out


def _listing_filter(cmd: str) -> re.Pattern | None:
    """The name pattern of ``ls dir | sed -n 's/^re$/.../p'`` (or grep)."""
    m = re.search(
        r"\|\s*(?:sed\s+-n\w*\s+[\"']s/|grep\s+-\w*E?\w*\s+[\"'])(\^[^/\"']*)", cmd
    )
    if not m:
        return None
    raw = m.group(1).replace("\\$", "\x01").replace("\\.", "\x00")
    rx = re.sub(r"\$\{?\w+\}?", ".*", raw).replace("\x00", "\\.").replace("\x01", "$")
    try:
        return re.compile(rx)
    except re.error:
        return None


def _split_commands(text: str) -> list[str]:
    out = []
    for inner in re.findall(r"\$\(([^()]*)\)", text):
        out.append(inner)
    for part in re.split(r"&&|\|\||;|\|", text):
        part = part.strip()
        if part:
            out.append(part)
    return out


def _tokens(cmd: str) -> list[str]:
    try:
        return shlex.split(cmd, comments=True)
    except ValueError:
        return cmd.split()


class JobMapper:
    def __init__(self, snap: Snapshot):
        self.snap = snap
        self.py = PyIndex(snap)
        self.vocab = hydra_cfg.vocabulary(snap)
        self._bash: dict[str, bashscan.BashScript] = {}

    # -- helpers ------------------------------------------------------------

    def bash(self, path: str) -> bashscan.BashScript:
        if path not in self._bash:
            self._bash[path] = bashscan.parse(self.snap.read(path) or "")
        return self._bash[path]

    def norm(self, p: str) -> str | None:
        p = p.strip().strip("'\"")
        p = posixpath.normpath(p.lstrip("/")) if p else p
        if p.startswith("./"):
            p = p[2:]
        if self.snap.exists(p) or p in self.snap.dirs:
            return p
        # Workflows that check the repo out under ``rlinf/``.
        if p.startswith("rlinf/") and (
            self.snap.exists(p[6:]) or p[6:] in self.snap.dirs
        ):
            return p[6:]
        return None

    def _pythonpath(
        self, commands: list[str], env: dict[str, set[str]]
    ) -> tuple[str, ...]:
        out = []
        for c in commands:
            m = re.search(r"PYTHONPATH=(\S+)", c)
            if not m:
                continue
            for value in expand(m.group(1), env):
                for part in value.split(":"):
                    d = self.norm(part)
                    if d is not None and d in self.snap.dirs and d not in out:
                        out.append(d)
        return tuple(out)

    # -- python -------------------------------------------------------------

    def python_call(
        self,
        job: Job,
        entry: str,
        config_path: str | None,
        config_name: str | None,
        overrides: list[str],
        extra_paths: tuple[str, ...],
    ) -> None:
        if entry in TREE_SCANNERS:
            job.py_values.append(None)
            job.files.update(self.snap.under(TREE_SCANNERS[entry]))
            job.files_unpruned.update(self.snap.under(TREE_SCANNERS[entry]))
        values = None
        if config_name:
            cdir = self.norm(config_path or posixpath.dirname(entry)) or ""
            files, vals = hydra_cfg.compose(self.snap, cdir, config_name, overrides)
            if not files:
                job.notes.append(f"config not found: {cdir}/{config_name}")
            job.files |= files
            values = frozenset(vals | RUNNER_VALUES.get(job.runs_on, set()))
        job.py_values.append(values)
        roots = [entry]
        for d in extra_paths:
            if self.snap.exists(f"{d}/sitecustomize.py"):
                roots.append(f"{d}/sitecustomize.py")
        py, data = self.py.closure(roots, values, self.vocab, extra_paths)
        job.files |= py | data
        py_all, data_all = self.py.closure(roots, None, self.vocab, extra_paths)
        job.files_unpruned |= py_all | data_all

    def _python_from_tokens(
        self, job: Job, toks: list[str], rest: list[str], extra_paths: tuple[str, ...]
    ) -> None:
        i = 0
        entry = None
        while i < len(toks):
            t = toks[i]
            if t == "-m" and i + 1 < len(toks):
                files = self.py.resolve(toks[i + 1], ("",) + extra_paths)
                entry = files[0] if files else None
                i += 2
                break
            if t.endswith(".py"):
                entry = self.norm(t)
                i += 1
                break
            i += 1
        if entry is None:
            return
        cpath = cname = None
        overrides: list[str] = []
        args = toks[i:]
        j = 0
        while j < len(args):
            a = args[j]
            if a in ("--config-path", "-cp", "--config-name", "-cn") and j + 1 < len(
                args
            ):
                if a in ("--config-path", "-cp"):
                    cpath = args[j + 1]
                else:
                    cname = args[j + 1]
                j += 2
                continue
            if a.startswith("--config-path="):
                cpath = a.split("=", 1)[1]
            elif a.startswith("--config-name="):
                cname = a.split("=", 1)[1]
            elif a in ("$@", '"$@"'):
                overrides += [r for r in rest if "=" in r]
            elif "=" in a and not a.startswith("-"):
                overrides.append(a)
            j += 1
        if cname and "$" in cname:
            cname = None
        self.python_call(job, entry, cpath, cname, overrides, extra_paths)

    # -- bash ---------------------------------------------------------------

    def install_call(self, job: Job, argv: list[str]) -> None:
        sc = self.bash(INSTALL)
        values = bashscan.args_to_values(sc, argv)
        job.bash_values.setdefault(INSTALL, set()).update(
            v for v in values.values() if v
        )
        run = bashscan.analyze(sc, values)
        job.add_lines(INSTALL, run.used)
        job.files.add("pyproject.toml")
        self._script_refs(job, run.commands, {"SCRIPT_DIR": {"requirements"}})

    def _script_refs(
        self, job: Job, commands: list[str], env: dict[str, set[str]]
    ) -> None:
        """Files a script names through ``$SCRIPT_DIR/...`` and the like."""
        for c in commands:
            for m in re.finditer(r"\$\{?SCRIPT_DIR\}?/([\w./${}*-]*)", c):
                rel = re.sub(r"\$\{?\w+\}?", "*", m.group(1)).rstrip("/") or "*"
                pattern = f"requirements/{rel}"
                if re.match(r"^\s*(ls|find)\s", c) and "*" not in pattern:
                    job.listings[pattern] = _listing_filter(c)
                    continue
                if "*" in pattern:
                    hits = self.snap.glob(pattern) or self.snap.glob(pattern + "*")
                    if posixpath.basename(pattern).startswith("*"):
                        # "$1_$2_${cuda}.txt": the leading variable is an
                        # argument such as the engine, so the file names it.
                        values = job.bash_values.get(INSTALL, set())
                        hits = [
                            h
                            for h in hits
                            if any(v and v in posixpath.basename(h) for v in values)
                        ]
                elif self.snap.exists(pattern):
                    hits = [pattern]
                else:
                    hits = self.snap.under(pattern)
                job.files.update(hits)

    def run_script(
        self,
        job: Job,
        path: str,
        argv: list[str],
        extra_paths: tuple[str, ...],
        depth: int = 0,
    ) -> None:
        sc = self.bash(path)
        values = bashscan.positional_values(sc, argv)
        job.bash_values.setdefault(path, set()).update(v for v in values.values() if v)
        run = bashscan.analyze(sc, values)
        job.add_lines(path, run.used)
        env = {k: {v} for k, v in values.items()}
        env["REPO_PATH"] = {""}
        for var, vals in run.assigned.items():
            if var not in env:
                env[var] = vals
        # Resolve assignments that refer to other assignments.
        for _ in range(3):
            for var, vals in list(env.items()):
                env[var] = {
                    e
                    for v in vals
                    for e in expand(v, {k: x for k, x in env.items() if k != var})
                }
        paths = extra_paths + self._pythonpath(run.commands, env)
        self.commands(job, run.commands, env, argv[1:], paths, depth + 1)

    def commands(self, job, commands, env, rest, extra_paths, depth=0) -> None:
        if depth > 4:
            return
        for logical in commands:
            for cmd in _split_commands(logical):
                for text in expand(cmd, env):
                    self._command(job, _tokens(text), env, rest, extra_paths, depth)

    def _command(self, job, toks, env, rest, extra_paths, depth) -> None:
        while toks and re.match(r"^\w+=", toks[0]):
            toks = toks[1:]
        if toks and toks[0] in ("timeout", "env", "time", "exec", "nohup"):
            toks = [
                t for t in toks[1:] if not re.match(r"^(-\S+|\d+[smh]?|\w+=.*)$", t)
            ]
        if not toks:
            return
        head = posixpath.basename(toks[0])
        if head in ("bash", "sh", "source", ".") or toks[0].endswith(".sh"):
            script_toks = toks[1:] if head in ("bash", "sh", "source", ".") else toks
            # Options before the script name belong to bash itself.
            while (
                head in ("bash", "sh")
                and script_toks
                and script_toks[0].startswith("-")
            ):
                script_toks = script_toks[1:]
            if not script_toks:
                return
            p = self.norm(script_toks[0])
            if p == INSTALL:
                self.install_call(job, script_toks[1:])
            elif p and p.endswith(".sh"):
                self.run_script(job, p, script_toks[1:], extra_paths, depth)
        elif re.match(r"^python[\d.]*$", head) or head in (
            "torchrun",
            "accelerate",
            "deepspeed",
        ):
            self._python_from_tokens(job, toks[1:], rest, extra_paths)
        elif head == "pytest" or (
            head.startswith("python") and toks[1:3] == ["-m", "pytest"]
        ):
            self._pytest(job, toks[1:], extra_paths)
        elif head == "uv":
            if "-r" in toks:
                p = self.norm(toks[toks.index("-r") + 1])
                if p:
                    job.files.add(p)
            if any(t in (".", "-e") for t in toks) or toks[1:2] in (
                ["sync"],
                ["build"],
            ):
                job.files.add("pyproject.toml")

    def _pytest(self, job: Job, args: list[str], extra_paths: tuple[str, ...]) -> None:
        doctest = "--doctest-modules" in args
        targets, skip = [], False
        for a in args:
            if skip:
                skip = False
                continue
            if a in (
                "-k",
                "-m",
                "-c",
                "-p",
                "-o",
                "--rootdir",
                "--deselect",
                "--ignore",
            ):
                skip = True
                continue
            if a.startswith("-"):
                continue
            p = self.norm(a.split("::")[0])
            if p:
                targets.append(p)
        roots: list[str] = []
        for t in targets:
            if t in self.snap.dirs:
                pyf = [f for f in self.snap.under(t) if f.endswith(".py")]
                roots += (
                    pyf
                    if doctest
                    else [f for f in pyf if posixpath.basename(f).startswith("test_")]
                )
            else:
                roots.append(t)
        conftests = set()
        for r in roots:
            d = posixpath.dirname(r)
            while True:
                for name in ("conftest.py", "pytest.ini"):
                    if self.snap.exists(f"{d}/{name}" if d else name):
                        conftests.add(f"{d}/{name}" if d else name)
                if not d:
                    break
                d = posixpath.dirname(d)
        job.files |= {c for c in conftests if c.endswith(".ini")}
        py_roots = roots + sorted(c for c in conftests if c.endswith(".py"))
        # Tests build their own configs, so nothing is pruned.
        job.py_values.append(None)
        py, data = self.py.closure(py_roots, None, self.vocab, extra_paths)
        job.files |= py | data
        job.files_unpruned |= py | data

    # -- docker -------------------------------------------------------------

    def docker_build(
        self, job: Job, dockerfile: str, build_args: dict[str, str]
    ) -> None:
        text = self.snap.read(dockerfile) or ""
        lines = text.split("\n")
        args: dict[str, str] = {}
        stages: list[tuple[str, str, int, int]] = []
        first_from = None
        for i, line in enumerate(lines):
            m = re.match(r"^\s*FROM\s+(\S+)(?:\s+AS\s+(\S+))?", line, re.I)
            if m:
                if first_from is None:
                    first_from = i
                if stages:
                    stages[-1] = (*stages[-1][:3], i - 1)
                stages.append((m.group(2) or "", m.group(1), i, len(lines) - 1))
            am = re.match(r"^\s*ARG\s+(\w+)(?:=(.*))?$", line)
            if am and am.group(2) is not None and am.group(1) not in args:
                args[am.group(1)] = am.group(2).strip()
        env = {k: {v} for k, v in args.items()}
        env.update({k: {v} for k, v in build_args.items()})
        env["INSTALL_MIRROR_OPTION"] = {""}
        env["RLINF_PLATFORM"] = env.get("PLATFORM", {"nvidia"})
        by_name = {s[0]: s for s in stages if s[0]}
        used = set(range(1, (first_from or 0) + 1))
        # The last stage is the build target; follow its base chain.
        todo = [stages[-1]] if stages else []
        seen = set()
        while todo:
            name, base, start, end = todo.pop()
            if (name, start) in seen:
                continue
            seen.add((name, start))
            used.update(range(start + 1, end + 2))
            for b in expand(base, env):
                if b in by_name:
                    todo.append(by_name[b])
        job.add_lines(dockerfile, used)
        # Logical RUN lines inside the used stages.
        logical, buf, buf_start = [], "", None
        for i, line in enumerate(lines):
            if buf_start is None:
                buf_start = i
            buf += line.rstrip("\\") + " "
            if not line.rstrip().endswith("\\"):
                logical.append((buf_start, buf))
                buf, buf_start = "", None
        for start, text in logical:
            if start + 1 in used and re.match(r"^\s*RUN\s", text):
                self.commands(job, [re.sub(r"^\s*RUN\s+", "", text)], env, [], ())
            cm = re.match(r"^\s*COPY\s+(.*)$", text)
            if start + 1 in used and cm:
                # A copied directory matters through the RUN steps that use
                # it, which are traced; a copied file is read as is.
                for src in _tokens(cm.group(1))[:-1]:
                    p = self.norm(src)
                    if p and not src.startswith("--") and self.snap.exists(p):
                        job.files.add(p)

    # -- workflows ----------------------------------------------------------

    def workflow_jobs(self, path: str) -> list[Job]:
        text = self.snap.read(path) or ""
        data = yaml.safe_load(text) or {}
        node = yaml.compose(text)
        spans: dict[str, tuple[int, int]] = {}
        for k, v in node.value:
            if k.value == "jobs":
                for jk, jv in v.value:
                    end = (
                        jv.end_mark.line if jv.end_mark.column else jv.end_mark.line - 1
                    )
                    spans[jk.value] = (jk.start_mark.line + 1, end + 1)
        n_lines = text.count("\n") + 1
        in_jobs = set()
        for s, e in spans.values():
            in_jobs.update(range(s, e + 1))
        header = set(range(1, n_lines + 1)) - in_jobs
        out = []
        stem = posixpath.splitext(posixpath.basename(path))[0]
        for jid, spec in (data.get("jobs") or {}).items():
            runs_on = spec.get("runs-on", "")
            runs_on = (
                runs_on if isinstance(runs_on, str) else ",".join(map(str, runs_on))
            )
            job = Job(f"{stem}/{jid}", path, runs_on)
            s, e = spans.get(jid, (0, -1))
            job.add_lines(path, set(range(s, e + 1)) | header)
            for step in spec.get("steps") or []:
                self._step(job, step)
            job.files_unpruned |= job.files
            out.append(job)
        return out

    def _step(self, job: Job, step: dict) -> None:
        uses = step.get("uses", "") or ""
        with_ = step.get("with") or {}
        if uses.startswith("docker/build-push-action"):
            build_args = {}
            for line in str(with_.get("build-args", "")).splitlines():
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    build_args[k] = v
            dockerfile = self.norm(str(with_.get("file", "docker/Dockerfile")))
            if dockerfile:
                self.docker_build(job, dockerfile, build_args)
            return
        run = step.get("run")
        if not run:
            return
        sc = bashscan.parse(run)
        result = bashscan.analyze(sc, None)
        env: dict[str, set[str]] = dict(result.assigned)
        env["REPO_PATH"] = {""}
        env["PACKAGE_PATH"] = {""}
        for m in re.finditer(r"for\s+(\w+)\s+in\s+([^;\n]+?)\s*;?\s*do\b", run):
            words = []
            for w in m.group(2).split():
                words += self.snap.glob(w.strip("'\"")) if "*" in w else [w]
            env[m.group(1)] = set(words)
        for var, vals in list(env.items()):
            env[var] = {
                e
                for v in vals
                for e in expand(v, {k: x for k, x in env.items() if k != var})
            }
        paths = self._pythonpath(result.commands, env)
        self.commands(job, result.commands, env, [], paths)


def caller_groups(snap: Snapshot) -> dict[str, str]:
    """Called workflow path -> the ci-tests.yml filter that gates it."""
    data = yaml.safe_load(snap.read(CALLER) or "") or {}
    out = {}
    for spec in (data.get("jobs") or {}).values():
        uses = spec.get("uses") or ""
        m = re.search(r"outputs\.(\w+)", str(spec.get("if", "")))
        if uses.startswith("./") and m:
            out[uses[2:]] = m.group(1)
    return out


def build_jobs(snap: Snapshot) -> dict[str, Job]:
    mapper = JobMapper(snap)
    jobs: dict[str, Job] = {}
    for wf in caller_groups(snap):
        if snap.exists(wf):
            for job in mapper.workflow_jobs(wf):
                jobs[job.name] = job
    return jobs
