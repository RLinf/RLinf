#!/usr/bin/env python3

# Copyright 2025 The RLinf Authors.
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

"""Drive the embodied install + e2e pipeline for one model/env.

This is the harness behind the `test-install` skill. It is the single source of
truth for "how does CI install and test <model>/<env>" — it reads that straight
out of .github/workflows/embodied-e2e-tests.yml instead of hard-coding it, so it
never drifts from CI.

Subcommands (run `driver.py <cmd> -h` for flags):

  list                       Table of every (model, env, job, configs) in CI.
  resolve  <model> [env]     Show the CI install command + test configs.
  check-paths <config>       Check the absolute model/ckpt paths in an e2e
                             config yaml actually exist on disk.
  check-all                  Run check-paths over every e2e config; report gaps.
  install  <model> [env]     Run CI's install command into --venv <path>.
  test     <config>          Run CI's e2e test for <config> with --venv <path>.
  run      <model> [env]     install -> check-paths -> test (the full pipeline).

Paths printed/used are relative to the repo root (the dir that contains
requirements/install.sh). Run this from there.
"""

import argparse
import os
import re
import subprocess
import sys

try:
    import yaml
except ImportError:
    # System python on the embodied runner has PyYAML; if not, bootstrap via uv.
    os.execvp("uv", ["uv", "run", "--with", "pyyaml", "python3", *sys.argv])

HERE = os.path.dirname(os.path.abspath(__file__))


def repo_root():
    """Walk up from CWD to the dir holding requirements/install.sh."""
    d = os.getcwd()
    while True:
        if os.path.isfile(os.path.join(d, "requirements", "install.sh")):
            return d
        if d == "/":
            sys.exit(
                "error: run this from inside the RLinf repo (no "
                "requirements/install.sh found above CWD)."
            )
        d = os.path.dirname(d)


ROOT = repo_root()
WF = os.path.join(ROOT, ".github", "workflows", "embodied-e2e-tests.yml")
E2E_DIR = os.path.join(ROOT, "tests", "e2e_tests", "embodied")

# Config keys whose value is an *input* artifact that must already exist.
REQUIRED_KEY = re.compile(
    r"(model_path|ckpt|checkpoint|pretrained|tokenizer|processor|lora|"
    r"weights?|_path$|_dir$)",
    re.I,
)
# Keys that are outputs (created by the run) — existence is not required.
OUTPUT_KEY = re.compile(r"(log_path|output|save|video|results?)", re.I)

RUN_RE = re.compile(r"tests/e2e_tests/embodied/(run\w*)\.sh\s+(\S+)(.*)")


# --------------------------------------------------------------------------- #
# Parse the CI workflow into a list of jobs.
# --------------------------------------------------------------------------- #
class Job:
    def __init__(self, name):
        self.name = name
        self.model = None
        self.env = None
        self.platform = None
        self.install_run = None  # full shell of the install step
        self.tests = []  # list of dicts: runner, config, args, run

    def __repr__(self):
        return f"<Job {self.name} model={self.model} env={self.env}>"


def load_jobs():
    with open(WF) as f:
        data = yaml.safe_load(f)
    jobs = []
    for name, spec in (data.get("jobs") or {}).items():
        job = Job(name)
        for step in spec.get("steps", []):
            run = step.get("run") or ""
            if "requirements/install.sh" in run:
                job.install_run = run
                m = re.search(r"--model\s+(\S+)", run)
                e = re.search(r"--env\s+(\S+)", run)
                p = re.search(r"--platform\s+(\S+)", run)
                job.model = m.group(1) if m else None
                job.env = e.group(1).rstrip(";") if e else None
                job.platform = p.group(1) if p else None
            if "tests/e2e_tests/embodied/run" in run:
                for line in run.splitlines():
                    rm = RUN_RE.search(line)
                    if rm:
                        job.tests.append(
                            {
                                "runner": rm.group(1),
                                "config": rm.group(2),
                                "args": rm.group(3).strip(),
                                "run": run,
                            }
                        )
        if job.install_run:
            jobs.append(job)
    return jobs


def match_jobs(jobs, model, env):
    """Match CI jobs by (model, env).

    With both args, require an exact match. With only one token (env is None),
    match it against either the model or the env — so `d4rl` finds the env-only
    d4rl job and `gr00t_n1d6` finds the gr00t job.
    """
    out = []
    for j in jobs:
        jm, je = j.model or "", j.env or ""
        if env is None:
            if model in (jm, je):
                out.append(j)
        elif jm == model and je == env:
            out.append(j)
    return out


# --------------------------------------------------------------------------- #
# Path checking for an e2e config yaml.
# --------------------------------------------------------------------------- #
def scan_config_paths(config):
    """Return list of (key, value, kind, exists) for absolute paths in config."""
    path = os.path.join(E2E_DIR, config + ".yaml")
    if not os.path.isfile(path):
        sys.exit(f"error: no e2e config at {os.path.relpath(path, ROOT)}")
    found = []
    seen = set()
    # Line-based scan: robust to Hydra structure, catches `key: /abs/path`.
    line_re = re.compile(r'^\s*([\w.]+)\s*:\s*["\']?(/[^"\'\s#]+)')
    with open(path) as f:
        for line in f:
            m = line_re.search(line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).rstrip("/")
            if (key, val) in seen:
                continue
            seen.add((key, val))
            if OUTPUT_KEY.search(key):
                kind = "output"
            elif REQUIRED_KEY.search(key):
                kind = "required"
            else:
                kind = "info"
            found.append((key, val, kind, os.path.exists(m.group(2).rstrip("/"))))
    return path, found


def cmd_check_paths(args):
    path, found = scan_config_paths(args.config)
    print(f"# {os.path.relpath(path, ROOT)}")
    missing_required = 0
    if not found:
        print("  (no absolute paths referenced)")
    for key, val, kind, exists in found:
        mark = "OK " if exists else "MISS"
        tag = {"required": "[model/input]", "output": "[output dir]", "info": "[path]"}[
            kind
        ]
        if not exists and kind == "required":
            missing_required += 1
        print(f"  {mark}  {tag:14} {key} = {val}")
    if missing_required:
        print(
            f"\n>>> {missing_required} required input path(s) MISSING — the "
            f"e2e test will fail before training."
        )
        return 1
    print("\nAll required input paths present.")
    return 0


def cmd_check_all(_args):
    rc = 0
    configs = sorted(f[:-5] for f in os.listdir(E2E_DIR) if f.endswith(".yaml"))
    for cfg in configs:
        _, found = scan_config_paths(cfg)
        miss = [(k, v) for k, v, kind, ex in found if kind == "required" and not ex]
        if miss:
            rc = 1
            print(f"MISSING  {cfg}")
            for k, v in miss:
                print(f"           {k} = {v}")
    if rc == 0:
        print(f"All required input paths present across {len(configs)} configs.")
    return rc


# --------------------------------------------------------------------------- #
# resolve / list
# --------------------------------------------------------------------------- #
def cmd_list(_args):
    jobs = load_jobs()
    print(f"{'MODEL':14} {'ENV':18} {'JOB':44} CONFIGS")
    for j in sorted(jobs, key=lambda x: (x.model or "", x.env or "")):
        cfgs = ",".join(t["config"] for t in j.tests) or "(none)"
        print(f"{(j.model or '-'):14} {(j.env or '-'):18} {j.name:44} {cfgs}")


def cmd_resolve(args):
    jobs = load_jobs()
    hits = match_jobs(jobs, args.model, args.env)
    if not hits:
        print(f"No CI job for model={args.model} env={args.env or '*'}.")
        print("Run `driver.py list` to see available combinations.")
        return 1
    for j in hits:
        print(
            f"=== job: {j.name}  (model={j.model or '-'} env={j.env or '-'}"
            f"{' platform=' + j.platform if j.platform else ''}) ==="
        )
        print("--- install (CI step, verbatim) ---")
        print(j.install_run.rstrip())
        for t in j.tests:
            print(f"--- test: {t['runner']}.sh {t['config']} {t['args']} ---")
        print()
    return 0


# --------------------------------------------------------------------------- #
# install / test / run  — execute CI's own shell with the venv path injected.
# --------------------------------------------------------------------------- #
def _run_shell(script, dry):
    print("+ " + script.strip().replace("\n", "\n+ "), file=sys.stderr)
    if dry:
        print("(dry-run: not executed)", file=sys.stderr)
        return 0
    return subprocess.call(["bash", "-c", script], cwd=ROOT)


def _install_script(job, venv, mirror=True):
    run = job.install_run

    # Inject --venv <path> and (by default) --use-mirror into the install.sh
    # invocation. Mirrors are on unless the caller opts out with --no-mirror.
    def add_flags(m):
        line = m.group(1)
        extra = f" --venv {venv}"
        if mirror and "--use-mirror" not in line:
            extra += " --use-mirror"
        return line + extra

    return re.sub(r"(bash\s+requirements/install\.sh[^\n]*)", add_flags, run, count=1)


def cmd_install(args):
    jobs = load_jobs()
    hits = match_jobs(jobs, args.model, args.env)
    if not hits:
        return cmd_resolve(args)
    job = hits[0]
    print(
        f"# installing {job.model or '-'}/{job.env or '-'} into {args.venv} "
        f"(from CI job {job.name})",
        file=sys.stderr,
    )
    return _run_shell(_install_script(job, args.venv, not args.no_mirror), args.dry_run)


def _test_script(test, venv):
    # CI's test step, verbatim, but pointed at the chosen venv.
    return test["run"].replace(".venv/bin/activate", f"{venv}/bin/activate")


def cmd_test(args):
    jobs = load_jobs()
    test = None
    for j in jobs:
        for t in j.tests:
            if t["config"] == args.config:
                test = t
                break
        if test:
            break
    if not test:
        # Config exists but isn't wired into CI: run it directly, ask the user
        # to confirm the runner.
        cfg_file = os.path.join(E2E_DIR, args.config + ".yaml")
        if not os.path.isfile(cfg_file):
            sys.exit(f"error: no e2e config '{args.config}' and not in CI.")
        runner = args.runner or "run"
        print(
            f"# '{args.config}' is not referenced by any CI job.\n"
            f"# Falling back to tests/e2e_tests/embodied/{runner}.sh — pass "
            f"--runner run|run_async|run_offline if that's wrong.",
            file=sys.stderr,
        )
        test = {
            "run": f"unset PYTHONPATH\nexport REPO_PATH=$(pwd)\n"
            f"source .venv/bin/activate\n"
            f"bash tests/e2e_tests/embodied/{runner}.sh {args.config}"
        }
    return _run_shell(_test_script(test, args.venv), args.dry_run)


def cmd_run(args):
    jobs = load_jobs()
    hits = match_jobs(jobs, args.model, args.env)
    if not hits:
        return cmd_resolve(args)
    job = hits[0]
    print("# === install ===", file=sys.stderr)
    rc = _run_shell(_install_script(job, args.venv, not args.no_mirror), args.dry_run)
    if rc:
        print("install failed; stopping.", file=sys.stderr)
        return rc
    # A (model, env) pair can map to several CI jobs — e.g. openvla/maniskill_libero
    # has separate SAC, async-SAC, and PPO jobs. Run the tests from every matching
    # job on the same platform we installed, so coverage matches CI instead of
    # silently stopping at the first job.
    tests = [t for h in hits if h.platform == job.platform for t in h.tests]
    if not tests:
        print(
            "# install OK; no CI job for this model/env has an e2e test — "
            "nothing to run. Ask the user what to run.",
            file=sys.stderr,
        )
        return 0
    rc = 0
    for t in tests:
        print(f"# === check-paths {t['config']} ===", file=sys.stderr)
        pr = scan_config_paths(t["config"])
        miss = [(k, v) for k, v, kind, ex in pr[1] if kind == "required" and not ex]
        for k, v in miss:
            print(f"  MISS [model/input] {k} = {v}", file=sys.stderr)
        if miss and not args.skip_missing:
            print(
                f"  required path(s) missing for {t['config']}; skipping its "
                f"test (pass --skip-missing to try anyway).",
                file=sys.stderr,
            )
            rc = 1
            continue
        print(f"# === test {t['config']} ===", file=sys.stderr)
        rc |= _run_shell(_test_script(t, args.venv), args.dry_run)
    return rc


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(func=cmd_list)

    sp = sub.add_parser("resolve")
    sp.add_argument("model")
    sp.add_argument("env", nargs="?")
    sp.set_defaults(func=cmd_resolve)

    sp = sub.add_parser("check-paths")
    sp.add_argument("config")
    sp.set_defaults(func=cmd_check_paths)

    sub.add_parser("check-all").set_defaults(func=cmd_check_all)

    for name, fn in (("install", cmd_install), ("run", cmd_run)):
        sp = sub.add_parser(name)
        sp.add_argument("model")
        sp.add_argument("env", nargs="?")
        sp.add_argument(
            "--venv",
            required=True,
            help="venv path to create/use (passed to install.sh --venv)",
        )
        sp.add_argument(
            "--no-mirror",
            action="store_true",
            help="don't add --use-mirror (mirrors are on by default)",
        )
        sp.add_argument("--dry-run", action="store_true")
        if name == "run":
            sp.add_argument(
                "--skip-missing",
                action="store_true",
                help="run a test even if required model paths are missing",
            )
        sp.set_defaults(func=fn)

    sp = sub.add_parser("test")
    sp.add_argument("config")
    sp.add_argument("--venv", required=True)
    sp.add_argument(
        "--runner",
        choices=["run", "run_async", "run_offline"],
        help="runner script if config is not wired into CI",
    )
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_test)

    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
