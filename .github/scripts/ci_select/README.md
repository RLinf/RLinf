# CI job selector

Picks the CI jobs a pull request needs from the files and lines it changes, instead of running every job in each path-filter group of `.github/workflows/ci-tests.yml`.

It is in **shadow mode**: the path filters still decide what runs. On every PR, the `ci-select-shadow` job records the selection, and after the test jobs finish, `ci-select-shadow-report` compares it with the jobs that ran. The comparison is in the job summary and the `ci-select-report` artifact. Neither job gates the merge.

## How a job is mapped

`jobs.py` reads each job's steps and records what the job depends on:

- the `install.sh` lines that run for its arguments, found by evaluating `case` and `[[ "$X" == ... ]]` branches with those arguments, and the requirement files those lines name;
- the `run*.sh` lines that run for its config name, and the Hydra configs that config composes through `defaults`;
- the Python import closure of the entry point (`pyscan.py`). An import inside a branch that compares a config value with a constant (`if env_type == SupportedEnvType.MANISKILL`, a builder registered under a model type) is followed only when the job's config sets that value;
- for a docker job, the Dockerfile stages it builds and the `install.sh` calls in them;
- its own block of the workflow file.

A changed line is checked against the job's lines; a changed Python line counts only if the job's config can reach it. Unit tests build their own configs, so nothing is pruned for them.

## Running it locally

```bash
pip install pyyaml
python .github/scripts/ci_select/select_jobs.py --base origin/main --head HEAD
python .github/scripts/ci_select/replay.py --rev origin/main -n 80 --out replay.json
```

`replay.py` runs merged PRs through the selector and the path filters, and lists the jobs that only config pruning drops.

## Known gaps

The map is static. It does not see imports whose module name is computed at runtime (files with such imports are listed in `PyIndex.unresolved_dynamic`), files opened through built paths, `subprocess` calls, or branches on environment variables. Shadow mode exists to measure what these gaps cost before the selector decides what runs.
