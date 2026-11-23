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

"""Shell hook generation, rc-file installation, and ``rlinf use`` helpers.

Pure Python standard library so this module works identically inside both the
zero-dependency ``rlinf-cli`` package and ``rlinf.cli``.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TextIO

SUPPORTED_SHELLS: tuple[str, ...] = ("bash", "zsh", "fish", "pwsh")
ACTIVATE_FILE_ENV = "RLINF_ACTIVATE_FILE"
HOOK_START = "# >>> rlinf shell hook >>>"
HOOK_END = "# <<< rlinf shell hook <<<"

_POSIX_HOOK = f"""{HOOK_START}
rlinf() {{
  local _rlinf_act_file=""
  local _rlinf_rc=0
  _rlinf_act_file="$(mktemp "${{TMPDIR:-/tmp}}/rlinf-activate.XXXXXX" 2>/dev/null || true)"
  if [ -n "$_rlinf_act_file" ]; then
    {ACTIVATE_FILE_ENV}="$_rlinf_act_file" command rlinf "$@"
    _rlinf_rc=$?
    if [ "$_rlinf_rc" -eq 0 ] && [ -s "$_rlinf_act_file" ]; then
      local _rlinf_venv=""
      IFS= read -r _rlinf_venv < "$_rlinf_act_file"
      if [ -n "$_rlinf_venv" ] && [ -f "$_rlinf_venv/bin/activate" ]; then
        . "$_rlinf_venv/bin/activate"
      fi
    fi
    rm -f "$_rlinf_act_file"
  else
    command rlinf "$@"
    _rlinf_rc=$?
  fi
  return "$_rlinf_rc"
}}
{HOOK_END}"""

_FISH_HOOK = f"""{HOOK_START}
function rlinf --wraps=rlinf --description "RLinf CLI with automatic venv activation"
    set -l _rlinf_tmp (set -q TMPDIR; and echo $TMPDIR; or echo /tmp)
    set -l _rlinf_act_file (mktemp "$_rlinf_tmp/rlinf-activate.XXXXXX" 2>/dev/null)
    if test -n "$_rlinf_act_file"
        env {ACTIVATE_FILE_ENV}="$_rlinf_act_file" command rlinf $argv
        set -l _rlinf_rc $status
        if test $_rlinf_rc -eq 0 -a -s "$_rlinf_act_file"
            set -l _rlinf_venv (string trim -- (cat "$_rlinf_act_file"))
            if test -n "$_rlinf_venv" -a -f "$_rlinf_venv/bin/activate.fish"
                source "$_rlinf_venv/bin/activate.fish"
            end
        end
        rm -f "$_rlinf_act_file"
        return $_rlinf_rc
    else
        command rlinf $argv
    end
end
{HOOK_END}"""

_PWSH_HOOK = f"""{HOOK_START}
function rlinf {{
    $actFile = [System.IO.Path]::GetTempFileName()
    $prev = $env:{ACTIVATE_FILE_ENV}
    $env:{ACTIVATE_FILE_ENV} = $actFile
    $rc = 0
    try {{
        $bin = (Get-Command rlinf -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
        & $bin @args
        $rc = $LASTEXITCODE
        if ($rc -eq 0 -and (Test-Path $actFile)) {{
            $venv = (Get-Content -Path $actFile -Raw).Trim()
            if ($venv) {{
                $script = if (Test-Path (Join-Path $venv "Scripts/Activate.ps1")) {{
                    Join-Path $venv "Scripts/Activate.ps1"
                }} else {{
                    Join-Path $venv "bin/Activate.ps1"
                }}
                if (Test-Path $script) {{
                    . $script
                }}
            }}
        }}
    }} finally {{
        if ($null -eq $prev) {{ Remove-Item Env:{ACTIVATE_FILE_ENV} -ErrorAction SilentlyContinue }} else {{ $env:{ACTIVATE_FILE_ENV} = $prev }}
        Remove-Item -Path $actFile -Force -ErrorAction SilentlyContinue
    }}
    $global:LASTEXITCODE = $rc
}}
{HOOK_END}"""

_HOOK_BLOCK_RE = re.compile(
    rf"\n?{re.escape(HOOK_START)}.*?{re.escape(HOOK_END)}\n?",
    re.DOTALL,
)


@dataclass(frozen=True)
class ShellHookResult:
    """Outcome of installing or removing the ``rlinf`` shell hook."""

    shell: str | None
    rc_path: Path | None
    status: str  # installed | updated | already_installed | removed | not_found | unsupported | error
    detail: str = ""


def detect_shell(env: Mapping[str, str] | None = None) -> str | None:
    """Detect the active user shell from environment variables."""
    env_map = os.environ if env is None else env
    shell_path = env_map.get("SHELL", "")
    if shell_path:
        name = Path(shell_path).name.lower()
        if name.endswith(".exe"):
            name = name[:-4]
        if name in ("bash", "zsh", "fish"):
            return name
        if name in ("pwsh", "powershell"):
            return "pwsh"
    if os.name == "nt" or env_map.get("PSModulePath"):
        return "pwsh"
    return None


def is_shell_hook_active(env: Mapping[str, str] | None = None) -> bool:
    """Return ``True`` when invoked through the ``rlinf`` shell hook function."""
    env_map = os.environ if env is None else env
    return bool(env_map.get(ACTIVATE_FILE_ENV))


def is_ci_environment(env: Mapping[str, str] | None = None) -> bool:
    """Return ``True`` when running inside a CI environment."""
    env_map = os.environ if env is None else env
    return bool(env_map.get("CI") or env_map.get("GITHUB_ACTIONS"))


def render_shell_hook(shell: str) -> str:
    """Return the shell hook function snippet for ``shell``."""
    normalized = shell.lower()
    if normalized in ("bash", "zsh"):
        return _POSIX_HOOK + "\n"
    if normalized == "fish":
        return _FISH_HOOK + "\n"
    if normalized in ("pwsh", "powershell"):
        return _PWSH_HOOK + "\n"
    raise ValueError(
        f"Unsupported shell '{shell}'. Expected one of: {', '.join(SUPPORTED_SHELLS)}."
    )


def shell_rc_path(
    shell: str,
    *,
    home: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Return the startup profile file path for ``shell``."""
    env_map = os.environ if env is None else env
    home_dir = Path.home() if home is None else home
    normalized = shell.lower()
    if normalized == "bash":
        bashrc = home_dir / ".bashrc"
        if sys.platform == "darwin" and not bashrc.exists():
            bash_profile = home_dir / ".bash_profile"
            if bash_profile.exists():
                return bash_profile
        return bashrc
    if normalized == "zsh":
        zdotdir = env_map.get("ZDOTDIR")
        base = Path(zdotdir).expanduser() if zdotdir else home_dir
        return base / ".zshrc"
    if normalized == "fish":
        xdg = env_map.get("XDG_CONFIG_HOME")
        cfg_root = Path(xdg).expanduser() if xdg else home_dir / ".config"
        return cfg_root / "fish" / "config.fish"
    if normalized in ("pwsh", "powershell"):
        if os.name == "nt":
            return (
                home_dir
                / "Documents"
                / "PowerShell"
                / "Microsoft.PowerShell_profile.ps1"
            )
        xdg = env_map.get("XDG_CONFIG_HOME")
        cfg_root = Path(xdg).expanduser() if xdg else home_dir / ".config"
        return cfg_root / "powershell" / "Microsoft.PowerShell_profile.ps1"
    raise ValueError(
        f"Unsupported shell '{shell}'. Expected one of: {', '.join(SUPPORTED_SHELLS)}."
    )


def manual_activate_command(
    venv_dir: Path | str,
    *,
    shell: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Format the manual activation command for ``venv_dir`` and ``shell``."""
    detected = shell or detect_shell(env) or "bash"
    venv_str = str(venv_dir)
    if detected == "fish":
        return f"source {venv_str}/bin/activate.fish"
    if detected == "pwsh":
        script = "Scripts/Activate.ps1" if os.name == "nt" else "bin/Activate.ps1"
        return f". {venv_str}/{script}"
    return f"source {venv_str}/bin/activate"


def manual_reload_command(rc_path: Path, shell: str) -> str:
    """Format the shell rc reload command for ``rc_path``."""
    if shell == "pwsh":
        return f". {rc_path}"
    return f"source {rc_path}"


def install_shell_hook(
    shell: str | None = None,
    *,
    home: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> ShellHookResult:
    """Idempotently install or update the ``rlinf`` shell hook in the user's rc file."""
    target_shell = shell or detect_shell(env)
    if target_shell is None:
        return ShellHookResult(
            shell=None,
            rc_path=None,
            status="unsupported",
            detail=(
                "Could not detect shell from $SHELL. Specify one explicitly: "
                f"rlinf shell-init --install <{'|'.join(SUPPORTED_SHELLS)}>"
            ),
        )

    hook_text = render_shell_hook(target_shell).rstrip("\n")
    rc = shell_rc_path(target_shell, home=home, env=env)
    try:
        rc.parent.mkdir(parents=True, exist_ok=True)
        existing = rc.read_text(encoding="utf-8") if rc.exists() else ""
        match = _HOOK_BLOCK_RE.search(existing)
        if match:
            current_block = match.group(0).strip("\n")
            if current_block == hook_text:
                return ShellHookResult(
                    shell=target_shell,
                    rc_path=rc,
                    status="already_installed",
                )
            updated = (
                existing[: match.start()].rstrip("\n")
                + ("\n\n" if existing[: match.start()].strip() else "")
                + hook_text
                + "\n"
                + existing[match.end() :].lstrip("\n")
            )
            rc.write_text(updated, encoding="utf-8")
            return ShellHookResult(shell=target_shell, rc_path=rc, status="updated")

        prefix = existing.rstrip("\n")
        content = (prefix + "\n\n" if prefix else "") + hook_text + "\n"
        rc.write_text(content, encoding="utf-8")
        return ShellHookResult(shell=target_shell, rc_path=rc, status="installed")
    except OSError as exc:
        return ShellHookResult(
            shell=target_shell,
            rc_path=rc,
            status="error",
            detail=str(exc),
        )


def uninstall_shell_hook(
    shell: str | None = None,
    *,
    home: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> ShellHookResult:
    """Remove the ``rlinf`` shell hook block from the user's rc file."""
    target_shell = shell or detect_shell(env)
    if target_shell is None:
        return ShellHookResult(
            shell=None,
            rc_path=None,
            status="unsupported",
            detail="Could not detect shell from $SHELL.",
        )

    rc = shell_rc_path(target_shell, home=home, env=env)
    if not rc.exists():
        return ShellHookResult(shell=target_shell, rc_path=rc, status="not_found")
    try:
        existing = rc.read_text(encoding="utf-8")
        if not _HOOK_BLOCK_RE.search(existing):
            return ShellHookResult(shell=target_shell, rc_path=rc, status="not_found")
        cleaned = _HOOK_BLOCK_RE.sub("\n", existing).strip("\n")
        rc.write_text((cleaned + "\n") if cleaned else "", encoding="utf-8")
        return ShellHookResult(shell=target_shell, rc_path=rc, status="removed")
    except OSError as exc:
        return ShellHookResult(
            shell=target_shell,
            rc_path=rc,
            status="error",
            detail=str(exc),
        )


def request_shell_activation(
    venv_dir: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Write ``venv_dir`` to ``RLINF_ACTIVATE_FILE`` when the shell hook is active.

    Returns:
        ``True`` if the activation target was written for the parent shell hook.
    """
    env_map = os.environ if env is None else env
    act_file = env_map.get(ACTIVATE_FILE_ENV)
    if not act_file:
        return False
    Path(act_file).write_text(str(venv_dir.resolve()) + "\n", encoding="utf-8")
    return True


def _has_venv_python(venv_dir: Path) -> bool:
    py = (
        venv_dir / "Scripts" / "python.exe"
        if os.name == "nt"
        else venv_dir / "bin" / "python"
    )
    return py.is_file()


def resolve_venv_dir(
    venv_arg: str | None,
    *,
    repo_root: Path | None = None,
) -> Path:
    """Resolve ``venv_arg`` (default ``.venv``) against cwd and optional ``repo_root``."""
    raw = venv_arg or ".venv"
    expanded = Path(raw).expanduser()
    candidates: list[Path] = []
    if expanded.is_absolute():
        candidates.append(expanded)
    else:
        candidates.append(Path.cwd() / expanded)
        if repo_root is not None:
            candidates.append(repo_root / expanded)

    for candidate in candidates:
        if _has_venv_python(candidate):
            return candidate.resolve()

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Virtual environment '{raw}' not found (checked: {searched}). "
        f"Create it with 'rlinf install ... --venv {raw}'."
    )


def discover_venvs(repo_root: Path | None = None) -> list[Path]:
    """List virtual environments matching ``.venv`` or ``.venv-*`` in cwd / repo root."""
    roots = [Path.cwd().resolve()]
    if repo_root is not None and repo_root.resolve() not in roots:
        roots.append(repo_root.resolve())

    found: list[Path] = []
    seen: set[Path] = set()
    for base in roots:
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name == ".venv" or entry.name.startswith(".venv"):
                resolved = entry.resolve()
                if resolved not in seen and _has_venv_python(resolved):
                    seen.add(resolved)
                    found.append(resolved)
    return found


def handle_shell_init(
    *,
    shell: str | None = None,
    install: bool = False,
    uninstall: bool = False,
    stream: TextIO | None = None,
    err_stream: TextIO | None = None,
    home: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """Execute ``rlinf shell-init`` (print snippet, ``--install``, or ``--uninstall``)."""
    out = sys.stdout if stream is None else stream
    err = sys.stderr if err_stream is None else err_stream

    if install and uninstall:
        err.write("error: Cannot pass both '--install' and '--uninstall'.\n")
        err.flush()
        return 2

    if install:
        result = install_shell_hook(shell, home=home, env=env)
        if result.status in ("installed", "updated", "already_installed"):
            assert result.rc_path is not None and result.shell is not None
            reload_cmd = manual_reload_command(result.rc_path, result.shell)
            if result.status == "already_installed":
                out.write(
                    f"[rlinf] Shell hook ({result.shell}) is already installed in {result.rc_path}.\n"
                )
            else:
                verb = "Installed" if result.status == "installed" else "Updated"
                out.write(
                    f"[rlinf] {verb} {result.shell} shell hook in {result.rc_path}.\n"
                )
            if not is_shell_hook_active(env):
                out.write(f"[rlinf] Activate in this shell now with:\n  {reload_cmd}\n")
            out.flush()
            return 0
        err.write(f"error: Could not install shell hook: {result.detail}\n")
        err.flush()
        return 2

    if uninstall:
        result = uninstall_shell_hook(shell, home=home, env=env)
        if result.status == "removed":
            out.write(
                f"[rlinf] Removed {result.shell} shell hook from {result.rc_path}.\n"
            )
            out.flush()
            return 0
        if result.status == "not_found":
            out.write(f"[rlinf] No shell hook found in {result.rc_path}.\n")
            out.flush()
            return 0
        err.write(f"error: Could not remove shell hook: {result.detail}\n")
        err.flush()
        return 2

    target_shell = shell or detect_shell(env) or "bash"
    out.write(render_shell_hook(target_shell))
    out.flush()
    return 0


def handle_use(
    venv_arg: str | None = None,
    *,
    list_venvs: bool = False,
    auto_install_shell_hook: bool = True,
    repo_root: Path | None = None,
    stream: TextIO | None = None,
    err_stream: TextIO | None = None,
    home: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """Execute ``rlinf use [venv]`` or ``rlinf use --list``."""
    out = sys.stdout if stream is None else stream
    err = sys.stderr if err_stream is None else err_stream
    env_map = os.environ if env is None else env

    if list_venvs:
        active = env_map.get("VIRTUAL_ENV")
        active_resolved = Path(active).resolve() if active else None
        venvs = discover_venvs(repo_root)
        if (
            active_resolved is not None
            and active_resolved not in venvs
            and _has_venv_python(active_resolved)
        ):
            venvs.insert(0, active_resolved)
        if not venvs:
            out.write(
                "No virtual environments found (.venv / .venv-*). "
                "Create one with 'rlinf install ...'.\n"
            )
            out.flush()
            return 0
        for item in venvs:
            marker = "*" if active_resolved == item else " "
            out.write(f"{marker} {item}\n")
        out.flush()
        return 0

    try:
        resolved = resolve_venv_dir(venv_arg, repo_root=repo_root)
    except FileNotFoundError as exc:
        err.write(f"error: {exc}\n")
        err.flush()
        return 2

    if request_shell_activation(resolved, env=env_map):
        out.write(f"[rlinf] Activated virtual environment: {resolved}\n")
        out.flush()
        return 0

    activate_cmd = manual_activate_command(resolved, env=env_map)
    reload_hint: str | None = None
    if auto_install_shell_hook and not is_ci_environment(env_map):
        hook_res = install_shell_hook(home=home, env=env_map)
        if hook_res.status in ("installed", "updated"):
            assert hook_res.rc_path is not None and hook_res.shell is not None
            verb = "Installed" if hook_res.status == "installed" else "Updated"
            out.write(
                f"[rlinf] {verb} {hook_res.shell} shell hook in {hook_res.rc_path} "
                "(future shells auto-activate).\n"
            )
            reload_hint = manual_reload_command(hook_res.rc_path, hook_res.shell)
        elif hook_res.status == "already_installed":
            assert hook_res.rc_path is not None and hook_res.shell is not None
            reload_hint = manual_reload_command(hook_res.rc_path, hook_res.shell)

    out.write(
        "[rlinf] Shell hook is not active in this shell, so 'rlinf use' cannot "
        "switch the parent shell automatically.\n"
        f"Activate manually:\n  {activate_cmd}\n"
    )
    if reload_hint is not None:
        out.write(f"Reload shell hook in this shell:\n  {reload_hint}\n")
    else:
        out.write("Enable automatic switching once:\n  rlinf shell-init --install\n")
    out.flush()
    return 0
