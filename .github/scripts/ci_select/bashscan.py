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

"""Which lines of a bash script run for given variable values.

The scripts CI calls (``install.sh``, ``tests/e2e_tests/*/run*.sh``) dispatch on
a few variables through ``case`` and ``[[ "$X" == pat ]]``, and otherwise call
functions. Given the values of those variables, ``analyze`` returns the lines
that can run: lines whose enclosing branches all match, inside functions
reachable from top-level code through such lines. A branch on anything it
cannot evaluate counts as taken.
"""

import fnmatch
import re
from dataclasses import dataclass, field

FUNC_DEF = re.compile(
    r"^\s*(?:function\s+)?([A-Za-z_][\w-]*)\s*\(\)\s*\{?\s*$|^\s*function\s+([A-Za-z_][\w-]*)\s*\{?\s*$"
)
VAR_REF = re.compile(r'^"?\$\{?([A-Za-z_]\w*|\d)(?::-[^}]*)?\}?"?$')
ASSIGN = re.compile(r"^\s*(?:export\s+|local\s+|readonly\s+)?([A-Za-z_]\w*)=(.*)$")
HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?")


@dataclass
class Guard:
    var: str | None  # None: cannot be evaluated, always taken
    patterns: tuple[str, ...]
    case_id: int | None = None  # set for case branches
    branch: int = 0


@dataclass
class BashScript:
    lines: list[str]
    # Per physical line (0-based): enclosing guard ids, innermost last.
    ctx: list[tuple[int, ...]]
    func_of: list[str | None]
    guards: list[Guard]
    cases: dict[int, list[tuple[str, ...]]]
    functions: dict[str, tuple[int, int]]
    # Logical lines: (first physical index, last physical index, joined text).
    logical: list[tuple[int, int, str]] = field(default_factory=list)


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t;"):
            break
        out.append(ch)
    return "".join(out)


def _subject(word: str) -> str | None:
    m = VAR_REF.match(word.strip())
    return m.group(1) if m else None


def _if_guard(cond: str) -> tuple[str | None, tuple[str, ...]] | None:
    """``[[ "$X" == a ]] || [[ "$X" == b ]]`` -> ("X", ("a", "b"))."""
    cond = re.sub(r";\s*then\s*$", "", cond).strip()
    alts = re.split(r"\|\|", cond)
    var, pats = None, []
    for alt in alts:
        alt = alt.split("&&")[0]
        m = re.search(
            r'\[\[?\s*("?\$\{?[A-Za-z_]\w*(?::-[^}]*)?\}?"?)\s*(==|=)\s*("[^"]*"|\'[^\']*\'|[^\s\]]+)\s*\]\]?',
            alt,
        )
        if not m:
            return None
        v = _subject(m.group(1))
        if v is None or (var is not None and v != var):
            return None
        var = v
        pats.append(m.group(3).strip("'\""))
    return (var, tuple(pats)) if var else None


def parse(text: str) -> BashScript:
    lines = text.split("\n")
    n = len(lines)
    ctx: list[tuple[int, ...]] = [()] * n
    func_of: list[str | None] = [None] * n
    guards: list[Guard] = []
    cases: dict[int, list[tuple[str, ...]]] = {}
    functions: dict[str, tuple[int, int]] = {}

    # Frames end with the active guard id (or None):
    # ["case", case_id, expect_pattern, subject_var, gid] or ["if", gid].
    stack: list[list] = []
    current_func: str | None = None
    func_start = 0
    heredoc: str | None = None
    logical: list[tuple[int, int, str]] = []
    i = 0
    while i < n:
        start = i
        raw = lines[i]
        joined = raw
        while joined.endswith("\\") and i + 1 < n:
            i += 1
            joined = joined[:-1] + " " + lines[i].lstrip()
        end = i
        i += 1

        def active() -> tuple[int, ...]:
            return tuple(f[-1] for f in stack if f[-1] is not None)

        if heredoc is not None:
            for k in range(start, end + 1):
                ctx[k] = active()
                func_of[k] = current_func
            if raw.strip() == heredoc:
                heredoc = None
            continue
        line = _strip_comment(joined).strip()
        m = HEREDOC.search(line)
        if m and "<<<" not in line:
            heredoc = m.group(1)

        fm = FUNC_DEF.match(line)
        if fm and current_func is None:
            current_func = fm.group(1) or fm.group(2)
            func_start = start
        elif current_func is None and re.match(
            r"^[A-Za-z_][\w-]*\s*\(\)\s*\{.*\}\s*;?$", line
        ):
            name = line.split("(")[0].strip()
            functions[name] = (start, end)
            for k in range(start, end + 1):
                func_of[k] = name
                ctx[k] = active()
            logical.append((start, end, line))
            continue

        # Close a case branch's pattern / open a new one.
        if (
            stack
            and stack[-1][0] == "case"
            and stack[-1][2]
            and line
            and line != "esac"
        ):
            pm = re.match(r"^\(?\s*([^()]*?)\s*\)\s*(.*)$", line)
            if pm:
                case_id = stack[-1][1]
                pats = tuple(p.strip().strip("'\"") for p in pm.group(1).split("|"))
                cases[case_id].append(pats)
                gid = len(guards)
                guards.append(
                    Guard(stack[-1][3], pats, case_id, len(cases[case_id]) - 1)
                )
                stack[-1][2] = False
                stack[-1][4] = gid
                line = pm.group(2).strip()

        if re.match(r"^case\s", line):
            m2 = re.match(r"^case\s+(.+?)\s+in\b(.*)$", line)
            case_id = len(cases)
            cases[case_id] = []
            stack.append(
                ["case", case_id, True, _subject(m2.group(1)) if m2 else None, None]
            )
        elif re.match(r"^(if|elif)\s", line):
            g = _if_guard(re.sub(r"^(if|elif)\s+", "", line))
            gid = None
            if g:
                gid = len(guards)
                guards.append(Guard(g[0], g[1]))
            one_line = re.search(r"\bthen\b.*\bfi\b\s*;?\s*$", line)
            if line.startswith("elif") and stack and stack[-1][0] == "if":
                stack[-1][-1] = gid
            elif not one_line:
                stack.append(["if", gid])
        elif re.match(r"^else\b", line) and stack and stack[-1][0] == "if":
            stack[-1][-1] = None

        for k in range(start, end + 1):
            ctx[k] = active()
            func_of[k] = current_func
        logical.append((start, end, line))

        if (
            re.search(r"(^|\s|;)(;;|;&|;;&)\s*$", line)
            and stack
            and stack[-1][0] == "case"
        ):
            stack[-1][2] = True
            stack[-1][4] = None
        if re.match(r"^esac\b", line) and stack and stack[-1][0] == "case":
            stack.pop()
        elif re.match(r"^fi\b", line) and stack and stack[-1][0] == "if":
            stack.pop()
        if (
            current_func is not None
            and re.match(r"^\}\s*$", raw)
            and start != func_start
        ):
            functions[current_func] = (func_start, end)
            current_func = None
            stack.clear()
    return BashScript(lines, ctx, func_of, guards, cases, functions, logical)


def _taken(script: BashScript, gid: int, values: dict[str, str]) -> bool:
    g = script.guards[gid]
    if g.var is None or g.var not in values:
        return True
    value = values[g.var]
    if g.case_id is None:
        return any(fnmatch.fnmatchcase(value, p) for p in g.patterns)
    for idx, pats in enumerate(script.cases[g.case_id]):
        if any(fnmatch.fnmatchcase(value, p if p != '""' else "") for p in pats):
            return idx == g.branch
    return False


@dataclass
class BashRun:
    used: frozenset[int]  # 1-based physical line numbers
    commands: list[str]  # logical lines that can run, in order
    assigned: dict[str, set[str]]


def analyze(script: BashScript, values: dict[str, str] | None) -> BashRun:
    """Lines that can run; ``values`` None means every branch is taken."""
    values = values or {}
    pinned = set(values)

    def line_ok(k: int) -> bool:
        return all(_taken(script, gid, values) for gid in script.ctx[k])

    names = sorted(script.functions, key=len, reverse=True)
    call_re = (
        re.compile(r"(?<![\w$-])(" + "|".join(map(re.escape, names)) + r")(?![\w-])")
        if names
        else None
    )

    reached: set[str | None] = {None}
    queue: list[str | None] = [None]
    while queue:
        fn = queue.pop()
        for start, end, text in script.logical:
            if script.func_of[start] != fn or not line_ok(start):
                continue
            if call_re and not FUNC_DEF.match(text):
                for m in call_re.finditer(text):
                    if m.group(1) not in reached:
                        reached.add(m.group(1))
                        queue.append(m.group(1))

    used: set[int] = set()
    commands: list[str] = []
    assigned: dict[str, set[str]] = {}
    for start, end, text in script.logical:
        if script.func_of[start] in reached and line_ok(start):
            used.update(range(start + 1, end + 2))
            commands.append(text)
            for part in re.split(r";|&&|\|\|", text):
                am = ASSIGN.match(part)
                if am and am.group(1) not in pinned:
                    assigned.setdefault(am.group(1), set()).add(
                        am.group(2).strip().strip("'\"")
                    )
    return BashRun(frozenset(used), commands, assigned)


POS_ARG = r"\$\{?%d(?::-[^}]*)?\}?"


def flag_vars(script: BashScript) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """``parse_args`` mapping: ``--flag -> VAR``, and ordered positional patterns."""
    flags: dict[str, str] = {}
    positional: list[tuple[str, str]] = []
    span = script.functions.get("parse_args")
    if not span:
        return flags, positional
    for start, _end, text in script.logical:
        if not span[0] <= start <= span[1]:
            continue
        case_guard = next(
            (
                script.guards[g]
                for g in reversed(script.ctx[start])
                if script.guards[g].case_id is not None
            ),
            None,
        )
        if case_guard is None:
            continue
        for part in re.split(r";", text):
            am = ASSIGN.match(part)
            if not am:
                continue
            rhs = am.group(2).strip().strip('"')
            for p in case_guard.patterns:
                if re.fullmatch(POS_ARG % 2, rhs) and p.startswith("-"):
                    flags[p] = am.group(1)
                elif re.fullmatch(POS_ARG % 1, rhs) and not p.startswith("-"):
                    if (p, am.group(1)) not in positional:
                        positional.append((p, am.group(1)))
    return flags, positional


def top_level_defaults(script: BashScript) -> dict[str, str]:
    """``VAR=literal`` and ``VAR="${VAR:-literal}"`` outside functions."""
    out: dict[str, str] = {}
    for start, _end, text in script.logical:
        if script.func_of[start] is not None or script.ctx[start]:
            continue
        am = ASSIGN.match(text)
        if not am:
            continue
        rhs = am.group(2).strip().strip("'\"")
        dm = re.fullmatch(r"\$\{" + am.group(1) + r":-([^}$]*)\}", rhs)
        if dm:
            out[am.group(1)] = dm.group(1)
        elif "$" not in rhs and "(" not in rhs:
            out[am.group(1)] = rhs
    return out


def positional_values(script: BashScript, argv: list[str]) -> dict[str, str]:
    """Values for a script called as ``script.sh argv...``.

    Binds ``$1``.. and top-level aliases such as ``CONFIG=$1``; an alias also
    assigned elsewhere (a fallback default, say) stays unknown.
    """
    values = {str(i + 1): a for i, a in enumerate(argv)}
    aliases: dict[str, set[str]] = {}
    for start, _end, text in script.logical:
        for part in re.split(r";|&&|\|\|", text):
            am = ASSIGN.match(part)
            if am:
                aliases.setdefault(am.group(1), set()).add(
                    am.group(2).strip().strip('"')
                )
    for var, rhss in aliases.items():
        if len(rhss) == 1:
            pm = re.fullmatch(r"\$\{?(\d)\}?", next(iter(rhss)))
            if pm and pm.group(1) in values:
                values[var] = values[pm.group(1)]
    return values


def args_to_values(script: BashScript, argv: list[str]) -> dict[str, str]:
    """Values ``parse_args`` sets for ``argv``; unset flag vars keep defaults."""
    flags, positional = flag_vars(script)
    defaults = top_level_defaults(script)
    values: dict[str, str] = {
        var: defaults.get(var, "")
        for var in {*flags.values(), *(v for _, v in positional)}
    }
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in flags and i + 1 < len(argv):
            values[flags[tok]] = argv[i + 1]
            i += 2
            continue
        if not tok.startswith("-"):
            for pat, var in positional:
                if fnmatch.fnmatchcase(tok, pat):
                    values[var] = tok
                    break
        i += 1
    return values
