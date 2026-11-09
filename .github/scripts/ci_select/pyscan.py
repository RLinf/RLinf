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

"""Static import closure of Python entry points, pruned by config values.

RLinf picks models, envs, workers, and backends at runtime from config values,
and imports the chosen one lazily inside the branch that selects it. Each
import is recorded with the constants its enclosing branches compare against
(``if x == "maniskill"``, ``in (A.X, A.Y)``, the key a builder function is
registered under, or the dict key a module path is stored under). A closure
drops an import only when such a guard names values that some config can set
and the job's own config sets none of them; a guard on anything else
(platform probes, ``sys.platform``) never prunes.

Imports inside a function are lazy: they run only if the function is called.
A module reached only as the parent package of a submodule runs its top-level
code, not its functions, so its lazy imports are followed only once some
import names the package itself.
"""

import ast
import posixpath
import re
from dataclasses import dataclass, field

from snapshot import Snapshot

#: Prefix of the config keys ``hydra_cfg.compose`` adds to a job's values.
KEY_MARK = "\x00key="

# A guard constant: ("s", value) for a string literal, ("a", owner, name) for
# an attribute such as SupportedEnvType.MANISKILL resolved later.
Sym = tuple
Guard = tuple[Sym, ...]
Guards = tuple[Guard, ...]

MODULE_STR = re.compile(
    r"^(rlinf|examples|toolkits|evaluations|tests)(\.[A-Za-z_]\w*)+(:[A-Za-z_]\w*)?$"
)
REL_MODULE_STR = re.compile(r"^\.{1,3}[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")
PATH_STR = re.compile(
    r"^[\w.${}:/-]+\.(py|ya?ml|json|txt|sh|toml|jinja|j2|xml|urdf|npy|csv)$"
)


@dataclass
class RawImport:
    level: int
    module: str | None
    names: tuple[str, ...]
    guards: Guards
    lazy: bool
    # ``import a.b.c`` binds ``a``, so every package on the path is used.
    binds_parents: bool = False


@dataclass
class FileInfo:
    imports: list[RawImport] = field(default_factory=list)
    module_strings: list[tuple[str, Guards, bool]] = field(default_factory=list)
    rel_module_strings: list[tuple[str, Guards, bool]] = field(default_factory=list)
    path_strings: list[str] = field(default_factory=list)
    dynamic: list[tuple[Guards, bool]] = field(default_factory=list)
    consts: dict[str, dict[str, str]] = field(default_factory=dict)
    # (key, default) of every ``cfg.get("key", "default")`` and ``getattr``.
    key_defaults: set[tuple[str, str]] = field(default_factory=set)
    # Names other modules can reach: imported names and attribute names.
    external_names: set[str] = field(default_factory=set)
    # (first line, last line, guards) of every statement, outermost first.
    spans: list[tuple[int, int, Guards]] = field(default_factory=list)
    parse_error: bool = False

    def line_guards(self, line: int) -> Guards | None:
        """Guards of the innermost statement on ``line``; None if none is."""
        best = None
        for start, end, guards in self.spans:
            if start <= line <= end and (
                best is None or end - start <= best[1] - best[0]
            ):
                best = (start, end, guards)
        return None if best is None else best[2]


def _const_sym(node: ast.AST | None) -> Sym | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return ("s", node.value.lower())
    if isinstance(node, ast.Attribute):
        if node.attr == "value" and isinstance(node.value, ast.Attribute):
            node = node.value
        owner = node.value
        owner_name = (
            owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", None)
        )
        if owner_name and node.attr.isupper():
            return ("a", owner_name, node.attr)
    return None


def _test_guard(test: ast.AST) -> Guard | None:
    """Constants an ``if`` test requires, or None when it is not a guard."""
    if isinstance(test, ast.BoolOp):
        parts = [_test_guard(v) for v in test.values]
        if isinstance(test.op, ast.Or):
            return (
                None
                if any(p is None for p in parts)
                else tuple(s for p in parts for s in p)
            )
        return next((p for p in parts if p is not None), None)
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op, left, right = test.ops[0], test.left, test.comparators[0]
        if isinstance(op, (ast.Eq, ast.Is)):
            sym = _const_sym(right) or _const_sym(left)
            return (sym,) if sym else None
        if isinstance(op, ast.In) and isinstance(right, (ast.Tuple, ast.List, ast.Set)):
            syms = [_const_sym(e) for e in right.elts]
            return None if not syms or any(s is None for s in syms) else tuple(syms)
    return None


def _exit_guard(stmt: ast.stmt) -> Guard | None:
    """``if x != K: return`` guards the rest of the block by ``x == K``."""
    if not isinstance(stmt, ast.If) or stmt.orelse or not stmt.body:
        return None
    if not isinstance(stmt.body[-1], (ast.Return, ast.Raise, ast.Continue)):
        return None
    return _negated_guard(stmt.test)


def _negated_guard(test: ast.AST) -> Guard | None:
    """Constants the code after ``if test: exit`` requires."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _test_guard(test.operand)
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
        # not (a or b) == (not a) and (not b): any one part is a guard.
        return next((g for g in map(_negated_guard, test.values) if g), None)
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op, left, right = test.ops[0], test.left, test.comparators[0]
        if isinstance(op, (ast.NotEq, ast.IsNot)):
            sym = _const_sym(right) or _const_sym(left)
            return (sym,) if sym else None
        if isinstance(op, ast.NotIn) and isinstance(
            right, (ast.Tuple, ast.List, ast.Set)
        ):
            syms = [_const_sym(e) for e in right.elts]
            return None if not syms or any(x is None for x in syms) else tuple(syms)
    return None


def _is_type_checking(test: ast.AST) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _registration_guards(tree: ast.AST) -> dict[str, Guard]:
    """Functions only ever referenced as the value of a registration key.

    ``register_model(SupportedModel.X.value, _build_x)``, ``{KEY: _build_x}``
    and ``@register("x")`` all make the function run only when the config
    selects that key, so its body is guarded by the key.
    """
    keyed: dict[str, list[Sym]] = {}
    keyed_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for i, arg in enumerate(node.args):
                if isinstance(arg, ast.Name):
                    syms = [_const_sym(a) for a in node.args[:i]]
                    if syms and all(syms):
                        keyed.setdefault(arg.id, []).extend(syms)
                        keyed_nodes.add(id(arg))
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(v, ast.Name) and _const_sym(k):
                    keyed.setdefault(v.id, []).append(_const_sym(k))
                    keyed_nodes.add(id(v))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and dec.args:
                    syms = [_const_sym(a) for a in dec.args]
                    if all(syms):
                        keyed.setdefault(node.name, []).extend(syms)
    # A function also called or stored anywhere else runs unconditionally.
    plain = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and id(node) not in keyed_nodes
    }
    return {
        name: (("f", name), *syms) for name, syms in keyed.items() if name not in plain
    }


def _called_names(nodes: list[ast.AST], into_functions: bool) -> set[str]:
    out: set[str] = set()
    stack = list(nodes)
    while stack:
        node = stack.pop()
        if not into_functions and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        ):
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            out.add(node.func.id)
        stack.extend(ast.iter_child_nodes(node))
    return out


def _load_time_functions(tree: ast.Module) -> set[str]:
    """Top-level functions that run while the module is imported."""
    funcs = {
        n.name: n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    todo = list(_called_names(tree.body, into_functions=False) & funcs.keys())
    out: set[str] = set()
    while todo:
        name = todo.pop()
        if name in out:
            continue
        out.add(name)
        todo += list(
            _called_names(funcs[name].body, into_functions=True) & funcs.keys()
        )
    return out


class _Scanner(ast.NodeVisitor):
    def visit_Module(self, node: ast.Module) -> None:
        self.visit_block(node.body)

    def __init__(
        self, info: FileInfo, reg_guards: dict[str, Guard], load_time: set[str]
    ):
        self.info = info
        self.reg_guards = reg_guards
        self.load_time = load_time
        self.guards: list[Guard] = []
        self.funcs: list[str] = []
        # name -> innermost guard (None if unguarded) at each reference.
        self.refs: dict[str, list[Guard | None]] = {}

    def _g(self) -> Guards:
        return tuple(self.guards)

    def _lazy(self) -> bool:
        return bool(self.funcs) and self.funcs[0] not in self.load_time

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking(node.test):
            for n in node.orelse:
                self.visit(n)
            return
        self.visit(node.test)
        # In a module ``__getattr__`` the test names the attribute an import
        # asks for, not a config value.
        guard = None if "__getattr__" in self.funcs else _test_guard(node.test)
        if guard:
            self.guards.append(guard)
        self.visit_block(node.body)
        if guard:
            self.guards.pop()
        self.visit_block(node.orelse)

    def visit_block(self, stmts: list[ast.stmt]) -> None:
        pushed = 0
        for stmt in stmts:
            guards = self._g()
            if isinstance(stmt, ast.If):
                # A constant comparison has no effect where it fails.
                own = None if "__getattr__" in self.funcs else _test_guard(stmt.test)
                if own:
                    guards += (own,)
                self.info.spans.append((stmt.lineno, stmt.test.end_lineno, guards))
            else:
                self.info.spans.append((stmt.lineno, stmt.end_lineno, guards))
            self.visit(stmt)
            guard = None if "__getattr__" in self.funcs else _exit_guard(stmt)
            if guard:
                self.guards.append(guard)
                pushed += 1
        for _ in range(pushed):
            self.guards.pop()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id not in self.funcs:
            self.refs.setdefault(node.id, []).append(
                self.guards[-1] if self.guards else None
            )

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for dec in node.decorator_list:
            self.visit(dec)
        guard = self.reg_guards.get(node.name)
        if guard:
            self.guards.append(guard)
        self.funcs.append(node.name)
        self.visit_block(node.body)
        self.funcs.pop()
        if guard:
            self.guards.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = _visit_func

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        self.info.consts.setdefault(node.name, {})[t.id] = (
                            stmt.value.value.lower()
                        )
        guard = self.reg_guards.get(node.name)
        if guard:
            self.guards.append(guard)
        self.generic_visit(node)
        if guard:
            self.guards.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        # SupportedModel.OPENVLA = SupportedModel.register("openvla", ...)
        value = node.value
        literal = None
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            literal = value.value
        elif (
            isinstance(value, ast.Call)
            and value.args
            and isinstance(value.args[0], ast.Constant)
            and isinstance(value.args[0].value, str)
        ):
            literal = value.args[0].value
        if literal is not None:
            for t in node.targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name):
                    self.info.consts.setdefault(t.value.id, {})[t.attr] = (
                        literal.lower()
                    )
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        # {"wan": "rlinf.envs.sim.world_model.backend.wan"}: loaded by key.
        for k, v in zip(node.keys, node.values):
            if k is not None:
                self.visit(k)
            sym = _const_sym(k)
            if sym and isinstance(v, ast.Constant) and isinstance(v.value, str):
                self.guards.append((sym,))
                self._string(v.value)
                self.guards.pop()
            else:
                self.visit(v)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.info.imports.append(
                RawImport(
                    0, alias.name, (), self._g(), self._lazy(), binds_parents=True
                )
            )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.info.external_names.add(node.attr)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        names = tuple(a.name for a in node.names if a.name != "*")
        self.info.external_names.update(names)
        self.info.imports.append(
            RawImport(node.level, node.module, names, self._g(), self._lazy())
        )

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        fname = (
            func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        )
        if fname in ("import_module", "__import__") and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                mod = arg.value
                level = len(mod) - len(mod.lstrip("."))
                self.info.imports.append(
                    RawImport(
                        level, mod.lstrip(".") or None, (), self._g(), self._lazy()
                    )
                )
            else:
                self.info.dynamic.append((self._g(), self._lazy()))
        # cfg.get("rollout_backend", "huggingface") and getattr(cfg, "x", "y")
        if fname in ("get", "pop", "setdefault") and len(node.args) >= 2:
            key, default = node.args[0], node.args[1]
        elif fname == "getattr" and len(node.args) >= 3:
            key, default = node.args[1], node.args[2]
        else:
            key = default = None
        if (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and isinstance(default, ast.Constant)
            and isinstance(default.value, str)
        ):
            self.info.key_defaults.add((key.value.lower(), default.value.lower()))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self._string(node.value)

    def _string(self, v: str) -> None:
        if len(v) >= 300:
            return
        if MODULE_STR.match(v):
            self.info.external_names.add(v.replace(":", ".").rsplit(".", 1)[-1])
            self.info.module_strings.append((v.split(":")[0], self._g(), self._lazy()))
        elif REL_MODULE_STR.match(v):
            self.info.rel_module_strings.append((v, self._g(), self._lazy()))
        elif "/" in v and PATH_STR.match(v):
            self.info.path_strings.append(v)


def scan_python(text: str) -> FileInfo:
    info = FileInfo()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        info.parse_error = True
        return info
    reg = _registration_guards(tree)
    load_time = _load_time_functions(tree)
    first = _Scanner(FileInfo(), reg, load_time)
    first.visit(tree)
    top_funcs = {
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    guards = dict(reg)
    for name in top_funcs - reg.keys() - load_time:
        sites = first.refs.get(name)
        # Every reference sits in a guarded branch: the body runs only then.
        # The ("f", name) marker voids the guard if another module uses it.
        if sites and all(g is not None for g in sites):
            syms = dict.fromkeys(x for g in sites for x in g if x[0] != "f")
            guards[name] = (("f", name), *syms)
    _Scanner(info, guards, load_time).visit(tree)
    return info


class PyIndex:
    """Import resolution and closures over one snapshot."""

    _info_cache: dict[str, FileInfo] = {}

    def __init__(self, snap: Snapshot):
        self.snap = snap
        self.py_files = sorted(f for f in snap.files if f.endswith(".py"))
        self.py_dirs = {posixpath.dirname(f) for f in self.py_files}
        # Dirs with Python at any depth; data under them belongs to that code.
        self.py_trees = set()
        for d in self.py_dirs:
            while d and d not in self.py_trees:
                self.py_trees.add(d)
                d = posixpath.dirname(d)
        self.children: dict[str, list[str]] = {}
        self.subdirs: dict[str, set[str]] = {}
        for f in snap.files:
            d = posixpath.dirname(f)
            self.children.setdefault(d, []).append(f)
            while d:
                parent = posixpath.dirname(d)
                self.subdirs.setdefault(parent, set()).add(d)
                d = parent
        self._closure_cache: dict = {}
        self._consts = None
        self._external = None
        self._key_defaults = None
        self._effective: dict = {}
        #: Files with a dynamic import whose target no string in the file names.
        self.unresolved_dynamic: set[str] = set()

    def info(self, path: str) -> FileInfo:
        key = self.snap.blob_id(path) or path
        if key not in self._info_cache:
            self._info_cache[key] = scan_python(self.snap.read(path) or "")
        return self._info_cache[key]

    @property
    def consts(self) -> dict[str, dict[str, str]]:
        if self._consts is None:
            # Owners are bare class names; a name two classes define with
            # different values resolves to None, which never prunes.
            merged: dict[str, dict[str, str | None]] = {}
            for f in self.py_files:
                for owner, table in self.info(f).consts.items():
                    slot = merged.setdefault(owner, {})
                    for attr, value in table.items():
                        if attr in slot and slot[attr] != value:
                            slot[attr] = None
                        else:
                            slot[attr] = value
            self._consts = merged
        return self._consts

    @property
    def external_names(self) -> frozenset[str]:
        if self._external is None:
            out: set[str] = set()
            for f in self.py_files:
                out |= self.info(f).external_names
            self._external = frozenset(out)
        return self._external

    @property
    def key_defaults(self) -> frozenset[tuple[str, str]]:
        if self._key_defaults is None:
            out: set[tuple[str, str]] = set()
            for f in self.py_files:
                out |= self.info(f).key_defaults
            self._key_defaults = frozenset(out)
        return self._key_defaults

    def effective(self, values: frozenset[str] | None) -> frozenset[str] | None:
        """Config values plus the code default of every key the config leaves unset.

        ``cfg.get("rollout_backend", "sglang")`` in ``rlinf/config.py`` makes
        SGLang the backend of every config without that key, wherever the
        branch on it lives.
        """
        if values is None:
            return None
        if values not in self._effective:
            self._effective[values] = values | {
                d for k, d in self.key_defaults if KEY_MARK + k not in values
            }
        return self._effective[values]

    def resolve_guard(self, guard: Guard) -> frozenset[str] | None:
        out = set()
        for sym in guard:
            if sym[0] == "f":
                if sym[1] in self.external_names:
                    return None
            elif sym[0] == "s":
                out.add(sym[1])
            else:
                value = self.consts.get(sym[1], {}).get(sym[2])
                if value is None:
                    return None
                out.add(value)
        return frozenset(out)

    # -- module resolution -------------------------------------------------

    def _module_files(self, dotted: str, base: str) -> list[str]:
        """The module's file first, then the ``__init__.py`` of each parent."""
        rel = dotted.replace(".", "/")
        prefix = f"{base}/" if base else ""
        for cand in (f"{prefix}{rel}.py", f"{prefix}{rel}/__init__.py"):
            if self.snap.exists(cand):
                out = [cand]
                parts = dotted.split(".")
                for i in range(len(parts) - 1, 0, -1):
                    init = f"{prefix}{'/'.join(parts[:i])}/__init__.py"
                    if self.snap.exists(init):
                        out.append(init)
                return out
        if f"{prefix}{rel}" in self.snap.dirs:
            # Namespace package: nothing to execute, but submodules resolve.
            return [f"{prefix}{rel}/"]
        return []

    def resolve(self, dotted: str, roots: tuple[str, ...]) -> list[str]:
        for base in roots:
            files = self._module_files(dotted, base)
            if files:
                return files
        return []

    def _targets(
        self, files: list[str], all_explicit: bool = False
    ) -> list[tuple[str, bool]]:
        return [
            (f, all_explicit or i == 0)
            for i, f in enumerate(files)
            if not f.endswith("/")
        ]

    def _import_targets(
        self, path: str, imp: RawImport, roots: tuple[str, ...]
    ) -> list[tuple[str, bool]]:
        here = posixpath.dirname(path)
        if imp.level:
            pkg = here
            for _ in range(imp.level - 1):
                pkg = posixpath.dirname(pkg)
            base_dotted = pkg.replace("/", ".")
            dotted = f"{base_dotted}.{imp.module}" if imp.module else base_dotted
            search: tuple[str, ...] = ("",)
        else:
            dotted = imp.module or ""
            search = roots + (here,)
        out = (
            self._targets(self.resolve(dotted, search), imp.binds_parents)
            if dotted
            else []
        )
        for name in imp.names:
            out += self._targets(
                self.resolve(f"{dotted}.{name}" if dotted else name, search)
            )
        return out

    def _data_files(self, d: str) -> list[str]:
        """Non-Python files a package dir can open by relative path.

        Hydra configs outside ``rlinf/`` are left out: a job reads the ones its
        config composes, which ``hydra_cfg.compose`` adds.
        """
        if d.startswith("rlinf/"):

            def keep(f: str) -> bool:
                return not f.endswith(".py")
        else:

            def keep(f: str) -> bool:
                return not f.endswith((".py", ".yaml", ".yml"))

        out = [f for f in self.children.get(d, []) if keep(f)]
        stack = [s for s in self.subdirs.get(d, ()) if s not in self.py_trees]
        while stack:
            s = stack.pop()
            out += [f for f in self.children.get(s, []) if keep(f)]
            stack += [c for c in self.subdirs.get(s, ()) if c not in self.py_trees]
        return out

    def line_ok(
        self, path: str, line: int, values: frozenset[str] | None, vocab: frozenset[str]
    ) -> bool:
        """Whether a job with ``values`` can run ``line`` of ``path``."""
        if values is None:
            return True
        info = self.info(path)
        guards = info.line_guards(line)
        if guards is None:
            return True
        return self._guard_ok(guards, self.effective(values), vocab)

    def _guard_ok(
        self, guards: Guards, values: frozenset[str] | None, vocab: frozenset[str]
    ) -> bool:
        if values is None:
            return True
        for guard in guards:
            resolved = self.resolve_guard(guard)
            if resolved is None or not (resolved & vocab):
                continue
            if not (resolved & values):
                return False
        return True

    def closure(
        self,
        roots: list[str],
        values: frozenset[str] | None,
        vocab: frozenset[str],
        extra_paths: tuple[str, ...] = (),
        parents: dict[str, str] | None = None,
    ) -> tuple[frozenset[str], frozenset[str]]:
        """Python files and data files reachable from ``roots``.

        ``values`` are the job's config values; None follows every import.
        ``vocab`` is every value any config sets; guards outside it never prune.
        ``parents``, when given, records the file that first reached each file.
        """
        values = self.effective(values)
        key = (tuple(sorted(roots)), values, extra_paths)
        if parents is None and key in self._closure_cache:
            return self._closure_cache[key]
        search = ("",) + extra_paths
        eager_done: set[str] = set()
        lazy_done: set[str] = set()
        data: set[str] = set()
        queue = [(r, True) for r in roots if self.snap.exists(r)]
        while queue:
            path, explicit = queue.pop()
            do_eager = path not in eager_done
            do_lazy = explicit and path not in lazy_done
            if not (do_eager or do_lazy):
                continue
            eager_done.add(path)
            if explicit:
                lazy_done.add(path)
            info = self.info(path)

            def wanted(lazy: bool, guards: Guards) -> bool:
                return (do_lazy if lazy else do_eager) and self._guard_ok(
                    guards, values, vocab
                )

            targets: list[tuple[str, bool]] = []
            for imp in info.imports:
                if wanted(imp.lazy, imp.guards):
                    targets += self._import_targets(path, imp, search)
            for dotted, guards, lazy in info.module_strings:
                if wanted(lazy, guards):
                    parts = dotted.split(".")
                    for i in range(len(parts), 0, -1):
                        files = self.resolve(".".join(parts[:i]), search)
                        if files:
                            targets += self._targets(files)
                            break
            for guards, lazy in info.dynamic:
                if wanted(lazy, guards):
                    pkg = posixpath.dirname(path).replace("/", ".")
                    resolved = []
                    for rel, rguards, _ in info.rel_module_strings:
                        if self._guard_ok(rguards, values, vocab):
                            level = len(rel) - len(rel.lstrip("."))
                            base = pkg
                            for _ in range(level - 1):
                                base = base.rpartition(".")[0]
                            resolved += self._targets(
                                self.resolve(f"{base}.{rel.lstrip('.')}", ("",))
                            )
                    if not resolved and not info.module_strings:
                        self.unresolved_dynamic.add(path)
                    targets += resolved
            if do_eager:
                for raw in info.path_strings:
                    p = raw.replace("${REPO_PATH}/", "").removeprefix("./")
                    for cand in (
                        p,
                        posixpath.normpath(posixpath.join(posixpath.dirname(path), p)),
                    ):
                        if self.snap.exists(cand):
                            if cand.endswith(".py"):
                                targets.append((cand, True))
                            else:
                                data.add(cand)
                            break
            for t, t_explicit in targets:
                if t not in eager_done or (t_explicit and t not in lazy_done):
                    queue.append((t, t_explicit))
                    if parents is not None:
                        parents.setdefault(t, path)
        for d in {posixpath.dirname(p) for p in eager_done}:
            data.update(self._data_files(d))
        result = (frozenset(eager_done), frozenset(data))
        if parents is None:
            self._closure_cache[key] = result
        return result

    def explain(self, roots, values, vocab, target, extra_paths=()) -> list[str]:
        """The import chain from a root to ``target``, for debugging the map."""
        parents: dict[str, str] = {}
        self.closure(roots, values, vocab, extra_paths, parents)
        chain = [target]
        while chain[-1] in parents and len(chain) < 50:
            chain.append(parents[chain[-1]])
        return chain[::-1]
