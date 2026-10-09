"""Loading architecture programs.

A *program* is a self-contained Python module that defines the block class of
the program contract (:mod:`openarchevo.contract`: class name, constructor
arguments and ``forward`` output).

Programs from the original search import helpers from ``src.models.*``.
:func:`install_legacy_spike_alias` maps those module paths onto
``openarchevo.models`` (see :mod:`openarchevo.models.legacy`) so archived
programs (``zoo/``) load unchanged. :func:`load_block_class` also checks, before
running a program, that every module it imports is installed: archived programs
wrap some imports in ``try``/``except`` blocks with placeholder fallbacks, and a
missing dependency would otherwise silently swap real layers for placeholders.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import linecache
import sys
import types
from pathlib import Path
from typing import Union

from torch import nn

from openarchevo.contract import BLOCK_CLASS
from openarchevo.models import legacy, neurons

#: Legacy module that did not exist when the archive was evaluated (opt-in shim).
LEGACY_SPIKE_NN = "src.models.spike.nn"

#: pip distribution names for import names, used in error messages.
_PIP_NAMES = {
    "fla": "flash-linear-attention", "flash_attn": "flash-attn", "einops": "einops",
    "transformers": "transformers", "torch": "torch",
}
_SHIM_FLAG = "__openarchevo_shim__"


def _shim_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    setattr(mod, _SHIM_FLAG, True)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def install_legacy_spike_alias(spike_nn: bool = False) -> None:
    """Register the legacy ``src.models.*`` modules that archived programs import.

    Always registered: ``src.models.spike.neuron`` (``registry``),
    ``src.models.spike.surrogate`` (``piecewise_quadratic_surrogate``) and
    ``src.models.nn.dropout`` (``DropoutNd``). ``src.models.spike.nn``
    (``SpikingLinear``) is registered only with ``spike_nn=True``: it was absent
    from the research code base, so by default the seven programs that import it
    run their in-file fallbacks, as they did when the archive was evaluated.
    Modules that already exist and were not created here are left untouched.
    """
    for name in ("src", "src.models", "src.models.spike", "src.models.nn"):
        pkg = sys.modules.setdefault(name, _shim_module(name))
        if not hasattr(pkg, "__path__"):
            pkg.__path__ = []  # mark as package
    shims = {
        "src.models.spike.neuron": _shim_module("src.models.spike.neuron", registry=neurons.registry),
        "src.models.spike.surrogate": _shim_module(
            "src.models.spike.surrogate",
            piecewise_quadratic_surrogate=neurons.piecewise_quadratic_surrogate),
        "src.models.nn.dropout": _shim_module("src.models.nn.dropout", DropoutNd=legacy.DropoutNd),
    }
    for name, mod in shims.items():
        sys.modules.setdefault(name, mod)
    current = sys.modules.get(LEGACY_SPIKE_NN)
    spike_pkg = sys.modules["src.models.spike"]
    if spike_nn and current is None:
        sys.modules[LEGACY_SPIKE_NN] = _shim_module(LEGACY_SPIKE_NN, SpikingLinear=legacy.LegacySpikingLinear)
    elif not spike_nn and current is not None and getattr(current, _SHIM_FLAG, False):
        del sys.modules[LEGACY_SPIKE_NN]
        if getattr(spike_pkg, "nn", None) is current:
            delattr(spike_pkg, "nn")
    # expose shim submodules as attributes of their parents (``import src.models.nn.dropout`` style)
    for name, mod in list(sys.modules.items()):
        if name.startswith("src.") and getattr(mod, _SHIM_FLAG, False):
            parent, _, child = name.rpartition(".")
            if parent in sys.modules:
                setattr(sys.modules[parent], child, mod)


def _is_type_checking_block(node: ast.AST) -> bool:
    test = getattr(node, "test", None)
    return isinstance(node, ast.If) and (
        (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
        or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"))


def _imports(tree: ast.AST):
    """Yield ``(module, names)`` for absolute imports outside ``if TYPE_CHECKING:`` blocks."""
    stack = [tree]
    while stack:
        node = stack.pop()
        if _is_type_checking_block(node):
            stack.extend(node.orelse)  # type: ignore[attr-defined]
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, ()
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module, tuple(a.name for a in node.names if a.name != "*")
        stack.extend(ast.iter_child_nodes(node))


def missing_imports(source: str, legacy_spike_nn: bool = False) -> list[str]:
    """Return a description of each import in ``source`` that cannot be satisfied.

    Imports the named modules (and checks imported names) without running the
    program. Call after :func:`install_legacy_spike_alias`.
    """
    problems: dict[str, str] = {}
    for module, names in _imports(ast.parse(source)):
        if module == "__future__" or (module == LEGACY_SPIKE_NN and not legacy_spike_nn):
            continue
        try:
            mod = importlib.import_module(module)
        except Exception as exc:  # ImportError, or e.g. a Triton/CUDA error raised at import
            problems.setdefault(module, f"{module} ({type(exc).__name__}: {exc})")
            continue
        for name in names:
            if hasattr(mod, name):
                continue
            try:
                importlib.import_module(f"{module}.{name}")
            except Exception:
                problems.setdefault(f"{module}.{name}", f"{module}.{name} (name not found in installed {module})")
    return list(problems.values())


def _missing_message(problems: list[str]) -> str:
    tops = sorted({p.split(" ")[0].split(".")[0] for p in problems})
    pips = [_PIP_NAMES.get(t, t) for t in tops if t != "src"]
    lines = ["This program imports modules that are not available:"] + [f"  - {p}" for p in problems]
    if pips:
        lines.append(f"Install them (archived zoo programs: pip install 'openarchevo[zoo]'; "
                     f"needed here: {', '.join(pips)}).")
    if "fla" in tops:
        lines.append("flash-linear-attention needs Triton; most FLA layers only run on a CUDA GPU.")
    if "src" in tops:
        lines.append("src.models.* modules other than spike.neuron, spike.surrogate, spike.nn and "
                     "nn.dropout are not provided by the legacy shim.")
    lines.append("Pass check_imports=False to load anyway (in-file placeholder fallbacks may then be used).")
    return "\n".join(lines)


def load_block_class(program: Union[str, Path], class_name: str = BLOCK_CLASS, *,
                     check_imports: bool = True, legacy_spike_nn: bool = False) -> type[nn.Module]:
    """Import a program (file path or source string) and return its block class.

    The source is executed as a fresh module. Source strings are registered with
    ``linecache`` so tracebacks, ``inspect`` and TorchDynamo work without a
    temporary file.

    ``check_imports``: raise :class:`ImportError` with install instructions if any
    imported module (or imported name) is unavailable, instead of letting the
    program fall back to placeholders. ``legacy_spike_nn``: provide
    ``src.models.spike.nn.SpikingLinear`` (see :func:`install_legacy_spike_alias`).
    """
    install_legacy_spike_alias(spike_nn=legacy_spike_nn)
    is_path = isinstance(program, Path) or ("\n" not in program and program.endswith(".py"))
    source = Path(program).read_text(encoding="utf-8") if is_path else str(program)
    if check_imports:
        problems = missing_imports(source, legacy_spike_nn=legacy_spike_nn)
        if problems:
            raise ImportError(_missing_message(problems))
    module_name = f"oae_program_{hashlib.sha1(source.encode()).hexdigest()[:16]}"
    filename = str(Path(program).resolve()) if is_path else f"<{module_name}>"
    if not is_path:
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    module = types.ModuleType(module_name)
    module.__file__ = filename
    sys.modules[module_name] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    block_cls = getattr(module, class_name, None)
    if block_cls is None:
        raise AttributeError(f"Program does not define {class_name}.")
    return block_cls
