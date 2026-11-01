"""VT→QD strategy bridge: compile a Vibe-Trading run directory into a
QuantDinger Strategy API V2 source.

Why a generator, not an adapter
-------------------------------
The two strategy contracts are opposite shapes:

- VT ``signal_engine.py`` is *vectorized*: ``SignalEngine().generate(data_map)``
  receives one frame per instrument for the whole period and returns a dict of
  target-weight Series. No per-bar loop, no ordering.
- QD Strategy API V2 is *event-driven*: ``initialize(context)`` declares a
  static universe + subscriptions, and ``handle_data(context, data)`` runs once
  per closed bar with ``get_history(count, frequency, field, symbol)`` and
  ``order_target_percent(symbol, target, reason=...)``.

Since a VT engine only needs ``data_map`` to produce weights, the honest
translation is to *embed the engine verbatim* into a V2 shell: on each bar,
reconstruct a trailing window of frames (QD serves history), build the
data_map, call ``generate()``, and translate the latest weights into
``order_target_percent`` calls. Look-ahead safety comes from VT's own
contract — ``generate`` is expected to emit weights aligned to the *last*
row of its inputs — so the shell always trims history to the current bar.

Generated source requirements (from QD's STRATEGY_DEV_GUIDE):
- triple-quoted docstring, first line = strategy name
- ``initialize(context)`` with ``context.set_universe`` + ``context.subscribe``
- ``handle_data(context, data)`` as the driving callback
- ``get_history(count, frequency, field, symbol)`` — count-first, no fields=
- no leveraged instruments unless declared (we emit long-only weights)

Usage:
    from src.tools.qd_bridge import compile_vt_run_to_qd
    src = compile_vt_run_to_qd(run_dir="runs/my-strat", name="my-strat")

The returned source string can be pasted into QD's Strategy IDE, or fed to
their MCP ``compile_strategy_code`` / ``save_strategy_source`` tools.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

__all__ = ["QDBridgeError", "compile_vt_run_to_qd", "qd_source_for_engine"]


class QDBridgeError(RuntimeError):
    """A VT run directory could not be translated into a QD V2 source."""


#: QD instrument prefix by VT market/source. QD symbols look like
#: ``USStock:SPY``, ``Crypto:BTC-USDT``. VT sources are loader names
#: (yfinance, okx, ccxt, akshare...) — market is the better key when present.
_MARKET_TO_PREFIX = {
    "us": "USStock",
    "hk": "HKStock",
    "cn": "CNStock",
    "crypto": "Crypto",
    "a": "CNStock",
}

#: Default QD frequency from VT engine hints. VT engines are daily-bar
#: vectorized; 1d is the only honest default.
_DEFAULT_FREQUENCY = "1d"

#: How many trailing bars the shell feeds the embedded engine. Must cover
#: the engine's warmup; VT manifests estimate this (strategy_manifest), and
#: callers can override via config ``qd_warmup`` / the ``warmup`` argument.
_DEFAULT_WARMUP = 260


def _qd_symbol(code: str, market: str | None) -> str:
    code = (code or "").strip()
    if not code:
        raise QDBridgeError("empty symbol in VT config codes")
    if ":" in code:
        return code  # already QD-shaped
    prefix = _MARKET_TO_PREFIX.get((market or "").strip().lower(), "USStock")
    return f"{prefix}:{code}"


def _split_engine_parts(engine_src: str) -> tuple[str, str, list[str], list[str]]:
    """Split a VT engine module into (class block, helper defs, literal assigns,
    non-literal assign setups).

    QD's sandbox contract (backend ``_validate_safe_module_top_level``) allows
    at top level ONLY: function defs, imports, string docstrings, *literal*
    assignments, and ``pass``. Notably **classes are forbidden** at top level.

    Translation applied here:
    - imports + function defs stay top-level (allowed);
    - module-level *literal* assignments stay top-level verbatim (allowed);
    - module-level *non-literal* assignments (``PARAMS = load()``) become
      ``global NAME; NAME = <expr>`` lines inside ``initialize`` — executed
      at QD compile time before the class body is defined;
    - anything else top-level is dropped (the sandbox would reject it).

    Values are never executed here — only AST-read.
    """
    try:
        tree = ast.parse(engine_src)
    except SyntaxError as exc:
        raise QDBridgeError(f"signal_engine.py does not parse: {exc}") from None

    class_blocks: list[str] = []
    helper_parts: list[str] = []
    literal_assigns: list[str] = []
    nonliteral_setups: list[str] = []
    lines = engine_src.splitlines()

    def _is_literal(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return True
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (str, bytes, int, float, complex, bool, type(None)))
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return all(_is_literal(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            return all(
                k is not None and _is_literal(k) and _is_literal(v)
                for k, v in zip(node.keys, node.values)
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return isinstance(node.operand, ast.Constant) and isinstance(
                node.operand.value, (int, float, complex)
            )
        return False

    def _expr_src(node: ast.AST) -> str:
        # Reconstruct the expression source with ast.unparse so the
        # translated module stays valid Python (avoids raw line-slice
        # drift from comments/blank lines inside the expression range).
        return ast.unparse(node)

    for node in tree.body:
        seg = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        if isinstance(node, ast.ClassDef):
            class_blocks.append(seg)
        elif isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef)):
            helper_parts.append(seg)
        elif isinstance(node, ast.Assign):
            # QD's top-level gate allows literal assigns including tuple
            # unpacking (``FAST, SLOW = 20, 50``): the Tuple target is a
            # "safe assignment target" and its elements are literal-checked
            # by _is_safe_static_value. Keep them when the RHS is literal.
            targets = []
            tuple_ok = True
            for t in node.targets:
                if isinstance(t, ast.Name):
                    targets.append(t)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    inner = [
                        e for e in (t.elts or []) if isinstance(e, ast.Name)
                    ]
                    if len(inner) != len(t.elts or []):
                        tuple_ok = False
                        break
                    targets.extend(inner)
                else:
                    tuple_ok = False
                    break
            if tuple_ok and targets and _is_literal(node.value):
                literal_assigns.append(seg)
            elif tuple_ok and targets:
                for target in targets:
                    nonliteral_setups.append(
                        f"    global {target.id}\n    {target.id} = "
                        + _expr_src(node.value)
                    )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None and _is_literal(node.value):
                # annotations need not survive; plain assign is allowed
                literal_assigns.append(f"{node.target.id} = {_expr_src(node.value)}")
            elif node.value is not None:
                nonliteral_setups.append(
                    f"    global {node.target.id}\n    {node.target.id} = "
                    + _expr_src(node.value, lines)
                )
        # else: bare expressions/calls at module level are dropped.

    if not class_blocks:
        raise QDBridgeError(
            "signal_engine.py defines no class; expected `class SignalEngine:`"
        )
    return (
        "\n\n".join(class_blocks),
        "\n".join(helper_parts),
        literal_assigns,
        nonliteral_setups,
    )


def _class_to_engine_functions(class_block: str) -> list[str]:
    """Rewrite a VT SignalEngine class into module-level engine functions.

    QD's sandbox has no ``__build_class__`` — classes cannot be *defined*
    anywhere in strategy code, not even inside ``initialize``. The VT
    contract, however, only needs ``SignalEngine().generate(data_map)``,
    and the common VT engine is stateless or instance-attribute-light.

    Translation (AST-based, never executed):

    - ``def generate(self, data_map)`` → ``def _vt_generate(data_map)``
      (leading ``self`` dropped);
    - every other method ``def name(self, ...)`` → ``def _vt_name(...)``
      (leading ``self`` dropped) so ``generate`` can call them;
    - ``self.<attr>`` reads/writes become module-global ``_vt_<attr>``
      (with ``global`` declarations on writes);
    - class-level literal assigns become module-level assigns (allowed);
    - the engine is instantiated once in ``initialize`` as a *function
      reference*: ``g.engine = _vt_engine()`` returns a tiny shim whose
      ``generate`` closes over the rewritten functions — also expressed
      as plain defs, so no class exists anywhere.

    Raises QDBridgeError if ``generate`` is missing or the class uses
    features this translation cannot express (inheritance, decorators,
    properties).
    """
    try:
        tree = ast.parse(class_block)
    except SyntaxError as exc:
        raise QDBridgeError(f"SignalEngine class does not parse: {exc}") from None

    cls = tree.body[0]
    if not isinstance(cls, ast.ClassDef):
        raise QDBridgeError("expected a class block")
    if cls.bases or cls.keywords or cls.decorator_list:
        raise QDBridgeError(
            "SignalEngine uses inheritance/decorators; the QD translation "
            "supports only a plain stateless class"
        )

    self_attrs: set[str] = set()
    out_funcs: list[str] = []
    class_consts: list[str] = []

    class _T(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            if node.id == "self":
                raise QDBridgeError("self used outside method args")
            return node

    for item in cls.body:
        if isinstance(item, ast.Pass):
            continue
        if isinstance(item, ast.Assign) and all(
            isinstance(t, ast.Name) for t in item.targets
        ):
            class_consts.append(ast.unparse(ast.Module(body=[item], type_ignores=[])))
            continue
        if isinstance(item, ast.AnnAssign):
            class_consts.append(ast.unparse(ast.Module(body=[ast.Assign(targets=[item.target], value=item.value)], type_ignores=[])) if item.value else "")
            continue
        if not isinstance(item, ast.FunctionDef):
            raise QDBridgeError(
                f"SignalEngine class body contains a {type(item).__name__}; "
                "the QD translation supports only methods and literal assigns"
            )
        if item.decorator_list:
            raise QDBridgeError("decorated methods are not translatable")

        args = item.args
        pos_args = list(args.posonlyargs) + list(args.args)
        if not pos_args or pos_args[0].arg != "self":
            raise QDBridgeError(f"method {item.name!r} lacks leading self")

        # Drop the leading `self` argument from the signature. Defaults
        # align to the END of the positional args, and `self` (first,
        # never defaulted) sits before them, so no default rebind is needed.
        if args.posonlyargs:
            args.posonlyargs = args.posonlyargs[1:]
        else:
            args.args = args.args[1:]

        # Collect self.attr uses to hoist as module globals.
        for node in ast.walk(item):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            ):
                self_attrs.add(node.attr)

        # Rewrite: drop self arg, rewrite self.attr -> _vt_attr, self.method -> _vt_method
        class _Rewrite(ast.NodeTransformer):
            def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
                self.generic_visit(node)
                if isinstance(node.value, ast.Name) and node.value.id == "self":
                    return ast.copy_location(
                        ast.Name(id=f"_vt_{node.attr}", ctx=node.ctx), node
                    )
                return node

            def visit_Call(self, node: ast.Call) -> ast.expr:
                self.generic_visit(node)
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                ):
                    return ast.copy_location(
                        ast.Call(
                            func=ast.Name(id=f"_vt_{node.func.attr}", ctx=ast.Load()),
                            args=node.args,
                            keywords=node.keywords,
                        ),
                        node,
                    )
                return node

        new_item = _Rewrite().visit(ast.fix_missing_locations(item))

        # global declarations for any _vt_attr writes
        written: set[str] = set()
        for node in ast.walk(new_item):
            if (
                isinstance(node, ast.Name)
                and node.id.startswith("_vt_")
                and isinstance(node.ctx, ast.Store)
            ):
                written.add(node.id)
        if written:
            glob = ast.Global(names=sorted(written))
            new_item.body.insert(0, ast.fix_missing_locations(glob))

        # Rename: generate -> _vt_generate; name -> _vt_name
        new_item.name = "_vt_generate" if item.name == "generate" else f"_vt_{item.name}"
        out_funcs.append(ast.unparse(new_item))

    if not any(f.startswith("def _vt_generate(") for f in out_funcs):
        raise QDBridgeError("SignalEngine class has no generate() method")

    return class_consts + out_funcs


_SHELL_TEMPLATE = '''\
def initialize(context):
    g.symbols = {symbols!r}
    g.engine = SignalEngine(){constants}
    context.set_universe(list(g.symbols))
    context.subscribe(
        frequency={frequency!r},
        fields=["open", "high", "low", "close", "volume"],
    )
    context.set_warmup({warmup})
    context.set_benchmark({benchmark!r})


def handle_data(context, data):
    import pandas as pd

    warmup = int(context.params.get("vt_warmup", {warmup}))
    data_map = {{}}
    for symbol in g.symbols:
        frame = get_history(warmup, {frequency!r}, "*", symbol)
        if frame is None or len(frame) == 0:
            continue
        data_map[symbol] = frame
    if not data_map:
        return
    weights = g.engine.generate(data_map)
    for symbol, target in weights.items():
        qd_sym = g.symbols.get(symbol, symbol)
        if hasattr(target, "iloc"):
            target = float(target.iloc[-1]) if len(target) else 0.0
        elif hasattr(target, "item"):
            target = float(target)
        else:
            target = float(target)
        order_target_percent(
            qd_sym,
            target,
            reason="vt_signal_engine",
        )
'''


def qd_source_for_engine(
    *,
    name: str,
    symbols: list[str],
    engine_src: str,
    frequency: str = _DEFAULT_FREQUENCY,
    warmup: int = _DEFAULT_WARMUP,
    benchmark: str | None = None,
) -> str:
    """Generate a QD Strategy API V2 source embedding a VT SignalEngine.

    QD's sandbox (``_validate_safe_module_top_level``) allows at module top
    level ONLY: function defs, imports, docstrings, *literal* assignments,
    and ``pass``. **Classes are forbidden at top level.** ``initialize``
    runs at compile time (manifest discovery), so the VT engine class is
    defined *inside* ``initialize`` — where it is also instantiated.

    Generated source layout:

    1. module docstring (strategy name, universe, risk summary)
    2. VT imports + helper defs (allowed top level)
    3. VT module-level *literal* assignments (``FAST = 20`` — allowed)
    4. ``initialize(context)``: non-literal setups as ``global`` writes,
       the indented VT class body, ``g.engine = SignalEngine()``,
       universe/subscription/warmup/benchmark
    5. ``handle_data(context, data)``: trailing data_map from QD history
       → ``g.engine.generate(data_map)`` → ``order_target_percent``
    """
    class_block, helper_defs, literal_assigns, nonliteral_setups = _split_engine_parts(
        engine_src
    )

    # QD's sandbox has no __build_class__: rewrite the class into plain
    # module-level engine functions (see _class_to_engine_functions).
    engine_funcs = _class_to_engine_functions(class_block)

    bench = benchmark or (symbols[0].split(":")[-1] if symbols else None)
    bench_qd = None
    if bench:
        if ":" in str(bench):
            bench_qd = str(bench)
        else:
            # benchmark shares the universe's market prefix
            prefix = (symbols[0].split(":")[0] if symbols else "USStock")
            bench_qd = f"{prefix}:{bench}"

    _dq3 = '"' * 3
    header = (
        f"{_dq3}{name}\n"
        "Vibe-Trading signal_engine translated to QuantDinger Strategy API V2.\n"
        f"Universe: {', '.join(symbols)}\n"
        f"Frequency: {frequency} bars; long-only target weights from VT generate().\n"
        "Risk: weights come from the embedded VT engine; QD protections (stop_loss)\n"
        "are declared by the engine's order calls if any.\n"
        f"{_dq3}"
    )

    # g.engine shim: a dict-like namespace bound to _vt_generate, defined
    # without classes (QD sandbox has no __build_class__).
    init_parts: list[str] = []
    if nonliteral_setups:
        init_parts.append("\n".join(nonliteral_setups))
    init_parts.append("    g.symbols = " + repr(symbols))
    init_parts.append("    context.set_universe(list(g.symbols))")
    init_parts.append(
        "\n".join(
            [
                "    context.subscribe(",
                f"        frequency={frequency!r},",
                '        fields=["open", "high", "low", "close", "volume"],',
                "    )",
            ]
        )
    )
    init_parts.append(f"    context.set_warmup({warmup})")
    init_parts.append(f"    context.set_benchmark({bench_qd!r})")

    handle_src = (
        "def handle_data(context, data):\n"
        "    import pandas as pd\n"
        "    warmup = int(context.params.get('vt_warmup', " + str(warmup) + "))\n"
        "    data_map = {}\n"
        "    for symbol in g.symbols:\n"
        "        frame = get_history(warmup, " + repr(frequency) + ", ['open', 'high', 'low', 'close', 'volume'], symbol)\n"
        "        if frame is None or len(frame) == 0:\n"
        "            continue\n"
        "        data_map[symbol] = frame\n"
        "    if not data_map:\n"
        "        return\n"
        "    weights = _vt_generate(data_map)\n"
        "    for symbol, target in weights.items():\n"
        "        qd_sym = g.symbols.get(symbol, symbol) if hasattr(g.symbols, 'get') else symbol\n"
        "        if hasattr(target, 'iloc'):\n"
        "            target = float(target.iloc[-1]) if len(target) else 0.0\n"
        "        elif hasattr(target, 'item'):\n"
        "            target = float(target)\n"
        "        else:\n"
        "            target = float(target)\n"
        "        order_target_percent(\n"
        "            qd_sym,\n"
        "            target,\n"
        "            reason='vt_signal_engine',\n"
        "        )"
    )

    parts = [header]
    if helper_defs.strip():
        parts.append(helper_defs)
    if literal_assigns:
        parts.append("\n".join(literal_assigns))
    parts.append("\n".join(engine_funcs))
    parts.append("def initialize(context):\n" + "\n".join(init_parts))
    parts.append(handle_src)
    return "\n\n".join(parts) + "\n"


def compile_vt_run_to_qd(
    *,
    run_dir: str | Path,
    name: str = "",
    warmup: int | None = None,
) -> dict[str, Any]:
    """Compile a VT run directory (config.json + code/signal_engine.py) to QD V2.

    Returns ``{"status": "ok", "name": ..., "source": <qd v2 python>,
    "symbols": [...], "warmup": n, "frequency": "1d"}``.

    Raises / errors (as dict): invalid or missing inputs.
    """
    root = Path(run_dir).expanduser().resolve()
    config_path = root / "config.json"
    engine_path = root / "code" / "signal_engine.py"

    if not config_path.is_file():
        return {"status": "error", "error": f"missing config.json under {root}"}
    if not engine_path.is_file():
        return {"status": "error", "error": f"missing code/signal_engine.py under {root}"}

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"status": "error", "error": f"config.json unparseable: {exc}"}
    if not isinstance(config, dict):
        return {"status": "error", "error": "config.json must be a JSON object"}

    codes = config.get("codes") or []
    if not isinstance(codes, list) or not codes:
        return {
            "status": "error",
            "error": "config codes missing or empty; QD needs a static universe",
        }
    market = config.get("market") or _market_from_source(config.get("source"))

    try:
        symbols = [_qd_symbol(str(c), market) for c in codes]
    except QDBridgeError as exc:
        return {"status": "error", "error": str(exc)}

    engine_src = engine_path.read_text(encoding="utf-8")

    if warmup is None:
        warmup = _estimate_warmup(engine_src, config)

    strategy_name = (name or root.name).strip()
    try:
        source = qd_source_for_engine(
            name=strategy_name,
            symbols=symbols,
            engine_src=engine_src,
            warmup=warmup,
            benchmark=config.get("benchmark"),
        )
    except QDBridgeError as exc:
        return {"status": "error", "error": str(exc)}

    return {
        "status": "ok",
        "name": strategy_name,
        "source": source,
        "symbols": symbols,
        "warmup": warmup,
        "frequency": _DEFAULT_FREQUENCY,
    }


def _market_from_source(source: str | None) -> str | None:
    """Best-effort market key from a VT loader name."""
    s = (source or "").lower()
    if "okx" in s or "ccxt" in s or "binance" in s:
        return "crypto"
    if "akshare" in s or "tushare" in s or "baostock" in s:
        return "cn"
    return "us"


def _estimate_warmup(engine_src: str, config: dict[str, Any]) -> int:
    """Longest rolling window referenced by the engine, floor 120."""
    declared = config.get("warmup_bars") or config.get("qd_warmup")
    if isinstance(declared, (int, float)) and declared > 0:
        return int(declared)
    windows = [
        int(m)
        for pattern in (
            r"rolling\(\s*(\d+)",
            r"window\s*=\s*(\d+)",
            r"\.mean\(\s*(\d+)\s*\)",
            r"\.shift\(\s*(\d+)\s*\)",
            r"tail\(\s*(\d+)\s*\)",
        )
        for m in re.findall(pattern, engine_src)
    ]
    return max([*windows, 120])