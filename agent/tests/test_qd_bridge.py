"""Regression tests for the VT→QD bridge (src/tools/qd_bridge.py).

Every test encodes a real contract discovery from the 2026-11-01 live E2E
against a running QuantDinger v5 stack. Each failure mode below was hit
for real; these tests keep the translation honest against all four
sandbox rules plus the two runtime contract details.

QD sandbox rules (backend contract.py ``_validate_safe_module_top_level``):
  1. module top level allows ONLY: function defs, imports, string
     docstrings, *literal* assignments, and ``pass``;
  2. classes are NOT in the allowlist (``topLevelExecutableNotAllowed``);
  3. ``__build_class__`` is absent from sandbox builtins — classes cannot
     be defined anywhere, not even inside ``initialize``;
  4. ``g`` strategy state must be JSON-serializable (no function values).

QD runtime contracts:
  5. ``get_history(count, frequency, field, symbol)`` — ``field`` accepts
     a field name or list; ``"*"`` is NOT a wildcard (filters to zero
     columns → ``KeyError: 'close'``);
  6. strategy-create payloads use camelCase (``sourceId``,
     ``initialCapital``, ``executionMode``); backtest payloads use
     ``startDate``/``endDate`` — snake_case fields are rejected as
     ``strategyV2.unsupportedFields``.
"""

from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.tools.qd_bridge import (
    QDBridgeError,
    compile_vt_run_to_qd,
    qd_source_for_engine,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUAL_EMA_ENGINE = '''\
import pandas as pd

FAST, SLOW = 20, 50

def _ma(series, n):
    return series.rolling(n).mean()

class SignalEngine:
    def generate(self, data_map):
        weights = {}
        for sym, df in data_map.items():
            close = df["close"]
            fast, slow = _ma(close, FAST), _ma(close, SLOW)
            weights[sym] = (fast > slow).astype(float) * 0.5
        return weights
'''


def _make_run_dir(tmp: Path, *, codes: list[str] | None = None,
                  engine: str = _DUAL_EMA_ENGINE) -> Path:
    rd = tmp / "strat"
    (rd / "code").mkdir(parents=True)
    (rd / "config.json").write_text(
        json.dumps({
            "source": "yfinance",
            "codes": codes or ["SPY", "QQQ"],
            "market": "us",
            "start_date": "2023-01-01",
            "end_date": "2025-12-31",
            "initial_capital": 100000,
            "benchmark": "SPY",
        }),
        encoding="utf-8",
    )
    (rd / "code" / "signal_engine.py").write_text(engine, encoding="utf-8")
    return rd


def _top_level_kinds(src: str) -> list[str]:
    return [type(node).__name__ for node in ast.parse(src).body]


# ---------------------------------------------------------------------------
# Rule 1 + 2: module top level must contain only allowed node kinds
# ---------------------------------------------------------------------------

def test_top_level_only_allowed_nodes() -> None:
    src = qd_source_for_engine(
        name="t", symbols=["USStock:SPY"], engine_src=_DUAL_EMA_ENGINE,
    )
    # QD's gate allows literal top-level assigns; Assign must appear only
    # with literal values (checked separately below).
    allowed = {"FunctionDef", "AsyncFunctionDef", "Import", "ImportFrom", "Expr", "Assign"}
    kinds = _top_level_kinds(src)
    assert set(kinds) <= allowed, f"forbidden top-level nodes: {set(kinds) - allowed}"


def test_top_level_exprs_are_only_docstring() -> None:
    src = qd_source_for_engine(
        name="t", symbols=["USStock:SPY"], engine_src=_DUAL_EMA_ENGINE,
    )
    for node in ast.parse(src).body:
        if isinstance(node, ast.Expr):
            assert isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, str
            ), "top-level Expr must be a docstring only"


def test_no_class_anywhere_in_output() -> None:
    # Rule 2 + 3: no ClassDef at any nesting level — QD has no __build_class__.
    src = qd_source_for_engine(
        name="t", symbols=["USStock:SPY"], engine_src=_DUAL_EMA_ENGINE,
    )
    for node in ast.walk(ast.parse(src)):
        assert not isinstance(node, ast.ClassDef), "class definition in output"


def test_top_level_assigns_are_literal() -> None:
    src = qd_source_for_engine(
        name="t", symbols=["USStock:SPY"], engine_src=_DUAL_EMA_ENGINE,
    )
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                # Name or a tuple/list of Names (e.g. FAST, SLOW = 20, 50)
                if isinstance(target, ast.Name):
                    continue
                assert isinstance(target, (ast.Tuple, ast.List)) and all(
                    isinstance(e, ast.Name) for e in target.elts
                ), f"unsupported target: {ast.unparse(target)[:40]}"
            # literal value check mirrors QD's _is_safe_static_value
            v = node.value
            ok = isinstance(v, (ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.UnaryOp))
            assert ok, f"non-literal top-level assign: {ast.unparse(v)[:60]}"


# ---------------------------------------------------------------------------
# Rule 4: g state must stay JSON-serializable (no function/closure values)
# ---------------------------------------------------------------------------

def test_initialize_assigns_json_compatible_g_state() -> None:
    src = qd_source_for_engine(
        name="t", symbols=["USStock:SPY", "USStock:QQQ"],
        engine_src=_DUAL_EMA_ENGINE,
    )
    tree = ast.parse(src)
    init = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "initialize"
    )
    for node in ast.walk(init):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "g"
        ):
            value = node.value
            assert not isinstance(value, ast.Lambda), "lambda stored in g"
            assert not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
            ), f"call result stored in g: {ast.unparse(value)[:50]}"


# ---------------------------------------------------------------------------
# Rule 5: get_history must use an explicit field list, never "*"
# ---------------------------------------------------------------------------

def test_get_history_uses_explicit_fields() -> None:
    src = qd_source_for_engine(
        name="t", symbols=["USStock:SPY"], engine_src=_DUAL_EMA_ENGINE,
    )
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "get_history"
        ):
            fields = node.args[2] if len(node.args) >= 3 else None
            assert fields is not None, "get_history without field argument"
            assert not (
                isinstance(fields, ast.Constant) and fields.value == "*"
            ), 'get_history(field="*") — not a wildcard, fails at runtime'
            if isinstance(fields, (ast.List, ast.Tuple)):
                names = [
                    e.value for e in fields.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                ]
                assert "close" in names, "field list must include close"


def test_handle_data_calls_vt_generate_directly() -> None:
    # After rule 3 (no classes) the engine is a plain function; after
    # rule 4 (JSON g) the call must not go through g.engine.
    src = qd_source_for_engine(
        name="t", symbols=["USStock:SPY"], engine_src=_DUAL_EMA_ENGINE,
    )
    assert "_vt_generate(data_map)" in src
    assert "g.engine" not in src


# ---------------------------------------------------------------------------
# Class→function rewrite correctness
# ---------------------------------------------------------------------------

def test_self_argument_is_dropped() -> None:
    src = qd_source_for_engine(
        name="t", symbols=["USStock:SPY"], engine_src=_DUAL_EMA_ENGINE,
    )
    tree = ast.parse(src)
    gen = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_vt_generate"
    )
    arg_names = [a.arg for a in list(gen.args.posonlyargs) + list(gen.args.args)]
    assert "self" not in arg_names, f"self survived: {arg_names}"
    assert arg_names == ["data_map"]


def test_self_attribute_and_method_refs_rewritten() -> None:
    engine = '''\
import pandas as pd

class SignalEngine:
    def generate(self, data_map):
        for sym, df in data_map.items():
            if self._ok(df):
                df["close"] = self._clean(df["close"])
        return {s: self._w(df) for s, df in data_map.items()}

    def _ok(self, df):
        return len(df) > 0

    def _clean(self, s):
        return s.dropna()

    def _w(self, df):
        return 0.5
'''
    src = qd_source_for_engine(
        name="t", symbols=["USStock:SPY"], engine_src=engine,
    )
    assert "self." not in src, "self.attr reference survived the rewrite"
    # helper methods are hoisted with '_vt_' prefix on their original names
    assert "_vt__ok(" in src and "_vt__clean(" in src and "_vt__w(" in src
    ast.parse(src)  # still valid python


def test_tuple_unpack_constants_survive() -> None:
    # FAST, SLOW = 20, 50 must be kept as a top-level literal assign —
    # dropping it was the "name 'FAST' is not defined" failure.
    src = qd_source_for_engine(
        name="t", symbols=["USStock:SPY"], engine_src=_DUAL_EMA_ENGINE,
    )
    assert "FAST, SLOW = 20, 50" in src


def test_nonliteral_module_assigns_become_initialize_globals() -> None:
    engine = '''\
import pandas as pd

import json as _json
PARAMS = _json.loads('{"a": 1}')

class SignalEngine:
    def generate(self, data_map):
        return {s: PARAMS["a"] for s in data_map}
'''
    src = qd_source_for_engine(
        name="t", symbols=["USStock:SPY"], engine_src=engine,
    )
    assert "global PARAMS" in src, "non-literal assign not hoisted into initialize"
    assert "PARAMS =" in src
    # the top-level assign must be gone (it would violate rule 1)
    top_assigns = [
        n for n in ast.parse(src).body
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PARAMS" for t in n.targets
        )
    ]
    assert not top_assigns, "non-literal PARAMS assign left at top level"


# ---------------------------------------------------------------------------
# Untranslatable inputs raise (never silently mis-translate)
# ---------------------------------------------------------------------------

def test_inheritance_rejected() -> None:
    engine = '''\
class SignalEngine(dict):
    def generate(self, data_map):
        return {}
'''
    with pytest.raises(QDBridgeError):
        qd_source_for_engine(
            name="t", symbols=["USStock:SPY"], engine_src=engine,
        )


def test_missing_generate_rejected() -> None:
    engine = '''\
class SignalEngine:
    def compute(self, data_map):
        return {}
'''
    with pytest.raises(QDBridgeError):
        qd_source_for_engine(
            name="t", symbols=["USStock:SPY"], engine_src=engine,
        )


def test_no_class_rejected() -> None:
    engine = "def generate(data_map):\n    return {}\n"
    with pytest.raises(QDBridgeError):
        qd_source_for_engine(
            name="t", symbols=["USStock:SPY"], engine_src=engine,
        )


# ---------------------------------------------------------------------------
# compile_vt_run_to_qd: symbol mapping + error paths (dict, not raise)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("market,code,expected", [
    ("us", "SPY", "USStock:SPY"),
    ("cn", "600519", "CNStock:600519"),
    ("hk", "0700", "HKStock:0700"),
    ("crypto", "BTC-USDT", "Crypto:BTC-USDT"),
])
def test_symbol_mapping(market: str, code: str, expected: str, tmp_path: Path) -> None:
    rd = _make_run_dir(tmp_path, codes=[code])
    cfg = json.loads((rd / "config.json").read_text())
    cfg["market"] = market
    (rd / "config.json").write_text(json.dumps(cfg))
    out = compile_vt_run_to_qd(run_dir=rd)
    assert out["status"] == "ok", out
    assert expected in out["source"], out["symbols"]


def test_market_inferred_from_source_when_missing(tmp_path: Path) -> None:
    rd = _make_run_dir(tmp_path, codes=["BTC-USDT"])
    cfg = json.loads((rd / "config.json").read_text())
    cfg.pop("market")
    cfg["source"] = "okx"
    (rd / "config.json").write_text(json.dumps(cfg))
    out = compile_vt_run_to_qd(run_dir=rd)
    assert out["status"] == "ok", out
    assert "Crypto:BTC-USDT" in out["source"]


def test_benchmark_gets_market_prefix(tmp_path: Path) -> None:
    rd = _make_run_dir(tmp_path)
    out = compile_vt_run_to_qd(run_dir=rd)
    assert out["status"] == "ok", out
    assert "context.set_benchmark('USStock:SPY')" in out["source"]


def test_error_missing_config(tmp_path: Path) -> None:
    out = compile_vt_run_to_qd(run_dir=tmp_path / "nope")
    assert out["status"] == "error"
    assert "config.json" in out["error"]


def test_error_missing_engine(tmp_path: Path) -> None:
    rd = tmp_path / "x"
    rd.mkdir()
    (rd / "config.json").write_text("{}")
    out = compile_vt_run_to_qd(run_dir=rd)
    assert out["status"] == "error"
    assert "signal_engine.py" in out["error"]


def test_error_empty_codes(tmp_path: Path) -> None:
    rd = _make_run_dir(tmp_path)
    cfg = json.loads((rd / "config.json").read_text())
    cfg["codes"] = []
    (rd / "config.json").write_text(json.dumps(cfg))
    out = compile_vt_run_to_qd(run_dir=rd)
    assert out["status"] == "error"


def test_error_no_config_codes_key(tmp_path: Path) -> None:
    rd = tmp_path / "y"
    (rd / "code").mkdir(parents=True)
    (rd / "config.json").write_text('{"source": "yfinance"}')
    (rd / "code" / "signal_engine.py").write_text(_DUAL_EMA_ENGINE)
    out = compile_vt_run_to_qd(run_dir=rd)
    assert out["status"] == "error"


# ---------------------------------------------------------------------------
# Warmup estimation
# ---------------------------------------------------------------------------

def test_warmup_estimated_from_engine_windows(tmp_path: Path) -> None:
    rd = _make_run_dir(tmp_path)
    out = compile_vt_run_to_qd(run_dir=rd)
    # engine uses rolling(50) → estimated max(window) floor 120
    assert out["warmup"] >= 50


def test_warmup_override(tmp_path: Path) -> None:
    rd = _make_run_dir(tmp_path)
    out = compile_vt_run_to_qd(run_dir=rd, warmup=260)
    assert out["warmup"] == 260


# ---------------------------------------------------------------------------
# End-to-end shape: the generated source parses and satisfies QD's contract
# ---------------------------------------------------------------------------

def test_generated_source_satisfies_qd_top_level_contract(tmp_path: Path) -> None:
    """The composite fixture through the full public entry point.

    Mirrors what the 2026-11-01 live E2E proved against the real compiler:
    the output module must pass QD's top-level walk AND stay runtime-valid
    (no NameError for constants, correct get_history field list).
    """
    rd = _make_run_dir(tmp_path)
    out = compile_vt_run_to_qd(run_dir=rd, name="dual-ema-vt")
    assert out["status"] == "ok", out
    src = out["source"]
    ast.parse(src)

    kinds = _top_level_kinds(src)
    assert set(kinds) <= {
        "FunctionDef", "AsyncFunctionDef", "Import", "ImportFrom", "Expr", "Assign",
    }
    assert "def initialize(context):" in src
    assert "def handle_data(context, data):" in src
    assert "context.set_universe(" in src
    assert "context.subscribe(" in src
    assert "order_target_percent(" in src
    # rule 5: explicit fields, includes close
    assert "['open', 'high', 'low', 'close', 'volume']" in src
    # constants survived (FAST defined at top level as literal assign)
    assert "FAST, SLOW = 20, 50" in src
    # no forbidden imports
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            mod = stripped.split()[1].split(".")[0].rstrip(",")
            assert mod not in {"os", "sys", "subprocess", "socket", "requests", "types"}, \
                f"forbidden import emitted: {mod}"