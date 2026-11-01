"""Manifest-driven scheduled paper-trading pass on the Alpaca paper account.

Closes the loop opened by the QuantDinger-inspired additions:

    compile_strategy → save_strategy_manifest → submit_paper_run(manifest_id)

One strategy definition (the manifest) now drives three proven surfaces:
the VT backtest, the QD backtest (via ``qd_bridge``), and — here — live
paper execution on the existing ``alpaca-paper-trade`` profile.

Design rules (stolen from ``src/scheduling/paper_run.py``, kept fixed):

* The manifest is content-addressed. Before any order, the run_dir is
  recompiled and its ``manifest_id`` compared — a drifted strategy never
  trades under a stale manifest id (fail closed).
* The engine contract is the VT runner contract: a no-arg ``SignalEngine``
  whose ``generate(data_map) -> dict[str, pd.Series]`` emits weights aligned
  to the *last* row of its inputs. The latest weight per symbol is the
  target for reconciliation.
* Bars come from the repository loader layer (``get_market_data``), not the
  broker feed. ``BTC-USDT`` loader data maps to the ``BTC/USD`` Alpaca order
  symbol; held/pending lookups normalize slashes away.
* Sizing is target-percent reconciliation: ``target_notional = weight ×
  equity`` versus current position value at last close; only the delta is
  ordered, with a skip threshold so a 0.01 weight wiggle does not churn.
* Long-only: a negative weight with ``allow_short`` unset sells to flat
  (shorting is out of paper scope); weights are clipped to the manifest's
  leverage policy with a hard per-order notional cap.
* Crypto orders use ``time_in_force="gtc"`` (Alpaca rejects ``day`` for
  crypto; run 2026-11-01: ``invalid crypto time_in_force``).
* Live profiles are unreachable from here: the profile id is fixed to
  ``alpaca-paper-trade`` — callers cannot widen it.
"""

from __future__ import annotations

import json
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.tools.strategy_manifest import ManifestError, compile_strategy

PROFILE_ID = "alpaca-paper-trade"

#: Hard per-order notional cap (USD). Mirrors paper_run.MAX_NOTIONAL_USD so a
#: misbehaving engine weight cannot push unbounded size at the sandbox.
MAX_NOTIONAL_USD = 5000.0

#: Reconciliation skip threshold: deltas under this notional are noise.
MIN_DELTA_NOTIONAL_USD = 50.0

#: Manifest engines run on daily bars here (VT backtests are daily by default).
_INTERVAL = "1D"
_LOOKBACK_DAYS = 400

_CRYPTO_SUFFIXES = ("USDT", "USD", "USDC")


def _norm(symbol: str) -> str:
    """Normalize an Alpaca symbol so ``BTC/USD`` == ``BTCUSD`` == ``btcusd``."""
    return str(symbol or "").strip().upper().replace("/", "")


def _alpaca_order_symbol(loader_symbol: str) -> tuple[str, bool]:
    """Map a VT loader symbol to its Alpaca order symbol.

    ``BTC-USDT`` → ``("BTC/USD", True)`` (crypto); ``SPY`` / ``AAPL.US`` →
    ``("SPY", False)`` / ``("AAPL", False)`` (US equity).
    """
    sym = str(loader_symbol or "").strip().upper()
    for suffix in _CRYPTO_SUFFIXES:
        if sym.endswith(f"-{suffix}"):
            base = sym[: -(len(suffix) + 1)]
            return f"{base}/USD", True
        if sym.endswith(suffix) and len(sym) > len(suffix):
            base = sym[: -len(suffix)]
            return f"{base}/USD", True
    if "." in sym:  # AAPL.US → AAPL
        sym = sym.split(".")[0]
    return sym, False


def _is_crypto_pair(order_symbol: str) -> bool:
    return "/" in str(order_symbol)


def _load_engine_class(run_dir: Path) -> type:
    """Import ``SignalEngine`` from ``run_dir/code/signal_engine.py``.

    The manifest was compiled from this file; the hash check in
    ``_verified_manifest`` already proved the content, so a plain
    importlib load (never ``exec`` of caller text) is safe here.
    """
    engine_path = Path(run_dir) / "code" / "signal_engine.py"
    if not engine_path.is_file():
        raise RuntimeError(f"missing engine file: {engine_path}")
    module_name = f"vt_paper_engine_{abs(hash(str(engine_path))) & 0xFFFFFFFF:x}"
    spec = importlib.util.spec_from_file_location(module_name, engine_path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib contract
        raise RuntimeError(f"cannot load engine module: {engine_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine_cls = getattr(module, "SignalEngine", None)
    if engine_cls is None or not isinstance(engine_cls, type):
        raise RuntimeError("signal_engine.py defines no class SignalEngine")
    return engine_cls


def _verified_manifest(manifest_id: str) -> dict[str, Any]:
    """Load a manifest and fail closed if its run_dir has drifted.

    Recompiles the manifest's run_dir and compares the content-addressed id.
    A strategy that changed on disk after compilation must never trade under
    the old manifest id — the caller re-compiles and submits the new id.
    """
    from src.tools.strategy_manifest import get_manifest_store

    store = get_manifest_store()
    manifest = store.get(manifest_id)
    run_dir = manifest.get("run_dir")
    if not run_dir:
        raise RuntimeError(f"manifest {manifest_id} has no run_dir")
    if not Path(run_dir).is_dir():
        raise RuntimeError(f"manifest run_dir no longer exists: {run_dir}")
    current = compile_strategy(run_dir=run_dir, name=manifest.get("name", ""))
    if current["manifest_id"] != manifest.get("manifest_id"):
        raise RuntimeError(
            "strategy drifted since compilation: manifest "
            f"{manifest.get('manifest_id')} != current {current['manifest_id']}; "
            "re-compile and submit the new manifest"
        )
    return manifest


def _load_rows(loader_symbol: str, interval: str, lookback_days: int) -> list[dict]:
    """Fetch chronological OHLCV rows through the repository loader layer."""
    from src.tools.market_data_tool import MarketDataTool

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=lookback_days)
    reply = MarketDataTool().execute(
        codes=[loader_symbol],
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        source="auto",
        interval=interval,
        max_rows=0,
    )
    payload = json.loads(reply)
    if payload.get("ok") is False or payload.get("error"):
        raise RuntimeError(str(payload.get("error") or payload)[:200])
    rows = payload.get(loader_symbol) or payload.get(str(loader_symbol).upper())
    if not isinstance(rows, list):
        return []
    return [
        row for row in rows
        if isinstance(row, dict) and isinstance(row.get("close"), (int, float))
    ]


def _account_equity() -> float | None:
    from src.trading.service import get_account

    account = get_account(PROFILE_ID)
    inner = account.get("account") or {}
    raw = inner.get("equity")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _positions_reply() -> dict:
    from src.trading.service import get_positions

    reply = get_positions(PROFILE_ID)
    if isinstance(reply.get("positions"), list):
        return reply
    return {"positions": reply.get("positions") or reply.get("data") or []}


def _held_map(positions_reply: dict) -> dict[str, float]:
    held: dict[str, float] = {}
    for row in positions_reply.get("positions") or []:
        if not isinstance(row, dict):
            continue
        symbol = _norm(row.get("symbol"))
        qty = row.get("quantity", row.get("qty"))
        try:
            qty = float(qty or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        if symbol and qty:
            held[symbol] = qty
    return held


def _latest_weight(weight: Any) -> float | None:
    """Extract a scalar latest target weight from an engine output value."""
    try:
        import pandas as pd  # noqa: F401 - pandas is a core dependency
    except ImportError:  # pragma: no cover
        pd = None
    if isinstance(weight, (int, float)) and not isinstance(weight, bool):
        return float(weight)
    if hasattr(weight, "iloc"):  # pd.Series / pandas-compat
        if len(weight) == 0:
            return None
        return float(weight.iloc[-1])
    if isinstance(weight, (list, tuple)):
        if not weight:
            return None
        return float(weight[-1])
    return None


def _clip_weight(raw: float | None, allow_short: bool, leverage: float) -> float:
    """Clip one weight into the manifest's risk policy."""
    weight = float(raw or 0.0)
    lo = -abs(float(leverage)) if allow_short else 0.0
    hi = abs(float(leverage))
    return min(max(weight, lo), hi)


def run_manifest_paper_pass(manifest_id: str) -> str:
    """Execute one manifest-driven pass on the Alpaca paper account.

    Returns the pass record as JSON (also appended to the paper audit log).
    Raises RuntimeError on any pre-trade failure (drifted manifest, missing
    engine, no equity) so the job model records it as ``error``.
    """
    from src.trading.service import place_order

    manifest = _verified_manifest(manifest_id)
    config = manifest.get("config") or {}
    codes = config.get("codes") or []
    if not codes:
        raise RuntimeError(f"manifest {manifest_id} has an empty universe")
    allow_short = bool((manifest.get("risk_policy") or {}).get("allow_short", False))
    leverage = float((manifest.get("risk_policy") or {}).get("leverage", 1.0) or 1.0)

    engine_cls = _load_engine_class(Path(manifest["run_dir"]))
    engine = engine_cls()

    equity = _account_equity()
    if equity is None or equity <= 0:
        raise RuntimeError("paper account equity unavailable; refusing to size orders")

    held = _held_map(_positions_reply())

    # Bars for every symbol, then one vectorized generate() like a backtest.
    data_map: dict[str, Any] = {}
    order_symbols: dict[str, tuple[str, bool]] = {}
    last_close: dict[str, float] = {}
    fetch_errors: list[dict] = []
    for code in codes:
        try:
            rows = _load_rows(code, _INTERVAL, _LOOKBACK_DAYS)
        except Exception as exc:  # noqa: BLE001 - one symbol must not abort the pass
            fetch_errors.append({"symbol": str(code), "reason": f"history error: {exc}"})
            continue
        if not rows:
            fetch_errors.append({"symbol": str(code), "reason": "no bars"})
            continue
        try:
            import pandas as pd

            frame = pd.DataFrame(rows)
        except ImportError:  # pragma: no cover
            frame = rows  # list-of-dicts fallback
        data_map[str(code)] = frame
        order_symbols[str(code)] = _alpaca_order_symbol(str(code))
        last_close[str(code)] = float(rows[-1]["close"])

    if not data_map:
        raise RuntimeError(f"no market data for any of {codes}; refusing to trade blind")

    weights = engine.generate(data_map) or {}

    actions: list[dict] = []
    for code, weight_value in weights.items():
        order_symbol, _ = order_symbols.get(code, _alpaca_order_symbol(code))
        key = _norm(order_symbol)
        is_crypto = _is_crypto_pair(order_symbol)
        tif = "gtc" if is_crypto else "day"

        raw = _latest_weight(weight_value)
        if raw is None:
            actions.append({"symbol": order_symbol, "action": "skip", "reason": "unusable weight"})
            continue
        weight = _clip_weight(raw, allow_short, leverage)

        target_notional = weight * equity
        target_notional = max(min(target_notional, MAX_NOTIONAL_USD), -MAX_NOTIONAL_USD)
        price = last_close.get(code)
        if price is None or price <= 0:
            actions.append({"symbol": order_symbol, "action": "skip", "reason": "no close for sizing"})
            continue
        target_qty = target_notional / price
        held_qty = held.get(key, 0.0)
        delta_qty = target_qty - held_qty
        delta_notional = delta_qty * price

        entry = {
            "symbol": order_symbol,
            "crypto": is_crypto,
            "weight": weight,
            "target_notional": target_notional,
            "held_qty": held_qty,
        }
        if abs(delta_notional) < MIN_DELTA_NOTIONAL_USD:
            actions.append({**entry, "action": "hold", "reason": "within threshold"})
            continue

        if delta_qty > 0:
            result = place_order(
                order_symbol,
                PROFILE_ID,
                side="buy",
                notional=min(delta_notional, MAX_NOTIONAL_USD),
                order_type="market",
                time_in_force=tif,
                session_id=f"manifest-paper:{manifest_id[:12]}",
            )
            actions.append({**entry, "action": "buy", "delta_notional": delta_notional, "result": result})
        else:
            sell_qty = min(abs(delta_qty), held_qty) if held_qty > 0 else abs(delta_qty)
            if held_qty <= 0:
                # Short of a flat book: long-only paper scope never opens shorts.
                actions.append({**entry, "action": "skip", "reason": "sell with no holding (short out of scope)"})
                continue
            result = place_order(
                order_symbol,
                PROFILE_ID,
                side="sell",
                quantity=sell_qty,
                order_type="market",
                time_in_force=tif,
                session_id=f"manifest-paper:{manifest_id[:12]}",
            )
            actions.append({**entry, "action": "sell", "delta_notional": delta_notional, "result": result})

    record = {
        "fired_at": datetime.now(timezone.utc).isoformat(),
        "profile": PROFILE_ID,
        "manifest_id": manifest_id,
        "manifest_name": manifest.get("name"),
        "equity_before": equity,
        "equity_after": _account_equity(),
        "fetch_errors": fetch_errors,
        "actions": actions,
    }
    _append_audit(record)
    return json.dumps(record, ensure_ascii=False, default=str)


def _append_audit(record: dict) -> None:
    log_path = Path.home() / ".vibe-trading" / "paper_pass_log.jsonl"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        # The audit log is best-effort; the orders already happened.
        pass


__all__ = [
    "PROFILE_ID",
    "MAX_NOTIONAL_USD",
    "MIN_DELTA_NOTIONAL_USD",
    "run_manifest_paper_pass",
]