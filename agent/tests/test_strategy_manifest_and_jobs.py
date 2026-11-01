"""Regression tests for the QuantDinger-inspired additions to Vibe-Trading:

- ``strategy_manifest.py`` — compiled strategy manifests with content-addressed
  versioning and warmup estimation (Feature 4, stolen from QD).
- ``job_model.py`` — bounded async job model with idempotent submit, atomic
  JSON persistence, and mandatory cancel confirmation (Feature 2).

Contract notes mirrored from the implementations:
- ManifestStore.save returns the file *path* (str), not an id; get by id.
- Manifests are content-addressed by (config_core + engine source) — name is
  metadata only and does NOT affect manifest_id.
- JobStore.submit is idempotent by idempotency_key: same key → replay snapshot
  with ``idempotent_replay: True``; payloads are never compared (the key is
  the caller's claim of uniqueness, matching QuantDinger's semantics).
- Job statuses: pending | running | completed | error | cancelled.
- cancel() refuses without confirm_cancel=True (QD's confirmation gate).
- job_snapshot_envelope returns a JSON *string* envelope.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from src.tools.strategy_manifest import (
    ManifestStore,
    ManifestError,
    compile_strategy,
    get_manifest_store,
)
from src.tools.job_model import (
    JobError,
    JobStore,
    job_snapshot_envelope,
)

# ---------------------------------------------------------------------------
# Fixtures: a minimal VT strategy run directory
# ---------------------------------------------------------------------------

@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "strategy_runs" / "dual_ema"
    (rd / "code").mkdir(parents=True)
    (rd / "config.json").write_text(
        json.dumps({
            "source": "yfinance",
            "codes": ["SPY", "QQQ"],
            "start_date": "2020-01-01",
            "end_date": "2025-01-01",
            "initial_capital": 100_000,
            "benchmark": "SPY",
        }),
        encoding="utf-8",
    )
    (rd / "code" / "signal_engine.py").write_text(
        "import pandas as pd\n"
        "FAST, SLOW = 20, 50\n"
        "class SignalEngine:\n"
        "    def generate(self, data_map):\n"
        "        w = {}\n"
        "        for s, df in data_map.items():\n"
        "            fast = df['close'].rolling(FAST).mean()\n"
        "            slow = df['close'].rolling(SLOW).shift(1).mean()\n"
        "            w[s] = (fast > slow).astype(float)\n"
        "        return w\n",
        encoding="utf-8",
    )
    return rd


# ---------------------------------------------------------------------------
# compile_strategy: warmup estimation + content addressing
# ---------------------------------------------------------------------------

def test_compile_strategy_warmup_estimated_from_rolling_window(run_dir: Path) -> None:
    manifest = compile_strategy(run_dir=run_dir, name="dual_ema")
    # engine references rolling(SLOW=50) and shift(1) → estimated max = 50
    assert manifest["warmup_estimated"] == 50
    # not declared in config → warmup_bars is None until a caller decides
    assert manifest["warmup_bars"] is None


def test_compile_strategy_warmup_declared_in_config_wins(run_dir: Path) -> None:
    cfg = json.loads((run_dir / "config.json").read_text())
    cfg["warmup_bars"] = 300
    (run_dir / "config.json").write_text(json.dumps(cfg))
    manifest = compile_strategy(run_dir=run_dir)
    assert manifest["warmup_bars"] == 300
    # estimate is not recomputed when a value is declared
    assert manifest["warmup_estimated"] is None


def test_compile_strategy_content_addressed_id(run_dir: Path) -> None:
    m1 = compile_strategy(run_dir=run_dir, name="dual_ema")
    m2 = compile_strategy(run_dir=run_dir, name="dual_ema")
    # same content → same id; name does not participate (metadata only)
    m_named = compile_strategy(run_dir=run_dir, name="totally_different")
    assert m1["manifest_id"] == m2["manifest_id"]
    assert m1["manifest_id"] == m_named["manifest_id"]


def test_compile_strategy_id_changes_with_content(run_dir: Path) -> None:
    m1 = compile_strategy(run_dir=run_dir, name="dual_ema")
    (run_dir / "code" / "signal_engine.py").write_text(
        "import pandas as pd\n"
        "class SignalEngine:\n"
        "    def generate(self, data_map):\n"
        "        return {s: 1.0 for s in data_map}\n",
        encoding="utf-8",
    )
    m2 = compile_strategy(run_dir=run_dir, name="dual_ema")
    assert m1["manifest_id"] != m2["manifest_id"]
    assert m1["engine_sha256"] != m2["engine_sha256"]


def test_compile_strategy_snapshot_shape(run_dir: Path) -> None:
    manifest = compile_strategy(run_dir=run_dir, name="dual_ema")
    assert manifest["name"] == "dual_ema"
    assert manifest["config"]["codes"] == ["SPY", "QQQ"]
    assert manifest["universe"] == {
        "source": "yfinance", "codes": ["SPY", "QQQ"], "market": None,
    }
    assert manifest["period"] == {"start": "2020-01-01", "end": "2025-01-01"}
    assert manifest["risk_policy"]["initial_capital"] == 100_000
    assert manifest["risk_policy"]["allow_short"] is False
    assert manifest["engine_sha256"]
    assert manifest["schema_version"] == 1


def test_compile_strategy_rejects_missing_config(tmp_path: Path) -> None:
    rd = tmp_path / "empty"
    rd.mkdir()
    with pytest.raises(ManifestError):
        compile_strategy(run_dir=rd, name="x")


def test_compile_strategy_rejects_missing_engine(tmp_path: Path) -> None:
    rd = tmp_path / "cfg_only"
    rd.mkdir()
    (rd / "config.json").write_text("{}")
    with pytest.raises(ManifestError):
        compile_strategy(run_dir=rd, name="x")


def test_compile_strategy_rejects_bad_json(tmp_path: Path) -> None:
    rd = tmp_path / "bad"
    (rd / "code").mkdir(parents=True)
    (rd / "config.json").write_text("{not json")
    (rd / "code" / "signal_engine.py").write_text("x = 1\n")
    with pytest.raises(ManifestError):
        compile_strategy(run_dir=rd, name="x")


# ---------------------------------------------------------------------------
# ManifestStore: save / get / list semantics
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path: Path) -> ManifestStore:
    return ManifestStore(root=tmp_path)


def test_save_and_get_roundtrip(store: ManifestStore, run_dir: Path) -> None:
    manifest = compile_strategy(run_dir=run_dir, name="dual_ema")
    path = store.save(manifest)
    assert path.endswith(f"{manifest['manifest_id']}.json")
    loaded = store.get(manifest["manifest_id"])
    assert loaded == manifest


def test_save_is_atomic_write(store: ManifestStore, run_dir: Path) -> None:
    # save goes through a .tmp + replace — no partial file is observable
    manifest = compile_strategy(run_dir=run_dir)
    store.save(manifest)
    directory = store.root / "strategy-manifests"
    assert not list(directory.glob("*.tmp")), "stray .tmp file left behind"


def test_get_missing_raises(store: ManifestStore) -> None:
    with pytest.raises(ManifestError):
        store.get("f" * 24)


def test_get_unsafe_id_rejected(store: ManifestStore) -> None:
    with pytest.raises(ManifestError):
        store.get("../../etc/passwd")


def test_save_requires_valid_id(store: ManifestStore) -> None:
    with pytest.raises(ManifestError):
        store.save({"manifest_id": "nope!"})


def test_list_manifests(store: ManifestStore, run_dir: Path) -> None:
    ids = set()
    for i in range(3):
        cfg_path = run_dir / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["initial_capital"] = 100_000 + i  # content changes → new id
        cfg_path.write_text(json.dumps(cfg))
        m = compile_strategy(run_dir=run_dir, name=f"strat_{i}")
        store.save(m)
        ids.add(m["manifest_id"])
    listed = store.list()
    assert {e["manifest_id"] for e in listed} == ids


def test_list_empty_before_any_save(tmp_path: Path) -> None:
    store = ManifestStore(root=tmp_path)
    assert store.list() == []


def test_get_manifest_store_is_singleton(monkeypatch, tmp_path: Path) -> None:
    import src.tools.strategy_manifest as sm
    monkeypatch.setattr(sm, "_manifest_store_singleton", None)
    import src.config.paths as paths_mod

    def _fake_root() -> str:
        return str(tmp_path)

    monkeypatch.setattr(paths_mod, "get_runtime_root", _fake_root)
    s1 = sm.get_manifest_store()
    s2 = sm.get_manifest_store()
    assert s1 is s2


# ---------------------------------------------------------------------------
# JobStore: submit / get / list / cancel / persistence
# ---------------------------------------------------------------------------

def _wait_terminal(store: JobStore, job_id: str, timeout: float = 10.0) -> dict[str, Any]:
    """Poll until the job reaches a terminal status (the MCP layer has
    wait_for_job; the store itself is poll-based)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = store.get(job_id)
        if snap["status"] in {"completed", "error", "cancelled"}:
            return snap
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} never reached terminal: {snap}")


@pytest.fixture()
def job_store(tmp_path: Path) -> JobStore:
    return JobStore(root=tmp_path)


def _work(value: int = 42, delay: float = 0.0) -> Any:
    def run() -> str:
        if delay:
            time.sleep(delay)
        return json.dumps({"value": value})
    return run


def test_submit_returns_running_snapshot(job_store: JobStore) -> None:
    snap = job_store.submit(
        kind="backtest",
        idempotency_key="k-1",
        request={"strategy": "dual_ema"},
        work=_work(value=7),
    )
    assert snap["job_id"]
    assert snap["status"] in {"running", "completed"}
    assert snap["kind"] == "backtest"


def test_submit_runs_work_to_completion(job_store: JobStore) -> None:
    snap = job_store.submit(
        kind="backtest", idempotency_key="k-2", request={}, work=_work(value=7),
    )
    final = _wait_terminal(job_store, snap["job_id"])
    assert final["status"] == "completed"
    assert json.loads(final["result_preview"])["value"] == 7
    assert final["ended_at"] >= final["started_at"]


def test_submit_is_idempotent_on_same_key(job_store: JobStore) -> None:
    s1 = job_store.submit(
        kind="backtest", idempotency_key="same", request={}, work=_work(value=1),
    )
    s2 = job_store.submit(
        kind="backtest", idempotency_key="same", request={}, work=_work(value=999),
    )
    # second submit replays the first job — the 999 work never runs
    assert s2["job_id"] == s1["job_id"]
    assert s2.get("idempotent_replay") is True
    _wait_terminal(job_store, s1["job_id"])
    snap = job_store.get(s1["job_id"])
    assert json.loads(snap["result_preview"])["value"] == 1


def test_submit_rejects_blank_key(job_store: JobStore) -> None:
    with pytest.raises(JobError):
        job_store.submit(
            kind="backtest", idempotency_key="   ", request={}, work=_work(),
        )


def test_get_unknown_job_raises(job_store: JobStore) -> None:
    with pytest.raises(JobError):
        job_store.get("a" * 32)  # shape-valid but never submitted


def test_get_rejects_unsafe_job_id(job_store: JobStore) -> None:
    with pytest.raises(JobError):
        job_store.get("../escape")


def test_list_jobs_newest_first(job_store: JobStore) -> None:
    ids = []
    for i in range(3):
        s = job_store.submit(
            kind="backtest", idempotency_key=f"list-{i}", request={}, work=_work(delay=0.05),
        )
        ids.append(s["job_id"])
    rows = job_store.list()
    # newest first == reverse submission order
    assert [r["job_id"] for r in rows] == list(reversed(ids))
    # limit truncates from the newest end
    assert [r["job_id"] for r in job_store.list(limit=1)] == [ids[-1]]


def test_cancel_requires_confirmation(job_store: JobStore) -> None:
    s = job_store.submit(
        kind="backtest", idempotency_key="cancel-1", request={}, work=_work(delay=5),
    )
    with pytest.raises(JobError, match="confirm_cancel"):
        job_store.cancel(s["job_id"])
    # job untouched by the refused cancel
    assert job_store.get(s["job_id"])["status"] == "running"


def test_cancel_with_confirmation_marks_cancelled(job_store: JobStore) -> None:
    s = job_store.submit(
        kind="backtest", idempotency_key="cancel-2", request={}, work=_work(delay=5),
    )
    snap = job_store.cancel(s["job_id"], confirm_cancel=True)
    assert snap["status"] == "cancelled"
    assert snap["ended_at"] is not None


def test_cancel_is_a_noop_on_terminal_job(job_store: JobStore) -> None:
    s = job_store.submit(
        kind="backtest", idempotency_key="cancel-3", request={}, work=_work(),
    )
    _wait_terminal(job_store, s["job_id"])
    snap = job_store.cancel(s["job_id"], confirm_cancel=True)
    assert snap["status"] == "completed"  # not overwritten to cancelled


def test_job_failure_recorded(job_store: JobStore) -> None:
    def boom() -> str:
        raise RuntimeError("worker boom")
    s = job_store.submit(
        kind="alpha_bench", idempotency_key="fail-1", request={}, work=boom,
    )
    final = _wait_terminal(job_store, s["job_id"])
    assert final["status"] == "error"
    assert "worker boom" in final["error"]


def test_job_persisted_to_disk(job_store: JobStore) -> None:
    s = job_store.submit(
        kind="backtest", idempotency_key="disk-1", request={}, work=_work(),
    )
    _wait_terminal(job_store, s["job_id"])
    record_path = job_store.root / "jobs" / f"{s['job_id']}.json"
    assert record_path.is_file()
    data = json.loads(record_path.read_text(encoding="utf-8"))
    assert data["status"] in {"completed", "error"}


def test_snapshot_envelope_is_json_string(job_store: JobStore) -> None:
    s = job_store.submit(
        kind="backtest", idempotency_key="env-1", request={}, work=_work(),
    )
    env = job_snapshot_envelope(job_store.get(s["job_id"]))
    assert isinstance(env, str)
    parsed = json.loads(env)
    assert parsed["status"] == "ok"
    assert parsed["job"]["job_id"] == s["job_id"]


def test_raw_idempotency_key_never_echoed(job_store: JobStore) -> None:
    # _public strips the raw key; only the digest correlation exists
    s = job_store.submit(
        kind="backtest", idempotency_key="SECRET-KEY-XYZ", request={}, work=_work(),
    )
    snap = job_store.get(s["job_id"])
    assert "SECRET-KEY-XYZ" not in json.dumps(snap)
    assert snap.get("idempotent") is True


def test_request_preview_is_bounded(job_store: JobStore) -> None:
    big_request = {f"k{i}": i for i in range(200)}
    s = job_store.submit(
        kind="backtest", idempotency_key="big-1", request=big_request, work=_work(),
    )
    snap = job_store.get(s["job_id"])
    assert len(snap["request_preview"]) <= 32